import { Architect, Network } from '../../../neataptic';

type TrainingSample = { input: number[]; output: number[] };

jest.retryTimes(5, { logErrorsBeforeRetry: true });
jest.setTimeout(30_000);

function createLinearCongruentialGenerator(seed: number): () => number {
  let state = seed >>> 0;

  return () => {
    state = (state * 1_664_525 + 1_013_904_223) >>> 0;
    return state / 0xffff_ffff;
  };
}

function createExclusiveOrDataset(): TrainingSample[] {
  return [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] },
  ];
}

function createInclusiveOrDataset(): TrainingSample[] {
  return [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [1] },
  ];
}

function createNormalizedSineDataset(
  sampleCount: number,
  seed: number,
): TrainingSample[] {
  const nextRandomValue = createLinearCongruentialGenerator(seed);

  return Array.from({ length: sampleCount }, () => {
    const angle = nextRandomValue() * Math.PI * 2;

    return {
      input: [angle / (Math.PI * 2)],
      output: [(Math.sin(angle) + 1) / 2],
    };
  });
}

function createRawSineDataset(
  sampleCount: number,
  seed: number,
): TrainingSample[] {
  const nextRandomValue = createLinearCongruentialGenerator(seed);

  return Array.from({ length: sampleCount }, () => {
    const angle = nextRandomValue() * Math.PI * 2;

    return {
      input: [angle],
      output: [Math.sin(angle)],
    };
  });
}

function createNormalizedSineCosineDataset(
  sampleCount: number,
  seed: number,
): TrainingSample[] {
  const nextRandomValue = createLinearCongruentialGenerator(seed);

  return Array.from({ length: sampleCount }, () => {
    const angle = nextRandomValue() * Math.PI * 2;

    return {
      input: [angle / (Math.PI * 2)],
      output: [(Math.sin(angle) + 1) / 2, (Math.cos(angle) + 1) / 2],
    };
  });
}

function createSequenceEchoDataset(): TrainingSample[] {
  return [0.1, 0.8, 0.2, 0.6].map((inputValue) => ({
    input: [inputValue],
    output: [inputValue],
  }));
}

function measureSequenceMeanAbsoluteError(
  network: Network,
  sequenceDataset: TrainingSample[],
): number {
  network.clear();

  const totalAbsoluteError = sequenceDataset.reduce(
    (runningTotal, trainingSample) => {
      const [actualOutput = 0] = network.activate(trainingSample.input);
      return runningTotal + Math.abs(trainingSample.output[0] - actualOutput);
    },
    0,
  );

  return totalAbsoluteError / sequenceDataset.length;
}

function trainSequenceDatasetWithManualEpochs(
  network: Network,
  sequenceDataset: TrainingSample[],
): void {
  for (let epochIndex = 0; epochIndex < 160; epochIndex += 1) {
    network.clear();

    for (const trainingSample of sequenceDataset) {
      network.activate(trainingSample.input, true);
      network.propagate(0.2, 0, true, trainingSample.output);
    }
  }
}

describe('network training learning capability chapter', () => {
  describe('logic-gate learning', () => {
    describe('given a perceptron trains on XOR', () => {
      describe('when sufficient iterations are allowed', () => {
        it('reduces the training error below 0.25', () => {
          // Arrange
          const network = Architect.perceptron(2, 10, 1);
          network.setSeed(201);

          // Act
          const trainingResult = network.train(createExclusiveOrDataset(), {
            iterations: 5_000,
            error: 0.25,
            shuffle: true,
            rate: 0.3,
            momentum: 0.9,
          });

          // Assert
          expect(trainingResult.error).toBeLessThan(0.25);
        });
      });

      describe('when iterations are intentionally constrained', () => {
        it('keeps the training error at or above 0.2', () => {
          // Arrange
          const network = Architect.perceptron(2, 10, 1);
          network.setSeed(202);

          // Act
          const trainingResult = network.train(createExclusiveOrDataset(), {
            iterations: 1,
            error: 0.25,
            shuffle: true,
            rate: 0.3,
            momentum: 0.9,
          });

          // Assert
          expect(trainingResult.error).toBeGreaterThanOrEqual(0.2);
        });
      });
    });

    describe('given a perceptron trains on OR', () => {
      describe('when sufficient iterations are allowed', () => {
        it('reduces the training error below 0.35', () => {
          // Arrange
          const network = Architect.perceptron(2, 3, 1);
          network.setSeed(203);

          // Act
          const trainingResult = network.train(createInclusiveOrDataset(), {
            iterations: 3_000,
            error: 0.01,
          });

          // Assert
          expect(trainingResult.error).toBeLessThan(0.35);
        });
      });

      describe('when iterations are intentionally constrained', () => {
        it('keeps the training error at or above 0.22', () => {
          // Arrange
          const network = Architect.perceptron(2, 1, 1);
          network.setSeed(204);

          // Act
          const trainingResult = network.train(createInclusiveOrDataset(), {
            iterations: 2,
            error: 0.01,
            rate: 0.1,
          });

          // Assert
          expect(trainingResult.error).toBeGreaterThanOrEqual(0.22);
        });
      });
    });
  });

  describe('function approximation learning', () => {
    describe('given a perceptron trains on a normalized sine curve', () => {
      describe('when sufficient iterations are allowed', () => {
        it('reduces the training error below 0.25', () => {
          // Arrange
          const network = Architect.perceptron(1, 15, 1);
          network.setSeed(205);

          // Act
          const trainingResult = network.train(
            createNormalizedSineDataset(100, 301),
            {
              iterations: 3_000,
              error: 0.2,
              shuffle: true,
              rate: 0.3,
              momentum: 0.9,
            },
          );

          // Assert
          expect(trainingResult.error).toBeLessThan(0.25);
        });
      });

      describe('when iterations are intentionally constrained', () => {
        it('keeps the training error at or above 0.11', () => {
          // Arrange
          const network = Architect.perceptron(1, 12, 1);
          network.setSeed(206);

          // Act
          const trainingResult = network.train(createRawSineDataset(40, 302), {
            iterations: 1,
            error: 0.01,
            rate: 0.3,
            momentum: 0.9,
          });

          // Assert
          expect(trainingResult.error).toBeGreaterThanOrEqual(0.11);
        });
      });
    });

    describe('given a perceptron trains on normalized sine and cosine targets', () => {
      describe('when sufficient iterations are allowed', () => {
        it('reduces the training error below 0.2', () => {
          // Arrange
          const network = Architect.perceptron(1, 20, 2);
          network.setSeed(207);

          // Act
          const trainingResult = network.train(
            createNormalizedSineCosineDataset(100, 303),
            {
              iterations: 6_000,
              error: 0.2,
              shuffle: true,
              rate: 0.3,
              momentum: 0.9,
            },
          );

          // Assert
          expect(trainingResult.error).toBeLessThan(0.2);
        });
      });

      describe('when iterations are intentionally constrained', () => {
        it('keeps the training error at or above 0.15', () => {
          // Arrange
          const network = Architect.perceptron(1, 20, 2);
          network.setSeed(208);

          // Act
          const trainingResult = network.train(
            createNormalizedSineCosineDataset(100, 304),
            {
              iterations: 1,
              error: 0.15,
              shuffle: true,
              rate: 0.3,
              momentum: 0.9,
            },
          );

          // Assert
          expect(trainingResult.error).toBeGreaterThanOrEqual(0.15);
        });
      });
    });

  });

  describe('dropout during training', () => {
    describe('given a small XOR-style dataset trains with dropout enabled', () => {
      describe('when a short training run completes', () => {
        it('keeps the reported training error at or below 0.5', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 209 });

          // Act
          const trainingResult = network.train(createExclusiveOrDataset(), {
            iterations: 10,
            error: 0.5,
            dropout: 0.5,
          });

          // Assert
          expect(trainingResult.error).toBeLessThanOrEqual(0.5);
        });
      });
    });

    describe('given one GRU builder trains on a tiny ordered sequence', () => {
      describe('when manual recurrent epochs are used', () => {
        it('reduces the sequence echo error below its initial baseline', () => {
          // Arrange
          const network = Architect.gru(1, 2, 1, { inputToOutput: true });
          network.setSeed(310);
          const sequenceDataset = createSequenceEchoDataset();
          const initialError = measureSequenceMeanAbsoluteError(
            network,
            sequenceDataset,
          );

          // Act
          trainSequenceDatasetWithManualEpochs(network, sequenceDataset);
          const trainedError = measureSequenceMeanAbsoluteError(
            network,
            sequenceDataset,
          );

          // Assert
          expect(trainedError).toBeLessThan(initialError);
        });
      });
    });
  });
});
