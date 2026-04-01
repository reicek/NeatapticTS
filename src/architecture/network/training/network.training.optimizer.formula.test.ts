import Network from '../network';

type OptimizerConnection = Network['connections'][number] & {
  firstMoment?: number;
  secondMoment?: number;
  infinityNorm?: number;
};

type OptimizerConfig = { type: string } & Record<string, unknown>;
type TrainingDataset = Array<{ input: number[]; output: number[] }>;

function createDeterministicNetwork(seed: number): Network {
  const network = new Network(1, 1, { seed });

  network.connections.forEach((connection) => {
    connection.weight = 0.5;
  });

  network.nodes
    .filter((node) => node.type !== 'input')
    .forEach((node) => {
      node.bias = 0;
    });

  return network;
}

function createRegressionDataset(): TrainingDataset {
  return Array.from({ length: 4 }, (_, sampleIndex) => {
    return {
      input: [sampleIndex + 1],
      output: [2 * (sampleIndex + 1)],
    };
  });
}

function runOptimizerTraining(
  network: Network,
  optimizer: OptimizerConfig,
  iterations: number,
): void {
  network.train(createRegressionDataset(), {
    iterations,
    rate: 0.01,
    error: 0,
    batchSize: 1,
    optimizer,
    cost: {
      fn: (target: number[], output: number[]) => {
        return (output[0] - target[0]) ** 2;
      },
      calculate: (target: number[], output: number[]) => {
        return (output[0] - target[0]) ** 2;
      },
    },
  });
}

describe('network training chapter', () => {
  describe('optimizer formula differences', () => {
    describe('given adamax and adam run over the same gradients', () => {
      let adamaxInfinityNorm = 0;
      let adamSecondMoment = 0;

      beforeAll(() => {
        const adamaxNetwork = createDeterministicNetwork(220);
        const adamNetwork = createDeterministicNetwork(221);

        runOptimizerTraining(adamaxNetwork, { type: 'adamax' }, 2);
        runOptimizerTraining(adamNetwork, { type: 'adam' }, 2);

        adamaxInfinityNorm =
          (adamaxNetwork.connections[0] as OptimizerConnection).infinityNorm ??
          NaN;
        adamSecondMoment =
          (adamNetwork.connections[0] as OptimizerConnection).secondMoment ??
          NaN;
      });

      describe('when the stored normalization terms are compared', () => {
        it('does not collapse adamax infinity norm to adam variance magnitude', () => {
          // Arrange
          const normalizationDifference = Math.abs(
            adamaxInfinityNorm - Math.sqrt(adamSecondMoment),
          );

          // Act
          const observedDifference = normalizationDifference;

          // Assert
          expect(observedDifference).toBeGreaterThan(1e-12);
        });
      });
    });

    describe('given nadam and adam take the same first step', () => {
      let stepMagnitudeDifference = 0;

      beforeAll(() => {
        const nadamNetwork = createDeterministicNetwork(222);
        const adamNetwork = createDeterministicNetwork(223);

        runOptimizerTraining(nadamNetwork, { type: 'nadam' }, 1);
        runOptimizerTraining(adamNetwork, { type: 'adam' }, 1);

        const nadamDelta = nadamNetwork.connections[0].weight - 0.5;
        const adamDelta = adamNetwork.connections[0].weight - 0.5;
        stepMagnitudeDifference = Math.abs(nadamDelta - adamDelta);
      });

      describe('when the first-step magnitudes are compared', () => {
        it('produces a different update from adam', () => {
          // Arrange
          const observedDifference = stepMagnitudeDifference;

          // Act
          const updateDifference = observedDifference;

          // Assert
          expect(updateDifference).toBeGreaterThan(1e-12);
        });
      });
    });

    describe('given radam runs before and after variance rectification stabilizes', () => {
      let rectificationDifference = 0;

      beforeAll(() => {
        const earlyRAdamNetwork = createDeterministicNetwork(224);
        const lateRAdamNetwork = createDeterministicNetwork(225);

        runOptimizerTraining(earlyRAdamNetwork, { type: 'radam' }, 1);
        runOptimizerTraining(lateRAdamNetwork, { type: 'radam' }, 10);

        const earlyStepMagnitude = Math.abs(
          earlyRAdamNetwork.connections[0].weight - 0.5,
        );
        const lateStepMagnitude = Math.abs(
          lateRAdamNetwork.connections[0].weight - 0.5,
        );
        rectificationDifference = Math.abs(
          lateStepMagnitude - earlyStepMagnitude,
        );
      });

      describe('when the early and late step magnitudes are compared', () => {
        it('changes its update magnitude after rectification', () => {
          // Arrange
          const observedDifference = rectificationDifference;

          // Act
          const updateDifference = observedDifference;

          // Assert
          expect(updateDifference).toBeGreaterThan(1e-12);
        });
      });
    });

    describe('given adabelief and adam see the same gradients', () => {
      let secondMomentDifference = 0;

      beforeAll(() => {
        const adabeliefNetwork = createDeterministicNetwork(226);
        const adamNetwork = createDeterministicNetwork(227);

        runOptimizerTraining(adabeliefNetwork, { type: 'adabelief' }, 2);
        runOptimizerTraining(adamNetwork, { type: 'adam' }, 2);

        const adabeliefSecondMoment =
          (adabeliefNetwork.connections[0] as OptimizerConnection)
            .secondMoment ?? NaN;
        const adamSecondMoment =
          (adamNetwork.connections[0] as OptimizerConnection).secondMoment ??
          NaN;
        secondMomentDifference = Math.abs(
          adabeliefSecondMoment - adamSecondMoment,
        );
      });

      describe('when the variance caches are compared', () => {
        it('stores a different second moment than adam', () => {
          // Arrange
          const observedDifference = secondMomentDifference;

          // Act
          const varianceDifference = observedDifference;

          // Assert
          expect(varianceDifference).toBeGreaterThan(1e-12);
        });
      });
    });
  });
});
