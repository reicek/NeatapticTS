import Network from '../network';

type TrainingStatsSnapshot = {
  lastGradNorm?: number;
  lastGradNormRaw?: number;
  layerwiseGroupCount?: number;
  mp?: unknown;
};

const TWO_INPUT_DATASET = [
  { input: [0, 0], output: [1] },
  { input: [0.25, 0.5], output: [1] },
  { input: [0.5, 0.25], output: [1] },
  { input: [0.75, 0.75], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [0, 1], output: [1] },
  { input: [1, 1], output: [1] },
  { input: [0.1, 0.9], output: [1] },
];

function cloneNetworkParameters(
  sourceNetwork: Network,
  targetNetwork: Network,
): void {
  for (
    let connectionIndex = 0;
    connectionIndex < sourceNetwork.connections.length;
    connectionIndex++
  ) {
    targetNetwork.connections[connectionIndex].weight =
      sourceNetwork.connections[connectionIndex].weight;
  }

  for (let nodeIndex = 0; nodeIndex < sourceNetwork.nodes.length; nodeIndex++) {
    Reflect.set(
      targetNetwork.nodes[nodeIndex],
      'bias',
      Reflect.get(sourceNetwork.nodes[nodeIndex], 'bias'),
    );
  }
}

function createTwoInputNetwork(seed: number): Network {
  return new Network(2, 1, { seed });
}

describe('network training chapter', () => {
  describe('gradient refinements', () => {
    describe('given accumulation reduction switches from average to sum', () => {
      describe('when the resulting activations are compared', () => {
        it('keeps the outputs close after rate adjustment', () => {
          // Arrange
          const averageReductionNetwork = createTwoInputNetwork(190);
          const sumReductionNetwork = createTwoInputNetwork(191);
          cloneNetworkParameters(averageReductionNetwork, sumReductionNetwork);

          averageReductionNetwork.train(TWO_INPUT_DATASET, {
            iterations: 1,
            rate: 0.01,
            batchSize: 1,
            accumulationSteps: 4,
            accumulationReduction: 'average',
            optimizer: 'adam',
          });
          sumReductionNetwork.train(TWO_INPUT_DATASET, {
            iterations: 1,
            rate: 0.01 / 4,
            batchSize: 1,
            accumulationSteps: 4,
            accumulationReduction: 'sum',
            optimizer: 'adam',
          });

          const averageOutput = averageReductionNetwork.activate([0.2, 0.3])[0];

          // Act
          const sumOutput = sumReductionNetwork.activate([0.2, 0.3])[0];

          // Assert
          expect(Math.abs(averageOutput - sumOutput)).toBeLessThan(0.08);
        });
      });
    });

    describe('given norm clipping runs during training', () => {
      let rawGradientNorm = 0;
      let trainingStats: TrainingStatsSnapshot;

      beforeAll(() => {
        const network = new Network(1, 1, { seed: 192 });
        const dataset = Array.from({ length: 6 }, (_, sampleIndex) => ({
          input: [sampleIndex],
          output: [2 * sampleIndex],
        }));

        network.train(dataset, {
          iterations: 1,
          rate: 0.1,
          batchSize: 3,
          gradientClip: { mode: 'norm', maxNorm: 0.05 },
        });

        rawGradientNorm = network.getRawGradientNorm();
        trainingStats = network.getTrainingStats() as TrainingStatsSnapshot;
      });

      describe('when the telemetry snapshot is inspected', () => {
        it('exposes the raw gradient norm', () => {
          // Arrange
          const capturedRawGradientNorm = rawGradientNorm;

          // Act
          const hasRawGradientNorm = capturedRawGradientNorm >= 0;

          // Assert
          expect(hasRawGradientNorm).toBe(true);
        });
      });

      describe('when raw and clipped norms are compared', () => {
        it('keeps the raw norm at or above the clipped norm', () => {
          // Arrange
          const clippedGradientNorm = trainingStats.lastGradNorm ?? 0;

          // Act
          const rawIsNotSmaller = rawGradientNorm >= clippedGradientNorm - 1e-6;

          // Assert
          expect(rawIsNotSmaller).toBe(true);
        });
      });
    });

    describe('given layerwise clipping runs during training', () => {
      describe('when the training stats are inspected afterward', () => {
        it('records at least one clipping group', () => {
          // Arrange
          const network = new Network(3, 2, { seed: 193 });

          network.train([{ input: [0.1, 0.2, 0.3], output: [0, 1] }], {
            iterations: 1,
            rate: 0.01,
            batchSize: 1,
            optimizer: 'adam',
            gradientClip: { mode: 'layerwiseNorm', maxNorm: 0.1 },
          });

          // Act
          const layerwiseGroupCount = network.getLastGradClipGroupCount();

          // Assert
          expect(layerwiseGroupCount).toBeGreaterThan(0);
        });
      });
    });

    describe('given dynamic mixed precision sees a forced overflow path', () => {
      describe('when the training stats are inspected afterward', () => {
        it('exposes mixed-precision telemetry', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 194 });
          const dataset = [{ input: [1], output: [2] }];

          network.train(dataset, {
            iterations: 1,
            batchSize: 1,
            mixedPrecision: {
              dynamic: true,
              lossScale: 128,
              scaleFactor: 2,
              scaleWindow: 1,
            },
          });
          network.testForceOverflow();
          network.train(dataset, {
            iterations: 1,
            batchSize: 1,
            mixedPrecision: {
              dynamic: true,
              lossScale: 128,
              scaleFactor: 2,
              scaleWindow: 1,
            },
          });

          const trainingStats =
            network.getTrainingStats() as TrainingStatsSnapshot;

          // Act
          const mixedPrecisionTelemetry = trainingStats.mp;

          // Assert
          expect(mixedPrecisionTelemetry).toBeDefined();
        });
      });
    });

    describe('given dynamic mixed precision sees many stable steps', () => {
      describe('when the runtime loss scale is inspected afterward', () => {
        it('increases up to the configured target scale', () => {
          // Arrange
          const network = createTwoInputNetwork(195);
          const initialScale = 64;
          const targetScale = initialScale * 2;

          network.train([{ input: [0, 0], output: [1] }], {
            iterations: 250,
            rate: 0.01,
            optimizer: 'adam',
            mixedPrecision: {
              lossScale: initialScale,
              dynamic: { increaseEvery: 100, maxScale: targetScale },
            },
          });

          // Act
          const lossScale = network.getLossScale();

          // Assert
          expect(lossScale).toBeGreaterThanOrEqual(targetScale);
        });
      });
    });
  });
});
