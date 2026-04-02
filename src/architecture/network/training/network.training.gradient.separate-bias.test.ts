import Network from '../network';

type TrainingStatsSnapshot = {
  lossScale: number;
};

describe('network training chapter', () => {
  describe('gradient grouping details', () => {
    describe('given separateBias is toggled for layerwise clipping', () => {
      describe('when the recorded group counts are compared', () => {
        it('uses at least as many groups when separateBias is enabled', () => {
          // Arrange
          const baselineNetwork = new Network(3, 2, { seed: 200 });
          const separateBiasNetwork = new Network(3, 2, { seed: 200 });

          baselineNetwork.train([{ input: [0, 0, 0], output: [0, 1] }], {
            iterations: 1,
            rate: 0.01,
            optimizer: 'adam',
            gradientClip: { mode: 'layerwiseNorm', maxNorm: 1 },
          });
          separateBiasNetwork.train([{ input: [0, 0, 0], output: [0, 1] }], {
            iterations: 1,
            rate: 0.01,
            optimizer: 'adam',
            gradientClip: {
              mode: 'layerwiseNorm',
              maxNorm: 1,
              separateBias: true,
            },
          });

          const baselineGroupCount =
            baselineNetwork.getLastGradClipGroupCount();

          // Act
          const separateBiasGroupCount =
            separateBiasNetwork.getLastGradClipGroupCount();

          // Assert
          expect(separateBiasGroupCount).toBeGreaterThanOrEqual(
            baselineGroupCount,
          );
        });
      });
    });

    describe('given training stats are captured after one clipped training step', () => {
      describe('when the runtime loss scale is inspected', () => {
        it('reports a positive loss scale', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 201 });

          network.train([{ input: [0, 0], output: [1] }], {
            iterations: 1,
            rate: 0.01,
            optimizer: 'adam',
            gradientClip: { mode: 'norm', maxNorm: 1 },
          });

          const trainingStats =
            network.getTrainingStats() as TrainingStatsSnapshot;

          // Act
          const lossScale = trainingStats.lossScale;

          // Assert
          expect(lossScale).toBeGreaterThan(0);
        });
      });
    });

    describe('given accumulation reduction switches to sum mode', () => {
      describe('when the recommended adjusted rate is computed', () => {
        it('divides the base rate by the accumulation step count', () => {
          // Arrange
          const baseRate = 0.01;

          // Act
          const adjustedRate = Network.adjustRateForAccumulation(
            baseRate,
            4,
            'sum',
          );

          // Assert
          expect(adjustedRate).toBeCloseTo(baseRate / 4, 10);
        });
      });
    });
  });
});
