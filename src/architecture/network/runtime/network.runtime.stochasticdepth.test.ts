import { Architect } from '../../../neataptic';

type StochasticDepthNetwork = {
  activate: (input: number[], training?: boolean) => number[];
  disableStochasticDepth: () => void;
  lastSkippedLayers: number[];
  setSeed: (seed: number) => void;
  setStochasticDepth: (survival: number[]) => void;
};

describe('network runtime chapter', () => {
  describe('stochastic depth', () => {
    describe('given two matching seeded networks share the same survival schedule', () => {
      describe('when repeated training activations run', () => {
        it('replays the same skipped-layer pattern', () => {
          // Arrange
          const firstNetwork = Architect.perceptron(
            4,
            8,
            8,
            8,
            2,
          ) as unknown as StochasticDepthNetwork;
          const secondNetwork = Architect.perceptron(
            4,
            8,
            8,
            8,
            2,
          ) as unknown as StochasticDepthNetwork;
          const survivalSchedule = [0.9, 0.5, 0.1];
          const inputVector = [0.1, 0.2, 0.3, 0.4];
          const firstPatterns: number[][] = [];
          const secondPatterns: number[][] = [];

          firstNetwork.setStochasticDepth(survivalSchedule);
          secondNetwork.setStochasticDepth(survivalSchedule);
          firstNetwork.setSeed(1234);
          secondNetwork.setSeed(1234);

          for (let iteration = 0; iteration < 20; iteration++) {
            firstNetwork.activate(inputVector, true);
            secondNetwork.activate(inputVector, true);
            firstPatterns.push(firstNetwork.lastSkippedLayers);
            secondPatterns.push(secondNetwork.lastSkippedLayers);
          }

          // Act
          const replayedIdentically =
            JSON.stringify(firstPatterns) === JSON.stringify(secondPatterns);

          // Assert
          expect(replayedIdentically).toBe(true);
        });
      });
    });

    describe('given stochastic depth was configured earlier', () => {
      describe('when stochastic depth is disabled', () => {
        it('clears the stored survival schedule', () => {
          // Arrange
          const network = Architect.perceptron(
            2,
            5,
            1,
          ) as unknown as StochasticDepthNetwork;
          network.setStochasticDepth([0.9]);

          // Act
          network.disableStochasticDepth();
          const configuredSchedule = Reflect.get(network, '_stochasticDepth');

          // Assert
          expect(
            Array.isArray(configuredSchedule) &&
              configuredSchedule.length === 0,
          ).toBe(true);
        });
      });
    });
  });
});
