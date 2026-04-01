import Network from '../network';
import { Architect } from '../../../neataptic';

function createRuntimeDropoutNetwork(seed: number): Network {
  const network = Architect.perceptron(2, 10, 1);
  network.setSeed(seed);
  return network;
}

function getHiddenMasks(network: Network): number[] {
  return network.nodes
    .filter((node) => node.type === 'hidden')
    .map((node) => node.mask);
}

describe('network runtime chapter', () => {
  describe('dropout forward pass', () => {
    describe('given training activation runs with a high dropout rate', () => {
      describe('when hidden-node masks are inspected', () => {
        it('drops at least one hidden node while keeping at least one active', () => {
          // Arrange
          const network = createRuntimeDropoutNetwork(41);
          network.dropout = 0.8;

          // Act
          network.activate([1, 1], true);
          const hiddenMasks = getHiddenMasks(network);
          const maskedCount = hiddenMasks.filter((mask) => mask === 0).length;

          // Assert
          expect(maskedCount > 0 && maskedCount < hiddenMasks.length).toBe(
            true,
          );
        });
      });
    });

    describe('given training activation runs with zero dropout', () => {
      describe('when hidden-node masks are inspected', () => {
        it('keeps every hidden node active', () => {
          // Arrange
          const network = createRuntimeDropoutNetwork(42);
          network.dropout = 0;

          // Act
          network.activate([1, 1], true);
          const hiddenMasks = getHiddenMasks(network);

          // Assert
          expect(hiddenMasks.every((mask) => mask === 1)).toBe(true);
        });
      });
    });

    describe('given training activation runs with full dropout pressure', () => {
      describe('when hidden-node masks are inspected', () => {
        it('keeps exactly one hidden node active', () => {
          // Arrange
          const network = createRuntimeDropoutNetwork(43);
          network.dropout = 1;

          // Act
          network.activate([1, 1], true);
          const hiddenMasks = getHiddenMasks(network);
          const maskedCount = hiddenMasks.filter((mask) => mask === 0).length;

          // Assert
          expect(maskedCount).toBe(hiddenMasks.length - 1);
        });
      });
    });

    describe('given a training activation already dropped some nodes', () => {
      describe('when dropout masks are reset explicitly', () => {
        it('restores every hidden-node mask to one', () => {
          // Arrange
          const network = createRuntimeDropoutNetwork(44);
          network.dropout = 0.8;
          network.activate([1, 1], true);

          // Act
          network.resetDropoutMasks();
          const hiddenMasks = getHiddenMasks(network);

          // Assert
          expect(hiddenMasks.every((mask) => mask === 1)).toBe(true);
        });
      });
    });
  });

  describe('dropout determinism', () => {
    describe('given two networks share the same seed and dropout rate', () => {
      describe('when repeated training activations run', () => {
        it('replays the same hidden-node mask sequence', () => {
          // Arrange
          const firstNetwork = new Network(4, 2);
          const secondNetwork = new Network(4, 2);
          const firstMaskHistory: number[][] = [];
          const secondMaskHistory: number[][] = [];

          firstNetwork.dropout = 0.5;
          secondNetwork.dropout = 0.5;
          firstNetwork.setSeed(99);
          secondNetwork.setSeed(99);

          for (let iteration = 0; iteration < 15; iteration++) {
            firstNetwork.activate([0.2, 0.1, 0.05, 0.9], true);
            secondNetwork.activate([0.2, 0.1, 0.05, 0.9], true);

            firstMaskHistory.push(getHiddenMasks(firstNetwork));
            secondMaskHistory.push(getHiddenMasks(secondNetwork));
          }

          // Act
          const replayedIdentically =
            JSON.stringify(firstMaskHistory) ===
            JSON.stringify(secondMaskHistory);

          // Assert
          expect(replayedIdentically).toBe(true);
        });
      });
    });
  });
});
