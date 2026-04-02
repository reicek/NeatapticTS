import type Network from '../network';
import { Architect } from '../../../neataptic';

type TrainingSample = { input: number[]; output: number[] };

const XOR_DATASET: TrainingSample[] = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

function createTrainingDropoutNetwork(seed: number): Network {
  const network = Architect.perceptron(2, 10, 1);
  network.setSeed(seed);
  return network;
}

function getHiddenMasks(network: Network): number[] {
  return network.nodes
    .filter((node) => node.type === 'hidden')
    .map((node) => node.mask);
}

describe('network training chapter', () => {
  describe('dropout lifecycle', () => {
    describe('given training runs with dropout enabled', () => {
      describe('when train() completes', () => {
        it('restores every hidden-node mask to one', () => {
          // Arrange
          const network = createTrainingDropoutNetwork(51);

          // Act
          network.train(XOR_DATASET, { iterations: 2, dropout: 0.5 });
          const hiddenMasks = getHiddenMasks(network);

          // Assert
          expect(hiddenMasks.every((mask) => mask === 1)).toBe(true);
        });
      });
    });

    describe('given a network was trained with dropout already', () => {
      describe('when test() completes', () => {
        it('keeps every hidden-node mask at one for inference', () => {
          // Arrange
          const network = createTrainingDropoutNetwork(52);
          network.train(XOR_DATASET, { iterations: 2, dropout: 0.5 });

          // Act
          network.test(XOR_DATASET);
          const hiddenMasks = getHiddenMasks(network);

          // Assert
          expect(hiddenMasks.every((mask) => mask === 1)).toBe(true);
        });
      });
    });

    describe('given train() uses dropout across several iterations', () => {
      describe('when activation calls are observed during training', () => {
        it('masks at least one hidden node during a training pass', () => {
          // Arrange
          const network = createTrainingDropoutNetwork(53);
          let maskedNodeSeen = false;
          const originalActivate = network.activate.bind(network);
          const activateSpy = jest
            .spyOn(network, 'activate')
            .mockImplementation((...activationArguments) => {
              const output = originalActivate(...activationArguments);

              if (
                activationArguments[1] === true &&
                getHiddenMasks(network).some((mask) => mask === 0)
              ) {
                maskedNodeSeen = true;
              }

              return output;
            });

          try {
            // Act
            network.train(XOR_DATASET, { iterations: 5, dropout: 0.5 });
          } finally {
            activateSpy.mockRestore();
          }

          // Assert
          expect(maskedNodeSeen).toBe(true);
        });
      });
    });

    describe('given one clone stays idle while its sibling trains with dropout', () => {
      describe('when the idle clone masks are inspected', () => {
        it('keeps every node mask at one', () => {
          // Arrange
          const trainedNetwork = createTrainingDropoutNetwork(54);
          const clonedNetwork = trainedNetwork.clone();

          // Act
          trainedNetwork.train([{ input: [0, 0], output: [0] }], {
            iterations: 5,
            dropout: 0.5,
          });
          const everyMaskIsOne = clonedNetwork.nodes.every(
            (node) => node.mask === 1,
          );

          // Assert
          expect(everyMaskIsOne).toBe(true);
        });
      });
    });
  });
});
