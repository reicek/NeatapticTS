jest.mock('./network.training.loop.utils', () => ({
  trainSetCore: jest.fn(() => 0.75),
}));

import Network from '../../network';
import { trainSetCore } from './network.training.loop.utils';
import {
  clearState,
  propagate,
  trainSetImpl,
} from './network.training.utils';
import type {
  CostDerivative,
  NetworkNode,
  OutputNodeWithCostDerivative,
} from './network.training.utils.types';

const mockedTrainSetCore = jest.mocked(trainSetCore);

describe('network training direct utility wrappers', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('propagate', () => {
    describe('given the training barrel re-exports one output-layer backpropagation call', () => {
      it('forwards the propagated arguments through the barrel export', () => {
        // Arrange
        const inputNode = createNetworkNode();
        const hiddenNode = createNetworkNode();
        const outputNode = createOutputNodeWithCostDerivative();
        const network = createNetworkFixture(
          [inputNode, hiddenNode, outputNode],
          1,
          1,
        );
        const costDerivative: CostDerivative = (target, output) =>
          target - output;

        // Act
        propagate.call(network, 0.3, 0.4, false, [0.75], 0.5, costDerivative);

        // Assert
        expect({
          outputCalls: outputNode.propagate.mock.calls,
          hiddenCalls: hiddenNode.propagate.mock.calls,
        }).toEqual({
          outputCalls: [[0.3, 0.4, false, 0.5, 0.75, costDerivative]],
          hiddenCalls: [[0.3, 0.4, false, 0.5]],
        });
      });
    });
  });

  describe('clearState', () => {
    describe('given the training barrel re-exports one runtime-state reset', () => {
      it('clears each node through the barrel export', () => {
        // Arrange
        const firstNode = createNetworkNode();
        const secondNode = createNetworkNode();
        const network = createNetworkFixture([firstNode, secondNode], 1, 0);

        // Act
        clearState.call(network);

        // Assert
        expect({
          firstClearCalls: firstNode.clear.mock.calls.length,
          secondClearCalls: secondNode.clear.mock.calls.length,
        }).toEqual({
          firstClearCalls: 1,
          secondClearCalls: 1,
        });
      });
    });
  });

  describe('trainSetImpl', () => {
    describe('given one normalized epoch-training call reaches the wrapper', () => {
      it('delegates the full call frame to the loop helper and returns its mean cost', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 1_601 });
        const trainingSet = [
          {
            input: [1],
            output: [0],
          },
        ];
        const regularization = {
          l1: 0,
          l2: 0.001,
        };
        const costFunction = (target: number[], output: number[]) =>
          Math.abs((target[0] ?? 0) - (output[0] ?? 0));
        const optimizer = {
          type: 'adam' as const,
        };

        // Act
        const meanCost = trainSetImpl(
          network,
          trainingSet,
          4,
          2,
          0.3,
          0.1,
          regularization,
          costFunction,
          optimizer,
        );

        // Assert
        expect({
          args: mockedTrainSetCore.mock.calls[0],
          meanCost,
        }).toEqual({
          args: [
            network,
            trainingSet,
            4,
            2,
            0.3,
            0.1,
            regularization,
            costFunction,
            optimizer,
          ],
          meanCost: 0.75,
        });
      });
    });
  });
});

type MockedNetworkNode = NetworkNode & {
  clear: jest.Mock<void, []>;
  propagate: jest.Mock<void, unknown[]>;
};

function createNetworkNode(): MockedNetworkNode {
  return {
    clear: jest.fn(),
    propagate: jest.fn(),
  } as unknown as MockedNetworkNode;
}

function createOutputNodeWithCostDerivative(): MockedNetworkNode & OutputNodeWithCostDerivative {
  return createNetworkNode() as MockedNetworkNode & OutputNodeWithCostDerivative;
}

function createNetworkFixture(
  nodes: NetworkNode[],
  input: number,
  output: number,
): Network {
  return {
    nodes,
    input,
    output,
  } as unknown as Network;
}