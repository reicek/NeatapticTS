import { NetworkTrainingOutputTargetLengthError } from './network.training.errors';
import { clearState, propagate } from './network.training.backprop.utils';
import type {
  CostDerivative,
  NetworkNode,
  OutputNodeWithCostDerivative,
} from './network.training.utils.types';
import type Network from '../../network/network';

describe('network training backprop utility chapter', () => {
  describe('propagate', () => {
    describe('given the target vector is missing', () => {
      it('throws the output-target-length error before propagating any node', () => {
        // Arrange
        const hiddenNode = createNetworkNode();
        const outputNode = createNetworkNode();
        const network = createNetworkFixture([hiddenNode, outputNode], 1, 1);

        // Act
        const propagateWithoutTarget = () => {
          propagate.call(
            network,
            0.1,
            0.2,
            true,
            undefined as unknown as number[],
          );
        };

        // Assert
        expect(propagateWithoutTarget).toThrow(
          NetworkTrainingOutputTargetLengthError,
        );
      });
    });

    describe('given one hidden node and one output node without a cost derivative override', () => {
      it('propagates the output node with the default regularization and then propagates the hidden node without a target', () => {
        // Arrange
        const inputNode = createNetworkNode();
        const hiddenNode = createNetworkNode();
        const outputNode = createNetworkNode();
        const network = createNetworkFixture(
          [inputNode, hiddenNode, outputNode],
          1,
          1,
        );

        // Act
        propagate.call(network, 0.1, 0.2, true, [0.9]);

        // Assert
        expect({
          outputCalls: outputNode.propagate.mock.calls,
          hiddenCalls: hiddenNode.propagate.mock.calls,
        }).toEqual({
          outputCalls: [[0.1, 0.2, true, 0, 0.9]],
          hiddenCalls: [[0.1, 0.2, true, 0]],
        });
      });
    });

    describe('given a custom cost derivative override is supplied for the output layer', () => {
      it('passes the override into the output-node propagate call while leaving hidden-node propagation unchanged', () => {
        // Arrange
        const inputNode = createNetworkNode();
        const hiddenNode = createNetworkNode();
        const outputNode = createOutputNodeWithCostDerivative();
        const network = createNetworkFixture(
          [inputNode, hiddenNode, outputNode],
          1,
          1,
        );
        const costDerivative: CostDerivative = (target, output) => target - output;

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
    describe('given the network holds multiple nodes with runtime state', () => {
      it('clears each node in order', () => {
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
  output: number,
  input: number,
): Network {
  return {
    nodes,
    output,
    input,
  } as unknown as Network;
}