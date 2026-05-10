import { config } from '../../../config';
import {
  acquireNode,
  nodePoolStats,
  releaseNode,
  resetNodePool,
} from '../../nodePool/nodePool';
import Network from '../network';
import { addNodeBetweenImpl } from './network.mutate.public.utils';

function countHiddenNodes(network: Network): number {
  return network.nodes.filter(
    (candidateNode) => candidateNode.type === 'hidden',
  ).length;
}

function createSingleConnectionNetwork(): Network {
  return new Network(1, 1, { seed: 7401 });
}

describe('network mutate public utility chapter', () => {
  const originalEnableNodePooling = config.enableNodePooling ?? false;

  afterEach(() => {
    config.enableNodePooling = originalEnableNodePooling;
    resetNodePool();
  });

  describe('addNodeBetweenImpl', () => {
    describe('given the sparsity budget cannot free enough room for one more split edge', () => {
      it('returns without adding a hidden node or changing the connection count', () => {
        // Arrange
        const network = createSingleConnectionNetwork();
        network.configureSparsityBudget({ maxConnections: 1 });
        const connectionCountBeforeMutation = network.connections.length;

        // Act
        addNodeBetweenImpl.call(network);
        const budgetSnapshot = network.getSparsityBudgetSnapshot();

        // Assert
        expect({
          connectionCount: network.connections.length,
          decision: budgetSnapshot?.decision,
          hiddenCount: countHiddenNodes(network),
          retainedOriginalCount:
            network.connections.length === connectionCountBeforeMutation,
        }).toEqual({
          connectionCount: 1,
          decision: 'deny',
          hiddenCount: 0,
          retainedOriginalCount: true,
        });
      });
    });

    describe('given the network has no remaining connections', () => {
      it('returns without adding a hidden node', () => {
        // Arrange
        const network = createSingleConnectionNetwork();
        const inputNode = network.nodes.find(
          (candidateNode) => candidateNode.type === 'input',
        );
        const outputNode = network.nodes.find(
          (candidateNode) => candidateNode.type === 'output',
        );

        if (!inputNode || !outputNode) {
          throw new Error('Expected one input node and one output node.');
        }

        network.disconnect(inputNode, outputNode);

        // Act
        addNodeBetweenImpl.call(network);

        // Assert
        expect(countHiddenNodes(network)).toBe(0);
      });
    });

    describe('given the random selector points past the available connection list', () => {
      it('returns without adding a hidden node', () => {
        // Arrange
        const network = createSingleConnectionNetwork();
        Reflect.set(network, '_rand', () => 1);

        // Act
        addNodeBetweenImpl.call(network);

        // Assert
        expect(countHiddenNodes(network)).toBe(0);
      });
    });

    describe('given node pooling is enabled and one pooled hidden node is available', () => {
      it('reuses a pooled hidden node for the inserted split node', () => {
        // Arrange
        config.enableNodePooling = true;
        const recycledNode = acquireNode({ type: 'hidden', rng: () => 0.5 });
        releaseNode(recycledNode);
        const network = createSingleConnectionNetwork();

        // Act
        addNodeBetweenImpl.call(network);

        // Assert
        expect(nodePoolStats().reused).toBe(1);
      });
    });

    describe('given node pooling is disabled for one valid split mutation', () => {
      it('adds one hidden node through the direct constructor path', () => {
        // Arrange
        config.enableNodePooling = false;
        const network = createSingleConnectionNetwork();

        // Act
        addNodeBetweenImpl.call(network);

        // Assert
        expect(countHiddenNodes(network)).toBe(1);
      });
    });
  });
});
