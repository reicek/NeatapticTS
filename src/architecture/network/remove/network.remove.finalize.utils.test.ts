import { config } from '../../../config';
import Node from '../../node';
import { nodePoolStats, resetNodePool } from '../../nodePool/nodePool';
import Network from '../network';
import {
  markNetworkRemovalDirtyFlags,
  removeNodeFromNetworkStorage,
} from './network.remove.finalize.utils';
import type {
  NetworkRemoveProps,
  NodeRemovalContext,
} from './network.remove.utils.types';

function createRemovalContext(): NodeRemovalContext {
  const network = new Network(1, 1, { seed: 9701 });
  const randomSource = Reflect.get(network, '_rand') as () => number;
  const hiddenNode = new Node('hidden', undefined, randomSource);

  network.nodes.splice(1, 0, hiddenNode);

  return {
    internalNetwork: {},
    network,
    targetNode: hiddenNode,
    targetNodeIndex: 1,
  };
}

describe('network remove finalize utility chapter', () => {
  const originalEnableNodePooling = config.enableNodePooling ?? false;

  afterEach(() => {
    config.enableNodePooling = originalEnableNodePooling;
    resetNodePool();
  });

  describe('markNetworkRemovalDirtyFlags', () => {
    describe('given one mutable internal network flag bag', () => {
      describe('when the finalize helper marks removal dirty state', () => {
        it('sets every removal-sensitive dirty flag to true', () => {
          // Arrange
          const internalNetwork: NetworkRemoveProps = {};

          // Act
          markNetworkRemovalDirtyFlags(internalNetwork);

          // Assert
          expect(internalNetwork).toEqual({
            _adjDirty: true,
            _nodeIndexDirty: true,
            _slabDirty: true,
            _topoDirty: true,
          });
        });
      });
    });
  });

  describe('removeNodeFromNetworkStorage', () => {
    describe('given node pooling is enabled for one valid hidden-node removal', () => {
      describe('when the finalize helper removes the node from storage', () => {
        it('releases the removed node into the node pool', () => {
          // Arrange
          config.enableNodePooling = true;
          const removalContext = createRemovalContext();

          // Act
          removeNodeFromNetworkStorage(removalContext);

          // Assert
          expect(nodePoolStats().size).toBe(1);
        });
      });
    });
  });
});
