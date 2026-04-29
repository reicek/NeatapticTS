import {
  acquireNode,
  nodePoolStats,
  releaseNode,
  resetNodePool,
} from './nodePool';

describe('nodePool', () => {
  afterEach(() => {
    resetNodePool();
  });

  describe('acquireNode()', () => {
    describe('given a requested hidden node', () => {
      describe('when acquiring a fresh node', () => {
        it('starts with zero activation', () => {
          // Arrange
          const acquiredNode = acquireNode({ type: 'hidden' });

          // Act
          const actualActivation = acquiredNode.activation;

          // Assert
          expect(actualActivation).toBe(0);
        });
      });
    });

    describe('given a requested input node', () => {
      describe('when acquiring a fresh node', () => {
        it('uses the requested type', () => {
          // Arrange
          const acquiredNode = acquireNode({ type: 'input' });

          // Act
          const actualType = acquiredNode.type;

          // Assert
          expect(actualType).toBe('input');
        });
      });
    });
  });

  describe('releaseNode() plus acquireNode()', () => {
    describe('given a recycled node with mutated runtime state', () => {
      describe('when re-acquiring it for a new lifecycle', () => {
        it('resets activation back to zero', () => {
          // Arrange
          const firstNode = acquireNode({ type: 'hidden' });
          firstNode.activation = 42;
          releaseNode(firstNode);

          // Act
          const recycledNode = acquireNode({ type: 'output' });

          // Assert
          expect(recycledNode.activation).toBe(0);
        });

        it('applies the newly requested type', () => {
          // Arrange
          const firstNode = acquireNode({ type: 'hidden' });
          releaseNode(firstNode);

          // Act
          const recycledNode = acquireNode({ type: 'output' });

          // Assert
          expect(recycledNode.type).toBe('output');
        });

        it('applies the requested activation function', () => {
          // Arrange
          const firstNode = acquireNode({ type: 'hidden' });
          releaseNode(firstNode);
          const requestedActivationFunction = (value: number): number => value;

          // Act
          const recycledNode = acquireNode({
            type: 'output',
            activationFn: requestedActivationFunction,
          });

          // Assert
          expect(recycledNode.squash).toBe(requestedActivationFunction);
        });

        it('sets bias to 0 when recycled as an input node', () => {
          // Arrange
          const firstNode = acquireNode({ type: 'hidden' });
          releaseNode(firstNode);

          // Act
          const recycledNode = acquireNode({ type: 'input' });

          // Assert
          expect(recycledNode.bias).toBe(0);
        });
      });
    });

    describe('given a pool whose high-water mark has already been reached', () => {
      describe('when releasing a node back to the prior peak size', () => {
        it('does not raise the high-water mark', () => {
          // Arrange – fill pool to highWaterMark = 2 then drain one slot
          const nodeA = acquireNode({ type: 'hidden' });
          const nodeB = acquireNode({ type: 'hidden' });
          releaseNode(nodeA);
          releaseNode(nodeB);
          // Pool is now at peak (size 2). Drain one.
          const recycled = acquireNode({ type: 'hidden' });

          // Act – releasing back to size 2 must NOT exceed highWaterMark
          releaseNode(recycled);

          // Assert – stats still reflect peak of 2
          const actualStats = nodePoolStats();
          expect(actualStats.highWaterMark).toBe(2);
        });
      });
    });
  });

  describe('nodePoolStats()', () => {
    describe('given one released node', () => {
      describe('when reading the pool stats', () => {
        it('reports one retained pooled node', () => {
          // Arrange
          const acquiredNode = acquireNode();
          releaseNode(acquiredNode);

          // Act
          const actualStats = nodePoolStats();

          // Assert
          expect(actualStats.size).toBe(1);
        });
      });
    });

    describe('given no nodes have been acquired since pool reset', () => {
      describe('when reading the pool stats', () => {
        it('returns 0 for recycled ratio', () => {
          // Arrange
          resetNodePool();

          // Act
          const actualStats = nodePoolStats();

          // Assert
          expect(actualStats.recycledRatio).toBe(0);
        });
      });
    });
  });
});
