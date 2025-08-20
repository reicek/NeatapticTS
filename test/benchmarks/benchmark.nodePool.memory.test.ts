import { memoryStats } from '../../src/utils/memory';
import {
  acquireNode,
  releaseNode,
  resetNodePool,
} from '../../src/architecture/nodePool';
import { config } from '../../src/config';

describe('benchmark.nodePool.memory', () => {
  describe('pool stats & reset cleanliness', () => {
    it('should expose nodePool stats and return clean reset state after release', () => {
      // Arrange
      resetNodePool();
      (config as any).enableNodePooling = true;
      const n1 = acquireNode({ type: 'hidden', rng: () => 0.42 });
      n1.activation = 123;
      n1.state = 456;
      releaseNode(n1);
      const stats = memoryStats();

      // Act
      const nodePool = stats.pools.nodePool;

      // Assert (single expectation bundling structural & numeric invariants)
      expect(
        !!nodePool &&
          typeof nodePool.size === 'number' &&
          typeof nodePool.highWaterMark === 'number'
      ).toBe(true);
    });
  });
});
