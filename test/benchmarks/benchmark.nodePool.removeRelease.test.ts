import Network from '../../src/architecture/network';
import { config } from '../../src/config';
import { nodePoolStats, resetNodePool } from '../../src/architecture/nodePool';

describe('benchmark.nodePool.removeRelease', () => {
  describe('remove() integration', () => {
    it('should recycle node into pool after removal when pooling enabled', () => {
      // Arrange
      resetNodePool();
      (config as any).enableNodePooling = true;
      const net = new Network(2, 1, { seed: 7, minHidden: 1 });
      const before = nodePoolStats().size;
      // Remove last hidden node if present (skip input/output)
      const hidden = net.nodes.find((n: any) => n.type === 'hidden');
      if (!hidden) throw new Error('Expected hidden node for test');
      net.remove(hidden);

      // Act
      const after = nodePoolStats().size;

      // Assert
      expect(after).toBeGreaterThan(before);
    });
  });
});
