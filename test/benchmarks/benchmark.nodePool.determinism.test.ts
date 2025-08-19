import Network from '../../src/architecture/network';
import { config } from '../../src/config';

describe('benchmark.nodePool.determinism', () => {
  describe('forward parity pooling off vs on', () => {
    it('should produce identical outputs with same seed (pool off vs on)', () => {
      // Arrange
      const seed = 1337;
      (config as any).enableNodePooling = false;
      const netA = new Network(3, 2, { seed });
      (config as any).enableNodePooling = true;
      const netB = new Network(3, 2, { seed });
      const input = [0.25, -0.1, 0.9];

      // Act
      const outA = netA.activate(input.slice());
      const outB = netB.activate(input.slice());

      // Assert
      expect(outA).toEqual(outB);
    });
  });
});
