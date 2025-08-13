import Network from '../../src/architecture/network';

/**
 * fastSlabActivate fallback branch when adjacency prerequisites missing (after manual corruption).
 * We simulate by nulling internal slab arrays post-build so guard triggers and generic path is used.
 */
describe('Network.fastSlabActivate prerequisites', () => {
  describe('Scenario: missing adjacency arrays triggers generic activate fallback', () => {
    it('returns output of expected length via standard activate', () => {
      // Arrange
      const net = new Network(2, 2, { seed: 22, enforceAcyclic: true });
      // Build slab once so fields exist, then corrupt them.
      (net as any).rebuildConnectionSlab(true);
      (net as any)._connWeights = null; // force missing prerequisite guard
      const input = [0.9, 0.1];
      // Act
      const out = (net as any)._fastSlabActivate(input); // call internal fast path wrapper
      // Assert
      expect(out.length).toBe(2);
    });
  });
});
