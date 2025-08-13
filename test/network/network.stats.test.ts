import Network from '../../src/architecture/network';

/** Regularization stats accessor tests (AAA pattern). */
describe('Network.stats.getRegularizationStats', () => {
  describe('Scenario: stats never recorded', () => {
    it('returns null when no stats present', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 1, enforceAcyclic: true });
      // Act
      const result = net.getRegularizationStats();
      // Assert
      expect(result).toBeNull();
    });
  });

  describe('Scenario: stats object is present', () => {
    it('returns a cloned object (not the same reference)', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 2, enforceAcyclic: true });
      const internalStats = {
        l1Penalty: 0.1,
        dropped: 0.25,
        custom: { depth: 3 },
      };
      (net as any)._lastStats = internalStats;
      // Act
      const snap = net.getRegularizationStats();
      // Assert
      expect(snap === internalStats).toBe(false);
    });
    describe('when mutating the returned clone', () => {
      it('does not reflect external deep mutation (defensive copy)', () => {
        // Arrange
        const net = new Network(1, 1, { seed: 3, enforceAcyclic: true });
        (net as any)._lastStats = { custom: { depth: 3 } };
        const snap = net.getRegularizationStats();
        // Act
        (snap as any).custom.depth = 99;
        const reread = net.getRegularizationStats();
        // Assert
        expect((reread as any).custom.depth).toBe(3);
      });
    });
  });
});
