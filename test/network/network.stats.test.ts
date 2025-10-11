import Network from '../../src/architecture/network';

interface RegularizationStatsSnapshot {
  l1Penalty?: number;
  dropped?: number;
  custom?: { depth: number };
  [key: string]: unknown;
}

const setRegularizationStats = (
  net: Network,
  stats: RegularizationStatsSnapshot | null
) => {
  Reflect.set(net, '_lastStats', stats);
};

const getRegularizationStatsSnapshot = (
  net: Network
): RegularizationStatsSnapshot | null =>
  net.getRegularizationStats() as RegularizationStatsSnapshot | null;

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
      const internalStats: RegularizationStatsSnapshot = {
        l1Penalty: 0.1,
        dropped: 0.25,
        custom: { depth: 3 },
      };
      setRegularizationStats(net, internalStats);
      // Act
      const snap = net.getRegularizationStats();
      // Assert
      expect(snap === internalStats).toBe(false);
    });
    describe('when mutating the returned clone', () => {
      it('does not reflect external deep mutation (defensive copy)', () => {
        // Arrange
        const net = new Network(1, 1, { seed: 3, enforceAcyclic: true });
        setRegularizationStats(net, { custom: { depth: 3 } });
        const snap = getRegularizationStatsSnapshot(net);
        // Act
        if (!snap || !snap.custom) {
          throw new Error('Regularization stats should include custom depth');
        }
        snap.custom.depth = 99;
        const reread = getRegularizationStatsSnapshot(net);
        // Assert
        if (!reread || !reread.custom) {
          throw new Error('Regularization stats should include custom depth');
        }
        expect(reread.custom.depth).toBe(3);
      });
    });
  });
});
