import Network from '../../src/architecture/network';

/** Deterministic RNG utilities tests (AAA pattern & single expectations). */
describe('Network.deterministic RNG utilities', () => {
  describe('Scenario: reproducible seeding', () => {
    it('produces identical first sampled value for same seed', () => {
      // Arrange
      const netA = new Network(1, 1, { seed: 123, enforceAcyclic: true });
      const netB = new Network(1, 1, { seed: 123, enforceAcyclic: true });
      // Act
      const a1 = (netA as any)._rand();
      const b1 = (netB as any)._rand();
      // Assert
      expect(a1).toBe(b1);
    });
  });

  describe('Scenario: snapshot & restore raw state', () => {
    it('snapshot contains numeric state', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 9, enforceAcyclic: true });
      // Act
      const snap = net.snapshotRNG();
      // Assert
      expect(typeof snap.state).toBe('number');
    });
    describe('after advancing RNG and restoring exact state word', () => {
      it('raw state matches original snapshot after setRNGState', () => {
        // Arrange
        const net = new Network(1, 1, { seed: 9, enforceAcyclic: true });
        const snap = net.snapshotRNG();
        (net as any)._rand();
        // Act
        net.setRNGState(snap.state as number);
        const roundTripped = net.getRNGState();
        // Assert
        expect(roundTripped).toBe(snap.state);
      });
    });
  });

  describe('Scenario: restoreRNG custom implementation', () => {
    it('uses injected custom RNG function', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 42, enforceAcyclic: true });
      // Act
      net.restoreRNG(() => 0.5);
      const val = (net as any)._rand();
      // Assert
      expect(val).toBe(0.5);
    });
    describe('after restoreRNG call, internal numeric state cleared', () => {
      it('rng state becomes undefined after restoreRNG', () => {
        // Arrange
        const net = new Network(1, 1, { seed: 42, enforceAcyclic: true });
        net.restoreRNG(() => 0.5);
        // Act
        const state = net.getRNGState();
        // Assert
        expect(state).toBeUndefined();
      });
    });
  });
});
