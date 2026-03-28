import Network from '../../src/architecture/network';

type InternalRand = () => number;

const invokeInternalRand = (network: Network): number => {
  const rand = Reflect.get(network, '_rand') as InternalRand;
  return rand.call(network);
};

/** Deterministic RNG utilities tests (AAA pattern & single expectations). */
describe('Network.deterministic RNG utilities', () => {
  describe('Scenario: reproducible seeding', () => {
    it('produces identical first sampled value for same seed', () => {
      // Arrange
      const netA = new Network(1, 1, { seed: 123, enforceAcyclic: true });
      const netB = new Network(1, 1, { seed: 123, enforceAcyclic: true });
      // Act
      const a1 = invokeInternalRand(netA);
      const b1 = invokeInternalRand(netB);
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
        invokeInternalRand(net);
        // Act
        const { state } = snap;
        if (typeof state !== 'number') {
          throw new Error(
            'Snapshot state should be numeric before restoration',
          );
        }
        net.setRNGState(state);
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
      const val = invokeInternalRand(net);
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
