import Network from '../../src/architecture/network';

describe('Network.activateBatch', () => {
  describe('single-sample equivalence', () => {
    const net = new Network(2, 1, { seed: 42, enforceAcyclic: true });
    const x = [0.1, -0.2];
    const y1 = net.activate(x);
    const yb = net.activateBatch([x]);
    it('returns same output length as activate()', () => {
      expect(yb[0].length).toBe(y1.length);
    });
    it('returns same value for output[0]', () => {
      expect(yb[0][0]).toBe(y1[0]);
    });
  });

  describe('multi-sample size', () => {
    const net = new Network(3, 2, { seed: 7, enforceAcyclic: true });
    const batch = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const out = net.activateBatch(batch);
    it('matches batch length', () => {
      expect(out.length).toBe(batch.length);
    });
  });
});
