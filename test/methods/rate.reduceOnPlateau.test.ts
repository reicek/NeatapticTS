import Rate from '../../src/methods/rate';

describe('Rate.reduceOnPlateau', () => {
  const base = 0.05;
  const rop = Rate.reduceOnPlateau({
    patience: 2,
    factor: 0.5,
    minRate: 0.005,
  });

  // simulate sequence of errors
  const seq = [0.5, 0.45, 0.46, 0.47, 0.4, 0.41, 0.41, 0.39];
  const rates: number[] = [];
  seq.forEach((err, i) => {
    rates.push(rop(base, i, err));
  });

  describe('Initial rate (iter 0)', () => {
    const value = rates[0];
    it('starts at base', () => {
      expect(value).toBeCloseTo(base, 10);
    });
  });

  describe('No reduction before patience (iter 1)', () => {
    const value = rates[1];
    it('unchanged early', () => {
      expect(value).toBe(base);
    });
  });

  describe('Reduction triggered after patience (iter 3)', () => {
    const value = rates[3];
    it('reduced after plateau', () => {
      expect(value).toBeLessThan(base);
    });
  });

  describe('Further improvement resets wait (iter 4)', () => {
    const value = rates[4];
    it('stays at reduced rate until new plateau', () => {
      expect(value).toBe(rates[3]);
    });
  });
});
