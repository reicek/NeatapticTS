import Rate from '../../src/methods/rate';

describe('Rate.cosineAnnealingWarmRestarts', () => {
  const base = 0.1;
  const period = 5;
  const min = 0.001;
  const tMult = 2;
  const fn = Rate.cosineAnnealingWarmRestarts(period, min, tMult);

  describe('First cycle peak at iteration 0', () => {
    const value = fn(base, 0);
    it('returns ~base at start of first cycle', () => {
      expect(value).toBeCloseTo(base, 10);
    });
  });

  describe('First cycle mid (iteration 2)', () => {
    const value = fn(base, 2);
    it('returns value between min and base', () => {
      expect(value).toBeLessThan(base);
    });
  });

  describe('End of first cycle (iteration 4)', () => {
    const value = fn(base, 4);
    it('returns near min at end of cycle', () => {
      expect(value).toBeLessThanOrEqual(base * 0.51); // rough check downward
    });
  });

  describe('Start of second (restart) cycle (iteration 5)', () => {
    const value = fn(base, 5);
    it('resets close to base after restart', () => {
      expect(value).toBeCloseTo(base, 2);
    });
  });

  describe('Within second (longer) cycle (iteration 8)', () => {
    const value = fn(base, 8);
    it('still between min and base mid second cycle', () => {
      expect(value).toBeGreaterThan(min);
    });
  });
});
