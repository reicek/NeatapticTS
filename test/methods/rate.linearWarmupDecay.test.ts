import Rate from '../../src/methods/rate';

describe('Rate.linearWarmupDecay', () => {
  const base = 0.2;
  const fn = Rate.linearWarmupDecay(20, 5, 0.01);

  describe('Warmup start (iter 0)', () => {
    const value = fn(base, 0);
    it('starts near 0', () => { expect(value).toBeCloseTo(0, 5); });
  });

  describe('Warmup mid (iter 3)', () => {
    const value = fn(base, 3);
    it('increases during warmup', () => { expect(value).toBeGreaterThan(0); });
  });

  describe('End warmup (iter 5)', () => {
    const value = fn(base, 5);
    it('reaches base at end warmup', () => { expect(value).toBeCloseTo(base, 5); });
  });

  describe('Decay mid (iter 10)', () => {
    const value = fn(base, 10);
    it('decays after warmup', () => { expect(value).toBeLessThan(base); });
  });

  describe('Final step (iter 20)', () => {
    const value = fn(base, 20);
    it('hits endRate at totalSteps', () => { expect(value).toBeCloseTo(0.01, 5); });
  });
});
