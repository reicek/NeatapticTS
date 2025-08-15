import Network from '../../src/architecture/network';

// Custom cost patterns to exercise smoothing & patience logic.

describe('Training Early Stopping Extensions', () => {
  describe('Scenario: movingAverageWindow > 1 smooths oscillating error', () => {
    const net = new Network(1, 1);
    const rawErrors: number[] = Array.from({ length: 20 }, (_, i) =>
      i % 2 === 0 ? 0.5 : 0.7
    );
    let idx = 0;
    const cost = () => rawErrors[Math.min(idx++, rawErrors.length - 1)];
    let result: any;
    beforeAll(() => {
      result = net.train([{ input: [0], output: [0] }], {
        iterations: 20,
        rate: 0.1,
        cost,
        movingAverageWindow: 4,
        error: 0.4, // target low enough so loop governed by iterations
        optimizer: 'sgd',
      });
    });
    it('performs all iterations when smoothed error stays above target', () => {
      expect(result.iterations).toBe(20);
    });
  });

  describe('Scenario: EMA smoothing behaves consistently', () => {
    const net = new Network(1, 1);
    const errs = [0.9, 0.8, 0.7, 0.6, 0.5];
    let i = 0;
    const cost = () => errs[Math.min(i++, errs.length - 1)];
    let result: any;
    beforeAll(() => {
      result = net.train([{ input: [0], output: [0] }], {
        iterations: 5,
        rate: 0.1,
        cost,
        movingAverageWindow: 5,
        movingAverageType: 'ema',
        error: 0.4,
        optimizer: 'sgd',
      });
    });
    it('reports final error lower than initial raw error due to EMA progression', () => {
      expect(result.error).toBeLessThan(0.9);
    });
  });

  describe('Scenario: earlyStopPatience triggers stop when no improvement', () => {
    const net = new Network(1, 1);
    const cost = () => 0.4;
    let result: any;
    beforeAll(() => {
      result = net.train([{ input: [0], output: [0] }], {
        iterations: 50,
        rate: 0.1,
        cost,
        earlyStopPatience: 3,
        error: 0.0,
        optimizer: 'sgd',
      });
    });
    it('stops before reaching max iterations due to earlyStopPatience', () => {
      expect(result.iterations).toBeLessThan(50);
    });
  });

  describe('Scenario: improvement resets earlyStopPatience', () => {
    const net = new Network(1, 1);
    const seq = [0.5, 0.49, 0.49, 0.48, 0.48, 0.47];
    let i = 0;
    const cost = () => seq[Math.min(i++, seq.length - 1)];
    let result: any;
    beforeAll(() => {
      result = net.train([{ input: [0], output: [0] }], {
        iterations: 30,
        rate: 0.1,
        cost,
        earlyStopPatience: 2,
        earlyStopMinDelta: 0.0005,
        error: 0.0,
        optimizer: 'sgd',
      });
    });
    it('runs past initial patience window because improvements occurred', () => {
      expect(result.iterations).toBeGreaterThan(4);
    });
  });
});
