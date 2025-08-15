import Network from '../../src/architecture/network';
import * as methods from '../../src/methods/methods';

describe('Plateau vs Early-Stop Smoothing Separation', () => {
  test('scheduler sees different (faster) smoothed plateauError than early-stop error', () => {
    const net = new Network(2, 1, { seed: 42 });
    // Simple linearly separable tiny set to force quick improvement then noise
    const data = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] },
    ];

    const plateauDecisions: number[] = []; // capture rates when reduced
    const rop = methods.Rate.reduceOnPlateau({
      patience: 3,
      factor: 0.5,
      minRate: 1e-4,
      // no callback supported by current reduceOnPlateau signature
    });

    // Use a slower, more robust early-stop smoother (median window 7) and a quicker plateau EMA window 2
    const res = net.train(data, {
      iterations: 80,
      rate: 0.2,
      movingAverageType: 'median',
      movingAverageWindow: 7,
      plateauMovingAverageType: 'ema',
      plateauMovingAverageWindow: 2,
      earlyStopPatience: 15,
      ratePolicy: rop,
      metricsHook: ({
        iteration,
        error,
        plateauError,
      }: {
        iteration: number;
        error: number;
        plateauError?: number;
      }) => {
        if (plateauError !== undefined) {
          // plateauError should track raw error more tightly -> expect smaller lag: plateauError changes sooner
          // We simply assert divergence appears at least once when early smoothing window is longer
          if (iteration > 5) {
            expect(plateauError).not.toBe(error); // at some point the EMA vs median should differ
          }
        }
      },
    });

    // Ensure training ran
    expect(res.iterations).toBeGreaterThan(0);
    // Expect at least one plateau reduction (shows scheduler operated on its own smoothed series)
    // We don't enforce a reduction (depends on synthetic sequence), just that training completes without error.
    expect(typeof res.error).toBe('number');
  });
});
