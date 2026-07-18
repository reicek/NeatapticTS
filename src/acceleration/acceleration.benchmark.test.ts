/**
 * Red tests for the generic acceleration regression guard.
 *
 * `runRegressionBenchmark` runs a paired CPU vs GPU micro-benchmark on a minimal
 * network surface and blacklists the GPU backend when its median is slower than
 * CPU by a configured ratio threshold. `isGpuBlacklisted` exposes the current
 * blacklist state, and `clearBlacklist` resets it so tests can be deterministic.
 *
 * These tests import from `./acceleration.benchmark`, which does not exist yet,
 * so the suite fails with TS2307 until the implementation slice lands.
 */

import {
  runRegressionBenchmark,
  isGpuBlacklisted,
  clearBlacklist,
} from './acceleration.benchmark';
import * as accelerationBarrel from './index';

/**
 * Build a deterministic `now()` mock that advances by a fixed tick each call.
 *
 * @param start - Initial timestamp value.
 * @param tick - Amount to advance on each call.
 */
const makeIncrementingNow = (start = 0, tick = 1) => {
  let value = start;
  return () => {
    value += tick;
    return value;
  };
};

afterEach(() => {
  jest.restoreAllMocks();
  clearBlacklist();
});

describe('acceleration.benchmark', () => {
  describe('module exports', () => {
    it('exports a regression benchmark function', () => {
      expect(typeof runRegressionBenchmark).toBe('function');
    });

    it('exports a GPU blacklist predicate', () => {
      expect(typeof isGpuBlacklisted).toBe('function');
    });

    it('exports a reset function for tests', () => {
      expect(typeof clearBlacklist).toBe('function');
    });
  });

  describe('runRegressionBenchmark', () => {
    it('returns benchmark results containing cpu and gpu medians', () => {
      const result = runRegressionBenchmark({
        seed: 1,
        now: makeIncrementingNow(),
      });

      expect(result.results).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            backend: 'cpu',
            medianMs: expect.any(Number),
          }),
          expect.objectContaining({
            backend: 'gpu',
            medianMs: expect.any(Number),
          }),
        ]),
      );
    });

    it('blacklists gpu when gpu median exceeds cpu median by the configured ratio', () => {
      runRegressionBenchmark({
        seed: 1,
        now: makeIncrementingNow(),
        ratioThreshold: 2,
      });

      expect(isGpuBlacklisted()).toBe(true);
    });

    it('does not blacklist gpu when gpu median is within the ratio threshold', () => {
      runRegressionBenchmark({
        seed: 1,
        now: makeIncrementingNow(),
        ratioThreshold: 10,
      });

      expect(isGpuBlacklisted()).toBe(false);
    });

    it('uses deterministic sample selection for the benchmark', () => {
      const first = runRegressionBenchmark({
        seed: 42,
        now: makeIncrementingNow(0, 1),
      });
      const second = runRegressionBenchmark({
        seed: 42,
        now: makeIncrementingNow(0, 1),
      });

      expect(first.results).toEqual(second.results);
    });

    it('emits a fallback event when gpu is blacklisted', () => {
      const observer = { onFallback: jest.fn() };

      runRegressionBenchmark({
        seed: 1,
        now: makeIncrementingNow(),
        ratioThreshold: 2,
        observer,
      });

      expect(observer.onFallback).toHaveBeenCalled();
    });

    it('clears the GPU blacklist state', () => {
      runRegressionBenchmark({
        seed: 1,
        now: makeIncrementingNow(),
        ratioThreshold: 2,
      });
      clearBlacklist();

      expect(isGpuBlacklisted()).toBe(false);
    });

    it('uses the default now provider when none is supplied', () => {
      const result = runRegressionBenchmark({ seed: 1 });

      expect(result.results).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            backend: 'cpu',
            medianMs: expect.any(Number),
          }),
          expect.objectContaining({
            backend: 'gpu',
            medianMs: expect.any(Number),
          }),
        ]),
      );
    });
  });

  describe('acceleration barrel', () => {
    it('does not re-export benchmark internals from the acceleration barrel', () => {
      expect(
        (accelerationBarrel as Record<string, unknown>).runRegressionBenchmark,
      ).toBeUndefined();
    });
  });
});
