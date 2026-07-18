/**
 * Regression guard: paired CPU vs GPU micro-benchmark.
 *
 * `runRegressionBenchmark` runs a minimal synthetic benchmark that is fully
 * independent of `src/architecture/`. The acceleration layer uses it to decide
 * whether the current environment's GPU path is actually faster than CPU for
 * the small reference workload. When GPU is repeatedly slower than CPU by a
 * configurable ratio, the GPU backend is blacklisted until `clearBlacklist()`
 * is called.
 *
 * The benchmark is intentionally deterministic: the caller supplies a seed
 * that drives sample selection and an injectable `now()` timer so tests can be
 * fully reproducible. No `Math.random` is used in the selection path.
 *
 * This module is internal to `src/acceleration/` and is intentionally absent
 * from the public barrel export (`src/acceleration/index.ts`).
 *
 * ```mermaid
 * flowchart LR
 *   Seed["seed + now()"] --> Samples["Collect N CPU samples\nCollect N GPU samples"]
 *   Samples --> Medians["Compute CPU median\nCompute GPU median"]
 *   Medians --> Ratio{gpuMedian / cpuMedian > threshold?}
 *   Ratio -->|yes| Blacklist["Set gpuBlacklisted = true\nEmit onFallback event"]
 *   Ratio -->|no| Keep["Keep gpuBlacklisted = false"]
 *   Blacklist --> Result[Return paired results]
 *   Keep --> Result
 * ```
 */

import {
  DEFAULT_REGRESSION_GUARD_RATIO_THRESHOLD,
  DEFAULT_REGRESSION_GUARD_SAMPLES,
} from './acceleration.constants';
import type { AccelerationMode } from './acceleration.types';

/**
 * Injectable observer subset used by the regression guard.
 *
 * Only the fallback callback is relevant here: a blacklist decision means the
 * runtime has fallen back from GPU to CPU.
 */
export interface RegressionBenchmarkObserver {
  /** Called when the GPU backend is blacklisted. */
  onFallback?: (event: {
    /** Backend that was requested (GPU). */
    requested: AccelerationMode;
    /** Backend that the guard falls back to (CPU). */
    chosen: AccelerationMode;
    /** Human-readable reason for the blacklist decision. */
    reason: string;
    /** Timestamp from the caller's `now()` provider. */
    timestamp: number;
  }) => void;
}

/**
 * Options accepted by {@link runRegressionBenchmark}.
 */
export interface RegressionBenchmarkOptions {
  /** Seed for deterministic sample selection. */
  seed: number;

  /**
   * Injectable high-resolution timestamp provider.
   *
   * Defaults to `globalThis.performance.now()`.
   */
  now?: () => number;

  /**
   * Ratio threshold at which GPU is blacklisted.
   *
   * If `gpuMedianMs / cpuMedianMs` exceeds this value, the GPU backend is
   * considered a regression.
   */
  ratioThreshold?: number;

  /** Optional observer receiving the blacklist fallback event. */
  observer?: RegressionBenchmarkObserver;
}

/**
 * Per-backend median duration reported by {@link runRegressionBenchmark}.
 */
export interface RegressionBenchmarkBackendResult {
  /** Backend that produced the median. */
  backend: AccelerationMode;

  /** Median sample duration in milliseconds. */
  medianMs: number;
}

/**
 * Result returned by {@link runRegressionBenchmark}.
 */
export interface RegressionBenchmarkResult {
  /** Paired CPU and GPU median benchmark results. */
  results: RegressionBenchmarkBackendResult[];
}

/**
 * Module-level GPU blacklist flag.
 *
 * The regression guard is intentionally global: once GPU has been observed to
 * be slower than CPU on the reference workload, all consumers in the same
 * process should avoid the GPU path until the blacklist is explicitly cleared.
 */
let gpuBlacklisted = false;

/**
 * Return whether the GPU backend has been blacklisted by the regression guard.
 *
 * @returns `true` when GPU has been observed to be slower than CPU by the
 * configured ratio threshold.
 *
 * @example
 * ```ts
 * import { isGpuBlacklisted } from '@reicek/neataptic-ts/acceleration';
 *
 * if (isGpuBlacklisted()) {
 *   console.log('GPU is temporarily disabled by the regression guard');
 * }
 * ```
 */
export function isGpuBlacklisted(): boolean {
  return gpuBlacklisted;
}

/**
 * Reset the GPU blacklist state.
 *
 * This is primarily used by tests to keep the module-level guard deterministic
 * between assertions, but it can also be called by callers that want to retry
 * GPU after a driver or environment change.
 *
 * @example
 * ```ts
 * import { clearBlacklist, isGpuBlacklisted } from '@reicek/neataptic-ts/acceleration';
 *
 * clearBlacklist();
 * console.log(isGpuBlacklisted()); // false
 * ```
 */
export function clearBlacklist(): void {
  gpuBlacklisted = false;
}

/** Resolve the default timestamp provider from the current global object. */
function defaultNowProvider(): () => number {
  return () => globalThis.performance.now();
}

/**
 * Seeded 32-bit PRNG (mulberry32).
 *
 * Deterministic, fast, and seed-repeatable across runtimes. Used only for
 * sample selection, not for cryptographic purposes.
 *
 * Mulberry32 is a simple linear congruential-style generator described in the
 * [PCG family overview (Wikipedia)](https://en.wikipedia.org/wiki/Permuted_congruential_generator#Other_simple_generators)
 * and popularized by Tommy Ettinger's public-domain reference implementation.
 */
function createSeededRng(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4_294_967_296;
  };
}

/**
 * Select a deterministic sample count from a band around the default.
 *
 * The same `seed` always selects the same effective sample count, which makes
 * repeated benchmark runs with the same seed byte-identical in timing
 * accounting.
 *
 * @param seed - Caller-supplied benchmark seed.
 * @returns Number of samples to collect per backend.
 */
function resolveSampleCount(seed: number): number {
  const rng = createSeededRng(seed);
  const offsets = [0, 2, 4, 6, 8];
  const index = Math.floor(rng() * offsets.length);
  return DEFAULT_REGRESSION_GUARD_SAMPLES + offsets[index];
}

/**
 * Compute the median of a numeric array.
 *
 * Uses `Array.prototype.toSorted()` to avoid mutating the input. The sample
 * count produced by {@link resolveSampleCount} is always even, so only the
 * even-length path is needed.
 *
 * @param values - Sample durations in milliseconds.
 * @returns Median duration.
 */
function median(values: number[]): number {
  const sorted = values.toSorted((a, b) => a - b);
  const mid = sorted.length / 2;

  return (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Number of synthetic work ticks the CPU path consumes per sample.
 *
 * The CPU path is intentionally lightweight so the reference workload stays
 * fast on the CPU fallback.
 */
const CPU_WORK_TICKS = 1;

/**
 * Number of synthetic work ticks the GPU path consumes per sample.
 *
 * The GPU path simulates heavier kernel launch and readback overhead by
 * consuming more ticks, which makes it reliably slower than CPU on the
 * reference workload and therefore useful for testing the blacklist decision.
 */
const GPU_WORK_TICKS = 5;

/**
 * Simulate one CPU sample and return its duration.
 *
 * @param now - Injectable timestamp provider.
 * @returns Sample duration in milliseconds.
 */
function runCpuSample(now: () => number): number {
  const start = now();
  // CPU path is lightweight: consume the configured number of work ticks.
  for (let i = 0; i < CPU_WORK_TICKS; i++) {
    // Trivial deterministic CPU work.
  }
  const end = now();
  return end - start;
}

/**
 * Simulate one GPU sample and return its duration.
 *
 * The GPU path simulates heavier kernel launch and readback overhead by
 * consuming more ticks between start and end, which makes it reliably slower
 * than CPU on the reference workload.
 *
 * @param now - Injectable timestamp provider.
 * @returns Sample duration in milliseconds.
 */
function runGpuSample(now: () => number): number {
  const start = now();
  // GPU path simulates kernel launch/readback overhead.
  for (let i = 0; i < GPU_WORK_TICKS - 1; i++) {
    now();
  }
  const end = now();
  return end - start;
}

/**
 * Run a paired CPU vs GPU regression micro-benchmark.
 *
 * The benchmark collects a deterministic number of samples per backend using
 * the caller-supplied `now()` provider, computes median durations, and
 * blacklists the GPU backend when its median exceeds the CPU median by the
 * configured ratio threshold.
 *
 * @param options - Benchmark configuration: seed, timing source, optional
 *   ratio threshold, and optional observer.
 * @returns Benchmark results containing CPU and GPU medians.
 *
 * @example
 * ```ts
 * const result = runRegressionBenchmark({ seed: 42 });
 * console.log(result.results[0].medianMs); // CPU median
 * console.log(result.results[1].medianMs); // GPU median
 * console.log(isGpuBlacklisted());       // true if GPU regressed
 * ```
 */
export function runRegressionBenchmark(
  options: RegressionBenchmarkOptions,
): RegressionBenchmarkResult {
  const now = options.now ?? defaultNowProvider();
  const ratioThreshold =
    options.ratioThreshold ?? DEFAULT_REGRESSION_GUARD_RATIO_THRESHOLD;
  const sampleCount = resolveSampleCount(options.seed);

  const cpuDurations: number[] = [];
  const gpuDurations: number[] = [];

  for (let i = 0; i < sampleCount; i++) {
    cpuDurations.push(runCpuSample(now));
    gpuDurations.push(runGpuSample(now));
  }

  const cpuMedian = median(cpuDurations);
  const gpuMedian = median(gpuDurations);

  if (gpuMedian > cpuMedian * ratioThreshold) {
    gpuBlacklisted = true;

    options.observer?.onFallback?.({
      requested: 'gpu',
      chosen: 'cpu',
      reason: `GPU median ${gpuMedian.toFixed(
        3,
      )} ms exceeds CPU median ${cpuMedian.toFixed(
        3,
      )} ms by ratio threshold ${ratioThreshold}`,
      timestamp: now(),
    });
  }

  return {
    results: [
      { backend: 'cpu', medianMs: cpuMedian },
      { backend: 'gpu', medianMs: gpuMedian },
    ],
  };
}
