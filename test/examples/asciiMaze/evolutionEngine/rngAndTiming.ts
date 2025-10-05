/**
 * RNG and timing helpers extracted from the evolution façade.
 * @remarks
 * All helpers accept an {@link EngineState} instance so they can read and mutate
 * shared state without relying on class statics.
 */

import {
  EngineState,
  RngCacheParameters,
  DEFAULT_RNG_CACHE_BATCH_SIZE,
  ensureRngCacheBatch,
  reseedRngState,
} from './engineState';

/** Linear congruential multiplier mirroring the historic façade RNG. */
const LCG_MULTIPLIER = 1_664_525;
/** Linear congruential increment mirroring the historic façade RNG. */
const LCG_INCREMENT = 1_013_904_223;
/** Bit shift applied when converting cached integers into uniform floats. */
const LCG_SHIFT = 9;
/** Scalar mapping shift-masked integers into the [0, 1) interval. */
const LCG_SCALE = 1 / 0x800000;

let cachedRngParameters: RngCacheParameters | undefined;

/**
 * Obtain a monotonic-ish timestamp suitable for profiling.
 * @returns Timestamp in milliseconds, preferring `performance.now` when available.
 * @example
 * const timestamp = readHighResolutionTime();
 */
export const readHighResolutionTime = (): number =>
  globalThis.performance?.now?.() ?? Date.now();

/**
 * Return a profiling start timestamp that mirrors the historic `#PROFILE_T0` helper.
 * @returns Millisecond timestamp representing the profiling start time.
 */
export const profilingStartTimestamp = (): number =>
  readHighResolutionTime();

/**
 * Provide cached congruential parameters used by the shared fast RNG helper.
 * @returns Immutable {@link RngCacheParameters} reference reused across draws.
 * @example
 * const parameters = resolveRngParameters();
 */
export const resolveRngParameters = (): RngCacheParameters => {
  if (!cachedRngParameters) {
    cachedRngParameters = Object.freeze({
      batchSize: DEFAULT_RNG_CACHE_BATCH_SIZE,
      multiplier: LCG_MULTIPLIER,
      increment: LCG_INCREMENT,
      shift: LCG_SHIFT,
      scale: LCG_SCALE,
    });
  }
  return cachedRngParameters;
};

/**
 * Accumulate a profiling duration under the supplied key when detail profiling is enabled.
 * @param state Shared engine state containing profiling configuration.
 * @param category Profiling segment key (for example `telemetry`).
 * @param deltaMs Millisecond duration to add to the accumulator.
 * @returns void.
 */
export const accumulateProfilingDuration = (
  state: EngineState,
  category: string,
  deltaMs: number,
): void => {
  if (!state.profiling.detailsEnabled) return;
  if (!Number.isFinite(deltaMs)) return;
  const accumulatorBucket = state.profiling.accumulators;
  accumulatorBucket[category] = (accumulatorBucket[category] ?? 0) + deltaMs;
};

/**
 * Generate a fast uniform random sample using the shared RNG cache.
 * @param state Shared engine state providing the RNG cache and seed.
 * @param parameters Congruential parameters forwarded to {@link ensureRngCacheBatch}.
 * @returns Uniform sample in the range [0, 1).
 * @example
 * const sample = drawFastRandom(sharedState, rngParameters);
 */
export const drawFastRandom = (
  state: EngineState,
  parameters: RngCacheParameters,
): number => {
  const rngHandles = ensureRngCacheBatch(parameters, state);
  const scratch = state.scratch;
  const nextIndex = scratch.rngCacheIndex++;
  return rngHandles.cache[nextIndex];
};

/**
 * Enable deterministic RNG mode and optionally reseed the shared RNG state.
 * @param state Shared engine state where the deterministic flag is stored.
 * @param seed Optional deterministic seed. Finite numeric inputs are normalised to u32.
 * @returns void.
 */
export const setDeterministicMode = (
  state: EngineState,
  seed?: number,
): void => {
  state.deterministicMode = true;
  if (typeof seed === 'number' && Number.isFinite(seed)) {
    reseedRngState(seed, state);
  }
};

/**
 * Disable deterministic RNG mode.
 * @param state Shared engine state where the deterministic flag is stored.
 * @returns void.
 */
export const clearDeterministicMode = (state: EngineState): void => {
  state.deterministicMode = false;
};

/**
 * Retrieve the deterministic RNG flag stored on the shared engine state.
 * @param state Shared engine state where the deterministic flag is stored.
 * @returns True when deterministic mode is active.
 */
export const isDeterministicModeEnabled = (state: EngineState): boolean =>
  state.deterministicMode;

/**
 * Determine whether detailed profiling accumulation is active.
 * @param state Shared engine state providing profiling configuration.
 * @returns True when detail profiling is enabled.
 */
export const isProfilingDetailsEnabled = (state: EngineState): boolean =>
  state.profiling.detailsEnabled;

/**
 * Provide direct access to the profiling accumulator map.
 * @param state Shared engine state containing the profiling accumulators.
 * @returns Mutable record of profiling accumulators keyed by category name.
 */
export const getProfilingAccumulators = (
  state: EngineState,
): Record<string, number> => state.profiling.accumulators;
