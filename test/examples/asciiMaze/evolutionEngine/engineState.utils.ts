/**
 * Pure utility helpers shared by the ASCII maze engine-state facade.
 */

import type {
  EngineScratchState,
  TelemetryScratchHandles,
  TelemetryScratchRequest,
} from './engineState.types';
import {
  ACTION_OUTPUT_DIMENSION,
  DEFAULT_LOGITS_RING_CAPACITY,
  DEFAULT_RNG_CACHE_BATCH_SIZE,
  DEFAULT_VISITED_HASH_LOAD_FACTOR,
  MAX_VISITED_HASH_LOAD_FACTOR,
  MIN_VISITED_HASH_LOAD_FACTOR,
  RNG_GOLDEN_RATIO_SEED,
} from './engineState.constants';

interface TelemetryCapacityHints {
  actionDimension: number;
  requiresHigherMoments: boolean;
  biasCount: number;
  stringLength: number;
}

/**
 * Build a logits ring sized to the requested default capacity.
 * @returns Array of Float32Array rows sized to the action dimension.
 */
export const createLogitsRing = (): Float32Array[] => {
  const rows = new Array<Float32Array>(DEFAULT_LOGITS_RING_CAPACITY);
  for (
    let ringIndex = 0;
    ringIndex < DEFAULT_LOGITS_RING_CAPACITY;
    ringIndex++
  ) {
    rows[ringIndex] = new Float32Array(ACTION_OUTPUT_DIMENSION);
  }
  return rows;
};

/**
 * Compute whether detailed profiling is enabled via the environment flag.
 * @returns True when the profiling environment flag is set.
 */
export const resolveProfilingEnabled = (): boolean => {
  try {
    return (
      typeof process !== 'undefined' &&
      process?.env?.ASCII_MAZE_PROFILE_DETAILS === '1'
    );
  } catch {
    return false;
  }
};

/**
 * Build a fresh profiling accumulator map seeded with zero totals.
 * @returns Accumulator record keyed by profiling segment name.
 */
export const createProfilingAccumulators = (): Record<string, number> => ({
  telemetry: 0,
  simplify: 0,
  snapshot: 0,
  prune: 0,
});

/**
 * Build the reusable snapshot metadata payload consumed by persistence helpers.
 * @returns Snapshot placeholder populated with neutral defaults.
 */
export const createSnapshotReusableObject = () => ({
  generation: 0,
  bestFitness: 0,
  simplifyMode: false,
  plateauCounter: 0,
  timestamp: 0,
  telemetryTail: undefined,
  top: undefined,
});

/**
 * Derive normalised capacity hints from the raw telemetry scratch request.
 * @param request Raw capacity request supplied by callers.
 * @returns Sanitised capacity values used during buffer initialisation.
 */
export const normaliseTelemetryCapacityHints = (
  request: TelemetryScratchRequest,
): TelemetryCapacityHints => {
  const normaliseSize = (value: number | undefined, fallback = 0): number => {
    if (!Number.isFinite(value as number)) {
      return Math.max(0, Math.floor(fallback));
    }
    return Math.max(0, Math.floor(value as number));
  };

  const biasCount = normaliseSize(request.biasCount);
  return {
    actionDimension: normaliseSize(request.actionDimension),
    requiresHigherMoments: Boolean(request.includeHigherMoments),
    biasCount,
    stringLength: normaliseSize(request.stringBufferLength, biasCount),
  };
};

/**
 * Ensure all Float64 scratch pools required for telemetry are adequately sized.
 * @param scratch Shared scratch state mutated in place.
 * @param hints Normalised capacity hints.
 * @returns void.
 */
export const ensureTelemetryFloatPools = (
  scratch: EngineScratchState,
  hints: TelemetryCapacityHints,
): void => {
  const minActionDimension = Math.max(4, hints.actionDimension);
  scratch.exps = ensureFloat64Pool(scratch.exps, minActionDimension, 4);
  scratch.means = ensureFloat64Pool(scratch.means, hints.actionDimension, 1);
  scratch.standardDeviations = ensureFloat64Pool(
    scratch.standardDeviations,
    hints.actionDimension,
    1,
  );
  scratch.secondMomentRaw = ensureFloat64Pool(
    scratch.secondMomentRaw,
    hints.actionDimension,
    1,
  );
  scratch.biasTelemetryScratch = ensureFloat64Pool(
    scratch.biasTelemetryScratch,
    hints.biasCount,
    1,
  );

  if (!hints.requiresHigherMoments) return;

  scratch.thirdMomentRaw = ensureOptionalFloat64Pool(
    scratch.thirdMomentRaw,
    hints.actionDimension,
  );
  scratch.fourthMomentRaw = ensureOptionalFloat64Pool(
    scratch.fourthMomentRaw,
    hints.actionDimension,
  );
  scratch.kurtosis = ensureOptionalFloat64Pool(
    scratch.kurtosis,
    hints.actionDimension,
  );
};

/**
 * Ensure the reusable string assembly buffer has sufficient capacity.
 * @param buffer Existing string buffer instance.
 * @param required Minimum number of slots needed for upcoming telemetry joins.
 * @returns Original buffer when large enough, otherwise a grown copy.
 */
export const ensureTelemetryStringBuffer = (
  buffer: string[],
  required: number,
): string[] => ensureArrayCapacity(buffer, required);

/**
 * Build typed handles referencing the ensured telemetry scratch buffers.
 * @param scratch Scratch state containing the prepared buffers.
 * @returns Structured handles consumed by telemetry helpers.
 */
export const buildTelemetryHandles = (
  scratch: EngineScratchState,
): TelemetryScratchHandles => ({
  exponentScratch: scratch.exps,
  meanScratch: scratch.means,
  standardDeviationScratch: scratch.standardDeviations,
  secondMomentScratch: scratch.secondMomentRaw,
  thirdMomentScratch: scratch.thirdMomentRaw,
  fourthMomentScratch: scratch.fourthMomentRaw,
  kurtosisScratch: scratch.kurtosis,
  biasScratch: scratch.biasTelemetryScratch,
  stringBuffer: scratch.stringAssemblyBuffer,
});

/**
 * Compute the next power-of-two for geometric growth.
 * @param candidate Raw size candidate.
 * @returns Smallest power-of-two greater than or equal to the candidate.
 */
export const nextPowerOfTwo = (candidate: number): number => {
  if (candidate <= 1) return 1;
  return 1 << Math.ceil(Math.log2(candidate));
};

/**
 * Clamp the requested target entry count to a non-negative integer.
 * @param requestedEntries Raw target entry count supplied by callers.
 * @returns Sanitised entry count used for capacity planning.
 */
export const normaliseVisitedHashEntries = (
  requestedEntries: number,
): number => {
  if (!Number.isFinite(requestedEntries)) return 0;
  return Math.max(0, Math.floor(requestedEntries));
};

/**
 * Ensure the visited hash load factor falls within a sensible range.
 * @param requestedLoadFactor Proposed load factor from configuration or state.
 * @returns Clamped load factor with a fallback to the project default.
 */
export const normaliseVisitedHashLoad = (
  requestedLoadFactor: number,
): number => {
  if (!Number.isFinite(requestedLoadFactor)) {
    return DEFAULT_VISITED_HASH_LOAD_FACTOR;
  }
  if (
    requestedLoadFactor <= MIN_VISITED_HASH_LOAD_FACTOR ||
    requestedLoadFactor >= MAX_VISITED_HASH_LOAD_FACTOR
  ) {
    return DEFAULT_VISITED_HASH_LOAD_FACTOR;
  }
  return requestedLoadFactor;
};

/**
 * Clamp the RNG cache batch size to a positive integer.
 * @param requestedBatchSize Raw batch size requested by callers.
 * @returns Valid batch size.
 */
export const normaliseRngBatchSize = (requestedBatchSize: number): number => {
  if (!Number.isFinite(requestedBatchSize)) {
    return DEFAULT_RNG_CACHE_BATCH_SIZE;
  }
  return Math.max(1, Math.floor(requestedBatchSize));
};

/**
 * Normalise a raw numeric seed into an unsigned 32-bit value.
 * @param rawSeed Raw seed provided by the caller.
 * @returns Unsigned 32-bit seed suitable for the congruential generator.
 */
export const normaliseRngSeed = (rawSeed: number): number => {
  const candidate = rawSeed >>> 0;
  return candidate === 0 ? RNG_GOLDEN_RATIO_SEED : candidate;
};

/**
 * Ensure a required Float64Array pool satisfies the requested capacity.
 * @param buffer Existing Float64Array instance reused by the engine.
 * @param required Minimum length the caller needs.
 * @param minimum Optional lower bound applied before comparison.
 * @returns Buffer with sufficient capacity.
 */
const ensureFloat64Pool = (
  buffer: Float64Array,
  required: number,
  minimum = 0,
): Float64Array => {
  const target = Math.max(required, minimum);
  if (target <= 0 || buffer.length >= target) return buffer;
  const nextSize = nextPowerOfTwo(target);
  const next = new Float64Array(nextSize);
  if (buffer.length > 0) {
    next.set(buffer.subarray(0, Math.min(buffer.length, nextSize)));
  }
  return next;
};

/**
 * Ensure an optional Float64Array is present and sized appropriately.
 * @param buffer Optional Float64Array reference to validate.
 * @param required Minimum required capacity.
 * @returns Buffer with sufficient capacity or undefined when no allocation is required.
 */
const ensureOptionalFloat64Pool = (
  buffer: Float64Array | undefined,
  required: number,
): Float64Array | undefined => {
  if (required <= 0) return buffer;
  if (!buffer) return new Float64Array(nextPowerOfTwo(required));
  if (buffer.length >= required) return buffer;
  const nextSize = nextPowerOfTwo(required);
  const next = new Float64Array(nextSize);
  next.set(buffer.subarray(0, Math.min(buffer.length, nextSize)));
  return next;
};

/**
 * Ensure a generic Array buffer has enough slots.
 * @param buffer Buffer instance to validate.
 * @param required Minimum number of elements required.
 * @returns Buffer with sufficient capacity.
 */
const ensureArrayCapacity = <T>(buffer: T[], required: number): T[] => {
  if (required <= 0 || buffer.length >= required) return buffer;
  const nextSize = nextPowerOfTwo(required);
  const grown = new Array<T>(nextSize);
  for (let index = 0; index < buffer.length; index++) {
    grown[index] = buffer[index];
  }
  return grown;
};
