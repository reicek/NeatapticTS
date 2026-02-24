import {
  RNG_DEFAULT_SEED_FALLBACK,
  RNG_NORMALIZATION_DIVISOR,
  RNG_POPULATION_OFFSET,
  RNG_SHIFT_LEFT_PRIMARY,
  RNG_SHIFT_LEFT_SECONDARY,
  RNG_SHIFT_RIGHT_PRIMARY,
  RNG_TIME_SCRAMBLE_CONSTANT,
} from './neat.rng.constants';

/**
 * Minimal host surface required by the RNG utilities.
 */
export interface RngHost {
  _rng?: () => number;
  _rngState?: number;
  population?: unknown[];
  options?: { rng?: () => number; seed?: unknown };
}

/**
 * Return a cached RNG or create a deterministic xorshift RNG when absent.
 *
 * The helper respects a user-provided RNG at `options.rng` when present.
 * Otherwise it seeds a xorshift32 RNG using the current time and population
 * size, guarding against the invalid zero seed.
 *
 * @param host - Object holding RNG state and configuration.
 * @returns A function that yields a uniform random value in [0, 1).
 */
export function getOrCreateRng(host: RngHost): () => number {
  if (host._rng) return host._rng;

  const userSuppliedRng = host.options?.rng;
  if (typeof userSuppliedRng === 'function') {
    host._rng = userSuppliedRng;
    return host._rng;
  }

  const restoredSeed = normalizeSeed(host._rngState);
  const optionSeed = normalizeSeed(
    (host.options as { seed?: unknown } | undefined)?.seed,
  );
  const initialSeed = guardSeedZero(
    restoredSeed ?? optionSeed ?? resolveInitialSeed(host),
  );
  host._rngState = initialSeed;

  host._rng = () => {
    let currentState = (host._rngState ?? RNG_DEFAULT_SEED_FALLBACK) >>> 0;

    currentState ^= currentState << RNG_SHIFT_LEFT_PRIMARY;
    currentState >>>= 0;

    currentState ^= currentState >> RNG_SHIFT_RIGHT_PRIMARY;
    currentState >>>= 0;

    currentState ^= currentState << RNG_SHIFT_LEFT_SECONDARY;
    currentState >>>= 0;

    host._rngState = currentState >>> 0;
    return (host._rngState ?? 0) / RNG_NORMALIZATION_DIVISOR;
  };

  return host._rng;
}

/**
 * Snapshot the current RNG state for deterministic replay.
 *
 * @param host - Object holding RNG state.
 * @returns The numeric RNG state or undefined when uninitialized.
 */
export function snapshotRngState(host: RngHost): number | undefined {
  return host._rngState;
}

/**
 * Restore a previously captured RNG state.
 *
 * @param host - Object holding RNG state.
 * @param state - Numeric RNG state to restore.
 */
export function restoreRngState(
  host: RngHost,
  state: number | string | undefined,
): void {
  host._rngState = normalizeSeed(state);
  host._rng = undefined;
}

/**
 * Alias for restoring RNG state kept for compatibility with prior surface.
 */
export function importRngState(
  host: RngHost,
  state: number | string | undefined,
): void {
  restoreRngState(host, state);
}

/**
 * Export the current RNG state for persistence.
 *
 * @param host - Object holding RNG state.
 * @returns The numeric RNG state or undefined when not set.
 */
export function exportRngState(host: RngHost): number | undefined {
  return host._rngState;
}

/**
 * Produce a sequence of random samples using the host RNG.
 *
 * @param host - Object holding RNG state.
 * @param sampleCount - Number of samples to generate.
 * @returns Array of random samples in [0, 1).
 */
export function sampleRandomSequence(
  host: RngHost,
  sampleCount: number,
): number[] {
  const rng = getOrCreateRng(host);
  const samples: number[] = [];
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    samples.push(rng());
  }
  return samples;
}

function resolveInitialSeed(host: RngHost): number {
  const populationSize = Array.isArray(host.population)
    ? host.population.length
    : 0;
  const scrambledSeed =
    (Date.now() ^
      ((populationSize + RNG_POPULATION_OFFSET) *
        RNG_TIME_SCRAMBLE_CONSTANT)) >>>
    0;
  const guardedSeed =
    scrambledSeed === 0 ? RNG_DEFAULT_SEED_FALLBACK : scrambledSeed;
  return guardedSeed >>> 0;
}

function normalizeSeed(candidate: unknown): number | undefined {
  if (typeof candidate === 'number' && Number.isFinite(candidate))
    return candidate >>> 0;
  if (typeof candidate === 'string' && candidate.trim().length > 0) {
    const parsedSeed = Number(candidate);
    if (Number.isFinite(parsedSeed)) return parsedSeed >>> 0;
  }
  return undefined;
}

function guardSeedZero(seed: number): number {
  if (seed === 0) return RNG_DEFAULT_SEED_FALLBACK;
  return seed >>> 0;
}
