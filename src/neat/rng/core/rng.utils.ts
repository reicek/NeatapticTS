/**
 * Replay utilities for the deterministic NEAT RNG.
 *
 * This file turns the small {@link RngHost} contract into one coherent
 * lifecycle: resolve who owns randomness, create or reuse the active stream,
 * capture a checkpoint before a risky operation, export that state when it must
 * cross a persistence boundary, and restore it later so the next draw resumes
 * from the same numeric position.
 *
 * The helpers are intentionally small and composable because different callers
 * care about different slices of that lifecycle. The controller mostly wants a
 * live RNG, tests often want checkpoints plus short sample runs, and export
 * logic usually only needs the compact numeric state.
 */

import {
  RNG_DEFAULT_SEED_FALLBACK,
  RNG_NORMALIZATION_DIVISOR,
  RNG_POPULATION_OFFSET,
  RNG_SHIFT_LEFT_PRIMARY,
  RNG_SHIFT_LEFT_SECONDARY,
  RNG_SHIFT_RIGHT_PRIMARY,
  RNG_TIME_SCRAMBLE_CONSTANT,
} from './rng.constants';
import type { RngHost } from './rng.types';

/**
 * Return a cached RNG or create a deterministic xorshift RNG when absent.
 *
 * This is the root runtime entrypoint for randomness. The helper resolves the
 * random stream in four ordered tiers:
 *
 * 1. reuse a previously created RNG when the stream already exists,
 * 2. prefer a user-supplied RNG when the caller wants to own randomness
 *    directly,
 * 3. otherwise rebuild the internal stream from restored numeric state or an
 *    explicit seed,
 * 4. if neither exists, derive a guarded default seed from lightweight host
 *    context.
 *
 * That order matters for replay. Once state has been restored, later random
 * draws should continue from the restored numeric state rather than silently
 * reseeding the controller. It also matters for ownership: an injected RNG is a
 * deliberate opt-out from the internal xorshift lifecycle, not just another
 * fallback.
 *
 * @example
 * ```ts
 * const rng = getOrCreateRng(neat);
 * const firstDraw = rng();
 * const checkpoint = snapshotRngState(neat);
 * ```
 *
 * @param host Object holding RNG state and configuration.
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
 * Use this when you want an in-memory checkpoint before a risky controller
 * action such as a mutation batch, debugging session, or deterministic test.
 * Unlike exporting a whole controller state, this is the smallest replay token:
 * it captures only the numeric RNG position.
 *
 * Prefer this helper when the state is staying in memory inside the current
 * process. Use `exportRngState()` when the same token is about to cross a wider
 * boundary such as JSON serialization, checkpoint files, or fixture snapshots.
 *
 * @param host Object holding RNG state.
 * @returns The numeric RNG state or undefined when uninitialized.
 */
export function snapshotRngState(host: RngHost): number | undefined {
  return host._rngState;
}

/**
 * Restore a previously captured RNG state.
 *
 * Restoring state clears the cached RNG function so the next call to
 * `getOrCreateRng()` rebuilds the stream from the restored numeric position
 * instead of continuing from an older closure. This separation is deliberate:
 * the restore step changes replay state immediately, while stream recreation is
 * deferred until a caller actually needs the next random draw.
 *
 * @example
 * ```ts
 * const savedState = exportRngState(neat);
 * restoreRngState(neat, savedState);
 * ```
 *
 * @param host Object holding RNG state.
 * @param state Numeric RNG state to restore.
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
 *
 * This exists so older callers can keep using the import-style name while the
 * underlying behavior remains the same replay boundary as `restoreRngState()`.
 *
 * @param host Object holding RNG state.
 * @param state Numeric RNG state to restore.
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
 * Use this when deterministic replay must cross a broader boundary such as
 * JSON export, checkpointing, or test snapshots. The returned number is the
 * compact controller-facing representation of the current random stream.
 *
 * Unlike `snapshotRngState()`, this helper is named for the portability use
 * case: the returned token is meant to leave the immediate call site and later
 * come back through `restoreRngState()` or `importRngState()`.
 *
 * @param host Object holding RNG state.
 * @returns The numeric RNG state or undefined when not set.
 */
export function exportRngState(host: RngHost): number | undefined {
  return host._rngState;
}

/**
 * Produce a sequence of random samples using the host RNG.
 *
 * This helper is mainly a diagnostics and testing convenience. It makes the
 * deterministic stream observable without forcing every caller to hand-roll its
 * own sampling loop, which is useful when comparing restored-state replay with
 * fresh execution.
 *
 * Sampling advances the same live stream used by the controller. Callers that
 * want a "peek" rather than a committed advance should snapshot first, sample,
 * then restore the saved state.
 *
 * @example
 * ```ts
 * const before = snapshotRngState(neat);
 * const samples = sampleRandomSequence(neat, 3);
 * restoreRngState(neat, before);
 * ```
 *
 * @param host Object holding RNG state.
 * @param sampleCount Number of samples to generate.
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
  if (typeof candidate === 'number' && Number.isFinite(candidate)) {
    return candidate >>> 0;
  }

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
