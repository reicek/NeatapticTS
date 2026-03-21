/**
 * Core contracts for deterministic RNG replay.
 *
 * This chapter explains the narrow host seam behind the NEAT controller's
 * random stream. The runtime does not need a heavyweight randomness service to
 * stay reproducible. It only needs a small place to keep the live generator,
 * remember its current numeric position, and accept caller-provided randomness
 * when a test or host wants to own that choice directly.
 *
 * Read this file as the state map for the replay lifecycle:
 *
 * 1. `options.rng` lets a caller fully own randomness,
 * 2. `options.seed` or `_rngState` gives the utilities enough information to
 *    rebuild a known stream,
 * 3. `_rng` caches the active closure between draws,
 * 4. `population` provides just enough context to derive a guarded default
 *    seed when no explicit source is available.
 *
 * `rng.utils.ts` owns the behavior. This file keeps the required state small
 * so export, restore, checkpoint, and diagnostics flows can all work without
 * coupling replay helpers to the full `Neat` implementation.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   inputs[Injected RNG, explicit seed, or restored state]:::base --> host[RngHost]:::accent
 *   host --> stream[Live cached RNG closure]:::base
 *   stream --> draws[Mutation, selection, and crossover draws]:::base
 *   stream --> state[Snapshot or export numeric state]:::base
 *   state --> host
 * ```
 */

/**
 * Minimal host surface required by the RNG replay utilities.
 *
 * This contract is deliberately smaller than the full controller. The replay
 * helpers only need four kinds of state: the live RNG closure, the numeric
 * checkpoint behind that closure, a little population context for fallback
 * seeding, and the optional hooks that let callers override the default path.
 *
 * That small seam is what makes deterministic replay portable. Tests,
 * diagnostics, import-export helpers, and the controller itself can all reuse
 * the same RNG utilities without pretending they share one large runtime type.
 *
 * @example
 * ```ts
 * const host: RngHost = {
 *   population: new Array(10),
 *   options: { seed: 42 },
 * };
 * ```
 */
export interface RngHost {
  /**
   * Cached RNG closure reused after the stream has been initialized.
   *
   * Clearing this value is the signal that the next read should rebuild the
   * stream from `_rngState` or another configured seed source.
   */
  _rng?: () => number;
  /**
   * Current numeric RNG position used for checkpoints, persistence, and replay.
   *
   * This is the compact token exported by the helpers when callers want to
   * pause a run and later resume random draws from the same point.
   */
  _rngState?: number;
  /**
   * Population context used only when deriving a guarded default seed.
   *
   * When callers provide neither a live RNG, a restored state, nor an explicit
   * seed, the utilities use population size as one input to the fallback seed
   * calculation so tiny or empty populations do not collapse toward zero.
   */
  population?: unknown[];
  /**
   * Optional caller-owned randomness hooks.
   *
   * `rng` fully bypasses the internal xorshift stream, while `seed` asks the
   * helpers to build a deterministic internal stream from a specific starting
   * value.
   */
  options?: { rng?: () => number; seed?: unknown };
}
