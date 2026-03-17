/**
 * Minimal host surface required by the RNG utilities.
 *
 * This contract stays intentionally small so deterministic replay can be used
 * by the main `Neat` controller, tests, and diagnostics without coupling every
 * caller to the full runtime implementation. The contract carries only the live
 * RNG function, the persisted numeric state, enough population context to build
 * a default seed, and the option hooks that can override that default flow.
 */
export interface RngHost {
  /** Cached RNG function reused once the stream has been initialized. */
  _rng?: () => number;
  /** Current numeric RNG state used for deterministic replay and persistence. */
  _rngState?: number;
  /** Population context used when deriving a default time-scrambled seed. */
  population?: unknown[];
  /** Optional injected RNG or explicit seed overriding the default initialization path. */
  options?: { rng?: () => number; seed?: unknown };
}
