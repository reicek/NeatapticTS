/**
 * Minimal host surface required by the RNG utilities.
 *
 * This contract stays intentionally small so deterministic replay can be used
 * by the main `Neat` controller, tests, and diagnostics without coupling every
 * caller to the full runtime implementation.
 */
export interface RngHost {
  _rng?: () => number;
  _rngState?: number;
  population?: unknown[];
  options?: { rng?: () => number; seed?: unknown };
}
