import type {
  NeatLike,
  SpeciesHistoryEntry,
  SpeciesLike,
  ConnectionLike,
} from './neat.types';
import {
  backfillExtendedHistory,
  shouldAugmentExtendedHistory,
} from './neat.species.utils';

/**
 * Get lightweight per-species statistics for the current population.
 *
 * This method intentionally returns a small, immutable-friendly summary per
 * species rather than exposing internal member lists. This avoids accidental
 * mutation of the library's internal state while still providing useful
 * telemetry for UIs, dashboards, or logging.
 *
 * Example:
 * ```ts
 * const stats = neat.getSpeciesStats();
 * // stats => [{ id: 1, size: 12, bestScore: 0.85, lastImproved: 42 }, ...]
 * ```
 *
 * Success criteria:
 * - Returns an array of objects each containing `id`, `size`, `bestScore`,
 *   and `lastImproved`.
 * - Does not expose or return references to internal member arrays.
 *
 * @returns Array of per-species summaries suitable for reporting.
 */
export function getSpeciesStats(
  this: NeatLike,
): { id: number; size: number; bestScore: number; lastImproved: number }[] {
  // Step 1: Read the internal species registry (kept private in the public API).
  /**
   * Array of species stored internally on the Neat instance.
   * This value is intentionally not documented in the public API; we only
   * expose the derived summary below.
   */
  const ctx = this as unknown as { _species?: unknown[] };
  const speciesArray = (ctx._species as SpeciesLike[]) || [];

  // Step 2: Map internal species to compact summaries.
  return speciesArray.map((species: SpeciesLike) => ({
    id: species.id,
    size: (species.members && species.members.length) || 0,
    bestScore: (species.bestScore as number) || 0,
    lastImproved: (species.lastImproved as number) || 0,
  }));
}

/**
 * Retrieve the recorded species history across generations.
 *
 * Each entry in the returned array corresponds to a recorded generation and
 * contains a snapshot of statistics for every species at that generation.
 * This is useful for plotting species sizes over time, tracking innovation
 * spread, or implementing population-level diagnostics.
 *
 * The shape of each entry is defined by `SpeciesHistoryEntry` in the public
 * types. When `options.speciesAllocation.extendedHistory` is enabled the
 * library attempts to include additional metrics such as `innovationRange`
 * and `enabledRatio`. When those extended metrics are missing they are
 * computed lazily from a representative genome to ensure historical data is
 * still useful for analysis.
 *
 * Example:
 * ```ts
 * const history = neat.getSpeciesHistory();
 * // history => [{ generation: 0, stats: [{ id:1, size:10, innovationRange:5, enabledRatio:0.9 }, ...] }, ...]
 * ```
 *
 * Notes for documentation:
 * - The function tries to avoid heavy computation. Extended metrics are
 *   computed only when explicitly requested via options.
 * - Computed extended metrics are conservative fallbacks; they use the
 *   available member connections and a fallback innovation extractor when
 *   connection innovation IDs are not present.
 *
 * @returns Array of generation-stamped species statistic snapshots.
 */
export function getSpeciesHistory(this: NeatLike): SpeciesHistoryEntry[] {
  /**
   * The raw species history array captured on the Neat instance. Each element
   * is a snapshot for a generation and includes a `stats` array of per-species
   * summaries.
   */
  const ctx = this as NeatLike & {
    _speciesHistory?: SpeciesHistoryEntry[];
    _species?: SpeciesLike[];
    _fallbackInnov?: (c: ConnectionLike) => number;
  };

  const speciesHistory = (ctx._speciesHistory as SpeciesHistoryEntry[]) || [];

  /**
   * The typed options for this Neat instance, when available.
   */
  const neatOptions = this.options as
    | import('./neat.types').NeatOptions
    | undefined;

  // Step 1: Backfill extended stats only when explicitly enabled.
  if (shouldAugmentExtendedHistory(neatOptions)) {
    backfillExtendedHistory(speciesHistory, ctx);
  }

  // Return the possibly-augmented history. Consumers should treat this as read-only.
  return speciesHistory;
}
