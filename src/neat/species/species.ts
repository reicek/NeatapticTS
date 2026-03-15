import type {
  NeatLike,
  SpeciesHistoryEntry,
} from '../shared/neat.shared.types';
import { getSpeciesHistory as getSpeciesHistoryImpl } from './history/read/species.history.read';
import { getSpeciesStats as getSpeciesStatsImpl } from './stats/species.stats';

/**
 * Species-reporting helpers for the NEAT controller.
 *
 * The root species chapter keeps the public reporting flow small: one path
 * returns the current live species summaries, and the other returns the
 * recorded cross-generation history.
 *
 * - `stats/` projects the live species registry into compact reporting rows.
 * - `core/` explains when extended species history is backfilled and how innovation-range summaries are derived.
 * - `history/` keeps the JSONL export surface scoped to one serialization concern.
 */

/**
 * Get lightweight per-species statistics for the current population.
 *
 * This is the shortest read path into the species system. It is useful when a
 * caller needs dashboard-friendly snapshots such as current species sizes,
 * recent improvement timestamps, or the best score per species without pulling
 * the heavier generation-by-generation history buffer.
 *
 * @param this - NEAT host exposing the internal species registry.
 * @returns Compact per-species summaries suitable for reporting.
 *
 * @example
 * ```ts
 * const speciesSummaries = neat.getSpeciesStats();
 * console.table(speciesSummaries);
 * ```
 */
export function getSpeciesStats(
  this: NeatLike,
): { id: number; size: number; bestScore: number; lastImproved: number }[] {
  // Step 1: Delegate the compact stats projection to the dedicated stats chapter.
  return getSpeciesStatsImpl(this);
}

/**
 * Retrieve the recorded species history across generations.
 *
 * @param this - NEAT host exposing species history, species records, fallback innovation logic, and options.
 * @returns Generation-stamped species history snapshots.
 */
export function getSpeciesHistory(this: NeatLike): SpeciesHistoryEntry[] {
  // Step 1: Delegate the history read flow to the dedicated history-read chapter.
  return getSpeciesHistoryImpl(this);
}
