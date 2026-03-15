import type { NeatLike, SpeciesLike } from '../../shared/neat.shared.types';

/**
 * Narrow host surface required by the species-stats chapter.
 *
 * Stats reads only the current species registry, so this helper keeps that
 * dependency explicit instead of coupling the chapter to the broader history
 * and fallback-innovation surface used by the root boundary.
 */
interface SpeciesStatsHost {
  _species?: SpeciesLike[];
}

/**
 * Species-summary projection helpers for the NEAT controller.
 *
 * This chapter owns the lightweight, read-only view used by dashboards, logs,
 * and quick diagnostics when callers only need the current species roster.
 * Keeping that projection here lets the root `species.ts` file focus on
 * orchestrating the broader reporting surface.
 */

/**
 * Get lightweight per-species statistics for the current population.
 *
 * The returned records are intentionally compact and detached from the live
 * species objects, which makes them safer to log, serialize, or hand to UI
 * code without leaking mutable internal member arrays.
 *
 * @param host - NEAT host exposing the internal species registry.
 * @returns Compact per-species summaries suitable for reporting.
 *
 * @example
 * ```ts
 * const summaries = getSpeciesStats(neat);
 * console.log(summaries.map((species) => `${species.id}:${species.size}`));
 * ```
 */
export function getSpeciesStats(
  host: NeatLike,
): { id: number; size: number; bestScore: number; lastImproved: number }[] {
  const speciesStatsHost = host as NeatLike & SpeciesStatsHost;
  const speciesRecords = speciesStatsHost._species ?? [];

  // Step 1: Project live species records into compact reporting snapshots.
  return speciesRecords.map((speciesRecord) => ({
    id: speciesRecord.id,
    size: speciesRecord.members?.length ?? 0,
    bestScore: speciesRecord.bestScore ?? 0,
    lastImproved: speciesRecord.lastImproved ?? 0,
  }));
}
