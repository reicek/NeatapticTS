import type { NeatLike, SpeciesHistoryEntry } from './neat.types';

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
  this: NeatLike
): { id: number; size: number; bestScore: number; lastImproved: number }[] {
  // `speciesArray` is a reference to the internal species registry. We map
  // to a minimal representation to avoid exposing the full objects.
  /** const JSDoc short descriptions above each constant */
  /**
   * Array of species stored internally on the Neat instance.
   * This value is intentionally not documented in the public API; we only
   * expose the derived summary below.
   */
  const speciesArray = (this as any)._species as any[];

  // Map internal species to compact summaries.
  return speciesArray.map((species: any) => ({
    id: species.id,
    size: species.members.length,
    bestScore: species.bestScore,
    lastImproved: species.lastImproved,
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
  /** const JSDoc short descriptions above each constant */
  /**
   * The raw species history array captured on the Neat instance. Each element
   * is a snapshot for a generation and includes a `stats` array of per-species
   * summaries.
   */
  const speciesHistory = (this as any)._speciesHistory as SpeciesHistoryEntry[];

  // If the user enabled extended history, ensure extended fields exist by
  // backfilling inexpensive fallbacks where possible.
  if (this.options?.speciesAllocation?.extendedHistory) {
    // Iterate over each generation snapshot
    for (const generationEntry of speciesHistory) {
      // Iterate over each per-species stat in the snapshot
      for (const speciesStat of generationEntry.stats as any[]) {
        // If extended fields already present, skip computation
        if ('innovationRange' in speciesStat && 'enabledRatio' in speciesStat)
          continue;

        // Find a representative species object in the current population by id
        // `speciesObj` is used to compute fallbacks when needed.
        const speciesObj = (this as any)._species.find(
          (s: any) => s.id === speciesStat.id
        );

        // If we have members, compute cheap fallbacks for innovationRange and enabledRatio
        if (speciesObj && speciesObj.members && speciesObj.members.length) {
          // Initialize tracking variables for the innovation id range and enabled/disabled counts
          let maxInnovation = -Infinity;
          let minInnovation = Infinity;
          let enabledCount = 0;
          let disabledCount = 0;

          // For each member genome in the species
          for (const member of speciesObj.members) {
            // For each connection in the genome, attempt to read an innovation id
            for (const connection of member.connections) {
              // Prefer an explicit `innovation` property; otherwise call internal
              // fallback innov extractor (if available) and finally default to 0.
              const innovationId =
                (connection as any).innovation ??
                (this as any)._fallbackInnov?.(connection) ??
                0;

              // Update min/max innovation trackers
              if (innovationId > maxInnovation) maxInnovation = innovationId;
              if (innovationId < minInnovation) minInnovation = innovationId;

              // Count enabled vs disabled connections (treat undefined as enabled)
              if ((connection as any).enabled === false) disabledCount++;
              else enabledCount++;
            }
          }

          // Compute innovationRange: positive difference when valid, otherwise 0
          (speciesStat as any).innovationRange =
            isFinite(maxInnovation) &&
            isFinite(minInnovation) &&
            maxInnovation > minInnovation
              ? maxInnovation - minInnovation
              : 0;

          // Compute enabledRatio: fraction of enabled connections when any exist
          (speciesStat as any).enabledRatio =
            enabledCount + disabledCount
              ? enabledCount / (enabledCount + disabledCount)
              : 0;
        }
      }
    }
  }

  // Return the possibly-augmented history. Consumers should treat this as read-only.
  return speciesHistory;
}
