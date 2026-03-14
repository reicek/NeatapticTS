/** Default slice size when exporting species history as JSONL. */
export const SPECIES_HISTORY_JSONL_MAX_DEFAULT = 200;

/**
 * Species-history export helpers for the NEAT controller.
 *
 * This chapter keeps the JSONL export surface separate from the broader
 * species-reporting helpers so generated docs stay tightly scoped to one
 * export concern.
 */

/**
 * Export species history records as JSON Lines.
 *
 * @param speciesHistory - Recorded species history entries to serialize.
 * @param maxEntries - Maximum number of recent entries to include.
 * @returns JSONL payload containing the requested recent history slice.
 */
export function exportSpeciesHistoryJsonl(
  speciesHistory: unknown[],
  maxEntries: number = SPECIES_HISTORY_JSONL_MAX_DEFAULT,
): string {
  const trimmedHistory = speciesHistory.slice(-maxEntries);
  return trimmedHistory.map((entry) => JSON.stringify(entry)).join('\n');
}