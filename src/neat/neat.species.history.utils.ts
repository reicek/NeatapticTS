/** Default slice size when exporting species history as JSONL. */
export const SPECIES_HISTORY_JSONL_MAX_DEFAULT = 200;

/**
 * Export species history records as JSON Lines.
 */
export function exportSpeciesHistoryJsonl(
  speciesHistory: unknown[],
  maxEntries: number = SPECIES_HISTORY_JSONL_MAX_DEFAULT,
): string {
  const trimmedHistory = speciesHistory.slice(-maxEntries);
  return trimmedHistory.map((entry) => JSON.stringify(entry)).join('\n');
}
