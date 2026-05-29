/**
 * Sanitizes user-controlled text before it is used in an SQLite FTS query.
 *
 * The durable-memory layer keeps its own local sanitizer so it does not depend
 * on the Repo Cortex indexing scripts.
 *
 * @param rawQuery - Raw query text that may contain FTS5 operator characters.
 * @returns Plain token text with operator syntax stripped and whitespace normalized.
 */
export function sanitizeFtsQuery(rawQuery: string): string {
  return rawQuery
    .replace(/[^\p{L}\p{N}\s']+/gu, ' ')
    .replace(/\s+/gu, ' ')
    .trim();
}
