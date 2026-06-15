/**
 * @module index-stats
 * @description Corpus index statistics tool for the Repo Cortex MCP server.
 *
 * Returns document, chunk, and family row counts plus the timestamp of the
 * most recent indexing run, useful for confirming index health in CI gates.
 *
 * Also surfaces relevance-feedback statistics so callers can observe how much
 * implicit and explicit signal data has been collected for the corpus.
 */
import { openCortexDatabase } from './cortex-db.mjs';

/** Weight applied to feedback boosts when blending them into ranking scores. */
export const FEEDBACK_WEIGHT = 1.0;

/** Half-life (in days) used for exponential time decay of feedback signals. */
export const FEEDBACK_HALF_LIFE_DAYS = 7;

/**
 * Aggregate feedback event and score statistics from the corpus database.
 *
 * @param {import('better-sqlite3').Database} database - Open SQLite connection.
 * @returns {{ total_events: number, events_by_type: Record<string, number>, chunks_with_feedback: number, average_feedback_boost: number | null, feedback_weight: number, feedback_half_life_days: number, last_recomputed_at: string | null }} Feedback statistics.
 */
function buildFeedbackStats(database) {
  const totalEvents = database
    .prepare('SELECT COUNT(*) AS count FROM feedback_events')
    .get().count;

  const eventsByTypeRows = database
    .prepare(
      'SELECT signal_type, COUNT(*) AS count FROM feedback_events GROUP BY signal_type',
    )
    .all();
  const eventsByType = Object.fromEntries(
    eventsByTypeRows.map((row) => [row.signal_type, row.count]),
  );

  const chunksWithFeedback = database
    .prepare('SELECT COUNT(DISTINCT chunk_id) AS count FROM feedback_scores')
    .get().count;
  const averageFeedbackBoostRow = database
    .prepare('SELECT AVG(feedback_boost) AS average FROM feedback_scores')
    .get();
  const lastRecomputedAt = database
    .prepare('SELECT MAX(last_feedback_at) AS value FROM feedback_scores')
    .get().value;

  const rawAverage = averageFeedbackBoostRow?.average;
  const averageFeedbackBoost =
    rawAverage === null || rawAverage === undefined
      ? null
      : Number(rawAverage);

  return {
    total_events: Number(totalEvents),
    events_by_type: eventsByType,
    chunks_with_feedback: Number(chunksWithFeedback),
    average_feedback_boost: averageFeedbackBoost,
    feedback_weight: FEEDBACK_WEIGHT,
    feedback_half_life_days: FEEDBACK_HALF_LIFE_DAYS,
    last_recomputed_at: lastRecomputedAt ?? null,
  };
}

/**
 * Return aggregate statistics for the indexed corpus.
 *
 * @param {object} [options={}] - Tool options.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @returns {Promise<{ total_documents: number, total_chunks: number, total_families: number, last_build_timestamp: string | null, feedback_stats: object }>} Index statistics.
 */
export async function indexStats(options = {}) {
  const database = openCortexDatabase(options.databasePath);

  try {
    const documentCount = database
      .prepare('SELECT COUNT(*) AS count FROM documents')
      .get().count;
    const chunkCount = database
      .prepare('SELECT COUNT(*) AS count FROM chunks')
      .get().count;
    const familyCount = database
      .prepare('SELECT COUNT(DISTINCT doc_family) AS count FROM documents')
      .get().count;
    const lastIndexedAt = database
      .prepare('SELECT MAX(indexed_at) AS value FROM documents')
      .get().value;

    return {
      total_documents: Number(documentCount),
      total_chunks: Number(chunkCount),
      total_families: Number(familyCount),
      last_build_timestamp: asIsoTimestamp(lastIndexedAt),
      feedback_stats: buildFeedbackStats(database),
    };
  } finally {
    database.close();
  }
}

/**
 * Convert a numeric Unix-millisecond timestamp to an ISO 8601 string.
 *
 * @param {unknown} value - Raw timestamp value (e.g. from a SQLite integer column).
 * @returns {string | null} ISO 8601 string, or `null` when the value is missing or non-numeric.
 */
function asIsoTimestamp(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) && numericValue > 0
    ? new Date(numericValue).toISOString()
    : null;
}
