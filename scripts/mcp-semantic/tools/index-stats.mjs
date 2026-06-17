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
import {
  DEFAULT_ANN_THRESHOLD,
  isHnswAvailable,
  resolveDenseStrategy,
} from './ann-strategy.mjs';

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
    rawAverage === null || rawAverage === undefined ? null : Number(rawAverage);

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
 * Build ANN index statistics for the corpus.
 *
 * @param {number} chunkCount - Total number of chunks in the corpus.
 * @returns {object} ANN stats: strategy, threshold, current_chunk_count, build_status, index_id, index_type.
 */
function buildAnnStats(chunkCount) {
  const strategy = resolveDenseStrategy({
    chunkCount,
    annThreshold: DEFAULT_ANN_THRESHOLD,
    indexStatus: 'missing',
    hnswAvailable: isHnswAvailable,
  });

  const isBelowThreshold = chunkCount < DEFAULT_ANN_THRESHOLD;

  let buildStatus;
  let indexId;
  let indexType;

  if (strategy === 'brute_force_cached' && isBelowThreshold) {
    buildStatus = 'not_applicable';
    indexId = null;
    indexType = null;
  } else if (strategy === 'hnsw') {
    buildStatus = 'ready';
    indexId = null;
    indexType = 'hnsw';
  } else {
    buildStatus = 'not_built';
    indexId = null;
    indexType = null;
  }

  return {
    strategy,
    threshold: DEFAULT_ANN_THRESHOLD,
    current_chunk_count: chunkCount,
    build_status: buildStatus,
    index_id: indexId,
    index_type: indexType,
  };
}

/**
 * Return aggregate statistics for the indexed corpus.
 *
 * @param {object} [options={}] - Tool options.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @returns {Promise<{ total_documents: number, total_chunks: number, total_families: number, last_build_timestamp: string | null, feedback_stats: object, ann: object }>} Index statistics.
 */
/** Metadata columns tracked for chunk-level coverage. */
const CHUNK_METADATA_COLUMNS = [
  'context_header',
  'symbol_name',
  'signature_text',
  'jsdoc_text',
  'export_type',
  'module_path',
  'arch_layer',
  'jsdoc_quality',
  'jsdoc_word_count',
  'cyclomatic_complexity',
  'test_coverage',
  'source_path_pattern',
];

/** Metadata columns tracked for document-level coverage. */
const DOCUMENT_METADATA_COLUMNS = [
  'arch_layer',
  'test_coverage',
  'source_path_pattern',
];

/**
 * Build chunk-level metadata coverage report.
 *
 * @param {import('better-sqlite3').Database} database - Open SQLite connection.
 * @param {number} totalChunks - Total number of chunks.
 * @returns {Record<string, { total: number, percent: number }>} Coverage per column.
 */
function buildChunkMetadataCoverage(database, totalChunks) {
  const coverage = {};
  for (const column of CHUNK_METADATA_COLUMNS) {
    const total = database
      .prepare(
        `SELECT COUNT(*) AS count FROM chunks WHERE ${column} IS NOT NULL`,
      )
      .get().count;
    coverage[column] = {
      total: Number(total),
      percent:
        totalChunks > 0 ? Number((Number(total) / totalChunks) * 100) : 0,
    };
  }
  return coverage;
}

/**
 * Build document-level metadata coverage report including value distribution.
 *
 * @param {import('better-sqlite3').Database} database - Open SQLite connection.
 * @param {number} totalDocuments - Total number of documents.
 * @returns {Record<string, { total: number, percent: number, distribution: Record<string, number> }>} Coverage per column.
 */
function buildDocumentMetadataCoverage(database, totalDocuments) {
  const coverage = {};
  for (const column of DOCUMENT_METADATA_COLUMNS) {
    const total = database
      .prepare(
        `SELECT COUNT(*) AS count FROM documents WHERE ${column} IS NOT NULL`,
      )
      .get().count;
    const distributionRows = database
      .prepare(
        `SELECT ${column} AS value, COUNT(*) AS count FROM documents WHERE ${column} IS NOT NULL GROUP BY ${column}`,
      )
      .all();
    const distribution = Object.fromEntries(
      distributionRows.map((row) => [
        String(row.value ?? 'null'),
        Number(row.count),
      ]),
    );
    coverage[column] = {
      total: Number(total),
      percent:
        totalDocuments > 0 ? Number((Number(total) / totalDocuments) * 100) : 0,
      distribution,
    };
  }
  return coverage;
}

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

    const result = {
      total_documents: Number(documentCount),
      total_chunks: Number(chunkCount),
      total_families: Number(familyCount),
      last_build_timestamp: asIsoTimestamp(lastIndexedAt),
      feedback_stats: buildFeedbackStats(database),
      ann: buildAnnStats(Number(chunkCount)),
    };

    if (options.include_metadata_coverage === true) {
      result.metadata_coverage = {
        chunks: buildChunkMetadataCoverage(database, result.total_chunks),
        documents: buildDocumentMetadataCoverage(
          database,
          result.total_documents,
        ),
      };
    }

    return result;
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
