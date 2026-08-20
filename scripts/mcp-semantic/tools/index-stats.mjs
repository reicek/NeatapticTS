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
import { getTursoClient } from './cortex-db.mjs';
import {
  DEFAULT_ANN_THRESHOLD,
  resolveDenseStrategy,
} from './ann-strategy.mjs';

/** Weight applied to feedback boosts when blending them into ranking scores. */
export const FEEDBACK_WEIGHT = 1.0;

/** Half-life (in days) used for exponential time decay of feedback signals. */
export const FEEDBACK_HALF_LIFE_DAYS = 7;

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
  });

  const isBelowThreshold = chunkCount < DEFAULT_ANN_THRESHOLD;

  let buildStatus;
  let indexId;
  let indexType;

  if (isBelowThreshold) {
    buildStatus = 'not_applicable';
    indexId = null;
    indexType = null;
  } else {
    buildStatus = 'not_built';
    indexId = null;
    indexType = 'diskann';
  }

  return {
    strategy,
    threshold: DEFAULT_ANN_THRESHOLD,
    current_chunk_count: chunkCount,
    build_status: buildStatus,
    index_id: indexId,
    index_type: indexType,
    vector_type: 'F8_BLOB',
    quantization: '8-bit',
  };
}

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

/**
 * Aggregate feedback event and score statistics from the corpus database.
 *
 * @param {import('@libsql/client').Client} client - libSQL client.
 * @returns {Promise<object>} Feedback statistics.
 */
async function buildFeedbackStatsAsync(client) {
  const totalResult = await client.execute(
    'SELECT COUNT(*) AS count FROM feedback_events',
  );
  const totalEvents = Number(totalResult.rows[0].count);

  const eventsByTypeResult = await client.execute(
    'SELECT signal_type, COUNT(*) AS count FROM feedback_events GROUP BY signal_type',
  );
  const eventsByType = Object.fromEntries(
    eventsByTypeResult.rows.map((row) => [row.signal_type, Number(row.count)]),
  );

  const chunksResult = await client.execute(
    'SELECT COUNT(DISTINCT chunk_id) AS count FROM feedback_scores',
  );
  const chunksWithFeedback = Number(chunksResult.rows[0].count);

  const avgResult = await client.execute(
    'SELECT AVG(feedback_boost) AS average FROM feedback_scores',
  );
  const rawAverage = avgResult.rows[0].average;
  const averageFeedbackBoost =
    rawAverage === null || rawAverage === undefined ? null : Number(rawAverage);

  const lastRecomputedResult = await client.execute(
    'SELECT MAX(last_feedback_at) AS value FROM feedback_scores',
  );
  const lastRecomputedAt = lastRecomputedResult.rows[0].value ?? null;

  return {
    total_events: totalEvents,
    events_by_type: eventsByType,
    chunks_with_feedback: chunksWithFeedback,
    average_feedback_boost: averageFeedbackBoost,
    feedback_weight: FEEDBACK_WEIGHT,
    feedback_half_life_days: FEEDBACK_HALF_LIFE_DAYS,
    last_recomputed_at: lastRecomputedAt,
  };
}

/**
 * Build chunk-level metadata coverage report.
 *
 * @param {import('@libsql/client').Client} client - libSQL client.
 * @param {number} totalChunks - Total number of chunks.
 * @returns {Promise<Record<string, { total: number, percent: number }>>} Coverage per column.
 */
async function buildChunkMetadataCoverageAsync(client, totalChunks) {
  const coverage = {};
  for (const column of CHUNK_METADATA_COLUMNS) {
    const result = await client.execute({
      sql: `SELECT COUNT(*) AS count FROM chunks WHERE ${column} IS NOT NULL`,
    });
    const total = Number(result.rows[0].count);
    coverage[column] = {
      total,
      percent: totalChunks > 0 ? Number((total / totalChunks) * 100) : 0,
    };
  }
  return coverage;
}

/**
 * Build document-level metadata coverage report including value distribution.
 *
 * @param {import('@libsql/client').Client} client - libSQL client.
 * @param {number} totalDocuments - Total number of documents.
 * @returns {Promise<Record<string, { total: number, percent: number, distribution: Record<string, number> }>>} Coverage per column.
 */
async function buildDocumentMetadataCoverageAsync(client, totalDocuments) {
  const coverage = {};
  for (const column of DOCUMENT_METADATA_COLUMNS) {
    const totalResult = await client.execute({
      sql: `SELECT COUNT(*) AS count FROM documents WHERE ${column} IS NOT NULL`,
    });
    const total = Number(totalResult.rows[0].count);
    const distributionResult = await client.execute({
      sql: `SELECT ${column} AS value, COUNT(*) AS count FROM documents WHERE ${column} IS NOT NULL GROUP BY ${column}`,
    });
    const distribution = Object.fromEntries(
      distributionResult.rows.map((row) => [
        String(
          /* istanbul ignore next -- SQL filters NULL values */ row.value ??
            'null',
        ),
        Number(row.count),
      ]),
    );
    coverage[column] = {
      total,
      percent: totalDocuments > 0 ? Number((total / totalDocuments) * 100) : 0,
      distribution,
    };
  }
  return coverage;
}

/**
 * Return aggregate statistics for the indexed corpus.
 *
 * @param {object} [options={}] - Tool options.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @param {import('@libsql/client').Client} [options.client] - Pre-existing libSQL client.
 * @param {boolean} [options.include_metadata_coverage] - When true, compute per-column coverage.
 * @returns {Promise<{ total_documents: number, total_chunks: number, total_families: number, last_build_timestamp: string | null, feedback_stats: object, ann: object }>} Index statistics.
 */
export async function indexStats(options = {}) {
  const client = options.client ?? (await getTursoClient(options.databasePath));

  const docResult = await client.execute(
    'SELECT COUNT(*) AS count FROM documents',
  );
  const chunkResult = await client.execute(
    'SELECT COUNT(*) AS count FROM chunks',
  );
  const familyResult = await client.execute(
    'SELECT COUNT(DISTINCT doc_family) AS count FROM documents',
  );
  const lastIndexedResult = await client.execute(
    'SELECT MAX(indexed_at) AS value FROM documents',
  );

  const documentCount = Number(docResult.rows[0].count);
  const chunkCount = Number(chunkResult.rows[0].count);
  const familyCount = Number(familyResult.rows[0].count);
  const lastIndexedAt = lastIndexedResult.rows[0].value;

  const result = {
    total_documents: documentCount,
    total_chunks: chunkCount,
    total_families: familyCount,
    last_build_timestamp: asIsoTimestamp(lastIndexedAt),
    feedback_stats: await buildFeedbackStatsAsync(client),
    ann: buildAnnStats(chunkCount),
  };

  if (options.include_metadata_coverage === true) {
    result.metadata_coverage = {
      chunks: await buildChunkMetadataCoverageAsync(client, chunkCount),
      documents: await buildDocumentMetadataCoverageAsync(
        client,
        documentCount,
      ),
    };
  }

  return result;
}
