import { requireString } from '../../agent-customization/mcp/mcp-utils.mjs';
import { queryDenseIndex } from '../../semantic-index/query-dense.mjs';
import { checkDenseReadiness } from '../../semantic-index/dense-readiness.mjs';
import { normalizeLimit, openCortexDatabase, readChunkRow, sanitizeFtsQuery } from './cortex-db.mjs';

let cachedDenseReadiness = null;

export async function searchCorpus(options = {}) {
  const rawQuery = requireString(options.query, 'query');
  const query = sanitizeFtsQuery(rawQuery);
  const limit = normalizeLimit(options.limit, 10);
  const family = typeof options.family === 'string' && options.family.trim() ? options.family.trim() : null;
  const useDense = options.use_dense !== false;

  if (!query) {
    if (!useDense) return createEmptyBm25Response({ family, limit, rawQuery });

    const readinessReport = await getDenseReadiness(options);
    if (readinessReport.state !== 'warm') {
      return createDegradedBm25Response({
        family,
        limit,
        query: rawQuery,
        readinessReport,
      });
    }

    return {
      alpha: normalizeAlpha(options.alpha),
      dense_state: 'warm',
      limit,
      ...(family ? { family } : {}),
      query: rawQuery,
      results: [],
      use_dense: true,
    };
  }

  if (useDense) {
    const readinessReport = await getDenseReadiness(options);
    if (readinessReport.state !== 'warm') {
      return createDegradedBm25Response({
        databasePath: options.databasePath,
        family,
        limit,
        query,
        readinessReport,
      });
    }

    const denseQuery = options.denseQuery ?? queryDenseIndex;
    const denseResult = await denseQuery({
      alpha: options.alpha,
      corpusDatabasePath: options.databasePath,
      dense: true,
      embeddingsDatabasePath: options.embeddingsDatabasePath,
      family,
      limit,
      modelDirectory: options.modelDirectory,
      modelId: options.modelId,
      query: rawQuery,
    });

    return {
      ...denseResult,
      dense_state: 'warm',
    };
  }

  return runBm25Search({ databasePath: options.databasePath, family, limit, query });
}

async function getDenseReadiness(options) {
  const readinessProbe = options.readinessProbe ?? checkDenseReadiness;
  const readinessOptions = {
    corpusDatabasePath: options.databasePath,
    embeddingsDatabasePath: options.embeddingsDatabasePath,
    modelDirectory: options.modelDirectory,
    modelId: options.modelId,
  };

  if (shouldBypassDenseReadinessCache()) {
    return readinessProbe(readinessOptions);
  }

  if (cachedDenseReadiness !== null) {
    return cachedDenseReadiness;
  }

  cachedDenseReadiness = await readinessProbe(readinessOptions);
  return cachedDenseReadiness;
}

function shouldBypassDenseReadinessCache() {
  const forcedState = typeof process.env.DENSE_FORCE_STATE === 'string' ? process.env.DENSE_FORCE_STATE.trim() : '';
  return forcedState === 'cold' || forcedState === 'model-only';
}

function createEmptyBm25Response({ family, limit, rawQuery }) {
  return {
    limit,
    ...(family ? { family } : {}),
    query: rawQuery,
    results: [],
    use_dense: false,
  };
}

function createDegradedBm25Response({ databasePath, family, limit, query, readinessReport }) {
  const bm25Response = query
    ? runBm25Search({ databasePath, family, limit, query })
    : createEmptyBm25Response({ family, limit, rawQuery: query });

  return {
    ...bm25Response,
    dense_degraded: true,
    dense_reason: normalizeDenseReason(readinessReport.reason, readinessReport.state),
    dense_state: readinessReport.state,
    use_dense: false,
  };
}

function normalizeDenseReason(reason, state) {
  const trimmedReason = typeof reason === 'string' ? reason.trim() : '';
  if (trimmedReason) return trimmedReason;
  return state === 'cold'
    ? 'Dense embeddings are unavailable because the model assets are absent.'
    : 'Dense embeddings are unavailable because the embeddings database is missing or incomplete.';
}

function normalizeAlpha(alpha) {
  const numericAlpha = Number(alpha ?? 0.5);
  return Number.isFinite(numericAlpha) ? numericAlpha : 0.5;
}

function runBm25Search({ databasePath, family, limit, query }) {
  const database = openCortexDatabase(databasePath);

  try {
    const familyFilter = family ? 'AND d.doc_family = @family' : '';
    const rows = database.prepare(`
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end, bm25(chunks_fts) AS score
      FROM chunks_fts
      JOIN chunks c ON c.chunk_id = chunks_fts.rowid
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE chunks_fts MATCH @query ${familyFilter}
      ORDER BY score
      LIMIT @limit
    `).all({ query, family, limit });

    return {
      query,
      limit,
      ...(family ? { family } : {}),
      use_dense: false,
      results: rows.map((row) => ({ ...readChunkRow(row), score: Number(row.score) })),
    };
  } finally {
    database.close();
  }
}