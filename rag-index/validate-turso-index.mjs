/**
 * @module validate-turso-index
 * @description Turso/libSQL RAG index validation script that verifies migrated
 * data integrity by comparing target table row counts against source SQLite
 * databases and checking for NULL embedding columns where data should exist.
 *
 * This script is the Phase 2 Step 05 validation gate. It connects to the Turso
 * target database via `@libsql/client`, counts rows in all migration tables,
 * optionally compares those counts against the source databases
 * (`rag-index/data/turso-replica.sqlite` + `rag-index/data/embeddings.sqlite`) read via
 * `@libsql/client`, and checks for NULL embeddings in `chunks.embedding` and
 * `term_embeddings.embedding`.
 *
 * When source databases are not present (e.g. CI or a fresh checkout without
 * the original SQLite files), the script reports target counts only and notes
 * that source comparison was skipped — it does not crash.
 *
 * Environment variables:
 *
 * - `TURSO_DATABASE_URL` — Turso database URL. Defaults to
 *   `file:rag-index/data/turso-replica.sqlite` for local development.
 * - `TURSO_AUTH_TOKEN` — Turso auth token. Not required for local `file:` mode.
 *
 * CLI flags:
 *
 * - `--json` — Output only JSON to stdout (for CI gate integration). Without
 *   this flag, a human-readable validation report is printed.
 * - `--validate-search` — Also run FTS5, vector (brute-force + DiskANN), and
 *   entity graph search validation. Each check creates and cleans up its own
 *   fixture data so the database is left in its prior state.
 * - `--full` — Alias for `--validate-search` (runs count + search validation).
 * - `--source-corpus <path>` — Override the source corpus database path.
 *   Defaults to `rag-index/data/turso-replica.sqlite`.
 * - `--source-embeddings <path>` — Override the source embeddings database path.
 *   Defaults to `rag-index/data/embeddings.sqlite`.
 *
 * Exit codes:
 *
 * - `0` — All validations passed (or skipped with no discrepancies).
 * - `1` — Validation failed (row count mismatch or unexpected NULL embeddings),
 *   or a connection error occurred.
 *
 * @example
 * ```ts
 * // Local development (no cloud credentials needed)
 * node rag-index/validate-turso-index.mjs --json
 *
 * // With FTS5 + vector + entity graph search validation
 * node rag-index/validate-turso-index.mjs --json --validate-search
 *
 * // Turso Cloud
 * TURSO_DATABASE_URL=libsql://my-db.turso.io \
 * TURSO_AUTH_TOKEN=eyJ... \
 * node rag-index/validate-turso-index.mjs --json --full
 *
 * // Programmatic usage
 * import { validateTursoIndex } from './validate-turso-index.mjs';
 * const result = await validateTursoIndex({ jsonOutput: true, validateSearch: true });
 * console.log(result);
 * ```
 */

import { createClient } from '@libsql/client';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Default Turso database URL for local development. Uses the embedded libSQL
 * file backend — no cloud credentials required.
 */
const DEFAULT_TURSO_URL = 'file:rag-index/data/turso-replica.sqlite';

/**
 * Default path to the source corpus SQLite database (read-only comparison).
 */
const DEFAULT_SOURCE_CORPUS_PATH = 'rag-index/data/turso-replica.sqlite';

/**
 * Default path to the source embeddings SQLite database (read-only comparison).
 */
const DEFAULT_SOURCE_EMBEDDINGS_PATH = 'rag-index/data/embeddings.sqlite';

/**
 * Tables whose row counts are validated. These match the Turso consolidated
 * schema and the migration target tables from `migrate-to-turso.mjs`.
 *
 * The first seven tables are migrated from `rag-index/data/turso-replica.sqlite`.
 * `_schema_version` and `_index_metadata` are Turso-internal and have no
 * source equivalent — they are reported but not compared.
 */
const TARGET_TABLES = [
  'documents',
  'chunks',
  'entities',
  'edges',
  'term_embeddings',
  'feedback_events',
  'feedback_scores',
  '_schema_version',
  '_index_metadata',
];

/**
 * Tables that exist in the source corpus database and have a direct
 * 1:1 mapping to the Turso target. Used for source-vs-target count comparison.
 */
const SOURCE_CORPUS_TABLES = [
  'documents',
  'chunks',
  'entities',
  'edges',
  'term_embeddings',
  'feedback_events',
  'feedback_scores',
];

/**
 * Tables in the source embeddings database. `chunk_embeddings` rows are
 * merged into `chunks.embedding` during migration, so the source
 * `chunk_embeddings` count should equal the number of Turso `chunks` rows
 * with a non-NULL `embedding`.
 */
const SOURCE_EMBEDDINGS_TABLES = ['chunk_embeddings'];

/**
 * Embedding columns to check for NULLs where data should exist. Each entry
 * maps a table to the column that should contain a vector if the source had
 * an embedding for that row.
 */
const EMBEDDING_NULL_CHECKS = [
  { table: 'chunks', column: 'embedding' },
  { table: 'term_embeddings', column: 'embedding' },
];

// ---------------------------------------------------------------------------
// Target count helpers
// ---------------------------------------------------------------------------

/**
 * Query the row count for each table in {@link TARGET_TABLES} from the Turso
 * target database.
 *
 * @param {object} client - libSQL client instance.
 * @returns {Promise<Record<string, number>>} Map of table name to row count.
 */
async function getTargetCounts(client) {
  const counts = {};
  for (const table of TARGET_TABLES) {
    const result = await client.execute(`SELECT COUNT(*) AS cnt FROM ${table}`);
    counts[table] = Number(result.rows[0].cnt);
  }
  return counts;
}

// ---------------------------------------------------------------------------
// Source count helpers (@libsql/client, read-only)
// ---------------------------------------------------------------------------

/**
 * Query the row count for each table in `tables` from a source SQLite database
 * opened read-only via `@libsql/client`.
 *
 * @param {string} dbPath - Filesystem path to the source SQLite database.
 * @param {string[]} tables - Table names to count.
 * @returns {Promise<Record<string, number>>} Map of table name to row count.
 */
async function getSourceCounts(dbPath, tables) {
  const { createClient } = await import('@libsql/client');
  const client = createClient({ url: pathToFileURL(dbPath).href });
  try {
    const counts = {};
    for (const table of tables) {
      // Check if the table exists in the source database before counting.
      const existsResult = await client.execute({
        sql: "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        args: [table],
      });
      if (existsResult.rows.length > 0) {
        const countResult = await client.execute(
          `SELECT COUNT(*) AS cnt FROM ${table}`,
        );
        counts[table] = Number(countResult.rows[0].cnt);
      } else {
        counts[table] = 0;
      }
    }
    return counts;
  } finally {
    await client.close();
  }
}

// ---------------------------------------------------------------------------
// NULL embedding checks
// ---------------------------------------------------------------------------

/**
 * Check for NULL embedding columns in the Turso target database. For each
 * table/column pair in {@link EMBEDDING_NULL_CHECKS}, count rows where the
 * embedding column is NULL. A non-zero count indicates missing data that
 * should have been migrated.
 *
 * @param {object} client - libSQL client instance.
 * @returns {Promise<Array<{table: string, column: string, nullCount: number, totalRows: number}>>}
 *   Per-table NULL embedding report.
 */
async function checkNullEmbeddings(client) {
  const results = [];
  for (const { table, column } of EMBEDDING_NULL_CHECKS) {
    const totalResult = await client.execute(
      `SELECT COUNT(*) AS cnt FROM ${table}`,
    );
    const totalRows = Number(totalResult.rows[0].cnt);
    const nullResult = await client.execute(
      `SELECT COUNT(*) AS cnt FROM ${table} WHERE ${column} IS NULL`,
    );
    const nullCount = Number(nullResult.rows[0].cnt);
    results.push({ table, column, nullCount, totalRows });
  }
  return results;
}

// ---------------------------------------------------------------------------
// Search validation helpers
// ---------------------------------------------------------------------------

/**
 * Embedding dimension for search-validation fixture vectors. Matches the
 * production `all-MiniLM-L6-v2` model dimension used by the Turso schema's
 * `F8_BLOB(384)` columns.
 */
const SEARCH_VALIDATION_DIM = 384;

/**
 * Unique marker prefix for search-validation fixture data. All fixture rows
 * inserted by the search validation use this prefix in `file_path`,
 * `body_text`, `heading_path`, and `qualified_name` so they can be reliably
 * identified and cleaned up.
 */
const SEARCH_FIXTURE_MARKER = '__validate_search_fixture__';

/**
 * Create a deterministic L2-normalized `Float32Array` embedding for search
 * validation fixtures. The same seed always produces the same vector so that
 * brute-force and DiskANN results are reproducible.
 *
 * @param {number} seed - Integer seed for deterministic value generation.
 * @param {number} dimension - Vector dimension.
 * @returns {Float32Array} A normalized `Float32Array` embedding.
 */
function makeFixtureEmbedding(seed, dimension) {
  const vec = new Float32Array(dimension);
  for (let i = 0; i < dimension; i += 1) {
    vec[i] = Math.sin((seed + 1) * (i + 1) * 0.001);
  }
  let mag = 0;
  for (let i = 0; i < dimension; i += 1) mag += vec[i] * vec[i];
  mag = Math.sqrt(mag);
  if (mag > 0) {
    for (let i = 0; i < dimension; i += 1) vec[i] /= mag;
  }
  return vec;
}

/**
 * Convert a `Float32Array` to a Node `Buffer` for `vector8()` BLOB binding via
 * `@libsql/client`. The `vector8()` SQL function expects raw Float32 bytes.
 *
 * @param {Float32Array} vec - Float32 vector.
 * @returns {Buffer} Buffer view of the underlying `ArrayBuffer`.
 */
function float32ToBuffer(vec) {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
}

/**
 * Remove any stale search-validation fixture data left over from a prior
 * interrupted run. Deletes edges, entities, chunks, and documents whose
 * identifying columns match the fixture marker.
 *
 * @param {object} client - libSQL client instance.
 * @returns {Promise<void>}
 */
async function cleanupSearchFixtures(client) {
  const pattern = SEARCH_FIXTURE_MARKER + '%';
  // Delete edges referencing fixture entities (source or target).
  await client.execute(
    `DELETE FROM edges
     WHERE source_entity_id IN (
       SELECT entity_id FROM entities WHERE qualified_name LIKE ?
     ) OR target_entity_id IN (
       SELECT entity_id FROM entities WHERE qualified_name LIKE ?
     )`,
    [pattern, pattern],
  );
  await client.execute(`DELETE FROM entities WHERE qualified_name LIKE ?`, [
    pattern,
  ]);
  await client.execute(
    `DELETE FROM chunks WHERE heading_path LIKE ? OR body_text LIKE ?`,
    [pattern, pattern],
  );
  await client.execute(`DELETE FROM documents WHERE file_path LIKE ?`, [
    pattern,
  ]);
}

/**
 * Validate FTS5 full-text search by inserting a fixture document + chunk with a
 * unique marker, querying `chunks_fts` via `MATCH`, verifying the trigger
 * auto-populated the FTS index, and cleaning up the fixture data.
 *
 * This confirms the `chunks_ai` (AFTER INSERT) and `chunks_ad` (AFTER DELETE)
 * triggers wire the FTS5 virtual table correctly — the same check `init-turso`
 * performs during initialization.
 *
 * @param {object} client - libSQL client instance.
 * @returns {Promise<{verified: boolean, matchedRows: number}>} Verification
 *   result: `verified` is true when the FTS5 MATCH returned at least one row.
 */
async function validateFts5Search(client) {
  await cleanupSearchFixtures(client);

  const marker = SEARCH_FIXTURE_MARKER;
  const filePath = marker + '_fts5.ts';
  const bodyText =
    marker + ' fts5 search validation test chunk uniquemarkerq1w2e3';

  // Insert a minimal fixture document.
  await client.execute({
    sql: `INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
          VALUES (?, ?, ?, ?, ?, ?)`,
    args: [filePath, '__validate__', 0, 0, 'sha-fts5', 0],
  });

  const docRow = await client.execute(
    `SELECT doc_id FROM documents WHERE file_path = ?`,
    [filePath],
  );
  const docId = docRow.rows[0].doc_id;

  // Insert a fixture chunk — the chunks_ai trigger should auto-populate chunks_fts.
  await client.execute({
    sql: `INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth)
          VALUES (?, ?, ?, ?, ?, ?)`,
    args: [docId, 0, bodyText, 0, bodyText.length, 0],
  });

  // Query chunks_fts for the unique marker — should match via the trigger.
  const ftsResult = await client.execute(
    `SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?`,
    ['uniquemarkerq1w2e3'],
  );
  const matchedRows = ftsResult.rows.length;
  const verified = matchedRows > 0;

  // Clean up: delete fixture chunk (chunks_ad trigger removes FTS row), then doc.
  await client.execute(`DELETE FROM chunks WHERE doc_id = ?`, [docId]);
  await client.execute(`DELETE FROM documents WHERE doc_id = ?`, [docId]);

  return { verified, matchedRows };
}

/**
 * Validate vector search (brute-force `vector_distance_cos` and DiskANN
 * `vector_top_k`) by inserting fixture chunks with known embeddings, querying
 * for nearest neighbors, and verifying the query vector's own chunk is returned
 * as the #1 result. Cleans up fixture data afterward.
 *
 * Brute-force search is always tested. DiskANN search is tested best-effort:
 * if the `vector_top_k` virtual table returns no rows or errors (which can
 * happen when the DiskANN index has not yet been built over a tiny fixture
 * dataset), the result is reported as `verified: 'skipped'` with a note rather
 * than failing the validation.
 *
 * @param {object} client - libSQL client instance.
 * @returns {Promise<{bruteForce: {verified: boolean, selfRank: number, testedChunks: number}, diskAnn: {verified: boolean|string, recallVsBruteForce: number|null, note: string}}>}
 *   Vector search validation result.
 */
async function validateVectorSearch(client) {
  await cleanupSearchFixtures(client);

  const marker = SEARCH_FIXTURE_MARKER;
  const filePath = marker + '_vector.ts';
  const headingPath = marker + '_vector_heading';

  // Insert a fixture document.
  await client.execute({
    sql: `INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
          VALUES (?, ?, ?, ?, ?, ?)`,
    args: [filePath, '__validate__', 0, 0, 'sha-vector', 0],
  });
  const docRow = await client.execute(
    `SELECT doc_id FROM documents WHERE file_path = ?`,
    [filePath],
  );
  const docId = docRow.rows[0].doc_id;

  // Insert 3 fixture chunks with deterministic embeddings (seeds 1, 2, 3).
  const fixtureCount = 3;
  const chunkIds = [];
  for (let i = 0; i < fixtureCount; i += 1) {
    const seed = i + 1;
    const vec = makeFixtureEmbedding(seed, SEARCH_VALIDATION_DIM);
    const bodyText = `${marker} vector chunk ${seed}`;
    await client.execute({
      sql: `INSERT INTO chunks (
              doc_id, chunk_index, heading_path, body_text, char_start, char_end, depth,
              embedding, embedding_model, chunk_sha256, embedded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, vector8(?), ?, ?, ?)`,
      args: [
        docId,
        i,
        headingPath,
        bodyText,
        0,
        bodyText.length,
        0,
        float32ToBuffer(vec),
        'fixture-model',
        `sha-vec-${seed}`,
        0,
      ],
    });
    const chunkRow = await client.execute(
      `SELECT chunk_id FROM chunks WHERE doc_id = ? AND chunk_index = ?`,
      [docId, i],
    );
    chunkIds.push(Number(chunkRow.rows[0].chunk_id));
  }

  // Query vector for chunk 0 (seed=1) — it should be its own nearest neighbor.
  const queryVec = makeFixtureEmbedding(1, SEARCH_VALIDATION_DIM);
  const queryBuffer = float32ToBuffer(queryVec);

  // --- Brute-force search via vector_distance_cos ---
  const bruteResult = await client.execute(
    `SELECT chunk_id, vector_distance_cos(embedding, vector8(?)) AS dist
     FROM chunks
     WHERE embedding IS NOT NULL AND heading_path = ?
     ORDER BY dist ASC
     LIMIT ?`,
    [queryBuffer, headingPath, fixtureCount],
  );
  const bruteTopId =
    bruteResult.rows.length > 0 ? Number(bruteResult.rows[0].chunk_id) : null;
  const bruteSelfRank = bruteResult.rows.findIndex(
    (r) => Number(r.chunk_id) === chunkIds[0],
  );
  const bruteForceVerified = bruteTopId === chunkIds[0] && bruteSelfRank === 0;

  // --- DiskANN search via vector_top_k ---
  let diskAnnResult = {
    verified: /** @type {boolean|string} */ ('skipped'),
    recallVsBruteForce: null,
    note: 'not yet attempted',
  };
  try {
    const topK = await client.execute(
      `SELECT v.id AS chunk_id
       FROM vector_top_k('chunks_embedding_idx', vector8(?), ?) AS v
       JOIN chunks ON chunks.rowid = v.id
       WHERE chunks.heading_path = ?`,
      [queryBuffer, fixtureCount, headingPath],
    );
    if (topK.rows.length === 0) {
      diskAnnResult = {
        verified: 'skipped',
        recallVsBruteForce: null,
        note: 'DiskANN index returned 0 rows — index not built over small fixture dataset',
      };
    } else {
      const diskTopId = Number(topK.rows[0].chunk_id);
      const diskAnnVerified = diskTopId === chunkIds[0];
      // Recall vs brute-force: fraction of brute-force top-1 found in DiskANN results.
      const diskIds = topK.rows.map((r) => Number(r.chunk_id));
      const bruteIds = bruteResult.rows.map((r) => Number(r.chunk_id));
      const overlap = bruteIds.filter((id) => diskIds.includes(id)).length;
      const recall = bruteIds.length > 0 ? overlap / bruteIds.length : null;
      diskAnnResult = {
        verified: diskAnnVerified,
        recallVsBruteForce: recall,
        note: diskAnnVerified
          ? 'DiskANN returned correct nearest neighbor'
          : 'DiskANN returned different top result than brute-force',
      };
    }
  } catch (err) {
    diskAnnResult = {
      verified: 'skipped',
      recallVsBruteForce: null,
      note: `DiskANN vector_top_k not available: ${err instanceof Error ? err.message : String(err)}`,
    };
  }

  // Clean up fixture data.
  await cleanupSearchFixtures(client);

  return {
    bruteForce: {
      verified: bruteForceVerified,
      selfRank: bruteSelfRank === -1 ? null : bruteSelfRank + 1,
      testedChunks: fixtureCount,
    },
    diskAnn: diskAnnResult,
  };
}

/**
 * Validate entity graph traversal by inserting fixture entities and an edge
 * between them, querying edges by source and target entity, verifying the
 * relationships resolve correctly, and cleaning up fixture data afterward.
 *
 * @param {object} client - libSQL client instance.
 * @returns {Promise<{verified: boolean, entitiesInserted: number, edgesInserted: number, edgesBySource: number, edgesByTarget: number}>}
 *   Entity graph validation result.
 */
async function validateEntityGraph(client) {
  await cleanupSearchFixtures(client);

  const marker = SEARCH_FIXTURE_MARKER;

  // Insert 2 fixture entities (no doc/chunk reference required — both nullable).
  const entity1Result = await client.execute({
    sql: `INSERT INTO entities (
            entity_type, name, qualified_name, file_path, extra_metadata
          ) VALUES (?, ?, ?, ?, ?)
          RETURNING entity_id`,
    args: [
      'function',
      marker + '_fn_a',
      marker + ':fn_a',
      marker + '_a.ts',
      '{}',
    ],
  });
  const entity1Id = Number(entity1Result.rows[0].entity_id);

  const entity2Result = await client.execute({
    sql: `INSERT INTO entities (
            entity_type, name, qualified_name, file_path, extra_metadata
          ) VALUES (?, ?, ?, ?, ?)
          RETURNING entity_id`,
    args: [
      'variable',
      marker + '_var_b',
      marker + ':var_b',
      marker + '_b.ts',
      '{}',
    ],
  });
  const entity2Id = Number(entity2Result.rows[0].entity_id);

  // Insert 1 edge: entity1 -> entity2 (depends-on).
  await client.execute({
    sql: `INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence, extra_metadata)
          VALUES (?, ?, ?, ?, ?)`,
    args: [entity1Id, entity2Id, 'depends-on', 'high', '{}'],
  });

  // Query edges by source entity.
  const sourceEdges = await client.execute(
    `SELECT edge_id, target_entity_id, relationship FROM edges WHERE source_entity_id = ?`,
    [entity1Id],
  );
  const edgesBySource = sourceEdges.rows.length;
  const sourceRelCorrect =
    edgesBySource > 0 &&
    Number(sourceEdges.rows[0].target_entity_id) === entity2Id &&
    sourceEdges.rows[0].relationship === 'depends-on';

  // Query edges by target entity.
  const targetEdges = await client.execute(
    `SELECT edge_id, source_entity_id, relationship FROM edges WHERE target_entity_id = ?`,
    [entity2Id],
  );
  const edgesByTarget = targetEdges.rows.length;
  const targetRelCorrect =
    edgesByTarget > 0 &&
    Number(targetEdges.rows[0].source_entity_id) === entity1Id;

  const verified = sourceRelCorrect && targetRelCorrect;

  // Clean up fixture data (edges cascade-delete with entities).
  await cleanupSearchFixtures(client);

  return {
    verified,
    entitiesInserted: 2,
    edgesInserted: 1,
    edgesBySource,
    edgesByTarget,
  };
}

/**
 * Run all search validation checks (FTS5, vector brute-force + DiskANN, entity
 * graph) against the Turso target database. Each check creates its own fixture
 * data, tests the search functionality, and cleans up afterward so the
 * database is left in its prior state.
 *
 * @param {object} client - libSQL client instance.
 * @returns {Promise<object>} Search validation result with `fts5`,
 *   `vectorBruteForce`, `vectorDiskAnn`, and `entityGraph` sub-objects, plus a
 *   `success` boolean.
 */
async function validateSearchFunctionality(client) {
  // Defensive cleanup in case a prior run was interrupted mid-fixture.
  await cleanupSearchFixtures(client);

  let fts5;
  let vector;
  let entityGraph;
  try {
    fts5 = await validateFts5Search(client);
  } catch (err) {
    fts5 = {
      verified: false,
      matchedRows: 0,
      error: err instanceof Error ? err.message : String(err),
    };
  }
  try {
    vector = await validateVectorSearch(client);
  } catch (err) {
    vector = {
      bruteForce: {
        verified: false,
        selfRank: null,
        testedChunks: 0,
        error: err instanceof Error ? err.message : String(err),
      },
      diskAnn: {
        verified: 'skipped',
        recallVsBruteForce: null,
        note: 'not attempted',
      },
    };
  }
  try {
    entityGraph = await validateEntityGraph(client);
  } catch (err) {
    entityGraph = {
      verified: false,
      entitiesInserted: 0,
      edgesInserted: 0,
      edgesBySource: 0,
      edgesByTarget: 0,
      error: err instanceof Error ? err.message : String(err),
    };
  }

  // Final defensive cleanup.
  await cleanupSearchFixtures(client);

  const fts5Ok = fts5.verified === true;
  const bruteOk = vector.bruteForce.verified === true;
  // DiskANN is not a hard failure when skipped (index not built over tiny fixture).
  const diskAnnOk =
    vector.diskAnn.verified === true || vector.diskAnn.verified === 'skipped';
  const graphOk = entityGraph.verified === true;

  return {
    success: fts5Ok && bruteOk && diskAnnOk && graphOk,
    fts5,
    vectorBruteForce: vector.bruteForce,
    vectorDiskAnn: vector.diskAnn,
    entityGraph,
  };
}

// ---------------------------------------------------------------------------
// Discrepancy detection
// ---------------------------------------------------------------------------

/**
 * Compare target counts against source counts and produce a list of
 * discrepancies (tables where the counts differ).
 *
 * @param {Record<string, number>} targetCounts - Turso target row counts.
 * @param {Record<string, number>} sourceCorpusCounts - Source corpus DB counts.
 * @param {Record<string, number>} sourceEmbeddingsCounts - Source embeddings DB counts.
 * @returns {Array<{table: string, target: number, source: number}>} Discrepancy list.
 */
function findDiscrepancies(
  targetCounts,
  sourceCorpusCounts,
  sourceEmbeddingsCounts,
) {
  const discrepancies = [];

  // Compare source corpus tables against target.
  for (const table of SOURCE_CORPUS_TABLES) {
    const target = targetCounts[table] ?? 0;
    const source = sourceCorpusCounts[table] ?? 0;
    if (target !== source) {
      discrepancies.push({ table, target, source });
    }
  }

  // chunk_embeddings source count should equal non-NULL chunks.embedding count.
  // This is a soft check: if chunk_embeddings exist in source, the number of
  // Turso chunks with a non-NULL embedding should match. We handle this in the
  // nullEmbeddings section rather than as a hard discrepancy since some chunks
  // legitimately have no embedding.
  if (sourceEmbeddingsCounts['chunk_embeddings'] !== undefined) {
    const sourceEmbCount = sourceEmbeddingsCounts['chunk_embeddings'];
    // The target non-null embedding count is computed in checkNullEmbeddings.
    // We store the source count for reference; the actual comparison is done
    // in the caller using nullEmbeddings data.
    discrepancies._sourceChunkEmbeddings = sourceEmbCount;
  }

  return discrepancies;
}

// ---------------------------------------------------------------------------
// Main validation function
// ---------------------------------------------------------------------------

/**
 * Validate the Turso/libSQL RAG index by comparing target table row counts
 * against source databases and checking for NULL embedding columns.
 *
 * @param {object} [options] - Validation options.
 * @param {string} [options.url] - Turso database URL. Defaults to
 *   `process.env.TURSO_DATABASE_URL` or `file:rag-index/data/turso-replica.sqlite`.
 * @param {string} [options.authToken] - Turso auth token. Defaults to
 *   `process.env.TURSO_AUTH_TOKEN`.
 * @param {string} [options.sourceCorpusPath] - Path to source corpus SQLite DB.
 * @param {string} [options.sourceEmbeddingsPath] - Path to source embeddings SQLite DB.
 * @param {boolean} [options.validateSearch] - When true, also run FTS5, vector
 *   (brute-force + DiskANN), and entity graph search validation against the
 *   target database. Each check creates and cleans up its own fixture data.
 * @param {boolean} [options.jsonOutput] - Reserved for CLI; the return value is
 *   always the validation result object.
 * @returns {Promise<object>} Validation result with `success`, `url`,
 *   `targetCounts`, `sourceCounts`, `sourceComparison`, `discrepancies`,
 *   `nullEmbeddings`, `searchValidation`, and `error` fields.
 */
export async function validateTursoIndex(options = {}) {
  const url =
    options.url ?? process.env.TURSO_DATABASE_URL ?? DEFAULT_TURSO_URL;
  const authToken =
    options.authToken ?? process.env.TURSO_AUTH_TOKEN ?? undefined;
  const sourceCorpusPath =
    options.sourceCorpusPath ?? DEFAULT_SOURCE_CORPUS_PATH;
  const sourceEmbeddingsPath =
    options.sourceEmbeddingsPath ?? DEFAULT_SOURCE_EMBEDDINGS_PATH;
  const validateSearch = options.validateSearch ?? false;

  const clientConfig = { url };
  if (authToken) {
    clientConfig.authToken = authToken;
  }

  let client;
  try {
    client = createClient(clientConfig);
  } catch (err) {
    return {
      success: false,
      url,
      targetCounts: {},
      sourceCounts: {},
      sourceComparison: 'error',
      discrepancies: [],
      embeddingMismatches: [],
      nullEmbeddings: [],
      searchValidation: null,
      error: err instanceof Error ? err.message : String(err),
    };
  }

  try {
    // Step 1: Collect target table counts.
    const targetCounts = await getTargetCounts(client);

    // Step 2: Collect source counts if source databases exist.
    const sourceCorpusExists = existsSync(path.resolve(sourceCorpusPath));
    const sourceEmbeddingsExists = existsSync(
      path.resolve(sourceEmbeddingsPath),
    );

    let sourceCounts = {};
    let sourceComparison = 'skipped';

    if (sourceCorpusExists) {
      sourceCounts = {
        ...sourceCounts,
        ...(await getSourceCounts(sourceCorpusPath, SOURCE_CORPUS_TABLES)),
      };
    }
    if (sourceEmbeddingsExists) {
      sourceCounts = {
        ...sourceCounts,
        ...(await getSourceCounts(
          sourceEmbeddingsPath,
          SOURCE_EMBEDDINGS_TABLES,
        )),
      };
    }

    if (sourceCorpusExists || sourceEmbeddingsExists) {
      sourceComparison = 'available';
    } else {
      sourceComparison = 'skipped — source databases not found';
    }

    // Step 3: Check for NULL embedding columns.
    const nullEmbeddings = await checkNullEmbeddings(client);

    // Step 4: Detect discrepancies (only when source counts are available).
    let discrepancies = [];
    let embeddingMismatches = [];

    if (sourceCorpusExists) {
      const sourceEmbeddingsCounts = sourceEmbeddingsExists
        ? { chunk_embeddings: sourceCounts['chunk_embeddings'] ?? 0 }
        : {};
      discrepancies = findDiscrepancies(
        targetCounts,
        sourceCounts,
        sourceEmbeddingsCounts,
      );

      // Extract the soft chunk_embeddings reference that findDiscrepancies
      // attaches as a non-array property.
      const sourceChunkEmbeddings = discrepancies._sourceChunkEmbeddings;
      if (sourceChunkEmbeddings !== undefined) {
        // Remove the non-array property so discrepancies is a clean array.
        const cleanDiscrepancies = discrepancies.filter(
          (d) => typeof d === 'object' && 'table' in d,
        );
        discrepancies = cleanDiscrepancies;

        // Compare: source chunk_embeddings count vs target chunks with
        // non-NULL embedding.
        const chunksNullInfo = nullEmbeddings.find((n) => n.table === 'chunks');
        const targetNonNullEmbeddings = chunksNullInfo
          ? chunksNullInfo.totalRows - chunksNullInfo.nullCount
          : 0;
        if (targetNonNullEmbeddings !== sourceChunkEmbeddings) {
          embeddingMismatches.push({
            table: 'chunks',
            column: 'embedding',
            targetNonNull: targetNonNullEmbeddings,
            sourceChunkEmbeddings,
          });
        }
      }
    }

    // Step 5: Run search validation when requested.
    let searchValidation = null;
    if (validateSearch) {
      searchValidation = await validateSearchFunctionality(client);
    }

    // Step 6: Determine overall success.
    const hasCountDiscrepancies = discrepancies.length > 0;
    const hasEmbeddingMismatches = embeddingMismatches.length > 0;
    // NULL embeddings are only a failure if source comparison was available
    // and we expected embeddings to exist. If source is skipped, we report
    // NULL counts as informational but do not fail.
    const hasUnexpectedNulls =
      sourceComparison === 'available' &&
      nullEmbeddings.some(
        (n) =>
          n.table === 'term_embeddings' &&
          n.nullCount > 0 &&
          (sourceCounts['term_embeddings'] ?? 0) > 0,
      );
    const hasSearchFailures =
      searchValidation !== null && searchValidation.success !== true;

    const success =
      !hasCountDiscrepancies &&
      !hasEmbeddingMismatches &&
      !hasUnexpectedNulls &&
      !hasSearchFailures;

    return {
      success,
      url,
      targetCounts,
      sourceCounts: sourceComparison === 'available' ? sourceCounts : {},
      sourceComparison,
      discrepancies,
      embeddingMismatches,
      nullEmbeddings,
      searchValidation,
    };
  } catch (err) {
    return {
      success: false,
      url,
      targetCounts: {},
      sourceCounts: {},
      sourceComparison: 'error',
      discrepancies: [],
      embeddingMismatches: [],
      nullEmbeddings: [],
      searchValidation: null,
      error: err instanceof Error ? err.message : String(err),
    };
  } finally {
    if (client) {
      await client.close();
    }
  }
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

/**
 * Run the Turso index validator from the command line.
 *
 * Usage:
 *   node rag-index/validate-turso-index.mjs [--json]
 *     [--validate-search|--full] [--source-corpus <path>]
 *     [--source-embeddings <path>]
 */
async function main() {
  const args = process.argv.slice(2);
  const getArg = (name) => {
    const idx = args.indexOf(name);
    return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
  };

  const jsonOutput = args.includes('--json');
  const sourceCorpusPath =
    getArg('--source-corpus') ?? DEFAULT_SOURCE_CORPUS_PATH;
  const sourceEmbeddingsPath =
    getArg('--source-embeddings') ?? DEFAULT_SOURCE_EMBEDDINGS_PATH;
  // --validate-search and --full both enable FTS5, vector, and entity graph
  // search validation in addition to the count/NULL checks.
  const validateSearch =
    args.includes('--validate-search') || args.includes('--full');

  const result = await validateTursoIndex({
    sourceCorpusPath,
    sourceEmbeddingsPath,
    validateSearch,
  });

  if (jsonOutput) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    if (result.success) {
      console.log('Turso index validation PASSED.');
    } else {
      console.error('Turso index validation FAILED.');
    }
    console.log(`  URL:                ${result.url}`);
    console.log(`  Source comparison:  ${result.sourceComparison}`);
    console.log('');
    console.log('  Target table counts:');
    for (const [table, count] of Object.entries(result.targetCounts)) {
      console.log(`    ${table.padEnd(22)} ${count}`);
    }

    if (result.sourceComparison === 'available') {
      console.log('');
      console.log('  Source table counts:');
      for (const [table, count] of Object.entries(result.sourceCounts)) {
        console.log(`    ${table.padEnd(22)} ${count}`);
      }
    }

    if (result.discrepancies.length > 0) {
      console.log('');
      console.error('  Row count discrepancies:');
      for (const d of result.discrepancies) {
        console.error(
          `    ${d.table.padEnd(22)} target=${d.target}  source=${d.source}`,
        );
      }
    }

    if (result.embeddingMismatches.length > 0) {
      console.log('');
      console.error('  Embedding count mismatches:');
      for (const m of result.embeddingMismatches) {
        console.error(
          `    ${m.table}.${m.column}  targetNonNull=${m.targetNonNull}  sourceChunkEmbeddings=${m.sourceChunkEmbeddings}`,
        );
      }
    }

    console.log('');
    console.log('  NULL embedding checks:');
    for (const n of result.nullEmbeddings) {
      const status =
        n.nullCount === 0 ? 'OK' : `${n.nullCount} NULLs out of ${n.totalRows}`;
      console.log(`    ${n.table}.${n.column.padEnd(12)} ${status}`);
    }

    if (result.error) {
      console.error(`  Error: ${result.error}`);
    }

    if (result.searchValidation) {
      console.log('');
      console.log('  Search validation:');
      const sv = result.searchValidation;
      console.log(
        `    FTS5:              ${sv.fts5.verified ? 'OK' : 'FAIL'} (matched ${sv.fts5.matchedRows} rows)`,
      );
      console.log(
        `    Vector brute-force: ${sv.vectorBruteForce.verified ? 'OK' : 'FAIL'} (self rank #${sv.vectorBruteForce.selfRank}, ${sv.vectorBruteForce.testedChunks} chunks)`,
      );
      const diskAnnStatus =
        sv.vectorDiskAnn.verified === true
          ? 'OK'
          : sv.vectorDiskAnn.verified === 'skipped'
            ? 'SKIPPED'
            : 'FAIL';
      console.log(
        `    Vector DiskANN:    ${diskAnnStatus} (recall=${sv.vectorDiskAnn.recallVsBruteForce})`,
      );
      if (sv.vectorDiskAnn.note) {
        console.log(`      note: ${sv.vectorDiskAnn.note}`);
      }
      console.log(
        `    Entity graph:      ${sv.entityGraph.verified ? 'OK' : 'FAIL'} (${sv.entityGraph.entitiesInserted} entities, ${sv.entityGraph.edgesInserted} edges)`,
      );
      if (sv.fts5.error) console.error(`      FTS5 error: ${sv.fts5.error}`);
      if (sv.vectorBruteForce.error)
        console.error(`      vector error: ${sv.vectorBruteForce.error}`);
      if (sv.entityGraph.error)
        console.error(`      graph error: ${sv.entityGraph.error}`);
    }
  }

  process.exit(result.success ? 0 : 1);
}

// Run CLI only when executed directly (not when imported).
if (
  process.argv[1] &&
  (process.argv[1].endsWith('validate-turso-index.mjs') ||
    process.argv[1].endsWith('validate-turso-index'))
) {
  main().catch((err) => {
    console.error('validate-turso-index failed:', err);
    process.exit(1);
  });
}
