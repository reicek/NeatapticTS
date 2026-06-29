/**
 * @module metadata-filter-turso.test
 * @description Red tests for Phase 4 Step 05 — metadata filtering for Turso
 * vector queries (WHERE clause for brute-force, post-filter JOIN for ANN,
 * partial index support, 14-predicate compatibility).
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because the implementation does not exist yet (not because of syntax errors).
 *
 * Coverage targets:
 * - query-dense.mjs imports compileFilterToSqlAliased for vector query filter compilation
 * - query-dense.mjs loadDenseRowsBruteForce accepts compiledFilter and applies it as WHERE clause
 * - query-dense.mjs loadAnnRows accepts compiledFilter and applies it as post-filter on vector_top_k results
 * - query-dense.mjs queryDenseIndex accepts metadataFilter/compiledFilter option
 * - query-dense.mjs queryDenseIndex passes compiledFilter to loadDenseRows
 * - search-corpus.mjs passes compiledFilter to queryDenseIndex for SQL-level filtering
 * - search-corpus.mjs removes applyPostRetrievalFilter from dense path (No Deferred Cleanup)
 * - ann-index.mjs supports partial index creation with WHERE clause
 * - ann-index.mjs buildAnnIndex accepts a partial filter option
 * - All 14 predicate types are compatible with Turso vector query paths
 * - Filter SQL is Turso/libSQL compatible (parameterized, no SQLite-only functions)
 *
 * Pure .mjs test — runs via Jest ESM project (no ts-jest).
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { closeTursoClient } from '../../scripts/mcp-semantic/tools/cortex-db.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const METADATA_FILTER_PATH = path.resolve(
  __dirname,
  '..',
  'metadata-filter.mjs',
);
const QUERY_DENSE_PATH = path.resolve(__dirname, '..', 'query-dense.mjs');
const SEARCH_CORPUS_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  'scripts',
  'mcp-semantic',
  'tools',
  'search-corpus.mjs',
);
const ANN_INDEX_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  'scripts',
  'mcp-semantic',
  'tools',
  'ann-index.mjs',
);

/**
 * Read a source file as UTF-8 text.
 *
 * @param {string} filePath - Absolute path to the source file.
 * @returns {Promise<string>} File contents.
 */
async function readSource(filePath) {
  return readFile(filePath, 'utf8');
}

// ---------------------------------------------------------------------------
// Teardown — close any Turso clients that may have been opened
// ---------------------------------------------------------------------------

afterEach(async () => {
  await closeTursoClient();
});

// ---------------------------------------------------------------------------
// WHERE clause for brute-force (vector_distance_cos path)
// ---------------------------------------------------------------------------

describe('metadata-filter-turso: brute-force vector_distance_cos WHERE clause', () => {
  it('query-dense.mjs imports compileFilterToSqlAliased from metadata-filter.mjs', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).toMatch(/compileFilterToSqlAliased/i);
  });

  it('query-dense.mjs loadDenseRowsBruteForce accepts a compiledFilter parameter', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The brute-force function signature should destructure compiledFilter.
    expect(source).toMatch(/loadDenseRowsBruteForce[\s\S]*?compiledFilter/i);
  });

  it('query-dense.mjs loadDenseRowsBruteForce applies compiledFilter.sql as a WHERE clause in the SQL', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The brute-force SQL should incorporate the compiled filter SQL fragment
    // as an additional WHERE condition (AND ${compiledFilter.sql}).
    expect(source).toMatch(
      /loadDenseRowsBruteForce[\s\S]*?compiledFilter\.sql/i,
    );
  });
});

// ---------------------------------------------------------------------------
// Post-filter JOIN for ANN (vector_top_k path)
// ---------------------------------------------------------------------------

describe('metadata-filter-turso: ANN vector_top_k post-filter', () => {
  it('query-dense.mjs loadAnnRows accepts a compiledFilter parameter', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).toMatch(/loadAnnRows[\s\S]*?compiledFilter/i);
  });

  it('query-dense.mjs loadAnnRows applies compiledFilter.sql in the SQL after vector_top_k results', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The ANN SQL should apply the compiled filter as a WHERE or JOIN condition
    // on the vector_top_k results (not JS-side post-retrieval filtering).
    expect(source).toMatch(/loadAnnRows[\s\S]*?compiledFilter\.sql/i);
  });

  it('query-dense.mjs loadDenseRows passes compiledFilter to both loadAnnRows and loadDenseRowsBruteForce', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The orchestrating loadDenseRows function should accept and forward
    // compiledFilter to both the ANN and brute-force sub-functions.
    expect(source).toMatch(/loadDenseRows[\s\S]*?compiledFilter/i);
  });
});

// ---------------------------------------------------------------------------
// queryDenseIndex integration — accepts and forwards metadata filter
// ---------------------------------------------------------------------------

describe('metadata-filter-turso: queryDenseIndex accepts metadata filter', () => {
  it('query-dense.mjs queryDenseIndex accepts a compiledFilter or metadataFilter option', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // queryDenseIndex should read options.compiledFilter or options.metadataFilter
    // so search-corpus can pass the filter through.
    expect(source).toMatch(/options\.(compiledFilter|metadataFilter)/i);
  });

  it('query-dense.mjs queryDenseIndex passes compiledFilter to loadDenseRows', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The loadDenseRows call inside queryDenseIndex should include compiledFilter.
    expect(source).toMatch(/loadDenseRows[\s\S]*?compiledFilter/i);
  });
});

// ---------------------------------------------------------------------------
// search-corpus integration — passes compiledFilter to queryDenseIndex
// ---------------------------------------------------------------------------

describe('metadata-filter-turso: search-corpus passes compiledFilter to queryDenseIndex', () => {
  it('search-corpus.mjs passes compiledFilter to the queryDenseIndex call for dense search', async () => {
    const source = await readSource(SEARCH_CORPUS_PATH);
    // The denseQuery({...}) call should include compiledFilter as a parameter
    // so SQL-level filtering is applied inside queryDenseIndex. We check that
    // compiledFilter appears INSIDE the denseQuery({...}) call block (before
    // the closing brace). Using [^}]* prevents matching across brace boundaries.
    expect(source).toMatch(/denseQuery(?:Fn)?\(\{[^}]*compiledFilter/i);
  });

  it('search-corpus.mjs does not use applyPostRetrievalFilter as the sole dense path filter', async () => {
    const source = await readSource(SEARCH_CORPUS_PATH);
    // After implementation, the dense path should use SQL-level filtering via
    // compiledFilter passed to queryDenseIndex, not JS-side
    // applyPostRetrievalFilter on dense results. The old post-retrieval call
    // must be removed (No Deferred Cleanup).
    expect(source).not.toMatch(/applyPostRetrievalFilter[\s\S]*?denseResult/i);
  });
});

// ---------------------------------------------------------------------------
// Partial index support in ann-index.mjs
// ---------------------------------------------------------------------------

describe('metadata-filter-turso: partial index WHERE clause in ann-index.mjs', () => {
  it('ann-index.mjs supports partial index creation with a WHERE clause in CREATE INDEX SQL', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    // The index builder should have a CREATE INDEX statement that includes
    // a WHERE clause for partial indexing (pre-filtered ANN).
    expect(source).toMatch(/CREATE\s+INDEX[\s\S]*?WHERE/i);
  });

  it('ann-index.mjs buildAnnIndex accepts a partial filter or whereClause option', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    // The buildAnnIndex function should accept an option for the partial index
    // WHERE clause so callers can create pre-filtered ANN indexes.
    expect(source).toMatch(
      /partialFilter|partialIndexFilter|whereClause|indexWhereClause/i,
    );
  });
});

// ---------------------------------------------------------------------------
// 14-predicate compatibility with Turso vector query paths
// ---------------------------------------------------------------------------

describe('metadata-filter-turso: 14-predicate compatibility with Turso vector queries', () => {
  it('query-dense.mjs imports validateFilter from metadata-filter.mjs to validate metadata filters before compilation', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // query-dense.mjs should import validateFilter so metadata filters are
    // validated before being compiled to SQL and applied to vector queries.
    expect(source).toMatch(/validateFilter/i);
  });

  it('query-dense.mjs merges compiledFilter.params into the SQL args array for parameterized vector query filtering', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The compiled filter produces parameterized SQL (using ? placeholders)
    // and a params array. query-dense.mjs must merge compiledFilter.params
    // into the args array passed to the SQL execute call so all 14 predicate
    // types are handled with proper parameterization.
    expect(source).toMatch(/compiledFilter\.params/i);
  });
});

// ---------------------------------------------------------------------------
// Turso-compatible filter SQL in vector queries
// ---------------------------------------------------------------------------

describe('metadata-filter-turso: Turso-compatible SQL in vector queries', () => {
  it('query-dense.mjs applies compiledFilter.sql as an AND condition in the vector query WHERE clause', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The compiled filter SQL fragment should be incorporated as an AND
    // condition in the vector query SQL (not applied as a JS post-filter).
    // This ensures the filter runs server-side in Turso/libSQL.
    expect(source).toMatch(
      /AND[\s\S]*?compiledFilter\.sql|compiledFilter\.sql[\s\S]*?AND/i,
    );
  });

  it('query-dense.mjs merges compiledFilter.params into args before the SQL execute call', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The compiled filter params must be spread into the args array used by
    // the SQL execute call. This ensures filter values are parameterized
    // server-side, not interpolated into SQL strings.
    expect(source).toMatch(
      /args[\s\S]*?compiledFilter\.params|compiledFilter\.params[\s\S]*?args/i,
    );
  });
});
