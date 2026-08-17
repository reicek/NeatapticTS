import { jest } from '@jest/globals';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// ---------------------------------------------------------------------------
// Import test helpers BEFORE mocking @libsql/client so they get the real impl
// ---------------------------------------------------------------------------
const { createSchemaClient } = await import(
  './__tests__/turso-test-helpers.mjs'
);

// ---------------------------------------------------------------------------
// Mock state
// ---------------------------------------------------------------------------
let throwOnCreate = false;
let targetClientRef = { current: null };
let sourceClientRef = { current: null };
let mockExistsSyncFn = () => false;

// ---------------------------------------------------------------------------
// Mock node:fs for existsSync
// ---------------------------------------------------------------------------
jest.unstable_mockModule('node:fs', () => ({
  existsSync: (...args) => mockExistsSyncFn(...args),
}));

// ---------------------------------------------------------------------------
// Mock @libsql/client
// ---------------------------------------------------------------------------
jest.unstable_mockModule('@libsql/client', () => ({
  createClient: (config) => {
    if (throwOnCreate) throw new Error('createClient failed');
    const url = config?.url ?? '';
    // Source DBs use file: URLs (from pathToFileURL in getSourceCounts)
    if (url.startsWith('file:')) return sourceClientRef.current;
    // Target uses :memory: or other non-file URLs
    return targetClientRef.current;
  },
}));

// ---------------------------------------------------------------------------
// Import module under test AFTER mocks are set up
// ---------------------------------------------------------------------------
const { validateTursoIndex } = await import('./validate-turso-index.mjs');

// ---------------------------------------------------------------------------
// Mock client factories
// ---------------------------------------------------------------------------
function createMockTargetClient(config = {}) {
  const tableCounts = config.tableCounts ?? {};
  const nullCounts = config.nullCounts ?? {};

  return {
    execute: jest.fn(async (sqlOrObj) => {
      const sql = typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;

      // NULL count query
      const nullMatch = sql.match(
        /^SELECT COUNT\(\*\) AS cnt FROM (\w+) WHERE (\w+) IS NULL$/,
      );
      if (nullMatch) {
        const key = `${nullMatch[1]}.${nullMatch[2]}`;
        return { rows: [{ cnt: nullCounts[key] ?? 0 }] };
      }

      // Total count query
      const countMatch = sql.match(/^SELECT COUNT\(\*\) AS cnt FROM (\w+)$/);
      if (countMatch) {
        return { rows: [{ cnt: tableCounts[countMatch[1]] ?? 0 }] };
      }

      return { rows: [] };
    }),
    batch: jest.fn(async () => []),
    close: jest.fn(async () => {}),
  };
}

function createMockSourceClient(config = {}) {
  const tableCounts = config.tableCounts ?? {};
  const existingTables = new Set(
    config.existingTables ?? Object.keys(tableCounts),
  );

  return {
    execute: jest.fn(async (sqlOrObj) => {
      const sql = typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
      const args = typeof sqlOrObj === 'string' ? [] : (sqlOrObj.args ?? []);

      // sqlite_master check
      if (sql.includes('sqlite_master')) {
        const tableName = args[0];
        if (existingTables.has(tableName)) {
          return { rows: [{ name: tableName }] };
        }
        return { rows: [] };
      }

      // COUNT(*) query
      const countMatch = sql.match(/^SELECT COUNT\(\*\) AS cnt FROM (\w+)$/);
      if (countMatch) {
        return { rows: [{ cnt: tableCounts[countMatch[1]] ?? 0 }] };
      }

      return { rows: [] };
    }),
    close: jest.fn(async () => {}),
  };
}

// ---------------------------------------------------------------------------
// Full mock client for search validation (handles all SQL patterns)
// ---------------------------------------------------------------------------
function createSearchMockClient(config = {}) {
  const tableCounts = config.tableCounts ?? {};
  const nullCounts = config.nullCounts ?? {};
  const fts5Rows = config.fts5Rows ?? [{ rowid: 1 }];
  const bruteForceRows = config.bruteForceRows ?? null;
  const diskAnnRows = config.diskAnnRows ?? null; // null = throw, [] = empty
  const throwOnFts5 = config.throwOnFts5 ?? false;
  const throwOnVector = config.throwOnVector ?? false;
  const throwOnEntityGraph = config.throwOnEntityGraph ?? false;

  let docIdCounter = 10;
  let chunkIdCounter = 100;
  let entityIdCounter = 200;
  const chunkIds = [];
  const entityIds = [];

  return {
    execute: jest.fn(async (sqlOrObj) => {
      const sql = typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
      const args = typeof sqlOrObj === 'string' ? [] : (sqlOrObj.args ?? []);

      // DELETE queries (cleanup)
      if (sql.startsWith('DELETE')) {
        return { rows: [] };
      }

      // NULL count query
      const nullMatch = sql.match(
        /^SELECT COUNT\(\*\) AS cnt FROM (\w+) WHERE (\w+) IS NULL$/,
      );
      if (nullMatch) {
        const key = `${nullMatch[1]}.${nullMatch[2]}`;
        return { rows: [{ cnt: nullCounts[key] ?? 0 }] };
      }

      // Total count query
      const countMatch = sql.match(/^SELECT COUNT\(\*\) AS cnt FROM (\w+)$/);
      if (countMatch) {
        return { rows: [{ cnt: tableCounts[countMatch[1]] ?? 0 }] };
      }

      // INSERT INTO documents
      if (sql.includes('INSERT INTO documents')) {
        return { rows: [] };
      }

      // SELECT doc_id FROM documents WHERE file_path = ?
      if (sql.includes('SELECT doc_id FROM documents WHERE file_path')) {
        return { rows: [{ doc_id: ++docIdCounter }] };
      }

      // INSERT INTO chunks
      if (sql.includes('INSERT INTO chunks') && !sql.startsWith('DELETE')) {
        return { rows: [] };
      }

      // SELECT chunk_id FROM chunks WHERE doc_id = ? AND chunk_index = ?
      if (sql.includes('SELECT chunk_id FROM chunks WHERE doc_id')) {
        const id = ++chunkIdCounter;
        chunkIds.push(id);
        return { rows: [{ chunk_id: id }] };
      }

      // FTS5 MATCH query
      if (sql.includes('chunks_fts MATCH')) {
        if (throwOnFts5) throw new Error('FTS5 search error');
        return { rows: fts5Rows };
      }

      // DELETE FROM chunks WHERE doc_id
      if (sql.includes('DELETE FROM chunks WHERE doc_id')) {
        return { rows: [] };
      }

      // DELETE FROM documents WHERE doc_id
      if (sql.includes('DELETE FROM documents WHERE doc_id')) {
        return { rows: [] };
      }

      // Brute-force vector search
      if (sql.includes('vector_distance_cos')) {
        if (throwOnVector) throw new Error('Vector search error');
        if (bruteForceRows !== null) return { rows: bruteForceRows };
        return { rows: [{ chunk_id: chunkIds[0] ?? 1, dist: 0 }] };
      }

      // DiskANN vector search
      if (sql.includes('vector_top_k')) {
        if (diskAnnRows === null) throw new Error('vector_top_k not available');
        return { rows: diskAnnRows };
      }

      // INSERT INTO entities ... RETURNING entity_id
      if (sql.includes('INSERT INTO entities')) {
        if (throwOnEntityGraph) throw new Error('Entity graph error');
        const id = ++entityIdCounter;
        entityIds.push(id);
        return { rows: [{ entity_id: id }] };
      }

      // INSERT INTO edges
      if (sql.includes('INSERT INTO edges')) {
        return { rows: [] };
      }

      // SELECT edge_id, target_entity_id, relationship FROM edges WHERE source_entity_id
      if (sql.includes('SELECT edge_id, target_entity_id')) {
        return {
          rows: [
            {
              edge_id: 1,
              target_entity_id: entityIds[1] ?? 201,
              relationship: 'depends-on',
            },
          ],
        };
      }

      // SELECT edge_id, source_entity_id, relationship FROM edges WHERE target_entity_id
      if (sql.includes('SELECT edge_id, source_entity_id')) {
        return {
          rows: [
            { edge_id: 1, source_entity_id: entityIds[0] ?? 200 },
          ],
        };
      }

      return { rows: [] };
    }),
    batch: jest.fn(async () => []),
    close: jest.fn(async () => {}),
  };
}

// ---------------------------------------------------------------------------
// Helper to create a real in-memory schema client with close as no-op
// ---------------------------------------------------------------------------
async function createRealSearchClient() {
  const client = await createSchemaClient();
  client.close = async () => {};
  return client;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe('validate-turso-index.mjs', () => {
  beforeEach(() => {
    throwOnCreate = false;
    targetClientRef.current = null;
    sourceClientRef.current = null;
    mockExistsSyncFn = () => false;
  });

  describe('validateTursoIndex', () => {
    it('returns error when createClient throws', async () => {
      throwOnCreate = true;
      const result = await validateTursoIndex({ url: ':memory:' });
      expect(result.success).toBe(false);
      expect(result.error).toBe('createClient failed');
      expect(result.sourceComparison).toBe('error');
    });

    it('skips source comparison when source databases not found', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: { documents: 5, chunks: 10 },
      });
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({ url: ':memory:' });
      expect(result.success).toBe(true);
      expect(result.sourceComparison).toBe('skipped — source databases not found');
      expect(result.sourceCounts).toEqual({});
      expect(result.discrepancies).toEqual([]);
      expect(result.nullEmbeddings).toHaveLength(2);
    });

    it('compares source corpus counts when source exists', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: {
          documents: 5,
          chunks: 10,
          entities: 3,
          edges: 2,
          term_embeddings: 4,
          feedback_events: 1,
          feedback_scores: 1,
          _schema_version: 1,
          _index_metadata: 1,
        },
        nullCounts: { 'chunks.embedding': 0, 'term_embeddings.embedding': 0 },
      });
      sourceClientRef.current = createMockSourceClient({
        tableCounts: {
          documents: 5,
          chunks: 10,
          entities: 3,
          edges: 2,
          term_embeddings: 4,
          feedback_events: 1,
          feedback_scores: 1,
        },
      });
      mockExistsSyncFn = (p) => String(p).includes('turso-replica');

      const result = await validateTursoIndex({
        url: ':memory:',
        sourceCorpusPath: 'fake/turso-replica.sqlite',
        sourceEmbeddingsPath: 'fake/embeddings.sqlite',
      });
      expect(result.success).toBe(true);
      expect(result.sourceComparison).toBe('available');
      expect(result.discrepancies).toEqual([]);
    });

    it('detects discrepancies when counts differ', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: {
          documents: 5,
          chunks: 10,
          entities: 3,
          edges: 2,
          term_embeddings: 4,
          feedback_events: 1,
          feedback_scores: 1,
          _schema_version: 1,
          _index_metadata: 1,
        },
        nullCounts: { 'chunks.embedding': 0, 'term_embeddings.embedding': 0 },
      });
      sourceClientRef.current = createMockSourceClient({
        tableCounts: {
          documents: 3, // different
          chunks: 10,
          entities: 3,
          edges: 2,
          term_embeddings: 4,
          feedback_events: 1,
          feedback_scores: 1,
        },
      });
      mockExistsSyncFn = (p) => String(p).includes('turso-replica');

      const result = await validateTursoIndex({
        url: ':memory:',
        sourceCorpusPath: 'fake/turso-replica.sqlite',
        sourceEmbeddingsPath: 'fake/embeddings.sqlite',
      });
      expect(result.success).toBe(false);
      expect(result.discrepancies).toContainEqual({
        table: 'documents',
        target: 5,
        source: 3,
      });
    });

    it('detects embedding mismatches when source embeddings count differs', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: {
          documents: 5,
          chunks: 10,
          entities: 0,
          edges: 0,
          term_embeddings: 0,
          feedback_events: 0,
          feedback_scores: 0,
          _schema_version: 1,
          _index_metadata: 1,
        },
        nullCounts: {
          'chunks.embedding': 5, // 10 total - 5 null = 5 non-null
          'term_embeddings.embedding': 0,
        },
      });
      // Source has different chunk_embeddings count
      sourceClientRef.current = createMockSourceClient({
        tableCounts: {
          documents: 5,
          chunks: 10,
          entities: 0,
          edges: 0,
          term_embeddings: 0,
          feedback_events: 0,
          feedback_scores: 0,
          chunk_embeddings: 8, // different from 5 non-null
        },
      });
      mockExistsSyncFn = () => true;

      const result = await validateTursoIndex({
        url: ':memory:',
        sourceCorpusPath: 'fake/turso-replica.sqlite',
        sourceEmbeddingsPath: 'fake/embeddings.sqlite',
      });
      expect(result.success).toBe(false);
      expect(result.embeddingMismatches).toContainEqual({
        table: 'chunks',
        column: 'embedding',
        targetNonNull: 5,
        sourceChunkEmbeddings: 8,
      });
    });

    it('passes when embedding counts match', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: {
          documents: 5,
          chunks: 10,
          entities: 0,
          edges: 0,
          term_embeddings: 0,
          feedback_events: 0,
          feedback_scores: 0,
          _schema_version: 1,
          _index_metadata: 1,
        },
        nullCounts: {
          'chunks.embedding': 2, // 10 - 2 = 8 non-null
          'term_embeddings.embedding': 0,
        },
      });
      sourceClientRef.current = createMockSourceClient({
        tableCounts: {
          documents: 5,
          chunks: 10,
          entities: 0,
          edges: 0,
          term_embeddings: 0,
          feedback_events: 0,
          feedback_scores: 0,
          chunk_embeddings: 8, // matches
        },
      });
      mockExistsSyncFn = () => true;

      const result = await validateTursoIndex({
        url: ':memory:',
        sourceCorpusPath: 'fake/turso-replica.sqlite',
        sourceEmbeddingsPath: 'fake/embeddings.sqlite',
      });
      expect(result.embeddingMismatches).toEqual([]);
    });

    it('detects unexpected nulls in term_embeddings when source has data', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: {
          documents: 0,
          chunks: 0,
          entities: 0,
          edges: 0,
          term_embeddings: 5,
          feedback_events: 0,
          feedback_scores: 0,
          _schema_version: 1,
          _index_metadata: 1,
        },
        nullCounts: {
          'chunks.embedding': 0,
          'term_embeddings.embedding': 3, // 3 nulls out of 5
        },
      });
      sourceClientRef.current = createMockSourceClient({
        tableCounts: {
          documents: 0,
          chunks: 0,
          entities: 0,
          edges: 0,
          term_embeddings: 5, // > 0, so nulls are unexpected
          feedback_events: 0,
          feedback_scores: 0,
        },
      });
      mockExistsSyncFn = (p) => String(p).includes('turso-replica');

      const result = await validateTursoIndex({
        url: ':memory:',
        sourceCorpusPath: 'fake/turso-replica.sqlite',
        sourceEmbeddingsPath: 'fake/embeddings.sqlite',
      });
      expect(result.success).toBe(false);
    });

    it('source embeddings only (no source corpus) - sourceComparison available', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: { documents: 0, chunks: 0 },
        nullCounts: { 'chunks.embedding': 0, 'term_embeddings.embedding': 0 },
      });
      sourceClientRef.current = createMockSourceClient({
        tableCounts: { chunk_embeddings: 0 },
      });
      // Only embeddings exists, not corpus
      mockExistsSyncFn = (p) => String(p).includes('embeddings');

      const result = await validateTursoIndex({
        url: ':memory:',
        sourceCorpusPath: 'fake/turso-replica.sqlite',
        sourceEmbeddingsPath: 'fake/embeddings.sqlite',
      });
      expect(result.sourceComparison).toBe('available');
      // No discrepancies because sourceCorpusExists is false
      expect(result.discrepancies).toEqual([]);
    });

    it('catches errors from getTargetCounts', async () => {
      const errorClient = {
        execute: jest.fn(async () => {
          throw new Error('Query failed');
        }),
        batch: jest.fn(async () => {}),
        close: jest.fn(async () => {}),
      };
      targetClientRef.current = errorClient;

      const result = await validateTursoIndex({ url: ':memory:' });
      expect(result.success).toBe(false);
      expect(result.error).toBe('Query failed');
      expect(result.sourceComparison).toBe('error');
    });

    it('passes authToken to clientConfig when provided', async () => {
      const mockClient = createMockTargetClient();
      targetClientRef.current = mockClient;
      const result = await validateTursoIndex({
        url: 'libsql://example.turso.io',
        authToken: 'test-token',
      });
      // The mock createClient doesn't check authToken, but the code should pass it
      expect(result.success).toBe(true);
    });

    it('uses env vars when options not provided', async () => {
      process.env.TURSO_DATABASE_URL = ':memory:';
      process.env.TURSO_AUTH_TOKEN = 'env-token';
      targetClientRef.current = createMockTargetClient();
      const result = await validateTursoIndex();
      expect(result.url).toBe(':memory:');
      delete process.env.TURSO_DATABASE_URL;
      delete process.env.TURSO_AUTH_TOKEN;
    });
  });

  describe('validateTursoIndex - search validation', () => {
    it('runs search validation with mock client (all succeed, DiskANN throws)', async () => {
      targetClientRef.current = createSearchMockClient({
        diskAnnRows: null, // throw
      });
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({
        url: ':memory:',
        validateSearch: true,
      });
      expect(result.searchValidation).not.toBeNull();
      expect(result.searchValidation.fts5.verified).toBe(true);
      expect(result.searchValidation.vectorBruteForce.verified).toBe(true);
      expect(result.searchValidation.vectorDiskAnn.verified).toBe('skipped');
      expect(result.searchValidation.entityGraph.verified).toBe(true);
      // DiskANN skipped is OK, so success should be true
      expect(result.searchValidation.success).toBe(true);
    });

    it('handles DiskANN returning 0 rows', async () => {
      targetClientRef.current = createSearchMockClient({
        diskAnnRows: [], // empty
      });
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({
        url: ':memory:',
        validateSearch: true,
      });
      expect(result.searchValidation.vectorDiskAnn.verified).toBe('skipped');
      expect(result.searchValidation.vectorDiskAnn.note).toContain(
        'returned 0 rows',
      );
    });

    it('handles DiskANN returning correct result', async () => {
      const client = createSearchMockClient({
        diskAnnRows: null, // will be set dynamically
      });
      // Override to return correct chunk ID
      const originalExecute = client.execute;
      let capturedChunkIds = [];
      client.execute = jest.fn(async (sqlOrObj) => {
        const sql = typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
        if (sql.includes('SELECT chunk_id FROM chunks WHERE doc_id')) {
          const result = await originalExecute(sqlOrObj);
          if (result.rows.length > 0) {
            capturedChunkIds.push(Number(result.rows[0].chunk_id));
          }
          return result;
        }
        if (sql.includes('vector_top_k')) {
          return { rows: [{ chunk_id: capturedChunkIds[0] }] };
        }
        return originalExecute(sqlOrObj);
      });

      targetClientRef.current = client;
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({
        url: ':memory:',
        validateSearch: true,
      });
      expect(result.searchValidation.vectorDiskAnn.verified).toBe(true);
      expect(result.searchValidation.vectorDiskAnn.note).toContain(
        'correct nearest neighbor',
      );
    });

    it('handles DiskANN returning wrong result', async () => {
      const client = createSearchMockClient({
        diskAnnRows: null,
      });
      const originalExecute = client.execute;
      let capturedChunkIds = [];
      client.execute = jest.fn(async (sqlOrObj) => {
        const sql = typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
        if (sql.includes('SELECT chunk_id FROM chunks WHERE doc_id')) {
          const result = await originalExecute(sqlOrObj);
          if (result.rows.length > 0) {
            capturedChunkIds.push(Number(result.rows[0].chunk_id));
          }
          return result;
        }
        if (sql.includes('vector_top_k')) {
          // Return wrong chunk (second instead of first)
          return { rows: [{ chunk_id: capturedChunkIds[1] ?? 999 }] };
        }
        return originalExecute(sqlOrObj);
      });

      targetClientRef.current = client;
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({
        url: ':memory:',
        validateSearch: true,
      });
      expect(result.searchValidation.vectorDiskAnn.verified).toBe(false);
      expect(result.searchValidation.vectorDiskAnn.note).toContain(
        'different top result',
      );
    });

    it('handles brute-force returning empty rows', async () => {
      targetClientRef.current = createSearchMockClient({
        bruteForceRows: [], // empty
        diskAnnRows: null,
      });
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({
        url: ':memory:',
        validateSearch: true,
      });
      expect(result.searchValidation.vectorBruteForce.verified).toBe(false);
      expect(result.searchValidation.vectorBruteForce.selfRank).toBeNull();
    });

    it('handles brute-force returning wrong top result (selfRank -1)', async () => {
      targetClientRef.current = createSearchMockClient({
        bruteForceRows: [{ chunk_id: 99999, dist: 0.5 }], // wrong chunk
        diskAnnRows: null,
      });
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({
        url: ':memory:',
        validateSearch: true,
      });
      expect(result.searchValidation.vectorBruteForce.verified).toBe(false);
      expect(result.searchValidation.vectorBruteForce.selfRank).toBeNull();
    });

    it('handles FTS5 search error', async () => {
      targetClientRef.current = createSearchMockClient({
        throwOnFts5: true,
        diskAnnRows: null,
      });
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({
        url: ':memory:',
        validateSearch: true,
      });
      expect(result.searchValidation.fts5.verified).toBe(false);
      expect(result.searchValidation.fts5.error).toBe('FTS5 search error');
      expect(result.searchValidation.success).toBe(false);
    });

    it('handles vector search error', async () => {
      targetClientRef.current = createSearchMockClient({
        throwOnVector: true,
        diskAnnRows: null,
      });
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({
        url: ':memory:',
        validateSearch: true,
      });
      expect(result.searchValidation.vectorBruteForce.verified).toBe(false);
      expect(result.searchValidation.vectorBruteForce.error).toBe(
        'Vector search error',
      );
      expect(result.searchValidation.success).toBe(false);
    });

    it('handles entity graph error', async () => {
      targetClientRef.current = createSearchMockClient({
        throwOnEntityGraph: true,
        diskAnnRows: null,
      });
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({
        url: ':memory:',
        validateSearch: true,
      });
      expect(result.searchValidation.entityGraph.verified).toBe(false);
      expect(result.searchValidation.entityGraph.error).toBe(
        'Entity graph error',
      );
      expect(result.searchValidation.success).toBe(false);
    });

    it('runs search validation with real in-memory client', async () => {
      const realClient = await createRealSearchClient();
      targetClientRef.current = realClient;
      mockExistsSyncFn = () => false;

      const result = await validateTursoIndex({
        url: ':memory:',
        validateSearch: true,
      });
      expect(result.searchValidation).not.toBeNull();
      // FTS5 should work with real schema
      expect(result.searchValidation.fts5.verified).toBe(true);
      // Entity graph should work with real schema
      expect(result.searchValidation.entityGraph.verified).toBe(true);
    });
  });

  describe('CLI main()', () => {
    let writeSpy;
    let errorSpy;
    let consoleErrorSpy;
    let consoleLogSpy;
    let originalExit;
    let exitCode;

    beforeEach(() => {
      writeSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
      errorSpy = jest.spyOn(process.stderr, 'write').mockImplementation(() => true);
      consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => true);
      consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => true);
      originalExit = process.exit;
      exitCode = null;
      process.exit = (code) => {
        exitCode = code;
      };
      // main() calls validateTursoIndex() without an explicit url, so it
      // falls back to TURSO_DATABASE_URL or DEFAULT_TURSO_URL (a file: URL).
      // The mock routes file: URLs to sourceClientRef, but CLI tests set
      // targetClientRef. Setting :memory: ensures the mock routes to
      // targetClientRef.current.
      process.env.TURSO_DATABASE_URL = ':memory:';
    });

    afterEach(() => {
      writeSpy.mockRestore();
      errorSpy.mockRestore();
      consoleErrorSpy.mockRestore();
      consoleLogSpy.mockRestore();
      process.exit = originalExit;
      delete process.env.TURSO_DATABASE_URL;
    });

    it('runs CLI with --json and successful validation', async () => {
      targetClientRef.current = createMockTargetClient();
      mockExistsSyncFn = () => false;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath, '--json'];

      const mod = await import(
        './validate-turso-index.mjs?cli-test=' + Date.now()
      );
      await new Promise((r) => setTimeout(r, 300));

      const stdout = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(stdout).toContain('"success"');
      expect(exitCode).toBe(0);
    });

    it('runs CLI with --json and failed validation', async () => {
      throwOnCreate = true;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath, '--json'];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + '2');
      await new Promise((r) => setTimeout(r, 300));

      const stdout = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(stdout).toContain('"success": false');
      expect(exitCode).toBe(1);
    });

    it('runs CLI with human-readable output on success', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: { documents: 5, chunks: 10 },
      });
      mockExistsSyncFn = () => false;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + '3');
      await new Promise((r) => setTimeout(r, 300));

      const stdout = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(stdout).toContain('PASSED');
      expect(stdout).toContain('Target table counts:');
      expect(exitCode).toBe(0);
    });

    it('runs CLI with human-readable output on failure', async () => {
      throwOnCreate = true;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + '4');
      await new Promise((r) => setTimeout(r, 300));

      const stderr = [...errorSpy.mock.calls, ...consoleErrorSpy.mock.calls]
        .map((c) => String(c[0])).join('');
      expect(stderr).toContain('FAILED');
      expect(exitCode).toBe(1);
    });

    it('runs CLI with --validate-search flag', async () => {
      targetClientRef.current = createSearchMockClient({
        diskAnnRows: null,
      });
      mockExistsSyncFn = () => false;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath, '--json', '--validate-search'];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + '5');
      await new Promise((r) => setTimeout(r, 300));

      const stdout = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(stdout).toContain('searchValidation');
      expect(exitCode).toBe(0);
    });

    it('runs CLI with --full flag and human-readable search output', async () => {
      targetClientRef.current = createSearchMockClient({
        diskAnnRows: [],
      });
      mockExistsSyncFn = () => false;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath, '--full'];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + '6');
      await new Promise((r) => setTimeout(r, 300));

      const stdout = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(stdout).toContain('Search validation:');
      expect(stdout).toContain('FTS5:');
      expect(stdout).toContain('Vector brute-force:');
      expect(stdout).toContain('Vector DiskANN:');
      expect(stdout).toContain('Entity graph:');
      expect(exitCode).toBe(0);
    });

    it('runs CLI with --source-corpus and --source-embeddings args', async () => {
      targetClientRef.current = createMockTargetClient();
      sourceClientRef.current = createMockSourceClient({
        tableCounts: {
          documents: 0,
          chunks: 0,
          entities: 0,
          edges: 0,
          term_embeddings: 0,
          feedback_events: 0,
          feedback_scores: 0,
        },
      });
      mockExistsSyncFn = () => true;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = [
        'node',
        scriptPath,
        '--json',
        '--source-corpus',
        'fake/corpus.sqlite',
        '--source-embeddings',
        'fake/embed.sqlite',
      ];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + '7');
      await new Promise((r) => setTimeout(r, 300));

      const stdout = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(stdout).toContain('"sourceComparison": "available"');
      expect(exitCode).toBe(0);
    });

    it('runs CLI with human-readable output showing discrepancies', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: {
          documents: 5,
          chunks: 0,
          entities: 0,
          edges: 0,
          term_embeddings: 0,
          feedback_events: 0,
          feedback_scores: 0,
          _schema_version: 0,
          _index_metadata: 0,
        },
        nullCounts: { 'chunks.embedding': 0, 'term_embeddings.embedding': 0 },
      });
      sourceClientRef.current = createMockSourceClient({
        tableCounts: {
          documents: 3,
          chunks: 0,
          entities: 0,
          edges: 0,
          term_embeddings: 0,
          feedback_events: 0,
          feedback_scores: 0,
          chunk_embeddings: 0,
        },
      });
      mockExistsSyncFn = () => true;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + '8');
      await new Promise((r) => setTimeout(r, 300));

      const stderr = [...errorSpy.mock.calls, ...consoleErrorSpy.mock.calls]
        .map((c) => String(c[0])).join('');
      expect(stderr).toContain('discrepancies');
      expect(exitCode).toBe(1);
    });

    it('runs CLI with human-readable output showing source counts', async () => {
      targetClientRef.current = createMockTargetClient();
      sourceClientRef.current = createMockSourceClient({
        tableCounts: {
          documents: 0,
          chunks: 0,
          entities: 0,
          edges: 0,
          term_embeddings: 0,
          feedback_events: 0,
          feedback_scores: 0,
        },
      });
      mockExistsSyncFn = (p) => String(p).includes('turso-replica');

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + '9');
      await new Promise((r) => setTimeout(r, 300));

      const stdout = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(stdout).toContain('Source table counts:');
      expect(exitCode).toBe(0);
    });

    it('runs CLI with embedding mismatches in human-readable output', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: {
          documents: 0,
          chunks: 10,
          entities: 0,
          edges: 0,
          term_embeddings: 0,
          feedback_events: 0,
          feedback_scores: 0,
          _schema_version: 0,
          _index_metadata: 0,
        },
        nullCounts: {
          'chunks.embedding': 5,
          'term_embeddings.embedding': 0,
        },
      });
      sourceClientRef.current = createMockSourceClient({
        tableCounts: {
          documents: 0,
          chunks: 10,
          entities: 0,
          edges: 0,
          term_embeddings: 0,
          feedback_events: 0,
          feedback_scores: 0,
          chunk_embeddings: 8,
        },
      });
      mockExistsSyncFn = () => true;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + 'a');
      await new Promise((r) => setTimeout(r, 300));

      const stderr = [...errorSpy.mock.calls, ...consoleErrorSpy.mock.calls]
        .map((c) => String(c[0])).join('');
      expect(stderr).toContain('Embedding count mismatches');
      expect(exitCode).toBe(1);
    });

    it('runs CLI with error in human-readable output', async () => {
      throwOnCreate = true;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + 'b');
      await new Promise((r) => setTimeout(r, 300));

      const stderr = [...errorSpy.mock.calls, ...consoleErrorSpy.mock.calls]
        .map((c) => String(c[0])).join('');
      expect(stderr).toContain('Error:');
      expect(exitCode).toBe(1);
    });

    it('runs CLI with NULL embedding checks showing OK status', async () => {
      targetClientRef.current = createMockTargetClient();
      mockExistsSyncFn = () => false;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + 'c');
      await new Promise((r) => setTimeout(r, 300));

      const stdout = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(stdout).toContain('NULL embedding checks:');
      expect(stdout).toContain('OK');
      expect(exitCode).toBe(0);
    });

    it('runs CLI with NULL embedding checks showing null counts', async () => {
      targetClientRef.current = createMockTargetClient({
        tableCounts: {
          documents: 0,
          chunks: 10,
          entities: 0,
          edges: 0,
          term_embeddings: 5,
          feedback_events: 0,
          feedback_scores: 0,
          _schema_version: 0,
          _index_metadata: 0,
        },
        nullCounts: {
          'chunks.embedding': 3,
          'term_embeddings.embedding': 2,
        },
      });
      mockExistsSyncFn = () => false;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + 'd');
      await new Promise((r) => setTimeout(r, 300));

      const stdout = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(stdout).toContain('NULLs out of');
      expect(exitCode).toBe(0); // No source comparison, so NULLs are informational
    });

    it('runs CLI with DiskANN OK note in human-readable output', async () => {
      const client = createSearchMockClient({ diskAnnRows: null });
      const originalExecute = client.execute;
      let capturedChunkIds = [];
      client.execute = jest.fn(async (sqlOrObj) => {
        const sql = typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
        if (sql.includes('SELECT chunk_id FROM chunks WHERE doc_id')) {
          const result = await originalExecute(sqlOrObj);
          if (result.rows.length > 0) {
            capturedChunkIds.push(Number(result.rows[0].chunk_id));
          }
          return result;
        }
        if (sql.includes('vector_top_k')) {
          return { rows: [{ chunk_id: capturedChunkIds[0] }] };
        }
        return originalExecute(sqlOrObj);
      });

      targetClientRef.current = client;
      mockExistsSyncFn = () => false;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath, '--full'];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + 'e');
      await new Promise((r) => setTimeout(r, 300));

      const stdout = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(stdout).toContain('Vector DiskANN:');
      expect(stdout).toContain('OK');
    });

    it('runs CLI with DiskANN FAIL and error notes in human-readable output', async () => {
      targetClientRef.current = createSearchMockClient({
        throwOnFts5: true,
        throwOnVector: true,
        throwOnEntityGraph: true,
        diskAnnRows: null,
      });
      mockExistsSyncFn = () => false;

      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'validate-turso-index.mjs',
      );
      process.argv = ['node', scriptPath, '--full'];

      await import('./validate-turso-index.mjs?cli-test=' + Date.now() + 'f');
      await new Promise((r) => setTimeout(r, 300));

      const stderr = [...errorSpy.mock.calls, ...consoleErrorSpy.mock.calls]
        .map((c) => String(c[0])).join('');
      expect(stderr).toContain('FTS5 error');
      expect(stderr).toContain('vector error');
      expect(stderr).toContain('graph error');
      expect(exitCode).toBe(1);
    });
  });
});