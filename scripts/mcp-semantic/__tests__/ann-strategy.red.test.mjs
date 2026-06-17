/**
 * @module ann-strategy.red.test
 * @description Red tests for Step 23 ANN dense-strategy selection.
 *
 * Verifies the expected public contract of the ann-strategy module before
 * implementation: threshold-based strategy resolution, brute-force LRU result
 * cache, HNSW availability detection, and incremental-vs-rebuild decisions.
 */

import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Database from 'better-sqlite3';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const DEFAULT_ANN_THRESHOLD = 50000;
const DEFAULT_CACHE_TTL_MS = 30 * 60 * 1000;

/**
 * Load the v2 corpus schema from the semantic-index scripts directory.
 * @returns {Promise<string>} SQL text.
 */
async function readCorpusSchema() {
  const schemaPath = path.join(__dirname, '../../semantic-index/schema-v2.sql');
  return readFile(schemaPath, 'utf8');
}

/**
 * Create a temporary corpus and embeddings database pair for fixtures.
 * @returns {Promise<{ corpusDbPath: string, embeddingsDbPath: string, tempDir: string }>}
 */
async function setupDatabases() {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'ann-strategy-test-'));
  const corpusDbPath = path.join(tempDir, 'corpus.sqlite');
  const embeddingsDbPath = path.join(tempDir, 'embeddings.sqlite');

  const corpusDb = new Database(corpusDbPath);
  corpusDb.exec(await readCorpusSchema());
  corpusDb.close();

  const embeddingsDb = new Database(embeddingsDbPath);
  embeddingsDb.exec(`
    CREATE TABLE IF NOT EXISTS chunk_embeddings (
      chunk_id INTEGER PRIMARY KEY,
      model_id TEXT NOT NULL,
      dimension INTEGER NOT NULL,
      embedding BLOB NOT NULL,
      chunk_sha256 TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS ann_index_meta (
      index_id TEXT PRIMARY KEY,
      index_type TEXT NOT NULL,
      model_id TEXT NOT NULL,
      model_sha256 TEXT NOT NULL,
      dimension INTEGER NOT NULL,
      metric TEXT NOT NULL,
      max_elements INTEGER NOT NULL,
      current_elements INTEGER NOT NULL DEFAULT 0,
      m_param INTEGER,
      ef_construction_param INTEGER,
      ef_search_param INTEGER,
      build_status TEXT NOT NULL DEFAULT 'pending',
      build_started_at TEXT,
      build_completed_at TEXT,
      build_duration_ms INTEGER,
      build_error TEXT,
      last_incremental_update_at TEXT,
      index_file_path TEXT,
      created_at TEXT NOT NULL DEFAULT(datetime('now')),
      updated_at TEXT NOT NULL DEFAULT(datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS ann_index_chunk_map (
      index_id TEXT NOT NULL,
      external_id INTEGER NOT NULL,
      chunk_id INTEGER NOT NULL,
      PRIMARY KEY(index_id, external_id)
    );
    CREATE TABLE IF NOT EXISTS ann_threshold_config (
      config_key TEXT PRIMARY KEY,
      config_value TEXT NOT NULL,
      updated_at TEXT NOT NULL DEFAULT(datetime('now'))
    );
  `);
  embeddingsDb.close();

  return { corpusDbPath, embeddingsDbPath, tempDir };
}

/**
 * Remove the temporary fixture directory.
 * @param {string} tempDir
 * @returns {Promise<void>}
 */
function teardown(tempDir) {
  return rm(tempDir, { recursive: true, force: true });
}

/**
 * Import the ann-strategy module, returning an empty object if it is missing.
 * This keeps red tests loadable before the production module is created.
 * @returns {Promise<Record<string, unknown>>}
 */
async function loadAnnStrategy() {
  return import('../tools/ann-strategy.mjs').catch(() => ({}));
}

/**
 * Import the ann-index module, returning an empty object if it is missing.
 * @returns {Promise<Record<string, unknown>>}
 */
async function loadAnnIndex() {
  return import('../tools/ann-index.mjs').catch(() => ({}));
}

describe('ann-strategy', () => {
  describe('module exports', () => {
    it('exports resolveDenseStrategy', async () => {
      const mod = await loadAnnStrategy();

      expect(mod.resolveDenseStrategy).toBeInstanceOf(Function);
    });

    it('exports a quantized hash helper for cache keys', async () => {
      const mod = await loadAnnStrategy();

      expect(mod.quantizedHash).toBeInstanceOf(Function);
    });

    it('exports cache accessors', async () => {
      const mod = await loadAnnStrategy();

      expect(mod.getQueryResultCache).toBeInstanceOf(Function);
      expect(mod.setQueryResultCache).toBeInstanceOf(Function);
    });

    it('exports HNSW availability state', async () => {
      const mod = await loadAnnStrategy();

      expect(typeof mod.isHnswAvailable).toBe('boolean');
    });

    it('exports incremental-update action detector', async () => {
      const mod = await loadAnnStrategy();

      expect(mod.detectIncrementalUpdateAction).toBeInstanceOf(Function);
    });
  });

  describe('strategy selection', () => {
    it('selects brute_force_cached when chunk count is below threshold', async () => {
      const { resolveDenseStrategy } = await loadAnnStrategy();

      const strategy = resolveDenseStrategy({
        chunkCount: 1000,
        annThreshold: DEFAULT_ANN_THRESHOLD,
      });

      expect(strategy).toBe('brute_force_cached');
    });

    it('selects hnsw when chunk count equals threshold and index is ready', async () => {
      const { resolveDenseStrategy } = await loadAnnStrategy();

      const strategy = resolveDenseStrategy({
        chunkCount: 50000,
        annThreshold: DEFAULT_ANN_THRESHOLD,
        indexStatus: 'ready',
      });

      expect(strategy).toBe('hnsw');
    });

    it('selects hnsw when chunk count is above threshold and index is ready', async () => {
      const { resolveDenseStrategy } = await loadAnnStrategy();

      const strategy = resolveDenseStrategy({
        chunkCount: 75000,
        annThreshold: DEFAULT_ANN_THRESHOLD,
        indexStatus: 'ready',
      });

      expect(strategy).toBe('hnsw');
    });

    it('falls back to brute_force_cached when index is stale', async () => {
      const { resolveDenseStrategy } = await loadAnnStrategy();

      const strategy = resolveDenseStrategy({
        chunkCount: 75000,
        annThreshold: DEFAULT_ANN_THRESHOLD,
        indexStatus: 'stale',
      });

      expect(strategy).toBe('brute_force_cached');
    });

    it('falls back to brute_force_cached when index is in error state', async () => {
      const { resolveDenseStrategy } = await loadAnnStrategy();

      const strategy = resolveDenseStrategy({
        chunkCount: 75000,
        annThreshold: DEFAULT_ANN_THRESHOLD,
        indexStatus: 'error',
      });

      expect(strategy).toBe('brute_force_cached');
    });

    it('falls back to brute_force_cached when hnswlib-node is unavailable', async () => {
      const { resolveDenseStrategy } = await loadAnnStrategy();

      const strategy = resolveDenseStrategy({
        chunkCount: 75000,
        annThreshold: DEFAULT_ANN_THRESHOLD,
        indexStatus: 'ready',
        hnswAvailable: false,
      });

      expect(strategy).toBe('brute_force_cached');
    });

    it('respects force=hnsw override regardless of chunk count', async () => {
      const { resolveDenseStrategy } = await loadAnnStrategy();

      const strategy = resolveDenseStrategy({
        chunkCount: 100,
        annThreshold: DEFAULT_ANN_THRESHOLD,
        forceStrategy: 'hnsw',
      });

      expect(strategy).toBe('hnsw');
    });

    it('respects force=brute_force_cached override regardless of threshold', async () => {
      const { resolveDenseStrategy } = await loadAnnStrategy();

      const strategy = resolveDenseStrategy({
        chunkCount: 75000,
        annThreshold: DEFAULT_ANN_THRESHOLD,
        indexStatus: 'ready',
        forceStrategy: 'brute_force_cached',
      });

      expect(strategy).toBe('brute_force_cached');
    });

    it('respects force=brute_force override regardless of threshold', async () => {
      const { resolveDenseStrategy } = await loadAnnStrategy();

      const strategy = resolveDenseStrategy({
        chunkCount: 75000,
        annThreshold: DEFAULT_ANN_THRESHOLD,
        indexStatus: 'ready',
        forceStrategy: 'brute_force',
      });

      expect(strategy).toBe('brute_force');
    });
  });

  describe('quantized hash', () => {
    it('returns the same hash for nearly identical embeddings', async () => {
      const { quantizedHash } = await loadAnnStrategy();

      const hashA = quantizedHash(
        new Float32Array([0.1000001, 0.2000001, 0.3000001]),
      );
      const hashB = quantizedHash(new Float32Array([0.1, 0.2, 0.3]));

      expect(hashA).toBe(hashB);
    });

    it('returns different hashes for clearly different embeddings', async () => {
      const { quantizedHash } = await loadAnnStrategy();

      const hashA = quantizedHash(new Float32Array([0.1, 0.2, 0.3]));
      const hashB = quantizedHash(new Float32Array([0.9, 0.8, 0.7]));

      expect(hashA).not.toBe(hashB);
    });
  });

  describe('brute_force_cached LRU cache', () => {
    it('stores and retrieves results by quantized embedding hash', async () => {
      const {
        getQueryResultCache,
        setQueryResultCache,
        clearQueryResultCache,
      } = await loadAnnStrategy();
      clearQueryResultCache?.();

      const embedding = new Float32Array([0.1, 0.2, 0.3]);
      const results = [{ chunk_id: 1, cosine_score: 0.9 }];
      setQueryResultCache({
        embedding,
        modelId: 'all-MiniLM-L6-v2',
        results,
        ttlMs: DEFAULT_CACHE_TTL_MS,
        maxEntries: 500,
      });

      const cached = getQueryResultCache({
        embedding,
        modelId: 'all-MiniLM-L6-v2',
      });

      expect(cached).toEqual(results);
    });

    it('returns undefined for embeddings that were not cached', async () => {
      const { getQueryResultCache, clearQueryResultCache } =
        await loadAnnStrategy();
      clearQueryResultCache?.();

      const cached = getQueryResultCache({
        embedding: new Float32Array([0.9, 0.8, 0.7]),
        modelId: 'all-MiniLM-L6-v2',
      });

      expect(cached).toBeUndefined();
    });

    it('evicts oldest entries when the cache exceeds max size', async () => {
      const {
        getQueryResultCache,
        setQueryResultCache,
        clearQueryResultCache,
      } = await loadAnnStrategy();
      clearQueryResultCache?.();

      for (let index = 0; index < 3; index++) {
        setQueryResultCache({
          embedding: new Float32Array([index, 0, 0]),
          modelId: 'all-MiniLM-L6-v2',
          results: [{ chunk_id: index }],
          ttlMs: DEFAULT_CACHE_TTL_MS,
          maxEntries: 2,
        });
      }

      const first = getQueryResultCache({
        embedding: new Float32Array([0, 0, 0]),
        modelId: 'all-MiniLM-L6-v2',
      });

      expect(first).toBeUndefined();
    });

    it('does not return expired cache entries', async () => {
      const {
        getQueryResultCache,
        setQueryResultCache,
        clearQueryResultCache,
      } = await loadAnnStrategy();
      clearQueryResultCache?.();

      const embedding = new Float32Array([0.1, 0.2, 0.3]);
      setQueryResultCache({
        embedding,
        modelId: 'all-MiniLM-L6-v2',
        results: [{ chunk_id: 1 }],
        ttlMs: -1,
        maxEntries: 500,
      });

      const cached = getQueryResultCache({
        embedding,
        modelId: 'all-MiniLM-L6-v2',
      });

      expect(cached).toBeUndefined();
    });
  });

  describe('HNSW wrapper and fallback', () => {
    it('exports an HNSW build function', async () => {
      const mod = await loadAnnIndex();

      expect(mod.buildAnnIndex).toBeInstanceOf(Function);
    });

    it('exports an HNSW search function', async () => {
      const mod = await loadAnnIndex();

      expect(mod.queryHnswIndex).toBeInstanceOf(Function);
    });

    it('gracefully falls back when hnswlib-node is unavailable', async () => {
      const { buildAnnIndex } = await loadAnnIndex();
      const { embeddingsDbPath, tempDir } = await setupDatabases();
      try {
        const result = await buildAnnIndex({
          embeddingsDatabasePath: embeddingsDbPath,
          indexFilePath: path.join(tempDir, 'hnsw.dat'),
          modelId: 'all-MiniLM-L6-v2',
          dimension: 3,
          hnswAvailable: false,
        });

        expect(result.build_status).not.toBe('ready');
      } finally {
        await teardown(tempDir);
      }
    });

    it('populates ann_index_chunk_map when building an HNSW index', async () => {
      const { buildAnnIndex } = await loadAnnIndex();
      const { embeddingsDbPath, tempDir } = await setupDatabases();
      try {
        const embeddingsDb = new Database(embeddingsDbPath);
        const buffer = Buffer.from(new Float32Array([1, 0, 0]).buffer);
        embeddingsDb
          .prepare(
            'INSERT INTO chunk_embeddings (chunk_id, model_id, dimension, embedding, chunk_sha256) VALUES (?, ?, ?, ?, ?)',
          )
          .run(1, 'all-MiniLM-L6-v2', 3, buffer, 'sha');
        embeddingsDb.close();

        await buildAnnIndex({
          embeddingsDatabasePath: embeddingsDbPath,
          indexFilePath: path.join(tempDir, 'hnsw.dat'),
          modelId: 'all-MiniLM-L6-v2',
          dimension: 3,
        });

        const db = new Database(embeddingsDbPath);
        const mapRow = db
          .prepare('SELECT COUNT(*) AS count FROM ann_index_chunk_map')
          .get();
        db.close();

        expect(Number(mapRow.count)).toBeGreaterThan(0);
      } finally {
        await teardown(tempDir);
      }
    });
  });

  describe('incremental update detection', () => {
    it('recommends incremental update when changed ratio is <= 5%', async () => {
      const { detectIncrementalUpdateAction } = await loadAnnStrategy();

      const action = detectIncrementalUpdateAction({
        currentElements: 1000,
        changedChunkCount: 50,
      });

      expect(action).toBe('incremental');
    });

    it('recommends full rebuild when changed ratio exceeds 5%', async () => {
      const { detectIncrementalUpdateAction } = await loadAnnStrategy();

      const action = detectIncrementalUpdateAction({
        currentElements: 1000,
        changedChunkCount: 51,
      });

      expect(action).toBe('rebuild');
    });

    it('recommends full rebuild when chunk count has changed', async () => {
      const { detectIncrementalUpdateAction } = await loadAnnStrategy();

      const action = detectIncrementalUpdateAction({
        currentElements: 1000,
        changedChunkCount: 10,
        chunkCountDelta: 50,
      });

      expect(action).toBe('rebuild');
    });
  });
});
