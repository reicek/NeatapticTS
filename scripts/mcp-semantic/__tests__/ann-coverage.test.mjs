/**
 * @module ann-coverage.test
 * @description Coverage tests for the ANN strategy and index modules.
 *
 * These tests exercise paths that cannot be reached in the red tests because
 * the optional `hnswlib-node` dependency is not installed in this environment.
 * A deterministic in-memory mock of `hnswlib-node` is used so the HNSW build and
 * query wiring can be validated at 100% coverage without a native build.
 *
 * The mock is injected through the exported `__hnswTestSeam` object in
 * `ann-strategy.mjs` rather than Jest's mock module registry, because the
 * optional dependency is not resolvable in this environment.
 *
 * This file follows the same coverage-test pattern as `eval-coverage.test.mjs`.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Database from 'better-sqlite3';
import * as strategyMod from '../tools/ann-strategy.mjs';
import * as indexMod from '../tools/ann-index.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Deterministic mock of the `hnswlib-node` HierarchicalNSW class.
 *
 * Persists the point list to the requested index file path so that separate
 * build/query instances can exercise the full read/write wiring. Search is
 * implemented as exact cosine similarity, so the mock returns the same top-k
 * as a brute-force scan for small synthetic corpora.
 */
function createMockHnsw() {
  class MockHierarchicalNSW {
    constructor(metric, dimension) {
      this.metric = metric;
      this.dimension = dimension;
      this.points = [];
    }

    initIndex(maxElements, M, efConstruction, randomSeed) {
      this.maxElements = maxElements;
    }

    addPoint(vector, externalId) {
      this.points.push({ externalId, vector: Array.from(vector) });
    }

    setEf(efSearch) {
      this.efSearch = efSearch;
    }

    writeIndexSync(filePath) {
      writeFileSync(
        filePath,
        JSON.stringify({
          metric: this.metric,
          dimension: this.dimension,
          points: this.points,
        }),
      );
    }

    readIndexSync(filePath) {
      const data = JSON.parse(readFileSync(filePath, 'utf8'));
      this.metric = data.metric;
      this.dimension = data.dimension;
      this.points = data.points;
    }

    searchKnn(queryVector, k) {
      const q = Array.from(queryVector);
      const scored = this.points.map(({ externalId, vector }) => {
        let dot = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < q.length; i++) {
          dot += vector[i] * q[i];
          normA += vector[i] * vector[i];
          normB += q[i] * q[i];
        }
        const similarity = dot / (Math.sqrt(normA) * Math.sqrt(normB)) || 0;
        return { externalId, score: similarity };
      });
      scored.sort((a, b) => b.score - a.score);
      const top = scored.slice(0, k);
      return {
        neighbors: top.map((p) => p.externalId),
        distances: top.map((p) => p.score),
      };
    }
  }

  return { HierarchicalNSW: MockHierarchicalNSW };
}

/**
 * Install the mock HNSW loader, refresh availability, run the callback, then
 * restore the real optional-dependency loader.
 * @param {() => Promise<void> | void} callback
 * @returns {Promise<void>}
 */
async function withMockHnsw(callback) {
  const original = strategyMod.__hnswTestSeam.importFn;
  strategyMod.__hnswTestSeam.importFn = () => Promise.resolve(createMockHnsw());
  await strategyMod.__refreshHnswAvailability();
  try {
    await callback();
  } finally {
    strategyMod.__hnswTestSeam.importFn = original;
    await strategyMod.__refreshHnswAvailability();
  }
}

/**
 * Create a temporary embeddings database with the minimal chunk_embeddings schema.
 * @returns {Promise<{ dbPath: string, tempDir: string }>}
 */
async function setupEmbeddingsDb() {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'ann-coverage-'));
  const dbPath = path.join(tempDir, 'embeddings.sqlite');
  const db = new Database(dbPath);
  db.exec(`
    CREATE TABLE IF NOT EXISTS chunk_embeddings (
      chunk_id INTEGER PRIMARY KEY,
      model_id TEXT NOT NULL,
      dimension INTEGER NOT NULL,
      embedding BLOB NOT NULL,
      chunk_sha256 TEXT NOT NULL
    );
  `);
  db.close();
  return { embeddingsDbPath: dbPath, tempDir };
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
 * Compute brute-force cosine top-k for a query against a list of embeddings.
 * @param {Float32Array} query
 * @param {Array<{ chunk_id: number, vector: Float32Array }>} embeddings
 * @param {number} k
 * @returns {Array<{ chunk_id: number, score: number }>}
 */
function bruteForceTopK(query, embeddings, k) {
  const q = Array.from(query);
  const scored = embeddings.map(({ chunk_id, vector }) => {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < q.length; i++) {
      dot += vector[i] * q[i];
      normA += vector[i] * vector[i];
      normB += q[i] * q[i];
    }
    const similarity = dot / (Math.sqrt(normA) * Math.sqrt(normB)) || 0;
    return { chunk_id, score: similarity };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

/**
 * Insert random unit-vector embeddings for a synthetic corpus.
 * @param {string} dbPath
 * @param {number} count
 * @param {number} dimension
 * @param {string} modelId
 * @returns {Array<{ chunk_id: number, vector: Float32Array }>}
 */
function insertSyntheticEmbeddings(dbPath, count, dimension, modelId) {
  const db = new Database(dbPath);
  const insert = db.prepare(
    'INSERT INTO chunk_embeddings (chunk_id, model_id, dimension, embedding, chunk_sha256) VALUES (?, ?, ?, ?, ?)',
  );
  const embeddings = [];
  for (let i = 1; i <= count; i++) {
    const vector = new Float32Array(dimension);
    let norm = 0;
    for (let d = 0; d < dimension; d++) {
      const value = Math.sin(i * (d + 1) * 0.5) + 0.0001;
      vector[d] = value;
      norm += value * value;
    }
    const invNorm = 1 / Math.sqrt(norm);
    for (let d = 0; d < dimension; d++) {
      vector[d] *= invNorm;
    }
    const buffer = Buffer.from(vector.buffer);
    insert.run(i, modelId, dimension, buffer, `sha-${i}`);
    embeddings.push({ chunk_id: i, vector });
  }
  db.close();
  return embeddings;
}

describe('ann-strategy coverage', () => {
  it('marks HNSW available when hnswlib-node is present', async () => {
    await withMockHnsw(() => {
      expect(strategyMod.isHnswAvailable).toBe(true);
    });
  });

  it('selects the hnsw strategy above threshold when dependency is present', async () => {
    await withMockHnsw(() => {
      const strategy = strategyMod.resolveDenseStrategy({
        chunkCount: 100000,
        annThreshold: 50000,
        indexStatus: 'ready',
      });

      expect(strategy).toBe('hnsw');
    });
  });

  it('uses default threshold and chunk count when inputs are omitted', () => {
    const strategy = strategyMod.resolveDenseStrategy({});

    expect(strategy).toBe('brute_force_cached');
  });

  it('handles nullish chunk count and threshold', () => {
    const strategy = strategyMod.resolveDenseStrategy({
      chunkCount: null,
      annThreshold: null,
      indexStatus: 'ready',
    });

    expect(strategy).toBe('brute_force_cached');
  });

  it('returns an empty hash for a nullish embedding', () => {
    const hash = strategyMod.quantizedHash(null);

    expect(hash).toBe('');
  });

  it('uses default TTL and max entries when storing cache results', () => {
    strategyMod.clearQueryResultCache();
    const embedding = new Float32Array([0.1, 0.2, 0.3]);
    const results = [{ chunk_id: 1 }];

    strategyMod.setQueryResultCache({
      embedding,
      modelId: 'default-cache-model',
      results,
    });

    expect(
      strategyMod.getQueryResultCache({
        embedding,
        modelId: 'default-cache-model',
      }),
    ).toEqual(results);
  });

  it('recommends rebuild when current elements are missing or zero', () => {
    expect(
      strategyMod.detectIncrementalUpdateAction({
        currentElements: 0,
        changedChunkCount: 1,
      }),
    ).toBe('rebuild');
  });

  it('recommends rebuild when current elements are omitted', () => {
    expect(strategyMod.detectIncrementalUpdateAction({})).toBe('rebuild');
  });

  it('treats a missing changed-chunk count as zero changes', () => {
    const action = strategyMod.detectIncrementalUpdateAction({
      currentElements: 100,
    });

    expect(action).toBe('incremental');
  });
});

describe('ann-index coverage', () => {
  it('throws when buildAnnIndex is called without options', async () => {
    await expect(indexMod.buildAnnIndex()).rejects.toThrow(
      'embeddingsDatabasePath is required',
    );
  });

  it('throws when buildAnnIndex is called without an embeddings database path', async () => {
    await expect(indexMod.buildAnnIndex({})).rejects.toThrow(
      'embeddingsDatabasePath is required',
    );
  });

  it('builds and queries using default model and dimension', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      const dimension = 384;
      const embeddings = insertSyntheticEmbeddings(
        embeddingsDbPath,
        3,
        dimension,
        'all-MiniLM-L6-v2',
      );

      await withMockHnsw(async () => {
        const buildResult = await indexMod.buildAnnIndex({
          embeddingsDatabasePath: embeddingsDbPath,
          indexFilePath: path.join(tempDir, 'hnsw-default.dat'),
          forceStrategy: 'hnsw',
        });

        expect(buildResult.build_status).toBe('ready');

        const results = await indexMod.queryHnswIndex(embeddings[0].vector, {
          embeddingsDatabasePath: embeddingsDbPath,
        });
        expect(results.length).toBeGreaterThan(0);
      });
    } finally {
      await teardown(tempDir);
    }
  });

  it('returns error when HNSW is forced but the dependency is unavailable', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      const result = await indexMod.buildAnnIndex({
        embeddingsDatabasePath: embeddingsDbPath,
        forceStrategy: 'hnsw',
        hnswAvailable: false,
        modelId: 'test-model',
        dimension: 3,
      });

      expect(result.build_status).toBe('error');
    } finally {
      await teardown(tempDir);
    }
  });

  it('decodes embeddings shorter than the declared dimension', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      const db = new Database(embeddingsDbPath);
      const shortBuffer = Buffer.from(new Float32Array([1.0]).buffer);
      db.prepare(
        'INSERT INTO chunk_embeddings (chunk_id, model_id, dimension, embedding, chunk_sha256) VALUES (?, ?, ?, ?, ?)',
      ).run(1, 'short-model', 3, shortBuffer, 'sha');
      db.close();

      await withMockHnsw(async () => {
        const result = await indexMod.buildAnnIndex({
          embeddingsDatabasePath: embeddingsDbPath,
          forceStrategy: 'hnsw',
          modelId: 'short-model',
          dimension: 3,
        });

        expect(result.build_status).toBe('ready');
      });
    } finally {
      await teardown(tempDir);
    }
  });

  it('builds an HNSW index and populates metadata when dependency is present', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      insertSyntheticEmbeddings(embeddingsDbPath, 5, 4, 'test-model');

      await withMockHnsw(async () => {
        const result = await indexMod.buildAnnIndex({
          embeddingsDatabasePath: embeddingsDbPath,
          indexFilePath: path.join(tempDir, 'hnsw.dat'),
          forceStrategy: 'hnsw',
          modelId: 'test-model',
          dimension: 4,
        });

        const db = new Database(embeddingsDbPath);
        const meta = db
          .prepare(
            'SELECT index_type, current_elements FROM ann_index_meta WHERE index_id = ?',
          )
          .get(result.index_id);
        const mapCount = db
          .prepare(
            'SELECT COUNT(*) AS count FROM ann_index_chunk_map WHERE index_id = ?',
          )
          .get(result.index_id);
        db.close();

        expect({
          build_status: result.build_status,
          index_type: meta?.index_type,
          current_elements: meta?.current_elements,
          map_count: mapCount?.count,
        }).toEqual({
          build_status: 'ready',
          index_type: 'hnsw',
          current_elements: 5,
          map_count: 5,
        });
      });
    } finally {
      await teardown(tempDir);
    }
  });

  it('queries an HNSW index by explicit index id', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      const embeddings = insertSyntheticEmbeddings(
        embeddingsDbPath,
        5,
        4,
        'test-model',
      );

      await withMockHnsw(async () => {
        const buildResult = await indexMod.buildAnnIndex({
          embeddingsDatabasePath: embeddingsDbPath,
          indexFilePath: path.join(tempDir, 'hnsw.dat'),
          forceStrategy: 'hnsw',
          modelId: 'test-model',
          dimension: 4,
        });

        const queryVector = embeddings[0].vector;
        const results = await indexMod.queryHnswIndex(queryVector, {
          embeddingsDatabasePath: embeddingsDbPath,
          indexId: buildResult.index_id,
          k: 3,
          modelId: 'test-model',
          dimension: 4,
        });

        expect(results.length).toBeGreaterThan(0);
      });
    } finally {
      await teardown(tempDir);
    }
  });

  it('queries an HNSW index by auto-resolving the newest index', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      const embeddings = insertSyntheticEmbeddings(
        embeddingsDbPath,
        5,
        4,
        'test-model',
      );

      await withMockHnsw(async () => {
        await indexMod.buildAnnIndex({
          embeddingsDatabasePath: embeddingsDbPath,
          indexFilePath: path.join(tempDir, 'hnsw.dat'),
          forceStrategy: 'hnsw',
          modelId: 'test-model',
          dimension: 4,
        });

        const queryVector = embeddings[0].vector;
        const results = await indexMod.queryHnswIndex(queryVector, {
          embeddingsDatabasePath: embeddingsDbPath,
          k: 3,
          modelId: 'test-model',
          dimension: 4,
        });

        expect(results.length).toBeGreaterThan(0);
      });
    } finally {
      await teardown(tempDir);
    }
  });

  it('throws when queryHnswIndex cannot find an index', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      await withMockHnsw(async () => {
        await expect(
          indexMod.queryHnswIndex(new Float32Array([1, 0, 0, 0]), {
            embeddingsDatabasePath: embeddingsDbPath,
            modelId: 'missing-model',
            dimension: 4,
            k: 3,
          }),
        ).rejects.toThrow('No HNSW index found');
      });
    } finally {
      await teardown(tempDir);
    }
  });

  it('throws when an explicit HNSW index id has no file path', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      const db = new Database(embeddingsDbPath);
      db.exec(`
        CREATE TABLE IF NOT EXISTS ann_index_meta (
          index_id TEXT PRIMARY KEY,
          index_type TEXT NOT NULL,
          model_id TEXT NOT NULL,
          dimension INTEGER NOT NULL,
          index_file_path TEXT,
          updated_at TEXT NOT NULL DEFAULT(datetime('now'))
        );
        INSERT INTO ann_index_meta (index_id, index_type, model_id, dimension, index_file_path)
        VALUES ('hnsw_test-model_4', 'hnsw', 'test-model', 4, NULL);
      `);
      db.close();

      await withMockHnsw(async () => {
        await expect(
          indexMod.queryHnswIndex(new Float32Array([1, 0, 0, 0]), {
            embeddingsDatabasePath: embeddingsDbPath,
            indexId: 'hnsw_test-model_4',
            modelId: 'test-model',
            dimension: 4,
            k: 3,
          }),
        ).rejects.toThrow('No HNSW index found');
      });
    } finally {
      await teardown(tempDir);
    }
  });

  it('throws when the fallback meta lookup returns no row', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      const db = new Database(embeddingsDbPath);
      db.exec(`
        CREATE TABLE IF NOT EXISTS ann_index_meta (
          index_id TEXT PRIMARY KEY,
          index_type TEXT NOT NULL,
          model_id TEXT NOT NULL,
          dimension INTEGER NOT NULL,
          index_file_path TEXT,
          updated_at TEXT NOT NULL DEFAULT(datetime('now'))
        );
      `);
      db.close();

      await withMockHnsw(async () => {
        await expect(
          indexMod.queryHnswIndex(new Float32Array([1, 0, 0, 0]), {
            embeddingsDatabasePath: embeddingsDbPath,
            modelId: 'no-index-model',
            dimension: 4,
            k: 3,
          }),
        ).rejects.toThrow('No HNSW index found');
      });
    } finally {
      await teardown(tempDir);
    }
  });

  it('skips chunk map rows that are missing', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      const embeddings = insertSyntheticEmbeddings(
        embeddingsDbPath,
        5,
        4,
        'test-model',
      );

      await withMockHnsw(async () => {
        const buildResult = await indexMod.buildAnnIndex({
          embeddingsDatabasePath: embeddingsDbPath,
          indexFilePath: path.join(tempDir, 'hnsw.dat'),
          forceStrategy: 'hnsw',
          modelId: 'test-model',
          dimension: 4,
        });

        const db = new Database(embeddingsDbPath);
        db.prepare(
          'DELETE FROM ann_index_chunk_map WHERE index_id = ? AND external_id = ?',
        ).run(buildResult.index_id, 1);
        db.close();

        const queryVector = embeddings[0].vector;
        const results = await indexMod.queryHnswIndex(queryVector, {
          embeddingsDatabasePath: embeddingsDbPath,
          indexId: buildResult.index_id,
          k: 5,
          modelId: 'test-model',
          dimension: 4,
        });

        expect(results.length).toBe(4);
      });
    } finally {
      await teardown(tempDir);
    }
  });

  it('accepts a plain array embedding in queryHnswIndex', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      insertSyntheticEmbeddings(embeddingsDbPath, 5, 4, 'test-model');

      await withMockHnsw(async () => {
        await indexMod.buildAnnIndex({
          embeddingsDatabasePath: embeddingsDbPath,
          indexFilePath: path.join(tempDir, 'hnsw.dat'),
          forceStrategy: 'hnsw',
          modelId: 'test-model',
          dimension: 4,
        });

        const results = await indexMod.queryHnswIndex([1, 0, 0, 0], {
          embeddingsDatabasePath: embeddingsDbPath,
          k: 3,
          modelId: 'test-model',
          dimension: 4,
        });

        expect(results.length).toBeGreaterThan(0);
      });
    } finally {
      await teardown(tempDir);
    }
  });

  it('throws when queryHnswIndex is called without an embeddings database path', async () => {
    await withMockHnsw(async () => {
      await expect(
        indexMod.queryHnswIndex(new Float32Array([1, 0, 0, 0]), { k: 3 }),
      ).rejects.toThrow('embeddingsDatabasePath is required');
    });
  });

  it('throws when queryHnswIndex is called without options', async () => {
    await withMockHnsw(async () => {
      await expect(
        indexMod.queryHnswIndex(new Float32Array([1, 0, 0, 0])),
      ).rejects.toThrow('embeddingsDatabasePath is required');
    });
  });

  it('throws when hnswlib-node is unavailable', async () => {
    await expect(
      indexMod.queryHnswIndex(new Float32Array([1, 0, 0, 0]), {
        embeddingsDatabasePath: './missing.sqlite',
        k: 3,
      }),
    ).rejects.toThrow('hnswlib-node is not available');
  });

  it('demonstrates Recall@10 >= 0.95 against brute-force', async () => {
    const { embeddingsDbPath, tempDir } = await setupEmbeddingsDb();
    try {
      const dimension = 8;
      const count = 100;
      const modelId = 'recall-model';
      const embeddings = insertSyntheticEmbeddings(
        embeddingsDbPath,
        count,
        dimension,
        modelId,
      );

      await withMockHnsw(async () => {
        await indexMod.buildAnnIndex({
          embeddingsDatabasePath: embeddingsDbPath,
          indexFilePath: path.join(tempDir, 'hnsw.dat'),
          forceStrategy: 'hnsw',
          modelId,
          dimension,
        });

        const query = new Float32Array(
          Array.from({ length: dimension }, () => Math.random()),
        );
        const hnswResults = await indexMod.queryHnswIndex(query, {
          embeddingsDatabasePath: embeddingsDbPath,
          k: 10,
          modelId,
          dimension,
        });
        const bruteResults = bruteForceTopK(query, embeddings, 10);

        const hnswIds = hnswResults.map((r) => r.chunk_id);
        const bruteIds = bruteResults.map((r) => r.chunk_id);
        const intersection = bruteIds.filter((id) => hnswIds.includes(id));
        const recall = intersection.length / bruteIds.length;

        expect(recall).toBeGreaterThanOrEqual(0.95);
      });
    } finally {
      await teardown(tempDir);
    }
  });
});
