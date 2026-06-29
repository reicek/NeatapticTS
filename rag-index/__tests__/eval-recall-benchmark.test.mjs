/**
 * @module eval-recall-benchmark.test
 * @description Tests for Phase 4 Step 06 — DiskANN vector search recall and
 * performance benchmark (`runRecallBenchmark` and `computeVectorRecall`).
 *
 * Validates that:
 * - `computeVectorRecall` correctly computes recall@k from chunk ID lists.
 * - `runRecallBenchmark` returns the expected JSON report structure.
 * - The benchmark gracefully handles missing DiskANN indexes (ANN fallback).
 * - No bulk embedding loading occurs (server-side vector search only).
 * - Latency and heap metrics are present and numeric.
 *
 * Uses an in-memory `@libsql/client` database with the Turso schema (DiskANN
 * indexes skipped) and a mock embedder so tests run without ONNX models or a
 * real corpus database.
 *
 * Pure .mjs test — runs via Jest ESM project (no ts-jest).
 */

import { fileURLToPath } from 'node:url';
import path from 'node:path';
import {
  createSchemaClient,
  insertTestFixtures,
  insertEmbeddingFixtures,
  TEST_DIMENSION,
  TEST_DOC_ID,
  TEST_MODEL_ID,
} from './turso-test-helpers.mjs';
import { computeVectorRecall, runRecallBenchmark } from '../eval-runner.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------
// Mock embedder — returns a deterministic 384-dim Float32Array
// ---------------------------------------------------------------------------

/**
 * Create a mock embedder function that returns a deterministic embedding
 * based on the input text hash. Mimics the ONNX embedder interface:
 * `{ text } -> { embedding: Float32Array }` and includes a `release()` method.
 *
 * @returns {{ (input: { text: string }) => Promise<{ embedding: Float32Array, release: () => void }> }}
 */
function createMockEmbedder() {
  const embed = async ({ text }) => {
    const embedding = new Float32Array(TEST_DIMENSION);
    // Simple deterministic hash → embedding so different texts produce
    // different vectors (needed for recall to be non-trivial).
    const hash = text
      .split('')
      .reduce((sum, char) => sum + char.charCodeAt(0), 0);
    for (let i = 0; i < TEST_DIMENSION; i += 1) {
      embedding[i] = ((hash + i * 7) % 100) / 100.0;
    }
    return embedding;
  };
  embed.release = () => {};
  return embed;
}

// ---------------------------------------------------------------------------
// Helper — insert multiple chunks with distinct embeddings
// ---------------------------------------------------------------------------

/**
 * Insert multiple chunks with distinct embeddings into a schema-loaded client.
 *
 * Creates N documents and N chunks, each with a deterministic but distinct
 * embedding vector so brute-force and ANN queries return meaningful results.
 *
 * @param {import('@libsql/client').Client} client - Schema-loaded client.
 * @param {number} count - Number of chunk+document pairs to insert.
 * @returns {Promise<number[]>} Array of inserted chunk IDs.
 */
async function insertMultipleChunks(client, count) {
  const chunkIds = [];
  for (let i = 0; i < count; i += 1) {
    const docId = TEST_DOC_ID + i + 1;
    const chunkId = 700000 + i;
    chunkIds.push(chunkId);

    await client.execute({
      sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at, arch_layer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      args: [
        docId,
        `src/bench-test-${i}.ts`,
        'bench-test',
        1000 + i,
        500,
        `bench-sha256-${i}`,
        1000 + i,
        'network',
      ],
    });

    await client.execute({
      sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth, symbol_name, module_path, arch_layer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      args: [
        chunkId,
        docId,
        0,
        `BenchTestModule${i}`,
        `bench test content ${i} for recall benchmark`,
        0,
        40,
        null,
        0,
        `BenchTestSymbol${i}`,
        `src/bench-test-${i}.ts`,
        'network',
      ],
    });

    // Insert a distinct embedding for this chunk.
    const embedding = new Float32Array(TEST_DIMENSION);
    for (let d = 0; d < TEST_DIMENSION; d += 1) {
      embedding[d] = ((i * 13 + d * 7) % 100) / 100.0;
    }
    const embeddingBuffer = Buffer.from(
      embedding.buffer,
      embedding.byteOffset,
      embedding.byteLength,
    );
    await client.execute({
      sql: `UPDATE chunks SET embedding = vector8(?), embedding_model = ?, chunk_sha256 = ?, embedded_at = ? WHERE chunk_id = ?`,
      args: [
        embeddingBuffer,
        TEST_MODEL_ID,
        `bench-chunk-sha-${i}`,
        1000,
        chunkId,
      ],
    });
  }
  return chunkIds;
}

// ---------------------------------------------------------------------------
// computeVectorRecall — pure function tests
// ---------------------------------------------------------------------------

describe('computeVectorRecall', () => {
  it('returns 1.0 when both lists are identical', () => {
    const ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const recall = computeVectorRecall(ids, ids, 10);
    expect(recall).toBe(1);
  });

  it('returns 0.0 when lists are completely disjoint', () => {
    const annIds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const bfIds = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    const recall = computeVectorRecall(annIds, bfIds, 10);
    expect(recall).toBe(0);
  });

  it('returns 0.8 when 4 of 5 ANN IDs match brute-force top-5', () => {
    const annIds = [1, 2, 3, 4, 5];
    const bfIds = [1, 2, 3, 4, 6];
    const recall = computeVectorRecall(annIds, bfIds, 5);
    expect(recall).toBe(0.8);
  });

  it('returns 0 when ANN list is empty', () => {
    const recall = computeVectorRecall([], [1, 2, 3], 3);
    expect(recall).toBe(0);
  });

  it('returns 0 when brute-force list is empty', () => {
    const recall = computeVectorRecall([1, 2, 3], [], 3);
    expect(recall).toBe(0);
  });

  it('respects the k parameter by only considering top-k from each list', () => {
    const annIds = [1, 2, 3, 4, 5, 6, 7, 8];
    const bfIds = [5, 6, 7, 8, 1, 2, 3, 4];
    // k=4: ANN top-4 = {1,2,3,4}, BF top-4 = {5,6,7,8} → 0 overlap
    const recall = computeVectorRecall(annIds, bfIds, 4);
    expect(recall).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// runRecallBenchmark — integration tests with in-memory client
// ---------------------------------------------------------------------------

describe('runRecallBenchmark', () => {
  it('returns a JSON report with the expected structure', async () => {
    const client = await createSchemaClient();
    try {
      await insertMultipleChunks(client, 15);
      const mockEmbedder = createMockEmbedder();

      const report = await runRecallBenchmark({
        client,
        embedText: mockEmbedder,
        modelId: TEST_MODEL_ID,
        dimension: TEST_DIMENSION,
        queries: [
          { query: 'bench test content 0' },
          { query: 'bench test content 1' },
        ],
        k: 10,
        minRecall: 0.9,
        maxLatencyMs: 5000,
        maxHeapDeltaMb: 500,
      });

      expect(report).toBeDefined();
      expect(report.benchmark).toBe('recall-benchmark');
      expect(report.query_count).toBe(2);
      expect(report.k).toBe(10);
      expect(typeof report.ann_available).toBe('boolean');
      expect(typeof report.recall_at_k).toBe('number');
      expect(typeof report.ann_latency_ms).toBe('number');
      expect(typeof report.brute_force_latency_ms).toBe('number');
      expect(typeof report.heap_before_mb).toBe('number');
      expect(typeof report.heap_after_mb).toBe('number');
      expect(typeof report.heap_delta_mb).toBe('number');
      expect(report.thresholds).toBeDefined();
      expect(report.criteria).toBeDefined();
      expect(typeof report.pass).toBe('boolean');
      expect(Array.isArray(report.per_query)).toBe(true);
      expect(report.per_query.length).toBe(2);
    } finally {
      await client.close();
    }
  });

  it('reports ann_available=false when DiskANN index is not present', async () => {
    const client = await createSchemaClient();
    try {
      await insertMultipleChunks(client, 5);
      const mockEmbedder = createMockEmbedder();

      const report = await runRecallBenchmark({
        client,
        embedText: mockEmbedder,
        modelId: TEST_MODEL_ID,
        dimension: TEST_DIMENSION,
        queries: [{ query: 'bench test content 0' }],
        k: 5,
      });

      // In-memory client skips DiskANN indexes, so ANN should be unavailable.
      expect(report.ann_available).toBe(false);
      // recall_pass should be null when ANN is unavailable.
      expect(report.criteria.recall_pass).toBeNull();
    } finally {
      await client.close();
    }
  });

  it('computes brute-force results correctly even when ANN is unavailable', async () => {
    const client = await createSchemaClient();
    try {
      await insertMultipleChunks(client, 10);
      const mockEmbedder = createMockEmbedder();

      const report = await runRecallBenchmark({
        client,
        embedText: mockEmbedder,
        modelId: TEST_MODEL_ID,
        dimension: TEST_DIMENSION,
        queries: [{ query: 'bench test content 0' }],
        k: 5,
      });

      // Even without ANN, brute-force should return chunk IDs.
      const perQuery = report.per_query[0];
      expect(perQuery).toBeDefined();
      expect(perQuery.brute_force_chunk_ids.length).toBeGreaterThan(0);
      expect(perQuery.brute_force_latency_ms).toBeGreaterThanOrEqual(0);
    } finally {
      await client.close();
    }
  });

  it('includes per-query recall, latency, and chunk ID arrays', async () => {
    const client = await createSchemaClient();
    try {
      await insertMultipleChunks(client, 8);
      const mockEmbedder = createMockEmbedder();

      const report = await runRecallBenchmark({
        client,
        embedText: mockEmbedder,
        modelId: TEST_MODEL_ID,
        dimension: TEST_DIMENSION,
        queries: [
          { query: 'bench test content 0' },
          { query: 'bench test content 1' },
        ],
        k: 5,
      });

      for (const perQuery of report.per_query) {
        expect(typeof perQuery.query).toBe('string');
        expect(Array.isArray(perQuery.ann_chunk_ids)).toBe(true);
        expect(Array.isArray(perQuery.brute_force_chunk_ids)).toBe(true);
        expect(typeof perQuery.recall_at_k).toBe('number');
        expect(typeof perQuery.ann_latency_ms).toBe('number');
        expect(typeof perQuery.brute_force_latency_ms).toBe('number');
      }
    } finally {
      await client.close();
    }
  });

  it('passes when thresholds are relaxed and brute-force path works', async () => {
    const client = await createSchemaClient();
    try {
      await insertMultipleChunks(client, 5);
      const mockEmbedder = createMockEmbedder();

      const report = await runRecallBenchmark({
        client,
        embedText: mockEmbedder,
        modelId: TEST_MODEL_ID,
        dimension: TEST_DIMENSION,
        queries: [{ query: 'bench test content 0' }],
        k: 5,
        minRecall: 0,
        maxLatencyMs: 60000,
        maxHeapDeltaMb: 1000,
      });

      // With relaxed thresholds, should pass (recall_pass is null since ANN
      // is unavailable, which is not false; latency and memory should pass).
      expect(report.pass).toBe(true);
      expect(report.criteria.latency_pass).toBe(true);
      expect(report.criteria.memory_pass).toBe(true);
    } finally {
      await client.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Source-level checks — no bulk embedding loading
// ---------------------------------------------------------------------------

describe('runRecallBenchmark: no JS-side bulk embedding loading', () => {
  it('eval-runner.mjs does not SELECT all embeddings from chunks', async () => {
    const { readFile } = await import('node:fs/promises');
    const source = await readFile(
      path.resolve(__dirname, '..', 'eval-runner.mjs'),
      'utf8',
    );
    // The recall benchmark must NOT load all embeddings into JS memory.
    // It should only SELECT chunk_id (not embedding column) for recall computation.
    expect(source).not.toMatch(/SELECT\s+embedding\s+FROM\s+chunks/i);
  });

  it('eval-runner.mjs recall benchmark only selects chunk_id for recall', async () => {
    const { readFile } = await import('node:fs/promises');
    const source = await readFile(
      path.resolve(__dirname, '..', 'eval-runner.mjs'),
      'utf8',
    );
    // The ANN and brute-force queries should only SELECT chunk_id, not the
    // embedding column, to avoid loading vectors into JS memory.
    expect(source).toMatch(/SELECT\s+c\.chunk_id/i);
  });
});
