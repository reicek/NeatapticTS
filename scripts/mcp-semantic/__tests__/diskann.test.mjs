/**
 * @module diskann.test
 * @description Red tests for Phase 4 Step 03 — DiskANN index migration
 * (hnswlib-node + brute_force fallback → libsql_vector_idx DiskANN).
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because the implementation does not exist yet (not because of syntax errors).
 *
 * Coverage targets:
 * - ann-index.mjs creates a DiskANN vector index using libsql_vector_idx on
 *   chunks.embedding
 * - DiskANN index is configured with metric=cosine, max_neighbors, alpha=1.2
 * - buildAnnIndex executes the DiskANN CREATE INDEX and no longer references
 *   hnswlib-node or HierarchicalNSW
 * - Old ANN tables (ann_index_meta, ann_index_chunk_map) REMOVED from
 *   ann-index.mjs
 * - Old ANN helpers (ensureAnnTablesAsync, populateChunkMapAsync,
 *   clearChunkMapAsync) REMOVED from ann-index.mjs
 * - ann-strategy.mjs simplified to a single DiskANN strategy (brute_force,
 *   brute_force_cached, hnsw strategies REMOVED)
 * - HNSW availability / construction parameters REMOVED from ann-strategy.mjs
 * - ann_build_index MCP tool force enum no longer lists the old strategies
 *
 * Pure .mjs test — runs via Jest ESM project (no ts-jest).
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { closeTursoClient } from '../tools/cortex-db.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ANN_INDEX_PATH = path.resolve(__dirname, '..', 'tools', 'ann-index.mjs');
const ANN_STRATEGY_PATH = path.resolve(
  __dirname,
  '..',
  'tools',
  'ann-strategy.mjs',
);
const REPO_CORTEX_MCP_PATH = path.resolve(
  __dirname,
  '..',
  'repo-cortex-mcp.mjs',
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
// DiskANN index creation SQL — ann-index.mjs
// ---------------------------------------------------------------------------

describe('diskann: DiskANN index creation SQL in ann-index.mjs', () => {
  it('creates a DiskANN vector index using libsql_vector_idx on chunks.embedding', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).toMatch(
      /CREATE\s+INDEX[^;]*libsql_vector_idx\s*\(\s*embedding\s*\)/i,
    );
  });

  it('configures the DiskANN index with metric=cosine', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).toMatch(/metric\s*=\s*['"]cosine['"]/i);
  });

  it('configures the DiskANN index with max_neighbors', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).toMatch(/max_neighbors/i);
  });

  it('configures the DiskANN index with alpha', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).toMatch(/alpha\s*=\s*1\.2/i);
  });
});

// ---------------------------------------------------------------------------
// buildAnnIndex DiskANN usage — ann-index.mjs
// ---------------------------------------------------------------------------

describe('diskann: buildAnnIndex uses DiskANN SQL', () => {
  it('buildAnnIndex executes CREATE INDEX with libsql_vector_idx', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).toMatch(/libsql_vector_idx/i);
  });

  it('buildAnnIndex no longer references hnswlib-node or HierarchicalNSW', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).not.toMatch(/hnswlib-node|HierarchicalNSW/i);
  });
});

// ---------------------------------------------------------------------------
// Old ANN tables and functions REMOVED — ann-index.mjs
// ---------------------------------------------------------------------------

describe('diskann: old ANN tables and functions REMOVED from ann-index.mjs', () => {
  it('does not create the ann_index_meta table', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).not.toMatch(/ann_index_meta/i);
  });

  it('does not create the ann_index_chunk_map table', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).not.toMatch(/ann_index_chunk_map/i);
  });

  it('does not define ensureAnnTablesAsync', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).not.toMatch(/ensureAnnTablesAsync/i);
  });

  it('does not define populateChunkMapAsync', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).not.toMatch(/populateChunkMapAsync/i);
  });

  it('does not define clearChunkMapAsync', async () => {
    const source = await readSource(ANN_INDEX_PATH);
    expect(source).not.toMatch(/clearChunkMapAsync/i);
  });
});

// ---------------------------------------------------------------------------
// Strategy simplification — ann-strategy.mjs
// ---------------------------------------------------------------------------

describe('diskann: ann-strategy.mjs simplified to DiskANN-only', () => {
  it('does not export the brute_force strategy', async () => {
    const source = await readSource(ANN_STRATEGY_PATH);
    expect(source).not.toMatch(/['"]brute_force['"]/i);
  });

  it('does not export the brute_force_cached strategy', async () => {
    const source = await readSource(ANN_STRATEGY_PATH);
    expect(source).not.toMatch(/brute_force_cached/i);
  });

  it('does not export the hnsw strategy', async () => {
    const source = await readSource(ANN_STRATEGY_PATH);
    expect(source).not.toMatch(/['"]hnsw['"]/i);
  });

  it('does not reference hnswlib-node availability', async () => {
    const source = await readSource(ANN_STRATEGY_PATH);
    expect(source).not.toMatch(/isHnswAvailable|hnswlib-node|__hnswTestSeam/i);
  });

  it('does not export HNSW construction parameters', async () => {
    const source = await readSource(ANN_STRATEGY_PATH);
    expect(source).not.toMatch(
      /DEFAULT_HNSW_M|DEFAULT_HNSW_EF_CONSTRUCTION|DEFAULT_HNSW_EF_SEARCH/i,
    );
  });

  it('references the DiskANN strategy', async () => {
    const source = await readSource(ANN_STRATEGY_PATH);
    expect(source).toMatch(/diskann/i);
  });
});

// ---------------------------------------------------------------------------
// MCP tool force enum — repo-cortex-mcp.mjs
// ---------------------------------------------------------------------------

describe('diskann: ann_build_index MCP tool force enum updated', () => {
  it('no longer accepts hnsw as a force strategy in the MCP tool schema', async () => {
    const source = await readSource(REPO_CORTEX_MCP_PATH);
    expect(source).not.toMatch(
      /['"]hnsw['"]\s*,\s*['"]brute_force_cached['"]\s*,\s*['"]brute_force['"]/,
    );
  });
});