/**
 * @module embed-turso.test
 * @description Red tests for Phase 4 Step 02 — embedding storage and read-path
 * migration (Float32 BLOB → F8_BLOB + server-side vector_distance_cos).
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because the implementation does not exist yet (not because of syntax errors).
 *
 * Coverage targets:
 * - embed-index.mjs writes embeddings using vector8() (F8_BLOB, not Float32 BLOB)
 * - build-term-index.mjs writes term_embeddings using vector8() (F8_BLOB)
 * - query-dense.mjs uses server-side vector_distance_cos() (no JS-side loading)
 * - Old chunk_embeddings table references REMOVED from query-dense.mjs
 * - loadDenseCandidateRows REMOVED from query-dense.mjs
 * - loadEmbeddingsForChunks REMOVED from query-dense.mjs
 * - computeCosineSimilarity usage REMOVED from query-dense.mjs
 * - decodeEmbeddingBlob REMOVED from query-dense.mjs
 * - F8 embedding buffer is 384 bytes (4x compression from 1536-byte F32)
 *
 * Pure .mjs test — runs via Jest ESM project (no ts-jest).
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { closeTursoClient } from '../../mcp-semantic/tools/cortex-db.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const EMBED_INDEX_PATH = path.resolve(__dirname, '..', 'embed-index.mjs');
const QUERY_DENSE_PATH = path.resolve(__dirname, '..', 'query-dense.mjs');
const BUILD_TERM_INDEX_PATH = path.resolve(
  __dirname,
  '..',
  'build-term-index.mjs',
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
// F8_BLOB embedding writes — embed-index.mjs
// ---------------------------------------------------------------------------

describe('embed-turso: embed-index.mjs F8_BLOB writes', () => {
  it('uses vector8() to quantize embeddings to F8_BLOB in the UPDATE SQL', async () => {
    const source = await readSource(EMBED_INDEX_PATH);
    expect(source).toMatch(/vector8\s*\(/i);
  });

  it('does not write embeddings to the old chunk_embeddings table', async () => {
    const source = await readSource(EMBED_INDEX_PATH);
    expect(source).not.toMatch(/chunk_embeddings/i);
  });

  it('writes embeddings to the chunks.embedding column', async () => {
    const source = await readSource(EMBED_INDEX_PATH);
    expect(source).toMatch(/UPDATE\s+chunks\s+SET\s+embedding/i);
  });
});

// ---------------------------------------------------------------------------
// F8_BLOB term embeddings — build-term-index.mjs
// ---------------------------------------------------------------------------

describe('embed-turso: build-term-index.mjs F8_BLOB term embeddings', () => {
  it('uses vector8() to quantize term embeddings to F8_BLOB', async () => {
    const source = await readSource(BUILD_TERM_INDEX_PATH);
    expect(source).toMatch(/vector8\s*\(/i);
  });
});

// ---------------------------------------------------------------------------
// 4x compression — F8 buffer size
// ---------------------------------------------------------------------------

describe('embed-turso: F8_BLOB 4x compression', () => {
  it('produces 384-byte embedding buffers for 384-dim vectors (not 1536)', async () => {
    const { buildEmbeddingIndex } = await import('../embed-index.mjs');
    // An F8 (int8) embedding for 384 dimensions is 384 bytes.
    // An F32 (float32) embedding for 384 dimensions is 1536 bytes.
    // The toBlobBuffer / F8 conversion helper must produce 384-byte buffers.
    //
    // We test the exported conversion path by creating a normalized 384-dim
    // vector and checking the quantized buffer length.
    const dimension = 384;
    const float32Vector = new Float32Array(dimension);
    for (let i = 0; i < dimension; i += 1) {
      float32Vector[i] = (i % 7) / 10.0;
    }

    // The implementation should export a quantization helper (e.g.
    // toF8BlobBuffer or the buildEmbeddingIndex internals should use it).
    // We attempt to import a quantization helper; if it does not exist yet
    // the import fails — which is the expected red state.
    const module = await import('../embed-index.mjs');
    const quantize =
      module.toF8BlobBuffer ?? module.quantizeToF8 ?? module.toF8Buffer;
    expect(quantize).toBeDefined();

    const f8Buffer = quantize(float32Vector);
    expect(f8Buffer.byteLength).toBe(384);
  });
});

// ---------------------------------------------------------------------------
// Server-side vector_distance_cos reads — query-dense.mjs
// ---------------------------------------------------------------------------

describe('embed-turso: query-dense.mjs server-side vector_distance_cos', () => {
  it('uses vector_distance_cos() in the dense query SQL', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).toMatch(/vector_distance_cos\s*\(/i);
  });

  it('passes the query embedding as a vector8() parameter in the SQL', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).toMatch(/vector8\s*\(\s*\?\s*\)/i);
  });
});

// ---------------------------------------------------------------------------
// Old code removal — query-dense.mjs
// ---------------------------------------------------------------------------

describe('embed-turso: old chunk_embeddings read path REMOVED', () => {
  it('does not reference the chunk_embeddings table', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).not.toMatch(/chunk_embeddings/i);
  });

  it('does not define or call loadDenseCandidateRows', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).not.toMatch(/loadDenseCandidateRows/i);
  });

  it('does not define or call loadEmbeddingsForChunks', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).not.toMatch(/loadEmbeddingsForChunks/i);
  });

  it('does not import or use computeCosineSimilarity', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).not.toMatch(/computeCosineSimilarity/i);
  });

  it('does not define or call decodeEmbeddingBlob', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).not.toMatch(/decodeEmbeddingBlob/i);
  });
});

// ---------------------------------------------------------------------------
// Old code removal — embed-index.mjs
// ---------------------------------------------------------------------------

describe('embed-turso: old Float32 BLOB write path REMOVED from embed-index.mjs', () => {
  it('does not use a Float32-only toBlobBuffer write path', async () => {
    const source = await readSource(EMBED_INDEX_PATH);
    // The old write path used toBlobBuffer(Float32Array) producing 1536-byte
    // BLOBs. After migration, the write path must use vector8() quantization.
    // We assert the source no longer contains a bare toBlobBuffer that
    // operates on Float32Array without F8 quantization.
    expect(source).not.toMatch(/toBlobBuffer\s*\(/i);
  });
});

// ---------------------------------------------------------------------------
// Incremental rule preservation
// ---------------------------------------------------------------------------

describe('embed-turso: incremental embedding rule preserved', () => {
  it('still checks chunk_sha256 and embedding_model for incremental skips', async () => {
    const source = await readSource(EMBED_INDEX_PATH);
    expect(source).toMatch(/chunk_sha256/i);
    expect(source).toMatch(/embedding_model/i);
  });
});