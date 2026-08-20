/**
 * @module embed-index.coverage.test
 * @description Coverage tests targeting rag-index/embed-index.mjs — exercises
 * buildEmbeddingIndex, normalizeEmbeddingVector, readModelMeta, createOnnxTextEmbedder,
 * toF8BlobBuffer, and internal helpers (extractSliceMetadata, createChunkSha256,
 * toFloat32Array, releaseEmbedText, readJsonFile, resolveSpecialToken, etc.).
 */
import { jest } from '@jest/globals';
import { createClient } from '@libsql/client';
import { writeFile, mkdir, rm, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Mock onnxruntime-node to prevent native bindings from loading under
// Jest ESM VM modules, which causes a native crash during cleanup.
jest.unstable_mockModule('onnxruntime-node', () => ({
  InferenceSession: {
    create: async () => {
      throw new Error(
        'Mocked: onnxruntime-node is not available in test environment',
      );
    },
  },
  Tensor: class MockTensor {},
  __esModule: true,
}));

const {
  buildEmbeddingIndex,
  normalizeEmbeddingVector,
  readModelMeta,
  createOnnxTextEmbedder,
  toF8BlobBuffer,
  DEFAULT_MODEL_ID,
  DEFAULT_MODEL_DIRECTORY,
} = await import('../../../rag-index/embed-index.mjs');

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEMP_DIR = path.join(__dirname, '.embed-tmp');

const savedEnv = { ...process.env };

beforeAll(async () => {
  await mkdir(TEMP_DIR, { recursive: true });
});

afterAll(async () => {
  await rm(TEMP_DIR, { recursive: true, force: true });
  process.env = savedEnv;
});

beforeEach(() => {
  delete process.env.DENSE_FORCE_STATE;
});

afterEach(() => {
  delete process.env.DENSE_FORCE_STATE;
});

// ---------------------------------------------------------------------------
// normalizeEmbeddingVector
// ---------------------------------------------------------------------------
describe('embed-index — normalizeEmbeddingVector', () => {
  it('normalizes a Float32Array', () => {
    const input = new Float32Array([3, 4]);
    const result = normalizeEmbeddingVector(input, 2);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result[0]).toBeCloseTo(0.6, 5);
    expect(result[1]).toBeCloseTo(0.8, 5);
  });

  it('returns zero vector unchanged when magnitude is zero', () => {
    const input = new Float32Array([0, 0]);
    const result = normalizeEmbeddingVector(input, 2);
    expect(result).toEqual(new Float32Array([0, 0]));
  });

  it('accepts a regular array', () => {
    const result = normalizeEmbeddingVector([3, 4], 2);
    expect(result[0]).toBeCloseTo(0.6, 5);
    expect(result[1]).toBeCloseTo(0.8, 5);
  });

  it('accepts an ArrayBuffer view (Int32Array)', () => {
    const view = new Int32Array([3, 4]);
    const result = normalizeEmbeddingVector(view, 2);
    expect(result[0]).toBeCloseTo(0.6, 5);
    expect(result[1]).toBeCloseTo(0.8, 5);
  });

  it('throws when dimension does not match', () => {
    expect(() =>
      normalizeEmbeddingVector(new Float32Array([1, 2, 3]), 2),
    ).toThrow('Expected embedding dimension 2, received 3');
  });

  it('handles single-element vector', () => {
    const result = normalizeEmbeddingVector(new Float32Array([5]), 1);
    expect(result[0]).toBeCloseTo(1, 5);
  });

  it('handles non-array, non-typed-array input (toFloat32Array fallback branch)', () => {
    // Pass a plain object — not Float32Array, not ArrayBuffer view, not array
    // toFloat32Array falls through to Float32Array.from([]) → empty, dimension 0 matches
    const result = normalizeEmbeddingVector({}, 0);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// toF8BlobBuffer
// ---------------------------------------------------------------------------
describe('embed-index — toF8BlobBuffer', () => {
  it('quantizes a Float32Array to int8 buffer', () => {
    const input = new Float32Array([1, -1, 0.5, -0.5, 0]);
    const buffer = toF8BlobBuffer(input);
    expect(buffer).toBeInstanceOf(Buffer);
    expect(buffer.length).toBe(5);
    expect(buffer.readInt8(0)).toBe(127);
    expect(buffer.readInt8(1)).toBe(-127);
    expect(buffer.readInt8(2)).toBe(Math.round(0.5 * 127));
    expect(buffer.readInt8(3)).toBe(Math.round(-0.5 * 127));
    expect(buffer.readInt8(4)).toBe(0);
  });

  it('clamps values outside [-1, 1]', () => {
    const input = new Float32Array([2, -2]);
    const buffer = toF8BlobBuffer(input);
    expect(buffer.readInt8(0)).toBe(127);
    expect(buffer.readInt8(1)).toBe(-127);
  });
});

// ---------------------------------------------------------------------------
// readModelMeta
// ---------------------------------------------------------------------------
describe('embed-index — readModelMeta', () => {
  it('returns modelMeta when passed directly', async () => {
    const meta = { dimension: 384, model_sha256: 'abc' };
    const result = await readModelMeta({ modelMeta: meta });
    expect(result).toBe(meta);
  });

  it('reads model-meta.json from modelMetaPath', async () => {
    const metaPath = path.join(TEMP_DIR, 'model-meta.json');
    await writeFile(
      metaPath,
      JSON.stringify({ dimension: 128, model_sha256: 'def' }),
    );
    const result = await readModelMeta({ modelMetaPath: metaPath });
    expect(result).toEqual({ dimension: 128, model_sha256: 'def' });
  });

  it('returns empty object when model-meta.json does not exist (ENOENT)', async () => {
    const result = await readModelMeta({
      modelMetaPath: path.join(TEMP_DIR, 'nonexistent.json'),
    });
    expect(result).toEqual({});
  });

  it('reads from modelDirectory/model-meta.json by default', async () => {
    const modelDir = path.join(TEMP_DIR, 'model-dir-' + Date.now());
    await mkdir(modelDir, { recursive: true });
    await writeFile(
      path.join(modelDir, 'model-meta.json'),
      JSON.stringify({ dimension: 256, model_sha256: 'xyz' }),
    );
    const result = await readModelMeta({ modelDirectory: modelDir });
    expect(result).toEqual({ dimension: 256, model_sha256: 'xyz' });
  });

  it('throws on invalid JSON in model-meta.json', async () => {
    const metaPath = path.join(TEMP_DIR, 'bad-meta.json');
    await writeFile(metaPath, '{ invalid json }');
    await expect(readModelMeta({ modelMetaPath: metaPath })).rejects.toThrow();
  });
});

// ---------------------------------------------------------------------------
// createOnnxTextEmbedder
// ---------------------------------------------------------------------------
describe('embed-index — createOnnxTextEmbedder (DENSE_FORCE_STATE)', () => {
  it('throws when DENSE_FORCE_STATE=cold', async () => {
    process.env.DENSE_FORCE_STATE = 'cold';
    await expect(createOnnxTextEmbedder({ dimension: 384 })).rejects.toThrow(
      'ONNX embedder disabled because DENSE_FORCE_STATE=cold',
    );
  });

  it('throws when DENSE_FORCE_STATE=model-only', async () => {
    process.env.DENSE_FORCE_STATE = 'model-only';
    await expect(createOnnxTextEmbedder({ dimension: 384 })).rejects.toThrow(
      'ONNX embedder disabled because DENSE_FORCE_STATE=model-only',
    );
  });
});

describe('embed-index — createOnnxTextEmbedder (tokenizer & config)', () => {
  /** Copy the real tokenizer.json from the models directory into a temp dir. */
  async function copyRealTokenizer(modelDir) {
    const realTokenizer = await readFile(
      path.join(DEFAULT_MODEL_DIRECTORY, 'tokenizer.json'),
      'utf8',
    );
    await writeFile(path.join(modelDir, 'tokenizer.json'), realTokenizer);
  }

  it('throws when tokenizer.json is missing vocabulary', async () => {
    const modelDir = path.join(TEMP_DIR, 'no-vocab-' + Date.now());
    await mkdir(modelDir, { recursive: true });
    await writeFile(
      path.join(modelDir, 'tokenizer.json'),
      JSON.stringify({ model: {} }),
    );
    await expect(
      createOnnxTextEmbedder({ modelDirectory: modelDir }),
    ).rejects.toThrow('tokenizer.json is missing the WordPiece vocabulary');
  });

  it('throws when tokenizer_config.json has invalid JSON', async () => {
    const modelDir = path.join(TEMP_DIR, 'bad-config-' + Date.now());
    await mkdir(modelDir, { recursive: true });
    await copyRealTokenizer(modelDir);
    await writeFile(
      path.join(modelDir, 'tokenizer_config.json'),
      '{ invalid json }',
    );
    await expect(
      createOnnxTextEmbedder({ modelDirectory: modelDir }),
    ).rejects.toThrow();
  });

  it('reaches ONNX session creation with valid tokenizer and string unk_token', async () => {
    const modelDir = path.join(TEMP_DIR, 'valid-str-' + Date.now());
    await mkdir(modelDir, { recursive: true });
    await copyRealTokenizer(modelDir);
    await writeFile(
      path.join(modelDir, 'tokenizer_config.json'),
      JSON.stringify({ do_lower_case: true }),
    );
    await writeFile(
      path.join(modelDir, 'special_tokens_map.json'),
      JSON.stringify({ unk_token: '[UNK]' }),
    );
    // Will fail at InferenceSession.create because model.onnx doesn't exist
    await expect(
      createOnnxTextEmbedder({ modelDirectory: modelDir }),
    ).rejects.toThrow();
  }, 30000);

  it('handles object unk_token in special_tokens_map', async () => {
    const modelDir = path.join(TEMP_DIR, 'obj-unk-' + Date.now());
    await mkdir(modelDir, { recursive: true });
    await copyRealTokenizer(modelDir);
    await writeFile(
      path.join(modelDir, 'tokenizer_config.json'),
      JSON.stringify({ do_lower_case: true }),
    );
    await writeFile(
      path.join(modelDir, 'special_tokens_map.json'),
      JSON.stringify({ unk_token: { type: 'SpecialToken', content: '[UNK]' } }),
    );
    await expect(
      createOnnxTextEmbedder({ modelDirectory: modelDir }),
    ).rejects.toThrow();
  }, 30000);

  it('uses fallback unk_token when special_tokens_map is missing', async () => {
    const modelDir = path.join(TEMP_DIR, 'no-stm-' + Date.now());
    await mkdir(modelDir, { recursive: true });
    await copyRealTokenizer(modelDir);
    await writeFile(
      path.join(modelDir, 'tokenizer_config.json'),
      JSON.stringify({ do_lower_case: true }),
    );
    // No special_tokens_map.json — resolveSpecialToken should use fallback
    await expect(
      createOnnxTextEmbedder({ modelDirectory: modelDir }),
    ).rejects.toThrow();
  }, 30000);

  it('handles missing tokenizer_config.json (ENOENT fallback)', async () => {
    const modelDir = path.join(TEMP_DIR, 'no-tc-' + Date.now());
    await mkdir(modelDir, { recursive: true });
    await copyRealTokenizer(modelDir);
    await writeFile(
      path.join(modelDir, 'special_tokens_map.json'),
      JSON.stringify({ unk_token: '[UNK]' }),
    );
    // No tokenizer_config.json — readJsonFile should return {} (ENOENT)
    await expect(
      createOnnxTextEmbedder({ modelDirectory: modelDir }),
    ).rejects.toThrow();
  }, 30000);
});

// ---------------------------------------------------------------------------
// buildEmbeddingIndex
// ---------------------------------------------------------------------------
describe('embed-index — buildEmbeddingIndex', () => {
  /**
   * Create an in-memory test database with chunks and documents tables.
   */
  async function createTestDb() {
    const client = createClient({ url: ':memory:' });
    await client.execute(`
      CREATE TABLE documents (
        doc_id INTEGER PRIMARY KEY,
        file_path TEXT NOT NULL UNIQUE,
        doc_family TEXT NOT NULL,
        mtime_ms INTEGER NOT NULL,
        file_size INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        indexed_at INTEGER NOT NULL
      )
    `);
    await client.execute(`
      CREATE TABLE chunks (
        chunk_id INTEGER PRIMARY KEY,
        doc_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        heading_path TEXT,
        body_text TEXT NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        parent_chunk_id INTEGER,
        depth INTEGER DEFAULT 0,
        context_header TEXT,
        symbol_name TEXT,
        signature_text TEXT,
        jsdoc_text TEXT,
        export_type TEXT,
        module_path TEXT,
        slice_id TEXT,
        step_number INTEGER,
        phase TEXT,
        status TEXT,
        arch_layer TEXT,
        test_coverage TEXT,
        jsdoc_quality TEXT,
        jsdoc_word_count INTEGER,
        cyclomatic_complexity INTEGER,
        source_path_pattern TEXT,
        chunk_sha256 TEXT,
        embedding BLOB,
        embedding_model TEXT,
        embedded_at INTEGER
      )
    `);
    await client.execute({
      sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
           VALUES (1, 'src/test.ts', 'ts-source', 1000, 200, 'abc', 1000)`,
    });
    await client.execute({
      sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_start, char_end, depth)
           VALUES (1, 1, 0, 'hello world test content', 0, 26, 0)`,
    });
    await client.execute({
      sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_start, char_end, depth)
           VALUES (2, 1, 1, 'second chunk body text', 26, 47, 0)`,
    });
    return client;
  }

  it('embeds all chunks with provided embedText and client', async () => {
    const client = await createTestDb();
    const embedText = jest.fn(async ({ text }) => {
      const v = new Float32Array(384);
      v.fill(0.1);
      return v;
    });

    const summary = await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    expect(summary.embedded).toBe(2);
    expect(summary.skipped).toBe(0);
    expect(summary.queued).toBe(0);
    expect(summary.dryRun).toBe(false);
    expect(summary.modelId).toBe('test-model');
    expect(embedText).toHaveBeenCalledTimes(2);
  });

  it('skips chunks that already have matching sha256 and model', async () => {
    const client = await createTestDb();

    // First run to set chunk_sha256
    const embedText = jest.fn(async () => {
      const v = new Float32Array(384);
      v.fill(0.1);
      return v;
    });
    await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    // Second run should skip all chunks
    embedText.mockClear();
    const summary = await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    expect(summary.skipped).toBe(2);
    expect(summary.embedded).toBe(0);
    expect(embedText).not.toHaveBeenCalled();
  });

  it('counts queued chunks in dry-run mode without writing', async () => {
    const client = await createTestDb();
    const embedText = jest.fn(async () => new Float32Array(384).fill(0.1));

    const summary = await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      dryRun: true,
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    expect(summary.queued).toBe(2);
    expect(summary.embedded).toBe(0);
    expect(summary.dryRun).toBe(true);
    // embedText should not be called in dry run
    expect(embedText).not.toHaveBeenCalled();
  });

  it('throws when dimension is invalid', async () => {
    await expect(
      buildEmbeddingIndex({
        client: createClient({ url: ':memory:' }),
        embedText: async () => new Float32Array(0),
        dimension: 0,
        modelSha256: 'sha',
        modelMeta: {},
      }),
    ).rejects.toThrow('Embedding dimension is required');
  });

  it('throws when modelSha256 is empty', async () => {
    await expect(
      buildEmbeddingIndex({
        client: createClient({ url: ':memory:' }),
        embedText: async () => new Float32Array(384),
        dimension: 384,
        modelSha256: '',
        modelMeta: {},
      }),
    ).rejects.toThrow('Model SHA-256 is required');
  });

  it('filters by files option', async () => {
    const client = await createTestDb();
    const embedText = jest.fn(async () => new Float32Array(384).fill(0.1));

    const summary = await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      files: ['src/test.ts'],
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    expect(summary.embedded).toBe(2);
    expect(embedText).toHaveBeenCalledTimes(2);
  });

  it('warns when files option matches no documents', async () => {
    const client = await createTestDb();
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const embedText = jest.fn(async () => new Float32Array(384).fill(0.1));

    const summary = await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      files: ['nonexistent.ts'],
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    expect(summary.skipped).toBe(2);
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('skips chunks not matching target files', async () => {
    const client = await createTestDb();
    // Add a chunk for a different file
    await client.execute({
      sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
           VALUES (2, 'docs/readme.md', 'md-doc', 2000, 300, 'def', 2000)`,
    });
    await client.execute({
      sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_start, char_end, depth)
           VALUES (3, 2, 0, 'doc content', 0, 11, 0)`,
    });

    const embedText = jest.fn(async () => new Float32Array(384).fill(0.1));

    const summary = await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      files: ['src/test.ts'],
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    expect(summary.embedded).toBe(2);
    expect(summary.skipped).toBe(1);
  });

  it('extracts slice metadata from plan-family chunks', async () => {
    const client = createClient({ url: ':memory:' });
    await client.execute(`
      CREATE TABLE documents (
        doc_id INTEGER PRIMARY KEY,
        file_path TEXT NOT NULL UNIQUE,
        doc_family TEXT NOT NULL,
        mtime_ms INTEGER NOT NULL,
        file_size INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        indexed_at INTEGER NOT NULL
      )
    `);
    await client.execute(`
      CREATE TABLE chunks (
        chunk_id INTEGER PRIMARY KEY,
        doc_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        heading_path TEXT,
        body_text TEXT NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        parent_chunk_id INTEGER,
        depth INTEGER DEFAULT 0,
        context_header TEXT,
        symbol_name TEXT,
        signature_text TEXT,
        jsdoc_text TEXT,
        export_type TEXT,
        module_path TEXT,
        slice_id TEXT,
        step_number INTEGER,
        phase TEXT,
        status TEXT,
        arch_layer TEXT,
        test_coverage TEXT,
        jsdoc_quality TEXT,
        jsdoc_word_count INTEGER,
        cyclomatic_complexity INTEGER,
        source_path_pattern TEXT,
        chunk_sha256 TEXT,
        embedding BLOB,
        embedding_model TEXT,
        embedded_at INTEGER
      )
    `);
    await client.execute({
      sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
           VALUES (1, 'plans/roadmap.md', 'plan', 1000, 200, 'abc', 1000)`,
    });
    const planBody = `Some plan text\n\`\`\`yaml\nphase: impl\nstep: 3\nstatus: green\nslices:\n  - slice_id: S1\n    description: do something\n\`\`\`\nmore text`;
    await client.execute({
      sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_start, char_end, depth)
           VALUES (1, 1, 0, ?, 0, ${planBody.length}, 0)`,
      args: [planBody],
    });

    const embedText = jest.fn(async () => new Float32Array(384).fill(0.1));

    await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    // Verify slice metadata was written
    const row = await client.execute({
      sql: 'SELECT slice_id, step_number, phase, status FROM chunks WHERE chunk_id = 1',
    });
    expect(row.rows[0].slice_id).toBe('S1');
    expect(row.rows[0].step_number).toBe(3);
    expect(row.rows[0].phase).toBe('impl');
    expect(row.rows[0].status).toBe('green');
  });

  it('handles plan-family chunk without YAML block', async () => {
    const client = createClient({ url: ':memory:' });
    await client.execute(`
      CREATE TABLE documents (
        doc_id INTEGER PRIMARY KEY,
        file_path TEXT NOT NULL UNIQUE,
        doc_family TEXT NOT NULL,
        mtime_ms INTEGER NOT NULL,
        file_size INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        indexed_at INTEGER NOT NULL
      )
    `);
    await client.execute(`
      CREATE TABLE chunks (
        chunk_id INTEGER PRIMARY KEY,
        doc_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        heading_path TEXT,
        body_text TEXT NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        parent_chunk_id INTEGER,
        depth INTEGER DEFAULT 0,
        context_header TEXT,
        symbol_name TEXT,
        signature_text TEXT,
        jsdoc_text TEXT,
        export_type TEXT,
        module_path TEXT,
        slice_id TEXT,
        step_number INTEGER,
        phase TEXT,
        status TEXT,
        arch_layer TEXT,
        test_coverage TEXT,
        jsdoc_quality TEXT,
        jsdoc_word_count INTEGER,
        cyclomatic_complexity INTEGER,
        source_path_pattern TEXT,
        chunk_sha256 TEXT,
        embedding BLOB,
        embedding_model TEXT,
        embedded_at INTEGER
      )
    `);
    await client.execute({
      sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
           VALUES (1, 'plans/note.md', 'plan', 1000, 200, 'abc', 1000)`,
    });
    await client.execute({
      sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_start, char_end, depth)
           VALUES (1, 1, 0, 'no yaml here', 0, 12, 0)`,
    });

    const embedText = jest.fn(async () => new Float32Array(384).fill(0.1));

    await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    // Slice metadata should be null
    const row = await client.execute({
      sql: 'SELECT slice_id, step_number, phase, status FROM chunks WHERE chunk_id = 1',
    });
    expect(row.rows[0].slice_id).toBeNull();
    expect(row.rows[0].step_number).toBeNull();
    expect(row.rows[0].phase).toBeNull();
    expect(row.rows[0].status).toBeNull();
  });

  it('uses embedText.release when available', async () => {
    const client = await createTestDb();
    const releaseFn = jest.fn(async () => {});
    const embedText = jest.fn(async () => new Float32Array(384).fill(0.1));
    embedText.release = releaseFn;

    await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    expect(releaseFn).toHaveBeenCalled();
  });

  it('updates indexed_at when files option is used (non-dry-run)', async () => {
    const client = await createTestDb();
    const embedText = jest.fn(async () => new Float32Array(384).fill(0.1));

    await buildEmbeddingIndex({
      client,
      embedText,
      modelId: 'test-model',
      modelSha256: 'test-sha',
      dimension: 384,
      files: ['src/test.ts'],
      modelMeta: { dimension: 384, model_sha256: 'test-sha' },
    });

    const doc = await client.execute(
      'SELECT indexed_at FROM documents WHERE doc_id = 1',
    );
    expect(Number(doc.rows[0].indexed_at)).toBeGreaterThan(1000);
  });
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
describe('embed-index — constants', () => {
  it('exports DEFAULT_MODEL_ID', () => {
    expect(DEFAULT_MODEL_ID).toBe('all-MiniLM-L6-v2');
  });

  it('exports DEFAULT_MODEL_DIRECTORY as a path containing "models"', () => {
    expect(DEFAULT_MODEL_DIRECTORY).toContain('models');
  });
});
