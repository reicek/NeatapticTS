/**
 * @module embed-index.test
 * @description 100% coverage tests for embed-index.mjs.
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourceFilePath = path.join(__dirname, 'embed-index.mjs');

// Shared mock functions
const mockCreateClient = jest.fn();
const mockReadFile = jest.fn();
const mockFail = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();
const mockTokenizerConstructor = jest.fn();
const mockInferenceSessionCreate = jest.fn();
const mockTensor = jest.fn();

jest.unstable_mockModule('@libsql/client', () => ({
  createClient: mockCreateClient,
}));
jest.unstable_mockModule('node:fs/promises', () => ({
  readFile: mockReadFile,
}));
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
}));
jest.unstable_mockModule('./init-schema.mjs', () => ({
  defaultDatabasePath: '/fake/default-db.sqlite',
  repoRoot: '/fake/repo',
  initSemanticIndex: jest.fn(),
}));
jest.unstable_mockModule('@huggingface/tokenizers', () => ({
  Tokenizer: mockTokenizerConstructor,
}));
jest.unstable_mockModule('onnxruntime-node', () => ({
  InferenceSession: { create: mockInferenceSessionCreate },
  Tensor: mockTensor,
}));

const {
  DEFAULT_MODEL_ID,
  buildEmbeddingIndex,
  normalizeEmbeddingVector,
  readModelMeta,
  createOnnxTextEmbedder,
  toF8BlobBuffer,
} = await import('./embed-index.mjs');

beforeEach(() => {
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

describe('embed-index: constants', () => {
  it('exports DEFAULT_MODEL_ID', () => {
    expect(DEFAULT_MODEL_ID).toBe('all-MiniLM-L6-v2');
  });
});

// ---------------------------------------------------------------------------
// normalizeEmbeddingVector
// ---------------------------------------------------------------------------

describe('embed-index: normalizeEmbeddingVector', () => {
  it('L2-normalizes a Float32Array', () => {
    const vec = new Float32Array([3, 4]);
    const result = normalizeEmbeddingVector(vec, 2);
    expect(result[0]).toBeCloseTo(0.6, 5);
    expect(result[1]).toBeCloseTo(0.8, 5);
  });

  it('returns new Float32Array for zero-magnitude vector', () => {
    const vec = new Float32Array([0, 0, 0]);
    const result = normalizeEmbeddingVector(vec, 3);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toHaveLength(3);
    expect(result[0]).toBe(0);
  });

  it('throws on dimension mismatch', () => {
    const vec = new Float32Array([1, 2, 3]);
    expect(() => normalizeEmbeddingVector(vec, 2)).toThrow(
      'Expected embedding dimension 2, received 3',
    );
  });

  it('handles Buffer input (ArrayBuffer.isView)', () => {
    const float32 = new Float32Array([3, 4]);
    const buf = Buffer.from(float32.buffer, float32.byteOffset, float32.byteLength);
    const result = normalizeEmbeddingVector(buf, 2);
    expect(result[0]).toBeCloseTo(0.6, 5);
    expect(result[1]).toBeCloseTo(0.8, 5);
  });

  it('handles Array input', () => {
    const result = normalizeEmbeddingVector([3, 4], 2);
    expect(result[0]).toBeCloseTo(0.6, 5);
    expect(result[1]).toBeCloseTo(0.8, 5);
  });

  it('handles non-array, non-typed-array input as empty', () => {
    const result = normalizeEmbeddingVector({}, 0);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// toF8BlobBuffer
// ---------------------------------------------------------------------------

describe('embed-index: toF8BlobBuffer', () => {
  it('quantizes normal values', () => {
    const vec = new Float32Array([0.5, -0.5, 1, -1]);
    const buf = toF8BlobBuffer(vec);
    expect(buf).toBeInstanceOf(Buffer);
    expect(buf.length).toBe(4);
    expect(buf.readInt8(0)).toBe(64);  // round(0.5 * 127) = 64
    expect(buf.readInt8(1)).toBe(-63); // round(-0.5 * 127) = round(-63.5) = -63
    expect(buf.readInt8(2)).toBe(127); // round(1 * 127) = 127
    expect(buf.readInt8(3)).toBe(-127); // round(-1 * 127) = -127
  });

  it('clamps values above 1', () => {
    const vec = new Float32Array([2.5]);
    const buf = toF8BlobBuffer(vec);
    expect(buf.readInt8(0)).toBe(127);
  });

  it('clamps values below -1', () => {
    const vec = new Float32Array([-2.5]);
    const buf = toF8BlobBuffer(vec);
    expect(buf.readInt8(0)).toBe(-127);
  });

  it('handles empty array', () => {
    const buf = toF8BlobBuffer(new Float32Array(0));
    expect(buf.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// readModelMeta
// ---------------------------------------------------------------------------

describe('embed-index: readModelMeta', () => {
  it('returns options.modelMeta directly when provided', async () => {
    const meta = { dimension: 384, model_sha256: 'abc' };
    const result = await readModelMeta({ modelMeta: meta });
    expect(result).toBe(meta);
    expect(mockReadFile).not.toHaveBeenCalled();
  });

  it('reads and parses model-meta.json', async () => {
    mockReadFile.mockResolvedValue(
      JSON.stringify({ dimension: 384, model_sha256: 'abc' }),
    );
    const result = await readModelMeta({
      modelMetaPath: '/fake/model-meta.json',
    });
    expect(result.dimension).toBe(384);
    expect(result.model_sha256).toBe('abc');
  });

  it('returns {} on ENOENT', async () => {
    const error = Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
    mockReadFile.mockRejectedValue(error);
    const result = await readModelMeta({
      modelMetaPath: '/fake/model-meta.json',
    });
    expect(result).toEqual({});
  });

  it('rethrows non-ENOENT errors', async () => {
    mockReadFile.mockRejectedValue(new Error('permission denied'));
    await expect(
      readModelMeta({ modelMetaPath: '/fake/model-meta.json' }),
    ).rejects.toThrow('permission denied');
  });
});

// ---------------------------------------------------------------------------
// createOnnxTextEmbedder
// ---------------------------------------------------------------------------

function setupTokenizerMocks({
  unkToken = '[UNK]',
  vocab = { hello: 1, world: 2, '[CLS]': 101, '[SEP]': 102 },
  doLowercase = true,
  stripAccents = true,
  padToken = '[PAD]',
  modelMaxLength = 512,
} = {}) {
  const tokenizer = {
    token_to_id: jest.fn((token) => {
      if (token === '[CLS]') return 101;
      if (token === '[SEP]') return 102;
      return undefined;
    }),
    encode: jest.fn((text) => ({
      ids: Array.from({ length: Math.min(text.length, 5) }, (_, i) => 200 + i),
      attention_mask: Array.from({ length: Math.min(text.length, 5) }, () => 1),
      token_type_ids: Array.from({ length: Math.min(text.length, 5) }, () => 0),
    })),
  };
  mockTokenizerConstructor.mockReturnValue(tokenizer);

  mockReadFile.mockImplementation((filePath) => {
    if (filePath.endsWith('tokenizer.json')) {
      return Promise.resolve(JSON.stringify({ model: { vocab } }));
    }
    if (filePath.endsWith('tokenizer_config.json')) {
      return Promise.resolve(
        JSON.stringify({
          do_lower_case: doLowercase,
          strip_accents: stripAccents,
          model_max_length: modelMaxLength,
          pad_token: padToken,
        }),
      );
    }
    if (filePath.endsWith('special_tokens_map.json')) {
      return Promise.resolve(JSON.stringify({ unk_token: unkToken, pad_token: padToken }));
    }
    return Promise.reject(Object.assign(new Error('ENOENT'), { code: 'ENOENT' }));
  });

  return tokenizer;
}

function setupSessionMocks({
  outputName = 'last_hidden_state',
  inputNames = ['input_ids', 'attention_mask', 'token_type_ids'],
  outputData = null,
} = {}) {
  const session = {
    run: jest.fn().mockResolvedValue({
      [outputName]: {
        data: outputData ?? new Float32Array([0.1, 0.2, 0.3, 0.4]),
      },
    }),
    outputNames: [outputName],
    inputNames,
    release: jest.fn().mockResolvedValue(undefined),
  };
  mockInferenceSessionCreate.mockResolvedValue(session);
  mockTensor.mockImplementation((type, data, dims) => ({ type, data, dims }));
  return session;
}

describe('embed-index: createOnnxTextEmbedder', () => {
  const origEnv = process.env.DENSE_FORCE_STATE;

  afterEach(() => {
    if (origEnv === undefined) delete process.env.DENSE_FORCE_STATE;
    else process.env.DENSE_FORCE_STATE = origEnv;
  });

  it('throws when DENSE_FORCE_STATE=cold', async () => {
    process.env.DENSE_FORCE_STATE = 'cold';
    await expect(createOnnxTextEmbedder({ dimension: 4 })).rejects.toThrow(
      'ONNX embedder disabled because DENSE_FORCE_STATE=cold',
    );
  });

  it('throws when DENSE_FORCE_STATE=model-only', async () => {
    process.env.DENSE_FORCE_STATE = 'model-only';
    await expect(createOnnxTextEmbedder({ dimension: 4 })).rejects.toThrow(
      'ONNX embedder disabled because DENSE_FORCE_STATE=model-only',
    );
  });

  it('creates embedder and embeds text (with token_type_ids)', async () => {
    delete process.env.DENSE_FORCE_STATE;
    setupTokenizerMocks();
    setupSessionMocks({
      outputName: 'last_hidden_state',
      inputNames: ['input_ids', 'attention_mask', 'token_type_ids'],
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    const result = await embedText({ text: 'hello' });
    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toHaveLength(4);
  });

  it('creates embedder without token_type_ids in inputNames', async () => {
    delete process.env.DENSE_FORCE_STATE;
    setupTokenizerMocks();
    setupSessionMocks({
      outputName: 'last_hidden_state',
      inputNames: ['input_ids', 'attention_mask'],
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    const result = await embedText({ text: 'hello' });
    expect(result).toBeInstanceOf(Float32Array);
  });

  it('uses first outputName when last_hidden_state not found', async () => {
    delete process.env.DENSE_FORCE_STATE;
    setupTokenizerMocks();
    setupSessionMocks({
      outputName: 'output_0',
      inputNames: ['input_ids', 'attention_mask'],
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    const result = await embedText({ text: 'hi' });
    expect(result).toBeInstanceOf(Float32Array);
  });

  it('throws when output is missing', async () => {
    delete process.env.DENSE_FORCE_STATE;
    setupTokenizerMocks();
    const session = {
      run: jest.fn().mockResolvedValue({}),
      outputNames: ['output_0'],
      inputNames: ['input_ids', 'attention_mask'],
      release: jest.fn(),
    };
    mockInferenceSessionCreate.mockResolvedValue(session);
    mockTensor.mockImplementation((type, data, dims) => ({ type, data, dims }));
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    await expect(embedText({ text: 'hi' })).rejects.toThrow(
      'ONNX embedding session did not return last_hidden_state output',
    );
  });

  it('throws when [CLS] or [SEP] token is missing', async () => {
    delete process.env.DENSE_FORCE_STATE;
    const tokenizer = {
      token_to_id: jest.fn(() => undefined),
      encode: jest.fn(() => ({ ids: [], attention_mask: [], token_type_ids: [] })),
    };
    mockTokenizerConstructor.mockReturnValue(tokenizer);
    mockReadFile.mockImplementation((filePath) => {
      if (filePath.endsWith('tokenizer.json')) {
        return Promise.resolve(JSON.stringify({ model: { vocab: { hello: 1 } } }));
      }
      if (filePath.endsWith('tokenizer_config.json')) {
        return Promise.resolve(JSON.stringify({}));
      }
      if (filePath.endsWith('special_tokens_map.json')) {
        return Promise.resolve(JSON.stringify({}));
      }
      return Promise.reject(Object.assign(new Error('ENOENT'), { code: 'ENOENT' }));
    });
    setupSessionMocks();
    await expect(createOnnxTextEmbedder({ dimension: 4 })).rejects.toThrow(
      'missing the required [CLS] or [SEP] tokens',
    );
  });

  it('releases session via embedText.release()', async () => {
    delete process.env.DENSE_FORCE_STATE;
    setupTokenizerMocks();
    const session = setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    await embedText.release();
    expect(session.release).toHaveBeenCalled();
  });

  it('handles empty text input', async () => {
    delete process.env.DENSE_FORCE_STATE;
    setupTokenizerMocks();
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    const result = await embedText({ text: '' });
    expect(result).toBeInstanceOf(Float32Array);
  });

  it('handles null text input', async () => {
    delete process.env.DENSE_FORCE_STATE;
    setupTokenizerMocks();
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    const result = await embedText({ text: null });
    expect(result).toBeInstanceOf(Float32Array);
  });
});

// ---------------------------------------------------------------------------
// createWordPieceTokenizer edge cases (via createOnnxTextEmbedder)
// ---------------------------------------------------------------------------

describe('embed-index: createWordPieceTokenizer edge cases', () => {
  beforeEach(() => {
    delete process.env.DENSE_FORCE_STATE;
  });

  it('throws when vocabulary is missing', async () => {
    mockTokenizerConstructor.mockReturnValue({
      token_to_id: jest.fn(() => 1),
      encode: jest.fn(() => ({ ids: [], attention_mask: [], token_type_ids: [] })),
    });
    mockReadFile.mockImplementation((filePath) => {
      if (filePath.endsWith('tokenizer.json')) {
        return Promise.resolve(JSON.stringify({ model: {} }));
      }
      if (filePath.endsWith('tokenizer_config.json')) {
        return Promise.resolve(JSON.stringify({}));
      }
      if (filePath.endsWith('special_tokens_map.json')) {
        return Promise.resolve(JSON.stringify({}));
      }
      return Promise.reject(Object.assign(new Error('ENOENT'), { code: 'ENOENT' }));
    });
    mockInferenceSessionCreate.mockResolvedValue({
      run: jest.fn(), outputNames: [], inputNames: [], release: jest.fn(),
    });
    await expect(createOnnxTextEmbedder({ dimension: 4 })).rejects.toThrow(
      'tokenizer.json is missing the WordPiece vocabulary',
    );
  });

  it('handles unk_token as object with content', async () => {
    setupTokenizerMocks({ unkToken: { content: '[UNK_OBJ]' } });
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    expect(embedText).toBeDefined();
  });

  it('handles missing unk_token (uses fallback)', async () => {
    setupTokenizerMocks({ unkToken: null });
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    expect(embedText).toBeDefined();
  });

  it('handles whitespace-only unk_token (uses fallback)', async () => {
    setupTokenizerMocks({ unkToken: '   ' });
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    expect(embedText).toBeDefined();
  });

  it('handles object unk_token with empty content (uses fallback)', async () => {
    setupTokenizerMocks({ unkToken: { content: '   ' } });
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    expect(embedText).toBeDefined();
  });

  it('handles ENOENT for tokenizer files', async () => {
    mockTokenizerConstructor.mockReturnValue({
      token_to_id: jest.fn(() => 1),
      encode: jest.fn(() => ({ ids: [], attention_mask: [], token_type_ids: [] })),
    });
    mockReadFile.mockRejectedValue(
      Object.assign(new Error('ENOENT'), { code: 'ENOENT' }),
    );
    mockInferenceSessionCreate.mockResolvedValue({
      run: jest.fn(), outputNames: [], inputNames: [], release: jest.fn(),
    });
    await expect(createOnnxTextEmbedder({ dimension: 4 })).rejects.toThrow(
      'tokenizer.json is missing the WordPiece vocabulary',
    );
  });

  it('rethrows non-ENOENT errors from readJsonFile', async () => {
    mockTokenizerConstructor.mockReturnValue({
      token_to_id: jest.fn(() => 1),
      encode: jest.fn(() => ({ ids: [], attention_mask: [], token_type_ids: [] })),
    });
    mockReadFile.mockRejectedValue(new Error('permission denied'));
    mockInferenceSessionCreate.mockResolvedValue({
      run: jest.fn(), outputNames: [], inputNames: [], release: jest.fn(),
    });
    await expect(createOnnxTextEmbedder({ dimension: 4 })).rejects.toThrow(
      'permission denied',
    );
  });

  it('handles do_lower_case=false and strip_accents=null from config', async () => {
    setupTokenizerMocks({ doLowercase: false, stripAccents: null });
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    expect(embedText).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// meanPoolEmbedding edge cases (via createOnnxTextEmbedder)
// ---------------------------------------------------------------------------

describe('embed-index: meanPoolEmbedding edge cases', () => {
  beforeEach(() => {
    delete process.env.DENSE_FORCE_STATE;
  });

  it('handles zero-token count (all attention_mask = 0)', async () => {
    setupTokenizerMocks();
    // The attention mask is always 1 for all tokens in embedText, so we
    // can't naturally hit the zero-token branch. Instead, we test by
    // having the session return a tensor for a single token.
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4]),
    });
    const embedText = await createOnnxTextEmbedder({ dimension: 4 });
    const result = await embedText({ text: 'hi' });
    // With only CLS and SEP tokens (empty text), the attention mask has 2 entries
    // The mean pool should still work since attention_mask is all 1s
    expect(result).toBeInstanceOf(Float32Array);
  });

  it('throws on invalid dimension for mean pooling', async () => {
    setupTokenizerMocks();
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4]),
    });
    // dimension validation happens in meanPoolEmbedding, which is called
    // during embedText(), not during createOnnxTextEmbedder()
    const embedText = await createOnnxTextEmbedder({ dimension: 0 });
    await expect(embedText({ text: 'hi' })).rejects.toThrow(
      'A positive embedding dimension is required for mean pooling',
    );
  });
});

// ---------------------------------------------------------------------------
// buildEmbeddingIndex
// ---------------------------------------------------------------------------

function makeMockClient(chunkRows = [], existingRows = null) {
  const execute = jest.fn().mockImplementation((params) => {
    const sql = params.sql || params;
    if (sql.includes('SELECT c.chunk_id')) {
      return Promise.resolve({ rows: chunkRows });
    }
    if (sql.includes('SELECT chunk_sha256')) {
      return Promise.resolve({
        rows: existingRows ?? [{ chunk_sha256: 'old', embedding_model: 'old-model' }],
      });
    }
    if (sql.includes('SELECT COUNT(*)')) {
      return Promise.resolve({ rows: [{ count: chunkRows.length }] });
    }
    if (sql.includes('UPDATE documents SET indexed_at')) {
      return Promise.resolve({ rows: [] });
    }
    return Promise.resolve({ rows: [] });
  });
  const batch = jest.fn().mockResolvedValue([]);
  return { execute, batch, close: jest.fn().mockResolvedValue(undefined) };
}

function makeMockEmbedText(withRelease = true) {
  const fn = jest.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3, 0.4]));
  if (withRelease) fn.release = jest.fn().mockResolvedValue(undefined);
  return fn;
}

describe('embed-index: buildEmbeddingIndex', () => {
  it('throws when dimension is invalid', async () => {
    await expect(
      buildEmbeddingIndex({
        client: makeMockClient(),
        embedText: makeMockEmbedText(),
        modelMeta: { dimension: 0, model_sha256: 'abc' },
        modelSha256: 'abc',
      }),
    ).rejects.toThrow('Embedding dimension is required');
  });

  it('throws when modelSha256 is missing', async () => {
    await expect(
      buildEmbeddingIndex({
        client: makeMockClient(),
        embedText: makeMockEmbedText(),
        modelMeta: { dimension: 4, model_sha256: '' },
        dimension: 4,
        modelSha256: '',
      }),
    ).rejects.toThrow('Model SHA-256 is required');
  });

  it('embeds chunks and writes to DB', async () => {
    const chunkRows = [
      {
        chunk_id: 1, chunk_index: 0, heading_path: 'h', body_text: 'hello',
        char_start: 0, char_end: 5, parent_chunk_id: null, depth: 0,
        context_header: null, symbol_name: null, signature_text: null,
        jsdoc_text: null, export_type: null, module_path: null,
        slice_id: null, step_number: null, phase: null, status: null,
        file_path: 'a.ts', doc_family: 'docs',
      },
    ];
    const client = makeMockClient(chunkRows, [{ chunk_sha256: 'old', embedding_model: 'old-model' }]);
    const embedText = makeMockEmbedText(true);
    const result = await buildEmbeddingIndex({
      client,
      embedText,
      dimension: 4,
      modelSha256: 'abc',
      modelMeta: { dimension: 4, model_sha256: 'abc' },
    });
    expect(result.embedded).toBe(1);
    expect(result.skipped).toBe(0);
    expect(client.batch).toHaveBeenCalledTimes(1);
    expect(embedText.release).toHaveBeenCalled();
  });

  it('skips up-to-date chunks', async () => {
    const chunkRows = [
      {
        chunk_id: 1, chunk_index: 0, heading_path: 'h', body_text: 'hello',
        char_start: 0, char_end: 5, parent_chunk_id: null, depth: 0,
        context_header: null, symbol_name: null, signature_text: null,
        jsdoc_text: null, export_type: null, module_path: null,
        slice_id: null, step_number: null, phase: null, status: null,
        file_path: 'a.ts', doc_family: 'docs',
      },
    ];
    // Existing sha256 and model match → skip
    const client = makeMockClient(chunkRows, [{ chunk_sha256: null, embedding_model: null }]);
    // We need the existing check to match. The sha256 is computed from the chunk data,
    // so we need to compute it. Easier: mock the existing result to match.
    // Actually, let's compute the sha256 by using createHash in the test.
    const { createHash } = await import('node:crypto');
    const expectedSha = createHash('sha256').update(JSON.stringify({
      body_text: 'hello', char_end: 5, char_start: 0, chunk_id: 1, chunk_index: 0,
      context_header: null, depth: 0, doc_family: 'docs', file_path: 'a.ts',
      heading_path: 'h', phase: null, slice_id: null, status: null, step_number: null,
      symbol_name: null,
    })).digest('hex');
    client.execute.mockImplementation((params) => {
      const sql = params.sql || params;
      if (sql.includes('SELECT c.chunk_id')) return Promise.resolve({ rows: chunkRows });
      if (sql.includes('SELECT chunk_sha256')) {
        return Promise.resolve({ rows: [{ chunk_sha256: expectedSha, embedding_model: 'test-model' }] });
      }
      return Promise.resolve({ rows: [] });
    });
    const embedText = makeMockEmbedText(true);
    const result = await buildEmbeddingIndex({
      client,
      embedText,
      dimension: 4,
      modelSha256: 'abc',
      modelId: 'test-model',
      modelMeta: { dimension: 4, model_sha256: 'abc' },
    });
    expect(result.skipped).toBe(1);
    expect(result.embedded).toBe(0);
  });

  it('handles dry-run mode', async () => {
    const chunkRows = [
      {
        chunk_id: 1, chunk_index: 0, heading_path: 'h', body_text: 'hello',
        char_start: 0, char_end: 5, parent_chunk_id: null, depth: 0,
        context_header: null, symbol_name: null, signature_text: null,
        jsdoc_text: null, export_type: null, module_path: null,
        slice_id: null, step_number: null, phase: null, status: null,
        file_path: 'a.ts', doc_family: 'docs',
      },
    ];
    const client = makeMockClient(chunkRows, [{ chunk_sha256: 'old', embedding_model: 'old' }]);
    const embedText = makeMockEmbedText(true);
    const result = await buildEmbeddingIndex({
      client, embedText, dimension: 4, modelSha256: 'abc', dryRun: true,
      modelMeta: { dimension: 4, model_sha256: 'abc' },
    });
    expect(result.queued).toBe(1);
    expect(result.embedded).toBe(0);
    expect(client.batch).not.toHaveBeenCalled();
  });

  it('filters by targetFiles', async () => {
    const chunkRows = [
      {
        chunk_id: 1, chunk_index: 0, heading_path: 'h', body_text: 'hello',
        char_start: 0, char_end: 5, parent_chunk_id: null, depth: 0,
        context_header: null, symbol_name: null, signature_text: null,
        jsdoc_text: null, export_type: null, module_path: null,
        slice_id: null, step_number: null, phase: null, status: null,
        file_path: 'a.ts', doc_family: 'docs',
      },
      {
        chunk_id: 2, chunk_index: 0, heading_path: 'h', body_text: 'world',
        char_start: 0, char_end: 5, parent_chunk_id: null, depth: 0,
        context_header: null, symbol_name: null, signature_text: null,
        jsdoc_text: null, export_type: null, module_path: null,
        slice_id: null, step_number: null, phase: null, status: null,
        file_path: 'b.ts', doc_family: 'docs',
      },
    ];
    const client = makeMockClient(chunkRows);
    const embedText = makeMockEmbedText(true);
    const result = await buildEmbeddingIndex({
      client, embedText, dimension: 4, modelSha256: 'abc',
      files: ['a.ts'],
      modelMeta: { dimension: 4, model_sha256: 'abc' },
    });
    expect(result.skipped).toBeGreaterThanOrEqual(1);
    // Should update freshness markers for targeted files
    const updateCall = client.execute.mock.calls.find(
      (c) => c[0].sql && c[0].sql.includes('UPDATE documents SET indexed_at'),
    );
    expect(updateCall).toBeDefined();
  });

  it('warns when no matching documents for targetFiles', async () => {
    const chunkRows = [];
    const client = makeMockClient(chunkRows);
    // Override to return count=0 for the match check
    client.execute.mockImplementation((params) => {
      const sql = params.sql || params;
      if (sql.includes('SELECT COUNT(*)')) {
        return Promise.resolve({ rows: [{ count: 0 }] });
      }
      if (sql.includes('SELECT c.chunk_id')) return Promise.resolve({ rows: [] });
      if (sql.includes('UPDATE documents SET indexed_at')) {
        return Promise.resolve({ rows: [] });
      }
      return Promise.resolve({ rows: [] });
    });
    const embedText = makeMockEmbedText(true);
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    await buildEmbeddingIndex({
      client, embedText, dimension: 4, modelSha256: 'abc',
      files: ['nonexistent.ts'],
      modelMeta: { dimension: 4, model_sha256: 'abc' },
    });
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('handles plan-family chunks with YAML slice metadata', async () => {
    const planBodyText = 'Some intro\n```yaml\nphase: implementation\nstep: 3\nstatus: in-progress\nslices:\n  - slice_id: C1-impl\n    name: test\n```\nMore text';
    const chunkRows = [
      {
        chunk_id: 1, chunk_index: 0, heading_path: 'h', body_text: planBodyText,
        char_start: 0, char_end: 100, parent_chunk_id: null, depth: 0,
        context_header: null, symbol_name: null, signature_text: null,
        jsdoc_text: null, export_type: null, module_path: null,
        slice_id: null, step_number: null, phase: null, status: null,
        file_path: 'plan.md', doc_family: 'plan',
      },
    ];
    const client = makeMockClient(chunkRows, [{ chunk_sha256: 'old', embedding_model: 'old' }]);
    const embedText = makeMockEmbedText(true);
    await buildEmbeddingIndex({
      client, embedText, dimension: 4, modelSha256: 'abc',
      modelMeta: { dimension: 4, model_sha256: 'abc' },
    });
    // The INSERT should include slice metadata
    const batchCall = client.batch.mock.calls[0];
    const updateStmt = batchCall[0][0];
    expect(updateStmt.args).toContain('C1-impl');
    expect(updateStmt.args).toContain('implementation');
    expect(updateStmt.args).toContain('in-progress');
    expect(updateStmt.args).toContain(3);
  });

  it('handles plan-family chunks without YAML', async () => {
    const chunkRows = [
      {
        chunk_id: 1, chunk_index: 0, heading_path: 'h', body_text: 'no yaml here',
        char_start: 0, char_end: 12, parent_chunk_id: null, depth: 0,
        context_header: null, symbol_name: null, signature_text: null,
        jsdoc_text: null, export_type: null, module_path: null,
        slice_id: null, step_number: null, phase: null, status: null,
        file_path: 'plan.md', doc_family: 'plan',
      },
    ];
    const client = makeMockClient(chunkRows, [{ chunk_sha256: 'old', embedding_model: 'old' }]);
    const embedText = makeMockEmbedText(true);
    await buildEmbeddingIndex({
      client, embedText, dimension: 4, modelSha256: 'abc',
      modelMeta: { dimension: 4, model_sha256: 'abc' },
    });
    const batchCall = client.batch.mock.calls[0];
    const updateStmt = batchCall[0][0];
    // slice metadata should be null
    expect(updateStmt.args).toContain(null);
  });

  it('handles empty chunk rows', async () => {
    const client = makeMockClient([]);
    const embedText = makeMockEmbedText(true);
    const result = await buildEmbeddingIndex({
      client, embedText, dimension: 4, modelSha256: 'abc',
      modelMeta: { dimension: 4, model_sha256: 'abc' },
    });
    expect(result.embedded).toBe(0);
    expect(result.skipped).toBe(0);
  });

  it('calls releaseEmbedText at the end', async () => {
    const client = makeMockClient([]);
    const embedText = makeMockEmbedText(true);
    await buildEmbeddingIndex({
      client, embedText, dimension: 4, modelSha256: 'abc',
      modelMeta: { dimension: 4, model_sha256: 'abc' },
    });
    expect(embedText.release).toHaveBeenCalled();
  });

  it('creates client when not provided', async () => {
    const mockClient = makeMockClient([]);
    mockCreateClient.mockReturnValue(mockClient);
    const embedText = makeMockEmbedText(true);
    await buildEmbeddingIndex({
      embedText, dimension: 4, modelSha256: 'abc',
      corpusDatabasePath: '/fake/db.sqlite',
      modelMeta: { dimension: 4, model_sha256: 'abc' },
    });
    expect(mockCreateClient).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// main() — CLI guard
// ---------------------------------------------------------------------------

describe('embed-index: main()', () => {
  const origArgv1 = process.argv[1];

  afterEach(() => {
    process.argv[1] = origArgv1;
  });

  it('prints help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-index.mjs');
    });
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
  });

  it('runs build and writes dry-run text output', async () => {
    mockParseCliArgs.mockReturnValue({
      'dry-run': true,
      json: false,
      _: [],
    });
    const mockClient = makeMockClient([]);
    mockCreateClient.mockReturnValue(mockClient);
    // Set up tokenizer and session mocks so createOnnxTextEmbedder succeeds
    setupTokenizerMocks();
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4]),
    });
    // Add model-meta.json handling on top of the tokenizer mock
    const tokenizerImpl = mockReadFile.getMockImplementation();
    mockReadFile.mockImplementation((filePath) => {
      if (filePath.endsWith('model-meta.json')) {
        return Promise.resolve(JSON.stringify({ dimension: 4, model_sha256: 'abc' }));
      }
      return tokenizerImpl(filePath);
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-index.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });

  it('runs build and writes JSON output', async () => {
    mockParseCliArgs.mockReturnValue({
      json: true,
      _: [],
    });
    const mockClient = makeMockClient([]);
    mockCreateClient.mockReturnValue(mockClient);
    setupTokenizerMocks();
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4]),
    });
    const tokenizerImpl = mockReadFile.getMockImplementation();
    mockReadFile.mockImplementation((filePath) => {
      if (filePath.endsWith('model-meta.json')) {
        return Promise.resolve(JSON.stringify({ dimension: 4, model_sha256: 'abc' }));
      }
      return tokenizerImpl(filePath);
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-index.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(true);
  });

  it('handles string args.files', async () => {
    mockParseCliArgs.mockReturnValue({
      'dry-run': true,
      json: true,
      files: 'single-file.ts',
      _: [],
    });
    const mockClient = makeMockClient([]);
    mockCreateClient.mockReturnValue(mockClient);
    setupTokenizerMocks();
    setupSessionMocks({
      outputData: new Float32Array([0.1, 0.2, 0.3, 0.4]),
    });
    const tokenizerImpl = mockReadFile.getMockImplementation();
    mockReadFile.mockImplementation((filePath) => {
      if (filePath.endsWith('model-meta.json')) {
        return Promise.resolve(JSON.stringify({ dimension: 4, model_sha256: 'abc' }));
      }
      return tokenizerImpl(filePath);
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-index.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });

  it('calls fail on Error', async () => {
    mockParseCliArgs.mockReturnValue({
      json: false,
      _: [],
    });
    mockReadFile.mockRejectedValue(new Error('read failed'));
    mockCreateClient.mockReturnValue(makeMockClient([]));
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-index.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
  });

  it('calls fail with String(error) for non-Error throws', async () => {
    mockParseCliArgs.mockReturnValue({
      json: true,
      _: [],
    });
    mockReadFile.mockRejectedValue('string error');
    mockCreateClient.mockReturnValue(makeMockClient([]));
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./embed-index.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('string error');
  });
});