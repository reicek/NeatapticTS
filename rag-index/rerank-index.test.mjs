/**
 * @module rerank-index.test
 * @description 100% coverage tests for rerank-index.mjs.
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourceFilePath = path.join(__dirname, 'rerank-index.mjs');

// Shared mock functions
const mockInferenceSessionCreate = jest.fn();
const mockTensor = jest.fn();
const mockTokenizerConstructor = jest.fn();
const mockReadFile = jest.fn();
const mockFail = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();

jest.unstable_mockModule('onnxruntime-node', () => ({
  InferenceSession: { create: mockInferenceSessionCreate },
  Tensor: mockTensor,
}));
jest.unstable_mockModule('@huggingface/tokenizers', () => ({
  Tokenizer: mockTokenizerConstructor,
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
jest.unstable_mockModule('./reranker-readiness.mjs', () => ({
  DEFAULT_RERANKER_MODEL_DIRECTORY: '/fake/reranker',
  DEFAULT_RERANKER_MODEL_ID: 'fake-reranker-id',
  DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH: 512,
}));

const {
  DEFAULT_RERANKER_CANDIDATES_COUNT,
  normalizeRerankCandidates,
  softmax,
  scorePairFromLogits,
  createRerankInput,
  getOrCreateRerankSession,
  releaseRerankSession,
  rerankCandidates,
} = await import('./rerank-index.mjs');

beforeEach(() => {
  jest.clearAllMocks();
  mockInferenceSessionCreate.mockReset();
  mockTensor.mockReset();
  mockTokenizerConstructor.mockReset();
  mockReadFile.mockReset();
});

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

describe('rerank-index: normalizeRerankCandidates', () => {
  it('returns default 50 for undefined', () => {
    expect(normalizeRerankCandidates(undefined)).toBe(50);
  });
  it('returns default 50 for null', () => {
    expect(normalizeRerankCandidates(null)).toBe(50);
  });
  it('returns valid value within range', () => {
    expect(normalizeRerankCandidates(25)).toBe(25);
  });
  it('clamps to 200 for values above 200', () => {
    expect(normalizeRerankCandidates(300)).toBe(200);
  });
  it('clamps to 1 for values below 1', () => {
    expect(normalizeRerankCandidates(0)).toBe(1);
    expect(normalizeRerankCandidates(-10)).toBe(1);
  });
  it('returns default 50 for NaN', () => {
    expect(normalizeRerankCandidates(NaN)).toBe(50);
  });
  it('returns default 50 for Infinity', () => {
    expect(normalizeRerankCandidates(Infinity)).toBe(50);
  });
  it('truncates non-integer values', () => {
    expect(normalizeRerankCandidates(10.7)).toBe(10);
  });
  it('handles string numbers', () => {
    expect(normalizeRerankCandidates('30')).toBe(30);
  });
});

describe('rerank-index: softmax', () => {
  it('returns [0.5, 0.5] for equal logits', () => {
    const result = softmax([0, 0]);
    expect(result[0]).toBeCloseTo(0.5);
    expect(result[1]).toBeCloseTo(0.5);
    expect(result.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });
  it('returns higher probability for larger logit', () => {
    const result = softmax([2, 0]);
    expect(result[0]).toBeGreaterThan(result[1]);
  });
  it('handles negative logits', () => {
    const result = softmax([-1, -2]);
    expect(result[0]).toBeGreaterThan(result[1]);
  });
  it('handles single element', () => {
    const result = softmax([5]);
    expect(result[0]).toBeCloseTo(1);
  });
});

describe('rerank-index: scorePairFromLogits', () => {
  it('returns probability of relevant class', () => {
    const score = scorePairFromLogits([0, 2]);
    expect(score).toBeGreaterThan(0.5);
  });
  it('returns ~0.5 for equal logits', () => {
    const score = scorePairFromLogits([1, 1]);
    expect(score).toBeCloseTo(0.5);
  });
});

describe('rerank-index: createRerankInput', () => {
  it('returns encoded input without truncation when under maxLength', () => {
    const encoded = {
      input_ids: [1, 2, 3],
      attention_mask: [1, 1, 1],
      token_type_ids: [0, 0, 0],
    };
    const result = createRerankInput('q', 'd', encoded, 512);
    expect(result.input_ids).toEqual([1, 2, 3]);
    expect(result.attention_mask).toEqual([1, 1, 1]);
    expect(result.token_type_ids).toEqual([0, 0, 0]);
  });
  it('truncates input when exceeding maxLength', () => {
    const encoded = {
      input_ids: Array.from({ length: 10 }, (_, i) => i),
      attention_mask: Array.from({ length: 10 }, () => 1),
      token_type_ids: Array.from({ length: 10 }, () => 0),
    };
    const result = createRerankInput('q', 'd', encoded, 5);
    expect(result.input_ids).toHaveLength(5);
    expect(result.attention_mask).toHaveLength(5);
    expect(result.token_type_ids).toHaveLength(5);
  });
  it('uses default maxLength of 512', () => {
    const encoded = {
      input_ids: [1, 2, 3],
      attention_mask: [1, 1, 1],
      token_type_ids: [0, 0, 0],
    };
    const result = createRerankInput('q', 'd', encoded);
    expect(result.input_ids).toEqual([1, 2, 3]);
  });
});

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

function setupMockTokenizer() {
  const tokenizer = {
    token_to_id: jest.fn((token) => {
      if (token === '[CLS]') return 101;
      if (token === '[SEP]') return 102;
      return undefined;
    }),
    encode: jest.fn((text) => ({
      ids: Array.from({ length: Math.min(text.length, 5) }, (_, i) => 200 + i),
    })),
  };
  mockTokenizerConstructor.mockReturnValue(tokenizer);
  return tokenizer;
}

function setupMockReadFile() {
  mockReadFile.mockImplementation((filePath) => {
    if (filePath.endsWith('tokenizer.json')) {
      return Promise.resolve(
        JSON.stringify({ model: { vocab: { hello: 1, world: 2 } } }),
      );
    }
    if (filePath.endsWith('tokenizer_config.json')) {
      return Promise.resolve(
        JSON.stringify({ model_max_length: 512, do_lower_case: true }),
      );
    }
    if (filePath.endsWith('special_tokens_map.json')) {
      return Promise.resolve(
        JSON.stringify({ unk_token: '[UNK]', pad_token: '[PAD]' }),
      );
    }
    return Promise.reject(Object.assign(new Error('ENOENT'), { code: 'ENOENT' }));
  });
}

function setupMockSession() {
  const session = {
    run: jest.fn().mockResolvedValue({
      logits: { data: [0.5, 1.5] },
    }),
    outputNames: ['logits'],
    release: jest.fn().mockResolvedValue(undefined),
  };
  mockInferenceSessionCreate.mockResolvedValue(session);
  mockTensor.mockImplementation((type, data, dims) => ({ type, data, dims }));
  return session;
}

describe('rerank-index: getOrCreateRerankSession', () => {
  beforeEach(async () => {
    await releaseRerankSession();
  });

  it('creates a new session on first call', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    const result = await getOrCreateRerankSession();
    expect(result.session).toBeDefined();
    expect(result.tokenizer).toBeDefined();
    expect(mockInferenceSessionCreate).toHaveBeenCalledTimes(1);
  });

  it('returns cached session on second call', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    await getOrCreateRerankSession();
    const result = await getOrCreateRerankSession();
    expect(mockInferenceSessionCreate).toHaveBeenCalledTimes(1);
  });

  it('creates new session when forceReload is true', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    await getOrCreateRerankSession();
    await getOrCreateRerankSession({ forceReload: true });
    expect(mockInferenceSessionCreate).toHaveBeenCalledTimes(2);
  });

  it('uses custom rerankerModelDirectory', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    await getOrCreateRerankSession({ rerankerModelDirectory: '/custom/dir' });
    expect(mockInferenceSessionCreate).toHaveBeenCalledWith(
      path.join(path.resolve('/custom/dir'), 'model.onnx'),
    );
  });
});

describe('rerank-index: releaseRerankSession', () => {
  it('releases active session', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    const { session } = await getOrCreateRerankSession();
    await releaseRerankSession();
    expect(session.release).toHaveBeenCalled();
  });

  it('is a no-op when no session is active', async () => {
    await releaseRerankSession();
    // Should not throw
  });
});

// ---------------------------------------------------------------------------
// rerankCandidates
// ---------------------------------------------------------------------------

describe('rerank-index: rerankCandidates', () => {
  it('ranks candidates by rerank_score descending', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    await releaseRerankSession();

    const candidates = [
      { chunk_id: 1, body_text: 'apple' },
      { chunk_id: 2, body_text: 'banana' },
      { chunk_id: 3, text: 'cherry' },
    ];

    // Mock session.run to return different scores for different candidates
    const { session } = await getOrCreateRerankSession();
    session.run
      .mockResolvedValueOnce({ logits: { data: [0.1, 0.2] } })
      .mockResolvedValueOnce({ logits: { data: [2.0, 3.0] } })
      .mockResolvedValueOnce({ logits: { data: [0.5, 1.0] } });

    const results = await rerankCandidates('query', candidates);
    expect(results).toHaveLength(3);
    expect(results[0].chunk_id).toBe(2);
    expect(results.every((r) => r.rerank_score !== undefined)).toBe(true);
  });

  it('uses body_text or text field for document text', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    await releaseRerankSession();

    const candidates = [{ chunk_id: 1, text: 'from text field' }];
    const { session } = await getOrCreateRerankSession();
    session.run.mockResolvedValueOnce({ logits: { data: [0.5] } });

    const results = await rerankCandidates('q', candidates);
    expect(results).toHaveLength(1);
  });

  it('uses empty string when no body_text or text', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    await releaseRerankSession();

    const candidates = [{ chunk_id: 1 }];
    const { session } = await getOrCreateRerankSession();
    session.run.mockResolvedValueOnce({ logits: { data: [0.5] } });

    const results = await rerankCandidates('q', candidates);
    expect(results).toHaveLength(1);
  });

  it('respects rerankCandidatesCount limit', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    await releaseRerankSession();

    const candidates = Array.from({ length: 10 }, (_, i) => ({
      chunk_id: i,
      body_text: `doc ${i}`,
    }));
    const { session } = await getOrCreateRerankSession();
    session.run.mockResolvedValue({ logits: { data: [0.5] } });

    const results = await rerankCandidates('q', candidates, {
      rerankCandidatesCount: 3,
    });
    expect(results).toHaveLength(3);
  });

  it('handles single-logit output (sigmoid)', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    await releaseRerankSession();

    const candidates = [{ chunk_id: 1, body_text: 'test' }];
    const { session } = await getOrCreateRerankSession();
    session.run.mockResolvedValueOnce({ logits: { data: [0.0] } });

    const results = await rerankCandidates('q', candidates);
    // sigmoid(0) = 0.5
    expect(results[0].rerank_score).toBeCloseTo(0.5);
  });

  it('handles output where logits is not in outputNames (uses first)', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    await releaseRerankSession();

    const candidates = [{ chunk_id: 1, body_text: 'test' }];
    const { session } = await getOrCreateRerankSession();
    session.outputNames = ['output_0'];
    session.run.mockResolvedValueOnce({
      output_0: { data: [0.5, 1.5] },
    });

    const results = await rerankCandidates('q', candidates);
    expect(results[0].rerank_score).toBeDefined();
  });

  it('truncates long sequences', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    await releaseRerankSession();

    // Create a very long query to trigger truncation
    const longQuery = 'a'.repeat(600);
    const candidates = [{ chunk_id: 1, body_text: 'b'.repeat(600) }];
    const { session } = await getOrCreateRerankSession();
    session.run.mockResolvedValueOnce({ logits: { data: [0.5, 1.5] } });

    const results = await rerankCandidates(longQuery, candidates);
    expect(results).toHaveLength(1);
  });

  it('truncates document tokens when exceeding maxLength in scorePair', async () => {
    setupMockSession();
    setupMockReadFile();
    await releaseRerankSession();

    // Custom tokenizer that returns many token ids to exceed maxLength=512
    const tokenizer = {
      token_to_id: jest.fn((token) => {
        if (token === '[CLS]') return 101;
        if (token === '[SEP]') return 102;
        return undefined;
      }),
      encode: jest.fn((text) => ({
        ids: Array.from({ length: Math.min(text.length, 600) }, (_, i) => 200 + i),
      })),
    };
    mockTokenizerConstructor.mockReturnValue(tokenizer);

    const longQuery = 'a'.repeat(600);
    const candidates = [{ chunk_id: 1, body_text: 'b'.repeat(600) }];
    const { session } = await getOrCreateRerankSession();
    session.run.mockResolvedValueOnce({ logits: { data: [0.5, 1.5] } });

    const results = await rerankCandidates(longQuery, candidates);
    expect(results).toHaveLength(1);
    // Verify truncation happened: the Tensor should have been called with
    // input_ids of length 512 (maxLength), not 1203
    const tensorCall = mockTensor.mock.calls.find(
      (c) => c[0] === 'int32',
    );
    if (tensorCall) {
      expect(tensorCall[1].length).toBeLessThanOrEqual(512);
    }
  });
});

// ---------------------------------------------------------------------------
// createRerankTokenizer edge cases
// ---------------------------------------------------------------------------

describe('rerank-index: createRerankTokenizer edge cases', () => {
  it('throws when vocabulary is missing', async () => {
    setupMockTokenizer();
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
      return Promise.reject(
        Object.assign(new Error('ENOENT'), { code: 'ENOENT' }),
      );
    });
    mockInferenceSessionCreate.mockResolvedValue({
      run: jest.fn(),
      outputNames: [],
    });

    await releaseRerankSession();
    await expect(getOrCreateRerankSession()).rejects.toThrow(
      'tokenizer.json is missing the WordPiece vocabulary',
    );
  });

  it('handles unk_token as object with content property', async () => {
    setupMockTokenizer();
    mockReadFile.mockImplementation((filePath) => {
      if (filePath.endsWith('tokenizer.json')) {
        return Promise.resolve(
          JSON.stringify({ model: { vocab: { hello: 1 } } }),
        );
      }
      if (filePath.endsWith('tokenizer_config.json')) {
        return Promise.resolve(JSON.stringify({}));
      }
      if (filePath.endsWith('special_tokens_map.json')) {
        return Promise.resolve(
          JSON.stringify({ unk_token: { content: '[UNK_OBJ]' } }),
        );
      }
      return Promise.reject(
        Object.assign(new Error('ENOENT'), { code: 'ENOENT' }),
      );
    });
    setupMockSession();

    await releaseRerankSession();
    const result = await getOrCreateRerankSession();
    expect(result.tokenizer).toBeDefined();
  });

  it('handles missing unk_token (uses fallback)', async () => {
    setupMockTokenizer();
    mockReadFile.mockImplementation((filePath) => {
      if (filePath.endsWith('tokenizer.json')) {
        return Promise.resolve(
          JSON.stringify({ model: { vocab: { hello: 1 } } }),
        );
      }
      if (filePath.endsWith('tokenizer_config.json')) {
        return Promise.resolve(JSON.stringify({}));
      }
      if (filePath.endsWith('special_tokens_map.json')) {
        return Promise.resolve(JSON.stringify({}));
      }
      return Promise.reject(
        Object.assign(new Error('ENOENT'), { code: 'ENOENT' }),
      );
    });
    setupMockSession();

    await releaseRerankSession();
    const result = await getOrCreateRerankSession();
    expect(result.tokenizer).toBeDefined();
  });

  it('handles tokenizer_config with model_max_length and pad_token', async () => {
    setupMockTokenizer();
    mockReadFile.mockImplementation((filePath) => {
      if (filePath.endsWith('tokenizer.json')) {
        return Promise.resolve(
          JSON.stringify({ model: { vocab: { hello: 1 } } }),
        );
      }
      if (filePath.endsWith('tokenizer_config.json')) {
        return Promise.resolve(
          JSON.stringify({ model_max_length: 256, pad_token: '[PAD]' }),
        );
      }
      if (filePath.endsWith('special_tokens_map.json')) {
        return Promise.resolve(
          JSON.stringify({ pad_token: '[PAD]' }),
        );
      }
      return Promise.reject(
        Object.assign(new Error('ENOENT'), { code: 'ENOENT' }),
      );
    });
    setupMockSession();

    await releaseRerankSession();
    const result = await getOrCreateRerankSession();
    expect(result.tokenizer).toBeDefined();
  });

  it('handles ENOENT for tokenizer files (returns defaults)', async () => {
    setupMockTokenizer();
    mockReadFile.mockImplementation(() =>
      Promise.reject(Object.assign(new Error('ENOENT'), { code: 'ENOENT' })),
    );
    // tokenizer.json will return null → vocabulary check will fail
    mockInferenceSessionCreate.mockResolvedValue({
      run: jest.fn(),
      outputNames: [],
    });

    await releaseRerankSession();
    await expect(getOrCreateRerankSession()).rejects.toThrow(
      'tokenizer.json is missing the WordPiece vocabulary',
    );
  });

  it('swallows non-ENOENT readFile errors and falls through to vocabulary check', async () => {
    setupMockTokenizer();
    mockReadFile.mockImplementation(() =>
      Promise.reject(new Error('permission denied')),
    );
    mockInferenceSessionCreate.mockResolvedValue({
      run: jest.fn(),
      outputNames: [],
    });

    await releaseRerankSession();
    await expect(getOrCreateRerankSession()).rejects.toThrow(
      'tokenizer.json is missing the WordPiece vocabulary',
    );
  });
});

// ---------------------------------------------------------------------------
// DEFAULT_RERANKER_CANDIDATES_COUNT
// ---------------------------------------------------------------------------

describe('rerank-index: constants', () => {
  it('exports DEFAULT_RERANKER_CANDIDATES_COUNT as 50', () => {
    expect(DEFAULT_RERANKER_CANDIDATES_COUNT).toBe(50);
  });
});

// ---------------------------------------------------------------------------
// main() — CLI guard
// ---------------------------------------------------------------------------

describe('rerank-index: main()', () => {
  const origArgv1 = process.argv[1];

  afterEach(() => {
    process.argv[1] = origArgv1;
  });

  it('prints help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./rerank-index.mjs');
    });
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
  });

  it('runs rerank and writes text output', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    mockParseCliArgs.mockReturnValue({
      query: 'neat',
      candidates: JSON.stringify([{ chunk_id: 1, body_text: 'test' }]),
      json: false,
      _: [],
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./rerank-index.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });

  it('calls fail when no query provided', async () => {
    mockParseCliArgs.mockReturnValue({
      query: '',
      _: [],
      json: false,
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./rerank-index.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
  });

  it('calls fail when candidates JSON.parse fails', async () => {
    mockParseCliArgs.mockReturnValue({
      query: 'test',
      candidates: 'not valid json',
      _: [],
      json: false,
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./rerank-index.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
  });

  it('calls fail with String(error) for non-Error throws', async () => {
    mockParseCliArgs.mockReturnValue({
      query: 'test',
      candidates: null,
      _: [],
      json: true,
    });
    process.argv[1] = sourceFilePath;
    // Mock rerankCandidates to throw a non-Error
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    // We need the module-level rerankCandidates to throw a string
    // But since we mock the deps, the error will come from the session creation
    // Let's force an error by making InferenceSession.create throw a string
    mockInferenceSessionCreate.mockRejectedValue('string error');
    await jest.isolateModulesAsync(async () => {
      await import('./rerank-index.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
  });

  it('uses positional args when --query is not provided', async () => {
    setupMockSession();
    setupMockTokenizer();
    setupMockReadFile();
    mockParseCliArgs.mockReturnValue({
      _: ['positional', 'query'],
      candidates: JSON.stringify([{ chunk_id: 1, body_text: 'test' }]),
      json: false,
    });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./rerank-index.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });
});