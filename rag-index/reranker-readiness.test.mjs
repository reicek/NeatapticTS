import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourcePath = path.resolve(__dirname, 'reranker-readiness.mjs');

const mockExistsSync = jest.fn();
const mockReadFile = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();
const mockFail = jest.fn();
const mockSessionCreate = jest.fn();
const mockSessionRelease = jest.fn();

jest.unstable_mockModule('node:fs', () => ({ existsSync: mockExistsSync, default: { existsSync: mockExistsSync } }));
jest.unstable_mockModule('node:fs/promises', () => ({ readFile: mockReadFile, default: { readFile: mockReadFile } }));
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
}));
jest.unstable_mockModule('./init-schema.mjs', () => ({
  repoRoot: '/fake/repo',
}));
jest.unstable_mockModule('onnxruntime-node', () => ({
  InferenceSession: { create: mockSessionCreate },
}));

const { checkRerankerReadiness, DEFAULT_RERANKER_MODEL_ID, DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH, DEFAULT_RERANKER_MODEL_DIRECTORY } = await import('./reranker-readiness.mjs');

afterEach(() => {
  jest.clearAllMocks();
  delete process.env.RERANKER_FORCE_STATE;
});

describe('constants', () => {
  it('has default model id', () => {
    expect(DEFAULT_RERANKER_MODEL_ID).toBe('cross-encoder/ms-marco-MiniLM-L-6-v2');
  });

  it('has default max sequence length', () => {
    expect(DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH).toBe(512);
  });

  it('has default model directory', () => {
    expect(DEFAULT_RERANKER_MODEL_DIRECTORY).toContain('reranker');
  });
});

describe('checkRerankerReadiness', () => {
  it('returns forced cold state from env', async () => {
    process.env.RERANKER_FORCE_STATE = 'cold';
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('cold');
    expect(result.ready).toBe(false);
    expect(result.reason).toContain('forced cold');
  });

  it('returns forced model-only state from env', async () => {
    process.env.RERANKER_FORCE_STATE = 'model-only';
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('model-only');
    expect(result.ready).toBe(false);
    expect(result.reason).toContain('forced model-only');
  });

  it('ignores invalid forced state', async () => {
    process.env.RERANKER_FORCE_STATE = 'invalid';
    mockExistsSync.mockReturnValue(false);
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('cold');
  });

  it('ignores whitespace-only forced state', async () => {
    process.env.RERANKER_FORCE_STATE = '  ';
    mockExistsSync.mockReturnValue(false);
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('cold');
  });

  it('uses forceState option over env', async () => {
    process.env.RERANKER_FORCE_STATE = 'cold';
    const result = await checkRerankerReadiness({ forceState: 'model-only' });
    expect(result.state).toBe('model-only');
  });

  it('returns cold when model file does not exist', async () => {
    mockExistsSync.mockReturnValue(false);
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('cold');
    expect(result.ready).toBe(false);
    expect(result.model_id).toBeNull();
    expect(result.max_sequence_length).toBeNull();
  });

  it('returns model-only when meta file is missing', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockRejectedValue(new Error('ENOENT'));
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('model-only');
    expect(result.reason).toContain('model-meta.json is missing or invalid');
  });

  it('returns model-only when meta file is invalid JSON', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockResolvedValue('invalid json{');
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('model-only');
    expect(result.reason).toContain('model-meta.json is missing or invalid');
  });

  it('returns model-only when meta is not an object', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockResolvedValue('"not an object"');
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('model-only');
    expect(result.reason).toContain('not a valid object');
  });

  it('returns model-only when meta is null', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockResolvedValue('null');
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('model-only');
    expect(result.reason).toContain('not a valid object');
  });

  it('returns warm when session creation succeeds', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockResolvedValue(JSON.stringify({
      model_id: 'custom-reranker',
      max_sequence_length: 256,
    }));
    mockSessionCreate.mockResolvedValue({ release: mockSessionRelease });
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('warm');
    expect(result.ready).toBe(true);
    expect(result.model_id).toBe('custom-reranker');
    expect(result.max_sequence_length).toBe(256);
    expect(mockSessionRelease).toHaveBeenCalled();
  });

  it('returns warm when session has no release method', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockResolvedValue(JSON.stringify({ model_id: 'test' }));
    mockSessionCreate.mockResolvedValue({});
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('warm');
    expect(result.ready).toBe(true);
  });

  it('uses default max_sequence_length when not in meta', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockResolvedValue(JSON.stringify({}));
    mockSessionCreate.mockResolvedValue({ release: jest.fn() });
    const result = await checkRerankerReadiness();
    expect(result.max_sequence_length).toBe(512);
  });

  it('uses default model_id when not in meta', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockResolvedValue(JSON.stringify({}));
    mockSessionCreate.mockResolvedValue({ release: jest.fn() });
    const result = await checkRerankerReadiness();
    expect(result.model_id).toBe(DEFAULT_RERANKER_MODEL_ID);
  });

  it('uses custom rerankerModelId as fallback when meta has no model_id', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockResolvedValue(JSON.stringify({}));
    mockSessionCreate.mockResolvedValue({ release: jest.fn() });
    const result = await checkRerankerReadiness({ rerankerModelId: 'custom-id' });
    expect(result.model_id).toBe('custom-id');
  });

  it('returns model-only when session creation fails with Error', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockResolvedValue(JSON.stringify({ model_id: 'test' }));
    mockSessionCreate.mockRejectedValue(new Error('ONNX load failed'));
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('model-only');
    expect(result.ready).toBe(false);
    expect(result.reason).toContain('ONNX load failed');
  });

  it('returns model-only when session creation fails with non-Error', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFile.mockResolvedValue(JSON.stringify({ model_id: 'test' }));
    mockSessionCreate.mockRejectedValue('string error');
    const result = await checkRerankerReadiness();
    expect(result.state).toBe('model-only');
    expect(result.reason).toContain('string error');
  });
});

describe('reranker-readiness main()', () => {
  it('shows help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    const savedArgv = [...process.argv];
    process.argv = [savedArgv[0], sourcePath, '--help'];
    try {
      jest.resetModules();
      await import('./reranker-readiness.mjs');
    } finally {
      process.argv = savedArgv;
    }
    expect(mockPrintHelp).toHaveBeenCalled();
  });

  it('writes report on success', async () => {
    mockParseCliArgs.mockReturnValue({ json: true });
    process.env.RERANKER_FORCE_STATE = 'cold';
    const savedArgv = [...process.argv];
    process.argv = [savedArgv[0], sourcePath, '--json'];
    try {
      jest.resetModules();
      await import('./reranker-readiness.mjs');
    } finally {
      process.argv = savedArgv;
      delete process.env.RERANKER_FORCE_STATE;
    }
    expect(mockWriteJsonOrText).toHaveBeenCalledWith(
      expect.objectContaining({ state: 'cold' }),
      true,
      expect.any(Function),
    );
  });

  it('calls fail on error', async () => {
    mockParseCliArgs.mockReturnValue({});
    mockExistsSync.mockImplementation(() => { throw new Error('crash'); });
    const savedArgv = [...process.argv];
    process.argv = [savedArgv[0], sourcePath];
    try {
      jest.resetModules();
      await import('./reranker-readiness.mjs');
    } finally {
      process.argv = savedArgv;
    }
    expect(mockFail).toHaveBeenCalledWith(expect.stringContaining('crash'), false);
  });

  it('calls fail with non-Error message', async () => {
    mockParseCliArgs.mockReturnValue({});
    mockExistsSync.mockImplementation(() => { throw 'string error'; });
    const savedArgv = [...process.argv];
    process.argv = [savedArgv[0], sourcePath];
    try {
      jest.resetModules();
      await import('./reranker-readiness.mjs');
    } finally {
      process.argv = savedArgv;
    }
    expect(mockFail).toHaveBeenCalledWith('string error', false);
  });
});