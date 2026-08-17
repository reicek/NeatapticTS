import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourcePath = path.resolve(__dirname, 'dense-readiness.mjs');

const mockExistsSync = jest.fn();
const mockValidateEmbeddings = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();
const mockFail = jest.fn();

jest.unstable_mockModule('node:fs', () => ({ existsSync: mockExistsSync, default: { existsSync: mockExistsSync } }));
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
}));
jest.unstable_mockModule('./embed-index.mjs', () => ({
  DEFAULT_MODEL_DIRECTORY: '/fake/model/dir',
  DEFAULT_MODEL_ID: 'fake-model-id',
}));
jest.unstable_mockModule('./init-schema.mjs', () => ({
  defaultDatabasePath: '/fake/db.sqlite',
}));
jest.unstable_mockModule('./validate-embeddings.mjs', () => ({
  validateEmbeddings: mockValidateEmbeddings,
}));

const { checkDenseReadiness } = await import('./dense-readiness.mjs');

afterEach(() => {
  jest.clearAllMocks();
  delete process.env.DENSE_FORCE_STATE;
});

describe('checkDenseReadiness', () => {
  it('returns forced cold state from env', async () => {
    process.env.DENSE_FORCE_STATE = 'cold';
    const result = await checkDenseReadiness();
    expect(result.state).toBe('cold');
    expect(result.ready).toBe(false);
    expect(result.reason).toContain('forced cold');
  });

  it('returns forced model-only state from env', async () => {
    process.env.DENSE_FORCE_STATE = 'model-only';
    const result = await checkDenseReadiness();
    expect(result.state).toBe('model-only');
    expect(result.ready).toBe(false);
    expect(result.reason).toContain('forced model-only');
  });

  it('ignores invalid forced state', async () => {
    process.env.DENSE_FORCE_STATE = 'invalid';
    mockExistsSync.mockReturnValue(false);
    const result = await checkDenseReadiness();
    expect(result.state).toBe('cold');
    expect(result.reason).toBe('Dense model assets are absent.');
  });

  it('ignores whitespace-only forced state', async () => {
    process.env.DENSE_FORCE_STATE = '  ';
    mockExistsSync.mockReturnValue(false);
    const result = await checkDenseReadiness();
    expect(result.state).toBe('cold');
  });

  it('returns cold when model file does not exist', async () => {
    mockExistsSync.mockReturnValue(false);
    const result = await checkDenseReadiness();
    expect(result.state).toBe('cold');
    expect(result.ready).toBe(false);
    expect(result.chunk_count).toBeNull();
    expect(result.embedding_count).toBeNull();
  });

  it('returns warm when validation passes', async () => {
    mockExistsSync.mockReturnValue(true);
    mockValidateEmbeddings.mockResolvedValue({
      pass: true,
      chunk_count: 100,
      embedding_count: 100,
    });
    const result = await checkDenseReadiness();
    expect(result.state).toBe('warm');
    expect(result.ready).toBe(true);
    expect(result.chunk_count).toBe(100);
    expect(result.embedding_count).toBe(100);
    expect(result.reason).toContain('100 chunks have embeddings');
  });

  it('returns model-only when validation fails with counts', async () => {
    mockExistsSync.mockReturnValue(true);
    mockValidateEmbeddings.mockResolvedValue({
      pass: false,
      chunk_count: 100,
      embedding_count: 50,
    });
    const result = await checkDenseReadiness();
    expect(result.state).toBe('model-only');
    expect(result.ready).toBe(false);
    expect(result.reason).toContain('expected 100, found 50');
  });

  it('returns model-only when validation fails with null counts', async () => {
    mockExistsSync.mockReturnValue(true);
    mockValidateEmbeddings.mockResolvedValue({
      pass: false,
      chunk_count: NaN,
      embedding_count: Infinity,
    });
    const result = await checkDenseReadiness();
    expect(result.state).toBe('model-only');
    expect(result.reason).toBe('Dense model is present but embeddings are missing or incomplete.');
  });

  it('returns model-only when validation fails with non-finite counts', async () => {
    mockExistsSync.mockReturnValue(true);
    mockValidateEmbeddings.mockResolvedValue({
      pass: false,
      chunk_count: 'not a number',
      embedding_count: undefined,
    });
    const result = await checkDenseReadiness();
    expect(result.state).toBe('model-only');
    expect(result.chunk_count).toBeNull();
    expect(result.embedding_count).toBeNull();
    expect(result.reason).toBe('Dense model is present but embeddings are missing or incomplete.');
  });

  it('returns model-only when validation throws', async () => {
    mockExistsSync.mockReturnValue(true);
    mockValidateEmbeddings.mockRejectedValue(new Error('DB connection failed'));
    const result = await checkDenseReadiness();
    expect(result.state).toBe('model-only');
    expect(result.ready).toBe(false);
    expect(result.reason).toContain('DB connection failed');
  });

  it('returns model-only when validation throws non-Error', async () => {
    mockExistsSync.mockReturnValue(true);
    mockValidateEmbeddings.mockRejectedValue('string error');
    const result = await checkDenseReadiness();
    expect(result.state).toBe('model-only');
    expect(result.reason).toContain('string error');
  });

  it('passes options through to validateEmbeddings', async () => {
    mockExistsSync.mockReturnValue(true);
    mockValidateEmbeddings.mockResolvedValue({ pass: true, chunk_count: 1, embedding_count: 1 });
    await checkDenseReadiness({ client: 'fake-client', modelId: 'custom-id' });
    expect(mockValidateEmbeddings).toHaveBeenCalledWith(expect.objectContaining({ modelId: 'custom-id', client: 'fake-client' }));
  });

  it('uses databasePath option as fallback for corpusDatabasePath', async () => {
    mockExistsSync.mockReturnValue(true);
    mockValidateEmbeddings.mockResolvedValue({ pass: true, chunk_count: 1, embedding_count: 1 });
    await checkDenseReadiness({ databasePath: '/custom/db.sqlite' });
    expect(mockValidateEmbeddings).toHaveBeenCalledWith(expect.objectContaining({ corpusDatabasePath: path.resolve('/custom/db.sqlite') }));
  });

  it('returns warm with 0 in reason when embedding_count is non-finite', async () => {
    mockExistsSync.mockReturnValue(true);
    mockValidateEmbeddings.mockResolvedValue({
      pass: true,
      chunk_count: 100,
      embedding_count: undefined,
    });
    const result = await checkDenseReadiness();
    expect(result.reason).toBe('0 chunks have embeddings.');
  });
});

describe('dense-readiness main()', () => {
  it('shows help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    const savedArgv = [...process.argv];
    process.argv = [savedArgv[0], sourcePath, '--help'];
    try {
      jest.resetModules();
      await import('./dense-readiness.mjs');
    } finally {
      process.argv = savedArgv;
    }
    expect(mockPrintHelp).toHaveBeenCalled();
  });

  it('writes report on success', async () => {
    mockParseCliArgs.mockReturnValue({ json: true });
    process.env.DENSE_FORCE_STATE = 'cold';
    const savedArgv = [...process.argv];
    process.argv = [savedArgv[0], sourcePath, '--json'];
    try {
      jest.resetModules();
      await import('./dense-readiness.mjs');
    } finally {
      process.argv = savedArgv;
      delete process.env.DENSE_FORCE_STATE;
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
      await import('./dense-readiness.mjs');
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
      await import('./dense-readiness.mjs');
    } finally {
      process.argv = savedArgv;
    }
    expect(mockFail).toHaveBeenCalledWith('string error', false);
  });

  it('formatter callback formats state and reason', async () => {
    mockParseCliArgs.mockReturnValue({ json: false });
    process.env.DENSE_FORCE_STATE = 'cold';
    const savedArgv = [...process.argv];
    process.argv = [savedArgv[0], sourcePath];
    try {
      jest.resetModules();
      await import('./dense-readiness.mjs');
    } finally {
      process.argv = savedArgv;
      delete process.env.DENSE_FORCE_STATE;
    }
    const formatter = mockWriteJsonOrText.mock.calls[0][2];
    expect(formatter({ state: 'cold', reason: 'forced cold readiness.' })).toBe('cold: forced cold readiness.');
  });
});