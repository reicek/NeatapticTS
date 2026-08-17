import { jest } from '@jest/globals';
import path from 'node:path';

// Mock node:fs
const mockExistsSync = jest.fn().mockReturnValue(true);
jest.unstable_mockModule('node:fs', () => ({
  existsSync: mockExistsSync,
  default: { existsSync: mockExistsSync },
}));

// Mock cortex-db
const mockGetTursoClient = jest.fn();
jest.unstable_mockModule('../../scripts/mcp-semantic/tools/cortex-db.mjs', () => ({
  getTursoClient: mockGetTursoClient,
  default: { getTursoClient: mockGetTursoClient },
}));

// Mock build-index
const mockBuildSemanticIndex = jest.fn();
jest.unstable_mockModule('../build-index.mjs', () => ({
  buildSemanticIndex: mockBuildSemanticIndex,
  default: { buildSemanticIndex: mockBuildSemanticIndex },
}));

// Mock embed-index
const mockBuildEmbeddingIndex = jest.fn();
jest.unstable_mockModule('../embed-index.mjs', () => ({
  buildEmbeddingIndex: mockBuildEmbeddingIndex,
  default: { buildEmbeddingIndex: mockBuildEmbeddingIndex },
}));

// Mock init-schema
jest.unstable_mockModule('../init-schema.mjs', () => ({
  defaultDatabasePath: '/fake/db.sqlite',
  repoRoot: '/fake/repo',
  default: { defaultDatabasePath: '/fake/db.sqlite', repoRoot: '/fake/repo' },
}));

// Mock cli-utils
const mockWriteJsonOrText = jest.fn();
jest.unstable_mockModule('../cli-utils.mjs', () => ({
  parseCliArgs: (args) => {
    const result = {};
    for (let i = 0; i < args.length; i++) {
      if (args[i] === '--help') result['help'] = true;
      else if (args[i] === '--json') result['json'] = true;
      else if (args[i] === '--skip-ann') result['skip-ann'] = true;
      else if (args[i] === '--database') result['database'] = args[++i];
      else if (args[i] === '--changed-file-globs') result['changed-file-globs'] = args[++i];
      else if (args[i].startsWith('--files=')) result['files'] = args[i].slice(8);
      else if (!args[i].startsWith('--')) {
        result['_'] = result['_'] || [];
        result['_'].push(args[i]);
      }
    }
    return result;
  },
  writeJsonOrText: mockWriteJsonOrText,
  default: { parseCliArgs: undefined, writeJsonOrText: mockWriteJsonOrText },
}));

const { createFreshnessHook, DEFAULT_CHANGED_FILE_GLOBS, runFreshnessHookCli, formatCliSummary, printUsage } = await import('./freshness-hooks.mjs');

function createMockClient(documents = [], filePaths = null) {
  return {
    execute: jest.fn(async (sql) => {
      if (sql.includes('SELECT file_path, doc_family')) {
        return { rows: documents };
      }
      if (sql.includes('SELECT file_path FROM documents')) {
        return { rows: (filePaths ?? documents).map((d) => ({ file_path: d.file_path })) };
      }
      if (sql.includes("sqlite_master")) {
        return { rows: [{ name: 'term_embeddings' }] };
      }
      return { rows: [] };
    }),
  };
}

afterEach(() => {
  jest.clearAllMocks();
  mockExistsSync.mockReturnValue(true);
});

// ---------------------------------------------------------------------------
// DEFAULT_CHANGED_FILE_GLOBS
// ---------------------------------------------------------------------------
describe('DEFAULT_CHANGED_FILE_GLOBS', () => {
  it('exports expected default globs', () => {
    expect(DEFAULT_CHANGED_FILE_GLOBS).toContain('src/**/*.ts');
    expect(DEFAULT_CHANGED_FILE_GLOBS).toContain('scripts/**/*.mjs');
    expect(DEFAULT_CHANGED_FILE_GLOBS).toContain('plans/**/*.md');
  });
});

describe('createFreshnessHook — defaults', () => {
  it('uses defaults when options are empty', () => {
    const hook = createFreshnessHook({});
    expect(hook).toBeDefined();
    expect(typeof hook.flush).toBe('function');
    expect(typeof hook.notifyWrite).toBe('function');
  });

  it('uses defaults when called with no arguments', () => {
    const hook = createFreshnessHook();
    expect(hook).toBeDefined();
    expect(typeof hook.flush).toBe('function');
  });

  it('schedules a debounce flush that fires automatically', async () => {
    const mockClient = createMockClient([{ file_path: 'src/foo.ts', doc_family: 'source-code' }]);
    mockGetTursoClient.mockResolvedValue(mockClient);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 10 });
    mockBuildEmbeddingIndex.mockResolvedValue({ indexed: 1, skipped: 0, elapsedMs: 10 });

    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
    });
    hook.notifyWrite('src/foo.ts');
    // Wait for the debounce timer to fire
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(mockBuildSemanticIndex).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// createFreshnessHook — notify methods
// ---------------------------------------------------------------------------
describe('createFreshnessHook — notify methods', () => {
  it('notifyWrite adds path and schedules flush', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: ['src/foo.ts'], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
    });
    hook.notifyWrite('src/foo.ts');
    const result = await hook.flush();
    expect(result.updated).toEqual(['src/foo.ts']);
    expect(runIncrementalBuild).toHaveBeenCalled();
  });

  it('notifyRename adds both old and new paths', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: [], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
    });
    hook.notifyRename('src/old.ts', 'src/new.ts');
    await hook.flush();
    const calledPaths = runIncrementalBuild.mock.calls[0][0];
    expect(calledPaths).toContain('src/old.ts');
    expect(calledPaths).toContain('src/new.ts');
  });

  it('notifyDelete adds path', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: [], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
    });
    hook.notifyDelete('src/deleted.ts');
    await hook.flush();
    const calledPaths = runIncrementalBuild.mock.calls[0][0];
    expect(calledPaths).toContain('src/deleted.ts');
  });

  it('normalizes backslash paths in notifyWrite', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: [], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src\\deep\\file.ts');
    await hook.flush();
    const calledPaths = runIncrementalBuild.mock.calls[0][0];
    expect(calledPaths).toContain('src/deep/file.ts');
  });

  it('strips leading slashes in notifyWrite', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: [], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('/src/foo.ts');
    await hook.flush();
    const calledPaths = runIncrementalBuild.mock.calls[0][0];
    expect(calledPaths).toContain('src/foo.ts');
  });

  it('handles null/undefined path in notifyWrite', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: [], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
    });
    hook.notifyWrite(null);
    const result = await hook.flush();
    // null becomes '' which doesn't match any glob → empty result
    expect(result.updated).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// createFreshnessHook — flush behavior
// ---------------------------------------------------------------------------
describe('createFreshnessHook — flush', () => {
  it('returns empty result when no pending paths', async () => {
    const runIncrementalBuild = jest.fn();
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient(),
      skip_ann: true,
    });
    const result = await hook.flush();
    expect(result.updated).toEqual([]);
    expect(result.failed).toEqual([]);
    expect(runIncrementalBuild).not.toHaveBeenCalled();
  });

  it('filters out paths not matching globs or indexed paths', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: [], failed: [] });
    const client = createMockClient([], []); // no indexed paths
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('random/file.txt'); // doesn't match src/**/*.ts
    const result = await hook.flush();
    expect(result.updated).toEqual([]);
    expect(runIncrementalBuild).not.toHaveBeenCalled();
  });

  it('allows paths that are already indexed even if not matching globs', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: ['old/file.txt'], failed: [] });
    const client = createMockClient(
      [{ file_path: 'old/file.txt', doc_family: 'unknown' }],
      [{ file_path: 'old/file.txt' }],
    );
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('old/file.txt');
    const result = await hook.flush();
    expect(result.updated).toEqual(['old/file.txt']);
  });

  it('returns failed result when runIncrementalBuild throws', async () => {
    const runIncrementalBuild = jest.fn().mockRejectedValue(new Error('Build failed'));
    const logWarning = jest.fn();
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
      logWarning,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/foo.ts');
    const result = await hook.flush();
    expect(result.failed).toHaveLength(1);
    expect(result.failed[0].path).toBe('src/foo.ts');
    expect(result.failed[0].error).toBe('Build failed');
    expect(logWarning).toHaveBeenCalled();
  });

  it('returns failed result with non-Error thrown', async () => {
    const runIncrementalBuild = jest.fn().mockRejectedValue('string error');
    const logWarning = jest.fn();
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
      logWarning,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/foo.ts');
    const result = await hook.flush();
    expect(result.failed[0].error).toBe('string error');
  });

  it('queues embedding update when skip_ann is false', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: ['src/foo.ts'], failed: [] });
    mockBuildEmbeddingIndex.mockResolvedValue({ indexed: 1, skipped: 0 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: false,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/foo.ts');
    await hook.flush();
    expect(mockBuildEmbeddingIndex).toHaveBeenCalled();
  });

  it('does not queue embedding update when skip_ann is true', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: ['src/foo.ts'], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/foo.ts');
    await hook.flush();
    expect(mockBuildEmbeddingIndex).not.toHaveBeenCalled();
  });

  it('returns same promise when flush is called concurrently', async () => {
    let resolveBuild;
    const buildPromise = new Promise((resolve) => { resolveBuild = resolve; });
    const runIncrementalBuild = jest.fn().mockReturnValue(buildPromise);
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/foo.ts');
    const flushPromise1 = hook.flush();
    const flushPromise2 = hook.flush();
    expect(flushPromise1).toStrictEqual(flushPromise2);
    resolveBuild({ updated: ['src/foo.ts'], failed: [] });
    await flushPromise1;
  });

  it('uses getTursoClient when no client provided for collectIndexedPaths', async () => {
    const mockClient = createMockClient([], []);
    mockGetTursoClient.mockResolvedValue(mockClient);
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: [], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/foo.ts');
    await hook.flush();
    expect(mockGetTursoClient).toHaveBeenCalled();
  });

  it('handles error from getTursoClient in collectIndexedPaths', async () => {
    mockGetTursoClient.mockRejectedValue(new Error('DB connection failed'));
    const logWarning = jest.fn();
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: [], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
      logWarning,
    });
    hook.notifyWrite('src/foo.ts');
    // collectIndexedPaths throws → flushNow catches? No, collectIndexedPaths is not in try/catch
    // Actually, looking at the code, collectIndexedPaths is called outside the try/catch
    // So the error propagates to the flush() caller
    await expect(hook.flush()).rejects.toThrow('DB connection failed');
  });

  it('queues embedding update and handles error', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: ['src/foo.ts'], failed: [] });
    mockBuildEmbeddingIndex.mockRejectedValue(new Error('Embedding failed'));
    const logWarning = jest.fn();
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: false,
      changed_file_globs: ['src/**/*.ts'],
      logWarning,
    });
    hook.notifyWrite('src/foo.ts');
    await hook.flush();
    // queueEmbeddingUpdate catches the error and calls logWarning
    // Wait a tick for the async catch
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(logWarning).toHaveBeenCalledWith(expect.stringContaining('Embedding failed'));
  });

  it('queues embedding update with non-Error rejection', async () => {
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: ['src/foo.ts'], failed: [] });
    mockBuildEmbeddingIndex.mockRejectedValue('string error');
    const logWarning = jest.fn();
    const hook = createFreshnessHook({
      debounce_ms: 0,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: false,
      changed_file_globs: ['src/**/*.ts'],
      logWarning,
    });
    hook.notifyWrite('src/foo.ts');
    await hook.flush();
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(logWarning).toHaveBeenCalledWith(expect.stringContaining('string error'));
  });
});

// ---------------------------------------------------------------------------
// createFreshnessHook — debounce timer
// ---------------------------------------------------------------------------
describe('createFreshnessHook — debounce', () => {
  it('schedules flush with debounce timer', async () => {
    jest.useFakeTimers({ doNotFake: ['setImmediate'] });
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: ['src/foo.ts'], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 100,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/foo.ts');
    expect(runIncrementalBuild).not.toHaveBeenCalled();
    jest.advanceTimersByTime(100);
    await new Promise((resolve) => setImmediate(resolve)); // let microtasks settle
    expect(runIncrementalBuild).toHaveBeenCalled();
    jest.useRealTimers();
  });

  it('resets timer on second notification', async () => {
    jest.useFakeTimers({ doNotFake: ['setImmediate'] });
    const runIncrementalBuild = jest.fn().mockResolvedValue({ updated: [], failed: [] });
    const hook = createFreshnessHook({
      debounce_ms: 100,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/foo.ts');
    jest.advanceTimersByTime(50);
    hook.notifyWrite('src/bar.ts');
    jest.advanceTimersByTime(50);
    // Timer was reset, only 50ms since last notify → not yet
    expect(runIncrementalBuild).not.toHaveBeenCalled();
    jest.advanceTimersByTime(50);
    await new Promise((resolve) => setImmediate(resolve));
    expect(runIncrementalBuild).toHaveBeenCalled();
    jest.useRealTimers();
  });

  it('handles error in debounced flush via logWarning', async () => {
    jest.useFakeTimers({ doNotFake: ['setImmediate'] });
    const runIncrementalBuild = jest.fn().mockRejectedValue(new Error('Build failed'));
    const logWarning = jest.fn();
    const hook = createFreshnessHook({
      debounce_ms: 100,
      runIncrementalBuild,
      client: createMockClient([], []),
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
      logWarning,
    });
    hook.notifyWrite('src/foo.ts');
    jest.advanceTimersByTime(100);
    await new Promise((resolve) => setImmediate(resolve));
    expect(logWarning).toHaveBeenCalledWith(expect.stringContaining('Build failed'));
    jest.useRealTimers();
  });
});

// ---------------------------------------------------------------------------
// createFreshnessHook — default runIncrementalBuild
// ---------------------------------------------------------------------------
describe('createFreshnessHook — defaultRunIncrementalBuild', () => {
  it('builds index with existing and new documents', async () => {
    const mockClient = createMockClient(
      [{ file_path: 'src/existing.ts', doc_family: 'ts-source' }],
    );
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 2, skipped: 0, chunks: 5, elapsedMs: 100 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/new.ts');
    const result = await hook.flush();
    expect(result.updated).toContain('src/new.ts');
    expect(result.summary.indexed).toBe(2);
    expect(mockBuildSemanticIndex).toHaveBeenCalled();
  });

  it('uses getTursoClient when no client provided', async () => {
    const mockClient = createMockClient([]);
    mockGetTursoClient.mockResolvedValue(mockClient);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/new.ts');
    await hook.flush();
    expect(mockGetTursoClient).toHaveBeenCalled();
  });

  it('excludes deleted files from corpus', async () => {
    const mockClient = createMockClient(
      [
        { file_path: 'src/exists.ts', doc_family: 'ts-source' },
        { file_path: 'src/deleted.ts', doc_family: 'ts-source' },
      ],
    );
    mockExistsSync.mockImplementation((p) => !p.includes('deleted'));
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/exists.ts');
    const result = await hook.flush();
    // Deleted file should be excluded from corpus
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const corpusPaths = buildCall.corpusDocuments.map((d) => d.filePath);
    expect(corpusPaths).toContain('src/exists.ts');
    expect(corpusPaths).not.toContain('src/deleted.ts');
  });

  it('infers family for new paths not in database', async () => {
    const mockClient = createMockClient([]);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts', 'plans/**/*.md', 'examples/**/README.md'],
    });
    hook.notifyWrite('src/new.ts');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const newDoc = buildCall.corpusDocuments.find((d) => d.filePath === 'src/new.ts');
    expect(newDoc.family).toBe('ts-source');
  });

  it('filters updated paths by existsSync', async () => {
    const mockClient = createMockClient([]);
    mockExistsSync.mockImplementation((p) => !p.includes('gone'));
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/exists.ts');
    hook.notifyWrite('src/gone.ts');
    const result = await hook.flush();
    expect(result.updated).toContain('src/exists.ts');
    expect(result.updated).not.toContain('src/gone.ts');
  });

  it('handles null doc_family in database rows', async () => {
    const mockClient = createMockClient(
      [{ file_path: 'src/existing.ts', doc_family: null }],
    );
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/existing.ts');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const doc = buildCall.corpusDocuments.find((d) => d.filePath === 'src/existing.ts');
    expect(doc.family).toBe('unknown');
  });
});

// ---------------------------------------------------------------------------
// inferFamily — tested via defaultRunIncrementalBuild
// ---------------------------------------------------------------------------
describe('inferFamily via defaultRunIncrementalBuild', () => {
  it('infers readme family for src/**/README.md', async () => {
    const mockClient = createMockClient([]);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['src/**/README.md'],
    });
    hook.notifyWrite('src/module/README.md');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const doc = buildCall.corpusDocuments.find((d) => d.filePath === 'src/module/README.md');
    expect(doc.family).toBe('readme');
  });

  it('infers plan family for plans/**/*.md', async () => {
    const mockClient = createMockClient([]);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['plans/**/*.md'],
    });
    hook.notifyWrite('plans/phase-1/step-1.md');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const doc = buildCall.corpusDocuments.find((d) => d.filePath === 'plans/phase-1/step-1.md');
    expect(doc.family).toBe('plan');
  });

  it('infers completed-plan for plans/completed/**/*.md', async () => {
    const mockClient = createMockClient([]);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['plans/completed/**/*.md'],
    });
    hook.notifyWrite('plans/completed/old-plan.md');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const doc = buildCall.corpusDocuments.find((d) => d.filePath === 'plans/completed/old-plan.md');
    expect(doc.family).toBe('completed-plan');
  });

  it('infers ts-source but ignores .d.ts files', async () => {
    const mockClient = createMockClient([]);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/types.d.ts');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const doc = buildCall.corpusDocuments.find((d) => d.filePath === 'src/types.d.ts');
    // .d.ts matches src/**/*.ts but is in ignore list → falls through to unknown
    expect(doc.family).toBe('unknown');
  });

  it('infers ts-source but ignores .test.ts files', async () => {
    const mockClient = createMockClient([]);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['src/**/*.ts'],
    });
    hook.notifyWrite('src/foo.test.ts');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const doc = buildCall.corpusDocuments.find((d) => d.filePath === 'src/foo.test.ts');
    expect(doc.family).toBe('unknown');
  });

  it('infers demo family for examples/**/*.ts', async () => {
    const mockClient = createMockClient([]);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['examples/**/*.ts'],
    });
    hook.notifyWrite('examples/demo.ts');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const doc = buildCall.corpusDocuments.find((d) => d.filePath === 'examples/demo.ts');
    expect(doc.family).toBe('demo');
  });

  it('infers benchmark family for benchmarks/**/*.test.ts', async () => {
    const mockClient = createMockClient([]);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['benchmarks/**/*.test.ts'],
    });
    hook.notifyWrite('benchmarks/perf.test.ts');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const doc = buildCall.corpusDocuments.find((d) => d.filePath === 'benchmarks/perf.test.ts');
    expect(doc.family).toBe('benchmark');
  });

  it('infers root-doc family for README.md', async () => {
    const mockClient = createMockClient([]);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['README.md'],
    });
    hook.notifyWrite('README.md');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const doc = buildCall.corpusDocuments.find((d) => d.filePath === 'README.md');
    expect(doc.family).toBe('root-doc');
  });

  it('infers unknown for unmatched paths', async () => {
    const mockClient = createMockClient([]);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 50 });
    const hook = createFreshnessHook({
      debounce_ms: 0,
      client: mockClient,
      skip_ann: true,
      changed_file_globs: ['random/**/*.xyz'],
    });
    hook.notifyWrite('random/file.xyz');
    await hook.flush();
    const buildCall = mockBuildSemanticIndex.mock.calls[0][0];
    const doc = buildCall.corpusDocuments.find((d) => d.filePath === 'random/file.xyz');
    expect(doc.family).toBe('unknown');
  });
});

// ---------------------------------------------------------------------------
// CLI entry point — runFreshnessHookCli, formatCliSummary, printUsage
// ---------------------------------------------------------------------------
describe('runFreshnessHookCli', () => {
  it('returns early when no files are provided', async () => {
    const summary = await runFreshnessHookCli({});
    expect(summary.notified).toBe(0);
    expect(summary.flushed).toBe(false);
    expect(summary.fatalError).toBeNull();
  });

  it('returns early when files array is empty', async () => {
    const summary = await runFreshnessHookCli({ files: [] });
    expect(summary.notified).toBe(0);
    expect(summary.flushed).toBe(false);
  });

  it('processes files from --files= csv string', async () => {
    const mockClient = createMockClient([{ file_path: 'src/foo.ts', doc_family: 'source-code' }]);
    mockGetTursoClient.mockResolvedValue(mockClient);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 10 });
    mockBuildEmbeddingIndex.mockResolvedValue({ indexed: 1, skipped: 0, elapsedMs: 10 });

    const summary = await runFreshnessHookCli({ files: 'src/foo.ts' });
    expect(summary.notified).toBe(1);
    expect(summary.flushed).toBe(true);
    expect(summary.result.updated).toEqual(['src/foo.ts']);
  });

  it('processes files from positional args array', async () => {
    const mockClient = createMockClient([{ file_path: 'src/bar.ts', doc_family: 'source-code' }]);
    mockGetTursoClient.mockResolvedValue(mockClient);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 10 });
    mockBuildEmbeddingIndex.mockResolvedValue({ indexed: 1, skipped: 0, elapsedMs: 10 });

    const summary = await runFreshnessHookCli({ _: ['src/bar.ts'] });
    expect(summary.notified).toBe(1);
    expect(summary.flushed).toBe(true);
  });

  it('processes files with skip-ann option', async () => {
    const mockClient = createMockClient([{ file_path: 'src/baz.ts', doc_family: 'source-code' }]);
    mockGetTursoClient.mockResolvedValue(mockClient);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 10 });

    const summary = await runFreshnessHookCli({ files: 'src/baz.ts', 'skip-ann': true });
    expect(summary.notified).toBe(1);
    expect(summary.flushed).toBe(true);
    expect(mockBuildEmbeddingIndex).not.toHaveBeenCalled();
  });

  it('processes files with custom changed-file-globs', async () => {
    const mockClient = createMockClient([{ file_path: 'random/file.xyz', doc_family: 'unknown' }]);
    mockGetTursoClient.mockResolvedValue(mockClient);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 10 });
    mockBuildEmbeddingIndex.mockResolvedValue({ indexed: 1, skipped: 0, elapsedMs: 10 });

    const summary = await runFreshnessHookCli({ files: 'random/file.xyz', 'changed-file-globs': 'random/**/*.xyz' });
    expect(summary.notified).toBe(1);
    expect(summary.flushed).toBe(true);
  });

  it('catches errors from hook.flush() and records fatalError', async () => {
    mockGetTursoClient.mockRejectedValue(new Error('client failed'));

    const summary = await runFreshnessHookCli({ files: 'src/err.ts' });
    expect(summary.flushed).toBe(false);
    expect(summary.fatalError).toBe('client failed');
  });

  it('catches non-Error throws and stringifies them', async () => {
    mockGetTursoClient.mockRejectedValue('string error');

    const summary = await runFreshnessHookCli({ files: 'src/err2.ts' });
    expect(summary.fatalError).toBe('string error');
  });

  it('uses custom database path from args', async () => {
    const mockClient = createMockClient([{ file_path: 'src/foo.ts', doc_family: 'source-code' }]);
    mockGetTursoClient.mockResolvedValue(mockClient);
    mockBuildSemanticIndex.mockResolvedValue({ indexed: 1, skipped: 0, chunks: 1, elapsedMs: 10 });
    mockBuildEmbeddingIndex.mockResolvedValue({ indexed: 1, skipped: 0, elapsedMs: 10 });

    const summary = await runFreshnessHookCli({ files: 'src/foo.ts', database: '/custom/db.sqlite' });
    expect(summary.databasePath).toContain('db.sqlite');
  });
});

describe('formatCliSummary', () => {
  it('formats summary without fatal error', () => {
    const summary = {
      notified: 2,
      result: { updated: ['a.ts', 'b.ts'], failed: [] },
      elapsedMs: 100,
      fatalError: null,
    };
    const text = formatCliSummary(summary);
    expect(text).toBe('freshness-hooks: notified=2 updated=2 failed=0 elapsed=100ms');
  });

  it('formats summary with fatal error', () => {
    const summary = {
      notified: 1,
      result: { updated: [], failed: ['x.ts'] },
      elapsedMs: 50,
      fatalError: 'something broke',
    };
    const text = formatCliSummary(summary);
    expect(text).toBe('freshness-hooks: notified=1 updated=0 failed=1 elapsed=50ms\nERROR: something broke');
  });
});

describe('printUsage', () => {
  it('prints usage text to console.log', () => {
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => undefined);
    printUsage();
    expect(logSpy).toHaveBeenCalled();
    const output = logSpy.mock.calls[0][0];
    expect(output).toContain('freshness-hooks');
    expect(output).toContain('Usage:');
    logSpy.mockRestore();
  });
});