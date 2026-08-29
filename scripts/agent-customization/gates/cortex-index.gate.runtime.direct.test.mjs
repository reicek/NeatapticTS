import { jest } from '@jest/globals';
import {
  getDefaultDeps,
  isPlanPath,
  readSnapshotCurrency,
  resolveFixHint,
  resolveWorkflowPlanPath,
  runWorkflowMcpSelfCheck,
} from './cortex-index.gate.runtime.mjs';

function fakeSnapshot(payload) {
  return JSON.stringify(
    payload ?? { generated_at: '2024-01-01T00:00:00.000Z' },
  );
}

describe('readSnapshotCurrency', () => {
  const fallback = {
    pass: false,
    indexDocuments: 0,
    snapshotAgeSeconds: 0,
    snapshotIndexedAt: null,
  };

  it('returns the fallback result when the snapshot file is missing', async () => {
    const result = await readSnapshotCurrency(
      {
        databasePath: 'db.sqlite',
        snapshotPath: 'missing.json',
        snapshotMaxAgeMs: 86_400_000,
      },
      { existsSync: () => false },
    );

    expect(result).toEqual(fallback);
  });

  it('returns the fallback result when the snapshot JSON is invalid', async () => {
    const result = await readSnapshotCurrency(
      {
        databasePath: 'db.sqlite',
        snapshotPath: 'bad.json',
        snapshotMaxAgeMs: 86_400_000,
      },
      {
        existsSync: () => true,
        readFile: async () => 'not-json',
      },
    );

    expect(result).toEqual(fallback);
  });

  it('returns the fallback result when generated_at is not a valid date', async () => {
    const result = await readSnapshotCurrency(
      {
        databasePath: 'db.sqlite',
        snapshotPath: 'bad-date.json',
        snapshotMaxAgeMs: 86_400_000,
      },
      {
        existsSync: () => true,
        readFile: async () => fakeSnapshot({ generated_at: 'not-a-date' }),
      },
    );

    expect(result).toEqual(fallback);
  });

  it('passes when the latest indexed_at is within the freshness window', async () => {
    const getTursoClient = jest.fn().mockResolvedValue({
      execute: jest.fn().mockResolvedValue({
        rows: [{ documents: 42, indexed_at: 1_704_067_200_000 }],
      }),
    });
    const closeTursoClient = jest.fn().mockResolvedValue(undefined);

    const result = await readSnapshotCurrency(
      {
        databasePath: 'db.sqlite',
        snapshotPath: 'fresh.json',
        snapshotMaxAgeMs: 86_400_000,
      },
      {
        existsSync: () => true,
        readFile: async () =>
          fakeSnapshot({ generated_at: '2023-12-31T00:00:00.000Z' }),
        getTursoClient,
        closeTursoClient,
      },
    );

    expect(result.pass).toBe(true);
    expect(result.indexDocuments).toBe(42);
    expect(result.snapshotAgeSeconds).toBeGreaterThan(0);
    expect(result.snapshotIndexedAt).toBe(
      new Date(1_704_067_200_000).toISOString(),
    );
    expect(getTursoClient).toHaveBeenCalledWith('db.sqlite');
    expect(closeTursoClient).toHaveBeenCalledWith('db.sqlite');
  });

  it('handles an empty row set by using defaults', async () => {
    const getTursoClient = jest.fn().mockResolvedValue({
      execute: jest.fn().mockResolvedValue({ rows: [] }),
    });

    const result = await readSnapshotCurrency(
      {
        databasePath: 'db.sqlite',
        snapshotPath: 'empty-row.json',
        snapshotMaxAgeMs: 86_400_000,
      },
      {
        existsSync: () => true,
        readFile: async () => fakeSnapshot(),
        getTursoClient,
        closeTursoClient: jest.fn(),
      },
    );

    expect(result.pass).toBe(false);
    expect(result.indexDocuments).toBe(0);
    expect(result.snapshotIndexedAt).toBeNull();
  });

  it('handles null row fields by using defaults', async () => {
    const getTursoClient = jest.fn().mockResolvedValue({
      execute: jest.fn().mockResolvedValue({
        rows: [{ documents: null, indexed_at: null }],
      }),
    });

    const result = await readSnapshotCurrency(
      {
        databasePath: 'db.sqlite',
        snapshotPath: 'null-fields.json',
        snapshotMaxAgeMs: 86_400_000,
      },
      {
        existsSync: () => true,
        readFile: async () => fakeSnapshot(),
        getTursoClient,
        closeTursoClient: jest.fn(),
      },
    );

    expect(result.pass).toBe(false);
    expect(result.indexDocuments).toBe(0);
    expect(result.snapshotIndexedAt).toBeNull();
  });

  it('fails when the snapshot is older than the freshness window', async () => {
    const getTursoClient = jest.fn().mockResolvedValue({
      execute: jest.fn().mockResolvedValue({
        rows: [{ documents: 5, indexed_at: 1_704_067_200_000 }],
      }),
    });
    const closeTursoClient = jest.fn().mockResolvedValue(undefined);

    const result = await readSnapshotCurrency(
      {
        databasePath: 'db.sqlite',
        snapshotPath: 'stale.json',
        snapshotMaxAgeMs: 1_000,
      },
      {
        existsSync: () => true,
        readFile: async () =>
          fakeSnapshot({ generated_at: '2023-01-01T00:00:00.000Z' }),
        getTursoClient,
        closeTursoClient,
      },
    );

    expect(result.pass).toBe(false);
    expect(result.indexDocuments).toBe(5);
  });

  it('fails when the database has not indexed any documents', async () => {
    const getTursoClient = jest.fn().mockResolvedValue({
      execute: jest.fn().mockResolvedValue({
        rows: [{ documents: 0, indexed_at: 0 }],
      }),
    });

    const result = await readSnapshotCurrency(
      {
        databasePath: 'db.sqlite',
        snapshotPath: 'empty.json',
        snapshotMaxAgeMs: 86_400_000,
      },
      {
        existsSync: () => true,
        readFile: async () => fakeSnapshot(),
        getTursoClient,
        closeTursoClient: jest.fn(),
      },
    );

    expect(result.pass).toBe(false);
    expect(result.snapshotIndexedAt).toBeNull();
  });
});

describe('runWorkflowMcpSelfCheck', () => {
  it('returns pass=false when the subprocess reports an error', () => {
    const spawnSync = jest
      .fn()
      .mockReturnValue({ error: new Error('spawn failure') });

    const result = runWorkflowMcpSelfCheck(
      { timeoutMs: 5_000, workflowPlanPath: 'plans/test.plans.md' },
      { spawnSync },
    );

    expect(result.pass).toBe(false);
    expect(result.report).toBeNull();
    expect(spawnSync).toHaveBeenCalledWith(
      process.execPath,
      expect.arrayContaining(['--self-check', '--json']),
      expect.objectContaining({ encoding: 'utf8', timeout: 5_000 }),
    );
  });

  it('returns pass=true when the subprocess exits 0 with ok=true', () => {
    const spawnSync = jest.fn().mockReturnValue({
      error: null,
      status: 0,
      stdout: JSON.stringify({ ok: true, detail: 'all good' }),
    });

    const result = runWorkflowMcpSelfCheck(
      { timeoutMs: 5_000, workflowPlanPath: 'plans/test.plans.md' },
      { spawnSync },
    );

    expect(result.pass).toBe(true);
    expect(result.report).toEqual({ ok: true, detail: 'all good' });
  });

  it('returns pass=false when the subprocess exits non-zero', () => {
    const spawnSync = jest.fn().mockReturnValue({
      error: null,
      status: 1,
      stdout: JSON.stringify({ ok: false }),
    });

    const result = runWorkflowMcpSelfCheck(
      { timeoutMs: 5_000, workflowPlanPath: 'plans/test.plans.md' },
      { spawnSync },
    );

    expect(result.pass).toBe(false);
  });

  it('returns pass=false when stdout is not valid JSON', () => {
    const spawnSync = jest.fn().mockReturnValue({
      error: null,
      status: 0,
      stdout: 'garbage',
    });

    const result = runWorkflowMcpSelfCheck(
      { timeoutMs: 5_000, workflowPlanPath: 'plans/test.plans.md' },
      { spawnSync },
    );

    expect(result.pass).toBe(false);
    expect(result.report).toBeNull();
  });
});

describe('resolveWorkflowPlanPath', () => {
  it('returns the default path when the config cannot be read', async () => {
    const result = await resolveWorkflowPlanPath({
      readFile: async () => {
        throw new Error('ENOENT');
      },
    });

    expect(result).toBe('plans/mcp-active-binding.plans.md');
  });

  it('returns the default path when no workflow server args exist', async () => {
    const result = await resolveWorkflowPlanPath({
      readFile: async () => JSON.stringify({ servers: {} }),
    });

    expect(result).toBe('plans/mcp-active-binding.plans.md');
  });

  it('extracts the plan argument from the workflow server args', async () => {
    const result = await resolveWorkflowPlanPath({
      readFile: async () =>
        JSON.stringify({
          servers: {
            'neataptic-workflow-mcp': {
              args: ['--plan=plans/custom.plans.md'],
            },
          },
        }),
    });

    expect(result).toBe('plans/custom.plans.md');
  });

  it('ignores non-string and unrelated args', async () => {
    const result = await resolveWorkflowPlanPath({
      readFile: async () =>
        JSON.stringify({
          servers: {
            'neataptic-workflow-mcp': {
              args: [123, '--verbose', 'plans/custom.plans.md'],
            },
          },
        }),
    });

    expect(result).toBe('plans/mcp-active-binding.plans.md');
  });
});

describe('resolveFixHint', () => {
  const okReports = {
    indexReport: { pass: true },
    snapshotCurrency: { pass: true },
    corpusMcpReport: { pass: true },
    workflowMcpReport: { pass: true },
  };

  it('returns null when everything passes', () => {
    expect(resolveFixHint(okReports)).toBeNull();
  });

  it('delegates to resolveStalePlanFixHint for stale-only plan paths', () => {
    const helper = jest.fn().mockReturnValue('stale plan hint');
    const reports = {
      ...okReports,
      indexReport: {
        pass: false,
        stale_paths: ['foo.plans.md'],
        missing_paths: [],
      },
    };

    expect(resolveFixHint(reports, { resolveStalePlanFixHint: helper })).toBe(
      'stale plan hint',
    );
    expect(helper).toHaveBeenCalledWith(['foo.plans.md']);
  });

  it('falls back to the generic build hint when non-stale issues exist', () => {
    const reports = {
      ...okReports,
      indexReport: {
        pass: false,
        stale_paths: ['foo.plans.md'],
        missing_paths: ['bar.md'],
      },
    };

    expect(resolveFixHint(reports)).toBe(
      'Run: node rag-index/build-index.mjs to rebuild stale index',
    );
  });

  it('falls back to the generic build hint when stale paths are not plan files', () => {
    const reports = {
      ...okReports,
      indexReport: {
        pass: false,
        stale_paths: ['foo.md'],
        missing_paths: [],
      },
    };

    expect(resolveFixHint(reports)).toBe(
      'Run: node rag-index/build-index.mjs to rebuild stale index',
    );
  });

  it('falls back to the generic build hint when no helper is supplied', () => {
    const reports = {
      ...okReports,
      indexReport: {
        pass: false,
        stale_paths: ['foo.plans.md'],
        missing_paths: [],
      },
    };

    expect(resolveFixHint(reports)).toBe(
      'Run: node rag-index/build-index.mjs to rebuild stale index',
    );
  });

  it('uses empty defaults when path arrays are omitted', () => {
    const reports = {
      ...okReports,
      indexReport: { pass: false },
    };

    expect(resolveFixHint(reports)).toBe(
      'Run: node rag-index/build-index.mjs to rebuild stale index',
    );
  });

  it('reports a stale snapshot', () => {
    const reports = {
      ...okReports,
      snapshotCurrency: { pass: false },
    };

    expect(resolveFixHint(reports)).toBe(
      'Run: npm run index:build-snapshot to regenerate snapshot',
    );
  });

  it('reports a dead corpus MCP', () => {
    const reports = {
      ...okReports,
      corpusMcpReport: { pass: false },
    };

    expect(resolveFixHint(reports)).toBe(
      'Check cortex server in .vscode/mcp.json; run cortex-mcp-smoke.mjs for details',
    );
  });

  it('reports a dead workflow MCP', () => {
    const reports = {
      ...okReports,
      workflowMcpReport: { pass: false },
    };

    expect(resolveFixHint(reports)).toBe(
      'Restart neataptic-workflow-mcp server to bind to active plan path',
    );
  });
});

describe('isPlanPath', () => {
  it('returns true for .plans.md paths', () => {
    expect(isPlanPath('plans/foo.plans.md')).toBe(true);
  });

  it('returns false for non-plan paths and non-strings', () => {
    expect(isPlanPath('plans/foo.md')).toBe(false);
    expect(isPlanPath(null)).toBe(false);
    expect(isPlanPath(undefined)).toBe(false);
    expect(isPlanPath(123)).toBe(false);
  });
});

describe('getDefaultDeps', () => {
  function buildImportMap(overrides = {}) {
    return {
      cortexDb: jest.fn().mockResolvedValue({
        getTursoClient: jest.fn().mockResolvedValue({ execute: jest.fn() }),
        closeTursoClient: jest.fn().mockResolvedValue(undefined),
      }),
      validateIndex: jest.fn().mockResolvedValue({
        validateDatabase: jest.fn().mockResolvedValue({ pass: true }),
      }),
      cortexMcpSmoke: jest.fn().mockResolvedValue({
        runCortexMcpSmoke: jest.fn().mockResolvedValue({ pass: true }),
      }),
      cortexTierTool: jest.fn().mockResolvedValue({
        rebuildIndex: jest.fn().mockResolvedValue({ success: true }),
      }),
      initSchema: jest.fn().mockResolvedValue({
        defaultDatabasePath: '/canonical/db.sqlite',
        repoRoot: '/repo',
      }),
      autoReindex: jest.fn().mockResolvedValue({
        resolveStalePlanFixHint: jest.fn().mockReturnValue('stale plan hint'),
      }),
      ...overrides,
    };
  }

  it('loads heavy modules through the import map and wires dependencies', async () => {
    const imports = buildImportMap();

    const deps = await getDefaultDeps({}, imports);

    expect(imports.cortexDb).toHaveBeenCalled();
    expect(imports.validateIndex).toHaveBeenCalled();
    expect(imports.cortexMcpSmoke).toHaveBeenCalled();
    expect(imports.cortexTierTool).toHaveBeenCalled();
    expect(imports.initSchema).toHaveBeenCalled();
    expect(imports.autoReindex).toHaveBeenCalled();

    expect(deps.databasePath).toContain('canonical');
    expect(deps.snapshotMaxAgeMs).toBe(86_400_000);
    expect(deps.timeoutMs).toBe(60_000);
    expect(typeof deps.indexValidator).toBe('function');
    expect(typeof deps.mcpSmoke).toBe('function');
    expect(typeof deps.workflowMcpCheck).toBe('function');
    expect(typeof deps.snapshotCurrency).toBe('function');
    expect(typeof deps.rebuildIndex).toBe('function');
    expect(typeof deps.resolveFixHint).toBe('function');
  });

  it('lets options override default paths and thresholds', async () => {
    const imports = buildImportMap();

    const deps = await getDefaultDeps(
      {
        databasePath: '/override/db.sqlite',
        snapshotPath: '/override/snapshot.json',
        snapshotMaxAgeMs: 1_000,
        timeoutMs: 5_000,
        workflowPlanPath: 'plans/override.plans.md',
      },
      imports,
    );

    expect(deps.databasePath).toContain('override');
    expect(deps.snapshotPath).toContain('override');
    expect(deps.snapshotMaxAgeMs).toBe(1_000);
    expect(deps.timeoutMs).toBe(5_000);
    expect(deps.workflowPlanPath).toBe('plans/override.plans.md');
  });

  it('binds resolveStalePlanFixHint into the returned resolveFixHint', async () => {
    const imports = buildImportMap();

    const deps = await getDefaultDeps({}, imports);
    const hint = deps.resolveFixHint({
      indexReport: {
        pass: false,
        stale_paths: ['x.plans.md'],
        missing_paths: [],
      },
      snapshotCurrency: { pass: true },
      corpusMcpReport: { pass: true },
      workflowMcpReport: { pass: true },
    });

    expect(hint).toBe('stale plan hint');
  });

  it('uses default dynamic imports when no import map is supplied', async () => {
    jest.unstable_mockModule('../../mcp-semantic/tools/cortex-db.mjs', () => ({
      getTursoClient: jest.fn().mockResolvedValue({
        execute: jest
          .fn()
          .mockResolvedValue({ rows: [{ documents: 3, indexed_at: 1 }] }),
      }),
      closeTursoClient: jest.fn().mockResolvedValue(undefined),
    }));
    jest.unstable_mockModule('../../../rag-index/validate-index.mjs', () => ({
      validateDatabase: jest.fn().mockResolvedValue({ pass: true }),
    }));
    jest.unstable_mockModule('./cortex-mcp-smoke.mjs', () => ({
      runCortexMcpSmoke: jest.fn().mockResolvedValue({ pass: true }),
    }));
    jest.unstable_mockModule('../mcp/cortex-tier-tool.mjs', () => ({
      rebuildIndex: jest.fn().mockResolvedValue({ success: true }),
    }));
    jest.unstable_mockModule('../../../rag-index/init-schema.mjs', () => ({
      defaultDatabasePath: '/canonical/db.sqlite',
      repoRoot: '/repo',
    }));
    jest.unstable_mockModule('../../../rag-index/auto-reindex.mjs', () => ({
      resolveStalePlanFixHint: jest.fn().mockReturnValue('stale default hint'),
    }));
    jest.resetModules();

    const { getDefaultDeps: getDefaultDepsFresh } =
      await import('./cortex-index.gate.runtime.mjs');

    const deps = await getDefaultDepsFresh({
      databasePath: '/test/db.sqlite',
      snapshotPath: '/test/snapshot.json',
      snapshotMaxAgeMs: 1_000,
      timeoutMs: 2_000,
      workflowPlanPath: 'plans/test.plans.md',
    });

    const indexReport = await deps.indexValidator({
      databasePath: '/test/db.sqlite',
    });
    const smoke = await deps.mcpSmoke({ databasePath: '/test/db.sqlite' });
    const workflow = await deps.workflowMcpCheck({
      timeoutMs: 2_000,
      workflowPlanPath: 'plans/test.plans.md',
    });
    const currency = await deps.snapshotCurrency({
      databasePath: '/test/db.sqlite',
      snapshotPath: '/test/snapshot.json',
      snapshotMaxAgeMs: 1_000,
    });
    const rebuild = await deps.rebuildIndex({
      databasePath: '/test/db.sqlite',
    });
    const hint = deps.resolveFixHint({
      indexReport: { pass: false, stale_paths: ['x.plans.md'] },
      snapshotCurrency: { pass: true },
      corpusMcpReport: { pass: true },
      workflowMcpReport: { pass: true },
    });

    expect(indexReport.pass).toBe(true);
    expect(smoke.pass).toBe(true);
    expect(typeof workflow).toBe('object');
    expect(typeof currency).toBe('object');
    expect(rebuild.success).toBe(true);
    expect(hint).toBe('stale default hint');
  });

  it('uses default options when called with no arguments', async () => {
    jest.unstable_mockModule('../../mcp-semantic/tools/cortex-db.mjs', () => ({
      getTursoClient: jest.fn().mockResolvedValue({
        execute: jest
          .fn()
          .mockResolvedValue({ rows: [{ documents: 3, indexed_at: 1 }] }),
      }),
      closeTursoClient: jest.fn().mockResolvedValue(undefined),
    }));
    jest.unstable_mockModule('../../../rag-index/validate-index.mjs', () => ({
      validateDatabase: jest.fn().mockResolvedValue({ pass: true }),
    }));
    jest.unstable_mockModule('./cortex-mcp-smoke.mjs', () => ({
      runCortexMcpSmoke: jest.fn().mockResolvedValue({ pass: true }),
    }));
    jest.unstable_mockModule('../mcp/cortex-tier-tool.mjs', () => ({
      rebuildIndex: jest.fn().mockResolvedValue({ success: true }),
    }));
    jest.unstable_mockModule('../../../rag-index/init-schema.mjs', () => ({
      defaultDatabasePath: '/canonical/db.sqlite',
      repoRoot: '/repo',
    }));
    jest.unstable_mockModule('../../../rag-index/auto-reindex.mjs', () => ({
      resolveStalePlanFixHint: jest.fn().mockReturnValue('stale default hint'),
    }));
    jest.resetModules();

    const { getDefaultDeps: getDefaultDepsFresh } =
      await import('./cortex-index.gate.runtime.mjs');

    const deps = await getDefaultDepsFresh();

    expect(deps).toMatchObject({
      snapshotCurrency: expect.any(Function),
      workflowMcpCheck: expect.any(Function),
      rebuildIndex: expect.any(Function),
    });
    expect(typeof deps.resolveFixHint).toBe('function');
  });
});
