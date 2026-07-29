import { jest } from '@jest/globals';
import { runCortexIndexGate, main } from './cortex-index.gate.mjs';

function makeDeps(overrides = {}) {
  return {
    databasePath: 'test.sqlite',
    snapshotPath: 'snapshot.json',
    snapshotMaxAgeMs: 86_400_000,
    timeoutMs: 60_000,
    workflowPlanPath: 'plans/test.plans.md',
    indexValidator: jest.fn().mockResolvedValue({ pass: true, documents: 7 }),
    mcpSmoke: jest.fn().mockResolvedValue({ pass: true }),
    workflowMcpCheck: jest.fn().mockResolvedValue({ pass: true }),
    snapshotCurrency: jest.fn().mockResolvedValue({
      pass: true,
      indexDocuments: 7,
      snapshotAgeSeconds: 10,
      snapshotIndexedAt: '2024-01-01T00:00:00.000Z',
    }),
    rebuildIndex: jest.fn().mockResolvedValue({ success: true }),
    resolveFixHint: jest.fn().mockReturnValue('mock fix hint'),
    ...overrides,
  };
}

describe('runCortexIndexGate', () => {
  let deps;

  beforeEach(() => {
    deps = makeDeps();
  });

  it('returns pass true when all sub-reports pass', async () => {
    const result = await runCortexIndexGate({}, deps);

    expect(result.pass).toBe(true);
    expect(result.evidence.index_documents).toBe(7);
    expect(result.evidence.index_fresh).toBe(true);
    expect(result.evidence.corpus_mcp_alive).toBe(true);
    expect(result.evidence.workflow_mcp_alive).toBe(true);
    expect(result.evidence.auto_rebuild_attempted).toBe(false);
    expect(result.fixHint).toBeNull();
    expect(deps.indexValidator).toHaveBeenCalledWith({
      databasePath: expect.stringContaining('test.sqlite'),
    });
    expect(deps.mcpSmoke).toHaveBeenCalledWith({
      databasePath: expect.stringContaining('test.sqlite'),
    });
    expect(deps.workflowMcpCheck).toHaveBeenCalledWith({
      timeoutMs: 60_000,
      workflowPlanPath: 'plans/test.plans.md',
    });
    expect(deps.snapshotCurrency).toHaveBeenCalledWith({
      databasePath: expect.stringContaining('test.sqlite'),
      snapshotPath: expect.stringContaining('snapshot.json'),
      snapshotMaxAgeMs: 86_400_000,
    });
  });

  it('returns pass false and a fixHint when the index is not fresh', async () => {
    deps.indexValidator.mockResolvedValue({
      pass: false,
      stale_paths: ['foo.plans.md'],
    });

    const result = await runCortexIndexGate({}, deps);

    expect(result.pass).toBe(false);
    expect(result.evidence.index_fresh).toBe(false);
    expect(result.fixHint).toBe('mock fix hint');
    expect(deps.resolveFixHint).toHaveBeenCalledWith(
      expect.objectContaining({
        indexReport: expect.objectContaining({ pass: false }),
        snapshotCurrency: expect.objectContaining({ pass: true }),
        corpusMcpReport: expect.objectContaining({ pass: true }),
        workflowMcpReport: expect.objectContaining({ pass: true }),
      }),
    );
  });

  it('auto-rebuilds a stale index and re-validates on success', async () => {
    deps.indexValidator
      .mockResolvedValueOnce({ pass: false, stale_paths: ['foo.plans.md'] })
      .mockResolvedValueOnce({ pass: true, documents: 9 });
    deps.rebuildIndex.mockResolvedValue({ success: true });

    const result = await runCortexIndexGate({ autoRebuild: true }, deps);

    expect(result.pass).toBe(true);
    expect(result.evidence.index_documents).toBe(9);
    expect(result.evidence.auto_rebuild_attempted).toBe(true);
    expect(result.evidence.auto_rebuild_success).toBe(true);
    expect(result.evidence.auto_rebuild_error).toBeNull();
    expect(deps.rebuildIndex).toHaveBeenCalledWith({
      databasePath: expect.stringContaining('test.sqlite'),
    });
    expect(deps.indexValidator).toHaveBeenCalledTimes(2);
  });

  it('reports an auto-rebuild failure and prefixes the fixHint', async () => {
    deps.indexValidator.mockResolvedValue({
      pass: false,
      stale_paths: ['a.plans.md'],
    });
    deps.rebuildIndex.mockResolvedValue({
      success: false,
      error: 'disk full',
    });

    const result = await runCortexIndexGate({ autoRebuild: true }, deps);

    expect(result.evidence.auto_rebuild_attempted).toBe(true);
    expect(result.evidence.auto_rebuild_success).toBe(false);
    expect(result.evidence.auto_rebuild_error).toBe('disk full');
    expect(result.fixHint).toBe(
      'Auto-rebuild failed: disk full. mock fix hint',
    );
  });

  it('does not auto-rebuild when the index is already fresh', async () => {
    const result = await runCortexIndexGate({ autoRebuild: true }, deps);

    expect(result.pass).toBe(true);
    expect(result.evidence.auto_rebuild_attempted).toBe(false);
    expect(deps.rebuildIndex).not.toHaveBeenCalled();
  });

  it('uses options to override dep defaults', async () => {
    await runCortexIndexGate(
      {
        databasePath: 'override.db',
        snapshotPath: 'override.json',
        snapshotMaxAgeMs: 1_000,
        timeoutMs: 5_000,
        workflowPlanPath: 'other.plans.md',
      },
      deps,
    );

    expect(deps.indexValidator).toHaveBeenCalledWith({
      databasePath: expect.stringContaining('override.db'),
    });
    expect(deps.snapshotCurrency).toHaveBeenCalledWith({
      databasePath: expect.stringContaining('override.db'),
      snapshotPath: expect.stringContaining('override.json'),
      snapshotMaxAgeMs: 1_000,
    });
    expect(deps.workflowMcpCheck).toHaveBeenCalledWith({
      timeoutMs: 5_000,
      workflowPlanPath: 'other.plans.md',
    });
  });

  it('uses option-level injected functions over deps', async () => {
    const optIndexValidator = jest
      .fn()
      .mockResolvedValue({ pass: true, documents: 11 });

    const result = await runCortexIndexGate(
      { indexValidator: optIndexValidator },
      deps,
    );

    expect(result.evidence.index_documents).toBe(11);
    expect(optIndexValidator).toHaveBeenCalledWith({
      databasePath: expect.stringContaining('test.sqlite'),
    });
    expect(deps.indexValidator).not.toHaveBeenCalled();
  });

  it('uses option-level resolveFixHint over deps', async () => {
    deps.indexValidator.mockResolvedValue({
      pass: false,
      stale_paths: ['x.plans.md'],
    });
    const customResolveFixHint = jest.fn().mockReturnValue('custom fix hint');

    const result = await runCortexIndexGate(
      { resolveFixHint: customResolveFixHint },
      deps,
    );

    expect(result.fixHint).toBe('custom fix hint');
    expect(customResolveFixHint).toHaveBeenCalled();
    expect(deps.resolveFixHint).not.toHaveBeenCalled();
  });

  it('uses default options and deps when called with no arguments', async () => {
    jest.unstable_mockModule('./cortex-index.gate.runtime.mjs', () => ({
      getDefaultDeps: jest.fn().mockResolvedValue(deps),
    }));
    jest.resetModules();

    const { runCortexIndexGate: freshGate } =
      await import('./cortex-index.gate.mjs');

    const result = await freshGate();

    expect(result.pass).toBe(true);
    expect(result.evidence.index_documents).toBe(7);
  });

  it('uses fallback values in evidence when reports omit optional fields', async () => {
    deps.indexValidator.mockResolvedValue({ pass: true });
    deps.snapshotCurrency.mockResolvedValue({
      pass: true,
      snapshotIndexedAt: '2024-02-01T00:00:00.000Z',
    });

    const result = await runCortexIndexGate({}, deps);

    expect(result.evidence.index_documents).toBe(0);
    expect(result.evidence.snapshot_age_seconds).toBe(0);
    expect(result.evidence.snapshot_indexed_at).toBe(
      '2024-02-01T00:00:00.000Z',
    );
  });

  it('reports auto-rebuild error without trailing fixHint when none is supplied', async () => {
    deps.indexValidator.mockResolvedValue({
      pass: false,
      stale_paths: ['a.plans.md'],
    });
    deps.rebuildIndex.mockResolvedValue({
      success: false,
      error: 'disk full',
    });
    deps.resolveFixHint.mockReturnValue(null);

    const result = await runCortexIndexGate({ autoRebuild: true }, deps);

    expect(result.fixHint).toBe('Auto-rebuild failed: disk full');
  });
});

describe('main CLI', () => {
  let logSpy;
  let deps;

  beforeEach(() => {
    logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    deps = makeDeps();
  });

  afterEach(() => {
    logSpy.mockRestore();
    delete process.exitCode;
  });

  it('prints usage and exits 0 for --help', async () => {
    const result = await main(['--help'], deps);

    expect(result).toBeUndefined();
    expect(process.exitCode).toBe(0);
    expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
  });

  it('emits JSON report and sets exit code 0 on pass', async () => {
    await main(['--json'], deps);

    expect(process.exitCode).toBe(0);
    const logged = JSON.parse(logSpy.mock.calls[0][0]);
    expect(logged.pass).toBe(true);
    expect(logged.owner).toBe('00-helping');
    expect(logged.schema_version).toBe(1);
  });

  it('emits human-readable failure and sets exit code 1', async () => {
    deps.indexValidator.mockResolvedValue({
      pass: false,
      stale_paths: ['bad.plans.md'],
    });

    await main([], deps);

    expect(process.exitCode).toBe(1);
    expect(logSpy).toHaveBeenCalledWith('FAIL cortex-index gate');
    expect(logSpy).toHaveBeenCalledWith('fixHint: mock fix hint');
  });

  it('passes auto-rebuild and override flags to runCortexIndexGate', async () => {
    deps.indexValidator
      .mockResolvedValueOnce({ pass: false, stale_paths: ['p.plans.md'] })
      .mockResolvedValueOnce({ pass: true });

    await main(
      [
        '--auto-rebuild',
        '--databasePath=other.db',
        '--plan=other.md',
        '--snapshot-max-age-ms=1234',
      ],
      deps,
    );

    expect(deps.rebuildIndex).toHaveBeenCalledWith({
      databasePath: expect.stringContaining('other.db'),
    });
    expect(deps.snapshotCurrency).toHaveBeenCalledWith(
      expect.objectContaining({ snapshotMaxAgeMs: 1234 }),
    );
  });

  it('uses default deps in main when none are supplied', async () => {
    jest.unstable_mockModule('./cortex-index.gate.runtime.mjs', () => ({
      getDefaultDeps: jest.fn().mockResolvedValue(deps),
    }));
    jest.resetModules();

    const { main: freshMain } = await import('./cortex-index.gate.mjs');

    await freshMain(['--json']);

    const logged = JSON.parse(logSpy.mock.calls[0][0]);
    expect(logged.pass).toBe(true);
    expect(process.exitCode).toBe(0);
  });

  it('runs with process.argv defaults and no injected deps', async () => {
    jest.unstable_mockModule('./cortex-index.gate.runtime.mjs', () => ({
      getDefaultDeps: jest.fn().mockResolvedValue(deps),
    }));
    jest.resetModules();

    const { main: freshMain } = await import('./cortex-index.gate.mjs');

    await freshMain();

    expect(process.exitCode).toBe(0);
    expect(logSpy).toHaveBeenCalledWith(
      expect.stringContaining('PASS cortex-index gate'),
    );
  });
});
