/**
 * @module cortex-index.gate.runtime
 * @description Production runtime defaults for the cortex-index gate.
 *
 * Heavy MCP / database modules are imported lazily inside
 * {@link getDefaultDeps} so that the file itself is safe to load in unit tests
 * without pulling in DB drivers or MCP servers. Helper functions accept
 * low-level primitives as parameters to stay testable with plain mocks.
 */
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
  '..',
);

const DEFAULT_SNAPSHOT_MAX_AGE_MS = 24 * 60 * 60 * 1000;
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_SNAPSHOT_PATH = path.join(
  REPO_ROOT,
  'rag-index',
  'snapshots',
  'semantic-snapshot.json',
);
const DEFAULT_WORKFLOW_PLAN_PATH = 'plans/mcp-active-binding.plans.md';
const WORKFLOW_MCP_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'mcp',
  'neataptic-workflow-mcp.mjs',
);
const WORKFLOW_MCP_CONFIG_PATH = path.join(REPO_ROOT, '.vscode', 'mcp.json');

/**
 * Read the semantic snapshot and compare its generated_at timestamp with the
 * maximum indexed_at value in the corpus database.
 *
 * @param {{ databasePath: string; snapshotPath: string; snapshotMaxAgeMs: number }} options
 * @param {{ existsSync?: typeof import('node:fs').existsSync; readFile?: typeof import('node:fs/promises').readFile; getTursoClient?: Function; closeTursoClient?: Function }} primitives
 * @returns {Promise<{ pass: boolean; indexDocuments: number; snapshotAgeSeconds: number; snapshotIndexedAt: string | null }>}
 */
export async function readSnapshotCurrency(
  { databasePath, snapshotPath, snapshotMaxAgeMs },
  primitives,
) {
  const exists = primitives.existsSync ?? existsSync;
  const read = primitives.readFile ?? readFile;
  const getTursoClient = primitives.getTursoClient;
  const closeTursoClient = primitives.closeTursoClient;

  const fallbackResult = {
    pass: false,
    indexDocuments: 0,
    snapshotAgeSeconds: 0,
    snapshotIndexedAt: null,
  };

  if (!exists(snapshotPath)) {
    return fallbackResult;
  }

  let snapshotGeneratedAtMs;

  try {
    const snapshotPayload = JSON.parse(await read(snapshotPath, 'utf8'));
    snapshotGeneratedAtMs = Date.parse(snapshotPayload.generated_at);
  } catch {
    return fallbackResult;
  }

  if (!Number.isFinite(snapshotGeneratedAtMs)) {
    return fallbackResult;
  }

  const client = await getTursoClient(databasePath);

  try {
    const snapshotResult = await client.execute({
      sql: 'SELECT COUNT(*) AS documents, MAX(indexed_at) AS indexed_at FROM documents',
      args: [],
    });
    const snapshotRow = snapshotResult.rows[0] ?? {};
    const indexedAtMs = Number(snapshotRow.indexed_at ?? 0);
    const snapshotAgeMs = Math.max(0, indexedAtMs - snapshotGeneratedAtMs);

    return {
      pass: indexedAtMs > 0 && snapshotAgeMs <= snapshotMaxAgeMs,
      indexDocuments: Number(snapshotRow.documents ?? 0),
      snapshotAgeSeconds: Math.trunc(snapshotAgeMs / 1000),
      snapshotIndexedAt:
        indexedAtMs > 0 ? new Date(indexedAtMs).toISOString() : null,
    };
  } finally {
    await closeTursoClient(databasePath);
  }
}

/**
 * Run the workflow MCP self-check as a subprocess and return a pass/fail report.
 *
 * @param {{ timeoutMs: number; workflowPlanPath: string }} options
 * @param {{ spawnSync?: typeof import('node:child_process').spawnSync }} primitives
 * @returns {{ pass: boolean; report: unknown }}
 */
export function runWorkflowMcpSelfCheck(
  { timeoutMs, workflowPlanPath },
  primitives = {},
) {
  const spawn = primitives.spawnSync ?? spawnSync;
  const spawned = spawn(
    process.execPath,
    [WORKFLOW_MCP_PATH, `--plan=${workflowPlanPath}`, '--self-check', '--json'],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: timeoutMs,
    },
  );

  if (spawned.error) {
    return {
      pass: false,
      report: null,
    };
  }

  let report = null;
  try {
    report = JSON.parse(spawned.stdout);
  } catch {
    report = null;
  }

  return {
    pass: spawned.status === 0 && report?.ok === true,
    report,
  };
}

/**
 * Resolve the workflow plan path from `.vscode/mcp.json` when possible.
 *
 * @param {{ readFile?: typeof import('node:fs/promises').readFile }} primitives
 * @returns {Promise<string>}
 */
export async function resolveWorkflowPlanPath(primitives = {}) {
  const read = primitives.readFile ?? readFile;
  try {
    const configPayload = JSON.parse(
      await read(WORKFLOW_MCP_CONFIG_PATH, 'utf8'),
    );
    const workflowArgs =
      configPayload?.servers?.['neataptic-workflow-mcp']?.args;
    const planArgument = Array.isArray(workflowArgs)
      ? workflowArgs.find(
          (argument) =>
            typeof argument === 'string' && argument.startsWith('--plan='),
        )
      : null;

    return typeof planArgument === 'string'
      ? planArgument.slice('--plan='.length)
      : DEFAULT_WORKFLOW_PLAN_PATH;
  } catch {
    return DEFAULT_WORKFLOW_PLAN_PATH;
  }
}

/**
 * Determine whether a path looks like a plan file.
 *
 * @param {unknown} filePath
 * @returns {boolean}
 */
export function isPlanPath(filePath) {
  return typeof filePath === 'string' && filePath.endsWith('.plans.md');
}

/**
 * Build a human-readable fix hint from the gate sub-reports.
 *
 * @param {{ indexReport: any; snapshotCurrency: any; corpusMcpReport: any; workflowMcpReport: any }} reports
 * @param {{ resolveStalePlanFixHint?: (stalePaths: string[]) => string }} helpers
 * @returns {string | null}
 */
export function resolveFixHint(
  { indexReport, snapshotCurrency, corpusMcpReport, workflowMcpReport },
  { resolveStalePlanFixHint } = {},
) {
  if (!indexReport.pass) {
    const stalePaths = indexReport.stale_paths ?? [];
    const missingPaths = indexReport.missing_paths ?? [];
    const overAgePaths = indexReport.over_age_paths ?? [];

    if (
      stalePaths.length > 0 &&
      missingPaths.length === 0 &&
      overAgePaths.length === 0 &&
      stalePaths.every(isPlanPath) &&
      typeof resolveStalePlanFixHint === 'function'
    ) {
      return resolveStalePlanFixHint(stalePaths);
    }

    return 'Run: node rag-index/build-index.mjs to rebuild stale index';
  }

  if (!snapshotCurrency.pass) {
    return 'Run: npm run index:build-snapshot to regenerate snapshot';
  }

  if (!corpusMcpReport.pass) {
    return 'Check cortex server in .vscode/mcp.json; run cortex-mcp-smoke.mjs for details';
  }

  if (!workflowMcpReport.pass) {
    return 'Restart neataptic-workflow-mcp server to bind to active plan path';
  }

  return null;
}

/**
 * Build the default dependency set used by the cortex-index gate in production.
 *
 * Heavy modules are loaded via dynamic import so that callers (and tests) that
 * inject their own `deps` never pay the cost or side effects. The optional
 * `imports` map makes the loading itself testable without touching real modules.
 *
 * @param {Record<string, unknown>} options
 * @param {Record<string, () => Promise<Record<string, unknown>>>} imports
 * @returns {Promise<Record<string, unknown>>}
 */
export async function getDefaultDeps(options = {}, imports = {}) {
  const importCortexDb =
    imports.cortexDb ??
    (() => import('../../mcp-semantic/tools/cortex-db.mjs'));
  const importValidateIndex =
    imports.validateIndex ??
    (() => import('../../../rag-index/validate-index.mjs'));
  const importCortexMcpSmoke =
    imports.cortexMcpSmoke ?? (() => import('./cortex-mcp-smoke.mjs'));
  const importCortexTierTool =
    imports.cortexTierTool ?? (() => import('../mcp/cortex-tier-tool.mjs'));
  const importInitSchema =
    imports.initSchema ?? (() => import('../../../rag-index/init-schema.mjs'));
  const importAutoReindex =
    imports.autoReindex ??
    (() => import('../../../rag-index/auto-reindex.mjs'));

  const [
    { getTursoClient, closeTursoClient },
    { validateDatabase },
    { runCortexMcpSmoke },
    { rebuildIndex },
    { defaultDatabasePath },
    { resolveStalePlanFixHint },
  ] = await Promise.all([
    importCortexDb(),
    importValidateIndex(),
    importCortexMcpSmoke(),
    importCortexTierTool(),
    importInitSchema(),
    importAutoReindex(),
  ]);

  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  const snapshotPath = path.resolve(
    options.snapshotPath ?? DEFAULT_SNAPSHOT_PATH,
  );
  const snapshotMaxAgeMs = Number(
    options.snapshotMaxAgeMs ?? DEFAULT_SNAPSHOT_MAX_AGE_MS,
  );
  const timeoutMs = Number(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const workflowPlanPath =
    options.workflowPlanPath ?? (await resolveWorkflowPlanPath());

  return {
    databasePath,
    snapshotPath,
    snapshotMaxAgeMs,
    timeoutMs,
    workflowPlanPath,
    indexValidator: async ({ databasePath: dbPath }) =>
      validateDatabase({ databasePath: dbPath }),
    mcpSmoke: async ({ databasePath: dbPath }) =>
      runCortexMcpSmoke({ databasePath: dbPath }),
    workflowMcpCheck: async ({ timeoutMs: tMs, workflowPlanPath: planPath }) =>
      runWorkflowMcpSelfCheck({ timeoutMs: tMs, workflowPlanPath: planPath }),
    snapshotCurrency: async (snapshotOptions) =>
      readSnapshotCurrency(snapshotOptions, {
        getTursoClient,
        closeTursoClient,
      }),
    rebuildIndex,
    resolveFixHint: (reports) =>
      resolveFixHint(reports, { resolveStalePlanFixHint }),
  };
}
