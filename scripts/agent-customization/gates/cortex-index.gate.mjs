#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  closeTursoClient,
  getTursoClient,
} from '../../mcp-semantic/tools/cortex-db.mjs';
import {
  defaultDatabasePath,
  repoRoot,
} from '../../semantic-index/init-schema.mjs';
import { validateDatabase } from '../../semantic-index/validate-index.mjs';
import { runCortexMcpSmoke } from './cortex-mcp-smoke.mjs';

const OWNER = '00-helping';
const DEFAULT_SNAPSHOT_MAX_AGE_MS = 24 * 60 * 60 * 1000;
const DEFAULT_TIMEOUT_MS = 10_000;
const DEFAULT_SNAPSHOT_PATH = path.join(
  repoRoot,
  'docs',
  'assets',
  'semantic-snapshot.json',
);
const DEFAULT_WORKFLOW_PLAN_PATH = 'plans/mcp-active-binding.plans.md';
const WORKFLOW_MCP_PATH = path.join(
  repoRoot,
  'scripts',
  'agent-customization',
  'mcp',
  'neataptic-workflow-mcp.mjs',
);
const WORKFLOW_MCP_CONFIG_PATH = path.join(repoRoot, '.vscode', 'mcp.json');

export async function runCortexIndexGate(options = {}) {
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

  const indexReport = await validateDatabase({ databasePath });
  const corpusMcpReport = await runCortexMcpSmoke({ databasePath });
  const workflowMcpReport = runWorkflowMcpSelfCheck({
    timeoutMs,
    workflowPlanPath,
  });
  const snapshotCurrency = await readSnapshotCurrency({
    databasePath,
    snapshotPath,
    snapshotMaxAgeMs,
  });

  const evidence = {
    index_documents: Number(
      indexReport.documents ?? snapshotCurrency.indexDocuments ?? 0,
    ),
    index_fresh: Boolean(indexReport.pass),
    corpus_mcp_alive: Boolean(corpusMcpReport.pass),
    workflow_mcp_alive: Boolean(workflowMcpReport.pass),
    snapshot_age_seconds: Number(snapshotCurrency.snapshotAgeSeconds ?? 0),
    snapshot_indexed_at: snapshotCurrency.snapshotIndexedAt,
  };
  const pass =
    evidence.index_fresh &&
    evidence.corpus_mcp_alive &&
    evidence.workflow_mcp_alive &&
    snapshotCurrency.pass;

  return {
    schema_version: 1,
    pass,
    evidence,
    fixHint: pass
      ? null
      : resolveFixHint({
          indexReport,
          snapshotCurrency,
          corpusMcpReport,
          workflowMcpReport,
        }),
    owner: OWNER,
  };
}

function parseArgs(argv) {
  return {
    help: argv.includes('--help') || argv.includes('-h'),
    json: argv.includes('--json'),
    databasePath: argv
      .find((argument) => argument.startsWith('--databasePath='))
      ?.slice('--databasePath='.length),
    workflowPlanPath: argv
      .find((argument) => argument.startsWith('--plan='))
      ?.slice('--plan='.length),
    snapshotMaxAgeMs: argv
      .find((argument) => argument.startsWith('--snapshot-max-age-ms='))
      ?.slice('--snapshot-max-age-ms='.length),
  };
}

function printUsage() {
  console.log(
    [
      'Cortex lifecycle gate',
      '',
      'Usage:',
      '  node scripts/agent-customization/gates/cortex-index.gate.mjs [--json] [--plan=<path>] [--databasePath=<path>]',
      '  node scripts/agent-customization/gates/cortex-index.gate.mjs --help',
      '',
      'Options:',
      '  --json                     Emit machine-readable gate JSON.',
      '  --plan=<path>              Optional workflow MCP self-check plan path.',
      '  --databasePath=<path>      Override the semantic index database path.',
      '  --snapshot-max-age-ms=<n>  Override the snapshot freshness threshold (default: 86400000).',
    ].join('\n'),
  );
}

async function readSnapshotCurrency({
  databasePath,
  snapshotPath,
  snapshotMaxAgeMs,
}) {
  const fallbackResult = {
    pass: false,
    indexDocuments: 0,
    snapshotAgeSeconds: 0,
    snapshotIndexedAt: null,
  };

  if (!existsSync(snapshotPath)) {
    return fallbackResult;
  }

  let snapshotGeneratedAtMs;

  try {
    const snapshotPayload = JSON.parse(await readFile(snapshotPath, 'utf8'));
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

function runWorkflowMcpSelfCheck({ timeoutMs, workflowPlanPath }) {
  const spawned = spawnSync(
    process.execPath,
    [WORKFLOW_MCP_PATH, `--plan=${workflowPlanPath}`, '--self-check', '--json'],
    {
      cwd: repoRoot,
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

  const report = parseJson(spawned.stdout);
  return {
    pass: spawned.status === 0 && report?.ok === true,
    report,
  };
}

async function resolveWorkflowPlanPath() {
  try {
    const configPayload = JSON.parse(
      await readFile(WORKFLOW_MCP_CONFIG_PATH, 'utf8'),
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

function resolveFixHint({
  indexReport,
  snapshotCurrency,
  corpusMcpReport,
  workflowMcpReport,
}) {
  if (!indexReport.pass) {
    return 'Run: node scripts/semantic-index/build-index.mjs to rebuild stale index';
  }

  if (!snapshotCurrency.pass) {
    return 'Run: npm run docs to regenerate snapshot';
  }

  if (!corpusMcpReport.pass) {
    return 'Check neataptic-cortex-mcp server in .vscode/mcp.json; run cortex-mcp-smoke.mjs for details';
  }

  if (!workflowMcpReport.pass) {
    return 'Restart neataptic-workflow-mcp server to bind to active plan path';
  }

  return null;
}

function parseJson(value) {
  if (typeof value !== 'string' || !value.trim()) {
    return null;
  }

  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  const report = await runCortexIndexGate(options);
  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(
      report.pass ? 'PASS cortex-index gate' : 'FAIL cortex-index gate',
    );
    if (!report.pass) console.log(`fixHint: ${report.fixHint}`);
  }

  process.exitCode = report.pass ? 0 : 1;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  await main();
}
