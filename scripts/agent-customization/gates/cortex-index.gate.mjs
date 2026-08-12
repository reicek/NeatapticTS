#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseArgs } from '../customization-utils.mjs';
import { getDefaultDeps } from './cortex-index.gate.runtime.mjs';

const OWNER = '00-helping';
const DEFAULT_SNAPSHOT_MAX_AGE_MS = 24 * 60 * 60 * 1000;
const DEFAULT_TIMEOUT_MS = 60_000;

export async function runCortexIndexGate(options = {}, deps = null) {
  const d = deps ?? (await getDefaultDeps(options));
  const databasePath = path.resolve(options.databasePath ?? d.databasePath);
  const snapshotPath = path.resolve(options.snapshotPath ?? d.snapshotPath);
  const snapshotMaxAgeMs = Number(
    options.snapshotMaxAgeMs ?? d.snapshotMaxAgeMs,
  );
  const timeoutMs = Number(options.timeoutMs ?? d.timeoutMs);
  const workflowPlanPath = options.workflowPlanPath ?? d.workflowPlanPath;
  const autoRebuild = Boolean(options.autoRebuild);

  const indexValidator = options.indexValidator ?? d.indexValidator;
  const mcpSmoke = options.mcpSmoke ?? d.mcpSmoke;
  const workflowMcpCheck = options.workflowMcpCheck ?? d.workflowMcpCheck;
  const snapshotCurrencyFn = options.snapshotCurrency ?? d.snapshotCurrency;
  const rebuildIndex = options.rebuildIndex ?? d.rebuildIndex;
  const resolveFixHint = options.resolveFixHint ?? d.resolveFixHint;

  let indexReport = await indexValidator({ databasePath });

  let autoRebuildAttempted = false;
  let autoRebuildSuccess = false;
  let autoRebuildError = null;

  if (autoRebuild && !indexReport.pass) {
    autoRebuildAttempted = true;
    const rebuildResult = await rebuildIndex({ databasePath });
    autoRebuildSuccess = Boolean(rebuildResult?.success);
    autoRebuildError = rebuildResult?.error ?? null;
    if (autoRebuildSuccess) {
      indexReport = await indexValidator({ databasePath });
    }
  }

  const corpusMcpReport = await mcpSmoke({ databasePath });
  const workflowMcpReport = await workflowMcpCheck({
    timeoutMs,
    workflowPlanPath,
  });
  const snapshotCurrency = await snapshotCurrencyFn({
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
    auto_rebuild_attempted: autoRebuildAttempted,
    auto_rebuild_success: autoRebuildSuccess,
    auto_rebuild_error: autoRebuildError,
  };
  const pass =
    evidence.index_fresh &&
    evidence.corpus_mcp_alive &&
    evidence.workflow_mcp_alive &&
    snapshotCurrency.pass;

  let fixHint = pass
    ? null
    : resolveFixHint({
        indexReport,
        snapshotCurrency,
        corpusMcpReport,
        workflowMcpReport,
      });

  if (!pass && autoRebuildError) {
    fixHint = `Auto-rebuild failed: ${autoRebuildError}${fixHint ? `. ${fixHint}` : ''}`;
  }

  return {
    schema_version: 1,
    pass,
    evidence,
    fixHint,
    owner: OWNER,
  };
}

function printUsage() {
  console.log(
    [
      'Cortex lifecycle gate',
      '',
      'Usage:',
      '  node scripts/agent-customization/gates/cortex-index.gate.mjs [--json] [--auto-rebuild] [--plan=<path>] [--databasePath=<path>]',
      '  node scripts/agent-customization/gates/cortex-index.gate.mjs --help',
      '',
      'Options:',
      '  --json                     Emit machine-readable gate JSON.',
      '  --auto-rebuild             Rebuild a stale semantic index and re-validate before reporting.',
      '  --plan=<path>              Optional workflow MCP self-check plan path.',
      '  --databasePath=<path>      Override the semantic index database path.',
      '  --snapshot-max-age-ms=<n>  Override the snapshot freshness threshold (default: 86400000).',
    ].join('\n'),
  );
}

export async function main(argv = process.argv.slice(2), deps = null) {
  const parsed = parseArgs(argv);
  if (parsed.help) {
    printUsage();
    process.exitCode = 0;
    return;
  }

  const options = {
    json: parsed.json,
    autoRebuild: argv.includes('--auto-rebuild'),
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

  const report = await runCortexIndexGate(options, deps);
  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(
      report.pass ? 'PASS cortex-index gate' : 'FAIL cortex-index gate',
    );
    if (!report.pass) console.log(`fixHint: ${report.fixHint}`);
  }

  process.exitCode = report.pass ? 0 : 1;
  return report;
}

/* istanbul ignore next */
const isMain = import.meta.url === pathToFileURL(process.argv[1] ?? '').href;
/* istanbul ignore next */
if (isMain) {
  main();
}
