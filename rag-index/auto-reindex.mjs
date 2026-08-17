#!/usr/bin/env node
/* global console, process */
/**
 * @module auto-reindex
 * @description Post-commit hook orchestration for re-indexing changed plan files.
 *
 * Detects `plans/*.plans.md` files that changed in the most recent commit and
 * runs targeted `build-index.mjs` / `embed-index.mjs` passes over just those
 * files. Failures are logged but never block the git commit: the post-commit
 * wrapper exits 0.
 */

import { spawnSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { toRepoRelative } from './cli-utils.mjs';
import { repoRoot } from './init-schema.mjs';

const PLAN_FILE_SUFFIX = '.plans.md';
const FRESHNESS_PROOFS_DIR = path.join(
  repoRoot,
  'rag-index',
  'freshness-proofs',
);
const REINDEX_LOG_PATH = path.join(FRESHNESS_PROOFS_DIR, 'plans-reindex.log');

/**
 * Re-index plan files that changed in the most recent commit.
 *
 * @param {object} [options={}] - Options.
 * @param {string[]} [options.changedFiles] - Repo-relative changed file paths.
 *   When omitted, changed `plans/*.plans.md` files are detected from
 *   `git diff --name-only HEAD~1 HEAD`.
 * @param {function(string[]): Promise<{success: boolean}>|{success: boolean}} [options.runCommand] - Injected command runner. Defaults to a `spawnSync` runner that treats the first element as the executable and the rest as arguments.
 * @returns {Promise<{ changed: string[]; commands: string[][]; exitCode: number; failures: string[][]; logPath: string }>}
 *   Summary of the re-index run.
 */
export async function reindexChangedPlans(options = {}) {
  const changedFiles = options.changedFiles ?? (await detectChangedPlanFiles());
  const planFiles = changedFiles.filter(isPlanFile);
  const commands = buildReindexCommands(planFiles);
  const runCommand = options.runCommand ?? defaultRunner;
  const failures = [];

  for (const command of commands) {
    const result = await runCommand(command);
    if (!result.success) {
      console.error(`auto-reindex: command failed: ${command.join(' ')}`);
      failures.push(command);
    }
  }

  const logPath = await writeReindexLog({
    changed: planFiles,
    commands,
    failures,
  });

  return { changed: planFiles, commands, exitCode: 0, failures, logPath };
}

/**
 * Produce a targeted fix-hint for stale plan files.
 *
 * @param {string[]} stalePaths - Repo-relative stale plan file paths.
 * @returns {string} A human-readable command hint that rebuilds and re-embeds
 *   exactly the listed files.
 */
export function resolveStalePlanFixHint(stalePaths) {
  const planFiles = stalePaths.filter(isPlanFile);
  if (planFiles.length === 0) {
    return 'Run: node rag-index/build-index.mjs to rebuild stale index';
  }

  const fileArgs = planFiles.map((filePath) => `--files=${filePath}`).join(' ');

  return `Run: node rag-index/build-index.mjs ${fileArgs} && node rag-index/embed-index.mjs ${fileArgs}`;
}

function isPlanFile(filePath) {
  return typeof filePath === 'string' && filePath.endsWith(PLAN_FILE_SUFFIX);
}

function buildReindexCommands(planFiles) {
  if (planFiles.length === 0) return [];

  const fileArgs = planFiles.map((filePath) => `--files=${filePath}`);

  return [
    ['node', 'rag-index/build-index.mjs', ...fileArgs],
    ['node', 'rag-index/embed-index.mjs', ...fileArgs],
  ];
}

async function detectChangedPlanFiles() {
  try {
    const result = spawnSync('git', ['diff', '--name-only', 'HEAD~1', 'HEAD'], {
      cwd: repoRoot,
      encoding: 'utf8',
      shell: false,
    });

    /* istanbul ignore if -- defensive: git returns non-zero only when HEAD~1 doesn't exist */
    if (result.status !== 0) return [];

    /* istanbul ignore next -- defensive: stdout is always a string when status is 0 */
    return (result.stdout ?? '')
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .filter(isPlanFile);
  } catch {
    /* istanbul ignore next -- defensive: spawnSync does not throw, errors are returned in result */
    return [];
  }
}

/* istanbul ignore next -- spawns real node processes, not suitable for unit tests */
async function defaultRunner(command) {
  const [executable, ...args] = command;
  const result = spawnSync(executable, args, {
    cwd: repoRoot,
    encoding: 'utf8',
    shell: false,
  });

  return { success: result.status === 0 && !result.error };
}

async function writeReindexLog(summary) {
  await mkdir(FRESHNESS_PROOFS_DIR, { recursive: true });
  const entry = { timestamp: new Date().toISOString(), ...summary };
  await writeFile(
    REINDEX_LOG_PATH,
    `${JSON.stringify(entry, null, 2)}\n`,
    'utf8',
  );

  return toRepoRelative(REINDEX_LOG_PATH);
}

export async function main() {
  const summary = await reindexChangedPlans();
  console.log(JSON.stringify(summary, null, 2));
}

/* istanbul ignore next -- CLI entry point guard */
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  await main();
}
