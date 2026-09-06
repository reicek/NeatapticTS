#!/usr/bin/env node
/* global console, process, setTimeout */
/**
 * @module reindex-plan-family
 * @description Synchronous post-save plan hook with a 60 second budget and
 * graceful degradation for the Cortex plan family.
 *
 * When a `plans/*.plans.md` file changes, this module synchronously re-runs
 * `build-index.mjs --files=...` (BM25) for exactly those paths so dispatch
 * context stays fresh, then queues `embed-index.mjs --files=...` in the
 * background (non-blocking) so dense embeddings catch up without stalling
 * the caller. When the synchronous build exceeds the budget the hook warns
 * and returns a `stale: true` report instead of blocking or throwing.
 *
 * `waitForPlanFresh` is the pre-dispatch integration point: orchestrators
 * call it before dispatching plan-dependent work to poll
 * `rag-index/data/freshness-manifest.json` for `families.plan.fresh`. It
 * graceful-degrades with `{ planFresh: false, stale: true }` on timeout and
 * treats JSON parse errors (an in-flight atomic write-then-rename) as
 * "not yet fresh" rather than throwing.
 *
 * The validator (`rag-index/validate-index.mjs`) is the single writer of the
 * freshness manifest; this module never writes it.
 */

import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseCliArgs } from './cli-utils.mjs';
import { repoRoot } from './init-schema.mjs';

const PLAN_FILE_SUFFIX = '.plans.md';
const BUILD_INDEX_SCRIPT = 'rag-index/build-index.mjs';
const EMBED_INDEX_SCRIPT = 'rag-index/embed-index.mjs';
const DEFAULT_MANIFEST_PATH = path.join(
  repoRoot,
  'rag-index',
  'data',
  'freshness-manifest.json',
);

/** Default synchronous BM25 budget in milliseconds (60 s, architecture Q2). */
export const DEFAULT_MAX_SYNC_WAIT_MS = 60_000;

/** Default manifest polling cadence used by {@link waitForPlanFresh}. */
export const DEFAULT_POLL_INTERVAL_MS = 250;

/**
 * Synchronously reindex changed plan files with a bounded budget.
 *
 * Runs the BM25 index rebuild for exactly the changed `.plans.md` paths and,
 * on success, queues the dense embedding pass in the background. Non-plan
 * paths are ignored; an empty plan set is trivially fresh.
 *
 * @param {string[]} changedPaths - Changed file paths (repo-relative or
 *   absolute); non-`.plans.md` entries are filtered out.
 * @param {object} [options={}] - Options.
 * @param {number} [options.maxSyncWaitMs=60000] - Synchronous BM25 budget in
 *   milliseconds.
 * @param {function(string[], { timeoutMs: number }): Promise<{ ok: boolean, timedOut?: boolean }>|{ ok: boolean, timedOut?: boolean }} [options.runSync]
 *   Injected synchronous command runner. Receives the command argv and the
 *   timeout; resolves with `ok` plus optional `timedOut`.
 * @param {function(string[]): unknown} [options.runBackground] - Injected
 *   background command runner used to queue the dense embed pass.
 * @param {function(string): void} [options.onWarning] - Injected warning sink.
 * @returns {Promise<{ syncFresh: boolean, planFresh: boolean, stale: boolean, files: string[], reason?: 'timeout'|'build-failed' }>}
 *   Freshness report. `stale: true` with `reason: 'timeout'` when the budget
 *   elapsed, or `reason: 'build-failed'` when the build exited non-zero.
 *
 * @example
 * ```js
 * const report = await reindexPlanFamily(['plans/Roadmap.plans.md']);
 * if (report.stale) console.error('plan context may be stale:', report.reason);
 * ```
 */
export async function reindexPlanFamily(changedPaths, options = {}) {
  const maxSyncWaitMs = Number(
    options.maxSyncWaitMs ?? DEFAULT_MAX_SYNC_WAIT_MS,
  );
  const runSync = options.runSync ?? defaultSyncRunner;
  const runBackground = options.runBackground ?? defaultBackgroundRunner;
  const warn = options.onWarning ?? defaultWarning;

  const files = normalizePlanPaths(changedPaths);
  if (files.length === 0) {
    return { syncFresh: true, planFresh: true, stale: false, files };
  }

  const buildCommand = [
    'node',
    BUILD_INDEX_SCRIPT,
    ...toFileArgs(files),
  ];
  const syncResult = await runSync(buildCommand, { timeoutMs: maxSyncWaitMs });

  if (!syncResult.ok) {
    const reason = syncResult.timedOut === true ? 'timeout' : 'build-failed';
    warn(
      `reindex-plan-family: synchronous BM25 reindex ${reason} ` +
        `for ${files.join(', ')} — plan family may be stale`,
    );
    return { syncFresh: false, planFresh: false, stale: true, reason, files };
  }

  // Dense embeddings catch up out of band; the build must finish first so
  // embed-index re-chunks the new content.
  runBackground(['node', EMBED_INDEX_SCRIPT, ...toFileArgs(files)]);

  return { syncFresh: true, planFresh: true, stale: false, files };
}

/**
 * Wait until the plan family is reported fresh, with graceful degradation.
 *
 * Polls the per-family freshness manifest every `pollIntervalMs` until
 * `families.plan.fresh` is true or `maxSyncWaitMs` elapses. Parse errors from
 * an in-flight atomic manifest write are treated as "not yet fresh" instead
 * of throwing, so a mid-write reader never crashes the dispatch path.
 *
 * @param {object} [options={}] - Options.
 * @param {number} [options.maxSyncWaitMs=60000] - Total wait budget in
 *   milliseconds.
 * @param {number} [options.pollIntervalMs=250] - Poll cadence in milliseconds.
 * @param {string} [options.manifestPath] - Manifest file to read. Defaults to
 *   `rag-index/data/freshness-manifest.json` in the repo root.
 * @param {function(): unknown} [options.readManifest] - Injected manifest
 *   reader. Must return the parsed manifest object or `null` when the
 *   manifest is absent or unreadable.
 * @param {function(string): void} [options.onWarning] - Injected warning sink.
 * @returns {Promise<{ planFresh: boolean, stale: boolean, waitedMs: number, reason?: 'timeout' }>}
 *   Freshness report. `{ planFresh: false, stale: true, reason: 'timeout' }`
 *   when the budget elapsed without the family becoming fresh.
 *
 * @example
 * ```js
 * const { planFresh, stale } = await waitForPlanFresh({ maxSyncWaitMs: 60_000 });
 * if (stale) dispatchWithStaleWarning();
 * ```
 */
export async function waitForPlanFresh(options = {}) {
  const maxSyncWaitMs = Number(
    options.maxSyncWaitMs ?? DEFAULT_MAX_SYNC_WAIT_MS,
  );
  const pollIntervalMs = Number(
    options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS,
  );
  /* istanbul ignore next -- default reads the live repo manifest; tests always pass manifestPath or readManifest */
  const manifestPath = options.manifestPath ?? DEFAULT_MANIFEST_PATH;
  const readManifest =
    options.readManifest ?? (() => readPlanManifest(manifestPath));
  const warn = options.onWarning ?? defaultWarning;

  const startedAt = Date.now();
  let manifest = readManifest();

  while (manifest?.families?.plan?.fresh !== true) {
    const waitedMs = Date.now() - startedAt;
    if (waitedMs >= maxSyncWaitMs) {
      warn(
        `reindex-plan-family: plan family still stale after ${waitedMs}ms ` +
          `(budget ${maxSyncWaitMs}ms) — proceeding with stale context`,
      );
      return {
        planFresh: false,
        stale: true,
        reason: 'timeout',
        waitedMs,
      };
    }
    await sleep(pollIntervalMs);
    manifest = readManifest();
  }

  return {
    planFresh: true,
    stale: false,
    waitedMs: Date.now() - startedAt,
  };
}

/**
 * Read and parse the freshness manifest, tolerating in-flight atomic writes.
 *
 * The validator writes the manifest via write-then-rename; a reader that
 * lands mid-replace can observe a partial or unparsable file. Those reads
 * resolve to `null`, which callers treat as "not yet fresh".
 *
 * @param {string} manifestPath - Absolute path to `freshness-manifest.json`.
 * @returns {unknown} The parsed manifest object, or `null` when absent or
 *   unparsable.
 */
export function readPlanManifest(manifestPath) {
  try {
    if (!existsSync(manifestPath)) return null;
    return JSON.parse(readFileSync(manifestPath, 'utf8'));
  } catch {
    return null;
  }
}

/**
 * CLI entry point. Two modes:
 *
 * - file mode (default): `node rag-index/reindex-plan-family.mjs
 *   --files=<path>... [--max-wait-ms=60000]` runs {@link reindexPlanFamily}.
 * - wait mode: `--wait-fresh [--max-wait-ms=60000] [--poll-ms=250]
 *   [--manifest=<path>]` runs {@link waitForPlanFresh}.
 *
 * Always prints the freshness report as JSON and exits 0 — the hook must
 * never block or fail the surrounding automation.
 *
 * @param {string[]} [argv=process.argv.slice(2)] - Raw CLI arguments.
 * @param {object} [deps={}] - Injectable dependencies forwarded to the
 *   underlying functions (`runSync`, `runBackground`, `readManifest`,
 *   `manifestPath`) plus `log` for output. Used by tests.
 * @returns {Promise<object>} The freshness report.
 */
export async function main(argv = process.argv.slice(2), deps = {}) {
  const flags = parseCliArgs(argv, { repeatableFlags: ['files'] });
  const log = deps.log ?? console.log;

  const options = {
    maxSyncWaitMs:
      flags['max-wait-ms'] === undefined ? undefined : Number(flags['max-wait-ms']),
    pollIntervalMs:
      flags['poll-ms'] === undefined ? undefined : Number(flags['poll-ms']),
    manifestPath: deps.manifestPath ?? flags.manifest,
    runSync: deps.runSync,
    runBackground: deps.runBackground,
    readManifest: deps.readManifest,
  };

  const files = flags.files === undefined ? [] : [flags.files].flat();
  const result =
    flags['wait-fresh'] === true
      ? await waitForPlanFresh(options)
      : await reindexPlanFamily(files, options);

  log(JSON.stringify(result, null, 2));
  return result;
}

/**
 * Normalize changed paths to repo-relative POSIX plan paths.
 *
 * @param {string[]} changedPaths - Raw changed paths.
 * @returns {string[]} Deduplicated, lexicographically sorted repo-relative
 *   `.plans.md` paths.
 */
function normalizePlanPaths(changedPaths) {
  const candidates = Array.isArray(changedPaths) ? changedPaths : [];
  const normalized = [];

  for (const candidate of candidates) {
    const repoRelative = toRepoRelativePlanPath(candidate);
    if (repoRelative !== null && !normalized.includes(repoRelative)) {
      normalized.push(repoRelative);
    }
  }

  return normalized.toSorted();
}

/**
 * Convert a candidate path to a repo-relative `.plans.md` path.
 *
 * @param {unknown} candidate - Raw path candidate.
 * @returns {string|null} Repo-relative POSIX path, or `null` when the
 *   candidate is not a non-empty `.plans.md` string.
 */
function toRepoRelativePlanPath(candidate) {
  if (typeof candidate !== 'string' || candidate.trim() === '') return null;
  const relative = path.isAbsolute(candidate)
    ? path.relative(repoRoot, candidate)
    : candidate;
  const normalized = relative.replaceAll(path.sep, '/');
  return normalized.endsWith(PLAN_FILE_SUFFIX) ? normalized : null;
}

/**
 * Build repeatable `--files=<path>` CLI arguments.
 *
 * @param {string[]} files - Repo-relative paths.
 * @returns {string[]} CLI arguments.
 */
function toFileArgs(files) {
  return files.map((file) => `--files=${file}`);
}

/**
 * Default synchronous runner: spawn `build-index.mjs` with a hard timeout.
 *
 * @param {string[]} command - Command argv (`[executable, ...args]`).
 * @param {{ timeoutMs: number }} options - Timeout in milliseconds.
 * @returns {{ ok: boolean, timedOut: boolean }} Success and timeout flags.
 */
/* istanbul ignore next -- spawns a real node process; unit tests always inject runSync */
function defaultSyncRunner(command, { timeoutMs }) {
  const [executable, ...args] = command;
  const result = spawnSync(executable, args, {
    cwd: repoRoot,
    encoding: 'utf8',
    shell: false,
    timeout: timeoutMs,
  });
  const timedOut =
    result.error?.code === 'ETIMEDOUT' || result.signal === 'SIGTERM';
  // spawnSync leaves `error` undefined on a normal exit; treat any set
  // error (spawn failure or timeout) as not-ok.
  return { ok: result.status === 0 && result.error == null, timedOut };
}

/**
 * Default background runner: spawn a detached, unref'd `embed-index.mjs`
 * process so the dense pass never blocks the caller.
 *
 * @param {string[]} command - Command argv (`[executable, ...args]`).
 */
/* istanbul ignore next -- spawns a detached node process; unit tests always inject runBackground */
function defaultBackgroundRunner(command) {
  const [executable, ...args] = command;
  try {
    const child = spawn(executable, args, {
      cwd: repoRoot,
      stdio: 'ignore',
      detached: true,
    });
    child.unref();
  } catch {
    // Best-effort: dense embeddings also refresh at the next validation pass.
  }
}

/**
 * Default warning sink.
 *
 * @param {string} message - Warning text.
 */
function defaultWarning(message) {
  console.error(message);
}

/**
 * Pause for the given number of milliseconds.
 *
 * @param {number} ms - Delay in milliseconds.
 * @returns {Promise<void>} Resolves after the delay.
 */
function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

/* istanbul ignore next -- CLI entry point guard */
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  await main();
}
