#!/usr/bin/env node
/**
 * @module pre-dispatch-freshness-hook
 * @description SessionStart / pre-dispatch hook that checks Cortex index
 *   freshness and triggers a background full reindex when the index is stale.
 *
 * The hook reads the hook input from stdin, determines whether the Cortex
 * corpus index is fresh enough to serve searches, and — if the index is
 * stale beyond a configurable grace window + staleness threshold — spawns a
 * detached background process to run a full reindex (`build-index.mjs` +
 * `embed-index.mjs`).
 *
 * The hook NEVER blocks dispatch. All tooling errors degrade gracefully: if
 * the freshness check cannot run (missing database, missing manifest, etc.),
 * the hook logs the error and returns `{ continue: true }`.
 *
 * Configuration via environment variables:
 * - `CORTEX_GRACE_WINDOW_S` (default 300) — grace window in seconds. If the
 *   last reindex was within this window, the index is considered fresh.
 * - `CORTEX_STALENESS_THRESHOLD_S` (default 300) — additional staleness
 *   threshold in seconds. If the index is older than grace + threshold,
 *   trigger a background reindex.
 *
 * For complex slices, the orchestrator MAY set `wait_for_reindex: true` in
 * the hook input (or env `CORTEX_WAIT_FOR_REINDEX=1`) to make the hook wait
 * for the background reindex to complete before returning. This should be
 * used sparingly as it blocks dispatch.
 */
import { spawn, spawnSync } from 'node:child_process';
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const buildIndexPath = path.join(repoRoot, 'rag-index', 'build-index.mjs');
const embedIndexPath = path.join(repoRoot, 'rag-index', 'embed-index.mjs');
const logPath = path.join(repoRoot, 'artifacts', 'pre-dispatch-freshness.log');
const manifestPath = path.join(
  repoRoot,
  'artifacts',
  'cortex-freshness-manifest.json',
);

const DEFAULT_GRACE_WINDOW_S = 300;
const DEFAULT_STALENESS_THRESHOLD_S = 300;

main().catch((error) => {
  safeLog(`[pre-dispatch-freshness] fatal: ${error?.message ?? error}`);
  writeHookOutput({ continue: true });
});

/**
 * Hook entry point.
 */
async function main() {
  const hookInput = readHookInput();
  const graceWindowS = parseEnvInt(
    'CORTEX_GRACE_WINDOW_S',
    DEFAULT_GRACE_WINDOW_S,
  );
  const stalenessThresholdS = parseEnvInt(
    'CORTEX_STALENESS_THRESHOLD_S',
    DEFAULT_STALENESS_THRESHOLD_S,
  );
  const waitForReindex =
    Boolean(hookInput.wait_for_reindex) ||
    process.env.CORTEX_WAIT_FOR_REINDEX === '1';

  const staleness = checkStaleness(graceWindowS, stalenessThresholdS);

  if (staleness.shouldReindex) {
    safeLog(
      `[pre-dispatch-freshness] index is stale (age ${staleness.ageS}s, threshold ${staleness.thresholdS}s) — triggering background reindex`,
    );
    if (waitForReindex) {
      runReindexSync();
    } else {
      runReindexBackground();
    }
    updateManifest();
  } else {
    safeLog(
      `[pre-dispatch-freshness] index is fresh enough (age ${staleness.ageS}s, grace ${graceWindowS}s)`,
    );
  }

  writeHookOutput({
    continue: true,
    hookSpecificOutput: {
      hookEventName: 'SessionStart',
      additionalContext: staleness.shouldReindex
        ? `Cortex index was stale (age ${staleness.ageS}s); background reindex triggered.`
        : `Cortex index is fresh (age ${staleness.ageS}s).`,
    },
  });
}

/**
 * Determine whether the index is stale enough to warrant a reindex.
 *
 * Uses a manifest file (`artifacts/cortex-freshness-manifest.json`) that
 * records the last reindex timestamp. If no manifest exists, the index is
 * considered stale (age = Infinity).
 *
 * @param {number} graceWindowS - Grace window in seconds.
 * @param {number} stalenessThresholdS - Additional staleness threshold in seconds.
 * @returns {{ shouldReindex: boolean, ageS: number, thresholdS: number }} Staleness report.
 */
export function checkStaleness(graceWindowS, stalenessThresholdS) {
  const now = Date.now();
  const lastReindex = readManifestTimestamp();
  const ageS =
    lastReindex === null ? Infinity : Math.floor((now - lastReindex) / 1000);
  const thresholdS = graceWindowS + stalenessThresholdS;
  const shouldReindex = ageS > thresholdS;
  return { shouldReindex, ageS, thresholdS };
}

/**
 * Read the last reindex timestamp from the manifest file.
 *
 * @returns {number|null} Timestamp in ms, or null if no manifest exists.
 */
export function readManifestTimestamp() {
  try {
    if (!existsSync(manifestPath)) return null;
    const content = readFileSync(manifestPath, 'utf8');
    const manifest = JSON.parse(content);
    const ts = Number(manifest.lastReindex);
    return Number.isFinite(ts) ? ts : null;
  } catch {
    return null;
  }
}

/**
 * Update the manifest with the current timestamp, marking a reindex as started.
 */
export function updateManifest() {
  try {
    mkdirSync(path.dirname(manifestPath), { recursive: true });
    writeFileSync(
      manifestPath,
      JSON.stringify(
        {
          lastReindex: Date.now(),
          updatedBy: 'pre-dispatch-freshness-hook',
        },
        null,
        2,
      ),
      'utf8',
    );
  } catch {
    // Best-effort; never throw.
  }
}

/**
 * Spawn a detached background process to run a full reindex.
 */
export function runReindexBackground() {
  try {
    const child = spawn(
      process.execPath,
      [buildIndexPath, '--json', '--force'],
      {
        cwd: repoRoot,
        stdio: 'ignore',
        detached: true,
      },
    );
    child.unref();
    safeLog(
      `[pre-dispatch-freshness] background build-index spawned (pid ${child.pid ?? '?'})`,
    );
    // Chain embed-index after build-index completes in the background.
    child.on('exit', (code) => {
      if (code === 0) {
        try {
          const embedChild = spawn(
            process.execPath,
            [embedIndexPath, '--json'],
            {
              cwd: repoRoot,
              stdio: 'ignore',
              detached: true,
            },
          );
          embedChild.unref();
          safeLog(
            `[pre-dispatch-freshness] background embed-index spawned (pid ${embedChild.pid ?? '?'})`,
          );
        } catch (embedSpawnError) {
          safeLog(
            `[pre-dispatch-freshness] embed-index spawn failed: ${embedSpawnError.message}`,
          );
        }
      } else {
        safeLog(
          `[pre-dispatch-freshness] background build-index exited with code ${code}`,
        );
      }
    });
  } catch (spawnError) {
    safeLog(
      `[pre-dispatch-freshness] background spawn failed: ${spawnError.message}`,
    );
  }
}

/**
 * Run a full reindex synchronously (blocking). Only used when
 * `wait_for_reindex: true` is set by the orchestrator.
 */
export function runReindexSync() {
  try {
    const buildResult = spawnSync(
      process.execPath,
      [buildIndexPath, '--json', '--force'],
      { cwd: repoRoot, encoding: 'utf8', timeout: 300_000 },
    );
    safeLog(
      `[pre-dispatch-freshness] sync build-index: status ${buildResult.status ?? '?'}`,
    );
    if (buildResult.status === 0) {
      const embedResult = spawnSync(
        process.execPath,
        [embedIndexPath, '--json'],
        { cwd: repoRoot, encoding: 'utf8', timeout: 300_000 },
      );
      safeLog(
        `[pre-dispatch-freshness] sync embed-index: status ${embedResult.status ?? '?'}`,
      );
    }
  } catch (syncError) {
    safeLog(
      `[pre-dispatch-freshness] sync reindex failed: ${syncError.message}`,
    );
  }
}

/**
 * Parse a positive integer from an environment variable.
 *
 * @param {string} envVar - Environment variable name.
 * @param {number} defaultValue - Fallback value.
 * @returns {number} Parsed integer or default.
 */
export function parseEnvInt(envVar, defaultValue) {
  const raw = process.env[envVar];
  if (!raw) return defaultValue;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : defaultValue;
}

/**
 * Append a timestamped line to the freshness log.
 *
 * @param {string} message - Log message.
 */
export function safeLog(message) {
  try {
    mkdirSync(path.dirname(logPath), { recursive: true });
    appendFileSync(logPath, `${new Date().toISOString()} ${message}\n`, 'utf8');
  } catch {
    // Best-effort; never throw.
  }
}

/**
 * Read and parse the hook input JSON from stdin (fd 0).
 *
 * @returns {Record<string, unknown>} Parsed hook input (or empty object).
 */
function readHookInput() {
  try {
    const rawInput = readFileSync(0, 'utf8').trim();
    if (!rawInput) return {};
    return JSON.parse(rawInput);
  } catch {
    return {};
  }
}

/**
 * Write the hook output JSON to stdout.
 *
 * @param {object} payload - Hook output payload.
 */
export function writeHookOutput(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}
