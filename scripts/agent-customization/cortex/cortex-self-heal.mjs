/**
 * @module cortex-self-heal
 * @description Detached repair orchestrator for the Cortex self-heal loop.
 *
 * The guard (`cortex-health-guard.mjs`) decides WHEN a repair should run and
 * spawns this orchestrator detached (`cortex-self-heal.mjs --state-dir <dir>`);
 * this module owns the repair execution and the shared state file's write
 * side (the guard writes state only at trigger time; everything else reads).
 * The run sequence per the plan's Architecture section:
 *
 *   1. Acquire the repair lock — adopt the guard's pre-created lock (the guard
 *      reserves it with the spawned orchestrator's pid), reclaim a stale lock
 *      (heartbeat at/beyond `CORTEX_SELFHEAL_LOCK_STALE_S`, or a dead holder
 *      pid, after a single forced unlink retry), or create it fresh. A lock
 *      held by a live foreign pid aborts the run without side effects.
 *   2. Hold the lock for the whole run, refreshing `heartbeat_at` on a
 *      heartbeat interval so a crash mid-run becomes reclaimable.
 *   3. Execute the composed repair sequence in classifier-first order,
 *      shelling out to the existing automation (`node` entry points and
 *      `npm run` scripts) — repair logic is never reimplemented here, and no
 *      `--force` flag is ever composed.
 *   4. Append the outcome and the measured duration to the state file's
 *      `history` (atomic write-then-rename) and clear `in_flight`.
 *   5. Release the lock on both success and failure.
 *
 * All external effects are injectable seams (`now`, `spawner`, `state`, `pid`)
 * so tests stay hermetic; the CLI (`--dry-run --json`, `--state-dir <dir>`)
 * composes the exact command list a real run would execute against the
 * on-disk state without executing anything or taking the lock.
 */
import { spawnSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import {
  closeSync,
  openSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');

/**
 * Parse an environment variable as a non-negative integer, falling back to
 * the default for missing, malformed, or negative values (same convention as
 * `CORTEX_GRACE_WINDOW_S` in `pre-dispatch-freshness-hook.mjs`).
 *
 * @param {string} name - Environment variable name.
 * @param {number} defaultValue - Fallback when the variable is unset/invalid.
 * @returns {number} Parsed knob value.
 */
function parseEnvInt(name, defaultValue) {
  const raw = process.env[name];
  if (!raw) return defaultValue;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : defaultValue;
}

const MS_PER_SECOND = 1000;
const MS_PER_MINUTE = 60 * MS_PER_SECOND;

const DEFAULT_COOLDOWN_S = 600;
const DEFAULT_BACKOFF_FACTOR = 2;
const DEFAULT_MAX_ATTEMPTS = 3;
const DEFAULT_LOCK_STALE_S = 2700;

/** Minimum seconds between repair triggers (`CORTEX_SELFHEAL_COOLDOWN_S`). */
const COOLDOWN_S = parseEnvInt(
  'CORTEX_SELFHEAL_COOLDOWN_S',
  DEFAULT_COOLDOWN_S,
);
/** Exponential multiplier per repeated attempt in the window (`CORTEX_SELFHEAL_BACKOFF_FACTOR`). */
const BACKOFF_FACTOR = parseEnvInt(
  'CORTEX_SELFHEAL_BACKOFF_FACTOR',
  DEFAULT_BACKOFF_FACTOR,
);
/** Attempt ceiling per window before pause-and-ask (`CORTEX_SELFHEAL_MAX_ATTEMPTS`). */
const MAX_ATTEMPTS = parseEnvInt(
  'CORTEX_SELFHEAL_MAX_ATTEMPTS',
  DEFAULT_MAX_ATTEMPTS,
);
/**
 * Stale-lock age threshold in milliseconds (`CORTEX_SELFHEAL_LOCK_STALE_S`,
 * default 2700 s). A lock is stale when its heartbeat is at or beyond this
 * threshold, or when the holder pid is no longer alive.
 */
const LOCK_STALE_MS =
  parseEnvInt('CORTEX_SELFHEAL_LOCK_STALE_S', DEFAULT_LOCK_STALE_S) *
  MS_PER_SECOND;
/** Kill switch — `1` disables automatic repair runs (dry-run still works). */
const KILL_SWITCH_ON = parseEnvInt('CORTEX_SELFHEAL_DISABLE', 0) === 1;
/**
 * Lock heartbeat interval (30 s) — far below the 2700 s staleness threshold,
 * so a healthy run misses ~90 heartbeats' worth of grace before reclaim.
 */
const HEARTBEAT_INTERVAL_MS = 30 * MS_PER_SECOND;

/** Shared state file name (this module is the state file's single writer). */
const STATE_FILE_NAME = 'cortex-self-heal-state.json';
/** Repair lock file name (mutual exclusion across repair-capable surfaces). */
const LOCK_FILE_NAME = 'cortex-self-heal.repair.lock';
const STATE_VERSION = 1;
/** History retention: the state file keeps the last 5 repair runs. */
const HISTORY_MAX_ENTRIES = 5;

/** Default state directory, matching the guard's trigger path. */
const DEFAULT_STATE_DIR = path.join(repoRoot, 'rag-index', 'data');

/** Composed repair sequence (classifier-first; existing automation only). */
const CLASSIFY_COMMAND = 'node rag-index/validate-index.mjs --json';
const CORPUS_BUILD_COMMAND = 'node rag-index/build-index.mjs --json';
const SNAPSHOT_COMMAND = 'npm run index:build-snapshot';
const PREWARM_COMMAND = 'npm run index:prewarm';
const GATE_COMMAND =
  'node scripts/agent-customization/gates/cortex-index.gate.mjs --auto-rebuild --json';
const SMOKE_COMMAND =
  'node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json';

/**
 * @param {*} value - Candidate value.
 * @returns {boolean} `true` for non-null plain objects.
 */
function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * @param {*} value - Candidate value.
 * @param {number} fallback - Fallback number.
 * @returns {number} The finite number, or the fallback.
 */
function finiteOr(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

/**
 * @param {*} value - Candidate value.
 * @returns {number|null} The finite number, or `null`.
 */
function finiteOrNull(value) {
  return Number.isFinite(value) ? value : null;
}

/**
 * @param {number} value - Value to round.
 * @returns {number} Value rounded to three decimal places.
 */
function round3(value) {
  return Math.round(value * 1000) / 1000;
}

/**
 * Probe whether a pid is alive. Signal 0 never kills; on Windows a recently
 * reused pid counts as alive. `EPERM` means the process exists but is owned
 * by another user, so it counts as alive too.
 *
 * @param {*} pid - Candidate pid.
 * @returns {boolean} `true` when the pid references a live process.
 */
function pidAlive(pid) {
  if (!Number.isFinite(pid) || pid <= 0) return false;
  if (pid === process.pid) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return Boolean(error) && error.code === 'EPERM';
  }
}

/**
 * @param {string} stateDir - Directory holding the lock file.
 * @returns {string} Absolute lock file path.
 */
function lockPathIn(stateDir) {
  return path.join(stateDir, LOCK_FILE_NAME);
}

/**
 * @param {string} stateDir - Directory holding the state file.
 * @returns {string} Absolute state file path.
 */
function statePathIn(stateDir) {
  return path.join(stateDir, STATE_FILE_NAME);
}

/**
 * Read and parse the lock record, tolerating a missing or corrupt file.
 *
 * @param {string} lockPath - Lock file path.
 * @returns {object|null} Lock record, or `null` when unreadable.
 */
function readLockRecord(lockPath) {
  try {
    const parsed = JSON.parse(readFileSync(lockPath, 'utf8'));
    return isPlainObject(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/** @returns {object} Fresh state seeded from the import-time knobs. */
function freshState() {
  return {
    version: STATE_VERSION,
    last_probe: null,
    cooldown: { cooldown_s: COOLDOWN_S, last_trigger_at: null },
    backoff: {
      attempts_in_window: 0,
      window_started_at: null,
      backoff_factor: BACKOFF_FACTOR,
      next_allowed_at: 0,
    },
    max_attempts: MAX_ATTEMPTS,
    in_flight: null,
    history: [],
  };
}

/**
 * Coerce a raw (possibly partial/corrupt) state file into the plan schema —
 * field-for-field identical to the guard's normalization so both modules
 * read and write the same shape.
 *
 * @param {object} raw - Parsed JSON of the on-disk state file.
 * @returns {object} Normalized state with every schema field populated.
 */
function normalizeState(raw) {
  const base = freshState();
  if (!isPlainObject(raw)) return base;
  const cooldown = isPlainObject(raw.cooldown) ? raw.cooldown : {};
  const backoff = isPlainObject(raw.backoff) ? raw.backoff : {};
  return {
    version: finiteOr(raw.version, STATE_VERSION),
    last_probe: isPlainObject(raw.last_probe) ? raw.last_probe : null,
    cooldown: {
      cooldown_s: finiteOr(cooldown.cooldown_s, COOLDOWN_S),
      last_trigger_at: finiteOrNull(cooldown.last_trigger_at),
    },
    backoff: {
      attempts_in_window: finiteOr(backoff.attempts_in_window, 0),
      window_started_at: finiteOrNull(backoff.window_started_at),
      backoff_factor: finiteOr(backoff.backoff_factor, BACKOFF_FACTOR),
      next_allowed_at: finiteOr(backoff.next_allowed_at, 0),
    },
    max_attempts: finiteOr(raw.max_attempts, MAX_ATTEMPTS),
    in_flight: isPlainObject(raw.in_flight) ? raw.in_flight : null,
    history: Array.isArray(raw.history) ? raw.history : [],
  };
}

/**
 * Load the normalized state, falling back to the injected seed (then fresh
 * defaults) when the on-disk file is missing or corrupt.
 *
 * @param {string} statePath - State file path.
 * @param {object|null} fallback - Seed state (typically the caller's snapshot).
 * @returns {object} Normalized state in the plan schema.
 */
function loadStateOrSeed(statePath, fallback = null) {
  try {
    return normalizeState(JSON.parse(readFileSync(statePath, 'utf8')));
  } catch {
    if (isPlainObject(fallback)) return normalizeState(fallback);
    return freshState();
  }
}

/**
 * Persist the state file atomically (write-then-rename, same convention as
 * `validate-index.mjs` freshness manifests).
 *
 * @param {string} statePath - State file path.
 * @param {object} state - Full state in the plan schema.
 * @returns {void}
 */
function persistState(statePath, state) {
  const tempPath = `${statePath}.${randomBytes(8).toString('hex')}.tmp`;
  writeFileSync(tempPath, JSON.stringify(state, null, 2));
  renameSync(tempPath, statePath);
}

/**
 * Production spawner: shell out to the composed command from the repo root.
 * A killed/undetermined process yields a nonzero code so the run records a
 * failure instead of hanging the state machine.
 *
 * @param {string} command - Shell command line to execute.
 * @returns {Promise<{code: number, stdout: string, stderr: string}>} Result.
 */
async function defaultSpawner(command) {
  const result = spawnSync(command, {
    shell: true,
    cwd: repoRoot,
    windowsHide: true,
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
  });
  return {
    code: Number.isFinite(result.status) ? result.status : 1,
    stdout: typeof result.stdout === 'string' ? result.stdout : '',
    stderr: typeof result.stderr === 'string' ? result.stderr : '',
  };
}

/**
 * Compose the classifier-first repair command list for the current dense
 * state — the exact sequence `runRepair` executes. Every entry shells out to
 * existing automation; no `--force` flag is ever composed. The snapshot and
 * prewarm steps are dropped when the last probe was `warm` (dense store
 * healthy), leaving classify → corpus build → revalidate gates.
 *
 * @param {{state?: object|null}} [seams] - Seams (state carries `last_probe`).
 * @returns {string[]} Command lines in execution order.
 */
export function dryRunReport(seams = {}) {
  const state =
    isPlainObject(seams) && isPlainObject(seams.state) ? seams.state : null;
  const lastProbeState =
    state !== null && isPlainObject(state.last_probe)
      ? state.last_probe.state
      : null;
  const denseRepairNeeded = lastProbeState !== 'warm';
  return [
    CLASSIFY_COMMAND,
    CORPUS_BUILD_COMMAND,
    ...(denseRepairNeeded ? [SNAPSHOT_COMMAND, PREWARM_COMMAND] : []),
    GATE_COMMAND,
    SMOKE_COMMAND,
  ];
}

/**
 * Create the repair lock exclusively (open with `'wx'` so a concurrent
 * creator gets `EEXIST` and sees `null`).
 *
 * @param {{
 *   stateDir: string,
 *   pid?: number|null,
 *   now?: () => number,
 *   attempt?: number,
 *   reason?: string,
 * }} seams - Lock creation inputs.
 * @returns {object|null} Lock record `{pid, started_at, heartbeat_at, attempt, reason}`, or `null` when the lock already exists.
 */
export function createLock(seams = {}) {
  const {
    stateDir,
    pid,
    now = Date.now,
    attempt = 1,
    reason = 'unknown',
  } = seams;
  const at = now();
  const record = { pid, started_at: at, heartbeat_at: at, attempt, reason };
  let handle;
  try {
    handle = openSync(lockPathIn(stateDir), 'wx');
  } catch (error) {
    if (error && error.code === 'EEXIST') return null;
    throw error;
  }
  try {
    writeFileSync(handle, JSON.stringify(record, null, 2));
  } finally {
    closeSync(handle);
  }
  return record;
}

/**
 * Refresh the lock heartbeat: bump `heartbeat_at` only, preserving every
 * other field of the record (notably `started_at`). When `pid` is provided,
 * a lock taken over by a foreign pid is never overwritten (lost ownership).
 *
 * @param {{stateDir: string, now?: () => number, pid?: number}} seams - Refresh inputs.
 * @returns {object|null} Updated lock record, or `null` when no lock exists (or ownership was lost).
 */
export function refreshLock(seams = {}) {
  const { stateDir, now = Date.now, pid } = seams;
  const lockPath = lockPathIn(stateDir);
  const record = readLockRecord(lockPath);
  if (record === null) return null;
  if (pid !== undefined && record.pid !== pid) return null;
  const updated = { ...record, heartbeat_at: now() };
  writeFileSync(lockPath, JSON.stringify(updated, null, 2));
  return updated;
}

/**
 * Release the lock, but only when it is still owned by `pid` — a foreign
 * holder's lock is never removed.
 *
 * @param {{stateDir: string, pid?: number}} seams - Release inputs.
 * @returns {boolean} `true` when the lock was removed.
 */
export function releaseLock(seams = {}) {
  const { stateDir, pid } = seams;
  const lockPath = lockPathIn(stateDir);
  const record = readLockRecord(lockPath);
  if (record === null) return false;
  if (record.pid !== pid) return false;
  unlinkSync(lockPath);
  return true;
}

/**
 * Reclaim a stale lock: stale means the heartbeat is at or beyond
 * `CORTEX_SELFHEAL_LOCK_STALE_S` **or** the holder pid is no longer alive.
 * After a single forced unlink retry, the caller re-creates the lock with
 * its own pid, preserving the original attempt and reason. A live holder
 * with a fresh heartbeat is never reclaimed.
 *
 * @param {{stateDir: string, now?: () => number, pid?: number|null}} seams - Reclaim inputs.
 * @returns {object|null} The new lock record, or `null` when the lock is not stale (or a concurrent creator won the re-create race).
 */
export function reclaimStaleLock(seams = {}) {
  const { stateDir, now = Date.now, pid } = seams;
  const lockPath = lockPathIn(stateDir);
  const record = readLockRecord(lockPath);
  if (record === null) return null;
  const heartbeatAt = finiteOrNull(record.heartbeat_at);
  const staleByAge = now() - (heartbeatAt ?? 0) >= LOCK_STALE_MS;
  const holderAlive = pidAlive(record.pid);
  if (!staleByAge && holderAlive) return null;

  let unlinkFailed = null;
  for (let attemptIndex = 0; attemptIndex < 2; attemptIndex += 1) {
    try {
      unlinkSync(lockPath);
      unlinkFailed = null;
      break;
    } catch (error) {
      unlinkFailed = error;
      if (error && error.code === 'ENOENT') {
        unlinkFailed = null;
        break;
      }
    }
  }
  if (unlinkFailed !== null) return null;

  return createLock({
    stateDir,
    pid,
    now,
    attempt: finiteOr(record.attempt, 1),
    reason: typeof record.reason === 'string' ? record.reason : 'reclaimed',
  });
}

/**
 * Run the full repair under the lock: acquire (adopt the guard's pre-created
 * lock / reclaim stale / create fresh), hold with heartbeat, execute the
 * composed commands in classifier-first order, append the outcome and
 * measured duration to the state history (atomic write-then-rename), clear
 * `in_flight`, and release the lock on both success and failure.
 *
 * @param {{
 *   stateDir: string,
 *   now?: () => number,
 *   spawner?: (command: string) => Promise<{code: number, stdout?: string, stderr?: string}>,
 *   state?: object|null,
 *   pid?: number,
 * }} seams - Run inputs (all external effects injectable).
 * @returns {Promise<{success: boolean, commands_executed?: number, skipped?: boolean, error?: string}>} Run outcome.
 */
export async function runRepair(seams = {}) {
  const {
    stateDir,
    now = Date.now,
    spawner = defaultSpawner,
    state = null,
    pid = process.pid,
  } = seams;
  if (typeof stateDir !== 'string' || stateDir.length === 0) {
    throw new TypeError('runRepair requires a stateDir string');
  }
  if (typeof spawner !== 'function') {
    throw new TypeError('runRepair requires a spawner function');
  }
  const lockPath = lockPathIn(stateDir);
  const statePath = statePathIn(stateDir);

  const seedState = loadStateOrSeed(statePath, state);
  const lastProbe = seedState.last_probe;
  const reason =
    lastProbe !== null && typeof lastProbe.state === 'string'
      ? lastProbe.state
      : 'unknown';
  const attempt = finiteOr(seedState.backoff.attempts_in_window, 0) + 1;

  // 1. Acquire the lock: adopt our own pre-created lock (the guard reserves
  //    it with the spawned orchestrator's pid), reclaim a stale one, or
  //    create it fresh. A live foreign holder aborts without side effects.
  let lock = readLockRecord(lockPath);
  if (lock !== null && lock.pid === pid) {
    lock = refreshLock({ stateDir, now, pid }) ?? lock;
  } else if (lock !== null) {
    lock = reclaimStaleLock({ stateDir, now, pid });
    if (lock === null) {
      return {
        success: false,
        skipped: true,
        error: 'repair lock held by a live holder',
      };
    }
  } else {
    lock = createLock({ stateDir, pid, now, attempt, reason });
    if (lock === null) {
      lock = reclaimStaleLock({ stateDir, now, pid });
      if (lock === null) {
        return {
          success: false,
          skipped: true,
          error: 'repair lock lost to a concurrent creator',
        };
      }
    }
  }

  // 2. Hold the lock for the whole run with a periodic heartbeat so a crash
  //    mid-run leaves a reclaimable record.
  const startedAtMs = now();
  const heartbeatTimer = setInterval(() => {
    refreshLock({ stateDir, now, pid });
  }, HEARTBEAT_INTERVAL_MS);
  if (typeof heartbeatTimer.unref === 'function') {
    heartbeatTimer.unref();
  }

  // 3. Execute the composed commands in classifier-first order.
  const commands = dryRunReport({ state });
  const executed = [];
  let failure = null;
  try {
    for (const command of commands) {
      const result = await spawner(command);
      executed.push(command);
      const code =
        isPlainObject(result) && Number.isFinite(result.code) ? result.code : 1;
      if (code !== 0) {
        failure = {
          command,
          code,
          stderr:
            isPlainObject(result) && typeof result.stderr === 'string'
              ? result.stderr
              : null,
        };
        break;
      }
    }
  } finally {
    clearInterval(heartbeatTimer);
  }

  // 4. Record the outcome + measured duration into the state history
  //    (atomic write-then-rename) and clear the guard's `in_flight` record.
  const finishedAtMs = now();
  const success = failure === null;
  const durationMin = round3(
    Math.max(0, (finishedAtMs - startedAtMs) / MS_PER_MINUTE),
  );
  const historyEntry = {
    success,
    duration_min: durationMin,
    started_at: startedAtMs,
    finished_at: finishedAtMs,
    attempt: finiteOr(lock.attempt, attempt),
    reason: typeof lock.reason === 'string' ? lock.reason : reason,
    commands_executed: executed.length,
  };
  if (!success) {
    historyEntry.error =
      `${failure.command} exited with code ${failure.code}` +
      (failure.stderr !== null ? `: ${failure.stderr}` : '');
  }
  const finalState = {
    ...seedState,
    in_flight: null,
    history: [...seedState.history, historyEntry].slice(-HISTORY_MAX_ENTRIES),
  };
  persistState(statePath, finalState);

  // 5. Release the lock on both success and failure.
  releaseLock({ stateDir, pid });

  const outcome = { success, commands_executed: executed.length };
  if (!success) outcome.error = historyEntry.error;
  return outcome;
}

/**
 * Parse CLI flags: `--dry-run`, `--json`, `--state-dir <dir>` (the guard
 * spawns `cortex-self-heal.mjs --state-dir <dir>`).
 *
 * @param {string[]} argv - Flags after the script path.
 * @returns {{dryRun: boolean, json: boolean, stateDir: string}} Parsed flags.
 */
function parseCliArgs(argv) {
  const flags = { dryRun: false, json: false, stateDir: DEFAULT_STATE_DIR };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--dry-run') {
      flags.dryRun = true;
    } else if (arg === '--json') {
      flags.json = true;
    } else if (arg === '--state-dir') {
      const value = argv[index + 1];
      if (typeof value === 'string' && value.length > 0) {
        flags.stateDir = value;
        index += 1;
      }
    } else if (typeof arg === 'string' && arg.startsWith('--state-dir=')) {
      const value = arg.slice('--state-dir='.length);
      if (value.length > 0) flags.stateDir = value;
    }
  }
  return flags;
}

/**
 * CLI entry: `--dry-run --json` composes and prints the repair command list
 * against the on-disk state (executing nothing, taking no lock); a real run
 * executes `runRepair` with the production spawner. The kill switch
 * (`CORTEX_SELFHEAL_DISABLE=1`) skips the real run with exit code 0.
 *
 * @param {string[]} argv - CLI flags.
 * @returns {Promise<number>} Process exit code.
 */
async function main(argv = []) {
  const flags = parseCliArgs(argv);
  const statePath = statePathIn(flags.stateDir);

  if (flags.dryRun) {
    const state = loadStateOrSeed(statePath);
    const commands = dryRunReport({ state });
    const report = {
      dry_run: true,
      state_dir: flags.stateDir,
      state_file: statePath,
      lock_file: lockPathIn(flags.stateDir),
      last_probe_state:
        state.last_probe !== null && typeof state.last_probe.state === 'string'
          ? state.last_probe.state
          : null,
      commands,
      ordering: 'classifier-first',
      notes: [
        'Every command shells out to existing automation; no repair logic is reimplemented.',
        'No --force flag is composed; stale rebuilds go through the auto-rebuild gate.',
        'A dry run executes nothing and takes no lock.',
      ],
    };
    if (flags.json) {
      process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    } else {
      process.stdout.write(
        `Cortex self-heal dry run (last_probe.state=${report.last_probe_state})\n` +
          `${commands.map((command) => `  - ${command}`).join('\n')}\n`,
      );
    }
    return 0;
  }

  if (KILL_SWITCH_ON) {
    const payload = {
      success: false,
      skipped: true,
      error:
        'CORTEX_SELFHEAL_DISABLE=1 — automatic repair is disabled; run the manual recovery commands instead.',
    };
    process.stdout.write(
      flags.json
        ? `${JSON.stringify(payload, null, 2)}\n`
        : `${payload.error}\n`,
    );
    return 0;
  }

  const state = loadStateOrSeed(statePath);
  const result = await runRepair({
    stateDir: flags.stateDir,
    state,
    pid: process.pid,
  });
  const payload = { ...result, state_file: statePath, finished: true };
  process.stdout.write(
    flags.json
      ? `${JSON.stringify(payload, null, 2)}\n`
      : `Cortex self-heal repair ${result.success ? 'succeeded' : 'failed'} (${result.commands_executed} commands).\n`,
  );
  return result.success ? 0 : 1;
}

const invokedDirectly =
  typeof process.argv[1] === 'string' &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;

if (invokedDirectly) {
  main(process.argv.slice(2))
    .then((exitCode) => {
      process.exitCode = exitCode;
    })
    .catch((error) => {
      process.stderr.write(`${error && error.stack ? error.stack : error}\n`);
      process.exitCode = 1;
    });
}
