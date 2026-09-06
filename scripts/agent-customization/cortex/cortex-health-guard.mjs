/**
 * @module cortex-health-guard
 * @description Decision engine for the Cortex self-heal loop.
 *
 * `evaluateSelfHeal(seams)` is consulted by search-time surfaces whenever dense
 * readiness looks degraded. It owns exactly one decision: should a repair be
 * triggered now, or should the caller keep waiting (in-flight repair, cooldown,
 * exhausted attempts, kill switch, or nothing because dense search is warm)?
 *
 * Decision order (Architecture section of plans/rag-self-heal-hooks.plans.md):
 *   1. Probe dense readiness (unless the per-process probe-cost memo is set).
 *      A warm probe short-circuits with zero filesystem activity.
 *   2. Kill switch — `CORTEX_SELFHEAL_DISABLE=1` blocks triggering/spawning
 *      while degraded guidance still reports.
 *   3. In-flight repair recorded in the shared state file (no re-trigger).
 *   4. Cooldown window + backoff `next_allowed_at` (no re-trigger).
 *   5. Attempt ceiling per window — pause-and-ask (T3) guidance.
 *   6. Trigger: reserve the repair lock atomically (`open` with `'wx'`),
 *      spawn the detached orchestrator, persist the lock record and the
      updated state file (write-then-rename), and return T1 guidance.
 *
 * The probe cost is measured with the injected clock; when it exceeds
 * `CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS` (default 600 ms, ~3x the measured
 * p50 190 ms / p95 215 ms of `checkDenseReadiness()`), the module memo engages
 * and later evaluations consult only the state file so a slow probe can never
 * tax the search path twice.
 *
 * All external effects are injectable seams (`probe`, `now`, `stateDir`,
 * `spawner`) so tests stay hermetic; defaults wrap the real
 * `checkDenseReadiness()`, `Date.now`, `rag-index/data`, and a detached spawn
 * of `cortex-self-heal.mjs`.
 */
import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { open, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');

/**
 * Parse an environment variable as a non-negative integer, falling back to the
 * default for missing, malformed, or negative values (same convention as
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

// Knobs are read once at import time so tests can isolate them per module
// instance (fresh import after `jest.resetModules()`).
const DEFAULT_COOLDOWN_S = 600;
const DEFAULT_BACKOFF_FACTOR = 2;
const DEFAULT_MAX_ATTEMPTS = 3;
const DEFAULT_ATTEMPT_WINDOW_S = 86_400;
const DEFAULT_PROBE_COST_THRESHOLD_MS = 600;

const MS_PER_SECOND = 1000;
const MS_PER_MINUTE = 60 * MS_PER_SECOND;
const MS_PER_HOUR = 60 * MS_PER_MINUTE;
const SECONDS_PER_MINUTE = 60;

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
/** Sliding window for counting attempts, in milliseconds (`CORTEX_SELFHEAL_ATTEMPT_WINDOW_S`). */
const ATTEMPT_WINDOW_MS =
  parseEnvInt('CORTEX_SELFHEAL_ATTEMPT_WINDOW_S', DEFAULT_ATTEMPT_WINDOW_S) *
  MS_PER_SECOND;
/** Probe-cost budget before the state-file-only fallback (`CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS`). */
const PROBE_COST_THRESHOLD_MS = parseEnvInt(
  'CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS',
  DEFAULT_PROBE_COST_THRESHOLD_MS,
);
/** Kill switch — `1` disables all triggering; read-side warnings still work. */
const KILL_SWITCH_ON = parseEnvInt('CORTEX_SELFHEAL_DISABLE', 0) === 1;

/** Shared state file name (single writer discipline; guard writes only on trigger). */
const STATE_FILE_NAME = 'cortex-self-heal-state.json';
/** Repair lock file name (mutual exclusion across repair-capable surfaces). */
const LOCK_FILE_NAME = 'cortex-self-heal.repair.lock';
const STATE_VERSION = 1;

/**
 * Chunk-count heuristic for repair-duration estimates: observed prewarm rate
 * (~25k chunks in ~5 min during the incident that motivated this plan).
 */
const CHUNKS_PER_MINUTE_HEURISTIC = 5000;
/** Fallback chunk count for the estimate when the probe reports no count. */
const FALLBACK_ESTIMATE_CHUNK_COUNT = 5000;

/** Exact manual recovery commands (T3 / disabled guidance and `manual_recovery`). */
const MANUAL_RECOVERY_COMMANDS = [
  'npm run index:session-start',
  'npm run index:prewarm',
  'node scripts/agent-customization/gates/cortex-index.gate.mjs --auto-rebuild --json',
];

/** One-line repair sequence summary rendered into T1 guidance. */
const REPAIR_SEQUENCE_SUMMARY =
  'classifier-first repair (validate-index, then targeted build/snapshot/prewarm, then cortex-index and cortex-mcp-smoke gates)';

const DENSE_READINESS_MODULE_PATH = path.resolve(
  repoRoot,
  'rag-index',
  'dense-readiness.mjs',
);
const ORCHESTRATOR_SCRIPT_PATH = path.join(__dirname, 'cortex-self-heal.mjs');
const DEFAULT_STATE_DIR = path.resolve(repoRoot, 'rag-index', 'data');

/** Warm-path decision: nothing to do, nothing to say, nothing to spawn. */
const WARM_DECISION = Object.freeze({
  action: null,
  guidanceFields: null,
  spawnDecision: null,
});

/**
 * Per-process probe-cost memo. Set when a measured probe exceeds the cost
 * threshold; later evaluations then consult only the state file. Isolated per
 * module instance (tests re-import with `jest.resetModules()`).
 *
 * @type {boolean}
 */
let probeCostMemoEngaged = false;

/**
 * Default probe adapter: wraps `checkDenseReadiness()` from
 * `rag-index/dense-readiness.mjs`. Imported lazily so the search path (and the
 * test suite) never pays for the module graph (libsql client) until a real
 * probe is needed.
 *
 * @returns {Promise<{ready: boolean, state: string, reason: string, chunk_count: number|null, embedding_count: number|null}>} Readiness report.
 */
async function defaultProbe() {
  const { checkDenseReadiness } = await import(
    pathToFileURL(DENSE_READINESS_MODULE_PATH).href
  );
  return checkDenseReadiness();
}

/**
 * Default spawner: detach the repair orchestrator CLI so the guard returns
 * immediately while repair continues in the background.
 *
 * @param {{ stateDir: string }} context - State directory hand-off for the orchestrator.
 * @returns {{ pid: number, command: string }} Spawned process descriptor.
 */
function defaultSpawner({ stateDir }) {
  const args = [ORCHESTRATOR_SCRIPT_PATH, '--state-dir', stateDir];
  const child = spawn(process.execPath, args, {
    detached: true,
    stdio: 'ignore',
  });
  child.unref();
  return { pid: child.pid, command: `${process.execPath} ${args.join(' ')}` };
}

/**
 * Decide whether a degraded dense-readiness signal should trigger repair now.
 *
 * Decision order: warm short-circuit → kill switch → in-flight → cooldown →
 * attempt ceiling → trigger. On trigger the guard reserves the repair lock,
 * spawns the orchestrator via the spawner seam, persists the lock record and
 * the updated state file (both under the lock), and returns T1 guidance. All
 * other degraded outcomes return the matching structured guidance block
 * without side effects.
 *
 * @param {{
 *   probe?: () => Promise<{ready: boolean, state: string, reason: string, chunk_count: number|null, embedding_count: number|null}>,
 *   now?: () => number,
 *   stateDir?: string,
 *   spawner?: (context: {stateDir: string, attempt: number, reason: string}) => Promise<{pid: number, command?: string}>,
 * }} [seams] - Injectable external effects (clock, probe, state dir, spawner).
 * @returns {Promise<{action: string|null, guidanceFields: object|null, spawnDecision: object|null}>} Decision: `null` fields on the warm path; otherwise `action` is one of `started` | `in_flight` | `cooldown` | `exhausted` | `disabled` with a structured `self_heal` guidance block in `guidanceFields` and, on `started`, advisory spawn metadata in `spawnDecision`.
 *
 * @example
 * ```js
 * import { evaluateSelfHeal } from './cortex-health-guard.mjs';
 * const decision = await evaluateSelfHeal();
 * if (decision.action === 'started') console.log('repair pid:', decision.spawnDecision.pid);
 * ```
 */
export async function evaluateSelfHeal(seams = {}) {
  const now = seams.now ?? Date.now;
  const stateDir = seams.stateDir ?? DEFAULT_STATE_DIR;
  const probe = seams.probe ?? defaultProbe;
  const spawner = seams.spawner ?? defaultSpawner;

  let liveReport = null;
  let probedAt = null;
  if (!probeCostMemoEngaged) {
    const probeStartedAt = now();
    liveReport = await probe();
    probedAt = now();
    if (probedAt - probeStartedAt > PROBE_COST_THRESHOLD_MS) {
      probeCostMemoEngaged = true;
    }
    if (isWarmReport(liveReport)) {
      return WARM_DECISION;
    }
  }

  const state = await loadState(stateDir);
  const signal = deriveSignal(liveReport, probedAt, state);
  if (signal === null) {
    return WARM_DECISION;
  }

  if (KILL_SWITCH_ON) {
    return buildDisabledDecision(state, signal);
  }
  if (state.in_flight !== null) {
    return buildInFlightDecision(state, signal, now());
  }
  const blockedUntil = cooldownBlockedUntil(state);
  if (blockedUntil !== null && now() < blockedUntil) {
    return buildCooldownDecision(state, signal, blockedUntil);
  }
  const attemptWindow = resolveAttemptWindow(state, now());
  if (attemptWindow.attempts >= MAX_ATTEMPTS) {
    return buildExhaustedDecision(state, signal, attemptWindow);
  }
  return triggerRepair({
    state,
    signal,
    stateDir,
    spawner,
    now,
    attemptWindow,
  });
}

/**
 * Trigger the repair: reserve the lock, spawn the orchestrator, persist the
 * lock record and the updated state file, and return the T1 decision.
 *
 * @param {{
 *   state: object,
 *   signal: object,
 *   stateDir: string,
 *   spawner: Function,
 *   now: () => number,
 *   attemptWindow: {attempts: number, windowExpired: boolean},
 * }} context - Everything the trigger sequence needs.
 * @returns {Promise<{action: string, guidanceFields: object, spawnDecision: object}>} `started` decision.
 */
async function triggerRepair(context) {
  const { state, signal, stateDir, spawner, now, attemptWindow } = context;
  const decisionAt = now();
  const attempt = attemptWindow.attempts + 1;
  const estimate = estimateRepairDuration(signal, state.history);
  const backoffDelayMs =
    COOLDOWN_S * MS_PER_SECOND * BACKOFF_FACTOR ** (attempt - 1);
  const nextAllowedAt = decisionAt + backoffDelayMs;
  const lockPath = path.join(stateDir, LOCK_FILE_NAME);

  let lockHandle;
  try {
    lockHandle = await open(lockPath, 'wx');
  } catch (error) {
    if (error && error.code === 'EEXIST') {
      // A concurrent repair-capable surface won the trigger race.
      return buildInFlightDecision(state, signal, now());
    }
    throw error;
  }

  let spawnSucceeded = false;
  try {
    const spawnResult = await spawner({
      stateDir,
      attempt,
      reason: signal.state,
    });
    spawnSucceeded = true;
    const triggerAt = now();
    const pid = extractPid(spawnResult);
    await lockHandle.writeFile(
      JSON.stringify(
        {
          pid,
          started_at: triggerAt,
          heartbeat_at: triggerAt,
          attempt,
          reason: signal.state,
        },
        null,
        2,
      ),
      'utf8',
    );
    await lockHandle.close();

    await persistState(
      stateDir,
      buildTriggeredState({
        state,
        signal,
        pid,
        attempt,
        triggerAt,
        nextAllowedAt,
        estimate,
        attemptWindow,
      }),
    );

    return {
      action: 'started',
      guidanceFields: buildStartedGuidanceFields({
        signal,
        attempt,
        estimate,
        nextAllowedAt,
      }),
      spawnDecision: { pid, command: extractCommand(spawnResult) },
    };
  } catch (error) {
    await lockHandle.close().catch(() => {});
    if (!spawnSucceeded) {
      // Release the reservation; nothing was spawned and no state changed.
      await unlink(lockPath).catch(() => {});
    }
    throw error;
  }
}

/**
 * Read the shared state file, falling back to fresh defaults on any read or
 * parse failure so the search path never throws on a missing/corrupt file.
 *
 * @param {string} stateDir - Directory holding the state file.
 * @returns {Promise<object>} Normalized state in the plan schema.
 */
async function loadState(stateDir) {
  try {
    const content = await readFile(
      path.join(stateDir, STATE_FILE_NAME),
      'utf8',
    );
    return normalizeState(JSON.parse(content));
  } catch {
    return freshState();
  }
}

/**
 * Persist the state file atomically (write-then-rename, same convention as
 * `validate-index.mjs` freshness manifests).
 *
 * @param {string} stateDir - Directory holding the state file.
 * @param {object} state - Full state in the plan schema.
 * @returns {Promise<void>} Resolves once the rename completed.
 */
async function persistState(stateDir, state) {
  const statePath = path.join(stateDir, STATE_FILE_NAME);
  const tempPath = `${statePath}.${randomBytes(8).toString('hex')}.tmp`;
  await writeFile(tempPath, JSON.stringify(state, null, 2), 'utf8');
  await rename(tempPath, statePath);
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
 * Coerce a raw (possibly partial/corrupt) state file into the plan schema.
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
 * Build the normalized degradation signal the decision engine and guidance
 * render from — either the live probe report or, when the probe-cost memo is
 * engaged, the state file's `last_probe` record.
 *
 * @param {object|null} liveReport - Live probe report (null on the memo path).
 * @param {number|null} probedAt - Clock reading taken after the live probe.
 * @param {object} state - Normalized shared state.
 * @returns {object|null} Signal, or `null` when no degradation evidence exists.
 */
function deriveSignal(liveReport, probedAt, state) {
  if (liveReport !== null) {
    return {
      state:
        typeof liveReport.state === 'string' ? liveReport.state : 'unknown',
      reason:
        typeof liveReport.reason === 'string'
          ? liveReport.reason
          : 'Dense readiness probe reported no reason.',
      chunk_count: finiteOrNull(liveReport.chunk_count),
      embedding_count: finiteOrNull(liveReport.embedding_count),
      probed_at: probedAt,
    };
  }
  const lastProbe = state.last_probe;
  if (lastProbe === null || lastProbe.state === 'warm') return null;
  return {
    state: lastProbe.state,
    reason: lastProbe.dense_reason,
    chunk_count: finiteOrNull(lastProbe.chunk_count),
    embedding_count: finiteOrNull(lastProbe.embedding_count),
    probed_at: finiteOrNull(lastProbe.probed_at),
  };
}

/**
 * @param {object|null} report - Probe report.
 * @returns {boolean} `true` when dense search is warm (no fs activity needed).
 */
function isWarmReport(report) {
  return (
    isPlainObject(report) && (report.ready === true || report.state === 'warm')
  );
}

/**
 * Resolve the attempt count in the current window, resetting the window when
 * it has elapsed.
 *
 * @param {object} state - Normalized shared state.
 * @param {number} nowMs - Current clock reading.
 * @returns {{attempts: number, windowExpired: boolean}} Effective attempts.
 */
function resolveAttemptWindow(state, nowMs) {
  const { window_started_at: windowStartedAt, attempts_in_window: attempts } =
    state.backoff;
  if (windowStartedAt === null || nowMs - windowStartedAt > ATTEMPT_WINDOW_MS) {
    return { attempts: 0, windowExpired: true };
  }
  return { attempts, windowExpired: false };
}

/**
 * Compute the timestamp until which re-triggering is blocked (cooldown after
 * the last trigger, backoff `next_allowed_at`, whichever is later).
 *
 * @param {object} state - Normalized shared state.
 * @returns {number|null} Blocked-until timestamp, or `null` when not blocked.
 */
function cooldownBlockedUntil(state) {
  const candidates = [];
  if (state.cooldown.last_trigger_at !== null) {
    candidates.push(
      state.cooldown.last_trigger_at + COOLDOWN_S * MS_PER_SECOND,
    );
  }
  if (state.backoff.next_allowed_at > 0) {
    candidates.push(state.backoff.next_allowed_at);
  }
  if (candidates.length === 0) return null;
  return Math.max(...candidates);
}

/**
 * Estimate the repair duration from successful history runs, falling back to
 * the chunk-count heuristic.
 *
 * @param {object} signal - Normalized degradation signal.
 * @param {Array<object>} history - Recorded repair runs (last 5, per plan).
 * @returns {{minutes: number, basis: string}} Estimate and its basis label.
 */
function estimateRepairDuration(signal, history) {
  const successfulDurations = (Array.isArray(history) ? history : [])
    .filter(
      (entry) =>
        isPlainObject(entry) &&
        entry.success === true &&
        Number.isFinite(entry.duration_min),
    )
    .map((entry) => entry.duration_min);
  if (successfulDurations.length > 0) {
    const average =
      successfulDurations.reduce((total, value) => total + value, 0) /
      successfulDurations.length;
    return { minutes: Math.max(1, Math.round(average)), basis: 'history' };
  }
  const chunkCount =
    signal.chunk_count !== null && signal.chunk_count > 0
      ? signal.chunk_count
      : FALLBACK_ESTIMATE_CHUNK_COUNT;
  return {
    minutes: Math.max(1, Math.ceil(chunkCount / CHUNKS_PER_MINUTE_HEURISTIC)),
    basis: 'chunk-count heuristic',
  };
}

/**
 * Build the on-disk state after a successful trigger (guard writes only while
 * it owns the lock; the orchestrator appends `history` later).
 *
 * @param {{state: object, signal: object, pid: number, attempt: number, triggerAt: number, nextAllowedAt: number, estimate: object, attemptWindow: object}} parts - Trigger facts.
 * @returns {object} Full state in the plan schema.
 */
function buildTriggeredState(parts) {
  const { state, signal, pid, attempt, triggerAt, nextAllowedAt, estimate } =
    parts;
  return {
    version: state.version,
    last_probe: {
      state: signal.state,
      dense_reason: signal.reason,
      chunk_count: signal.chunk_count,
      embedding_count: signal.embedding_count,
      probed_at: signal.probed_at,
    },
    cooldown: { cooldown_s: COOLDOWN_S, last_trigger_at: triggerAt },
    backoff: {
      attempts_in_window: attempt,
      window_started_at:
        parts.attemptWindow.windowExpired ||
        state.backoff.window_started_at === null
          ? triggerAt
          : state.backoff.window_started_at,
      backoff_factor: BACKOFF_FACTOR,
      next_allowed_at: nextAllowedAt,
    },
    max_attempts: MAX_ATTEMPTS,
    in_flight: {
      pid,
      started_at: triggerAt,
      attempt,
      reason: signal.state,
      est_duration_min: estimate.minutes,
    },
    history: state.history,
  };
}

/**
 * T2 decision: a repair is already in flight (or a concurrent surface just won
 * the trigger race); no re-trigger, no side effects.
 *
 * @param {object} state - Normalized shared state.
 * @param {object} signal - Normalized degradation signal.
 * @param {number} nowMs - Current clock reading.
 * @returns {{action: string, guidanceFields: object, spawnDecision: null}} Decision.
 */
function buildInFlightDecision(state, signal, nowMs) {
  const inFlight = state.in_flight ?? {};
  const startedAt = finiteOrNull(inFlight.started_at);
  const elapsedMin =
    startedAt !== null
      ? Math.max(0, Math.round((nowMs - startedAt) / MS_PER_MINUTE))
      : null;
  const recordedEstimate = finiteOrNull(inFlight.est_duration_min);
  const estMin =
    recordedEstimate !== null
      ? recordedEstimate
      : estimateRepairDuration(signal, state.history).minutes;
  const remainingMin =
    elapsedMin !== null ? Math.max(0, estMin - elapsedMin) : estMin;
  const attempt = finiteOrNull(inFlight.attempt);
  const nextAllowedAt =
    state.cooldown.last_trigger_at !== null
      ? state.cooldown.last_trigger_at + COOLDOWN_S * MS_PER_SECOND
      : null;
  return {
    action: 'in_flight',
    guidanceFields: {
      state: signal.state,
      reason: signal.reason,
      action: 'in_flight',
      attempt,
      max_attempts: MAX_ATTEMPTS,
      cooldown_s: COOLDOWN_S,
      next_allowed_at: nextAllowedAt,
      est_duration_min: remainingMin,
      manual_recovery: null,
      guidance:
        `Cortex dense search is degraded (${signal.reason}) and self-heal is ALREADY RUNNING ` +
        `(started ${elapsedMin ?? 'an unknown number of'} min ago, attempt ${attempt ?? '?'}/${MAX_ATTEMPTS}, ` +
        `est. remaining ~${remainingMin} min). BM25-only results are being returned — continue with ` +
        'reduced recall or fall back to native tools. No re-trigger needed; results recover ' +
        'automatically when repair completes. Wait and re-try later rather than spawning new repair work.',
    },
    spawnDecision: null,
  };
}

/**
 * Cooldown decision: a repair was triggered recently and the cooldown (or
 * backoff delay) has not elapsed; no re-trigger, no side effects.
 *
 * @param {object} state - Normalized shared state.
 * @param {object} signal - Normalized degradation signal.
 * @param {number} blockedUntil - Timestamp until which triggering is blocked.
 * @returns {{action: string, guidanceFields: object, spawnDecision: null}} Decision.
 */
function buildCooldownDecision(state, signal, blockedUntil) {
  const attempts = state.backoff.attempts_in_window;
  const cooldownMinutes = Math.round(COOLDOWN_S / SECONDS_PER_MINUTE);
  return {
    action: 'cooldown',
    guidanceFields: {
      state: signal.state,
      reason: signal.reason,
      action: 'cooldown',
      attempt: attempts,
      max_attempts: MAX_ATTEMPTS,
      cooldown_s: COOLDOWN_S,
      next_allowed_at: blockedUntil,
      est_duration_min: null,
      manual_recovery: null,
      guidance:
        `Cortex dense search is degraded (${signal.reason}) and self-heal is in cooldown: a repair ` +
        `was triggered recently and the ${cooldownMinutes} min cooldown has not elapsed ` +
        `(attempt ${attempts} of ${MAX_ATTEMPTS} in the current window; next automatic attempt at ` +
        `${new Date(blockedUntil).toISOString()}). BM25-only results are being returned — continue with ` +
        'reduced recall or fall back to native tools and re-try later. Do NOT re-trigger repair or ' +
        'spawn new repair work.',
    },
    spawnDecision: null,
  };
}

/**
 * T3 decision: the attempt ceiling is reached — pause-and-ask with the exact
 * manual recovery commands; no re-trigger, no side effects.
 *
 * @param {object} state - Normalized shared state.
 * @param {object} signal - Normalized degradation signal.
 * @param {{attempts: number}} attemptWindow - Effective attempt window.
 * @returns {{action: string, guidanceFields: object, spawnDecision: null}} Decision.
 */
function buildExhaustedDecision(state, signal, attemptWindow) {
  const history = Array.isArray(state.history) ? state.history : [];
  const lastFailure = history.findLast(
    (entry) => isPlainObject(entry) && entry.success === false,
  );
  const shortError =
    isPlainObject(lastFailure) && typeof lastFailure.error === 'string'
      ? lastFailure.error
      : 'unknown';
  const windowHours = Math.round(ATTEMPT_WINDOW_MS / MS_PER_HOUR);
  const windowStartedAt = state.backoff.window_started_at;
  const nextAllowedAt =
    windowStartedAt !== null ? windowStartedAt + ATTEMPT_WINDOW_MS : null;
  return {
    action: 'exhausted',
    guidanceFields: {
      state: signal.state,
      reason: signal.reason,
      action: 'exhausted',
      attempt: attemptWindow.attempts,
      max_attempts: MAX_ATTEMPTS,
      cooldown_s: COOLDOWN_S,
      next_allowed_at: nextAllowedAt,
      est_duration_min: null,
      manual_recovery: MANUAL_RECOVERY_COMMANDS,
      guidance:
        `Cortex dense search is degraded (${signal.reason}) and automatic self-heal has FAILED ` +
        `${MAX_ATTEMPTS} times in the last ${windowHours} h (last failure: ${shortError}). Auto-repair ` +
        'is paused to avoid a blocker loop. PAUSE your work on RAG-dependent steps and ask the user ' +
        'to run `npm run index:session-start`, then `npm run index:prewarm`, then `node ' +
        'scripts/agent-customization/gates/cortex-index.gate.mjs --auto-rebuild --json`. ' +
        'BM25-only results are being returned meanwhile.',
    },
    spawnDecision: null,
  };
}

/**
 * Kill-switch decision: triggering and spawning are blocked, degraded
 * read-side guidance still reports, and manual recovery is offered.
 *
 * @param {object} state - Normalized shared state.
 * @param {object} signal - Normalized degradation signal.
 * @returns {{action: string, guidanceFields: object, spawnDecision: null}} Decision.
 */
function buildDisabledDecision(state, signal) {
  const manualRecovery = MANUAL_RECOVERY_COMMANDS.join('`, then `');
  return {
    action: 'disabled',
    guidanceFields: {
      state: signal.state,
      reason: signal.reason,
      action: 'disabled',
      attempt: state.backoff.attempts_in_window,
      max_attempts: MAX_ATTEMPTS,
      cooldown_s: COOLDOWN_S,
      next_allowed_at: null,
      est_duration_min: null,
      manual_recovery: MANUAL_RECOVERY_COMMANDS,
      guidance:
        `Cortex dense search is degraded (${signal.reason}; state=${signal.state}) and automatic ` +
        'self-heal is DISABLED via CORTEX_SELFHEAL_DISABLE=1, so no repair was started. ' +
        'BM25-only results are being returned — continue with reduced recall or fall back to ' +
        `native tools (grep/glob/view). Manual recovery when convenient: run \`${manualRecovery}\`.`,
    },
    spawnDecision: null,
  };
}

/**
 * T1 decision fields after a successful trigger.
 *
 * @param {{signal: object, attempt: number, estimate: object, nextAllowedAt: number}} parts - Trigger facts.
 * @returns {object} Structured `self_heal` guidance block.
 */
function buildStartedGuidanceFields(parts) {
  const { signal, attempt, estimate, nextAllowedAt } = parts;
  const cooldownMinutes = Math.round(COOLDOWN_S / SECONDS_PER_MINUTE);
  const chunkCount =
    signal.chunk_count !== null ? signal.chunk_count : 'unknown';
  return {
    state: signal.state,
    reason: signal.reason,
    action: 'started',
    attempt,
    max_attempts: MAX_ATTEMPTS,
    cooldown_s: COOLDOWN_S,
    next_allowed_at: nextAllowedAt,
    est_duration_min: estimate.minutes,
    manual_recovery: null,
    guidance:
      `Cortex dense search is degraded (${signal.reason}; state=${signal.state}). Background ` +
      `self-heal has been STARTED: ${REPAIR_SEQUENCE_SUMMARY} (estimated ~${estimate.minutes} min ` +
      `for ${chunkCount} chunks; estimate based on ${estimate.basis}). BM25-only results are being ` +
      'returned below — continue with reduced recall or fall back to native tools (grep/glob/view) ' +
      'and re-try Cortex after the repair completes. Do NOT re-trigger repair: a cooldown of ' +
      `${cooldownMinutes} min applies (attempt ${attempt} of ${MAX_ATTEMPTS} in the current window). ` +
      `If automatic repair fails ${MAX_ATTEMPTS} times, pause and ask the user to run ` +
      '`npm run index:session-start` followed by `npm run index:prewarm`, then re-run the Cortex gate.',
  };
}

/**
 * @param {object|null} spawnResult - Spawner return value.
 * @returns {number|null} Spawned pid when reported.
 */
function extractPid(spawnResult) {
  return isPlainObject(spawnResult) && Number.isFinite(spawnResult.pid)
    ? spawnResult.pid
    : null;
}

/**
 * @param {object|null} spawnResult - Spawner return value.
 * @returns {string|null} Spawned command when reported.
 */
function extractCommand(spawnResult) {
  return isPlainObject(spawnResult) && typeof spawnResult.command === 'string'
    ? spawnResult.command
    : null;
}

/**
 * @param {*} value - Candidate value.
 * @returns {number|null} The finite number, or `null`.
 */
function finiteOrNull(value) {
  return Number.isFinite(value) ? value : null;
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
 * @returns {boolean} `true` for non-null plain objects.
 */
function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
