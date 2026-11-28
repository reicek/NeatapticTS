#!/usr/bin/env node
/**
 * Shared wall-clock budgets for the NeatapticTS Tier-1 gate tooling.
 *
 * This module is the single source of truth for every timeout ceiling and
 * every output-buffer limit the gate stack applies, so the outer ceilings
 * can never drift below the inner ones they wrap.
 *
 * Why a shared module
 * -------------------
 * The gate stack nests three levels of wall-clock ceilings:
 *
 *   1. `neataptic-gate-mcp.mjs` spawns ONE gate script
 *      (outer spawn budget).
 *   2. `slice-advancement.gate.mjs` spawns each sub-gate sequentially
 *      (per-sub-gate budget).
 *   3. `shared-validation.gate.mjs` spawns its runners (jest, build, lint)
 *      (per-runner backstop).
 *
 * The defect this module exists to prevent: an outer ceiling that is not
 * larger than the work it wraps. When the outer ceiling fires, the child is
 * killed mid-run, its stdout is truncated or empty, and the caller cannot
 * parse a gate contract — it reports the opaque string
 * "Gate '<id>' did not return valid JSON" and hides the real cause (a
 * timeout). The same inversion existed one level down, where the
 * per-sub-gate ceiling was not larger than the shared-validation workload it
 * wrapped.
 *
 * Sizing rules encoded here
 * -------------------------
 * - A budget that WRAPS other budgets must exceed their SUM plus overhead,
 *   so the inner layer always terminates first and reports a clean, specific
 *   failure instead of being truncated from the outside.
 * - The slice-advancement budget must exceed the SUM of its sub-gate budgets
 *   because the sub-gates run sequentially.
 * - Every gate the facade can spawn has an EXPLICIT budget. There is no
 *   silent fallback for a gate that someone forgot to size — the fallback is
 *   only a safety net and `assertEveryGateHasBudget` proves coverage.
 * - Every spawn that carries a gate contract sets an explicit `maxBuffer`.
 *   Node's default (1 MiB) silently truncates a large gate contract, which
 *   produces the SAME opaque "did not return valid JSON" symptom as a
 *   timeout, for a completely different reason.
 */

/** Milliseconds in one second, for readable budget arithmetic. */
export const MILLISECONDS_PER_SECOND = 1_000;

/**
 * Maximum bytes a gate spawn may buffer.
 *
 * Node's `child_process` default is 1 MiB. A gate contract embeds the
 * captured stdout/stderr of its runners (jest reports, webpack output), so a
 * large-but-legitimate run can exceed 1 MiB and be truncated — producing the
 * SAME opaque "did not return valid JSON" symptom as a timeout, for a
 * completely different reason. 16 MiB leaves ample headroom for the gate
 * contracts this stack produces.
 */
export const GATE_MAX_BUFFER_BYTES = 16 * 1024 * 1024;

/**
 * Per-runner backstops enforced INSIDE `shared-validation.gate.mjs`.
 *
 * These are the innermost ceilings: each runner is killed and reported as a
 * clean runner failure once its own budget expires.
 */
export const SHARED_VALIDATION_RUNNER_TIMEOUTS = Object.freeze({
  tests: 650_000,
  build: 300_000,
  lint: 300_000,
});

/**
 * Sum of the shared-validation runner backstops.
 *
 * The outer shared-validation budget MUST be larger than this value,
 * otherwise the outer kill truncates a run whose inner backstop would have
 * produced a specific failure report.
 */
export const SHARED_VALIDATION_RUNNER_TOTAL_MS =
  SHARED_VALIDATION_RUNNER_TIMEOUTS.tests +
  SHARED_VALIDATION_RUNNER_TIMEOUTS.build +
  SHARED_VALIDATION_RUNNER_TIMEOUTS.lint;

/**
 * Outer budget for the `shared-validation` sub-gate.
 *
 * Sized as runner-total (1 250 000 ms) plus ~20 % headroom for process
 * startup, artifact I/O, and scheduler contention.
 */
export const SHARED_VALIDATION_TIMEOUT_MS = 1_500_000;

/**
 * Default budget for a sub-gate that only reads files.
 *
 * Used when a gate is added to the consolidated set without an explicit
 * budget; `assertEveryGateHasBudget` keeps that from silently shipping.
 */
export const DEFAULT_SUB_GATE_TIMEOUT_MS = 120_000;

/**
 * Per-sub-gate budgets applied by `slice-advancement.gate.mjs`.
 *
 * The heavy gate is `shared-validation` (jest + build + lint); the rest are
 * file-reading gates that finish in seconds.
 */
export const SUB_GATE_TIMEOUT_BUDGETS = Object.freeze({
  'plan-sync': 60_000,
  'step-packet': 60_000,
  'plan-slice-quality': 60_000,
  'plan-command-lint': 180_000,
  'shared-validation': SHARED_VALIDATION_TIMEOUT_MS,
  'code-coverage': 180_000,
  'specialist-review': 120_000,
});

/** Trivial-slice sub-gates — always run. */
export const TRIVIAL_SUB_GATES = Object.freeze([
  'plan-sync',
  'step-packet',
  'plan-slice-quality',
  'plan-command-lint',
]);

/** Full-slice sub-gates — run in addition to the trivial set. */
export const FULL_SUB_GATES = Object.freeze([
  'shared-validation',
  'code-coverage',
  'specialist-review',
]);

/**
 * Every sub-gate the consolidated `slice-advancement` gate can spawn, in
 * execution order. FULL gates are appended after the TRIVIAL gates.
 */
export const SLICE_ADVANCEMENT_SUB_GATES = Object.freeze([
  ...TRIVIAL_SUB_GATES,
  ...FULL_SUB_GATES,
]);

/**
 * Slack added to any budget that WRAPS other budgets, covering process
 * startup, JSON serialization, and artifact writes.
 */
export const GATE_TIMEOUT_OVERHEAD_MS = 60_000;

/**
 * Resolves the budget for a single sub-gate.
 *
 * @param {string} gateName - Sub-gate name (for example `shared-validation`).
 * @returns {number} Budget in milliseconds.
 */
export function resolveSubGateTimeoutMs(gateName) {
  return SUB_GATE_TIMEOUT_BUDGETS[gateName] ?? DEFAULT_SUB_GATE_TIMEOUT_MS;
}

/**
 * Computes the outer budget for the consolidated `slice-advancement` gate.
 *
 * The sub-gates run SEQUENTIALLY, so the outer budget must cover the sum of
 * their budgets plus overhead.
 *
 * @returns {number} Budget in milliseconds.
 */
export function computeSliceAdvancementTimeoutMs() {
  const subGateTotal = SLICE_ADVANCEMENT_SUB_GATES.reduce(
    (total, gateName) => total + resolveSubGateTimeoutMs(gateName),
    0,
  );
  return subGateTotal + GATE_TIMEOUT_OVERHEAD_MS;
}

/**
 * Outer budget for the consolidated `slice-advancement` gate.
 *
 * Currently 2 220 000 ms (37 min) — a CEILING, not an expectation. With the
 * scoped runners in `shared-validation.gate.mjs` the consolidated gate
 * normally completes in tens of seconds; the ceiling exists so a legitimate
 * slow run is never killed and mis-reported as invalid JSON.
 */
export const SLICE_ADVANCEMENT_TIMEOUT_MS = computeSliceAdvancementTimeoutMs();

/**
 * Default budget for a gate the facade can spawn directly.
 */
export const DEFAULT_GATE_TIMEOUT_MS = 120_000;

/**
 * Explicit budgets for every gate the facade can spawn.
 *
 * `slice-advancement` is derived (see above); every other entry is a
 * deliberate, hand-sized value.
 */
export const GATE_TIMEOUT_BUDGETS = Object.freeze({
  'plan-sync': 60_000,
  'step-packet': 60_000,
  'plan-slice-quality': 60_000,
  'plan-command-lint': 180_000,
  'agent-graph': 120_000,
  'agent-quality': 120_000,
  'tier-enforcement': 120_000,
  'routing-table-freshness': 120_000,
  'learning-event': 60_000,
  'stale-wip-plans': 60_000,
  'cortex-index': 180_000,
  'cortex-first-search': 180_000,
  'devtools-coverage': 120_000,
  'delegate-skill-coverage': 120_000,
  'code-coverage': 180_000,
  'specialist-review': 120_000,
  'slice-advancement': SLICE_ADVANCEMENT_TIMEOUT_MS,
});

/**
 * Environment variable an operator can set to RAISE the outer spawn budget.
 *
 * The budget table is sized for a typical workstation. A slower CI runner
 * (or a fully loaded machine) can need more headroom, and the operator should
 * not have to edit a source file to get it. The override is RAISE-ONLY: a
 * value below the built-in budget is ignored, because lowering a ceiling
 * recreates exactly the truncation defect this module exists to prevent.
 */
export const GATE_TIMEOUT_ENV_VAR = 'NEATAPTIC_GATE_TIMEOUT_MS';

/**
 * Reads the environment override, ignoring malformed or non-positive values.
 *
 * @param {NodeJS.ProcessEnv} [env] - Environment to read (defaults to `process.env`).
 * @returns {number} The override in milliseconds, or 0 when unset/invalid.
 */
function readTimeoutOverride(env = process.env) {
  const raw = env[GATE_TIMEOUT_ENV_VAR];
  if (typeof raw !== 'string' || raw.trim() === '') return 0;
  const parsed = Number.parseInt(raw.trim(), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return 0;
  return parsed;
}

/**
 * Resolves the outer spawn budget for a gate.
 *
 * The built-in budget applies unless `NEATAPTIC_GATE_TIMEOUT_MS` asks for a
 * LARGER one (see `GATE_TIMEOUT_ENV_VAR`).
 *
 * @param {string} gateId - Gate identifier (for example `slice-advancement`).
 * @param {NodeJS.ProcessEnv} [env] - Environment to read (defaults to `process.env`).
 * @returns {number} Budget in milliseconds.
 */
export function resolveGateTimeoutMs(gateId, env = process.env) {
  const base = GATE_TIMEOUT_BUDGETS[gateId] ?? DEFAULT_GATE_TIMEOUT_MS;
  return Math.max(base, readTimeoutOverride(env));
}

/**
 * Checks that every gate the facade exposes has an explicit budget.
 *
 * Guards against the silent-fallback failure mode: a gate added to the
 * facade without a budget would otherwise inherit a default that may be too
 * small, reintroducing exactly the defect this module prevents.
 *
 * @param {string[]} gateIds - Gate identifiers exposed by the facade.
 * @returns {string[]} Names of gates missing an explicit budget (empty when complete).
 */
export function findGatesWithoutBudget(gateIds) {
  return gateIds.filter((gateId) => !(gateId in GATE_TIMEOUT_BUDGETS));
}

/**
 * Renders the budget table for human-readable diagnostics.
 *
 * @returns {object} Structured budget report plus a rendered text summary.
 */
export function describeTimeoutBudgets() {
  const subGateSum = SLICE_ADVANCEMENT_SUB_GATES.reduce(
    (total, gateName) => total + resolveSubGateTimeoutMs(gateName),
    0,
  );
  const envOverrideMs = readTimeoutOverride();

  const lines = [
    'Gate timeout budgets (ms):',
    ...Object.entries(GATE_TIMEOUT_BUDGETS)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(
        ([gateId, budget]) =>
          `  ${gateId}: ${budget}${envOverrideMs > budget ? ` (raised to ${envOverrideMs} by ${GATE_TIMEOUT_ENV_VAR})` : ''}`,
      ),
    '',
    `shared-validation runner backstops: tests=${SHARED_VALIDATION_RUNNER_TIMEOUTS.tests}, build=${SHARED_VALIDATION_RUNNER_TIMEOUTS.build}, lint=${SHARED_VALIDATION_RUNNER_TIMEOUTS.lint} (sum=${SHARED_VALIDATION_RUNNER_TOTAL_MS})`,
    `shared-validation outer budget: ${SHARED_VALIDATION_TIMEOUT_MS}`,
    `slice-advancement sub-gate sum: ${subGateSum} + overhead ${GATE_TIMEOUT_OVERHEAD_MS} = ${SLICE_ADVANCEMENT_TIMEOUT_MS}`,
    `gate spawn maxBuffer: ${GATE_MAX_BUFFER_BYTES} bytes`,
    `environment override (${GATE_TIMEOUT_ENV_VAR}): ${envOverrideMs === 0 ? 'not set' : envOverrideMs}`,
  ];

  return {
    gateBudgets: { ...GATE_TIMEOUT_BUDGETS },
    subGateBudgets: { ...SUB_GATE_TIMEOUT_BUDGETS },
    sharedValidationRunnerTimeouts: { ...SHARED_VALIDATION_RUNNER_TIMEOUTS },
    sharedValidationRunnerTotalMs: SHARED_VALIDATION_RUNNER_TOTAL_MS,
    sharedValidationTimeoutMs: SHARED_VALIDATION_TIMEOUT_MS,
    sliceAdvancementSubGateSumMs: subGateSum,
    sliceAdvancementTimeoutMs: SLICE_ADVANCEMENT_TIMEOUT_MS,
    gateTimeoutOverheadMs: GATE_TIMEOUT_OVERHEAD_MS,
    gateMaxBufferBytes: GATE_MAX_BUFFER_BYTES,
    envOverrideMs,
    envOverrideVar: GATE_TIMEOUT_ENV_VAR,
    report: lines.join('\n'),
  };
}

/**
 * Detects whether a failure was a TIMEOUT rather than a crash.
 *
 * Node reports a timeout through several different signals depending on the
 * platform and the Node version, and the signal lands on a different object
 * depending on which API produced it:
 *
 * - `execFileSync` throws an error carrying `killed: true`, `signal:
 *   'SIGTERM'`, and an `ETIMEDOUT` code (sometimes nested under `cause`).
 * - `spawnSync` returns a result object carrying `killed: true`, `signal:
 *   'SIGTERM'`, and an `error` whose `code` is `ETIMEDOUT`.
 *
 * A timeout is NOT a graceful degradation: the child was KILLED before it
 * produced a verdict, so the work was never validated. Callers must report
 * it as a failure rather than skipping the check.
 *
 * @param {object} candidate - An error object or a `spawnSync` result.
 * @returns {boolean} True when the failure was a timeout.
 */
export function isTimeoutError(candidate) {
  if (!candidate) return false;
  if (candidate.killed === true) return true;
  if (candidate.signal === 'SIGTERM') return true;
  const code =
    candidate.code ??
    candidate.cause?.code ??
    candidate.error?.code ??
    candidate.error?.cause?.code;
  return code === 'ETIMEDOUT';
}

