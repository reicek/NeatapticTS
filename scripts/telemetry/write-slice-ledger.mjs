#!/usr/bin/env node
/**
 * @module write-slice-ledger
 * @description The repo-owned DONE WRITER of the per-slice telemetry ledger:
 * the only component that writes a `telemetry_ledger` block into a companion
 * log, and the only component that consumes the transient dispatch record of a
 * slice.
 *
 * Slice: OT-V1-2 of `plans/opentelemetry-slice-telemetry.plans.md`.
 * Acceptance criteria implemented by this module:
 *   - AC-V1-F (the DONE writer and the first real DONE ledger entry)
 *   - AC-002 (the versioned, keyed, idempotent, forward-only ledger)
 *   - AC-004 (an unavailable counter is null with a reason, never zero)
 *   - AC-012 (layer B produces every per-slice measurement)
 *
 * THE COMMAND LINE:
 *
 *   node scripts/telemetry/write-slice-ledger.mjs \
 *     --plan <planPath> --slice <sliceId> \
 *     [--plan-id <planId>] [--status <status>] [--otel-state <state>] \
 *     [--validation <json>] [--dispatch-dir <dir>] [--recover]
 *
 *   --plan         REQUIRED. The plan file. The companion log is its sibling
 *                  `<name>.logs.md`, so `x.plans.md` is written into
 *                  `x.logs.md`.
 *   --slice        REQUIRED. The slice id.
 *   --plan-id      OPTIONAL. The plan id; defaults to the plan basename minus
 *                  the `.plans.md` suffix.
 *   --status       OPTIONAL. The slice status; defaults to `[DONE]`.
 *   --otel-state   OPTIONAL. The layer-A reporting state, one of `ENABLED`,
 *                  `DISABLED`, `STALE`, `CORRUPT`, `NO-DATA`.
 *   --validation   OPTIONAL. A JSON object `{total, passed, failed, skipped,
 *                  results:[{name, status, duration_ms}]}`; it is persisted
 *                  verbatim in the validation block.
 *   --dispatch-dir OPTIONAL. The transient dispatch-record directory; it
 *                  defaults to `logs/slice-ledger/dispatch` under the repo
 *                  root.
 *   --recover      The recovery mode for an INTERRUPTED slice.
 *
 * BOTH FLAG SPELLINGS ARE ACCEPTED: `--plan <path>` and `--plan=<path>` are
 * the same flag, because a caller that builds a command line by hand and a
 * caller that builds one from a dictionary must not disagree about it.
 *
 * THE EXIT CONTRACT: the writer exits 0 when it wrote a ledger (or when the
 * ledger was ALREADY there, which is the idempotent no-op) and NON-ZERO when
 * it refused. A refusal NEVER writes a block.
 *
 * THE TOOL-EXECUTION CONTRACT: the tool-execution counter is measured from
 * the REPO-OWNED observation log the tool-execution observer appends to, NOT
 * from a hard-coded reason. The writer resolves that log through the SAME
 * seam the observer uses - the `NEATAPTIC_OBSERVATION_LOG_PATH` environment
 * variable, and otherwise the SIBLING `tool-executions.jsonl` of the resolved
 * dispatch-record directory - reads the observations the observer attributed
 * to THIS slice and persists the REAL counts. An observation the log does not
 * carry is NEVER invented: a counter that cannot be measured is
 * `available: false` with the CONCRETE reason (no observation log, an
 * unreadable log, a log that carries no observation, or a log that carries no
 * observation for this slice) and with NULL numeric fields, never a zero.
 *
 * THE MEASUREMENT CONTRACT: the duration is the MONOTONIC delta of the
 * transient dispatch record, measured with the same clock the record was
 * written with, and the record is DELETED once the block is written. A
 * missing, unreadable or negative-delta record yields
 * `measurement_state: UNAVAILABLE` with a reason and a NULL duration, never a
 * zero. A CORRUPT record is REFUSED, because a corrupt record is repaired,
 * never guessed.
 *
 * THE PRUNING CONTRACT: the closure path of a slice also PRUNES the transient
 * dispatch-record directory, so a record no DONE writer will ever claim does
 * not stay in the pointer directory forever. The pruner removes a record ONLY
 * when the record's own bytes violate the persisted-artefact contract, so
 * every contract-clean record - and above all the OPEN dispatch record of a
 * slice that is still running - SURVIVES the prune. The prune report carries
 * COUNTS and never a path, because a removed record may carry a forbidden
 * value in its NAME.
 *
 * THE IDEMPOTENCE CONTRACT: a repeated write of the same `(plan_id,
 * slice_id)` leaves the persisted bytes IDENTICAL and keeps exactly one
 * block. The existing, schema-valid block is therefore AUTHORITATIVE: the
 * writer does not recompute it, because the transient record it was derived
 * from is gone by then and a recomputation would produce a DIFFERENT block
 * (a different duration, a different heap reading) under the same key.
 *
 * THE LAYER INDEPENDENCE: this writer imports NO `@opentelemetry*` package
 * and NO `src/observability` module. The repo-owned measurement must not
 * depend on the optional OpenTelemetry layer; the writer may READ a layer-A
 * state report but never REQUIRES one, and a layer-A state never changes the
 * layer-B measurement state.
 *
 * @example
 * node scripts/telemetry/write-slice-ledger.mjs \
 *   --plan plans/completed/opentelemetry-slice-telemetry.plans.md \
 *   --slice OT-V1-2 --plan-id opentelemetry-slice-telemetry \
 *   --status '[DONE]' --otel-state DISABLED
 */

import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pruneDispatchRecords } from './prune-dispatch-records.mjs';
import {
  DISPATCH_RECORD_DIR,
  SCHEMA_VERSION,
  TOOL_EXECUTION_BUCKETS,
  deleteDispatchRecord,
  emitLedgerBlock,
  readDispatchRecord,
  validateLedgerBlock,
} from './telemetry-contract.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');

/**
 * The default status of a written ledger.
 *
 * @type {string}
 */
const DEFAULT_STATUS = '[DONE]';

/**
 * The default layer-A reporting state. `NO-DATA` is the honest state of a
 * layer that was never enabled, and it NEVER changes the layer-B measurement
 * state.
 *
 * @type {string}
 */
const DEFAULT_OTEL_STATE = 'NO-DATA';

/**
 * The reason token of an unavailable token counter whose dispatch record is
 * missing.
 *
 * @type {string}
 */
const REASON_DISPATCH_RECORD_MISSING = 'dispatch_record_missing';

/**
 * The reason token of an unavailable token counter whose dispatch record
 * carries no monotonic base.
 *
 * @type {string}
 */
const REASON_DISPATCH_RECORD_WITHOUT_BASE = 'dispatch_record_without_monotonic_base';

/**
 * The reason token of a monotonic delta that came out negative.
 *
 * @type {string}
 */
const REASON_NEGATIVE_MONOTONIC_DELTA = 'negative_monotonic_delta';

/**
 * The reason token of a recovered slice: the slice was interrupted, so it
 * carries no live measurement.
 *
 * @type {string}
 */
const REASON_SLICE_INTERRUPTED = 'slice_interrupted';

/**
 * The seam the tool-execution observer resolves the observation log through.
 * The writer resolves it through the SAME seam, so the writer and the
 * observer read the SAME file without either of them hard-coding the
 * repository layout.
 *
 * @type {string}
 */
const OBSERVATION_LOG_ENV = 'NEATAPTIC_OBSERVATION_LOG_PATH';

/**
 * The file name of the observation log: the SIBLING of the dispatch-record
 * directory.
 *
 * @type {string}
 */
const OBSERVATION_LOG_NAME = 'tool-executions.jsonl';

/**
 * The source of the tool-execution counter: the post-hook observation window
 * the tool-execution observer opens inside the slice dispatch window.
 *
 * @type {string}
 */
const TOOL_EXECUTION_SOURCE = 'copilot-cli-posthook-observation-window';

/**
 * What the tool-execution counter measures, stated exactly. A SUCCESSFUL and
 * a FAILED completion are counted as separately named measurements, and a
 * call that was DENIED before it executed never completes and is never
 * observed.
 *
 * @type {string}
 */
const TOOL_EXECUTION_MEASURES =
  'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window';

/**
 * The scope of the tool-execution counter: the observations the observer
 * attributed while this slice's dispatch record was open.
 *
 * @type {string}
 */
const TOOL_EXECUTION_SCOPE = 'slice-dispatch-window';

/**
 * The outcome of an observation that completed.
 *
 * @type {string}
 */
const OUTCOME_COMPLETED = 'completed';

/**
 * The outcome of an observation that failed.
 *
 * @type {string}
 */
const OUTCOME_FAILED = 'failed';

/**
 * The reason token of a tool-execution counter whose observation log does not
 * exist: the observer has produced no observation log at all.
 *
 * @type {string}
 */
const REASON_OBSERVATION_LOG_MISSING = 'observation_log_missing';

/**
 * The reason token of a tool-execution counter whose observation log cannot
 * be read.
 *
 * @type {string}
 */
const REASON_OBSERVATION_LOG_UNREADABLE = 'observation_log_unreadable';

/**
 * The reason token of a tool-execution counter whose observation log carries
 * no observation the contract can read.
 *
 * @type {string}
 */
const REASON_OBSERVATION_LOG_EMPTY = 'observation_log_empty';

/**
 * The reason token of a tool-execution counter whose observation log carries
 * observations, but none of them is attributed to this slice.
 *
 * @type {string}
 */
const REASON_NO_OBSERVATIONS_FOR_SLICE = 'no_observations_for_slice';

/**
 * The error a malformed command line raises.
 *
 * @class UsageError
 * @extends {Error}
 */
class UsageError extends Error {}

/**
 * Parses the command line. Both `--flag value` and `--flag=value` are
 * accepted.
 *
 * @param {string[]} argv - The arguments after the script path.
 * @returns {Record<string, string | boolean>} The parsed options.
 * @example
 * parseArgs(['--plan', 'p.plans.md', '--recover']);
 * // => { plan: 'p.plans.md', recover: true, ... }
 */
export function parseArgs(argv) {
  const options = {
    plan: null,
    slice: null,
    planId: null,
    status: null,
    otelState: null,
    validation: null,
    dispatchDir: null,
    recover: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--recover') {
      options.recover = true;
      continue;
    }
    if (!token.startsWith('--')) {
      throw new UsageError(`unknown argument: ${token}`);
    }
    const equals = token.indexOf('=');
    const name = equals >= 0 ? token.slice(0, equals) : token;
    let value = equals >= 0 ? token.slice(equals + 1) : null;
    if (value === null) {
      const next = argv[index + 1];
      if (next === undefined || next.startsWith('--')) {
        throw new UsageError(`the flag ${name} carries no value`);
      }
      value = next;
      index += 1;
    }
    switch (name) {
      case '--plan':
        options.plan = value;
        break;
      case '--slice':
        options.slice = value;
        break;
      case '--plan-id':
        options.planId = value;
        break;
      case '--status':
        options.status = value;
        break;
      case '--otel-state':
        options.otelState = value;
        break;
      case '--validation':
        options.validation = value;
        break;
      case '--dispatch-dir':
        options.dispatchDir = value;
        break;
      default:
        throw new UsageError(`unknown flag: ${name}`);
    }
  }
  return options;
}

/**
 * The stem of a plan file: the basename without the `.plans.md` suffix.
 *
 * @param {string} planPath - The path of the plan file.
 * @returns {string} The stem.
 * @example
 * planStem('plans/x.plans.md'); // => 'x'
 */
export function planStem(planPath) {
  const base = path.basename(path.resolve(planPath));
  if (base.endsWith('.plans.md')) return base.slice(0, -'.plans.md'.length);
  return base.replace(/\.md$/, '');
}

/**
 * The companion log of a plan: its sibling `<name>.logs.md`.
 *
 * @param {string} planPath - The path of the plan file.
 * @returns {string} The absolute path of the companion log.
 * @example
 * companionLogPath('plans/x.plans.md'); // => '...\\plans\\x.logs.md'
 */
export function companionLogPath(planPath) {
  const resolved = path.resolve(planPath);
  return path.join(path.dirname(resolved), `${planStem(resolved)}.logs.md`);
}

/**
 * Reads a scalar of the emitted YAML subset.
 *
 * @param {string} body - The body of a fenced block.
 * @param {string} key - The root-level key.
 * @returns {string | null} The value, or null when the key is absent.
 */
function readRootScalar(body, key) {
  const pattern = new RegExp(`^ {2}${key}:[ \\t]*(.*)$`, 'm');
  const match = pattern.exec(body);
  if (!match) return null;
  const raw = match[1].trim();
  if (raw.length >= 2 && raw.startsWith("'") && raw.endsWith("'")) {
    return raw.slice(1, -1).replace(/''/g, "'");
  }
  return raw;
}

/**
 * Reads every fenced `yaml` block of a companion log that is ROOTED at the
 * ledger root key. A block with another root (an archived step packet, for
 * instance) is not a ledger block and is ignored.
 *
 * The fence tolerates BOTH line endings: a companion log written on Windows
 * carries CRLF, and a block whose fence is CRLF-terminated is a REAL block. A
 * pattern that only accepted LF would MISS an existing block and the writer
 * would append a SECOND block under the same key.
 *
 * @param {string} logText - The text of the companion log.
 * @returns {Array<{ start: number, end: number, planId: string | null, sliceId: string | null, schemaVersion: number | null }>} The ledger blocks.
 */
export function findLedgerBlocks(logText) {
  const blocks = [];
  const pattern = /```yaml\r?\n([\s\S]*?)```/g;
  let match = pattern.exec(logText);
  while (match !== null) {
    const body = match[1];
    const firstLine = body.split(/\r?\n/).find((line) => line.trim().length > 0);
    if (firstLine !== undefined && firstLine.trim() === 'telemetry_ledger:') {
      const rawVersion = readRootScalar(body, 'schema_version');
      blocks.push({
        start: match.index,
        end: match.index + match[0].length,
        planId: readRootScalar(body, 'plan_id'),
        sliceId: readRootScalar(body, 'slice_id'),
        schemaVersion:
          rawVersion !== null && Number.isFinite(Number(rawVersion))
            ? Number(rawVersion)
            : null,
      });
    }
    match = pattern.exec(logText);
  }
  return blocks;
}

/**
 * Parses the monotonic base of a dispatch record.
 *
 * @param {Record<string, unknown> | null} record - The dispatch record.
 * @returns {bigint | null} The base, or null when the record carries none.
 */
function readMonotonicBase(record) {
  if (!record) return null;
  const raw = record.started_monotonic_ns;
  if (typeof raw !== 'string' || raw.length === 0) return null;
  try {
    return BigInt(raw);
  } catch {
    return null;
  }
}

/**
 * Resolves the observation log the tool-execution observer appends to.
 *
 * The seam is the ONE the observer uses: the environment variable wins, and
 * the fallback is the SIBLING `tool-executions.jsonl` of the resolved
 * dispatch-record directory. The writer therefore reads the SAME log the
 * observer writes, and neither of them hard-codes the repository layout.
 *
 * @param {string} dispatchDir - The resolved dispatch-record directory.
 * @param {NodeJS.ProcessEnv} [env] - The environment to read.
 * @returns {string} The absolute path of the observation log.
 * @example
 * resolveObservationLog('C:\\repo\\logs\\slice-ledger\\dispatch');
 * // => 'C:\\repo\\logs\\slice-ledger\\tool-executions.jsonl'
 */
export function resolveObservationLog(dispatchDir, env = process.env) {
  const configured = env[OBSERVATION_LOG_ENV];
  if (typeof configured === 'string' && configured.trim().length > 0) {
    return path.resolve(configured.trim());
  }
  return path.join(
    path.dirname(path.resolve(dispatchDir)),
    OBSERVATION_LOG_NAME,
  );
}

/**
 * Decides whether a parsed observation line carries an observation the
 * contract can read.
 *
 * The observer writes `{ seq, slice_id, session_key, bucket, outcome }` and
 * nothing else. A line that carries no such observation is NOT counted: it
 * is not evidence of a tool execution, and counting it would make the
 * counter a count of lines rather than a count of executions.
 *
 * @param {unknown} parsed - The parsed line.
 * @returns {boolean} Whether the line carries an observation.
 */
function isObservation(parsed) {
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed))
    return false;
  const entry = /** @type {Record<string, unknown>} */ (parsed);
  if (typeof entry.slice_id !== 'string') return false;
  if (typeof entry.session_key !== 'string') return false;
  if (typeof entry.bucket !== 'string') return false;
  return (
    entry.outcome === OUTCOME_COMPLETED || entry.outcome === OUTCOME_FAILED
  );
}

/**
 * Builds the per-bucket breakdown, with EVERY safe bucket present - the
 * buckets that saw no execution carry a zero, because a breakdown that
 * omits a bucket cannot be told apart from a bucket that was never counted.
 *
 * @returns {Record<string, number>} The empty breakdown.
 */
function emptyBreakdown() {
  const breakdown = {};
  for (const bucket of TOOL_EXECUTION_BUCKETS) breakdown[bucket] = 0;
  return breakdown;
}

/**
 * Reads the tool observations the observer attributed to ONE slice.
 *
 * The reader is deliberately TOLERANT and deliberately HONEST: a line it
 * cannot read is a line it does not count, an observation that belongs to
 * another slice is never attributed to this one, and NO observation is ever
 * invented. The four unavailable states are distinguished, because they are
 * four different facts and the reason token names the fact.
 *
 * @param {string} logPath - The observation log.
 * @param {string} sliceId - The slice id.
 * @returns {{ state: string, total: number, completed_executions: number, failed_executions: number, sessions_observed: number, breakdown: Record<string, number> }} The observations.
 * @example
 * readSliceObservations('C:\\repo\\logs\\slice-ledger\\tool-executions.jsonl', 'OT-V2-3');
 * // => { state: 'AVAILABLE', total: 3, ... }
 */
export function readSliceObservations(logPath, sliceId) {
  const counts = {
    total: 0,
    completed_executions: 0,
    failed_executions: 0,
    sessions_observed: 0,
    breakdown: emptyBreakdown(),
  };
  if (!existsSync(logPath)) return { ...counts, state: 'MISSING' };
  let text;
  try {
    text = readFileSync(logPath, 'utf8');
  } catch {
    return { ...counts, state: 'UNREADABLE' };
  }
  const observations = [];
  for (const line of text.split(/\r?\n/)) {
    if (line.trim().length === 0) continue;
    let parsed;
    try {
      parsed = JSON.parse(line);
    } catch {
      continue;
    }
    if (!isObservation(parsed)) continue;
    observations.push(/** @type {Record<string, unknown>} */ (parsed));
  }
  if (observations.length === 0) return { ...counts, state: 'EMPTY' };
  const mine = observations.filter((entry) => entry.slice_id === sliceId);
  if (mine.length === 0) return { ...counts, state: 'NO_OBSERVATIONS' };
  const breakdown = emptyBreakdown();
  const sessions = new Set();
  let completed = 0;
  let failed = 0;
  for (const entry of mine) {
    // A bucket the contract does not know is still an execution: it is
    // counted in the OTHER bucket, so the breakdown always sums up to the
    // total and no execution is silently dropped.
    const bucket = TOOL_EXECUTION_BUCKETS.includes(entry.bucket)
      ? entry.bucket
      : 'other';
    breakdown[bucket] += 1;
    sessions.add(entry.session_key);
    if (entry.outcome === OUTCOME_COMPLETED) completed += 1;
    else failed += 1;
  }
  return {
    state: 'AVAILABLE',
    total: completed + failed,
    completed_executions: completed,
    failed_executions: failed,
    sessions_observed: sessions.size,
    breakdown,
  };
}

/**
 * Builds the tool-execution counter of a block.
 *
 * An AVAILABLE counter carries the REAL counts the observation log holds for
 * this slice. An UNAVAILABLE counter carries the CONCRETE reason and NULL
 * numeric fields - never a zero, because a zero is a measurement.
 *
 * @param {string} logPath - The observation log.
 * @param {string} sliceId - The slice id.
 * @returns {Record<string, unknown>} The counter.
 * @example
 * buildToolExecutions('C:\\repo\\logs\\slice-ledger\\tool-executions.jsonl', 'OT-V2-3');
 * // => { available: true, total: 3, ... }
 */
export function buildToolExecutions(logPath, sliceId) {
  const observations = readSliceObservations(logPath, sliceId);
  const reason =
    {
      MISSING: REASON_OBSERVATION_LOG_MISSING,
      UNREADABLE: REASON_OBSERVATION_LOG_UNREADABLE,
      EMPTY: REASON_OBSERVATION_LOG_EMPTY,
      NO_OBSERVATIONS: REASON_NO_OBSERVATIONS_FOR_SLICE,
    }[observations.state] ?? null;
  if (reason !== null) {
    return {
      available: false,
      source: TOOL_EXECUTION_SOURCE,
      measures: TOOL_EXECUTION_MEASURES,
      scope: TOOL_EXECUTION_SCOPE,
      total: null,
      completed_executions: null,
      failed_executions: null,
      sessions_observed: null,
      breakdown: null,
      unavailable_reason: reason,
    };
  }
  return {
    available: true,
    source: TOOL_EXECUTION_SOURCE,
    measures: TOOL_EXECUTION_MEASURES,
    scope: TOOL_EXECUTION_SCOPE,
    total: observations.total,
    completed_executions: observations.completed_executions,
    failed_executions: observations.failed_executions,
    sessions_observed: observations.sessions_observed,
    breakdown: observations.breakdown,
    unavailable_reason: null,
  };
}

/**
 * Builds the usage counters of a block.
 *
 * @param {Record<string, unknown> | null} record - The dispatch record, when there is one.
 * @param {boolean} measured - Whether the slice produced a live measurement.
 * @param {string} unavailableReason - The reason token of an unavailable token counter.
 * @param {Record<string, unknown>} toolExecutions - The tool-execution counter.
 * @returns {{ token_usage: Record<string, unknown>, tool_executions: Record<string, unknown> }} The counters.
 */
function buildUsage(record, measured, unavailableReason, toolExecutions) {
  const delivered =
    measured && record && typeof record.context_tokens_delivered === 'number'
      ? record.context_tokens_delivered
      : null;
  const tokenUsage = {
    available: delivered !== null,
    source: 'dispatch_record',
    measures: 'the delivered context-token count',
    context_tokens_delivered: delivered,
    prompt_tokens: null,
    completion_tokens: null,
    total_tokens: null,
    unavailable_reason: delivered === null ? unavailableReason : null,
  };
  return { token_usage: tokenUsage, tool_executions: toolExecutions };
}

/**
 * Prunes the contract-violating residue of the dispatch-record directory.
 *
 * The prune is the SAFE residue prune: it removes a record ONLY when the
 * record's own bytes violate the persisted-artefact contract, that is when
 * the CURRENT writer could never have produced it. Every contract-clean
 * record - and above all the OPEN dispatch record of a slice that is still
 * running - therefore SURVIVES the prune.
 *
 * @param {string} dispatchDir - The absolute dispatch-record directory.
 * @returns {{ inspected: number, removed: number, kept: number, failed: number }} The prune counts.
 * @example
 * pruneDispatchResidue('C:\\repo\\logs\\slice-ledger\\dispatch');
 * // => { inspected: 7, removed: 1, kept: 6, failed: 0 }
 */
function pruneDispatchResidue(dispatchDir) {
  const report = pruneDispatchRecords(dispatchDir);
  return {
    inspected: report.inspected,
    removed: report.removed.length,
    kept: report.kept,
    failed: report.failed.length,
  };
}

/**
 * Reports the prune counts of the dispatch-record residue.
 *
 * THE REPORT CARRIES COUNTS AND NEVER PATHS: a removed record may carry a
 * forbidden value in its NAME, so the report must never be persisted into a
 * plan or a log.
 *
 * @param {{ inspected: number, removed: number, kept: number, failed: number }} counts - The prune counts.
 * @returns {void} Nothing.
 * @example
 * reportDispatchResidue({ inspected: 7, removed: 1, kept: 6, failed: 0 });
 * // => writes "the dispatch-record residue pruned: 7 inspected, 1 removed, 6 kept"
 */
function reportDispatchResidue(counts) {
  process.stdout.write(
    `the dispatch-record residue pruned: ${counts.inspected} inspected, ${counts.removed} removed, ${counts.kept} kept\n`,
  );
  if (counts.failed > 0) {
    process.stderr.write(
      `the dispatch-record residue could not be fully pruned: ${counts.failed} record(s) resisted removal and are pruned again on the next closure\n`,
    );
  }
}

/**
 * Runs the writer.
 *
 * @param {string[]} argv - The arguments after the script path.
 * @returns {number} The exit code.
 */
export function main(argv) {
  let options;
  try {
    options = parseArgs(argv);
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    return 2;
  }
  if (!options.plan) {
    process.stderr.write('the flag --plan is required\n');
    return 2;
  }
  if (!options.slice) {
    process.stderr.write('the flag --slice is required\n');
    return 2;
  }
  const planPath = path.resolve(options.plan);
  if (!existsSync(planPath)) {
    process.stderr.write(`the plan file does not exist: ${planPath}\n`);
    return 2;
  }
  const planId = options.planId ?? planStem(planPath);
  const sliceId = options.slice;
  const logPath = companionLogPath(planPath);
  const dispatchDir = options.dispatchDir
    ? path.resolve(options.dispatchDir)
    : path.resolve(REPO_ROOT, DISPATCH_RECORD_DIR);
  const dispatchOptions = options.dispatchDir ? { directory: dispatchDir } : {};

  // The observation log is the SIBLING of the dispatch-record directory, so
  // the writer and the observer read the SAME log without either of them
  // hard-coding the repository layout.
  const observationLogPath = resolveObservationLog(dispatchDir);

  // The record is read BEFORE anything else: a CORRUPT record is refused, and
  // a refusal never writes a block.
  const read = readDispatchRecord(
    { plan_id: planId, slice_id: sliceId },
    dispatchOptions,
  );
  if (read.status === 'CORRUPT') {
    process.stderr.write(
      `refusing to write the ledger: the transient dispatch record is corrupt and is repaired, never guessed: ${read.path}\n`,
    );
    return 1;
  }

  // The idempotence contract: an existing, schema-valid block for this key is
  // AUTHORITATIVE. The writer does not recompute it, because the transient
  // record it was derived from is gone by then and a recomputation would
  // produce a different block under the same key.
  const existingText = existsSync(logPath) ? readFileSync(logPath, 'utf8') : '';
  const existing = findLedgerBlocks(existingText).find(
    (block) => block.planId === planId && block.sliceId === sliceId,
  );
  if (existing && existing.schemaVersion === SCHEMA_VERSION) {
    // The transient state is removed on the IDEMPOTENT path too. An
    // idempotent write still CLOSES the slice, so a dispatch record that
    // survived the first write is removed here instead of lingering in the
    // dispatch directory forever (R-8).
    const removed = deleteDispatchRecord(
      { plan_id: planId, slice_id: sliceId },
      dispatchOptions,
    );
    // The closure path prunes the residue too: an idempotent write still
    // CLOSES the slice, so a record that no DONE writer will ever claim does
    // not survive this closure either.
    reportDispatchResidue(pruneDispatchResidue(dispatchDir));
    process.stdout.write(
      `the ledger for ${planId}/${sliceId} is already recorded at ${logPath}; nothing to write${removed ? ', the transient dispatch record is removed' : ''}\n`,
    );
    return 0;
  }

  const recover = options.recover === true;
  let measurementState = 'UNAVAILABLE';
  let durationMs = null;
  let tokenReason = REASON_DISPATCH_RECORD_MISSING;
  const record = read.status === 'OK' ? read.record : null;
  if (recover) {
    // A recovered slice is an INTERRUPTED slice: it carries no live
    // measurement, so the block states the interruption instead of a number.
    tokenReason = REASON_SLICE_INTERRUPTED;
  } else if (record) {
    const base = readMonotonicBase(record);
    if (base === null) {
      tokenReason = REASON_DISPATCH_RECORD_WITHOUT_BASE;
    } else {
      const delta = process.hrtime.bigint() - base;
      if (delta < 0n) {
        tokenReason = REASON_NEGATIVE_MONOTONIC_DELTA;
      } else {
        measurementState = 'MEASURED';
        durationMs = Number(delta / 1_000_000n);
        tokenReason = null;
      }
    }
  }

  let validation = {
    total: null,
    passed: null,
    failed: null,
    skipped: null,
    results: null,
  };
  if (options.validation) {
    try {
      validation = JSON.parse(options.validation);
    } catch {
      process.stderr.write('the flag --validation carries no parsable JSON\n');
      return 2;
    }
  }

  const usage = buildUsage(
    record,
    measurementState === 'MEASURED',
    tokenReason,
    buildToolExecutions(observationLogPath, sliceId),
  );
  const memory = process.memoryUsage();
  const block = {
    schema_version: SCHEMA_VERSION,
    plan_id: planId,
    slice_id: sliceId,
    status: options.status ?? DEFAULT_STATUS,
    measurement_state: measurementState,
    otel_state: options.otelState ?? DEFAULT_OTEL_STATE,
    duration_ms: durationMs,
    runtime: {
      node_version: process.version,
      heap_used_bytes: memory.heapUsed,
      rss_bytes: memory.rss,
    },
    usage: {
      token_usage: usage.token_usage,
      tool_executions: usage.tool_executions,
    },
    validation,
  };

  const verdict = validateLedgerBlock(block);
  if (!verdict.valid) {
    process.stderr.write(
      `refusing to write the ledger into ${logPath}: ${verdict.errors.join('; ')}\n`,
    );
    return 1;
  }

  const emitted = emitLedgerBlock(block);
  const directory = path.dirname(logPath);
  if (!existsSync(directory)) mkdirSync(directory, { recursive: true });
  if (existing) {
    // A block of an UNKNOWN schema version is STALE: it is replaced, not
    // kept, and the replacement keeps the ledger at ONE block per key.
    const withoutStale =
      existingText.slice(0, existing.start) + existingText.slice(existing.end);
    const separator =
      withoutStale.length === 0 || withoutStale.endsWith('\n') ? '\n' : '\n\n';
    writeFileSync(logPath, `${withoutStale}${separator}${emitted}`, 'utf8');
  } else {
    const separator =
      existingText.length === 0
        ? ''
        : existingText.endsWith('\n')
          ? '\n'
          : '\n\n';
    appendFileSync(logPath, `${separator}${emitted}`, 'utf8');
  }

  // The transient state is removed once the slice is closed (R-8).
  const removed = deleteDispatchRecord(
    { plan_id: planId, slice_id: sliceId },
    dispatchOptions,
  );

  // The closure path prunes the residue: a record that no DONE writer will
  // ever claim does not stay in the pointer directory forever. The prune
  // NEVER removes the OPEN dispatch record of a slice that is still running,
  // because such a record is contract-clean and the prune removes
  // contract-violating bytes only.
  reportDispatchResidue(pruneDispatchResidue(dispatchDir));

  process.stdout.write(
    `the ledger for ${planId}/${sliceId} is written into ${logPath} (measurement_state: ${measurementState}${removed ? ', the transient dispatch record is removed' : ''})\n`,
  );
  return 0;
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedDirectly) {
  process.exitCode = main(process.argv.slice(2));
}

export { REPO_ROOT };
