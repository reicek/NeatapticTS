#!/usr/bin/env node
/**
 * @module slice-usage-hook
 * @description The REPO-OWNED tool-execution observer of the vertical slice B
 * (AC-V2-D). It is a POST-EVENT hook: it reads the host payload as JSON on
 * STANDARD INPUT and appends ONE observation line to the tool-execution
 * observation log.
 *
 * WHY THE OBSERVER IS FAIL-OPEN: a hook that throws into the host breaks the
 * tool call it was supposed to observe, so EVERY failure path of this module
 * exits `0` and prints NOTHING. A failed observation is a missing measurement,
 * never a broken tool call.
 *
 * WHAT THE OBSERVER PERSISTS - AND WHAT IT NEVER PERSISTS: the observation
 * line carries EXACTLY five fields - `seq`, `slice_id`, `session_key`,
 * `bucket` and `outcome` - and nothing else. No tool name, no argument, no
 * result, no error content, no path, no instant and no raw session id ever
 * reach the log (R-3, R-5). The tool name is reduced to one of the five safe
 * buckets of `scripts/telemetry/telemetry-contract.mjs`, and the raw session
 * id is reduced to a TRUNCATED SHA-256 digest.
 *
 * THE SEAMS THE HOST SEES:
 *   - the transient dispatch-record directory comes from
 *     `NEATAPTIC_DISPATCH_RECORD_DIR` and falls back to
 *     `logs/slice-ledger/dispatch` under the repository root;
 *   - the observation log comes from `NEATAPTIC_OBSERVATION_LOG_PATH` and
 *     falls back to the SIBLING `tool-executions.jsonl` of the resolved
 *     dispatch-record directory;
 *   - the payload is tolerated in BOTH host spellings (the camelCase
 *     `sessionId`/`toolName` and the VS Code `session_id`/`tool_name`).
 *
 * THE SLICE IS NEVER GUESSED: when no dispatch record is open, when no record
 * carries a slice id, or when the MOST RECENTLY dispatched record cannot be
 * told apart from another one (a genuine tie on both order keys), the
 * observer writes NOTHING.
 *
 * THE OPEN SLICE IS THE MOST RECENTLY DISPATCHED ONE: the dispatch directory
 * is a POINTER DIRECTORY, not a single-slot register, so it may hold one
 * record per dispatched slice. The observer therefore resolves the open slice
 * by the DISPATCH ORDER the records themselves carry - the dispatch sequence
 * number first, the monotonic start base second - and never by a wall clock,
 * a file time or the shape of a file name. Both keys are PERSISTED fields of
 * the record, so the order stays stable across a restart.
 *
 * @example
 * echo '{"sessionId":"s-1","toolName":"functions.view"}' \
 *   | NEATAPTIC_DISPATCH_RECORD_DIR=logs/slice-ledger/dispatch \
 *     node scripts/agent-customization/hooks/slice-usage-hook.mjs
 */
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..', '..');

/** The seam the repository already uses for the transient dispatch-record directory. */
const DISPATCH_DIR_ENV = 'NEATAPTIC_DISPATCH_RECORD_DIR';
/** The seam this module pins for the observation log. */
const OBSERVATION_LOG_ENV = 'NEATAPTIC_OBSERVATION_LOG_PATH';
/** The repository-relative default of the transient dispatch-record directory. */
const DEFAULT_DISPATCH_DIR = path.join('logs', 'slice-ledger', 'dispatch');
/** The file name of the observation log, the SIBLING of the dispatch-record directory. */
const OBSERVATION_LOG_NAME = 'tool-executions.jsonl';
/** The bounded number of append attempts a transient write failure may take. */
const MAX_WRITE_ATTEMPTS = 3;
/** The length of the truncated SHA-256 session digest. */
const SESSION_KEY_LENGTH = 12;
/** The two completion outcomes the observation vocabulary is closed to. */
const OUTCOME_COMPLETED = 'completed';
const OUTCOME_FAILED = 'failed';
/** The bucket every tool that matches no rule falls into. */
const DEFAULT_BUCKET = 'other';

/**
 * The bucket rules, in the order they are applied. The rules are ANCHORED on
 * the whole tool name (the namespace prefix is stripped first), so a tool name
 * can never match a rule by accident and a tool name never reaches the log.
 */
const TOOL_BUCKET_RULES = Object.freeze([
  { bucket: 'exec', pattern: /^(?:exec|run|shell|bash|powershell|terminal|command|spawn|task)$/i },
  {
    bucket: 'edit',
    pattern:
      /^(?:edit|create|create_?file|write|write_?file|apply_?patch|patch|replace(_string_in_file)?|delete|remove|move|rename|multiedit|notebookedit|update)$/i,
  },
  {
    bucket: 'search',
    pattern:
      /^(?:grep|search|find|code_?search|file_?search|glob_?search|web_?search|webfetch|fetch|query)$/i,
  },
  { bucket: 'read', pattern: /^(?:view|read|read_?file|cat|open|list|ls|glob|notebookread|get_?file)$/i },
]);

/**
 * Parses the host payload from the TEXT the host wrote on standard input.
 *
 * A payload that is absent, empty or unparseable is NOT an error: the
 * observer has nothing to observe and writes nothing. The parsing is a PURE
 * function of the text, so the edge that reads standard input stays at the
 * very edge of the module and the observer itself can be proved in process.
 *
 * @param {unknown} [raw] The text the host wrote on standard input.
 * @returns {Record<string, unknown>} The payload, or an empty object.
 * @example
 * readPayload('{"sessionId":"s-1","toolName":"functions.view"}');
 * // => { sessionId: 's-1', toolName: 'functions.view' }
 */
export function readPayload(raw) {
  try {
    const text = typeof raw === 'string' ? raw.trim() : '';
    if (text.length === 0) return {};
    const parsed = JSON.parse(text);
    return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/**
 * Resolves the transient dispatch-record directory.
 *
 * @param {NodeJS.ProcessEnv} [env] The environment to read.
 * @returns {string} The absolute path of the dispatch-record directory.
 * @example
 * resolveDispatchDir({ NEATAPTIC_DISPATCH_RECORD_DIR: 'C:\\tmp\\d' });
 * // => 'C:\\tmp\\d'
 */
export function resolveDispatchDir(env = process.env) {
  const configured = env[DISPATCH_DIR_ENV];
  if (typeof configured === 'string' && configured.trim().length > 0) {
    return path.resolve(configured.trim());
  }
  return path.resolve(REPO_ROOT, DEFAULT_DISPATCH_DIR);
}

/**
 * Resolves the observation log.
 *
 * The fallback is the SIBLING `tool-executions.jsonl` of the resolved
 * dispatch-record directory, which is the repository path
 * `logs/slice-ledger/tool-executions.jsonl`.
 *
 * @param {string} dispatchDir The resolved dispatch-record directory.
 * @param {NodeJS.ProcessEnv} [env] The environment to read.
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
  return path.join(path.dirname(path.resolve(dispatchDir)), OBSERVATION_LOG_NAME);
}

/**
 * Lists the transient dispatch records of a directory.
 *
 * Only regular `*.json` files are records: a directory or a temporary
 * sibling is not one.
 *
 * @param {string} dispatchDir The dispatch-record directory.
 * @returns {string[]} The absolute paths of the records.
 * @example
 * listDispatchRecords('C:\\repo\\logs\\slice-ledger\\dispatch');
 * // => ['C:\\repo\\logs\\slice-ledger\\dispatch\\p__S-1.json']
 */
export function listDispatchRecords(dispatchDir) {
  if (!existsSync(dispatchDir)) return [];
  let entries = [];
  try {
    entries = readdirSync(dispatchDir, { withFileTypes: true });
  } catch {
    // A path that cannot be listed (it is not a directory) holds no records.
    return [];
  }
  return entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => path.join(dispatchDir, entry.name));
}

/**
 * Reads the DISPATCH ORDER of one transient dispatch record.
 *
 * The order is the pair of PERSISTED fields the record itself carries:
 *
 *   - the dispatch sequence number - the observation-log line index at
 *     dispatch time, a repository-wide counter that grows with every
 *     dispatched slice;
 *   - the monotonic start base - the hex nanosecond base of the dispatching
 *     process.
 *
 * A field that is absent or malformed falls back to its LOWEST value
 * (`-1` and `0n`), so a record that carries no order at all can never win
 * the resolution against a record that carries one.
 *
 * @param {unknown} record The parsed dispatch record.
 * @returns {{ dispatchSeq: number, startedMonotonicNs: bigint }} The order.
 * @example
 * readDispatchOrder({ dispatch_seq: 3, started_monotonic_ns: '0x10' });
 * // => { dispatchSeq: 3, startedMonotonicNs: 10n }
 */
export function readDispatchOrder(record) {
  const source = record !== null && typeof record === 'object' ? record : {};
  const rawSeq = source.dispatch_seq;
  const dispatchSeq = typeof rawSeq === 'number' && Number.isFinite(rawSeq) ? rawSeq : -1;
  const rawBase = source.started_monotonic_ns;
  let startedMonotonicNs = 0n;
  if (typeof rawBase === 'string') {
    const trimmed = rawBase.trim();
    // The pattern admits only a hexadecimal literal, so the conversion cannot fail.
    if (/^0x[0-9a-f]+$/i.test(trimmed)) startedMonotonicNs = BigInt(trimmed);
  }
  return { dispatchSeq, startedMonotonicNs };
}

/**
 * Compares two dispatch records by their dispatch order, most recent first.
 *
 * The comparison is a TOTAL order over the persisted fields only: the
 * dispatch sequence number, then the monotonic start base, then the record
 * path (so two records that agree on both keys still have a stable order).
 *
 * @param {{ dispatchSeq: number, startedMonotonicNs: bigint, file: string }} a The first record.
 * @param {{ dispatchSeq: number, startedMonotonicNs: bigint, file: string }} b The second record.
 * @returns {number} The comparison result.
 * @example
 * compareDispatchOrder(
 *   { dispatchSeq: 2, startedMonotonicNs: 16n, file: 'b.json' },
 *   { dispatchSeq: 1, startedMonotonicNs: 16n, file: 'a.json' },
 * );
 * // => -1
 */
export function compareDispatchOrder(a, b) {
  if (a.dispatchSeq !== b.dispatchSeq) return b.dispatchSeq - a.dispatchSeq;
  if (a.startedMonotonicNs !== b.startedMonotonicNs) {
    return a.startedMonotonicNs > b.startedMonotonicNs ? -1 : 1;
  }
  // The record path is the LAST key: two records that agree on both order
  // keys still have a stable order. The path of a record is unique inside
  // the directory, so the comparison never meets the same path twice.
  return a.file < b.file ? -1 : 1;
}

/**
 * Reads every transient dispatch record of a directory, in DISPATCH ORDER.
 *
 * A record that cannot be parsed, that is not an object or that carries no
 * slice id carries no usable identity and is skipped: it can never be the
 * open dispatch.
 *
 * @param {string} dispatchDir The dispatch-record directory.
 * @returns {Array<{ file: string, slice_id: string, dispatchSeq: number, startedMonotonicNs: bigint }>}
 *   The records, most recently dispatched first.
 * @example
 * readDispatchRecords('C:\\repo\\logs\\slice-ledger\\dispatch');
 * // => [{ file: '...\\p__S-1.json', slice_id: 'S-1', ... }]
 */
export function readDispatchRecords(dispatchDir) {
  const records = [];
  for (const file of listDispatchRecords(dispatchDir)) {
    let parsed = null;
    try {
      parsed = JSON.parse(readFileSync(file, 'utf8'));
    } catch {
      parsed = null;
    }
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) continue;
    const sliceId = typeof parsed.slice_id === 'string' ? parsed.slice_id.trim() : '';
    if (sliceId.length === 0) continue;
    const order = readDispatchOrder(parsed);
    records.push({ file, slice_id: sliceId, dispatchSeq: order.dispatchSeq, startedMonotonicNs: order.startedMonotonicNs });
  }
  records.sort(compareDispatchOrder);
  return records;
}

/**
 * Reads the open slice id of the MOST RECENTLY DISPATCHED record.
 *
 * The slice id is NEVER guessed: no record, a record without a slice id and
 * an AMBIGUOUS directory (two records that agree on BOTH order keys but name
 * different slices) all yield `null`, and a `null` slice id means the
 * observer writes nothing.
 *
 * @param {string} dispatchDir The dispatch-record directory.
 * @returns {string | null} The open slice id, or null when there is none.
 * @example
 * readOpenSliceId('C:\\repo\\logs\\slice-ledger\\dispatch');
 * // => 'OT-V2-3'
 */
export function readOpenSliceId(dispatchDir) {
  const records = readDispatchRecords(dispatchDir);
  if (records.length === 0) return null;
  const newest = records[0];
  const tied = records.filter(
    (record) =>
      record.dispatchSeq === newest.dispatchSeq && record.startedMonotonicNs === newest.startedMonotonicNs,
  );
  const slices = new Set(tied.map((record) => record.slice_id));
  if (slices.size > 1) return null;
  return newest.slice_id;
}

/**
 * Derives the safe bucket of a tool name.
 *
 * @param {unknown} toolName The raw tool name.
 * @returns {string} One of the five safe buckets.
 * @example
 * deriveBucket('functions.view');
 * // => 'read'
 */
export function deriveBucket(toolName) {
  const raw = typeof toolName === 'string' ? toolName.trim() : '';
  if (raw.length === 0) return DEFAULT_BUCKET;
  const bare = raw.includes('.') ? raw.slice(raw.lastIndexOf('.') + 1) : raw;
  for (const rule of TOOL_BUCKET_RULES) {
    if (rule.pattern.test(bare)) return rule.bucket;
  }
  return DEFAULT_BUCKET;
}

/**
 * Tells whether the payload reports a FAILED completion.
 *
 * The host reports a failure with an `error` field; the field counts as a
 * failure when it carries content. The error CONTENT itself is never
 * persisted.
 *
 * @param {Record<string, unknown>} payload The host payload.
 * @returns {boolean} True when the completion failed.
 * @example
 * hasError({ error: 'boom' });
 * // => true
 */
export function hasError(payload) {
  const candidates = [payload.error, payload.error_message, payload.errorMessage];
  return candidates.some((value) => {
    if (value === null || value === undefined) return false;
    if (typeof value === 'string') return value.trim().length > 0;
    if (Array.isArray(value)) return value.length > 0;
    if (typeof value === 'object') return Object.keys(value).length > 0;
    if (typeof value === 'boolean') return value;
    return true;
  });
}

/**
 * Derives the truncated SHA-256 digest of the raw session id.
 *
 * The raw session id is NEVER persisted: only a truncated digest is, and the
 * digest is short enough that it cannot be replayed into a session.
 *
 * @param {unknown} sessionId The raw session id.
 * @returns {string} The truncated digest.
 * @example
 * deriveSessionKey('session-camel-9f3a7c21');
 * // => '3f0c1a9d2b4e'
 */
export function deriveSessionKey(sessionId) {
  const raw = typeof sessionId === 'string' ? sessionId : '';
  return createHash('sha256').update(raw).digest('hex').slice(0, SESSION_KEY_LENGTH);
}

/**
 * Counts the non-empty lines of the observation log, which is the
 * append-time LINE INDEX the next observation carries.
 *
 * @param {string} logPath The observation log.
 * @returns {number} The number of non-empty lines.
 * @example
 * countObservationLines('C:\\repo\\logs\\slice-ledger\\tool-executions.jsonl');
 * // => 3
 */
export function countObservationLines(logPath) {
  try {
    if (!existsSync(logPath)) return 0;
    return readFileSync(logPath, 'utf8')
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0).length;
  } catch {
    return 0;
  }
}

/**
 * Appends ONE observation line to the log.
 *
 * The append is ONE write of ONE complete line, so two concurrent writers
 * never interleave inside a line. A transient write failure is retried a
 * BOUNDED number of times and is then swallowed: the observer stays
 * FAIL-OPEN.
 *
 * @param {string} logPath The observation log.
 * @param {Record<string, unknown>} record The observation.
 * @returns {boolean} True when the line was appended.
 * @example
 * appendObservation('C:\\log.jsonl', { seq: 0, slice_id: 'S-1' });
 * // => true
 */
export function appendObservation(logPath, record) {
  const line = `${JSON.stringify(record)}\n`;
  for (let attempt = 0; attempt < MAX_WRITE_ATTEMPTS; attempt += 1) {
    try {
      appendFileSync(logPath, line, 'utf8');
      return true;
    } catch (error) {
      if (error && error.code === 'ENOENT') {
        try {
          mkdirSync(path.dirname(logPath), { recursive: true });
        } catch {
          return false;
        }
        continue;
      }
    }
  }
  return false;
}

/**
 * Observes ONE tool execution and appends its observation line.
 *
 * @param {Record<string, unknown>} payload The host payload.
 * @param {NodeJS.ProcessEnv} [env] The environment to read; the process
 *   environment when it is absent.
 * @returns {boolean} True when a line was appended.
 * @example
 * observe({ sessionId: 's-1', toolName: 'functions.view' });
 * // => true
 */
export function observe(payload, env) {
  const dispatchDir = resolveDispatchDir(env);
  const sliceId = readOpenSliceId(dispatchDir);
  if (sliceId === null) return false;
  const logPath = resolveObservationLog(dispatchDir, env);
  const record = {
    seq: countObservationLines(logPath),
    slice_id: sliceId,
    session_key: deriveSessionKey(payload.sessionId ?? payload.session_id ?? payload.sessionID ?? ''),
    bucket: deriveBucket(payload.toolName ?? payload.tool_name ?? ''),
    outcome: hasError(payload) ? OUTCOME_FAILED : OUTCOME_COMPLETED,
  };
  return appendObservation(logPath, record);
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
/* istanbul ignore next -- the CLI entry path is the ONLY place that reads standard input, and it is proved by the child-process lanes of the home, not from inside the process. */
if (invokedDirectly) {
  try {
    let raw = '';
    try {
      raw = readFileSync(0, 'utf8');
    } catch {
      raw = '';
    }
    observe(readPayload(raw), process.env);
  } catch {
    // FAIL-OPEN: a failed observation never fails the tool call it observed.
  }
  process.exitCode = 0;
}
