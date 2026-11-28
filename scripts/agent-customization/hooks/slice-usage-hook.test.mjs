/**
 * @module slice-usage-hook.test
 * @description IN-PROCESS contract home for the tool-execution observer of the
 * vertical slice B (AC-V2-D).
 *
 * WHY THIS HOME EXISTS NEXT TO THE CHILD-PROCESS HOME: the observer is
 * FAIL-OPEN by contract, so its refusal to write and its resolution of the
 * open slice are properties of its FUNCTIONS, not only of its exit code. The
 * child-process home (`scripts/telemetry/slice-usage-hook.test.mjs`) proves
 * the process contract; THIS home proves the FUNCTION contract IN PROCESS, so
 * the observer is measured by the coverage of the file itself instead of by a
 * process that only ever runs its CLI entry.
 *
 * THE SUBJECT UNDER TEST IS `scripts/agent-customization/hooks/slice-usage-hook.mjs`.
 * Nothing is mocked: the lanes write REAL dispatch records into a REAL
 * temporary directory and read REAL observation logs back.
 *
 * THE ANTI-VACUITY LANE (AC-011): the manifest registration is checked
 * against the OBSERVER PATH, under BOTH post event keys, with the SAME command
 * and with NO matcher. The lane that removes the failure-event registration
 * from a COPY of the tracked manifest MUST fail the verdict: a home that
 * cannot fail proves nothing.
 *
 * WHAT THE OBSERVER NEVER PERSISTS: the observation line carries EXACTLY five
 * fields - `seq`, `slice_id`, `session_key`, `bucket` and `outcome` - and no
 * tool name, no argument, no result, no path, no instant and no raw session
 * id ever reach the log (R-3, R-5).
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  appendObservation,
  compareDispatchOrder,
  countObservationLines,
  deriveBucket,
  deriveSessionKey,
  hasError,
  listDispatchRecords,
  observe,
  readDispatchOrder,
  readDispatchRecords,
  readOpenSliceId,
  readPayload,
  resolveDispatchDir,
  resolveObservationLog,
} from './slice-usage-hook.mjs';
import {
  OBSERVER_EVENT_KEYS,
  OBSERVER_RELPATH,
  inspectObserverRegistration,
} from '../../telemetry/telemetry-contract.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..', '..');
/** The tracked hook manifest the registration contract is checked against. */
const MANIFEST_PATH = path.join(REPO_ROOT, '.github', 'hooks', 'cortex-refresh.json');

/** The temporary workspace of the running lane. */
let workspace;
/** The transient dispatch-record directory of the running lane. */
let dispatchDir;
/** The observation log of the running lane. */
let observationLog;

/**
 * Writes one transient dispatch record.
 *
 * @param {string} name The record file name.
 * @param {Record<string, unknown>} record The record body.
 * @returns {string} The record path.
 */
function writeRecord(name, record) {
  const target = path.join(dispatchDir, name);
  writeFileSync(target, typeof record === 'string' ? record : JSON.stringify(record), 'utf8');
  return target;
}

/**
 * Reads the observation log back as parsed lines.
 *
 * @returns {Array<Record<string, unknown>>} The observation lines.
 */
function readObservations() {
  if (!existsSync(observationLog)) return [];
  return readFileSync(observationLog, 'utf8')
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line));
}

beforeEach(() => {
  workspace = mkdtempSync(path.join(tmpdir(), 'ot-v2-3-observer-'));
  dispatchDir = path.join(workspace, 'dispatch');
  observationLog = path.join(workspace, 'tool-executions.jsonl');
  mkdirSync(dispatchDir, { recursive: true });
});

afterEach(() => {
  rmSync(workspace, { recursive: true, force: true });
});

describe('the payload the host writes on standard input', () => {
  it('parses a JSON object payload', () => {
    expect(readPayload('{"sessionId":"s-1","toolName":"functions.view"}')).toEqual({
      sessionId: 's-1',
      toolName: 'functions.view',
    });
  });

  it('tolerates an absent, empty or blank payload', () => {
    expect(readPayload(undefined)).toEqual({});
    expect(readPayload('')).toEqual({});
    expect(readPayload('   \n ')).toEqual({});
  });

  it('tolerates a payload that is not a JSON object', () => {
    expect(readPayload('not json at all')).toEqual({});
    expect(readPayload('null')).toEqual({});
    expect(readPayload('[1,2,3]')).toEqual({});
    expect(readPayload('42')).toEqual({});
  });
});

describe('the seams the host sees', () => {
  it('resolves the dispatch directory from the environment seam', () => {
    const configured = path.join(workspace, 'records');
    expect(resolveDispatchDir({ NEATAPTIC_DISPATCH_RECORD_DIR: `  ${configured}  ` })).toBe(
      path.resolve(configured),
    );
  });

  it('falls back to the repository dispatch directory when the seam is blank or absent', () => {
    const fallback = path.resolve(REPO_ROOT, 'logs', 'slice-ledger', 'dispatch');
    expect(resolveDispatchDir({ NEATAPTIC_DISPATCH_RECORD_DIR: '   ' })).toBe(fallback);
    expect(resolveDispatchDir({})).toBe(fallback);
    expect(resolveDispatchDir()).toBe(fallback);
  });

  it('resolves the observation log from the environment seam', () => {
    const configured = path.join(workspace, 'obs.jsonl');
    expect(resolveObservationLog(dispatchDir, { NEATAPTIC_OBSERVATION_LOG_PATH: configured })).toBe(
      path.resolve(configured),
    );
  });

  it('falls back to the sibling observation log of the dispatch directory', () => {
    expect(resolveObservationLog(dispatchDir, { NEATAPTIC_OBSERVATION_LOG_PATH: '  ' })).toBe(
      path.join(workspace, 'tool-executions.jsonl'),
    );
    expect(resolveObservationLog(dispatchDir, {})).toBe(path.join(workspace, 'tool-executions.jsonl'));
    expect(resolveObservationLog(dispatchDir)).toBe(path.join(workspace, 'tool-executions.jsonl'));
  });
});

describe('the transient dispatch records', () => {
  it('lists no record for a directory that does not exist', () => {
    expect(listDispatchRecords(path.join(workspace, 'absent'))).toEqual([]);
  });

  it('lists no record for a path that cannot be listed', () => {
    const notADirectory = path.join(workspace, 'a-file.txt');
    writeFileSync(notADirectory, 'x', 'utf8');
    expect(listDispatchRecords(notADirectory)).toEqual([]);
  });

  it('lists only the regular JSON files of the directory', () => {
    const record = writeRecord('plan__S-1.json', { slice_id: 'S-1' });
    writeFileSync(path.join(dispatchDir, 'notes.txt'), 'x', 'utf8');
    mkdirSync(path.join(dispatchDir, 'nested.json'));
    expect(listDispatchRecords(dispatchDir)).toEqual([record]);
  });
});

describe('the dispatch order the records carry', () => {
  it('reads the dispatch sequence number and the monotonic start base', () => {
    expect(readDispatchOrder({ dispatch_seq: 3, started_monotonic_ns: '0x10' })).toEqual({
      dispatchSeq: 3,
      startedMonotonicNs: 16n,
    });
  });

  it('falls back to the lowest order for a field that is absent or malformed', () => {
    expect(readDispatchOrder({ dispatch_seq: 'seven', started_monotonic_ns: 'nope' })).toEqual({
      dispatchSeq: -1,
      startedMonotonicNs: 0n,
    });
    expect(readDispatchOrder({ dispatch_seq: Number.NaN, started_monotonic_ns: 42 })).toEqual({
      dispatchSeq: -1,
      startedMonotonicNs: 0n,
    });
  });

  it('tolerates a record that is not an object', () => {
    expect(readDispatchOrder(null)).toEqual({ dispatchSeq: -1, startedMonotonicNs: 0n });
    expect(readDispatchOrder('not a record')).toEqual({ dispatchSeq: -1, startedMonotonicNs: 0n });
  });

  it('orders two records by the dispatch sequence number, most recent first', () => {
    const older = { dispatchSeq: 1, startedMonotonicNs: 0n, file: 'a.json' };
    const newer = { dispatchSeq: 2, startedMonotonicNs: 0n, file: 'b.json' };
    expect(compareDispatchOrder(newer, older)).toBe(-1);
    expect(compareDispatchOrder(older, newer)).toBe(1);
  });

  it('orders two records that share a sequence number by their start base, most recent first', () => {
    const older = { dispatchSeq: 2, startedMonotonicNs: 16n, file: 'a.json' };
    const newer = { dispatchSeq: 2, startedMonotonicNs: 32n, file: 'b.json' };
    expect(compareDispatchOrder(newer, older)).toBe(-1);
    expect(compareDispatchOrder(older, newer)).toBe(1);
  });

  it('orders two records that share both keys by their path', () => {
    const first = { dispatchSeq: 2, startedMonotonicNs: 16n, file: 'a.json' };
    const second = { dispatchSeq: 2, startedMonotonicNs: 16n, file: 'b.json' };
    expect(compareDispatchOrder(first, second)).toBe(-1);
    expect(compareDispatchOrder(second, first)).toBe(1);
  });
});

describe('the records the observer reads', () => {
  it('skips a record that carries no usable identity', () => {
    writeRecord('broken.json', 'this is not json');
    writeRecord('array.json', [1, 2, 3]);
    writeRecord('no-slice.json', { dispatch_seq: 1 });
    writeRecord('blank-slice.json', { slice_id: '   ' });
    writeRecord('wrong-type.json', { slice_id: 7 });
    expect(readDispatchRecords(dispatchDir)).toEqual([]);
  });

  it('orders the records by the dispatch order they carry', () => {
    writeRecord('older.json', { slice_id: 'S-1', dispatch_seq: 1, started_monotonic_ns: '0x10' });
    writeRecord('newest.json', { slice_id: 'S-3', dispatch_seq: 3, started_monotonic_ns: '0x10' });
    writeRecord('middle.json', { slice_id: 'S-2', dispatch_seq: 2, started_monotonic_ns: '0x10' });
    expect(readDispatchRecords(dispatchDir).map((record) => record.slice_id)).toEqual(['S-3', 'S-2', 'S-1']);
  });

  it('orders two records that share a sequence number by their start base', () => {
    writeRecord('slow.json', { slice_id: 'S-1', dispatch_seq: 2, started_monotonic_ns: '0x10' });
    writeRecord('fast.json', { slice_id: 'S-2', dispatch_seq: 2, started_monotonic_ns: '0x20' });
    expect(readDispatchRecords(dispatchDir).map((record) => record.slice_id)).toEqual(['S-2', 'S-1']);
  });

  it('orders two records that share a sequence number and a start base by their path', () => {
    writeRecord('a.json', { slice_id: 'S-1', dispatch_seq: 2, started_monotonic_ns: '0x10' });
    writeRecord('b.json', { slice_id: 'S-2', dispatch_seq: 2, started_monotonic_ns: '0x10' });
    expect(readDispatchRecords(dispatchDir).map((record) => record.slice_id)).toEqual(['S-1', 'S-2']);
  });

  it('orders the records of a directory that mixes every order key', () => {
    writeRecord('a-high.json', { slice_id: 'S-HIGH', dispatch_seq: 2, started_monotonic_ns: '0x20' });
    writeRecord('b-low.json', { slice_id: 'S-LOW', dispatch_seq: 2, started_monotonic_ns: '0x10' });
    writeRecord('c-newest.json', { slice_id: 'S-NEWEST', dispatch_seq: 3, started_monotonic_ns: '0x01' });
    writeRecord('d-oldest.json', { slice_id: 'S-OLDEST', dispatch_seq: 1, started_monotonic_ns: '0xff' });
    expect(readDispatchRecords(dispatchDir).map((record) => record.slice_id)).toEqual([
      'S-NEWEST',
      'S-HIGH',
      'S-LOW',
      'S-OLDEST',
    ]);
  });
});

describe('the open slice the observer resolves', () => {
  it('resolves no slice when the directory holds no record', () => {
    expect(readOpenSliceId(dispatchDir)).toBeNull();
  });

  it('resolves the most recently dispatched slice', () => {
    writeRecord('older.json', { slice_id: 'S-1', dispatch_seq: 1 });
    writeRecord('newest.json', { slice_id: 'S-3', dispatch_seq: 3 });
    expect(readOpenSliceId(dispatchDir)).toBe('S-3');
  });

  it('resolves no slice when two records tie on both keys and name different slices', () => {
    writeRecord('a.json', { slice_id: 'S-1', dispatch_seq: 2, started_monotonic_ns: '0x10' });
    writeRecord('b.json', { slice_id: 'S-2', dispatch_seq: 2, started_monotonic_ns: '0x10' });
    expect(readOpenSliceId(dispatchDir)).toBeNull();
  });

  it('resolves the slice when two records tie on both keys and name the SAME slice', () => {
    writeRecord('a.json', { slice_id: 'S-1', dispatch_seq: 2, started_monotonic_ns: '0x10' });
    writeRecord('b.json', { slice_id: 'S-1', dispatch_seq: 2, started_monotonic_ns: '0x10' });
    expect(readOpenSliceId(dispatchDir)).toBe('S-1');
  });
});

describe('the safe bucket of a tool name', () => {
  it('reduces a tool name to one of the safe buckets', () => {
    expect(deriveBucket('functions.view')).toBe('read');
    expect(deriveBucket('functions.edit')).toBe('edit');
    expect(deriveBucket('functions.grep')).toBe('search');
    expect(deriveBucket('functions.exec')).toBe('exec');
  });

  it('reduces a bare tool name to one of the safe buckets', () => {
    expect(deriveBucket('view')).toBe('read');
    expect(deriveBucket('edit')).toBe('edit');
    expect(deriveBucket('grep')).toBe('search');
    expect(deriveBucket('exec')).toBe('exec');
  });

  it('falls back to the other bucket for a name that matches no rule', () => {
    expect(deriveBucket('functions.unknown')).toBe('other');
    expect(deriveBucket('')).toBe('other');
    expect(deriveBucket(undefined)).toBe('other');
  });
});

describe('the failure the payload reports', () => {
  it('reports no failure for a payload that carries none', () => {
    expect(hasError({})).toBe(false);
    expect(hasError({ error: null })).toBe(false);
    expect(hasError({ error: undefined })).toBe(false);
    expect(hasError({ error: '   ' })).toBe(false);
    expect(hasError({ error: [] })).toBe(false);
    expect(hasError({ error: {} })).toBe(false);
    expect(hasError({ error: false })).toBe(false);
  });

  it('reports a failure for every spelling that carries content', () => {
    expect(hasError({ error: 'boom' })).toBe(true);
    expect(hasError({ error: ['boom'] })).toBe(true);
    expect(hasError({ error: { message: 'boom' } })).toBe(true);
    expect(hasError({ error: true })).toBe(true);
    expect(hasError({ error: 7 })).toBe(true);
    expect(hasError({ error_message: 'boom' })).toBe(true);
    expect(hasError({ errorMessage: 'boom' })).toBe(true);
  });
});

describe('the session key the observer persists', () => {
  it('persists a truncated digest instead of the raw session id', () => {
    const key = deriveSessionKey('session-camel-9f3a7c21');
    expect(key).toMatch(/^[0-9a-f]{12}$/);
    expect(key).not.toContain('9f3a7c21');
  });

  it('digests the empty string for a payload that carries no session id', () => {
    expect(deriveSessionKey(undefined)).toBe(deriveSessionKey(''));
  });
});

describe('the observation log', () => {
  it('counts no line for a log that does not exist', () => {
    expect(countObservationLines(path.join(workspace, 'absent.jsonl'))).toBe(0);
  });

  it('counts only the non-empty lines of the log', () => {
    writeFileSync(observationLog, '{"seq":0}\n\n{"seq":1}\n', 'utf8');
    expect(countObservationLines(observationLog)).toBe(2);
  });

  it('counts no line for a log that cannot be read', () => {
    expect(countObservationLines(dispatchDir)).toBe(0);
  });

  it('appends one complete line', () => {
    expect(appendObservation(observationLog, { seq: 0, slice_id: 'S-1' })).toBe(true);
    expect(readFileSync(observationLog, 'utf8')).toBe('{"seq":0,"slice_id":"S-1"}\n');
  });

  it('creates the missing directory of the log and writes into it', () => {
    const nested = path.join(workspace, 'deep', 'nested', 'obs.jsonl');
    expect(appendObservation(nested, { seq: 0 })).toBe(true);
    expect(existsSync(nested)).toBe(true);
  });

  it('reports the failure when the directory of the log cannot be created', () => {
    const parent = path.join(workspace, 'a-file.txt');
    writeFileSync(parent, 'x', 'utf8');
    expect(appendObservation(path.join(parent, 'obs.jsonl'), { seq: 0 })).toBe(false);
  });

  it('reports the failure when the log itself cannot be appended to', () => {
    expect(appendObservation(dispatchDir, { seq: 0 })).toBe(false);
  });
});

describe('the observation the observer writes', () => {
  it('writes one line carrying exactly the five safe fields', () => {
    writeRecord('plan__OT-V2-3.json', { slice_id: 'OT-V2-3', dispatch_seq: 4 });
    const env = {
      NEATAPTIC_DISPATCH_RECORD_DIR: dispatchDir,
      NEATAPTIC_OBSERVATION_LOG_PATH: observationLog,
    };
    expect(observe({ sessionId: 'session-camel-9f3a7c21', toolName: 'functions.edit' }, env)).toBe(true);
    const lines = readObservations();
    expect(lines).toHaveLength(1);
    expect(Object.keys(lines[0]).sort()).toEqual(['bucket', 'outcome', 'seq', 'session_key', 'slice_id']);
    expect(lines[0].slice_id).toBe('OT-V2-3');
    expect(lines[0].bucket).toBe('edit');
    expect(lines[0].outcome).toBe('completed');
    expect(lines[0].seq).toBe(0);
    expect(lines[0].session_key).toBe(deriveSessionKey('session-camel-9f3a7c21'));
  });

  it('reports the failure of the tool call it observed', () => {
    writeRecord('plan__OT-V2-3.json', { slice_id: 'OT-V2-3', dispatch_seq: 4 });
    const env = {
      NEATAPTIC_DISPATCH_RECORD_DIR: dispatchDir,
      NEATAPTIC_OBSERVATION_LOG_PATH: observationLog,
    };
    expect(observe({ session_id: 's-2', tool_name: 'functions.view', error: 'boom' }, env)).toBe(true);
    const lines = readObservations();
    expect(lines[0].outcome).toBe('failed');
    expect(lines[0].bucket).toBe('read');
    expect(lines[0].session_key).toBe(deriveSessionKey('s-2'));
  });

  it('writes nothing when no dispatch record is open', () => {
    const env = {
      NEATAPTIC_DISPATCH_RECORD_DIR: dispatchDir,
      NEATAPTIC_OBSERVATION_LOG_PATH: observationLog,
    };
    expect(observe({ sessionId: 's-1', toolName: 'functions.view' }, env)).toBe(false);
    expect(existsSync(observationLog)).toBe(false);
  });

  it('observes a payload that carries no tool name and no session id', () => {
    writeRecord('plan__OT-V2-3.json', { slice_id: 'OT-V2-3', dispatch_seq: 4 });
    const env = {
      NEATAPTIC_DISPATCH_RECORD_DIR: dispatchDir,
      NEATAPTIC_OBSERVATION_LOG_PATH: observationLog,
    };
    expect(observe({ sessionID: 's-3' }, env)).toBe(true);
    const lines = readObservations();
    expect(lines[0].session_key).toBe(deriveSessionKey('s-3'));
    expect(lines[0].bucket).toBe('other');
    expect(lines[0].outcome).toBe('completed');
  });

  it('observes a payload that carries no session id at all', () => {
    writeRecord('plan__OT-V2-3.json', { slice_id: 'OT-V2-3', dispatch_seq: 4 });
    const env = {
      NEATAPTIC_DISPATCH_RECORD_DIR: dispatchDir,
      NEATAPTIC_OBSERVATION_LOG_PATH: observationLog,
    };
    expect(observe({ toolName: 'functions.view' }, env)).toBe(true);
    expect(readObservations()[0].session_key).toBe(deriveSessionKey(''));
  });
});

describe('the manifest registration of the observer', () => {
  it('registers the observer under both post event keys, with the same command and no matcher', () => {
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, 'utf8'));
    const verdict = inspectObserverRegistration(manifest);
    expect(verdict.registered).toBe(true);
    expect(verdict.complete).toBe(true);
    expect(verdict.missing).toEqual([]);
    expect(verdict.registered_under).toEqual([...OBSERVER_EVENT_KEYS]);
    expect(verdict.same_command).toBe(true);
    expect(verdict.matcher_free).toBe(true);
    expect(verdict.script_path).toBe(OBSERVER_RELPATH);
    expect(existsSync(path.join(REPO_ROOT, OBSERVER_RELPATH))).toBe(true);
  });

  it('THE ANTI-VACUITY LANE: the verdict FAILS when the failure event key is removed', () => {
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, 'utf8'));
    const stripped = {
      ...manifest,
      hooks: { ...manifest.hooks, PostToolUseFailure: [] },
    };
    const verdict = inspectObserverRegistration(stripped);
    expect(verdict.registered).toBe(false);
    expect(verdict.complete).toBe(false);
    expect(verdict.missing).toContain('PostToolUseFailure');
  });

  it('THE ANTI-VACUITY LANE: the verdict FAILS when a matcher is added to the registration', () => {
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, 'utf8'));
    const matched = {
      ...manifest,
      hooks: {
        ...manifest.hooks,
        PostToolUse: manifest.hooks.PostToolUse.map((entry) =>
          entry.command.includes(OBSERVER_RELPATH) ? { ...entry, matcher: 'view' } : entry,
        ),
      },
    };
    const verdict = inspectObserverRegistration(matched);
    expect(verdict.complete).toBe(false);
    expect(verdict.matcher_free).toBe(false);
  });

  it('THE ANTI-VACUITY LANE: the verdict FAILS when the two registrations carry different commands', () => {
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, 'utf8'));
    const drifted = {
      ...manifest,
      hooks: {
        ...manifest.hooks,
        PostToolUseFailure: [{ command: 'node scripts/agent-customization/hooks/other-hook.mjs' }],
      },
    };
    const verdict = inspectObserverRegistration(drifted);
    expect(verdict.complete).toBe(false);
    expect(verdict.same_command).toBe(false);
    expect(verdict.missing).toContain('PostToolUseFailure');
  });
});
