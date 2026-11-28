/**
 * @module prune-dispatch-records.test
 * @description The home of the SAFE RESIDUE PRUNER of the transient
 * dispatch-record directory (R-8).
 *
 * Slice: OT-V2-3 of `plans/opentelemetry-slice-telemetry.plans.md`.
 *
 * THE SUBJECT UNDER TEST IS `scripts/telemetry/prune-dispatch-records.mjs`,
 * THE PRUNER. The lanes drive the REAL pruner IN PROCESS over a REAL
 * temporary dispatch directory and the REAL CLI as a REAL child process.
 *
 * THE ONE RULE THE PRUNER OWNS: a record is removed ONLY when the ONE
 * scanner of the contract module (`scanPersistedText`) reports a violation in
 * its own bytes, that is when the CURRENT writer could never have produced
 * it. Every contract-clean record - including every live pointer of a
 * dispatched slice - is KEPT, no matter how old it is.
 *
 * WHY THIS HOME EXISTS: the pruner is the command that keeps the dispatch
 * directory a POINTER DIRECTORY instead of a pile of residue. A pruner
 * without a home of its own is a pruner nobody can prove is safe, so the
 * home lives next to the pruner.
 *
 * THE PRUNER NEVER GUESSES: a record whose bytes cannot be read is KEPT,
 * because a record nobody could judge is not a record nobody may remove.
 *
 * THE REPORT IS STDOUT-ONLY: a removed record may carry a forbidden value in
 * its NAME, so the report must never be persisted into a plan or a log.
 */

import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  DEFAULT_DISPATCH_DIR,
  EXIT_UNKNOWN_FLAG,
  judgeRecord,
  listRecords,
  main,
  parseArgs,
  pruneDispatchRecords,
} from './prune-dispatch-records.mjs';
import { writeDispatchRecord } from './telemetry-contract.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');

/** The repo-relative path of the pruner this home owns. */
const PRUNER_RELPATH = 'scripts/telemetry/prune-dispatch-records.mjs';
/** The absolute path of the pruner this home owns. */
const PRUNER_PATH = path.join(REPO_ROOT, 'scripts', 'telemetry', 'prune-dispatch-records.mjs');

/** The bytes of a LIVE POINTER of a dispatched slice: contract-clean. */
const LIVE_RECORD = `${JSON.stringify(
  {
    plan_id: 'opentelemetry-slice-telemetry',
    slice_id: 'OT-V2-3',
    context_tokens_delivered: 9963,
    dispatch_seq: 0,
    started_monotonic_ns: '0x23408ef5fd628',
  },
  null,
  2,
)}\n`;

/** The bytes of a RESIDUE record: they carry a forbidden key. */
const RESIDUE_RECORD = `${JSON.stringify(
  { plan_id: 'opentelemetry-slice-telemetry', slice_id: 'OT-TEST-1', timestamp: 1764147453321 },
  null,
  2,
)}\n`;

/** The temporary workspace of the running lane. */
let workspace;
/** The temporary dispatch-record directory of the running lane. */
let dispatchDir;

/**
 * Creates the temporary workspace of the running lane.
 *
 * @returns {{ root: string, dispatchDir: string }} The workspace.
 */
function createWorkspace() {
  const root = mkdtempSync(path.join(tmpdir(), 'prune-dispatch-records-'));
  const dir = path.join(root, 'dispatch');
  mkdirSync(dir, { recursive: true });
  return { root, dispatchDir: dir };
}

/**
 * Writes a dispatch record into the temporary dispatch directory.
 *
 * @param {string} name The file name of the record.
 * @param {string} content The bytes of the record.
 */
function writeRecord(name, content) {
  writeFileSync(path.join(dispatchDir, name), content, 'utf8');
}

/**
 * Runs the REAL pruner as a REAL child process.
 *
 * @param {string[]} [extra] The extra command-line arguments.
 * @returns {{ status: number|null, stdout: string, stderr: string, error?: Error }} The run result.
 */
function runPruner(extra = []) {
  const result = spawnSync(process.execPath, [PRUNER_PATH, '--json', ...extra], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 60000,
  });
  return {
    status: result.status,
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
    error: result.error,
  };
}

/**
 * Parses the JSON report a command prints.
 *
 * @param {string} text The output text.
 * @returns {Record<string, unknown>|null} The report, or null when there is none.
 */
function parseJsonReport(text) {
  const candidates = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.startsWith('{'));
  for (let index = candidates.length - 1; index >= 0; index -= 1) {
    try {
      const parsed = JSON.parse(candidates[index]);
      if (parsed !== null && typeof parsed === 'object') return parsed;
    } catch {
      // not the report line
    }
  }
  return null;
}

/**
 * Asserts that the subject is on disk, so a lane that fails afterwards fails
 * on the SUBJECT's behaviour rather than on a missing file.
 *
 * @param {string} subjectPath The absolute path of the subject.
 * @param {string} subjectRelpath The repo-relative path of the subject.
 */
function expectSubjectAvailable(subjectPath, subjectRelpath) {
  expect({ subject: subjectRelpath, exists: existsSync(subjectPath) }).toEqual({
    subject: subjectRelpath,
    exists: true,
  });
}

beforeEach(() => {
  workspace = createWorkspace();
  dispatchDir = workspace.dispatchDir;
});

afterEach(() => {
  rmSync(workspace.root, { recursive: true, force: true });
});

describe('OT-V2-3: the dispatch-record pruner', () => {
  it('exists at the path the contract declares', () => {
    expectSubjectAvailable(PRUNER_PATH, PRUNER_RELPATH);
  });

  it('keeps a contract-clean record: a live pointer is never removed', () => {
    writeRecord('opentelemetry-slice-telemetry__OT-V2-3.json', LIVE_RECORD);
    const report = pruneDispatchRecords(dispatchDir);
    expect(report.inspected).toBe(1);
    expect(report.kept).toBe(1);
    expect(report.removed).toEqual([]);
    expect(existsSync(path.join(dispatchDir, 'opentelemetry-slice-telemetry__OT-V2-3.json'))).toBe(true);
  });

  it('keeps the OPEN record the REAL dispatch-record producer writes', () => {
    const target = writeDispatchRecord(
      {
        plan_id: 'opentelemetry-slice-telemetry',
        slice_id: 'OT-V2-3',
        context_tokens_delivered: 9963,
        dispatch_seq: 0,
        started_monotonic_ns: `0x${process.hrtime.bigint().toString(16)}`,
      },
      { directory: dispatchDir },
    );
    const report = pruneDispatchRecords(dispatchDir);
    expect(report.inspected).toBe(1);
    expect(report.kept).toBe(1);
    expect(report.removed).toEqual([]);
    expect(existsSync(target)).toBe(true);
  });

  it('removes a record whose own bytes violate the persisted-artefact contract', () => {
    writeRecord('opentelemetry-slice-telemetry__OT-TEST-1.json', RESIDUE_RECORD);
    const report = pruneDispatchRecords(dispatchDir);
    expect(report.inspected).toBe(1);
    expect(report.removed).toEqual([path.join(dispatchDir, 'opentelemetry-slice-telemetry__OT-TEST-1.json')]);
    expect(report.kept).toBe(0);
    expect(existsSync(path.join(dispatchDir, 'opentelemetry-slice-telemetry__OT-TEST-1.json'))).toBe(false);
  });

  it('keeps the live pointer and removes the residue in a MIXED directory', () => {
    writeRecord('opentelemetry-slice-telemetry__OT-V2-3.json', LIVE_RECORD);
    writeRecord('opentelemetry-slice-telemetry__OT-TEST-1.json', RESIDUE_RECORD);
    const report = pruneDispatchRecords(dispatchDir);
    expect(report.inspected).toBe(2);
    expect(report.kept).toBe(1);
    expect(report.removed).toHaveLength(1);
    expect(existsSync(path.join(dispatchDir, 'opentelemetry-slice-telemetry__OT-V2-3.json'))).toBe(true);
  });

  it('removes NOTHING in a dry run but NAMES the removable records', () => {
    writeRecord('opentelemetry-slice-telemetry__OT-TEST-1.json', RESIDUE_RECORD);
    const report = pruneDispatchRecords(dispatchDir, { dryRun: true });
    expect(report.dry_run).toBe(true);
    expect(report.removed).toHaveLength(1);
    expect(existsSync(path.join(dispatchDir, 'opentelemetry-slice-telemetry__OT-TEST-1.json'))).toBe(true);
  });

  it('keeps a record nobody could judge', () => {
    const verdict = judgeRecord(path.join(dispatchDir, 'absent.json'));
    expect(verdict.readable).toBe(false);
    expect(verdict.removable).toBe(false);
    expect(verdict.violations).toBe(0);
  });

  it('takes only regular .json files as candidates', () => {
    writeRecord('live.json', LIVE_RECORD);
    writeRecord('notes.txt', 'not a record\n');
    mkdirSync(path.join(dispatchDir, 'folder.json'), { recursive: true });
    const candidates = listRecords(dispatchDir);
    expect(candidates).toEqual([path.join(dispatchDir, 'live.json')]);
  });

  it('inspects NOTHING when the dispatch directory does not exist', () => {
    const report = pruneDispatchRecords(path.join(workspace.root, 'absent'));
    expect(report.inspected).toBe(0);
    expect(report.kept).toBe(0);
    expect(report.removed).toEqual([]);
  });

  it('inspects NOTHING when the dispatch path is a FILE', () => {
    const file = path.join(workspace.root, 'not-a-directory');
    writeFileSync(file, 'x\n', 'utf8');
    expect(listRecords(file)).toEqual([]);
    const report = pruneDispatchRecords(file);
    expect(report.inspected).toBe(0);
  });
});

describe('OT-V2-3: the dispatch-record pruner CLI', () => {
  it('prints the report and exits zero when nothing failed', () => {
    writeRecord('opentelemetry-slice-telemetry__OT-V2-3.json', LIVE_RECORD);
    const result = runPruner([`--dispatch-dir=${dispatchDir}`]);
    expect(result.status).toBe(0);
    const report = parseJsonReport(result.stdout);
    expect(report?.inspected ?? null).toBe(1);
    expect(report?.kept ?? null).toBe(1);
    expect(report?.removed ?? null).toEqual([]);
    expect(report?.dispatch_dir ?? null).toBe(path.resolve(dispatchDir));
  });

  it('removes the residue when it really runs', () => {
    writeRecord('opentelemetry-slice-telemetry__OT-TEST-1.json', RESIDUE_RECORD);
    const result = runPruner([`--dispatch-dir=${dispatchDir}`]);
    expect(result.status).toBe(0);
    const report = parseJsonReport(result.stdout);
    expect(report?.removed ?? null).toHaveLength(1);
    expect(existsSync(path.join(dispatchDir, 'opentelemetry-slice-telemetry__OT-TEST-1.json'))).toBe(false);
  });

  it('removes NOTHING in a dry run', () => {
    writeRecord('opentelemetry-slice-telemetry__OT-TEST-1.json', RESIDUE_RECORD);
    const result = runPruner([`--dispatch-dir=${dispatchDir}`, '--dry-run']);
    expect(result.status).toBe(0);
    const report = parseJsonReport(result.stdout);
    expect(report?.dry_run ?? null).toBe(true);
    expect(report?.removed ?? null).toHaveLength(1);
    expect(existsSync(path.join(dispatchDir, 'opentelemetry-slice-telemetry__OT-TEST-1.json'))).toBe(true);
  });

  it('refuses an unknown flag and removes NOTHING', () => {
    writeRecord('opentelemetry-slice-telemetry__OT-TEST-1.json', RESIDUE_RECORD);
    const result = runPruner(['--dispatch-dir', dispatchDir, '--dryrun']);
    expect(result.status).toBe(EXIT_UNKNOWN_FLAG);
    expect(parseJsonReport(result.stderr)?.error ?? null).toBe('unknown_flag');
    expect(existsSync(path.join(dispatchDir, 'opentelemetry-slice-telemetry__OT-TEST-1.json'))).toBe(true);
  });

  it('resolves a relative dispatch directory against the repository root', () => {
    const result = runPruner([`--dispatch-dir=${DEFAULT_DISPATCH_DIR}`, '--dry-run']);
    expect(result.status).toBe(0);
    const report = parseJsonReport(result.stdout);
    expect(report?.dispatch_dir ?? null).toBe(path.resolve(REPO_ROOT, DEFAULT_DISPATCH_DIR));
  });
});

describe('OT-V2-3: the dispatch-record pruner IN PROCESS', () => {
  /** @type {import('@jest/globals').jest.SpyInstance} */
  let stdoutSpy;
  /** @type {import('@jest/globals').jest.SpyInstance} */
  let stderrSpy;

  beforeEach(() => {
    stdoutSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
    stderrSpy = jest.spyOn(process.stderr, 'write').mockImplementation(() => true);
  });

  afterEach(() => {
    stdoutSpy.mockRestore();
    stderrSpy.mockRestore();
  });

  it('parses both spellings of the dispatch-directory flag', () => {
    expect(parseArgs(['--json', '--dry-run', '--dispatch-dir=logs/x'])).toEqual({
      json: true,
      dispatchDir: 'logs/x',
      dryRun: true,
      unknown: [],
    });
    expect(parseArgs(['--dispatch-dir', 'logs/x'])).toEqual({
      json: false,
      dispatchDir: 'logs/x',
      dryRun: false,
      unknown: [],
    });
  });

  it('reports an unknown flag instead of ignoring it', () => {
    expect(parseArgs(['--nope']).unknown).toEqual(['--nope']);
    expect(parseArgs(['--dispatch-dir']).unknown).toEqual(['--dispatch-dir']);
  });

  it('prints the report and exits zero', () => {
    writeRecord('live.json', LIVE_RECORD);
    stdoutSpy.mockClear();
    const status = main(['--json', `--dispatch-dir=${dispatchDir}`]);
    expect(status).toBe(0);
    const report = parseJsonReport(stdoutSpy.mock.calls.map((call) => String(call[0])).join(''));
    expect(report?.inspected ?? null).toBe(1);
    expect(report?.kept ?? null).toBe(1);
  });

  it('refuses an unknown flag without removing anything', () => {
    writeRecord('opentelemetry-slice-telemetry__OT-TEST-1.json', RESIDUE_RECORD);
    stdoutSpy.mockClear();
    stderrSpy.mockClear();
    const status = main(['--json', `--dispatch-dir=${dispatchDir}`, '--nope']);
    expect(status).toBe(EXIT_UNKNOWN_FLAG);
    expect(stdoutSpy.mock.calls.map((call) => String(call[0])).join('')).toBe('');
    expect(parseJsonReport(stderrSpy.mock.calls.map((call) => String(call[0])).join(''))?.error ?? null).toBe(
      'unknown_flag',
    );
    expect(existsSync(path.join(dispatchDir, 'opentelemetry-slice-telemetry__OT-TEST-1.json'))).toBe(true);
  });
});
