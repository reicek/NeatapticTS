/**
 * @module privacy-sweep.test
 * @description The home of the privacy sweep CLI of the vertical slice B.
 *
 * Slice: OT-V2-3 of `plans/opentelemetry-slice-telemetry.plans.md`.
 * Acceptance criteria pinned by this home: AC-V2-F (the privacy sweep) and
 * the OT-V2-3 iteration-1 repair of the sweep's DEFAULT target set, of its
 * per-finding `file` attribution and of its refusal of an unknown flag.
 *
 * THE SUBJECT UNDER TEST IS `scripts/telemetry/privacy-sweep.mjs`, THE
 * PRIVACY-SWEEP CLI. The lanes drive the REAL CLI as a REAL child process
 * over REAL temporary artefacts and - for the exported helpers - IN PROCESS.
 *
 * WHY THE SWEEP HAS ITS OWN HOME: the sweep is a subject in its own right,
 * not a helper of the enablement command. Its home used to live inside the
 * enablement home, where a missing file was invisible: the enablement lanes
 * could not fail because the sweep had no lane of its own. The sweep home
 * lives here, next to its subject, so `scripts/telemetry/privacy-sweep.test.mjs`
 * exists and the sweep has a home of its own.
 *
 * THE SWEEP IS A THIN WRAPPER: the rules live in exactly ONE place -
 * `scanPersistedText` of `scripts/telemetry/telemetry-contract.mjs` - and the
 * sweep introduces NO second implementation of them. A lane reads the source
 * and proves that no second copy of the rules was added.
 *
 * THE SWEEP COVERS EVERYTHING: the report carries an ALWAYS EMPTY `excluded`
 * list, because a persisted artefact is never excused from the privacy rules.
 *
 * THE DEFAULT TARGET SET MUST BE REAL: every default target names an
 * artefact that REALLY EXISTS, and a default run really scans at least one
 * file. A default set of phantom files would make the sweep vacuously green,
 * so the anti-vacuity lanes below assert BOTH the existence of every default
 * target AND a non-empty scan.
 *
 * THE SWEEP REFUSES AN UNKNOWN FLAG: a mistyped option is an ERROR, not a
 * silently ignored token, so `--targtes=...` can never make the sweep scan
 * the default set instead of the set the operator asked for. An unknown flag
 * makes the command exit `2` and scan NOTHING.
 *
 * EVERY FINDING NAMES ITS FILE: a violation carries the REPO-RELATIVE path of
 * the artefact it was found in, so a report says WHERE a violation is.
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

import { scanPersistedText } from './telemetry-contract.mjs';
import {
  DEFAULT_TARGETS,
  EXIT_UNKNOWN_FLAG,
  expandTarget,
  main,
  parseArgs,
  resolveTarget,
  scanTargets,
  splitTargets,
  toRepoRelative,
} from './privacy-sweep.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');

/** The repo-relative path of the sweep CLI this home owns. */
const SWEEP_RELPATH = 'scripts/telemetry/privacy-sweep.mjs';
/** The absolute path of the sweep CLI this home owns. */
const SWEEP_PATH = path.join(REPO_ROOT, 'scripts', 'telemetry', 'privacy-sweep.mjs');

/** The temporary workspace of the running lane. */
let workspace;

/**
 * Creates the temporary workspace of the running lane.
 *
 * @returns {{ root: string }} The workspace.
 */
function createWorkspace() {
  return { root: mkdtempSync(path.join(tmpdir(), 'privacy-sweep-')) };
}

/**
 * Runs the REAL sweep CLI as a REAL child process.
 *
 * @param {string[]} [extra] The extra command-line arguments.
 * @returns {{ status: number|null, stdout: string, stderr: string, output: string, error?: Error }} The run result.
 */
function runSweep(extra = []) {
  const result = spawnSync(process.execPath, [SWEEP_PATH, '--json', ...extra], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 60000,
  });
  const stdout = result.stdout ?? '';
  const stderr = result.stderr ?? '';
  return {
    status: result.status,
    stdout,
    stderr,
    output: `${stdout}\n${stderr}`,
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
});

afterEach(() => {
  rmSync(workspace.root, { recursive: true, force: true });
});

describe('OT-V2-3: the privacy sweep CLI', () => {
  it('exists at the path the contract declares', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
  });

  it('reports zero violations for a clean artefact', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const clean = path.join(workspace.root, 'clean.jsonl');
    writeFileSync(clean, '{"bucket":"read","outcome":"completed","seq":1}\n', 'utf8');
    const result = runSweep([`--targets=${clean}`]);
    expect(result.status).toBe(0);
    const report = parseJsonReport(result.stdout);
    expect(report?.violations ?? null).toEqual([]);
  });

  it('reports exactly the findings of the ONE scanner, verbatim', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const dirty = path.join(workspace.root, 'dirty.jsonl');
    const text = '  timestamp: 1764147453321\n  bucket: read\n';
    writeFileSync(dirty, text, 'utf8');
    const result = runSweep([`--targets=${dirty}`]);
    const expected = scanPersistedText(text);
    expect(expected.length).toBeGreaterThan(0);
    const report = parseJsonReport(result.stdout);
    const reported = report?.violations ?? null;
    expect(reported).toHaveLength(expected.length);
    expect(reported).toEqual(
      expect.arrayContaining(expected.map((violation) => expect.objectContaining(violation))),
    );
  });

  it('refuses when it finds a violation', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const dirty = path.join(workspace.root, 'dirty.jsonl');
    writeFileSync(dirty, '  timestamp: 1764147453321\n', 'utf8');
    const result = runSweep([`--targets=${dirty}`]);
    expect(result.status).not.toBe(0);
  });

  it('covers every target artefact with no exclusion', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const first = path.join(workspace.root, 'first.jsonl');
    const second = path.join(workspace.root, 'second.jsonl');
    const third = path.join(workspace.root, 'third.jsonl');
    writeFileSync(first, '{"seq":1}\n', 'utf8');
    writeFileSync(second, '{"seq":2}\n', 'utf8');
    writeFileSync(third, '  timestamp: 1764147453321\n', 'utf8');
    const result = runSweep([`--targets=${[first, second, third].join(',')}`]);
    const report = parseJsonReport(result.stdout);
    const scanned = report?.scanned ?? [];
    expect(scanned.length).toBe(3);
    expect(report?.excluded ?? null).toEqual([]);
    expect(scanPersistedText(readFileSync(third, 'utf8')).length).toBeGreaterThan(0);
  });

  it('names the artefact of every finding', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const dirty = path.join(workspace.root, 'dirty.jsonl');
    writeFileSync(dirty, '  timestamp: 1764147453321\n', 'utf8');
    const result = runSweep([`--targets=${dirty}`]);
    const report = parseJsonReport(result.stdout);
    const violations = report?.violations ?? [];
    expect(violations.length).toBeGreaterThan(0);
    for (const violation of violations) {
      expect(typeof violation.file).toBe('string');
      // The artefact lives OUTSIDE the repository root, so it keeps its
      // absolute path; a repo-relative form does not exist for it.
      expect(violation.file).toBe(path.resolve(dirty));
      expect(violation.file).toContain('dirty.jsonl');
    }
  });

  it('treats a target that does not exist as NOTHING, not as a violation', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const missing = path.join(workspace.root, 'missing.jsonl');
    const result = runSweep([`--targets=${missing}`]);
    expect(result.status).toBe(0);
    const report = parseJsonReport(result.stdout);
    expect(report?.scanned ?? null).toEqual([]);
    expect(report?.violations ?? null).toEqual([]);
  });

  it('is a thin wrapper: it introduces no second implementation of the rules', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const source = readFileSync(SWEEP_PATH, 'utf8');
    expect(source).toContain('scanPersistedText');
    expect(source).toContain('telemetry-contract.mjs');
    // A second implementation would carry its own copy of the patterns; the
    // calendar-date pattern is the sharpest probe for that.
    expect(source).not.toContain('\\d{4}-\\d{2}-\\d{2}');
  });

  it('never rewrites the artefacts it scans', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const dirty = path.join(workspace.root, 'dirty.jsonl');
    const text = '  timestamp: 1764147453321\n';
    writeFileSync(dirty, text, 'utf8');
    runSweep([`--targets=${dirty}`]);
    expect(readFileSync(dirty, 'utf8')).toBe(text);
  });
});

describe('OT-V2-3: the privacy sweep DEFAULT target set is REAL', () => {
  it('falls back to the DEFAULT artefact set of the lane when no target is named', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const result = runSweep();
    const report = parseJsonReport(result.stdout);
    expect(report?.targets ?? null).toEqual([...DEFAULT_TARGETS]);
  });

  it('names ONLY artefacts that really exist, so the default set is not vacuous', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    for (const target of DEFAULT_TARGETS) {
      const resolved = resolveTarget(target);
      expect({ target, exists: existsSync(resolved) }).toEqual({
        target,
        exists: true,
      });
    }
  });

  it('really scans at least one artefact when it runs with the default set', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const result = runSweep();
    const report = parseJsonReport(result.stdout);
    const scanned = report?.scanned ?? [];
    expect(scanned.length).toBeGreaterThan(0);
    for (const file of scanned) {
      expect(typeof file).toBe('string');
      expect(file).not.toMatch(/^[A-Za-z]:/u);
    }
  });

  it('finds no forbidden value in the live artefacts of the lane', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const result = runSweep();
    expect(result.status).toBe(0);
    const report = parseJsonReport(result.stdout);
    expect(report?.violations ?? null).toEqual([]);
  });
});

describe('OT-V2-3: the privacy sweep REFUSES an unknown flag', () => {
  it('exits with the refusal code and reports the unknown flag on stderr', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const result = runSweep(['--targtes=logs/telemetry']);
    expect(result.status).toBe(EXIT_UNKNOWN_FLAG);
    const report = parseJsonReport(result.stderr);
    expect(report?.error ?? null).toBe('unknown_flag');
    expect(report?.unknown ?? null).toEqual(['--targtes=logs/telemetry']);
  });

  it('scans NOTHING when the flag is unknown', () => {
    expectSubjectAvailable(SWEEP_PATH, SWEEP_RELPATH);
    const result = runSweep(['--targtes=logs/telemetry']);
    expect(result.stdout.trim()).toBe('');
  });
});

describe('OT-V2-3: the privacy sweep IN PROCESS', () => {
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

  /**
   * Runs the sweep in process and captures its report.
   *
   * @param {string[]} argv The command line.
   * @returns {{ status: number, stdout: string, stderr: string, report: Record<string, unknown>|null }} The run result.
   */
  function runInProcess(argv) {
    stdoutSpy.mockClear();
    stderrSpy.mockClear();
    const status = main(argv);
    const stdout = stdoutSpy.mock.calls.map((call) => String(call[0])).join('');
    const stderr = stderrSpy.mock.calls.map((call) => String(call[0])).join('');
    return { status, stdout, stderr, report: parseJsonReport(stdout) };
  }

  it('parses both spellings of the target flag', () => {
    expect(parseArgs(['--json', '--targets=a.jsonl,b.jsonl'])).toEqual({
      json: true,
      targets: ['a.jsonl', 'b.jsonl'],
      unknown: [],
    });
    expect(parseArgs(['--targets', 'a.jsonl'])).toEqual({
      json: false,
      targets: ['a.jsonl'],
      unknown: [],
    });
  });

  it('reports an unknown flag instead of ignoring it', () => {
    expect(parseArgs(['--json', '--nope']).unknown).toEqual(['--nope']);
    expect(parseArgs(['--targets']).unknown).toEqual(['--targets']);
  });

  it('splits a target list and drops the empty entries', () => {
    expect(splitTargets('a.jsonl, b.jsonl')).toEqual(['a.jsonl', 'b.jsonl']);
    expect(splitTargets('')).toEqual([]);
    expect(splitTargets(undefined)).toEqual([]);
  });

  it('resolves a relative target against the repository root', () => {
    expect(resolveTarget('logs/telemetry')).toBe(path.resolve(REPO_ROOT, 'logs/telemetry'));
    expect(resolveTarget(path.resolve(workspace.root, 'x.jsonl'))).toBe(
      path.resolve(workspace.root, 'x.jsonl'),
    );
  });

  it('renders a repo-relative path in the forward-slash form', () => {
    expect(toRepoRelative(path.join(REPO_ROOT, 'logs', 'telemetry', 'trace.jsonl'))).toBe(
      'logs/telemetry/trace.jsonl',
    );
    expect(toRepoRelative(REPO_ROOT)).toBe('.');
  });

  it('keeps the absolute path of a file outside the repository root', () => {
    const outside = path.resolve(workspace.root, 'outside.jsonl');
    expect(toRepoRelative(outside)).toBe(outside);
  });

  it('expands a file target, a directory target and a missing target', () => {
    const file = path.join(workspace.root, 'one.jsonl');
    writeFileSync(file, '{"seq":1}\n', 'utf8');
    const directory = path.join(workspace.root, 'nested');
    mkdirSync(directory, { recursive: true });
    writeFileSync(path.join(directory, 'two.jsonl'), '{"seq":2}\n', 'utf8');
    expect(expandTarget(file)).toEqual([file]);
    expect(expandTarget(directory)).toEqual([path.join(directory, 'two.jsonl')]);
    expect(expandTarget(path.join(workspace.root, 'absent'))).toEqual([]);
  });

  it('scans a repeated target only once', () => {
    const file = path.join(workspace.root, 'one.jsonl');
    writeFileSync(file, '{"seq":1}\n', 'utf8');
    const report = scanTargets([file, file]);
    expect(report.scanned).toEqual([file]);
    expect(report.excluded).toEqual([]);
    expect(report.violations).toEqual([]);
  });

  it('exits zero and prints the report for a clean target', () => {
    const file = path.join(workspace.root, 'clean.jsonl');
    writeFileSync(file, '{"seq":1}\n', 'utf8');
    const run = runInProcess(['--json', `--targets=${file}`]);
    expect(run.status).toBe(0);
    expect(run.report?.violations ?? null).toEqual([]);
    expect(run.report?.targets ?? null).toEqual([file]);
  });

  it('refuses an unknown flag without scanning anything', () => {
    const run = runInProcess(['--json', '--nope']);
    expect(run.status).toBe(EXIT_UNKNOWN_FLAG);
    expect(run.stdout).toBe('');
    expect(parseJsonReport(run.stderr)?.error ?? null).toBe('unknown_flag');
  });
});
