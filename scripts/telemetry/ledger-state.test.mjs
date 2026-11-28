/**
 * @module ledger-state.test
 * @description RED contract home for the telemetry state classifier (AC-V2-C).
 *
 * Slice: OT-V2-1 of `plans/opentelemetry-slice-telemetry.plans.md`.
 *
 * THE SUBJECT UNDER TEST IS `scripts/telemetry/ledger-state.mjs`, the
 * CLASSIFIER. It decides the layer-A reporting state and the layer-B
 * measurement state of a slice and REPORTS them. It never writes a ledger,
 * never runs the writer's `--recover` mode, never repairs a transient artefact
 * and never creates or deletes one - the recovery is owned by the WRITER
 * (`scripts/telemetry/write-slice-ledger.mjs`).
 *
 * Every lane drives the REAL classifier as a REAL child process over REAL
 * temporary artefacts, and the "still yields a ledger" and "refuses the
 * transition" rules are proven through the REAL writer rather than through a
 * description. Nothing is mocked, and the subject does not exist yet, so every
 * lane is RED BY ASSERTION - never by an import error that would abort the
 * file.
 *
 * THE COMMAND LINE THIS HOME DEMANDS:
 *
 *   node scripts/telemetry/ledger-state.mjs --json --slice=<sliceId> \
 *     [--settings=<path>] [--sink=<dir>] [--dispatch-record=<path>] [--ledger=<path>]
 *
 *   --slice            REQUIRED. The slice whose state is classified.
 *   --settings         OPTIONAL. The local settings file. It defaults to
 *                      `logs/telemetry/settings.json` under the repository root.
 *   --sink             OPTIONAL. The local JSONL sink that holds the layer-A
 *                      records. It defaults to the settings directory.
 *   --dispatch-record  OPTIONAL. The transient dispatch record of the slice.
 *   --ledger           OPTIONAL. The artefact that carries the ledger block
 *                      (the companion log of the tracker).
 *   --json             The report is printed as ONE JSON object.
 *
 * THE REPORT THIS HOME DEMANDS:
 *   - `state`             - the layer-A reporting state: exactly one of
 *                           `ENABLED`, `DISABLED`, `STALE`, `CORRUPT` or
 *                           `NO-DATA`;
 *   - `artefact`          - the artefact that produced the verdict;
 *   - `reason`            - why that verdict was reached;
 *   - `repair_command`    - the exact remediation;
 *   - `measurement_state` - the layer-B measurement state: exactly one of
 *                           `MEASURED`, `UNAVAILABLE` or `CORRUPT`;
 *   - `duration_ms`       - the measured duration, or `null`.
 *
 * THE LAYER-A DETECTION RULES:
 *   - the local settings are MISSING, or they say `enabled: false` -> DISABLED;
 *   - the local settings carry an OLDER `schema_version` -> STALE;
 *   - the local settings are UNPARSEABLE -> CORRUPT;
 *   - the settings are current and enabled, but the sink holds NO layer-A
 *     record for THIS slice -> NO-DATA;
 *   - the settings are current and enabled and the sink holds a record for
 *     THIS slice -> ENABLED.
 *
 * THE LAYER-B DETECTION RULES:
 *   - the ledger says `MEASURED` -> MEASURED, with a numeric duration;
 *   - the ledger says `UNAVAILABLE`, or there is no ledger at all -> UNAVAILABLE,
 *     with a NULL duration - never a zero;
 *   - the transient dispatch record is unparseable -> CORRUPT.
 *
 * THE TWO RULES THAT PROVE THE INDEPENDENCE OF THE LAYERS:
 *   - a DISABLED or NO-DATA layer A still YIELDS a ledger: the real writer
 *     writes one whose `measurement_state` is `UNAVAILABLE` and whose
 *     `duration_ms` is `null`;
 *   - a CORRUPT layer-B artefact REFUSES the transition: the real writer
 *     refuses to write a block and exits non-zero.
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SCHEMA_VERSION } from './telemetry-contract.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');

/** The repo-relative path of the classifier this home owns. */
const CLASSIFIER_RELPATH = 'scripts/telemetry/ledger-state.mjs';
/** The absolute path of the classifier this home owns. */
const CLASSIFIER_PATH = path.join(REPO_ROOT, 'scripts', 'telemetry', 'ledger-state.mjs');
/** The repo-relative path of the DONE writer this home drives. */
const WRITER_RELPATH = 'scripts/telemetry/write-slice-ledger.mjs';
/** The absolute path of the DONE writer this home drives. */
const WRITER_PATH = path.join(REPO_ROOT, 'scripts', 'telemetry', 'write-slice-ledger.mjs');

const PLAN_ID = 'fixture-plan';
const SLICE_ID = 'OT-V2-9';

/** The temporary workspace of the running lane. */
let workspace;

/**
 * Creates the temporary workspace: a tracker, a local settings directory, a
 * sink directory and a transient dispatch-record directory.
 *
 * @returns {{ root: string, planPath: string, logPath: string, settingsDir: string, settingsPath: string, dispatchDir: string }} The workspace.
 */
function createWorkspace() {
  const root = mkdtempSync(path.join(tmpdir(), 'ledger-state-'));
  const plansDir = path.join(root, 'plans');
  const settingsDir = path.join(root, 'logs', 'telemetry');
  const dispatchDir = path.join(root, 'logs', 'slice-ledger', 'dispatch');
  mkdirSync(plansDir, { recursive: true });
  mkdirSync(settingsDir, { recursive: true });
  mkdirSync(dispatchDir, { recursive: true });
  const planPath = path.join(plansDir, `${PLAN_ID}.plans.md`);
  writeFileSync(planPath, '# Fixture tracker\n', 'utf8');
  return {
    root,
    planPath,
    logPath: path.join(plansDir, `${PLAN_ID}.logs.md`),
    settingsDir,
    settingsPath: path.join(settingsDir, 'settings.json'),
    dispatchDir,
  };
}

/**
 * Writes the local settings file.
 *
 * @param {unknown} settings The settings to write.
 */
function writeSettings(settings) {
  writeFileSync(
    workspace.settingsPath,
    typeof settings === 'string' ? settings : `${JSON.stringify(settings, null, 2)}\n`,
    'utf8',
  );
}

/**
 * Builds a valid local settings object.
 *
 * @param {Record<string, unknown>} [overrides] The settings overrides.
 * @returns {Record<string, unknown>} The settings.
 */
function validSettings(overrides = {}) {
  return {
    schema_version: SCHEMA_VERSION,
    enabled: true,
    signals: { traces: true, metrics: true, logs: true },
    exporter: { type: 'jsonl', path: path.join(workspace.settingsDir, 'telemetry.jsonl') },
    ...overrides,
  };
}

/**
 * Writes a layer-A record for this slice into the local sink.
 */
function writeLayerARecord() {
  writeFileSync(
    path.join(workspace.settingsDir, 'telemetry.jsonl'),
    `${JSON.stringify({ slice_id: SLICE_ID, span: 'the slice window', duration_ms: 12 })}\n`,
    'utf8',
  );
}

/**
 * Runs the REAL classifier as a REAL child process.
 *
 * @param {string[]} [extra] The extra command-line arguments.
 * @returns {{ status: number|null, stdout: string, stderr: string, output: string, error?: Error }} The run result.
 */
function runClassifier(extra = []) {
  const result = spawnSync(
    process.execPath,
    [
      CLASSIFIER_PATH,
      '--json',
      '--slice',
      SLICE_ID,
      '--settings',
      workspace.settingsPath,
      '--sink',
      workspace.settingsDir,
      ...extra,
    ],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 60000,
    },
  );
  const stdout = result.stdout ?? '';
  const stderr = result.stderr ?? '';
  return { status: result.status, stdout, stderr, output: `${stdout}\n${stderr}`, error: result.error };
}

/**
 * Runs the REAL DONE writer as a REAL child process.
 *
 * @param {string[]} [extra] The extra command-line arguments.
 * @returns {{ status: number|null, stdout: string, stderr: string, output: string, error?: Error }} The run result.
 */
function runWriter(extra = []) {
  const result = spawnSync(
    process.execPath,
    [
      WRITER_PATH,
      '--plan',
      workspace.planPath,
      '--slice',
      SLICE_ID,
      '--plan-id',
      PLAN_ID,
      '--dispatch-dir',
      workspace.dispatchDir,
      ...extra,
    ],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 60000,
    },
  );
  const stdout = result.stdout ?? '';
  const stderr = result.stderr ?? '';
  return { status: result.status, stdout, stderr, output: `${stdout}\n${stderr}`, error: result.error };
}

/**
 * Parses the JSON report the classifier prints.
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
 * Asserts that the classifier is on disk, so a lane that fails afterwards
 * fails on the CLASSIFIER's behaviour rather than on a missing file.
 */
function expectClassifierAvailable() {
  expect({ subject: CLASSIFIER_RELPATH, exists: existsSync(CLASSIFIER_PATH) }).toEqual({
    subject: CLASSIFIER_RELPATH,
    exists: true,
  });
}

beforeEach(() => {
  workspace = createWorkspace();
});

afterEach(() => {
  rmSync(workspace.root, { recursive: true, force: true });
});

describe('OT-V2-1: the layer-A reporting state', () => {
  it('exists at the path the contract declares', () => {
    expectClassifierAvailable();
  });

  it('reports DISABLED when the local settings are missing', () => {
    expectClassifierAvailable();
    rmSync(workspace.settingsPath, { force: true });
    const result = runClassifier();
    expect(result.status).toBe(0);
    expect(parseJsonReport(result.output)?.state).toBe('DISABLED');
  });

  it('reports DISABLED when the local settings say disabled', () => {
    expectClassifierAvailable();
    writeSettings(validSettings({ enabled: false }));
    const result = runClassifier();
    expect(result.status).toBe(0);
    expect(parseJsonReport(result.output)?.state).toBe('DISABLED');
  });

  it('reports STALE when the local settings carry an older schema version', () => {
    expectClassifierAvailable();
    writeSettings(validSettings({ schema_version: SCHEMA_VERSION - 1 }));
    const result = runClassifier();
    expect(result.status).toBe(0);
    expect(parseJsonReport(result.output)?.state).toBe('STALE');
  });

  it('reports CORRUPT when the local settings are unparseable', () => {
    expectClassifierAvailable();
    writeSettings('{"schema_version": 1, "signals": {"traces": true,,}\n');
    const result = runClassifier();
    expect(result.status).toBe(0);
    expect(parseJsonReport(result.output)?.state).toBe('CORRUPT');
  });

  it('reports NO-DATA when layer A is enabled and healthy but produced no record for this slice', () => {
    expectClassifierAvailable();
    writeSettings(validSettings());
    const result = runClassifier();
    expect(result.status).toBe(0);
    expect(parseJsonReport(result.output)?.state).toBe('NO-DATA');
  });

  it('reports ENABLED when layer A produced a record for this slice', () => {
    expectClassifierAvailable();
    writeSettings(validSettings());
    writeLayerARecord();
    const result = runClassifier();
    expect(result.status).toBe(0);
    expect(parseJsonReport(result.output)?.state).toBe('ENABLED');
  });

  it('names the artefact that produced the verdict', () => {
    expectClassifierAvailable();
    writeSettings(validSettings({ enabled: false }));
    const result = runClassifier();
    const report = parseJsonReport(result.output);
    expect(typeof report?.artefact).toBe('string');
    expect(String(report?.artefact)).toContain(path.basename(workspace.settingsPath));
  });

  it('carries a reason for the verdict', () => {
    expectClassifierAvailable();
    writeSettings(validSettings({ enabled: false }));
    const result = runClassifier();
    const report = parseJsonReport(result.output);
    expect(typeof report?.reason).toBe('string');
    expect(String(report?.reason).length).toBeGreaterThan(0);
  });

  it('carries an actionable repair command that names the enablement command', () => {
    expectClassifierAvailable();
    writeSettings(validSettings({ schema_version: SCHEMA_VERSION - 1 }));
    const result = runClassifier();
    const report = parseJsonReport(result.output);
    expect(typeof report?.repair_command).toBe('string');
    expect(String(report?.repair_command)).toContain('scripts/telemetry/ensure-local-telemetry.mjs');
  });
});

describe('OT-V2-1: the classifier performs no recovery of its own', () => {
  it('leaves every artefact byte-identical and creates no file', () => {
    expectClassifierAvailable();
    writeSettings(validSettings());
    writeLayerARecord();
    const settingsBefore = readFileSync(workspace.settingsPath, 'utf8');
    const sinkBefore = readFileSync(path.join(workspace.settingsDir, 'telemetry.jsonl'), 'utf8');
    const entriesBefore = readdirSync(workspace.settingsDir).sort();
    const result = runClassifier();
    expect(result.status).toBe(0);
    expect(readFileSync(workspace.settingsPath, 'utf8')).toBe(settingsBefore);
    expect(readFileSync(path.join(workspace.settingsDir, 'telemetry.jsonl'), 'utf8')).toBe(sinkBefore);
    expect(readdirSync(workspace.settingsDir).sort()).toEqual(entriesBefore);
  });

  it('does not run the writer and writes no ledger', () => {
    expectClassifierAvailable();
    writeSettings(validSettings());
    const result = runClassifier();
    expect(result.status).toBe(0);
    expect(existsSync(workspace.logPath)).toBe(false);
  });
});

describe('OT-V2-1: the layer-B measurement state', () => {
  it('reports UNAVAILABLE, never MEASURED, for a ledger the writer wrote with a DISABLED layer', () => {
    expectClassifierAvailable();
    writeSettings(validSettings({ enabled: false }));
    const written = runWriter(['--otel-state', 'DISABLED']);
    expect(written.status).toBe(0);
    expect(existsSync(workspace.logPath)).toBe(true);
    const block = readFileSync(workspace.logPath, 'utf8');
    expect(block).toContain("measurement_state: 'UNAVAILABLE'");
    expect(block).toContain('duration_ms: null');
    const result = runClassifier(['--ledger', workspace.logPath]);
    const report = parseJsonReport(result.output);
    expect(report?.measurement_state).toBe('UNAVAILABLE');
    expect(report?.duration_ms ?? null).toBeNull();
  });

  it('reports UNAVAILABLE, never MEASURED, for a ledger the writer wrote with a NO-DATA layer', () => {
    expectClassifierAvailable();
    writeSettings(validSettings());
    const written = runWriter(['--otel-state', 'NO-DATA']);
    expect(written.status).toBe(0);
    const result = runClassifier(['--ledger', workspace.logPath]);
    const report = parseJsonReport(result.output);
    expect(report?.measurement_state).toBe('UNAVAILABLE');
    expect(report?.duration_ms ?? null).toBeNull();
  });

  it('reports CORRUPT for an unparseable transient dispatch record and the transition is refused', () => {
    expectClassifierAvailable();
    writeSettings(validSettings());
    const recordPath = path.join(workspace.dispatchDir, `${PLAN_ID}__${SLICE_ID}.json`);
    writeFileSync(recordPath, '{"plan_id": "fixture-plan",,}\n', 'utf8');
    const result = runClassifier(['--dispatch-record', recordPath]);
    expect(result.status).toBe(0);
    expect(parseJsonReport(result.output)?.measurement_state).toBe('CORRUPT');
    const written = runWriter();
    expect(written.status).not.toBe(0);
    expect(existsSync(workspace.logPath)).toBe(false);
  });

  it('reports a recovered ledger honestly: UNAVAILABLE, a null duration and no zero counter', () => {
    expectClassifierAvailable();
    writeSettings(validSettings());
    const written = runWriter(['--recover']);
    expect(written.status).toBe(0);
    const result = runClassifier(['--ledger', workspace.logPath]);
    const report = parseJsonReport(result.output);
    expect(report?.measurement_state).toBe('UNAVAILABLE');
    expect(report?.measurement_state).not.toBe('MEASURED');
    expect(report?.duration_ms ?? null).toBeNull();
    const block = readFileSync(workspace.logPath, 'utf8');
    expect(block).toContain('duration_ms: null');
    expect(block).toContain('available: false');
    expect(block).not.toContain('total: 0');
    expect(block).not.toContain('completed_executions: 0');
    expect(block).not.toContain('failed_executions: 0');
  });
});
