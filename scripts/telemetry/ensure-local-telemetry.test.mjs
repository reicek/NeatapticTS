/**
 * @module ensure-local-telemetry.test
 * @description RED contract home for the local telemetry enablement command
 * (AC-V2-C).
 *
 * Slice: OT-V2-1 of `plans/opentelemetry-slice-telemetry.plans.md`.
 *
 * This home owns ONE subject:
 *   - `scripts/telemetry/ensure-local-telemetry.mjs` - the enablement command
 *     that creates, upgrades and quarantines the LOCAL settings.
 *
 * The privacy sweep CLI has a home OF ITS OWN - `scripts/telemetry/privacy-
 * sweep.test.mjs` - because a subject whose home belongs to another subject
 * is a subject whose absence is invisible: a missing sweep file could not
 * fail a lane of the enablement home.
 *
 * Every lane drives the REAL command as a REAL child process over REAL
 * temporary artefacts. Nothing is mocked, and the subjects do not exist yet,
 * so every lane is RED BY ASSERTION - never by an import error that would
 * abort the file.
 *
 * THE ENABLEMENT COMMAND:
 *
 *   node scripts/telemetry/ensure-local-telemetry.mjs --json [--check] [--settings-dir=<dir>]
 *
 *   --settings-dir  OPTIONAL. The directory that holds the LOCAL settings
 *                   file. It defaults to `logs/telemetry` under the repository
 *                   root. The settings file inside it is `settings.json`.
 *   --json          The report is printed as ONE JSON object.
 *   --check         The READ-ONLY probe. It REPORTS the state of the settings
 *                   and leaves their bytes BYTE-IDENTICAL.
 *
 * THE COMMAND REFUSES AN UNKNOWN FLAG: a mistyped option is an ERROR, not a
 * silently ignored token, so `--chek` can never run the MUTATING enablement
 * by accident. An unknown flag makes the command exit `2`, report the
 * `unknown_flag` error and change NOTHING.
 *
 * THE LOCAL SETTINGS ARE NOT A TRACKED FILE. They live under `logs/telemetry/`,
 * which `.gitignore` already covers, so the command may create, upgrade and
 * quarantine them freely. THE TRACKED FILE - `.github/hooks/cortex-refresh.json`
 * - IS NEVER REWRITTEN AT RUNTIME: the command REPORTS the absence of the
 * tracked observer registration and prints the exact remediation instead.
 *
 * THE SETTINGS SHAPE THIS HOME DEMANDS:
 *   - `schema_version` is the contract's `SCHEMA_VERSION`;
 *   - the three signals `traces`, `metrics` and `logs` are ALL enabled;
 *   - the exporter points at the LOCAL JSONL sink under the settings
 *     directory;
 *   - no URL, no host and no e-mail address appear anywhere in the settings.
 *
 * THE SETTINGS LIFECYCLE THIS HOME DEMANDS:
 *   - MISSING settings are CREATED;
 *   - a STALE `schema_version` is UPGRADED;
 *   - a CORRUPT settings file is QUARANTINED to a `.bak` sibling - whose bytes
 *     are the ORIGINAL corrupt bytes - and the settings are RECREATED;
 *   - the command is IDEMPOTENT: a second run changes no byte and reports no
 *     changed path;
 *   - the command PRINTS the exact changed paths;
 *   - the `--check` probe REPORTS the state of the settings and leaves their
 *     bytes BYTE-IDENTICAL.
 *
 * THE LOCAL SETTINGS CARRY NO ABSOLUTE PATH: the exporter path is RELATIVE
 * to the settings directory, so the settings file never carries a machine
 * path and never carries a forbidden value.
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

/** The repo-relative path of the enablement command this home owns. */
const ENSURE_RELPATH = 'scripts/telemetry/ensure-local-telemetry.mjs';
/** The absolute path of the enablement command this home owns. */
const ENSURE_PATH = path.join(REPO_ROOT, 'scripts', 'telemetry', 'ensure-local-telemetry.mjs');
/** The tracked manifest the ensure command may REPORT but never rewrite. */
const MANIFEST_RELPATH = '.github/hooks/cortex-refresh.json';
/** The absolute path of the tracked manifest. */
const MANIFEST_PATH = path.join(REPO_ROOT, '.github', 'hooks', 'cortex-refresh.json');
/** The file name of the local settings file inside the settings directory. */
const SETTINGS_FILE_NAME = 'settings.json';

/** The temporary workspace of the running lane. */
let workspace;

/**
 * Creates the temporary workspace: a settings directory and a sink directory.
 *
 * @returns {{ root: string, settingsDir: string, settingsPath: string }} The workspace.
 */
function createWorkspace() {
  const root = mkdtempSync(path.join(tmpdir(), 'ensure-local-telemetry-'));
  const settingsDir = path.join(root, 'logs', 'telemetry');
  mkdirSync(settingsDir, { recursive: true });
  return {
    root,
    settingsDir,
    settingsPath: path.join(settingsDir, SETTINGS_FILE_NAME),
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
    exporter: { type: 'jsonl', path: '.' },
    ...overrides,
  };
}

/**
 * Runs the REAL enablement command as a REAL child process.
 *
 * @param {string[]} [extra] The extra command-line arguments.
 * @returns {{ status: number|null, stdout: string, stderr: string, output: string, error?: Error }} The run result.
 */
function runEnsure(extra = []) {
  const result = spawnSync(
    process.execPath,
    [ENSURE_PATH, '--json', '--settings-dir', workspace.settingsDir, ...extra],
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
 * Collects every value of the named keys in a parsed settings object.
 *
 * @param {unknown} node The node to walk.
 * @param {string} key The key to collect.
 * @returns {unknown[]} The values.
 */
function collectValues(node, key) {
  if (node === null || typeof node !== 'object') return [];
  if (Array.isArray(node)) return node.flatMap((entry) => collectValues(entry, key));
  const found = [];
  for (const [name, value] of Object.entries(node)) {
    if (name === key) found.push(value);
    found.push(...collectValues(value, key));
  }
  return found;
}

/**
 * Collects every string value in a parsed settings object.
 *
 * @param {unknown} node The node to walk.
 * @returns {string[]} The strings.
 */
function collectStrings(node) {
  if (typeof node === 'string') return [node];
  if (node === null || typeof node !== 'object') return [];
  if (Array.isArray(node)) return node.flatMap((entry) => collectStrings(entry));
  return Object.values(node).flatMap((value) => collectStrings(value));
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

describe('OT-V2-1: the local telemetry enablement command', () => {
  it('exists at the path the contract declares', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
  });

  it('creates the local settings when they are missing', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    rmSync(workspace.settingsPath, { force: true });
    const result = runEnsure();
    expect(result.status).toBe(0);
    expect(existsSync(workspace.settingsPath)).toBe(true);
    expect(() => JSON.parse(readFileSync(workspace.settingsPath, 'utf8'))).not.toThrow();
  });

  it('enables traces, metrics and logs', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    rmSync(workspace.settingsPath, { force: true });
    runEnsure();
    const settings = JSON.parse(readFileSync(workspace.settingsPath, 'utf8'));
    for (const signal of ['traces', 'metrics', 'logs']) {
      const values = collectValues(settings, signal);
      expect(values.length).toBeGreaterThan(0);
      expect(values.every((value) => value === true)).toBe(true);
    }
  });

  it('points the exporter at the local JSONL sink RELATIVE to the settings directory', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    rmSync(workspace.settingsPath, { force: true });
    runEnsure();
    const settings = JSON.parse(readFileSync(workspace.settingsPath, 'utf8'));
    expect(settings.exporter?.type).toBe('jsonl');
    expect(settings.exporter?.path).toBe('.');
  });

  it('carries NO absolute machine path into the local settings', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    rmSync(workspace.settingsPath, { force: true });
    runEnsure();
    const settings = JSON.parse(readFileSync(workspace.settingsPath, 'utf8'));
    const absolute = collectStrings(settings).filter((value) => /^[A-Za-z]:[\\/]/u.test(value));
    expect(absolute).toEqual([]);
  });

  it('puts no URL and no host into the local settings', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    rmSync(workspace.settingsPath, { force: true });
    runEnsure();
    const text = readFileSync(workspace.settingsPath, 'utf8');
    expect(text).not.toMatch(/https?:\/\//i);
    expect(text).not.toMatch(/wss?:\/\//i);
    expect(text).not.toMatch(/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/);
  });

  it('upgrades a stale schema version', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    writeSettings(validSettings({ schema_version: SCHEMA_VERSION - 1 }));
    const result = runEnsure();
    expect(result.status).toBe(0);
    const settings = JSON.parse(readFileSync(workspace.settingsPath, 'utf8'));
    expect(settings.schema_version).toBe(SCHEMA_VERSION);
  });

  it('quarantines a corrupt settings file to a .bak sibling and recreates it', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    const corrupt = '{"schema_version": 1, "signals": {"traces": true,,}\n';
    writeSettings(corrupt);
    const result = runEnsure();
    expect(result.status).toBe(0);
    const siblings = readdirSync(workspace.settingsDir);
    const backup = siblings.find((name) => name.includes('.bak'));
    expect(backup).toBeDefined();
    expect(() => JSON.parse(readFileSync(workspace.settingsPath, 'utf8'))).not.toThrow();
  });

  it('preserves the original corrupt bytes in the .bak sibling', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    const corrupt = '{"schema_version": 1, "signals": {"traces": true,,}\n';
    writeSettings(corrupt);
    runEnsure();
    const backup = readdirSync(workspace.settingsDir).find((name) => name.includes('.bak'));
    expect(readFileSync(path.join(workspace.settingsDir, backup), 'utf8')).toBe(corrupt);
  });

  it('is idempotent: a second run changes no byte and reports no changed path', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    rmSync(workspace.settingsPath, { force: true });
    runEnsure();
    const after = readFileSync(workspace.settingsPath, 'utf8');
    const result = runEnsure();
    expect(result.status).toBe(0);
    expect(readFileSync(workspace.settingsPath, 'utf8')).toBe(after);
    const report = parseJsonReport(result.output);
    expect(report?.changed ?? []).toEqual([]);
  });

  it('prints the exact changed paths', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    rmSync(workspace.settingsPath, { force: true });
    const result = runEnsure();
    const report = parseJsonReport(result.output);
    const changed = report?.changed ?? [];
    expect(Array.isArray(changed)).toBe(true);
    expect(changed.some((entry) => String(entry).endsWith(SETTINGS_FILE_NAME))).toBe(true);
  });

  it('reports the tracked hook registration and never rewrites the tracked manifest', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    const before = readFileSync(MANIFEST_PATH, 'utf8');
    const result = runEnsure();
    expect(result.status).toBe(0);
    expect(readFileSync(MANIFEST_PATH, 'utf8')).toBe(before);
    expect(result.output).toContain(MANIFEST_RELPATH);
  });

  it('never contacts the network', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    rmSync(workspace.settingsPath, { force: true });
    const result = runEnsure();
    expect(result.output).not.toMatch(/https?:\/\//i);
  });

  it('refuses an unknown flag: it exits 2, reports the unknown_flag error and changes nothing', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    rmSync(workspace.settingsPath, { force: true });
    const result = runEnsure(['--bogus-flag']);
    const report = parseJsonReport(result.output);
    expect({
      status: result.status,
      error: report === null ? null : report.error ?? null,
      unknown: report === null ? null : report.unknown ?? null,
      settingsCreated: existsSync(workspace.settingsPath),
    }).toEqual({
      status: 2,
      error: 'unknown_flag',
      unknown: ['--bogus-flag'],
      settingsCreated: false,
    });
  });

  it('reports the registered state through the --check probe and leaves the settings byte-identical', () => {
    expectSubjectAvailable(ENSURE_PATH, ENSURE_RELPATH);
    // The settings are stored in a byte layout the enablement never writes,
    // so a probe that REWROTE the file would change these bytes. The byte
    // check is the ONLY witness of such a probe: the mutating enablement
    // leaves the canonical CONTENT alone and never rewrites the layout.
    writeSettings(JSON.stringify(validSettings()));
    const before = readFileSync(workspace.settingsPath);
    const result = runEnsure(['--check']);
    const report = parseJsonReport(result.output);
    expect({
      status: result.status,
      checked: report === null ? null : report.checked ?? null,
      state: report === null ? null : report.state ?? null,
      canonical: report === null ? null : report.canonical ?? null,
      changed: report === null ? null : report.changed ?? null,
      quarantined: report === null ? null : report.quarantined ?? null,
      enabled: report === null || report.settings === null ? null : report.settings.enabled ?? null,
      hookRegistered:
        report === null || report.tracked_hook_registration === null
          ? null
          : report.tracked_hook_registration.registered ?? null,
      bytesIdentical: readFileSync(workspace.settingsPath).equals(before),
    }).toEqual({
      status: 0,
      checked: true,
      state: 'OK',
      canonical: true,
      changed: [],
      quarantined: [],
      enabled: true,
      hookRegistered: true,
      bytesIdentical: true,
    });
  });
});

describe('OT-V2-1: the privacy sweep CLI', () => {
  it('lives in a home of its own', () => {
    expect(
      existsSync(path.join(REPO_ROOT, 'scripts', 'telemetry', 'privacy-sweep.test.mjs')),
    ).toBe(true);
  });
});
