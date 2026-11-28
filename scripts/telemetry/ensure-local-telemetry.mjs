#!/usr/bin/env node
/**
 * @module ensure-local-telemetry
 * @description The LOCAL ENABLEMENT COMMAND of the vertical slice B
 * (AC-V2-F). It creates, upgrades and quarantines the LOCAL telemetry
 * settings under `logs/telemetry` and it NEVER rewrites a tracked file.
 *
 * WHAT THE COMMAND OWNS: the local settings file `logs/telemetry/settings.json`.
 * The file is NOT tracked - `logs/` is already ignored - so the command may
 * create it, upgrade it and quarantine it freely.
 *
 * WHAT THE COMMAND NEVER DOES: it NEVER rewrites the tracked hook manifest
 * `.github/hooks/cortex-refresh.json`. A manifest change is not live in the
 * session that wrote it, because the CLI loads the repository hooks at
 * startup, so the command REPORTS the registration state and prints the exact
 * remediation instead of touching the tracked bytes.
 *
 * THE LIFECYCLE: MISSING settings are CREATED, a STALE `schema_version` is
 * UPGRADED, a CORRUPT settings file is QUARANTINED to a `.bak` sibling that
 * holds the ORIGINAL corrupt bytes and is then RECREATED, and the command is
 * IDEMPOTENT: a second run changes no byte and reports no changed path.
 *
 * THE SETTINGS GOVERN THE REAL SINKS: the `exporter.path` of the settings is
 * the sink root RELATIVE TO THE SETTINGS DIRECTORY (the canonical value is
 * `.`, that is the settings directory itself), so the settings never carry an
 * absolute machine path and the sinks of the local layer are the ones the
 * settings name. The `enabled` flag of the settings is what the exporters
 * read, so `enabled: false` really stops the emission.
 *
 * THE COMMAND REFUSES AN UNKNOWN FLAG: a mistyped option is an ERROR, not a
 * silently ignored token, so `--chek` can never run the MUTATING enablement
 * by accident. An unknown flag makes the command exit `2` and change
 * NOTHING.
 *
 * THE COMMAND NEVER CONTACTS THE NETWORK: it opens no socket, it resolves no
 * host and it writes no URL into the settings.
 *
 * THE COMMAND LINE:
 *
 *   node scripts/telemetry/ensure-local-telemetry.mjs --json [--check] [--enable|--disable] [--settings-dir=<dir>]
 *
 * @example
 * node scripts/telemetry/ensure-local-telemetry.mjs --json
 * // => { "changed": ["...\\logs\\telemetry\\settings.json"], "quarantined": [], ... }
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SCHEMA_VERSION, OBSERVER_RELPATH, inspectObserverRegistration } from './telemetry-contract.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');

/** The repo-relative directory that holds the LOCAL settings. */
const DEFAULT_SETTINGS_DIR = path.join('logs', 'telemetry');
/** The file name of the local settings file inside the settings directory. */
const SETTINGS_FILE_NAME = 'settings.json';
/** The repo-relative path of the TRACKED hook manifest the command reports but never rewrites. */
const MANIFEST_RELPATH = '.github/hooks/cortex-refresh.json';
/**
 * The sink root of the local settings, RELATIVE TO THE SETTINGS DIRECTORY.
 *
 * The canonical value is `.` - the sink root IS the settings directory, so
 * the three JSONL sinks sit next to the settings that name them. The value
 * is RELATIVE on purpose: an absolute machine path in a persisted artefact
 * is a privacy violation, and the settings are a persisted artefact.
 *
 * @type {string}
 */
const SINK_ROOT_RELPATH = '.';

/**
 * The exit code of a REFUSED command line: an unknown flag.
 *
 * @type {number}
 */
export const EXIT_UNKNOWN_FLAG = 2;

/**
 * Tells whether a value is a non-empty string.
 *
 * @param {unknown} value The value to test.
 * @returns {boolean} True for a non-empty string.
 * @example
 * isNonEmptyString('logs/telemetry');
 * // => true
 */
function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

/**
 * Parses the command line of the enablement command.
 *
 * Both spellings of the directory flag are accepted: `--settings-dir <dir>`
 * and `--settings-dir=<dir>`. An UNKNOWN flag is REPORTED instead of
 * ignored, so a mistyped option can never run the MUTATING enablement by
 * accident.
 *
 * @param {string[]} argv The arguments after the script path.
 * @returns {{ json: boolean, settingsDir: string|null, check: boolean, enabled: boolean, unknown: string[] }} The parsed options.
 * @example
 * parseArgs(['--json', '--settings-dir', 'logs/telemetry']);
 * // => { json: true, settingsDir: 'logs/telemetry', check: false, enabled: true, unknown: [] }
 */
export function parseArgs(argv) {
  const options = { json: false, settingsDir: null, check: false, enabled: true, unknown: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = String(argv[index]);
    if (argument === '--json') {
      options.json = true;
      continue;
    }
    if (argument === '--check') {
      options.check = true;
      continue;
    }
    if (argument === '--enable') {
      options.enabled = true;
      continue;
    }
    if (argument === '--disable') {
      options.enabled = false;
      continue;
    }
    if (argument.startsWith('--settings-dir=')) {
      options.settingsDir = argument.slice('--settings-dir='.length);
      continue;
    }
    if (argument === '--settings-dir') {
      const value = argv[index + 1];
      if (value !== undefined) {
        options.settingsDir = value;
        index += 1;
        continue;
      }
      options.unknown.push(argument);
      continue;
    }
    options.unknown.push(argument);
  }
  return options;
}

/**
 * Builds the CANONICAL local settings of a settings directory.
 *
 * The settings carry the current schema version, the enabled flag, the three
 * signals and the local JSONL exporter. No URL, no host, no e-mail address
 * and no ABSOLUTE machine path ever reaches the file: the sink root is the
 * RELATIVE `.`, that is the settings directory itself.
 *
 * @param {boolean} [enabled] The enabled flag of the settings.
 * @returns {Record<string, unknown>} The canonical settings.
 * @example
 * buildSettings();
 * // => { schema_version: 1, enabled: true, signals: { ... }, exporter: { ... } }
 */
export function buildSettings(enabled = true) {
  return {
    schema_version: SCHEMA_VERSION,
    enabled: enabled === true,
    signals: { traces: true, metrics: true, logs: true },
    exporter: { type: 'jsonl', path: SINK_ROOT_RELPATH },
  };
}

/**
 * Renders the settings as the exact bytes the command writes.
 *
 * @param {Record<string, unknown>} settings The settings.
 * @returns {string} The rendered text.
 * @example
 * renderSettings(buildSettings('d'));
 * // => '{\n  "schema_version": 1,\n  ...\n}\n'
 */
export function renderSettings(settings) {
  return `${JSON.stringify(settings, null, 2)}\n`;
}

/**
 * Reads the local settings and tells the three states apart.
 *
 * @param {string} settingsPath The settings path.
 * @returns {{ status: 'OK' | 'MISSING' | 'CORRUPT', value: unknown }} The read result.
 * @example
 * readSettings('C:\\missing\\settings.json');
 * // => { status: 'MISSING', value: null }
 */
export function readSettings(settingsPath) {
  if (!existsSync(settingsPath)) return { status: 'MISSING', value: null };
  let text = '';
  try {
    text = readFileSync(settingsPath, 'utf8');
  } catch {
    return { status: 'CORRUPT', value: null };
  }
  try {
    const parsed = JSON.parse(text);
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { status: 'CORRUPT', value: null };
    }
    return { status: 'OK', value: parsed };
  } catch {
    return { status: 'CORRUPT', value: null };
  }
}

/**
 * Tells whether the current settings already ARE the canonical settings.
 *
 * @param {unknown} current The current settings.
 * @param {Record<string, unknown>} desired The canonical settings.
 * @returns {boolean} True when nothing has to change.
 * @example
 * isCanonical(buildSettings('d'), buildSettings('d'));
 * // => true
 */
export function isCanonical(current, desired) {
  if (current === null || typeof current !== 'object') return false;
  if (current.schema_version !== desired.schema_version) return false;
  if (current.enabled !== desired.enabled) return false;
  const signals = current.signals;
  if (signals === null || typeof signals !== 'object' || Array.isArray(signals)) return false;
  for (const signal of Object.keys(desired.signals)) {
    if (signals[signal] !== true) return false;
  }
  const exporter = current.exporter;
  if (exporter === null || typeof exporter !== 'object' || Array.isArray(exporter)) return false;
  if (exporter.type !== desired.exporter.type) return false;
  if (exporter.path !== desired.exporter.path) return false;
  return true;
}

/**
 * Finds the free `.bak` sibling of the settings file.
 *
 * The quarantine never overwrites an earlier quarantine, so the evidence of
 * every corrupt file is kept.
 *
 * @param {string} settingsPath The settings path.
 * @returns {string} The quarantine path.
 * @example
 * findQuarantinePath('C:\\logs\\telemetry\\settings.json');
 * // => 'C:\\logs\\telemetry\\settings.json.bak'
 */
export function findQuarantinePath(settingsPath) {
  const first = `${settingsPath}.bak`;
  if (!existsSync(first)) return first;
  for (let index = 1; index < 1000; index += 1) {
    const candidate = `${settingsPath}.bak.${index}`;
    if (!existsSync(candidate)) return candidate;
  }
  return first;
}

/**
 * Ensures the local settings exist and are canonical.
 *
 * @param {string} settingsDir The absolute settings directory.
 * @param {{ enabled?: boolean }} [options] The options; `enabled` is the flag the
 *   canonical settings carry.
 * @returns {{ changed: string[], quarantined: string[], settingsPath: string, settings: Record<string, unknown>, state: string }} The outcome.
 * @example
 * ensureSettings('C:\\repo\\logs\\telemetry');
 * // => { changed: ['...\\settings.json'], quarantined: [], ... }
 */
export function ensureSettings(settingsDir, options = {}) {
  const settingsPath = path.join(settingsDir, SETTINGS_FILE_NAME);
  const desired = buildSettings(options.enabled !== false);
  const changed = [];
  const quarantined = [];
  const current = readSettings(settingsPath);
  let state = current.status;
  if (current.status === 'CORRUPT') {
    const quarantinePath = findQuarantinePath(settingsPath);
    copyFileSync(settingsPath, quarantinePath);
    quarantined.push(quarantinePath);
    state = 'QUARANTINED';
  }
  if (!existsSync(settingsPath) || !isCanonical(current.value, desired)) {
    mkdirSync(settingsDir, { recursive: true });
    writeFileSync(settingsPath, renderSettings(desired), 'utf8');
    changed.push(settingsPath);
    if (state !== 'QUARANTINED') state = current.status === 'OK' ? 'UPGRADED' : 'CREATED';
  }
  return { changed, quarantined, settingsPath, settings: desired, state };
}

/**
 * Inspects the TRACKED hook manifest and reports the observer registration.
 *
 * The manifest is READ, never written. The verdict comes from the ONE
 * registration checker of the contract module, so the command and the
 * observer's own home cannot disagree about what "registered" means.
 *
 * @returns {{ manifest: string, observer: string, registered: boolean, complete: boolean, registered_under: string[], missing: string[], remediation: string }} The report.
 * @example
 * inspectManifest();
 * // => { manifest: '.github/hooks/cortex-refresh.json', registered: true, ... }
 */
export function inspectManifest() {
  const manifestPath = path.join(REPO_ROOT, ...MANIFEST_RELPATH.split('/'));
  let verdict = null;
  try {
    verdict = inspectObserverRegistration(JSON.parse(readFileSync(manifestPath, 'utf8')));
  } catch {
    verdict = null;
  }
  const registered = verdict !== null && verdict.registered;
  const complete = verdict !== null && verdict.complete;
  return {
    manifest: MANIFEST_RELPATH,
    observer: OBSERVER_RELPATH,
    registered,
    complete,
    registered_under: verdict === null ? [] : verdict.registered_under,
    missing: verdict === null ? ['PostToolUse', 'PostToolUseFailure'] : verdict.missing,
    remediation: registered
      ? `the observer is registered in the tracked manifest ${MANIFEST_RELPATH}; the CLI loads the repository hooks at startup, so a fresh session in the repository is what makes the registration live`
      : `the observer is NOT registered in the tracked manifest ${MANIFEST_RELPATH}: register it under PostToolUse AND PostToolUseFailure with the SAME command and with NO matcher, then restart the CLI in the repository - the CLI loads the repository hooks at startup, so a registration change is not live in the session that wrote it`,
  };
}

/**
 * Runs the enablement command.
 *
 * @param {string[]} argv The arguments after the script path.
 * @returns {number} The exit code.
 * @example
 * main(['--json']);
 * // => 0
 */
export function main(argv) {
  const options = parseArgs(argv);
  if (options.unknown.length > 0) {
    process.stderr.write(
      `${JSON.stringify({
        error: 'unknown_flag',
        unknown: options.unknown,
        detail: 'the enablement command refuses an unknown flag: it changes NOTHING',
      })}\n`,
    );
    return EXIT_UNKNOWN_FLAG;
  }
  const settingsDir = path.resolve(
    REPO_ROOT,
    isNonEmptyString(options.settingsDir) ? options.settingsDir : DEFAULT_SETTINGS_DIR,
  );
  const manifest = inspectManifest();
  if (options.check) {
    const current = readSettings(path.join(settingsDir, SETTINGS_FILE_NAME));
    const canonical = isCanonical(current.value, buildSettings(options.enabled));
    const report = {
      checked: true,
      changed: [],
      quarantined: [],
      state: current.status,
      canonical,
      settings_path: path.join(settingsDir, SETTINGS_FILE_NAME),
      settings: current.value,
      tracked_hook_registration: manifest,
    };
    process.stdout.write(`${JSON.stringify(report)}\n`);
    return canonical ? 0 : 1;
  }
  mkdirSync(settingsDir, { recursive: true });
  const outcome = ensureSettings(settingsDir, { enabled: options.enabled });
  const report = {
    checked: false,
    changed: outcome.changed,
    quarantined: outcome.quarantined,
    state: outcome.state,
    settings: {
      path: outcome.settingsPath,
      schema_version: outcome.settings.schema_version,
      enabled: outcome.settings.enabled,
      signals: outcome.settings.signals,
      exporter: outcome.settings.exporter,
    },
    tracked_hook_registration: manifest,
  };
  process.stdout.write(`${JSON.stringify(report)}\n`);
  return 0;
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedDirectly) {
  process.exitCode = main(process.argv.slice(2));
}

export { DEFAULT_SETTINGS_DIR, SETTINGS_FILE_NAME, MANIFEST_RELPATH, OBSERVER_RELPATH, SINK_ROOT_RELPATH, REPO_ROOT };
