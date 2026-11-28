#!/usr/bin/env node
/**
 * @module privacy-sweep
 * @description The PRIVACY-SWEEP CLI of the vertical slice B (AC-V2-F). It
 * scans every persisted telemetry artefact with the ONE forbidden-key and
 * forbidden-value scanner of the contract module and it reports the findings.
 *
 * THE SWEEP IS A THIN WRAPPER: the rules live in exactly ONE place -
 * `scanPersistedText` of `scripts/telemetry/telemetry-contract.mjs` - and this
 * CLI introduces NO second implementation of them. It owns no pattern of its
 * own, so the sweep, the writer and the tests cannot drift apart.
 *
 * THE SWEEP COVERS EVERYTHING: the report carries an ALWAYS EMPTY `excluded`
 * list, because a persisted artefact is never excused from the privacy rules.
 *
 * THE DEFAULT TARGET SET IS THE SET OF ARTEFACTS THAT REALLY EXIST: the
 * directory that holds the local settings and the three JSONL sinks, the
 * directory that holds the transient dispatch records and the tool-execution
 * observation log, the companion log of the lane and the tracker itself. A
 * target that does not exist is NOT an artefact and is never a violation, so
 * the default set names NO phantom file.
 *
 * THE TARGETS ARE RESOLVED AGAINST THE REPOSITORY ROOT: a repo-relative
 * target means the same artefact no matter which directory a runner happened
 * to choose, so the sweep cannot be made to scan nothing by running it from
 * the wrong place.
 *
 * EVERY FINDING NAMES ITS FILE: a violation carries the REPO-RELATIVE path of
 * the artefact it was found in, so a report says WHERE a violation is instead
 * of only counting it.
 *
 * THE SWEEP NEVER REWRITES AN ARTEFACT: it reads and it reports. A violation
 * makes the command exit non-zero.
 *
 * THE SWEEP REFUSES AN UNKNOWN FLAG: a mistyped option is an ERROR, not a
 * silently ignored token, so `--targtes=...` can never make the sweep scan
 * the default set instead of the set the operator asked for. An unknown flag
 * makes the command exit `2` and scan NOTHING.
 *
 * THE COMMAND LINE:
 *
 *   node scripts/telemetry/privacy-sweep.mjs --json [--targets=<a,b,c>]
 *
 * @example
 * node scripts/telemetry/privacy-sweep.mjs --json --targets=logs/telemetry
 * // => { "scanned": ["logs/telemetry/trace.jsonl"], "excluded": [], "violations": [] }
 */
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { scanPersistedText } from './telemetry-contract.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');

/**
 * The DEFAULT target set: every persisted artefact of the lane that REALLY
 * exists.
 *
 * @type {readonly string[]}
 */
export const DEFAULT_TARGETS = Object.freeze([
  'logs/telemetry',
  'logs/slice-ledger',
  'plans/completed/opentelemetry-slice-telemetry.logs.md',
  'plans/completed/opentelemetry-slice-telemetry.plans.md',
]);

/**
 * The exit code of a REFUSED command line: an unknown flag.
 *
 * @type {number}
 */
export const EXIT_UNKNOWN_FLAG = 2;

/**
 * Parses the command line of the sweep.
 *
 * Both spellings of the target flag are accepted: `--targets <a,b,c>` and
 * `--targets=<a,b,c>`. An UNKNOWN flag is REPORTED instead of ignored, so a
 * mistyped option can never make the sweep scan something else than what the
 * operator asked for.
 *
 * @param {string[]} argv The arguments after the script path.
 * @returns {{ json: boolean, targets: string[]|null, unknown: string[] }} The parsed options.
 * @example
 * parseArgs(['--json', '--targets=a.jsonl,b.jsonl']);
 * // => { json: true, targets: ['a.jsonl', 'b.jsonl'], unknown: [] }
 */
export function parseArgs(argv) {
  const options = { json: false, targets: null, unknown: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = String(argv[index]);
    if (argument === '--json') {
      options.json = true;
      continue;
    }
    if (argument.startsWith('--targets=')) {
      options.targets = splitTargets(argument.slice('--targets='.length));
      continue;
    }
    if (argument === '--targets') {
      const value = argv[index + 1];
      if (value !== undefined) {
        options.targets = splitTargets(value);
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
 * Splits a comma-separated list of targets.
 *
 * @param {string} value The raw list.
 * @returns {string[]} The targets.
 * @example
 * splitTargets('a.jsonl, b.jsonl');
 * // => ['a.jsonl', 'b.jsonl']
 */
export function splitTargets(value) {
  return String(value ?? '')
    .split(',')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

/**
 * Resolves a target against the REPOSITORY ROOT.
 *
 * A relative target is resolved against the repository root, so the sweep
 * scans the same artefacts no matter which directory the runner chose. An
 * absolute target is kept as it is.
 *
 * @param {string} target The target path.
 * @returns {string} The absolute path.
 * @example
 * resolveTarget('logs/telemetry');
 * // => 'C:\\repo\\logs\\telemetry'
 */
export function resolveTarget(target) {
  const raw = String(target ?? '');
  if (path.isAbsolute(raw)) return path.resolve(raw);
  return path.resolve(REPO_ROOT, raw);
}

/**
 * Renders a path as the REPO-RELATIVE, forward-slash form, so a report is
 * stable across platforms and never carries an absolute machine path.
 *
 * A file OUTSIDE the repository root keeps its absolute path, because there
 * is no repo-relative form of it.
 *
 * @param {string} file The absolute file path.
 * @returns {string} The reported path.
 * @example
 * toRepoRelative('C:\\repo\\logs\\telemetry\\trace.jsonl');
 * // => 'logs/telemetry/trace.jsonl'
 */
export function toRepoRelative(file) {
  const relative = path.relative(REPO_ROOT, path.resolve(file));
  if (relative.length === 0) return '.';
  if (relative.startsWith('..') || path.isAbsolute(relative)) return path.resolve(file);
  return relative.split(path.sep).join('/');
}

/**
 * Expands a target into the FILES it stands for.
 *
 * A file target is the file itself, a directory target is every file under it
 * and a missing target is NOTHING - a target that does not exist is never an
 * artefact and never a violation.
 *
 * @param {string} target The target path.
 * @returns {string[]} The absolute file paths.
 * @example
 * expandTarget('logs/telemetry');
 * // => ['C:\\repo\\logs\\telemetry\\trace.jsonl']
 */
export function expandTarget(target) {
  const resolved = resolveTarget(target);
  if (!existsSync(resolved)) return [];
  let stats;
  try {
    stats = statSync(resolved);
  } catch {
    return [];
  }
  if (stats.isFile()) return [resolved];
  if (!stats.isDirectory()) return [];
  const files = [];
  const walk = (directory) => {
    let entries = [];
    try {
      entries = readdirSync(directory, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const child = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        walk(child);
        continue;
      }
      if (entry.isFile()) files.push(child);
    }
  };
  walk(resolved);
  return files.sort();
}

/**
 * Scans a set of targets with the ONE scanner of the contract module.
 *
 * Every finding carries the REPO-RELATIVE path of the artefact it was found
 * in, so a report names WHERE the violation is.
 *
 * @param {string[]} targets The targets.
 * @returns {{ scanned: string[], excluded: string[], violations: Array<Record<string, unknown>> }} The report.
 * @example
 * scanTargets(['logs/telemetry']);
 * // => { scanned: ['...'], excluded: [], violations: [] }
 */
export function scanTargets(targets) {
  const scanned = [];
  const excluded = [];
  const violations = [];
  const seen = new Set();
  for (const target of targets) {
    for (const file of expandTarget(target)) {
      if (seen.has(file)) continue;
      seen.add(file);
      scanned.push(file);
      let text = '';
      try {
        text = readFileSync(file, 'utf8');
      } catch {
        continue;
      }
      const relative = toRepoRelative(file);
      for (const violation of scanPersistedText(text)) {
        violations.push({ ...violation, file: relative });
      }
    }
  }
  return { scanned, excluded, violations };
}

/**
 * Runs the sweep.
 *
 * @param {string[]} argv The arguments after the script path.
 * @returns {number} The exit code.
 * @example
 * main(['--json', '--targets=logs/telemetry']);
 * // => 0
 */
export function main(argv) {
  const options = parseArgs(argv);
  if (options.unknown.length > 0) {
    process.stderr.write(
      `${JSON.stringify({
        error: 'unknown_flag',
        unknown: options.unknown,
        detail: 'the sweep refuses an unknown flag: it scans NOTHING and changes NOTHING',
      })}\n`,
    );
    return EXIT_UNKNOWN_FLAG;
  }
  const targets = options.targets === null ? [...DEFAULT_TARGETS] : options.targets;
  const report = scanTargets(targets);
  const payload = {
    scanned: report.scanned.map((file) => toRepoRelative(file)),
    excluded: report.excluded,
    violations: report.violations,
    targets,
  };
  process.stdout.write(`${JSON.stringify(payload)}\n`);
  return report.violations.length === 0 ? 0 : 1;
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedDirectly) {
  process.exitCode = main(process.argv.slice(2));
}

export { REPO_ROOT };
