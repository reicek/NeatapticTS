#!/usr/bin/env node
/**
 * @module prune-dispatch-records
 * @description The SAFE RESIDUE PRUNER of the transient dispatch-record
 * directory (R-8). It removes a dispatch record ONLY when the record's own
 * bytes violate the persisted-artefact contract, and it never removes a
 * contract-clean record.
 *
 * WHY THE PRUNER EXISTS: the dispatch directory is a POINTER DIRECTORY - one
 * record per dispatched slice - and the DONE writer deletes the record of a
 * slice when it writes the ledger block. A record that no DONE writer will
 * ever claim (a record whose plan id or slice id is a TEST identity, for
 * instance) would otherwise stay in the directory forever and turn the
 * pointer directory into a pile of residue that hides the real pointers.
 *
 * WHY THE PRUNER IS SAFE - THE ONE RULE: a record is removed ONLY when the
 * ONE scanner of the contract module (`scanPersistedText`) reports a
 * violation in its own bytes, that is when the CURRENT writer could never
 * have produced it. Every contract-clean record - including every live
 * pointer of a dispatched slice - is KEPT, no matter how old it is. The
 * pruner therefore never removes a pointer the observer still needs.
 *
 * WHY THE PRUNER NEVER GUESSES: a record whose bytes cannot be read is
 * KEPT, because a record nobody could judge is not a record nobody may
 * remove.
 *
 * THE REPORT IS STDOUT-ONLY: the report names the removed records, and a
 * removed record may carry a forbidden value in its NAME, so the report must
 * never be persisted into a plan or a log.
 *
 * THE COMMAND LINE:
 *
 *   node scripts/telemetry/prune-dispatch-records.mjs --json [--dispatch-dir=<dir>] [--dry-run]
 *
 * @example
 * node scripts/telemetry/prune-dispatch-records.mjs --json --dry-run
 * // => { "inspected": 312, "removed": [], "kept": 306, "dry_run": true }
 */
import { existsSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { scanPersistedText } from './telemetry-contract.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');

/**
 * The repo-relative default of the transient dispatch-record directory.
 *
 * @type {string}
 */
export const DEFAULT_DISPATCH_DIR = 'logs/slice-ledger/dispatch';

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
 * isNonEmptyString('logs/slice-ledger');
 * // => true
 */
function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

/**
 * Parses the command line of the pruner.
 *
 * An UNKNOWN flag is REPORTED instead of ignored, so a mistyped option can
 * never make the pruner do something the operator did not ask for.
 *
 * @param {string[]} argv The arguments after the script path.
 * @returns {{ json: boolean, dispatchDir: string|null, dryRun: boolean, unknown: string[] }} The parsed options.
 * @example
 * parseArgs(['--json', '--dry-run']);
 * // => { json: true, dispatchDir: null, dryRun: true, unknown: [] }
 */
export function parseArgs(argv) {
  const options = { json: false, dispatchDir: null, dryRun: false, unknown: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = String(argv[index]);
    if (argument === '--json') {
      options.json = true;
      continue;
    }
    if (argument === '--dry-run') {
      options.dryRun = true;
      continue;
    }
    if (argument.startsWith('--dispatch-dir=')) {
      options.dispatchDir = argument.slice('--dispatch-dir='.length);
      continue;
    }
    if (argument === '--dispatch-dir') {
      const value = argv[index + 1];
      if (value !== undefined) {
        options.dispatchDir = value;
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
 * Lists the candidate records of a dispatch-record directory.
 *
 * Only regular `*.json` files are candidates: a directory or a temporary
 * sibling is not a record.
 *
 * @param {string} dispatchDir The dispatch-record directory.
 * @returns {string[]} The absolute paths of the candidates.
 * @example
 * listRecords('C:\\repo\\logs\\slice-ledger\\dispatch');
 * // => ['C:\\repo\\logs\\slice-ledger\\dispatch\\p__S-1.json']
 */
export function listRecords(dispatchDir) {
  try {
    if (!existsSync(dispatchDir) || !statSync(dispatchDir).isDirectory()) return [];
    return readdirSync(dispatchDir)
      .filter((name) => name.endsWith('.json'))
      .map((name) => path.join(dispatchDir, name))
      .filter((candidate) => {
        try {
          return statSync(candidate).isFile();
        } catch {
          return false;
        }
      })
      .sort();
  } catch {
    return [];
  }
}

/**
 * Judges ONE record: a record is REMOVABLE only when its own bytes violate
 * the persisted-artefact contract.
 *
 * @param {string} file The record path.
 * @returns {{ removable: boolean, violations: number, readable: boolean }} The verdict.
 * @example
 * judgeRecord('C:\\repo\\logs\\slice-ledger\\dispatch\\p__S-1.json');
 * // => { removable: false, violations: 0, readable: true }
 */
export function judgeRecord(file) {
  let text = '';
  try {
    text = readFileSync(file, 'utf8');
  } catch {
    return { removable: false, violations: 0, readable: false };
  }
  const violations = scanPersistedText(text).length;
  return { removable: violations > 0, violations, readable: true };
}

/**
 * Prunes the contract-violating records of a dispatch-record directory.
 *
 * @param {string} dispatchDir The dispatch-record directory.
 * @param {{ dryRun?: boolean }} [options] The options.
 * @returns {{ inspected: number, removed: string[], kept: number, failed: string[], dry_run: boolean }} The report.
 * @example
 * pruneDispatchRecords('C:\\repo\\logs\\slice-ledger\\dispatch', { dryRun: true });
 * // => { inspected: 312, removed: [], kept: 306, failed: [], dry_run: true }
 */
export function pruneDispatchRecords(dispatchDir, options = {}) {
  const dryRun = options.dryRun === true;
  const records = listRecords(dispatchDir);
  const removed = [];
  const failed = [];
  let kept = 0;
  for (const file of records) {
    const verdict = judgeRecord(file);
    if (!verdict.removable) {
      kept += 1;
      continue;
    }
    if (dryRun) {
      removed.push(file);
      continue;
    }
    try {
      rmSync(file, { force: true });
      removed.push(file);
    } catch {
      failed.push(file);
    }
  }
  return { inspected: records.length, removed, kept, failed, dry_run: dryRun };
}

/**
 * Runs the pruner.
 *
 * @param {string[]} argv The arguments after the script path.
 * @returns {number} The exit code.
 * @example
 * main(['--json', '--dry-run']);
 * // => 0
 */
export function main(argv) {
  const options = parseArgs(argv);
  if (options.unknown.length > 0) {
    process.stderr.write(
      `${JSON.stringify({
        error: 'unknown_flag',
        unknown: options.unknown,
        detail: 'the pruner refuses an unknown flag: it removes NOTHING',
      })}\n`,
    );
    return EXIT_UNKNOWN_FLAG;
  }
  const dispatchDir = path.resolve(
    REPO_ROOT,
    isNonEmptyString(options.dispatchDir) ? options.dispatchDir : DEFAULT_DISPATCH_DIR,
  );
  const report = pruneDispatchRecords(dispatchDir, { dryRun: options.dryRun });
  process.stdout.write(`${JSON.stringify({ ...report, dispatch_dir: dispatchDir })}\n`);
  return report.failed.length === 0 ? 0 : 1;
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedDirectly) {
  process.exitCode = main(process.argv.slice(2));
}

export { REPO_ROOT };
