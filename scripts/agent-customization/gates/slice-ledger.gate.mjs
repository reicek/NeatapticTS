#!/usr/bin/env node
/**
 * @module slice-ledger.gate
 * @description The fail-closed DONE-CLOSURE GATE of the vertical slice B
 * (AC-V2-E). It scans a tracker for the slices that carry `[DONE]` and
 * demands, for every one of them that is IN SCOPE, exactly ONE valid ledger
 * block in the companion log.
 *
 * WHAT MAKES THE GATE FAIL-CLOSED: a DONE slice with no ledger, with a
 * malformed ledger, with a duplicated ledger, with a ledger of an unknown
 * schema version, with a ledger that carries a forbidden field, with an
 * available counter that claims no source, no measures string or no scope,
 * with an unavailable counter that was written as a zero, or with a
 * `measurement_state: CORRUPT` is a HARD FAILURE and the DONE transition is
 * refused.
 *
 * WHAT IS NEVER AN EXEMPTION: the layer-A reporting states (`ENABLED`,
 * `DISABLED`, `STALE`, `NO-DATA`) are NOT exemptions. The OpenTelemetry
 * layer is optional and fail-open; the repo-owned measurement is not. Only
 * the layer-B `CORRUPT` state blocks the transition, and a RECOVERED ledger
 * (`UNAVAILABLE`, a null duration and null counters) is HONEST and accepted.
 *
 * THE FORWARD-ONLY RULE (R-7): the gate binds only FROM the activation
 * marker, so a slice that closed BEFORE the writer and the gate landed is
 * never refused and is never back-filled. The marker is the tracker's own
 * `ledger_activation_marker` when it declares one and the built-in marker of
 * the slice that lands the gate otherwise; a slice that is ordered BEFORE
 * the marker is out of scope. The measured pre-existing set can also be
 * given on the command line.
 *
 * THE GATE NEVER REWRITES ANYTHING: not the tracker, not the companion log.
 * It reads and it reports.
 *
 * THE COMMAND LINE:
 *
 *   node scripts/agent-customization/gates/slice-ledger.gate.mjs \
 *     --json --plan=<planPath> \
 *     [--logs=<logsPath>] [--pre-existing-done=<csv slice ids>]
 *
 * @example
 * node scripts/agent-customization/gates/slice-ledger.gate.mjs --json \
 *   --plan plans/completed/opentelemetry-slice-telemetry.plans.md
 * // => { "pass": true, ... }
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { validateLedgerBlock } from '../../telemetry/telemetry-contract.mjs';
import { extractLedgerBlocks, isDone, readTrackerSlices } from '../../telemetry/ledger-state.mjs';
import { companionLogPath, planStem } from '../../telemetry/write-slice-ledger.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..', '..');

/** The repo-relative path of the writer the refusal names. */
const WRITER_RELPATH = 'scripts/telemetry/write-slice-ledger.mjs';
/** The built-in activation marker: the slice that lands the writer and the gate. */
const BUILT_IN_ACTIVATION_MARKER = 'OT-V2-2';
/** The key a tracker declares its own activation marker under. */
const ACTIVATION_MARKER_KEY = 'ledger_activation_marker';
/** The machine-stable reason a report carries when the gate checked NOTHING. */
const VACUITY_REASON = 'no-done-slice-in-scope';

/**
 * Tells whether a value is a non-empty string.
 *
 * @param {unknown} value The value to test.
 * @returns {boolean} True for a non-empty string.
 * @example
 * isNonEmptyString('scope');
 * // => true
 */
function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

/**
 * Parses the command line of the gate.
 *
 * Both spellings of every flag are accepted: `--flag value` and
 * `--flag=value`.
 *
 * @param {string[]} argv The arguments after the script path.
 * @returns {{ plan: string|null, logs: string|null, preExistingDone: string[], json: boolean }} The parsed options.
 * @example
 * parseArgs(['--json', '--plan', 'plans/x.plans.md']);
 * // => { plan: 'plans/x.plans.md', logs: null, preExistingDone: [], json: true }
 */
export function parseArgs(argv) {
  const options = { plan: null, logs: null, preExistingDone: [], json: false };
  const valueFlags = new Map([
    ['--plan', 'plan'],
    ['--logs', 'logs'],
    ['--pre-existing-done', 'preExistingDone'],
  ]);
  for (let index = 0; index < argv.length; index += 1) {
    const argument = String(argv[index]);
    if (argument === '--json') {
      options.json = true;
      continue;
    }
    const separator = argument.indexOf('=');
    if (separator > 0) {
      const name = argument.slice(0, separator);
      const key = valueFlags.get(name);
      if (key === undefined) continue;
      const value = argument.slice(separator + 1);
      options[key] = key === 'preExistingDone' ? splitSliceIds(value) : value;
      continue;
    }
    const key = valueFlags.get(argument);
    if (key === undefined) continue;
    const value = argv[index + 1];
    if (value === undefined) continue;
    options[key] = key === 'preExistingDone' ? splitSliceIds(value) : value;
    index += 1;
  }
  return options;
}

/**
 * Splits a comma-separated list of slice ids.
 *
 * @param {string} value The raw list.
 * @returns {string[]} The slice ids.
 * @example
 * splitSliceIds('OT-V1-1, OT-V1-2');
 * // => ['OT-V1-1', 'OT-V1-2']
 */
export function splitSliceIds(value) {
  return String(value ?? '')
    .split(',')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

/**
 * Measures the pre-existing DONE set of a tracker.
 *
 * The set is the union of the MEASURED set given on the command line and
 * the slices that the tracker orders BEFORE the activation marker. A tracker
 * that declares no marker and a marker that the tracker does not carry
 * leave the set to the command line alone - the gate never guesses a
 * boundary it cannot measure.
 *
 * @param {{ slices: Array<{ slice_id: string, status: string, index: number }>, activation_marker: string|null }} tracker The tracker.
 * @param {string[]} measured The measured pre-existing slice ids.
 * @returns {{ preExisting: string[], marker: string|null, markerIndex: number }} The measured set.
 * @example
 * measurePreExisting({ slices: [{ slice_id: 'OT-V2-2', status: '[DONE]', index: 0 }], activation_marker: null }, []);
 * // => { preExisting: [], marker: 'OT-V2-2', markerIndex: -1 }
 */
export function measurePreExisting(tracker, measured) {
  const marker = isNonEmptyString(tracker.activation_marker)
    ? tracker.activation_marker.trim()
    : BUILT_IN_ACTIVATION_MARKER;
  const markerIndex = tracker.slices.findIndex((entry) => entry.slice_id === marker);
  const preExisting = new Set(measured);
  if (markerIndex > 0) {
    for (const entry of tracker.slices.slice(0, markerIndex)) preExisting.add(entry.slice_id);
  }
  return { preExisting: [...preExisting], marker, markerIndex };
}

/**
 * Checks the layer-B rules the shared validator does not cover.
 *
 * The shared validator of the contract module checks the required keys, the
 * schema version, the forbidden keys, the two state enums and the
 * availability of every counter. It does NOT check the `scope` of an
 * available counter, and it accepts `measurement_state: CORRUPT` as a legal
 * enum - the two rules the gate adds.
 *
 * @param {Record<string, unknown>} block The ledger block.
 * @returns {string[]} The reasons the block is refused, or an empty list.
 * @example
 * checkGateOnlyRules(block);
 * // => []
 */
export function checkGateOnlyRules(block) {
  const reasons = [];
  const usage = block.usage !== null && typeof block.usage === 'object' ? block.usage : {};
  const toolExecutions =
    usage.tool_executions !== null && typeof usage.tool_executions === 'object'
      ? usage.tool_executions
      : null;
  if (toolExecutions !== null && toolExecutions.available === true) {
    if (!isNonEmptyString(toolExecutions.scope)) {
      reasons.push('an available `usage.tool_executions` counter carries no `scope`');
    }
  }
  if (block.measurement_state === 'CORRUPT') {
    reasons.push('the ledger block records a corrupt measurement');
  }
  return reasons;
}

/**
 * Checks one DONE slice against the ledger blocks of the companion log.
 *
 * @param {{ sliceId: string, planId: string, blocks: Array<{ block: Record<string, unknown>|null, error: string|null }> }} request The request.
 * @returns {{ failures: Array<{ slice_id: string, kind: string, detail: string }> }} The failures.
 * @example
 * checkSlice({ sliceId: 'OT-V2-2', planId: 'p', blocks: [] });
 * // => { failures: [{ slice_id: 'OT-V2-2', kind: 'missing-ledger', detail: '...' }] }
 */
export function checkSlice(request) {
  const { sliceId, planId, blocks } = request;
  const failures = [];
  const mine = blocks.filter(
    (entry) =>
      entry.block !== null &&
      entry.block.slice_id === sliceId &&
      (entry.block.plan_id === undefined || entry.block.plan_id === planId),
  );
  if (mine.length === 0) {
    failures.push({
      slice_id: sliceId,
      kind: 'missing-ledger',
      detail: `no ledger block for ${planId}/${sliceId} is in the companion log`,
    });
    return { failures };
  }
  if (mine.length > 1) {
    failures.push({
      slice_id: sliceId,
      kind: 'duplicate-ledger',
      detail: `the companion log carries ${mine.length} ledger blocks for ${planId}/${sliceId}; exactly one is demanded`,
    });
    return { failures };
  }
  const block = mine[0].block;
  const verdict = validateLedgerBlock(block);
  if (!verdict.valid) {
    failures.push({
      slice_id: sliceId,
      kind: 'invalid-ledger',
      detail: verdict.errors.join('; '),
    });
    return { failures };
  }
  const reasons = checkGateOnlyRules(block);
  if (reasons.length > 0) {
    failures.push({
      slice_id: sliceId,
      kind: 'invalid-ledger',
      detail: reasons.join('; '),
    });
  }
  return { failures };
}

/**
 * Runs the gate.
 *
 * @param {string[]} argv The arguments after the script path.
 * @returns {number} The exit code.
 * @example
 * main(['--json', '--plan', 'plans/x.plans.md']);
 * // => 0
 */
export function main(argv) {
  const options = parseArgs(argv);
  if (!isNonEmptyString(options.plan)) {
    process.stderr.write('the flag --plan is required\n');
    return 2;
  }
  const planPath = path.resolve(options.plan);
  if (!existsSync(planPath)) {
    process.stderr.write(`the plan file does not exist: ${planPath}\n`);
    return 2;
  }
  const planId = planStem(planPath);
  const logPath = isNonEmptyString(options.logs)
    ? path.resolve(options.logs)
    : companionLogPath(planPath);
  const planText = readFileSync(planPath, 'utf8');
  const tracker = readTrackerSlices(planText);
  const { preExisting, marker } = measurePreExisting(tracker, options.preExistingDone);
  const preExistingSet = new Set(preExisting);
  const logText = existsSync(logPath) ? readFileSync(logPath, 'utf8') : '';
  const blocks = extractLedgerBlocks(logText);

  const failures = [];
  for (const entry of blocks) {
    if (entry.block === null) {
      failures.push({
        slice_id: null,
        kind: 'unattributable-ledger',
        detail: `a ledger block of the companion log carries no parsable slice id: ${entry.error ?? 'the block does not parse'}`,
      });
      continue;
    }
    if (!isNonEmptyString(entry.block.slice_id)) {
      failures.push({
        slice_id: null,
        kind: 'unattributable-ledger',
        detail: 'a ledger block of the companion log declares no slice id and is attributed to no slice',
      });
    }
  }

  const inScope = [];
  for (const entry of tracker.slices) {
    if (!isDone(entry)) continue;
    if (preExistingSet.has(entry.slice_id)) continue;
    inScope.push(entry.slice_id);
    failures.push(...checkSlice({ sliceId: entry.slice_id, planId, blocks }).failures);
  }

  // WHY THE VACUITY IS REPORTED: a gate that checked NOTHING must not
  // present itself as a gate that VERIFIED something. The count of checked
  // slices is reported together with an explicit verdict, so a green report
  // that proves nothing can never be mistaken for a green report that
  // proved the DONE transition.
  const vacuous = inScope.length === 0;

  const report = {
    pass: failures.length === 0,
    plan: planPath,
    logs: logPath,
    plan_id: planId,
    activation_marker: marker,
    in_scope: inScope,
    pre_existing_done: preExisting,
    failures,
    checked: inScope.length,
    vacuous,
    verdict: vacuous ? 'VACUOUS' : 'VERIFIED',
    vacuity_reason: vacuous ? VACUITY_REASON : null,
  };
  process.stdout.write(`${JSON.stringify(report)}\n`);
  if (failures.length > 0) {
    for (const failure of failures) {
      const slice = failure.slice_id ?? 'an unattributable ledger block';
      process.stderr.write(
        `refusing the DONE transition of ${slice}: ${failure.kind} - ${failure.detail}\n` +
          `the ledger belongs in ${path.basename(logPath)} and is written by: node ${WRITER_RELPATH} --plan ${planPath} --slice ${failure.slice_id ?? '<slice-id>'} --plan-id ${planId}\n` +
          'the rule: a disabled OpenTelemetry layer does not exempt a slice from its ledger; the ledger is the repo-owned measurement and layer A is optional.\n',
      );
    }
  }
  return failures.length === 0 ? 0 : 1;
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
/* istanbul ignore next -- the CLI entry path writes the report to standard output and is proved by the child-process lanes of the home, not from inside the process. */
if (invokedDirectly) {
  process.exitCode = main(process.argv.slice(2));
}

export { BUILT_IN_ACTIVATION_MARKER, ACTIVATION_MARKER_KEY, REPO_ROOT };
