/**
 * @module slice-ledger.gate.test
 * @description RED contract home for the DONE-closure gate of the vertical slice B.
 *
 * Slice: OT-V2-1 of `plans/opentelemetry-slice-telemetry.plans.md`.
 * Acceptance criteria pinned by this home:
 *   - AC-V2-B (the DONE gate home), which carries forward AC-007-A, AC-007-B,
 *     AC-006-S4-A, AC-011-C and AC-007-C (the disabled-layer rule).
 *
 * THE SUBJECT UNDER TEST IS `scripts/agent-customization/gates/slice-ledger.gate.mjs`,
 * THE DONE-CLOSURE GATE. Every lane drives the REAL gate as a REAL child
 * process over a REAL temporary tracker and a REAL temporary companion log.
 * Nothing is mocked: not the gate, not the tracker, not the companion log, and
 * not the shared contract module, which the gate is expected to reuse.
 *
 * WHY THE GATE IS DRIVEN AS A CHILD PROCESS: the gate is a CLI, its contract
 * IS its command line, its exit code and its report, and a CLI that is driven
 * through an import cannot prove its refusal. The gate does not exist yet, so
 * every lane is RED BY ASSERTION: the lanes fail on the exit code, on the
 * report or on the missing block - never on an import error that would abort
 * the file.
 *
 * THE COMMAND LINE THIS HOME DEMANDS:
 *
 *   node scripts/agent-customization/gates/slice-ledger.gate.mjs \
 *     --json --plan=<planPath> \
 *     [--logs=<logsPath>] [--pre-existing-done=<csv slice ids>]
 *
 *   --plan               REQUIRED. The tracker file. The companion log is its
 *                        sibling `<name>.logs.md`.
 *   --logs               OPTIONAL. The companion log path; it defaults to the
 *                        sibling of the tracker.
 *   --pre-existing-done  OPTIONAL. The MEASURED set of slice ids that already
 *                        carried `[DONE]` before the ledger contract was
 *                        activated (R-7, the forward-only rule). A slice in
 *                        that set is NEVER demanded a ledger and is NEVER
 *                        backfilled; when the flag is absent the gate
 *                        measures the set from the tracker's own activation
 *                        marker.
 *   --json               The report is printed as ONE JSON object carrying a
 *                        boolean `pass`.
 *
 * THE EXIT CONTRACT: the gate exits 0 when every DONE slice in scope holds
 * exactly one valid ledger block, and NON-ZERO when any of them does not.
 * A refusal never rewrites the tracker (R-6: no script ever rewrites the
 * tracker).
 *
 * THE LEDGER CONTRACT (the block the gate demands):
 *   - exactly ONE ledger block per DONE slice in scope, keyed by `plan_id`
 *     plus `slice_id`;
 *   - the block is VALID against the shared validator of
 *     `scripts/telemetry/telemetry-contract.mjs`, so the gate and the writer
 *     share ONE notion of a valid ledger;
 *   - a MISSING, MALFORMED, DUPLICATED, SCHEMA-UNKNOWN or FORBIDDEN-FIELD
 *     ledger is a HARD FAILURE: the DONE transition is refused;
 *   - a counter that claims `available: true` without a `source`, without a
 *     `measures` string or without a `scope` is refused;
 *   - an UNAVAILABLE counter written as `0` is refused (an unavailable
 *     counter is null, never zero);
 *   - the layer-A reporting states (ENABLED, DISABLED, STALE, NO-DATA) are
 *     NEVER an exemption from the ledger, so only the layer-B CORRUPT state
 *     blocks the transition;
 *   - a RECOVERED ledger (`measurement_state: UNAVAILABLE`, a null duration and
 *     null counters) is HONEST and is accepted.
 *
 * THE ACTIONABLE REFUSAL (the message the gate must carry): it names the
 * slice, the companion log it expects, the EXACT writer command that produces
 * the missing block and the rule that a disabled OpenTelemetry layer does not
 * exempt the slice.
 */

import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { emitLedgerBlock, SCHEMA_VERSION } from '../../telemetry/telemetry-contract.mjs';
import {
  ACTIVATION_MARKER_KEY,
  BUILT_IN_ACTIVATION_MARKER,
  checkGateOnlyRules,
  checkSlice,
  main,
  measurePreExisting,
  parseArgs,
  splitSliceIds,
} from './slice-ledger.gate.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..', '..');

/** The repo-relative path of the gate this home owns. */
const GATE_RELPATH = 'scripts/agent-customization/gates/slice-ledger.gate.mjs';
/** The absolute path of the gate this home owns. */
const GATE_PATH = path.join(REPO_ROOT, 'scripts', 'agent-customization', 'gates', 'slice-ledger.gate.mjs');
/** The repo-relative path of the DONE writer the refusal must name. */
const WRITER_RELPATH = 'scripts/telemetry/write-slice-ledger.mjs';

const PLAN_ID = 'fixture-plan';
const SLICE_ID = 'OT-V2-9';
const PRE_EXISTING_SLICE_ID = 'OT-OLD-1';

/** The temporary workspace of the running lane. */
let workspace;

/**
 * The valid ledger block every positive lane starts from.
 *
 * @param {Record<string, unknown>} [overrides] The block overrides.
 * @returns {Record<string, unknown>} The ledger block.
 */
function validBlock(overrides = {}) {
  return {
    schema_version: SCHEMA_VERSION,
    plan_id: PLAN_ID,
    slice_id: SLICE_ID,
    status: '[DONE]',
    measurement_state: 'MEASURED',
    otel_state: 'DISABLED',
    duration_ms: 250,
    runtime: {
      node_version: 'v22.15.0',
      heap_used_bytes: 12345,
      rss_bytes: 67890,
    },
    usage: {
      token_usage: {
        available: true,
        source: 'cortex-get-slice-context',
        measures: 'context tokens DELIVERED to the slice, not model consumption',
        context_tokens_delivered: 1200,
        prompt_tokens: null,
        completion_tokens: null,
        total_tokens: null,
        unavailable_reason: null,
      },
      tool_executions: {
        available: true,
        source: 'copilot-cli-posthook-observation-window',
        measures: 'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
        scope: 'slice-dispatch-window',
        total: 3,
        completed_executions: 2,
        failed_executions: 1,
        sessions_observed: 1,
        breakdown: { read: 1, edit: 1, search: 1, exec: 0, other: 0 },
      },
    },
    validation: {
      total: 1,
      passed: 1,
      failed: 0,
      skipped: 0,
      results: [{ name: 'unit', status: 'passed', duration_ms: 12 }],
    },
    ...overrides,
  };
}

/**
 * Extracts the body of an emitted ledger block.
 *
 * @param {Record<string, unknown>} block The ledger block.
 * @returns {string} The body.
 */
function blockBody(block) {
  const fenced = emitLedgerBlock(block);
  const match = /```yaml\n([\s\S]*?)\n```/.exec(fenced);
  if (match === null) throw new Error('the emitter produced no fenced block');
  return match[1];
}

/**
 * Wraps a body back into the fenced block the companion log carries.
 *
 * @param {string} body The body.
 * @returns {string} The fenced block.
 */
function fenced(body) {
  return `\`\`\`yaml\n${body}\n\`\`\`\n`;
}

/**
 * Removes a child key from a nested mapping of an emitted body.
 *
 * @param {string} body The emitted body.
 * @param {string} parentKey The parent mapping key.
 * @param {string} childKey The child key to remove.
 * @returns {string} The mutated body.
 */
function removeChildKey(body, parentKey, childKey) {
  const lines = body.split('\n');
  const parentIndex = lines.findIndex((line) => line.trim() === `${parentKey}:`);
  if (parentIndex < 0) throw new Error(`the emitted body carries no \`${parentKey}\` mapping`);
  const parentIndent = lines[parentIndex].length - lines[parentIndex].trimStart().length;
  for (let index = parentIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    const indent = line.length - line.trimStart().length;
    if (indent <= parentIndent) break;
    if (line.trim().startsWith(`${childKey}:`)) {
      lines.splice(index, 1);
      return lines.join('\n');
    }
  }
  throw new Error(`the emitted body carries no \`${childKey}\` under \`${parentKey}\``);
}

/**
 * Inserts a line right below the ledger root key.
 *
 * @param {string} body The emitted body.
 * @param {string} line The line to insert.
 * @returns {string} The mutated body.
 */
function insertUnderRoot(body, line) {
  const lines = body.split('\n');
  lines.splice(1, 0, line);
  return lines.join('\n');
}

/**
 * Creates the temporary workspace: a tracker and its companion log.
 *
 * @returns {{ root: string, planPath: string, logPath: string }} The workspace.
 */
function createWorkspace() {
  const root = mkdtempSync(path.join(tmpdir(), 'slice-ledger-gate-'));
  const plansDir = path.join(root, 'plans');
  mkdirSync(plansDir, { recursive: true });
  const planPath = path.join(plansDir, `${PLAN_ID}.plans.md`);
  return {
    root,
    planPath,
    logPath: path.join(plansDir, `${PLAN_ID}.logs.md`),
  };
}

/**
 * Writes the fixture tracker with the given DONE slices.
 *
 * @param {string[]} sliceIds The DONE slice ids.
 */
function writePlan(sliceIds) {
  const entries = sliceIds.map(
    (sliceId) => `  - slice_id: '${sliceId}'\n    title: 'the fixture slice'\n    status: '[DONE]'`,
  );
  writeFileSync(
    workspace.planPath,
    ['# Fixture tracker', '', '```yaml', 'slices:', ...entries, '```', ''].join('\n'),
    'utf8',
  );
}

/**
 * Writes the fixture tracker with CRLF line endings, the shape a tracker
 * written on Windows carries.
 *
 * @param {string[]} sliceIds The DONE slice ids.
 */
function writeCrlfPlan(sliceIds) {
  const entries = sliceIds.map(
    (sliceId) => `  - slice_id: '${sliceId}'\r\n    title: 'the fixture slice'\r\n    status: '[DONE]'`,
  );
  writeFileSync(
    workspace.planPath,
    ['# Fixture tracker', '', '```yaml', 'slices:', ...entries, '```', ''].join('\r\n'),
    'utf8',
  );
}

/**
 * Appends fenced ledger blocks to the companion log.
 *
 * @param {string[]} bodies The block bodies.
 */
function writeLedgerBodies(bodies) {
  const text = bodies.map((body) => fenced(body)).join('\n');
  writeFileSync(workspace.logPath, text, 'utf8');
}

/**
 * Runs the REAL gate as a REAL child process.
 *
 * @param {string[]} [extra] The extra command-line arguments.
 * @returns {{ status: number|null, stdout: string, stderr: string, output: string, error?: Error }} The run result.
 */
function runGate(extra = []) {
  const result = spawnSync(
    process.execPath,
    [GATE_PATH, '--json', '--plan', workspace.planPath, '--logs', workspace.logPath, ...extra],
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
 * Parses the JSON report the gate prints.
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
 * Asserts that the gate is on disk, so a lane that fails afterwards fails on
 * the GATE's behaviour rather than on a missing file.
 */
function expectGateAvailable() {
  expect({ gate: GATE_RELPATH, exists: existsSync(GATE_PATH) }).toEqual({
    gate: GATE_RELPATH,
    exists: true,
  });
}

beforeEach(() => {
  workspace = createWorkspace();
});

afterEach(() => {
  rmSync(workspace.root, { recursive: true, force: true });
});

describe('OT-V2-1: the DONE-closure gate', () => {
  it('exists at the path the contract declares', () => {
    expectGateAvailable();
  });

  it('refuses a DONE slice that has NO ledger', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const result = runGate();
    expect(result.status).not.toBe(0);
    expect(parseJsonReport(result.output)?.pass).toBe(false);
  });

  it('names the slice that has no ledger', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const result = runGate();
    expect(result.output).toContain(SLICE_ID);
  });

  it('names the companion log it expects the ledger in', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const result = runGate();
    expect(result.output).toContain(path.basename(workspace.logPath));
  });

  it('names the EXACT writer command that produces the missing ledger', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const result = runGate();
    expect(result.output).toContain(WRITER_RELPATH);
  });

  it('states that a disabled OpenTelemetry layer does not exempt the slice', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const result = runGate();
    expect(result.output).toMatch(/opentelemetry/i);
    expect(result.output).toMatch(/does not exempt/i);
  });

  it('refuses a MALFORMED ledger', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    writeLedgerBodies(['telemetry_ledger:\n  schema_version: 1\n  plan_id: \'fixture-plan\'']);
    const result = runGate();
    expect(result.status).not.toBe(0);
  });

  it('refuses a DUPLICATED ledger', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const body = blockBody(validBlock());
    writeLedgerBodies([body, body]);
    const result = runGate();
    expect(result.status).not.toBe(0);
  });

  it('refuses a ledger whose schema version this tracker does not know', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    writeLedgerBodies([blockBody(validBlock({ schema_version: 999 }))]);
    const result = runGate();
    expect(result.status).not.toBe(0);
  });

  it('refuses a ledger that carries a forbidden field', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const body = insertUnderRoot(blockBody(validBlock()), "  timestamp: '1764147453321'");
    writeLedgerBodies([body]);
    const result = runGate();
    expect(result.status).not.toBe(0);
  });

  it('refuses an available counter that carries no source', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const body = removeChildKey(blockBody(validBlock()), 'tool_executions', 'source');
    writeLedgerBodies([body]);
    const result = runGate();
    expect(result.status).not.toBe(0);
  });

  it('refuses an available counter that carries no measures string', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const body = removeChildKey(blockBody(validBlock()), 'tool_executions', 'measures');
    writeLedgerBodies([body]);
    const result = runGate();
    expect(result.status).not.toBe(0);
  });

  it('refuses an available counter that carries no scope', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const body = removeChildKey(blockBody(validBlock()), 'tool_executions', 'scope');
    writeLedgerBodies([body]);
    const result = runGate();
    expect(result.status).not.toBe(0);
  });

  it('refuses an unavailable counter that was written as zero', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const block = validBlock();
    block.usage.tool_executions = {
      available: false,
      source: 'copilot-cli-posthook-observation-window',
      measures: 'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
      scope: 'slice-dispatch-window',
      total: 0,
      completed_executions: 0,
      failed_executions: 0,
      sessions_observed: 0,
      breakdown: null,
      unavailable_reason: 'observer_not_loaded',
    };
    writeLedgerBodies([blockBody(block)]);
    const result = runGate();
    expect(result.status).not.toBe(0);
  });

  it('never demands a ledger from a slice that was DONE before the activation marker', () => {
    expectGateAvailable();
    writePlan([PRE_EXISTING_SLICE_ID, SLICE_ID]);
    writeLedgerBodies([blockBody(validBlock())]);
    const result = runGate([`--pre-existing-done=${PRE_EXISTING_SLICE_ID}`]);
    expect(result.status).toBe(0);
    expect(parseJsonReport(result.output)?.pass).toBe(true);
  });

  it('accepts a valid ledger and never rewrites the tracker', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    writeLedgerBodies([blockBody(validBlock())]);
    const before = readFileSync(workspace.planPath, 'utf8');
    const result = runGate();
    expect(result.status).toBe(0);
    expect(parseJsonReport(result.output)?.pass).toBe(true);
    expect(readFileSync(workspace.planPath, 'utf8')).toBe(before);
  });

  it('reads a CRLF tracker: a DONE slice with no ledger is refused', () => {
    expectGateAvailable();
    writeCrlfPlan([SLICE_ID]);
    writeLedgerBodies([]);
    const result = runGate();
    const report = parseJsonReport(result.output);
    // The slice is IN SCOPE: a tracker whose fences are CRLF-terminated is a
    // real tracker, and a tracker that is read as EMPTY would make the gate
    // pass vacuously.
    expect(report?.in_scope ?? null).toContain(SLICE_ID);
    expect(result.status).not.toBe(0);
  });

  it('never rewrites the COMPANION LOG it reads', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    writeLedgerBodies([blockBody(validBlock())]);
    const before = readFileSync(workspace.logPath, 'utf8');
    const result = runGate();
    expect(result.status).toBe(0);
    expect(readFileSync(workspace.logPath, 'utf8')).toBe(before);
  });

  it('reports the pre-existing DONE set it exempted', () => {
    expectGateAvailable();
    writePlan([PRE_EXISTING_SLICE_ID, SLICE_ID]);
    writeLedgerBodies([blockBody(validBlock())]);
    const result = runGate([`--pre-existing-done=${PRE_EXISTING_SLICE_ID}`]);
    const report = parseJsonReport(result.output);
    expect(report?.pre_existing_done ?? null).toEqual([PRE_EXISTING_SLICE_ID]);
    expect(report?.checked ?? null).toBe(1);
  });
});

describe('OT-V2-1: the layer-A reporting states are never an exemption', () => {
  for (const otelState of ['ENABLED', 'DISABLED', 'STALE', 'NO-DATA']) {
    it(`accepts a valid ledger whose otel_state is ${otelState}`, () => {
      expectGateAvailable();
      writePlan([SLICE_ID]);
      writeLedgerBodies([blockBody(validBlock({ otel_state: otelState }))]);
      const result = runGate();
      expect(result.status).toBe(0);
    });
  }
});

describe('OT-V2-1: the layer-B measurement state', () => {
  it('refuses a ledger whose measurement state is CORRUPT', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    writeLedgerBodies([blockBody(validBlock({ measurement_state: 'CORRUPT' }))]);
    const result = runGate();
    expect(result.status).not.toBe(0);
  });

  it('accepts a RECOVERED ledger: UNAVAILABLE, a null duration and null counters', () => {
    expectGateAvailable();
    writePlan([SLICE_ID]);
    const block = validBlock({ measurement_state: 'UNAVAILABLE', duration_ms: null });
    block.usage.token_usage = {
      available: false,
      source: 'host-model-accounting',
      measures: 'the host model token consumption',
      context_tokens_delivered: null,
      prompt_tokens: null,
      completion_tokens: null,
      total_tokens: null,
      unavailable_reason: 'the dispatch record is missing',
    };
    block.usage.tool_executions = {
      available: false,
      source: 'copilot-cli-posthook-observation-window',
      measures: 'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
      scope: 'slice-dispatch-window',
      total: null,
      completed_executions: null,
      failed_executions: null,
      sessions_observed: null,
      breakdown: null,
      unavailable_reason: 'observer_not_loaded',
    };
    writeLedgerBodies([blockBody(block)]);
    const result = runGate();
    expect(result.status).toBe(0);
  });
});

/**
 * THE IN-PROCESS HOME (OT-V2-3).
 *
 * WHY IT EXISTS NEXT TO THE CHILD-PROCESS LANES: the child-process lanes
 * prove the CLI CONTRACT of the gate - its exit code, its report and its
 * refusal. These lanes prove the FUNCTIONS the report is built from, so the
 * gate is measured by the coverage of its own source instead of only by its
 * process. The report is captured by replacing the two stream writers, so
 * nothing is printed and no lane depends on the order of the output.
 */
describe('OT-V2-3: the DONE gate IN PROCESS', () => {
  /** The captured standard output of the running lane. */
  let stdoutSpy;
  /** The captured standard error of the running lane. */
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
   * Runs the gate in process and captures its report.
   *
   * @param {string[]} argv The command line.
   * @returns {{ status: number, report: Record<string, unknown>|null, stderr: string }} The run result.
   */
  function runInProcess(argv) {
    const status = main(argv);
    const text = stdoutSpy.mock.calls.map((call) => String(call[0])).join('');
    const stderr = stderrSpy.mock.calls.map((call) => String(call[0])).join('');
    return {
      status,
      report: text.trim().length > 0 ? JSON.parse(text) : null,
      stderr,
    };
  }

  /**
   * Writes a tracker whose slices carry the given statuses.
   *
   * @param {Array<{ slice_id: string, status: string }>} entries The tracker entries.
   */
  function writeTrackerWithStatuses(entries) {
    const body = entries
      .map((entry) => `  - slice_id: '${entry.slice_id}'\n    status: '${entry.status}'`)
      .join('\n');
    writeFileSync(
      workspace.planPath,
      ['# Fixture tracker', '', '```yaml', 'slices:', body, '```', ''].join('\n'),
      'utf8',
    );
  }

  it('parses both spellings of every flag', () => {
    expect(parseArgs(['--json', '--plan', 'plans/x.plans.md'])).toEqual({
      plan: 'plans/x.plans.md',
      logs: null,
      preExistingDone: [],
      json: true,
    });
    expect(
      parseArgs([
        '--plan=plans/y.plans.md',
        '--logs=logs/y.logs.md',
        '--pre-existing-done=OT-V1-1, OT-V1-2',
      ]),
    ).toEqual({
      plan: 'plans/y.plans.md',
      logs: 'logs/y.logs.md',
      preExistingDone: ['OT-V1-1', 'OT-V1-2'],
      json: false,
    });
  });

  it('ignores an unknown flag and a flag that carries no value', () => {
    expect(parseArgs(['--unknown', 'value', '--plan'])).toEqual({
      plan: null,
      logs: null,
      preExistingDone: [],
      json: false,
    });
    expect(parseArgs(['--unknown=value', '--plan=plans/x.plans.md']).plan).toBe('plans/x.plans.md');
    expect(parseArgs(['--plan=']).plan).toBe('');
  });

  it('splits a comma-separated list of slice ids', () => {
    expect(splitSliceIds('OT-V1-1, OT-V1-2')).toEqual(['OT-V1-1', 'OT-V1-2']);
    expect(splitSliceIds('')).toEqual([]);
    expect(splitSliceIds(' , ')).toEqual([]);
    expect(splitSliceIds(undefined)).toEqual([]);
  });

  it('measures the pre-existing set from the activation marker of the tracker', () => {
    const tracker = {
      slices: [
        { slice_id: 'OT-V2-2', status: '[DONE]', index: 0 },
        { slice_id: 'OT-V2-3', status: '[DONE]', index: 1 },
      ],
      activation_marker: 'OT-V2-3',
    };
    expect(measurePreExisting(tracker, [])).toEqual({
      preExisting: ['OT-V2-2'],
      marker: 'OT-V2-3',
      markerIndex: 1,
    });
  });

  it('falls back to the built-in activation marker and to the measured set', () => {
    const tracker = {
      slices: [{ slice_id: 'OT-V2-2', status: '[DONE]', index: 0 }],
      activation_marker: null,
    };
    expect(measurePreExisting(tracker, ['OT-V1-9'])).toEqual({
      preExisting: ['OT-V1-9'],
      marker: BUILT_IN_ACTIVATION_MARKER,
      markerIndex: 0,
    });
    expect(measurePreExisting({ ...tracker, activation_marker: '   ' }, []).marker).toBe(
      BUILT_IN_ACTIVATION_MARKER,
    );
    expect(ACTIVATION_MARKER_KEY).toBe('ledger_activation_marker');
  });

  it('checks the rules the shared validator does not cover', () => {
    expect(checkGateOnlyRules(validBlock())).toEqual([]);
    expect(checkGateOnlyRules({ ...validBlock(), measurement_state: 'CORRUPT' })).toEqual([
      'the ledger block records a corrupt measurement',
    ]);
    const withoutScope = validBlock();
    delete withoutScope.usage.tool_executions.scope;
    expect(checkGateOnlyRules(withoutScope)).toEqual([
      'an available `usage.tool_executions` counter carries no `scope`',
    ]);
    expect(checkGateOnlyRules({ usage: null })).toEqual([]);
    expect(checkGateOnlyRules({ usage: { tool_executions: 'not an object' } })).toEqual([]);
  });

  it('accepts exactly one valid ledger for the slice', () => {
    const block = validBlock();
    expect(checkSlice({ sliceId: SLICE_ID, planId: PLAN_ID, blocks: [{ block, error: null }] })).toEqual({
      failures: [],
    });
  });

  it('refuses a missing, a duplicated and a malformed ledger', () => {
    expect(checkSlice({ sliceId: SLICE_ID, planId: PLAN_ID, blocks: [] }).failures[0].kind).toBe(
      'missing-ledger',
    );
    const block = validBlock();
    expect(
      checkSlice({
        sliceId: SLICE_ID,
        planId: PLAN_ID,
        blocks: [
          { block, error: null },
          { block, error: null },
        ],
      }).failures[0].kind,
    ).toBe('duplicate-ledger');
    expect(
      checkSlice({
        sliceId: SLICE_ID,
        planId: PLAN_ID,
        blocks: [{ block: { slice_id: SLICE_ID }, error: null }],
      }).failures[0].kind,
    ).toBe('invalid-ledger');
  });

  it('ignores a ledger that belongs to another slice or another plan', () => {
    const other = validBlock({ slice_id: 'OT-V2-8' });
    const otherPlan = validBlock({ plan_id: 'another-plan' });
    expect(
      checkSlice({
        sliceId: SLICE_ID,
        planId: PLAN_ID,
        blocks: [
          { block: null, error: 'unparsable' },
          { block: other, error: null },
          { block: otherPlan, error: null },
        ],
      }).failures[0].kind,
    ).toBe('missing-ledger');
  });

  it('refuses a ledger that breaks a gate-only rule', () => {
    const block = validBlock({ measurement_state: 'CORRUPT' });
    const failures = checkSlice({
      sliceId: SLICE_ID,
      planId: PLAN_ID,
      blocks: [{ block, error: null }],
    }).failures;
    expect(failures[0].kind).toBe('invalid-ledger');
    expect(failures[0].detail).toContain('corrupt measurement');
  });

  it('demands the flag --plan', () => {
    const result = runInProcess([]);
    expect(result.status).toBe(2);
    expect(result.stderr).toContain('the flag --plan is required');
  });

  it('refuses a plan file that does not exist', () => {
    const result = runInProcess(['--plan', path.join(workspace.root, 'absent.plans.md')]);
    expect(result.status).toBe(2);
    expect(result.stderr).toContain('does not exist');
  });

  it('VERIFIES a DONE slice that holds exactly one valid ledger', () => {
    writePlan([SLICE_ID]);
    writeLedgerBodies([blockBody(validBlock())]);
    const result = runInProcess(['--plan', workspace.planPath, '--logs', workspace.logPath]);
    expect(result.status).toBe(0);
    expect(result.report.pass).toBe(true);
    expect(result.report.checked).toBe(1);
    expect(result.report.in_scope).toEqual([SLICE_ID]);
    expect(result.report.vacuous).toBe(false);
    expect(result.report.verdict).toBe('VERIFIED');
    expect(result.report.vacuity_reason).toBeNull();
  });

  it('THE ANTI-VACUITY LANE: a report that checked NOTHING is VACUOUS, never VERIFIED', () => {
    writeTrackerWithStatuses([{ slice_id: SLICE_ID, status: '[OPEN]' }]);
    writeLedgerBodies([]);
    const result = runInProcess(['--plan', workspace.planPath, '--logs', workspace.logPath]);
    expect(result.status).toBe(0);
    expect(result.report.pass).toBe(true);
    expect(result.report.checked).toBe(0);
    expect(result.report.in_scope).toEqual([]);
    expect(result.report.vacuous).toBe(true);
    expect(result.report.verdict).toBe('VACUOUS');
    expect(result.report.vacuity_reason).toBe('no-done-slice-in-scope');
  });

  it('refuses a DONE slice that holds no ledger at all', () => {
    writePlan([SLICE_ID]);
    writeLedgerBodies([]);
    const result = runInProcess(['--plan', workspace.planPath, '--logs', workspace.logPath]);
    expect(result.status).toBe(1);
    expect(result.report.pass).toBe(false);
    expect(result.report.failures[0].kind).toBe('missing-ledger');
    expect(result.report.vacuous).toBe(false);
    expect(result.report.verdict).toBe('VERIFIED');
  });

  it('refuses a companion log that carries an unattributable ledger block', () => {
    writePlan([SLICE_ID]);
    writeLedgerBodies([blockBody(validBlock())]);
    writeFileSync(
      workspace.logPath,
      [
        fenced('telemetry_ledger:\n'),
        fenced('telemetry_ledger:\n  a: 1\n   b: 2\n'),
        fenced('telemetry_ledger:\n  schema_version: 1\n'),
      ].join('\n'),
      'utf8',
    );
    const result = runInProcess(['--plan', workspace.planPath, '--logs', workspace.logPath]);
    expect(result.status).toBe(1);
    const failures = result.report.failures;
    expect(failures.filter((failure) => failure.kind === 'unattributable-ledger')).toHaveLength(3);
    expect(failures[0].detail).toContain('carries no parsable slice id');
    expect(failures[1].detail).toContain('unexpected indentation');
    expect(failures[2].detail).toContain('declares no slice id');
  });

  it('reads the companion log that sits beside the tracker when no log flag is given', () => {
    writePlan([SLICE_ID]);
    writeFileSync(workspace.logPath, fenced(blockBody(validBlock())), 'utf8');
    const result = runInProcess(['--plan', workspace.planPath]);
    expect(result.status).toBe(0);
    expect(result.report.checked).toBe(1);
    expect(result.report.verdict).toBe('VERIFIED');
  });

  it('never demands a ledger of a slice that closed before the activation marker', () => {
    writePlan([SLICE_ID]);
    rmSync(workspace.logPath, { force: true });
    const result = runInProcess([
      '--plan',
      workspace.planPath,
      '--logs',
      workspace.logPath,
      '--pre-existing-done',
      SLICE_ID,
    ]);
    expect(result.status).toBe(0);
    expect(result.report.pre_existing_done).toEqual([SLICE_ID]);
    expect(result.report.checked).toBe(0);
    expect(result.report.vacuous).toBe(true);
  });
});
