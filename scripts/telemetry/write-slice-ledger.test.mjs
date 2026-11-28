/**
 * @module write-slice-ledger.test
 * @description RED contract home for the DONE writer of the vertical slice A.
 *
 * Slice: OT-V1-1 of `plans/opentelemetry-slice-telemetry.plans.md`.
 * Acceptance criteria pinned by this home:
 *   - AC-V1-C (the DONE writer home), which carries forward AC-002-S4-A,
 *     AC-002-S4-B, AC-012-B, AC-012-C, AC-011-B, AC-006-E and AC-004-D.
 *
 * THE SUBJECT UNDER TEST IS `scripts/telemetry/write-slice-ledger.mjs`, THE
 * REPO-OWNED DONE WRITER. Every lane drives the REAL writer as a REAL child
 * process over a REAL temporary companion log and a REAL temporary dispatch
 * directory. Nothing is mocked: not the writer, not the contract module, not
 * the dispatcher. The only thing this home reads instead of executing is the
 * writer SOURCE, and only for the independence rule (AC-001-D) - and that lane
 * reads the real file the writer owns.
 *
 * WHY THE WRITER IS DRIVEN AS A CHILD PROCESS: the writer is a CLI, its
 * contract IS its command line, and a CLI that is driven through an import
 * cannot prove its exit code, its refusal or its idempotence. The writer does
 * not exist yet, so every lane is RED BY ASSERTION: the lanes fail on the
 * exit code, on the missing block or on the missing file - never on an import
 * error that would abort the file.
 *
 * THE COMMAND LINE THIS HOME DEMANDS:
 *
 *   node scripts/telemetry/write-slice-ledger.mjs \
 *     --plan=<planPath> --slice=<sliceId> \
 *     [--plan-id=<planId>] [--status=<status>] [--otel-state=<state>] \
 *     [--validation=<json>] [--dispatch-dir=<dir>] [--recover]
 *
 *   --plan         REQUIRED. The plan file. The companion log is its sibling
 *                  `<name>.logs.md`, so `x.plans.md` is written into
 *                  `x.logs.md`.
 *   --slice        REQUIRED. The slice id.
 *   --plan-id      OPTIONAL. The plan id; defaults to the plan basename minus
 *                  the `.plans.md` suffix.
 *   --status       OPTIONAL. The slice status; defaults to `[DONE]`.
 *   --otel-state   OPTIONAL. The layer-A reporting state, one of `ENABLED`,
 *                  `DISABLED`, `STALE`, `CORRUPT`, `NO-DATA`.
 *   --validation   OPTIONAL. A JSON object
 *                  `{total, passed, failed, skipped, results:[{name, status,
 *                  duration_ms}]}`; it is persisted verbatim in the validation
 *                  block.
 *   --dispatch-dir OPTIONAL. The transient dispatch-record directory; it
 *                  defaults to `logs/slice-ledger/dispatch` under the repo
 *                  root. The home always points it at a temporary directory,
 *                  so no lane ever touches the repository's own transient
 *                  state.
 *   --recover      The recovery mode for an INTERRUPTED slice.
 *
 * THE EXIT CONTRACT: the writer exits 0 when it wrote a ledger and NON-ZERO
 * when it refused, and a refusal never writes a block.
 *
 * THE LEDGER CONTRACT (the block the writer persists):
 *   - it is ONE fenced `yaml` block, keyed by `plan_id` plus `slice_id`, and
 *     it is written through the ONE deterministic emitter of
 *     `scripts/telemetry/telemetry-contract.mjs`, so the same facts produce
 *     the same bytes;
 *   - a repeated write of the same key leaves the persisted bytes IDENTICAL
 *     (AC-002-S4-A: "the two bytes are identical"), so the upsert is
 *     idempotent in the strict sense;
 *   - it carries the schema version, the status, the duration, the runtime
 *     metadata set (the Node version, the heap usage and the RSS usage, and
 *     NOTHING else), the usage block and the validation block;
 *   - the layer-A state (`otel_state`) and the layer-B measurement state
 *     (`measurement_state`) live in SEPARATE fields, and a layer-A state never
 *     changes the layer-B measurement state (AC-001-D);
 *   - the duration is the MONOTONIC delta of the transient dispatch record,
 *     measured with the same clock the record was written with
 *     (`process.hrtime.bigint()`), and the transient record is DELETED when
 *     the block is written (R-8);
 *   - a missing, unreadable or negative-delta record yields
 *     `measurement_state: UNAVAILABLE` with a reason and a NULL duration,
 *     never a zero (AC-004-D, the M-1 never-zero rule);
 *   - an unavailable counter is `available: false` with a reason and NULL
 *     counters - never a zero;
 *   - the recovered ledger of an interrupted slice carries NO live
 *     measurement, so recovering twice yields a byte-identical block
 *     (AC-006-E);
 *   - a CORRUPT dispatch record is REFUSED: the writer reports the exact
 *     record path and exits non-zero, and writes NO block, because a corrupt
 *     record is repaired, never guessed (AC-006-E);
 *   - the writer imports NO OpenTelemetry package and NO `src/observability`
 *     module, so the repo-owned layer stays provably independent of the
 *     OpenTelemetry layer (AC-001-D).
 *
 * WHY THE HOME PARSES THE LEDGER ITSELF: the persisted block is the contract,
 * so the home reads it as DATA instead of trusting the writer's own report. The
 * reader is a small YAML subset parser that understands exactly what the
 * deterministic emitter writes (two-space indentation, single-quoted scalars,
 * `null`, and lists whose first key rides on the dash). The parser is proven
 * by a round-trip lane over the REAL emitter, so a reader bug can never be
 * mistaken for a writer bug.
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
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
  dispatchRecordPath,
  emitLedgerBlock,
  scanPersistedText,
  SCHEMA_VERSION,
  validateLedgerBlock,
  validateUsageCounter,
  writeDispatchRecord,
} from './telemetry-contract.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
const WRITER_PATH = path.join(REPO_ROOT, 'scripts', 'telemetry', 'write-slice-ledger.mjs');
const WRITER_RELPATH = 'scripts/telemetry/write-slice-ledger.mjs';

const PLAN_ID = 'fixture-plan';
const SLICE_ID = 'OT-V1-9';
const DELIVERED_TOKENS = 1200;

/** The seam the observer resolves the observation log through. */
const OBSERVATION_LOG_ENV = 'NEATAPTIC_OBSERVATION_LOG_PATH';

/** The file name of the observation log, the SIBLING of the dispatch directory. */
const OBSERVATION_LOG_NAME = 'tool-executions.jsonl';

/** The validation summary a lane hands to the writer. */
const VALIDATION_SUMMARY = {
  total: 2,
  passed: 1,
  failed: 1,
  skipped: 0,
  results: [
    { name: 'npm run jest:mjs', status: 'failed', duration_ms: 12 },
    { name: 'npm run jest:base', status: 'passed', duration_ms: 40 },
  ],
};

/** The fixture plan the writer is pointed at. */
const FIXTURE_PLAN = [
  '# Fixture plan',
  '',
  '## Phase 1 - the fixture slice',
  '',
  '```yaml',
  'slices:',
  `  - slice_id: '${SLICE_ID}'`,
  "    title: 'the fixture slice'",
  "    status: '[DONE]'",
  '```',
  '',
].join('\n');

/**
 * Reads a scalar of the emitted YAML subset.
 *
 * @param {string} raw The raw scalar text.
 * @returns {unknown} The parsed value.
 */
function parseScalar(raw) {
  const text = raw.trim();
  if (text === '' || text === 'null' || text === '~') return null;
  if (text === 'true') return true;
  if (text === 'false') return false;
  if (/^-?\d+$/.test(text)) return Number(text);
  if (/^-?\d*\.\d+$/.test(text)) return Number(text);
  if (text.length >= 2 && text.startsWith("'") && text.endsWith("'")) {
    return text.slice(1, -1).replace(/''/g, "'");
  }
  if (text.length >= 2 && text.startsWith('"') && text.endsWith('"')) {
    try {
      return JSON.parse(text);
    } catch {
      return text.slice(1, -1);
    }
  }
  return text;
}

/**
 * Counts the leading spaces of a line.
 *
 * @param {string} line The line.
 * @returns {number} The indentation.
 */
function indentOf(line) {
  return line.length - line.trimStart().length;
}

/**
 * Parses a YAML mapping at a given indentation.
 *
 * @param {string[]} lines The lines.
 * @param {number} start The first line index.
 * @param {number} indent The indentation of the mapping.
 * @returns {[Record<string, unknown>, number]} The mapping and the next line index.
 */
function parseMapping(lines, start, indent) {
  const result = {};
  let index = start;
  while (index < lines.length) {
    const line = lines[index];
    const lineIndent = indentOf(line);
    if (lineIndent < indent) break;
    if (lineIndent > indent) {
      throw new Error(`unexpected indentation in the emitted YAML: ${JSON.stringify(line)}`);
    }
    const content = line.trim();
    const separator = content.indexOf(':');
    if (separator < 0) {
      throw new Error(`unparsable emitted YAML line: ${JSON.stringify(content)}`);
    }
    const key = content.slice(0, separator).trim().replace(/^["']|["']$/g, '');
    const rest = content.slice(separator + 1).trim();
    if (rest.length > 0) {
      result[key] = parseScalar(rest);
      index += 1;
      continue;
    }
    const next = index + 1;
    if (next < lines.length && indentOf(lines[next]) > indent) {
      const childIndent = indentOf(lines[next]);
      const childIsList = lines[next].trim().startsWith('-');
      if (childIsList) {
        const [items, after] = parseSequence(lines, next, childIndent);
        result[key] = items;
        index = after;
        continue;
      }
      const [child, after] = parseMapping(lines, next, childIndent);
      result[key] = child;
      index = after;
      continue;
    }
    result[key] = null;
    index += 1;
  }
  return [result, index];
}

/**
 * Parses a YAML list at a given indentation.
 *
 * @param {string[]} lines The lines.
 * @param {number} start The first line index.
 * @param {number} indent The indentation of the list.
 * @returns {[unknown[], number]} The items and the next line index.
 */
function parseSequence(lines, start, indent) {
  const working = lines.slice();
  const items = [];
  let index = start;
  while (index < working.length) {
    const lineIndent = indentOf(working[index]);
    if (lineIndent < indent) break;
    if (lineIndent > indent) {
      throw new Error(`unexpected list indentation in the emitted YAML: ${JSON.stringify(working[index])}`);
    }
    const content = working[index].trim();
    if (!content.startsWith('-')) break;
    const inner = content.slice(1).replace(/^\s+/, '');
    working[index] = ' '.repeat(lineIndent + 2) + inner;
    const [item, after] = parseMapping(working, index, lineIndent + 2);
    items.push(item);
    index = after;
  }
  return [items, index];
}

/**
 * Parses the YAML subset the deterministic emitter writes.
 *
 * @param {string} text The YAML text.
 * @returns {Record<string, unknown>} The parsed mapping.
 */
function parseYamlSubset(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.replace(/\s+$/, ''))
    .filter((line) => line.trim().length > 0);
  if (lines.length === 0) return {};
  const [value] = parseMapping(lines, 0, indentOf(lines[0]));
  return value;
}

/**
 * Extracts every fenced `yaml` block of a companion log.
 *
 * @param {string} logText The companion log text.
 * @returns {string[]} The block bodies.
 */
function extractFencedBlocks(logText) {
  const bodies = [];
  const pattern = /```yaml\r?\n([\s\S]*?)```/g;
  let match = pattern.exec(logText);
  while (match !== null) {
    bodies.push(match[1]);
    match = pattern.exec(logText);
  }
  return bodies;
}

/**
 * Reads the persisted ledger blocks of a companion log.
 *
 * @param {string} logPath The companion log path.
 * @returns {Record<string, unknown>[]} The ledger blocks.
 */
function readLedgerBlocks(logPath) {
  if (!existsSync(logPath)) return [];
  return extractFencedBlocks(readFileSync(logPath, 'utf8')).map((body) => {
    const parsed = parseYamlSubset(body);
    const block = parsed.telemetry_ledger;
    if (block === null || typeof block !== 'object') {
      throw new Error('the persisted block is not rooted at `telemetry_ledger`');
    }
    return block;
  });
}

/**
 * Collects every module specifier of a source text.
 *
 * @param {string} source The source text.
 * @returns {string[]} The specifiers.
 */
function collectImportSpecifiers(source) {
  const specifiers = [];
  const patterns = [
    /\bfrom\s*['"]([^'"]+)['"]/g,
    /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
    /\bimport\s+['"]([^'"]+)['"]/g,
    /\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
  ];
  for (const pattern of patterns) {
    for (const match of source.matchAll(pattern)) specifiers.push(match[1]);
  }
  return specifiers;
}

/** The temporary workspace of the running lane. */
let workspace;

/**
 * Creates the temporary workspace: a plan file, the companion log path that
 * belongs to it, a temporary dispatch-record directory and the observation
 * log that sits BESIDE that directory.
 *
 * @returns {{ root: string, planPath: string, logPath: string, dispatchDir: string, observationLogPath: string }} The workspace.
 */
function createWorkspace() {
  const root = mkdtempSync(path.join(tmpdir(), 'telemetry-ledger-'));
  const plansDir = path.join(root, 'plans');
  mkdirSync(plansDir, { recursive: true });
  const planPath = path.join(plansDir, `${PLAN_ID}.plans.md`);
  writeFileSync(planPath, FIXTURE_PLAN, 'utf8');
  const dispatchDir = path.join(root, 'dispatch');
  mkdirSync(dispatchDir, { recursive: true });
  return {
    root,
    planPath,
    logPath: path.join(plansDir, `${PLAN_ID}.logs.md`),
    dispatchDir,
    observationLogPath: path.join(root, OBSERVATION_LOG_NAME),
  };
}

/**
 * Runs the REAL writer as a REAL child process.
 *
 * The observation-log seam is CLEARED, so the writer resolves the log the
 * way the observer does - as the sibling of the dispatch-record directory -
 * and no lane can be decided by an ambient environment variable.
 *
 * @param {string[]} args The command-line arguments.
 * @returns {{ status: number|null, stdout: string, stderr: string, output: string }} The run result.
 */
function runWriter(args) {
  const env = { ...process.env };
  delete env[OBSERVATION_LOG_ENV];
  const result = spawnSync(process.execPath, [WRITER_PATH, ...args], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 60000,
    env,
  });
  const stdout = result.stdout ?? '';
  const stderr = result.stderr ?? '';
  return {
    status: result.status,
    stdout,
    stderr,
    output: `${stdout}\n${stderr}`,
  };
}

/**
 * Asserts that the writer is on disk, so a lane that fails afterwards fails
 * on the WRITER's behaviour rather than on a missing file.
 */
function expectWriterAvailable() {
  expect({ writer: WRITER_RELPATH, exists: existsSync(WRITER_PATH) }).toEqual({
    writer: WRITER_RELPATH,
    exists: true,
  });
}

/**
 * Builds the argument list of a normal write.
 *
 * @param {string[]} extra The extra arguments.
 * @returns {string[]} The argument list.
 */
function normalArgs(extra = []) {
  return [
    '--plan',
    workspace.planPath,
    '--slice',
    SLICE_ID,
    '--plan-id',
    PLAN_ID,
    '--status',
    '[DONE]',
    '--otel-state',
    'DISABLED',
    '--dispatch-dir',
    workspace.dispatchDir,
    ...extra,
  ];
}

/**
 * Builds the argument list of a recovery.
 *
 * @param {string[]} extra The extra arguments.
 * @returns {string[]} The argument list.
 */
function recoverArgs(extra = []) {
  return [...normalArgs(extra), '--recover'];
}

/**
 * Writes a transient dispatch record with a monotonic base that sits the
 * requested number of milliseconds in the past.
 *
 * @param {number} millisecondsAgo How far in the past the base sits.
 * @returns {string} The record path.
 */
function writeDispatchRecordInThePast(millisecondsAgo) {
  const base = process.hrtime.bigint() - BigInt(Math.round(millisecondsAgo * 1_000_000));
  return writeDispatchRecord(
    {
      plan_id: PLAN_ID,
      slice_id: SLICE_ID,
      context_tokens_delivered: DELIVERED_TOKENS,
      dispatch_seq: 7,
      started_monotonic_ns: `0x${base.toString(16)}`,
    },
    { directory: workspace.dispatchDir },
  );
}

/**
 * Writes the observation log the tool-execution observer appends to, with
 * the REAL line shape it writes: `{ seq, slice_id, session_key, bucket,
 * outcome }` and nothing else.
 *
 * @param {Array<{ slice_id?: string, session_key?: string, bucket?: string, outcome?: string }>} entries - The observations.
 * @returns {string} The observation log path.
 */
function writeObservationLog(entries) {
  const lines = entries.map((entry, index) =>
    JSON.stringify({
      seq: index,
      slice_id: entry.slice_id ?? SLICE_ID,
      session_key: entry.session_key ?? 'aaaabbbbcccc',
      bucket: entry.bucket ?? 'read',
      outcome: entry.outcome ?? 'completed',
    }),
  );
  writeFileSync(workspace.observationLogPath, `${lines.join('\n')}\n`, 'utf8');
  return workspace.observationLogPath;
}

/**
 * The slice id of a slice that is STILL RUNNING: its OPEN dispatch record
 * must survive the prune of a slice that closes.
 *
 * @type {string}
 */
const RUNNING_SLICE_ID = 'OT-V1-RUNNING';

/**
 * The slice id of the RESIDUE record: a record no DONE writer will ever
 * claim, because its own bytes carry a forbidden key.
 *
 * @type {string}
 */
const RESIDUE_SLICE_ID = 'OT-TEST-1';

/**
 * The epoch-millisecond literal the residue record carries: a forbidden
 * value shape.
 *
 * @type {number}
 */
const FORBIDDEN_INSTANT = 1764147453321;

/**
 * Writes the OPEN dispatch record of a slice that is STILL RUNNING, with the
 * REAL producer the contract module owns.
 *
 * @returns {string} The path of the written record.
 */
function writeRunningSliceRecord() {
  return writeDispatchRecord(
    {
      plan_id: PLAN_ID,
      slice_id: RUNNING_SLICE_ID,
      context_tokens_delivered: DELIVERED_TOKENS,
      dispatch_seq: 1,
      started_monotonic_ns: `0x${process.hrtime.bigint().toString(16)}`,
    },
    { directory: workspace.dispatchDir },
  );
}

/**
 * Writes a RESIDUE record: a record whose own bytes violate the
 * persisted-artefact contract, so the pruner MUST remove it.
 *
 * @returns {string} The path of the written record.
 */
function writeResidueRecord() {
  const target = path.join(workspace.dispatchDir, `${PLAN_ID}__${RESIDUE_SLICE_ID}.json`);
  writeFileSync(
    target,
    `${JSON.stringify(
      { plan_id: PLAN_ID, slice_id: RESIDUE_SLICE_ID, timestamp: FORBIDDEN_INSTANT },
      null,
      2,
    )}\n`,
    'utf8',
  );
  return target;
}

/**
 * Writes a CORRUPT dispatch record of the slice the writer is about to
 * close, so the writer REFUSES instead of guessing.
 *
 * @returns {string} The path of the written record.
 */
function writeCorruptDispatchRecord() {
  const target = dispatchRecordPath(
    { plan_id: PLAN_ID, slice_id: SLICE_ID },
    { directory: workspace.dispatchDir },
  );
  writeFileSync(target, 'not a dispatch record\n', 'utf8');
  return target;
}

beforeEach(() => {
  workspace = createWorkspace();
});
afterEach(() => {
  rmSync(workspace.root, { recursive: true, force: true });
});

describe('OT-V1-1: the YAML reader this home uses', () => {
  it('reads back exactly what the deterministic emitter wrote', () => {
    // The reader is the instrument of every lane below, so it is proven
    // against the REAL emitter: a reader bug can never be mistaken for a
    // writer bug.
    const emitted = {
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
          source: 'dispatch_record',
          measures: 'the delivered context-token count',
          context_tokens_delivered: DELIVERED_TOKENS,
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          unavailable_reason: null,
        },
        tool_executions: {
          available: true,
          source: 'copilot-cli-posthook-observation-window',
          measures:
            'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
          scope: 'slice-dispatch-window',
          total: 3,
          completed_executions: 2,
          failed_executions: 1,
          sessions_observed: 1,
          breakdown: { read: 1, edit: 1, search: 0, exec: 1, other: 0 },
          unavailable_reason: null,
        },
      },
      validation: {
        total: 2,
        passed: 1,
        failed: 1,
        skipped: 0,
        results: [
          { name: 'npm run jest:mjs', status: 'failed', duration_ms: 12 },
          { name: 'npm run jest:base', status: 'passed', duration_ms: 40 },
        ],
      },
    };
    const body = extractFencedBlocks(emitLedgerBlock(emitted))[0];
    const parsed = parseYamlSubset(body);
    // The emitted block is rooted at `telemetry_ledger`, so the reader hands
    // back the root and the block is its single child.
    expect(Object.keys(parsed)).toEqual(['telemetry_ledger']);
    expect(parsed.telemetry_ledger).toEqual(emitted);
  });
});

describe('OT-V1-1: the DONE writer', () => {
  it('exists at the path the contract declares', () => {
    expect({ writer: WRITER_RELPATH, exists: existsSync(WRITER_PATH) }).toEqual({
      writer: WRITER_RELPATH,
      exists: true,
    });
  });

  it('writes exactly one ledger block for a completed slice', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    const result = runWriter(normalArgs());
    expect(result.status).toBe(0);
    expect(readLedgerBlocks(workspace.logPath)).toHaveLength(1);
  });

  it('declares the schema version, the plan id, the slice id and the status', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    expect(block.schema_version).toBe(SCHEMA_VERSION);
    expect(String(block.plan_id)).toContain(PLAN_ID);
    expect(block.slice_id).toBe(SLICE_ID);
    expect(String(block.status)).toContain('DONE');
  });

  it('records the measured monotonic duration of the slice window', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    expect(typeof block.duration_ms).toBe('number');
    expect(block.duration_ms).toBeGreaterThan(0);
  });

  it('writes the runtime metadata set and nothing else', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    expect(Object.keys(block.runtime).sort()).toEqual([
      'heap_used_bytes',
      'node_version',
      'rss_bytes',
    ]);
    expect(String(block.runtime.node_version)).toMatch(/^v?\d+\.\d+\.\d+/);
    expect(typeof block.runtime.heap_used_bytes).toBe('number');
    expect(block.runtime.heap_used_bytes).toBeGreaterThan(0);
    expect(typeof block.runtime.rss_bytes).toBe('number');
    expect(block.runtime.rss_bytes).toBeGreaterThan(0);
  });

  it('keeps the layer-A state and the layer-B measurement state in separate fields', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs(['--otel-state', 'DISABLED']));
    const [block] = readLedgerBlocks(workspace.logPath);
    expect(block.otel_state).toBe('DISABLED');
    expect(block.measurement_state).toBe('MEASURED');
  });

  it('writes the delivered context tokens and leaves the LLM token consumption null', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    const tokenUsage = block.usage.token_usage;
    expect(tokenUsage.available).toBe(true);
    expect(typeof tokenUsage.source).toBe('string');
    expect(tokenUsage.source.length).toBeGreaterThan(0);
    expect(typeof tokenUsage.measures).toBe('string');
    expect(tokenUsage.measures.length).toBeGreaterThan(0);
    expect(tokenUsage.context_tokens_delivered).toBe(DELIVERED_TOKENS);
    expect(tokenUsage.prompt_tokens).toBeNull();
    expect(tokenUsage.completion_tokens).toBeNull();
    expect(tokenUsage.total_tokens).toBeNull();
  });

  it('writes the tool-execution counter as unavailable with a reason and NULL counters', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    const toolExecutions = block.usage.tool_executions;
    expect(toolExecutions.available).toBe(false);
    expect(toolExecutions.scope).toBe('slice-dispatch-window');
    const reason = toolExecutions.unavailable_reason ?? toolExecutions.reason;
    expect(reason).toBe('observation_log_missing');
    expect(toolExecutions.total).toBeNull();
    expect(toolExecutions.completed_executions).toBeNull();
    expect(toolExecutions.failed_executions).toBeNull();
    expect(toolExecutions.sessions_observed).toBeNull();
    expect(toolExecutions.breakdown).toBeNull();
  });

  it('writes no zero for an unavailable counter', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    const counters = [block.usage.token_usage, block.usage.tool_executions];
    for (const counter of counters) {
      if (counter.available !== false) continue;
      for (const [key, value] of Object.entries(counter)) {
        if (
          [
            'available',
            'source',
            'measures',
            'scope',
            'reason',
            'unavailable_reason',
          ].includes(key)
        ) {
          continue;
        }
        expect({ key, value }).toEqual({ key, value: null });
      }
    }
  });

  it('round-trips the validation summary it is given', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs(['--validation', JSON.stringify(VALIDATION_SUMMARY)]));
    const [block] = readLedgerBlocks(workspace.logPath);
    expect(block.validation.total).toBe(VALIDATION_SUMMARY.total);
    expect(block.validation.passed).toBe(VALIDATION_SUMMARY.passed);
    expect(block.validation.failed).toBe(VALIDATION_SUMMARY.failed);
    expect(block.validation.skipped).toBe(VALIDATION_SUMMARY.skipped);
    expect(block.validation.results).toHaveLength(VALIDATION_SUMMARY.results.length);
    expect(block.validation.results[0].name).toBe(VALIDATION_SUMMARY.results[0].name);
    expect(block.validation.results[0].status).toBe(VALIDATION_SUMMARY.results[0].status);
    expect(block.validation.results[0].duration_ms).toBe(VALIDATION_SUMMARY.results[0].duration_ms);
  });

  it('writes a ledger the shared validator accepts', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    const verdict = validateLedgerBlock(block);
    expect(verdict.valid).toBe(true);
    expect(verdict.errors).toEqual([]);
  });

  it('upserts exactly ONE block for a repeated write and leaves the bytes identical', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    const first = runWriter(normalArgs());
    expect(first.status).toBe(0);
    const firstBytes = readFileSync(workspace.logPath, 'utf8');
    runWriter(normalArgs());
    const secondBytes = readFileSync(workspace.logPath, 'utf8');
    expect(readLedgerBlocks(workspace.logPath)).toHaveLength(1);
    expect(secondBytes).toBe(firstBytes);
  });

  it('deletes the transient dispatch record when the ledger block is written', () => {
    expectWriterAvailable();
    const recordPath = writeDispatchRecordInThePast(50);
    expect(existsSync(recordPath)).toBe(true);
    runWriter(normalArgs());
    expect(existsSync(recordPath)).toBe(false);
  });

  it('removes the transient dispatch record on an IDEMPOTENT write too', () => {
    expectWriterAvailable();
    // The first write closes the slice and removes the record. A record that
    // is written AGAIN for the SAME slice must not survive the idempotent
    // write: an idempotent write still CLOSES the slice, so the transient
    // state is closed either way (R-8).
    writeDispatchRecordInThePast(50);
    expect(runWriter(normalArgs()).status).toBe(0);
    const recordPath = writeDispatchRecordInThePast(50);
    expect(existsSync(recordPath)).toBe(true);
    const second = runWriter(normalArgs());
    expect(second.status).toBe(0);
    expect(second.output).toContain('the transient dispatch record is removed');
    expect(existsSync(recordPath)).toBe(false);
    expect(readLedgerBlocks(workspace.logPath)).toHaveLength(1);
  });

  it('finds the existing block in a CRLF companion log and writes no second block', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    expect(runWriter(normalArgs()).status).toBe(0);
    // A companion log written on Windows carries CRLF. The idempotent write
    // must still FIND the block that is already there, or it appends a second
    // block under the same key.
    const crlf = readFileSync(workspace.logPath, 'utf8').replace(/\r?\n/g, '\r\n');
    writeFileSync(workspace.logPath, crlf, 'utf8');
    const second = runWriter(normalArgs());
    expect(second.status).toBe(0);
    expect(readLedgerBlocks(workspace.logPath)).toHaveLength(1);
  });

  it('writes no forbidden key and no forbidden value into the companion log', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs());
    expect(existsSync(workspace.logPath)).toBe(true);
    const violations = scanPersistedText(readFileSync(workspace.logPath, 'utf8'));
    expect(violations).toEqual([]);
  });

  it('imports no OpenTelemetry package and no src/observability module', () => {
    expectWriterAvailable();
    const source = readFileSync(WRITER_PATH, 'utf8');
    const specifiers = collectImportSpecifiers(source);
    const forbidden = specifiers.filter(
      (specifier) =>
        specifier.startsWith('@opentelemetry') || specifier.includes('src/observability'),
    );
    expect(forbidden).toEqual([]);
  });
});

describe('OT-V2-3 D13: the DONE writer consumes the REAL per-slice tool observations', () => {
  it('writes the tool-execution counter the observation log really carries for the slice', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeObservationLog([
      { bucket: 'read', outcome: 'completed' },
      { bucket: 'edit', outcome: 'completed' },
      { bucket: 'exec', outcome: 'failed' },
    ]);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    const toolExecutions = block.usage.tool_executions;
    expect({
      available: toolExecutions.available,
      total: toolExecutions.total,
      completed_executions: toolExecutions.completed_executions,
      failed_executions: toolExecutions.failed_executions,
    }).toEqual({
      available: true,
      total: 3,
      completed_executions: 2,
      failed_executions: 1,
    });
  });

  it('writes the per-bucket breakdown with EVERY safe bucket present', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeObservationLog([
      { bucket: 'read', outcome: 'completed' },
      { bucket: 'edit', outcome: 'completed' },
      { bucket: 'exec', outcome: 'failed' },
    ]);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    expect(block.usage.tool_executions.breakdown).toEqual({
      read: 1,
      edit: 1,
      search: 0,
      exec: 1,
      other: 0,
    });
  });

  it('counts the DISTINCT sessions the observations of the slice carry', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeObservationLog([
      { session_key: 'aaaabbbbcccc', bucket: 'read', outcome: 'completed' },
      { session_key: 'aaaabbbbcccc', bucket: 'edit', outcome: 'completed' },
      { session_key: 'ddddeeeeffff', bucket: 'read', outcome: 'completed' },
    ]);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    expect(block.usage.tool_executions.sessions_observed).toBe(2);
  });

  it('gives an available tool-execution counter the scope the DONE gate demands', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeObservationLog([{ bucket: 'read', outcome: 'completed' }]);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    expect(block.usage.tool_executions.scope).toBe('slice-dispatch-window');
  });

  it('writes the tool-execution counter as unavailable with the missing-log reason', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    const toolExecutions = block.usage.tool_executions;
    expect(toolExecutions.available).toBe(false);
    expect(toolExecutions.unavailable_reason ?? toolExecutions.reason).toBe(
      'observation_log_missing',
    );
  });

  it('writes the tool-execution counter as unavailable with the empty-log reason', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeFileSync(workspace.observationLogPath, '', 'utf8');
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    const toolExecutions = block.usage.tool_executions;
    expect(toolExecutions.available).toBe(false);
    expect(toolExecutions.unavailable_reason ?? toolExecutions.reason).toBe(
      'observation_log_empty',
    );
  });

  it('writes the tool-execution counter as unavailable with the no-observation reason', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeObservationLog([
      { slice_id: 'OT-V1-8', bucket: 'read', outcome: 'completed' },
    ]);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    const toolExecutions = block.usage.tool_executions;
    expect(toolExecutions.available).toBe(false);
    expect(toolExecutions.unavailable_reason ?? toolExecutions.reason).toBe(
      'no_observations_for_slice',
    );
  });

  it('writes NO zero for an unavailable tool-execution counter while the log exists', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeObservationLog([
      { slice_id: 'OT-V1-8', bucket: 'read', outcome: 'completed' },
    ]);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    const toolExecutions = block.usage.tool_executions;
    expect({
      total: toolExecutions.total,
      completed_executions: toolExecutions.completed_executions,
      failed_executions: toolExecutions.failed_executions,
      sessions_observed: toolExecutions.sessions_observed,
      breakdown: toolExecutions.breakdown,
    }).toEqual({
      total: null,
      completed_executions: null,
      failed_executions: null,
      sessions_observed: null,
      breakdown: null,
    });
  });

  it('attributes NO observation of another slice to this slice', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeObservationLog([
      { slice_id: 'OT-V1-8', bucket: 'read', outcome: 'completed' },
      { slice_id: 'OT-V1-8', bucket: 'edit', outcome: 'completed' },
      { slice_id: 'OT-V1-8', bucket: 'exec', outcome: 'completed' },
    ]);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    const toolExecutions = block.usage.tool_executions;
    expect({
      available: toolExecutions.available,
      total: toolExecutions.total,
    }).toEqual({
      available: false,
      total: null,
    });
  });

  it('writes no forbidden key and no forbidden value for an available counter', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeObservationLog([
      { bucket: 'read', outcome: 'completed' },
      { bucket: 'edit', outcome: 'failed' },
    ]);
    runWriter(normalArgs());
    expect(scanPersistedText(readFileSync(workspace.logPath, 'utf8'))).toEqual(
      [],
    );
  });

  it('writes a ledger the shared validator accepts for an available counter', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeObservationLog([
      { bucket: 'read', outcome: 'completed' },
      { bucket: 'edit', outcome: 'failed' },
    ]);
    runWriter(normalArgs());
    const [block] = readLedgerBlocks(workspace.logPath);
    expect(validateLedgerBlock(block).valid).toBe(true);
  });
});

describe('OT-V1-1: the interrupted slice and the recovery mode', () => {
  it('never writes a MEASURED ledger for a slice with no dispatch record', () => {
    expectWriterAvailable();
    runWriter(normalArgs());
    const blocks = readLedgerBlocks(workspace.logPath);
    expect(
      blocks.filter((block) => block.measurement_state === 'MEASURED'),
    ).toHaveLength(0);
    for (const block of blocks) {
      expect(block.duration_ms).toBeNull();
    }
  });

  it('writes an explicit UNAVAILABLE ledger for an interrupted slice when --recover is run', () => {
    expectWriterAvailable();
    const result = runWriter(recoverArgs());
    expect(result.status).toBe(0);
    const blocks = readLedgerBlocks(workspace.logPath);
    expect(blocks).toHaveLength(1);
    const [block] = blocks;
    expect(block.measurement_state).toBe('UNAVAILABLE');
    expect(block.duration_ms).toBeNull();
    const tokenUsage = block.usage.token_usage;
    expect(tokenUsage.available).toBe(false);
    const tokenReason = tokenUsage.unavailable_reason ?? tokenUsage.reason;
    expect(typeof tokenReason).toBe('string');
    expect(tokenReason.length).toBeGreaterThan(0);
    expect(tokenUsage.context_tokens_delivered).toBeNull();
    const toolExecutions = block.usage.tool_executions;
    expect(toolExecutions.available).toBe(false);
    const toolReason =
      toolExecutions.unavailable_reason ?? toolExecutions.reason;
    expect(toolReason).toBe('observation_log_missing');
    expect(toolExecutions.total).toBeNull();
  });

  it('writes the same bytes when --recover is run twice', () => {
    expectWriterAvailable();
    const first = runWriter(recoverArgs());
    expect(first.status).toBe(0);
    const firstBytes = readFileSync(workspace.logPath, 'utf8');
    const second = runWriter(recoverArgs());
    expect(second.status).toBe(0);
    expect(readFileSync(workspace.logPath, 'utf8')).toBe(firstBytes);
  });

  it('refuses to overwrite a CORRUPT dispatch record', () => {
    expectWriterAvailable();
    const recordPath = dispatchRecordPath(
      { plan_id: PLAN_ID, slice_id: SLICE_ID },
      { directory: workspace.dispatchDir },
    );
    mkdirSync(path.dirname(recordPath), { recursive: true });
    writeFileSync(recordPath, '{ this is not json', 'utf8');
    const result = runWriter(recoverArgs());
    expect(result.status).not.toBe(0);
    expect(readLedgerBlocks(workspace.logPath)).toHaveLength(0);
    expect(result.output).toContain(path.basename(recordPath));
    expect(result.output).toMatch(/corrupt|repair|recover|invalid|unreadable/i);
  });

  it('reports a negative monotonic delta as UNAVAILABLE with a NULL duration', () => {
    expectWriterAvailable();
    const base = process.hrtime.bigint() + 60_000_000_000n;
    writeDispatchRecord(
      {
        plan_id: PLAN_ID,
        slice_id: SLICE_ID,
        context_tokens_delivered: DELIVERED_TOKENS,
        dispatch_seq: 7,
        started_monotonic_ns: `0x${base.toString(16)}`,
      },
      { directory: workspace.dispatchDir },
    );
    runWriter(normalArgs());
    const blocks = readLedgerBlocks(workspace.logPath);
    expect(blocks).toHaveLength(1);
    const [block] = blocks;
    expect(block.measurement_state).toBe('UNAVAILABLE');
    expect(block.duration_ms).toBeNull();
    expect(block.usage.tool_executions.available).toBe(false);
  });
});

describe('OT-V1-1: the availability rule the shared contract owns', () => {
  it('rejects a ledger in which an unavailable counter was written as zero', () => {
    // The DONE-closure gate and the writer share ONE notion of validity, so
    // the never-zero rule is proven through the ONE contract module rather
    // than through a second implementation.
    const verdict = validateUsageCounter({
      available: false,
      source: 'observation_log',
      measures: 'the tool-execution count',
      scope: 'slice-dispatch-window',
      total: 0,
      unavailable_reason: 'observation_log_missing',
    });
    expect(verdict.valid).toBe(false);
    expect(verdict.errors.length).toBeGreaterThan(0);
  });
});

describe('OT-V2-3 D14: the DONE writer prunes the dispatch-record residue', () => {
  it('wires the SAFE residue pruner into the writer', () => {
    expectWriterAvailable();
    const specifiers = collectImportSpecifiers(readFileSync(WRITER_PATH, 'utf8'));
    expect(specifiers).toContain('./prune-dispatch-records.mjs');
  });

  it('removes the contract-violating residue when it closes a slice', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    const residue = writeResidueRecord();
    const result = runWriter(normalArgs());
    expect(result.status).toBe(0);
    expect(existsSync(residue)).toBe(false);
  });

  it('keeps the OPEN dispatch record of a slice that is still running', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    const running = writeRunningSliceRecord();
    const result = runWriter(normalArgs());
    expect(result.status).toBe(0);
    expect(existsSync(running)).toBe(true);
  });

  it('keeps the running slice and removes the residue in a MIXED directory', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    const running = writeRunningSliceRecord();
    const residue = writeResidueRecord();
    const result = runWriter(normalArgs());
    expect(result.status).toBe(0);
    expect(existsSync(running)).toBe(true);
    expect(existsSync(residue)).toBe(false);
  });

  it('prunes the residue on the IDEMPOTENT closure path too', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    const first = runWriter(normalArgs());
    const residue = writeResidueRecord();
    const second = runWriter(normalArgs());
    expect(first.status).toBe(0);
    expect(second.status).toBe(0);
    expect(existsSync(residue)).toBe(false);
  });

  it('reports the prune COUNTS and never names a removed record', () => {
    expectWriterAvailable();
    writeDispatchRecordInThePast(50);
    writeResidueRecord();
    const result = runWriter(normalArgs());
    expect(result.stdout).toContain('the dispatch-record residue pruned:');
    expect(result.stdout).toContain('1 removed');
    expect(result.stdout).not.toContain(RESIDUE_SLICE_ID);
  });

  it('prunes NOTHING when it refuses to write the ledger', () => {
    expectWriterAvailable();
    writeCorruptDispatchRecord();
    const residue = writeResidueRecord();
    const result = runWriter(normalArgs());
    expect(result.status).toBe(1);
    expect(existsSync(residue)).toBe(true);
  });
});
