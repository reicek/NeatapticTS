#!/usr/bin/env node
/**
 * @module ledger-state
 * @description The telemetry STATE CLASSIFIER of the vertical slice B
 * (AC-V2-F). It answers two questions about ONE slice and answers them
 * HONESTLY:
 *
 *   - the LAYER-A reporting state - is the OpenTelemetry layer `ENABLED`,
 *     `DISABLED`, `STALE`, `CORRUPT` or `NO-DATA` for this slice?
 *   - the LAYER-B measurement state - is the repo-owned measurement `MEASURED`,
 *     `UNAVAILABLE` or `CORRUPT` for this slice?
 *
 * THE CLASSIFIER PERFORMS NO RECOVERY. It never writes a ledger, never runs
 * the writer's `--recover` mode, never repairs a transient artefact and never
 * creates or deletes one: the recovery is owned by the WRITER
 * (`scripts/telemetry/write-slice-ledger.mjs`) and the enablement is owned by
 * `scripts/telemetry/ensure-local-telemetry.mjs`. The classifier only reads.
 *
 * THE TWO LAYERS STAY SEPARATE. A `DISABLED` or `NO-DATA` layer A still
 * YIELDS a ledger - the writer writes one whose `measurement_state` is
 * `UNAVAILABLE` and whose duration is `null` - and a `CORRUPT` layer-B
 * artefact REFUSES the transition. An unavailable measurement is `null`,
 * NEVER zero.
 *
 * THE MODULE ALSO OWNS THE SHARED YAML-SUBSET PARSER. The deterministic
 * emitter of `scripts/telemetry/telemetry-contract.mjs` writes one YAML
 * subset, and the DONE gate has to READ that same subset, so the parser
 * lives here and is exported for the gate.
 *
 * THE COMMAND LINE:
 *
 *   node scripts/telemetry/ledger-state.mjs --json --slice=<sliceId> \
 *     [--settings=<path>] [--sink=<dir>] [--dispatch-record=<path>] [--ledger=<path>]
 *
 * @example
 * node scripts/telemetry/ledger-state.mjs --json --slice=OT-V2-2
 * // => { "slice_id": "OT-V2-2", "state": "DISABLED", ... }
 */
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LEDGER_ROOT_KEY, SCHEMA_VERSION } from './telemetry-contract.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');

/** The repository-relative directory of the LOCAL settings. */
const DEFAULT_SETTINGS_DIR = path.join('logs', 'telemetry');
/** The file name of the LOCAL settings inside that directory. */
const SETTINGS_FILE_NAME = 'settings.json';
/** The repository-relative directory of the TRANSIENT dispatch records. */
const DEFAULT_DISPATCH_DIR = path.join('logs', 'slice-ledger', 'dispatch');
/** The repo-relative enablement command the repair instruction names. */
const ENABLEMENT_COMMAND = 'node scripts/telemetry/ensure-local-telemetry.mjs --json';
/** The five layer-A reporting states. */
const LAYER_A_STATES = Object.freeze(['ENABLED', 'DISABLED', 'STALE', 'CORRUPT', 'NO-DATA']);
/** The three layer-B measurement states. */
const LAYER_B_STATES = Object.freeze(['MEASURED', 'UNAVAILABLE', 'CORRUPT']);

/**
 * Parses a scalar of the emitted YAML subset.
 *
 * @param {string} raw The raw scalar text.
 * @returns {unknown} The parsed value.
 * @example
 * parseScalar("'the slice'");
 * // => 'the slice'
 */
export function parseScalar(raw) {
  const text = String(raw ?? '').trim();
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
 * @example
 * indentOf('  slice_id: x');
 * // => 2
 */
export function indentOf(line) {
  return String(line).length - String(line).trimStart().length;
}

/**
 * Tells whether a line opens a MAPPING rather than carrying a scalar.
 *
 * A quoted scalar is never a mapping, whatever it contains, and a plain
 * scalar only opens a mapping when a colon separates a key from a value.
 *
 * @param {string} content The trimmed line content.
 * @returns {boolean} True when the content opens a mapping.
 * @example
 * isMappingEntry("slice_id: 'OT-V2-2'");
 * // => true
 */
export function isMappingEntry(content) {
  const text = String(content ?? '');
  if (text.startsWith("'") || text.startsWith('"')) return false;
  const separator = text.indexOf(':');
  if (separator < 0) return false;
  const rest = text.slice(separator + 1);
  return rest.length === 0 || rest.startsWith(' ');
}

/**
 * Parses a YAML mapping at a given indentation.
 *
 * @param {string[]} lines The lines.
 * @param {number} start The first line index.
 * @param {number} indent The indentation of the mapping.
 * @returns {[Record<string, unknown>, number]} The mapping and the next line index.
 * @example
 * parseMapping(['a: 1'], 0, 0);
 * // => [{ a: 1 }, 1]
 */
export function parseMapping(lines, start, indent) {
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
      if (lines[next].trim().startsWith('-')) {
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
 * @example
 * parseSequence(['- a'], 0, 0);
 * // => [['a'], 1]
 */
export function parseSequence(lines, start, indent) {
  const working = lines.slice();
  const items = [];
  let index = start;
  while (index < working.length) {
    const lineIndent = indentOf(working[index]);
    if (lineIndent < indent) break;
    if (lineIndent > indent) {
      throw new Error(
        `unexpected list indentation in the emitted YAML: ${JSON.stringify(working[index])}`,
      );
    }
    const content = working[index].trim();
    if (!content.startsWith('-')) break;
    const inner = content.slice(1).replace(/^\s+/, '');
    if (!isMappingEntry(inner)) {
      items.push(parseScalar(inner));
      index += 1;
      continue;
    }
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
 * @example
 * parseYamlSubset('telemetry_ledger:\n  slice_id: \'OT-V2-2\'');
 * // => { telemetry_ledger: { slice_id: 'OT-V2-2' } }
 */
export function parseYamlSubset(text) {
  const lines = String(text ?? '')
    .split(/\r?\n/)
    .map((line) => line.replace(/\s+$/, ''))
    .filter((line) => line.trim().length > 0);
  if (lines.length === 0) return {};
  const [value] = parseMapping(lines, 0, indentOf(lines[0]));
  return value;
}

/**
 * Extracts every fenced `yaml` block of a text.
 *
 * The fence tolerates BOTH line endings: a tracker written on Windows carries
 * CRLF, and a block whose fence is CRLF-terminated is a REAL block. A pattern
 * that only accepted LF would silently skip every CRLF tracker and the gate
 * would scan an EMPTY tracker.
 *
 * @param {string} text The text.
 * @returns {string[]} The block bodies.
 * @example
 * extractFencedBlocks('```yaml\na: 1\n```');
 * // => ['a: 1\n']
 */
export function extractFencedBlocks(text) {
  const bodies = [];
  const pattern = /```ya?ml\r?\n([\s\S]*?)```/g;
  let match = pattern.exec(String(text ?? ''));
  while (match !== null) {
    bodies.push(match[1]);
    match = pattern.exec(String(text ?? ''));
  }
  return bodies;
}

/**
 * Extracts every LEDGER block of a companion log and parses it.
 *
 * A block is a ledger block when its first line is the ledger root key. A
 * ledger block that does not parse is REPORTED with its error instead of
 * being dropped, because an unparsable ledger is a HARD FAILURE and never
 * an invisible artefact.
 *
 * @param {string} logText The companion log text.
 * @returns {Array<{ body: string, block: Record<string, unknown>|null, error: string|null }>} The ledger blocks.
 * @example
 * extractLedgerBlocks('```yaml\ntelemetry_ledger:\n  slice_id: x\n```');
 * // => [{ body: '...', block: { slice_id: 'x' }, error: null }]
 */
export function extractLedgerBlocks(logText) {
  const blocks = [];
  for (const body of extractFencedBlocks(logText)) {
    const firstLine = body.split(/\r?\n/).find((line) => line.trim().length > 0);
    if (firstLine === undefined || firstLine.trim() !== `${LEDGER_ROOT_KEY}:`) continue;
    try {
      const parsed = parseYamlSubset(body);
      const block = parsed[LEDGER_ROOT_KEY];
      blocks.push({
        body,
        block: block !== null && typeof block === 'object' && !Array.isArray(block) ? block : null,
        error: null,
      });
    } catch (error) {
      blocks.push({
        body,
        block: null,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
  return blocks;
}

/**
 * Reads the tracker of a plan and returns its ORDERED slice list.
 *
 * The tracker of a plan may carry SEVERAL `slices:` lists - one per phase
 * packet - so every fenced block is read and the lists are concatenated in
 * DOCUMENT order. The order is what the forward-only rule needs: a slice
 * that closed BEFORE the activation marker is never demanded a ledger.
 *
 * @param {string} planText The tracker text.
 * @returns {{ slices: Array<{ slice_id: string, status: string, index: number }>, activation_marker: string|null }} The slices and the declared activation marker.
 * @example
 * readTrackerSlices('# tracker');
 * // => { slices: [], activation_marker: null }
 */
export function readTrackerSlices(planText) {
  const slices = [];
  let activationMarker = null;
  for (const body of extractFencedBlocks(planText)) {
    let parsed;
    try {
      parsed = parseYamlSubset(body);
    } catch {
      continue;
    }
    if (typeof parsed.ledger_activation_marker === 'string') {
      activationMarker = parsed.ledger_activation_marker.trim();
    }
    const list = parsed.slices;
    if (!Array.isArray(list)) continue;
    for (const entry of list) {
      if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) continue;
      const sliceId = typeof entry.slice_id === 'string' ? entry.slice_id.trim() : '';
      if (sliceId.length === 0) continue;
      slices.push({
        slice_id: sliceId,
        status: typeof entry.status === 'string' ? entry.status.trim() : '',
        index: slices.length,
      });
    }
  }
  return { slices, activation_marker: activationMarker };
}

/**
 * Tells whether a tracker slice is closed.
 *
 * @param {{ status: string }} entry The tracker slice.
 * @returns {boolean} True when the slice carries the `[DONE]` marker.
 * @example
 * isDone({ status: '[DONE]' });
 * // => true
 */
export function isDone(entry) {
  return /\[\s*DONE\s*\]/i.test(String(entry?.status ?? ''));
}

/**
 * Reads a JSON file and tells the three states apart.
 *
 * @param {string} filePath The file path.
 * @returns {{ status: 'OK' | 'MISSING' | 'CORRUPT', value: unknown }} The read result.
 * @example
 * readJsonFile('C:\\missing.json');
 * // => { status: 'MISSING', value: null }
 */
export function readJsonFile(filePath) {
  if (!existsSync(filePath)) return { status: 'MISSING', value: null };
  try {
    const parsed = JSON.parse(readFileSync(filePath, 'utf8'));
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { status: 'CORRUPT', value: null };
    }
    return { status: 'OK', value: parsed };
  } catch {
    return { status: 'CORRUPT', value: null };
  }
}

/**
 * Finds the layer-A record of a slice in the local sink.
 *
 * @param {string} sinkPath The local JSONL sink (a file or a directory).
 * @param {string} sliceId The slice id.
 * @returns {Record<string, unknown>|null} The record, or null when there is none.
 * @example
 * findLayerARecord('C:\\logs\\telemetry', 'OT-V2-2');
 * // => { slice_id: 'OT-V2-2', ... }
 */
export function findLayerARecord(sinkPath, sliceId) {
  const files = [];
  try {
    if (!existsSync(sinkPath)) return null;
    if (statSync(sinkPath).isDirectory()) {
      for (const name of readdirSync(sinkPath)) {
        if (!name.endsWith('.jsonl')) continue;
        const candidate = path.join(sinkPath, name);
        if (statSync(candidate).isFile()) files.push(candidate);
      }
    } else {
      files.push(sinkPath);
    }
  } catch {
    return null;
  }
  for (const file of files) {
    let text = '';
    try {
      text = readFileSync(file, 'utf8');
    } catch {
      continue;
    }
    for (const line of text.split(/\r?\n/)) {
      if (line.trim().length === 0) continue;
      let record = null;
      try {
        record = JSON.parse(line);
      } catch {
        continue;
      }
      if (record !== null && typeof record === 'object' && record.slice_id === sliceId) {
        return record;
      }
    }
  }
  return null;
}

/**
 * Classifies the LAYER-A reporting state of a slice.
 *
 * @param {{ sliceId: string, settingsPath: string, sinkPath: string }} request The request.
 * @returns {{ state: string, artefact: string, reason: string }} The verdict.
 * @example
 * classifyLayerA({ sliceId: 'OT-V2-2', settingsPath: 's.json', sinkPath: 'd' });
 * // => { state: 'DISABLED', artefact: 's.json', reason: '...' }
 */
export function classifyLayerA(request) {
  const { sliceId, settingsPath, sinkPath } = request;
  if (!existsSync(settingsPath)) {
    return {
      state: 'DISABLED',
      artefact: settingsPath,
      reason: 'the local telemetry settings are missing',
    };
  }
  const read = readJsonFile(settingsPath);
  if (read.status === 'CORRUPT') {
    return {
      state: 'CORRUPT',
      artefact: settingsPath,
      reason: 'the local telemetry settings carry no parsable JSON object',
    };
  }
  const settings = read.value;
  if (settings.schema_version !== SCHEMA_VERSION) {
    return {
      state: 'STALE',
      artefact: settingsPath,
      reason: `the local telemetry settings declare schema_version ${String(settings.schema_version)}, not ${SCHEMA_VERSION}`,
    };
  }
  if (settings.enabled !== true) {
    return {
      state: 'DISABLED',
      artefact: settingsPath,
      reason: 'the local telemetry settings are disabled',
    };
  }
  if (findLayerARecord(sinkPath, sliceId) === null) {
    return {
      state: 'NO-DATA',
      artefact: sinkPath,
      reason: 'the local telemetry is enabled and healthy, but the local sink holds no record for this slice',
    };
  }
  return {
    state: 'ENABLED',
    artefact: sinkPath,
    reason: 'the local telemetry is enabled and the local sink holds a record for this slice',
  };
}

/**
 * Finds the transient dispatch record of a slice in the default directory.
 *
 * @param {string} sliceId The slice id.
 * @param {string} [dispatchDir] The dispatch-record directory.
 * @returns {string|null} The record path, or null when there is none.
 * @example
 * findDispatchRecord('OT-V2-2');
 * // => 'C:\\...\\logs\\slice-ledger\\dispatch\\plan__OT-V2-2.json'
 */
export function findDispatchRecord(sliceId, dispatchDir = path.join(REPO_ROOT, DEFAULT_DISPATCH_DIR)) {
  const suffix = `__${String(sliceId).replace(/[^A-Za-z0-9._-]/g, '_')}.json`;
  try {
    if (!existsSync(dispatchDir) || !statSync(dispatchDir).isDirectory()) return null;
    const match = readdirSync(dispatchDir).find((name) => name.endsWith(suffix));
    if (match === undefined) return null;
    return path.join(dispatchDir, match);
  } catch {
    return null;
  }
}

/**
 * Classifies the LAYER-B measurement state of a slice.
 *
 * A CORRUPT transient dispatch record wins over everything else: a corrupt
 * artefact is reported, never guessed around.
 *
 * @param {{ sliceId: string, ledgerPath: string|null, dispatchRecord: string|null }} request The request.
 * @returns {{ measurement_state: string, duration_ms: number|null, reason: string }} The verdict.
 * @example
 * classifyLayerB({ sliceId: 'OT-V2-2', ledgerPath: null, dispatchRecord: null });
 * // => { measurement_state: 'UNAVAILABLE', duration_ms: null, reason: '...' }
 */
export function classifyLayerB(request) {
  const { sliceId, ledgerPath, dispatchRecord } = request;
  if (dispatchRecord !== null && dispatchRecord !== undefined) {
    const read = readJsonFile(dispatchRecord);
    if (read.status === 'CORRUPT') {
      return {
        measurement_state: 'CORRUPT',
        duration_ms: null,
        reason: 'the transient dispatch record is corrupt and is repaired, never guessed',
      };
    }
  }
  if (ledgerPath !== null && ledgerPath !== undefined && existsSync(ledgerPath)) {
    let text = '';
    try {
      text = readFileSync(ledgerPath, 'utf8');
    } catch {
      text = '';
    }
    const blocks = extractLedgerBlocks(text).filter(
      (entry) => entry.block !== null && entry.block.slice_id === sliceId,
    );
    if (blocks.length > 0) {
      const block = blocks[blocks.length - 1].block;
      const state = block.measurement_state;
      if (state === 'MEASURED') {
        return {
          measurement_state: 'MEASURED',
          duration_ms: typeof block.duration_ms === 'number' ? block.duration_ms : null,
          reason: 'the ledger block records a live measurement of this slice',
        };
      }
      if (state === 'CORRUPT') {
        return {
          measurement_state: 'CORRUPT',
          duration_ms: null,
          reason: 'the ledger block records a corrupt measurement of this slice',
        };
      }
      return {
        measurement_state: 'UNAVAILABLE',
        duration_ms: null,
        reason: 'the ledger block records an unavailable measurement of this slice',
      };
    }
  }
  return {
    measurement_state: 'UNAVAILABLE',
    duration_ms: null,
    reason: 'no ledger block records a measurement of this slice',
  };
}

/**
 * Classifies the state of a slice and returns the full report.
 *
 * @param {{ sliceId: string, settingsPath?: string, sinkPath?: string, ledgerPath?: string|null, dispatchRecord?: string|null }} request The request.
 * @returns {Record<string, unknown>} The report.
 * @example
 * classifySlice({ sliceId: 'OT-V2-2' });
 * // => { slice_id: 'OT-V2-2', state: 'DISABLED', measurement_state: 'UNAVAILABLE', ... }
 */
export function classifySlice(request) {
  const sliceId = String(request.sliceId ?? '').trim();
  const settingsPath = request.settingsPath
    ? path.resolve(request.settingsPath)
    : path.join(REPO_ROOT, DEFAULT_SETTINGS_DIR, SETTINGS_FILE_NAME);
  const sinkPath = request.sinkPath
    ? path.resolve(request.sinkPath)
    : path.dirname(settingsPath);
  const ledgerPath = request.ledgerPath ? path.resolve(request.ledgerPath) : null;
  const dispatchRecord =
    request.dispatchRecord === null || request.dispatchRecord === undefined
      ? findDispatchRecord(sliceId)
      : path.resolve(request.dispatchRecord);
  const layerA = classifyLayerA({ sliceId, settingsPath, sinkPath });
  const layerB = classifyLayerB({ sliceId, ledgerPath, dispatchRecord });
  return {
    slice_id: sliceId,
    state: layerA.state,
    artefact: layerA.artefact,
    reason: layerA.reason,
    repair_command: ENABLEMENT_COMMAND,
    measurement_state: layerB.measurement_state,
    duration_ms: layerB.duration_ms,
    measurement_reason: layerB.reason,
    settings: settingsPath,
    sink: sinkPath,
    ledger: ledgerPath,
    dispatch_record: dispatchRecord,
  };
}

/**
 * Parses the command line of the classifier.
 *
 * Both spellings of every flag are accepted: `--flag value` and
 * `--flag=value`.
 *
 * @param {string[]} argv The arguments after the script path.
 * @returns {{ slice: string|null, settings: string|null, sink: string|null, dispatchRecord: string|null, ledger: string|null, json: boolean }} The parsed options.
 * @example
 * parseArgs(['--json', '--slice', 'OT-V2-2']);
 * // => { slice: 'OT-V2-2', ..., json: true }
 */
export function parseArgs(argv) {
  const options = {
    slice: null,
    settings: null,
    sink: null,
    dispatchRecord: null,
    ledger: null,
    json: false,
  };
  const valueFlags = new Map([
    ['--slice', 'slice'],
    ['--settings', 'settings'],
    ['--sink', 'sink'],
    ['--dispatch-record', 'dispatchRecord'],
    ['--ledger', 'ledger'],
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
      options[key] = argument.slice(separator + 1);
      continue;
    }
    const key = valueFlags.get(argument);
    if (key === undefined) continue;
    const value = argv[index + 1];
    if (value === undefined) continue;
    options[key] = value;
    index += 1;
  }
  return options;
}

/**
 * Runs the classifier.
 *
 * @param {string[]} argv The arguments after the script path.
 * @returns {number} The exit code.
 * @example
 * main(['--json', '--slice', 'OT-V2-2']);
 * // => 0
 */
export function main(argv) {
  const options = parseArgs(argv);
  if (options.slice === null || options.slice.trim().length === 0) {
    process.stderr.write('the flag --slice is required\n');
    return 2;
  }
  const report = classifySlice({
    sliceId: options.slice,
    settingsPath: options.settings,
    sinkPath: options.sink,
    ledgerPath: options.ledger,
    dispatchRecord: options.dispatchRecord,
  });
  process.stdout.write(`${JSON.stringify(report)}\n`);
  return 0;
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedDirectly) {
  process.exitCode = main(process.argv.slice(2));
}

export { LAYER_A_STATES, LAYER_B_STATES, REPO_ROOT };
