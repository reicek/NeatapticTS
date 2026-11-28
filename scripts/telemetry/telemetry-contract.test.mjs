/**
 * @module telemetry-contract.test
 * @description The focused home of the contract declaration module.
 *
 * Slice: OT-S1-2 of `plans/opentelemetry-slice-telemetry.plans.md`.
 * Acceptance criteria covered by this home:
 *   - AC-003-B (the declaration)
 *   - AC-001-A (the layer independence)
 *
 * The RED home of the step (`scripts/telemetry/slice-ledger-contract.test.mjs`,
 * owned by OT-S1-1) pins the CONTRACT of the module from the outside. This home
 * pins the parts of the DECLARATION that the contract home deliberately leaves
 * open, because the tracker names them explicitly and the module must not
 * drift away from them:
 *   - the availability enum, the layer-A reporting states and the layer-B
 *     measurement states, named and enumerated exactly;
 *   - the FIXED key order and the SORTED breakdown keys of the deterministic
 *     emitter, which are the two properties that make the rewrite idempotent
 *     (R-2);
 *   - the ONE-scanner property: the module exports exactly one scanner, so the
 *     privacy sweep CLI, the ledger gate and the tests cannot drift apart
 *     (AC-003-B);
 *   - the layer independence of AC-001-A, asserted against the module SOURCE
 *     and not against a convention;
 *   - the transient dispatch-record helper, which the dispatch-time producer
 *     (OT-S1-4) uses and which must never persist a forbidden field (R-8).
 *   - the DEFENSIVE lanes of the declared API (a non-string text, a non-finite
 *     number, a bigint, an exotic value, a usage mapping that is not an object
 *     and the default dispatch-record directory), so every REACHABLE branch of
 *     the declaration is pinned by a lane of this home and not only by the
 *     contract home.
 *
 * The subject under test is resolved LAZILY inside every assertion, so an
 * unimplemented module is an ASSERTION FAILURE that names the missing module
 * and never an import failure that aborts the whole file.
 *
 * @example
 * node --test scripts/telemetry/telemetry-contract.test.mjs
 */

import { describe, it, expect } from '@jest/globals';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
const MODULE_PATH = path.join(HERE, 'telemetry-contract.mjs');

/**
 * Resolves the contract module inside the calling lane and asserts that it
 * exists.
 *
 * @returns {Promise<Record<string, unknown>>} The module namespace.
 */
async function requireContract() {
  let contract;
  try {
    contract = await import('./telemetry-contract.mjs');
  } catch (error) {
    const message = String(error?.message ?? error);
    expect(message).toBe('');
    throw error;
  }
  expect(contract).toBeDefined();
  return contract;
}

/**
 * Reads the module source, so a structural property (the absence of an import)
 * can be asserted against the bytes rather than against a convention.
 *
 * @returns {string} The module source.
 */
function readModuleSource() {
  return readFileSync(MODULE_PATH, 'utf8');
}

/**
 * Collects every import specifier of the module source.
 *
 * @param {string} source The module source.
 * @returns {string[]} The import specifiers.
 */
function collectImportSpecifiers(source) {
  const specifiers = [];
  const pattern = /(?:^|\n)\s*(?:import|export)\s[^;\n]*?from\s*['"]([^'"]+)['"]/g;
  let match = pattern.exec(source);
  while (match !== null) {
    specifiers.push(match[1]);
    match = pattern.exec(source);
  }
  const bare = /(?:^|\n)\s*import\s*['"]([^'"]+)['"]/g;
  let bareMatch = bare.exec(source);
  while (bareMatch !== null) {
    specifiers.push(bareMatch[1]);
    bareMatch = bare.exec(source);
  }
  return specifiers;
}

/**
 * Extracts the lines that belong to a nested YAML block, so a lane can assert
 * on ONE block instead of on the whole emitted text.
 *
 * @param {string} text The emitted text.
 * @param {string} header The header of the block, for example `breakdown:`.
 * @returns {string[]} The lines of the block.
 */
function extractNestedBlock(text, header) {
  const lines = text.split('\n');
  const startIndex = lines.findIndex((line) => line.trim() === header);
  if (startIndex === -1) return [];
  const headerIndent = lines[startIndex].length - lines[startIndex].trimStart().length;
  const block = [];
  for (let index = startIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim().length === 0) continue;
    const indent = line.length - line.trimStart().length;
    if (indent <= headerIndent) break;
    block.push(line);
  }
  return block;
}

/**
 * Removes the comments of a source, so a structural property can be asserted
 * against the CODE and not against the prose that documents it.
 *
 * @param {string} source The module source.
 * @returns {string} The source without its comments.
 */
function stripComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^[ \t]*\/\/.*$/gm, '');
}

/**
 * Tells whether the text of a sequence item starts a MAPPING (and is not a
 * quoted scalar that merely contains a colon).
 *
 * @param {string} text The text of the sequence item.
 * @returns {boolean} True for a mapping item.
 */
function isMappingItem(text) {
  return /^[A-Za-z_][A-Za-z0-9_]*:(\s|$)/.test(text);
}

/**
 * Parses the restricted YAML subset the emitter writes (mappings, sequences
 * and scalars) and returns the reconstructed value.
 *
 * The parser is deliberately STRICT about the indentation grammar, because the
 * defect class it guards against IS an indentation defect: a mapping entry
 * whose indent does not match the indent of its siblings is invalid YAML, and
 * the emitted block is persisted VERBATIM by the DONE writer, so an invalid
 * block is a broken ledger rather than a cosmetic difference.
 *
 * It is written here instead of imported, because the repository declares no
 * YAML parser dependency: a transitive `js-yaml` would be a phantom
 * dependency that could vanish on a clean install and break this home.
 *
 * @param {string} text The YAML text.
 * @returns {unknown} The parsed value.
 * @example
 * parseYamlSubset("root:\n  items:\n    - name: 'a'\n      count: 1\n");
 * // => { root: { items: [ { name: 'a', count: 1 } ] } }
 */
function parseYamlSubset(text) {
  const lines = text.split('\n').filter((line) => line.trim().length > 0);
  let cursor = 0;

  const indentOf = (line) => line.length - line.trimStart().length;
  const contentOf = (line) => line.trimStart();

  const fail = (message, lineNumber) => {
    throw new Error(`invalid YAML at line ${lineNumber}: ${message}`);
  };

  const parseScalar = (raw, lineNumber) => {
    const value = raw.trim();
    if (value === '' || value === 'null' || value === '~') return null;
    if (value === 'true') return true;
    if (value === 'false') return false;
    if (/^-?\d+(\.\d+)?$/.test(value)) return Number(value);
    if (value.startsWith("'")) {
      if (value.length < 2 || !value.endsWith("'")) {
        fail('an unterminated single-quoted scalar', lineNumber);
      }
      return value.slice(1, -1).replace(/''/g, "'");
    }
    if (value.startsWith('"')) {
      try {
        return JSON.parse(value);
      } catch {
        return fail('an unparsable double-quoted scalar', lineNumber);
      }
    }
    return value;
  };

  const parseMapping = (indent) => {
    const map = {};
    while (cursor < lines.length) {
      const line = lines[cursor];
      const currentIndent = indentOf(line);
      if (currentIndent < indent) break;
      if (currentIndent > indent) {
        fail('bad indentation of a mapping entry', cursor + 1);
      }
      const content = contentOf(line);
      if (content.startsWith('- ')) {
        fail('a sequence cannot share the indent of the mapping keys above it', cursor + 1);
      }
      const colon = content.indexOf(':');
      if (colon === -1) fail('a mapping entry needs a key', cursor + 1);
      const key = content.slice(0, colon).trim();
      const rawValue = content.slice(colon + 1).trim();
      cursor += 1;
      if (rawValue.length > 0) {
        map[key] = parseScalar(rawValue, cursor);
        continue;
      }
      if (cursor >= lines.length) {
        map[key] = null;
        continue;
      }
      const childIndent = indentOf(lines[cursor]);
      if (childIndent <= indent) {
        map[key] = null;
        continue;
      }
      map[key] = parseNode(childIndent);
    }
    return map;
  };

  const parseSequence = (indent) => {
    const items = [];
    while (cursor < lines.length) {
      const line = lines[cursor];
      const currentIndent = indentOf(line);
      if (currentIndent < indent) break;
      if (currentIndent > indent) {
        fail('bad indentation of a sequence entry', cursor + 1);
      }
      const content = contentOf(line);
      if (!content.startsWith('- ')) break;
      const rest = content.slice(2);
      if (isMappingItem(rest)) {
        lines[cursor] = ' '.repeat(indent + 2) + rest;
        items.push(parseMapping(indent + 2));
        continue;
      }
      cursor += 1;
      items.push(parseScalar(rest, cursor));
    }
    return items;
  };

  function parseNode(indent) {
    if (contentOf(lines[cursor]).startsWith('- ')) return parseSequence(indent);
    return parseMapping(indent);
  }

  if (lines.length === 0) return null;
  const parsed = parseNode(indentOf(lines[0]));
  if (cursor !== lines.length) fail('trailing content after the document', cursor + 1);
  return parsed;
}

/**
 * Strips the fences of the emitted block, so the BODY can be parsed.
 *
 * @param {string} emitted The emitted block.
 * @returns {string} The YAML body.
 */
function ledgerBody(emitted) {
  return emitted.replace(/^```ya?ml\r?\n/, '').replace(/\r?\n```\s*$/, '');
}

/**
 * The canonical schema-v1 ledger block of the tracker, used as the fixture of
 * the emitter lanes.
 *
 * @returns {Record<string, unknown>} The canonical block.
 */
function canonicalBlock() {
  return {
    schema_version: 1,
    plan_id: 'opentelemetry-slice-telemetry',
    slice_id: 'OT-S3-2',
    status: '[DONE]',
    measurement_state: 'MEASURED',
    otel_state: 'DISABLED',
    duration_ms: 41233,
    runtime: {
      node_version: '22.14.0',
      heap_used_bytes: 41234567,
      rss_bytes: 98765432,
    },
    usage: {
      token_usage: {
        available: true,
        source: 'cortex-get-slice-context',
        measures: 'context tokens DELIVERED to the slice, not model consumption',
        context_tokens_delivered: 18432,
        prompt_tokens: null,
        completion_tokens: null,
        total_tokens: null,
        unavailable_reason: 'the repo has no access to the host model accounting',
      },
      tool_executions: {
        available: true,
        source: 'copilot-cli-posthook-observation-window',
        measures: 'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
        scope: 'slice-dispatch-window',
        total: 17,
        completed_executions: 16,
        failed_executions: 1,
        sessions_observed: 1,
        breakdown: { read: 9, edit: 4, search: 4, exec: 0, other: 0 },
      },
    },
    validation: {
      total: 3,
      passed: 3,
      failed: 0,
      skipped: 0,
      results: [{ name: 'unit', status: 'passed', duration_ms: 412 }],
    },
  };
}

describe('telemetry contract module', () => {
  describe('AC-003-B - the declaration', () => {
    it('declares the schema version as the integer 1', async () => {
      const contract = await requireContract();
      expect(Number.isInteger(contract.SCHEMA_VERSION)).toBe(true);
      expect(contract.SCHEMA_VERSION).toBe(1);
    });

    it('declares the required key set in the canonical order', async () => {
      const contract = await requireContract();
      expect([...contract.REQUIRED_LEDGER_KEYS]).toEqual([
        'schema_version',
        'plan_id',
        'slice_id',
        'status',
        'measurement_state',
        'otel_state',
        'duration_ms',
        'runtime',
        'usage',
        'validation',
      ]);
    });

    it('declares the R-3 forbidden key set, including the modification markers', async () => {
      const contract = await requireContract();
      const forbidden = [...contract.FORBIDDEN_KEYS];
      for (const key of [
        'timestamp',
        'time',
        'startTime',
        'endTime',
        'timeUnixNano',
        'observedTimestamp',
        'startedAt',
        'completedAt',
        'date',
        'datetime',
        'epoch',
        'mtime',
        'mtimeMs',
        'lastModified',
        'lastParsedAt',
        'parsedAt',
        'writtenAt',
      ]) {
        expect(forbidden).toContain(key);
      }
    });

    it('declares the availability enum as available true and unavailable false', async () => {
      const contract = await requireContract();
      expect(contract.AVAILABILITY.AVAILABLE).toBe(true);
      expect(contract.AVAILABILITY.UNAVAILABLE).toBe(false);
    });

    it('declares the layer-A reporting states', async () => {
      const contract = await requireContract();
      expect([...contract.LAYER_A_REPORTING_STATES]).toEqual([
        'ENABLED',
        'DISABLED',
        'STALE',
        'CORRUPT',
        'NO-DATA',
      ]);
    });

    it('declares the layer-B measurement states', async () => {
      const contract = await requireContract();
      expect([...contract.LAYER_B_MEASUREMENT_STATES]).toEqual([
        'MEASURED',
        'UNAVAILABLE',
        'CORRUPT',
      ]);
    });

    it('declares exactly the five safe tool-execution buckets', async () => {
      const contract = await requireContract();
      expect([...contract.TOOL_EXECUTION_BUCKETS]).toEqual([
        'read',
        'edit',
        'search',
        'exec',
        'other',
      ]);
    });

    it('declares the ledger writer path inside scripts/telemetry', async () => {
      const contract = await requireContract();
      expect(contract.LEDGER_WRITER_PATH).toBe('scripts/telemetry/write-slice-ledger.mjs');
      const resolved = path.resolve(REPO_ROOT, contract.LEDGER_WRITER_PATH);
      expect(resolved.startsWith(path.join(REPO_ROOT, 'scripts', 'telemetry'))).toBe(true);
    });

    it('exports exactly ONE scanner, so the sweep, the gate and the tests cannot drift apart', async () => {
      const contract = await requireContract();
      const scanners = Object.entries(contract).filter(
        ([name, value]) => typeof value === 'function' && /scan/i.test(name),
      );
      expect(scanners.map(([name]) => name)).toEqual(['scanPersistedText']);
    });

    it('reports the offending key AND the offending pattern through that one scanner', async () => {
      const contract = await requireContract();
      const keyViolations = contract.scanPersistedText('  timestamp: 1764147453321');
      expect(keyViolations.length).toBeGreaterThan(0);
      expect(keyViolations.some((violation) => violation.key === 'timestamp')).toBe(true);

      const valueViolations = contract.scanPersistedText("  node_version: 'C:\\Users\\alice\\app'");
      expect(valueViolations.length).toBeGreaterThan(0);
      expect(
        valueViolations.some((violation) => violation.kind === 'forbidden-value'),
      ).toBe(true);
    });

    it('reports no violation for a clean persisted text', async () => {
      const contract = await requireContract();
      expect(contract.scanPersistedText("plan_id: 'opentelemetry-slice-telemetry'")).toEqual([]);
    });
  });

  describe('AC-003-B - the deterministic emitter', () => {
    it('emits a fixed key order, so the rewrite is byte-identical (R-2)', async () => {
      const contract = await requireContract();
      const emitted = contract.emitLedgerBlock(canonicalBlock());
      const positions = [
        'schema_version',
        'plan_id',
        'slice_id',
        'status',
        'measurement_state',
        'otel_state',
        'duration_ms',
        'runtime',
        'usage',
        'validation',
      ].map((key) => emitted.indexOf(`  ${key}:`));
      for (const position of positions) {
        expect(position).toBeGreaterThan(-1);
      }
      expect([...positions].toSorted((left, right) => left - right)).toEqual(positions);
    });

    it('emits the breakdown keys sorted, so the producer cannot leak its own ordering', async () => {
      const contract = await requireContract();
      const emitted = contract.emitLedgerBlock(canonicalBlock());
      const buckets = extractNestedBlock(emitted, 'breakdown:').map((line) =>
        line.trim().split(':')[0],
      );
      expect(buckets).toEqual(['edit', 'exec', 'other', 'read', 'search']);
    });

    it('emits the identical bytes for identical input', async () => {
      const contract = await requireContract();
      expect(contract.emitLedgerBlock(canonicalBlock())).toBe(
        contract.emitLedgerBlock(canonicalBlock()),
      );
    });

    it('emits the fenced yaml block the companion log carries', async () => {
      const contract = await requireContract();
      const emitted = contract.emitLedgerBlock(canonicalBlock());
      expect(emitted.startsWith('```yaml\n')).toBe(true);
      expect(emitted.trimEnd().endsWith('```')).toBe(true);
      expect(emitted).toContain('telemetry_ledger:');
    });

    it('emits a body that PARSES as yaml and round-trips to the block it was given', async () => {
      const contract = await requireContract();
      const body = ledgerBody(contract.emitLedgerBlock(canonicalBlock()));
      const parsed = parseYamlSubset(body);
      expect(parsed).toEqual({ telemetry_ledger: canonicalBlock() });
    });

    it('aligns the sibling keys of a sequence item with its first key', async () => {
      const contract = await requireContract();
      const body = ledgerBody(contract.emitLedgerBlock(canonicalBlock()));
      const lines = body.split('\n');
      const markerIndex = lines.findIndex((line) => line.trimStart().startsWith('- '));
      expect(markerIndex).toBeGreaterThan(-1);
      const markerLine = lines[markerIndex];
      const markerIndent = markerLine.length - markerLine.trimStart().length;
      const siblings = [];
      for (let index = markerIndex + 1; index < lines.length; index += 1) {
        const line = lines[index];
        if (line.trim().length === 0) continue;
        const indent = line.length - line.trimStart().length;
        if (indent <= markerIndent) break;
        siblings.push({ indent, key: line.trim().split(':')[0] });
      }
      expect(siblings.map((sibling) => sibling.key)).toEqual(['status', 'duration_ms']);
      for (const sibling of siblings) {
        expect(sibling.indent).toBe(markerIndent + 2);
      }
    });

    it('removes a forbidden key instead of exempting it (R-3)', async () => {
      const contract = await requireContract();
      const block = { ...canonicalBlock(), timestamp: 1764147453321 };
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).not.toMatch(/^[ \t]*timestamp[ \t]*:/m);
      expect(contract.scanPersistedText(emitted)).toEqual([]);
    });

    it('emits an unavailable counter as null and never as a zero (R-4)', async () => {
      const contract = await requireContract();
      const block = canonicalBlock();
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
        unavailable_reason: 'the host CLI exposes no post-execution hook on this machine',
      };
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).toMatch(/total: null/);
      expect(emitted).toMatch(/unavailable_reason: 'the host CLI exposes no post-execution hook on this machine'/);
      expect(emitted).not.toMatch(/['"]?total['"]?\s*:\s*0\b/);
    });

    it('never emits a value the scanner itself forbids', async () => {
      const contract = await requireContract();
      expect(contract.scanPersistedText(contract.emitLedgerBlock(canonicalBlock()))).toEqual([]);
    });
  });

  describe('AC-003-B - the availability validator', () => {
    it('accepts an available counter that names its source and what it measures', async () => {
      const contract = await requireContract();
      const verdict = contract.validateUsageCounter({
        available: true,
        source: 'copilot-cli-posthook-observation-window',
        measures: 'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
        total: 17,
      });
      expect(verdict.valid).toBe(true);
      expect(verdict.errors).toEqual([]);
    });

    it('accepts an unavailable counter whose numeric fields are null and which carries a reason', async () => {
      const contract = await requireContract();
      const verdict = contract.validateUsageCounter({
        available: false,
        source: 'copilot-cli-posthook-observation-window',
        measures: 'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
        total: null,
        unavailable_reason: 'the host CLI exposes no post-execution hook on this machine',
      });
      expect(verdict.valid).toBe(true);
    });

    it('refuses an unavailable counter that carries a zero instead of a null', async () => {
      const contract = await requireContract();
      const verdict = contract.validateUsageCounter({
        available: false,
        source: 'copilot-cli-posthook-observation-window',
        measures: 'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
        total: 0,
        unavailable_reason: 'the host CLI exposes no post-execution hook on this machine',
      });
      expect(verdict.valid).toBe(false);
      expect(verdict.errors.length).toBeGreaterThan(0);
    });

    it('refuses an available counter that names no source', async () => {
      const contract = await requireContract();
      const verdict = contract.validateUsageCounter({
        available: true,
        measures: 'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
        total: 17,
      });
      expect(verdict.valid).toBe(false);
    });
  });

  describe('AC-003-B - the forward-only rule', () => {
    it('does not demand a ledger for a slice that completed before the activation marker', async () => {
      const contract = await requireContract();
      expect(
        contract.demandsLedger({
          slice_id: 'OT-S1-1',
          pre_existing_done_slice_ids: ['OT-S1-1'],
        }),
      ).toBe(false);
    });

    it('demands a ledger for a slice that completed after the activation marker', async () => {
      const contract = await requireContract();
      expect(
        contract.demandsLedger({
          slice_id: 'OT-S1-2',
          pre_existing_done_slice_ids: ['OT-S1-1'],
        }),
      ).toBe(true);
    });
  });

  describe('AC-003-B - the transient dispatch record (R-8)', () => {
    it('writes, reads and deletes a dispatch record that carries no forbidden field', async () => {
      const contract = await requireContract();
      const directory = mkdtempSync(path.join(os.tmpdir(), 'telemetry-dispatch-'));
      try {
        const identity = { plan_id: 'opentelemetry-slice-telemetry', slice_id: 'OT-S1-4' };
        const written = contract.writeDispatchRecord(
          {
            ...identity,
            context_tokens_delivered: 18432,
            dispatch_seq: 7,
            timestamp: 1764147453321,
          },
          { directory },
        );
        expect(written.startsWith(directory)).toBe(true);

        const read = contract.readDispatchRecord(identity, { directory });
        expect(read.status).toBe('OK');
        expect(read.record.plan_id).toBe('opentelemetry-slice-telemetry');
        expect(read.record.slice_id).toBe('OT-S1-4');
        expect(read.record.context_tokens_delivered).toBe(18432);
        expect(read.record.dispatch_seq).toBe(7);
        expect(Object.keys(read.record).toSorted()).toEqual([...contract.DISPATCH_RECORD_FIELDS].toSorted());
        expect(contract.scanPersistedText(JSON.stringify(read.record))).toEqual([]);

        expect(contract.deleteDispatchRecord(identity, { directory })).toBe(true);
        expect(contract.readDispatchRecord(identity, { directory }).status).toBe('MISSING');
        expect(contract.deleteDispatchRecord(identity, { directory })).toBe(false);
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    });
  });

  describe('AC-003-B - the ledger block validator (the exported contract API)', () => {
    it('accepts the canonical schema-v1 block', async () => {
      const contract = await requireContract();
      expect(contract.validateLedgerBlock(canonicalBlock()).valid).toBe(true);
    });

    it('reports a block that declares no required key', async () => {
      const contract = await requireContract();
      const block = canonicalBlock();
      delete block.status;
      const verdict = contract.validateLedgerBlock(block);
      expect(verdict.errors.some((error) => error.includes('`status`'))).toBe(true);
    });

    it('reports a block that declares the wrong schema version', async () => {
      const contract = await requireContract();
      const verdict = contract.validateLedgerBlock({ ...canonicalBlock(), schema_version: 2 });
      expect(verdict.errors.some((error) => error.includes('schema_version'))).toBe(true);
    });

    it('reports a block whose key field is empty', async () => {
      const contract = await requireContract();
      const verdict = contract.validateLedgerBlock({ ...canonicalBlock(), plan_id: '   ' });
      expect(verdict.errors.some((error) => error.includes('non-empty `plan_id`'))).toBe(true);
    });

    it('reports a forbidden key that sits anywhere in the tree', async () => {
      const contract = await requireContract();
      const block = canonicalBlock();
      block.usage.tool_executions.breakdown = { read: 1, timestamp: 1764147453321 };
      const verdict = contract.validateLedgerBlock(block);
      expect(verdict.errors.some((error) => error.includes('forbidden key `timestamp`'))).toBe(
        true,
      );
    });

    it('reports an unknown measurement_state', async () => {
      const contract = await requireContract();
      const verdict = contract.validateLedgerBlock({
        ...canonicalBlock(),
        measurement_state: 'PARTLY-MEASURED',
      });
      expect(
        verdict.errors.some((error) => error.includes('unknown measurement_state')),
      ).toBe(true);
    });

    it('reports an unknown otel_state', async () => {
      const contract = await requireContract();
      const verdict = contract.validateLedgerBlock({ ...canonicalBlock(), otel_state: 'MAYBE' });
      expect(verdict.errors.some((error) => error.includes('unknown otel_state'))).toBe(true);
    });

    it('reports a block that declares no usage counter', async () => {
      const contract = await requireContract();
      const block = canonicalBlock();
      delete block.usage.tool_executions;
      const verdict = contract.validateLedgerBlock(block);
      expect(verdict.errors.some((error) => error.includes('usage.tool_executions'))).toBe(true);
    });

    it('reports a block that is not an object', async () => {
      const contract = await requireContract();
      expect(contract.validateLedgerBlock('not a block').valid).toBe(false);
    });

    it('reports a usage counter that is not an object', async () => {
      const contract = await requireContract();
      expect(contract.validateUsageCounter('not a counter').errors).toContain(
        'the usage counter is not an object',
      );
    });

    it('reports a usage counter that declares no boolean available flag', async () => {
      const contract = await requireContract();
      const verdict = contract.validateUsageCounter({ source: 's', measures: 'm' });
      expect(verdict.errors.some((error) => error.includes('boolean `available`'))).toBe(true);
    });
  });

  describe('AC-003-B - the emitter sequence shapes', () => {
    it('emits a scalar sequence item as a scalar', async () => {
      const contract = await requireContract();
      const block = canonicalBlock();
      block.validation.results = ['unit'];
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).toContain("- 'unit'");
    });

    it('emits a fully-forbidden sequence item as an empty mapping', async () => {
      const contract = await requireContract();
      const block = canonicalBlock();
      block.validation.results = [{ timestamp: 1764147453321 }];
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).toContain('- {}');
    });

    it('emits a nested mapping that rides on the first key of a sequence item', async () => {
      const contract = await requireContract();
      const block = canonicalBlock();
      block.validation.results = [{ meta: { tag: 'unit' } }];
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).toContain('- meta:');
      expect(emitted).toContain("tag: 'unit'");
    });

    it('emits a nested sequence that rides on the first key of a sequence item', async () => {
      const contract = await requireContract();
      const block = canonicalBlock();
      block.validation.results = [{ meta: ['a', 'b'] }];
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).toContain('- meta:');
      expect(emitted).toContain("- 'a'");
    });
  });

  describe('AC-003-B - the scanner reports one offending key exactly once', () => {
    it('reports a quoted forbidden key exactly once', async () => {
      const contract = await requireContract();
      const violations = contract.scanPersistedText('  "timestamp": 1');
      expect(violations.length).toBe(1);
      expect(violations[0].key).toBe('timestamp');
    });

    it('reports an unquoted forbidden key exactly once', async () => {
      const contract = await requireContract();
      expect(contract.scanPersistedText('  timestamp: 1').length).toBe(1);
    });

    it('still reports a quoted forbidden key that does not start the line', async () => {
      const contract = await requireContract();
      const violations = contract.scanPersistedText('{"timestamp": 1}');
      expect(violations.map((violation) => violation.key)).toEqual(['timestamp']);
    });

    it('reports a forbidden key once even when one line spells it twice', async () => {
      const contract = await requireContract();
      const violations = contract.scanPersistedText("  'timestamp': 'timestamp': 1");
      expect(violations.filter((violation) => violation.kind === 'forbidden-key').length).toBe(1);
    });

    it('reports a forbidden key once even when one line spells it in two raw forms', async () => {
      const contract = await requireContract();
      const violations = contract.scanPersistedText('  start_time: 1 "startTime": 2');
      const forbiddenKeys = violations.filter((violation) => violation.kind === 'forbidden-key');
      expect(forbiddenKeys.length).toBe(1);
      expect(['start_time', 'startTime']).toContain(forbiddenKeys[0].key);
    });

    it('still reports two DIFFERENT forbidden keys that share one line', async () => {
      const contract = await requireContract();
      const violations = contract.scanPersistedText('  timestamp: 1 "startedAt": 2');
      const forbiddenKeys = violations.filter((violation) => violation.kind === 'forbidden-key');
      expect(forbiddenKeys.length).toBe(2);
    });
  });

  describe('AC-003-B - an exotic value is emitted as null, never as an empty mapping', () => {
    it('emits a Date value as null instead of as an empty mapping', async () => {
      const contract = await requireContract();
      const block = { ...canonicalBlock(), observed_at: new Date(0) };
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).toContain('  observed_at: null');
    });

    it('never persists the own properties of an exotic value', async () => {
      const contract = await requireContract();
      class Secretive {
        constructor() {
          this.secret = 'do-not-persist';
        }
      }
      const block = { ...canonicalBlock(), observed_at: new Secretive() };
      expect(contract.emitLedgerBlock(block)).not.toContain('do-not-persist');
    });
  });

  describe('AC-003-B - the transient dispatch record (R-8), the corrupt and the monotonic lanes', () => {
    it('reports a dispatch record that carries unparseable JSON as CORRUPT', async () => {
      const contract = await requireContract();
      const directory = mkdtempSync(path.join(os.tmpdir(), 'telemetry-corrupt-'));
      try {
        const identity = { plan_id: 'opentelemetry-slice-telemetry', slice_id: 'OT-S1-4' };
        const target = contract.dispatchRecordPath(identity, { directory });
        mkdirSync(path.dirname(target), { recursive: true });
        writeFileSync(target, '{ not json', 'utf8');
        expect(contract.readDispatchRecord(identity, { directory }).status).toBe('CORRUPT');
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    });

    it('reports a dispatch record that carries non-object JSON as CORRUPT', async () => {
      const contract = await requireContract();
      const directory = mkdtempSync(path.join(os.tmpdir(), 'telemetry-corrupt-'));
      try {
        const identity = { plan_id: 'opentelemetry-slice-telemetry', slice_id: 'OT-S1-4' };
        const target = contract.dispatchRecordPath(identity, { directory });
        mkdirSync(path.dirname(target), { recursive: true });
        writeFileSync(target, '"just a string"', 'utf8');
        expect(contract.readDispatchRecord(identity, { directory }).status).toBe('CORRUPT');
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    });

    it('writes the caller-supplied monotonic start base verbatim', async () => {
      const contract = await requireContract();
      const directory = mkdtempSync(path.join(os.tmpdir(), 'telemetry-monotonic-'));
      try {
        const identity = { plan_id: 'opentelemetry-slice-telemetry', slice_id: 'OT-S1-4' };
        const written = contract.writeDispatchRecord(
          {
            ...identity,
            context_tokens_delivered: 18432,
            dispatch_seq: 7,
            started_monotonic_ns: '0x2f',
          },
          { directory },
        );
        const record = JSON.parse(readFileSync(written, 'utf8'));
        expect(record.started_monotonic_ns).toBe('0x2f');
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    });

    it('refuses a dispatch record that is not an object', async () => {
      const contract = await requireContract();
      const directory = mkdtempSync(path.join(os.tmpdir(), 'telemetry-invalid-'));
      try {
        expect(() => contract.writeDispatchRecord('not a record', { directory })).toThrow(TypeError);
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    });

    it('refuses a dispatch record that declares no plan_id and slice_id', async () => {
      const contract = await requireContract();
      const directory = mkdtempSync(path.join(os.tmpdir(), 'telemetry-invalid-'));
      try {
        expect(() => contract.writeDispatchRecord({ plan_id: 'p' }, { directory })).toThrow(
          TypeError,
        );
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    });
  });

  describe('AC-003-B - the defensive lanes of the declared API', () => {
    it('emits a value that carries a control character as a double-quoted scalar', async () => {
      const contract = await requireContract();
      const controlCharacter = String.fromCharCode(7);
      const label = `a${controlCharacter}b`;
      const block = { ...canonicalBlock(), label };
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).toContain(`label: ${JSON.stringify(label)}`);
    });

    it('emits a non-finite number as null', async () => {
      const contract = await requireContract();
      const block = { ...canonicalBlock(), ratio: Number.POSITIVE_INFINITY };
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).toContain('  ratio: null');
    });

    it('emits a bigint as its decimal string', async () => {
      const contract = await requireContract();
      const block = { ...canonicalBlock(), retries: 3n };
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).toContain('  retries: 3');
    });

    it('emits a block that is not an object as an empty ledger body', async () => {
      const contract = await requireContract();
      expect(contract.emitLedgerBlock('not a block')).toBe('```yaml\ntelemetry_ledger:\n```\n');
    });

    it('scans a null and an undefined text as clean', async () => {
      const contract = await requireContract();
      expect(contract.scanPersistedText(null)).toEqual([]);
      expect(contract.scanPersistedText(undefined)).toEqual([]);
    });

    it('scans a non-string text as its string form', async () => {
      const contract = await requireContract();
      expect(contract.scanPersistedText(42)).toEqual([]);
    });

    it('reports a usage counter that is present but invalid', async () => {
      const contract = await requireContract();
      const block = canonicalBlock();
      block.usage.token_usage = { available: true };
      const verdict = contract.validateLedgerBlock(block);
      expect(verdict.errors.some((error) => error.startsWith('usage.token_usage: '))).toBe(true);
    });

    it('reports a usage mapping that is not an object', async () => {
      const contract = await requireContract();
      const verdict = contract.validateLedgerBlock({ ...canonicalBlock(), usage: 'not-a-mapping' });
      expect(verdict.errors.some((error) => error.includes('usage.token_usage'))).toBe(true);
      expect(verdict.errors.some((error) => error.includes('usage.tool_executions'))).toBe(true);
    });

    it('demands a ledger for a request that carries no pre-existing slice list', async () => {
      const contract = await requireContract();
      expect(contract.demandsLedger()).toBe(true);
      expect(contract.demandsLedger({ slice_id: 'OT-S1-1' })).toBe(true);
    });

    it('demands a ledger for a null request', async () => {
      const contract = await requireContract();
      expect(contract.demandsLedger(null)).toBe(true);
    });

    it('resolves the dispatch-record path of the default directory', async () => {
      const contract = await requireContract();
      expect(contract.dispatchRecordPath()).toBe(
        path.join(REPO_ROOT, 'logs', 'slice-ledger', 'dispatch', 'unknown__unknown.json'),
      );
    });

    it('writes into the default dispatch directory and removes its artefact again', async () => {
      const contract = await requireContract();
      const identity = {
        plan_id: 'opentelemetry-slice-telemetry',
        slice_id: 'OT-S1-2-default-directory-probe',
      };
      try {
        const written = contract.writeDispatchRecord({
          ...identity,
          context_tokens_delivered: 18432,
          dispatch_seq: 7,
        });
        expect(written).toBe(contract.dispatchRecordPath(identity));
        expect(contract.readDispatchRecord(identity).status).toBe('OK');
      } finally {
        contract.deleteDispatchRecord(identity);
      }
      expect(contract.readDispatchRecord(identity).status).toBe('MISSING');
    });

    it('nulls the counters a caller did not supply', async () => {
      const contract = await requireContract();
      const directory = mkdtempSync(path.join(os.tmpdir(), 'telemetry-nullable-'));
      try {
        const written = contract.writeDispatchRecord(
          { plan_id: 'opentelemetry-slice-telemetry', slice_id: 'OT-S1-2' },
          { directory },
        );
        const record = JSON.parse(readFileSync(written, 'utf8'));
        expect(record.context_tokens_delivered).toBeNull();
        expect(record.dispatch_seq).toBeNull();
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    });

    it('reports no dispatch record for the default identity', async () => {
      const contract = await requireContract();
      expect(contract.readDispatchRecord().status).toBe('MISSING');
      expect(contract.deleteDispatchRecord()).toBe(false);
    });
  });

  describe('AC-001-A - the layer independence', () => {
    it('imports no OpenTelemetry package and nothing from src/observability', async () => {
      await requireContract();
      const specifiers = collectImportSpecifiers(readModuleSource());
      expect(specifiers.length).toBeGreaterThan(0);
      for (const specifier of specifiers) {
        expect(specifier).not.toMatch(/^@opentelemetry\//);
        expect(specifier).not.toMatch(/observability/);
      }
    });

    it('keeps the OpenTelemetry layer out of the module code entirely', async () => {
      await requireContract();
      const code = stripComments(readModuleSource());
      expect(code).not.toMatch(/@opentelemetry\//);
      expect(code).not.toMatch(/src\/observability/);
    });
  });
});
