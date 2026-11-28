/**
 * @module telemetry-contract
 * @description The DECLARATION of the repo-owned per-slice telemetry ledger
 * contract (schema v1) of `plans/opentelemetry-slice-telemetry.plans.md`.
 *
 * Slice: OT-S1-2 of `plans/opentelemetry-slice-telemetry.plans.md`.
 * Acceptance criteria implemented by this module:
 *   - AC-003-B (the declaration)
 *   - AC-001-A (the layer independence)
 *
 * WHAT this module is: the single place where the persisted-artifact rules are
 * DECLARED - the schema version, the required key set, the forbidden key set,
 * the forbidden value patterns, the availability semantics, the two layer state
 * enums and the deterministic emitter - plus the ONE scanner that applies the
 * forbidden sets to any persisted text.
 *
 * WHY it exists: the tracker pins the forbidden sets in TWO languages (this
 * module and, later, the TypeScript layer-A policy) and pins that the privacy
 * sweep CLI, the ledger gate and the tests must all reuse ONE implementation of
 * those rules. A second implementation is a drift bug, so the rules live here
 * once and every other component calls them.
 *
 * WHEN it is used: at dispatch time (the transient dispatch record), at DONE
 * time (the ledger block), by the privacy sweep and by the DONE-closure gate.
 *
 * LAYER INDEPENDENCE (AC-001-A): this module imports NOTHING from
 * `@opentelemetry/*` and NOTHING from `src/observability`. The repo-owned
 * layer-B measurement must not depend on the optional OpenTelemetry layer-A
 * packages; the only way to keep that true is to make the dependency
 * impossible rather than merely discouraged.
 *
 * @example
 * import { scanPersistedText, emitLedgerBlock } from './telemetry-contract.mjs';
 *
 * const block = emitLedgerBlock({ schema_version: 1, plan_id: 'p', slice_id: 'S-1' });
 * const violations = scanPersistedText(block);
 */

import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');

/**
 * The schema version of the persisted ledger block. It is an INTEGER (R-1): a
 * block whose `schema_version` this tracker does not know is reported as
 * `STALE` and repaired, never silently accepted.
 *
 * @type {number}
 */
export const SCHEMA_VERSION = 1;

/**
 * The root key of the persisted ledger block. The companion log carries ONE
 * fenced `yaml` block per completed slice and every one of them is rooted at
 * this key, so the DONE writer can find a block by its shape alone.
 *
 * @type {string}
 */
export const LEDGER_ROOT_KEY = 'telemetry_ledger';

/**
 * The language tag of the fence that wraps a persisted ledger block.
 *
 * @type {string}
 */
export const LEDGER_FENCE_LANGUAGE = 'yaml';

/**
 * The keys a persisted ledger block MUST declare, in the canonical order of
 * the schema-v1 example of the tracker. The order is the FIXED key order of the
 * emitter (R-2), so a block written twice from the same facts is
 * byte-identical.
 *
 * @type {readonly string[]}
 */
export const REQUIRED_LEDGER_KEYS = Object.freeze([
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

/**
 * The fields the ledger is KEYED on (R-1): the plan id plus the slice id. There
 * is exactly one ledger per newly completed slice, so the key is the pair.
 *
 * @type {readonly string[]}
 */
export const LEDGER_KEY_FIELDS = Object.freeze(['plan_id', 'slice_id']);

/**
 * The forbidden KEY set (R-3): every key that would carry an instant, a clock
 * time, a date, a modification marker or a PII-ish identifier. A persisted
 * field that would carry an instant is REMOVED, never exempted.
 *
 * The list is shared with the TypeScript layer-A policy; a drift guard
 * compares the two lists and fails on any difference.
 *
 * @type {readonly string[]}
 */
export const FORBIDDEN_KEYS = Object.freeze([
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
]);

/**
 * The forbidden VALUE patterns (R-3): the shapes a persisted value must never
 * take. They are plain regular expressions so that the drift guard against the
 * TypeScript layer-A policy can compare `source` and `flags` directly.
 *
 * The patterns are deliberately NOT global, so `RegExp.prototype.test` and
 * `RegExp.prototype.exec` are stateless and a caller can reuse them safely.
 *
 * @type {readonly RegExp[]}
 */
export const FORBIDDEN_VALUE_PATTERNS = Object.freeze([
  // An ISO-8601 datetime, with or without a fraction and with or without an offset.
  /\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?/,
  // A calendar date on its own.
  /\b\d{4}-\d{2}-\d{2}\b/,
  // An epoch-millisecond literal (13 decimal digits).
  /\b\d{13}\b/,
  // An absolute Windows path, drive letter included.
  /\b[A-Za-z]:[\\/][^\s'"`|]*/,
  // An absolute POSIX user path.
  /\/(?:Users|home)\/[^\s'"`|]*/,
  // Any other absolute POSIX path shape.
  /(?<![\w.])\/(?:[\w.@+-]+\/)+[\w.@+-]+/,
  // An e-mail address.
  /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/,
]);

/**
 * The five SAFE tool-execution buckets (R-5). The breakdown is a COUNT per
 * bucket and never an argument list, so no tool name, no argument, no result
 * and no path can reach the persisted artefact.
 *
 * @type {readonly string[]}
 */
export const TOOL_EXECUTION_BUCKETS = Object.freeze(['read', 'edit', 'search', 'exec', 'other']);

/**
 * The repo-relative path of the layer-B DONE writer. The path is DECLARED here
 * rather than probed on disk, because the writer is owned by a later slice and
 * a lane that demanded the file to exist would keep this step permanently red.
 *
 * @type {string}
 */
export const LEDGER_WRITER_PATH = 'scripts/telemetry/write-slice-ledger.mjs';

/**
 * The availability enum (R-4). A usage counter is either AVAILABLE - it
 * carries a `source`, a `measures` string and real numbers - or UNAVAILABLE,
 * in which case it carries a reason and NULL numeric fields, never zero.
 *
 * @type {Readonly<{ AVAILABLE: true, UNAVAILABLE: false }>}
 */
export const AVAILABILITY = Object.freeze({ AVAILABLE: true, UNAVAILABLE: false });

/**
 * The layer-A (the OpenTelemetry layer) reporting states. Layer A is optional
 * and fail-open, so it has its own vocabulary: the repo-owned measurement of
 * layer B never depends on it, which is why `otel_state: 'DISABLED'` can sit
 * next to a complete `measurement_state: 'MEASURED'` block.
 *
 * @type {readonly string[]}
 */
export const LAYER_A_REPORTING_STATES = Object.freeze([
  'ENABLED',
  'DISABLED',
  'STALE',
  'CORRUPT',
  'NO-DATA',
]);

/**
 * The layer-B (the repo-owned measurement) states. A missing, interrupted or
 * corrupt dispatch record yields `UNAVAILABLE` with null counters and a
 * reason; `CORRUPT` is reported when a persisted artefact cannot be trusted.
 *
 * @type {readonly string[]}
 */
export const LAYER_B_MEASUREMENT_STATES = Object.freeze(['MEASURED', 'UNAVAILABLE', 'CORRUPT']);

/**
 * The repo-relative directory of the TRANSIENT dispatch record (R-8). The
 * record holds the open-slice pointer and is deleted by the layer-B DONE
 * writer when the ledger block is written.
 *
 * @type {string}
 */
export const DISPATCH_RECORD_DIR = 'logs/slice-ledger/dispatch';

/**
 * The exact fields a transient dispatch record may carry: the slice identity,
 * the delivered context-token count, the dispatch sequence number and the
 * monotonic start base. Nothing else may be written, so the helper that writes
 * the record builds the payload from this list alone.
 *
 * @type {readonly string[]}
 */
export const DISPATCH_RECORD_FIELDS = Object.freeze([
  'plan_id',
  'slice_id',
  'context_tokens_delivered',
  'dispatch_seq',
  'started_monotonic_ns',
]);

/**
 * The repo-relative path of the tool-execution observer (AC-011). The manifest
 * registration is checked against this path, so the check cannot drift into
 * "some hook is registered".
 *
 * @type {string}
 */
export const OBSERVER_RELPATH = 'scripts/agent-customization/hooks/slice-usage-hook.mjs';

/**
 * The two post event keys the observer MUST be registered under: a completed
 * tool execution and a failed one. Both registrations carry the same command
 * and NO matcher, so the observer sees every tool execution.
 *
 * @type {readonly string[]}
 */
export const OBSERVER_EVENT_KEYS = Object.freeze(['PostToolUse', 'PostToolUseFailure']);

const ROOT_KEY_ORDER = REQUIRED_LEDGER_KEYS;

const NESTED_KEY_ORDER = Object.freeze({
  runtime: Object.freeze(['node_version', 'heap_used_bytes', 'rss_bytes']),
  usage: Object.freeze(['token_usage', 'tool_executions']),
  'usage.token_usage': Object.freeze([
    'available',
    'source',
    'measures',
    'context_tokens_delivered',
    'prompt_tokens',
    'completion_tokens',
    'total_tokens',
    'unavailable_reason',
  ]),
  'usage.tool_executions': Object.freeze([
    'available',
    'source',
    'measures',
    'scope',
    'total',
    'completed_executions',
    'failed_executions',
    'sessions_observed',
    'breakdown',
  ]),
  validation: Object.freeze(['total', 'passed', 'failed', 'skipped', 'results']),
  'validation.results[]': Object.freeze(['name', 'status', 'duration_ms']),
});

const FORBIDDEN_KEY_INDEX = new Set(FORBIDDEN_KEYS.map(normalizeKey));

const YAML_KEY_PATTERN = /^[ \t]*(?:[-*][ \t]+)?["']?([A-Za-z_][A-Za-z0-9_-]*)["']?[ \t]*:/;

const QUOTED_KEY_PATTERN = /["']([A-Za-z_][A-Za-z0-9_-]*)["'][ \t]*:/g;

/**
 * Normalises a key so that `startTime`, `start_time` and `START-TIME` are one
 * and the same forbidden key. The comparison is exact on the normalised form,
 * so `duration_ms` and `runtime` stay legal.
 *
 * @param {unknown} key The key to normalise.
 * @returns {string} The normalised key.
 */
function normalizeKey(key) {
  return String(key).toLowerCase().replace(/[_-]/g, '');
}

/**
 * Tells whether a key is forbidden (R-3).
 *
 * @param {unknown} key The key to test.
 * @returns {boolean} True when the key is in the forbidden set.
 */
function isForbiddenKey(key) {
  return FORBIDDEN_KEY_INDEX.has(normalizeKey(key));
}

/**
 * Formats a string as a YAML scalar. A plain single-quoted scalar keeps the
 * value a string even when it looks like a number or a boolean, and a value
 * that carries a control character falls back to a double-quoted scalar.
 *
 * @param {string} value The value to format.
 * @returns {string} The formatted scalar.
 */
function formatString(value) {
  if (/[\u0000-\u001f\u007f]/.test(value)) return JSON.stringify(value);
  return `'${value.replace(/'/g, "''")}'`;
}

/**
 * Formats any scalar as a YAML scalar.
 *
 * @param {unknown} value The value to format.
 * @returns {string} The formatted scalar.
 */
function formatScalar(value) {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : 'null';
  if (typeof value === 'bigint') return String(value);
  if (typeof value === 'string') return formatString(value);
  return 'null';
}

/**
 * Tells whether a value is a REAL plain object (and not an array, and not an
 * exotic object such as a `Date`, a `Map` or a class instance).
 *
 * WHY the prototype is tested: an exotic object passes a `typeof`-object test
 * but is NOT a mapping, so the emitter would render it as an EMPTY MAPPING and
 * silently LOSE the value (and, for a class instance, persist its own
 * enumerable properties). Restricting the predicate to a real plain object
 * lets every exotic value take the scalar path, where it is emitted as `null`
 * instead of being dropped or leaked.
 *
 * @param {unknown} value The value to test.
 * @returns {boolean} True for a real plain object.
 */
function isPlainObject(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/**
 * Returns the declared key order of the object at a path, or null when the
 * path has no declared order (an unknown object is emitted with its keys
 * sorted alphabetically, which is deterministic too).
 *
 * @param {string} objectPath The dotted path of the object.
 * @returns {readonly string[] | null} The declared key order, or null.
 */
function declaredKeyOrder(objectPath) {
  if (objectPath === '') return ROOT_KEY_ORDER;
  return NESTED_KEY_ORDER[objectPath] ?? null;
}

/**
 * Returns the keys of an object in their emission order: the declared order
 * first, then every remaining key sorted alphabetically.
 *
 * @param {Record<string, unknown>} source The object to read.
 * @param {string} objectPath The dotted path of the object.
 * @returns {string[]} The ordered keys.
 */
function orderedKeys(source, objectPath) {
  const declared = declaredKeyOrder(objectPath);
  const own = Object.keys(source);
  if (!declared) return own.toSorted();
  const known = declared.filter((key) => key in source);
  const extra = own.filter((key) => !declared.includes(key)).toSorted();
  return [...known, ...extra];
}

/**
 * Renders one key of an object.
 *
 * @param {string} key The key to render.
 * @param {unknown} value The value of the key.
 * @param {string} indent The indentation of the key line.
 * @param {string} childPath The dotted path of the child.
 * @param {string[]} lines The accumulator of rendered lines.
 */
function renderEntry(key, value, indent, childPath, lines) {
  if (isPlainObject(value)) {
    lines.push(`${indent}${key}:`);
    renderObject(value, `${indent}  `, childPath, lines);
    return;
  }
  if (Array.isArray(value)) {
    lines.push(`${indent}${key}:`);
    renderArray(value, `${indent}  `, childPath, lines);
    return;
  }
  lines.push(`${indent}${key}: ${formatScalar(value)}`);
}

/**
 * Renders an object as YAML lines. A forbidden key is REMOVED rather than
 * exempted, so an injected timestamp never reaches the persisted artefact.
 *
 * @param {Record<string, unknown>} source The object to render.
 * @param {string} indent The indentation of the object's own lines.
 * @param {string} objectPath The dotted path of the object.
 * @param {string[]} lines The accumulator of rendered lines.
 */
function renderObject(source, indent, objectPath, lines) {
  for (const key of orderedKeys(source, objectPath)) {
    if (isForbiddenKey(key)) continue;
    const childPath = objectPath === '' ? key : `${objectPath}.${key}`;
    renderEntry(key, source[key], indent, childPath, lines);
  }
}

/**
 * Renders an array as YAML lines. The first key of an object item rides on the
 * `- ` marker and every following key is aligned under it.
 *
 * @param {unknown[]} items The array to render.
 * @param {string} indent The indentation of the items.
 * @param {string} objectPath The dotted path of the array.
 * @param {string[]} lines The accumulator of rendered lines.
 */
function renderArray(items, indent, objectPath, lines) {
  const itemPath = `${objectPath}[]`;
  for (const item of items) {
    if (!isPlainObject(item)) {
      lines.push(`${indent}- ${formatScalar(item)}`);
      continue;
    }
    const keys = orderedKeys(item, itemPath).filter((key) => !isForbiddenKey(key));
    if (keys.length === 0) {
      lines.push(`${indent}- {}`);
      continue;
    }
    const [first, ...rest] = keys;
    const firstValue = item[first];
    if (isPlainObject(firstValue) || Array.isArray(firstValue)) {
      lines.push(`${indent}- ${first}:`);
      const childPath = `${itemPath}.${first}`;
      if (isPlainObject(firstValue)) {
        renderObject(firstValue, `${indent}    `, childPath, lines);
      } else {
        renderArray(firstValue, `${indent}    `, childPath, lines);
      }
    } else {
      lines.push(`${indent}- ${first}: ${formatScalar(firstValue)}`);
    }
    for (const key of rest) {
      const childPath = `${itemPath}.${key}`;
      renderEntry(key, item[key], `${indent}  `, childPath, lines);
    }
  }
}

/**
 * Renders the body of a ledger block (the fenced block without its fences).
 *
 * @param {Record<string, unknown>} block The ledger block.
 * @returns {string} The rendered YAML body.
 */
function renderLedgerBody(block) {
  const source = isPlainObject(block) ? block : {};
  const lines = [`${LEDGER_ROOT_KEY}:`];
  renderObject(source, '  ', '', lines);
  return lines.join('\n');
}

/**
 * The ONE scanner of the forbidden key set and the forbidden value patterns
 * (AC-003-B). Every component that has to decide whether a persisted artefact
 * is safe - the privacy sweep CLI, the DONE-closure gate, the ledger scanner
 * and the tests - calls THIS function, so there is exactly one implementation
 * of the rules and therefore no drift.
 *
 * The scanner reports the EXACT offending key or pattern, with the line it was
 * found on, so a report can name the offending line instead of only counting
 * it. One offending key is reported EXACTLY ONCE per line: the YAML-key pass
 * and the quoted-key pass are mutually exclusive and the reports are
 * de-duplicated by (key, line).
 *
 * @param {string} text The persisted text to scan.
 * @returns {Array<{ kind: string, key?: string, pattern?: string, flags?: string, sample?: string, line: number }>} The
 * violations, in the order they were found. An empty array means the text is
 * clean.
 * @example
 * import { scanPersistedText } from './telemetry-contract.mjs';
 *
 * scanPersistedText('  timestamp: 1764147453321');
 * // => [{ kind: 'forbidden-key', key: 'timestamp', line: 1, ... }]
 */
export function scanPersistedText(text) {
  const violations = [];
  const content = typeof text === 'string' ? text : text === null || text === undefined ? '' : String(text);
  if (content.length === 0) return violations;
  const lines = content.split(/\r?\n/);
  lines.forEach((line, index) => {
    const lineNumber = index + 1;
    const yamlKey = YAML_KEY_PATTERN.exec(line);
    if (yamlKey && isForbiddenKey(yamlKey[1])) {
      violations.push({
        kind: 'forbidden-key',
        key: yamlKey[1],
        line: lineNumber,
        detail: `the persisted text carries the forbidden key \`${yamlKey[1]}\``,
      });
    }
    // The quoted-key pass scans ONLY the region the YAML-key pass did not
    // already cover, so the two passes are MUTUALLY EXCLUSIVE and a quoted
    // forbidden key is reported EXACTLY ONCE instead of twice.
    const quotedRegion = yamlKey ? line.slice(yamlKey.index + yamlKey[0].length) : line;
    for (const match of quotedRegion.matchAll(QUOTED_KEY_PATTERN)) {
      if (!isForbiddenKey(match[1])) continue;
      violations.push({
        kind: 'forbidden-key',
        key: match[1],
        line: lineNumber,
        detail: `the persisted text carries the forbidden key \`${match[1]}\``,
      });
    }
    for (const pattern of FORBIDDEN_VALUE_PATTERNS) {
      const match = pattern.exec(line);
      if (!match) continue;
      violations.push({
        kind: 'forbidden-value',
        pattern: pattern.source,
        flags: pattern.flags,
        sample: match[0],
        line: lineNumber,
        detail: `the persisted text carries the forbidden value \`${match[0]}\``,
      });
    }
  });
  return dedupeForbiddenKeys(violations);
}

/**
 * De-duplicates the forbidden-key reports of one line, so the same offending
 * key is reported EXACTLY ONCE per line.
 *
 * The two scanner passes are already mutually exclusive, so this is the SAFETY
 * NET for the pathological case of one line that spells the same forbidden
 * key twice: the report still names the offending key, but it names it once.
 *
 * The fingerprint is built from the NORMALISED key, not from the raw
 * spelling, because `normalizeKey` declares `startTime`, `start_time` and
 * `START-TIME` to be ONE and the SAME forbidden key. Fingerprinting the raw
 * spelling would let one line report one canonical forbidden key twice - once
 * per spelling - which is the very double report this function exists to
 * prevent. The forbidden-VALUE reports are never de-duplicated, because two
 * different forbidden values on one line are two different findings.
 *
 * @param {Array<{ kind: string, key?: string, pattern?: string, flags?: string, sample?: string, line: number }>} violations The
 * collected violations.
 * @returns {Array<{ kind: string, key?: string, pattern?: string, flags?: string, sample?: string, line: number }>} The
 * de-duplicated violations, in the order they were found.
 */
function dedupeForbiddenKeys(violations) {
  const reported = new Set();
  return violations.filter((violation) => {
    if (violation.kind !== 'forbidden-key') return true;
    const fingerprint = `${violation.line}:${normalizeKey(violation.key)}`;
    if (reported.has(fingerprint)) return false;
    reported.add(fingerprint);
    return true;
  });
}

/**
 * The DETERMINISTIC emitter (R-2). The output is byte-identical for identical
 * input: the keys are emitted in a FIXED order, the breakdown keys are emitted
 * sorted, no timestamp, no random id and no host-dependent ordering reaches
 * the text, and a forbidden key is removed instead of being exempted.
 *
 * The emitted text is the fenced `yaml` block the companion log carries, so
 * the DONE writer can write the return value verbatim.
 *
 * @param {Record<string, unknown>} block The ledger block to emit.
 * @returns {string} The fenced YAML block.
 * @example
 * import { emitLedgerBlock } from './telemetry-contract.mjs';
 *
 * emitLedgerBlock({ schema_version: 1, plan_id: 'p', slice_id: 'S-1' });
 * // => '```yaml\ntelemetry_ledger:\n  schema_version: 1\n  ...\n```\n'
 */
export function emitLedgerBlock(block) {
  const body = renderLedgerBody(block);
  return `\`\`\`${LEDGER_FENCE_LANGUAGE}\n${body}\n\`\`\`\n`;
}

/**
 * Validates a usage counter against the availability semantics (R-4).
 *
 * An AVAILABLE counter carries a non-empty `source` and a non-empty `measures`
 * string that states exactly what it measures. An UNAVAILABLE counter carries a
 * reason - spelled `reason` or `unavailable_reason` - and NULL numeric fields:
 * an unavailable counter is null, NEVER zero, because a zero is a measurement.
 *
 * The verdict is a value, never a throw, so a caller can report the reasons
 * instead of crashing on a malformed counter.
 *
 * @param {Record<string, unknown>} counter The counter to validate.
 * @returns {{ valid: boolean, errors: string[] }} The verdict and its reasons.
 * @example
 * import { validateUsageCounter } from './telemetry-contract.mjs';
 *
 * validateUsageCounter({ available: false, source: 's', measures: 'm', total: null, reason: 'why' });
 * // => { valid: true, errors: [] }
 */
export function validateUsageCounter(counter) {
  const errors = [];
  if (!isPlainObject(counter)) {
    return { valid: false, errors: ['the usage counter is not an object'] };
  }
  const available = counter.available;
  if (available !== true && available !== false) {
    errors.push('the usage counter declares no boolean `available` flag');
  }
  if (!isNonEmptyString(counter.source)) {
    errors.push('the usage counter carries no non-empty `source`');
  }
  if (!isNonEmptyString(counter.measures)) {
    errors.push('the usage counter carries no non-empty `measures` string');
  }
  if (available === false) {
    const reason = counter.reason ?? counter.unavailable_reason;
    if (!isNonEmptyString(reason)) {
      errors.push('an unavailable usage counter carries no `reason`');
    }
    for (const [key, value] of Object.entries(counter)) {
      if (key === 'available' || key === 'source' || key === 'measures' || key === 'scope') {
        continue;
      }
      if (key === 'reason' || key === 'unavailable_reason') continue;
      if (value === null || value === undefined) continue;
      errors.push(
        `an unavailable usage counter carries \`${key}\` instead of null; an unavailable counter is null, never zero`,
      );
    }
  }
  return { valid: errors.length === 0, errors };
}

/**
 * Validates a whole ledger block against the schema-v1 contract.
 *
 * The validator checks the required key set, the schema version, the two state
 * enums, the forbidden keys and the availability of every usage counter, so
 * the DONE writer and the DONE-closure gate share ONE notion of a valid
 * ledger.
 *
 * @param {Record<string, unknown>} block The ledger block to validate.
 * @returns {{ valid: boolean, errors: string[] }} The verdict and its reasons.
 * @example
 * import { validateLedgerBlock } from './telemetry-contract.mjs';
 *
 * validateLedgerBlock({ schema_version: 1, plan_id: 'p', slice_id: 'S-1' }).valid;
 * // => false (the required keys are missing)
 */
export function validateLedgerBlock(block) {
  const errors = [];
  if (!isPlainObject(block)) {
    return { valid: false, errors: ['the ledger block is not an object'] };
  }
  for (const key of REQUIRED_LEDGER_KEYS) {
    if (!(key in block)) errors.push(`the ledger block declares no \`${key}\``);
  }
  if (block.schema_version !== SCHEMA_VERSION) {
    errors.push(`the ledger block declares schema_version \`${String(block.schema_version)}\`, not \`${SCHEMA_VERSION}\``);
  }
  for (const key of LEDGER_KEY_FIELDS) {
    if (!isNonEmptyString(block[key])) errors.push(`the ledger block declares no non-empty \`${key}\``);
  }
  for (const key of collectKeys(block)) {
    if (isForbiddenKey(key)) errors.push(`the ledger block carries the forbidden key \`${key}\``);
  }
  if (typeof block.measurement_state === 'string' && !LAYER_B_MEASUREMENT_STATES.includes(block.measurement_state)) {
    errors.push(`the ledger block declares the unknown measurement_state \`${block.measurement_state}\``);
  }
  if (typeof block.otel_state === 'string' && !LAYER_A_REPORTING_STATES.includes(block.otel_state)) {
    errors.push(`the ledger block declares the unknown otel_state \`${block.otel_state}\``);
  }
  const usage = isPlainObject(block.usage) ? block.usage : {};
  for (const name of ['token_usage', 'tool_executions']) {
    if (!(name in usage)) {
      errors.push(`the ledger block declares no \`usage.${name}\` counter`);
      continue;
    }
    const verdict = validateUsageCounter(usage[name]);
    for (const error of verdict.errors) errors.push(`usage.${name}: ${error}`);
  }
  return { valid: errors.length === 0, errors };
}

/**
 * Decides whether a slice is DEMANDED a ledger (R-7, the forward-only rule).
 *
 * A slice that already carried `[DONE]` before the ledger contract was
 * activated is never demanded a ledger and is never backfilled; every slice
 * that completed at or after the activation marker is.
 *
 * @param {{ slice_id?: string, pre_existing_done_slice_ids?: Iterable<string> }} request The demand request.
 * @returns {boolean} True when the slice is demanded a ledger.
 * @example
 * import { demandsLedger } from './telemetry-contract.mjs';
 *
 * demandsLedger({ slice_id: 'OT-S1-2', pre_existing_done_slice_ids: ['OT-S1-1'] });
 * // => true
 */
export function demandsLedger(request = {}) {
  const { slice_id: sliceId, pre_existing_done_slice_ids: preExisting } = request ?? {};
  const exempt = new Set(
    preExisting === null || preExisting === undefined ? [] : [...preExisting],
  );
  return !exempt.has(sliceId);
}

/**
 * The absolute path of the transient dispatch record of a slice.
 *
 * @param {{ plan_id?: string, slice_id?: string }} identity The slice identity.
 * @param {{ directory?: string }} [options] An optional directory override.
 * @returns {string} The absolute path of the dispatch record.
 * @example
 * import { dispatchRecordPath } from './telemetry-contract.mjs';
 *
 * dispatchRecordPath({ plan_id: 'p', slice_id: 'S-1' });
 * // => 'C:\\...\\logs\\slice-ledger\\dispatch\\p__S-1.json'
 */
export function dispatchRecordPath(identity = {}, options = {}) {
  const planId = sanitizeIdentityPart(identity.plan_id);
  const sliceId = sanitizeIdentityPart(identity.slice_id);
  const directory = options.directory
    ? path.resolve(options.directory)
    : path.resolve(REPO_ROOT, DISPATCH_RECORD_DIR);
  return path.join(directory, `${planId}__${sliceId}.json`);
}

/**
 * Writes the TRANSIENT dispatch record of a slice (R-8).
 *
 * The record is the open-slice pointer: it carries the slice identity, the
 * delivered context-token count, the dispatch sequence number and the
 * monotonic start base, and NOTHING else - the payload is built from
 * `DISPATCH_RECORD_FIELDS` alone, so no tool argument, no tool result, no
 * session id and no instant can leak into it.
 *
 * The monotonic start base is stored as a HEX string on purpose: it is a
 * nanosecond counter, not a wall-clock instant, and a decimal encoding of a
 * long-running host could look exactly like the epoch-millisecond literal the
 * forbidden value patterns forbid.
 *
 * @param {Record<string, unknown>} record The record to write.
 * @param {{ directory?: string }} [options] An optional directory override.
 * @returns {string} The absolute path of the written record.
 * @example
 * import { writeDispatchRecord } from './telemetry-contract.mjs';
 *
 * writeDispatchRecord({ plan_id: 'p', slice_id: 'S-1', context_tokens_delivered: 1200, dispatch_seq: 7 });
 * // => 'C:\\...\\logs\\slice-ledger\\dispatch\\p__S-1.json'
 */
export function writeDispatchRecord(record, options = {}) {
  if (!isPlainObject(record)) {
    throw new TypeError('the dispatch record is not an object');
  }
  if (!isNonEmptyString(record.plan_id) || !isNonEmptyString(record.slice_id)) {
    throw new TypeError('the dispatch record declares no `plan_id` and `slice_id`');
  }
  const payload = {};
  for (const field of DISPATCH_RECORD_FIELDS) {
    if (field === 'started_monotonic_ns') {
      payload[field] =
        typeof record.started_monotonic_ns === 'string' && record.started_monotonic_ns.length > 0
          ? record.started_monotonic_ns
          : `0x${process.hrtime.bigint().toString(16)}`;
      continue;
    }
    payload[field] = record[field] ?? null;
  }
  const target = dispatchRecordPath(record, options);
  const directory = path.dirname(target);
  mkdirSync(directory, { recursive: true });
  const temporary = `${target}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  renameSync(temporary, target);
  return target;
}

/**
 * Reads the transient dispatch record of a slice (R-8).
 *
 * The reader tells the three states apart, because the tracker treats them
 * differently: a MISSING record and a CORRUPT record both yield an
 * `UNAVAILABLE` measurement, but a corrupt record is reported as such.
 *
 * @param {{ plan_id?: string, slice_id?: string }} identity The slice identity.
 * @param {{ directory?: string }} [options] An optional directory override.
 * @returns {{ status: 'OK' | 'MISSING' | 'CORRUPT', record: Record<string, unknown> | null, path: string }} The read result.
 * @example
 * import { readDispatchRecord } from './telemetry-contract.mjs';
 *
 * readDispatchRecord({ plan_id: 'p', slice_id: 'S-1' }).status;
 * // => 'MISSING'
 */
export function readDispatchRecord(identity = {}, options = {}) {
  const target = dispatchRecordPath(identity, options);
  if (!existsSync(target)) {
    return { status: 'MISSING', record: null, path: target };
  }
  try {
    const parsed = JSON.parse(readFileSync(target, 'utf8'));
    if (!isPlainObject(parsed)) {
      return { status: 'CORRUPT', record: null, path: target };
    }
    return { status: 'OK', record: parsed, path: target };
  } catch {
    return { status: 'CORRUPT', record: null, path: target };
  }
}

/**
 * Deletes the transient dispatch record of a slice (R-8).
 *
 * The deletion is performed by the layer-B DONE writer when the ledger block
 * is written; no layer-A module creates, reads, measures or deletes a layer-B
 * transient artefact.
 *
 * @param {{ plan_id?: string, slice_id?: string }} identity The slice identity.
 * @param {{ directory?: string }} [options] An optional directory override.
 * @returns {boolean} True when a record was removed.
 * @example
 * import { deleteDispatchRecord } from './telemetry-contract.mjs';
 *
 * deleteDispatchRecord({ plan_id: 'p', slice_id: 'S-1' });
 * // => false when there was nothing to delete
 */
export function deleteDispatchRecord(identity = {}, options = {}) {
  const target = dispatchRecordPath(identity, options);
  if (!existsSync(target)) return false;
  rmSync(target, { force: true });
  return true;
}

/**
 * Inspects a hook manifest and reports whether the tool-execution observer is
 * registered the way the contract demands (AC-011).
 *
 * The verdict is ONE implementation on purpose: the enablement CLI reports it
 * to the operator and the observer's own home asserts it, so a manifest that
 * lost the failure-event registration cannot pass a lane that only asks
 * "is some hook registered".
 *
 * @param {unknown} manifest The parsed manifest (a `hooks` object or a whole
 *   manifest document that carries one).
 * @param {{ scriptPath?: string }} [options] An optional script path override.
 * @returns {{
 *   registered: boolean,
 *   complete: boolean,
 *   registered_under: string[],
 *   missing: string[],
 *   same_command: boolean,
 *   matcher_free: boolean,
 *   script_path: string,
 *   commands: Record<string, string[]>
 * }} The registration verdict.
 * @example
 * import { inspectObserverRegistration } from './telemetry-contract.mjs';
 *
 * const verdict = inspectObserverRegistration(manifest);
 * // => { registered: true, complete: true, missing: [], ... }
 */
export function inspectObserverRegistration(manifest, options = {}) {
  const scriptPath = isNonEmptyString(options.scriptPath)
    ? options.scriptPath.trim()
    : OBSERVER_RELPATH;
  const hooks = isPlainObject(manifest)
    ? (isPlainObject(manifest.hooks) ? manifest.hooks : manifest)
    : {};
  const commands = {};
  const registeredUnder = [];
  const missing = [];
  for (const key of OBSERVER_EVENT_KEYS) {
    const entries = Array.isArray(hooks[key]) ? hooks[key] : [];
    const matched = [];
    for (const entry of entries) {
      const command = isPlainObject(entry) && typeof entry.command === 'string' ? entry.command : '';
      if (command.includes(scriptPath)) matched.push(entry);
    }
    commands[key] = matched.map((entry) => String(entry.command));
    if (matched.length > 0) registeredUnder.push(key);
    else missing.push(key);
  }
  const firstCommands = OBSERVER_EVENT_KEYS.map((key) => commands[key][0] ?? null);
  const allPresent = firstCommands.every((command) => command !== null);
  const sameCommand = allPresent && firstCommands.every((command) => command === firstCommands[0]);
  const matcherFree = OBSERVER_EVENT_KEYS.every((key) =>
    (Array.isArray(hooks[key]) ? hooks[key] : []).every(
      (entry) => !(isPlainObject(entry) && Object.prototype.hasOwnProperty.call(entry, 'matcher')),
    ));
  const registered = missing.length === 0;
  return {
    registered,
    complete: registered && sameCommand && matcherFree,
    registered_under: registeredUnder,
    missing,
    same_command: sameCommand,
    matcher_free: matcherFree,
    script_path: scriptPath,
    commands,
  };
}

/**
 * Tells whether a value is a non-empty string.
 *
 * @param {unknown} value The value to test.
 * @returns {boolean} True for a non-empty string.
 */
function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

/**
 * Collects every key of an object tree, so a validator can report a forbidden
 * key wherever it sits.
 *
 * @param {unknown} source The object to walk.
 * @param {string[]} [collected] The accumulator of collected keys.
 * @returns {string[]} The collected keys.
 */
function collectKeys(source, collected = []) {
  if (Array.isArray(source)) {
    for (const item of source) collectKeys(item, collected);
    return collected;
  }
  if (!isPlainObject(source)) return collected;
  for (const [key, value] of Object.entries(source)) {
    collected.push(key);
    collectKeys(value, collected);
  }
  return collected;
}

/**
 * Sanitises one part of a dispatch-record file name, so a plan id or a slice
 * id can never escape the dispatch directory.
 *
 * @param {unknown} value The raw identity part.
 * @returns {string} The sanitised identity part.
 */
function sanitizeIdentityPart(value) {
  const raw = isNonEmptyString(value) ? value.trim() : 'unknown';
  return raw.replace(/[^A-Za-z0-9._-]/g, '_');
}
