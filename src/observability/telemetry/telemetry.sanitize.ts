/**
 * The export boundary of the local OpenTelemetry signal layer: the ALLOW-LIST
 * pass and the DENY-LIST pass that every attribute has to survive before it
 * may reach the local JSONL sink.
 *
 * THE TWO PASSES, IN THIS ORDER:
 *
 *   1. THE ALLOW-LIST PASS. An attribute survives only when its NAME is
 *      explicitly permitted for the signal that writes it. A name that is not
 *      permitted is dropped without a trace, because an allow-list is a rule
 *      and not an anomaly.
 *   2. THE DENY-LIST PASS. A name the allow-list permits can still be
 *      REJECTED by its VALUE: a non-scalar bag (a tool argument list, a file
 *      content, a nested object) and a value that carries a forbidden shape
 *      (an absolute path, a calendar date, an epoch literal, an e-mail
 *      address) never reach the sink, even when the name is allowed.
 *
 * WHY THE DENY PASS IS NOT REDUNDANT: the allow-list names the KEYS, and a
 * key that is allowed can still carry a value nobody meant to publish. The
 * allow-list is the rule; the deny pass is the safety net under it.
 *
 * WHY A THROWING ACCESSOR IS A WARNING AND NOT AN EXCEPTION: an attribute bag
 * is caller-supplied data, and a getter that throws is a defect in the
 * CALLER, not in the telemetry path. The path warns with a short
 * machine-stable token, drops that one attribute and keeps going, because a
 * broken attribute must never fail the work the telemetry path observes.
 *
 * WHY THE WARNINGS ARE TOKENS: a warning is persisted in a log like any other
 * artefact, so a warning carries no absolute path, no calendar date and no
 * free text that could carry one. The tokens are greppable and stable across
 * releases.
 *
 * LAYER INDEPENDENCE: this module imports NOTHING from the repo-owned
 * measurement tooling in `scripts/telemetry` and NOTHING from the
 * NEAT-algorithm telemetry tree in `src/neat/telemetry`. The sanitizer of the
 * signal layer knows nothing about the repo-owned per-slice ledger.
 *
 * @example
 * import { sanitizeAttributes } from './telemetry.sanitize';
 *
 * sanitizeAttributes(
 *   { 'neat.telemetry.signal': 'trace', 'not.allowed': 'x' },
 *   { allowedAttributeNames: ['neat.telemetry.signal'] },
 * );
 * // => { attributes: { 'neat.telemetry.signal': 'trace' }, dropped: ['not.allowed'] }
 *
 * @module telemetry.sanitize
 */

import {
  FORBIDDEN_KEYS,
  FORBIDDEN_VALUE_PATTERNS,
  type TelemetrySignal,
  getAllowListEntry,
} from './telemetry.policy';

/**
 * The warning token for an attribute whose accessor threw. The attribute is
 * dropped and the path continues.
 *
 * @type {string}
 */
export const REASON_ATTRIBUTE_ACCESSOR_FAILED = 'telemetry_attribute_accessor_failed';

/**
 * The warning token for an allow-listed attribute whose VALUE was rejected by
 * the deny pass.
 *
 * @type {string}
 */
export const REASON_ATTRIBUTE_VALUE_REJECTED = 'telemetry_attribute_value_rejected';

/**
 * The warning token for an attribute whose NAME is not in the allow-list. The
 * attribute is dropped silently by default; the token exists for a caller
 * that asks for the strict mode.
 *
 * @type {string}
 */
export const REASON_ATTRIBUTE_NOT_ALLOWED = 'telemetry_attribute_not_allowed';

/**
 * The options of one sanitisation pass. The permitted attribute names may be
 * given EXPLICITLY, or the pass reads them from the allow-list of the signal
 * that writes the record.
 *
 * @interface SanitizeAttributesOptions
 * @property {readonly string[]} [allowedAttributeNames] - The permitted attribute names.
 * @property {TelemetrySignal} [signal] - The signal, used to read the allow-list when no explicit list is given.
 * @property {(message: string) => void} [warn] - The warning sink.
 * @property {boolean} [strict] - Whether a dropped allow-list name warns.
 */
export interface SanitizeAttributesOptions {
  readonly allowedAttributeNames?: readonly string[];
  readonly signal?: TelemetrySignal;
  readonly warn?: (message: string) => void;
  readonly strict?: boolean;
}

/**
 * The result of one sanitisation pass: the attributes that survived BOTH
 * passes, and the names that were dropped, in the order they were seen.
 *
 * @interface SanitizeAttributesResult
 * @property {Record<string, unknown>} attributes - The attributes that survived both passes.
 * @property {string[]} dropped - The names that were dropped, in the order they were seen.
 */
export interface SanitizeAttributesResult {
  readonly attributes: Record<string, unknown>;
  readonly dropped: string[];
}

/**
 * The keys the exporter reserves for itself. An attribute that carries one of
 * them is dropped, so a caller can never overwrite the record name or the
 * single duration field.
 *
 * @type {readonly string[]}
 */
export const RESERVED_RECORD_KEYS: readonly string[] = Object.freeze([
  'name',
  'duration_ms',
]);

/**
 * Normalises a key the way the forbidden set is normalised, so `startTime`,
 * `start_time` and `START-TIME` are one and the same forbidden key.
 *
 * @param {unknown} key - The key to normalise.
 * @returns {string} The normalised key.
 */
function normalizeKey(key: unknown): string {
  return String(key).toLowerCase().replace(/[_-]/g, '');
}

/**
 * The forbidden key set, normalised once.
 *
 * @type {Set<string>}
 */
const FORBIDDEN_KEY_INDEX: ReadonlySet<string> = new Set(
  FORBIDDEN_KEYS.map((key) => normalizeKey(key)),
);

/**
 * Tells whether a key is forbidden.
 *
 * @param {unknown} key - The key to test.
 * @returns {boolean} True when the key is forbidden.
 */
function isForbiddenKey(key: unknown): boolean {
  return FORBIDDEN_KEY_INDEX.has(normalizeKey(key));
}

/**
 * Tells whether a value is a SCALAR the sink may write. An object, an array, a
 * function, a symbol and a bigint-free exotic value are all non-scalars: a
 * non-scalar bag is where tool arguments and file contents live.
 *
 * @param {unknown} value - The value to test.
 * @returns {boolean} True when the value is a writable scalar.
 */
export function isWritableScalar(value: unknown): boolean {
  return (
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  );
}

/**
 * Tells whether a value carries a forbidden shape. A number is matched as its
 * decimal text, so an epoch-millisecond literal cannot hide behind a number
 * type.
 *
 * @param {unknown} value - The value to test.
 * @returns {boolean} True when the value carries a forbidden shape.
 */
export function carriesForbiddenShape(value: unknown): boolean {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return true;
    return FORBIDDEN_VALUE_PATTERNS.some((pattern) =>
      pattern.test(String(value)),
    );
  }
  if (typeof value !== 'string') return false;
  return FORBIDDEN_VALUE_PATTERNS.some((pattern) => pattern.test(value));
}

/**
 * Runs both passes over a caller-supplied attribute bag. An attribute has to
 * survive the ALLOW-LIST pass AND the DENY-LIST pass to reach the returned
 * bag; everything else is dropped and named in `dropped`.
 *
 * @param {Record<string, unknown> | undefined | null} attributes - The attribute bag.
 * @param {SanitizeAttributesOptions} [options] - The pass options.
 * @returns {SanitizeAttributesResult} The surviving attributes and the dropped names.
 * @example
 * sanitizeAttributes({ 'neat.telemetry.signal': 'trace' }, {
 *   allowedAttributeNames: ['neat.telemetry.signal'],
 * }).attributes;
 * // => { 'neat.telemetry.signal': 'trace' }
 */
export function sanitizeAttributes(
  attributes: Record<string, unknown> | undefined | null,
  options: SanitizeAttributesOptions = {},
): SanitizeAttributesResult {
  const allowed = new Set(
    options.allowedAttributeNames ??
      (options.signal ? getAllowListEntry(options.signal).attributeNames : []),
  );
  const warn = options.warn;
  const strict = options.strict === true;
  const kept: Record<string, unknown> = {};
  const dropped: string[] = [];
  if (attributes === null || attributes === undefined) {
    return { attributes: kept, dropped };
  }
  for (const key of Object.keys(attributes)) {
    if (!allowed.has(key)) {
      dropped.push(key);
      if (strict && warn) warn(REASON_ATTRIBUTE_NOT_ALLOWED);
      continue;
    }
    if (isForbiddenKey(key) || RESERVED_RECORD_KEYS.includes(key)) {
      dropped.push(key);
      continue;
    }
    let value: unknown;
    try {
      value = attributes[key];
    } catch {
      // A caller-supplied accessor that throws is the CALLER's defect: the
      // attribute is dropped, the path warns and the work continues.
      dropped.push(key);
      if (warn) warn(REASON_ATTRIBUTE_ACCESSOR_FAILED);
      continue;
    }
    if (!isWritableScalar(value) || carriesForbiddenShape(value)) {
      dropped.push(key);
      if (warn) warn(REASON_ATTRIBUTE_VALUE_REJECTED);
      continue;
    }
    kept[key] = value;
  }
  return { attributes: kept, dropped };
}
