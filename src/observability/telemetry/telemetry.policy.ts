/**
 * The DECLARATION of the local OpenTelemetry signal policy of the
 * observability boundary: the sink root, the telemetry engine floor, the
 * per-signal allow-list and the two forbidden sets.
 *
 * WHAT this module is: the ONE place where the privacy rules of the local
 * signal layer are DECLARED. The exporter boundary is an ALLOW-LIST first - a
 * field is written only when it is explicitly permitted for that signal - and
 * a DENY-LIST second, so a bug in the allow-list still cannot leak a
 * forbidden field.
 *
 * WHY the forbidden sets live here AND in `scripts/telemetry/telemetry-contract.mjs`:
 * the rules are pinned in two languages (this TypeScript policy and the
 * repo-owned JavaScript contract), and the drift guard compares them. A second
 * implementation would be a drift bug, so the two lists are kept IDENTICAL and
 * the guard fails on any difference.
 *
 * WHY the engine floor is declared HERE and not in the manifest: the root
 * manifest floor stays at `>= 22.0.0` because the OpenTelemetry 3.x line is
 * optional for a consumer. The telemetry path needs a HIGHER floor of its own,
 * and a manifest cannot express two floors for one package, so the telemetry
 * path declares and enforces its own floor in source.
 *
 * LAYER INDEPENDENCE: this module imports NOTHING from the repo-owned
 * measurement tooling in `scripts/telemetry` and NOTHING from the
 * NEAT-algorithm telemetry tree in `src/neat/telemetry`. The signal layer
 * produces SIGNALS; the repo-owned per-slice ledger produces the
 * MEASUREMENTS, and neither depends on the other.
 *
 * @example
 * import { TELEMETRY_ALLOW_LIST } from './telemetry.policy';
 *
 * TELEMETRY_ALLOW_LIST.trace.recordNames;
 * // => ['neat.telemetry.trace']
 *
 * @module telemetry.policy
 */

/**
 * The repo-relative root of the local JSONL sink. The sink is the ONLY thing
 * the signal layer writes to disk, and `.gitignore` already covers it.
 *
 * @type {string}
 */
export const TELEMETRY_SINK_ROOT = 'logs/telemetry';

/**
 * The engine floor of the TELEMETRY path, which is strictly higher than the
 * root manifest floor (`>= 22.0.0`): the ruled OpenTelemetry 3.1.1 packages
 * require `>= 22.15.0` themselves. A runtime below this floor degrades the
 * layer warn-only instead of failing the caller's work.
 *
 * @type {string}
 */
export const TELEMETRY_NODE_FLOOR = '>= 22.15.0';

/**
 * The machine-stable reason token the telemetry path reports when the runtime
 * is below its own floor. A TOKEN, not a sentence: it has to be greppable and
 * stable across releases.
 *
 * @type {string}
 */
export const REASON_NODE_BELOW_TELEMETRY_FLOOR = 'node_below_telemetry_floor';

/**
 * The three signals the local OpenTelemetry layer emits. A signal names the
 * KIND of observation - a span, a metric data point or a log record - and
 * every record written to the local sink belongs to exactly one of them.
 *
 * @type {'trace' | 'metric' | 'log'}
 */
export type TelemetrySignal = 'trace' | 'metric' | 'log';

/**
 * The per-signal allow-list entry: the record names that may be written for
 * that signal and the attribute names that may be written with them.
 *
 * @interface TelemetryAllowListEntry
 * @property {readonly string[]} recordNames - The record names the signal may write.
 * @property {readonly string[]} attributeNames - The attribute names the signal may write.
 */
export interface TelemetryAllowListEntry {
  readonly recordNames: readonly string[];
  readonly attributeNames: readonly string[];
}

/**
 * The ALLOW-LIST of the export boundary, one entry per signal.
 *
 * WHY THE LISTS ARE SHORT: an allow-list only protects what it names, and
 * every name here is a repo-owned namespace. The metric list deliberately
 * carries NO `neat.acceleration` instrument: the acceleration variants emit an
 * unconditional inference duration of zero, so a metric instrument built on it
 * would report a MEASUREMENT OF ZERO for work that was never measured - and a
 * zero is a measurement, so it may only be written when a source really
 * reported zero.
 *
 * @type {Readonly<Record<TelemetrySignal, TelemetryAllowListEntry>>}
 */
export const TELEMETRY_ALLOW_LIST: Readonly<
  Record<TelemetrySignal, TelemetryAllowListEntry>
> = Object.freeze({
  trace: Object.freeze({
    recordNames: Object.freeze(['neat.telemetry.trace']),
    attributeNames: Object.freeze([
      'neat.telemetry.signal',
      'neat.telemetry.component',
      'neat.telemetry.outcome',
    ]),
  }),
  metric: Object.freeze({
    recordNames: Object.freeze(['neat.telemetry.metric']),
    attributeNames: Object.freeze([
      'neat.telemetry.signal',
      'neat.telemetry.component',
      'neat.telemetry.unit',
    ]),
  }),
  log: Object.freeze({
    recordNames: Object.freeze(['neat.telemetry.log']),
    attributeNames: Object.freeze([
      'neat.telemetry.signal',
      'neat.telemetry.severity',
      'neat.telemetry.component',
    ]),
  }),
});

/**
 * The forbidden KEY set: every key that would carry an instant, a clock time,
 * a date, a modification marker or a PII-ish identifier. A persisted field
 * that would carry an instant is REMOVED, never exempted.
 *
 * The list is shared with `scripts/telemetry/telemetry-contract.mjs`; a drift
 * guard compares the two lists and fails on any difference.
 *
 * @type {readonly string[]}
 */
export const FORBIDDEN_KEYS: readonly string[] = Object.freeze([
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
 * The forbidden VALUE patterns: the shapes a persisted value must never take
 * - ISO-8601 datetimes, calendar dates, epoch-millisecond literals, absolute
 * Windows paths, absolute POSIX user paths, any other absolute path shape and
 * e-mail addresses.
 *
 * The patterns are deliberately NOT global, so `RegExp.prototype.test` and
 * `RegExp.prototype.exec` are stateless and a caller can reuse them safely.
 *
 * @type {readonly RegExp[]}
 */
export const FORBIDDEN_VALUE_PATTERNS: readonly RegExp[] = Object.freeze([
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
 * The signals the layer knows, in the canonical order. The order is the
 * canonical one - trace, then metric, then log - so a rendered listing of
 * the supported signals stays stable across releases and comparable between
 * runs.
 *
 * @type {readonly TelemetrySignal[]}
 */
export const TELEMETRY_SIGNALS: readonly TelemetrySignal[] = Object.freeze([
  'trace',
  'metric',
  'log',
]);

/**
 * Tells whether a value names one of the three known signals.
 *
 * @param {unknown} value - The value to test.
 * @returns {boolean} True when the value is a known signal.
 * @example
 * isTelemetrySignal('trace'); // => true
 * isTelemetrySignal('span'); // => false
 */
export function isTelemetrySignal(value: unknown): value is TelemetrySignal {
  return (
    typeof value === 'string' &&
    (TELEMETRY_SIGNALS as readonly string[]).includes(value)
  );
}

/**
 * Returns the allow-list entry of a signal. An unknown signal resolves to the
 * EMPTY entry, so a typo can never widen the allow-list.
 *
 * @param {TelemetrySignal} signal - The signal.
 * @returns {TelemetryAllowListEntry} The allow-list entry of the signal.
 * @example
 * getAllowListEntry('metric').recordNames; // => ['neat.telemetry.metric']
 */
export function getAllowListEntry(
  signal: TelemetrySignal,
): TelemetryAllowListEntry {
  return (
    TELEMETRY_ALLOW_LIST[signal] ??
    Object.freeze({ recordNames: Object.freeze([]), attributeNames: Object.freeze([]) })
  );
}

/**
 * Tells whether a key is one of the forbidden key names. The comparison is
 * exact on the normalised form, so `startTime`, `start_time` and `START-TIME`
 * are one and the same forbidden key, while `duration_ms` stays legal.
 *
 * @param {unknown} key - The key to test.
 * @returns {boolean} True when the key is forbidden.
 * @example
 * isForbiddenKey('startTime'); // => true
 * isForbiddenKey('duration_ms'); // => false
 */
export function isForbiddenKey(key: unknown): boolean {
  return FORBIDDEN_KEYS.some(
    (forbidden) =>
      String(forbidden).toLowerCase().replace(/[_-]/g, '') ===
      String(key).toLowerCase().replace(/[_-]/g, ''),
  );
}

/**
 * Tells whether a value carries a forbidden shape. A string is matched
 * against every forbidden value pattern; a number is matched as its decimal
 * text, so an epoch-millisecond literal cannot hide behind a number type.
 *
 * @param {unknown} value - The value to test.
 * @returns {boolean} True when the value carries a forbidden shape.
 * @example
 * isForbiddenValue('2024-01-01'); // => true
 * isForbiddenValue('probe'); // => false
 */
export function isForbiddenValue(value: unknown): boolean {
  if (typeof value === 'number' || typeof value === 'bigint') {
    return FORBIDDEN_VALUE_PATTERNS.some((pattern) =>
      pattern.test(String(value)),
    );
  }
  if (typeof value !== 'string') return false;
  return FORBIDDEN_VALUE_PATTERNS.some((pattern) => pattern.test(value));
}
