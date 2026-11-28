/**
 * The THREE local JSONL exporters of the observability boundary: the trace
 * exporter, the metric exporter and the log exporter, all of them writing one
 * sanitized JSON record per line under `logs/telemetry/`.
 *
 * WHAT AN EXPORTER DOES, IN THIS ORDER:
 *
 *   1. It checks the RECORD NAME against the allow-list of its signal. A
 *      record the allow-list does not cover is not written at all - not
 *      written with fewer fields, NOT written: the sink stays empty, so an
 *      instrument nobody declared cannot show up in the artefact.
 *   2. It sanitises the attributes (allow-list pass, then deny pass).
 *   3. It converts the MONOTONIC INTERVAL into the single duration field
 *      `duration_ms` and drops every instant (`startTime`, `endTime`,
 *      `timeUnixNano`, `observedTimestamp`): a persisted artefact carries a
 *      DURATION, never a moment.
 *
 * WHY A CALLER-SUPPLIED DURATION IS IGNORED: the duration is a MEASUREMENT,
 * and the only honest measurement of a span is the monotonic interval the
 * exporter itself was handed. A caller that also passes `durationMs` gets the
 * interval-derived value, so a second duration field can never appear and a
 * caller cannot smuggle a duration of its own into the artefact. When there
 * is no interval at all, NO duration field is written - a record without a
 * measurement carries no measurement.
 *
 * WHY THE EXPORTER NEVER THROWS: the telemetry path is fail-open by
 * contract. A sink that cannot be created, a sink that cannot be written and
 * an attribute accessor that throws all produce a short machine-stable
 * warning and never an exception into the caller, because the work the
 * telemetry path observes must never fail because of the telemetry path.
 *
 * WHY THE SINK DIRECTORY IS CREATED ON DEMAND: an exporter that is
 * constructed but never used must leave no trace on disk, and the sink
 * directory of a repo whose telemetry was never enabled must not appear in
 * the working tree.
 *
 * WHY THE LOCAL SETTINGS GOVERN THE SINKS: the sink root is the
 * `exporter.path` of `logs/telemetry/settings.json`, resolved against the
 * directory of that file, and an EXPLICIT `enabled: false` in that file
 * stops the emission. The local enablement command owns the file, so the
 * settings an operator writes are the settings the exporters obey - they
 * are not a note nobody reads.
 *
 * THE PIPELINE OF ONE RECORD, IN ORDER:
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   record[One record arrives]:::base --> allowed{Record name on the signal allow-list?}:::accent
 *   allowed -->|No| dropped[Nothing is written at all]:::base
 *   allowed -->|Yes| sanitize[Sanitise attributes, allow-list pass then deny-list pass]:::base
 *   sanitize --> span{Monotonic interval present?}:::accent
 *   span -->|Yes| derived[duration_ms derived from the interval]:::base
 *   span -->|No| noduration[No duration field is written]:::base
 *   derived --> instants[Every instant is dropped]:::base
 *   noduration --> instants
 *   instants --> enabled{Layer enabled by the local settings?}:::accent
 *   enabled -->|No| silent[Nothing is written at all]:::base
 *   enabled -->|Yes| append[Append one JSON line to the sink file]:::base
 *   append --> created[Sink directory created on demand]:::base
 * ```
 *
 * WHY THE FILE FORMAT IS JSON LINES: each record is written as ONE JSON
 * object on ONE line, which is the [JSON Lines](https://en.wikipedia.org/wiki/JSON_Lines)
 * convention. A line-oriented format keeps an append-only sink cheap, keeps
 * a truncated write readable, and lets a reader parse one record without
 * parsing the whole file.
 *
 * WHY THE ALLOW-LIST IS A GATE AND NOT A FILTER: the list of accepted
 * record names is an [allow list](https://en.wikipedia.org/wiki/Allow_list),
 * so an instrument nobody declared has no path into the artefact. A record
 * the list does not cover is not written with fewer fields - it is not
 * written at all.
 *
 * LAYER INDEPENDENCE: this module imports NOTHING from the repo-owned
 * measurement tooling in `scripts/telemetry` and NOTHING from the
 * NEAT-algorithm telemetry tree in `src/neat/telemetry`. It measures nothing
 * of a slice window and it owns no slice instant.
 *
 * @example
 * const exporter = createJsonlExporter({ signal: 'trace' });
 * await exporter.exportRecord({
 *   name: 'neat.telemetry.trace',
 *   attributes: { 'neat.telemetry.signal': 'trace' },
 *   interval: { startMonotonicMs: 1000, endMonotonicMs: 1250 },
 * });
 * await exporter.forceFlush();
 *
 * @module telemetry.jsonl-exporters
 */

import { appendFileSync, existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import {
  type TelemetrySignal,
  getAllowListEntry,
} from './telemetry.policy';
import {
  resolveTelemetryEnabled,
  resolveTelemetrySinkRoot,
} from './telemetry.local-settings';
import { sanitizeAttributes } from './telemetry.sanitize';

/**
 * The warning token for a sink that cannot be created or written.
 *
 * @type {string}
 */
export const REASON_SINK_UNWRITABLE = 'telemetry_sink_unwritable';

/**
 * The warning token for a record whose name the allow-list does not cover.
 *
 * @type {string}
 */
export const REASON_RECORD_NOT_ALLOWED = 'telemetry_record_not_allowed';

/**
 * A record the exporter may write. The exporter decides the final shape: it
 * drops every instant, derives the single duration field from the monotonic
 * interval, and writes only the attributes that the allow-list and the
 * deny-list both accept.
 *
 * @interface TelemetryRecordInput
 * @property {string} name - The record name.
 * @property {Record<string, unknown>} [attributes] - The attributes of the record.
 * @property {{ startMonotonicMs: number, endMonotonicMs: number }} [interval] - The monotonic interval of the record.
 * @property {number} [durationMs] - A caller-supplied duration; it never becomes a second duration field.
 * @property {unknown} [startTime] - An instant; it is dropped.
 * @property {unknown} [endTime] - An instant; it is dropped.
 * @property {unknown} [timeUnixNano] - An instant; it is dropped.
 * @property {unknown} [observedTimestamp] - An instant; it is dropped.
 */
export interface TelemetryRecordInput {
  readonly name: string;
  readonly attributes?: Record<string, unknown>;
  readonly interval?: {
    readonly startMonotonicMs: number;
    readonly endMonotonicMs: number;
  };
  readonly durationMs?: number;
  readonly startTime?: unknown;
  readonly endTime?: unknown;
  readonly timeUnixNano?: unknown;
  readonly observedTimestamp?: unknown;
}

/**
 * The options of one exporter: the sink directory it writes into, the signal
 * whose allow-list governs the records it accepts, and the warning sink that
 * receives its fail-open warnings.
 *
 * TWO OF THE OPTIONS COME FROM THE LOCAL SETTINGS: `repositoryRoot` says
 * WHICH tree the settings are read from, and `enabled` says whether the
 * exporter may emit at all. When `enabled` is left out, the `enabled` flag
 * of `logs/telemetry/settings.json` decides, and when `repositoryRoot` is
 * left out, the repository root of the working directory is used. An
 * explicit `enabled` therefore OVERRIDES the settings file, which is what
 * makes a caller able to switch the layer off without touching the file.
 *
 * @interface JsonlExporterOptions
 * @property {string} [sinkDirectory] - The sink directory; defaults to the sink root the LOCAL SETTINGS name.
 * @property {TelemetrySignal} [signal] - The signal the exporter writes; defaults to `trace`.
 * @property {(message: string) => void} [warn] - The warning sink.
 * @property {string} [repositoryRoot] - The repository root the LOCAL SETTINGS are read from; defaults to the repository root of the working directory.
 * @property {boolean} [enabled] - Whether the exporter may emit; defaults to the `enabled` flag of the LOCAL SETTINGS.
 */
export interface JsonlExporterOptions {
  readonly sinkDirectory?: string;
  readonly signal?: TelemetrySignal;
  readonly warn?: (message: string) => void;
  readonly repositoryRoot?: string;
  readonly enabled?: boolean;
}

/**
 * The exporter handle. Every method of the handle is fail-open: a record the
 * allow-list does not cover is not written at all, and a sink that cannot be
 * created or written produces a warning instead of an exception.
 *
 * @interface JsonlExporter
 * @property {string} sinkDirectory - The absolute sink directory.
 * @property {(record: TelemetryRecordInput) => void | Promise<void>} exportRecord - Writes one record, or nothing at all.
 * @property {() => void | Promise<void>} forceFlush - Flushes the pending writes.
 * @property {() => void | Promise<void>} shutdown - Flushes and closes the exporter.
 */
export interface JsonlExporter {
  readonly sinkDirectory: string;
  exportRecord(record: TelemetryRecordInput): void | Promise<void>;
  forceFlush(): void | Promise<void>;
  shutdown(): void | Promise<void>;
}

/**
 * Walks up from the working directory to the repository root, so the default
 * sink directory does not depend on the directory a runner happened to choose.
 *
 * @returns {string} The absolute repository root.
 */
function findRepositoryRoot(): string {
  let current = process.cwd();
  for (let depth = 0; depth < 8; depth += 1) {
    if (existsSync(path.join(current, 'package.json'))) return current;
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return process.cwd();
}

/**
 * Resolves the DEFAULT sink directory: the sink root the LOCAL SETTINGS name.
 *
 * THE LOCAL SETTINGS GOVERN THE SINKS: the sink root is the `exporter.path`
 * of `logs/telemetry/settings.json`, resolved against the directory of that
 * file, so the file the local enablement command writes is the file the
 * exporters obey. When the settings are missing, corrupt or carry an
 * ABSOLUTE path, the DECLARED default root is used instead. The directory
 * itself is NOT created here - it is created on demand, by the first write.
 *
 * @param {string} [repositoryRoot] - The repository root; defaults to the repository root of the working directory.
 * @returns {string} The absolute sink directory.
 * @example
 * resolveDefaultSinkDirectory();
 * // => 'C:\\...\\logs\\telemetry'
 * resolveDefaultSinkDirectory('/repo');
 * // => '/repo/logs/telemetry'
 */
export function resolveDefaultSinkDirectory(repositoryRoot?: string): string {
  return resolveTelemetrySinkRoot(repositoryRoot ?? findRepositoryRoot());
}

/**
 * Creates one local JSONL exporter for one signal. The sink directory is
 * created on demand by the first write, so an exporter that is constructed
 * and never used leaves no trace on disk. An exporter whose LOCAL SETTINGS
 * say `enabled: false` validates every record as usual but writes NOTHING
 * into the sink and never creates the sink directory.
 *
 * @param {JsonlExporterOptions} [options] - The exporter options.
 * @returns {JsonlExporter} The exporter.
 * @example
 * const exporter = createJsonlExporter({ signal: 'metric' });
 * exporter.sinkDirectory; // => 'C:\\...\\logs\\telemetry'
 *
 * const quiet = createJsonlExporter({ signal: 'log', enabled: false });
 * quiet.exportRecord({ name: 'neat.telemetry.log' });
 * // => nothing is written and the sink directory is never created
 */
export function createJsonlExporter(
  options: JsonlExporterOptions = {},
): JsonlExporter {
  const signal: TelemetrySignal = options.signal ?? 'trace';
  const repositoryRoot = options.repositoryRoot ?? findRepositoryRoot();
  const sinkDirectory = options.sinkDirectory
    ? path.resolve(options.sinkDirectory)
    : resolveDefaultSinkDirectory(repositoryRoot);
  const enabled = options.enabled ?? resolveTelemetryEnabled(repositoryRoot);
  const warn =
    options.warn ??
    ((message: string): void => {
      console.warn(message);
    });
  const allowedRecordNames = new Set(getAllowListEntry(signal).recordNames);
  const sinkFile = path.join(sinkDirectory, `${signal}.jsonl`);
  let closed = false;

  /**
   * Writes one line into the sink, creating the sink directory on demand.
   * Every failure is a warning: the exporter never throws into the caller.
   *
   * @param line - The line to append.
   */
  const appendLine = (line: string): void => {
    try {
      mkdirSync(sinkDirectory, { recursive: true });
      appendFileSync(sinkFile, line, 'utf8');
    } catch {
      warn(REASON_SINK_UNWRITABLE);
    }
  };

  return {
    sinkDirectory,
    exportRecord(record: TelemetryRecordInput): void {
      if (closed) return;
      if (!allowedRecordNames.has(record.name)) {
        warn(REASON_RECORD_NOT_ALLOWED);
        return;
      }
      const { attributes } = sanitizeAttributes(record.attributes, {
        signal,
        warn,
      });
      const written: Record<string, unknown> = { name: record.name };
      const interval = record.interval;
      if (
        interval !== undefined &&
        interval !== null &&
        Number.isFinite(interval.startMonotonicMs) &&
        Number.isFinite(interval.endMonotonicMs)
      ) {
        const duration = interval.endMonotonicMs - interval.startMonotonicMs;
        if (Number.isFinite(duration) && duration >= 0) {
          written.duration_ms = duration;
        }
      }
      for (const [key, value] of Object.entries(attributes)) {
        written[key] = value;
      }
      // THE LOCAL SWITCH GOVERNS THE SINK, NOT THE VALIDATION: a disabled
      // layer still refuses a record the allow-list does not cover - and
      // says so - but it never appends a line and never creates the sink
      // directory.
      if (!enabled) return;
      appendLine(`${JSON.stringify(written)}\n`);
    },
    forceFlush(): void {
      // Every write is appended synchronously, so there is nothing buffered
      // to flush. The method exists because the OpenTelemetry exporters call
      // it, and it keeps the flush deterministic for the caller.
    },
    shutdown(): void {
      closed = true;
    },
  };
}
