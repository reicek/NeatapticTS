import type {
  NeatLike,
  TelemetryEntry,
  SpeciesHistoryStat,
  SpeciesHistoryEntry,
} from '../../shared/neat.shared.types';
import {
  collectBaseKeys,
  collectGroupedMetricKeys,
  collectDiversityLineageMetrics,
  collectOptionalColumnPresence,
  ensureSpeciesHistoryArray,
  ensureMinimalSpeciesSnapshot,
  collectSpeciesHistoryHeaders,
  serializeSpeciesHistoryRow,
  type TelemetryHeaderCollectionState,
} from './telemetry.exports.utils';

/**
 * Telemetry export helpers for stream-friendly logs and spreadsheet-oriented analysis.
 *
 * This chapter keeps the public serialization story together after the telemetry
 * facade split: JSONL helpers preserve full-fidelity event payloads for log
 * pipelines, while the CSV helpers flatten the most useful telemetry and
 * species-history metrics into a deterministic tabular shape for notebooks,
 * spreadsheets, and quick audits.
 *
 * The neighboring `telemetry/facade/*` chapters own public `Neat` wrappers.
 * This exports chapter owns the actual serialization mechanics those wrappers
 * delegate to.
 *
 * The main design constraint is determinism across uneven telemetry windows.
 * A long-running experiment rarely records the same optional fields in every
 * entry, especially when objective sets, lineage metadata, or Pareto state can
 * change over time. The helpers in this file therefore do two jobs at once:
 * they serialize entries, and they first discover a stable export shape so the
 * resulting CSV can still be compared, plotted, or diffed reliably.
 *
 * Read this chapter when you want to understand how NeatapticTS turns rich but
 * irregular telemetry objects into bounded JSONL streams and spreadsheet-ready
 * CSV output without losing the meaning of sparse runtime fields.
 *
 * Read the boundary as three export decisions instead of one big serialization
 * shelf:
 *
 * 1. `exportTelemetryJSONL()` when downstream tools want each full-fidelity
 *    runtime entry on its own line,
 * 2. `exportTelemetryCSV()` when a notebook or spreadsheet needs one stable
 *    telemetry table across a bounded recent window,
 * 3. `exportSpeciesHistoryCSV()` when the question is specifically about
 *    species turnover, growth, stagnation, and timing over generations.
 *
 * The distinction matters because "export telemetry" is not one reader need.
 * Log pipelines want raw objects that preserve nested detail. Human analysis
 * tools usually want a fixed rectangular table. Species history often deserves
 * its own table entirely because one generation can contain multiple species
 * rows and because early runs may need a synthesized first snapshot before the
 * broader telemetry stream becomes interesting.
 *
 * ```mermaid
 * flowchart LR
 *   Buffer[Telemetry buffer and species history] --> Choice{What do you need to inspect?}
 *   Choice --> JSONL[Full-fidelity event stream\nexportTelemetryJSONL]
 *   Choice --> TelemetryCsv[Bounded telemetry table\nexportTelemetryCSV]
 *   Choice --> SpeciesCsv[Species timeline table\nexportSpeciesHistoryCSV]
 *   TelemetryCsv --> Headers[Discover stable headers\nacross the sampled window]
 *   SpeciesCsv --> Backfill[Backfill an early snapshot\nwhen history has not started yet]
 * ```
 */

/**
 * Shape describing collected telemetry header discovery info.
 *
 * Think of this as the temporary export blueprint for one CSV window. Before a
 * telemetry entry can be flattened into cells, the exporter needs to know which
 * top-level fields exist, which nested metric groups need prefixed columns, and
 * which optional arrays or maps appeared anywhere in the sampled window.
 */
interface TelemetryHeaderInfo {
  /** Top-level keys (excluding explicit grouped objects). */
  baseKeys: Set<string>;
  /** Nested metric keys under complexity group. */
  complexityKeys: Set<string>;
  /** Nested metric keys under performance group. */
  perfKeys: Set<string>;
  /** Nested metric keys under lineage group. */
  lineageKeys: Set<string>;
  /** Selected diversity lineage metric keys. */
  diversityLineageKeys: Set<string>;
  /** Flag: include ops column. */
  includeOps: boolean;
  /** Flag: include objectives column. */
  includeObjectives: boolean;
  /** Flag: include objective ages column. */
  includeObjAges: boolean;
  /** Flag: include species allocation column. */
  includeSpeciesAlloc: boolean;
  /** Flag: include objective events column. */
  includeObjEvents: boolean;
  /** Flag: include objective importance column. */
  includeObjImportance: boolean;
}

/** Group prefix for complexity nested metrics when flattened. */
const COMPLEXITY_PREFIX = 'complexity.';
/** Group prefix for performance nested metrics when flattened. */
const PERF_PREFIX = 'perf.';
/** Group prefix for lineage nested metrics when flattened. */
const LINEAGE_PREFIX = 'lineage.';
/** Group prefix for diversity nested metrics when flattened. */
const DIVERSITY_PREFIX = 'diversity.';

/** Header label for Pareto front arrays column. */
const HEADER_FRONTS = 'fronts';
/** Header label for operations array column. */
const HEADER_OPS = 'ops';
/** Header label for objectives vector column. */
const HEADER_OBJECTIVES = 'objectives';
/** Header label for objective ages map column. */
const HEADER_OBJ_AGES = 'objAges';
/** Header label for species allocation array column. */
const HEADER_SPECIES_ALLOC = 'speciesAlloc';
/** Header label for objective events list column. */
const HEADER_OBJ_EVENTS = 'objEvents';
/** Header label for objective importance map column. */
const HEADER_OBJ_IMPORTANCE = 'objImportance';
/** Header label for generation column in species history CSV. */
const HEADER_GENERATION = 'generation';

/**
 * Default max entries for species history CSV exports.
 *
 * This keeps spreadsheet-oriented history exports bounded by default so a long
 * run does not accidentally dump an unmanageably large CSV when the caller only
 * wants a recent analytical window.
 *
 * The default is intentionally generous enough for short trend analysis while
 * still nudging callers toward deliberate windowing. Species history is often
 * most useful as a recent timeline, not as an accidental full-run dump.
 */
export const DEFAULT_SPECIES_HISTORY_MAX_ENTRIES = 200;
/**
 * Default fallback species id when a synthesized history row lacks an id.
 *
 * This only appears in the synthetic early-history path where live species
 * exist but a formal history snapshot has not been archived yet.
 */
export const DEFAULT_SPECIES_ID = -1;
/**
 * Default fallback species size when a synthesized history row lacks a size.
 *
 * The exporter prefers an explicit neutral size over leaving the cell absent so
 * spreadsheets and notebooks can still treat the backfilled row as a stable
 * member of the same column contract.
 */
export const DEFAULT_SPECIES_SIZE = 0;
/**
 * Default fallback best score when a synthesized history row lacks a score.
 *
 * This preserves a numeric best-score column even when the runtime only has a
 * minimally reconstructed species snapshot.
 */
export const DEFAULT_SPECIES_BEST_SCORE = 0;
/**
 * Default fallback last-improved generation when a synthesized row lacks one.
 *
 * A neutral default keeps the CSV shape deterministic without pretending the
 * exporter knows more improvement history than the controller has actually
 * recorded.
 */
export const DEFAULT_SPECIES_LAST_IMPROVED = 0;
/**
 * Default fallback generation used when backfilling an early species snapshot.
 *
 * The synthetic row still prefers the controller's live generation when it is
 * available. This constant is the final floor for partially initialized hosts.
 */
export const DEFAULT_SPECIES_HISTORY_GENERATION = 0;

/**
 * Serialize telemetry as JSON Lines for stream-friendly log exports.
 *
 * JSONL keeps the export path almost lossless: each telemetry entry is emitted
 * as one JSON object on one line, which makes the format easy to append to a
 * file, stream through command-line tools, or reload in notebook code without
 * flattening nested structures first.
 *
 * @param this - Neat instance exposing the internal telemetry buffer.
 * @returns JSONL payload with one telemetry object per line.
 *
 * @example
 * ```ts
 * const jsonl = exportTelemetryJSONL.call(neat);
 * console.log(jsonl.split('\n').length);
 * ```
 */
export function exportTelemetryJSONL(
  this: NeatLike & { _telemetry: TelemetryEntry[] },
): string {
  return this._telemetry
    .map((entry: TelemetryEntry) => JSON.stringify(entry))
    .join('\n');
}

/**
 * Export recent telemetry entries to a CSV string.
 *
 * This is the human-scanning export path. Instead of preserving the exact
 * object graph like JSONL does, it discovers a stable column set across the
 * sampled window and then flattens grouped metrics into prefixed columns.
 *
 * Flattening rules:
 * - nested `complexity`, `perf`, `lineage`, and selected `diversity` fields
 *   become `group.key` columns,
 * - optional arrays and maps are serialized as JSON cells only when present in
 *   the sampled window,
 * - the most recent `maxEntries` records are exported to keep output bounded.
 *
 * Use this path when the reader question is comparative rather than archival:
 * you want to sort, filter, chart, or diff recent runtime behavior in a tool
 * that understands rows and columns better than nested objects.
 *
 * @param this - Neat instance exposing the internal telemetry buffer.
 * @param maxEntries - Maximum number of recent telemetry rows to include.
 * @returns CSV string containing headers plus one row per exported entry.
 * @example
 * ```ts
 * const csv = exportTelemetryCSV.call(neat, 50);
 * console.log(csv.split('\n')[0]);
 * ```
 */
export function exportTelemetryCSV(
  this: NeatLike & { _telemetry: TelemetryEntry[] },
  maxEntries = 500,
): string {
  const recentTelemetry: TelemetryEntry[] = Array.isArray(this._telemetry)
    ? this._telemetry.slice(-maxEntries)
    : [];
  if (!recentTelemetry.length) return '';

  // Step 1: Collect structural + header metadata across entries.
  const headerInfo = collectTelemetryHeaderInfo(recentTelemetry);

  // Step 2: Materialize header list from collected metadata.
  const headers = buildTelemetryHeaders(headerInfo);

  // Step 3: Serialize the header row plus each telemetry entry.
  const csvLines: string[] = [headers.join(',')];
  for (const telemetryEntry of recentTelemetry) {
    csvLines.push(serializeTelemetryEntry(telemetryEntry, headers));
  }

  return csvLines.join('\n');
}

/**
 * Export species history snapshots to CSV.
 *
 * The exporter preserves runtime-added species stat fields by discovering the
 * union of keys across the requested history window. When history has not been
 * recorded yet but live species data exists, the helper synthesizes a minimal
 * snapshot so early-run CSV exports remain deterministic.
 *
 * This is the most timeline-oriented export in the chapter. Instead of one row
 * per telemetry entry, it expands each generation snapshot into one row per
 * species so a reader can inspect turnover, stagnation, and size changes with
 * ordinary tabular tools.
 *
 * @param this - Neat instance exposing species history and optional live species.
 * @param maxEntries - Maximum number of recent history snapshots to include.
 * @returns CSV payload describing one species row per generation snapshot.
 * @example
 * ```ts
 * const csv = exportSpeciesHistoryCSV.call(neat, 25);
 * console.log(csv.split('\n').slice(0, 2).join('\n'));
 * ```
 */
export function exportSpeciesHistoryCSV(
  this: NeatLike & {
    _speciesHistory?: SpeciesHistoryEntry[];
    _species?: SpeciesHistoryStat[];
    generation?: number;
  },
  maxEntries = DEFAULT_SPECIES_HISTORY_MAX_ENTRIES,
): string {
  const speciesHistory = ensureSpeciesHistoryArray(this);

  // Step 1: Backfill a minimal snapshot when only live species data exists.
  ensureMinimalSpeciesSnapshot(
    this,
    speciesHistory,
    DEFAULT_SPECIES_HISTORY_GENERATION,
    DEFAULT_SPECIES_ID,
    DEFAULT_SPECIES_SIZE,
    DEFAULT_SPECIES_BEST_SCORE,
    DEFAULT_SPECIES_LAST_IMPROVED,
  );

  // Step 2: Bound the export window to the most recent history entries.
  const recentHistory: SpeciesHistoryEntry[] =
    speciesHistory.slice(-maxEntries);
  if (!recentHistory.length) return 'generation,id,size,best,lastImproved';

  // Step 3: Discover dynamic headers, then serialize the final CSV.
  const headers = collectSpeciesHistoryHeaders(
    recentHistory,
    HEADER_GENERATION,
  );
  return buildSpeciesHistoryCsv(recentHistory, headers);
}

/**
 * Collect header metadata from the sampled telemetry entries.
 *
 * This is the discovery pass that makes CSV exports deterministic. Instead of
 * assuming every telemetry entry has the same shape, the helper scans the
 * window first and records which base fields, grouped metrics, and optional
 * columns actually appear anywhere in the sample.
 *
 * Conceptually, this is where an irregular event stream becomes a table plan.
 * The rest of the CSV path is simpler because this helper commits to one export
 * shape before row serialization starts.
 *
 * @param entries - Telemetry entries included in the export window.
 * @returns Flattened header discovery state.
 */
function collectTelemetryHeaderInfo(
  entries: TelemetryEntry[],
): TelemetryHeaderInfo {
  const headerState: TelemetryHeaderCollectionState = {
    baseKeys: new Set<string>(),
    complexityKeys: new Set<string>(),
    perfKeys: new Set<string>(),
    lineageKeys: new Set<string>(),
    diversityLineageKeys: new Set<string>(),
    includeOps: false,
    includeObjectives: false,
    includeObjAges: false,
    includeSpeciesAlloc: false,
    includeObjEvents: false,
    includeObjImportance: false,
  };

  // Step 1: Declaratively collect structural metadata per entry.
  entries.forEach((entry) => {
    collectBaseKeys(entry, headerState, HEADER_FRONTS);
    collectGroupedMetricKeys(entry, headerState);
    collectDiversityLineageMetrics(entry, headerState);
    collectOptionalColumnPresence(entry, headerState);
  });

  // Step 2: Return the fully collected header state.
  return headerState;
}

/**
 * Build the ordered header list for telemetry CSV output.
 *
 * Once discovery is complete, this helper turns the collected key sets into the
 * final column order. Group prefixes such as `complexity.` and `perf.` preserve
 * meaning after flattening so spreadsheet readers can still tell which runtime
 * family a value came from.
 *
 * The output order is intentionally grouped rather than purely alphabetical. It
 * keeps top-level run facts first, then nested metric families, then sparse
 * optional payload columns that behave more like attachments.
 *
 * @param info - Collected header discovery state.
 * @returns Ordered header names used for serialization.
 */
function buildTelemetryHeaders(info: TelemetryHeaderInfo): string[] {
  const headers: string[] = [
    ...info.baseKeys,
    ...[...info.complexityKeys].map(
      (keyName) => `${COMPLEXITY_PREFIX}${keyName}`,
    ),
    ...[...info.perfKeys].map((keyName) => `${PERF_PREFIX}${keyName}`),
    ...[...info.lineageKeys].map((keyName) => `${LINEAGE_PREFIX}${keyName}`),
    ...[...info.diversityLineageKeys].map(
      (keyName) => `${DIVERSITY_PREFIX}${keyName}`,
    ),
  ];

  if (info.includeOps) headers.push(HEADER_OPS);
  if (info.includeObjectives) headers.push(HEADER_OBJECTIVES);
  if (info.includeObjAges) headers.push(HEADER_OBJ_AGES);
  if (info.includeSpeciesAlloc) headers.push(HEADER_SPECIES_ALLOC);
  if (info.includeObjEvents) headers.push(HEADER_OBJ_EVENTS);
  if (info.includeObjImportance) headers.push(HEADER_OBJ_IMPORTANCE);

  return headers;
}

/**
 * Serialize one telemetry entry into a CSV row using the ordered headers.
 *
 * The serializer follows the header contract built for the whole export window.
 * That means each row is shaped against the same column set even when a given
 * entry is missing optional fields. Missing values become empty cells, while
 * nested structures that do exist are serialized into the cell that matches the
 * previously discovered column.
 *
 * This separation is what makes the exporter robust: header discovery decides
 * what the table means, and row serialization only answers how one entry fits
 * inside that already chosen shape.
 *
 * @param entry - Telemetry entry being serialized.
 * @param headers - Ordered headers for the whole export window.
 * @returns CSV row string.
 */
function serializeTelemetryEntry(
  entry: TelemetryEntry,
  headers: string[],
): string {
  const row: string[] = [];

  for (const header of headers) {
    switch (true) {
      case header.startsWith(COMPLEXITY_PREFIX): {
        const keyName = header.slice(COMPLEXITY_PREFIX.length);
        const complexity = entry.complexity as
          | Record<string, unknown>
          | undefined;
        row.push(
          complexity && keyName in complexity
            ? JSON.stringify(complexity[keyName])
            : '',
        );
        break;
      }
      case header.startsWith(PERF_PREFIX): {
        const keyName = header.slice(PERF_PREFIX.length);
        const perf = entry.perf as Record<string, unknown> | undefined;
        row.push(perf && keyName in perf ? JSON.stringify(perf[keyName]) : '');
        break;
      }
      case header.startsWith(LINEAGE_PREFIX): {
        const keyName = header.slice(LINEAGE_PREFIX.length);
        const lineage = entry.lineage as Record<string, unknown> | undefined;
        row.push(
          lineage && keyName in lineage ? JSON.stringify(lineage[keyName]) : '',
        );
        break;
      }
      case header.startsWith(DIVERSITY_PREFIX): {
        const keyName = header.slice(DIVERSITY_PREFIX.length);
        const diversity = entry.diversity as
          | Record<string, unknown>
          | undefined;
        row.push(
          diversity && keyName in diversity
            ? JSON.stringify(diversity[keyName])
            : '',
        );
        break;
      }
      case header === HEADER_FRONTS: {
        row.push(
          Array.isArray(entry.fronts) ? JSON.stringify(entry.fronts) : '',
        );
        break;
      }
      case header === HEADER_OPS: {
        row.push(Array.isArray(entry.ops) ? JSON.stringify(entry.ops) : '');
        break;
      }
      case header === HEADER_OBJECTIVES: {
        row.push(
          Array.isArray(entry.objectives)
            ? JSON.stringify(entry.objectives)
            : '',
        );
        break;
      }
      case header === HEADER_OBJ_AGES: {
        row.push(entry.objAges ? JSON.stringify(entry.objAges) : '');
        break;
      }
      case header === HEADER_SPECIES_ALLOC: {
        row.push(
          Array.isArray(entry.speciesAlloc)
            ? JSON.stringify(entry.speciesAlloc)
            : '',
        );
        break;
      }
      case header === HEADER_OBJ_EVENTS: {
        row.push(
          Array.isArray(entry.objEvents) ? JSON.stringify(entry.objEvents) : '',
        );
        break;
      }
      case header === HEADER_OBJ_IMPORTANCE: {
        row.push(
          entry.objImportance ? JSON.stringify(entry.objImportance) : '',
        );
        break;
      }
      default: {
        row.push(
          JSON.stringify((entry as unknown as Record<string, unknown>)[header]),
        );
        break;
      }
    }
  }

  return row.join(',');
}

/**
 * Build the full CSV string for species history entries.
 *
 * Species history exports are row-expanded: each generation snapshot can produce
 * multiple rows, one for each species stat recorded inside that generation. The
 * helper therefore writes a single shared header line and then flattens the
 * nested history structure into one CSV row per species snapshot.
 *
 * The important teaching point is that species history is not a one-row-per-
 * generation export. A generation can contain several contemporaneous species,
 * so the serializer preserves that multiplicity instead of forcing species data
 * into one over-packed cell.
 *
 * @param recentHistory - History entries to export.
 * @param headers - Ordered headers for the export window.
 * @returns Complete CSV payload.
 */
function buildSpeciesHistoryCsv(
  recentHistory: SpeciesHistoryEntry[],
  headers: string[],
): string {
  // Step 1: Materialize the header row.
  const headerLine = headers.join(',');

  // Step 2: Serialize one row per species stat in each generation snapshot.
  const dataLines = recentHistory.flatMap((historyEntry) =>
    historyEntry.stats.map((speciesStat) =>
      serializeSpeciesHistoryRow(
        historyEntry,
        speciesStat,
        headers,
        HEADER_GENERATION,
      ),
    ),
  );

  // Step 3: Join the header and data rows.
  return [headerLine, ...dataLines].join('\n');
}
