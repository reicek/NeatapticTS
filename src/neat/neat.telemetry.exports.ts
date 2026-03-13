import type {
  NeatLike,
  TelemetryEntry,
  SpeciesHistoryStat,
  SpeciesHistoryEntry,
} from './neat.types';
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
} from './neat.telemetry.exports.utils';

/**
 * Shape describing collected telemetry header discovery info.
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
const COMPLEXITY_PREFIX = 'complexity.'; // complexity.* flattened headers
/** Group prefix for performance nested metrics when flattened. */
const PERF_PREFIX = 'perf.'; // perf.* flattened headers
/** Group prefix for lineage nested metrics when flattened. */
const LINEAGE_PREFIX = 'lineage.'; // lineage.* flattened headers
/** Group prefix for diversity nested metrics when flattened. */
const DIVERSITY_PREFIX = 'diversity.'; // diversity.* flattened headers

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

/** Default max entries for species history CSV exports. */
export const DEFAULT_SPECIES_HISTORY_MAX_ENTRIES = 200;
/** Default fallback species id when missing. */
export const DEFAULT_SPECIES_ID = -1;
/** Default fallback species size when missing. */
export const DEFAULT_SPECIES_SIZE = 0;
/** Default fallback best score when missing. */
export const DEFAULT_SPECIES_BEST_SCORE = 0;
/** Default fallback last improved when missing. */
export const DEFAULT_SPECIES_LAST_IMPROVED = 0;
/** Default fallback generation when missing. */
export const DEFAULT_SPECIES_HISTORY_GENERATION = 0;

/**
 * Telemetry export helpers extracted from `neat.ts`.
 *
 * This module exposes small helpers intended to serialize the internal
 * telemetry gathered by the NeatapticTS `Neat` runtime into common
 * data-export formats (JSONL and CSV). The functions intentionally
 * operate against `this` so they can be attached to instances.
 */
export function exportTelemetryJSONL(
  this: NeatLike & { _telemetry: TelemetryEntry[] },
): string {
  /**
   * Serialize the internal telemetry array to JSON Lines (JSONL).
   * Each telemetry entry is stringified and separated by a newline.
   *
   * Example:
   * ```ts
   * // Attach to a neat instance and call:
   * const jsonl = neatInstance.exportTelemetryJSONL();
   * // jsonl now contains one JSON object per line
   * ```
   *
   * Notes for docs: JSONL is useful for streaming telemetry into
   * log processors and line-based parsers. Each line is independent
   * and can be parsed with JSON.parse.
   */
  return this._telemetry
    .map((entry: TelemetryEntry) => JSON.stringify(entry))
    .join('\n');
}

/**
 * Export recent telemetry entries to a CSV string.
 *
 * Responsibilities:
 * - Collect a bounded slice (`maxEntries`) of recent telemetry records.
 * - Discover and flatten dynamic header keys (top-level + grouped metrics).
 * - Serialize each entry into a CSV row with stable, parseable values.
 *
 * Flattening Rules:
 * - Nested groups (complexity, perf, lineage, diversity) become group.key columns.
 * - Optional arrays/maps (ops, objectives, objAges, speciesAlloc, objEvents, objImportance, fronts) included only if present.
 *
 * @param this Neat instance (expects `_telemetry` array field).
 * @param maxEntries Maximum number of most recent telemetry entries to include (default 500).
 * @returns CSV string (headers + rows) or empty string when no telemetry.
 */
export function exportTelemetryCSV(
  this: NeatLike & { _telemetry: TelemetryEntry[] },
  maxEntries = 500,
): string {
  /**
   * Recent telemetry entries to export. Contains at most `maxEntries` items.
   */
  const recentTelemetry: TelemetryEntry[] = Array.isArray(this._telemetry)
    ? this._telemetry.slice(-maxEntries)
    : [];
  if (!recentTelemetry.length) return '';

  // 1. Collect structural + header metadata across entries
  /** Metadata describing all discovered headers across sampled entries. */
  const headerInfo = collectTelemetryHeaderInfo(recentTelemetry);

  // 2. Materialize header list (ordered) from collected metadata
  /** Ordered list of CSV header names (flattened). */
  const headers = buildTelemetryHeaders(headerInfo);

  // 3. Serialize: header row + data rows
  /** Accumulator of CSV lines starting with the header row. */
  const csvLines: string[] = [headers.join(',')];

  for (const telemetryEntry of recentTelemetry) {
    csvLines.push(serializeTelemetryEntry(telemetryEntry, headers));
  }

  return csvLines.join('\n');
}

/**
 * Collect header metadata from the raw telemetry entries.
 * - Discovers base (top‑level) keys excluding grouped objects.
 * - Discovers nested keys inside complexity, perf, lineage, diversity groups.
 * - Tracks presence of optional multi-value structures (ops, objectives, etc.).
 */
function collectTelemetryHeaderInfo(
  entries: TelemetryEntry[],
): TelemetryHeaderInfo {
  /** Accumulator for header metadata and optional column flags. */
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

  // Step 2: Fold all collected metadata into a single info object.
  return headerState;
}

/**
 * Build the ordered list of CSV headers from collected metadata.
 * Flattened nested metrics are emitted using group prefixes (group.key).
 */
function buildTelemetryHeaders(info: TelemetryHeaderInfo): string[] {
  /** Aggregated headers list (ordered). */
  const headers: string[] = [
    ...info.baseKeys,
    ...[...info.complexityKeys].map((k) => `${COMPLEXITY_PREFIX}${k}`),
    ...[...info.perfKeys].map((k) => `${PERF_PREFIX}${k}`),
    ...[...info.lineageKeys].map((k) => `${LINEAGE_PREFIX}${k}`),
    ...[...info.diversityLineageKeys].map((k) => `${DIVERSITY_PREFIX}${k}`),
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
 * Serialize one telemetry entry into a CSV row using previously computed headers.
 * Uses a `switch(true)` pattern instead of a long if/else chain to reduce
 * cognitive complexity while preserving readability of each scenario.
 */
function serializeTelemetryEntry(
  entry: TelemetryEntry,
  headers: string[],
): string {
  /** Accumulator for serialized cell values for one telemetry row. */
  const row: string[] = [];
  for (const header of headers) {
    switch (true) {
      // Grouped complexity metrics
      case header.startsWith(COMPLEXITY_PREFIX): {
        // Complexity metrics describe structural attributes of evolved networks
        // (e.g., node counts, connection counts, depth). We flatten them as
        // complexity.<metric>. Missing metrics serialize as an empty cell.
        const key = header.slice(COMPLEXITY_PREFIX.length);
        // TypeScript: safe dynamic access
        const complexity = entry.complexity as
          | Record<string, unknown>
          | undefined;
        row.push(
          complexity && key in complexity
            ? JSON.stringify(complexity[key])
            : '',
        );
        break;
      }
      // Grouped performance metrics
      case header.startsWith(PERF_PREFIX): {
        const key = header.slice(PERF_PREFIX.length);
        const perf = entry.perf as Record<string, unknown> | undefined;
        row.push(perf && key in perf ? JSON.stringify(perf[key]) : '');
        break;
      }
      // Grouped lineage metrics
      case header.startsWith(LINEAGE_PREFIX): {
        const key = header.slice(LINEAGE_PREFIX.length);
        const lineage = entry.lineage as Record<string, unknown> | undefined;
        row.push(lineage && key in lineage ? JSON.stringify(lineage[key]) : '');
        break;
      }
      // Grouped diversity metrics
      case header.startsWith(DIVERSITY_PREFIX): {
        const key = header.slice(DIVERSITY_PREFIX.length);
        const diversity = entry.diversity as
          | Record<string, unknown>
          | undefined;
        row.push(
          diversity && key in diversity ? JSON.stringify(diversity[key]) : '',
        );
        break;
      }
      // Array-like and optional multi-value columns
      case header === HEADER_FRONTS: {
        // fronts: Pareto fronts (multi-objective optimization). Each element
        // is typically an index set or representation of a front. Serialized
        // as JSON array for downstream MOEA visualization.
        row.push(
          Array.isArray(entry.fronts) ? JSON.stringify(entry.fronts) : '',
        );
        break;
      }
      case header === HEADER_OPS: {
        // ops: chronological list of evolutionary operations executed during
        // the generation (mutations, crossovers, pruning, etc.). Enables
        // audit and frequency analysis of algorithmic behaviors.
        row.push(Array.isArray(entry.ops) ? JSON.stringify(entry.ops) : '');
        break;
      }
      case header === HEADER_OBJECTIVES: {
        // objectives: current scalar objective scores (single or multi‑objective)
        // maintained for the individual / population snapshot. Represented as
        // JSON array to keep numeric precision and ordering.
        row.push(
          Array.isArray(entry.objectives)
            ? JSON.stringify(entry.objectives)
            : '',
        );
        break;
      }
      case header === HEADER_OBJ_AGES: {
        // objAges: age (iterations since last improvement) per objective.
        // Helps scheduling adaptive pressure or annealing strategies.
        row.push(entry.objAges ? JSON.stringify(entry.objAges) : '');
        break;
      }
      case header === HEADER_SPECIES_ALLOC: {
        // speciesAlloc: allocation proportions / counts assigned to species
        // for reproduction. Valuable for diagnosing speciation balancing.
        row.push(
          Array.isArray(entry.speciesAlloc)
            ? JSON.stringify(entry.speciesAlloc)
            : '',
        );
        break;
      }
      case header === HEADER_OBJ_EVENTS: {
        // objEvents: timeline of objective-related events (e.g., dominance
        // shifts, re-weighting). Provides temporal context to objective trends.
        row.push(
          Array.isArray(entry.objEvents) ? JSON.stringify(entry.objEvents) : '',
        );
        break;
      }
      case header === HEADER_OBJ_IMPORTANCE: {
        // objImportance: dynamic importance weights per objective applied by
        // adaptive multi-objective strategies; used for post-hoc analysis of
        // weight schedules.
        row.push(
          entry.objImportance ? JSON.stringify(entry.objImportance) : '',
        );
        break;
      }
      // Default: treat as top-level column
      default: {
        // All remaining headers correspond to primitive / object top‑level
        // properties (e.g., generation, population size, best score). Use
        // JSON.stringify so objects/arrays stay parseable and commas safe.
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
 * Export species history snapshots to CSV.
 *
 * Each row represents a single species at a specific generation; the generation
 * value is repeated per species. Dynamically discovers species stat keys so
 * custom metadata added at runtime is preserved.
 *
 * Behavior:
 * - If `_speciesHistory` is absent/empty but `_species` exists, synthesizes a
 *   minimal snapshot to ensure deterministic headers early in a run.
 * - Returns a header-only CSV when there is no history or species.
 *
 * @param this Neat instance (expects `_speciesHistory` and optionally `_species`).
 * @param maxEntries Maximum number of most recent history snapshots (generations) to include (default 200).
 * @returns CSV string (headers + rows) describing species evolution timeline.
 */
export function exportSpeciesHistoryCSV(
  this: NeatLike & {
    _speciesHistory?: SpeciesHistoryEntry[];
    _species?: SpeciesHistoryStat[];
    generation?: number;
  },
  maxEntries = DEFAULT_SPECIES_HISTORY_MAX_ENTRIES,
): string {
  /**
   * Ensure the species history structure exists on the instance.
   * Returned array is the canonical backing store used throughout export.
   */
  const speciesHistory = ensureSpeciesHistoryArray(this);

  /**
   * If species history is empty but species are present, create a minimal
   * snapshot so the CSV exporter still produces a header row. This helps
   * early debugging and deterministic exports before speciation/evolution
   * has run.
   */
  ensureMinimalSpeciesSnapshot(
    this,
    speciesHistory,
    DEFAULT_SPECIES_HISTORY_GENERATION,
    DEFAULT_SPECIES_ID,
    DEFAULT_SPECIES_SIZE,
    DEFAULT_SPECIES_BEST_SCORE,
    DEFAULT_SPECIES_LAST_IMPROVED,
  );

  /** Recent slice of the species history we will export. */
  const recentHistory: SpeciesHistoryEntry[] =
    speciesHistory.slice(-maxEntries);
  if (!recentHistory.length) {
    // Emit header-only CSV for deterministic empty export
    return 'generation,id,size,best,lastImproved';
  }

  /** Final ordered header list for CSV output. */
  const headers = collectSpeciesHistoryHeaders(
    recentHistory,
    HEADER_GENERATION,
  );

  // Delegate CSV line materialization to helper for readability & testability
  return buildSpeciesHistoryCsv(recentHistory, headers);
}

/**
 * Build the full CSV string for species history given ordered headers and
 * a slice of history entries.
 *
 * Implementation notes:
 * - The history is a 2‑level structure (generation entry -> species stats[]).
 * - We emit one CSV row per species stat, repeating the generation value.
 * - Values are JSON.stringify'd to remain safe for commas/quotes.
 */
function buildSpeciesHistoryCsv(
  recentHistory: SpeciesHistoryEntry[],
  headers: string[],
): string {
  // Step 1: Materialize the header row.
  const headerLine = headers.join(',');

  // Step 2: Collect all data rows as strings.
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

  // Step 3: Fold into a final CSV string.
  return [headerLine, ...dataLines].join('\n');
}
