/**
 * Telemetry export helpers extracted from `neat.ts`.
 *
 * This module exposes small helpers intended to serialize the internal
 * telemetry gathered by the NeatapticTS `Neat` runtime into common
 * data-export formats (JSONL and CSV). The functions intentionally
 * operate against `this` so they can be attached to instances.
 */
export function exportTelemetryJSONL(this: any): string {
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
  return this._telemetry.map((entry: any) => JSON.stringify(entry)).join('\n');
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
export function exportTelemetryCSV(this: any, maxEntries = 500): string {
  /**
   * Recent telemetry entries to export. Contains at most `maxEntries` items.
   */
  const recentTelemetry = Array.isArray(this._telemetry)
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

/**
 * Collect header metadata from the raw telemetry entries.
 * - Discovers base (top‑level) keys excluding grouped objects.
 * - Discovers nested keys inside complexity, perf, lineage, diversity groups.
 * - Tracks presence of optional multi-value structures (ops, objectives, etc.).
 */
function collectTelemetryHeaderInfo(entries: any[]): TelemetryHeaderInfo {
  /** Discovered base keys (excluding grouped containers). */
  const baseKeys = new Set<string>();
  /** Discovered complexity metric keys. */
  const complexityKeys = new Set<string>();
  /** Discovered performance metric keys. */
  const perfKeys = new Set<string>();
  /** Discovered lineage metric keys. */
  const lineageKeys = new Set<string>();
  /** Selected diversity lineage metric keys. */
  const diversityLineageKeys = new Set<string>();

  /** Presence: operations array. */
  let includeOps = false;
  /** Presence: objectives array. */
  let includeObjectives = false;
  /** Presence: objective ages map. */
  let includeObjAges = false;
  /** Presence: species allocation array. */
  let includeSpeciesAlloc = false;
  /** Presence: objective events array. */
  let includeObjEvents = false;
  /** Presence: objective importance map. */
  let includeObjImportance = false;

  for (const entry of entries) {
    // (A) Discover base keys (excluding grouped containers we flatten separately)
    Object.keys(entry).forEach((k) => {
      if (
        k !== 'complexity' &&
        k !== 'perf' &&
        k !== 'ops' &&
        k !== HEADER_FRONTS
      ) {
        baseKeys.add(k);
      }
    });

    // (B) Add fronts as a base key only when it's an array
    if (Array.isArray(entry.fronts)) baseKeys.add(HEADER_FRONTS);

    // (C) Discover nested group keys
    if (entry.complexity)
      Object.keys(entry.complexity).forEach((k) => complexityKeys.add(k));
    if (entry.perf) Object.keys(entry.perf).forEach((k) => perfKeys.add(k));
    if (entry.lineage)
      Object.keys(entry.lineage).forEach((k) => lineageKeys.add(k));

    // (D) Diversity: export only curated lineage metrics for stability
    if (entry.diversity) {
      if ('lineageMeanDepth' in entry.diversity)
        diversityLineageKeys.add('lineageMeanDepth');
      if ('lineageMeanPairDist' in entry.diversity)
        diversityLineageKeys.add('lineageMeanPairDist');
    }

    // (E) Guarantee rng is surfaced (primitive or object)
    if ('rng' in entry) baseKeys.add('rng');

    // (F) Presence tracking for optional array/map columns
    if (Array.isArray(entry.ops) && entry.ops.length) includeOps = true;
    if (Array.isArray(entry.objectives)) includeObjectives = true;
    if (entry.objAges) includeObjAges = true;
    if (Array.isArray(entry.speciesAlloc)) includeSpeciesAlloc = true;
    if (Array.isArray(entry.objEvents) && entry.objEvents.length)
      includeObjEvents = true;
    if (entry.objImportance) includeObjImportance = true;
  }

  return {
    baseKeys,
    complexityKeys,
    perfKeys,
    lineageKeys,
    diversityLineageKeys,
    includeOps,
    includeObjectives,
    includeObjAges,
    includeSpeciesAlloc,
    includeObjEvents,
    includeObjImportance,
  };
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
function serializeTelemetryEntry(entry: any, headers: string[]): string {
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
        row.push(
          entry.complexity && key in entry.complexity
            ? JSON.stringify(entry.complexity[key])
            : ''
        );
        break;
      }
      // Grouped performance metrics
      case header.startsWith(PERF_PREFIX): {
        // Performance (perf.*) captures runtime / evaluation timing or cost
        // figures (e.g., ms per generation, fitness evaluation cost). Allows
        // profiling & trend analysis. Empty when metric not present.
        const key = header.slice(PERF_PREFIX.length);
        row.push(
          entry.perf && key in entry.perf ? JSON.stringify(entry.perf[key]) : ''
        );
        break;
      }
      // Grouped lineage metrics
      case header.startsWith(LINEAGE_PREFIX): {
        // Lineage metrics (lineage.*) reflect genealogical statistics such as
        // ancestor depth, branch factors, or identifiers helpful for tracing
        // evolutionary history.
        const key = header.slice(LINEAGE_PREFIX.length);
        row.push(
          entry.lineage && key in entry.lineage
            ? JSON.stringify(entry.lineage[key])
            : ''
        );
        break;
      }
      // Grouped diversity metrics
      case header.startsWith(DIVERSITY_PREFIX): {
        // Diversity metrics (diversity.*) quantify population variety to guard
        // against premature convergence (e.g., mean lineage depth / pairwise
        // distance). Only curated subset exported for header stability.
        const key = header.slice(DIVERSITY_PREFIX.length);
        row.push(
          entry.diversity && key in entry.diversity
            ? JSON.stringify(entry.diversity[key])
            : ''
        );
        break;
      }
      // Array-like and optional multi-value columns
      case header === HEADER_FRONTS: {
        // fronts: Pareto fronts (multi-objective optimization). Each element
        // is typically an index set or representation of a front. Serialized
        // as JSON array for downstream MOEA visualization.
        row.push(
          Array.isArray(entry.fronts) ? JSON.stringify(entry.fronts) : ''
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
            : ''
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
            : ''
        );
        break;
      }
      case header === HEADER_OBJ_EVENTS: {
        // objEvents: timeline of objective-related events (e.g., dominance
        // shifts, re-weighting). Provides temporal context to objective trends.
        row.push(
          Array.isArray(entry.objEvents) ? JSON.stringify(entry.objEvents) : ''
        );
        break;
      }
      case header === HEADER_OBJ_IMPORTANCE: {
        // objImportance: dynamic importance weights per objective applied by
        // adaptive multi-objective strategies; used for post-hoc analysis of
        // weight schedules.
        row.push(
          entry.objImportance ? JSON.stringify(entry.objImportance) : ''
        );
        break;
      }
      // Default: treat as top-level column
      default: {
        // All remaining headers correspond to primitive / object top‑level
        // properties (e.g., generation, population size, best score). Use
        // JSON.stringify so objects/arrays stay parseable and commas safe.
        row.push(JSON.stringify(entry[header]));
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
export function exportSpeciesHistoryCSV(this: any, maxEntries = 200): string {
  /** Ensure the species history structure exists on the instance. */
  if (!Array.isArray(this._speciesHistory)) this._speciesHistory = [];

  /**
   * If species history is empty but species are present, create a minimal
   * snapshot so the CSV exporter still produces a header row. This helps
   * early debugging and deterministic exports before speciation/evolution
   * has run.
   */
  if (
    !this._speciesHistory.length &&
    Array.isArray(this._species) &&
    this._species.length
  ) {
    // Create a minimal snapshot on demand so early exports (before evolve/speciate) still yield a header row
    const stats = this._species.map((sp: any) => ({
      /** Unique identifier for the species (or -1 when missing). */
      id: sp.id ?? -1,
      /** Current size (number of members) in the species. */
      size: Array.isArray(sp.members) ? sp.members.length : 0,
      /** Best score observed in the species (fallback 0). */
      best: sp.bestScore ?? 0,
      /** Generation index when the species last improved (fallback 0). */
      lastImproved: sp.lastImproved ?? 0,
    }));
    this._speciesHistory.push({ generation: this.generation || 0, stats });
  }

  /** Recent slice of the species history we will export. */
  const recentHistory = this._speciesHistory.slice(-maxEntries);
  if (!recentHistory.length) {
    // Emit header-only CSV for deterministic empty export
    return 'generation,id,size,best,lastImproved';
  }

  /** Set of discovered keys to use as headers; starts with `generation`. */
  const headerKeySet = new Set<string>(['generation']);
  for (const entry of recentHistory)
    for (const speciesStat of entry.stats)
      Object.keys(speciesStat).forEach((k) => headerKeySet.add(k));

  /** Final ordered header list for CSV output. */
  const headers = Array.from(headerKeySet);

  // Delegate CSV line materialization to helper for readability & testability
  return buildSpeciesHistoryCsv(recentHistory, headers);
}

/** Header label for generation column in species history CSV. */
const HEADER_GENERATION = 'generation';

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
  recentHistory: Array<{ generation: number; stats: any[] }>,
  headers: string[]
): string {
  /** Accumulates lines; seeded with header row. */
  const lines: string[] = [headers.join(',')];
  // Iterate each generation snapshot
  for (const historyEntry of recentHistory) {
    // Each species stat becomes its own CSV data row
    for (const speciesStat of historyEntry.stats) {
      /** Cell accumulator for a single row. */
      const rowCells: string[] = [];
      // Maintain header order while extracting values
      for (const header of headers) {
        // Special-case generation (lives on outer entry rather than per species)
        if (header === HEADER_GENERATION) {
          rowCells.push(JSON.stringify(historyEntry.generation));
          continue;
        }
        // Generic species stat field (may be undefined -> serialized as undefined)
        rowCells.push(JSON.stringify((speciesStat as any)[header]));
      }
      lines.push(rowCells.join(','));
    }
  }
  return lines.join('\n');
}
