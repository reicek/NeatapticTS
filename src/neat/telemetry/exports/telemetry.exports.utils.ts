import type {
  TelemetryEntry,
  NeatLike,
  SpeciesHistoryEntry,
  SpeciesHistoryStat,
} from '../../shared/neat.shared.types';

/**
 * Shared header-discovery and species-row helpers for the telemetry exports chapter.
 *
 * The main exports file owns the user-facing JSONL and CSV entrypoints, while
 * this companion utility file keeps the lower-level bookkeeping focused on two
 * jobs: discover a stable flattened column set for telemetry CSV windows, and
 * normalize species-history rows so early-run exports remain deterministic even
 * when the live controller state is still sparse.
 *
 * In other words, this file is where the exporter does its careful prep work.
 * The public helpers can promise deterministic CSV output because these utility
 * functions first answer the low-level questions: which columns exist in this
 * window, which optional structures actually appeared, which species-history
 * fields must be backfilled, and how should one value become one safe CSV cell.
 */

/**
 * Mutable state container used while collecting telemetry header metadata.
 *
 * The header collector builds this structure incrementally while scanning a
 * telemetry window. It acts as a temporary map from irregular runtime objects
 * to the stable column layout that later CSV rows must follow.
 */
export interface TelemetryHeaderCollectionState {
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
 * Collect base (top-level) telemetry keys for a single entry.
 *
 * This is the first header-discovery pass. It records ordinary top-level fields
 * while deliberately skipping grouped containers that will later be flattened
 * under prefixed columns.
 *
 * @param entry - Telemetry entry to inspect.
 * @param state - Mutable header collection state.
 * @param frontsHeader - Header label for fronts column.
 * @returns void. Mutates `state.baseKeys`.
 */
export function collectBaseKeys(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
  frontsHeader: string,
): void {
  // Step 1: Discover base keys (excluding grouped containers we flatten separately).
  Object.keys(entry).forEach((keyName) => {
    if (
      keyName !== 'complexity' &&
      keyName !== 'perf' &&
      keyName !== 'ops' &&
      keyName !== frontsHeader
    ) {
      state.baseKeys.add(keyName);
    }
  });

  // Step 2: Add fronts as a base key only when it's an array.
  if (Array.isArray(entry.fronts)) state.baseKeys.add(frontsHeader);

  // Step 3: Guarantee rng is surfaced (primitive or object).
  if ('rng' in entry) state.baseKeys.add('rng');
}

/**
 * Collect nested metric keys for grouped telemetry fields.
 *
 * Grouped telemetry objects such as `complexity`, `perf`, and `lineage` become
 * prefixed CSV columns. This helper discovers those nested keys without mixing
 * them into the base top-level header set.
 *
 * @param entry - Telemetry entry to inspect.
 * @param state - Mutable header collection state.
 * @returns void. Mutates complexity/perf/lineage key sets.
 */
export function collectGroupedMetricKeys(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
): void {
  // Step 1: Discover nested group keys for complexity/perf/lineage.
  if (entry.complexity)
    Object.keys(entry.complexity).forEach((keyName) =>
      state.complexityKeys.add(keyName),
    );
  if (entry.perf)
    Object.keys(entry.perf).forEach((keyName) => state.perfKeys.add(keyName));
  if (entry.lineage)
    Object.keys(entry.lineage).forEach((keyName) =>
      state.lineageKeys.add(keyName),
    );
}

/**
 * Collect curated diversity lineage metrics for stable CSV exports.
 *
 * Diversity objects can contain more information than the CSV surface needs.
 * This helper intentionally exports only the lineage-related diversity metrics
 * that are useful and stable enough to deserve fixed columns.
 *
 * @param entry - Telemetry entry to inspect.
 * @param state - Mutable header collection state.
 * @returns void. Mutates diversity lineage key set.
 */
export function collectDiversityLineageMetrics(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
): void {
  // Step 1: Export only curated lineage metrics for stability.
  if (!entry.diversity) return;
  if ('lineageMeanDepth' in entry.diversity)
    state.diversityLineageKeys.add('lineageMeanDepth');
  if ('lineageMeanPairDist' in entry.diversity)
    state.diversityLineageKeys.add('lineageMeanPairDist');
}

/**
 * Collect presence flags for optional telemetry columns.
 *
 * Some telemetry fields are sparse by design. Rather than creating columns for
 * every optional structure unconditionally, the exporter first checks whether a
 * field actually appears in the sampled window and only then enables the
 * corresponding column.
 *
 * @param entry - Telemetry entry to inspect.
 * @param state - Mutable header collection state.
 * @returns void. Mutates optional-column flags.
 */
export function collectOptionalColumnPresence(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
): void {
  // Step 1: Track presence for optional array/map columns.
  if (Array.isArray(entry.ops) && entry.ops.length) state.includeOps = true;
  if (Array.isArray(entry.objectives)) state.includeObjectives = true;
  if (entry.objAges) state.includeObjAges = true;
  if (Array.isArray(entry.speciesAlloc)) state.includeSpeciesAlloc = true;
  if (Array.isArray(entry.objEvents) && entry.objEvents.length)
    state.includeObjEvents = true;
  if (entry.objImportance) state.includeObjImportance = true;
}

/**
 * Ensure the species history array exists on the Neat instance.
 *
 * This helper gives the CSV path one stable backing array to work with, even in
 * early or partially initialized controller states. It keeps later export code
 * focused on serialization instead of defensive host checks.
 *
 * @param neatInstance - Neat instance holding species history.
 * @returns Species history backing array (ensured on instance).
 */
export function ensureSpeciesHistoryArray(
  neatInstance: NeatLike & { _speciesHistory?: SpeciesHistoryEntry[] },
): SpeciesHistoryEntry[] {
  // Step 1: Create the history array if missing.
  if (!Array.isArray(neatInstance._speciesHistory))
    neatInstance._speciesHistory = [];
  return neatInstance._speciesHistory;
}

/**
 * Ensure a minimal species snapshot exists for deterministic CSV headers.
 *
 * Early in a run, live species data can exist before any formal history entry
 * has been recorded. This helper bridges that gap by synthesizing one minimal
 * history snapshot so header discovery and CSV output remain deterministic
 * instead of depending on whether the first archival step has happened yet.
 *
 * @param neatInstance - Neat instance with optional species history and species.
 * @param history - Species history backing array.
 * @param fallbackGeneration - Generation fallback when missing.
 * @param defaultSpeciesId - Default species id when missing.
 * @param defaultSpeciesSize - Default species size when missing.
 * @param defaultBestScore - Default best score when missing.
 * @param defaultLastImproved - Default last improved when missing.
 * @returns void. Mutates history when a minimal snapshot is needed.
 * @example
 * const history = ensureSpeciesHistoryArray(neat);
 * ensureMinimalSpeciesSnapshot(neat, history, 0, -1, 0, 0, 0);
 */
export function ensureMinimalSpeciesSnapshot(
  neatInstance: NeatLike & {
    _speciesHistory?: SpeciesHistoryEntry[];
    _species?: SpeciesHistoryStat[];
    generation?: number;
  },
  history: SpeciesHistoryEntry[],
  fallbackGeneration: number,
  defaultSpeciesId: number,
  defaultSpeciesSize: number,
  defaultBestScore: number,
  defaultLastImproved: number,
): void {
  // Step 1: Only synthesize a snapshot when history is empty and species exist.
  if (
    history.length ||
    !Array.isArray(neatInstance._species) ||
    !neatInstance._species.length
  )
    return;

  // Step 2: Defensive mapping for legacy or incomplete species objects.
  const stats = buildSpeciesHistoryStats(
    neatInstance._species,
    defaultSpeciesId,
    defaultSpeciesSize,
    defaultBestScore,
    defaultLastImproved,
  );

  // Step 3: Push the synthesized snapshot for deterministic headers.
  history.push({
    generation: neatInstance.generation ?? fallbackGeneration,
    stats,
  });
}

/**
 * Collect ordered header keys for species history CSV export.
 *
 * Species history rows can grow new fields over time. This helper scans the
 * selected history window and builds the union of all encountered stat keys so
 * every emitted row follows one shared, predictable column order.
 *
 * @param history - Recent species history entries.
 * @param generationHeader - Header label for generation column.
 * @returns Ordered header list for CSV output.
 */
export function collectSpeciesHistoryHeaders(
  history: SpeciesHistoryEntry[],
  generationHeader: string,
): string[] {
  // Step 1: Seed header set with generation column.
  const headerKeySet = new Set<string>([generationHeader]);

  // Step 2: Discover dynamic species stat keys across history.
  for (const historyEntry of history)
    for (const speciesStat of historyEntry.stats)
      Object.keys(speciesStat).forEach((speciesStatKey) =>
        headerKeySet.add(speciesStatKey),
      );

  // Step 3: Return ordered headers for stable CSV output.
  return Array.from(headerKeySet);
}

/**
 * Normalize raw species records into exportable history stats.
 *
 * Runtime species objects can arrive with slightly different shapes depending
 * on when they were recorded or which legacy path produced them. This helper
 * converts those permissive records into one compact export-ready model with
 * explicit fallbacks for id, size, score, and last-improved generation.
 *
 * @param speciesList - Raw species records to normalize.
 * @param defaultSpeciesId - Default species id when missing.
 * @param defaultSpeciesSize - Default species size when missing.
 * @param defaultBestScore - Default best score when missing.
 * @param defaultLastImproved - Default last improved when missing.
 * @returns Normalized stats for CSV export.
 * @example
 * const stats = buildSpeciesHistoryStats(neat._species ?? [], -1, 0, 0, 0);
 * console.log(stats.length);
 */
export function buildSpeciesHistoryStats(
  speciesList: SpeciesHistoryStat[],
  defaultSpeciesId: number,
  defaultSpeciesSize: number,
  defaultBestScore: number,
  defaultLastImproved: number,
): SpeciesHistoryStat[] {
  // Step 1: Normalize to a permissive record shape for legacy compatibility.
  const speciesRecords = speciesList as unknown as Record<string, unknown>[];
  /** Accumulator for normalized species history stats. */
  const stats: SpeciesHistoryStat[] = [];

  // Step 2: Extract fields with explicit fallbacks.
  for (const speciesRecord of speciesRecords) {
    const id = resolveSpeciesId(speciesRecord, defaultSpeciesId);
    const size = resolveSpeciesSize(speciesRecord, defaultSpeciesSize);
    const bestScore = resolveSpeciesBestScore(speciesRecord, defaultBestScore);
    const lastImproved = resolveSpeciesLastImproved(
      speciesRecord,
      defaultLastImproved,
    );
    stats.push({ id, size, bestScore, lastImproved });
  }

  // Step 3: Return normalized stats.
  return stats;

  /**
   * @param speciesRecord - Raw species record.
   * @param fallbackId - Default id when missing.
   * @returns Resolved species id with fallback.
   */
  function resolveSpeciesId(
    speciesRecord: Record<string, unknown>,
    fallbackId: number,
  ): number {
    // Step 1: Prefer explicit id when numeric.
    const idValue = readNumericProperty(speciesRecord, 'id');
    if (idValue !== undefined) return idValue;

    // Step 2: Apply default fallback.
    return fallbackId;
  }

  /**
   * @param speciesRecord - Raw species record.
   * @param fallbackSize - Default size when missing.
   * @returns Resolved species size with fallback.
   */
  function resolveSpeciesSize(
    speciesRecord: Record<string, unknown>,
    fallbackSize: number,
  ): number {
    // Step 1: Prefer members length when present.
    const membersSize = readMembersSize(speciesRecord);
    if (membersSize !== undefined) return membersSize;

    // Step 2: Fall back to explicit size when numeric.
    const sizeValue = readNumericProperty(speciesRecord, 'size');
    if (sizeValue !== undefined) return sizeValue;

    // Step 3: Default fallback.
    return fallbackSize;
  }

  /**
   * @param speciesRecord - Raw species record.
   * @param fallbackBestScore - Default best score when missing.
   * @returns Resolved best score with fallback.
   */
  function resolveSpeciesBestScore(
    speciesRecord: Record<string, unknown>,
    fallbackBestScore: number,
  ): number {
    // Step 1: Prefer bestScore when numeric.
    const bestScoreValue = readNumericProperty(speciesRecord, 'bestScore');
    if (bestScoreValue !== undefined) return bestScoreValue;

    // Step 2: Fall back to legacy best when numeric.
    const bestValue = readNumericProperty(speciesRecord, 'best');
    if (bestValue !== undefined) return bestValue;

    // Step 3: Default fallback.
    return fallbackBestScore;
  }

  /**
   * @param speciesRecord - Raw species record.
   * @param fallbackLastImproved - Default last improved when missing.
   * @returns Resolved last improved with fallback.
   */
  function resolveSpeciesLastImproved(
    speciesRecord: Record<string, unknown>,
    fallbackLastImproved: number,
  ): number {
    // Step 1: Prefer explicit lastImproved when numeric.
    const lastImprovedValue = readNumericProperty(
      speciesRecord,
      'lastImproved',
    );
    if (lastImprovedValue !== undefined) return lastImprovedValue;

    // Step 2: Default fallback.
    return fallbackLastImproved;
  }

  /**
   * @param speciesRecord - Raw species record.
   * @returns Members length when present.
   */
  function readMembersSize(
    speciesRecord: Record<string, unknown>,
  ): number | undefined {
    // Step 1: Ensure members is an array before reading length.
    const membersValue = speciesRecord.members;
    if (Array.isArray(membersValue)) return membersValue.length;
    return undefined;
  }

  /**
   * @param speciesRecord - Raw species record.
   * @param propertyName - Property to read.
   * @returns Numeric property value when present.
   */
  function readNumericProperty(
    speciesRecord: Record<string, unknown>,
    propertyName: string,
  ): number | undefined {
    // Step 1: Return numeric values only.
    const candidateValue = speciesRecord[propertyName];
    if (typeof candidateValue === 'number') return candidateValue;
    return undefined;
  }
}

/**
 * Serialize one species history row using the provided headers.
 *
 * Species history CSV output is built row by row from one generation snapshot
 * plus one species stat record. This helper guarantees that each row follows
 * the same previously collected header order, even when some species records
 * omit optional fields.
 *
 * @param historyEntry - A single generation snapshot.
 * @param speciesStat - A single species stat record for that generation.
 * @param orderedHeaders - Ordered header list for stable CSV.
 * @param generationHeader - Column header name for generation.
 * @returns CSV row string matching the provided header order.
 */
export function serializeSpeciesHistoryRow(
  historyEntry: SpeciesHistoryEntry,
  speciesStat: SpeciesHistoryStat,
  orderedHeaders: string[],
  generationHeader: string,
): string {
  // Step 1: Convert each header to its corresponding cell value.
  const cells = orderedHeaders.map((headerName) =>
    resolveSpeciesHistoryCellValue(
      historyEntry,
      speciesStat,
      headerName,
      generationHeader,
    ),
  );

  // Step 2: Join the row cells.
  return cells.join(',');
}

/**
 * Resolve a single species history cell value for the provided header.
 *
 * The generation column is special because it lives on the outer history entry,
 * not on the species stat itself. All remaining headers are treated as dynamic
 * species-stat fields and are read through the same record-style lookup path.
 *
 * @param historyEntry - A single generation snapshot.
 * @param speciesStat - A single species stat record.
 * @param headerName - Column header name.
 * @param generationHeader - Column header name for generation.
 * @returns Serialized cell (JSON) or empty string for missing values.
 */
export function resolveSpeciesHistoryCellValue(
  historyEntry: SpeciesHistoryEntry,
  speciesStat: SpeciesHistoryStat,
  headerName: string,
  generationHeader: string,
): string {
  // Step 1: Generation is stored on the history entry, not the species stat.
  if (headerName === generationHeader)
    return safeStringifyCell(historyEntry.generation);

  // Step 2: All other headers are treated as dynamic species stat fields.
  const speciesStatRecord = speciesStat as unknown as Record<string, unknown>;
  return safeStringifyCell(speciesStatRecord[headerName]);
}

/**
 * Serialize a CSV cell with JSON.stringify safeguards.
 *
 * The exporter uses JSON stringification for cells so numbers, strings, arrays,
 * and nested values all flow through one escaping path. Undefined values remain
 * empty cells to preserve the exporter’s existing sparse-column behavior.
 *
 * @param value - Any value to stringify.
 * @returns JSON string or empty string when JSON.stringify returns undefined.
 */
export function safeStringifyCell(value: unknown): string {
  // Step 1: Preserve the existing "undefined -> empty cell" CSV behavior.
  return JSON.stringify(value) ?? '';
}
