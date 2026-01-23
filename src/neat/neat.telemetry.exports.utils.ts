import type {
  TelemetryEntry,
  NeatLike,
  SpeciesHistoryEntry,
  SpeciesHistoryStat,
} from './neat.types';

/**
 * Mutable state container used while collecting telemetry header metadata.
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
 * @param neatInstance - Neat instance with optional species history and species.
 * @param history - Species history backing array.
 * @param fallbackGeneration - Generation fallback when missing.
 * @param defaultSpeciesId - Default species id when missing.
 * @param defaultSpeciesSize - Default species size when missing.
 * @param defaultBestScore - Default best score when missing.
 * @param defaultLastImproved - Default last improved when missing.
 * @returns void. Mutates history when a minimal snapshot is needed.
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
 * @param speciesList - Raw species records to normalize.
 * @param defaultSpeciesId - Default species id when missing.
 * @param defaultSpeciesSize - Default species size when missing.
 * @param defaultBestScore - Default best score when missing.
 * @param defaultLastImproved - Default last improved when missing.
 * @returns Normalized stats for CSV export.
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
 * @param value - Any value to stringify.
 * @returns JSON string or empty string when JSON.stringify returns undefined.
 */
export function safeStringifyCell(value: unknown): string {
  // Step 1: Preserve the existing "undefined -> empty cell" CSV behavior.
  return JSON.stringify(value) ?? '';
}
