import type {
  ConnectionLike,
  GenomeDetailed,
  NeatOptions,
  SpeciesHistoryEntry,
  SpeciesHistoryStatExtended,
  SpeciesLike,
} from './neat.types';

/** Default innovation id when none is present. */
export const SPECIES_HISTORY_DEFAULT_INNOVATION_ID = 0;
/** Default innovation range when data is missing. */
export const SPECIES_HISTORY_DEFAULT_INNOVATION_RANGE = 0;
/** Default enabled ratio when no connections exist. */
export const SPECIES_HISTORY_DEFAULT_ENABLED_RATIO = 0;
/** Shared zero value for counters and defaults. */
export const SPECIES_HISTORY_ZERO = 0;
/** Initial max tracker for innovation range aggregation. */
export const SPECIES_HISTORY_INITIAL_MAX_INNOVATION = -Infinity;
/** Initial min tracker for innovation range aggregation. */
export const SPECIES_HISTORY_INITIAL_MIN_INNOVATION = Infinity;

/**
 * @param options - Current Neat options.
 * @returns True when extended history is enabled.
 */
export function shouldAugmentExtendedHistory(
  options: NeatOptions | undefined,
): boolean {
  // Step 1: Read the extended-history flag safely.
  return Boolean(options?.speciesAllocation?.extendedHistory);
}

/**
 * @param history - Recorded history to enrich in place.
 * @param context - Neat instance context for lookups.
 */
export function backfillExtendedHistory(
  history: SpeciesHistoryEntry[],
  context: {
    _species?: SpeciesLike[];
    _fallbackInnov?: (c: ConnectionLike) => number;
  },
): void {
  // Step 1: Iterate over each generation snapshot.
  for (const generationEntry of history) {
    // Step 2: Enrich each per-species stat where fields are missing.
    for (const speciesStat of generationEntry.stats as SpeciesHistoryStatExtended[]) {
      if (hasExtendedFields(speciesStat)) {
        continue;
      }

      const speciesRecord = findSpeciesById(
        context._species || [],
        speciesStat.id,
      );
      if (!speciesRecord?.members?.length) {
        continue;
      }

      const summary = summarizeSpeciesConnections(
        speciesRecord.members as GenomeDetailed[],
        context._fallbackInnov,
      );

      applyExtendedSummary(speciesStat, summary);
    }
  }

  /**
   * @param speciesStat - Per-species stat entry.
   * @returns True when both extended fields exist.
   */
  function hasExtendedFields(speciesStat: SpeciesHistoryStatExtended): boolean {
    // Step 1: Check for the presence of extended properties.
    return 'innovationRange' in speciesStat && 'enabledRatio' in speciesStat;
  }

  /**
   * @param speciesList - Species records to search.
   * @param speciesId - Identifier to match.
   * @returns Matching species or undefined.
   */
  function findSpeciesById(
    speciesList: SpeciesLike[],
    speciesId: number,
  ): SpeciesLike | undefined {
    // Step 1: Locate the matching species record.
    return speciesList.find((speciesRecord) => speciesRecord.id === speciesId);
  }

  /**
   * @param members - Member genomes to summarize.
   * @param fallbackInnov - Optional innovation fallback extractor.
   * @returns Extended summary metrics derived from members.
   */
  function summarizeSpeciesConnections(
    members: GenomeDetailed[],
    fallbackInnov: ((c: ConnectionLike) => number) | undefined,
  ): { innovationRange: number; enabledRatio: number } {
    // Step 1: Initialize aggregation values.
    let maxInnovation = SPECIES_HISTORY_INITIAL_MAX_INNOVATION;
    let minInnovation = SPECIES_HISTORY_INITIAL_MIN_INNOVATION;
    let enabledCount = SPECIES_HISTORY_ZERO;
    let disabledCount = SPECIES_HISTORY_ZERO;

    // Step 2: Aggregate innovation range and enabled counts.
    for (const member of members) {
      const connections = member.connections as ConnectionLike[];
      for (const connection of connections) {
        const innovationId =
          connection.innovation ??
          fallbackInnov?.(connection) ??
          SPECIES_HISTORY_DEFAULT_INNOVATION_ID;

        if (innovationId > maxInnovation) maxInnovation = innovationId;
        if (innovationId < minInnovation) minInnovation = innovationId;

        if (connection.enabled === false) {
          disabledCount++;
        } else {
          enabledCount++;
        }
      }
    }

    // Step 3: Fold aggregates into the summary object.
    return {
      innovationRange:
        isFinite(maxInnovation) &&
        isFinite(minInnovation) &&
        maxInnovation > minInnovation
          ? maxInnovation - minInnovation
          : SPECIES_HISTORY_DEFAULT_INNOVATION_RANGE,
      enabledRatio:
        enabledCount + disabledCount
          ? enabledCount / (enabledCount + disabledCount)
          : SPECIES_HISTORY_DEFAULT_ENABLED_RATIO,
    };
  }

  /**
   * @param speciesStat - Target stat entry to mutate.
   * @param summary - Computed extended metrics.
   */
  function applyExtendedSummary(
    speciesStat: SpeciesHistoryStatExtended,
    summary: { innovationRange: number; enabledRatio: number },
  ): void {
    // Step 1: Assign computed values into the stat record.
    (speciesStat as SpeciesHistoryStatExtended).innovationRange =
      summary.innovationRange;
    (speciesStat as SpeciesHistoryStatExtended).enabledRatio =
      summary.enabledRatio;
  }
}
