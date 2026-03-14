import type {
  ConnectionLike,
  GenomeDetailed,
  NeatOptions,
  SpeciesHistoryEntry,
  SpeciesHistoryStatExtended,
  SpeciesLike,
} from '../../neat.types';

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
 * Species-history augmentation mechanics used by NEAT reporting helpers.
 *
 * This chapter holds the opt-in extended-history backfill logic that derives
 * innovation-range and enabled-ratio summaries from the current species state.
 */

/**
 * Check whether extended species history should be augmented.
 *
 * @param options - Current NEAT options.
 * @returns `true` when extended history is enabled.
 */
export function shouldAugmentExtendedHistory(
  options: NeatOptions | undefined,
): boolean {
  return Boolean(options?.speciesAllocation?.extendedHistory);
}

/**
 * Backfill missing extended history fields in place.
 *
 * @param history - Recorded species history to enrich.
 * @param context - NEAT context exposing current species and optional fallback innovations.
 * @returns Nothing. The history entries are mutated in place when backfill succeeds.
 */
export function backfillExtendedHistory(
  history: SpeciesHistoryEntry[],
  context: {
    _species?: SpeciesLike[];
    _fallbackInnov?: (connection: ConnectionLike) => number;
  },
): void {
  for (const generationEntry of history) {
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
}

function hasExtendedFields(speciesStat: SpeciesHistoryStatExtended): boolean {
  return 'innovationRange' in speciesStat && 'enabledRatio' in speciesStat;
}

function findSpeciesById(
  speciesList: SpeciesLike[],
  speciesId: number,
): SpeciesLike | undefined {
  return speciesList.find((speciesRecord) => speciesRecord.id === speciesId);
}

function summarizeSpeciesConnections(
  members: GenomeDetailed[],
  fallbackInnov: ((connection: ConnectionLike) => number) | undefined,
): { innovationRange: number; enabledRatio: number } {
  let maxInnovation = SPECIES_HISTORY_INITIAL_MAX_INNOVATION;
  let minInnovation = SPECIES_HISTORY_INITIAL_MIN_INNOVATION;
  let enabledCount = SPECIES_HISTORY_ZERO;
  let disabledCount = SPECIES_HISTORY_ZERO;

  for (const member of members) {
    const connections = member.connections as ConnectionLike[];
    for (const connection of connections) {
      const innovationId =
        connection.innovation ??
        fallbackInnov?.(connection) ??
        SPECIES_HISTORY_DEFAULT_INNOVATION_ID;

      if (innovationId > maxInnovation) {
        maxInnovation = innovationId;
      }
      if (innovationId < minInnovation) {
        minInnovation = innovationId;
      }

      if (connection.enabled === false) {
        disabledCount += 1;
      } else {
        enabledCount += 1;
      }
    }
  }

  return {
    innovationRange:
      Number.isFinite(maxInnovation) &&
      Number.isFinite(minInnovation) &&
      maxInnovation > minInnovation
        ? maxInnovation - minInnovation
        : SPECIES_HISTORY_DEFAULT_INNOVATION_RANGE,
    enabledRatio:
      enabledCount + disabledCount
        ? enabledCount / (enabledCount + disabledCount)
        : SPECIES_HISTORY_DEFAULT_ENABLED_RATIO,
  };
}

function applyExtendedSummary(
  speciesStat: SpeciesHistoryStatExtended,
  summary: { innovationRange: number; enabledRatio: number },
): void {
  speciesStat.innovationRange = summary.innovationRange;
  speciesStat.enabledRatio = summary.enabledRatio;
}