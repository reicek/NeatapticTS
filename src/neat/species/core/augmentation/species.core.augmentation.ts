import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciesHistoryEntry,
  SpeciesHistoryStatExtended,
  SpeciesLike,
} from '../../../shared/neat.shared.types';
import {
  summarizeSpeciesConnections,
  type SpeciesConnectionSummary,
} from '../shared/species.core.shared';

/**
 * Runtime surface required to backfill extended species-history fields.
 *
 * The augmentation chapter only needs the current live species registry plus an
 * optional innovation fallback for legacy connection records.
 */
export type SpeciesHistoryBackfillContext = {
  _species?: SpeciesLike[];
  _fallbackInnov?: (connection: ConnectionLike) => number;
};

/**
 * Backfill missing extended history fields in place.
 *
 * This chapter owns the concrete augmentation workflow so the root
 * `species.core.ts` file can stay focused on policy decisions such as whether
 * the backfill should run at all.
 *
 * @param history - Recorded species history to enrich.
 * @param context - NEAT context exposing current species and optional fallback innovations.
 * @returns Nothing. The history entries are mutated in place when backfill succeeds.
 */
export function backfillExtendedHistoryEntries(
  history: SpeciesHistoryEntry[],
  context: SpeciesHistoryBackfillContext,
): void {
  const speciesRecords = context._species ?? [];

  for (const generationEntry of history) {
    for (const speciesStat of generationEntry.stats as SpeciesHistoryStatExtended[]) {
      if (hasExtendedFields(speciesStat)) {
        continue;
      }

      const speciesRecord = findSpeciesById(speciesRecords, speciesStat.id);
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

function applyExtendedSummary(
  speciesStat: SpeciesHistoryStatExtended,
  summary: SpeciesConnectionSummary,
): void {
  speciesStat.innovationRange = summary.innovationRange;
  speciesStat.enabledRatio = summary.enabledRatio;
}
