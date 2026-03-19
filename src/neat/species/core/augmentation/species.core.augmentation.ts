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
 * Concrete extended-history augmentation helpers for the species core chapter.
 *
 * The root `species.core.ts` file decides whether augmentation should run at
 * all. This companion file owns the narrower mechanics that actually walk the
 * history rows, find matching live species records, and copy derived structural
 * summaries into entries that are still missing them.
 *
 * The implementation deliberately stays conservative:
 * - it enriches only rows that lack the extended fields,
 * - it skips history rows whose species no longer exist in the live registry,
 * - it derives only compact summary values instead of copying full genomes or
 *   connection lists into history.
 */

/**
 * Runtime surface required to backfill extended species-history fields.
 *
 * The augmentation chapter only needs the current live species registry plus an
 * optional innovation fallback for legacy connection records.
 *
 * That narrowness is intentional. The backfill path is a read-side enrichment
 * step, not a second controller facade, so it should depend only on the live
 * species evidence and the small fallback hook needed to interpret older
 * connection records.
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
 * The helper walks recorded generations first and species rows second. For each
 * row, it performs a small three-step decision:
 * 1. skip rows that already contain the extended metrics,
 * 2. look up the matching live species by id,
 * 3. if that species still exists and has members, summarize its innovation
 *    coverage and copy the result into the history row.
 *
 * This makes the backfill useful for read-time teaching and diagnostics without
 * pretending that old history can always be reconstructed perfectly from the
 * current runtime state.
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
