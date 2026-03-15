import type {
  NeatOptions,
  SpeciesHistoryEntry,
} from '../../shared/neat.shared.types';
import {
  backfillExtendedHistoryEntries,
  type SpeciesHistoryBackfillContext,
} from './augmentation/species.core.augmentation';

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
  context: SpeciesHistoryBackfillContext,
): void {
  backfillExtendedHistoryEntries(history, context);
}
