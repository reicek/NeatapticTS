import type {
  NeatLike,
  SpeciesHistoryEntry,
} from '../../../shared/neat.shared.types';
import {
  backfillExtendedHistory,
  shouldAugmentExtendedHistory,
} from '../../core/species.core';
import { resolveSpeciesHistoryContext } from '../context/species.history.context';

/**
 * Read the recorded species history for a NEAT host.
 *
 * This history-read chapter keeps the public species facade thin by owning the
 * only remaining orchestration in the history path: resolve the stored history
 * context, optionally backfill extended metrics, then return the normalized
 * history buffer.
 *
 * @param host - NEAT host exposing species history, species records, fallback innovation logic, and options.
 * @returns Generation-stamped species history snapshots.
 *
 * @example
 * ```ts
 * const historyEntries = getSpeciesHistory(neat);
 * console.log(historyEntries.at(-1));
 * ```
 */
export function getSpeciesHistory(host: NeatLike): SpeciesHistoryEntry[] {
  const historyContext = resolveSpeciesHistoryContext(host);

  // Step 1: Backfill extended fields only when explicitly enabled.
  if (shouldAugmentExtendedHistory(historyContext.neatOptions)) {
    backfillExtendedHistory(
      historyContext.speciesHistory,
      historyContext.backfillContext,
    );
  }

  return historyContext.speciesHistory;
}
