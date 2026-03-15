import type {
  ConnectionLike,
  NeatLike,
  NeatOptions,
  SpeciesHistoryEntry,
  SpeciesLike,
} from '../../../shared/neat.shared.types';
import type { SpeciesHistoryBackfillContext } from '../../core/augmentation/species.core.augmentation';

/**
 * Resolved host data needed to serve a species-history read.
 *
 * The public `getSpeciesHistory()` facade should not need to know how the NEAT
 * controller stores its history buffer or which internal fields are required to
 * support optional extended-history augmentation. This context keeps that
 * plumbing in one place.
 */
export type ResolvedSpeciesHistoryContext = {
  speciesHistory: SpeciesHistoryEntry[];
  backfillContext: SpeciesHistoryBackfillContext;
  neatOptions: NeatOptions | undefined;
};

type SpeciesHistoryHost = NeatLike & {
  _speciesHistory?: SpeciesHistoryEntry[];
  _species?: SpeciesLike[];
  _fallbackInnov?: (connection: ConnectionLike) => number;
};

/**
 * Resolve the stored history buffer, augmentation context, and options needed
 * by the public species-history read path.
 *
 * @param host - NEAT host exposing species history, species records, fallback innovation logic, and options.
 * @returns Normalized history-read context for the species facade.
 *
 * @example
 * ```ts
 * const historyContext = resolveSpeciesHistoryContext(neat);
 * console.log(historyContext.speciesHistory.length);
 * ```
 */
export function resolveSpeciesHistoryContext(
  host: NeatLike,
): ResolvedSpeciesHistoryContext {
  const historyHost = host as SpeciesHistoryHost;

  return {
    speciesHistory: historyHost._speciesHistory ?? [],
    backfillContext: {
      _species: historyHost._species,
      _fallbackInnov: historyHost._fallbackInnov,
    },
    neatOptions: host.options as NeatOptions | undefined,
  };
}
