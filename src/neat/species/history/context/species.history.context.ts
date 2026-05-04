import type {
  ConnectionLike,
  NeatLike,
  NeatOptions,
  SpeciesHistoryEntry,
  SpeciesLike,
} from '../../../shared/neat.shared.types';
import type { SpeciesHistoryBackfillContext } from '../../core/augmentation/species.core.augmentation';

/**
 * Setup seam for species-history reads.
 *
 * The public species facade answers a simple question: "what does the recent
 * species story look like?" The controller data behind that answer is less
 * simple. The history read path may need the stored history buffer, the live
 * species registry, the explicit legacy/import fallback resolver used by
 * optional backfill, and controller options that decide whether augmentation
 * should happen at all.
 *
 * This file keeps that setup work narrow and explicit. Instead of making the
 * public history read helper reach into several internal host fields directly,
 * it gathers those pieces once, normalizes missing values into predictable
 * shapes, and hands one compact context object to the next stage.
 *
 * Read this chapter in two steps:
 *
 * 1. `ResolvedSpeciesHistoryContext` shows the normalized handoff shape used by
 *    the history read path.
 * 2. `resolveSpeciesHistoryContext()` shows how the NEAT host is translated
 *    into that shape without leaking controller storage details into the public
 *    facade.
 */

/**
 * Resolved host data needed to serve a species-history read.
 *
 * The public `getSpeciesHistory()` facade should not need to know how the NEAT
 * controller stores its history buffer or which internal fields are required to
 * support optional extended-history augmentation. This context keeps that
 * plumbing in one place.
 *
 * The three fields map to the three setup concerns behind a history read:
 *
 * - `speciesHistory` is the stored generation-by-generation buffer,
 * - `backfillContext` carries the internal data needed only when extended
 *   history enrichment is allowed,
 * - `neatOptions` carries the policy switches that decide whether that
 *   enrichment path should run.
 *
 * Read this type as the boundary between raw controller storage and the richer
 * history-read logic. Once this object has been resolved, later helpers can ask
 * "should we augment?" and "what data do we already have?" without knowing
 * where the controller originally stored each ingredient.
 */
export type ResolvedSpeciesHistoryContext = {
  /** Recorded species-history rows, normalized to an empty array when absent. */
  speciesHistory: SpeciesHistoryEntry[];
  /** Internal data used when optional extended-history backfill is enabled. */
  backfillContext: SpeciesHistoryBackfillContext;
  /** Controller options that gate optional enrichment behavior. */
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
 * This helper is intentionally small because its job is normalization, not
 * interpretation. It does three things:
 *
 * 1. read the controller's stored species-history buffer,
 * 2. gather the internal fields needed only by optional backfill,
 * 3. pass through the current options so later read helpers can decide whether
 *    extended enrichment is allowed.
 *
 * That split keeps `getSpeciesHistory()` easier to read. The facade can stay
 * focused on serving historical data, while this helper owns the one-time host
 * translation from controller internals into a stable read-side context.
 *
 * @param host - NEAT host exposing species history, species records, explicit legacy/import fallback innovation logic, and options.
 * @returns Normalized history-read context for the species facade.
 *
 * @example
 * ```ts
 * const historyContext = resolveSpeciesHistoryContext(neat);
 *
 * if (historyContext.neatOptions?.enableSpeciesAugmentation) {
 *   console.log(historyContext.backfillContext._species?.length ?? 0);
 * }
 *
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
