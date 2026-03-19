/**
 * Measurement helpers for species-history augmentation.
 *
 * The surrounding `species/core` chapter decides *when* older history rows may
 * be enriched. This shared chapter answers the narrower question underneath
 * that policy gate: *what compact evidence should the controller extract from a
 * live species once enrichment is allowed?*
 *
 * The answer is intentionally small. Instead of persisting every connection for
 * every member genome, the read-side history pipeline reduces one species to a
 * pair of explanatory signals:
 *
 * - how wide the innovation span is across the current members,
 * - how much of that structural material is still enabled.
 *
 * Those two metrics are enough to make historical species rows more legible in
 * dashboards and exports without dragging a large connection-level payload into
 * the reporting surface.
 *
 * Read this file when the open question is about measurement rather than
 * orchestration. If you need to know whether augmentation should happen at all,
 * go back up to `species/core`. If you need to know how missing rows are walked
 * and filled in place, continue into `augmentation/`.
 *
 * ```mermaid
 * flowchart TD
 *   Members[Live species members] --> Resolve[Resolve innovation ids]
 *   Resolve --> Count[Count enabled and disabled connections]
 *   Count --> Fold[Fold into innovation range and enabled ratio]
 *   Fold --> History[Write compact metrics into history rows]
 * ```
 */
import type {
  ConnectionLike,
  GenomeDetailed,
} from '../../../shared/neat.shared.types';

/**
 * Shared arithmetic helpers for species-history augmentation.
 *
 * The surrounding `species/core` chapter decides when extended history should
 * be backfilled, and the `augmentation/` chapter walks the recorded rows. This
 * shared helper file owns the smallest reusable arithmetic underneath both: how
 * to summarize one live species into compact innovation-coverage evidence.
 *
 * Read this file when the missing question is not policy but measurement:
 * which values are extracted from current species members, how legacy
 * connection records fall back to an innovation resolver, and why the result is
 * stored as a compact summary instead of raw connection lists.
 */

const SPECIES_HISTORY_DEFAULT_INNOVATION_ID = 0;
const SPECIES_HISTORY_DEFAULT_INNOVATION_RANGE = 0;
const SPECIES_HISTORY_DEFAULT_ENABLED_RATIO = 0;
const SPECIES_HISTORY_ZERO = 0;
const SPECIES_HISTORY_INITIAL_MAX_INNOVATION = -Infinity;
const SPECIES_HISTORY_INITIAL_MIN_INNOVATION = Infinity;

/**
 * Aggregated innovation coverage for the genomes currently assigned to one species.
 *
 * The history backfill path uses this compact shape to answer two questions that
 * are useful in telemetry dashboards:
 *
 * - how wide the inherited innovation span is across the species members,
 * - how many of those structural genes are still enabled.
 *
 * Those two values are enough to make extended history rows much more
 * explanatory without forcing the species-reporting surface to retain every
 * connection-level detail from every generation.
 *
 * Treat it as a reporting summary, not as a lossless reconstruction format.
 * The goal is to explain structural breadth and retention at a glance, not to
 * preserve every innovation id for downstream mutation logic.
 */
export type SpeciesConnectionSummary = {
  innovationRange: number;
  enabledRatio: number;
};

/**
 * Summarize innovation spread and enabled-connection ratio for one species.
 *
 * This helper stays separate from the history backfill orchestration so the
 * arithmetic can be reused and documented independently from the policy that
 * decides when augmentation should happen.
 *
 * Read the fold in three stages:
 *
 * 1. walk every member connection,
 * 2. resolve an innovation id from the connection or fallback resolver,
 * 3. reduce the seen ids and enabled flags into one compact summary.
 *
 * The fold preserves three small rules:
 * - connection innovation ids come from the connection itself when present,
 * - legacy connections may fall back to the supplied innovation resolver,
 * - empty or fully unresolved inputs collapse to safe zero-style defaults
 *   instead of producing `NaN` or `Infinity` noise in history output.
 *
 * @param members - Detailed member genomes for a single species.
 * @param fallbackInnov - Optional innovation resolver for legacy connections that do not carry a direct innovation id.
 * @returns Compact innovation-range and enabled-ratio telemetry for the species.
 *
 * @example
 * ```ts
 * const summary = summarizeSpeciesConnections(species.members, neat._fallbackInnov);
 * console.log(summary.innovationRange, summary.enabledRatio);
 * ```
 */
export function summarizeSpeciesConnections(
  members: GenomeDetailed[],
  fallbackInnov: ((connection: ConnectionLike) => number) | undefined,
): SpeciesConnectionSummary {
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
