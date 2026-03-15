import type {
  ConnectionLike,
  GenomeDetailed,
} from '../../../shared/neat.shared.types';

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
