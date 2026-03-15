import type Network from '../../../architecture/network';
import type {
  NeatLikeWithMultiObjective,
  NetworkWithMOAnnotations,
} from '../shared/multiobjective.types';

/**
 * Compact Pareto archive helpers for the NEAT controller.
 *
 * This chapter keeps the write-side archive mechanics together: when
 * multi-objective mode is enabled, it snapshots the leading Pareto fronts into
 * a small archive that downstream telemetry and visualization helpers can
 * inspect without retaining whole genomes.
 */

/** Maximum number of top Pareto fronts to retain per archive snapshot. */
export const MAX_PARETO_ARCHIVE_FRONTS = 3;

/** Maximum number of archive snapshots to retain. */
export const MAX_PARETO_ARCHIVE_LENGTH = 100;

/**
 * Archives a compact snapshot of the current Pareto fronts when
 * multi-objective mode is enabled.
 *
 * This is intended for visualization/debugging:
 * - Stores only genome `_id` values, not full genomes.
 * - Keeps only the top `MAX_PARETO_ARCHIVE_FRONTS` fronts.
 * - Maintains a bounded archive by shifting the oldest entry.
 *
 * @param neatInstance - Neat instance.
 * @param fronts - Pareto fronts to archive.
 */
export function archiveParetoFrontsIfEnabled(
  neatInstance: NeatLikeWithMultiObjective,
  fronts: Network[][],
): void {
  if (!neatInstance.options.multiObjective?.enabled) return;

  // Step 1: Store a compact snapshot of the top fronts.
  neatInstance._paretoArchive.push({
    generation: neatInstance.generation,
    fronts: fronts
      .slice(0, MAX_PARETO_ARCHIVE_FRONTS)
      .map((front) =>
        front.map((genome) => (genome as NetworkWithMOAnnotations)._id || 0),
      ),
  });

  // Step 2: Keep the archive bounded.
  if (neatInstance._paretoArchive.length > MAX_PARETO_ARCHIVE_LENGTH) {
    neatInstance._paretoArchive.shift();
  }
}
