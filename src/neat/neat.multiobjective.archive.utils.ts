import type Network from '../architecture/network';
import type {
  NeatLikeWithMultiObjective,
  NetworkWithMOAnnotations,
} from './neat.multiobjective.utils.types';

/**
 * Maximum number of top Pareto fronts to retain per archive snapshot.
 *
 * Archival stores a compact representation (IDs only) for visualization or
 * debugging.
 */
export const MAX_PARETO_ARCHIVE_FRONTS = 3;

/**
 * Maximum number of archive snapshots to retain.
 *
 * When the archive exceeds this length, the oldest snapshot is dropped.
 */
export const MAX_PARETO_ARCHIVE_LENGTH = 100;

/**
 * Archives a compact snapshot of the current Pareto fronts when
 * multi-objective mode is enabled.
 *
 * This is intended for visualization/debugging:
 * - Stores only genome `_id` values (not full genomes).
 * - Keeps only the top {@link MAX_PARETO_ARCHIVE_FRONTS} fronts.
 * - Maintains a ring-buffer-like cap of {@link MAX_PARETO_ARCHIVE_LENGTH}
 *   snapshots by shifting the oldest entry.
 *
 * Behavior note:
 * - This currently gates only on `neatInstance.options.multiObjective?.enabled`.
 *   If you want a separate archive toggle, ensure the caller configures
 *   `enabled` accordingly.
 *
 * @param neatInstance - Neat instance.
 * @param fronts - Pareto fronts to archive.
 */
export function archiveParetoFrontsIfEnabled(
  neatInstance: NeatLikeWithMultiObjective,
  fronts: Network[][],
): void {
  if (!neatInstance.options.multiObjective?.enabled) return;

  // Step 1: store a compact snapshot of top fronts.
  neatInstance._paretoArchive.push({
    generation: neatInstance.generation,
    fronts: fronts
      .slice(0, MAX_PARETO_ARCHIVE_FRONTS)
      .map((front) =>
        front.map((genome) => (genome as NetworkWithMOAnnotations)._id || 0),
      ),
  });

  if (neatInstance._paretoArchive.length > MAX_PARETO_ARCHIVE_LENGTH) {
    neatInstance._paretoArchive.shift();
  }
}
