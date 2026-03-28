import type Network from '../../../architecture/network/network';
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
 *
 * The archive boundary is intentionally smaller than the main ranking pass.
 * `objectives/`, `dominance/`, `fronts/`, and `crowding/` decide which genomes
 * belong on the frontier and how spread out they are. This file runs only
 * after that work is complete. Its job is to preserve a lightweight historical
 * trace of the leading fronts so later reads can answer questions such as
 * "which genomes were on the frontier a few generations ago?" without copying
 * the full population into long-lived telemetry state.
 *
 * The write path stays compact on purpose:
 * - keep only the top fronts that are most useful for inspection,
 * - store stable genome identifiers instead of genome objects,
 * - label each snapshot with the current generation when available,
 * - trim the archive so long runs do not grow without bound.
 *
 * ```mermaid
 * flowchart TD
 *   A[Ordered Pareto fronts] --> B{Archive enabled?}
 *   B -->|No| C[Skip archive write]
 *   B -->|Yes| D[Take top fronts only]
 *   D --> E[Convert genomes to compact _id lists]
 *   E --> F[Push generation-labelled snapshot]
 *   F --> G{Archive too long?}
 *   G -->|No| H[Keep snapshot]
 *   G -->|Yes| I[Drop oldest snapshot]
 * ```
 */

/**
 * Maximum number of top Pareto fronts to retain per archive snapshot.
 *
 * This keeps each saved snapshot focused on the most competitively relevant
 * layers instead of persisting every dominated tail front from the population.
 */
export const MAX_PARETO_ARCHIVE_FRONTS = 3;

/**
 * Maximum number of archive snapshots to retain.
 *
 * The archive is meant to be a rolling inspection aid, not an unbounded event
 * log. Older snapshots are discarded once this limit is exceeded.
 */
export const MAX_PARETO_ARCHIVE_LENGTH = 100;

/**
 * Archives a compact snapshot of the current Pareto fronts when
 * multi-objective mode is enabled.
 *
 * This helper converts the frontier result of the current ranking pass into a
 * compact historical record. It does not recompute ranks, crowding, or
 * dominance; it trusts the incoming ordered `fronts` array and only decides
 * whether and how to persist a smaller snapshot.
 *
 * The saved snapshot is intentionally lightweight:
 * - stores only genome `_id` values, not full genomes,
 * - keeps only the top `MAX_PARETO_ARCHIVE_FRONTS` fronts,
 * - labels the snapshot with `generation` when the caller tracks one,
 * - maintains a bounded archive by dropping the oldest entry when necessary.
 *
 * Read this as an inspection aid rather than a replay format. The archive is
 * useful for telemetry, charts, and quick frontier-history questions, but it is
 * not designed to reconstruct full genome state later.
 *
 * @param neatInstance - Neat instance.
 * @param fronts - Ordered Pareto fronts from the current ranking pass.
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
