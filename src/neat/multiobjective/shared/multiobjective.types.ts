import type Network from '../../../architecture/network';

/**
 * Shared contracts for the multi-objective ranking helpers.
 *
 * These types describe the smallest stable surface the multi-objective helpers
 * need: how objectives read values from genomes, which runtime fields a
 * Neat-like host must expose for Pareto archiving, and the transient `_mo*`
 * annotations attached to genomes during ranking.
 */

/**
 * Describes how to evaluate a single objective for a genome.
 *
 * The order of objective descriptors defines the order of each genome's
 * objective vector and therefore the columns of the values matrix.
 *
 * Notes:
 * - `accessor` should be deterministic for a given genome state.
 * - `direction` controls Pareto dominance comparisons:
 *   - `'max'`: higher is better
 *   - `'min'`: lower is better
 * - If `direction` is omitted, it defaults to `'max'`.
 *
 * @example
 * ```ts
 * const objectives: ObjectiveDescriptor[] = [
 *   { accessor: (g) => g.score ?? 0, direction: 'max' },
 *   { accessor: (g) => g.cost ?? 0, direction: 'min' },
 * ];
 * ```
 */
export type ObjectiveDescriptor = {
  /**
   * Extracts the raw objective value from a genome.
   *
   * This function should return a finite number. If it throws, callers that
   * use `readObjectiveValue` will treat it as `0`.
   */
  accessor: (genome: Network) => number;

  /**
   * Whether higher or lower values are preferred during Pareto dominance.
   *
   * Defaults to `'max'`.
   */
  direction?: 'max' | 'min';
};

/**
 * Minimal Neat-like interface required by the multi-objective helpers.
 *
 * This intentionally models only the fields used for archiving Pareto fronts
 * and retrieving objective descriptors. It allows these helpers to be used
 * without depending on the full Neat class type.
 */
export interface NeatLikeWithMultiObjective {
  /** Returns the objective descriptors (schema) used to evaluate genomes. */
  _getObjectives: () => ObjectiveDescriptor[];

  options: {
    multiObjective?: {
      /** Enables multi-objective mode. */
      enabled?: boolean;

      /** Optional config flag for archiving Pareto fronts. */
      archiveParetoFronts?: boolean;
    };
  };

  /** Rolling archive of compact Pareto snapshots. */
  _paretoArchive: Array<{ generation?: number; fronts: number[][] }>;

  /** Current generation index, if tracked by the caller. */
  generation?: number;
}

/**
 * Extends a genome/network with multi-objective annotations.
 *
 * These properties are used as transient metadata during selection.
 */
export interface NetworkWithMOAnnotations extends Network {
  /** Pareto front rank (0 = best front). */
  _moRank?: number;

  /** Crowding distance within the front. */
  _moCrowd?: number;

  /** Optional stable identifier used for compact Pareto archiving. */
  _id?: number;
}
