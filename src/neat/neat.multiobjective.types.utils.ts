import type Network from '../architecture/network';

/**
 * Describes how to evaluate a single objective for a genome.
 *
 * The order of objective descriptors defines the order of each genome’s
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
   * use {@link readObjectiveValue} will treat it as `0`.
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
  /**
   * Returns the objective descriptors (schema) used to evaluate genomes.
   */
  _getObjectives: () => ObjectiveDescriptor[];

  options: {
    multiObjective?: {
      /**
       * Enables multi-objective mode.
       *
       * When `false`/unset, helpers like archiving are expected to be no-ops.
       */
      enabled?: boolean;

      /**
       * Optional config flag for archiving Pareto fronts.
       *
       * Note: {@link archiveParetoFrontsIfEnabled} currently gates only on
       * `enabled`. If you want this flag to be authoritative, incorporate it
       * into the call site or update the gating behavior.
       */
      archiveParetoFronts?: boolean;
    };
  };

  /**
   * Rolling archive of compact Pareto snapshots.
   *
   * Each snapshot stores:
   * - `generation`: optional generation number
   * - `fronts`: array of fronts, where each front is an array of genome `_id`
   *   values.
   */
  _paretoArchive: Array<{ generation?: number; fronts: number[][] }>;

  /** Current generation index, if tracked by the caller. */
  generation?: number;
}

/**
 * Extends a genome/network with multi-objective annotations.
 *
 * These properties are used as transient metadata during selection.
 *
 * - `_moRank`: Pareto front rank (0 = best front)
 * - `_moCrowd`: crowding distance within the front (higher = more isolated;
 *   boundary genomes are typically `Infinity`)
 * - `_id`: optional stable identifier used for compact archiving
 */
export interface NetworkWithMOAnnotations extends Network {
  /** Pareto front rank (0 = best front). */
  _moRank?: number;

  /**
   * Crowding distance within the front.
   *
   * Higher means more isolated. Boundary genomes for an objective are
   * typically assigned `Infinity`.
   */
  _moCrowd?: number;

  /** Optional stable identifier used for compact Pareto archiving. */
  _id?: number;
}
