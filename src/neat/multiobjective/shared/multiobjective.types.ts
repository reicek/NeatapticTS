import type Network from '../../../architecture/network';

/**
 * Shared contracts for the multi-objective ranking helpers.
 *
 * This chapter is the common language layer behind the multi-objective
 * pipeline. The root `multiobjective/` chapter explains the ranking story at a
 * high level, while `objectives/`, `dominance/`, `fronts/`, and `crowding/`
 * each own one narrow stage of the mechanics. This file exists so those helper
 * chapters can agree on a deliberately small contract surface instead of
 * depending on the full `Neat` controller shape.
 *
 * The shared surface is organized into three practical contract families:
 * - objective descriptors define how one genome becomes one ordered value
 *   vector,
 * - the minimal host contract exposes only the objective schema and compact
 *   Pareto archive state,
 * - transient genome annotations carry rank and crowding evidence forward after
 *   sorting.
 *
 * The main invariant to keep in mind is stable ordering. Descriptor order
 * becomes vector-column order, vector-column order feeds pairwise dominance,
 * and the same row positions are preserved through frontier peeling and
 * crowding assignment. If that ordering drifts, the later helper chapters would
 * still run, but they would be reasoning about the wrong objectives or genomes.
 */

/**
 * Describes how to evaluate one objective for one genome.
 *
 * `objectives/` uses these descriptors to assemble one ordered value vector per
 * genome. `dominance/` then compares those vectors column by column, so the
 * descriptor array is more than configuration data: it is the schema that tells
 * every later helper what each column means and whether larger or smaller
 * values should win.
 *
 * Two rules matter most:
 * - keep descriptor order stable for the duration of one ranking pass,
 * - keep each accessor deterministic for a given genome state so pairwise
 *   comparisons do not change mid-pass.
 *
 * Notes:
 * - `accessor` should return a finite numeric signal for the current genome.
 * - `direction` controls Pareto dominance comparisons:
 *   - `'max'`: higher is better
 *   - `'min'`: lower is better
 * - If `direction` is omitted, it defaults to `'max'` so single-score style
 *   objectives keep their usual interpretation.
 *
 * @example
 * ```ts
 * const objectives: ObjectiveDescriptor[] = [
 *   { accessor: (genome) => genome.score ?? 0, direction: 'max' },
 *   { accessor: (genome) => genome.cost ?? 0, direction: 'min' },
 * ];
 * ```
 */
export type ObjectiveDescriptor = {
  /**
   * Extracts the raw objective value from a genome.
   *
   * This is the read-side seam between the controller's genome state and the
   * dense numeric matrix used by the ranking helpers. If it throws, callers
   * such as `readObjectiveValue()` treat the failure as `0` so the ranking pass
   * can stay total instead of aborting the whole frontier computation.
   */
  accessor: (genome: Network) => number;

  /**
   * Whether higher or lower values are preferred during Pareto dominance.
   *
   * This flag is consumed directly by the pairwise dominance helpers. Defaults
   * to `'max'` so callers can omit it for ordinary "higher is better" fitness
   * style reads.
   */
  direction?: 'max' | 'min';
};

/**
 * Minimal Neat-like interface required by the multi-objective helpers.
 *
 * This host contract stays intentionally small so the multi-objective helpers
 * can be reused without depending on the entire `Neat` controller surface.
 * The boundary owns only two kinds of state:
 * - objective-schema access for the start of the ranking pass,
 * - optional Pareto-archive state for the end of the ranking pass.
 *
 * Everything else stays outside this interface on purpose. `objectives/`,
 * `dominance/`, `fronts/`, and `crowding/` operate on prepared vectors,
 * bookkeeping structures, and annotated genomes rather than reaching back into
 * controller internals mid-pass.
 */
export interface NeatLikeWithMultiObjective {
  /**
   * Returns the active objective schema for the current ranking pass.
   *
   * The returned descriptor order defines the vector-column order used by every
   * downstream helper in the same pass.
   */
  _getObjectives: () => ObjectiveDescriptor[];

  options: {
    multiObjective?: {
      /** Enables multi-objective ranking for the current controller run. */
      enabled?: boolean;

      /**
       * Enables compact archival snapshots of the leading fronts after ranking.
       */
      archiveParetoFronts?: boolean;
    };
  };

  /**
   * Rolling archive of compact Pareto snapshots.
   *
   * The orchestration layer writes lightweight front membership snapshots here
   * so later telemetry or inspection code can examine past frontier shape
   * without storing full genome copies.
   */
  _paretoArchive: Array<{ generation?: number; fronts: number[][] }>;

  /** Current generation index, if the caller tracks one for archive labeling. */
  generation?: number;
}

/**
 * Extends a genome/network with multi-objective annotations.
 *
 * These properties are transient ranking metadata. They are attached after the
 * multi-objective helpers compute fronts and crowding distances, then consumed
 * by later selection or inspection code as a compact summary of where a genome
 * landed on the current Pareto surface.
 *
 * Treat these fields as derived evidence, not durable genome state. A later
 * ranking pass is free to recompute or overwrite them.
 */
export interface NetworkWithMOAnnotations extends Network {
  /** Pareto front rank where `0` denotes the current non-dominated front. */
  _moRank?: number;

  /**
   * Crowding distance within the current front.
   *
   * Larger values indicate genomes in less crowded tradeoff regions. Boundary
   * genomes may receive `Infinity` when crowding preserves the extremes.
   */
  _moCrowd?: number;

  /**
   * Optional stable identifier used when compact Pareto archives record front
   * membership without persisting full genome objects.
   */
  _id?: number;
}
