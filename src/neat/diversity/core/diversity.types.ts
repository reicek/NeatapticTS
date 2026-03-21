/**
 * Contracts for the bounded diversity-reporting pipeline.
 *
 * The diversity core chapter builds one cheap population snapshot from four
 * complementary signals:
 * structural size, structural shape, compatibility distance, and lineage
 * spread. These contracts stay intentionally small so telemetry and diagnostic
 * reads can project just the fields that report needs instead of importing the
 * entire `Neat` runtime surface.
 *
 * Read the chapter in this order:
 * - `GenomeWithMetrics` describes the minimal per-genome projection.
 * - `CompatComputer` supplies the one expensive cross-genome measurement.
 * - `DiversityStats` names the aggregated output fields that consumers compare
 *   across generations.
 */

/**
 * Minimal node interface used by diversity computations.
 *
 * Diversity helpers only need each node's outgoing connection count to build a
 * structural-entropy fingerprint, so this type keeps the reporting boundary
 * narrower than the full runtime node model.
 */
export interface NodeWithConnections {
  connections: {
    out: unknown[];
  };
}

/**
 * Minimal genome shape used by diversity computations.
 *
 * This projection is the per-genome input to the diversity report. It exposes
 * exactly the data needed to answer four cheap read-side questions:
 *
 * - how large genomes are right now,
 * - how uneven that structural size has become,
 * - how far sampled ancestry depth has spread,
 * - and how compatible each genome remains with sampled peers.
 *
 * Keeping the contract this small lets diagnostics and telemetry reuse the
 * diversity helpers against genome-like snapshots rather than the full NEAT
 * controller state.
 */
export interface GenomeWithMetrics {
  nodes: NodeWithConnections[];
  connections: unknown[];
  _depth?: number;
}

/**
 * Minimal interface for computing compatibility distance between genomes.
 *
 * Compatibility is the one report dimension that needs a cross-genome callback
 * instead of direct field reads. This seam keeps the diversity chapter focused
 * on sampling and aggregation while the broader NEAT controller owns the actual
 * compatibility policy.
 */
export interface CompatComputer {
  /**
   * Compute a compatibility distance between two genomes.
   *
   * Diversity reporting treats this as a reusable measurement primitive: the
   * helper samples a bounded set of genome pairs, asks the host for distances,
   * then folds those distances into one trend value that can be tracked over
   * time.
   *
   * @param firstGenome - First genome-like object.
   * @param secondGenome - Second genome-like object.
   * @returns Non-negative compatibility distance where larger means more different.
   */
  _compatibilityDistance(
    firstGenome: GenomeWithMetrics,
    secondGenome: GenomeWithMetrics,
  ): number;
}

/**
 * Diversity statistics returned by sampled population analysis.
 *
 * Treat this as a compact population-health report rather than as a single
 * scalar "diversity score." The fields are grouped deliberately:
 *
 * - lineage fields show whether ancestry depth is spreading or collapsing,
 * - node and connection fields show average structural size and unevenness,
 * - compatibility sampling estimates how genetically separated sampled peers
 *   remain,
 * - entropy adds a shape signal that raw size counts cannot capture.
 *
 * In practice, telemetry consumers compare this object across generations to
 * see whether mutation, speciation, and pruning are still producing meaningful
 * variation without paying for exhaustive all-pairs analysis.
 */
export interface DiversityStats {
  /** Mean lineage depth across genomes that expose `_depth`. */
  lineageMeanDepth: number;
  /** Mean sampled pairwise lineage-depth distance. */
  lineageMeanPairDist: number;
  /** Mean number of nodes across the population. */
  meanNodes: number;
  /** Mean number of connections across the population. */
  meanConns: number;
  /** Variance of node counts across the population. */
  nodeVar: number;
  /** Variance of connection counts across the population. */
  connVar: number;
  /** Mean sampled compatibility distance across genome pairs. */
  meanCompat: number;
  /** Mean structural entropy across sampled genomes. */
  graphletEntropy: number;
  /** Population size represented by the report. */
  population: number;
}
