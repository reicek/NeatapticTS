/**
 * Minimal node interface used by diversity computations.
 *
 * Diversity helpers only need the outgoing connection count, so this type keeps
 * the telemetry boundary narrower than the full runtime node model.
 */
export interface NodeWithConnections {
  connections: {
    out: unknown[];
  };
}

/**
 * Minimal genome shape used by diversity computations.
 *
 * The diversity chapter focuses on structural size, lineage depth, and sampled
 * compatibility, so only those fields are modeled here.
 */
export interface GenomeWithMetrics {
  nodes: NodeWithConnections[];
  connections: unknown[];
  _depth?: number;
}

/**
 * Minimal interface for computing compatibility distance between genomes.
 *
 * The narrow contract keeps diversity telemetry independent from the broader
 * NEAT controller implementation.
 */
export interface CompatComputer {
  /**
   * Compute a compatibility distance between two genomes.
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
 * Each field captures one aggregate lens on the current population: lineage
 * spread, structural size, compatibility separation, or entropy.
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
