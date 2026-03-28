/**
 * Crossover methods for genetic algorithms.
 *
 * These methods implement the crossover strategies described in the Instinct algorithm,
 * enabling the creation of offspring with unique combinations of parent traits.
 *
 * Read this file as an inheritance-policy shelf: each method answers a
 * different question about how aggressively two parents should be mixed.
 *
 * - `SINGLE_POINT` preserves one contiguous prefix from one parent and the
 *   remaining suffix from the other,
 * - `TWO_POINT` preserves a middle segment boundary instead of only one split,
 * - `UNIFORM` treats each gene as an independent coin flip,
 * - `AVERAGE` blends compatible numeric genes instead of copying segments.
 *
 * A practical chooser for first experiments:
 *
 * - start with `UNIFORM` when you want broad mixing and do not need contiguous
 *   blocks of structure to stay together,
 * - use `SINGLE_POINT` or `TWO_POINT` when adjacency matters and you want to
 *   preserve larger parent segments,
 * - choose `AVERAGE` when the genome is meaningfully numeric and interpolation
 *   is more useful than hard parent switching.
 *
 * Minimal workflow:
 *
 * ```ts
 * const broadMixing = crossover.UNIFORM;
 *
 * const oneCut = crossover.SINGLE_POINT;
 *
 * const twoCut = {
 *   ...crossover.TWO_POINT,
 *   config: [0.25, 0.75],
 * };
 *
 * const blendedOffspring = crossover.AVERAGE;
 * ```
 *
 * ```mermaid
 * flowchart LR
 *   Parents[Two parent genomes] --> Segment[Segment-preserving crossover]
 *   Parents --> GeneWise[Gene-wise crossover]
 *   Parents --> Blend[Numeric blending]
 *   Segment --> Single[SINGLE_POINT]
 *   Segment --> Double[TWO_POINT]
 *   GeneWise --> Uniform[UNIFORM]
 *   Blend --> Average[AVERAGE]
 * ```
 *
 * @see Instinct Algorithm - Section 2 Crossover
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6}
 * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)}
 */
export const crossover = {
  /**
   * Single-point crossover.
   * A single crossover point is selected, and genes are exchanged between parents up to this point.
   * This method is particularly useful for binary-encoded genomes.
   *
   * Use this when a single clean cut is enough to preserve a meaningful prefix
   * from one parent and a suffix from the other.
   *
   * @property {string} name - The name of the crossover method.
   * @property {number[]} config - Configuration for the crossover point.
   * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#One-point_crossover}
   */
  SINGLE_POINT: {
    name: 'SINGLE_POINT',
    config: [0.4],
  },

  /**
   * Two-point crossover.
   * Two crossover points are selected, and genes are exchanged between parents between these points.
   * This method is an extension of single-point crossover and is often used for more complex genomes.
   *
   * Use this when one cut feels too coarse and you want to preserve a central
   * segment without giving up all positional structure.
   *
   * @property {string} name - The name of the crossover method.
   * @property {number[]} config - Configuration for the two crossover points.
   * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Two-point_and_k-point_crossover}
   */
  TWO_POINT: {
    name: 'TWO_POINT',
    config: [0.4, 0.9],
  },

  /**
   * Uniform crossover.
   * Each gene is selected randomly from one of the parents with equal probability.
   * This method provides a high level of genetic diversity in the offspring.
   *
   * This is the most fine-grained mixing policy in the file: inheritance is
   * decided gene by gene instead of by preserving long contiguous blocks.
   *
   * @property {string} name - The name of the crossover method.
   * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Uniform_crossover}
   */
  UNIFORM: {
    name: 'UNIFORM',
  },

  /**
   * Average crossover.
   * The offspring's genes are the average of the parents' genes.
   * This method is particularly useful for real-valued genomes.
   *
   * Use this when interpolation is meaningful and you want offspring to land
   * between both parents instead of inheriting one parent's exact segment.
   *
   * @property {string} name - The name of the crossover method.
   * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Arithmetic_recombination}
   */
  AVERAGE: {
    name: 'AVERAGE',
  },
};
