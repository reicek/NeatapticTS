/**
 * Crossover methods for genetic algorithms.
 *
 * These methods implement the crossover strategies described in the Instinct algorithm,
 * enabling the creation of offspring with unique combinations of parent traits.
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
   * @property {string} name - The name of the crossover method.
   * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Arithmetic_recombination}
   */
  AVERAGE: {
    name: 'AVERAGE',
  },
};
