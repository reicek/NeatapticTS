/**
 * Crossover methods for genetic algorithms.
 * @see {@link https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)}
 */
export const crossover = {
  /**
   * Single-point crossover.
   * A single crossover point is selected, and genes are exchanged between parents up to this point.
   * @property {string} name - The name of the crossover method.
   * @property {number[]} config - Configuration for the crossover point.
   */
  SINGLE_POINT: {
    name: 'SINGLE_POINT',
    config: [0.4],
  },

  /**
   * Two-point crossover.
   * Two crossover points are selected, and genes are exchanged between parents between these points.
   * @property {string} name - The name of the crossover method.
   * @property {number[]} config - Configuration for the two crossover points.
   */
  TWO_POINT: {
    name: 'TWO_POINT',
    config: [0.4, 0.9],
  },

  /**
   * Uniform crossover.
   * Each gene is selected randomly from one of the parents.
   * @property {string} name - The name of the crossover method.
   */
  UNIFORM: {
    name: 'UNIFORM',
  },

  /**
   * Average crossover.
   * The offspring's genes are the average of the parents' genes.
   * @property {string} name - The name of the crossover method.
   */
  AVERAGE: {
    name: 'AVERAGE',
  },
};
