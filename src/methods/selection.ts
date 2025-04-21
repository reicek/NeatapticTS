/**
 * Selection methods for genetic algorithms.
 *
 * Selection determines which individuals in a population are chosen for
 * reproduction. Methods like Fitness Proportionate Selection, Tournament
 * Selection, and Rank Selection balance exploration and exploitation in
 * evolutionary algorithms.
 *
 * Selection pressure influences convergence speed and diversity. High
 * pressure accelerates convergence but risks premature stagnation, while
 * low pressure ensures diversity but slows progress.
 *
 * @see {@link https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)}
 * @see {@link https://en.wikipedia.org/wiki/Evolutionary_algorithm}
 */
export const selection = {
  /**
   * Fitness Proportionate Selection (Roulette Wheel Selection).
   * Selects individuals based on their fitness proportion relative to the population.
   */
  FITNESS_PROPORTIONATE: {
    name: 'FITNESS_PROPORTIONATE',
  },

  /**
   * Power Selection.
   * Selects individuals with a bias towards higher fitness using a power factor.
   * @property {number} power - The power factor used to bias selection.
   */
  POWER: {
    name: 'POWER',
    power: 4,
  },

  /**
   * Tournament Selection.
   * Selects individuals by running tournaments among a subset of the population.
   * @property {number} size - The size of the tournament.
   * @property {number} probability - Probability of selecting the best individual in the tournament.
   */
  TOURNAMENT: {
    name: 'TOURNAMENT',
    size: 5,
    probability: 0.5,
  },
};
