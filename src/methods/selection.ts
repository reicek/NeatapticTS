/**
 * Selection methods for genetic algorithms.
 * @see {@link https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)}
 */
export const selection = {
  /**
   * Fitness Proportionate Selection (also known as Roulette Wheel Selection).
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
   * @property {number} probability - The probability of selecting the best individual in the tournament.
   */
  TOURNAMENT: {
    name: 'TOURNAMENT',
    size: 5,
    probability: 0.5,
  },
};
