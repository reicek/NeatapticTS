/**
 * Defines various selection methods used in genetic algorithms to choose individuals
 * for reproduction based on their fitness scores.
 *
 * Selection is a crucial step that determines which genetic traits are passed on
 * to the next generation. Different methods offer varying balances between
 * exploration (maintaining diversity) and exploitation (favoring high-fitness individuals).
 * The choice of selection method significantly impacts the algorithm's convergence
 * speed and the diversity of the population. High selection pressure (strongly
 * favoring the fittest) can lead to faster convergence but may result in premature
 * stagnation at suboptimal solutions. Conversely, lower pressure maintains diversity
 * but can slow down the search process.
 *
 * @see {@link https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)|Selection (genetic algorithm) - Wikipedia}
 * @see {@link https://en.wikipedia.org/wiki/Evolutionary_algorithm|Evolutionary algorithm - Wikipedia}
 */
export const selection = {
  /**
   * Fitness Proportionate Selection (also known as Roulette Wheel Selection).
   *
   * Individuals are selected based on their fitness relative to the total fitness
   * of the population. An individual's chance of being selected is directly
   * proportional to its fitness score. Higher fitness means a higher probability
   * of selection. This method can struggle if fitness values are very close or
   * if there are large disparities.
   */
  FITNESS_PROPORTIONATE: {
    name: 'FITNESS_PROPORTIONATE',
  },

  /**
   * Power Selection.
   *
   * Similar to Fitness Proportionate Selection, but fitness scores are raised
   * to a specified power before calculating selection probabilities. This increases
   * the selection pressure towards individuals with higher fitness scores, making
   * them disproportionately more likely to be selected compared to FITNESS_PROPORTIONATE.
   *
   * @property {number} power - The exponent applied to each individual's fitness score. Higher values increase selection pressure. Must be a positive number. Defaults to 4.
   */
  POWER: {
    name: 'POWER',
    power: 4,
  },

  /**
   * Tournament Selection.
   *
   * Selects individuals by holding competitions ('tournaments') among randomly
   * chosen subsets of the population. In each tournament, a fixed number (`size`)
   * of individuals are compared, and the fittest individual is chosen with a
   * certain `probability`. If not chosen (with probability 1 - `probability`),
   * the next fittest individual in the tournament might be selected (implementation dependent),
   * or another tournament might be run. This method is less sensitive to the scale
   * of fitness values compared to fitness proportionate methods.
   *
   * @property {number} size - The number of individuals participating in each tournament. Must be a positive integer. Defaults to 5.
   * @property {number} probability - The probability (between 0 and 1) of selecting the absolute fittest individual from the tournament participants. Defaults to 0.5.
   */
  TOURNAMENT: {
    name: 'TOURNAMENT',
    size: 5,
    probability: 0.5,
  },
};
