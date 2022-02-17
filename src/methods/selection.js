/**
 * Selection
 * @see {@link https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)}
 */
const selection = {
  FITNESS_PROPORTIONATE: {
    name: 'FITNESS_PROPORTIONATE',
  },
  POWER: {
    name: 'POWER',
    power: 4,
  },
  TOURNAMENT: {
    name: 'TOURNAMENT',
    size: 5,
    probability: 0.5,
  },
};

module.exports = selection;
