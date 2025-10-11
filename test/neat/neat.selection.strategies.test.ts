import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/** Tests covering selection strategies (POWER, TOURNAMENT error path). */
describe('NEAT Selection Strategies', () => {
  describe('POWER selection sorts when unsorted', () => {
    /** Fitness equals negative node count to force varied scores. */
    const powerFitness = (network: Network) => -network.nodes.length;
    /** Instance with POWER selection. */
    const powerNeat = new Neat(2, 1, powerFitness, {
      popsize: 6,
      seed: 555,
      selection: { name: 'POWER', power: 1 },
    });
    beforeAll(async () => {
      await powerNeat.evaluate();
      // Arrange: intentionally swap two genome scores to simulate unsorted state
      const firstGenome = powerNeat.population[0];
      const secondGenome = powerNeat.population[1];
      // Force score inversion (unsafe cast for test only)
      firstGenome.score = 1;
      secondGenome.score = 5; // ensure second > first triggers sort branch
    });
    test('getParent returns a genome (post-sort path executed)', () => {
      // Act: select parent
      const parent = (powerNeat as Neat & { getParent(): Network }).getParent();
      // Assert: parent object returned
      expect(typeof parent).toBe('object');
    });
  });
  describe('TOURNAMENT selection invalid size error', () => {
    /** Constant fitness (scores irrelevant). */
    const tournamentFitness = (network: Network) => {
      void network;
      return 1;
    };
    /** Instance with small population so oversized tournament triggers error. */
    const tournamentNeat = new Neat(2, 1, tournamentFitness, {
      popsize: 3,
      seed: 556,
      selection: { name: 'TOURNAMENT', size: 10, probability: 0.5 },
    });
    test('throws when tournament size > population and not suppressed', () => {
      // Arrange: callable invoking selection
      const act = () =>
        (tournamentNeat as Neat & { getParent(): Network }).getParent();
      // Act & Assert: expect throw
      expect(act).toThrow();
    });
  });
});
