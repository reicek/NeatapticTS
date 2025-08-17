import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/** Tests covering selection strategies (POWER, TOURNAMENT error path). */
describe('NEAT Selection Strategies', () => {
  describe('POWER selection sorts when unsorted', () => {
    /** Fitness equals negative node count to force varied scores. */
    const fitness = (n: Network) => -n.nodes.length;
    /** Instance with POWER selection. */
    const neat = new Neat(2, 1, fitness, {
      popsize: 6,
      seed: 555,
      selection: { name: 'POWER', power: 1 },
    });
    beforeAll(async () => {
      await neat.evaluate();
      // Arrange: intentionally swap two genome scores to simulate unsorted state
      const a = neat.population[0];
      const b = neat.population[1];
      // Force score inversion (unsafe cast for test only)
      (a as any).score = 1;
      (b as any).score = 5; // ensure second > first triggers sort branch
    });
    test('getParent returns a genome (post-sort path executed)', () => {
      // Act: select parent
      const parent = (neat as any).getParent();
      // Assert: parent object returned
      expect(typeof parent).toBe('object');
    });
  });
  describe('TOURNAMENT selection invalid size error', () => {
    /** Constant fitness (scores irrelevant). */
    const fitness = (_: Network) => 1;
    /** Instance with small population so oversized tournament triggers error. */
    const neat = new Neat(2, 1, fitness, {
      popsize: 3,
      seed: 556,
      selection: { name: 'TOURNAMENT', size: 10, probability: 0.5 },
    });
    test('throws when tournament size > population and not suppressed', () => {
      // Arrange: callable invoking selection
      const act = () => (neat as any).getParent();
      // Act & Assert: expect throw
      expect(act).toThrow();
    });
  });
});
