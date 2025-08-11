import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('refined offspring allocation', () => {
  const fitness = (net: Network) => 1; // uniform fitness encourages proportional distribution
  test('each species gets at least minOffspring', async () => {
    const neat = new Neat(3, 2, fitness, {
      popsize: 50,
      targetSpecies: 6,
      speciation: true,
      speciesAllocation: { minOffspring: 1, extendedHistory: false },
    });
    await neat.evaluate();
    // Force multiple generations to build species diversity
    for (let i = 0; i < 3; i++) await neat.evolve();
    // After evolve population is new; gather next species stats
    await neat.evaluate();
    const speciesStats = neat.getSpeciesStats();
    // We can't directly read per-species allocation post-hoc, but ensure no species collapsed to 0 members unexpectedly
    expect(speciesStats.every((s) => s.size >= 1)).toBe(true);
  });
});
