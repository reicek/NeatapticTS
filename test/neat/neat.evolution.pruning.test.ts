import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

// Single expectation per test.

describe('Evolution-time pruning', () => {
  const fitness = (net: Network) => net.connections.length; // deterministic proxy
  // Pruning starts at generation 1 (i.e., after first evolve), mutation disabled for determinism.
  const neat = new Neat(2, 1, fitness, {
    popsize: 6,
    seed: 101,
    speciation: false,
    mutationRate: 0,
    mutationAmount: 0,
    evolutionPruning: {
      startGeneration: 1,
      interval: 1,
      targetSparsity: 0.5,
      rampGenerations: 0,
      method: 'magnitude',
    },
  });
  let postFirstEvolveAvg: number | undefined;
  test('records average after first evolve (pre-pruning baseline)', async () => {
    await neat.evaluate(); // gen 0
    await neat.evolve(); // gen 0->1 (no pruning yet)
    postFirstEvolveAvg =
      neat.population.reduce((s, g) => s + g.connections.length, 0) /
      neat.population.length;
    expect(postFirstEvolveAvg).toBeGreaterThan(0);
  });
  test('pruning at generation 1 reduces or maintains average connections by generation 2', async () => {
    await neat.evolve(); // gen 1->2 triggers pruning
    const avg =
      neat.population.reduce((s, g) => s + g.connections.length, 0) /
      neat.population.length;
    expect(avg).toBeLessThanOrEqual(postFirstEvolveAvg!);
  });
});
