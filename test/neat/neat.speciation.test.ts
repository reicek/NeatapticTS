import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

// All tests single expectation each (project guideline).

describe('Speciation', () => {
  describe('after several generations with structural diversity', () => {
    const neat = new Neat(2, 1, (net: Network) => net.connections.length, {
      popsize: 20,
      seed: 42,
      speciation: true,
      mutationRate: 0.3,
      mutationAmount: 1,
    });
    beforeAll(async () => {
      for (let i = 0; i < 3; i++) await neat.evolve();
    });
    test('forms at least one species', () => {
      expect(neat.getSpeciesStats().length).toBeGreaterThan(0);
    });
    test('assigns all members across species', () => {
      const stats = neat.getSpeciesStats();
      const totalMembers = stats.reduce((s, sp) => s + sp.size, 0);
      expect(totalMembers).toBe(neat.population.length);
    });
  });
  describe('stagnation pruning', () => {
    const neat = new Neat(2, 1, () => 1, {
      popsize: 15,
      seed: 7,
      speciation: true,
      stagnationGenerations: 2,
      mutationRate: 0.4,
      mutationAmount: 1,
    });
    beforeAll(async () => {
      for (let i = 0; i < 6; i++) await neat.evolve();
    });
    test('keeps at least one species after culling', () => {
      expect(neat.getSpeciesStats().length).toBeGreaterThanOrEqual(1);
    });
  });
});

describe('Kernel Fitness Sharing', () => {
  // Kernel sharing (sigma>0) yields per-genome divisors <= species size, so mean adjusted score should be >= simple averaging baseline.
  test('kernel sharing produces equal or higher mean adjusted score', async () => {
    const baseFitness = (net: Network) => net.connections.length;
    const neatA = new Neat(2, 1, baseFitness, {
      popsize: 12,
      seed: 120,
      speciation: true,
      sharingSigma: 0,
    });
    await neatA.evaluate();
    // Clone population JSON to reuse identical raw scores in second instance
    const exported = neatA.export();
    const neatB = new Neat(2, 1, baseFitness, {
      popsize: 12,
      seed: 120,
      speciation: true,
      sharingSigma: 2,
    });
    neatB.import(exported); // identical genomes before evaluation
    await neatB.evaluate();
    const meanA =
      neatA.population.reduce((s, g) => s + (g.score || 0), 0) /
      neatA.population.length;
    const meanB =
      neatB.population.reduce((s, g) => s + (g.score || 0), 0) /
      neatB.population.length;
    expect(meanB).toBeGreaterThanOrEqual(meanA);
  });
});

describe('Global Stagnation Injection', () => {
  test('injects fresh genomes after configured stagnation window', async () => {
    const neat = new Neat(2, 1, () => 1, {
      popsize: 12,
      seed: 55,
      speciation: true,
      globalStagnationGenerations: 2,
      mutationRate: 0.2,
      mutationAmount: 1,
    });
    // Force stagnation by constant fitness
    await neat.evolve(); // gen 0->1
    const beforeIds = neat.population.map((g) => g.nodes.length); // baseline structure counts
    await neat.evolve(); // gen 1->2 (still within window)
    await neat.evolve(); // gen 2->3 triggers injection (>=2 since last improvement)
    const afterIds = neat.population.map((g) => g.nodes.length);
    // Expect at least one structural baseline difference (some genomes replaced => different size probable)
    const changed = afterIds.some((v, i) => v !== beforeIds[i]);
    expect(changed).toBe(true);
  });
});
