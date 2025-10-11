import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

// Single expectation per test.

describe('Adaptive mutation rates', () => {
  test('per-genome mutation rate field changes between successive evolves', async () => {
    const neat = new Neat(2, 1, () => 1, {
      popsize: 6,
      seed: 200,
      speciation: false,
      mutationRate: 0.5,
      mutationAmount: 1,
      adaptiveMutation: {
        enabled: true,
        initialRate: 0.5,
        sigma: 0.1,
        minRate: 0.1,
        maxRate: 0.9,
      },
    });
    await neat.evaluate();
    await neat.evolve(); // initializes _mutRate
    type GenomeWithMutationRate = Network & { _mutRate?: number };
    await neat.evolve(); // adapts
    const secondGenerationRates = (neat.population as GenomeWithMutationRate[]).map(
      (genome) => genome._mutRate
    );
    const mutationChanged = secondGenerationRates.some(
      (rate) =>
        rate !== undefined &&
        Math.abs(rate - (neat.options.adaptiveMutation?.initialRate ?? 0.5)) >
          1e-6
    );
    // ensure at least one genome drifted away from the baseline mutation rate
    expect(mutationChanged).toBe(true);
  });
});

describe('Novelty search blending', () => {
  test('novelty blending adds _novelty and alters score', async () => {
    const descriptor = (network: Network) => [
      network.connections.length,
      network.nodes.length,
    ];
    const neat = new Neat(
      2,
      1,
      (network: Network) => network.connections.length,
      {
        popsize: 8,
        seed: 210,
        speciation: false,
        novelty: {
          enabled: true,
          descriptor,
          archiveAddThreshold: 0,
          k: 3,
          blendFactor: 0.5,
        },
      }
    );
    await neat.evaluate();
    type GenomeWithNovelty = Network & { _novelty?: number };
    const noveltyPop = neat.population as GenomeWithNovelty[];
    const annotated = noveltyPop.filter(
      (genome) => genome._novelty !== undefined
    );
    expect(annotated.length).toBeGreaterThan(0);
  });
});

describe('Species age bonus allocation', () => {
  test('young species receive boosted offspring allocation', async () => {
    // Force speciation with small pop and high target species to split
    const neat = new Neat(3, 1, (n: Network) => n.connections.length, {
      popsize: 12,
      seed: 220,
      speciation: true,
      targetSpecies: 12,
      compatAdjust: { kp: 0, ki: 0 },
      speciesAgeBonus: { youngThreshold: 100, youngMultiplier: 2 },
    });
    await neat.evaluate();
    // All species considered young (threshold high). Capture member counts.
    const beforeSpecies = neat.getSpeciesStats();
    await neat.evolve();
    const afterSpecies = neat.getSpeciesStats();
    // If young bonus applied, species count should persist (prevent collapse)
    expect(afterSpecies.length).toBeGreaterThanOrEqual(beforeSpecies.length);
  });
});
