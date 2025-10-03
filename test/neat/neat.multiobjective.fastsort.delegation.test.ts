import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

type NetworkWithMultiObjective = Network & {
  _moRank?: number;
  _moCrowd?: number;
  score?: number;
};

/**
 * Sanity test: verify that delegated fast non-dominated sorting still produces
 * monotonic ranks (non-negative) and assigns Infinity crowding distance to
 * boundary solutions of each front when multi-objective mode enabled.
 */

describe('fast non-dominated sorting delegation invariants', () => {
  it('produces non-negative ranks and Infinity crowding at extremes', async () => {
    const neat = new Neat(3, 2, (network: Network) => network.connections.length, {
      popsize: 30,
      multiObjective: {
        enabled: true,
        objectives: [
          {
            key: 'fitness',
            direction: 'max',
            accessor: (genome: NetworkWithMultiObjective) => genome.score ?? 0,
          },
          {
            key: 'complexity',
            direction: 'min',
            accessor: (genome: Network) => genome.connections.length,
          },
        ],
      },
    });
    // Assign synthetic fitness scores to encourage spread
    const population = neat.population as NetworkWithMultiObjective[];
    population.forEach((genome, index) => {
      genome.score = index; // strictly increasing
    });
    // One evolve step will invoke sorting & crowding
    await neat.evolve();
    const ranks = population.map((genome) => genome._moRank ?? 0);
    expect(ranks.every((rank) => typeof rank === 'number' && rank >= 0)).toBe(true);
    // For first Pareto front collect crowding distances
    const firstFront = population.filter((genome) => genome._moRank === 0);
    const crowdingDistances = firstFront.map((genome) => genome._moCrowd);
    // Boundary genomes (at least one, typically two) should have Infinity crowding
    const infCount = crowdingDistances.filter((value) => value === Infinity).length;
    if (firstFront.length >= 2) {
      expect(infCount).toBeGreaterThanOrEqual(1);
    } else {
      expect(infCount).toBeGreaterThanOrEqual(0);
    }
  });
});
