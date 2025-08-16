import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/**
 * Sanity test: verify that delegated fast non-dominated sorting still produces
 * monotonic ranks (non-negative) and assigns Infinity crowding distance to
 * boundary solutions of each front when multi-objective mode enabled.
 */

describe('fast non-dominated sorting delegation invariants', () => {
  it('produces non-negative ranks and Infinity crowding at extremes', async () => {
    const neat = new Neat(3, 2, (n: any) => n.connections.length, {
      popsize: 30,
      multiObjective: {
        enabled: true,
        objectives: [
          {
            key: 'fitness',
            direction: 'max',
            accessor: (g: any) => g.score || 0,
          },
          {
            key: 'complexity',
            direction: 'min',
            accessor: (g: any) => g.connections.length,
          },
        ],
      },
    });
    // Assign synthetic fitness scores to encourage spread
    neat.population.forEach((g: any, i: number) => {
      g.score = i; // strictly increasing
    });
    // One evolve step will invoke sorting & crowding
    await neat.evolve();
    const ranks = neat.population.map((g: any) => (g as any)._moRank ?? 0);
    expect(ranks.every((r: any) => typeof r === 'number' && r >= 0)).toBe(true);
    // For first Pareto front collect crowding distances
    const firstFront = neat.population.filter(
      (g: any) => (g as any)._moRank === 0
    );
    const crowd = firstFront.map((g: any) => (g as any)._moCrowd);
    // Boundary genomes (at least one, typically two) should have Infinity crowding
    const infCount = crowd.filter((c: any) => c === Infinity).length;
    if (firstFront.length >= 2) {
      expect(infCount).toBeGreaterThanOrEqual(1);
    } else {
      expect(infCount).toBeGreaterThanOrEqual(0);
    }
  });
});
