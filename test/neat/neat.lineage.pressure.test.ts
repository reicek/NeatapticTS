import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

type NetworkWithLineage = Network & {
  _depth?: number;
  score?: number;
};

describe('Lineage pressure feature', () => {
  test('penalizeDeep reduces deep genome scores', async () => {
    const neat = new Neat(3, 1, () => 1, {
      popsize: 12,
      seed: 7,
      lineageTracking: true,
      lineagePressure: {
        enabled: true,
        mode: 'penalizeDeep',
        targetMeanDepth: 1,
        strength: 0.05,
      },
      telemetry: { enabled: true },
    });
    await neat.evolve();
    // Force another generation to build depth
    await neat.evolve();
    const population = neat.population as NetworkWithLineage[];
    const depths = population.map((genome) => genome._depth ?? 0);
    const scores = population.map((genome) => genome.score ?? 0);
    // Check if any genome with depth > target has score < 1 (penalized)
    const penalized = depths.some(
      (depthValue, index) => depthValue > 1 && scores[index] < 0.99
    );
    expect(penalized).toBe(true);
  });
});
