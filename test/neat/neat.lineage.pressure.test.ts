import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Lineage pressure feature', () => {
  test('penalizeDeep reduces deep genome scores', async () => {
    const neat = new Neat(3, 1, (n: Network) => 1, {
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
    const depths = (neat as any).population.map((g: any) => g._depth || 0);
    const scores = (neat as any).population.map((g: any) => g.score || 0);
    // Check if any genome with depth > target has score < 1 (penalized)
    const penalized = depths.some(
      (d: number, i: number) => d > 1 && scores[i] < 0.99
    );
    expect(penalized).toBe(true);
  });
});
