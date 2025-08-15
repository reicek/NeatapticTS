import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Telemetry CSV lineage flattening', () => {
  test('includes lineage and diversity lineage fields', async () => {
    const neat = new Neat(2, 1, (n: Network) => Math.random(), {
      popsize: 8,
      lineageTracking: true,
      telemetry: { enabled: true, logEvery: 1 },
      diversityMetrics: { enabled: true },
    });
    await neat.evolve();
    await neat.evolve();
    const csv = (neat as any).exportTelemetryCSV(10) as string;
    expect(csv).toMatch(/lineage.depthBest/);
    // diversity lineage fields may appear (lineageMeanDepth)
    expect(csv).toMatch(/diversity.lineageMeanDepth/);
  });
});
