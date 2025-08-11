import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Lineage & Auto-Entropy Objective', () => {
  test('lineage parents tracked and entropy objective added', async () => {
    const neat = new Neat(
      2,
      1,
      (n: Network) => {
        return Math.random();
      },
      {
        popsize: 12,
        multiObjective: { enabled: true, autoEntropy: true },
        telemetry: { enabled: true, logEvery: 1 },
        lineageTracking: true,
        mutation: [] as any, // disable mutation for speed
      }
    );
    // run a couple generations
    for (let i = 0; i < 3; i++) await neat.evolve();
    const telem = neat.getTelemetry();
    expect(telem.length).toBeGreaterThan(0);
    const last = telem[telem.length - 1];
    // lineage field exists with depth metrics
    expect(last.lineage).toBeDefined();
    expect(typeof last.lineage.depthBest).toBe('number');
    expect(typeof last.lineage.meanDepth).toBe('number');
    // diversity stats include lineage depth if diversity metrics enabled (may be 0 if disabled)
    if (last.diversity) {
      expect(last.diversity.lineageMeanDepth).toBeDefined();
      expect(last.diversity.lineageMeanPairDist).toBeDefined();
    }
    // objectives include entropy
    const objKeys = neat.getObjectives().map((o) => o.key);
    expect(objKeys).toContain('entropy');
    const lineageSnap = neat.getLineageSnapshot();
    // offspring after first generation should have parents array
    const withParents = lineageSnap.filter(
      (e) => e.parents && e.parents.length > 0
    );
    expect(withParents.length).toBeGreaterThan(0);
  });

  test('can disable lineage tracking', async () => {
    const neat = new Neat(2, 1, (n: Network) => Math.random(), {
      popsize: 10,
      lineageTracking: false,
      telemetry: { enabled: true },
    });
    await neat.evolve();
    const telem = neat.getTelemetry();
    const last = telem[telem.length - 1];
    expect(last.lineage).toBeUndefined();
  });
});
