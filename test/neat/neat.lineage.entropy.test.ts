import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import type { TelemetryEntry } from '../../src/neat/neat.types';

describe('Lineage & Auto-Entropy Objective', () => {
  test('lineage parents tracked and entropy objective added', async () => {
    const neat = new Neat(
      2,
      1,
      (network: Network) => {
        void network;
        return Math.random();
      },
      {
        popsize: 12,
        multiObjective: { enabled: true, autoEntropy: true },
        telemetry: { enabled: true, logEvery: 1 },
        lineageTracking: true,
        mutation: [], // disable mutation for speed
      },
    );
    // run a couple generations
    for (let generationIndex = 0; generationIndex < 3; generationIndex += 1) {
      await neat.evolve();
    }
    const telemetryEntries = neat.getTelemetry() as TelemetryEntry[];
    expect(telemetryEntries.length).toBeGreaterThan(0);
    const lastEntry = telemetryEntries.at(-1) as TelemetryEntry;
    // lineage field exists with depth metrics
    expect(lastEntry.lineage).toBeDefined();
    expect(typeof lastEntry.lineage?.depthBest).toBe('number');
    expect(typeof lastEntry.lineage?.meanDepth).toBe('number');
    // diversity stats include lineage depth if diversity metrics enabled (may be 0 if disabled)
    if (lastEntry.diversity) {
      expect(lastEntry.diversity.lineageMeanDepth).toBeDefined();
      expect(lastEntry.diversity.lineageMeanPairDist).toBeDefined();
    }
    // objectives include entropy
    const objKeys = neat.getObjectives().map((o) => o.key);
    expect(objKeys).toContain('entropy');
  const lineageSnapshot = neat.getLineageSnapshot();
    // offspring after first generation should have parents array
    const withParents = lineageSnapshot.filter(
      (snapshot) => snapshot.parents && snapshot.parents.length > 0,
    );
    expect(withParents.length).toBeGreaterThan(0);
  });

  test('can disable lineage tracking', async () => {
    const neat = new Neat(2, 1, (network: Network) => {
      void network;
      return Math.random();
    }, {
      popsize: 10,
      lineageTracking: false,
      telemetry: { enabled: true },
    });
    await neat.evolve();
    const telemetryEntries = neat.getTelemetry() as TelemetryEntry[];
    const lastEntry = telemetryEntries.at(-1);
    expect(lastEntry?.lineage).toBeUndefined();
  });
});
