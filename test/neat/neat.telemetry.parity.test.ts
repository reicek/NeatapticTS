import Neat from '../../src/neat';

/**
 * Parity test: ensure telemetry export (CSV vs JSONL) includes core fields and
 * that CSV headers cover nested complexity/perf/lineage fields present in JSON entries.
 */

describe('telemetry export parity', () => {
  it('includes core fields and consistent nested flattening', async () => {
    const neat = new Neat(3, 2, (n: any) => n.connections.length, {
      popsize: 20,
      telemetry: { enabled: true, complexity: true, performance: true },
      multiObjective: { enabled: true, objectives: [] },
      speciation: false,
    });
    // Run a few generations to accumulate telemetry snapshots
    for (let i = 0; i < 3; i++) await neat.evolve();
    const jsonl = neat.exportTelemetryJSONL();
    const csv = neat.exportTelemetryCSV();
    expect(jsonl.length).toBeGreaterThan(0);
    const lines = jsonl
      .trim()
      .split(/\n+/)
      .map((l) => JSON.parse(l));
    const last = lines[lines.length - 1];
    // Core presence
    ['gen', 'best', 'species', 'hyper', 'ops'].forEach((k) =>
      expect(last).toHaveProperty(k)
    );
    // If complexity present in JSON ensure CSV header contains complexity.meanNodes
    if (last.complexity) {
      expect(csv.split('\n')[0]).toContain('complexity.meanNodes');
    }
    if (last.perf) {
      expect(csv.split('\n')[0]).toContain('perf.evalMs');
    }
    // lineage optional; if present ensure CSV header has lineage.depthBest
    if (last.lineage) {
      expect(csv.split('\n')[0]).toContain('lineage.depthBest');
    }
  });
});
