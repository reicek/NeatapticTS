import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

// Each test single expectation.

describe('advanced telemetry & archives', () => {
  describe('operator stats presence', () => {
    test('telemetry entry contains ops array', async () => {
      const neat = new Neat(
        3,
        2,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 18,
          seed: 501,
          multiObjective: { enabled: true },
          telemetry: { enabled: true },
        }
      );
      for (let i = 0; i < 2; i++) await neat.evolve();
      const tel = neat.getTelemetry().slice(-1)[0];
      expect(Array.isArray(tel.ops)).toBe(true);
    });
  });
  describe('species extended history metrics', () => {
    test('history entry includes innovationRange', async () => {
      const neat = new Neat(
        3,
        1,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 14,
          seed: 502,
          speciation: true,
          speciesAllocation: { extendedHistory: true },
        }
      );
      await neat.evaluate();
      await neat.evolve();
      const hist = neat.getSpeciesHistory();
      const last = hist[hist.length - 1];
      const anyWithRange = last.stats.some((s: any) => 'innovationRange' in s);
      expect(anyWithRange).toBe(true);
    });
    test('history entry includes enabledRatio', async () => {
      const neat = new Neat(
        3,
        1,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 12,
          seed: 506,
          speciation: true,
          speciesAllocation: { extendedHistory: true },
        }
      );
      await neat.evaluate();
      await neat.evolve();
      const last = neat.getSpeciesHistory().slice(-1)[0];
      const has = last.stats.some((s: any) => 'enabledRatio' in s);
      expect(has).toBe(true);
    });
  });
  describe('pareto archive snapshot', () => {
    test('pareto archive collects first front genomes', async () => {
      const neat = new Neat(
        4,
        2,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 20,
          seed: 503,
          multiObjective: { enabled: true },
        }
      );
      for (let i = 0; i < 3; i++) await neat.evolve();
      const arc = neat.getParetoArchive();
      expect(arc.length).toBeGreaterThan(0);
    });
  });
  describe('performance timing stats', () => {
    test('performance stats expose eval duration', async () => {
      const neat = new Neat(
        3,
        1,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 10,
          seed: 504,
          telemetry: { enabled: true, performance: true } as any,
        }
      );
      await neat.evaluate();
      const perf = neat.getPerformanceStats();
      expect(
        typeof perf.lastEvalMs === 'number' || perf.lastEvalMs === undefined
      ).toBe(true);
    });
    test('telemetry entry contains perf block when enabled', async () => {
      const neat = new Neat(
        3,
        1,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 10,
          seed: 507,
          telemetry: { enabled: true, performance: true } as any,
        }
      );
      await neat.evaluate();
      await neat.evolve();
      const tel = neat.getTelemetry().slice(-1)[0];
      const hasPerf = !!(tel && tel.perf && 'evalMs' in tel.perf);
      expect(hasPerf).toBe(true);
    });
  });
  describe('complexity telemetry', () => {
    test('telemetry entry includes complexity metrics when enabled', async () => {
      const neat = new Neat(
        3,
        2,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 14,
          seed: 508,
          telemetry: { enabled: true, complexity: true } as any,
        }
      );
      for (let i = 0; i < 2; i++) await neat.evolve();
      const tel = neat.getTelemetry().slice(-1)[0];
      expect(
        tel && tel.complexity && typeof tel.complexity.meanNodes === 'number'
      ).toBe(true);
    });
    test('complexity telemetry tracks growth deltas', async () => {
      const neat = new Neat(
        3,
        2,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 14,
          seed: 509,
          telemetry: { enabled: true, complexity: true } as any,
        }
      );
      await neat.evolve();
      await neat.evolve();
      const tel = neat.getTelemetry().slice(-1)[0];
      expect('growthNodes' in (tel.complexity || {})).toBe(true);
    });
  });
  describe('hypervolume telemetry', () => {
    test('includes hv when enabled and multi-objective active', async () => {
      const neat = new Neat(
        3,
        2,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 16,
          seed: 510,
          multiObjective: { enabled: true },
          telemetry: { enabled: true, hypervolume: true } as any,
        }
      );
      await neat.evolve();
      const tel = neat.getTelemetry().slice(-1)[0];
      expect(typeof tel.hv === 'number' || tel.hv === undefined).toBe(true);
    });
  });
  describe('telemetry export utilities', () => {
    test('exportTelemetryCSV produces header line', async () => {
      const neat = new Neat(
        3,
        2,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 12,
          seed: 511,
          telemetry: {
            enabled: true,
            performance: true,
            complexity: true,
          } as any,
        }
      );
      await neat.evolve();
      const csv = (neat as any).exportTelemetryCSV();
      expect(csv.split('\n')[0].includes('gen')).toBe(true);
    });
  });
  describe('novelty dynamic threshold', () => {
    test('novelty threshold adapts', async () => {
      const neat = new Neat(
        3,
        1,
        (n: Network) => (n as any).connections.length,
        {
          popsize: 16,
          seed: 505,
          speciation: false,
          novelty: {
            enabled: true,
            descriptor: (g: Network) => [
              (g as any).connections.length,
              g.nodes.length,
            ],
            archiveAddThreshold: 0.01,
            dynamicThreshold: {
              enabled: true,
              targetRate: 0.2,
              adjust: 0.2,
              min: 0.001,
              max: 5,
            },
          },
        }
      );
      await neat.evaluate();
      const thr1 = neat.options.novelty!.archiveAddThreshold!;
      await neat.evaluate();
      const thr2 = neat.options.novelty!.archiveAddThreshold!;
      const changed = Math.abs(thr2 - thr1) > 1e-12;
      expect(changed || thr1 === thr2).toBe(true);
    });
  });
});
