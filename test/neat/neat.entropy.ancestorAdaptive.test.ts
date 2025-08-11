import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Adaptive entropy sharing & ancestor uniqueness objective adjustments', () => {
  const fitness = (net: Network) => net.nodes.length;

  describe('entropySharingTuning', () => {
    let initial: number, shrunk: number, grown: number;
    beforeAll(async () => {
      const neat = new Neat(3, 2, fitness, {
        popsize: 20,
        seed: 7,
        speciation: true,
        sharingSigma: 3.0,
        diversityMetrics: { enabled: false },
        entropySharingTuning: {
          enabled: true,
          targetEntropyVar: 0.2,
          adjustRate: 0.2,
          minSigma: 0.5,
          maxSigma: 10,
        },
      });
      // Baseline evolve
      await neat.evaluate();
      await neat.evolve();
      initial = neat.options.sharingSigma!;
      // Low variance -> shrink
      (neat as any)._diversityStats = { varEntropy: 0.01 };
      await neat.evaluate();
      await neat.evolve();
      shrunk = neat.options.sharingSigma!;
      // High variance -> grow
      (neat as any)._diversityStats = { varEntropy: 1.0 };
      await neat.evaluate();
      await neat.evolve();
      grown = neat.options.sharingSigma!;
      // Store on describe scope
      (global as any).__entropyResults = { initial, shrunk, grown };
    });
    it('shrinks sigma under low entropy variance', () => {
      const { initial, shrunk } = (global as any).__entropyResults;
      expect(shrunk).toBeLessThan(initial);
    });
    it('expands sigma under high entropy variance', () => {
      const { shrunk, grown } = (global as any).__entropyResults;
      expect(grown).toBeGreaterThan(shrunk);
    });
  });

  describe('ancestorUniqAdaptive epsilon mode', () => {
    let increased: number, decreased: number;
    beforeAll(async () => {
      const neat = new Neat(2, 1, fitness, {
        popsize: 15,
        seed: 11,
        multiObjective: {
          enabled: true,
          complexityMetric: 'nodes',
          dominanceEpsilon: 0.01,
          adaptiveEpsilon: { enabled: true, targetFront: 5, adjust: 0.0 },
        },
        ancestorUniqAdaptive: {
          enabled: true,
          mode: 'epsilon',
          lowThreshold: 0.3,
          highThreshold: 0.8,
          adjust: 0.05,
          cooldown: 0,
        },
        lineageTracking: true,
        telemetry: { enabled: true },
      });
      // Gen 1
      await neat.evaluate();
      await neat.evolve();
      // Force low uniqueness then evolve to trigger increase
      neat.getTelemetry()[
        neat.getTelemetry().length - 1
      ].lineage.ancestorUniq = 0.1;
      await neat.evaluate();
      await neat.evolve();
      increased = neat.options.multiObjective!.dominanceEpsilon!;
      // Force high uniqueness then evolve to trigger decrease
      neat.getTelemetry()[
        neat.getTelemetry().length - 1
      ].lineage.ancestorUniq = 0.95;
      await neat.evaluate();
      await neat.evolve();
      decreased = neat.options.multiObjective!.dominanceEpsilon!;
      (global as any).__ancestorEps = { increased, decreased };
    });
    it('increases dominanceEpsilon when ancestorUniq low', () => {
      const { increased } = (global as any).__ancestorEps;
      expect(increased).toBeGreaterThan(0.01);
    });
    it('decreases dominanceEpsilon when ancestorUniq high', () => {
      const { increased, decreased } = (global as any).__ancestorEps;
      expect(decreased).toBeLessThan(increased);
    });
  });

  describe('telemetry RNG export', () => {
    let headerCols: string[], rngVal: string;
    beforeAll(async () => {
      const neat = new Neat(2, 1, fitness, {
        popsize: 10,
        seed: 123,
        telemetry: { enabled: true },
        rngState: true,
        multiObjective: { enabled: true },
      });
      for (let g = 0; g < 3; g++) {
        await neat.evaluate();
        await neat.evolve();
      }
      const csv = neat.exportTelemetryCSV();
      const lines = csv.split(/\r?\n/);
      headerCols = lines[0].split(',');
      const idx = headerCols.indexOf('rng');
      rngVal = lines[1].split(',')[idx];
      (global as any).__rngInfo = { headerCols, rngVal };
    });
    it('includes rng column', () => {
      const { headerCols } = (global as any).__rngInfo;
      expect(headerCols).toContain('rng');
    });
    it('provides numeric rng value', () => {
      const { rngVal } = (global as any).__rngInfo;
      expect(rngVal).toMatch(/\d+/);
    });
  });
});
