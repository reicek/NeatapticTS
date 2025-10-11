import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

type DiversityStats = {
  meanEntropy?: number;
  varEntropy?: number;
};

type TelemetryEntry = ReturnType<Neat['getTelemetry']>[number];

const entropyResults: {
  initial?: number;
  shrunk?: number;
  grown?: number;
} = {};
const ancestorEpsilonResults: { increased?: number; decreased?: number } = {};
const rngExportInfo: { headerCols?: string[]; rngVal?: string } = {};

describe('Adaptive entropy sharing & ancestor uniqueness objective adjustments', () => {
  const fitness = (net: Network) => net.nodes.length;

  describe('entropySharingTuning', () => {
    let initial: number;
    let shrunk: number;
    let grown: number;
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
      const lowVarianceStats: DiversityStats = { varEntropy: 0.01 };
      Reflect.set(neat, '_diversityStats', lowVarianceStats);
      await neat.evaluate();
      await neat.evolve();
      shrunk = neat.options.sharingSigma!;
      // High variance -> grow
      const highVarianceStats: DiversityStats = { varEntropy: 1.0 };
      Reflect.set(neat, '_diversityStats', highVarianceStats);
      await neat.evaluate();
      await neat.evolve();
      grown = neat.options.sharingSigma!;
      // Store on describe scope
      Object.assign(entropyResults, { initial, shrunk, grown });
    });
    it('shrinks sigma under low entropy variance', () => {
      expect(entropyResults.shrunk!).toBeLessThan(entropyResults.initial!);
    });
    it('expands sigma under high entropy variance', () => {
      expect(entropyResults.grown!).toBeGreaterThan(entropyResults.shrunk!);
    });
  });

  describe('ancestorUniqAdaptive epsilon mode', () => {
    let increased: number;
    let decreased: number;
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
      const telemetryEntriesLow = neat.getTelemetry();
      const lastEntryLow = telemetryEntriesLow.at(-1) as
        | TelemetryEntry
        | undefined;
      if (lastEntryLow?.lineage) {
        lastEntryLow.lineage.ancestorUniq = 0.1;
      }
      await neat.evaluate();
      await neat.evolve();
      increased = neat.options.multiObjective!.dominanceEpsilon!;
      // Force high uniqueness then evolve to trigger decrease
      const telemetryEntriesHigh = neat.getTelemetry();
      const lastEntryHigh = telemetryEntriesHigh.at(-1) as
        | TelemetryEntry
        | undefined;
      if (lastEntryHigh?.lineage) {
        lastEntryHigh.lineage.ancestorUniq = 0.95;
      }
      await neat.evaluate();
      await neat.evolve();
      decreased = neat.options.multiObjective!.dominanceEpsilon!;
      Object.assign(ancestorEpsilonResults, { increased, decreased });
    });
    it('increases dominanceEpsilon when ancestorUniq low', () => {
      expect(ancestorEpsilonResults.increased).toBeGreaterThan(0.01);
    });
    it('decreases dominanceEpsilon when ancestorUniq high', () => {
      expect(ancestorEpsilonResults.decreased!).toBeLessThan(
        ancestorEpsilonResults.increased!
      );
    });
  });

  describe('telemetry RNG export', () => {
    let headerCols: string[];
    let rngVal: string;
    beforeAll(async () => {
      const neat = new Neat(2, 1, fitness, {
        popsize: 10,
        seed: 123,
        telemetry: { enabled: true },
        rngState: true,
        multiObjective: { enabled: true },
      });
      for (let generationIndex = 0; generationIndex < 3; generationIndex += 1) {
        await neat.evaluate();
        await neat.evolve();
      }
      const csv = neat.exportTelemetryCSV();
      const lines = csv.split(/\r?\n/);
      headerCols = lines[0].split(',');
      const rngColumnIndex = headerCols.indexOf('rng');
      rngVal = lines[1].split(',')[rngColumnIndex];
      Object.assign(rngExportInfo, { headerCols, rngVal });
    });
    it('includes rng column', () => {
      expect(rngExportInfo.headerCols!).toContain('rng');
    });
    it('provides numeric rng value', () => {
      expect(rngExportInfo.rngVal!).toMatch(/\d+/);
    });
  });
});
