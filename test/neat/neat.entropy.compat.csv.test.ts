import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

// Common trivial fitness (structure-dependent for slight variation)
const fitness = (net: Network) => net.nodes.length;

describe('Telemetry & adaptive entropy/speciation extensions', () => {
  describe('entropyCompatTuning', () => {
    let thrBaseline: number;
    let thrLow: number;
    let thrHigh: number;
    beforeAll(async () => {
      const neat = new Neat(3, 2, fitness, {
        popsize: 18,
        seed: 77,
        compatibilityThreshold: 3.0,
        entropyCompatTuning: {
          enabled: true,
          targetEntropy: 0.5,
          adjustRate: 0.2,
          deadband: 0.05,
          minThreshold: 0.5,
          maxThreshold: 6,
        },
      });
      await neat.evaluate();
      await neat.evolve();
      thrBaseline = neat.options.compatibilityThreshold!;
      // Low entropy -> decrease threshold
      (neat as any)._diversityStats = { meanEntropy: 0.1 };
      neat.population.forEach((g) => (g.score = undefined));
      await neat.evaluate(); // triggers tuning
      thrLow = neat.options.compatibilityThreshold!;
      // High entropy -> increase threshold
      (neat as any)._diversityStats = { meanEntropy: 1.0 };
      neat.population.forEach((g) => (g.score = undefined));
      await neat.evaluate();
      thrHigh = neat.options.compatibilityThreshold!;
      (global as any).__compat = { thrBaseline, thrLow, thrHigh };
    });
    it('decreases threshold under low entropy', () => {
      const { thrBaseline, thrLow } = (global as any).__compat;
      expect(thrLow).toBeLessThan(thrBaseline);
    });
    it('increases threshold under high entropy', () => {
      const { thrLow, thrHigh } = (global as any).__compat;
      expect(thrHigh).toBeGreaterThan(thrLow);
    });
  });

  describe('CSV export ops & objectives columns', () => {
    let headers: string[];
    let objectivesCell: string;
    let opsPresent: boolean;
    beforeAll(async () => {
      const neat = new Neat(2, 1, fitness, {
        popsize: 20,
        seed: 1234,
        telemetry: { enabled: true },
        rngState: true,
        multiObjective: { enabled: true, autoEntropy: true },
      });
      // Seed operator stats to guarantee ops column emission
      (neat as any)._operatorStats.set('ADD_NODE', { success: 3, attempts: 5 });
      for (let g = 0; g < 4; g++) {
        await neat.evaluate();
        await neat.evolve();
      }
      const csv = neat.exportTelemetryCSV();
      const lines = csv.split(/\r?\n/);
      headers = lines[0].split(',');
      const idxObj = headers.indexOf('objectives');
      objectivesCell = idxObj >= 0 ? lines[1].split(',')[idxObj] : '';
      opsPresent = headers.includes('ops');
      (global as any).__csvInfo = { headers, objectivesCell, opsPresent };
    });
    it('includes ops column when operator stats present', () => {
      const { opsPresent } = (global as any).__csvInfo;
      expect(opsPresent).toBe(true);
    });
    it('includes objectives column with active objective keys', () => {
      const { objectivesCell } = (global as any).__csvInfo;
      expect(objectivesCell.length).toBeGreaterThan(2);
    });
  });

  describe('telemetry objectives list', () => {
    let hasFitness: boolean;
    beforeAll(async () => {
      const neat = new Neat(2, 1, fitness, {
        popsize: 12,
        seed: 999,
        telemetry: { enabled: true },
        multiObjective: { enabled: true, autoEntropy: true },
      });
      await neat.evaluate();
      await neat.evolve();
      const last = neat.getTelemetry().slice(-1)[0];
      hasFitness =
        Array.isArray(last.objectives) && last.objectives.includes('fitness');
      (global as any).__telObj = { hasFitness };
    });
    it('records objectives array in telemetry', () => {
      const { hasFitness } = (global as any).__telObj;
      expect(hasFitness).toBe(true);
    });
  });
});
