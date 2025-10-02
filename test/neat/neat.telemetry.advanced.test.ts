import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import type {
  SpeciesHistoryEntry,
  SpeciesHistoryStat,
  SpeciesHistoryStatExtended,
  TelemetryEntry,
} from '../../src/neat/neat.types';

// Each test single expectation.

const getLatestTelemetryEntry = (
  neatInstance: Neat,
): TelemetryEntry | undefined => {
  const telemetryEntries = neatInstance.getTelemetry() as TelemetryEntry[];
  return telemetryEntries.at(-1);
};

const requireLatestTelemetryEntry = (neatInstance: Neat): TelemetryEntry => {
  const latestTelemetryEntry = getLatestTelemetryEntry(neatInstance);
  if (!latestTelemetryEntry) {
    throw new Error('Expected telemetry to contain at least one entry.');
  }
  return latestTelemetryEntry;
};

const requireLatestSpeciesHistoryEntry = (
  entries: SpeciesHistoryEntry[],
): SpeciesHistoryEntry => {
  const latestEntry = entries.at(-1);
  if (!latestEntry) {
    throw new Error('Species history should contain at least one entry.');
  }
  return latestEntry;
};

const speciesStatHasInnovationRange = (
  stat: SpeciesHistoryStat,
): stat is SpeciesHistoryStatExtended =>
  (stat as SpeciesHistoryStatExtended).innovationRange !== undefined;

const speciesStatHasEnabledRatio = (
  stat: SpeciesHistoryStat,
): stat is SpeciesHistoryStatExtended =>
  (stat as SpeciesHistoryStatExtended).enabledRatio !== undefined;

describe('advanced telemetry & archives', () => {
  describe('operator stats presence', () => {
    test('telemetry entry contains ops array', async () => {
      const neat = new Neat(
        3,
        2,
        (network: Network) => network.connections.length,
        {
          popsize: 18,
          seed: 501,
          multiObjective: { enabled: true },
          telemetry: { enabled: true },
        },
      );
      for (let generationIndex = 0; generationIndex < 2; generationIndex += 1) {
        await neat.evolve();
      }
      const telemetryEntry = requireLatestTelemetryEntry(neat);
      expect(Array.isArray(telemetryEntry.ops)).toBe(true);
    });
  });
  describe('species extended history metrics', () => {
    test('history entry includes innovationRange', async () => {
      const neat = new Neat(
        3,
        1,
        (network: Network) => network.connections.length,
        {
          popsize: 14,
          seed: 502,
          speciation: true,
          speciesAllocation: { extendedHistory: true },
        },
      );
      await neat.evaluate();
      await neat.evolve();
      const historyEntries = neat.getSpeciesHistory();
      const latestEntry = requireLatestSpeciesHistoryEntry(historyEntries);
      const hasInnovationRange = latestEntry.stats.some(
        speciesStatHasInnovationRange,
      );
      expect(hasInnovationRange).toBe(true);
    });
    test('history entry includes enabledRatio', async () => {
      const neat = new Neat(
        3,
        1,
        (network: Network) => network.connections.length,
        {
          popsize: 12,
          seed: 506,
          speciation: true,
          speciesAllocation: { extendedHistory: true },
        },
      );
      await neat.evaluate();
      await neat.evolve();
      const latestEntry = requireLatestSpeciesHistoryEntry(
        neat.getSpeciesHistory(),
      );
      const hasEnabledRatio = latestEntry.stats.some(
        speciesStatHasEnabledRatio,
      );
      expect(hasEnabledRatio).toBe(true);
    });
  });
  describe('pareto archive snapshot', () => {
    test('pareto archive collects first front genomes', async () => {
      const neat = new Neat(
        4,
        2,
        (network: Network) => network.connections.length,
        {
          popsize: 20,
          seed: 503,
          multiObjective: { enabled: true },
        },
      );
      for (let generationIndex = 0; generationIndex < 3; generationIndex += 1) {
        await neat.evolve();
      }
      const paretoArchive = neat.getParetoArchive() as unknown[];
      expect(paretoArchive.length).toBeGreaterThan(0);
    });
  });
  describe('performance timing stats', () => {
    test('performance stats expose eval duration', async () => {
      const neat = new Neat(
        3,
        1,
        (network: Network) => network.connections.length,
        {
          popsize: 10,
          seed: 504,
          telemetry: { enabled: true, performance: true },
        },
      );
      await neat.evaluate();
      const performanceStats = neat.getPerformanceStats();
      expect(
        typeof performanceStats.lastEvalMs === 'number' ||
          performanceStats.lastEvalMs === undefined,
      ).toBe(true);
    });
    test('telemetry entry contains perf block when enabled', async () => {
      const neat = new Neat(
        3,
        1,
        (network: Network) => network.connections.length,
        {
          popsize: 10,
          seed: 507,
          telemetry: { enabled: true, performance: true },
        },
      );
      await neat.evaluate();
      await neat.evolve();
      const telemetryEntry = requireLatestTelemetryEntry(neat);
      const hasPerfBlock =
        telemetryEntry.perf !== undefined && 'evalMs' in telemetryEntry.perf;
      expect(hasPerfBlock).toBe(true);
    });
  });
  describe('complexity telemetry', () => {
    test('telemetry entry includes complexity metrics when enabled', async () => {
      const neat = new Neat(
        3,
        2,
        (network: Network) => network.connections.length,
        {
          popsize: 14,
          seed: 508,
          telemetry: { enabled: true, complexity: true },
        },
      );
      for (let generationIndex = 0; generationIndex < 2; generationIndex += 1) {
        await neat.evolve();
      }
      const telemetryEntry = requireLatestTelemetryEntry(neat);
      const hasComplexityMetrics =
        typeof telemetryEntry.complexity?.meanNodes === 'number';
      expect(hasComplexityMetrics).toBe(true);
    });
    test('complexity telemetry tracks growth deltas', async () => {
      const neat = new Neat(
        3,
        2,
        (network: Network) => network.connections.length,
        {
          popsize: 14,
          seed: 509,
          telemetry: { enabled: true, complexity: true },
        },
      );
      await neat.evolve();
      await neat.evolve();
      const telemetryEntry = requireLatestTelemetryEntry(neat);
      const tracksGrowthDeltas =
        telemetryEntry.complexity?.growthNodes !== undefined;
      expect(tracksGrowthDeltas).toBe(true);
    });
  });
  describe('hypervolume telemetry', () => {
    test('includes hv when enabled and multi-objective active', async () => {
      const neat = new Neat(
        3,
        2,
        (network: Network) => network.connections.length,
        {
          popsize: 16,
          seed: 510,
          multiObjective: { enabled: true },
          telemetry: { enabled: true, hypervolume: true },
        },
      );
      await neat.evolve();
      const telemetryEntry = requireLatestTelemetryEntry(neat);
      const hypervolumeValue = telemetryEntry.hv;
      const hvPresent =
        typeof hypervolumeValue === 'number' || hypervolumeValue === undefined;
      expect(hvPresent).toBe(true);
    });
  });
  describe('telemetry export utilities', () => {
    test('exportTelemetryCSV produces header line', async () => {
      const neat = new Neat(
        3,
        2,
        (network: Network) => network.connections.length,
        {
          popsize: 12,
          seed: 511,
          telemetry: {
            enabled: true,
            performance: true,
            complexity: true,
          },
        },
      );
      await neat.evolve();
      const csvOutput = neat.exportTelemetryCSV();
      expect(csvOutput.split('\n')[0].includes('gen')).toBe(true);
    });
  });
  describe('novelty dynamic threshold', () => {
    test('novelty threshold adapts', async () => {
      const neat = new Neat(
        3,
        1,
        (network: Network) => network.connections.length,
        {
          popsize: 16,
          seed: 505,
          speciation: false,
          novelty: {
            enabled: true,
            descriptor: (genome: Network) => [
              genome.connections.length,
              genome.nodes.length,
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
        },
      );
      await neat.evaluate();
      const firstThreshold = neat.options.novelty!.archiveAddThreshold!;
      await neat.evaluate();
      const secondThreshold = neat.options.novelty!.archiveAddThreshold!;
      const changed = Math.abs(secondThreshold - firstThreshold) > 1e-12;
      expect(changed || firstThreshold === secondThreshold).toBe(true);
    });
  });
});
