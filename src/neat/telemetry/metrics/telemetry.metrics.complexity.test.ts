import type {
  GenomeDetailed,
  NeatOptions,
} from '../../shared/neat.shared.types';
import type {
  TelemetryDiversityOptions,
  TelemetryEntryRecord,
} from '../types/telemetry.types';
import {
  applyComplexityStatsMonoObjective,
  applyComplexityStatsMultiObjective,
  buildComplexityEntry,
  computeAndStoreGrowthValues,
  computeEnabledRatios,
  computeMaxCounts,
  computeMeanCounts,
  computeMeanEnabledRatio,
} from './telemetry.metrics.complexity';

describe('neat telemetry complexity metrics chapter', () => {
  describe('computeMeanCounts', () => {
    describe('given both count arrays are empty', () => {
      it('returns zero means instead of dividing by zero', () => {
        // Arrange
        const counts = {
          connectionCounts: [],
          nodeCounts: [],
        };

        // Act
        const meanCounts = computeMeanCounts(counts);

        // Assert
        expect(meanCounts).toEqual({ meanConns: 0, meanNodes: 0 });
      });
    });
  });

  describe('computeMaxCounts', () => {
    describe('given both count arrays are empty', () => {
      it('returns zero maxima', () => {
        // Arrange
        const counts = {
          connectionCounts: [],
          nodeCounts: [],
        };

        // Act
        const maxCounts = computeMaxCounts(counts);

        // Assert
        expect(maxCounts).toEqual({ maxConns: 0, maxNodes: 0 });
      });
    });
  });

  describe('computeEnabledRatios', () => {
    describe('given one genome has both enabled and disabled connections while another has none', () => {
      it('counts disabled connections and returns zero for empty genomes', () => {
        // Arrange
        const populationSnapshot = [
          createGenomeDetailed({
            connections: [{ enabled: true }, { enabled: false }],
            nodeCount: 2,
          }),
          createGenomeDetailed({ nodeCount: 1 }),
        ];

        // Act
        const enabledRatios = computeEnabledRatios(populationSnapshot);

        // Assert
        expect(enabledRatios).toEqual([0.5, 0]);
      });
    });
  });

  describe('computeMeanEnabledRatio', () => {
    describe('given the ratio list is empty', () => {
      it('returns zero instead of dividing by zero', () => {
        // Arrange
        const enabledRatios: number[] = [];

        // Act
        const meanEnabledRatio = computeMeanEnabledRatio(enabledRatios);

        // Assert
        expect(meanEnabledRatio).toBe(0);
      });
    });
  });

  describe('buildComplexityEntry', () => {
    describe('given budget ceilings are omitted from telemetry options', () => {
      it('defaults both budget values to zero while rounding the metrics', () => {
        // Arrange
        const telemetryOptions = createTelemetryOptions({});

        // Act
        const complexityEntry = buildComplexityEntry(
          telemetryOptions,
          { meanConns: 3.456, meanNodes: 2.345 },
          { maxConns: 5, maxNodes: 4 },
          0.6666,
          { growthConns: 1.234, growthNodes: -0.444 },
        );

        // Assert
        expect(complexityEntry).toEqual({
          budgetMaxConns: 0,
          budgetMaxNodes: 0,
          growthConns: 1.23,
          growthNodes: -0.44,
          maxConns: 5,
          maxNodes: 4,
          meanConns: 3.46,
          meanEnabledRatio: 0.667,
          meanNodes: 2.35,
        });
      });
    });
  });

  describe('computeAndStoreGrowthValues', () => {
    describe('given no previous mean values are stored on the context', () => {
      it('returns zero growth while storing the current means', () => {
        // Arrange
        const telemetryContext: {
          _lastMeanConns?: number;
          _lastMeanNodes?: number;
        } = {};

        // Act
        const growthValues = computeAndStoreGrowthValues(telemetryContext, {
          meanConns: 1.5,
          meanNodes: 3,
        });

        // Assert
        expect({ growthValues, telemetryContext }).toEqual({
          growthValues: {
            growthConns: 0,
            growthNodes: 0,
          },
          telemetryContext: {
            _lastMeanConns: 1.5,
            _lastMeanNodes: 3,
          },
        });
      });
    });
  });

  describe('applyComplexityStatsMultiObjective', () => {
    describe('given complexity telemetry is disabled', () => {
      it('leaves the entry and growth context unchanged', () => {
        // Arrange
        const telemetryContext = {
          _lastMeanConns: 3,
          _lastMeanNodes: 2,
        };
        const telemetryOptions = createTelemetryOptions({
          complexityEnabled: false,
        });
        const entry = createTelemetryEntry();

        // Act
        applyComplexityStatsMultiObjective(
          telemetryContext,
          telemetryOptions,
          [createGenomeDetailed({ connectionCount: 1, nodeCount: 2 })],
          entry,
        );

        // Assert
        expect({ entry, telemetryContext }).toEqual({
          entry: createTelemetryEntry(),
          telemetryContext: {
            _lastMeanConns: 3,
            _lastMeanNodes: 2,
          },
        });
      });
    });

    describe('given complexity telemetry is enabled for a mixed population snapshot', () => {
      it('stores the rounded complexity snapshot and updates the growth baselines', () => {
        // Arrange
        const telemetryContext = {
          _lastMeanConns: 1,
          _lastMeanNodes: 2,
        };
        const telemetryOptions = createTelemetryOptions({
          maxConns: 20,
          maxNodes: 10,
        });
        const population = [
          createGenomeDetailed({
            connections: [{ enabled: true }, { enabled: false }],
            nodeCount: 3,
          }),
          createGenomeDetailed({
            connections: [{ enabled: true }],
            nodeCount: 5,
          }),
        ];
        const entry = createTelemetryEntry();

        // Act
        applyComplexityStatsMultiObjective(
          telemetryContext,
          telemetryOptions,
          population,
          entry,
        );

        // Assert
        expect({
          complexity: entry.complexity,
          telemetryContext,
        }).toEqual({
          complexity: {
            budgetMaxConns: 20,
            budgetMaxNodes: 10,
            growthConns: 0.5,
            growthNodes: 2,
            maxConns: 2,
            maxNodes: 5,
            meanConns: 1.5,
            meanEnabledRatio: 0.75,
            meanNodes: 4,
          },
          telemetryContext: {
            _lastMeanConns: 1.5,
            _lastMeanNodes: 4,
          },
        });
      });
    });
  });

  describe('applyComplexityStatsMonoObjective', () => {
    describe('given complexity telemetry is disabled', () => {
      it('leaves the entry unchanged', () => {
        // Arrange
        const telemetryContext = {};
        const telemetryOptions = createTelemetryOptions({
          complexityEnabled: false,
        });
        const entry = createTelemetryEntry();

        // Act
        applyComplexityStatsMonoObjective(
          telemetryContext,
          telemetryOptions,
          [createGenomeDetailed({ connectionCount: 1, nodeCount: 2 })],
          entry,
        );

        // Assert
        expect(entry).toEqual(createTelemetryEntry());
      });
    });

    describe('given complexity telemetry is enabled for a mono-objective population snapshot', () => {
      it('stores the rounded complexity snapshot on the entry', () => {
        // Arrange
        const telemetryContext = {
          _lastMeanConns: 0,
          _lastMeanNodes: 0,
        };
        const telemetryOptions = createTelemetryOptions({
          maxConns: 12,
          maxNodes: 8,
        });
        const entry = createTelemetryEntry();
        const populationSnapshot = [
          createGenomeDetailed({
            connections: [{ enabled: false }],
            nodeCount: 2,
          }),
          createGenomeDetailed({
            connections: [{ enabled: true }, { enabled: true }],
            nodeCount: 4,
          }),
        ];

        // Act
        applyComplexityStatsMonoObjective(
          telemetryContext,
          telemetryOptions,
          populationSnapshot,
          entry,
        );

        // Assert
        expect(entry.complexity).toEqual({
          budgetMaxConns: 12,
          budgetMaxNodes: 8,
          growthConns: 1.5,
          growthNodes: 3,
          maxConns: 2,
          maxNodes: 4,
          meanConns: 1.5,
          meanEnabledRatio: 0.5,
          meanNodes: 3,
        });
      });
    });
  });
});

function createTelemetryOptions(input: {
  complexityEnabled?: boolean;
  maxNodes?: number;
  maxConns?: number;
}): NeatOptions & TelemetryDiversityOptions {
  return {
    maxConns: input.maxConns,
    maxNodes: input.maxNodes,
    telemetry: {
      complexity: input.complexityEnabled ?? true,
    },
  } as NeatOptions & TelemetryDiversityOptions;
}

function createTelemetryEntry(): TelemetryEntryRecord {
  return {
    best: 0,
    gen: 0,
    hyper: 0,
    objImportance: {},
    ops: [],
    species: 0,
  };
}

function createGenomeDetailed(input: {
  nodeCount: number;
  connectionCount?: number;
  connections?: Array<{ enabled?: boolean }>;
}): GenomeDetailed {
  const backingNodes = Array.from(
    { length: Math.max(input.nodeCount, 1) },
    (_unusedNode, nodeIndex) => ({ geneId: nodeIndex }),
  );
  const connectionEntries =
    input.connections ??
    Array.from({ length: input.connectionCount ?? 0 }, () => ({
      enabled: true,
    }));

  return {
    _id: nextGenomeId++,
    connections: connectionEntries.map((connectionEntry) => ({
      enabled: connectionEntry.enabled,
      from: backingNodes[0],
      to: backingNodes.at(-1) ?? backingNodes[0],
    })),
    nodes: backingNodes.slice(0, input.nodeCount),
  };
}

let nextGenomeId = 1;
