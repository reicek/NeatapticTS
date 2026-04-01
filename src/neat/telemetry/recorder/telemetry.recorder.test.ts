import {
  applyTelemetrySelect,
  buildTelemetryEntry,
  structuralEntropy,
} from './telemetry.recorder';
import type {
  DiversityStats,
  GenomeDetailed,
  ObjectiveEvent,
  TelemetryEntry,
} from '../../shared/neat.shared.types';

type TelemetryContextStub = Record<string, unknown> & {
  generation: number;
  population: GenomeDetailed[];
  _species: unknown[];
  _operatorStats: Map<string, { success: number; attempts: number }>;
  _diversityStats: DiversityStats;
  _getObjectives: () => Array<{ key: string }>;
  _getRNG: () => () => number;
  options: {
    multiObjective: { enabled: boolean };
    telemetry: { complexity: boolean };
  };
  _pendingObjectiveAdds?: string[];
  _pendingObjectiveRemoves?: string[];
  _objectiveEvents?: ObjectiveEvent[];
  _lineageEnabled?: boolean;
};

type TelemetrySelectContextStub = Record<string, unknown> & {
  generation: number;
  _getRNG: () => () => number;
  options: Record<string, unknown>;
  _telemetrySelect?: Set<string>;
};

type GraphStub = {
  nodes: Array<{ geneId: number }>;
  connections: Array<{
    from: { geneId: number };
    to: { geneId: number };
    enabled: boolean;
  }>;
  _entropyGen?: number;
  _entropyVal?: number;
};

function createTelemetryContext(): TelemetryContextStub {
  const population: GenomeDetailed[] = [
    {
      score: 1,
      nodes: [{ geneId: 1 }, { geneId: 2 }],
      connections: [
        {
          from: { geneId: 1 },
          to: { geneId: 2 },
          enabled: true,
        },
      ],
      _moRank: 0,
      _depth: 2,
      _parents: [1, 2],
      _id: 1,
    },
    {
      score: 0.5,
      nodes: [{ geneId: 3 }, { geneId: 4 }],
      connections: [
        {
          from: { geneId: 3 },
          to: { geneId: 4 },
          enabled: true,
        },
      ],
      _moRank: 1,
      _depth: 1,
      _parents: [1],
      _id: 2,
    },
  ];
  const diversityStats: DiversityStats = {
    meanCompat: 0.5,
    varCompat: 0.1,
    meanEntropy: 0.2,
    varEntropy: 0.05,
    graphletEntropy: 0.3,
    lineageMeanDepth: 1,
    lineageMeanPairDist: 0.4,
  };

  return {
    generation: 3,
    population,
    _species: [],
    _operatorStats: new Map<string, { success: number; attempts: number }>(),
    _diversityStats: diversityStats,
    _getObjectives: () => [],
    _getRNG: () => () => 0.42,
    options: {
      multiObjective: { enabled: false },
      telemetry: { complexity: false },
    },
  };
}

function buildMonoObjectiveEntryWithObjectiveEvents(): {
  telemetryContext: TelemetryContextStub;
  telemetryEntry: TelemetryEntry;
} {
  const telemetryContext = createTelemetryContext();
  telemetryContext._pendingObjectiveAdds = ['a', 'b'];
  telemetryContext._pendingObjectiveRemoves = ['c'];
  telemetryContext._objectiveEvents = [];
  telemetryContext._lineageEnabled = true;

  const telemetryEntry = buildTelemetryEntry.call(
    telemetryContext,
    telemetryContext.population[0],
  ) as TelemetryEntry;

  return { telemetryContext, telemetryEntry };
}

function buildMonoObjectiveEntryWithoutLineage(): TelemetryEntry {
  const telemetryContext = createTelemetryContext();

  return buildTelemetryEntry.call(
    telemetryContext,
    telemetryContext.population[0],
  ) as TelemetryEntry;
}

function createTelemetrySelectContext(): TelemetrySelectContextStub {
  return {
    generation: 0,
    _getRNG: () => () => Math.random(),
    options: {},
  };
}

describe('neat telemetry recorder chapter', () => {
  describe('applyTelemetrySelect', () => {
    describe('given a telemetry selection whitelist exists', () => {
      it('removes non-core keys that are not whitelisted', () => {
        // Arrange
        const telemetryContext = createTelemetrySelectContext();
        telemetryContext._telemetrySelect = new Set(['keep']);
        const telemetryEntry: Record<string, unknown> = {
          gen: 1,
          best: 0.5,
          species: 3,
          keep: 1,
          drop: 2,
        };

        // Act
        const filteredEntry = applyTelemetrySelect.call(
          telemetryContext,
          telemetryEntry,
        );

        // Assert
        expect(filteredEntry).toEqual({
          gen: 1,
          best: 0.5,
          species: 3,
          keep: 1,
        });
      });
    });

    describe('given no telemetry selection whitelist exists', () => {
      it('returns the original entry reference unchanged', () => {
        // Arrange
        const telemetryContext = createTelemetrySelectContext();
        const telemetryEntry: Record<string, unknown> = {
          gen: 2,
          best: 1,
          species: 1,
          keep: 1,
        };

        // Act
        const filteredEntry = applyTelemetrySelect.call(
          telemetryContext,
          telemetryEntry,
        );

        // Assert
        expect(filteredEntry).toBe(telemetryEntry);
      });
    });
  });

  describe('structuralEntropy', () => {
    describe('given the same graph is inspected twice in the same generation', () => {
      it('reuses the cached entropy value', () => {
        // Arrange
        const telemetryContext = createTelemetrySelectContext();
        const graph: GraphStub = {
          nodes: [{ geneId: 1 }, { geneId: 2 }],
          connections: [
            {
              from: { geneId: 1 },
              to: { geneId: 2 },
              enabled: true,
            },
          ],
        };

        // Act
        const firstEntropy = structuralEntropy.call(telemetryContext, graph);
        graph.connections.push({
          from: { geneId: 1 },
          to: { geneId: 2 },
          enabled: true,
        });
        const secondEntropy = structuralEntropy.call(telemetryContext, graph);

        // Assert
        expect(secondEntropy).toBe(firstEntropy);
      });
    });
  });

  describe('buildTelemetryEntry', () => {
    describe('given pending objective lifecycle changes exist in mono-objective mode', () => {
      it('records add and remove events onto the telemetry entry', () => {
        // Arrange
        const { telemetryEntry } = buildMonoObjectiveEntryWithObjectiveEvents();

        // Assert
        expect(telemetryEntry.objEvents).toEqual([
          { gen: 3, type: 'add', key: 'a' },
          { gen: 3, type: 'add', key: 'b' },
          { gen: 3, type: 'remove', key: 'c' },
        ]);
      });

      it('persists the flushed lifecycle events onto the recorder context', () => {
        // Arrange
        const { telemetryContext } =
          buildMonoObjectiveEntryWithObjectiveEvents();

        // Assert
        expect(telemetryContext._objectiveEvents).toEqual([
          { gen: 3, type: 'add', key: 'a' },
          { gen: 3, type: 'add', key: 'b' },
          { gen: 3, type: 'remove', key: 'c' },
        ]);
      });

      it('clears the pending objective queues after recording the entry', () => {
        // Arrange
        const { telemetryContext } =
          buildMonoObjectiveEntryWithObjectiveEvents();

        // Assert
        expect([
          telemetryContext._pendingObjectiveAdds,
          telemetryContext._pendingObjectiveRemoves,
        ]).toEqual([[], []]);
      });
    });

    describe('given lineage telemetry is enabled', () => {
      it('attaches the fittest genome lineage block with the derived mean depth', () => {
        // Arrange
        const { telemetryEntry } = buildMonoObjectiveEntryWithObjectiveEvents();

        // Assert
        expect(telemetryEntry.lineage).toMatchObject({
          parents: [1, 2],
          depthBest: 2,
          meanDepth: 1.5,
        });
      });

      it('exposes lineage depth and pair-distance stats inside the diversity block', () => {
        // Arrange
        const { telemetryEntry } = buildMonoObjectiveEntryWithObjectiveEvents();

        // Assert
        expect(telemetryEntry.diversity).toMatchObject({
          lineageMeanDepth: 1,
          lineageMeanPairDist: 0.4,
        });
      });
    });

    describe('given lineage telemetry is disabled', () => {
      it('omits the lineage block from the entry', () => {
        // Arrange
        const telemetryEntry = buildMonoObjectiveEntryWithoutLineage();

        // Assert
        expect(telemetryEntry.lineage).toBeUndefined();
      });
    });
  });
});
