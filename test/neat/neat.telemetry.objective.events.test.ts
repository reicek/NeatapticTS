/**
 * Tests covering objective events & lineage block in buildTelemetryEntry.
 * Single expectation per test.
 */
import { buildTelemetryEntry } from '../../src/neat/telemetry/recorder/telemetry.recorder';
import type {
  DiversityStats,
  GenomeDetailed,
  TelemetryEntry,
} from '../../src/neat/shared/neat.shared.types';

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
  _objectiveEvents?: Array<Record<string, unknown>>;
  _lineageEnabled?: boolean;
};

/** Helper to construct a minimal Neat-like context */
const makeCtx = (): TelemetryContextStub => {
  /** dummy genomes */
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
  /** context object mimicking internal Neat state */
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
};

describe('Telemetry objective events & lineage', () => {
  test('buildTelemetryEntry consumes pending objective events', () => {
    // Arrange
    const ctx = makeCtx();
    ctx._pendingObjectiveAdds = ['a', 'b'];
    ctx._pendingObjectiveRemoves = ['c'];
    ctx._objectiveEvents = [];
    ctx._lineageEnabled = true;
    // Act
    const entry = buildTelemetryEntry.call(
      ctx,
      ctx.population[0],
    ) as TelemetryEntry;
    // Assert
    const objectiveEvents = Array.isArray(entry.objEvents)
      ? (entry.objEvents as unknown[])
      : [];
    expect(objectiveEvents.length).toBe(3);
  });
});
