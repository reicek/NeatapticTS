import type {
  NeatLike,
  SpeciesHistoryEntry,
  SpeciesHistoryStat,
  TelemetryEntry,
} from '../../shared/neat.shared.types';
import {
  buildSpeciesHistoryStats,
  collectBaseKeys,
  collectDiversityLineageMetrics,
  collectGroupedMetricKeys,
  collectOptionalColumnPresence,
  collectSpeciesHistoryHeaders,
  ensureMinimalSpeciesSnapshot,
  ensureSpeciesHistoryArray,
  serializeSpeciesHistoryRow,
  type TelemetryHeaderCollectionState,
} from './telemetry.exports.utils';

type SpeciesHistoryHost = NeatLike & {
  _speciesHistory?: SpeciesHistoryEntry[];
  _species?: SpeciesHistoryStat[];
  generation?: number;
};

function createTelemetryEntry(
  overrides: Record<string, unknown> = {},
): TelemetryEntry {
  return {
    gen: 3,
    best: 7,
    species: 2,
    hyper: 0,
    ...overrides,
  } as TelemetryEntry;
}

function createHeaderState(): TelemetryHeaderCollectionState {
  return {
    baseKeys: new Set<string>(),
    complexityKeys: new Set<string>(),
    perfKeys: new Set<string>(),
    lineageKeys: new Set<string>(),
    diversityLineageKeys: new Set<string>(),
    includeOps: false,
    includeObjectives: false,
    includeObjAges: false,
    includeSpeciesAlloc: false,
    includeObjEvents: false,
    includeObjImportance: false,
  };
}

describe('neat telemetry exports utility chapter', () => {
  describe('collectBaseKeys', () => {
    describe('when the telemetry entry includes grouped blocks plus fronts and rng', () => {
      it('records only the base keys together with fronts and rng', () => {
        // Arrange
        const telemetryEntry = createTelemetryEntry({
          complexity: { meanNodes: 4 },
          perf: { evalMs: 12 },
          ops: [{ op: 'ADD_NODE', succ: 1, att: 1 }],
          fronts: [1, 2],
          rng: 7,
        });
        const headerState = createHeaderState();

        // Act
        collectBaseKeys(telemetryEntry, headerState, 'fronts');

        // Assert
        expect(Array.from(headerState.baseKeys).toSorted()).toEqual([
          'best',
          'fronts',
          'gen',
          'hyper',
          'rng',
          'species',
        ]);
      });
    });

    describe('when fronts is not an array and rng is absent', () => {
      it('leaves those optional keys out of the discovered base header set', () => {
        // Arrange
        const telemetryEntry = {
          gen: 3,
          best: 7,
          species: 2,
          hyper: 0,
          fronts: { current: 1 },
        } as unknown as TelemetryEntry;
        const headerState = createHeaderState();

        // Act
        collectBaseKeys(telemetryEntry, headerState, 'fronts');

        // Assert
        expect(Array.from(headerState.baseKeys).toSorted()).toEqual([
          'best',
          'gen',
          'hyper',
          'species',
        ]);
      });
    });
  });

  describe('collectGroupedMetricKeys', () => {
    describe('when grouped metric families are absent', () => {
      it('leaves each grouped header set empty', () => {
        // Arrange
        const telemetryEntry = createTelemetryEntry();
        const headerState = createHeaderState();

        // Act
        collectGroupedMetricKeys(telemetryEntry, headerState);

        // Assert
        expect({
          complexityKeys: Array.from(headerState.complexityKeys),
          lineageKeys: Array.from(headerState.lineageKeys),
          perfKeys: Array.from(headerState.perfKeys),
        }).toEqual({
          complexityKeys: [],
          lineageKeys: [],
          perfKeys: [],
        });
      });
    });

    describe('when grouped metric families are present', () => {
      it('records complexity, performance, and lineage keys under their own sets', () => {
        // Arrange
        const telemetryEntry = createTelemetryEntry({
          complexity: { meanNodes: 4, meanConns: 6 },
          perf: { evalMs: 12, evolveMs: 7 },
          lineage: { depthBest: 2, ancestorUniq: 0.6 },
        });
        const headerState = createHeaderState();

        // Act
        collectGroupedMetricKeys(telemetryEntry, headerState);

        // Assert
        expect({
          complexityKeys: Array.from(headerState.complexityKeys).toSorted(),
          lineageKeys: Array.from(headerState.lineageKeys).toSorted(),
          perfKeys: Array.from(headerState.perfKeys).toSorted(),
        }).toEqual({
          complexityKeys: ['meanConns', 'meanNodes'],
          lineageKeys: ['ancestorUniq', 'depthBest'],
          perfKeys: ['evalMs', 'evolveMs'],
        });
      });
    });
  });

  describe('collectDiversityLineageMetrics', () => {
    describe('when the telemetry entry has no diversity block', () => {
      it('leaves the lineage diversity header set empty', () => {
        // Arrange
        const telemetryEntry = createTelemetryEntry();
        const headerState = createHeaderState();

        // Act
        collectDiversityLineageMetrics(telemetryEntry, headerState);

        // Assert
        expect(Array.from(headerState.diversityLineageKeys)).toEqual([]);
      });
    });

    describe('when the diversity block omits the curated lineage fields', () => {
      it('keeps the lineage diversity header set empty', () => {
        // Arrange
        const telemetryEntry = createTelemetryEntry({
          diversity: {
            meanCompat: 0.5,
            meanEntropy: 0.2,
          },
        });
        const headerState = createHeaderState();

        // Act
        collectDiversityLineageMetrics(telemetryEntry, headerState);

        // Assert
        expect(Array.from(headerState.diversityLineageKeys)).toEqual([]);
      });
    });

    describe('when the diversity block includes both curated lineage fields', () => {
      it('records the lineage depth and pair-distance columns', () => {
        // Arrange
        const telemetryEntry = createTelemetryEntry({
          diversity: {
            lineageMeanDepth: 1.2,
            lineageMeanPairDist: 0.4,
          },
        });
        const headerState = createHeaderState();

        // Act
        collectDiversityLineageMetrics(telemetryEntry, headerState);

        // Assert
        expect(Array.from(headerState.diversityLineageKeys).toSorted()).toEqual([
          'lineageMeanDepth',
          'lineageMeanPairDist',
        ]);
      });
    });
  });

  describe('collectOptionalColumnPresence', () => {
    describe('when optional telemetry structures are absent or empty', () => {
      it('keeps every optional export flag disabled', () => {
        // Arrange
        const telemetryEntry = createTelemetryEntry({
          ops: [],
          objEvents: [],
        });
        const headerState = createHeaderState();

        // Act
        collectOptionalColumnPresence(telemetryEntry, headerState);

        // Assert
        expect({
          includeObjAges: headerState.includeObjAges,
          includeObjEvents: headerState.includeObjEvents,
          includeObjImportance: headerState.includeObjImportance,
          includeObjectives: headerState.includeObjectives,
          includeOps: headerState.includeOps,
          includeSpeciesAlloc: headerState.includeSpeciesAlloc,
        }).toEqual({
          includeObjAges: false,
          includeObjEvents: false,
          includeObjImportance: false,
          includeObjectives: false,
          includeOps: false,
          includeSpeciesAlloc: false,
        });
      });
    });

    describe('when optional telemetry structures are populated', () => {
      it('enables every matching optional export flag', () => {
        // Arrange
        const telemetryEntry = createTelemetryEntry({
          ops: [{ op: 'ADD_NODE', succ: 1, att: 1 }],
          objectives: ['fitness'],
          objAges: { fitness: 7 },
          speciesAlloc: [{ id: 1, alloc: 2 }],
          objEvents: [{ gen: 3, key: 'fitness', type: 'add' }],
          objImportance: { fitness: { range: 1, var: 0.25 } },
        });
        const headerState = createHeaderState();

        // Act
        collectOptionalColumnPresence(telemetryEntry, headerState);

        // Assert
        expect({
          includeObjAges: headerState.includeObjAges,
          includeObjEvents: headerState.includeObjEvents,
          includeObjImportance: headerState.includeObjImportance,
          includeObjectives: headerState.includeObjectives,
          includeOps: headerState.includeOps,
          includeSpeciesAlloc: headerState.includeSpeciesAlloc,
        }).toEqual({
          includeObjAges: true,
          includeObjEvents: true,
          includeObjImportance: true,
          includeObjectives: true,
          includeOps: true,
          includeSpeciesAlloc: true,
        });
      });
    });
  });

  describe('ensureSpeciesHistoryArray', () => {
    describe('when the host is missing its history shelf', () => {
      it('creates and returns an empty history array', () => {
        // Arrange
        const telemetryHost = {} as SpeciesHistoryHost;

        // Act
        const history = ensureSpeciesHistoryArray(telemetryHost);

        // Assert
        expect({
          historyLength: history.length,
          sameReference: history === telemetryHost._speciesHistory,
        }).toEqual({
          historyLength: 0,
          sameReference: true,
        });
      });
    });

    describe('when the host already has a history shelf', () => {
      it('preserves the existing array reference', () => {
        // Arrange
        const existingHistory: SpeciesHistoryEntry[] = [
          { generation: 2, stats: [] },
        ];
        const telemetryHost = {
          _speciesHistory: existingHistory,
        } as SpeciesHistoryHost;

        // Act
        const history = ensureSpeciesHistoryArray(telemetryHost);

        // Assert
        expect(history).toBe(existingHistory);
      });
    });
  });

  describe('ensureMinimalSpeciesSnapshot', () => {
    describe('when history already exists', () => {
      it('leaves the history shelf unchanged', () => {
        // Arrange
        const history: SpeciesHistoryEntry[] = [{ generation: 5, stats: [] }];
        const telemetryHost = {
          _species: [{ id: 4, size: 2 }],
        } as unknown as SpeciesHistoryHost;

        // Act
        ensureMinimalSpeciesSnapshot(telemetryHost, history, 9, -1, 0, 0, 0);

        // Assert
        expect(history).toEqual([{ generation: 5, stats: [] }]);
      });
    });

    describe('when live species are absent', () => {
      it('does not synthesize a snapshot', () => {
        // Arrange
        const history: SpeciesHistoryEntry[] = [];
        const telemetryHost = {} as SpeciesHistoryHost;

        // Act
        ensureMinimalSpeciesSnapshot(telemetryHost, history, 9, -1, 0, 0, 0);

        // Assert
        expect(history).toEqual([]);
      });
    });

    describe('when the live species list is empty', () => {
      it('does not synthesize a snapshot', () => {
        // Arrange
        const history: SpeciesHistoryEntry[] = [];
        const telemetryHost = {
          _species: [],
        } as SpeciesHistoryHost;

        // Act
        ensureMinimalSpeciesSnapshot(telemetryHost, history, 9, -1, 0, 0, 0);

        // Assert
        expect(history).toEqual([]);
      });
    });

    describe('when live species exist but generation is missing', () => {
      it('synthesizes one snapshot using the fallback generation', () => {
        // Arrange
        const history: SpeciesHistoryEntry[] = [];
        const telemetryHost = {
          _species: [{ size: 2 }],
        } as unknown as SpeciesHistoryHost;

        // Act
        ensureMinimalSpeciesSnapshot(telemetryHost, history, 9, -1, 0, 0, 0);

        // Assert
        expect(history).toEqual([
          {
            generation: 9,
            stats: [{ id: -1, size: 2, bestScore: 0, lastImproved: 0 }],
          },
        ]);
      });
    });

    describe('when live species exist and generation is present', () => {
      it('synthesizes one snapshot using the live generation value', () => {
        // Arrange
        const history: SpeciesHistoryEntry[] = [];
        const telemetryHost = {
          generation: 12,
          _species: [
            {
              id: 4,
              members: [{}, {}],
              bestScore: 8,
              lastImproved: 11,
            },
          ],
        } as unknown as SpeciesHistoryHost;

        // Act
        ensureMinimalSpeciesSnapshot(telemetryHost, history, 9, -1, 0, 0, 0);

        // Assert
        expect(history).toEqual([
          {
            generation: 12,
            stats: [{ id: 4, size: 2, bestScore: 8, lastImproved: 11 }],
          },
        ]);
      });
    });
  });

  describe('collectSpeciesHistoryHeaders', () => {
    describe('when different history rows introduce different stat fields', () => {
      it('returns the generation column followed by the union of discovered stat keys', () => {
        // Arrange
        const history: SpeciesHistoryEntry[] = [
          {
            generation: 3,
            stats: [{ id: 1, size: 2 } as SpeciesHistoryStat],
          },
          {
            generation: 4,
            stats: [
              {
                id: 2,
                trend: 'up',
              } as unknown as SpeciesHistoryStat,
            ],
          },
        ];

        // Act
        const headers = collectSpeciesHistoryHeaders(history, 'generation');

        // Assert
        expect(headers).toEqual(['generation', 'id', 'size', 'trend']);
      });
    });
  });

  describe('buildSpeciesHistoryStats', () => {
    describe('when legacy and incomplete species records are mixed together', () => {
      it('normalizes ids, sizes, scores, and last-improved values with explicit fallbacks', () => {
        // Arrange
        const speciesRecords = [
          {
            id: 11,
            members: [{}, {}, {}],
            bestScore: 9,
            lastImproved: 4,
          },
          {
            size: 5,
            best: 7,
          },
          {
            id: 'bad',
            size: 'bad',
            bestScore: 'bad',
            best: 'bad',
            lastImproved: 'bad',
          },
        ] as unknown as SpeciesHistoryStat[];

        // Act
        const normalizedStats = buildSpeciesHistoryStats(
          speciesRecords,
          -1,
          0,
          0,
          0,
        );

        // Assert
        expect(normalizedStats).toEqual([
          { id: 11, size: 3, bestScore: 9, lastImproved: 4 },
          { id: -1, size: 5, bestScore: 7, lastImproved: 0 },
          { id: -1, size: 0, bestScore: 0, lastImproved: 0 },
        ]);
      });
    });
  });

  describe('serializeSpeciesHistoryRow', () => {
    describe('when a requested stat field is missing from the species snapshot', () => {
      it('serializes generation and known fields while leaving the missing cell empty', () => {
        // Arrange
        const historyEntry = {
          generation: 8,
          stats: [],
        } as SpeciesHistoryEntry;
        const speciesStat = { id: 5 } as SpeciesHistoryStat;

        // Act
        const csvRow = serializeSpeciesHistoryRow(
          historyEntry,
          speciesStat,
          ['generation', 'id', 'notes'],
          'generation',
        );

        // Assert
        expect(csvRow).toBe('8,5,');
      });
    });
  });
});