import type { DiversityStats } from '../../diversity/core/diversity.types';
import type { TelemetryEntry } from '../../shared/neat.shared.types';
import {
  LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
  buildLineageSnapshot,
  clearTelemetryBuffer,
  getCachedDiversityStats,
  getObjectiveEventsSnapshot,
  getPerformanceStatsSnapshot,
  getTelemetryBuffer,
  type TelemetryAccessorHost,
} from './telemetry.accessors';

type ObjectiveEvent = NonNullable<TelemetryAccessorHost['_objectiveEvents']>[number];

function createTelemetryHost(
  overrides: TelemetryAccessorHost = {},
): TelemetryAccessorHost {
  return { ...overrides };
}

function createTelemetryEntry(entryId: number): TelemetryEntry {
  return {
    gen: entryId,
    best: entryId,
    species: 1,
    hyper: 0,
    ops: [],
    objImportance: {},
  };
}

function createObjectiveEvent(
  generation: number,
  type: 'add' | 'remove',
  key: string,
): ObjectiveEvent {
  return { gen: generation, type, key };
}

function createDiversityStats(): DiversityStats {
  return {
    lineageMeanDepth: 1,
    lineageMeanPairDist: 0.25,
    meanNodes: 8,
    meanConns: 12,
    nodeVar: 0.5,
    connVar: 0.75,
    meanCompat: 1.5,
    graphletEntropy: 0.6,
    population: 10,
  };
}

describe('neat telemetry accessors chapter', () => {
  describe('getTelemetryBuffer', () => {
    describe('given the host already has a telemetry buffer', () => {
      it('returns the existing buffer reference', () => {
        // Arrange
        const telemetryEntries = [createTelemetryEntry(1)];
        const telemetryHost = createTelemetryHost({
          _telemetry: telemetryEntries,
        });

        // Act
        const telemetryBuffer = getTelemetryBuffer(telemetryHost);

        // Assert
        expect(telemetryBuffer).toBe(telemetryEntries);
      });
    });

    describe('given the host has not initialized telemetry yet', () => {
      it('returns an empty buffer snapshot', () => {
        // Arrange
        const telemetryHost = createTelemetryHost();

        // Act
        const telemetryBuffer = getTelemetryBuffer(telemetryHost);

        // Assert
        expect(telemetryBuffer).toEqual([]);
      });
    });
  });

  describe('clearTelemetryBuffer', () => {
    describe('given the host currently stores telemetry entries', () => {
      it('replaces the host buffer with an empty array', () => {
        // Arrange
        const telemetryHost = createTelemetryHost({
          _telemetry: [createTelemetryEntry(1)],
        });

        // Act
        clearTelemetryBuffer(telemetryHost);

        // Assert
        expect(telemetryHost._telemetry).toEqual([]);
      });
    });
  });

  describe('getObjectiveEventsSnapshot', () => {
    describe('given the host has not recorded any objective events', () => {
      it('returns an empty event snapshot', () => {
        // Arrange
        const telemetryHost = createTelemetryHost();

        // Act
        const objectiveEvents = getObjectiveEventsSnapshot(telemetryHost);

        // Assert
        expect(objectiveEvents).toEqual([]);
      });
    });

    describe('given the host stores objective events', () => {
      it('returns a shallow copy that callers can mutate safely', () => {
        // Arrange
        const recordedEvents = [createObjectiveEvent(2, 'add', 'entropy')];
        const telemetryHost = createTelemetryHost({
          _objectiveEvents: recordedEvents,
        });

        // Act
        const objectiveEvents = getObjectiveEventsSnapshot(telemetryHost);
        objectiveEvents.push(createObjectiveEvent(3, 'remove', 'entropy'));

        // Assert
        expect(telemetryHost._objectiveEvents).toEqual(recordedEvents);
      });
    });
  });

  describe('buildLineageSnapshot', () => {
    describe('given lineage metadata is partial and the caller provides a limit', () => {
      it('returns compact lineage entries with id and parents fallbacks', () => {
        // Arrange
        const parentIds = [10, 11];
        const population = [{ _id: 7, _parents: parentIds }, {}];

        // Act
        const lineageSnapshot = buildLineageSnapshot(population, 2);
        parentIds.push(12);

        // Assert
        expect(lineageSnapshot).toEqual([
          { id: 7, parents: [10, 11] },
          { id: -1, parents: [] },
        ]);
      });
    });

    describe('given the caller omits the limit', () => {
      it('uses the default lineage snapshot cap', () => {
        // Arrange
        const population = Array.from(
          { length: LINEAGE_SNAPSHOT_DEFAULT_LIMIT + 5 },
          (_unusedEntry, genomeIndex) => ({ _id: genomeIndex }),
        );

        // Act
        const lineageSnapshot = buildLineageSnapshot(population);

        // Assert
        expect(lineageSnapshot).toHaveLength(LINEAGE_SNAPSHOT_DEFAULT_LIMIT);
      });
    });
  });

  describe('getCachedDiversityStats', () => {
    describe('given the host already has cached diversity metrics', () => {
      it('returns the cached diversity snapshot', () => {
        // Arrange
        const diversityStats = createDiversityStats();
        const telemetryHost = createTelemetryHost({
          _diversityStats: diversityStats,
        });

        // Act
        const cachedDiversityStats = getCachedDiversityStats(telemetryHost);

        // Assert
        expect(cachedDiversityStats).toBe(diversityStats);
      });
    });
  });

  describe('getPerformanceStatsSnapshot', () => {
    describe('given the host stores the latest evaluation and evolution timings', () => {
      it('returns the coarse performance timing snapshot', () => {
        // Arrange
        const telemetryHost = createTelemetryHost({
          _lastEvalDuration: 12,
          _lastEvolveDuration: 34,
        });

        // Act
        const performanceStats = getPerformanceStatsSnapshot(telemetryHost);

        // Assert
        expect(performanceStats).toEqual({
          lastEvalMs: 12,
          lastEvolveMs: 34,
        });
      });
    });
  });
});