import { SPECIES_HISTORY_JSONL_MAX_DEFAULT } from '../../species/history/species.history';
import {
  clearParetoArchive,
  clearTelemetry,
  clearTelemetryObjectives,
  exportSpeciesHistoryCSV,
  exportSpeciesHistoryJSONL,
  exportParetoFrontJSONL,
  exportTelemetryCSV,
  exportTelemetryJSONL,
  getDiversityStats,
  getLineageSnapshot,
  getMultiObjectiveMetrics,
  getNoveltyArchiveSize,
  getObjectiveEvents,
  getObjectiveKeys,
  getOperatorStats,
  getParetoArchive,
  getParetoFronts,
  getPerformanceStats,
  getSpeciesHistory,
  getSpeciesStats,
  getTelemetry,
  getObjectives,
  registerTelemetryObjective,
  resetNoveltyArchive,
  type NeatTelemetryFacadeHost,
} from './telemetry.facade';
import {
  clearParetoArchive as clearTelemetryFacadeArchive,
  exportParetoFrontJSONL as exportTelemetryFacadeArchiveJsonl,
  getMultiObjectiveMetrics as getTelemetryFacadeArchiveMetrics,
  getParetoArchive as getTelemetryFacadeArchiveEntries,
  getParetoFronts as getTelemetryFacadeArchiveFronts,
} from './archive/telemetry.facade.archive';
import {
  clearTelemetry as clearTelemetryFacadeBuffer,
  exportTelemetryCSV as exportTelemetryFacadeBufferCsv,
  exportTelemetryJSONL as exportTelemetryFacadeBufferJsonl,
  getTelemetry as getTelemetryFacadeBuffer,
} from './buffer/telemetry.facade.buffer';
import {
  getLineageSnapshot as getTelemetryFacadeLineageSnapshot,
  LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
} from './lineage/telemetry.facade.lineage';
import {
  getNoveltyArchiveSize as getTelemetryFacadeNoveltyArchiveSize,
  resetNoveltyArchive as resetTelemetryFacadeNoveltyArchive,
} from './novelty/telemetry.facade.novelty';
import {
  clearTelemetryObjectives as clearTelemetryFacadeObjectives,
  getObjectiveEvents as getTelemetryFacadeObjectiveEvents,
  getObjectiveKeys as getTelemetryFacadeObjectiveKeys,
  getObjectives as getTelemetryFacadeObjectives,
  registerTelemetryObjective as registerTelemetryFacadeObjective,
} from './objectives/telemetry.facade.objectives';
import { getOperatorStats as getTelemetryFacadeOperatorStats } from './operator-stats/telemetry.facade.operator-stats';
import {
  getDiversityStats as getTelemetryFacadeRuntimeDiversityStats,
  getPerformanceStats as getTelemetryFacadeRuntimePerformanceStats,
} from './runtime/telemetry.facade.runtime';
import {
  exportSpeciesHistoryCSV as exportTelemetryFacadeSpeciesHistoryCsv,
  exportSpeciesHistoryJSONL as exportTelemetryFacadeSpeciesHistoryJsonl,
  getSpeciesHistory as getTelemetryFacadeSpeciesHistory,
  getSpeciesStats as getTelemetryFacadeSpeciesStats,
} from './species/telemetry.facade.species';

jest.mock('./archive/telemetry.facade.archive', () => ({
  clearParetoArchive: jest.fn(),
  exportParetoFrontJSONL: jest.fn(),
  getMultiObjectiveMetrics: jest.fn(),
  getParetoArchive: jest.fn(),
  getParetoFronts: jest.fn(),
}));

jest.mock('./buffer/telemetry.facade.buffer', () => ({
  clearTelemetry: jest.fn(),
  exportTelemetryCSV: jest.fn(),
  exportTelemetryJSONL: jest.fn(),
  getTelemetry: jest.fn(),
}));

jest.mock('./lineage/telemetry.facade.lineage', () => {
  const actualModule = jest.requireActual('./lineage/telemetry.facade.lineage');

  return {
    ...actualModule,
    getLineageSnapshot: jest.fn(),
  };
});

jest.mock('./novelty/telemetry.facade.novelty', () => ({
  getNoveltyArchiveSize: jest.fn(),
  resetNoveltyArchive: jest.fn(),
}));

jest.mock('./objectives/telemetry.facade.objectives', () => ({
  clearTelemetryObjectives: jest.fn(),
  getObjectiveEvents: jest.fn(),
  getObjectiveKeys: jest.fn(),
  getObjectives: jest.fn(),
  registerTelemetryObjective: jest.fn(),
}));

jest.mock('./operator-stats/telemetry.facade.operator-stats', () => ({
  getOperatorStats: jest.fn(),
}));

jest.mock('./runtime/telemetry.facade.runtime', () => ({
  getDiversityStats: jest.fn(),
  getPerformanceStats: jest.fn(),
}));

jest.mock('./species/telemetry.facade.species', () => ({
  exportSpeciesHistoryCSV: jest.fn(),
  exportSpeciesHistoryJSONL: jest.fn(),
  getSpeciesHistory: jest.fn(),
  getSpeciesStats: jest.fn(),
}));

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function createTelemetryFacadeHost(): NeatTelemetryFacadeHost {
  return {
    population: [],
    _speciesHistory: [],
    _telemetry: [],
  } as unknown as NeatTelemetryFacadeHost;
}

describe('neat telemetry facade root chapter', () => {
  const mockClearTelemetryFacadeArchive = jest.mocked(
    clearTelemetryFacadeArchive,
  );
  const mockExportTelemetryFacadeArchiveJsonl = jest.mocked(
    exportTelemetryFacadeArchiveJsonl,
  );
  const mockGetTelemetryFacadeArchiveEntries = jest.mocked(
    getTelemetryFacadeArchiveEntries,
  );
  const mockGetTelemetryFacadeArchiveFronts = jest.mocked(
    getTelemetryFacadeArchiveFronts,
  );
  const mockGetTelemetryFacadeArchiveMetrics = jest.mocked(
    getTelemetryFacadeArchiveMetrics,
  );
  const mockClearTelemetryFacadeBuffer = jest.mocked(clearTelemetryFacadeBuffer);
  const mockExportTelemetryFacadeBufferCsv = jest.mocked(
    exportTelemetryFacadeBufferCsv,
  );
  const mockExportTelemetryFacadeBufferJsonl = jest.mocked(
    exportTelemetryFacadeBufferJsonl,
  );
  const mockGetTelemetryFacadeBuffer = jest.mocked(getTelemetryFacadeBuffer);
  const mockGetTelemetryFacadeLineageSnapshot = jest.mocked(
    getTelemetryFacadeLineageSnapshot,
  );
  const mockGetTelemetryFacadeNoveltyArchiveSize = jest.mocked(
    getTelemetryFacadeNoveltyArchiveSize,
  );
  const mockResetTelemetryFacadeNoveltyArchive = jest.mocked(
    resetTelemetryFacadeNoveltyArchive,
  );
  const mockClearTelemetryFacadeObjectives = jest.mocked(
    clearTelemetryFacadeObjectives,
  );
  const mockGetTelemetryFacadeObjectiveEvents = jest.mocked(
    getTelemetryFacadeObjectiveEvents,
  );
  const mockGetTelemetryFacadeObjectiveKeys = jest.mocked(
    getTelemetryFacadeObjectiveKeys,
  );
  const mockGetTelemetryFacadeObjectives = jest.mocked(
    getTelemetryFacadeObjectives,
  );
  const mockGetTelemetryFacadeOperatorStats = jest.mocked(
    getTelemetryFacadeOperatorStats,
  );
  const mockGetTelemetryFacadeRuntimeDiversityStats = jest.mocked(
    getTelemetryFacadeRuntimeDiversityStats,
  );
  const mockGetTelemetryFacadeRuntimePerformanceStats = jest.mocked(
    getTelemetryFacadeRuntimePerformanceStats,
  );
  const mockRegisterTelemetryFacadeObjective = jest.mocked(
    registerTelemetryFacadeObjective,
  );
  const mockExportTelemetryFacadeSpeciesHistoryCsv = jest.mocked(
    exportTelemetryFacadeSpeciesHistoryCsv,
  );
  const mockExportTelemetryFacadeSpeciesHistoryJsonl = jest.mocked(
    exportTelemetryFacadeSpeciesHistoryJsonl,
  );
  const mockGetTelemetryFacadeSpeciesHistory = jest.mocked(
    getTelemetryFacadeSpeciesHistory,
  );
  const mockGetTelemetryFacadeSpeciesStats = jest.mocked(
    getTelemetryFacadeSpeciesStats,
  );

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('exportTelemetryJSONL()', () => {
    describe('given telemetry entries are available', () => {
      it('delegates JSONL export to the telemetry buffer chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        mockExportTelemetryFacadeBufferJsonl.mockReturnValue('jsonl-payload');

        // Act
        const exportPayload = exportTelemetryJSONL(host);

        // Assert
        expect({
          calls: mockExportTelemetryFacadeBufferJsonl.mock.calls,
          exportPayload,
        }).toEqual({
          calls: [[host]],
          exportPayload: 'jsonl-payload',
        });
      });
    });
  });

  describe('getObjectiveKeys()', () => {
    describe('given objective keys are registered on the host', () => {
      it('returns the stable objective-key list from the objectives chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const objectiveKeys = ['score', 'novelty'];
        mockGetTelemetryFacadeObjectiveKeys.mockReturnValue(objectiveKeys);

        // Act
        const keys = getObjectiveKeys(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeObjectiveKeys.mock.calls,
          keys,
        }).toEqual({
          calls: [[host]],
          keys: objectiveKeys,
        });
      });
    });
  });

  describe('getTelemetry()', () => {
    describe('given telemetry entries are already buffered', () => {
      it('returns the in-memory buffer from the telemetry buffer chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const telemetryEntries = [
          { generation: 3, bestScore: 12 } as unknown as ReturnType<
            typeof getTelemetry
          >[number],
        ];
        mockGetTelemetryFacadeBuffer.mockReturnValue(telemetryEntries);

        // Act
        const telemetryWindow = getTelemetry(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeBuffer.mock.calls,
          telemetryWindow,
        }).toEqual({
          calls: [[host]],
          telemetryWindow: telemetryEntries,
        });
      });
    });
  });

  describe('exportTelemetryCSV()', () => {
    describe('given the caller omits a maximum entry window', () => {
      it('uses the root facade default window size', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        mockExportTelemetryFacadeBufferCsv.mockReturnValue('csv-payload');

        // Act
        const exportPayload = exportTelemetryCSV(host);

        // Assert
        expect({
          calls: mockExportTelemetryFacadeBufferCsv.mock.calls,
          exportPayload,
        }).toEqual({
          calls: [[host, 500]],
          exportPayload: 'csv-payload',
        });
      });
    });
  });

  describe('clearTelemetry()', () => {
    describe('given the caller wants a fresh observation window', () => {
      it('delegates clearing to the telemetry buffer chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();

        // Act
        clearTelemetry(host);

        // Assert
        expect(mockClearTelemetryFacadeBuffer.mock.calls).toEqual([[host]]);
      });
    });
  });

  describe('clearTelemetryObjectives()', () => {
    describe('given the caller wants to remove custom telemetry objectives', () => {
      it('delegates clearing to the objectives chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();

        // Act
        clearTelemetryObjectives(host);

        // Assert
        expect(mockClearTelemetryFacadeObjectives.mock.calls).toEqual([[host]]);
      });
    });
  });

  describe('getObjectives()', () => {
    describe('given compact objective summaries are available', () => {
      it('returns the summaries from the objectives chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const objectives = [{ key: 'score', direction: 'max' as const }];
        mockGetTelemetryFacadeObjectives.mockReturnValue(objectives);

        // Act
        const summaries = getObjectives(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeObjectives.mock.calls,
          summaries,
        }).toEqual({
          calls: [[host]],
          summaries: objectives,
        });
      });
    });
  });

  describe('registerTelemetryObjective()', () => {
    describe('given a custom objective accessor is provided', () => {
      it('delegates registration to the objectives chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const accessor = jest.fn(() => 7);

        // Act
        registerTelemetryObjective(host, 'novelty', 'max', accessor);

        // Assert
        expect(mockRegisterTelemetryFacadeObjective.mock.calls).toEqual([
          [host, 'novelty', 'max', accessor],
        ]);
      });
    });
  });

  describe('getObjectiveEvents()', () => {
    describe('given the host has recorded objective lifecycle events', () => {
      it('returns the events from the objectives chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const objectiveEvents = [{ gen: 12, type: 'add' as const, key: 'novelty' }];
        mockGetTelemetryFacadeObjectiveEvents.mockReturnValue(objectiveEvents);

        // Act
        const events = getObjectiveEvents(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeObjectiveEvents.mock.calls,
          events,
        }).toEqual({
          calls: [[host]],
          events: objectiveEvents,
        });
      });
    });
  });

  describe('getLineageSnapshot()', () => {
    describe('given the caller omits a lineage sample limit', () => {
      it('uses the lineage chapter default limit', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const lineageSnapshot = [{ id: 7, parents: [3, 4] }];
        mockGetTelemetryFacadeLineageSnapshot.mockReturnValue(lineageSnapshot);

        // Act
        const snapshot = getLineageSnapshot(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeLineageSnapshot.mock.calls,
          snapshot,
        }).toEqual({
          calls: [[host, LINEAGE_SNAPSHOT_DEFAULT_LIMIT]],
          snapshot: lineageSnapshot,
        });
      });
    });
  });

  describe('exportSpeciesHistoryCSV()', () => {
    describe('given the caller omits a maximum history window', () => {
      it('uses the facade default history size for CSV export', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        mockExportTelemetryFacadeSpeciesHistoryCsv.mockReturnValue('species-csv');

        // Act
        const exportPayload = exportSpeciesHistoryCSV(host);

        // Assert
        expect({
          calls: mockExportTelemetryFacadeSpeciesHistoryCsv.mock.calls,
          exportPayload,
        }).toEqual({
          calls: [[host, 200]],
          exportPayload: 'species-csv',
        });
      });
    });
  });

  describe('exportSpeciesHistoryJSONL()', () => {
    describe('given the caller omits a maximum history window', () => {
      it('uses the species-history JSONL default window', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        mockExportTelemetryFacadeSpeciesHistoryJsonl.mockReturnValue(
          'species-jsonl',
        );

        // Act
        const exportPayload = exportSpeciesHistoryJSONL(host);

        // Assert
        expect({
          calls: mockExportTelemetryFacadeSpeciesHistoryJsonl.mock.calls,
          exportPayload,
        }).toEqual({
          calls: [[host, SPECIES_HISTORY_JSONL_MAX_DEFAULT]],
          exportPayload: 'species-jsonl',
        });
      });
    });
  });

  describe('getSpeciesStats()', () => {
    describe('given live species summaries are available', () => {
      it('returns the species summaries from the species chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const speciesStats = [
          { id: 2, size: 6, bestScore: 9, lastImproved: 14 },
        ];
        mockGetTelemetryFacadeSpeciesStats.mockReturnValue(speciesStats);

        // Act
        const summaries = getSpeciesStats(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeSpeciesStats.mock.calls,
          summaries,
        }).toEqual({
          calls: [[host]],
          summaries: speciesStats,
        });
      });
    });
  });

  describe('getSpeciesHistory()', () => {
    describe('given the host stores species-history snapshots', () => {
      it('returns the history from the species chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const speciesHistory = [
          { generation: 8, species: [] } as unknown as ReturnType<
            typeof getSpeciesHistory
          >[number],
        ];
        mockGetTelemetryFacadeSpeciesHistory.mockReturnValue(speciesHistory);

        // Act
        const history = getSpeciesHistory(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeSpeciesHistory.mock.calls,
          history,
        }).toEqual({
          calls: [[host]],
          history: speciesHistory,
        });
      });
    });
  });

  describe('getNoveltyArchiveSize()', () => {
    describe('given the host has archived novelty descriptors', () => {
      it('returns the novelty archive size from the novelty chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        mockGetTelemetryFacadeNoveltyArchiveSize.mockReturnValue(8);

        // Act
        const archiveSize = getNoveltyArchiveSize(host);

        // Assert
        expect({
          archiveSize,
          calls: mockGetTelemetryFacadeNoveltyArchiveSize.mock.calls,
        }).toEqual({
          archiveSize: 8,
          calls: [[host]],
        });
      });
    });
  });

  describe('getOperatorStats()', () => {
    describe('given operator outcomes have been recorded', () => {
      it('returns the operator summaries from the operator-stats chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const operatorStats = [{ name: 'ADD_CONN', success: 3, attempts: 5 }];
        mockGetTelemetryFacadeOperatorStats.mockReturnValue(operatorStats);

        // Act
        const summaries = getOperatorStats(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeOperatorStats.mock.calls,
          summaries,
        }).toEqual({
          calls: [[host]],
          summaries: operatorStats,
        });
      });
    });
  });

  describe('getMultiObjectiveMetrics()', () => {
    describe('given compact Pareto metrics are available', () => {
      it('returns the metrics from the archive chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const multiObjectiveMetrics = [
          { rank: 0, crowding: 1.2, score: 14, nodes: 5, connections: 7 },
        ];
        mockGetTelemetryFacadeArchiveMetrics.mockReturnValue(
          multiObjectiveMetrics,
        );

        // Act
        const metrics = getMultiObjectiveMetrics(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeArchiveMetrics.mock.calls,
          metrics,
        }).toEqual({
          calls: [[host]],
          metrics: multiObjectiveMetrics,
        });
      });
    });
  });

  describe('getParetoFronts()', () => {
    describe('given the caller requests a bounded number of fronts', () => {
      it('delegates front reconstruction to the archive chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const paretoFronts = [[{ id: 1 }]] as unknown as ReturnType<
          typeof getParetoFronts
        >;
        mockGetTelemetryFacadeArchiveFronts.mockReturnValue(paretoFronts);

        // Act
        const fronts = getParetoFronts(host, 2);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeArchiveFronts.mock.calls,
          fronts,
        }).toEqual({
          calls: [[host, 2]],
          fronts: paretoFronts,
        });
      });
    });
  });

  describe('getParetoArchive()', () => {
    describe('given the caller requests a recent archive window', () => {
      it('returns the archived entries from the archive chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const paretoArchive = [
          { generation: 9, fronts: [] } as unknown as ReturnType<
            typeof getParetoArchive
          >[number],
        ];
        mockGetTelemetryFacadeArchiveEntries.mockReturnValue(paretoArchive);

        // Act
        const archive = getParetoArchive(host, 4);

        // Assert
        expect({
          archive,
          calls: mockGetTelemetryFacadeArchiveEntries.mock.calls,
        }).toEqual({
          archive: paretoArchive,
          calls: [[host, 4]],
        });
      });
    });
  });

  describe('exportParetoFrontJSONL()', () => {
    describe('given a recent archive window is requested', () => {
      it('delegates JSONL export to the archive chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        mockExportTelemetryFacadeArchiveJsonl.mockReturnValue('pareto-jsonl');

        // Act
        const exportPayload = exportParetoFrontJSONL(host, 6);

        // Assert
        expect({
          calls: mockExportTelemetryFacadeArchiveJsonl.mock.calls,
          exportPayload,
        }).toEqual({
          calls: [[host, 6]],
          exportPayload: 'pareto-jsonl',
        });
      });
    });
  });

  describe('getPerformanceStats()', () => {
    describe('given runtime timing metrics are available', () => {
      it('returns the timing snapshot from the runtime chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const performanceStats = {
          lastEvalMs: 4,
          lastEvolveMs: 9,
        } as ReturnType<typeof getPerformanceStats>;
        mockGetTelemetryFacadeRuntimePerformanceStats.mockReturnValue(
          performanceStats,
        );

        // Act
        const timing = getPerformanceStats(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeRuntimePerformanceStats.mock.calls,
          timing,
        }).toEqual({
          calls: [[host]],
          timing: performanceStats,
        });
      });
    });
  });

  describe('getDiversityStats()', () => {
    describe('given runtime diversity metrics are available', () => {
      it('returns the diversity snapshot from the runtime chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();
        const diversityStats = {
          structuralEntropy: 1.5,
          uniqueStructures: 3,
        } as unknown as ReturnType<typeof getDiversityStats>;
        mockGetTelemetryFacadeRuntimeDiversityStats.mockReturnValue(
          diversityStats,
        );

        // Act
        const diversity = getDiversityStats(host);

        // Assert
        expect({
          calls: mockGetTelemetryFacadeRuntimeDiversityStats.mock.calls,
          diversity,
        }).toEqual({
          calls: [[host]],
          diversity: diversityStats,
        });
      });
    });
  });

  describe('resetNoveltyArchive()', () => {
    describe('given the caller wants to restart novelty observation', () => {
      it('delegates resetting to the novelty chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();

        // Act
        resetNoveltyArchive(host);

        // Assert
        expect(mockResetTelemetryFacadeNoveltyArchive.mock.calls).toEqual([
          [host],
        ]);
      });
    });
  });

  describe('clearParetoArchive()', () => {
    describe('given the caller wants a fresh Pareto archive window', () => {
      it('delegates clearing to the archive chapter', () => {
        // Arrange
        const host = createTelemetryFacadeHost();

        // Act
        clearParetoArchive(host);

        // Assert
        expect(mockClearTelemetryFacadeArchive.mock.calls).toEqual([[host]]);
      });
    });
  });
});