jest.mock('../../exports/telemetry.exports', () => ({
  exportSpeciesHistoryCSV: jest.fn(function (this: unknown, maxEntries: number) {
    return JSON.stringify({
      maxEntries,
      receivedHost: this,
    });
  }),
}));

jest.mock('../../../species/species', () => ({
  getSpeciesHistory: jest.fn(() => []),
  getSpeciesStats: jest.fn(function (this: unknown) {
    return [
      {
        bestScore: 8,
        id: 3,
        lastImproved: 2,
        size: 5,
      },
    ];
  }),
}));

jest.mock('../../../species/history/species.history', () => {
  const actualModule = jest.requireActual('../../../species/history/species.history');

  return {
    ...actualModule,
    exportSpeciesHistoryJsonl: jest.fn(
      (historyEntries: unknown[], maxEntries: number) =>
        JSON.stringify({
          historyEntries,
          maxEntries,
        }),
    ),
  };
});

import { exportSpeciesHistoryCSV as exportSpeciesHistoryCsvImpl } from '../../exports/telemetry.exports';
import {
  exportSpeciesHistoryJsonl,
  SPECIES_HISTORY_JSONL_MAX_DEFAULT,
} from '../../../species/history/species.history';
import { getSpeciesStats as getSpeciesStatsImpl } from '../../../species/species';
import {
  exportSpeciesHistoryCSV,
  exportSpeciesHistoryJSONL,
  getSpeciesStats,
  type TelemetryFacadeSpeciesHost,
} from './telemetry.facade.species';

const mockedExportSpeciesHistoryCsvImpl = jest.mocked(exportSpeciesHistoryCsvImpl);
const mockedExportSpeciesHistoryJsonl = jest.mocked(exportSpeciesHistoryJsonl);
const mockedGetSpeciesStatsImpl = jest.mocked(getSpeciesStatsImpl);

describe('neat telemetry facade species direct helpers', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('exportSpeciesHistoryCSV', () => {
    describe('given the caller omits the max-entry override', () => {
      it('delegates with the default csv history limit', () => {
        // Arrange
        const host: TelemetryFacadeSpeciesHost = {
          options: {},
        };

        // Act
        exportSpeciesHistoryCSV(host);

        // Assert
        expect({
          args: mockedExportSpeciesHistoryCsvImpl.mock.calls[0],
          context: mockedExportSpeciesHistoryCsvImpl.mock.contexts[0],
        }).toEqual({
          args: [200],
          context: host,
        });
      });
    });
  });

  describe('exportSpeciesHistoryJSONL', () => {
    describe('given the caller omits both history and max-entry overrides', () => {
      it('delegates with an empty history array and the jsonl default limit', () => {
        // Arrange
        const host: TelemetryFacadeSpeciesHost = {
          options: {},
        };

        // Act
        exportSpeciesHistoryJSONL(host);

        // Assert
        expect(mockedExportSpeciesHistoryJsonl.mock.calls[0]).toEqual([
          [],
          SPECIES_HISTORY_JSONL_MAX_DEFAULT,
        ]);
      });
    });
  });

  describe('getSpeciesStats', () => {
    describe('given the facade forwards a live species host to the species reader', () => {
      it('returns the delegated roster summary', () => {
        // Arrange
        const host: TelemetryFacadeSpeciesHost = {
          options: {},
        };

        // Act
        const speciesStats = getSpeciesStats(host);

        // Assert
        expect({
          context: mockedGetSpeciesStatsImpl.mock.contexts[0],
          speciesStats,
        }).toEqual({
          context: host,
          speciesStats: [
            {
              bestScore: 8,
              id: 3,
              lastImproved: 2,
              size: 5,
            },
          ],
        });
      });
    });
  });
});