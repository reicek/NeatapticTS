import { getSpeciesHistory, getSpeciesStats } from './species';
import { getSpeciesHistory as getSpeciesHistoryImpl } from './history/read/species.history.read';
import { getSpeciesStats as getSpeciesStatsImpl } from './stats/species.stats';

jest.mock('./history/read/species.history.read', () => ({
  getSpeciesHistory: jest.fn(),
}));

jest.mock('./stats/species.stats', () => ({
  getSpeciesStats: jest.fn(),
}));

const mockedGetSpeciesHistoryImpl = jest.mocked(getSpeciesHistoryImpl);
const mockedGetSpeciesStatsImpl = jest.mocked(getSpeciesStatsImpl);

describe('neat species root chapter', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('getSpeciesStats', () => {
    describe('given a reporting host asks for the current species roster', () => {
      it('delegates to the stats chapter with the same host and returns its snapshot', () => {
        // Arrange
        const speciesHost = { _species: [{ id: 4 }] };
        const speciesStats = [
          {
            id: 4,
            size: 1,
            bestScore: 9,
            lastImproved: 3,
          },
        ];
        mockedGetSpeciesStatsImpl.mockReturnValue(speciesStats);

        // Act
        const returnedSpeciesStats = getSpeciesStats.call(speciesHost as never);

        // Assert
        expect({
          helperHost: mockedGetSpeciesStatsImpl.mock.calls[0]?.[0],
          returnedSpeciesStats,
        }).toEqual({
          helperHost: speciesHost,
          returnedSpeciesStats: speciesStats,
        });
      });
    });
  });

  describe('getSpeciesHistory', () => {
    describe('given a reporting host asks for the cross-generation species story', () => {
      it('delegates to the history-read chapter with the same host and returns its snapshots', () => {
        // Arrange
        const speciesHost = { _speciesHistory: [{ generation: 2, stats: [] }] };
        const speciesHistory = [{ generation: 2, stats: [] }];
        mockedGetSpeciesHistoryImpl.mockReturnValue(speciesHistory);

        // Act
        const returnedSpeciesHistory = getSpeciesHistory.call(speciesHost as never);

        // Assert
        expect({
          helperHost: mockedGetSpeciesHistoryImpl.mock.calls[0]?.[0],
          returnedSpeciesHistory,
        }).toEqual({
          helperHost: speciesHost,
          returnedSpeciesHistory: speciesHistory,
        });
      });
    });
  });
});