import Network from '../../../../architecture/network';
import Neat from '../../../../neat';
import { buildEmptyDiversityStats } from '../../../diversity/diversity';
import { getDiversityStats } from './telemetry.facade.runtime';
import * as telemetryAccessors from '../../accessors/telemetry.accessors';
import type { DiversityStats } from '../../../diversity/diversity';

function createDiversityStatsHost(input: {
  populationSize: number;
  diversityStats?: DiversityStats;
  computedStats: DiversityStats;
}) {
  const computeDiversityStats = jest.fn(() => input.computedStats);

  return {
    population: { length: input.populationSize },
    _diversityStats: input.diversityStats,
    _computeDiversityStats: computeDiversityStats,
    computeDiversityStats,
  };
}

describe('neat telemetry facade runtime chapter', () => {
  describe('getDiversityStats', () => {
    describe('given no cached diversity snapshot exists yet', () => {
      it('computes the diversity metrics on demand', () => {
        // Arrange
        const computedStats = buildEmptyDiversityStats(4);
        const runtimeHost = createDiversityStatsHost({
          populationSize: 4,
          computedStats,
        });

        // Act
        const diversityStats = getDiversityStats(runtimeHost);

        // Assert
        expect({
          diversityStats,
          computeCalls: runtimeHost.computeDiversityStats.mock.calls.length,
        }).toEqual({
          diversityStats: computedStats,
          computeCalls: 1,
        });
      });
    });

    describe('given a cached diversity flag exists but the shared accessor returns nothing', () => {
      it('falls back to an empty snapshot sized to the current population', () => {
        // Arrange
        const cachedStats = {
          ...buildEmptyDiversityStats(7),
          meanCompat: 1.5,
        };
        const runtimeHost = createDiversityStatsHost({
          populationSize: 7,
          diversityStats: cachedStats,
          computedStats: buildEmptyDiversityStats(99),
        });
        const cachedDiversitySpy = jest
          .spyOn(telemetryAccessors, 'getCachedDiversityStats')
          .mockReturnValue(undefined);

        try {
          // Act
          const diversityStats = getDiversityStats(runtimeHost);

          // Assert
          expect({
            diversityStats,
            computeCalls: runtimeHost.computeDiversityStats.mock.calls.length,
          }).toEqual({
            diversityStats: buildEmptyDiversityStats(7),
            computeCalls: 0,
          });
        } finally {
          cachedDiversitySpy.mockRestore();
        }
      });
    });

    describe('given a telemetry-enabled controller after one evaluation and evolution pass', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let diversityStats: ReturnType<Neat['getDiversityStats']>;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(4, 2, scoreByConnectionCount, {
          popsize: 25,
          seed: 77,
          speciation: true,
          telemetry: { enabled: true },
          diversityMetrics: {
            enabled: true,
            pairSample: 30,
            graphletSample: 40,
          },
        });

        await neat.evaluate();

        // Act
        await neat.evolve();
        diversityStats = neat.getDiversityStats();
      });

      describe('when the cached diversity snapshot is inspected', () => {
        it('returns sampled compatibility and graphlet metrics', () => {
          // Assert
          expect(diversityStats).toEqual(
            expect.objectContaining({
              meanCompat: expect.any(Number),
              graphletEntropy: expect.any(Number),
            }),
          );
        });
      });
    });
  });

  describe('getPerformanceStats', () => {
    describe('given telemetry performance tracking is enabled across evaluation and evolution', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let performanceStats: ReturnType<Neat['getPerformanceStats']>;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 1, scoreByConnectionCount, {
          popsize: 10,
          seed: 504,
          telemetry: { enabled: true, performance: true },
        });

        await neat.evaluate();
        await neat.evolve();

        // Act
        performanceStats = neat.getPerformanceStats();
      });

      describe('when the latest timing snapshot is inspected', () => {
        it('returns the latest recorded evolution duration', () => {
          // Assert
          expect(performanceStats).toEqual(
            expect.objectContaining({
              lastEvolveMs: expect.any(Number),
            }),
          );
        });
      });
    });
  });
});
