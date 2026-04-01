import Network from '../../../../architecture/network';
import Neat from '../../../../neat';

describe('neat telemetry facade runtime chapter', () => {
  describe('getDiversityStats', () => {
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
