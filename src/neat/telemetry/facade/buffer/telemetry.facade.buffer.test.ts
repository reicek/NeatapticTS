import Network from '../../../../architecture/network';
import Neat from '../../../../neat';

describe('neat telemetry facade buffer chapter', () => {
  describe('getTelemetry', () => {
    describe('given a telemetry-enabled controller after one evaluation and evolution pass', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let latestTelemetryEntry:
        | ReturnType<Neat['getTelemetry']>[number]
        | undefined;

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
        await neat.evolve();

        // Act
        latestTelemetryEntry = neat.getTelemetry().at(-1);
      });

      describe('when the latest recorded generation is inspected', () => {
        it('exposes the recorded diversity block', () => {
          // Assert
          expect(latestTelemetryEntry?.diversity).toEqual(
            expect.objectContaining({
              meanCompat: expect.any(Number),
              graphletEntropy: expect.any(Number),
            }),
          );
        });
      });
    });

    describe('given a multi-objective controller after several evolution passes', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let latestTelemetryEntry:
        | ReturnType<Neat['getTelemetry']>[number]
        | undefined;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 2, scoreByConnectionCount, {
          popsize: 30,
          multiObjective: { enabled: true, complexityMetric: 'connections' },
          telemetry: { enabled: true },
        });

        for (let generationIndex = 0; generationIndex < 3; generationIndex++) {
          await neat.evolve();
        }

        // Act
        latestTelemetryEntry = neat.getTelemetry().at(-1);
      });

      describe('when the latest recorded generation is inspected', () => {
        it('exposes the recorded hyper proxy value', () => {
          // Assert
          expect(latestTelemetryEntry?.hyper).toEqual(expect.any(Number));
        });
      });
    });

    describe('given a multi-objective telemetry-enabled controller after two evolution passes', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let latestTelemetryEntry:
        | ReturnType<Neat['getTelemetry']>[number]
        | undefined;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 2, scoreByConnectionCount, {
          popsize: 18,
          seed: 501,
          multiObjective: { enabled: true },
          telemetry: { enabled: true },
        });

        for (let generationIndex = 0; generationIndex < 2; generationIndex++) {
          await neat.evolve();
        }

        // Act
        latestTelemetryEntry = neat.getTelemetry().at(-1);
      });

      describe('when the latest recorded generation is inspected', () => {
        it('exposes the recorded operator statistics array', () => {
          // Assert
          expect(latestTelemetryEntry?.ops).toEqual(expect.any(Array));
        });
      });
    });
    describe('given multi-objective speciation telemetry is recorded across several generations', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let latestTelemetryEntry:
        | ReturnType<Neat['getTelemetry']>[number]
        | undefined;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 2, scoreByConnectionCount, {
          popsize: 25,
          seed: 9,
          telemetry: { enabled: true },
          multiObjective: { enabled: true },
          speciation: true,
        });

        for (let generationIndex = 0; generationIndex < 5; generationIndex++) {
          await neat.evaluate();
          await neat.evolve();
        }

        // Act
        latestTelemetryEntry = neat.getTelemetry().at(-1);
      });

      describe('when the latest recorded generation is inspected', () => {
        it('exposes the recorded species allocation array', () => {
          // Assert
          expect(Array.isArray(latestTelemetryEntry?.speciesAlloc)).toBe(true);
        });
      });
    });

    describe('given telemetry performance and complexity blocks are enabled', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let latestTelemetryEntry:
        | ReturnType<Neat['getTelemetry']>[number]
        | undefined;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 2, scoreByConnectionCount, {
          popsize: 14,
          seed: 508,
          telemetry: { enabled: true, performance: true, complexity: true },
        });

        await neat.evolve();
        await neat.evolve();

        // Act
        latestTelemetryEntry = neat.getTelemetry().at(-1);
      });

      describe('when the latest recorded generation is inspected', () => {
        it('exposes the recorded performance block', () => {
          // Assert
          expect(latestTelemetryEntry?.perf).toEqual(
            expect.objectContaining({
              evolveMs: expect.any(Number),
            }),
          );
        });

        it('exposes the recorded complexity metrics block', () => {
          // Assert
          expect(latestTelemetryEntry?.complexity).toEqual(
            expect.objectContaining({
              meanNodes: expect.any(Number),
            }),
          );
        });

        it('tracks complexity growth deltas on the recorded block', () => {
          // Assert
          expect(latestTelemetryEntry?.complexity).toEqual(
            expect.objectContaining({
              growthNodes: expect.any(Number),
            }),
          );
        });
      });
    });

    describe('given hypervolume telemetry is enabled in multi-objective mode', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let latestTelemetryEntry:
        | ReturnType<Neat['getTelemetry']>[number]
        | undefined;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 2, scoreByConnectionCount, {
          popsize: 16,
          seed: 510,
          multiObjective: { enabled: true },
          telemetry: { enabled: true, hypervolume: true },
        });

        await neat.evolve();

        // Act
        latestTelemetryEntry = neat.getTelemetry().at(-1);
      });

      describe('when the latest recorded generation is inspected', () => {
        it('exposes the rounded hypervolume field', () => {
          // Assert
          expect(latestTelemetryEntry?.hv).toEqual(expect.any(Number));
        });
      });
    });
  });

  describe('exportTelemetryCSV', () => {
    describe('given telemetry is recorded before export', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let headerRow = '';

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 2, scoreByConnectionCount, {
          popsize: 12,
          seed: 511,
          telemetry: {
            enabled: true,
            performance: true,
            complexity: true,
          },
        });

        await neat.evolve();

        // Act
        headerRow = neat.exportTelemetryCSV().split('\n')[0];
      });

      describe('when the CSV payload is inspected', () => {
        it('includes the generation header column', () => {
          // Assert
          expect(headerRow).toContain('gen');
        });
      });
    });
  });
});
