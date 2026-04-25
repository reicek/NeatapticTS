import Network from '../../../../architecture/network';
import Neat from '../../../../neat';
import { structuralEntropy } from '../../../diversity/diversity';
import {
  exportParetoFrontJSONL,
  getParetoArchive,
  getParetoFronts,
  type TelemetryFacadeArchiveHost,
} from './telemetry.facade.archive';

type ParetoArchiveJsonLine = {
  gen?: number;
  vectors?: {
    values?: number[];
  }[];
};

function scoreByNodeBalance(network: Network): number {
  return network.nodes.length - network.connections.length * 0.01;
}

function createDynamicMultiObjectiveNeat(): Neat {
  const neat = new Neat(3, 2, scoreByNodeBalance, {
    popsize: 30,
    seed: 42,
    speciation: false,
    multiObjective: {
      enabled: true,
      complexityMetric: 'nodes',
    },
  });

  neat.registerObjective('entropy', 'max', (genome) =>
    structuralEntropy(genome as Network),
  );

  return neat;
}

describe('neat telemetry facade archive chapter', () => {
  describe('getParetoArchive', () => {
    describe('given a multi-objective controller after several evolution passes', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let paretoArchiveLength = 0;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(4, 2, scoreByConnectionCount, {
          popsize: 20,
          seed: 503,
          multiObjective: { enabled: true },
        });

        for (let generationIndex = 0; generationIndex < 3; generationIndex++) {
          await neat.evolve();
        }

        // Act
        paretoArchiveLength = neat.getParetoArchive().length;
      });

      describe('when the recent archive slice is inspected', () => {
        it('returns retained Pareto archive entries', () => {
          // Assert
          expect(paretoArchiveLength).toBeGreaterThan(0);
        });
      });
    });
  });

  describe('exportParetoFrontJSONL', () => {
    describe('given a controller with archived Pareto objective vectors after one evolved generation', () => {
      let exportSummary = {
        generation: -1,
        hasVectors: false,
        firstVectorHasValues: false,
      };

      beforeAll(async () => {
        // Arrange
        const neat = createDynamicMultiObjectiveNeat();

        // Act
        await neat.evolve();

        const parsedArchiveLine = JSON.parse(
          neat.exportParetoFrontJSONL().split('\n').at(0) ?? '{}',
        ) as ParetoArchiveJsonLine;

        exportSummary = {
          generation: parsedArchiveLine.gen ?? -1,
          hasVectors:
            Array.isArray(parsedArchiveLine.vectors) &&
            parsedArchiveLine.vectors.length > 0,
          firstVectorHasValues: Array.isArray(
            parsedArchiveLine.vectors?.at(0)?.values,
          ),
        };
      });

      it('serializes the archived objective vectors as JSONL entries', () => {
        // Assert
        expect(exportSummary).toEqual({
          generation: 0,
          hasVectors: true,
          firstVectorHasValues: true,
        });
      });
    });
  });

  describe('live Pareto inspection', () => {
    describe('given a controller with a registered entropy objective after one evolved generation', () => {
      let paretoFrontCount = 0;
      let rankZeroMetricCount = 0;

      beforeAll(async () => {
        // Arrange
        const neat = createDynamicMultiObjectiveNeat();

        // Act
        await neat.evaluate();
        await neat.evolve();

        paretoFrontCount = neat.getParetoFronts(2).length;
        rankZeroMetricCount = neat
          .getMultiObjectiveMetrics()
          .filter((metric) => metric.rank === 0).length;
      });

      it('reconstructs at least one live Pareto front from the ranked population', () => {
        // Assert
        expect(paretoFrontCount).toBeGreaterThan(0);
      });

      it('reports at least one rank-zero genome in the compact metrics view', () => {
        // Assert
        expect(rankZeroMetricCount).toBeGreaterThan(0);
      });
    });
  });

  describe('clearParetoArchive', () => {
    describe('given a controller with archived Pareto entries after several evolution passes', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let clearedArchiveLength = -1;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(4, 2, scoreByConnectionCount, {
          popsize: 20,
          seed: 504,
          multiObjective: { enabled: true },
        });

        for (let generationIndex = 0; generationIndex < 3; generationIndex++) {
          await neat.evolve();
        }

        // Act
        neat.clearParetoArchive();
        clearedArchiveLength = neat.getParetoArchive().length;
      });

      it('empties the retained Pareto archive slice', () => {
        // Assert
        expect(clearedArchiveLength).toBe(0);
      });
    });
  });

  describe('archive helper defaults', () => {
    describe('given a minimal archive host', () => {
      describe('when the helper defaults are omitted directly', () => {
        it('uses the default archive and front window sizes', () => {
          // Arrange
          const archiveHost = {
            population: [],
            options: { multiObjective: { enabled: true } },
            _paretoArchive: [],
            _paretoObjectivesArchive: [],
          } as TelemetryFacadeArchiveHost;

          // Act
          const defaultFronts = getParetoFronts(archiveHost);
          const defaultArchive = getParetoArchive(archiveHost);
          const defaultJsonl = exportParetoFrontJSONL(archiveHost);

          // Assert
          expect({
            frontsLength: defaultFronts.length,
            archiveLength: defaultArchive.length,
            jsonlLength: defaultJsonl.length,
          }).toEqual({ frontsLength: 0, archiveLength: 0, jsonlLength: 0 });
        });
      });
    });
  });
});
