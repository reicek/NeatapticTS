import type Network from './architecture/network/network';
import Neat, {
  DEFAULT_COMPATIBILITY_THRESHOLD,
  DEFAULT_DISJOINT_COEFF,
  DEFAULT_DIVERSITY_GRAPHLET_SAMPLE,
  DEFAULT_DIVERSITY_PAIR_SAMPLE,
  DEFAULT_ELITISM,
  DEFAULT_EXCESS_COEFF,
  DEFAULT_MAX_CONNS,
  DEFAULT_MAX_GATES,
  DEFAULT_MAX_NODES,
  DEFAULT_MUTATION_AMOUNT,
  DEFAULT_MUTATION_RATE,
  DEFAULT_NOVELTY_K,
  DEFAULT_POPULATION_SIZE,
  DEFAULT_PROVENANCE,
  DEFAULT_WEIGHT_DIFF_COEFF,
} from './neat';
import {
  buildEmptyDiversityStats,
  type DiversityStats,
} from './neat/diversity/diversity';
import {
  DEFAULT_COMPATIBILITY_THRESHOLD as ROOT_DEFAULT_COMPATIBILITY_THRESHOLD,
  DEFAULT_DISJOINT_COEFF as ROOT_DEFAULT_DISJOINT_COEFF,
  DEFAULT_DIVERSITY_GRAPHLET_SAMPLE as ROOT_DEFAULT_DIVERSITY_GRAPHLET_SAMPLE,
  DEFAULT_DIVERSITY_PAIR_SAMPLE as ROOT_DEFAULT_DIVERSITY_PAIR_SAMPLE,
  DEFAULT_ELITISM as ROOT_DEFAULT_ELITISM,
  DEFAULT_EXCESS_COEFF as ROOT_DEFAULT_EXCESS_COEFF,
  DEFAULT_MAX_CONNS as ROOT_DEFAULT_MAX_CONNS,
  DEFAULT_MAX_GATES as ROOT_DEFAULT_MAX_GATES,
  DEFAULT_MAX_NODES as ROOT_DEFAULT_MAX_NODES,
  DEFAULT_MUTATION_AMOUNT as ROOT_DEFAULT_MUTATION_AMOUNT,
  DEFAULT_MUTATION_RATE as ROOT_DEFAULT_MUTATION_RATE,
  DEFAULT_NOVELTY_K as ROOT_DEFAULT_NOVELTY_K,
  DEFAULT_POPULATION_SIZE as ROOT_DEFAULT_POPULATION_SIZE,
  DEFAULT_PROVENANCE as ROOT_DEFAULT_PROVENANCE,
  DEFAULT_WEIGHT_DIFF_COEFF as ROOT_DEFAULT_WEIGHT_DIFF_COEFF,
} from './neat/neat.defaults.constants';
import { warnIfNoBestGenome } from './neat/evolve/warnings/evolve.warnings.utils';
import { DEFAULT_MAX_PARETO_FRONTS } from './neat/multiobjective/metrics/multiobjective.metrics';
import { LINEAGE_SNAPSHOT_DEFAULT_LIMIT } from './neat/telemetry/accessors/telemetry.accessors';
import { createPool, spawnFromParent } from './neat/helpers/neat.helpers';
import { initializeNeatConstructor } from './neat/init/neat.init';
import { selectMutationMethod } from './neat/mutation/mutation';
import * as neatPopulationSummaryFacade from './neat/selection/facade/selection.facade';
import * as neatPruningFacade from './neat/pruning/facade/pruning.facade';
import * as neatTelemetryFacade from './neat/telemetry/facade/telemetry.facade';

jest.mock('./neat/init/neat.init', () => ({
  initializeNeatConstructor: jest.fn(),
}));

jest.mock('./neat/helpers/neat.helpers', () => ({
  createPool: jest.fn(),
  spawnFromParent: jest.fn(),
  addGenome: jest.fn(),
}));

jest.mock('./neat/mutation/mutation', () => ({
  selectMutationMethod: jest.fn(),
  mutate: jest.fn(),
  mutateAddNodeReuse: jest.fn(),
  mutateAddConnReuse: jest.fn(),
}));

jest.mock('./neat/evolve/warnings/evolve.warnings.utils', () => ({
  warnIfNoBestGenome: jest.fn(),
}));

jest.mock('./neat/pruning/facade/pruning.facade', () => ({
  applyEvolutionPruning: jest.fn(),
  applyAdaptivePruning: jest.fn(),
}));

jest.mock('./neat/selection/facade/selection.facade', () => ({
  sort: jest.fn(),
  getFittest: jest.fn(),
  getAverage: jest.fn(),
}));

jest.mock('./neat/telemetry/facade/telemetry.facade', () => ({
  getObjectiveKeys: jest.fn(),
  getTelemetry: jest.fn(),
  exportTelemetryJSONL: jest.fn(),
  exportTelemetryCSV: jest.fn(),
  clearTelemetry: jest.fn(),
  getObjectives: jest.fn(),
  registerTelemetryObjective: jest.fn(),
  clearTelemetryObjectives: jest.fn(),
  getObjectiveEvents: jest.fn(),
  getLineageSnapshot: jest.fn(),
  exportSpeciesHistoryCSV: jest.fn(),
  exportSpeciesHistoryJSONL: jest.fn(),
  getSpeciesStats: jest.fn(),
  getSpeciesHistory: jest.fn(),
  getNoveltyArchiveSize: jest.fn(),
  getMultiObjectiveMetrics: jest.fn(),
  getOperatorStats: jest.fn(),
  getParetoFronts: jest.fn(),
  getParetoArchive: jest.fn(),
  exportParetoFrontJSONL: jest.fn(),
  getPerformanceStats: jest.fn(),
  getDiversityStats: jest.fn(),
  resetNoveltyArchive: jest.fn(),
  clearParetoArchive: jest.fn(),
}));

type MockedNeatHelpersModule = {
  createPool: unknown;
  spawnFromParent: unknown;
  addGenome: unknown;
};

type NeatDiversityInternals = {
  population: Network[];
  _diversityStats?: DiversityStats;
  _computeDiversityStats: () => DiversityStats;
};

function createNeatRootHost(): Neat {
  return Object.create(Neat.prototype) as Neat;
}

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('neat root coverage chapter', () => {
  const mockInitializeNeatConstructor = jest.mocked(initializeNeatConstructor);
  const mockCreatePool = jest.mocked(createPool);
  const mockSpawnFromParent = jest.mocked(spawnFromParent);
  const mockSelectMutationMethod = jest.mocked(selectMutationMethod);
  const mockWarnIfNoBestGenome = jest.mocked(warnIfNoBestGenome);
  const mockApplyEvolutionPruning = jest.mocked(
    neatPruningFacade.applyEvolutionPruning,
  );
  const mockApplyAdaptivePruning = jest.mocked(
    neatPruningFacade.applyAdaptivePruning,
  );
  const mockGetFittest = jest.mocked(neatPopulationSummaryFacade.getFittest);
  const mockGetAverage = jest.mocked(neatPopulationSummaryFacade.getAverage);
  const mockExportTelemetryJsonl = jest.mocked(
    neatTelemetryFacade.exportTelemetryJSONL,
  );
  const mockClearTelemetry = jest.mocked(neatTelemetryFacade.clearTelemetry);
  const mockClearTelemetryObjectives = jest.mocked(
    neatTelemetryFacade.clearTelemetryObjectives,
  );
  const mockGetObjectiveEvents = jest.mocked(
    neatTelemetryFacade.getObjectiveEvents,
  );
  const mockGetLineageSnapshot = jest.mocked(
    neatTelemetryFacade.getLineageSnapshot,
  );
  const mockGetSpeciesStats = jest.mocked(neatTelemetryFacade.getSpeciesStats);
  const mockGetNoveltyArchiveSize = jest.mocked(
    neatTelemetryFacade.getNoveltyArchiveSize,
  );
  const mockGetOperatorStats = jest.mocked(
    neatTelemetryFacade.getOperatorStats,
  );
  const mockGetParetoFronts = jest.mocked(neatTelemetryFacade.getParetoFronts);
  const mockResetNoveltyArchive = jest.mocked(
    neatTelemetryFacade.resetNoveltyArchive,
  );

  beforeEach(() => {
    const helpersModule = jest.requireMock(
      './neat/helpers/neat.helpers',
    ) as MockedNeatHelpersModule;

    jest.clearAllMocks();
    helpersModule.createPool = mockCreatePool;
  });

  describe('constructor', () => {
    describe('given every constructor argument is omitted', () => {
      it('normalizes the root defaults before handing control to initialization', () => {
        // Arrange
        mockInitializeNeatConstructor.mockImplementation(() => undefined);

        // Act
        const neat = new Neat();
        const [, initializationRequest] =
          mockInitializeNeatConstructor.mock.calls[0] ?? [];

        // Assert
        expect({
          input: neat.input,
          output: neat.output,
          defaultFitnessValue: neat.fitness(undefined as never),
          fitnessType: typeof neat.fitness,
          rawOptions: initializationRequest?.rawOptions,
          optionBagMatches: initializationRequest?.optionBag === neat.options,
        }).toEqual({
          input: 0,
          output: 0,
          defaultFitnessValue: 0,
          fitnessType: 'function',
          rawOptions: {},
          optionBagMatches: true,
        });
      });
    });

    describe('given the caller passes a null option bag at runtime', () => {
      it('falls back to an empty option object before initialization runs', () => {
        // Arrange
        mockInitializeNeatConstructor.mockImplementation(() => undefined);

        // Act
        const neat = new Neat(
          undefined,
          undefined,
          undefined,
          null as unknown as ConstructorParameters<typeof Neat>[3],
        );
        const [, initializationRequest] =
          mockInitializeNeatConstructor.mock.calls[0] ?? [];

        // Assert
        expect({
          rawOptions: initializationRequest?.rawOptions,
          options: neat.options,
          optionBagMatches: initializationRequest?.optionBag === neat.options,
        }).toEqual({
          rawOptions: null,
          options: {},
          optionBagMatches: true,
        });
      });
    });
  });

  describe('population bootstrap bridge', () => {
    describe('given the caller reads the root default constants through the public facade', () => {
      it('re-exports the same default values as the defaults chapter', () => {
        // Arrange
        const exportedDefaults = {
          DEFAULT_COMPATIBILITY_THRESHOLD,
          DEFAULT_DISJOINT_COEFF,
          DEFAULT_DIVERSITY_GRAPHLET_SAMPLE,
          DEFAULT_DIVERSITY_PAIR_SAMPLE,
          DEFAULT_ELITISM,
          DEFAULT_EXCESS_COEFF,
          DEFAULT_MAX_CONNS,
          DEFAULT_MAX_GATES,
          DEFAULT_MAX_NODES,
          DEFAULT_MUTATION_AMOUNT,
          DEFAULT_MUTATION_RATE,
          DEFAULT_NOVELTY_K,
          DEFAULT_POPULATION_SIZE,
          DEFAULT_PROVENANCE,
          DEFAULT_WEIGHT_DIFF_COEFF,
        };

        // Act
        const chapterDefaults = {
          DEFAULT_COMPATIBILITY_THRESHOLD: ROOT_DEFAULT_COMPATIBILITY_THRESHOLD,
          DEFAULT_DISJOINT_COEFF: ROOT_DEFAULT_DISJOINT_COEFF,
          DEFAULT_DIVERSITY_GRAPHLET_SAMPLE:
            ROOT_DEFAULT_DIVERSITY_GRAPHLET_SAMPLE,
          DEFAULT_DIVERSITY_PAIR_SAMPLE: ROOT_DEFAULT_DIVERSITY_PAIR_SAMPLE,
          DEFAULT_ELITISM: ROOT_DEFAULT_ELITISM,
          DEFAULT_EXCESS_COEFF: ROOT_DEFAULT_EXCESS_COEFF,
          DEFAULT_MAX_CONNS: ROOT_DEFAULT_MAX_CONNS,
          DEFAULT_MAX_GATES: ROOT_DEFAULT_MAX_GATES,
          DEFAULT_MAX_NODES: ROOT_DEFAULT_MAX_NODES,
          DEFAULT_MUTATION_AMOUNT: ROOT_DEFAULT_MUTATION_AMOUNT,
          DEFAULT_MUTATION_RATE: ROOT_DEFAULT_MUTATION_RATE,
          DEFAULT_NOVELTY_K: ROOT_DEFAULT_NOVELTY_K,
          DEFAULT_POPULATION_SIZE: ROOT_DEFAULT_POPULATION_SIZE,
          DEFAULT_PROVENANCE: ROOT_DEFAULT_PROVENANCE,
          DEFAULT_WEIGHT_DIFF_COEFF: ROOT_DEFAULT_WEIGHT_DIFF_COEFF,
        };

        // Assert
        expect(exportedDefaults).toEqual(chapterDefaults);
      });
    });

    describe('given the helper export remains callable', () => {
      it('delegates pool creation through the helper module', () => {
        // Arrange
        const neat = createNeatRootHost();
        const seedNetwork = { key: 'seed' } as unknown as Network;

        // Act
        neat.createPool(seedNetwork);

        // Assert
        expect({
          calls: mockCreatePool.mock.calls,
          contexts: mockCreatePool.mock.contexts,
        }).toEqual({
          calls: [[seedNetwork]],
          contexts: [neat],
        });
      });
    });

    describe('given the helper export is truthy but no longer callable', () => {
      it('skips pool creation without throwing at the root surface', () => {
        // Arrange
        const neat = createNeatRootHost();
        const helpersModule = jest.requireMock(
          './neat/helpers/neat.helpers',
        ) as MockedNeatHelpersModule;
        let didThrow = false;
        helpersModule.createPool = 'not-callable';

        // Act
        try {
          neat.createPool(null);
        } catch {
          didThrow = true;
        }

        // Assert
        expect({
          didThrow,
          helperCalls: mockCreatePool.mock.calls.length,
        }).toEqual({
          didThrow: false,
          helperCalls: 0,
        });
      });
    });

    describe('given the helper throws during pool creation', () => {
      it('swallows the helper failure to preserve the root bootstrap flow', () => {
        // Arrange
        const neat = createNeatRootHost();
        let didThrow = false;
        mockCreatePool.mockImplementation(() => {
          throw new Error('pool creation failed');
        });

        // Act
        try {
          neat.createPool(null);
        } catch {
          didThrow = true;
        }

        // Assert
        expect({
          didThrow,
          helperCalls: mockCreatePool.mock.calls.length,
        }).toEqual({
          didThrow: false,
          helperCalls: 1,
        });
      });
    });
  });

  describe('pruning bridges', () => {
    describe('given the controller exposes the missing-best-genome warning hook', () => {
      it('delegates the warning emission to the evolve warnings chapter', () => {
        // Arrange
        const neat = createNeatRootHost();

        // Act
        neat._warnIfNoBestGenome();

        // Assert
        expect(mockWarnIfNoBestGenome.mock.calls).toEqual([[]]);
      });
    });

    describe('given both pruning entry points are triggered from the root controller', () => {
      it('delegates the scheduled and adaptive pruning requests to the pruning facade', async () => {
        // Arrange
        const neat = createNeatRootHost();
        mockApplyEvolutionPruning.mockResolvedValue(undefined);
        mockApplyAdaptivePruning.mockResolvedValue(undefined);

        // Act
        await neat.applyEvolutionPruning();
        await neat.applyAdaptivePruning();

        // Assert
        expect({
          evolutionCalls: mockApplyEvolutionPruning.mock.calls,
          adaptiveCalls: mockApplyAdaptivePruning.mock.calls,
        }).toEqual({
          evolutionCalls: [[neat]],
          adaptiveCalls: [[neat]],
        });
      });
    });
  });

  describe('reproduction bridges', () => {
    describe('given the caller spawns from a parent without overriding mutate count', () => {
      it('forwards the default single-mutation pass through the helper module', () => {
        // Arrange
        const neat = createNeatRootHost();
        const parentGenome = { id: 1 } as unknown as Network;
        const childGenome = { id: 2 } as unknown as Network;
        mockSpawnFromParent.mockImplementation(() => childGenome as never);

        // Act
        const offspring = neat.spawnFromParent(parentGenome);

        // Assert
        expect({
          calls: mockSpawnFromParent.mock.calls,
          contexts: mockSpawnFromParent.mock.contexts,
          offspring,
        }).toEqual({
          calls: [[parentGenome, 1]],
          contexts: [neat],
          offspring: childGenome,
        });
      });
    });

    describe('given mutation-method selection succeeds with the default test return flag', () => {
      it('forwards the default raw-return toggle through the mutation chapter', async () => {
        // Arrange
        const neat = createNeatRootHost();
        const genome = { id: 9 } as unknown as Network;
        const selectionResult = {
          method: 'ADD_NODE',
        } as unknown as Awaited<ReturnType<Neat['selectMutationMethod']>>;
        mockSelectMutationMethod.mockResolvedValue(selectionResult);

        // Act
        const result = await neat.selectMutationMethod(genome);

        // Assert
        expect({
          calls: mockSelectMutationMethod.mock.calls,
          contexts: mockSelectMutationMethod.mock.contexts,
          result,
        }).toEqual({
          calls: [[genome, true]],
          contexts: [neat],
          result: selectionResult,
        });
      });
    });

    describe('given mutation-method selection throws inside the mutation chapter', () => {
      it('returns null instead of letting the root helper escape the error', async () => {
        // Arrange
        const neat = createNeatRootHost();
        const genome = { id: 10 } as unknown as Network;
        mockSelectMutationMethod.mockRejectedValue(
          new Error('selection failed'),
        );

        // Act
        const result = await neat.selectMutationMethod(genome, false);

        // Assert
        expect(result).toBeNull();
      });
    });
  });

  describe('population summary bridges', () => {
    describe('given the caller reads the fittest genome and average score through the root API', () => {
      it('delegates both summary lookups to the population-summary facade', () => {
        // Arrange
        const neat = createNeatRootHost();
        const fittestGenome = { score: 12 } as unknown as Network;
        mockGetFittest.mockReturnValue(fittestGenome);
        mockGetAverage.mockReturnValue(4.5);

        // Act
        const fittest = neat.getFittest();
        const average = neat.getAverage();

        // Assert
        expect({
          fittestCalls: mockGetFittest.mock.calls,
          averageCalls: mockGetAverage.mock.calls,
          fittest,
          average,
        }).toEqual({
          fittestCalls: [[neat]],
          averageCalls: [[neat]],
          fittest: fittestGenome,
          average: 4.5,
        });
      });
    });
  });

  describe('telemetry bridges', () => {
    describe('given diversity must be recomputed for an empty population snapshot', () => {
      it('falls back to the empty diversity summary and caches it on the controller', () => {
        // Arrange
        const neat = createNeatRootHost() as unknown as NeatDiversityInternals;
        neat.population = [];

        // Act
        const diversityStats = neat._computeDiversityStats();

        // Assert
        expect({
          diversityStats,
          cachedStats: neat._diversityStats,
        }).toEqual({
          diversityStats: buildEmptyDiversityStats(0),
          cachedStats: buildEmptyDiversityStats(0),
        });
      });
    });

    describe('given telemetry buffer export and reset are requested through the root API', () => {
      it('delegates both buffer operations to the telemetry facade', () => {
        // Arrange
        const neat = createNeatRootHost();
        mockExportTelemetryJsonl.mockReturnValue('jsonl-buffer');

        // Act
        const jsonlPayload = neat.exportTelemetryJSONL();
        neat.clearTelemetry();

        // Assert
        expect({
          exportCalls: mockExportTelemetryJsonl.mock.calls,
          clearCalls: mockClearTelemetry.mock.calls,
          jsonlPayload,
        }).toEqual({
          exportCalls: [[neat]],
          clearCalls: [[neat]],
          jsonlPayload: 'jsonl-buffer',
        });
      });
    });

    describe('given objective history is managed through the root API', () => {
      it('delegates objective clearing and event reads to the telemetry facade', () => {
        // Arrange
        const neat = createNeatRootHost();
        const objectiveEvents = [{ gen: 3, type: 'add', key: 'score' }];
        mockGetObjectiveEvents.mockReturnValue(
          objectiveEvents as ReturnType<Neat['getObjectiveEvents']>,
        );

        // Act
        neat.clearObjectives();
        const events = neat.getObjectiveEvents();

        // Assert
        expect({
          clearCalls: mockClearTelemetryObjectives.mock.calls,
          eventCalls: mockGetObjectiveEvents.mock.calls,
          events,
        }).toEqual({
          clearCalls: [[neat]],
          eventCalls: [[neat]],
          events: objectiveEvents,
        });
      });
    });

    describe('given lineage and species summaries are requested without explicit limits', () => {
      it('forwards the default lineage window and returns the current species stats', () => {
        // Arrange
        const neat = createNeatRootHost();
        const lineageSnapshot = [{ id: 7, parents: [1, 2] }];
        const speciesStats = [
          { id: 4, size: 9, bestScore: 12, lastImproved: 3 },
        ];
        mockGetLineageSnapshot.mockReturnValue(
          lineageSnapshot as ReturnType<Neat['getLineageSnapshot']>,
        );
        mockGetSpeciesStats.mockReturnValue(
          speciesStats as ReturnType<Neat['getSpeciesStats']>,
        );

        // Act
        const returnedLineage = neat.getLineageSnapshot();
        const returnedSpeciesStats = neat.getSpeciesStats();

        // Assert
        expect({
          lineageCalls: mockGetLineageSnapshot.mock.calls,
          speciesCalls: mockGetSpeciesStats.mock.calls,
          returnedLineage,
          returnedSpeciesStats,
        }).toEqual({
          lineageCalls: [[neat, LINEAGE_SNAPSHOT_DEFAULT_LIMIT]],
          speciesCalls: [[neat]],
          returnedLineage: lineageSnapshot,
          returnedSpeciesStats: speciesStats,
        });
      });
    });

    describe('given novelty and Pareto diagnostics are inspected through the root API', () => {
      it('delegates archive size, operator stats, default-front reconstruction, and archive reset', () => {
        // Arrange
        const neat = createNeatRootHost();
        const operatorStats = [{ name: 'ADD_NODE', success: 2, attempts: 3 }];
        const paretoFronts = [[{ score: 1 } as unknown as Network]];
        mockGetNoveltyArchiveSize.mockReturnValue(6);
        mockGetOperatorStats.mockReturnValue(
          operatorStats as ReturnType<Neat['getOperatorStats']>,
        );
        mockGetParetoFronts.mockReturnValue(
          paretoFronts as ReturnType<Neat['getParetoFronts']>,
        );

        // Act
        const noveltyArchiveSize = neat.getNoveltyArchiveSize();
        const returnedOperatorStats = neat.getOperatorStats();
        const returnedParetoFronts = neat.getParetoFronts();
        neat.resetNoveltyArchive();

        // Assert
        expect({
          noveltyCalls: mockGetNoveltyArchiveSize.mock.calls,
          operatorCalls: mockGetOperatorStats.mock.calls,
          paretoCalls: mockGetParetoFronts.mock.calls,
          resetCalls: mockResetNoveltyArchive.mock.calls,
          noveltyArchiveSize,
          returnedOperatorStats,
          returnedParetoFronts,
        }).toEqual({
          noveltyCalls: [[neat]],
          operatorCalls: [[neat]],
          paretoCalls: [[neat, DEFAULT_MAX_PARETO_FRONTS]],
          resetCalls: [[neat]],
          noveltyArchiveSize: 6,
          returnedOperatorStats: operatorStats,
          returnedParetoFronts: paretoFronts,
        });
      });
    });
  });
});
