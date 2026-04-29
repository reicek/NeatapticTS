import {
  DEFAULT_ELITISM,
  DEFAULT_MAX_CONNS,
  DEFAULT_MAX_GATES,
  DEFAULT_MAX_NODES,
  DEFAULT_MUTATION_AMOUNT,
  DEFAULT_MUTATION_RATE,
  DEFAULT_NEAT_CONSTRUCTOR_DEFAULTS,
  DEFAULT_POPULATION_SIZE,
  DEFAULT_PROVENANCE,
} from '../neat.defaults.constants';
import {
  initializeNeatConstructor,
  type NeatInitializationHost,
} from './neat.init';

type InitializationHostHarness = NeatInitializationHost & {
  createPool: jest.Mock;
};

type InitializeNeatConstructorFunction = typeof initializeNeatConstructor;

type FastModeOptionBag = {
  fastMode: boolean;
  novelty: { enabled: boolean; k?: number };
  diversityMetrics: {
    enabled?: boolean;
    pairSample?: number;
    graphletSample?: number;
  };
  multiObjective: { enabled: boolean; objectives?: unknown[] };
};

type CatalogFallbackOptionBag = {
  crossover?: unknown;
  mutation?: unknown[];
  selection?: unknown;
};

type ConcreteBootstrapOptionBag = CatalogFallbackOptionBag & {
  popsize: number;
  elitism: number;
  provenance: number;
  mutationRate: number;
  mutationAmount: number;
  fitnessPopulation: boolean;
  clear: boolean;
  equal: boolean;
  compatibilityThreshold: number;
  maxNodes: number;
  maxConns: number;
  maxGates: number;
  excessCoeff: number;
  disjointCoeff: number;
  weightDiffCoeff: number;
  novelty: { enabled: boolean; k?: number };
  diversityMetrics: {
    enabled: boolean;
    pairSample?: number;
    graphletSample?: number;
  };
  speciation: boolean;
  lineage: { enabled: boolean };
  lineageTracking: boolean;
};

function createInitializationHost(): InitializationHostHarness {
  return {
    options: {},
    population: [],
    _lineageEnabled: false,
    _getRNG() {
      return () => 0.5;
    },
    createPool: jest.fn(),
  } as unknown as InitializationHostHarness;
}

function createConstructorDefaults(
  overrides: Partial<typeof DEFAULT_NEAT_CONSTRUCTOR_DEFAULTS> = {},
) {
  return {
    ...DEFAULT_NEAT_CONSTRUCTOR_DEFAULTS,
    ...overrides,
  };
}

function loadMockedInitializeNeatConstructor(input: {
  methodsModule: Record<string, unknown>;
  selectionModule: Record<string, unknown>;
}): InitializeNeatConstructorFunction {
  let mockedInitializeNeatConstructor:
    | InitializeNeatConstructorFunction
    | undefined;

  jest.resetModules();

  jest.isolateModules(() => {
    jest.doMock('../../methods/methods', () => input.methodsModule);
    jest.doMock(
      '../../methods/selection/selection',
      () => input.selectionModule,
    );

    const neatInitModule =
      require('./neat.init') as typeof import('./neat.init');
    mockedInitializeNeatConstructor = neatInitModule.initializeNeatConstructor;
  });

  jest.dontMock('../../methods/methods');
  jest.dontMock('../../methods/selection/selection');

  if (!mockedInitializeNeatConstructor) {
    throw new Error('Expected mocked neat init module to load.');
  }

  return mockedInitializeNeatConstructor;
}

afterEach(() => {
  jest.resetModules();
  jest.clearAllMocks();
  jest.dontMock('../../methods/methods');
  jest.dontMock('../../methods/selection/selection');
});

describe('neat init chapter', () => {
  describe('initializeNeatConstructor', () => {
    describe('given the caller passes an empty options bag', () => {
      it('materializes the public constructor defaults onto that bag', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const optionBag = {};

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: {},
          defaults: DEFAULT_NEAT_CONSTRUCTOR_DEFAULTS,
        });

        // Assert
        expect(optionBag).toEqual(
          expect.objectContaining({
            popsize: DEFAULT_POPULATION_SIZE,
            equal: false,
            clear: false,
            elitism: DEFAULT_ELITISM,
            provenance: DEFAULT_PROVENANCE,
            mutationRate: DEFAULT_MUTATION_RATE,
            mutationAmount: DEFAULT_MUTATION_AMOUNT,
            fitnessPopulation: false,
            selection: expect.anything(),
            crossover: expect.anything(),
            mutation: expect.anything(),
            maxNodes: DEFAULT_MAX_NODES,
            maxConns: DEFAULT_MAX_CONNS,
            maxGates: DEFAULT_MAX_GATES,
          }),
        );
      });
    });

    describe('given explicit population and elitism settings are provided', () => {
      it('preserves those values and still bootstraps the initial pool', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const optionBag = { popsize: 10, elitism: 2 };

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: { ...optionBag },
          defaults: DEFAULT_NEAT_CONSTRUCTOR_DEFAULTS,
        });

        // Assert
        expect({
          popsize: optionBag.popsize,
          elitism: optionBag.elitism,
          createPoolCalls: initializationHost.createPool.mock.calls,
        }).toEqual({
          popsize: 10,
          elitism: 2,
          createPoolCalls: [[null]],
        });
      });
    });

    describe('given the caller already provides concrete bootstrap values', () => {
      it('preserves those values and skips pool creation when popsize is zero', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const optionBag: ConcreteBootstrapOptionBag = {
          popsize: 0,
          elitism: 4,
          provenance: 0,
          mutationRate: 0.9,
          mutationAmount: 6,
          fitnessPopulation: true,
          clear: true,
          equal: true,
          compatibilityThreshold: 2.5,
          maxNodes: 42,
          maxConns: 84,
          maxGates: 21,
          excessCoeff: 0.3,
          disjointCoeff: 0.4,
          weightDiffCoeff: 0.5,
          mutation: ['CUSTOM_MUTATION'],
          selection: 'CUSTOM_SELECTION',
          crossover: 'CUSTOM_CROSSOVER',
          novelty: { enabled: false },
          diversityMetrics: { enabled: false },
          speciation: true,
          lineage: { enabled: false },
          lineageTracking: false,
        };

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: { ...optionBag },
          defaults: DEFAULT_NEAT_CONSTRUCTOR_DEFAULTS,
        });

        // Assert
        expect({
          createPoolCalls: initializationHost.createPool.mock.calls,
          lineageEnabled: initializationHost._lineageEnabled,
          optionBag,
        }).toEqual({
          createPoolCalls: [],
          lineageEnabled: false,
          optionBag: {
            popsize: 0,
            elitism: 4,
            provenance: 0,
            mutationRate: 0.9,
            mutationAmount: 6,
            fitnessPopulation: true,
            clear: true,
            equal: true,
            compatibilityThreshold: 2.5,
            maxNodes: 42,
            maxConns: 84,
            maxGates: 21,
            excessCoeff: 0.3,
            disjointCoeff: 0.4,
            weightDiffCoeff: 0.5,
            mutation: ['CUSTOM_MUTATION'],
            selection: 'CUSTOM_SELECTION',
            crossover: 'CUSTOM_CROSSOVER',
            novelty: { enabled: false },
            diversityMetrics: { enabled: false },
            speciation: true,
            lineage: { enabled: false },
            lineageTracking: false,
          },
        });
      });
    });

    describe('given a seed network is supplied during bootstrap', () => {
      it('creates the initial pool from that network reference', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const seedNetwork = {} as never;
        const optionBag = { network: seedNetwork };

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: { ...optionBag },
          defaults: createConstructorDefaults({ provenance: 0 }),
        });

        // Assert
        expect(initializationHost.createPool.mock.calls).toEqual([
          [seedNetwork],
        ]);
      });
    });

    describe('given fast mode starts from sparse constructor state', () => {
      it('hydrates the missing host state and fast-mode sampling defaults', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const defaults = createConstructorDefaults({ provenance: 0 });
        const optionBag: FastModeOptionBag = {
          fastMode: true,
          novelty: { enabled: true },
          diversityMetrics: {},
          multiObjective: { enabled: true },
        };

        Reflect.set(initializationHost, 'population', undefined);

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: { ...optionBag },
          defaults,
        });

        // Assert
        expect({
          diversityMetrics: optionBag.diversityMetrics,
          hasInnovationTracker:
            initializationHost._innovationTracker !== undefined,
          multiObjectiveObjectives: optionBag.multiObjective?.objectives,
          nextSpeciesId: initializationHost._nextSpeciesId,
          novelty: optionBag.novelty,
          noveltyArchive: initializationHost._noveltyArchive,
          objectiveAgesIsMap: initializationHost._objectiveAges instanceof Map,
          objectiveStaleIsMap:
            initializationHost._objectiveStale instanceof Map,
          pendingObjectiveAdds: initializationHost._pendingObjectiveAdds,
          pendingObjectiveRemoves: initializationHost._pendingObjectiveRemoves,
          population: initializationHost.population,
          prevSpeciesMembersIsMap:
            initializationHost._prevSpeciesMembers instanceof Map,
          species: initializationHost._species,
          speciesCreatedIsMap:
            initializationHost._speciesCreated instanceof Map,
          speciesLastStatsIsMap:
            initializationHost._speciesLastStats instanceof Map,
        }).toEqual({
          diversityMetrics: {
            graphletSample: defaults.diversityGraphletSample,
            pairSample: defaults.diversityPairSample,
          },
          hasInnovationTracker: true,
          multiObjectiveObjectives: [],
          nextSpeciesId: 1,
          novelty: { enabled: true, k: defaults.noveltyK },
          noveltyArchive: [],
          objectiveAgesIsMap: true,
          objectiveStaleIsMap: true,
          pendingObjectiveAdds: [],
          pendingObjectiveRemoves: [],
          population: [],
          prevSpeciesMembersIsMap: true,
          species: [],
          speciesCreatedIsMap: true,
          speciesLastStatsIsMap: true,
        });
      });
    });

    describe('given fast mode and controller bookkeeping are already initialized', () => {
      it('preserves the existing state containers and sampling values', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const innovationTracker = { marker: 'tracker' };
        const species = ['existing-species'];
        const speciesCreated = new Map([[1, 2]]);
        const prevSpeciesMembers = new Map([[1, new Set([2])]]);
        const speciesLastStats = new Map([[1, { bestScore: 4 }]]);
        const objectiveAges = new Map([['fitness', 3]]);
        const pendingObjectiveAdds = ['add-objective'];
        const pendingObjectiveRemoves = ['remove-objective'];
        const objectiveStale = new Map([['fitness', false]]);
        const defaults = createConstructorDefaults({ provenance: 0 });
        const optionBag: FastModeOptionBag = {
          fastMode: true,
          novelty: { enabled: true, k: 11 },
          diversityMetrics: {
            enabled: true,
            pairSample: 7,
            graphletSample: 5,
          },
          multiObjective: { enabled: true, objectives: ['fitness'] },
        };

        initializationHost._innovationTracker = innovationTracker as never;
        initializationHost._species = species;
        initializationHost._nextSpeciesId = 9;
        initializationHost._speciesCreated = speciesCreated;
        initializationHost._prevSpeciesMembers = prevSpeciesMembers;
        initializationHost._speciesLastStats = speciesLastStats;
        initializationHost._objectiveAges = objectiveAges;
        initializationHost._pendingObjectiveAdds = pendingObjectiveAdds;
        initializationHost._pendingObjectiveRemoves = pendingObjectiveRemoves;
        initializationHost._objectiveStale = objectiveStale;

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: { ...optionBag },
          defaults,
        });

        // Assert
        expect({
          diversityMetrics: optionBag.diversityMetrics,
          innovationTrackerPreserved:
            initializationHost._innovationTracker === innovationTracker,
          multiObjectiveObjectives: optionBag.multiObjective?.objectives,
          nextSpeciesId: initializationHost._nextSpeciesId,
          novelty: optionBag.novelty,
          objectiveAgesPreserved:
            initializationHost._objectiveAges === objectiveAges,
          objectiveStalePreserved:
            initializationHost._objectiveStale === objectiveStale,
          pendingObjectiveAddsPreserved:
            initializationHost._pendingObjectiveAdds === pendingObjectiveAdds,
          pendingObjectiveRemovesPreserved:
            initializationHost._pendingObjectiveRemoves ===
            pendingObjectiveRemoves,
          prevSpeciesMembersPreserved:
            initializationHost._prevSpeciesMembers === prevSpeciesMembers,
          speciesCreatedPreserved:
            initializationHost._speciesCreated === speciesCreated,
          speciesLastStatsPreserved:
            initializationHost._speciesLastStats === speciesLastStats,
          speciesPreserved: initializationHost._species === species,
        }).toEqual({
          diversityMetrics: {
            enabled: true,
            graphletSample: 5,
            pairSample: 7,
          },
          innovationTrackerPreserved: true,
          multiObjectiveObjectives: ['fitness'],
          nextSpeciesId: 9,
          novelty: { enabled: true, k: 11 },
          objectiveAgesPreserved: true,
          objectiveStalePreserved: true,
          pendingObjectiveAddsPreserved: true,
          pendingObjectiveRemovesPreserved: true,
          prevSpeciesMembersPreserved: true,
          speciesCreatedPreserved: true,
          speciesLastStatsPreserved: true,
          speciesPreserved: true,
        });
      });
    });

    describe('given provenance does not enable lineage automatically', () => {
      it('still enables lineage when the explicit lineage flag is on', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const optionBag = { lineage: { enabled: true } };

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: { ...optionBag },
          defaults: createConstructorDefaults({ provenance: 0 }),
        });

        // Assert
        expect(initializationHost._lineageEnabled).toBe(true);
      });
    });

    describe('given constructor defaults carry positive provenance', () => {
      it('enables lineage even when the explicit lineage flag is absent', () => {
        // Arrange
        const initializationHost = createInitializationHost();

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag: {},
          rawOptions: {},
          defaults: createConstructorDefaults({ provenance: 2 }),
        });

        // Assert
        expect(initializationHost._lineageEnabled).toBe(true);
      });
    });

    describe('given lineage is explicitly disabled while provenance stays positive', () => {
      it('still enables lineage from the positive provenance branch', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const optionBag = { lineage: { enabled: false } };

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: { ...optionBag },
          defaults: createConstructorDefaults({ provenance: 2 }),
        });

        // Assert
        expect(initializationHost._lineageEnabled).toBe(true);
      });
    });

    describe('given provenance is explicitly null during bootstrap', () => {
      it('falls back to zero for the provenance lineage check', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const optionBag = { provenance: null } as unknown as {
          provenance?: number;
        };

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: {},
          defaults: createConstructorDefaults({ provenance: 0 }),
        });

        // Assert
        expect(initializationHost._lineageEnabled).toBe(false);
      });
    });

    describe('given lineage tracking is requested explicitly', () => {
      it('enables lineage even when provenance remains disabled', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const optionBag = { lineageTracking: true };

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: { ...optionBag },
          defaults: createConstructorDefaults({ provenance: 0 }),
        });

        // Assert
        expect(initializationHost._lineageEnabled).toBe(true);
      });
    });

    describe('given raw lineage pressure is enabled before lineage tracking is on', () => {
      it('turns on lineage tracking during bootstrap', () => {
        // Arrange
        const initializationHost = createInitializationHost();

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag: {},
          rawOptions: { lineagePressure: { enabled: true } },
          defaults: createConstructorDefaults({ provenance: 0 }),
        });

        // Assert
        expect(initializationHost._lineageEnabled).toBe(true);
      });
    });

    describe('given the methods catalog omits crossover and the shared mutation list', () => {
      it('falls back to the feed-forward mutation and method-catalog tournament selection', () => {
        // Arrange
        const initializeNeatConstructorWithMocks =
          loadMockedInitializeNeatConstructor({
            methodsModule: {
              mutation: { ALL: undefined, FFW: 'FFW_ONLY' },
              crossover: undefined,
              selection: { TOURNAMENT: 'METHOD_TOURNAMENT' },
            },
            selectionModule: {
              selection: {
                TOURNAMENT: undefined,
                FITNESS_PROPORTIONATE: 'FITNESS_ONLY',
              },
            },
          });
        const initializationHost = createInitializationHost();
        const optionBag: CatalogFallbackOptionBag = {};

        // Act
        initializeNeatConstructorWithMocks(initializationHost, {
          optionBag,
          rawOptions: {},
          defaults: createConstructorDefaults({ provenance: 0 }),
        });

        // Assert
        expect({
          crossover: optionBag.crossover,
          mutation: optionBag.mutation,
          selection: optionBag.selection,
        }).toEqual({
          crossover: undefined,
          mutation: ['FFW_ONLY'],
          selection: 'METHOD_TOURNAMENT',
        });
      });
    });

    describe('given neither tournament selection source nor feed-forward mutation exist', () => {
      it('falls back to an empty mutation list and fitness-proportionate selection', () => {
        // Arrange
        const initializeNeatConstructorWithMocks =
          loadMockedInitializeNeatConstructor({
            methodsModule: {
              mutation: { ALL: undefined, FFW: undefined },
              crossover: { SINGLE_POINT: 'SINGLE_POINT' },
              selection: {},
            },
            selectionModule: {
              selection: {
                TOURNAMENT: undefined,
                FITNESS_PROPORTIONATE: 'FITNESS_ONLY',
              },
            },
          });
        const initializationHost = createInitializationHost();
        const optionBag: CatalogFallbackOptionBag = {};

        // Act
        initializeNeatConstructorWithMocks(initializationHost, {
          optionBag,
          rawOptions: {},
          defaults: createConstructorDefaults({ provenance: 0 }),
        });

        // Assert
        expect({
          mutation: optionBag.mutation,
          selection: optionBag.selection,
        }).toEqual({
          mutation: [],
          selection: 'FITNESS_ONLY',
        });
      });
    });
  });
});
