import { Neat, methods } from '../../../src/neataptic';
import type Network from '../../../src/architecture/network';
import { createGenomeFromNetwork } from '../../../src/neat/genome/genome';
import * as flappyEvaluation from '../flappyEvaluation';
import {
  FLAPPY_BROWSER_ELITISM_COUNT,
  FLAPPY_BROWSER_POPULATION_SIZE,
  FLAPPY_MAX_FRAMES_PER_EPISODE,
  FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from '../constants/constants';
import { buildExampleArchitectureProfileNetwork } from '../../architectureProfiles';
import type { WorkerInitMessage } from './flappy-evolution-worker.types';
import { createInitializedWorkerRuntime } from './flappy-evolution-worker.runtime.service';
import { warmStartWorkerGenerationZeroIfNeeded } from './flappy-evolution-worker.warm-start.service';

interface WorkerRuntimeWithSeedNetwork {
  options: {
    allowRecurrent?: boolean;
    mutation?: Array<{ name: string }>;
    network: {
      describeArchitecture: () => {
        hiddenLayerSizes: number[];
      };
      inputNodeIds: number[];
      outputNodeIds: number[];
    };
  };
}

interface WorkerRuntimeWithPopulation extends Neat {
  fitness: (network: Network) => number;
  population: Network[];
}

afterEach(() => {
  jest.restoreAllMocks();
});

describe('createInitializedWorkerRuntime', () => {
  it('keeps the default MLP worker runtime on the feed-forward mutation shelf', () => {
    const initPayload: WorkerInitMessage['payload'] = {
      populationSize: 8,
      elitismCount: 2,
      rngSeed: 12345,
    };
    const neatRuntime = createInitializedWorkerRuntime(
      initPayload,
    ) as unknown as WorkerRuntimeWithSeedNetwork;

    expect({
      allowRecurrent: neatRuntime.options.allowRecurrent,
      mutationNames: neatRuntime.options.mutation?.map(
        (mutationMethod) => mutationMethod.name,
      ),
    }).toEqual({
      allowRecurrent: false,
      mutationNames: methods.mutation.FFW.map(
        (mutationMethod) => mutationMethod.name,
      ),
    });
  });

  it('uses the shared Flappy MLP profile as the default worker seed network', () => {
    const initPayload: WorkerInitMessage['payload'] = {
      populationSize: 8,
      elitismCount: 2,
      rngSeed: 12345,
    };
    const neatRuntime = createInitializedWorkerRuntime(
      initPayload,
    ) as unknown as WorkerRuntimeWithSeedNetwork;

    expect({
      hiddenLayerSizes:
        neatRuntime.options.network.describeArchitecture().hiddenLayerSizes,
      inputNodeIds: neatRuntime.options.network.inputNodeIds.length,
      outputNodeIds: neatRuntime.options.network.outputNodeIds.length,
    }).toEqual({
      hiddenLayerSizes: FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
      inputNodeIds: FLAPPY_NETWORK_INPUT_SIZE,
      outputNodeIds: FLAPPY_NETWORK_OUTPUT_SIZE,
    });
  });

  it('accepts an explicit shared recurrent profile when the browser requests one', () => {
    const initPayload: WorkerInitMessage['payload'] = {
      architectureProfileId: 'narx',
      populationSize: 8,
      elitismCount: 2,
      rngSeed: 12345,
    };
    const neatRuntime = createInitializedWorkerRuntime(
      initPayload,
    ) as unknown as WorkerRuntimeWithSeedNetwork;

    expect({
      hiddenLayerSizes:
        neatRuntime.options.network.describeArchitecture().hiddenLayerSizes,
      inputNodeIds: neatRuntime.options.network.inputNodeIds.length,
      outputNodeIds: neatRuntime.options.network.outputNodeIds.length,
    }).toEqual({
      hiddenLayerSizes: [24],
      inputNodeIds: FLAPPY_NETWORK_INPUT_SIZE,
      outputNodeIds: 2,
    });
  });

  it('enables recurrent mutation growth for temporal Flappy profiles', () => {
    const initPayload: WorkerInitMessage['payload'] = {
      architectureProfileId: 'gru',
      populationSize: 8,
      elitismCount: 2,
      rngSeed: 12345,
    };
    const neatRuntime = createInitializedWorkerRuntime(
      initPayload,
    ) as unknown as WorkerRuntimeWithSeedNetwork;

    expect({
      allowRecurrent: neatRuntime.options.allowRecurrent,
      mutationNames: neatRuntime.options.mutation?.map(
        (mutationMethod) => mutationMethod.name,
      ),
    }).toEqual({
      allowRecurrent: true,
      mutationNames: [
        ...methods.mutation.FFW,
        methods.mutation.ADD_BACK_CONN,
        methods.mutation.SUB_BACK_CONN,
        methods.mutation.ADD_SELF_CONN,
        methods.mutation.SUB_SELF_CONN,
      ].map((mutationMethod) => mutationMethod.name),
    });
  });

  it('keeps the default MLP worker runtime on the single-rollout browser objective', () => {
    const singleRolloutFitnessSpy = jest
      .spyOn(flappyEvaluation, 'evaluateFlappyFitness')
      .mockReturnValue(123);
    const sharedSeedFitnessSpy = jest.spyOn(
      flappyEvaluation,
      'evaluateFlappyFitnessAcrossSeeds',
    );
    const initPayload: WorkerInitMessage['payload'] = {
      populationSize: 8,
      elitismCount: 2,
      rngSeed: 12345,
    };
    const neatRuntime = createInitializedWorkerRuntime(
      initPayload,
    ) as WorkerRuntimeWithPopulation;
    const resolvedFitness = neatRuntime.fitness(neatRuntime.population[0]);

    expect({
      aggregateCallCount: sharedSeedFitnessSpy.mock.calls.length,
      resolvedFitness,
      singleRolloutCallCount: singleRolloutFitnessSpy.mock.calls.length,
      singleRolloutOptions: singleRolloutFitnessSpy.mock.calls.at(-1)?.[1],
    }).toEqual({
      aggregateCallCount: 0,
      resolvedFitness: 123,
      singleRolloutCallCount: 1,
      singleRolloutOptions: {
        enableEarlyTermination: true,
        maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
      },
    });
  });

  it.each([
    {
      architectureProfileId: 'narx' as const,
      expectedSharedSeedCount: 1,
    },
    {
      architectureProfileId: 'gru' as const,
      expectedSharedSeedCount: 1,
    },
    {
      architectureProfileId: 'lstm' as const,
      expectedSharedSeedCount: 4,
    },
  ])(
    'uses a pipe-first shared-seed browser scalar for the $architectureProfileId worker profile',
    ({ architectureProfileId, expectedSharedSeedCount }) => {
      const singleRolloutFitnessSpy = jest
        .spyOn(flappyEvaluation, 'evaluateFlappyFitness')
        .mockReturnValue(999);
      const sharedSeedFitnessSpy = jest
        .spyOn(flappyEvaluation, 'evaluateFlappyFitnessAcrossSeeds')
        .mockReturnValue({
          seedCount: 1,
          meanFitness: 0,
          medianFitness: 0,
          p90Fitness: 0,
          fitnessStdDev: 10,
          robustFitness: 321,
          meanPipesPassed: 2,
          meanFramesSurvived: 150,
        });
      const initPayload: WorkerInitMessage['payload'] = {
        architectureProfileId,
        populationSize: 8,
        elitismCount: 2,
        rngSeed: 12345,
      };
      const neatRuntime = createInitializedWorkerRuntime(
        initPayload,
      ) as WorkerRuntimeWithPopulation;
      const resolvedFitness = neatRuntime.fitness(neatRuntime.population[0]);
      const sharedSeedFitnessCall = sharedSeedFitnessSpy.mock.calls.at(-1);

      expect({
        aggregateOptions: sharedSeedFitnessCall?.[2],
        resolvedFitness,
        sharedSeedCount: sharedSeedFitnessCall?.[1]?.length,
        singleRolloutCallCount: singleRolloutFitnessSpy.mock.calls.length,
      }).toEqual({
        aggregateOptions: {
          enableEarlyTermination: true,
          maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
          normalizeFitness: true,
          pipeProgressTarget: 12,
        },
        resolvedFitness: 20_145,
        sharedSeedCount: expectedSharedSeedCount,
        singleRolloutCallCount: 0,
      });
    },
  );

  it('rotates the LSTM shared rollout seed batch after each evolved generation', () => {
    const sharedSeedFitnessSpy = jest
      .spyOn(flappyEvaluation, 'evaluateFlappyFitnessAcrossSeeds')
      .mockReturnValue({
        seedCount: 2,
        meanFitness: 0,
        medianFitness: 0,
        p90Fitness: 0,
        fitnessStdDev: 10,
        robustFitness: 321,
        meanPipesPassed: 2,
        meanFramesSurvived: 150,
      });
    const neatRuntime = createInitializedWorkerRuntime({
      architectureProfileId: 'lstm',
      populationSize: 8,
      elitismCount: 2,
      rngSeed: 12345,
    }) as WorkerRuntimeWithPopulation & { generation: number };

    neatRuntime.fitness(neatRuntime.population[0]);
    const firstGenerationSeeds = [
      ...(sharedSeedFitnessSpy.mock.calls.at(-1)?.[1] ?? []),
    ];

    neatRuntime.fitness(neatRuntime.population[0]);
    const repeatedGenerationSeeds = [
      ...(sharedSeedFitnessSpy.mock.calls.at(-1)?.[1] ?? []),
    ];

    neatRuntime.generation = 1;
    neatRuntime.fitness(neatRuntime.population[0]);
    const nextGenerationSeeds = [
      ...(sharedSeedFitnessSpy.mock.calls.at(-1)?.[1] ?? []),
    ];

    expect({
      firstGenerationSeedCount: firstGenerationSeeds.length,
      repeatsWithinGeneration:
        JSON.stringify(firstGenerationSeeds) ===
        JSON.stringify(repeatedGenerationSeeds),
      rotatesAcrossGenerations:
        JSON.stringify(firstGenerationSeeds) !==
        JSON.stringify(nextGenerationSeeds),
      secondGenerationSeedCount: nextGenerationSeeds.length,
    }).toEqual({
      firstGenerationSeedCount: 4,
      repeatsWithinGeneration: true,
      rotatesAcrossGenerations: true,
      secondGenerationSeedCount: 4,
    });
  });

  it.each(['gru', 'lstm'] as const)(
    'keeps the generation-zero worker population structurally valid before warm-start for the %s profile',
    (architectureProfileId) => {
      const initPayload: WorkerInitMessage['payload'] = {
        architectureProfileId,
        populationSize: FLAPPY_BROWSER_POPULATION_SIZE,
        elitismCount: FLAPPY_BROWSER_ELITISM_COUNT,
        rngSeed: 12345,
      };
      const neatRuntime = createInitializedWorkerRuntime(initPayload);
      const generationZeroPopulation = (
        neatRuntime as unknown as {
          population: Network[];
        }
      ).population;
      const firstInvalidPopulationIndex = generationZeroPopulation.findIndex(
        (populationNetwork) => {
          try {
            createGenomeFromNetwork(populationNetwork);
            return false;
          } catch {
            return true;
          }
        },
      );

      expect(firstInvalidPopulationIndex).toBe(-1);
    },
  );

  it.each(['gru', 'lstm'] as const)(
    'keeps the cloned %s seed structurally valid when dead-end repair runs under a fresh tracker',
    (architectureProfileId) => {
      const repairProbeRuntime = new Neat(
        FLAPPY_NETWORK_INPUT_SIZE,
        FLAPPY_NETWORK_OUTPUT_SIZE,
        () => 0,
        {
          popsize: 0,
          allowRecurrent: true,
          mutation: [
            ...methods.mutation.FFW,
            methods.mutation.ADD_BACK_CONN,
            methods.mutation.SUB_BACK_CONN,
            methods.mutation.ADD_SELF_CONN,
            methods.mutation.SUB_SELF_CONN,
          ],
        },
      );
      const seedNetwork = buildExampleArchitectureProfileNetwork(
        'flappy-bird',
        architectureProfileId,
      ).clone();
      let strictGenomeOk = true;

      repairProbeRuntime.ensureNoDeadEnds(seedNetwork);

      try {
        createGenomeFromNetwork(seedNetwork);
      } catch {
        strictGenomeOk = false;
      }

      expect(strictGenomeOk).toBe(true);
    },
  );

  it.each(['gru', 'lstm'] as const)(
    'keeps the first worker generation structurally valid for the %s profile',
    async (architectureProfileId) => {
      const initPayload: WorkerInitMessage['payload'] = {
        architectureProfileId,
        populationSize: FLAPPY_BROWSER_POPULATION_SIZE,
        elitismCount: FLAPPY_BROWSER_ELITISM_COUNT,
        rngSeed: 12345,
      };
      const neatRuntime = createInitializedWorkerRuntime(initPayload);

      await warmStartWorkerGenerationZeroIfNeeded(neatRuntime, {
        architectureProfileId,
        workerInitSeed: initPayload.rngSeed,
        generationZeroWarmStartApplied: false,
      });

      let firstInvalidPopulationIndex = -1;

      for (let generationStep = 0; generationStep < 2; generationStep++) {
        await neatRuntime.evolve();
        const evolvedPopulation = (
          neatRuntime as unknown as {
            population: Network[];
          }
        ).population;
        firstInvalidPopulationIndex = evolvedPopulation.findIndex(
          (populationNetwork) => {
            try {
              createGenomeFromNetwork(populationNetwork);
              return false;
            } catch {
              return true;
            }
          },
        );

        if (firstInvalidPopulationIndex !== -1) {
          break;
        }
      }

      expect(firstInvalidPopulationIndex).toBe(-1);
    },
  );
});
