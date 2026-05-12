import { Neat, methods } from '../../../src/neataptic';
import type Network from '../../../src/architecture/network';
import { createGenomeFromNetwork } from '../../../src/neat/genome/genome';
import * as flappyEvaluation from '../flappyEvaluation';
import {
  FLAPPY_BROWSER_ELITISM_COUNT,
  FLAPPY_BROWSER_POPULATION_SIZE,
  FLAPPY_MAX_FRAMES_PER_EPISODE,
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
  fitness: (network: Network) => number | Promise<number>;
  population: Network[];
}

interface WorkerRuntimeWithPopulationFitness extends Neat {
  fitness: (population: Network[]) => Promise<void>;
  population: Network[];
  options: {
    fitnessPopulation?: boolean;
  };
}

afterEach(() => {
  jest.restoreAllMocks();
});

describe('createInitializedWorkerRuntime', () => {
  it('keeps the default worker runtime on the feed-forward mutation shelf', () => {
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

  it('uses the shared Flappy default profile dimensions for the default worker seed network', () => {
    const initPayload: WorkerInitMessage['payload'] = {
      populationSize: 8,
      elitismCount: 2,
      rngSeed: 12345,
    };
    const neatRuntime = createInitializedWorkerRuntime(
      initPayload,
    ) as unknown as WorkerRuntimeWithSeedNetwork;

    expect({
      hasHiddenNodes:
        neatRuntime.options.network.describeArchitecture().hiddenLayerSizes
          .length > 0,
      inputNodeIds: neatRuntime.options.network.inputNodeIds.length,
      outputNodeIds: neatRuntime.options.network.outputNodeIds.length,
    }).toEqual({
      hasHiddenNodes: true,
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

  it('uses the saved browser-local champion network when init payload provides one', () => {
    const championNetwork = buildExampleArchitectureProfileNetwork(
      'flappy-bird',
      'mlp',
    );
    const hiddenNode = championNetwork.nodes.find(
      (candidateNode) => candidateNode.type === 'hidden',
    );

    if (!hiddenNode) {
      throw new Error(
        'Expected the saved champion probe network to have one hidden node.',
      );
    }

    hiddenNode.bias = 42;

    const initPayload: WorkerInitMessage['payload'] = {
      architectureProfileId: 'mlp',
      championNetworkJson: championNetwork.toJSON(),
      populationSize: 8,
      elitismCount: 2,
      rngSeed: 12345,
    };
    const neatRuntime = createInitializedWorkerRuntime(
      initPayload,
    ) as unknown as {
      options: {
        network: Network;
      };
    };
    const restoredHiddenNode = neatRuntime.options.network.nodes.find(
      (candidateNode) => candidateNode.type === 'hidden',
    );

    expect({
      restoredHiddenBias: restoredHiddenNode?.bias,
    }).toEqual({
      restoredHiddenBias: 42,
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

  it('keeps the default worker runtime on the single-rollout browser objective', async () => {
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
    const resolvedFitness = await neatRuntime.fitness(
      neatRuntime.population[0],
    );

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
      expectedSharedSeedCount: 3,
    },
    {
      architectureProfileId: 'gru' as const,
      expectedSharedSeedCount: 3,
    },
    {
      architectureProfileId: 'lstm' as const,
      expectedSharedSeedCount: 4,
    },
  ])(
    'uses a pipe-first shared-seed browser scalar for the $architectureProfileId worker profile',
    async ({ architectureProfileId, expectedSharedSeedCount }) => {
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
      const resolvedFitness = await neatRuntime.fitness(
        neatRuntime.population[0],
      );
      const sharedSeedFitnessCall = sharedSeedFitnessSpy.mock.calls.at(-1);

      expect({
        aggregateCallCount: sharedSeedFitnessSpy.mock.calls.length,
        aggregateOptions: sharedSeedFitnessCall?.[2],
        resolvedFitness,
        sharedSeedCount: sharedSeedFitnessCall?.[1]?.length,
        singleRolloutCallCount: singleRolloutFitnessSpy.mock.calls.length,
      }).toEqual({
        aggregateCallCount: 2,
        aggregateOptions: {
          enableEarlyTermination: true,
          maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
          normalizeFitness: true,
          pipeProgressTarget: 12,
        },
        resolvedFitness: 20_316,
        sharedSeedCount: expectedSharedSeedCount,
        singleRolloutCallCount: 0,
      });
    },
  );

  it('keeps zero-pipe recurrent genomes on the first shared seed only', async () => {
    const sharedSeedFitnessSpy = jest
      .spyOn(flappyEvaluation, 'evaluateFlappyFitnessAcrossSeeds')
      .mockReturnValue({
        seedCount: 1,
        meanFitness: 0,
        medianFitness: 0,
        p90Fitness: 0,
        fitnessStdDev: 0,
        robustFitness: 50,
        meanPipesPassed: 0,
        meanFramesSurvived: 50,
      });
    const neatRuntime = createInitializedWorkerRuntime({
      architectureProfileId: 'gru',
      populationSize: 8,
      elitismCount: 2,
      rngSeed: 12345,
    }) as WorkerRuntimeWithPopulation;

    const resolvedFitness = await neatRuntime.fitness(
      neatRuntime.population[0],
    );

    expect({
      aggregateCallCount: sharedSeedFitnessSpy.mock.calls.length,
      resolvedFitness,
      sharedSeedCount: sharedSeedFitnessSpy.mock.calls.at(-1)?.[1]?.length,
    }).toEqual({
      aggregateCallCount: 1,
      resolvedFitness: 50,
      sharedSeedCount: 1,
    });
  });

  it('rotates the LSTM shared rollout seed batch after each evolved generation', async () => {
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

    await neatRuntime.fitness(neatRuntime.population[0]);
    const firstGenerationSeeds = [
      ...(sharedSeedFitnessSpy.mock.calls.at(-1)?.[1] ?? []),
    ];

    await neatRuntime.fitness(neatRuntime.population[0]);
    const repeatedGenerationSeeds = [
      ...(sharedSeedFitnessSpy.mock.calls.at(-1)?.[1] ?? []),
    ];

    neatRuntime.generation = 1;
    await neatRuntime.fitness(neatRuntime.population[0]);
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
      jest
        .spyOn(flappyEvaluation, 'evaluateFlappyFitnessAcrossSeeds')
        .mockReturnValue({
          seedCount: 3,
          meanFitness: 0,
          medianFitness: 0,
          p90Fitness: 0,
          fitnessStdDev: 0,
          robustFitness: 0,
          meanPipesPassed: 0,
          meanFramesSurvived: 0,
        });
      jest.spyOn(flappyEvaluation, 'evaluateFlappyFitness').mockReturnValue(0);

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

  it('uses the shared-memory evaluation pool for recurrent browser profiles when one is available', async () => {
    const workerPool = {
      evaluateGenomesAcrossSeeds: jest.fn(
        async (population: Network[]) =>
          new Map([
            [population[0], createSharedAggregate(2, 321, 10)],
            [population[1], createSharedAggregate(1, 100, 0)],
          ]),
      ),
    };
    const initPayload: WorkerInitMessage['payload'] = {
      architectureProfileId: 'gru',
      populationSize: 2,
      elitismCount: 1,
      rngSeed: 12345,
    };
    const neatRuntime = createInitializedWorkerRuntime(initPayload, {
      workerPool: workerPool as never,
    }) as WorkerRuntimeWithPopulationFitness;

    await neatRuntime.fitness(neatRuntime.population);

    expect({
      fitnessPopulation: neatRuntime.options.fitnessPopulation,
      scoredPopulation: neatRuntime.population.map((populationNetwork) =>
        Number(populationNetwork.score ?? 0),
      ),
      workerPoolCallCount:
        workerPool.evaluateGenomesAcrossSeeds.mock.calls.length,
    }).toEqual({
      fitnessPopulation: true,
      scoredPopulation: [20_316, 10_100],
      workerPoolCallCount: 2,
    });
  });
});

function createSharedAggregate(
  meanPipesPassed: number,
  robustFitness: number,
  fitnessStdDev: number,
): flappyEvaluation.FlappySeedBatchEvaluation {
  return {
    seedCount: 3,
    meanFitness: robustFitness,
    medianFitness: robustFitness,
    p90Fitness: robustFitness,
    fitnessStdDev,
    robustFitness,
    meanPipesPassed,
    meanFramesSurvived: robustFitness,
  };
}
