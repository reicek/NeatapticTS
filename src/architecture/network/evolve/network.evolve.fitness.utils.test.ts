import { config } from '../../../config';
import Multi from '../../../multithreading/multi';
import type {
  TestWorkerConstructor,
  TestWorkerInstance,
} from '../../../multithreading/types';
import Network from '../network';
import {
  buildMultiThreadFitness,
  buildSingleThreadFitness,
  computeComplexityPenalty,
  evaluateGenomeWithWorker,
  installWorkerTerminationHook,
} from './network.evolve.fitness.utils';
import { DEFAULT_THREAD_COUNT } from './network.evolve.utils.types';

type TrainingSet = Parameters<typeof buildSingleThreadFitness>[0];
type CostFunction = Parameters<typeof buildSingleThreadFitness>[1];
type CostReference = Parameters<typeof buildMultiThreadFitness>[1];

const TRAINING_SET: TrainingSet = [{ input: [0.2], output: [0.8] }];
const COST_FUNCTION = (() => 0) as CostFunction;
const ORIGINAL_WORKERS = Multi.workers;
const ORIGINAL_WARNINGS = config.warnings;
const ORIGINAL_PROCESS_DESCRIPTOR = Object.getOwnPropertyDescriptor(
  globalThis,
  'process',
);

if (!ORIGINAL_PROCESS_DESCRIPTOR) {
  throw new Error('Expected the global process descriptor to exist');
}

describe('network evolve fitness utility chapter', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    Multi.workers = ORIGINAL_WORKERS;
    config.warnings = ORIGINAL_WARNINGS;
    Object.defineProperty(globalThis, 'process', ORIGINAL_PROCESS_DESCRIPTOR);
  });

  describe('computeComplexityPenalty', () => {
    describe('given the same genome is measured before and after a structural change', () => {
      it('reuses the cached complexity base until the structure counts change', () => {
        // Arrange
        const genome = new Network(1, 1, { seed: 701 });
        const growth = 0.5;

        // Act
        const initialPenalty = computeComplexityPenalty(genome, growth);
        const cachedPenalty = computeComplexityPenalty(genome, growth);
        genome.connections.push(
          {} as unknown as (typeof genome.connections)[number],
        );
        const invalidatedPenalty = computeComplexityPenalty(genome, growth);

        // Assert
        expect({
          cacheWasReused: initialPenalty === cachedPenalty,
          cacheWasInvalidated: invalidatedPenalty !== cachedPenalty,
        }).toEqual({
          cacheWasReused: true,
          cacheWasInvalidated: true,
        });
      });
    });
  });

  describe('buildSingleThreadFitness', () => {
    describe('given genome evaluation throws while warnings are disabled', () => {
      it('returns negative infinity without logging a warning', () => {
        // Arrange
        config.warnings = false;
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        const genome = new Network(1, 1, { seed: 702 });
        jest.spyOn(genome, 'test').mockImplementation(() => {
          throw new Error('disabled warning failure');
        });
        const fitnessFunction = buildSingleThreadFitness(
          TRAINING_SET,
          COST_FUNCTION,
          1,
          0.2,
        );

        // Act
        const fitnessScore = fitnessFunction(genome);

        // Assert
        expect({ fitnessScore, warningCount: warnSpy.mock.calls.length }).toEqual({
          fitnessScore: -Infinity,
          warningCount: 0,
        });
      });
    });

    describe('given genome evaluation throws while warnings are enabled', () => {
      it('logs the fallback warning message before penalizing the genome', () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        const genome = new Network(1, 1, { seed: 703 });
        jest.spyOn(genome, 'test').mockImplementation(() => {
          throw 'string failure';
        });
        const fitnessFunction = buildSingleThreadFitness(
          TRAINING_SET,
          COST_FUNCTION,
          1,
          0.2,
        );

        // Act
        void fitnessFunction(genome);

        // Assert
        expect(String(warnSpy.mock.calls[0]?.[0])).toContain('string failure');
      });
    });

    describe('given evaluation produces a NaN error value', () => {
      it('returns negative infinity instead of propagating NaN', () => {
        // Arrange
        const genome = new Network(1, 1, { seed: 704 });
        jest.spyOn(genome, 'test').mockReturnValue({
          error: Number.NaN,
        } as ReturnType<Network['test']>);
        const fitnessFunction = buildSingleThreadFitness(
          TRAINING_SET,
          COST_FUNCTION,
          1,
          0.2,
        );

        // Act
        const fitnessScore = fitnessFunction(genome);

        // Assert
        expect(fitnessScore).toBe(-Infinity);
      });
    });

    describe('given evaluation produces a finite error value', () => {
      it('returns a finite normalized fitness score', () => {
        // Arrange
        const genome = new Network(1, 1, { seed: 7041 });
        jest.spyOn(genome, 'test').mockReturnValue({
          error: 0.25,
        } as ReturnType<Network['test']>);
        const fitnessFunction = buildSingleThreadFitness(
          TRAINING_SET,
          COST_FUNCTION,
          2,
          0.2,
        );

        // Act
        const fitnessScore = fitnessFunction(genome);

        // Assert
        expect(Number.isFinite(fitnessScore)).toBe(true);
      });
    });
  });

  describe('evaluateGenomeWithWorker', () => {
    describe('given the worker returns a nonnumeric payload', () => {
      it('leaves the genome score unchanged', async () => {
        // Arrange
        const genome = new Network(1, 1, { seed: 705 });
        genome.score = 42;
        const worker = {
          evaluate: async () => 'not-a-number',
        } as unknown as TestWorkerInstance;

        // Act
        await evaluateGenomeWithWorker(worker, genome, 0.2);

        // Assert
        expect(genome.score).toBe(42);
      });
    });

    describe('given the worker returns NaN', () => {
      it('stores negative infinity as the penalized score', async () => {
        // Arrange
        const genome = new Network(1, 1, { seed: 706 });
        const worker = {
          evaluate: async () => Number.NaN,
        } as unknown as TestWorkerInstance;

        // Act
        await evaluateGenomeWithWorker(worker, genome, 0.2);

        // Assert
        expect(genome.score).toBe(-Infinity);
      });
    });
  });

  describe('installWorkerTerminationHook', () => {
    describe('given one worker throws during termination', () => {
      it('continues terminating the remaining workers', () => {
        // Arrange
        let didReachSecondWorker = false;
        const options: Record<string, unknown> = {};
        const workers = [
          {
            terminate: () => {
              throw new Error('terminate failure');
            },
          },
          {
            terminate: () => {
              didReachSecondWorker = true;
            },
          },
        ] as unknown as TestWorkerInstance[];
        installWorkerTerminationHook(options, workers);
        const terminateWorkers = options._workerTerminators as () => void;

        // Act
        terminateWorkers();

        // Assert
        expect(didReachSecondWorker).toBe(true);
      });
    });
  });

  describe('buildMultiThreadFitness', () => {
    describe('given node-worker discovery is unavailable in the current runtime', () => {
      it('falls back to the default single-thread worker count', async () => {
        // Arrange
        Multi.workers = {} as unknown as typeof Multi.workers;
        const options: Record<string, unknown> = {};

        // Act
        const fitnessSetup = await buildMultiThreadFitness(
          TRAINING_SET,
          COST_FUNCTION,
          1,
          0.2,
          3,
          options,
        );

        // Assert
        expect(fitnessSetup.threads).toBe(DEFAULT_THREAD_COUNT);
      });
    });

    describe('given browser worker discovery is used in a browser-like runtime', () => {
      it('uses the browser worker factory and marks the options for population fitness', async () => {
        // Arrange
        let capturedCostName = '';
        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: { versions: {} },
        });
        Multi.workers = {
          getBrowserTestWorker: async () =>
            createWorkerConstructor({
              onConstruct: (_serializedSet, workerCost) => {
                capturedCostName = workerCost.name;
              },
            }),
        } as unknown as typeof Multi.workers;
        const options: Record<string, unknown> = {};
        const browserCost = { name: 'browser-cost' } as CostReference;

        // Act
        await buildMultiThreadFitness(
          TRAINING_SET,
          browserCost,
          1,
          0.2,
          1,
          options,
        );

        // Assert
        expect({
          fitnessPopulation: options.fitnessPopulation,
          capturedCostName,
        }).toEqual({
          fitnessPopulation: true,
          capturedCostName: 'browser-cost',
        });
      });
    });

    describe('given worker discovery throws while warnings are enabled', () => {
      it('logs the single-thread fallback warning', async () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        Multi.workers = {
          getNodeTestWorker: async () => {
            throw 'worker unavailable';
          },
        } as unknown as typeof Multi.workers;

        // Act
        await buildMultiThreadFitness(
          TRAINING_SET,
          COST_FUNCTION,
          1,
          0.2,
          2,
          {},
        );

        // Assert
        expect(String(warnSpy.mock.calls[0]?.[1])).toBe('worker unavailable');
      });
    });

    describe('given worker discovery throws while warnings are disabled', () => {
      it('falls back without emitting a warning', async () => {
        // Arrange
        config.warnings = false;
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        Multi.workers = {
          getNodeTestWorker: async () => {
            throw new Error('silent fallback');
          },
        } as unknown as typeof Multi.workers;

        // Act
        const fitnessSetup = await buildMultiThreadFitness(
          TRAINING_SET,
          COST_FUNCTION,
          1,
          0.2,
          2,
          {},
        );

        // Assert
        expect({ threads: fitnessSetup.threads, warningCount: warnSpy.mock.calls.length }).toEqual({
          threads: DEFAULT_THREAD_COUNT,
          warningCount: 0,
        });
      });
    });

    describe('given every worker construction attempt fails while warnings are enabled', () => {
      it('logs the worker spawn failure warning', async () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        Multi.workers = {
          getNodeTestWorker: async () =>
            createWorkerConstructor({ shouldThrowOnConstruct: true }),
        } as unknown as typeof Multi.workers;

        // Act
        await buildMultiThreadFitness(
          TRAINING_SET,
          COST_FUNCTION,
          1,
          0.2,
          2,
          {},
        );

        // Assert
        expect(String(warnSpy.mock.calls[0]?.[0])).toContain(
          'Worker spawn failed',
        );
      });
    });

    describe('given no workers end up available for population evaluation', () => {
      it('resolves the population fitness function immediately', async () => {
        // Arrange
        Multi.workers = {
          getNodeTestWorker: async () =>
            createWorkerConstructor({ shouldThrowOnConstruct: true }),
        } as unknown as typeof Multi.workers;
        const options: Record<string, unknown> = {};
        const fitnessSetup = await buildMultiThreadFitness(
          TRAINING_SET,
          Object.create(null) as CostReference,
          1,
          0.2,
          2,
          options,
        );
        let didResolve = false;
        const populationFitness = fitnessSetup.fitnessFunction as unknown as (
          population: Network[],
        ) => Promise<void>;

        // Act
        await populationFitness([new Network(1, 1, { seed: 707 })]);
        didResolve = true;

        // Assert
        expect(didResolve).toBe(true);
      });
    });

    describe('given spawned workers can evaluate queued genomes successfully', () => {
      it('drains the queue and assigns finite penalized scores', async () => {
        // Arrange
        Multi.workers = {
          getNodeTestWorker: async () =>
            createWorkerConstructor({
              evaluate: async () => 0.3,
            }),
        } as unknown as typeof Multi.workers;
        const fitnessSetup = await buildMultiThreadFitness(
          TRAINING_SET,
          COST_FUNCTION,
          1,
          0.2,
          2,
          {},
        );
        const populationFitness = fitnessSetup.fitnessFunction as unknown as (
          population: Network[],
        ) => Promise<void>;
        const genome = new Network(1, 1, { seed: 708 });

        // Act
        await populationFitness([genome]);

        // Assert
        expect(Number.isFinite(genome.score as number)).toBe(true);
      });
    });

    describe('given one worker evaluation rejects during traversal', () => {
      it('continues draining the queue after the rejection', async () => {
        // Arrange
        let evaluationCount = 0;
        Multi.workers = {
          getNodeTestWorker: async () =>
            createWorkerConstructor({
              evaluate: async () => {
                evaluationCount += 1;

                if (evaluationCount === 1) {
                  throw new Error('reject once');
                }

                return 0.4;
              },
            }),
        } as unknown as typeof Multi.workers;
        const fitnessSetup = await buildMultiThreadFitness(
          TRAINING_SET,
          COST_FUNCTION,
          1,
          0.2,
          1,
          {},
        );
        const populationFitness = fitnessSetup.fitnessFunction as unknown as (
          population: Network[],
        ) => Promise<void>;
        const firstGenome = new Network(1, 1, { seed: 709 });
        const secondGenome = new Network(1, 1, { seed: 710 });

        // Act
        await populationFitness([firstGenome, secondGenome]);

        // Assert
        expect(Number.isFinite(secondGenome.score as number)).toBe(true);
      });
    });
  });
});

function createWorkerConstructor(input: {
  onConstruct?: (
    serializedSet: ReturnType<typeof Multi.serializeDataSet>,
    cost: { name: string },
  ) => void;
  evaluate?: (candidate: Network) => unknown | Promise<unknown>;
  terminate?: () => void;
  shouldThrowOnConstruct?: boolean;
}): TestWorkerConstructor {
  const {
    onConstruct,
    evaluate,
    terminate,
    shouldThrowOnConstruct = false,
  } = input;

  return class MockTestWorker {
    ['worker']: unknown = null;

    constructor(
      serializedSet: ReturnType<typeof Multi.serializeDataSet>,
      cost: { name: string },
    ) {
      onConstruct?.(serializedSet, cost);

      if (shouldThrowOnConstruct) {
        throw new Error('spawn failure');
      }
    }

    async evaluate(candidate: Network): Promise<unknown> {
      return evaluate ? evaluate(candidate) : 0;
    }

    terminate(): void {
      terminate?.();
    }
  } as unknown as TestWorkerConstructor;
}