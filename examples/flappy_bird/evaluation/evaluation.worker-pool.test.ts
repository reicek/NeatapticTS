import { openSharedInferenceWorker } from '../../../src/neataptic';
import { exportTransferableInferencePayload } from '../../../src/neataptic';
import { evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker } from './evaluation.fitness.utils';
import { FlappyEvaluationWorkerPool } from './evaluation.worker-pool';

jest.mock('../../../src/neataptic', () => {
  const actualModule = jest.requireActual('../../../src/neataptic');

  return {
    ...actualModule,
    exportTransferableInferencePayload: jest.fn((genome) => ({
      inputCount: 2,
      outputCount: 1,
      strategy: 'transferable',
      version: 1,
      genomeId: genome._id,
    })),
    openSharedInferenceWorker: jest.fn(),
  };
});

jest.mock('./evaluation.fitness.utils', () => ({
  evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker: jest.fn(),
}));

type DeferredEvaluation = {
  aggregate: {
    fitnessStdDev: number;
    meanFitness: number;
    meanFramesSurvived: number;
    meanPipesPassed: number;
    medianFitness: number;
    p90Fitness: number;
    robustFitness: number;
    seedCount: number;
  };
  resolve: () => void;
};

type MockSharedInferenceWorker = {
  awaitOutput: jest.Mock<Promise<Float64Array>, []>;
  infer: jest.Mock<Promise<Float64Array>, [ReadonlyArray<number>]>;
  isReady: boolean;
  release: jest.Mock<Promise<void>, []>;
  reset: jest.Mock<Promise<void>, []>;
  strategy: 'shared-memory';
  submitInput: jest.Mock<void, [ReadonlyArray<number>]>;
};

describe('FlappyEvaluationWorkerPool', () => {
  it('keeps shared-worker concurrency bounded while preserving caller order', async () => {
    const deferredEvaluationByGenomeId = new Map<number, DeferredEvaluation>();
    const genomes = [{ _id: 11 }, { _id: 22 }, { _id: 33 }];
    let activeWorkerCount = 0;
    let peakWorkerCount = 0;

    (openSharedInferenceWorker as jest.Mock).mockImplementation(
      (): MockSharedInferenceWorker => {
        activeWorkerCount += 1;
        peakWorkerCount = Math.max(peakWorkerCount, activeWorkerCount);

        return {
          awaitOutput: jest.fn(async () => new Float64Array([0.75])),
          infer: jest.fn(async (_input) => new Float64Array([0.75])),
          isReady: true,
          release: jest.fn(async () => {
            activeWorkerCount -= 1;
          }),
          reset: jest.fn(async () => undefined),
          strategy: 'shared-memory',
          submitInput: jest.fn(),
        };
      },
    );

    (
      evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker as jest.Mock
    ).mockImplementation(async (_sharedWorker, sharedSeeds, options) => {
      const nextDeferredEvaluation: DeferredEvaluation = {
        aggregate: {
          fitnessStdDev: 0,
          meanFitness: options.networkId,
          meanFramesSurvived: options.networkId,
          meanPipesPassed: options.networkId / 10,
          medianFitness: options.networkId,
          p90Fitness: options.networkId,
          robustFitness: options.networkId,
          seedCount: sharedSeeds.length,
        },
        resolve: () => undefined,
      };

      await new Promise<void>((resolve) => {
        nextDeferredEvaluation.resolve = resolve;
        deferredEvaluationByGenomeId.set(options.networkId, nextDeferredEvaluation);
      });

      return nextDeferredEvaluation.aggregate;
    });

    const workerPool = new FlappyEvaluationWorkerPool(2);
    await workerPool.initialize(genomes as never);

    const orderedEvaluationsPromise = workerPool.evaluateGenomesAcrossSeeds(
      genomes as never,
      [101, 202],
      {},
    );

    await waitForCondition(() => deferredEvaluationByGenomeId.has(11));
    await waitForCondition(() => deferredEvaluationByGenomeId.has(22));
    deferredEvaluationByGenomeId.get(11)?.resolve();
    await waitForCondition(() => deferredEvaluationByGenomeId.has(33));
    deferredEvaluationByGenomeId.get(22)?.resolve();
    deferredEvaluationByGenomeId.get(33)?.resolve();

    const orderedEvaluations = await orderedEvaluationsPromise;

    expect({
      exportedGenomeIds: (exportTransferableInferencePayload as jest.Mock).mock
        .calls.map(([genome]) => genome._id),
      orderedGenomeIds: [...orderedEvaluations.keys()].map((genome) => genome._id),
      peakWorkerCount,
      resolvedRobustFitnesses: [...orderedEvaluations.values()].map(
        (aggregate) => aggregate.robustFitness,
      ),
    }).toEqual({
      exportedGenomeIds: [11, 22, 33],
      orderedGenomeIds: [11, 22, 33],
      peakWorkerCount: 2,
      resolvedRobustFitnesses: [11, 22, 33],
    });
  });

  it('caps the default browser worker count instead of fanning out to full hardware concurrency', async () => {
    const deferredEvaluationByGenomeId = new Map<number, DeferredEvaluation>();
    const genomes = [
      { _id: 11 },
      { _id: 22 },
      { _id: 33 },
      { _id: 44 },
      { _id: 55 },
      { _id: 66 },
    ];
    const originalHardwareConcurrency = globalThis.navigator.hardwareConcurrency;
    let activeWorkerCount = 0;
    let peakWorkerCount = 0;

    Object.defineProperty(globalThis.navigator, 'hardwareConcurrency', {
      configurable: true,
      value: 12,
    });

    (openSharedInferenceWorker as jest.Mock).mockImplementation(
      (): MockSharedInferenceWorker => {
        activeWorkerCount += 1;
        peakWorkerCount = Math.max(peakWorkerCount, activeWorkerCount);

        return {
          awaitOutput: jest.fn(async () => new Float64Array([0.75])),
          infer: jest.fn(async (_input) => new Float64Array([0.75])),
          isReady: true,
          release: jest.fn(async () => {
            activeWorkerCount -= 1;
          }),
          reset: jest.fn(async () => undefined),
          strategy: 'shared-memory',
          submitInput: jest.fn(),
        };
      },
    );

    (
      evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker as jest.Mock
    ).mockImplementation(async (_sharedWorker, sharedSeeds, options) => {
      const nextDeferredEvaluation: DeferredEvaluation = {
        aggregate: {
          fitnessStdDev: 0,
          meanFitness: options.networkId,
          meanFramesSurvived: options.networkId,
          meanPipesPassed: options.networkId / 10,
          medianFitness: options.networkId,
          p90Fitness: options.networkId,
          robustFitness: options.networkId,
          seedCount: sharedSeeds.length,
        },
        resolve: () => undefined,
      };

      await new Promise<void>((resolve) => {
        nextDeferredEvaluation.resolve = resolve;
        deferredEvaluationByGenomeId.set(options.networkId, nextDeferredEvaluation);
      });

      return nextDeferredEvaluation.aggregate;
    });

    try {
      const workerPool = new FlappyEvaluationWorkerPool();
      await workerPool.initialize(genomes as never);

      const orderedEvaluationsPromise = workerPool.evaluateGenomesAcrossSeeds(
        genomes as never,
        [101, 202],
        {},
      );

      await waitForCondition(() => deferredEvaluationByGenomeId.has(11));
      await waitForCondition(() => deferredEvaluationByGenomeId.has(22));
      await waitForCondition(() => deferredEvaluationByGenomeId.has(33));
      await waitForCondition(() => deferredEvaluationByGenomeId.has(44));
      deferredEvaluationByGenomeId.get(11)?.resolve();
      await waitForCondition(() => deferredEvaluationByGenomeId.has(55));
      deferredEvaluationByGenomeId.get(22)?.resolve();
      await waitForCondition(() => deferredEvaluationByGenomeId.has(66));
      deferredEvaluationByGenomeId.get(33)?.resolve();
      deferredEvaluationByGenomeId.get(44)?.resolve();
      deferredEvaluationByGenomeId.get(55)?.resolve();
      deferredEvaluationByGenomeId.get(66)?.resolve();

      const orderedEvaluations = await orderedEvaluationsPromise;

      expect({
        orderedGenomeIds: [...orderedEvaluations.keys()].map((genome) => genome._id),
        peakWorkerCount,
        resolvedRobustFitnesses: [...orderedEvaluations.values()].map(
          (aggregate) => aggregate.robustFitness,
        ),
      }).toEqual({
        orderedGenomeIds: [11, 22, 33, 44, 55, 66],
        peakWorkerCount: 4,
        resolvedRobustFitnesses: [11, 22, 33, 44, 55, 66],
      });
    } finally {
      Object.defineProperty(globalThis.navigator, 'hardwareConcurrency', {
        configurable: true,
        value: originalHardwareConcurrency,
      });
    }
  });
});

async function waitForCondition(
  predicate: () => boolean,
  maximumAttempts = 25,
): Promise<void> {
  for (let attemptIndex = 0; attemptIndex < maximumAttempts; attemptIndex += 1) {
    if (predicate()) {
      return;
    }

    await Promise.resolve();
  }

  throw new Error('Timed out while waiting for the worker-pool test condition.');
}