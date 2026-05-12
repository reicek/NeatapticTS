import { evaluateFlappyFitnessAcrossSeeds } from '../../flappyEvaluation';
import { evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker } from '../../evaluation/evaluation.fitness.utils';
import { evaluateSpecificGenomesAcrossSeeds } from './trainer.evaluation.service.services';

jest.mock('../../flappyEvaluation', () => ({
  evaluateFlappyFitnessAcrossSeeds: jest.fn(),
}));

jest.mock('../../evaluation/evaluation.fitness.utils', () => ({
  evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker: jest.fn(),
}));

describe('evaluateSpecificGenomesAcrossSeeds', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('routes aggregate evaluation through the worker pool when provided', async () => {
    const firstGenome = { _id: 11, activate: () => [0.1, 0.9] };
    const secondGenome = { _id: 22, activate: () => [0.2, 0.8] };
    const aggregateByGenome = new Map();
    const workerPool = {
      resolveOrderedPayloads: jest
        .fn()
        .mockResolvedValue([{ id: 11 }, { id: 22 }]),
      parallelWorkerPool: {
        evaluateOrderedBatch: jest.fn(
          async (
            payloads: Array<{ id: number }>,
            evaluateWithWorker: (
              worker: { id: number; release: jest.Mock<Promise<void>, []> },
              payload: { id: number },
              payloadIndex: number,
            ) => Promise<ReturnType<typeof createAggregate>>,
          ) => {
            return Promise.all(
              payloads.map((payload, payloadIndex) =>
                evaluateWithWorker(
                  {
                    id: payload.id,
                    release: jest.fn(async () => undefined),
                  },
                  payload,
                  payloadIndex,
                ),
              ),
            );
          },
        ),
      },
    };

    (evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker as jest.Mock)
      .mockResolvedValueOnce(createAggregate(11))
      .mockResolvedValueOnce(createAggregate(22));

    await evaluateSpecificGenomesAcrossSeeds(
      [firstGenome, secondGenome],
      [101, 202],
      { normalizeFitness: true },
      aggregateByGenome,
      { workerPool: workerPool as never },
    );

    expect({
      directEvaluationCalls: (evaluateFlappyFitnessAcrossSeeds as jest.Mock)
        .mock.calls.length,
      parallelBatchCallCount:
        workerPool.parallelWorkerPool.evaluateOrderedBatch.mock.calls.length,
      resolvedFitnesses: [...aggregateByGenome.values()].map(
        (aggregate) => aggregate.robustFitness,
      ),
      payloadResolutionCallCount:
        workerPool.resolveOrderedPayloads.mock.calls.length,
    }).toEqual({
      directEvaluationCalls: 0,
      parallelBatchCallCount: 1,
      resolvedFitnesses: [11, 22],
      payloadResolutionCallCount: 1,
    });
  });

  it('falls back to direct sequential evaluation when no worker pool exists', async () => {
    const firstGenome = { _id: 11, activate: () => [0.1, 0.9] };
    const secondGenome = { _id: 22, activate: () => [0.2, 0.8] };
    const aggregateByGenome = new Map();

    (evaluateFlappyFitnessAcrossSeeds as jest.Mock)
      .mockReturnValueOnce(createAggregate(11))
      .mockReturnValueOnce(createAggregate(22));

    await evaluateSpecificGenomesAcrossSeeds(
      [firstGenome, secondGenome],
      [101, 202],
      { normalizeFitness: true },
      aggregateByGenome,
    );

    expect({
      directEvaluationCalls: (evaluateFlappyFitnessAcrossSeeds as jest.Mock)
        .mock.calls.length,
      resolvedFitnesses: [...aggregateByGenome.values()].map(
        (aggregate) => aggregate.robustFitness,
      ),
    }).toEqual({
      directEvaluationCalls: 2,
      resolvedFitnesses: [11, 22],
    });
  });
});

function createAggregate(robustFitness: number) {
  return {
    seedCount: 2,
    meanFitness: robustFitness,
    medianFitness: robustFitness,
    p90Fitness: robustFitness,
    fitnessStdDev: 0,
    robustFitness,
    meanPipesPassed: robustFitness / 10,
    meanFramesSurvived: robustFitness,
  };
}
