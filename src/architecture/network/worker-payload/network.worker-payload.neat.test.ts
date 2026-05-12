import { createNeatParallelPopulationEvaluator } from './network.worker-payload';

interface MockGenome {
  id: number;
  localFitness: number;
  score?: number;
}

interface MockPayload {
  id: number;
}

interface MockWorker {
  id: number;
  release: jest.Mock<Promise<void>, []>;
}

describe('createNeatParallelPopulationEvaluator', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('assigns per-genome scores through the local evaluation path when parallel is disabled', async () => {
    const population: MockGenome[] = [
      { id: 11, localFitness: 1 },
      { id: 22, localFitness: 2 },
    ];
    const evaluatePopulation = createNeatParallelPopulationEvaluator({
      parallel: false,
      evaluateGenome: async (genome: MockGenome) => genome.localFitness,
    });

    await evaluatePopulation(population);

    expect(population.map((genome) => genome.score)).toEqual([1, 2]);
  });

  it('assigns ordered scores through workers when parallel execution is enabled', async () => {
    const population: MockGenome[] = [
      { id: 11, localFitness: 1 },
      { id: 22, localFitness: 2 },
    ];
    const openWorker = jest.fn(async (payload: MockPayload) => ({
      id: payload.id,
      release: jest.fn(async () => undefined),
    }));
    const evaluatePopulation = createNeatParallelPopulationEvaluator<
      MockGenome,
      MockPayload,
      MockWorker,
      { fitness: number }
    >({
      parallel: true,
      openWorker,
      resolvePayload: (genome: MockGenome) => ({ id: genome.id }),
      evaluateGenome: async (genome: MockGenome) => ({
        fitness: genome.localFitness,
      }),
      evaluateWithWorker: async (
        worker: MockWorker,
        genome: MockGenome,
        _inputIndex: number,
        payload: MockPayload,
      ) => ({
        fitness: genome.localFitness + worker.id + payload.id,
      }),
      assignResult: (genome: MockGenome, result: { fitness: number }) => {
        genome.score = result.fitness;
      },
    });

    await evaluatePopulation(population);

    expect({
      openWorkerCallCount: openWorker.mock.calls.length,
      scores: population.map((genome) => genome.score),
    }).toEqual({
      openWorkerCallCount: 2,
      scores: [23, 46],
    });
  });

  it('throws when ordered evaluation does not resolve every genome result', async () => {
    const population: MockGenome[] = [
      { id: 11, localFitness: 1 },
      { id: 22, localFitness: 2 },
    ];
    const evaluatePopulation = createNeatParallelPopulationEvaluator({
      parallel: true,
      evaluateGenome: async (genome: MockGenome) => genome.localFitness,
      evaluateWithWorker: async (worker: MockWorker, genome: MockGenome) => {
        return genome.localFitness + worker.id;
      },
      workerPool: {
        evaluateOrderedBatch: jest.fn(async () => [99]),
      } as never,
    });

    await expect(evaluatePopulation(population)).rejects.toThrow(
      'Expected NEAT population evaluation result for genome 1.',
    );
  });

  it('throws when non-numeric results are returned without an assignment override', async () => {
    const population: MockGenome[] = [{ id: 11, localFitness: 1 }];
    const evaluatePopulation = createNeatParallelPopulationEvaluator({
      parallel: false,
      evaluateGenome: async () => ({ fitness: 11 }),
    });

    await expect(evaluatePopulation(population)).rejects.toThrow(
      'createNeatParallelPopulationEvaluator requires assignResult when genome 0 returns a non-numeric result.',
    );
  });
});
