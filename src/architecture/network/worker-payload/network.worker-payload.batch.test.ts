import {
  evaluateInWorkers,
  ParallelInferencePool,
} from './network.worker-payload';

interface MockPayload {
  id: number;
}

interface MockWorker {
  id: number;
  release: jest.Mock<Promise<void>, []>;
}

describe('evaluateInWorkers', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('returns ordered results, stable task ids, and parallel mode when workers are available', async () => {
    const openWorker = jest.fn(async (payload: MockPayload) => ({
      id: payload.id,
      release: jest.fn(async () => undefined),
    }));

    const batchResult = await evaluateInWorkers<
      MockPayload,
      MockPayload,
      MockWorker,
      string
    >({
      inputs: [{ id: 11 }, { id: 22 }],
      openWorker,
      evaluateWithWorker: async (worker: MockWorker, input: MockPayload) =>
        `${String(input.id)}:${String(worker.id)}`,
    });

    expect({
      openWorkerCallCount: openWorker.mock.calls.length,
      mode: batchResult.mode,
      results: batchResult.results,
      taskIds: batchResult.taskIds,
    }).toEqual({
      openWorkerCallCount: 2,
      mode: 'parallel',
      results: ['11:11', '22:22'],
      taskIds: [0, 1],
    });
  });

  it('falls back to single-thread evaluation when no worker configuration exists', async () => {
    const batchResult = await evaluateInWorkers<
      MockPayload,
      MockPayload,
      MockWorker,
      string
    >({
      inputs: [{ id: 11 }, { id: 22 }],
      evaluateLocally: async (input: MockPayload) =>
        `${String(input.id)}:local`,
    });

    expect({
      mode: batchResult.mode,
      results: batchResult.results,
      taskIds: batchResult.taskIds,
    }).toEqual({
      mode: 'single-thread',
      results: ['11:local', '22:local'],
      taskIds: [0, 1],
    });
  });

  it('supports custom payload resolution when worker payloads differ from logical inputs', async () => {
    const openWorker = jest.fn(async (payload: MockPayload) => ({
      id: payload.id,
      release: jest.fn(async () => undefined),
    }));

    const batchResult = await evaluateInWorkers<
      number,
      MockPayload,
      MockWorker,
      string
    >({
      inputs: [11, 22],
      openWorker,
      resolvePayload: (input: number) => ({ id: input + 100 }),
      evaluateWithWorker: async (
        worker: MockWorker,
        input: number,
        _inputIndex: number,
        payload: MockPayload,
      ) => {
        return `${String(input)}:${String(payload.id)}:${String(worker.id)}`;
      },
    });

    expect({
      openWorkerCallCount: openWorker.mock.calls.length,
      results: batchResult.results,
    }).toEqual({
      openWorkerCallCount: 2,
      results: ['11:111:111', '22:122:122'],
    });
  });

  it('falls back to Date timing when performance.now is unavailable', async () => {
    const restorePerformance = replacePerformance(undefined);

    const batchResult = await evaluateInWorkers<
      MockPayload,
      MockPayload,
      MockWorker,
      string
    >({
      inputs: [{ id: 11 }],
      evaluateLocally: async (input: MockPayload) =>
        `${String(input.id)}:local`,
    });

    restorePerformance();

    expect({
      elapsedMsIsNumber: typeof batchResult.elapsedMs,
      mode: batchResult.mode,
      results: batchResult.results,
    }).toEqual({
      elapsedMsIsNumber: 'number',
      mode: 'single-thread',
      results: ['11:local'],
    });
  });

  it('reuses a provided pool without reopening workers across repeated calls', async () => {
    const inputs = [{ id: 11 }, { id: 22 }];
    const openWorker = jest.fn(async (payload: MockPayload) => ({
      id: payload.id,
      release: jest.fn(async () => undefined),
    }));
    const workerPool = new ParallelInferencePool<MockPayload, MockWorker>({
      openWorker,
      workerCount: 2,
    });

    const firstBatchResult = await evaluateInWorkers<
      MockPayload,
      MockPayload,
      MockWorker,
      string
    >({
      inputs,
      workerPool,
      evaluateWithWorker: async (worker: MockWorker, input: MockPayload) =>
        `${String(input.id)}:${String(worker.id)}`,
    });
    const secondBatchResult = await evaluateInWorkers<
      MockPayload,
      MockPayload,
      MockWorker,
      string
    >({
      inputs,
      workerPool,
      evaluateWithWorker: async (worker: MockWorker, input: MockPayload) =>
        `${String(input.id)}:${String(worker.id)}`,
    });
    await workerPool.dispose();

    expect({
      firstResults: firstBatchResult.results,
      openWorkerCallCount: openWorker.mock.calls.length,
      secondResults: secondBatchResult.results,
    }).toEqual({
      firstResults: ['11:11', '22:22'],
      openWorkerCallCount: 2,
      secondResults: ['11:11', '22:22'],
    });
  });

  it('throws when neither worker execution nor local fallback can run the batch', async () => {
    await expect(
      evaluateInWorkers<MockPayload, MockPayload, MockWorker, string>({
        inputs: [{ id: 11 }],
      }),
    ).rejects.toThrow(
      'evaluateInWorkers requires either worker execution or a local fallback evaluator.',
    );
  });
});

function replacePerformance(value: Performance | undefined): () => void {
  const originalDescriptor = Object.getOwnPropertyDescriptor(
    globalThis,
    'performance',
  );

  Object.defineProperty(globalThis, 'performance', {
    configurable: true,
    value,
    writable: true,
  });

  return () => {
    if (originalDescriptor) {
      Object.defineProperty(globalThis, 'performance', originalDescriptor);
      return;
    }

    delete (globalThis as { performance?: Performance }).performance;
  };
}
