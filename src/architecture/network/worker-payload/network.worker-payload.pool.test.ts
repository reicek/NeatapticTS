import { ParallelInferencePool } from './network.worker-payload';

interface MockPayload {
  id: number;
}

interface MockWorker {
  id: number;
  release: jest.Mock<Promise<void>, []>;
}

describe('ParallelInferencePool', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('returns an empty result shelf without opening workers for empty batches', async () => {
    const openWorker = jest.fn(async (payload: MockPayload) => ({
      id: payload.id,
      release: jest.fn(async () => undefined),
    }));
    const pool = new ParallelInferencePool<MockPayload, MockWorker>({
      openWorker,
      workerCount: 2,
    });

    const results = await pool.evaluateOrderedBatch([], async (worker, payload) => {
      return `${String(payload.id)}:${String(worker.id)}`;
    });
    await pool.dispose();

    expect({ openWorkerCallCount: openWorker.mock.calls.length, results }).toEqual({
      openWorkerCallCount: 0,
      results: [],
    });
  });

  it('keeps worker usage bounded while preserving caller result order', async () => {
    const payloads: MockPayload[] = [{ id: 11 }, { id: 22 }, { id: 33 }];
    const deferredByPayloadId = new Map<number, () => void>();
    let activeWorkerCount = 0;
    let peakWorkerCount = 0;

    const pool = new ParallelInferencePool<MockPayload, MockWorker>({
      openWorker: async (payload) => {
        activeWorkerCount += 1;
        peakWorkerCount = Math.max(peakWorkerCount, activeWorkerCount);

        return {
          id: payload.id,
          release: jest.fn(async () => {
            activeWorkerCount -= 1;
          }),
        };
      },
      workerCount: 2,
    });

    await pool.initialize(payloads);

    const evaluationPromise = pool.evaluateOrderedBatch(
      payloads,
      async (worker, payload) => {
        await new Promise<void>((resolve) => {
          deferredByPayloadId.set(payload.id, resolve);
        });

        return `${String(payload.id)}:${String(worker.id)}`;
      },
    );

    await waitForCondition(() => deferredByPayloadId.has(11));
    await waitForCondition(() => deferredByPayloadId.has(22));
    deferredByPayloadId.get(11)?.();
    await waitForCondition(() => deferredByPayloadId.has(33));
    deferredByPayloadId.get(22)?.();
    deferredByPayloadId.get(33)?.();

    const results = await evaluationPromise;
    await pool.dispose();

    expect({ peakWorkerCount, results }).toEqual({
      peakWorkerCount: 2,
      results: ['11:11', '22:22', '33:33'],
    });
  });

  it('reuses warm workers across repeated batches for the same payload shelf', async () => {
    const payloads: MockPayload[] = [{ id: 11 }, { id: 22 }];
    const openWorker = jest.fn(async (payload: MockPayload) => ({
      id: payload.id,
      release: jest.fn(async () => undefined),
    }));
    const pool = new ParallelInferencePool<MockPayload, MockWorker>({
      openWorker,
      workerCount: 2,
    });

    await pool.initialize(payloads);
    const firstResults = await pool.evaluateOrderedBatch(
      payloads,
      async (worker, payload) => `${String(payload.id)}:${String(worker.id)}`,
    );
    const secondResults = await pool.evaluateOrderedBatch(
      payloads,
      async (worker, payload) => `${String(payload.id)}:${String(worker.id)}`,
    );
    await pool.dispose();

    expect({
      firstResults,
      openWorkerCallCount: openWorker.mock.calls.length,
      secondResults,
    }).toEqual({
      firstResults: ['11:11', '22:22'],
      openWorkerCallCount: 2,
      secondResults: ['11:11', '22:22'],
    });
  });

  it('releases every active worker during pool disposal', async () => {
    const payloads: MockPayload[] = [{ id: 11 }, { id: 22 }];
    const releaseCalls: number[] = [];
    const pool = new ParallelInferencePool<MockPayload, MockWorker>({
      openWorker: async (payload) => ({
        id: payload.id,
        release: jest.fn(async () => {
          releaseCalls.push(payload.id);
        }),
      }),
      workerCount: 2,
    });

    await pool.initialize(payloads);
    await pool.evaluateOrderedBatch(
      payloads,
      async (worker, payload) => `${String(payload.id)}:${String(worker.id)}`,
    );
    await pool.dispose();

    expect(releaseCalls).toEqual([11, 22]);
  });

  it('lazily initializes slots when evaluation starts before explicit initialization', async () => {
    const restoreNavigator = replaceNavigator({ hardwareConcurrency: 1 });
    const openWorker = jest.fn(async (payload: MockPayload) => ({
      id: payload.id,
      release: jest.fn(async () => undefined),
    }));
    const pool = new ParallelInferencePool<MockPayload, MockWorker>({
      openWorker,
    });

    const results = await pool.evaluateOrderedBatch(
      [{ id: 11 }],
      async (worker, payload) => `${String(payload.id)}:${String(worker.id)}`,
    );
    await pool.dispose();
    restoreNavigator();

    expect({ openWorkerCallCount: openWorker.mock.calls.length, results }).toEqual({
      openWorkerCallCount: 1,
      results: ['11:11'],
    });
  });

  it('skips sparse payload entries while preserving defined results', async () => {
    const sparsePayloads = new Array<MockPayload | undefined>(2);
    sparsePayloads[1] = { id: 22 };
    const openWorker = jest.fn(async (payload: MockPayload | undefined) => ({
      id: payload?.id ?? -1,
      release: jest.fn(async () => undefined),
    }));
    const pool = new ParallelInferencePool<MockPayload | undefined, MockWorker>({
      openWorker,
      workerCount: 2,
    });

    const results = await pool.evaluateOrderedBatch(
      sparsePayloads,
      async (worker, payload) => `${String(payload?.id)}:${String(worker.id)}`,
    );
    await pool.dispose();

    expect({ openWorkerCallCount: openWorker.mock.calls.length, results }).toEqual({
      openWorkerCallCount: 1,
      results: [undefined, '22:22'],
    });
  });

  it('falls back to the built-in worker count when navigator concurrency is unavailable', async () => {
    const restoreNavigator = replaceNavigator(undefined);
    const payloads: MockPayload[] = [
      { id: 11 },
      { id: 22 },
      { id: 33 },
      { id: 44 },
      { id: 55 },
    ];
    const deferredByPayloadId = new Map<number, () => void>();
    let activeWorkerCount = 0;
    let peakWorkerCount = 0;
    const pool = new ParallelInferencePool<MockPayload, MockWorker>({
      openWorker: async (payload) => {
        activeWorkerCount += 1;
        peakWorkerCount = Math.max(peakWorkerCount, activeWorkerCount);

        return {
          id: payload.id,
          release: jest.fn(async () => {
            activeWorkerCount -= 1;
          }),
        };
      },
    });

    const evaluationPromise = pool.evaluateOrderedBatch(payloads, async (worker, payload) => {
      await new Promise<void>((resolve) => {
        deferredByPayloadId.set(payload.id, resolve);
      });

      return `${String(payload.id)}:${String(worker.id)}`;
    });

    await waitForCondition(() => deferredByPayloadId.size === 4);
    deferredByPayloadId.get(11)?.();
    await waitForCondition(() => deferredByPayloadId.has(55));
    deferredByPayloadId.get(22)?.();
    deferredByPayloadId.get(33)?.();
    deferredByPayloadId.get(44)?.();
    deferredByPayloadId.get(55)?.();

    const results = await evaluationPromise;
    await pool.dispose();
    restoreNavigator();

    expect({ peakWorkerCount, results }).toEqual({
      peakWorkerCount: 4,
      results: ['11:11', '22:22', '33:33', '44:44', '55:55'],
    });
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

  throw new Error('Timed out while waiting for the pool test condition.');
}

function replaceNavigator(
  value: { hardwareConcurrency?: number } | undefined,
): () => void {
  const originalDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');

  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value,
    writable: true,
  });

  return () => {
    if (originalDescriptor) {
      Object.defineProperty(globalThis, 'navigator', originalDescriptor);
      return;
    }

    delete (globalThis as { navigator?: { hardwareConcurrency?: number } }).navigator;
  };
}