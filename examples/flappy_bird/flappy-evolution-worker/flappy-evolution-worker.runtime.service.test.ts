import type { WorkerInitMessage } from './flappy-evolution-worker.types';
import { createInitializedWorkerRuntime } from './flappy-evolution-worker.runtime.service';

describe('createInitializedWorkerRuntime', () => {
  it('pins the worker runtime to feed-forward growth', () => {
    const initPayload: WorkerInitMessage['payload'] = {
      populationSize: 8,
      elitismCount: 2,
      rngSeed: 12345,
    };
    const neatRuntime = createInitializedWorkerRuntime(initPayload);

    expect(neatRuntime.options.allowRecurrent).toBe(false);
  });
});