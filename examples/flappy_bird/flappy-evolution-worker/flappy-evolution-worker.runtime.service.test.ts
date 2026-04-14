import type { WorkerInitMessage } from './flappy-evolution-worker.types';
import { createInitializedWorkerRuntime } from './flappy-evolution-worker.runtime.service';

interface WorkerRuntimeWithSeedNetwork {
  options: {
    allowRecurrent?: boolean;
    network: {
      describeArchitecture: () => {
        hiddenLayerSizes: number[];
      };
      inputNodeIds: number[];
      outputNodeIds: number[];
    };
  };
}

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
      hiddenLayerSizes: neatRuntime.options.network.describeArchitecture()
        .hiddenLayerSizes,
      inputNodeIds: neatRuntime.options.network.inputNodeIds.length,
      outputNodeIds: neatRuntime.options.network.outputNodeIds.length,
    }).toEqual({
      hiddenLayerSizes: [16, 8, 4],
      inputNodeIds: 38,
      outputNodeIds: 2,
    });
  });
});