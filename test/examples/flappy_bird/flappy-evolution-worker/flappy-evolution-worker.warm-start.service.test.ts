import type { Neat } from '../../../../src/neataptic';
import type Network from '../../../../src/architecture/network';
import {
  warmStartWorkerGenerationZeroIfNeeded,
  type WorkerWarmStartDependencies,
} from './flappy-evolution-worker.warm-start.service';

describe('warmStartWorkerGenerationZeroIfNeeded', () => {
  it('seeds generation-zero genomes from the rollout-optimized template', async () => {
    const optimizedTemplateNetwork = createMockNetwork({
      nodeBiases: [7],
      connectionWeights: [11],
    });
    const optimizeWarmStartTemplateNetwork = jest
      .fn<Network, [Network, number]>()
      .mockReturnValue(optimizedTemplateNetwork);
    const dependencies: WorkerWarmStartDependencies = {
      buildHeuristicPretrainSet: () => [
        { input: [0, 0], output: [1, 0] },
        { input: [1, 1], output: [0, 1] },
      ],
      optimizeWarmStartTemplateNetwork,
    };
    const firstGenome = createMockNetwork();
    const secondGenome = createMockNetwork();
    const neatController = {
      generation: 0,
      population: [firstGenome, secondGenome],
    } as unknown as Neat;

    await warmStartWorkerGenerationZeroIfNeeded(
      neatController,
      {
        workerInitSeed: 123,
        generationZeroWarmStartApplied: false,
      },
      dependencies,
    );

    expect(optimizeWarmStartTemplateNetwork).toHaveBeenCalledTimes(1);
    expect(firstGenome.nodes[0].bias).toBeGreaterThan(6.5);
    expect(firstGenome.connections[0].weight).toBeGreaterThan(10.5);
    expect(secondGenome.nodes[0].bias).toBeGreaterThan(6.5);
    expect(secondGenome.connections[0].weight).toBeGreaterThan(10.5);
    expect(firstGenome.score).toBeUndefined();
    expect(secondGenome.score).toBeUndefined();
  });
});

function createMockNetwork(options?: {
  nodeBiases?: number[];
  connectionWeights?: number[];
}): Network {
  const nodeBiases = [...(options?.nodeBiases ?? [0])];
  const connectionWeights = [...(options?.connectionWeights ?? [0])];

  return {
    nodes: nodeBiases.map((biasValue) => ({ bias: biasValue })),
    connections: connectionWeights.map((weightValue) => ({
      weight: weightValue,
    })),
    score: 5,
    clone() {
      return createMockNetwork({
        nodeBiases,
        connectionWeights,
      });
    },
    train: jest.fn(),
  } as unknown as Network;
}
