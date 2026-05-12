import {
  Architect,
  exportTransferableInferencePayload,
  getTransferList,
  methods,
  type Neat,
} from '../../../src/neataptic';
import type Network from '../../../src/architecture/network';
import {
  evolveAndBuildGenerationReadyMessage,
  resolveGenerationReadyMessageTransferList,
} from './flappy-evolution-worker.evolution.service';

describe('evolveAndBuildGenerationReadyMessage', () => {
  it('publishes generation-zero startup population before the first expensive evolve pass', async () => {
    const firstPopulationNetwork = createConcreteNetwork(3, 0.1);
    const secondPopulationNetwork = createConcreteNetwork(7, 0.2);
    const setCurrentPopulation = jest.fn();
    const markStartupPopulationPublished = jest.fn();
    const warmStartGenerationZeroIfNeeded = jest.fn(async () => undefined);
    const neatRuntime = {
      generation: 0,
      population: [firstPopulationNetwork, secondPopulationNetwork],
      evolve: jest.fn().mockResolvedValue(secondPopulationNetwork),
    } as unknown as Neat;

    const generationReadyMessage = await evolveAndBuildGenerationReadyMessage({
      architectureProfileId: 'lstm',
      neatRuntime,
      isStopped: () => false,
      warmStartGenerationZeroIfNeeded,
      setCurrentPopulation,
      publishStartupPopulationBeforeFirstEvolution: true,
      markStartupPopulationPublished,
    });

    expect({
      bestFitness: generationReadyMessage.payload.bestFitness,
      didMarkPublished: markStartupPopulationPublished.mock.calls.length === 1,
      didSkipEvolve: (neatRuntime.evolve as jest.Mock).mock.calls.length === 0,
      didSkipWarmStart: warmStartGenerationZeroIfNeeded.mock.calls.length === 0,
      generation: generationReadyMessage.payload.generation,
      population: setCurrentPopulation.mock.calls[0]?.[0],
    }).toEqual({
      bestFitness: 7,
      didMarkPublished: true,
      didSkipEvolve: true,
      didSkipWarmStart: true,
      generation: 0,
      population: [firstPopulationNetwork, secondPopulationNetwork],
    });
  });

  it('includes transferable generation payloads in stable playback order alongside the visualization bridge', async () => {
    const bestNetwork = createConcreteNetwork(42, 0.3);
    const firstPopulationNetwork = createConcreteNetwork(0, 0.1);
    const secondPopulationNetwork = createConcreteNetwork(0, 0.2);
    setFirstNonInputActivation(bestNetwork, methods.Activation.swish);
    setFirstNonInputActivation(firstPopulationNetwork, methods.Activation.gelu);
    setFirstNonInputActivation(
      secondPopulationNetwork,
      methods.Activation.mish,
    );
    const neatRuntime = {
      generation: 7,
      population: [firstPopulationNetwork, secondPopulationNetwork],
      evolve: jest.fn().mockResolvedValue(bestNetwork),
    } as unknown as Neat;

    const generationReadyMessage = await evolveAndBuildGenerationReadyMessage({
      architectureProfileId: 'random-sparse',
      neatRuntime,
      isStopped: () => false,
      warmStartGenerationZeroIfNeeded: async () => undefined,
      setCurrentPopulation: () => undefined,
    });

    expect(generationReadyMessage).toEqual({
      type: 'generation-ready',
      payload: {
        architectureProfileId: 'random-sparse',
        generation: 7,
        bestFitness: 42,
        bestNetworkJson: bestNetwork.toJSON(),
        bestNetworkPayload: exportTransferableInferencePayload(bestNetwork),
        populationNetworksJson: [
          firstPopulationNetwork.toJSON(),
          secondPopulationNetwork.toJSON(),
        ],
        populationNetworkPayloads: [
          exportTransferableInferencePayload(firstPopulationNetwork),
          exportTransferableInferencePayload(secondPopulationNetwork),
        ],
      },
    });
  });

  it('collects the transferable buffers for the generation-ready payload in postMessage order', async () => {
    const bestNetwork = createConcreteNetwork(42, 0.3);
    const firstPopulationNetwork = createConcreteNetwork(0, 0.1);
    const secondPopulationNetwork = createConcreteNetwork(0, 0.2);
    const neatRuntime = {
      generation: 7,
      population: [firstPopulationNetwork, secondPopulationNetwork],
      evolve: jest.fn().mockResolvedValue(bestNetwork),
    } as unknown as Neat;

    const generationReadyMessage = await evolveAndBuildGenerationReadyMessage({
      architectureProfileId: 'random-sparse',
      neatRuntime,
      isStopped: () => false,
      warmStartGenerationZeroIfNeeded: async () => undefined,
      setCurrentPopulation: () => undefined,
    });

    expect(
      resolveGenerationReadyMessageTransferList(generationReadyMessage),
    ).toEqual([
      ...getTransferList(generationReadyMessage.payload.bestNetworkPayload!),
      ...generationReadyMessage.payload.populationNetworkPayloads!.flatMap(
        (populationNetworkPayload) => getTransferList(populationNetworkPayload),
      ),
    ]);
  });

  it('evolves normally when optional generation-zero warm-start fails', async () => {
    const bestNetwork = createConcreteNetwork(42, 0.3);
    const neatRuntime = {
      generation: 0,
      population: [bestNetwork],
      evolve: jest.fn().mockResolvedValue(bestNetwork),
    } as unknown as Neat;

    await evolveAndBuildGenerationReadyMessage({
      architectureProfileId: 'lstm',
      neatRuntime,
      isStopped: () => false,
      warmStartGenerationZeroIfNeeded: async () => {
        throw new Error('synthetic warm-start failure');
      },
      setCurrentPopulation: () => undefined,
    });

    expect((neatRuntime.evolve as jest.Mock).mock.calls.length).toBe(1);
  });
});

function createConcreteNetwork(score = 0, biasOffset = 0): Network {
  const network = Architect.perceptron(2, 2, 1);
  network.score = score;

  network.nodes.forEach((node, nodeIndex) => {
    node.bias += biasOffset + nodeIndex * 0.01;
  });

  return network;
}

function setFirstNonInputActivation(
  network: Network,
  activationFunction: typeof methods.Activation.swish,
): void {
  const targetNode = network.nodes.find((node) => node.type !== 'input');

  if (!targetNode) {
    throw new Error(
      'Expected a non-input node for the activation coverage test.',
    );
  }

  targetNode.squash = activationFunction;
}
