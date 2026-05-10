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
