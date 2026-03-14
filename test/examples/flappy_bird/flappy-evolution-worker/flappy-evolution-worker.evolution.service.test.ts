import type { Neat } from '../../../../src/neataptic';
import type Network from '../../../../src/architecture/network';
import { evolveAndBuildGenerationReadyMessage } from './flappy-evolution-worker.evolution.service';

describe('evolveAndBuildGenerationReadyMessage', () => {
  it('includes the serialized generation population in stable playback order', async () => {
    const bestNetworkJson = { id: 'best-network' };
    const firstPopulationNetworkJson = { id: 'population-network-0' };
    const secondPopulationNetworkJson = { id: 'population-network-1' };
    const bestNetwork = createMockNetwork(bestNetworkJson, 42);
    const neatRuntime = {
      generation: 7,
      population: [
        createMockNetwork(firstPopulationNetworkJson),
        createMockNetwork(secondPopulationNetworkJson),
      ],
      evolve: jest.fn().mockResolvedValue(bestNetwork),
    } as unknown as Neat;

    const generationReadyMessage = await evolveAndBuildGenerationReadyMessage({
      neatRuntime,
      isStopped: () => false,
      warmStartGenerationZeroIfNeeded: async () => undefined,
      setCurrentPopulation: () => undefined,
    });

    expect(generationReadyMessage).toEqual({
      type: 'generation-ready',
      payload: {
        generation: 7,
        bestFitness: 42,
        bestNetworkJson,
        populationNetworksJson: [
          firstPopulationNetworkJson,
          secondPopulationNetworkJson,
        ],
      },
    });
  });
});

function createMockNetwork(
  jsonValue: Record<string, unknown>,
  score = 0,
): Network {
  return {
    score,
    toJSON: () => jsonValue,
  } as unknown as Network;
}