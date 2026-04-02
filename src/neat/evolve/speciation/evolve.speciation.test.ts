import { applyGlobalStagnationInjectionIfNeeded } from './evolve.speciation.utils';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
} from '../evolve.types';

function createGenome(genomeId: number): GenomeWithMetadata {
  return { _id: genomeId } as GenomeWithMetadata;
}

describe('neat evolve speciation chapter', () => {
  describe('applyGlobalStagnationInjectionIfNeeded', () => {
    describe('given a run that exceeded the configured global stagnation window', () => {
      it('replaces only the bounded non-elite tail and resets the improvement generation', async () => {
        // Arrange
        const evolutionHost = {
          generation: 6,
          population: [
            createGenome(1),
            createGenome(2),
            createGenome(3),
            createGenome(4),
          ],
          options: {
            elitism: 1,
            globalStagnationGenerations: 2,
          },
          _lastGlobalImproveGeneration: 3,
        } as NeatControllerForEvolution;
        let nextFreshGenomeId = 100;

        // Act
        await applyGlobalStagnationInjectionIfNeeded(evolutionHost, {
          buildFreshGenomeForStagnation: async () =>
            createGenome(nextFreshGenomeId++),
          replaceFraction: 0.5,
        });

        // Assert
        expect({
          populationIds: evolutionHost.population.map((genome) => genome._id),
          lastGlobalImproveGeneration:
            evolutionHost._lastGlobalImproveGeneration,
        }).toEqual({
          populationIds: [1, 2, 100, 101],
          lastGlobalImproveGeneration: 6,
        });
      });
    });
  });
});
