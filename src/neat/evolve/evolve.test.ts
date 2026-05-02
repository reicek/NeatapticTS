import type Network from '../../architecture/network/network';
import * as methods from '../../methods/methods';
import Neat from '../../neat';
import { createGenomeFromNetwork } from '../genome/genome';

const XOR_DATASET = [
  { expectedOutput: 0, inputValues: [0, 0] },
  { expectedOutput: 1, inputValues: [0, 1] },
  { expectedOutput: 1, inputValues: [1, 0] },
  { expectedOutput: 0, inputValues: [1, 1] },
] as const;

function scoreXorFitness(network: Network): number {
  return XOR_DATASET.reduce((score, xorSample) => {
    const outputValue = network.activate([...xorSample.inputValues])[0] ?? 0;
    return score + (1 - Math.abs(xorSample.expectedOutput - outputValue));
  }, 0);
}

describe('neat evolve root chapter', () => {
  describe('evolve', () => {
    describe('given the current population still has missing scores', () => {
      it('forces evaluation before the generation advances', async () => {
        // Arrange
        const evolutionController = new Neat(2, 1, () => 1, {
          popsize: 2,
          seed: 701,
        });
        evolutionController.population[1].score = undefined;
        const evaluateSpy = jest.spyOn(evolutionController, 'evaluate');

        // Act
        await evolutionController.evolve();

        // Assert
        expect(evaluateSpy).toHaveBeenCalled();
      });
    });

    describe('given the current population is already scored', () => {
      it('increments the generation after rebuilding the next population', async () => {
        // Arrange
        const evolutionController = new Neat(2, 1, () => 1, {
          popsize: 2,
          seed: 702,
        });
        evolutionController.population[0].score = 1;
        evolutionController.population[1].score = 2;

        // Act
        await evolutionController.evolve();

        // Assert
        expect(evolutionController.generation).toBe(1);
      });

      it('builds fresh genomes when global stagnation exceeds the configured threshold', async () => {
        // Arrange
        const evolutionController = new Neat(2, 1, () => 1, {
          popsize: 2,
          seed: 703,
          elitism: 0,
          globalStagnationGenerations: 1,
        });
        evolutionController.population[0].score = 1;
        evolutionController.population[1].score = 2;
        (
          evolutionController as unknown as {
            _lastGlobalImproveGeneration?: number;
            _bestGlobalScore: number;
          }
        )._lastGlobalImproveGeneration = -1;
        (
          evolutionController as unknown as {
            _lastGlobalImproveGeneration?: number;
            _bestGlobalScore: number;
            _bestScoreLastGen?: number;
          }
        )._bestGlobalScore = Number.POSITIVE_INFINITY;
        (
          evolutionController as unknown as {
            _lastGlobalImproveGeneration?: number;
            _bestGlobalScore: number;
            _bestScoreLastGen?: number;
          }
        )._bestScoreLastGen = Number.POSITIVE_INFINITY;

        // Act
        await evolutionController.evolve();

        // Assert
        expect(
          (
            evolutionController as unknown as {
              _lastGlobalImproveGeneration?: number;
            }
          )._lastGlobalImproveGeneration,
        ).toBe(1);
      });

      it('keeps the population strict-genome clean across early feed-forward XOR generations', async () => {
        // Arrange
        const evolutionController = new Neat(2, 1, scoreXorFitness, {
          elitism: 5,
          fastMode: true,
          mutation: methods.mutation.FFW,
          mutationAmount: 2,
          mutationRate: 0.8,
          popsize: 100,
          seed: 42,
        });
        let firstFailureGeneration: number | undefined;

        // Act
        for (
          let generationIndex = 0;
          generationIndex < 5 && firstFailureGeneration === undefined;
          generationIndex += 1
        ) {
          try {
            await evolutionController.evaluate();
            await evolutionController.evolve();

            const hasInvalidGenome = evolutionController.population.some(
              (genome) => {
                try {
                  createGenomeFromNetwork(genome);
                  return false;
                } catch {
                  return true;
                }
              },
            );

            if (hasInvalidGenome) {
              firstFailureGeneration = evolutionController.generation;
            }
          } catch {
            firstFailureGeneration = evolutionController.generation;
          }
        }

        // Assert
        expect(firstFailureGeneration).toBeUndefined();
      });
    });
  });
});
