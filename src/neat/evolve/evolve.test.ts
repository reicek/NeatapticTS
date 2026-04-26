import Neat from '../../neat';

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
    });
  });
});
