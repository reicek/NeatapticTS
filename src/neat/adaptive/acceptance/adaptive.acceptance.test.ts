import { applyMinimalCriterionAdaptive } from './adaptive.acceptance';
import Network from '../../../architecture/network';
import Neat from '../../../neat';
import type { NeatLikeWithAdaptive } from '../core/adaptive.core.types';

type AcceptanceConfig = NonNullable<
  NeatLikeWithAdaptive['options']['minimalCriterionAdaptive']
>;

function createAcceptanceController(
  scoreSnapshot: number[],
  minimalCriterionAdaptive: AcceptanceConfig,
): NeatLikeWithAdaptive {
  return {
    options: { minimalCriterionAdaptive },
    population: scoreSnapshot.map((score) => ({ score })),
    input: 2,
    output: 1,
    generation: 0,
  };
}

describe('neat adaptive acceptance chapter', () => {
  describe('applyMinimalCriterionAdaptive', () => {
    describe('given a generation whose acceptance rate exceeds the configured target band', () => {
      it('raises the threshold and rejects genomes that still fall below the updated bar', () => {
        // Arrange
        const adaptiveController = createAcceptanceController(
          [0.05, 0.2, 0.3, 0.4],
          {
            enabled: true,
            initialThreshold: 0.1,
            targetAcceptance: 0.5,
            adjustRate: 0.5,
          },
        );

        // Act
        applyMinimalCriterionAdaptive.call(adaptiveController);

        // Assert
        expect({
          threshold: Number(adaptiveController._mcThreshold?.toFixed(2)),
          scores: adaptiveController.population.map((genome) => genome.score),
        }).toEqual({
          threshold: 0.15,
          scores: [0, 0.2, 0.3, 0.4],
        });
      });
    });

    describe('given evolve runs with adaptive minimal criterion enabled', () => {
      it('raises the persisted threshold above the configured starting value', async () => {
        // Arrange
        const fitness = (network: Network) => network.connections.length;
        const neatController = new Neat(3, 2, fitness, {
          popsize: 20,
          seed: 1_301,
          speciation: true,
          mutationRate: 1,
          mutationAmount: 1,
          minimalCriterionAdaptive: {
            enabled: true,
            initialThreshold: 0.1,
            targetAcceptance: 0.6,
            adjustRate: 0.2,
          },
        });

        // Act
        await neatController.evaluate();
        for (
          let generationIndex = 0;
          generationIndex < 4;
          generationIndex += 1
        ) {
          await neatController.evolve();
        }
        await neatController.evaluate();

        // Assert
        expect(
          Reflect.get(neatController as object, '_mcThreshold') as number,
        ).toBeGreaterThan(0.1);
      });
    });
  });
});
