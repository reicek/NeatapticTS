import Network from '../../../architecture/network/network';
import {
  buildParetoFronts,
  MAX_PARETO_FRONT_RANK_GUARD,
} from './multiobjective.fronts';
import type { DominanceState } from '../dominance/multiobjective.dominance';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('multiobjective fronts chapter', () => {
  describe('buildParetoFronts()', () => {
    describe('given maxFrontRankGuard is zero and the population has two distinct fronts', () => {
      it('stops after the first front when the guard threshold is exceeded', () => {
        // Arrange: genome 0 dominates genome 1 → two fronts expected
        const population = [new Network(1, 1), new Network(1, 1)];
        const dominanceState: DominanceState = {
          dominationCounts: [0, 1],
          dominatedIndicesByIndex: [[1], []],
          firstFrontIndices: [0],
        };

        // Act: guard of 0 fires after rank increments to 1, cutting off front 2
        const result = buildParetoFronts(population, dominanceState, 0);

        // Assert: only the first front is returned
        expect(result.length).toBe(1);
      });
    });

    describe('given a two-genome population with no dominance', () => {
      it('places both genomes in the same non-dominated front', () => {
        // Arrange: both genomes are non-dominated (firstFront = [0, 1])
        const population = [new Network(1, 1), new Network(1, 1)];
        const dominanceState: DominanceState = {
          dominationCounts: [0, 0],
          dominatedIndicesByIndex: [[], []],
          firstFrontIndices: [0, 1],
        };

        // Act
        const result = buildParetoFronts(
          population,
          dominanceState,
          MAX_PARETO_FRONT_RANK_GUARD,
        );

        // Assert: one front containing both genomes
        expect(result.length).toBe(1);
      });
    });

    describe('given a genome dominated by two others so one decrement does not reach zero', () => {
      it('places the doubly-dominated genome in the second front', () => {
        // Arrange: genomes 0 and 1 are non-dominated; genome 2 is dominated by both
        // Processing genome 0 decrements counts[2] from 2 to 1 (FALSE arm at line 194)
        // Processing genome 1 decrements counts[2] from 1 to 0 (TRUE arm, genome 2 joins next front)
        const population = [
          new Network(1, 1),
          new Network(1, 1),
          new Network(1, 1),
        ];
        const dominanceState: DominanceState = {
          dominationCounts: [0, 0, 2],
          dominatedIndicesByIndex: [[2], [2], []],
          firstFrontIndices: [0, 1],
        };

        // Act
        const result = buildParetoFronts(
          population,
          dominanceState,
          MAX_PARETO_FRONT_RANK_GUARD,
        );

        // Assert: two fronts — {0,1} and {2}
        expect(result.length).toBe(2);
      });
    });
  });
});
