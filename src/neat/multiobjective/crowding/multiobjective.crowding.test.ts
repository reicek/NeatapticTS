import Network from '../../../architecture/network';
import {
  accumulateCrowdingForObjective,
  assignCrowdingDistances,
  markBoundaryCrowding,
  resolveGenomeIndex,
} from './multiobjective.crowding';
import { MultiobjectiveCrowdingGenomeIndexResolutionError } from './multiobjective.crowding.errors';

describe('multiobjective crowding chapter', () => {
  describe('resolveGenomeIndex', () => {
    describe('given a genome reference that is not present in the index map', () => {
      it('throws the crowding genome index resolution error', () => {
        // Arrange
        const indexedGenome = new Network(1, 1, { seed: 1001 });
        const missingGenome = new Network(1, 1, { seed: 1002 });
        const genomeIndexByReference = new Map<Network, number>([
          [indexedGenome, 0],
        ]);

        // Act
        const resolveMissingGenome = () =>
          resolveGenomeIndex(genomeIndexByReference, missingGenome);

        // Assert
        expect(resolveMissingGenome).toThrow(
          MultiobjectiveCrowdingGenomeIndexResolutionError,
        );
      });
    });
  });

  describe('markBoundaryCrowding()', () => {
    describe('given an empty sorted front', () => {
      it('returns early without throwing (line 165 branch 0 — !firstGenome)', () => {
        // Act + Assert: empty front → !firstGenome → early return
        expect(() => markBoundaryCrowding([])).not.toThrow();
      });
    });
  });

  describe('accumulateCrowdingForObjective()', () => {
    describe('given an empty sorted front', () => {
      it('returns early when resolveBoundaryGenomes returns null (lines 224+254 branch 0)', () => {
        // Arrange: empty front → resolveBoundaryGenomes → !firstGenome → return null
        const emptyGenomeIndex = new Map<Network, number>();

        // Act + Assert: must not throw
        expect(() =>
          accumulateCrowdingForObjective([], [[]], emptyGenomeIndex, 0),
        ).not.toThrow();
      });
    });
  });

  describe('assignCrowdingDistances()', () => {
    describe('given fronts containing an empty front', () => {
      it('skips the empty front without throwing (line 441 branch 0 — shouldSkipCrowdingFront)', () => {
        // Arrange: one empty front → shouldSkipCrowdingFront([]) = true → continue
        const population: Network[] = [];
        // Act + Assert: must not throw even though a front is empty
        expect(() =>
          assignCrowdingDistances([[]], [], [], population),
        ).not.toThrow();
      });
    });
  });
});
