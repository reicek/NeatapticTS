import Network from '../../../architecture/network';
import { resolveGenomeIndex } from './multiobjective.crowding';
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
});
