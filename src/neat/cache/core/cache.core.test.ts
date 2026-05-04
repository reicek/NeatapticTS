import { GENOME_CACHE_FIELD_KEYS } from './cache.constants';
import { invalidateGenomeCaches } from './cache.core';

describe('neat cache core chapter', () => {
  describe('invalidateGenomeCaches', () => {
    describe('given the candidate is not an object', () => {
      it('returns without throwing', () => {
        // Arrange
        const genomeCandidate = undefined;

        // Act
        const invalidateNonObjectCandidate = () => {
          invalidateGenomeCaches(genomeCandidate);
        };

        // Assert
        expect(invalidateNonObjectCandidate).not.toThrow();
      });
    });

    describe('given the candidate carries genome cache fields', () => {
      it('deletes every configured cache field while preserving other data', () => {
        // Arrange
        const genomeCandidate = {
          _compatCache: { distance: 0.42 },
          _outputCache: [1, 0],
          _traceCache: { steps: 3 },
          persistentField: 'keep-me',
        };

        // Act
        invalidateGenomeCaches(genomeCandidate);

        // Assert
        expect({
          removedAllCacheFields: GENOME_CACHE_FIELD_KEYS.every(
            (cacheFieldKey) => !(cacheFieldKey in genomeCandidate),
          ),
          persistentField: genomeCandidate.persistentField,
        }).toEqual({
          removedAllCacheFields: true,
          persistentField: 'keep-me',
        });
      });
    });
  });
});
