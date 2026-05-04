import Node from '../../node';
import {
  applyRestoredConnectionIdentity,
  hydrateNodeGeneIdWhenProvided,
} from './network.serialize.runtime.utils';

describe('network serialize runtime utilities chapter', () => {
  describe('hydrateNodeGeneIdWhenProvided', () => {
    describe('given the persisted gene id is not numeric', () => {
      describe('when node identity hydration runs', () => {
        it('keeps the existing runtime gene id unchanged', () => {
          // Arrange
          const runtimeNode = new Node('hidden');
          const existingGeneId = runtimeNode.geneId;

          // Act
          hydrateNodeGeneIdWhenProvided(runtimeNode, null);

          // Assert
          expect(runtimeNode.geneId).toBe(existingGeneId);
        });
      });
    });
  });

  describe('applyRestoredConnectionIdentity', () => {
    describe('given the created connection is missing', () => {
      describe('when persisted identity metadata is applied', () => {
        it('returns without throwing', () => {
          // Arrange
          let didThrow = false;

          // Act
          try {
            applyRestoredConnectionIdentity(undefined, {
              enabled: true,
              innovation: 41,
            });
          } catch {
            didThrow = true;
          }

          // Assert
          expect(didThrow).toBe(false);
        });
      });
    });
  });
});
