import {
  createInnovationTracker,
  prepareInnovationTrackerForGeneration,
  prepareInnovationTrackerForMutation,
  recordConnectionInnovation,
  recordNodeSplitRecord,
  restoreInnovationTracker,
  serializeInnovationTracker,
} from './innovation-tracker';

describe('neat innovation-tracker boundary', () => {
  describe('prepareInnovationTrackerForGeneration', () => {
    describe('given one mutation window has already recorded structural reuse data', () => {
      it('clears the generation-local registries while preserving the next innovation cursor', () => {
        // Arrange
        const innovationTracker = createInnovationTracker();
        innovationTracker.nextInnovationId = 19;
        recordConnectionInnovation(innovationTracker, '2->3', 11);
        recordNodeSplitRecord(innovationTracker, 'splitConnectionInnovation:11', {
          newNodeGeneId: 7,
          inInnov: 11,
          outInnov: 12,
        });

        // Act
        prepareInnovationTrackerForGeneration(innovationTracker, 1);

        // Assert
        expect({
          activeGeneration: innovationTracker.activeGeneration,
          nextInnovationId: innovationTracker.nextInnovationId,
          connectionCount: innovationTracker.connectionInnovations.size,
          splitCount: innovationTracker.nodeSplitRecords.size,
        }).toEqual({
          activeGeneration: 1,
          nextInnovationId: 19,
          connectionCount: 0,
          splitCount: 0,
        });
      });
    });
  });

  describe('prepareInnovationTrackerForMutation', () => {
    describe('given a restored tracker is already mid-mutation for a later generation', () => {
      it('keeps the in-flight generation window instead of clearing it back to the controller generation', () => {
        // Arrange
        const innovationTracker = createInnovationTracker();
        innovationTracker.activeGeneration = 3;
        recordConnectionInnovation(innovationTracker, '2->3', 11);

        // Act
        prepareInnovationTrackerForMutation(innovationTracker, 2);

        // Assert
        expect({
          activeGeneration: innovationTracker.activeGeneration,
          connectionCount: innovationTracker.connectionInnovations.size,
        }).toEqual({
          activeGeneration: 3,
          connectionCount: 1,
        });
      });
    });
  });

  describe('serializeInnovationTracker', () => {
    describe('given the tracker contains an active generation window and reuse entries', () => {
      it('round-trips the tracker through checkpoint JSON without losing mutation-window state', () => {
        // Arrange
        const innovationTracker = createInnovationTracker();
        innovationTracker.activeGeneration = 4;
        innovationTracker.nextInnovationId = 23;
        recordConnectionInnovation(innovationTracker, '2->3', 11);
        recordNodeSplitRecord(innovationTracker, 'splitConnectionInnovation:11', {
          newNodeGeneId: 9,
          inInnov: 11,
          outInnov: 12,
        });

        // Act
        const restoredTracker = restoreInnovationTracker(
          serializeInnovationTracker(innovationTracker),
        );

        // Assert
        expect(serializeInnovationTracker(restoredTracker)).toEqual(
          serializeInnovationTracker(innovationTracker),
        );
      });
    });
  });
});