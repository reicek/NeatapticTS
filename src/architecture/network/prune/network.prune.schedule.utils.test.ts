import Connection from '../../connection';
import Node from '../../node/node';
import {
  buildPruneSelection,
  resolvePruningMethod,
  shouldRunScheduledPrune,
} from './network.prune.schedule.utils';

describe('network prune schedule utility chapter', () => {
  describe('resolvePruningMethod()', () => {
    describe('given the pruning method is omitted', () => {
      it('falls back to magnitude pruning', () => {
        // Act
        const resolvedMethod = resolvePruningMethod(undefined);

        // Assert
        expect(resolvedMethod).toBe('magnitude');
      });
    });
  });

  describe('shouldRunScheduledPrune()', () => {
    describe('given the schedule frequency is missing or invalid', () => {
      it('uses the default cadence when the configured frequency is zero', () => {
        // Act
        const shouldRunNow = shouldRunScheduledPrune(1, {
          end: 3,
          frequency: 0,
          lastPruneIter: undefined,
          method: 'magnitude',
          regrowFraction: 0,
          start: 1,
          targetSparsity: 0.5,
        });

        // Assert
        expect(shouldRunNow).toBe(true);
      });
    });

    describe('given the current iteration was already pruned', () => {
      it('skips duplicate pruning for the same iteration', () => {
        // Act
        const shouldRunNow = shouldRunScheduledPrune(2, {
          end: 5,
          frequency: 1,
          lastPruneIter: 2,
          method: 'magnitude',
          regrowFraction: 0,
          start: 0,
          targetSparsity: 0.5,
        });

        // Assert
        expect(shouldRunNow).toBe(false);
      });
    });
  });

  describe('buildPruneSelection()', () => {
    describe('given SNIP ranking has no gradient statistics', () => {
      it('falls back to magnitude ordering', () => {
        // Arrange
        const sourceNode = new Node('hidden');
        const targetNode = new Node('hidden');
        const lowMagnitudeConnection = new Connection(
          sourceNode,
          targetNode,
          0.1,
        );
        const highMagnitudeConnection = new Connection(
          new Node('hidden'),
          new Node('hidden'),
          0.4,
        );

        // Act
        const selectedConnections = buildPruneSelection({
          connections: [highMagnitudeConnection, lowMagnitudeConnection],
          method: 'snip',
          removalCount: 1,
        }).connectionsToPrune;

        // Assert
        expect(selectedConnections[0]).toBe(lowMagnitudeConnection);
      });
    });

    describe('given SNIP ranking has gradient statistics', () => {
      it('sorts by weighted saliency instead of raw magnitude', () => {
        // Arrange
        const sourceNode = new Node('hidden');
        const targetNode = new Node('hidden');
        const lowerSaliencyConnection = new Connection(
          sourceNode,
          targetNode,
          0.4,
        );
        lowerSaliencyConnection.totalDeltaWeight = 1;

        const higherSaliencyConnection = new Connection(
          new Node('hidden'),
          new Node('hidden'),
          0.1,
        );
        higherSaliencyConnection.totalDeltaWeight = 5;

        // Act
        const selectedConnections = buildPruneSelection({
          connections: [higherSaliencyConnection, lowerSaliencyConnection],
          method: 'snip',
          removalCount: 1,
        }).connectionsToPrune;

        // Assert
        expect(selectedConnections[0]).toBe(lowerSaliencyConnection);
      });
    });
  });
});