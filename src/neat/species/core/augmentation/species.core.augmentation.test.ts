import type { SpeciesHistoryEntry } from '../../../shared/neat.shared.types';
import { backfillExtendedHistoryEntries } from './species.core.augmentation';

function createHistoryEntries(): SpeciesHistoryEntry[] {
  return [
    {
      generation: 0,
      stats: [
        {
          bestScore: 1,
          id: 7,
          lastImproved: 0,
          size: 2,
        },
      ],
    },
  ];
}

describe('species core augmentation chapter', () => {
  describe('backfillExtendedHistoryEntries', () => {
    describe('given the backfill context omits the live species registry', () => {
      describe('when one recorded row still lacks extended fields', () => {
        it('leaves the recorded row unchanged', () => {
          // Arrange
          const historyEntries = createHistoryEntries();
          const expectedHistoryEntries = createHistoryEntries();

          // Act
          backfillExtendedHistoryEntries(historyEntries, {});

          // Assert
          expect(historyEntries).toEqual(expectedHistoryEntries);
        });
      });
    });
  });
});