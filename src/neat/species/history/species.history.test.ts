import type { SpeciesHistoryEntry } from '../../shared/neat.shared.types';
import { exportSpeciesHistoryJsonl } from './species.history';

describe('species history helpers', () => {
  describe('exportSpeciesHistoryJsonl()', () => {
    describe('given the export window is omitted', () => {
      describe('when history rows are serialized directly', () => {
        it('uses the default recent-history window', () => {
          // Arrange
          const speciesHistory = [
            { generation: 1 },
            { generation: 2 },
          ] as SpeciesHistoryEntry[];

          // Act
          const jsonl = exportSpeciesHistoryJsonl(speciesHistory);

          // Assert
          expect(jsonl).toBe('{"generation":1}\n{"generation":2}');
        });
      });
    });
  });
});
