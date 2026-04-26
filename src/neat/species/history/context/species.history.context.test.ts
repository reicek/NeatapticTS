import { resolveSpeciesHistoryContext } from './species.history.context';
import type {
  ConnectionLike,
  NeatOptions,
  SpeciesHistoryEntry,
  SpeciesLike,
} from '../../../shared/neat.shared.types';

type SpeciesHistoryContextHost = {
  options?: NeatOptions;
  _speciesHistory?: SpeciesHistoryEntry[];
  _species?: SpeciesLike[];
  _fallbackInnov?: (connection: ConnectionLike) => number;
};

describe('species history context', () => {
  describe('resolveSpeciesHistoryContext()', () => {
    describe('given the host has no stored species history buffer', () => {
      it('returns an empty history array', () => {
        // Arrange
        const speciesHistoryHost: SpeciesHistoryContextHost = {
          options: {},
        };

        // Act
        const resolvedContext = resolveSpeciesHistoryContext(
          speciesHistoryHost as never,
        );

        // Assert
        expect(resolvedContext.speciesHistory).toEqual([]);
      });
    });

    describe('given the host provides backfill context fields', () => {
      it('preserves live-species references for optional augmentation', () => {
        // Arrange
        const liveSpecies: SpeciesLike[] = [
          {
            id: 7,
            members: [],
            bestScore: 10,
            lastImproved: 2,
          },
        ];
        const speciesHistoryHost: SpeciesHistoryContextHost = {
          options: {},
          _species: liveSpecies,
        };

        // Act
        const resolvedContext = resolveSpeciesHistoryContext(
          speciesHistoryHost as never,
        );

        // Assert
        expect(resolvedContext.backfillContext._species).toBe(liveSpecies);
      });
    });
  });
});