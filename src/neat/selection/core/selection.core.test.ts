import {
  calculateTotalScore,
  ensurePopulationEvaluated,
  ensurePopulationSortedDescending,
  selectParentByStrategy,
} from './selection.core';
import type { NeatLikeWithSelection } from './selection.types';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function buildHost(
  overrides: Partial<NeatLikeWithSelection> = {},
): NeatLikeWithSelection {
  return {
    population: [{ score: 5 }, { score: 3 }],
    options: {},
    _getRNG: () => () => 0.5,
    sort: jest.fn(),
    ...overrides,
  } as unknown as NeatLikeWithSelection;
}

describe('selection core chapter', () => {
  describe('selectParentByStrategy()', () => {
    describe('given an unknown strategy name', () => {
      it('returns the first population member (line 216 default arm)', () => {
        // Arrange: unknown strategy name triggers the switch default arm
        const host = buildHost({
          options: { selection: { name: 'UNKNOWN_STRATEGY' } },
          population: [{ score: 42 }, { score: 1 }],
        });

        // Act
        const result = selectParentByStrategy(host);

        // Assert: first genome returned without sorting
        expect(result.score).toBe(42);
      });
    });
  });

  describe('ensurePopulationEvaluated()', () => {
    describe('given a population whose last genome already has a score', () => {
      it('does not call evaluate (false arm of isUnevaluated)', () => {
        // Arrange: score is defined → isUnevaluated = false
        let evaluateCalled = false;
        const host = buildHost({
          population: [{ score: 7 }],
          evaluate: () => {
            evaluateCalled = true;
          },
        } as unknown as Partial<NeatLikeWithSelection>);

        // Act
        ensurePopulationEvaluated(host);

        // Assert: evaluate was skipped
        expect(evaluateCalled).toBe(false);
      });
    });

    describe('given a population whose last genome has no score', () => {
      it('calls evaluate to produce scores (true arm of isUnevaluated)', () => {
        // Arrange: score is undefined → isUnevaluated = true → evaluate fires
        let evaluateCalled = false;
        const host = buildHost({
          population: [{ score: undefined }],
          evaluate: () => {
            evaluateCalled = true;
          },
        } as unknown as Partial<NeatLikeWithSelection>);

        // Act
        ensurePopulationEvaluated(host);

        // Assert: evaluate was triggered
        expect(evaluateCalled).toBe(true);
      });
    });
  });

  describe('ensurePopulationSortedDescending()', () => {
    describe('given a population already in descending score order', () => {
      it('skips sort when first score exceeds second (false arm of isOutOfOrder)', () => {
        // Arrange: first > second → isOutOfOrder = false
        const sortFn = jest.fn();
        const host = buildHost({
          population: [{ score: 8 }, { score: 3 }],
          sort: sortFn,
        });

        // Act
        ensurePopulationSortedDescending(host);

        // Assert: sort was not called
        expect(sortFn).not.toHaveBeenCalled();
      });
    });

    describe('given a population that is out of descending order', () => {
      it('calls sort to restore descending order (true arm of isOutOfOrder)', () => {
        // Arrange: first < second → isOutOfOrder = true
        const sortFn = jest.fn();
        const host = buildHost({
          population: [{ score: 2 }, { score: 9 }],
          sort: sortFn,
        });

        // Act
        ensurePopulationSortedDescending(host);

        // Assert: sort was called once
        expect(sortFn).toHaveBeenCalledTimes(1);
      });
    });

    describe('given a population where first genome has undefined score', () => {
      it('treats undefined as DEFAULT_SCORE for first genome (line 258 ?? fallback)', () => {
        // Arrange: first score = undefined → ?? DEFAULT_SCORE fires on line 258
        const sortFn = jest.fn();
        const host = buildHost({
          population: [{ score: undefined }, { score: 3 }],
          sort: sortFn,
        });

        // Act
        ensurePopulationSortedDescending(host);

        // Assert: sort called since DEFAULT_SCORE(0) < 3
        expect(sortFn).toHaveBeenCalledTimes(1);
      });
    });

    describe('given a population where second genome has undefined score', () => {
      it('treats undefined as DEFAULT_SCORE for second genome (line 259 ?? fallback)', () => {
        // Arrange: second score = undefined → ?? DEFAULT_SCORE fires on line 259
        const sortFn = jest.fn();
        const host = buildHost({
          population: [{ score: 5 }, { score: undefined }],
          sort: sortFn,
        });

        // Act
        ensurePopulationSortedDescending(host);

        // Assert: 5 > DEFAULT_SCORE(0) → isOutOfOrder = false → no sort
        expect(sortFn).not.toHaveBeenCalled();
      });
    });
  });

  describe('calculateTotalScore()', () => {
    describe('given a population with defined and undefined scores', () => {
      it('sums scores and treats undefined as zero (DEFAULT_SCORE fallback)', () => {
        // Arrange: one undefined score → uses DEFAULT_SCORE (0)
        const population = [{ score: 4 }, { score: undefined }, { score: 6 }];

        // Act
        const total = calculateTotalScore(population);

        // Assert: 4 + 0 + 6 = 10
        expect(total).toBe(10);
      });
    });
  });

  describe('POWER selection', () => {
    describe('given no power option configured', () => {
      it('uses DEFAULT_POWER as the bias exponent (line 307 ?? fallback)', () => {
        // Arrange: power omitted → ?? DEFAULT_POWER fires
        const host = buildHost({
          options: { selection: { name: 'POWER' } },
          population: [{ score: 5 }, { score: 3 }],
          _getRNG: () => () => 0.1,
        });

        // Act
        const result = selectParentByStrategy(host);

        // Assert: a genome is selected without error
        expect(result).toBeDefined();
      });
    });

    describe('given a population already sorted in descending order', () => {
      it('skips the sort in the power guard (line 403 false arm)', () => {
        // Arrange: population already sorted → ensurePopulationSortedDescendingForPower skips sort
        const sortFn = jest.fn();
        const host = buildHost({
          options: { selection: { name: 'POWER', power: 1 } },
          population: [{ score: 8 }, { score: 3 }],
          sort: sortFn,
          _getRNG: () => () => 0.0,
        });

        // Act
        selectParentByStrategy(host);

        // Assert: no sort called because population was already ordered
        expect(sortFn).not.toHaveBeenCalled();
      });
    });
  });

  describe('TOURNAMENT selection', () => {
    describe('given no size option configured', () => {
      it('uses DEFAULT_TOURNAMENT_SIZE as the bracket size (line 368 ?? fallback)', () => {
        // Arrange: size omitted → ?? DEFAULT_TOURNAMENT_SIZE fires
        const host = buildHost({
          options: { selection: { name: 'TOURNAMENT', probability: 1 } },
          population: [{ score: 5 }, { score: 3 }, { score: 1 }],
          _getRNG: () => () => 0.0,
        });

        // Act
        const result = selectParentByStrategy(host);

        // Assert: a genome is selected
        expect(result).toBeDefined();
      });
    });

    describe('given no probability option configured', () => {
      it('uses DEFAULT_TOURNAMENT_PROBABILITY as the win probability (line 544 ?? fallback)', () => {
        // Arrange: probability omitted → ?? DEFAULT_TOURNAMENT_PROBABILITY fires
        const host = buildHost({
          options: { selection: { name: 'TOURNAMENT', size: 2 } },
          population: [{ score: 5 }, { score: 3 }, { score: 1 }],
          _getRNG: () => () => 0.5,
        });

        // Act
        const result = selectParentByStrategy(host);

        // Assert: a genome is selected without error
        expect(result).toBeDefined();
      });
    });

    describe('given a population containing genomes with undefined scores', () => {
      it('uses DEFAULT_SCORE in the toSorted comparator (lines 380-381 ?? fallback)', () => {
        // Arrange: undefined-scored genome in pool → ?? DEFAULT_SCORE fires in toSorted
        const host = buildHost({
          options: {
            selection: { name: 'TOURNAMENT', size: 2, probability: 1 },
          },
          population: [{ score: undefined }, { score: 3 }, { score: 1 }],
          _getRNG: () => () => 0.0,
        });

        // Act
        const result = selectParentByStrategy(host);

        // Assert: tournament completes and returns a genome
        expect(result).toBeDefined();
      });
    });

    describe('given a tournament size larger than population with suppression enabled', () => {
      it('returns a random member instead of throwing (line 495 suppressed false arm)', () => {
        // Arrange: overflow + _suppressTournamentError=true → getRandomPopulationMember
        const host = buildHost({
          options: { selection: { name: 'TOURNAMENT', size: 10 } },
          population: [{ score: 5 }, { score: 3 }],
          _suppressTournamentError: true,
          _getRNG: () => () => 0,
        });

        // Act
        const result = selectParentByStrategy(host);

        // Assert: first member returned (index = floor(0 * 2) = 0)
        expect(result.score).toBe(5);
      });
    });
  });

  describe('FITNESS_PROPORTIONATE selection', () => {
    describe('given a genome with undefined score in the population', () => {
      it('treats undefined score as zero in the shifted roulette scan (line 435 ?? fallback)', () => {
        // Arrange: one genome has no score → DEFAULT_SCORE used in calculateFitnessTotals
        const host = buildHost({
          options: { selection: { name: 'FITNESS_PROPORTIONATE' } },
          population: [{ score: undefined }, { score: 4 }],
          _getRNG: () => () => 0.1,
        });

        // Act
        const result = selectParentByStrategy(host);

        // Assert: selection completes without error
        expect(result).toBeDefined();
      });
    });
  });
});
