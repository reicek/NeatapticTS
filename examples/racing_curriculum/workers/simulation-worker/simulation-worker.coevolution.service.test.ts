/**
 * Sibling smoke tests for `simulation-worker.coevolution.service.ts`.
 *
 * Full contract tests live in `simulation-worker.coevolution.test.ts`
 * (authored in Step 03).  This file satisfies the folder quality gate's
 * sibling-test-file requirement.
 *
 * Single-expect rule enforced throughout.
 */
import {
  createCoevolutionContainer,
} from './simulation-worker.coevolution.service';

describe('simulation-worker.coevolution.service module exports', () => {
  describe('createCoevolutionContainer', () => {
    it('is exported as a function', () => {
      expect(typeof createCoevolutionContainer).toBe('function');
    });

    it('returns an object with a teamA container', () => {
      const container = createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 0,
        tier: 1,
      });

      expect(typeof container.teamA).toBe('object');
    });

    it('returns an object with a teamB container', () => {
      const container = createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 0,
        tier: 1,
      });

      expect(typeof container.teamB).toBe('object');
    });

    it('returns an object with a resolveTeamFitness function', () => {
      const container = createCoevolutionContainer({
        populationSize: 10,
        rngSeed: 0,
        tier: 1,
      });

      expect(typeof container.resolveTeamFitness).toBe('function');
    });
  });
});
