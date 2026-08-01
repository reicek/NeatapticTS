import { describe, expect, it } from '@jest/globals';

import type * as Fitness from './fitness';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/fitness.ts.
 *
 * Covers AC-305: the CombatQualitySignal formula, sign convention, default
 * weights, and the 800-3000 synapse/neuron parsimony band.
 */

interface FitnessModule {
  computeCombatQualitySignal: typeof Fitness.computeCombatQualitySignal;
  NEATENSTEIN_PARSIMONY_LOWER_BOUND: typeof Fitness.NEATENSTEIN_PARSIMONY_LOWER_BOUND;
  NEATENSTEIN_PARSIMONY_UPPER_BOUND: typeof Fitness.NEATENSTEIN_PARSIMONY_UPPER_BOUND;
}

describe('Neatenstein harness fitness', () => {
  describe('AC-305: CombatQualitySignal exports', () => {
    it('exports computeCombatQualitySignal as a function', async () => {
      const mod = (await import('./fitness.ts')) as Record<string, unknown>;
      expect(typeof mod.computeCombatQualitySignal).toBe('function');
    });

    it('exports parsimony lower bound equal to 800', async () => {
      const mod = (await import('./fitness.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_PARSIMONY_LOWER_BOUND as number).toBe(800);
    });

    it('exports parsimony upper bound equal to 3000', async () => {
      const mod = (await import('./fitness.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_PARSIMONY_UPPER_BOUND as number).toBe(3000);
    });
  });

  describe('AC-305: CombatQualitySignal sign convention', () => {
    it('returns higher fitness when survivalTicks increases', async () => {
      const { computeCombatQualitySignal } =
        (await import('./fitness.ts')) as FitnessModule;
      const low = computeCombatQualitySignal({
        survivalTicks: 10,
        damageDealt: 0,
        kills: 0,
        damageTaken: 0,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
      const high = computeCombatQualitySignal({
        survivalTicks: 100,
        damageDealt: 0,
        kills: 0,
        damageTaken: 0,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
      expect(high).toBeGreaterThan(low);
    });

    it('returns higher fitness when damageDealt increases', async () => {
      const { computeCombatQualitySignal } =
        (await import('./fitness.ts')) as FitnessModule;
      const low = computeCombatQualitySignal({
        survivalTicks: 0,
        damageDealt: 5,
        kills: 0,
        damageTaken: 0,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
      const high = computeCombatQualitySignal({
        survivalTicks: 0,
        damageDealt: 50,
        kills: 0,
        damageTaken: 0,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
      expect(high).toBeGreaterThan(low);
    });

    it('returns higher fitness when kills increase', async () => {
      const { computeCombatQualitySignal } =
        (await import('./fitness.ts')) as FitnessModule;
      const low = computeCombatQualitySignal({
        survivalTicks: 0,
        damageDealt: 0,
        kills: 0,
        damageTaken: 0,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
      const high = computeCombatQualitySignal({
        survivalTicks: 0,
        damageDealt: 0,
        kills: 3,
        damageTaken: 0,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
      expect(high).toBeGreaterThan(low);
    });

    it('returns lower fitness when damageTaken increases', async () => {
      const { computeCombatQualitySignal } =
        (await import('./fitness.ts')) as FitnessModule;
      const low = computeCombatQualitySignal({
        survivalTicks: 0,
        damageDealt: 0,
        kills: 0,
        damageTaken: 50,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
      const high = computeCombatQualitySignal({
        survivalTicks: 0,
        damageDealt: 0,
        kills: 0,
        damageTaken: 5,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
      expect(high).toBeGreaterThan(low);
    });

    it('returns lower fitness when aimMissRate increases', async () => {
      const { computeCombatQualitySignal } =
        (await import('./fitness.ts')) as FitnessModule;
      const good = computeCombatQualitySignal({
        survivalTicks: 0,
        damageDealt: 0,
        kills: 0,
        damageTaken: 0,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
      const bad = computeCombatQualitySignal({
        survivalTicks: 0,
        damageDealt: 0,
        kills: 0,
        damageTaken: 0,
        aimMissRate: 0.9,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
      expect(good).toBeGreaterThan(bad);
    });
  });

  describe('AC-305: parsimony band', () => {
    it('applies a penalty below the lower bound', async () => {
      const { computeCombatQualitySignal, NEATENSTEIN_PARSIMONY_LOWER_BOUND } =
        (await import('./fitness.ts')) as FitnessModule;
      const inside = computeCombatQualitySignal(
        {
          survivalTicks: 100,
          damageDealt: 0,
          kills: 0,
          damageTaken: 0,
          aimMissRate: 0,
          complexityBonus: 0,
          parsimonyDensityPenalty: 0,
        },
        NEATENSTEIN_PARSIMONY_LOWER_BOUND + 100,
      );
      const below = computeCombatQualitySignal(
        {
          survivalTicks: 100,
          damageDealt: 0,
          kills: 0,
          damageTaken: 0,
          aimMissRate: 0,
          complexityBonus: 0,
          parsimonyDensityPenalty: 0,
        },
        NEATENSTEIN_PARSIMONY_LOWER_BOUND - 100,
      );
      expect(below).toBeLessThan(inside);
    });

    it('applies a penalty above the upper bound', async () => {
      const { computeCombatQualitySignal, NEATENSTEIN_PARSIMONY_UPPER_BOUND } =
        (await import('./fitness.ts')) as FitnessModule;
      const inside = computeCombatQualitySignal(
        {
          survivalTicks: 100,
          damageDealt: 0,
          kills: 0,
          damageTaken: 0,
          aimMissRate: 0,
          complexityBonus: 0,
          parsimonyDensityPenalty: 0,
        },
        NEATENSTEIN_PARSIMONY_UPPER_BOUND - 100,
      );
      const above = computeCombatQualitySignal(
        {
          survivalTicks: 100,
          damageDealt: 0,
          kills: 0,
          damageTaken: 0,
          aimMissRate: 0,
          complexityBonus: 0,
          parsimonyDensityPenalty: 0,
        },
        NEATENSTEIN_PARSIMONY_UPPER_BOUND + 100,
      );
      expect(above).toBeLessThan(inside);
    });
  });

  describe('AC-503: team-level enemy fitness', () => {
    it('exports computeEnemyTeamFitness as a function', async () => {
      const mod = (await import('./fitness.ts')) as Record<string, unknown>;
      expect(typeof mod.computeEnemyTeamFitness).toBe('function');
    });

    it('returns a higher scalar when collective damage dealt increases', async () => {
      const { computeEnemyTeamFitness } =
        (await import('./fitness.ts')) as unknown as {
          computeEnemyTeamFitness: (
            damageDealt: number,
            enemiesSurvived: number,
            config?: Record<string, unknown>,
          ) => number;
        };
      const low = computeEnemyTeamFitness(10, 1);
      const high = computeEnemyTeamFitness(100, 1);
      expect(high).toBeGreaterThan(low);
    });

    it('returns a higher scalar when more enemies survive', async () => {
      const { computeEnemyTeamFitness } =
        (await import('./fitness.ts')) as unknown as {
          computeEnemyTeamFitness: (
            damageDealt: number,
            enemiesSurvived: number,
            config?: Record<string, unknown>,
          ) => number;
        };
      const low = computeEnemyTeamFitness(0, 2);
      const high = computeEnemyTeamFitness(0, 6);
      expect(high).toBeGreaterThan(low);
    });

    it('honours custom weights from the optional config', async () => {
      const { computeEnemyTeamFitness } =
        (await import('./fitness.ts')) as unknown as {
          computeEnemyTeamFitness: (
            damageDealt: number,
            enemiesSurvived: number,
            config?: { damageWeight?: number; survivalWeight?: number },
          ) => number;
        };
      const defaultScore = computeEnemyTeamFitness(10, 2);
      const weightedScore = computeEnemyTeamFitness(10, 2, {
        damageWeight: 10,
        survivalWeight: 0,
      });
      expect(weightedScore).not.toBe(defaultScore);
      expect(weightedScore).toBeGreaterThan(defaultScore);
    });
  });
});
