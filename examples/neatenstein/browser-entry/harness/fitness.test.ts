import { describe, expect, it } from '@jest/globals';

import type * as Fitness from './fitness';
import type { EnemyEpisodeTelemetry } from './types';

/**
 * Contract tests for examples/neatenstein/browser-entry/harness/fitness.ts.
 *
 * Covers AC-305: the CombatQualitySignal formula, sign convention, default
 * weights, and the 800-3000 synapse/neuron parsimony band.
 * Covers AC-10.5e: composite navigation+combat fitness with per-step telemetry.
 */

interface FitnessModule {
  computeCombatQualitySignal: typeof Fitness.computeCombatQualitySignal;
  computeEnemyNavigationFitness: typeof Fitness.computeEnemyNavigationFitness;
  computeEnemyTeamFitness: typeof Fitness.computeEnemyTeamFitness;
  NEATENSTEIN_PARSIMONY_LOWER_BOUND: typeof Fitness.NEATENSTEIN_PARSIMONY_LOWER_BOUND;
  NEATENSTEIN_PARSIMONY_UPPER_BOUND: typeof Fitness.NEATENSTEIN_PARSIMONY_UPPER_BOUND;
}

/** Helper: build a minimal telemetry object for fitness tests. */
function makeTelemetry(
  overrides: Partial<EnemyEpisodeTelemetry> = {},
): EnemyEpisodeTelemetry {
  return {
    position: { x: 60, y: 60 },
    bfsDistances: [10, 10, 10],
    damageDealt: 0,
    enemiesSurvived: 1,
    cellsVisited: 1,
    stagnationTicks: 0,
    finalDistance: 10,
    ...overrides,
  };
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

  describe('AC-10.5e-001: computeEnemyNavigationFitness', () => {
    it('exports computeEnemyNavigationFitness as a function', async () => {
      const mod = (await import('./fitness.ts')) as Record<string, unknown>;
      expect(typeof mod.computeEnemyNavigationFitness).toBe('function');
    });

    it('rewards progress: decreasing BFS distance → positive fitness', async () => {
      const { computeEnemyNavigationFitness } =
        (await import('./fitness.ts')) as FitnessModule;
      const telemetry = makeTelemetry({
        bfsDistances: [20, 18, 16, 14, 12],
        finalDistance: 10,
        cellsVisited: 5,
        stagnationTicks: 0,
      });
      const fitness = computeEnemyNavigationFitness(telemetry);
      expect(fitness).toBeGreaterThan(0);
    });

    it('approaching enemy scores higher than retreating enemy', async () => {
      const { computeEnemyNavigationFitness } =
        (await import('./fitness.ts')) as FitnessModule;
      const approaching = makeTelemetry({
        bfsDistances: [20, 15, 10, 5],
        finalDistance: 2,
        cellsVisited: 4,
        stagnationTicks: 0,
      });
      const retreating = makeTelemetry({
        bfsDistances: [5, 10, 15, 20],
        finalDistance: 25,
        cellsVisited: 4,
        stagnationTicks: 0,
      });
      expect(computeEnemyNavigationFitness(approaching)).toBeGreaterThan(
        computeEnemyNavigationFitness(retreating),
      );
    });

    it('rewards exploration: more unique cells → higher fitness', async () => {
      const { computeEnemyNavigationFitness } =
        (await import('./fitness.ts')) as FitnessModule;
      const low = makeTelemetry({
        bfsDistances: [10, 10, 10],
        finalDistance: 10,
        cellsVisited: 2,
        stagnationTicks: 0,
      });
      const high = makeTelemetry({
        bfsDistances: [10, 10, 10],
        finalDistance: 10,
        cellsVisited: 10,
        stagnationTicks: 0,
      });
      expect(computeEnemyNavigationFitness(high)).toBeGreaterThan(
        computeEnemyNavigationFitness(low),
      );
    });

    it('penalizes stagnation above threshold (~80 ticks)', async () => {
      const { computeEnemyNavigationFitness } =
        (await import('./fitness.ts')) as FitnessModule;
      const lowStagnation = makeTelemetry({
        bfsDistances: [10, 10, 10],
        finalDistance: 10,
        cellsVisited: 1,
        stagnationTicks: 10,
      });
      const highStagnation = makeTelemetry({
        bfsDistances: [10, 10, 10],
        finalDistance: 10,
        cellsVisited: 1,
        stagnationTicks: 120,
      });
      expect(computeEnemyNavigationFitness(lowStagnation)).toBeGreaterThan(
        computeEnemyNavigationFitness(highStagnation),
      );
    });

    it('does not penalize stagnation below threshold', async () => {
      const { computeEnemyNavigationFitness } =
        (await import('./fitness.ts')) as FitnessModule;
      const belowThreshold = makeTelemetry({
        bfsDistances: [10, 10, 10],
        finalDistance: 10,
        cellsVisited: 1,
        stagnationTicks: 50,
      });
      const zeroStagnation = makeTelemetry({
        bfsDistances: [10, 10, 10],
        finalDistance: 10,
        cellsVisited: 1,
        stagnationTicks: 0,
      });
      // 50 < 80 threshold → no penalty applied for either
      expect(computeEnemyNavigationFitness(belowThreshold)).toBe(
        computeEnemyNavigationFitness(zeroStagnation),
      );
    });
  });

  describe('AC-10.5e-003: computeEnemyTeamFitness (composite)', () => {
    it('exports computeEnemyTeamFitness as a function', async () => {
      const mod = (await import('./fitness.ts')) as Record<string, unknown>;
      expect(typeof mod.computeEnemyTeamFitness).toBe('function');
    });

    it('accepts EnemyEpisodeTelemetry and returns a number', async () => {
      const { computeEnemyTeamFitness } =
        (await import('./fitness.ts')) as FitnessModule;
      const telemetry = makeTelemetry();
      const score = computeEnemyTeamFitness(telemetry);
      expect(typeof score).toBe('number');
    });

    it('returns higher fitness for an enemy that dealt more damage', async () => {
      const { computeEnemyTeamFitness } =
        (await import('./fitness.ts')) as FitnessModule;
      const low = makeTelemetry({ damageDealt: 0 });
      const high = makeTelemetry({ damageDealt: 100 });
      expect(computeEnemyTeamFitness(high)).toBeGreaterThan(
        computeEnemyTeamFitness(low),
      );
    });

    it('returns higher fitness for an enemy that approached the player', async () => {
      const { computeEnemyTeamFitness } =
        (await import('./fitness.ts')) as FitnessModule;
      const stuck = makeTelemetry({
        bfsDistances: [20, 20, 20, 20],
        finalDistance: 20,
        cellsVisited: 1,
        damageDealt: 0,
      });
      const approaching = makeTelemetry({
        bfsDistances: [20, 15, 10, 5],
        finalDistance: 2,
        cellsVisited: 4,
        damageDealt: 0,
      });
      expect(computeEnemyTeamFitness(approaching)).toBeGreaterThan(
        computeEnemyTeamFitness(stuck),
      );
    });

    it('honours custom navigation/combat weights from config', async () => {
      const { computeEnemyTeamFitness } =
        (await import('./fitness.ts')) as FitnessModule;
      const telemetry = makeTelemetry({
        damageDealt: 50,
        bfsDistances: [20, 15, 10, 5],
        finalDistance: 2,
        cellsVisited: 4,
      });
      const defaultScore = computeEnemyTeamFitness(telemetry);
      const navFocused = computeEnemyTeamFitness(telemetry, {
        navigationWeight: 10,
        combatWeight: 0,
      });
      const combatFocused = computeEnemyTeamFitness(telemetry, {
        navigationWeight: 0,
        combatWeight: 10,
      });
      expect(navFocused).not.toBe(defaultScore);
      expect(combatFocused).not.toBe(defaultScore);
    });

    it('returns zero when all weights are zero', async () => {
      const { computeEnemyTeamFitness } =
        (await import('./fitness.ts')) as FitnessModule;
      const telemetry = makeTelemetry({ damageDealt: 100 });
      const score = computeEnemyTeamFitness(telemetry, {
        navigationWeight: 0,
        combatWeight: 0,
        damageWeight: 0,
        survivalWeight: 0,
      });
      expect(score).toBe(0);
    });
  });
});
