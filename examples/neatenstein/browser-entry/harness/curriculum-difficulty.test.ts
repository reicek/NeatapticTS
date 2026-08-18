/**
 * Red-phase contract tests for Step B4 item 8:
 * Curriculum-based respawn difficulty scaling.
 *
 * Scales enemy capability based on real player performance telemetry,
 * replacing the previous RNG-based difficulty metrics. Uses existing
 * arms-race.ts wave logic but replaces RNG metrics with real telemetry.
 *
 * All tests in this file MUST fail (RED) until the B4 implementation lands.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import * as CurriculumDifficultyModule from './curriculum-difficulty';

/** Access unknown (future) exports on the curriculum-difficulty module. */
const curriculumDifficulty = CurriculumDifficultyModule as Record<
  string,
  unknown
>;

/** Player performance telemetry fixture. */
interface PlayerPerformanceTelemetry {
  survivalTicks: number;
  damageDealt: number;
  damageTaken: number;
  kills: number;
  deaths: number;
}

describe('B4.8: Curriculum-based respawn difficulty', () => {
  describe('B4.8: difficulty computation', () => {
    it('exports computeCurriculumDifficulty as a function', () => {
      expect(typeof curriculumDifficulty.computeCurriculumDifficulty).toBe(
        'function',
      );
    });

    it('returns a difficulty value in [0, 1] for strong player performance', () => {
      const telemetry: PlayerPerformanceTelemetry = {
        survivalTicks: 500,
        damageDealt: 80,
        damageTaken: 10,
        kills: 5,
        deaths: 0,
      };
      const difficulty = (curriculumDifficulty.computeCurriculumDifficulty as (...a: unknown[]) => number)(
        telemetry,
      );
      expect(difficulty).toBeGreaterThanOrEqual(0);
      expect(difficulty).toBeLessThanOrEqual(1);
    });

    it('returns higher difficulty when the player is performing well', () => {
      const strongPlayer: PlayerPerformanceTelemetry = {
        survivalTicks: 500,
        damageDealt: 80,
        damageTaken: 10,
        kills: 5,
        deaths: 0,
      };
      const weakPlayer: PlayerPerformanceTelemetry = {
        survivalTicks: 50,
        damageDealt: 5,
        damageTaken: 80,
        kills: 0,
        deaths: 3,
      };
      const strongDifficulty = (curriculumDifficulty.computeCurriculumDifficulty as (...a: unknown[]) => number)(
        strongPlayer,
      );
      const weakDifficulty = (curriculumDifficulty.computeCurriculumDifficulty as (...a: unknown[]) => number)(
        weakPlayer,
      );
      expect(strongDifficulty).toBeGreaterThan(weakDifficulty);
    });
  });

  describe('B4.8: difficulty-to-capability mapping', () => {
    it('exports scaleEnemyCapability as a function', () => {
      expect(typeof curriculumDifficulty.scaleEnemyCapability).toBe('function');
    });

    it('scales enemy mutation sigma proportional to difficulty', () => {
      const baseSigma = 0.08;
      const highDifficulty = 0.9;
      const scaled = (curriculumDifficulty.scaleEnemyCapability as (...a: unknown[]) => number)(
        baseSigma,
        highDifficulty,
      );
      expect(scaled).toBeGreaterThan(baseSigma);
    });

    it('keeps enemy capability at baseline when difficulty is zero', () => {
      const baseSigma = 0.08;
      const scaled = (curriculumDifficulty.scaleEnemyCapability as (...a: unknown[]) => number)(
        baseSigma,
        0,
      );
      expect(scaled).toBeCloseTo(baseSigma, 5);
    });
  });

  describe('B4.8: wave difficulty progression', () => {
    it('exports computeWaveDifficulty as a function', () => {
      expect(typeof curriculumDifficulty.computeWaveDifficulty).toBe('function');
    });

    it('increases difficulty across waves when player survives consistently', () => {
      const wave1 = (curriculumDifficulty.computeWaveDifficulty as (...a: unknown[]) => number)(
        { wave: 1, playerSurvivalRate: 0.9 },
      );
      const wave5 = (curriculumDifficulty.computeWaveDifficulty as (...a: unknown[]) => number)(
        { wave: 5, playerSurvivalRate: 0.9 },
      );
      expect(wave5).toBeGreaterThan(wave1);
    });
  });
});