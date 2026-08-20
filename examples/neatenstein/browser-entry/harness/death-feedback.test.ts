import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for
 * examples/neatenstein/browser-entry/harness/death-feedback.ts.
 *
 * Covers:
 * - AC-601-S05-003: death-feedback.ts computes an adaptation signal (enemy
 *   behavior delta) from consecutive generations fed by replay entries.
 */

interface GenerationSnapshot {
  generation: number;
  enemyBehaviorMetrics: {
    aggression: number;
    movementPattern: number;
    positioning: number;
  };
}

interface AdaptationSignal {
  direction: 'stronger' | 'weaker' | 'shifted';
  aggressionDelta: number;
  movementDelta: number;
  positioningDelta: number;
}

interface EnemyBehaviorMetrics {
  aggression: number;
  movementPattern: number;
  positioning: number;
}

interface DeathFeedbackModule {
  computeAdaptationSignal: (
    prevGeneration: GenerationSnapshot,
    currGeneration: GenerationSnapshot,
  ) => AdaptationSignal;
  computeEnemyBehaviorMetrics: (telemetry: {
    damageDealt: number;
    survivalTicks: number;
    meanDistanceToPlayer: number;
    dirChangeCount: number;
    maxMapDistance: number;
  }) => EnemyBehaviorMetrics;
}

const prevSnapshot: GenerationSnapshot = {
  generation: 1,
  enemyBehaviorMetrics: {
    aggression: 0.3,
    movementPattern: 0.5,
    positioning: 0.2,
  },
};

const currSnapshot: GenerationSnapshot = {
  generation: 2,
  enemyBehaviorMetrics: {
    aggression: 0.7,
    movementPattern: 0.4,
    positioning: 0.6,
  },
};

describe('Neatenstein harness death feedback', () => {
  describe('AC-601-S05-003: adaptation signal from consecutive generations', () => {
    it('exports computeAdaptationSignal as a function', async () => {
      const mod =
        (await import('./death-feedback.ts')) as unknown as DeathFeedbackModule;

      expect(typeof mod.computeAdaptationSignal).toBe('function');
    });

    it('returns an adaptation signal with a direction field', async () => {
      const { computeAdaptationSignal } =
        (await import('./death-feedback.ts')) as unknown as DeathFeedbackModule;
      const signal = computeAdaptationSignal(prevSnapshot, currSnapshot);

      expect(signal.direction).toBeDefined();
    });

    it('computes aggression delta between consecutive generations', async () => {
      const { computeAdaptationSignal } =
        (await import('./death-feedback.ts')) as unknown as DeathFeedbackModule;
      const signal = computeAdaptationSignal(prevSnapshot, currSnapshot);

      expect(signal.aggressionDelta).toBeCloseTo(0.4, 5);
    });

    it('returns weaker direction when aggression decreases beyond threshold', async () => {
      const { computeAdaptationSignal } =
        (await import('./death-feedback.ts')) as unknown as DeathFeedbackModule;
      const prev: GenerationSnapshot = {
        generation: 1,
        enemyBehaviorMetrics: {
          aggression: 0.7,
          movementPattern: 0.5,
          positioning: 0.2,
        },
      };
      const curr: GenerationSnapshot = {
        generation: 2,
        enemyBehaviorMetrics: {
          aggression: 0.3,
          movementPattern: 0.4,
          positioning: 0.6,
        },
      };
      const signal = computeAdaptationSignal(prev, curr);

      expect(signal.direction).toBe('weaker');
      expect(signal.aggressionDelta).toBeCloseTo(-0.4, 5);
    });

    it('returns shifted direction when aggression change is within threshold', async () => {
      const { computeAdaptationSignal } =
        (await import('./death-feedback.ts')) as unknown as DeathFeedbackModule;
      const prev: GenerationSnapshot = {
        generation: 1,
        enemyBehaviorMetrics: {
          aggression: 0.5,
          movementPattern: 0.5,
          positioning: 0.2,
        },
      };
      const curr: GenerationSnapshot = {
        generation: 2,
        enemyBehaviorMetrics: {
          aggression: 0.55,
          movementPattern: 0.4,
          positioning: 0.6,
        },
      };
      const signal = computeAdaptationSignal(prev, curr);

      expect(signal.direction).toBe('shifted');
      expect(signal.aggressionDelta).toBeCloseTo(0.05, 5);
    });
  });

  describe('AC-A4-003: real telemetry behavior metrics', () => {
    it('exports computeEnemyBehaviorMetrics as a function', async () => {
      const mod =
        (await import('./death-feedback.ts')) as unknown as DeathFeedbackModule;

      expect(typeof mod.computeEnemyBehaviorMetrics).toBe('function');
    });

    it('computes aggression from damage dealt and survival ticks', async () => {
      const { computeEnemyBehaviorMetrics } =
        (await import('./death-feedback.ts')) as unknown as DeathFeedbackModule;
      const telemetry = {
        damageDealt: 10,
        survivalTicks: 50,
        meanDistanceToPlayer: 5,
        dirChangeCount: 3,
        maxMapDistance: 20,
      };
      const result = computeEnemyBehaviorMetrics(telemetry);
      expect(result.aggression).toBeCloseTo(10 / 15, 5);
    });

    it('computes positioning from mean distance', async () => {
      const { computeEnemyBehaviorMetrics } =
        (await import('./death-feedback.ts')) as unknown as DeathFeedbackModule;
      const telemetry = {
        damageDealt: 10,
        survivalTicks: 50,
        meanDistanceToPlayer: 5,
        dirChangeCount: 3,
        maxMapDistance: 20,
      };
      const result = computeEnemyBehaviorMetrics(telemetry);
      expect(result.positioning).toBeCloseTo(0.25, 5);
    });

    it('computes movementPattern from direction changes', async () => {
      const { computeEnemyBehaviorMetrics } =
        (await import('./death-feedback.ts')) as unknown as DeathFeedbackModule;
      const telemetry = {
        damageDealt: 10,
        survivalTicks: 50,
        meanDistanceToPlayer: 5,
        dirChangeCount: 3,
        maxMapDistance: 20,
      };
      const result = computeEnemyBehaviorMetrics(telemetry);
      expect(result.movementPattern).toBeCloseTo(0.06, 5);
    });

    it('clamps all metrics to [0, 1]', async () => {
      const { computeEnemyBehaviorMetrics } =
        (await import('./death-feedback.ts')) as unknown as DeathFeedbackModule;
      const telemetry = {
        damageDealt: 0,
        survivalTicks: 0,
        meanDistanceToPlayer: 0,
        dirChangeCount: 0,
        maxMapDistance: 1,
      };
      const result = computeEnemyBehaviorMetrics(telemetry);
      const clamped = (value: number) => value >= 0 && value <= 1;
      expect(clamped(result.aggression)).toBe(true);
      expect(clamped(result.positioning)).toBe(true);
      expect(clamped(result.movementPattern)).toBe(true);
    });
  });
});
