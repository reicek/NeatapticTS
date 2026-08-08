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

interface DeathFeedbackModule {
  computeAdaptationSignal: (
    prevGeneration: GenerationSnapshot,
    currGeneration: GenerationSnapshot,
  ) => AdaptationSignal;
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
});
