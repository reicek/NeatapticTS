/**
 * Red-phase contract tests for Step B4 items 4 + 5 + 6:
 * Transition-level experience replay, prioritized death replay, and
 * CERL-style shared replay for the main agent.
 *
 * All tests in this file MUST fail (RED) until the B4 implementation lands.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import * as TransitionReplayModule from './transition-replay';

/** Access unknown (future) exports on the transition-replay module. */
const transitionReplay = TransitionReplayModule as Record<string, unknown>;

/** Fixed MLP weight count for the 6→6→4→4 topology (90 parameters). */
const MLP_WEIGHT_COUNT = 6 * 6 + 6 * 4 + 4 * 4 + 6 + 4 + 4;

/** Build a deterministic weight vector for fixture use. */
function makeWeights(): Float32Array {
  return new Float32Array(MLP_WEIGHT_COUNT);
}

/** Build a minimal transition fixture. */
function makeTransition(
  tick: number,
  input: number[],
  output: number[],
  reward: number,
) {
  return {
    tick,
    input: new Float32Array(input),
    output: new Float32Array(output),
    reward,
    done: false,
  };
}

describe('B4.4: Transition replay for Lamarckian updates', () => {
  describe('B4.4: transition buffer creation', () => {
    it('exports createTransitionBuffer as a function', () => {
      expect(typeof transitionReplay.createTransitionBuffer).toBe('function');
    });

    it('creates a bounded FIFO buffer with the given capacity', () => {
      const buffer = (
        transitionReplay.createTransitionBuffer as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )(100);
      expect(buffer).toBeDefined();
      expect(typeof buffer.push).toBe('function');
      expect(typeof buffer.size).toBe('function');
      expect(typeof buffer.sample).toBe('function');
    });

    it('starts empty and grows as transitions are pushed', () => {
      const buffer = (
        transitionReplay.createTransitionBuffer as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )(100);
      expect((buffer.size as () => number)()).toBe(0);
      (buffer.push as (...a: unknown[]) => void)(
        makeTransition(0, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0], 0.1),
      );
      expect((buffer.size as () => number)()).toBe(1);
    });

    it('evicts oldest transitions when capacity is exceeded', () => {
      const buffer = (
        transitionReplay.createTransitionBuffer as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )(3);
      (buffer.push as (...a: unknown[]) => void)(
        makeTransition(0, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0], 0.1),
      );
      (buffer.push as (...a: unknown[]) => void)(
        makeTransition(1, [0, 1, 0, 0, 0, 0], [0, 0, 0, 0], 0.2),
      );
      (buffer.push as (...a: unknown[]) => void)(
        makeTransition(2, [0, 0, 1, 0, 0, 0], [0, 0, 0, 0], 0.3),
      );
      (buffer.push as (...a: unknown[]) => void)(
        makeTransition(3, [0, 0, 0, 1, 0, 0], [0, 0, 0, 0], 0.4),
      );
      expect((buffer.size as () => number)()).toBe(3);
    });
  });

  describe('B4.4: Lamarckian replay updates', () => {
    it('exports runReplayUpdates as a function', () => {
      expect(typeof transitionReplay.runReplayUpdates).toBe('function');
    });

    it('runs N backprop steps on the transition buffer and returns updated weights', () => {
      const buffer = (
        transitionReplay.createTransitionBuffer as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )(100);
      for (let i = 0; i < 10; i++) {
        (buffer.push as (...a: unknown[]) => void)(
          makeTransition(i, [0.5, 0.3, 0.2, 0.1, 0.4, 0.6], [1, 0, 0, 0], 0.5),
        );
      }
      const result = (
        transitionReplay.runReplayUpdates as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )({
        weights: makeWeights(),
        buffer,
        steps: 5,
        learningRate: 0.01,
        seed: 42,
      });
      expect(result).toBeDefined();
      expect(result.weights).toBeInstanceOf(Float32Array);
      expect((result.weights as Float32Array).length).toBe(MLP_WEIGHT_COUNT);
    });

    it('produces weights that differ from the input after replay', () => {
      const buffer = (
        transitionReplay.createTransitionBuffer as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )(100);
      for (let i = 0; i < 10; i++) {
        (buffer.push as (...a: unknown[]) => void)(
          makeTransition(i, [0.5, 0.3, 0.2, 0.1, 0.4, 0.6], [1, 0, 0, 0], 0.5),
        );
      }
      const inputWeights = makeWeights();
      const result = (
        transitionReplay.runReplayUpdates as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )({
        weights: new Float32Array(inputWeights),
        buffer,
        steps: 5,
        learningRate: 0.01,
        seed: 42,
      });
      expect(Array.from(result.weights as Float32Array)).not.toEqual(
        Array.from(inputWeights),
      );
    });
  });
});

describe('B4.5: Prioritized death replay by surprise', () => {
  describe('B4.5: surprise score computation', () => {
    it('exports computeDeathSurprise as a function', () => {
      expect(typeof transitionReplay.computeDeathSurprise).toBe('function');
    });

    it('computes a high surprise for a quick death (low survival ticks)', () => {
      const surprise = (
        transitionReplay.computeDeathSurprise as (...a: unknown[]) => number
      )({ survivalTicks: 5, healthAtDeath: 90, expectedDirection: 0 });
      expect(surprise).toBeGreaterThan(0);
      expect(surprise).toBeLessThanOrEqual(1);
    });

    it('computes a lower surprise for a long survival with low health at death', () => {
      const highSurprise = (
        transitionReplay.computeDeathSurprise as (...a: unknown[]) => number
      )({ survivalTicks: 5, healthAtDeath: 90, expectedDirection: 0 });
      const lowSurprise = (
        transitionReplay.computeDeathSurprise as (...a: unknown[]) => number
      )({ survivalTicks: 200, healthAtDeath: 5, expectedDirection: 0 });
      expect(lowSurprise).toBeLessThan(highSurprise);
    });
  });

  describe('B4.5: replay pressure from surprise', () => {
    it('exports computeReplayPressureFromSurprise as a function', () => {
      expect(typeof transitionReplay.computeReplayPressureFromSurprise).toBe(
        'function',
      );
    });

    it('clamps replay pressure to [0, 1]', () => {
      const pressure = (
        transitionReplay.computeReplayPressureFromSurprise as (
          ...a: unknown[]
        ) => number
      )({ surpriseScore: 0.5, maxSurprise: 1.0 });
      expect(pressure).toBeGreaterThanOrEqual(0);
      expect(pressure).toBeLessThanOrEqual(1);
    });

    it('returns 1.0 when surprise equals max surprise', () => {
      const pressure = (
        transitionReplay.computeReplayPressureFromSurprise as (
          ...a: unknown[]
        ) => number
      )({ surpriseScore: 1.0, maxSurprise: 1.0 });
      expect(pressure).toBeCloseTo(1.0, 5);
    });
  });
});

describe('B4.6: CERL-style shared replay for main agent', () => {
  describe('B4.6: shared replay buffer creation', () => {
    it('exports createSharedReplayBuffer as a function', () => {
      expect(typeof transitionReplay.createSharedReplayBuffer).toBe('function');
    });

    it('creates a shared buffer that all hero variants contribute to', () => {
      const shared = (
        transitionReplay.createSharedReplayBuffer as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )(200);
      expect(shared).toBeDefined();
      expect(typeof shared.push).toBe('function');
      expect(typeof shared.sample).toBe('function');
    });

    it('accepts transitions tagged with variantId', () => {
      const shared = (
        transitionReplay.createSharedReplayBuffer as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )(200);
      (shared.push as (...a: unknown[]) => void)({
        ...makeTransition(0, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0], 0.1),
        variantId: 3,
      });
      expect((shared.size as () => number)()).toBe(1);
    });
  });

  describe('B4.6: warm-start from shared replay', () => {
    it('exports warmStartFromSharedReplay as a function', () => {
      expect(typeof transitionReplay.warmStartFromSharedReplay).toBe(
        'function',
      );
    });

    it('produces warm-started weights from shared replay samples', () => {
      const shared = (
        transitionReplay.createSharedReplayBuffer as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )(200);
      for (let i = 0; i < 20; i++) {
        (shared.push as (...a: unknown[]) => void)({
          ...makeTransition(
            i,
            [0.5, 0.3, 0.2, 0.1, 0.4, 0.6],
            [1, 0, 0, 0],
            0.5,
          ),
          variantId: i % 4,
        });
      }
      const result = (
        transitionReplay.warmStartFromSharedReplay as (
          ...a: unknown[]
        ) => Float32Array
      )({ sharedBuffer: shared, steps: 10, learningRate: 0.01, seed: 42 });
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(MLP_WEIGHT_COUNT);
    });
  });
});
