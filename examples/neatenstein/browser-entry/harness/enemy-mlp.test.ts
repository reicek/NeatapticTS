import { describe, expect, it } from '@jest/globals';

import type * as EnemyMlp from './enemy-mlp';
import type { EnemyVariant } from './types';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/enemy-mlp.ts.
 *
 * Covers:
 * - AC-501.1: MLP enemy uses a fixed 8→6→4→4 topology with per-layer bias
 *   and a 4-output move/strafe/turn/fire interpretation.
 * - AC-303: MLP enemy population update fires only on N % 5 == 0.
 */

interface EnemyMlpModule {
  createMlpEnemyPopulation: typeof EnemyMlp.createMlpEnemyPopulation;
}

/** Connection weight count for the fixed 8→6→4→4 topology. */
const MLP_CONNECTION_COUNT = 8 * 6 + 6 * 4 + 4 * 4;

/** Per-layer bias count for the fixed 8→6→4→4 topology (one bias per non-input neuron). */
const MLP_BIAS_COUNT = 6 + 4 + 4;

describe('Neatenstein harness enemy-mlp', () => {
  describe('AC-501.1: MLP 8→6→4→4 topology with bias', () => {
    it('exports the fixed 8→6→4→4 topology constant', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_MLP_TOPOLOGY as number[]).toEqual([8, 6, 4, 4]);
    });

    it('exports createMlpEnemyPopulation as a function', async () => {
      const mod = (await import('./enemy-mlp.ts')) as Record<string, unknown>;
      expect(typeof mod.createMlpEnemyPopulation).toBe('function');
    });

    it('produces a population of 32 variants', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      expect(population.size).toBe(32);
    });

    it('produces weight vectors sized for the topology plus per-layer biases', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const variant = population.sample(0) as EnemyVariant;
      expect(variant.weights.length).toBe(
        MLP_CONNECTION_COUNT + MLP_BIAS_COUNT,
      );
    });

    it('samples distinct weights for different variant ids', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const a = population.sample(0) as EnemyVariant;
      const b = population.sample(1) as EnemyVariant;
      expect(a.weights).not.toEqual(b.weights);
    });

    it('produces deterministic samples for the same seed', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const first = createMlpEnemyPopulation({ seed: 7 });
      const second = createMlpEnemyPopulation({ seed: 7 });
      expect((first.sample(0) as EnemyVariant).weights).toEqual(
        (second.sample(0) as EnemyVariant).weights,
      );
    });

    it('defaults to seed 0 when no options are provided', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const defaulted = createMlpEnemyPopulation();
      const explicit = createMlpEnemyPopulation({ seed: 0 });
      expect((defaulted.sample(0) as EnemyVariant).weights).toEqual(
        (explicit.sample(0) as EnemyVariant).weights,
      );
    });

    it('falls back to variant 0 for invalid sample indices', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      expect(population.sample(-1)).toEqual(population.sample(0));
    });
  });

  describe('AC-501.1: MLP 4-output move/strafe/turn/fire mapping', () => {
    it('exports the four output labels in order', async () => {
      const mod = (await import('./enemy-mlp.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_MLP_OUTPUT_LABELS as string[]).toEqual([
        'move',
        'strafe',
        'turn',
        'fire',
      ]);
    });

    it('exports an activation helper that returns four outputs', async () => {
      const mod = (await import('./enemy-mlp.ts')) as Record<string, unknown>;
      expect(typeof mod.activateMlp).toBe('function');
      const result = (
        mod.activateMlp as (
          weights: Float32Array,
          inputs: Float32Array,
        ) => Float32Array
      )(
        new Float32Array(MLP_CONNECTION_COUNT + MLP_BIAS_COUNT),
        new Float32Array(8),
      );
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(4);
    });

    it('exports an output interpreter keyed by label', async () => {
      const mod = (await import('./enemy-mlp.ts')) as Record<string, unknown>;
      expect(typeof mod.interpretMlpOutputs).toBe('function');
      const interpreted = (
        mod.interpretMlpOutputs as (
          outputs: Float32Array,
        ) => Record<string, number>
      )(new Float32Array([0.1, 0.2, 0.3, 0.4]));
      expect(interpreted).toEqual({
        move: 0.1,
        strafe: 0.2,
        turn: 0.3,
        fire: 0.4,
      });
    });
  });

  describe('AC-303: MLP update gating', () => {
    it('returns the same snapshot on generations not divisible by 5', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = population.snapshot();
      const after = population.update({ generation: 4 });
      expect(after).toBe(before);
    });

    it('returns a new snapshot on generation divisible by 5', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = population.snapshot();
      const after = population.update({ generation: 5 });
      expect(after).not.toBe(before);
    });
  });

  describe('AC-501.1: MLP activation helpers', () => {
    it('rejects mismatched input sizes in activateMlp', async () => {
      const mod = (await import('./enemy-mlp.ts')) as Record<string, unknown>;
      expect(() =>
        (mod.activateMlp as (w: Float32Array, i: Float32Array) => Float32Array)(
          new Float32Array(102),
          new Float32Array(7),
        ),
      ).toThrow();
    });

    it('rejects mismatched weight vector lengths in activateMlp', async () => {
      const mod = (await import('./enemy-mlp.ts')) as Record<string, unknown>;
      expect(() =>
        (mod.activateMlp as (w: Float32Array, i: Float32Array) => Float32Array)(
          new Float32Array(101),
          new Float32Array(8),
        ),
      ).toThrow();
    });

    it('rejects mismatched output and label counts in interpretMlpOutputs', async () => {
      const mod = (await import('./enemy-mlp.ts')) as Record<string, unknown>;
      expect(() =>
        (
          mod.interpretMlpOutputs as (o: Float32Array) => Record<string, number>
        )(new Float32Array([0.1, 0.2, 0.3])),
      ).toThrow();
    });
  });
});
