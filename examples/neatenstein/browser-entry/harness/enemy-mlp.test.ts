import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/enemy-mlp.ts.
 *
 * Covers:
 * - AC-303: MLP enemy population update fires only on N % 5 == 0.
 * - AC-306: MLP backend uses a fixed 8→6→4→2 topology, weight-only
 *   evolution, and 32 variants.
 */

describe('Neatenstein harness enemy-mlp', () => {
  describe('AC-306: MLP backend shape', () => {
    it('exports createMlpEnemyPopulation as a function', async () => {
      const mod = (await import('./enemy-mlp.ts')) as Record<string, unknown>;
      expect(typeof mod.createMlpEnemyPopulation).toBe('function');
    });

    it('produces a population of 32 variants', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as Record<string, any>;
      const population = createMlpEnemyPopulation({ seed: 1 });
      expect(population.size).toBe(32);
    });

    it('samples distinct weights for different variant ids', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as Record<string, any>;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const a = population.sample(0);
      const b = population.sample(1);
      expect(a.weights).not.toEqual(b.weights);
    });

    it('produces deterministic samples for the same seed', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as Record<string, any>;
      const first = createMlpEnemyPopulation({ seed: 7 });
      const second = createMlpEnemyPopulation({ seed: 7 });
      expect(first.sample(0).weights).toEqual(second.sample(0).weights);
    });

    it('defaults to seed 0 when no options are provided', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as Record<string, any>;
      const defaulted = createMlpEnemyPopulation();
      const explicit = createMlpEnemyPopulation({ seed: 0 });
      expect(defaulted.sample(0).weights).toEqual(explicit.sample(0).weights);
    });

    it('falls back to variant 0 for invalid sample indices', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as Record<string, any>;
      const population = createMlpEnemyPopulation({ seed: 1 });
      expect(population.sample(-1)).toEqual(population.sample(0));
    });
  });

  describe('AC-303: MLP update gating', () => {
    it('returns the same snapshot on generations not divisible by 5', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as Record<string, any>;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = population.snapshot();
      const after = population.update({ generation: 4 });
      expect(after).toBe(before);
    });

    it('returns a new snapshot on generation divisible by 5', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as Record<string, any>;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = population.snapshot();
      const after = population.update({ generation: 5 });
      expect(after).not.toBe(before);
    });
  });
});
