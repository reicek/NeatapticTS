import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for
 * examples/neatenstein/browser-entry/harness/enemy-evolution.ts.
 *
 * Covers AC-A4: real per-death enemy NEAT evolution replaces the previous
 * deterministic weight reseeding with parent selection, mutation, and a new
 * stable variant id.
 */

interface EnemyEvolutionModule {
  evolveEnemyOnDeath: (
    parentWeights: Float32Array,
    fitnessRecord: {
      damageDealt: number;
      survivalTicks: number;
      kills: number;
      deaths: number;
      damageTaken: number;
    },
    mutationSeed: number,
    options?: { mutationRate?: number },
  ) => { weights: Float32Array; variantId: number };
}

/** Connection weight count for the fixed 6→6→4→4 topology. */
const MLP_CONNECTION_COUNT = 6 * 6 + 6 * 4 + 4 * 4;

/** Per-layer bias count for the fixed 6→6→4→4 topology (one bias per non-input neuron). */
const MLP_BIAS_COUNT = 6 + 4 + 4;

/** Total weight-vector length for the fixed MLP enemy topology. */
const MLP_WEIGHT_COUNT = MLP_CONNECTION_COUNT + MLP_BIAS_COUNT;

describe('Neatenstein harness enemy-evolution', () => {
  describe('AC-A4-001: per-death enemy evolution', () => {
    it('exports evolveEnemyOnDeath as a function', async () => {
      const mod = (await import('./enemy-evolution.ts')) as Record<
        string,
        unknown
      >;
      expect(typeof mod.evolveEnemyOnDeath).toBe('function');
    });

    it('produces mutated weights that differ from the parent', async () => {
      const { evolveEnemyOnDeath } =
        (await import('./enemy-evolution.ts')) as EnemyEvolutionModule;
      const parentWeights = new Float32Array(MLP_WEIGHT_COUNT);
      const fitnessRecord = {
        damageDealt: 5,
        survivalTicks: 100,
        kills: 1,
        deaths: 1,
        damageTaken: 3,
      };
      const result = evolveEnemyOnDeath(parentWeights, fitnessRecord, 42);
      expect(result.weights).toBeInstanceOf(Float32Array);
      expect(result.weights.length).toBe(MLP_WEIGHT_COUNT);
      expect(result.weights).not.toEqual(parentWeights);
    });

    it('produces a new variantId', async () => {
      const { evolveEnemyOnDeath } =
        (await import('./enemy-evolution.ts')) as EnemyEvolutionModule;
      const parentWeights = new Float32Array(MLP_WEIGHT_COUNT);
      const fitnessRecord = {
        damageDealt: 5,
        survivalTicks: 100,
        kills: 1,
        deaths: 1,
        damageTaken: 3,
      };
      const result = evolveEnemyOnDeath(parentWeights, fitnessRecord, 42);
      expect(typeof result.variantId).toBe('number');
      expect(result.variantId).toBeGreaterThanOrEqual(0);
    });

    it('is deterministic for the same mutation seed', async () => {
      const { evolveEnemyOnDeath } =
        (await import('./enemy-evolution.ts')) as EnemyEvolutionModule;
      const parentWeights = new Float32Array(MLP_WEIGHT_COUNT);
      const fitnessRecord = {
        damageDealt: 5,
        survivalTicks: 100,
        kills: 1,
        deaths: 1,
        damageTaken: 3,
      };
      const first = evolveEnemyOnDeath(parentWeights, fitnessRecord, 42);
      const second = evolveEnemyOnDeath(parentWeights, fitnessRecord, 42);
      expect(first.weights).toEqual(second.weights);
      expect(first.variantId).toBe(second.variantId);
    });

    it('preserves weight vector length matching the MLP topology', async () => {
      const { evolveEnemyOnDeath } =
        (await import('./enemy-evolution.ts')) as EnemyEvolutionModule;
      const parentWeights = new Float32Array(MLP_WEIGHT_COUNT);
      const fitnessRecord = {
        damageDealt: 5,
        survivalTicks: 100,
        kills: 1,
        deaths: 1,
        damageTaken: 3,
      };
      const result = evolveEnemyOnDeath(parentWeights, fitnessRecord, 42);
      expect(result.weights.length).toBe(MLP_WEIGHT_COUNT);
    });
  });
});
