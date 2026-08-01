import { describe, expect, it } from '@jest/globals';

import { createMlpEnemyPopulation } from './enemy-mlp';
import { runEnemyWaveRunner } from './enemy-runner';

const VARIANT_COUNT = 32;

describe('Neatenstein headless enemy wave runner', () => {
  describe('runEnemyWaveRunner(population, seed, config?)', () => {
    it('exports runEnemyWaveRunner as a function', async () => {
      const mod = (await import('./enemy-runner.ts')) as Record<
        string,
        unknown
      >;
      expect(typeof mod.runEnemyWaveRunner).toBe('function');
    });

    it('returns a champion, per-variant scores, generation, and seed pack', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const result = runEnemyWaveRunner(population, 123);

      expect(result).toHaveProperty('champion');
      expect(result).toHaveProperty('scores');
      expect(result).toHaveProperty('generation');
      expect(result).toHaveProperty('seedPack');
      expect(result.scores).toHaveLength(VARIANT_COUNT);
      expect(result.seedPack.seeds).toHaveLength(VARIANT_COUNT);
      expect(result.generation).toBe(123);
      expect(result.champion.id).toBeGreaterThanOrEqual(0);
      expect(result.champion.id).toBeLessThan(VARIANT_COUNT);
      expect(typeof result.champion.score).toBe('number');
    });

    it('evaluates all 32 variants with unique ids in population order', () => {
      const population = createMlpEnemyPopulation({ seed: 2 });
      const result = runEnemyWaveRunner(population, 456);
      const ids = result.scores.map((entry) => entry.id);

      expect(new Set(ids).size).toBe(VARIANT_COUNT);
      expect(ids).toEqual(
        Array.from({ length: VARIANT_COUNT }, (_, index) => index),
      );
    });

    it('selects the champion with deterministic lowest-id tie-breaking', () => {
      const population = createMlpEnemyPopulation({ seed: 3 });
      const result = runEnemyWaveRunner(population, 789, {
        fitness: { damageWeight: 0, survivalWeight: 0 },
      });

      // With zero weights every score is zero, so the lowest id wins.
      expect(result.champion.id).toBe(0);
      expect(result.scores.every((entry) => entry.score === 0)).toBe(true);
    });

    it('uses the generation override from config when provided', () => {
      const population = createMlpEnemyPopulation({ seed: 4 });
      const result = runEnemyWaveRunner(population, 999, { generation: 42 });

      expect(result.generation).toBe(42);
      expect(result.seedPack.generation).toBe(999);
    });

    it('returns a frozen champion snapshot that does not alias live weights', () => {
      const population = createMlpEnemyPopulation({ seed: 5 });
      const result = runEnemyWaveRunner(population, 111);

      expect(Object.isFrozen(result.champion.snapshot)).toBe(true);
      expect(Object.isFrozen(result.champion.snapshot.weights)).toBe(true);

      const liveVariant = population.sample(result.champion.id) as {
        weights: Float32Array;
      };
      const before = liveVariant.weights[0];
      liveVariant.weights[0] = before + 9_999;

      expect(result.champion.snapshot.weights[0]).not.toBe(
        liveVariant.weights[0],
      );
    });

    it('is deterministic for the same seed and population', () => {
      const populationA = createMlpEnemyPopulation({ seed: 6 });
      const populationB = createMlpEnemyPopulation({ seed: 6 });

      const resultA = runEnemyWaveRunner(populationA, 222);
      const resultB = runEnemyWaveRunner(populationB, 222);

      expect(resultA.champion.id).toBe(resultB.champion.id);
      expect(resultA.scores.map((entry) => entry.score)).toEqual(
        resultB.scores.map((entry) => entry.score),
      );
      expect(resultA.seedPack.seeds).toEqual(resultB.seedPack.seeds);
    });

    it('honours custom fitness weights through the optional config', () => {
      const population = createMlpEnemyPopulation({ seed: 7 });

      const defaultRun = runEnemyWaveRunner(population, 333);
      const weightedRun = runEnemyWaveRunner(population, 333, {
        fitness: { damageWeight: 10, survivalWeight: 0 },
      });

      expect(weightedRun.scores.length).toBe(VARIANT_COUNT);
      expect(
        weightedRun.scores.some(
          (entry, index) => entry.score !== defaultRun.scores[index].score,
        ),
      ).toBe(true);
    });

    it('returns different champions for different seeds', () => {
      const populationA = createMlpEnemyPopulation({ seed: 8 });
      const populationB = createMlpEnemyPopulation({ seed: 8 });

      const resultA = runEnemyWaveRunner(populationA, 444);
      const resultB = runEnemyWaveRunner(populationB, 555);

      expect(resultA.seedPack.seeds).not.toEqual(resultB.seedPack.seeds);
    });
  });
});
