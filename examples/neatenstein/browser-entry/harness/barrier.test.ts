import { describe, expect, it } from '@jest/globals';

import { createMlpEnemyPopulation } from './enemy-mlp';
import type {
  EnemyPopulation,
  MlpSnapshot,
  SeedPack,
  Snapshot,
  SwarmSnapshot,
} from './types';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/barrier.ts.
 *
 * Covers AC-504: the generation barrier must expose
 * buildEnemyEvaluationBarrier(generation, population, seedPack) returning
 * { snapshot, seedPack } where the snapshot is frozen and the seed pack is
 * deterministic for the generation, so replay evaluation never reads from the
 * live mutable MLP population.
 */

interface BarrierModule {
  buildEnemyEvaluationBarrier: (
    generation: number,
    population: EnemyPopulation,
    seedPack: SeedPack,
  ) => { snapshot: Snapshot; seedPack: SeedPack };
  genBarrier: (options: { seed: number; generation: number }) => {
    generation: number;
    mainSnapshot: { id: number };
    enemySnapshot: Snapshot;
    seed: number;
  };
  hashEnemySnapshot: (snapshot: Snapshot) => string;
}

const VARIANT_COUNT = 32;

const makePack = (generation: number): SeedPack => ({
  generation,
  seeds: Array.from({ length: VARIANT_COUNT }, (_, i) => generation * 1000 + i),
});

describe('Neatenstein enemy evaluation barrier', () => {
  describe('buildEnemyEvaluationBarrier(generation, population, seedPack)', () => {
    it('exports buildEnemyEvaluationBarrier as a function', async () => {
      const mod = (await import('./barrier.ts')) as Record<string, unknown>;
      expect(typeof mod.buildEnemyEvaluationBarrier).toBe('function');
    });

    it('returns an object with frozen snapshot and the provided seed pack', async () => {
      const { buildEnemyEvaluationBarrier } =
        (await import('./barrier.ts')) as unknown as BarrierModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const seedPack = makePack(5);
      const barrier = buildEnemyEvaluationBarrier(5, population, seedPack);

      expect(barrier).toHaveProperty('snapshot');
      expect(barrier).toHaveProperty('seedPack');
      expect(Object.isFrozen(barrier.snapshot)).toBe(true);
      expect(barrier.seedPack).toBe(seedPack);
    });

    it('does not alias the live MLP population weights', async () => {
      const { buildEnemyEvaluationBarrier } =
        (await import('./barrier.ts')) as unknown as BarrierModule;
      const population = createMlpEnemyPopulation({ seed: 3 });
      const seedPack = makePack(7);
      const barrier = buildEnemyEvaluationBarrier(7, population, seedPack);
      const before = (barrier.snapshot as MlpSnapshot).weights[0];

      const variant = population.sample(0) as { weights: Float32Array };
      variant.weights[0] = 999;

      const after = (barrier.snapshot as MlpSnapshot).weights[0];
      expect(after).toBe(before);
    });

    it('produces the same barrier snapshot for the same population seed', async () => {
      const { buildEnemyEvaluationBarrier } =
        (await import('./barrier.ts')) as unknown as BarrierModule;
      const populationA = createMlpEnemyPopulation({ seed: 1 });
      const populationB = createMlpEnemyPopulation({ seed: 1 });
      const seedPack = makePack(2);
      const barrierA = buildEnemyEvaluationBarrier(2, populationA, seedPack);
      const barrierB = buildEnemyEvaluationBarrier(2, populationB, seedPack);

      expect((barrierA.snapshot as MlpSnapshot).weights).toEqual(
        (barrierB.snapshot as MlpSnapshot).weights,
      );
      expect(barrierA.seedPack).toEqual(barrierB.seedPack);
    });

    it('keeps the seed pack deterministic for the generation', async () => {
      const { buildEnemyEvaluationBarrier } =
        (await import('./barrier.ts')) as unknown as BarrierModule;
      const population = createMlpEnemyPopulation({ seed: 4 });
      const seedPack = makePack(9);
      const barrier = buildEnemyEvaluationBarrier(9, population, seedPack);

      expect(barrier.seedPack.generation).toBe(9);
      expect(barrier.seedPack.seeds).toEqual(seedPack.seeds);
    });
  });

  describe('genBarrier(options)', () => {
    it('returns a barrier state for generation 0', async () => {
      const { genBarrier } =
        (await import('./barrier.ts')) as unknown as BarrierModule;
      const barrier = genBarrier({ seed: 1, generation: 0 });

      expect(barrier.generation).toBe(0);
      expect(barrier.seed).toBe(1);
      expect(barrier.mainSnapshot).toHaveProperty('id');
      expect(barrier.enemySnapshot.kind).toBe('mlp');
    });

    it('returns a barrier state for a non-refresh generation', async () => {
      const { genBarrier } =
        (await import('./barrier.ts')) as unknown as BarrierModule;
      const barrier = genBarrier({ seed: 2, generation: 1 });

      expect(barrier.generation).toBe(1);
      expect(barrier.enemySnapshot.kind).toBe('mlp');
    });
  });

  describe('hashEnemySnapshot(snapshot)', () => {
    it('returns a stable hash for an MLP snapshot', async () => {
      const { hashEnemySnapshot } =
        (await import('./barrier.ts')) as unknown as BarrierModule;
      const snapshot: MlpSnapshot = {
        kind: 'mlp',
        weights: new Float32Array([0.1, -0.2, 0.3]),
      };

      expect(hashEnemySnapshot(snapshot)).toMatch(/^mlp:/);
    });

    it('returns a stable hash for a SWARM snapshot', async () => {
      const { hashEnemySnapshot } =
        (await import('./barrier.ts')) as unknown as BarrierModule;
      const snapshot: SwarmSnapshot = {
        kind: 'swarm',
        dna: 'abc',
        coordinates: [
          { x: 1.1, y: 2.2 },
          { x: 3.3, y: 4.4 },
        ],
      };

      expect(hashEnemySnapshot(snapshot)).toMatch(/^swarm:/);
    });
  });
});
