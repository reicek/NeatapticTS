import { describe, expect, it } from '@jest/globals';

import { createMlpEnemyPopulation } from './enemy-mlp';
import type { EnemyPopulation, MlpSnapshot } from './types';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/snapshot.ts.
 *
 * Covers the rolling enemy weight snapshot store:
 *  - refreshEnemySnapshots(population) refreshes the store from a population.
 *  - getEnemySnapshot(variantId) returns a frozen snapshot of that variant.
 *  - Snapshots must never alias the live mutable weights.
 */

interface SnapshotModule {
  refreshEnemySnapshots: (population: EnemyPopulation) => void;
  getEnemySnapshot: (variantId: number) => MlpSnapshot;
  shouldRefreshMlpSnapshot: (generation: number) => boolean;
  shouldRefreshSwarmSnapshot: (generation: number) => boolean;
}

describe('Neatenstein enemy snapshot store', () => {
  describe('refreshEnemySnapshots + getEnemySnapshot', () => {
    it('exports refreshEnemySnapshots and getEnemySnapshot as functions', async () => {
      const mod = (await import('./snapshot.ts')) as Record<string, unknown>;
      expect(typeof mod.refreshEnemySnapshots).toBe('function');
      expect(typeof mod.getEnemySnapshot).toBe('function');
    });

    it('returns a frozen MLP snapshot for a variant', async () => {
      const { refreshEnemySnapshots, getEnemySnapshot } =
        (await import('./snapshot.ts')) as unknown as SnapshotModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      refreshEnemySnapshots(population);

      const snapshot = getEnemySnapshot(0);
      expect(snapshot).toHaveProperty('kind', 'mlp');
      expect(snapshot).toHaveProperty('weights');
      expect(Object.isFrozen(snapshot)).toBe(true);
      expect(Object.isFrozen(snapshot.weights)).toBe(true);
    });

    it('does not alias live population weights', async () => {
      const { refreshEnemySnapshots, getEnemySnapshot } =
        (await import('./snapshot.ts')) as unknown as SnapshotModule;
      const population = createMlpEnemyPopulation({ seed: 2 });
      refreshEnemySnapshots(population);

      const before = getEnemySnapshot(0).weights[0];
      const variant = population.sample(0) as { weights: Float32Array };
      variant.weights[0] = 999;
      const after = getEnemySnapshot(0).weights[0];

      expect(after).toBe(before);
    });

    it('returns distinct snapshots for distinct variants', async () => {
      const { refreshEnemySnapshots, getEnemySnapshot } =
        (await import('./snapshot.ts')) as unknown as SnapshotModule;
      const population = createMlpEnemyPopulation({ seed: 3 });
      refreshEnemySnapshots(population);

      const a = getEnemySnapshot(0);
      const b = getEnemySnapshot(1);
      expect(a.weights).not.toBe(b.weights);
    });

    it('throws when requesting a variant missing from the store', async () => {
      const { refreshEnemySnapshots, getEnemySnapshot } =
        (await import('./snapshot.ts')) as unknown as SnapshotModule;
      const emptyPopulation: EnemyPopulation = {
        kind: 'swarm',
        size: 0,
        sample: () => ({}),
        snapshot: () => ({ kind: 'swarm', dna: '', coordinates: [] }),
      };
      refreshEnemySnapshots(emptyPopulation);

      expect(() => getEnemySnapshot(0)).toThrow(
        'No enemy snapshot for variant 0; call refreshEnemySnapshots first.',
      );
    });

    it('leaves the store empty for a non-MLP population', async () => {
      const { refreshEnemySnapshots, getEnemySnapshot } =
        (await import('./snapshot.ts')) as unknown as SnapshotModule;
      const population: EnemyPopulation = {
        kind: 'swarm',
        size: 4,
        sample: () => ({}),
        snapshot: () => ({ kind: 'swarm', dna: '', coordinates: [] }),
      };
      refreshEnemySnapshots(population);

      expect(() => getEnemySnapshot(0)).toThrow(
        'No enemy snapshot for variant 0; call refreshEnemySnapshots first.',
      );
    });
  });

  describe('refresh gates', () => {
    it('reports MLP refresh boundaries correctly', async () => {
      const { shouldRefreshMlpSnapshot } =
        (await import('./snapshot.ts')) as unknown as SnapshotModule;

      expect(shouldRefreshMlpSnapshot(0)).toBe(true);
      expect(shouldRefreshMlpSnapshot(1)).toBe(false);
      expect(shouldRefreshMlpSnapshot(5)).toBe(true);
    });

    it('reports SWARM refresh boundaries correctly', async () => {
      const { shouldRefreshSwarmSnapshot } =
        (await import('./snapshot.ts')) as unknown as SnapshotModule;

      expect(shouldRefreshSwarmSnapshot(0)).toBe(true);
      expect(shouldRefreshSwarmSnapshot(1)).toBe(false);
      expect(shouldRefreshSwarmSnapshot(3)).toBe(true);
    });
  });
});
