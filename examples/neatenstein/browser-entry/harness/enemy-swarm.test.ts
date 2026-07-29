import { describe, expect, it } from '@jest/globals';

import type * as Constants from './constants';
import type * as EnemySwarm from './enemy-swarm';
import type { SwarmVariant } from './enemy-swarm';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/enemy-swarm.ts.
 *
 * Covers AC-306: the WeightSharedCohort backend uses a single shared DNA,
 * shared weights, and distinct per-enemy coordinate injection.
 */

interface EnemySwarmModule {
  createSwarmEnemyPopulation: typeof EnemySwarm.createSwarmEnemyPopulation;
}

interface ConstantsModule {
  NEATENSTEIN_SWARM_MAX_SIZE: typeof Constants.NEATENSTEIN_SWARM_MAX_SIZE;
}

describe('Neatenstein harness enemy-swarm', () => {
  describe('AC-306: WeightSharedCohort backend', () => {
    it('exports createSwarmEnemyPopulation as a function', async () => {
      const mod = (await import('./enemy-swarm.ts')) as Record<string, unknown>;
      expect(typeof mod.createSwarmEnemyPopulation).toBe('function');
    });

    it('uses a single shared DNA across all members', async () => {
      const { createSwarmEnemyPopulation } =
        (await import('./enemy-swarm.ts')) as EnemySwarmModule;
      const population = createSwarmEnemyPopulation({ seed: 1, size: 8 });
      const a = population.sample(0) as SwarmVariant;
      const b = population.sample(1) as SwarmVariant;
      expect(a.dna).toBe(b.dna);
    });

    it('injects distinct per-enemy coordinates', async () => {
      const { createSwarmEnemyPopulation } =
        (await import('./enemy-swarm.ts')) as EnemySwarmModule;
      const population = createSwarmEnemyPopulation({ seed: 1, size: 8 });
      const a = population.sample(0) as SwarmVariant;
      const b = population.sample(1) as SwarmVariant;
      expect(a.coordinates).not.toEqual(b.coordinates);
    });

    it('produces deterministic samples for the same seed', async () => {
      const { createSwarmEnemyPopulation } =
        (await import('./enemy-swarm.ts')) as EnemySwarmModule;
      const first = createSwarmEnemyPopulation({ seed: 7, size: 8 });
      const second = createSwarmEnemyPopulation({ seed: 7, size: 8 });
      expect((first.sample(0) as SwarmVariant).coordinates).toEqual(
        (second.sample(0) as SwarmVariant).coordinates,
      );
    });

    it('defaults to seed 0 when no options are provided', async () => {
      const { createSwarmEnemyPopulation } =
        (await import('./enemy-swarm.ts')) as EnemySwarmModule;
      const defaulted = createSwarmEnemyPopulation();
      const explicit = createSwarmEnemyPopulation({ seed: 0 });
      expect((defaulted.sample(0) as SwarmVariant).dna).toBe(
        (explicit.sample(0) as SwarmVariant).dna,
      );
    });

    it('defaults to the maximum cohort size when no size is provided', async () => {
      const { createSwarmEnemyPopulation } =
        (await import('./enemy-swarm.ts')) as EnemySwarmModule;
      const { NEATENSTEIN_SWARM_MAX_SIZE } =
        (await import('./constants')) as ConstantsModule;
      const population = createSwarmEnemyPopulation({ seed: 1 });
      expect(population.size).toBe(NEATENSTEIN_SWARM_MAX_SIZE);
    });

    it('falls back to member 0 for invalid sample indices', async () => {
      const { createSwarmEnemyPopulation } =
        (await import('./enemy-swarm.ts')) as EnemySwarmModule;
      const population = createSwarmEnemyPopulation({ seed: 1, size: 8 });
      expect(population.sample(-1)).toEqual(population.sample(0));
    });

    it('refreshes the champion snapshot on refresh generations', async () => {
      const { createSwarmEnemyPopulation } =
        (await import('./enemy-swarm.ts')) as EnemySwarmModule;
      const population = createSwarmEnemyPopulation({ seed: 1, size: 8 });
      const before = population.snapshot();
      const after = population.update({ generation: 3 });
      expect(after).not.toBe(before);
    });

    it('reuses the champion snapshot on non-refresh generations', async () => {
      const { createSwarmEnemyPopulation } =
        (await import('./enemy-swarm.ts')) as EnemySwarmModule;
      const population = createSwarmEnemyPopulation({ seed: 1, size: 8 });
      const before = population.snapshot();
      const after = population.update({ generation: 1 });
      expect(after).toBe(before);
    });
  });
});
