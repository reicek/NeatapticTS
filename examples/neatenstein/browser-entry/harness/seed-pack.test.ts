import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/seed-pack.ts.
 *
 * Covers makeEnemySeedPack(seed): returns a fixed, frozen set of seeds for one
 * generation so every variant evaluated in that generation sees the same
 * deterministic environment.
 */

interface SeedPackModule {
  makeEnemySeedPack: (seed: number) => { seeds: number[] };
  createSeedPack: (options: { generation: number; variantCount?: number }) => {
    generation: number;
    seeds: number[];
  };
}

const VARIANT_COUNT = 32;

describe('Neatenstein enemy seed-pack', () => {
  describe('makeEnemySeedPack(seed)', () => {
    it('exports makeEnemySeedPack as a function', async () => {
      const mod = (await import('./seed-pack.ts')) as Record<string, unknown>;
      expect(typeof mod.makeEnemySeedPack).toBe('function');
    });

    it('returns a fixed pack with one seed per enemy variant', async () => {
      const { makeEnemySeedPack } =
        (await import('./seed-pack.ts')) as unknown as SeedPackModule;
      const pack = makeEnemySeedPack(123);

      expect(pack).toHaveProperty('seeds');
      expect(Array.isArray(pack.seeds)).toBe(true);
      expect(pack.seeds).toHaveLength(VARIANT_COUNT);
    });

    it('returns identical seeds for the same seed', async () => {
      const { makeEnemySeedPack } =
        (await import('./seed-pack.ts')) as unknown as SeedPackModule;
      const first = makeEnemySeedPack(7);
      const second = makeEnemySeedPack(7);

      expect(first.seeds).toEqual(second.seeds);
    });

    it('returns different seeds for different seeds', async () => {
      const { makeEnemySeedPack } =
        (await import('./seed-pack.ts')) as unknown as SeedPackModule;
      const first = makeEnemySeedPack(7);
      const second = makeEnemySeedPack(8);

      expect(first.seeds).not.toEqual(second.seeds);
    });

    it('returns a frozen seed pack', async () => {
      const { makeEnemySeedPack } =
        (await import('./seed-pack.ts')) as unknown as SeedPackModule;
      const pack = makeEnemySeedPack(1);

      expect(Object.isFrozen(pack)).toBe(true);
      expect(Object.isFrozen(pack.seeds)).toBe(true);
    });

    it('throws when the seed is not a finite integer', async () => {
      const { makeEnemySeedPack } =
        (await import('./seed-pack.ts')) as unknown as SeedPackModule;

      expect(() => makeEnemySeedPack(Number.NaN)).toThrow(
        'seed must be a finite integer',
      );
      expect(() => makeEnemySeedPack(1.5)).toThrow(
        'seed must be a finite integer',
      );
    });
  });

  describe('createSeedPack(options)', () => {
    it('returns a frozen seed pack for a generation', async () => {
      const { createSeedPack } =
        (await import('./seed-pack.ts')) as unknown as SeedPackModule;
      const pack = createSeedPack({ generation: 5 });

      expect(pack.generation).toBe(5);
      expect(pack.seeds).toHaveLength(VARIANT_COUNT);
      expect(Object.isFrozen(pack)).toBe(true);
      expect(Object.isFrozen(pack.seeds)).toBe(true);
    });

    it('throws when generation is negative or non-integer', async () => {
      const { createSeedPack } =
        (await import('./seed-pack.ts')) as unknown as SeedPackModule;

      expect(() => createSeedPack({ generation: -1 })).toThrow(
        'generation must be a non-negative integer',
      );
      expect(() => createSeedPack({ generation: 1.5 })).toThrow(
        'generation must be a non-negative integer',
      );
      expect(() => createSeedPack({ generation: Number.NaN })).toThrow(
        'generation must be a non-negative integer',
      );
    });

    it('throws when variantCount is not a positive integer', async () => {
      const { createSeedPack } =
        (await import('./seed-pack.ts')) as unknown as SeedPackModule;

      expect(() => createSeedPack({ generation: 0, variantCount: 0 })).toThrow(
        'variantCount must be a positive integer',
      );
      expect(() => createSeedPack({ generation: 0, variantCount: -1 })).toThrow(
        'variantCount must be a positive integer',
      );
      expect(() =>
        createSeedPack({ generation: 0, variantCount: 1.5 }),
      ).toThrow('variantCount must be a positive integer');
    });
  });
});
