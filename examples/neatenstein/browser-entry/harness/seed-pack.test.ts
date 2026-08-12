import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/seed-pack.ts.
 *
 * Covers createSeedPack(options): returns a fixed, frozen set of seeds for one
 * generation so every variant evaluated in that generation sees the same
 * deterministic environment.
 */

interface SeedPackModule {
  createSeedPack: (options: {
    generation: number;
    variantCount?: number;
    seed?: number;
  }) => {
    generation: number;
    seeds: number[];
  };
}

const VARIANT_COUNT = 32;

describe('Neatenstein enemy seed-pack', () => {
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

    it('returns a deterministic pack when a fixed root seed is provided', async () => {
      const { createSeedPack } =
        (await import('./seed-pack.ts')) as unknown as SeedPackModule;

      const packA = createSeedPack({ generation: 5, seed: 42 });
      const packB = createSeedPack({ generation: 5, seed: 42 });

      expect(packA.generation).toBe(5);
      expect(packA.seeds).toHaveLength(VARIANT_COUNT);
      expect(packA.seeds).toEqual(packB.seeds);
      expect(packA.seeds).not.toEqual(createSeedPack({ generation: 5 }).seeds);
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
