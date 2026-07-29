import { describe, expect, it } from '@jest/globals';

import type * as SeedPack from './seed-pack';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/seed-pack.ts.
 *
 * Covers AC-304: all variants evaluated in a single generation see an
 * identical frozen seed pack.
 */

interface SeedPackModule {
  createSeedPack: typeof SeedPack.createSeedPack;
}

describe('Neatenstein harness seed-pack', () => {
  describe('AC-304: seed-pack fairness', () => {
    it('exports createSeedPack as a function', async () => {
      const mod = (await import('./seed-pack.ts')) as Record<string, unknown>;
      expect(typeof mod.createSeedPack).toBe('function');
    });

    it('returns identical frozen seeds for the same generation', async () => {
      const { createSeedPack } =
        (await import('./seed-pack.ts')) as SeedPackModule;
      const first = createSeedPack({ generation: 5, variantCount: 32 });
      const second = createSeedPack({ generation: 5, variantCount: 32 });
      expect(first.seeds).toEqual(second.seeds);
    });

    it('returns different frozen seeds for different generations', async () => {
      const { createSeedPack } =
        (await import('./seed-pack.ts')) as SeedPackModule;
      const gen5 = createSeedPack({ generation: 5, variantCount: 32 });
      const gen6 = createSeedPack({ generation: 6, variantCount: 32 });
      expect(gen5.seeds).not.toEqual(gen6.seeds);
    });

    it('returns one seed per variant', async () => {
      const { createSeedPack } =
        (await import('./seed-pack.ts')) as SeedPackModule;
      const pack = createSeedPack({ generation: 1, variantCount: 2048 });
      expect(pack.seeds.length).toBe(2048);
    });

    it('exposes the requested generation on the pack', async () => {
      const { createSeedPack } =
        (await import('./seed-pack.ts')) as SeedPackModule;
      const pack = createSeedPack({ generation: 9, variantCount: 32 });
      expect(pack.generation).toBe(9);
    });

    it('rejects a negative generation', async () => {
      const { createSeedPack } =
        (await import('./seed-pack.ts')) as SeedPackModule;
      expect(() => createSeedPack({ generation: -1 })).toThrow(
        'generation must be a non-negative integer',
      );
    });

    it('rejects a non-positive variant count', async () => {
      const { createSeedPack } =
        (await import('./seed-pack.ts')) as SeedPackModule;
      expect(() => createSeedPack({ generation: 1, variantCount: 0 })).toThrow(
        'variantCount must be a positive integer',
      );
    });
  });
});
