import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/constants.ts.
 *
 * These tests define the numeric constants that the co-evolution harness
 * depends on. The source module does not exist yet, so every test fails with
 * a module-not-found error until the 03-harness-scaffold slice provides it.
 *
 * Coverage mapping:
 * - AC-302/AC-303: refresh/update cadence constants.
 * - AC-306: MLP topology, variant count, and SWARM size constants.
 * - AC-305: CombatQualitySignal default weight constants.
 * - AC-308 is a lint-hygiene criterion; it is not validated by runtime tests.
 */

describe('Neatenstein harness constants', () => {
  describe('AC-302 / AC-303: refresh cadence constants', () => {
    it('exports MLP refresh interval equal to 5 generations', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_MLP_REFRESH_INTERVAL_GENERATIONS as number).toBe(
        5,
      );
    });

    it('exports SWARM refresh interval equal to 3 generations', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_SWARM_REFRESH_INTERVAL_GENERATIONS as number).toBe(
        3,
      );
    });
  });

  describe('AC-306: MLP topology and variant constants', () => {
    it('exports MLP topology as 6, 6, 4, 4', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_MLP_TOPOLOGY as number[]).toEqual([6, 6, 4, 4]);
    });

    it('exports MLP variant count equal to 32', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_MLP_VARIANT_COUNT as number).toBe(32);
    });

    it('exports SWARM maximum size equal to 8', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_SWARM_MAX_SIZE as number).toBe(8);
    });
  });

  describe('AC-305: CombatQualitySignal default weights', () => {
    it('exports weight for survivalTicks equal to 1', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_WEIGHT_SURVIVAL_TICKS as number).toBe(1);
    });

    it('exports weight for damageDealt equal to 2', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_WEIGHT_DAMAGE_DEALT as number).toBe(2);
    });

    it('exports weight for kills equal to 5', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_WEIGHT_KILLS as number).toBe(5);
    });

    it('exports penalty weight for damageTaken equal to 1', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_WEIGHT_DAMAGE_TAKEN as number).toBe(1);
    });

    it('exports penalty weight for aimMissRate equal to 20 (scaled in P4S1)', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_WEIGHT_AIM_MISS_RATE as number).toBe(20);
    });

    it('exports complexity bonus weight equal to 0.1', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_WEIGHT_COMPLEXITY_BONUS as number).toBe(0.1);
    });

    it('exports parsimony density penalty weight equal to 0.01', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_WEIGHT_PARSIMONY_DENSITY_PENALTY as number).toBe(
        0.01,
      );
    });
  });
});
