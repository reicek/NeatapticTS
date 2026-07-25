import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/constants.ts.
 *
 * These tests define the numeric constants that the rest of the game logic
 * depends on. The source module does not exist yet, so every test fails with
 * a module-not-found error until the 02-game-scaffold slice provides them.
 */

describe('Neatenstein game constants', () => {
  describe('AC-215: exported numeric constants', () => {
    it('exports a positive fixed timestep in milliseconds', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect({
        isNumber: typeof mod.NEATENSTEIN_FIXED_TIMESTEP_MS === 'number',
        isPositive: (mod.NEATENSTEIN_FIXED_TIMESTEP_MS as number) > 0,
      }).toEqual({
        isNumber: true,
        isPositive: true,
      });
    });

    it('exports a positive player maximum health', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect({
        isNumber: typeof mod.NEATENSTEIN_PLAYER_MAX_HEALTH === 'number',
        isPositive: (mod.NEATENSTEIN_PLAYER_MAX_HEALTH as number) > 0,
      }).toEqual({
        isNumber: true,
        isPositive: true,
      });
    });

    it('exports a positive player maximum ammo', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect({
        isNumber: typeof mod.NEATENSTEIN_PLAYER_MAX_AMMO === 'number',
        isPositive: (mod.NEATENSTEIN_PLAYER_MAX_AMMO as number) > 0,
      }).toEqual({
        isNumber: true,
        isPositive: true,
      });
    });

    it('exports an enemy concurrent cap equal to 8', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_ENEMY_MAX_CONCURRENT as number).toBe(8);
    });

    it('exports dash invulnerability equal to 200 ms', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_DASH_INVULNERABILITY_MS as number).toBe(200);
    });

    it('exports a dash cooldown strictly greater than invulnerability', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect((mod.NEATENSTEIN_DASH_COOLDOWN_MS as number) > 200).toBe(true);
    });

    it('exports episode duration bounds between 15000 ms and 25000 ms', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect({
        min: mod.NEATENSTEIN_EPISODE_MIN_DURATION_MS as number,
        max: mod.NEATENSTEIN_EPISODE_MAX_DURATION_MS as number,
      }).toEqual({
        min: 15000,
        max: 25000,
      });
    });

    it('exports a minimum generation cadence of at least 2 per minute', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect((mod.NEATENSTEIN_MIN_GENERATIONS_PER_MINUTE as number) >= 2).toBe(
        true,
      );
    });
  });

  describe('AC-216 / 03-red: 60x60 map gameplay constants', () => {
    it('exports a beam max range that reaches across a 60x60 map', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(
        mod.NEATENSTEIN_BEAM_MAX_RANGE_CELLS as number,
      ).toBeGreaterThanOrEqual(60 * Math.SQRT2);
    });

    it('exports player spawn coordinates at the center of a 60x60 map', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect({
        x: mod.NEATENSTEIN_SPAWN_CENTER_X as number,
        y: mod.NEATENSTEIN_SPAWN_CENTER_Y as number,
      }).toEqual({
        x: 30.5,
        y: 30.5,
      });
    });
  });
});
