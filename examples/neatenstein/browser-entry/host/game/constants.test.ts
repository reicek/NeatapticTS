import { describe, expect, it } from '@jest/globals';

import {
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_LIGHT_TOGGLE_KEY,
  NEATENSTEIN_MAP_SIZE,
} from './constants.ts';

/**
 * Contract tests for examples/neatenstein/browser-entry/host/game/constants.ts.
 *
 * These tests lock the numeric constants that the rest of the game logic
 * depends on. The source module is implemented; updates here must stay in sync
 * with the exported values in the source file.
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

    it('re-exports the shared light toggle key', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(typeof mod.NEATENSTEIN_LIGHT_TOGGLE_KEY).toBe('string');
    });

    it('re-exports the shared map size constant', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(typeof mod.NEATENSTEIN_MAP_SIZE).toBe('number');
      expect((mod.NEATENSTEIN_MAP_SIZE as number) > 0).toBe(true);
    });
  });

  it('re-exports the fixed timestep and shared constants via named imports', () => {
    expect(typeof NEATENSTEIN_FIXED_TIMESTEP_MS).toBe('number');
    expect(typeof NEATENSTEIN_LIGHT_TOGGLE_KEY).toBe('string');
    expect(NEATENSTEIN_MAP_SIZE).toBeGreaterThan(0);
  });

  it('re-exports the shared constants through the namespace object', async () => {
    const mod = await import('./constants.ts');
    expect(typeof mod.NEATENSTEIN_FIXED_TIMESTEP_MS).toBe('number');
    expect(typeof mod.NEATENSTEIN_LIGHT_TOGGLE_KEY).toBe('string');
    expect(mod.NEATENSTEIN_MAP_SIZE).toBeGreaterThan(0);
  });

  describe('AC-216 / 03-red: 120x120 map gameplay constants', () => {
    it('exports a positive bolt speed in cells per second', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(
        mod.NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND as number,
      ).toBeGreaterThan(0);
    });

    it('exports a bolt max range of 30 cells', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect(mod.NEATENSTEIN_BOLT_MAX_RANGE_CELLS as number).toBe(30);
    });

    it('exports player spawn coordinates at the center of a 120x120 map', async () => {
      const mod = (await import('./constants.ts')) as Record<string, unknown>;
      expect({
        x: mod.NEATENSTEIN_SPAWN_CENTER_X as number,
        y: mod.NEATENSTEIN_SPAWN_CENTER_Y as number,
      }).toEqual({
        x: 60.5,
        y: 60.5,
      });
    });
  });
});
