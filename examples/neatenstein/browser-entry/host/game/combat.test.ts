import { describe, expect, it } from '@jest/globals';

import { fireNeonBeam } from './combat';
import {
  NEATENSTEIN_BEAM_DAMAGE,
  NEATENSTEIN_BEAM_MAX_RANGE_CELLS,
  NEATENSTEIN_MUZZLE_OFFSET_CELLS,
  NEATENSTEIN_TEST_ENEMY_BEHIND_WALL_DISTANCE_CELLS,
  NEATENSTEIN_TEST_ENEMY_BEYOND_RANGE_OFFSET_CELLS,
  NEATENSTEIN_TEST_ENEMY_FAR_DISTANCE_CELLS,
  NEATENSTEIN_TEST_ENEMY_HEALTH,
  NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
  NEATENSTEIN_TEST_ENEMY_OFF_BEAM_OFFSET_CELLS,
  NEATENSTEIN_TEST_SEED,
} from './constants';
import { createGameState } from './state';
import type { GameState } from './types';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/combat.ts.
 *
 * Covers AC-204 (hitscan neon beam) and AC-210 (only the beam weapon exists).
 */

describe('Neatenstein game combat', () => {
  describe('AC-204 / AC-210: neon beam hitscan contract', () => {
    it('exports fireNeonBeam', () => {
      expect(typeof fireNeonBeam).toBe('function');
    });

    it('returns a tracer object for the fired frame', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireNeonBeam(state);
      expect(typeof result.tracer).toBe('object');
    });

    it('does not fire when ammo is zero', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, ammo: 0 },
      };
      const result = fireNeonBeam(state);
      expect(result.fired).toBe(false);
    });

    it('consumes one unit of ammo when fired', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireNeonBeam(state);
      expect(result.state.player.ammo).toBe(state.player.ammo - 1);
    });

    it('appends a single tracer to the new state when fired', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireNeonBeam(state);
      expect(result.state.tracers.length).toBe(state.tracers.length + 1);
    });

    it('returns a tracer with beam origin, endpoint, and hit metadata', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0 },
      };
      const result = fireNeonBeam(state);
      expect(result.tracer).toMatchObject({
        origin: {
          x: state.player.position.x + NEATENSTEIN_MUZZLE_OFFSET_CELLS,
          y: state.player.position.y,
        },
        hitType: expect.any(String),
        distance: expect.any(Number),
        color: expect.any(String),
      });
    });

    it('damages the first enemy in the beam path', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0, ammo: base.player.maxAmmo },
        enemies: [
          {
            position: {
              x:
                base.player.position.x +
                NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
              y: base.player.position.y,
            },
            health: NEATENSTEIN_TEST_ENEMY_HEALTH,
          },
        ],
      };
      const result = fireNeonBeam(state);
      expect(result.state.enemies[0].health).toBe(
        NEATENSTEIN_TEST_ENEMY_HEALTH - NEATENSTEIN_BEAM_DAMAGE,
      );
    });

    it('skips an enemy behind the beam origin', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0, ammo: base.player.maxAmmo },
        enemies: [
          {
            position: {
              x:
                base.player.position.x -
                NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
              y: base.player.position.y,
            },
            health: NEATENSTEIN_TEST_ENEMY_HEALTH,
          },
        ],
      };
      const result = fireNeonBeam(state);
      expect(result.tracer?.hitType).toBe('wall');
    });

    it('skips an enemy beyond the beam max range', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0, ammo: base.player.maxAmmo },
        enemies: [
          {
            position: {
              x:
                base.player.position.x +
                NEATENSTEIN_BEAM_MAX_RANGE_CELLS +
                NEATENSTEIN_TEST_ENEMY_BEYOND_RANGE_OFFSET_CELLS,
              y: base.player.position.y,
            },
            health: NEATENSTEIN_TEST_ENEMY_HEALTH,
          },
        ],
      };
      const result = fireNeonBeam(state);
      expect(result.state.enemies[0].health).toBe(
        NEATENSTEIN_TEST_ENEMY_HEALTH,
      );
    });

    it('stops at the wall when an enemy is behind it', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0, ammo: base.player.maxAmmo },
        enemies: [
          {
            position: {
              x:
                base.player.position.x +
                NEATENSTEIN_TEST_ENEMY_BEHIND_WALL_DISTANCE_CELLS,
              y: base.player.position.y,
            },
            health: NEATENSTEIN_TEST_ENEMY_HEALTH,
          },
        ],
      };
      const result = fireNeonBeam(state);
      expect({
        hitType: result.tracer?.hitType,
        enemyHealth: result.state.enemies[0].health,
      }).toEqual({
        hitType: 'wall',
        enemyHealth: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
    });

    it('counts a kill when the beam reduces an enemy to zero health', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0, ammo: base.player.maxAmmo },
        enemies: [
          {
            position: {
              x:
                base.player.position.x +
                NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
              y: base.player.position.y,
            },
            health: NEATENSTEIN_BEAM_DAMAGE,
          },
        ],
      };
      const result = fireNeonBeam(state);
      expect(result.state.kills).toBe(1);
    });

    it('hits the nearest enemy when two enemies share the beam', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0, ammo: base.player.maxAmmo },
        enemies: [
          {
            position: {
              x:
                base.player.position.x +
                NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
              y: base.player.position.y,
            },
            health: NEATENSTEIN_TEST_ENEMY_HEALTH,
          },
          {
            position: {
              x:
                base.player.position.x +
                NEATENSTEIN_TEST_ENEMY_FAR_DISTANCE_CELLS,
              y: base.player.position.y,
            },
            health: NEATENSTEIN_TEST_ENEMY_HEALTH,
          },
        ],
      };
      const result = fireNeonBeam(state);
      expect({
        hitType: result.tracer?.hitType,
        nearHealth: result.state.enemies[0].health,
        farHealth: result.state.enemies[1].health,
      }).toEqual({
        hitType: 'enemy',
        nearHealth: NEATENSTEIN_TEST_ENEMY_HEALTH - NEATENSTEIN_BEAM_DAMAGE,
        farHealth: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
    });

    it('skips a dead enemy', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0, ammo: base.player.maxAmmo },
        enemies: [
          {
            position: {
              x:
                base.player.position.x +
                NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
              y: base.player.position.y,
            },
            health: 0,
          },
        ],
      };
      const result = fireNeonBeam(state);
      expect(result.tracer?.hitType).toBe('wall');
    });

    it('misses an enemy too far perpendicular to the beam', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0, ammo: base.player.maxAmmo },
        enemies: [
          {
            position: {
              x:
                base.player.position.x +
                NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
              y:
                base.player.position.y +
                NEATENSTEIN_TEST_ENEMY_OFF_BEAM_OFFSET_CELLS,
            },
            health: NEATENSTEIN_TEST_ENEMY_HEALTH,
          },
        ],
      };
      const result = fireNeonBeam(state);
      expect(result.state.enemies[0].health).toBe(
        NEATENSTEIN_TEST_ENEMY_HEALTH,
      );
    });

    it('hits the wall when no enemies are in range', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireNeonBeam(state);
      expect(result.tracer?.hitType).toBe('wall');
    });

    it('AC-210: does not export weapon-switching helpers', async () => {
      const mod = (await import('./combat')) as Record<string, unknown>;
      expect(
        mod.switchWeapon ?? mod.weaponIndex ?? mod.weaponState,
      ).toBeUndefined();
    });
  });
});
