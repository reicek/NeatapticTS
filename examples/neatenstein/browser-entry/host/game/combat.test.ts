import { describe, expect, it } from '@jest/globals';

import {
  fireBolt,
  NEATENSTEIN_BOLT_DAMAGE,
  NEATENSTEIN_MUZZLE_OFFSET_CELLS,
} from './combat';
import {
  NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
  NEATENSTEIN_TEST_ENEMY_HEALTH,
  NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
  NEATENSTEIN_TEST_ENEMY_OFF_BOLT_OFFSET_CELLS,
  NEATENSTEIN_TEST_SEED,
} from './constants';
import { createGameState } from './state';
import type { GameState } from './types';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/combat.ts.
 *
 * Covers AC-106 (traveling plasma bolt) and AC-210 (only the bolt weapon exists).
 */

describe('Neatenstein game combat', () => {
  describe('AC-106: traveling plasma bolt contract', () => {
    it('exports fireBolt', () => {
      expect(typeof fireBolt).toBe('function');
    });

    it('exports a positive bolt damage constant', () => {
      expect(NEATENSTEIN_BOLT_DAMAGE).toBeGreaterThan(0);
    });

    it('exports a positive muzzle offset constant', () => {
      expect(NEATENSTEIN_MUZZLE_OFFSET_CELLS).toBeGreaterThan(0);
    });

    it('returns a bolt object when the weapon fires', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireBolt(state);
      expect(typeof result.bolt).toBe('object');
    });

    it('does not fire when ammo is zero', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = { ...base, player: { ...base.player, ammo: 0 } };
      const result = fireBolt(state);
      expect(result.fired).toBe(false);
    });

    it('consumes one unit of ammo when fired', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireBolt(state);
      expect(result.state.player.ammo).toBe(state.player.ammo - 1);
    });

    it('appends a bolt to GameState.bolts when fired', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireBolt(state);
      expect(result.state.bolts?.length ?? 0).toBe(
        (state.bolts?.length ?? 0) + 1,
      );
    });

    it('returns a bolt object when spawning ahead of the player', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0 },
      };
      const result = fireBolt(state);
      expect(result.bolt).not.toBeNull();
    });

    it('spawns the bolt ahead of the player along the aim direction', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0 },
      };
      const result = fireBolt(state);
      expect(result.bolt!.position.x).toBeGreaterThan(
        state.player.position.x + NEATENSTEIN_MUZZLE_OFFSET_CELLS - 0.01,
      );
    });

    it('damages the first enemy in the bolt path', () => {
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
      const result = fireBolt(state);
      expect(result.state.enemies[0].health).toBe(
        NEATENSTEIN_TEST_ENEMY_HEALTH - NEATENSTEIN_BOLT_DAMAGE,
      );
    });

    it('skips an enemy behind the bolt origin', () => {
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
      const result = fireBolt(state);
      expect(result.state.enemies[0].health).toBe(
        NEATENSTEIN_TEST_ENEMY_HEALTH,
      );
    });

    it('counts a kill when the bolt reduces an enemy to zero health', () => {
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
            health: NEATENSTEIN_BOLT_DAMAGE,
          },
        ],
      };
      const result = fireBolt(state);
      expect(result.state.kills).toBe(1);
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
      const result = fireBolt(state);
      expect(result.bolt).not.toBeNull();
    });

    it('misses an enemy too far perpendicular to the bolt', () => {
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
                NEATENSTEIN_TEST_ENEMY_OFF_BOLT_OFFSET_CELLS,
            },
            health: NEATENSTEIN_TEST_ENEMY_HEALTH,
          },
        ],
      };
      const result = fireBolt(state);
      expect(result.state.enemies[0].health).toBe(
        NEATENSTEIN_TEST_ENEMY_HEALTH,
      );
    });

    it('hits the wall when no enemies are in range', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireBolt(state);
      expect(result.state.impacts.length).toBeGreaterThan(0);
    });

    it('returns a defined bolt when recording creation time', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireBolt(state);
      expect(result.bolt).toBeDefined();
    });

    it('records bolt creation time at the current simulation time', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireBolt(state);
      expect(result.bolt!.createdAtMs).toBe(state.simTimeMs);
    });

    it('records a finite bolt creation time', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireBolt(state);
      expect(Number.isFinite(result.bolt!.createdAtMs)).toBe(true);
    });

    describe('records bolt travel time on wall impacts so rendering can gate arrival', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireBolt(state);

      it('produces at least one wall impact', () => {
        expect(result.state.impacts.length).toBeGreaterThan(0);
      });

      result.state.impacts.forEach((impact, index) => {
        it(`records the canonical travel duration on wall impact ${index}`, () => {
          expect(impact.boltTravelTimeMs).toBe(
            NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
          );
        });

        it(`records a finite travel duration on wall impact ${index}`, () => {
          expect(Number.isFinite(impact.boltTravelTimeMs)).toBe(true);
        });
      });
    });

    it('caps bolt targetDistance at the max range', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireBolt(state);
      expect(result.bolt!.targetDistance).toBeLessThanOrEqual(
        NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
      );
    });

    it('does not damage an enemy beyond the max range', () => {
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state: GameState = {
        ...base,
        player: { ...base.player, angleRad: 0, ammo: base.player.maxAmmo },
        enemies: [
          {
            position: {
              x: base.player.position.x + NEATENSTEIN_BOLT_MAX_RANGE_CELLS + 5,
              y: base.player.position.y,
            },
            health: NEATENSTEIN_TEST_ENEMY_HEALTH,
          },
        ],
      };
      const result = fireBolt(state);
      expect(result.state.enemies[0].health).toBe(
        NEATENSTEIN_TEST_ENEMY_HEALTH,
      );
    });

    it('does not append legacy hitscan tracers', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const result = fireBolt(state);
      expect('tracers' in result.state).toBe(false);
    });

    it('AC-210: does not export the legacy fireNeonBeam helper', async () => {
      const mod = (await import('./combat')) as Record<string, unknown>;
      expect(mod.fireNeonBeam).toBeUndefined();
    });

    it('AC-210: does not export weapon-switching helpers', async () => {
      const mod = (await import('./combat')) as Record<string, unknown>;
      expect(
        mod.switchWeapon ?? mod.weaponIndex ?? mod.weaponState,
      ).toBeUndefined();
    });
  });
});
