import { describe, expect, it } from '@jest/globals';
import {
  applyEnemyDamage,
  fireBolt,
  NEATENSTEIN_BOLT_DAMAGE,
  NEATENSTEIN_MUZZLE_OFFSET_CELLS,
} from './combat';
import {
  NEATENSTEIN_BOLT_HIT_RADIUS_CELLS,
  NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
  NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS,
  NEATENSTEIN_TEST_ENEMY_HEALTH,
  NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
  NEATENSTEIN_TEST_ENEMY_OFF_BOLT_OFFSET_CELLS,
  NEATENSTEIN_TEST_SEED,
} from './constants';
import { castRayDDAFromFlatMap } from '../../renderer/raycast';
import { createGameState } from './state';
import type { BoltState, GameState } from './types';

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

    describe('AC-10.2c-003: fireBolt ignores inactive enemies', () => {
      it('does not damage an inactive enemy on the bolt path', () => {
        const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        const state: GameState = {
          ...base,
          player: {
            ...base.player,
            angleRad: 0,
            ammo: base.player.maxAmmo,
          },
          enemies: [
            {
              position: {
                x:
                  base.player.position.x +
                  NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
                y: base.player.position.y,
              },
              health: NEATENSTEIN_TEST_ENEMY_HEALTH,
              active: false,
            },
          ],
        };
        const result = fireBolt(state);
        expect(result.state.enemies[0].health).toBe(
          NEATENSTEIN_TEST_ENEMY_HEALTH,
        );
      });
    });

    describe('Coverage: edge cases', () => {
      it('returns Infinity when a ray exceeds the render-distance cap without hitting a wall', () => {
        const flatMap = new Uint8Array(64 * 64).fill(0);
        const hit = castRayDDAFromFlatMap(flatMap, 64, 32.5, 32.5, 1, 0);
        expect(hit.perpWallDist).toBe(Number.POSITIVE_INFINITY);
      });

      it('appends a bolt to a non-empty existing bolts array', () => {
        const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        const existingBolt: BoltState = {
          position: { x: 0, y: 0 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 1,
          active: true,
          createdAtMs: 0,
          origin: { x: 0, y: 0 },
          targetDistance: 1,
        };
        const state: GameState = {
          ...base,
          player: { ...base.player, ammo: base.player.maxAmmo },
          bolts: [existingBolt],
        };
        const result = fireBolt(state);
        expect(result.state.bolts).toHaveLength(2);
        expect(result.state.bolts![0]).toBe(existingBolt);
      });

      it('does not damage enemies outside the hit index', () => {
        const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        const state: GameState = {
          ...base,
          enemies: [
            {
              position: { x: 0, y: 0 },
              health: NEATENSTEIN_BOLT_DAMAGE,
            },
            {
              position: { x: 1, y: 0 },
              health: NEATENSTEIN_TEST_ENEMY_HEALTH,
            },
          ],
        };
        const after = applyEnemyDamage(state, 0);
        expect(after.enemies[0].health).toBe(0);
        expect(after.enemies[1].health).toBe(NEATENSTEIN_TEST_ENEMY_HEALTH);
      });

      it('caps the bolt at max range when the wall ray exceeds the render-distance cap', () => {
        const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        const state: GameState = {
          ...base,
          player: {
            ...base.player,
            position: { x: 52.5, y: 52.5 },
            angleRad: 0.5934119456780721,
            ammo: base.player.maxAmmo,
          },
        };
        const result = fireBolt(state);
        expect(result.state.impacts.length).toBe(0);
        expect(result.bolt!.targetDistance).toBe(
          NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
        );
      });

      it('creates a bolts array when none exists', () => {
        const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        const { bolts, ...stateWithoutBolts } = {
          ...base,
          player: { ...base.player, ammo: base.player.maxAmmo },
        };
        void bolts;
        const result = fireBolt(stateWithoutBolts as GameState);
        expect(result.state.bolts).toHaveLength(1);
      });
    });
  });
});

describe('AC-10.2d-006: bolt collision radius matches enemy body radius', () => {
  it('uses the enemy body radius as the bolt hit radius', () => {
    expect(NEATENSTEIN_BOLT_HIT_RADIUS_CELLS).toBe(
      NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS,
    );
  });

  it('misses an enemy just outside the body radius', () => {
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
              NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS +
              0.01,
          },
          health: NEATENSTEIN_TEST_ENEMY_HEALTH,
        },
      ],
    };
    const result = fireBolt(state);
    expect(result.state.enemies[0].health).toBe(NEATENSTEIN_TEST_ENEMY_HEALTH);
  });

  it('records bolt radius and hitEnemyIndex on the spawned bolt', () => {
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
    const bolt = result.bolt as unknown as Record<string, unknown>;
    expect(typeof bolt.radius).toBe('number');
    expect(Number.isInteger(bolt.hitEnemyIndex)).toBe(true);
  });
});
