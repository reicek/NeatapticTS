import { describe, expect, it } from '@jest/globals';
import {
  applyEnemyDamage,
  fireBolt,
  fireEnemyBolt,
  NEATENSTEIN_BOLT_DAMAGE,
  NEATENSTEIN_MUZZLE_OFFSET_CELLS,
} from './combat';
import {
  NEATENSTEIN_BOLT_HIT_RADIUS_CELLS,
  NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
  NEATENSTEIN_ENEMY_BOLT_DAMAGE,
  NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS,
  NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT,
  NEATENSTEIN_ENEMY_MAX_HEALTH,
  NEATENSTEIN_ENEMY_STUN_DURATION_MS,
  NEATENSTEIN_TEST_ENEMY_HEALTH,
  NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
  NEATENSTEIN_TEST_ENEMY_OFF_BOLT_OFFSET_CELLS,
  NEATENSTEIN_TEST_SEED,
} from './constants';
import { NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS } from '../../constants';
import { castRayDDAFromFlatMap } from '../../renderer/raycast';
import { createGameState } from './state';
import type { BoltState, EnemyImpactSpot, GameState } from './types';

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
            angleRad: 0.5934,
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

describe('AC-11b-002: bolt damage is 20', () => {
  it('exports NEATENSTEIN_BOLT_DAMAGE as 20', () => {
    expect(NEATENSTEIN_BOLT_DAMAGE).toBe(20);
  });
});

describe('AC-11b-001: five non-lethal hits reduce health 100→0', () => {
  it('reduces enemy health by 20 per hit and kills on the 5th hit', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    let state: GameState = {
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
          health: NEATENSTEIN_ENEMY_MAX_HEALTH,
          maxHealth: NEATENSTEIN_ENEMY_MAX_HEALTH,
          stunTimerMs: 0,
        },
      ],
    };

    // Hits 1–4: non-lethal (100→80→60→40→20)
    const expectedHealthAfter = [80, 60, 40, 20];
    for (let i = 0; i < 4; i += 1) {
      // Clear stun so the next hit lands
      state = {
        ...state,
        enemies: state.enemies.map((e) => ({ ...e, stunTimerMs: 0 })),
      };
      state = fireBolt(state).state;
      expect(state.enemies[0].health).toBe(expectedHealthAfter[i]);
    }

    // 5th hit: lethal (20→0)
    state = {
      ...state,
      enemies: state.enemies.map((e) => ({ ...e, stunTimerMs: 0 })),
    };
    state = fireBolt(state).state;
    expect(state.enemies[0].health).toBe(0);
    expect(state.kills).toBe(1);
  });
});

describe('AC-11b-006: invincibility during stun', () => {
  it('skips damage when stunTimerMs > 0', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: { x: 10, y: 10 },
          health: 100,
          stunTimerMs: 100,
        },
      ],
    };
    const result = applyEnemyDamage(state, 0);
    expect(result).toBe(state);
  });

  it('applies damage when stunTimerMs is 0', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: { x: 10, y: 10 },
          health: 100,
          stunTimerMs: 0,
        },
      ],
    };
    const result = applyEnemyDamage(state, 0);
    expect(result.enemies[0].health).toBe(80);
  });

  it('applies damage when stunTimerMs is undefined', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: { x: 10, y: 10 },
          health: 100,
        },
      ],
    };
    const result = applyEnemyDamage(state, 0);
    expect(result.enemies[0].health).toBe(80);
  });
});

describe('AC-11b-003: hit-stun on non-lethal hits', () => {
  it('sets stunTimerMs on a non-lethal hit', () => {
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
          health: 100,
          stunTimerMs: 0,
        },
      ],
    };
    const result = fireBolt(state);
    expect(result.state.enemies[0].stunTimerMs).toBe(
      NEATENSTEIN_ENEMY_STUN_DURATION_MS,
    );
  });

  it('does not set stunTimerMs on a lethal hit', () => {
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
          stunTimerMs: 0,
        },
      ],
    };
    const result = fireBolt(state);
    expect(result.state.enemies[0].stunTimerMs).toBe(0);
  });
});

describe('AC-11b-005: pushback on non-lethal hit', () => {
  it('pushes the enemy away from the player on a non-lethal hit', () => {
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
          health: 100,
          stunTimerMs: 0,
        },
      ],
    };
    const beforeX = state.enemies[0].position.x;
    const result = fireBolt(state);
    const afterX = result.state.enemies[0].position.x;
    // Enemy is to the right of the player; pushback should move it further right.
    expect(afterX).toBeGreaterThanOrEqual(beforeX);
  });

  it('does not push back on a lethal hit', () => {
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
          stunTimerMs: 0,
        },
      ],
    };
    const beforeX = state.enemies[0].position.x;
    const result = fireBolt(state);
    expect(result.state.enemies[0].position.x).toBe(beforeX);
  });
});

describe('AC-11c-001: EnemyImpactSpot creation in fireBolt hitscan path', () => {
  it('creates an EnemyImpactSpot when a bolt hits an enemy', () => {
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

    // AC-11c-001: EnemyImpactSpot created at enemy position
    expect(result.state.enemyImpacts).toBeDefined();
    expect(result.state.enemyImpacts!.length).toBe(1);
    const impact = result.state.enemyImpacts![0];
    expect(impact.position.x).toBe(state.enemies[0].position.x);
    expect(impact.position.y).toBe(state.enemies[0].position.y);
  });

  it('sets boltTravelTimeMs to NEATENSTEIN_BOLT_TRAVEL_DURATION_MS in the hitscan path', () => {
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
    const impact = result.state.enemyImpacts![0];

    expect(impact.boltTravelTimeMs).toBe(NEATENSTEIN_BOLT_TRAVEL_DURATION_MS);
  });

  it('sets createdAtMs to the state simTimeMs', () => {
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
    const impact = result.state.enemyImpacts![0];

    expect(impact.createdAtMs).toBe(state.simTimeMs);
  });

  it('sets lifetimeMs to NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS', () => {
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
    const impact = result.state.enemyImpacts![0];

    expect(impact.lifetimeMs).toBe(NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS);
  });

  it('does not create an EnemyImpactSpot when the bolt hits a wall', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    // No enemies → bolt will hit a wall, not an enemy
    const result = fireBolt(base);

    expect(result.state.enemyImpacts ?? []).toHaveLength(0);
  });

  it('caps enemyImpacts at NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    // Pre-fill enemyImpacts to near the cap
    const existing: EnemyImpactSpot[] = Array.from(
      { length: NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT },
      () => ({
        position: { x: 0, y: 0 },
        createdAtMs: 0,
        lifetimeMs: NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
        boltTravelTimeMs: 0,
      }),
    );
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
      enemyImpacts: existing,
    };
    const result = fireBolt(state);

    // After adding 1, the cap should drop the oldest, keeping exactly MAX
    expect(result.state.enemyImpacts!.length).toBe(
      NEATENSTEIN_ENEMY_IMPACT_MAX_CONCURRENT,
    );
  });
});

describe('AC-11-enemy-fire coverage: stun invincibility branch (line 323)', () => {
  it('returns the same state when enemy has active stun timer', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: { x: 5, y: 5 },
          health: 100,
          stunTimerMs: 250,
        },
      ],
    };
    const result = applyEnemyDamage(state, 0);
    // Stun invincibility: damage is skipped, state returned unchanged
    expect(result).toBe(state);
    expect(result.enemies[0].health).toBe(100);
  });
});

describe('AC-11-enemy-fire coverage: iteration 3 branch gaps', () => {
  it('skips pushback when enemy is at the exact same position as player (dist=0, line 359)', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: { ...base.player.position },
          health: 100,
          stunTimerMs: 0,
        },
      ],
    };
    const result = applyEnemyDamage(state, 0);
    // dist=0 → pushback skipped, but damage still applied
    expect(result.enemies[0].health).toBe(100 - NEATENSTEIN_BOLT_DAMAGE);
    // Position unchanged (no pushback)
    expect(result.enemies[0].position.x).toBe(base.player.position.x);
    expect(result.enemies[0].position.y).toBe(base.player.position.y);
  });

  it('uses default damage when input.damage is not provided (line 435)', () => {
    const bolt = fireEnemyBolt(
      { origin: { x: 10, y: 10 }, direction: { x: 1, y: 0 } },
      1000,
    );
    expect(bolt.damage).toBe(NEATENSTEIN_ENEMY_BOLT_DAMAGE);
    expect(bolt.damage).toBe(10);
  });
});

describe('AC-801-S05-002: ammo pickup spawn on enemy kill', () => {
  it('spawns an ammo pickup at the enemy position when the enemy is killed', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const enemyPos = { x: 5, y: 5 };
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: enemyPos,
          health: NEATENSTEIN_BOLT_DAMAGE, // 20 — one shot kills
        },
      ],
    };
    const result = applyEnemyDamage(state, 0);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- ammoPickups not yet in GameState; red test
    const pickups = (result as any).ammoPickups as unknown[] | undefined;
    expect(pickups).toBeDefined();
    expect(pickups).toHaveLength(1);
    expect(pickups![0]).toMatchObject({ position: enemyPos, active: true });
  });

  it('does not spawn an ammo pickup on a non-lethal hit', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: { x: 5, y: 5 },
          health: 100, // > 20, so non-lethal
        },
      ],
    };
    const result = applyEnemyDamage(state, 0);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- ammoPickups not yet in GameState; red test
    const pickups = (result as any).ammoPickups as unknown[] | undefined;
    // On non-lethal hits, no pickup should be spawned
    expect(pickups ?? []).toHaveLength(0);
  });

  it('uses ?? [] fallback when ammoPickups is undefined on a non-lethal hit', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    // Strip ammoPickups so the property is entirely absent from the state,
    // exercising the `state.ammoPickups ?? []` nullish branch at line 403.
    const { ammoPickups: _stripped, ...stateWithoutPickups } = base;
    void _stripped;
    const state: GameState = {
      ...stateWithoutPickups,
      enemies: [
        {
          position: { x: 5, y: 5 },
          health: 100, // > NEATENSTEIN_BOLT_DAMAGE, so non-lethal
        },
      ],
    };
    const result = applyEnemyDamage(state, 0);
    // The ?? [] fallback fires because state.ammoPickups is undefined.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- ammoPickups not yet in GameState; red test
    const pickups = (result as any).ammoPickups as unknown[] | undefined;
    expect(pickups).toEqual([]);
  });

  it('uses ?? [] fallback when ammoPickups is undefined on a KILL', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    // Strip ammoPickups so the property is entirely absent from the state,
    // exercising the `state.ammoPickups ?? []` nullish branch at line 395
    // (the kill-path spread).
    const { ammoPickups: _stripped, ...stateWithoutPickups } = base;
    void _stripped;
    const enemyPos = { x: 7, y: 7 };
    const state: GameState = {
      ...stateWithoutPickups,
      enemies: [
        {
          position: enemyPos,
          health: NEATENSTEIN_BOLT_DAMAGE, // 20 — one shot kills
        },
      ],
    };
    const result = applyEnemyDamage(state, 0);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- ammoPickups not yet in GameState; red test
    const pickups = (result as any).ammoPickups as unknown[] | undefined;
    // The kill path spawns a new pickup despite ammoPickups being undefined,
    // because the ?? [] fallback yields an empty array that is then spread
    // alongside the new pickup.
    expect(pickups).toBeDefined();
    expect(pickups).toHaveLength(1);
    expect(pickups![0]).toMatchObject({ position: enemyPos, active: true });
  });

  it('spawned pickup has the correct amount field', () => {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: { x: 5, y: 5 },
          health: NEATENSTEIN_BOLT_DAMAGE,
        },
      ],
    };
    const result = applyEnemyDamage(state, 0);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- ammoPickups not yet in GameState; red test
    const pickups = (result as any).ammoPickups as
      Array<Record<string, unknown>> | undefined;
    expect(pickups).toBeDefined();
    expect(pickups!.length).toBeGreaterThan(0);
    expect(typeof pickups![0].amount).toBe('number');
  });
});
