import { describe, expect, it } from '@jest/globals';
import { NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS } from '../../constants';
import { buildNeatensteinMap, createCollisionMap } from '../../renderer/map';
import { fireBolt } from './combat';
import {
  NEATENSTEIN_BOLT_DAMAGE,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_TEST_ENEMY_HEALTH,
  NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
  NEATENSTEIN_TEST_SEED,
} from './constants';
import { createGameState } from './state';
import type { BoltState, EnemyState, ImpactSpot } from './types';

/**
 * Minimal impact-spot factory for tests that only care about lifetime semantics.
 *
 * @param lifetimeMs - Remaining visibility time for the impact spot.
 * @returns A valid {@link ImpactSpot} with placeholder geometry.
 */
function makeImpact(lifetimeMs: number): ImpactSpot {
  return {
    wallHit: { mapX: 1, mapY: 1, side: 0, wallX: 0.5 },
    position: { x: 1, y: 1 },
    createdAtMs: 0,
    lifetimeMs,
    perpWallDist: 1,
    boltTravelTimeMs: 0,
  };
}

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/tick.ts.
 *
 * Covers AC-201: a deterministic tick function advances world state by one
 * fixed timestep given an input snapshot.
 */

describe('Neatenstein game tick', () => {
  describe('AC-201: deterministic fixed-timestep tick contract', () => {
    it('exports gameTick', async () => {
      const mod = (await import('./tick.ts')) as Record<string, unknown>;
      expect(typeof mod.gameTick).toBe('function');
    });

    it('exports the fixed timestep constant', async () => {
      const mod = (await import('./tick.ts')) as Record<string, unknown>;
      expect(typeof mod.NEATENSTEIN_FIXED_TIMESTEP_MS).toBe('number');
    });

    it('advances simTime by exactly the fixed timestep', async () => {
      const { createGameState, gameTick, NEATENSTEIN_FIXED_TIMESTEP_MS } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const snapshot = {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: false,
        dash: false,
      };
      const next = gameTick(state, snapshot);
      expect(next.simTimeMs).toBe(
        state.simTimeMs + NEATENSTEIN_FIXED_TIMESTEP_MS,
      );
    });

    it('produces identical state for identical seed and input snapshot', async () => {
      const { createGameState, gameTick } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const snapshot = {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: false,
        dash: false,
      };
      const stateA = createGameState({ seed: 3 });
      const stateB = createGameState({ seed: 3 });
      expect(gameTick(stateA, snapshot).player.position).toEqual(
        gameTick(stateB, snapshot).player.position,
      );
    });

    it('applies a dash when the input requests one', async () => {
      const { createGameState, gameTick } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const next = gameTick(state, { dash: true });
      expect(next.player.dashTimeRemainingMs).toBeGreaterThan(
        state.player.dashTimeRemainingMs,
      );
    });

    it('falls back to the fixed timestep for invalid dt values', async () => {
      const { createGameState, gameTick, NEATENSTEIN_FIXED_TIMESTEP_MS } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const next = gameTick(state, {}, undefined, Number.NaN);
      expect(next.simTimeMs).toBe(
        state.simTimeMs + NEATENSTEIN_FIXED_TIMESTEP_MS,
      );
    });

    it('uses an explicit collision map when provided', async () => {
      const { createGameState, gameTick } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const flatMap = buildNeatensteinMap(state.seed);
      const collisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);
      const next = gameTick(state, {}, collisionMap);
      expect(next.simTimeMs).toBe(
        state.simTimeMs + NEATENSTEIN_FIXED_TIMESTEP_MS,
      );
    });

    it('rotates the player by a non-zero look delta', async () => {
      const { createGameState, gameTick } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const lookDelta = 0.1;
      const next = gameTick(state, { lookDelta });
      expect(next.player.angleRad).toBeCloseTo(
        state.player.angleRad + lookDelta,
        6,
      );
    });

    it('filters bolts that expire during the tick from next state', async () => {
      const { createGameState, gameTick, NEATENSTEIN_BOLT_TRAVEL_DURATION_MS } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const bolt = {
        position: { x: 0, y: 0 },
        direction: { x: 1, y: 0 },
        speedCellsPerSecond: 1,
        active: true,
        createdAtMs: 0,
      };
      const stateWithBolt = { ...state, bolts: [bolt] };
      const next = gameTick(
        stateWithBolt,
        { move: { x: 0, y: 0 }, lookDelta: 0 },
        undefined,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
      );
      expect(next.bolts.length).toBe(0);
    });

    it('keeps a freshly fired bolt active in next state', async () => {
      const { createGameState, gameTick } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const { NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./constants.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const next = gameTick(state, {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: true,
      });
      expect(next.bolts.length).toBeGreaterThan(0);
      expect(next.bolts.every((b: { active: boolean }) => b.active)).toBe(true);
      expect(next.gun.recoilOffset).toBe(NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX);
    });

    it('retains existing bolts that remain active through the tick', async () => {
      const { createGameState, gameTick } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const bolt = {
        position: { x: 0, y: 0 },
        direction: { x: 1, y: 0 },
        speedCellsPerSecond: 1,
        active: true,
        createdAtMs: state.simTimeMs,
      };
      const stateWithBolt = { ...state, bolts: [bolt] };
      const next = gameTick(
        stateWithBolt,
        { move: { x: 0, y: 0 }, lookDelta: 0 },
        undefined,
        16,
      );
      expect(next.bolts.length).toBe(1);
      expect(next.bolts[0].active).toBe(true);
    });

    it('tolerates a missing bolts array during the tick', async () => {
      const { createGameState, gameTick } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const stateWithoutBolts = { ...state, bolts: undefined };
      const next = gameTick(
        stateWithoutBolts,
        { move: { x: 0, y: 0 }, lookDelta: 0 },
        undefined,
        16,
      );
      expect(next.bolts).toEqual([]);
    });

    it('tolerates a missing gun state during the tick', async () => {
      const { createGameState, gameTick } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const stateWithoutGun = { ...state, gun: undefined };
      const next = gameTick(
        stateWithoutGun,
        { move: { x: 0, y: 0 }, lookDelta: 0 },
        undefined,
        16,
      );
      expect(next.gun.recoilOffset).toBe(0);
    });

    it('does not set gun recoil when firing with no ammo', async () => {
      const { createGameState, gameTick } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const emptyAmmoState = {
        ...state,
        player: { ...state.player, ammo: 0 },
      };
      const next = gameTick(emptyAmmoState, {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: true,
      });
      expect(next.bolts.length).toBe(0);
      expect(next.gun.recoilOffset).toBe(0);
    });
  });

  describe('ageImpacts helper', () => {
    it('removes impact spots whose lifetime has expired', async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
      const { ageImpacts } = (await import('./tick.ts')) as Record<string, any>;
      const impacts = [
        makeImpact(NEATENSTEIN_FIXED_TIMESTEP_MS),
        makeImpact(NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS),
      ];
      const aged = ageImpacts(impacts, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(aged.map((i: ImpactSpot) => i.lifetimeMs)).toEqual([
        NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS - NEATENSTEIN_FIXED_TIMESTEP_MS,
      ]);
    });
  });

  describe('AC-107: bolt movement and expiry', () => {
    it('exports updateBolts', async () => {
      const { updateBolts } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      expect(typeof updateBolts).toBe('function');
    });

    it('advances a bolt by speed multiplied by dt', async () => {
      const { updateBolts, NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const bolts = [
        {
          position: { x: 0, y: 0 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
          active: true,
          createdAtMs: 0,
        },
      ];
      const next = updateBolts(bolts, 1000, 100);
      expect(next[0].position.x).toBeCloseTo(
        NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
        6,
      );
    });

    it('stops movement but keeps a bolt active when it leaves the world bounds', async () => {
      const { updateBolts, NEATENSTEIN_BOLT_TRAVEL_DURATION_MS } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const bolts = [
        {
          position: { x: -1, y: 0 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 1,
          active: true,
          createdAtMs: 0,
        },
      ];
      const next = updateBolts(
        bolts,
        16,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS - 1,
      );
      expect(next[0].active).toBe(true);
      expect(next[0].position.x).toBeCloseTo(-1, 6);
    });

    it('removes inactive bolts from the returned array', async () => {
      const { updateBolts } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const bolts = [
        {
          position: { x: 0, y: 0 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 1,
          active: false,
          createdAtMs: 0,
        },
      ];
      const next = updateBolts(bolts, 16, 16);
      expect(next.length).toBe(0);
    });

    it('stops movement but keeps a bolt active when it exceeds the max travel range', async () => {
      const { updateBolts, NEATENSTEIN_BOLT_TRAVEL_DURATION_MS } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const { NEATENSTEIN_BOLT_MAX_RANGE_CELLS } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./constants.ts')) as Record<string, any>;
      const bolts = [
        {
          position: { x: NEATENSTEIN_BOLT_MAX_RANGE_CELLS, y: 0 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 1,
          active: true,
          createdAtMs: 0,
          origin: { x: 0, y: 0 },
        },
      ];
      const next = updateBolts(
        bolts,
        16,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS - 1,
      );
      expect(next[0].active).toBe(true);
      expect(next[0].position.x).toBeCloseTo(
        NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
        6,
      );
    });

    it('keeps a bolt active just before its screen travel duration expires', async () => {
      const { updateBolts, NEATENSTEIN_BOLT_TRAVEL_DURATION_MS } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const { NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./constants.ts')) as Record<string, any>;
      const bolts = [
        {
          position: { x: 0, y: 0 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
          active: true,
          createdAtMs: 0,
        },
      ];
      const stillAlive = updateBolts(
        bolts,
        16,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS - 1,
      );
      expect(stillAlive[0].active).toBe(true);
    });

    it('deactivates a bolt once its screen travel duration expires', async () => {
      const { updateBolts, NEATENSTEIN_BOLT_TRAVEL_DURATION_MS } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const { NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./constants.ts')) as Record<string, any>;
      const bolts = [
        {
          position: { x: 0, y: 0 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
          active: true,
          createdAtMs: 0,
        },
      ];
      const expired = updateBolts(
        bolts,
        16,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
      );
      expect(expired[0].active).toBe(false);
    });
  });

  describe('AC-107: gun recoil decay', () => {
    it('exports decayGunRecoil', async () => {
      const { decayGunRecoil } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      expect(typeof decayGunRecoil).toBe('function');
    });

    it('reduces recoil offset toward zero over time', async () => {
      const { decayGunRecoil } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const gun = { recoilOffset: 10 };
      const next = decayGunRecoil(gun, 16);
      expect(next.recoilOffset).toBeLessThan(gun.recoilOffset);
    });
  });
});

describe('AC-10.2d-001: traveling bolt collides with enemy and stops at impact point', () => {
  function createEmptyCollisionMap(): { isSolid: () => false } {
    return { isSolid: () => false };
  }

  it('deactivates a bolt at the enemy impact point', async () => {
    const { updateBolts } = (await import('./tick.ts')) as Record<
      string,
      unknown
    >;
    const collisionMap = createEmptyCollisionMap();
    const bolt = {
      position: { x: 0, y: 0 },
      direction: { x: 1, y: 0 },
      speedCellsPerSecond: 36,
      active: true,
      createdAtMs: 0,
      origin: { x: 0, y: 0 },
    };
    const enemies = [{ position: { x: 2, y: 0 }, health: 100, active: true }];
    const next = (
      updateBolts as unknown as (
        bolts: unknown[],
        dtMs: number,
        currentTimeMs: number,
        collisionMap?: unknown,
        enemies?: unknown[],
      ) => Array<{
        active: boolean;
        position: { x: number; y: number };
        [key: string]: unknown;
      }>
    )([bolt], 1000, 100, collisionMap, enemies);
    const updated = next[0];
    const hitEnemyIndex = updated.hitEnemyIndex;
    expect(updated.active === false || hitEnemyIndex === 0).toBe(true);
    expect(updated.position.x).toBeCloseTo(2, 0);
  });

  it('records a finite bolt radius on the spawned bolt', () => {
    const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const stateWithAmmo = {
      ...state,
      player: { ...state.player, ammo: state.player.maxAmmo },
      enemies: [
        {
          position: {
            x:
              state.player.position.x +
              NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
            y: state.player.position.y,
          },
          health: NEATENSTEIN_TEST_ENEMY_HEALTH,
        },
      ],
    };
    const result = fireBolt(stateWithAmmo);
    expect(result.bolt).not.toBeNull();
    expect(typeof result.bolt!.radius).toBe('number');
    expect(Number.isFinite(result.bolt!.radius)).toBe(true);
  });

  it('skips a dead enemy in the bolt path', async () => {
    const { updateBolts } = (await import('./tick.ts')) as Record<
      string,
      unknown
    >;
    const collisionMap = createEmptyCollisionMap();
    const bolt: BoltState = {
      position: { x: 0, y: 0 },
      direction: { x: 1, y: 0 },
      speedCellsPerSecond: 36,
      active: true,
      createdAtMs: 0,
      origin: { x: 0, y: 0 },
    };
    const enemies: EnemyState[] = [
      { position: { x: 2, y: 0 }, health: 0, active: true } as EnemyState,
    ];
    const next = (
      updateBolts as unknown as (
        bolts: BoltState[],
        dtMs: number,
        currentTimeMs: number,
        collisionMap?: { isSolid: () => boolean },
        enemies?: EnemyState[],
      ) => BoltState[]
    )([bolt], 1000, 100, collisionMap, enemies);
    expect(next[0].active).toBe(true);
    expect(next[0].hitEnemyIndex).toBeUndefined();
  });

  it('skips an enemy that sits behind the bolt origin', async () => {
    const { updateBolts } = (await import('./tick.ts')) as Record<
      string,
      unknown
    >;
    const collisionMap = createEmptyCollisionMap();
    const bolt: BoltState = {
      position: { x: 10, y: 0 },
      direction: { x: 1, y: 0 },
      speedCellsPerSecond: 36,
      active: true,
      createdAtMs: 0,
      origin: { x: 10, y: 0 },
    };
    const enemies: EnemyState[] = [
      { position: { x: 5, y: 0 }, health: 100, active: true } as EnemyState,
    ];
    const next = (
      updateBolts as unknown as (
        bolts: BoltState[],
        dtMs: number,
        currentTimeMs: number,
        collisionMap?: { isSolid: () => boolean },
        enemies?: EnemyState[],
      ) => BoltState[]
    )([bolt], 1000, 100, collisionMap, enemies);
    expect(next[0].active).toBe(true);
    expect(next[0].hitEnemyIndex).toBeUndefined();
  });

  it('hits the nearest enemy when multiple enemies are on the bolt path', async () => {
    const { updateBolts } = (await import('./tick.ts')) as Record<
      string,
      unknown
    >;
    const collisionMap = createEmptyCollisionMap();
    const bolt: BoltState = {
      position: { x: 0, y: 0 },
      direction: { x: 1, y: 0 },
      speedCellsPerSecond: 36,
      active: true,
      createdAtMs: 0,
      origin: { x: 0, y: 0 },
    };
    const enemies: EnemyState[] = [
      { position: { x: 4, y: 0 }, health: 100, active: true } as EnemyState,
      { position: { x: 2, y: 0 }, health: 100, active: true } as EnemyState,
    ];
    const next = (
      updateBolts as unknown as (
        bolts: BoltState[],
        dtMs: number,
        currentTimeMs: number,
        collisionMap?: { isSolid: () => boolean },
        enemies?: EnemyState[],
      ) => BoltState[]
    )([bolt], 1000, 100, collisionMap, enemies);
    expect(next[0].active).toBe(false);
    expect(next[0].hitEnemyIndex).toBe(1);
  });

  it('keeps the nearest enemy index when a farther enemy is checked later', async () => {
    const { updateBolts } = (await import('./tick.ts')) as Record<
      string,
      unknown
    >;
    const collisionMap = createEmptyCollisionMap();
    const bolt: BoltState = {
      position: { x: 0, y: 0 },
      direction: { x: 1, y: 0 },
      speedCellsPerSecond: 36,
      active: true,
      createdAtMs: 0,
      origin: { x: 0, y: 0 },
    };
    const enemies: EnemyState[] = [
      { position: { x: 2, y: 0 }, health: 100, active: true } as EnemyState,
      { position: { x: 4, y: 0 }, health: 100, active: true } as EnemyState,
    ];
    const next = (
      updateBolts as unknown as (
        bolts: BoltState[],
        dtMs: number,
        currentTimeMs: number,
        collisionMap?: { isSolid: () => boolean },
        enemies?: EnemyState[],
      ) => BoltState[]
    )([bolt], 1000, 100, collisionMap, enemies);
    expect(next[0].active).toBe(false);
    expect(next[0].hitEnemyIndex).toBe(0);
  });

  it('uses the bolt radius when present and misses outside it', async () => {
    const { updateBolts } = (await import('./tick.ts')) as Record<
      string,
      unknown
    >;
    const collisionMap = createEmptyCollisionMap();
    const bolt: BoltState = {
      position: { x: 0, y: 0 },
      direction: { x: 1, y: 0 },
      speedCellsPerSecond: 36,
      active: true,
      createdAtMs: 0,
      origin: { x: 0, y: 0 },
      radius: 0.1,
    };
    const enemies: EnemyState[] = [
      { position: { x: 2, y: 0.5 }, health: 100, active: true } as EnemyState,
    ];
    const next = (
      updateBolts as unknown as (
        bolts: BoltState[],
        dtMs: number,
        currentTimeMs: number,
        collisionMap?: { isSolid: () => boolean },
        enemies?: EnemyState[],
      ) => BoltState[]
    )([bolt], 1000, 100, collisionMap, enemies);
    expect(next[0].active).toBe(true);
    expect(next[0].hitEnemyIndex).toBeUndefined();
  });

  it('applies enemy damage through gameTick when a bolt hits an active enemy', async () => {
    const { gameTick } =
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
      (await import('./tick.ts')) as Record<string, any>;
    const collisionMap = createEmptyCollisionMap();
    const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    // Suppress wave spawning so the pre-placed enemy index stays at 0.
    const stateWithBoltAndEnemy = {
      ...state,
      spawnCount: 999,
      bolts: [
        {
          position: { x: 5, y: 5 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 36,
          active: true,
          createdAtMs: 0,
          origin: { x: 5, y: 5 },
        },
      ],
      enemies: [
        {
          position: { x: 5.3, y: 5 },
          health: 100,
          active: true,
        },
      ],
    };
    const snapshot = {
      move: { x: 0, y: 0 },
      lookDelta: 0,
      fire: false,
      dash: false,
    };
    const next = gameTick(stateWithBoltAndEnemy, snapshot, collisionMap);
    expect(next.enemies[0].health).toBe(100 - NEATENSTEIN_BOLT_DAMAGE);
    expect(next.bolts).toHaveLength(0);
  });

  it('stops a bolt when it reaches its target distance', async () => {
    const { updateBolts } = (await import('./tick.ts')) as Record<
      string,
      unknown
    >;
    const collisionMap = createEmptyCollisionMap();
    const bolt: BoltState = {
      position: { x: 0, y: 0 },
      direction: { x: 1, y: 0 },
      speedCellsPerSecond: 36,
      active: true,
      createdAtMs: 0,
      origin: { x: 0, y: 0 },
      targetDistance: 5,
    };
    const next = (
      updateBolts as unknown as (
        bolts: BoltState[],
        dtMs: number,
        currentTimeMs: number,
        collisionMap?: { isSolid: () => boolean },
        enemies?: EnemyState[],
      ) => BoltState[]
    )([bolt], 200, 200, collisionMap);
    // Bolt would travel 7.2 cells (≥ targetDistance 5, < maxRange 30, < 300ms)
    // so reachedTarget stops movement while keeping the bolt active.
    expect(next[0].active).toBe(true);
    expect(next[0].position).toEqual({ x: 0, y: 0 });
  });
});
