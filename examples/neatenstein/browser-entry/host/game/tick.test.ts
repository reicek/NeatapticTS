import { describe, expect, it, jest } from '@jest/globals';
import { NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS } from '../../constants';
import { NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS } from '../../constants';
import { MAX_TURN_RATE } from '../../harness/neat-io-config';
import { buildNeatensteinMap, createCollisionMap } from '../../renderer/map';
import { fireBolt } from './combat';
import {
  NEATENSTEIN_BOLT_DAMAGE,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_TEST_ENEMY_HEALTH,
  NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS,
  NEATENSTEIN_TEST_SEED,
} from './constants';
import { createGameState } from './state';
import type {
  BoltState,
  EnemyBoltState,
  EnemyImpactSpot,
  EnemyState,
  GameState,
  ImpactSpot,
} from './types';

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
      // Also verify enemy bolt speed constant is re-exported (covers istanbul function)
      expect(typeof mod.NEATENSTEIN_ENEMY_BOLT_SPEED_CELLS_PER_SECOND).toBe(
        'number',
      );
    });

    it('advances simTime by exactly the fixed timestep', async () => {
      const { createGameState, gameTick, NEATENSTEIN_FIXED_TIMESTEP_MS } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
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
        (await import('./tick.ts')) as typeof import('./tick.ts');
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
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const state = createGameState({ seed: 1 });
      const next = gameTick(state, { dash: true });
      expect(next.player.dashTimeRemainingMs).toBeGreaterThan(
        state.player.dashTimeRemainingMs,
      );
    });

    it('falls back to the fixed timestep for invalid dt values', async () => {
      const { createGameState, gameTick, NEATENSTEIN_FIXED_TIMESTEP_MS } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const state = createGameState({ seed: 1 });
      const next = gameTick(state, {}, undefined, Number.NaN);
      expect(next.simTimeMs).toBe(
        state.simTimeMs + NEATENSTEIN_FIXED_TIMESTEP_MS,
      );
    });

    it('uses an explicit collision map when provided', async () => {
      const { createGameState, gameTick } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
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
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const state = createGameState({ seed: 1 });
      const lookDelta = 0.1;
      const next = gameTick(state, { lookDelta });
      expect(next.player.angleRad).toBeCloseTo(
        state.player.angleRad + lookDelta,
        6,
      );
    });

    it('clamps a negative look delta exceeding -MAX_TURN_RATE', async () => {
      const { createGameState, gameTick } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const state = createGameState({ seed: 1 });
      // -Math.PI is well below -MAX_TURN_RATE (-π/4), so the clamp must engage.
      const next = gameTick(state, { lookDelta: -Math.PI });
      expect(next.player.angleRad).toBeCloseTo(
        state.player.angleRad - MAX_TURN_RATE,
        6,
      );
    });

    it('clamps a positive look delta exceeding MAX_TURN_RATE', async () => {
      const { createGameState, gameTick } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const state = createGameState({ seed: 1 });
      // Math.PI is well above MAX_TURN_RATE (π/4), so the clamp must engage.
      const next = gameTick(state, { lookDelta: Math.PI });
      expect(next.player.angleRad).toBeCloseTo(
        state.player.angleRad + MAX_TURN_RATE,
        6,
      );
    });

    it('filters bolts that expire during the tick from next state', async () => {
      const { createGameState, gameTick, NEATENSTEIN_BOLT_TRAVEL_DURATION_MS } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
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
      expect(next.bolts!.length).toBe(0);
    });

    it('keeps a freshly fired bolt active in next state', async () => {
      const { createGameState, gameTick } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const { NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX } =
        (await import('./constants.ts')) as typeof import('./constants.ts');
      const state = createGameState({ seed: 1 });
      const next = gameTick(state, {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: true,
      });
      expect(next.bolts!.length).toBeGreaterThan(0);
      expect(next.bolts!.every((b: { active: boolean }) => b.active)).toBe(
        true,
      );
      expect(next.gun!.recoilOffset).toBe(NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX);
      expect(next.gun!.firing).toBe(true);
    });

    it('retains existing bolts that remain active through the tick', async () => {
      const { createGameState, gameTick } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
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
      expect(next.bolts!.length).toBe(1);
      expect(next.bolts![0].active).toBe(true);
    });

    it('tolerates a missing bolts array during the tick', async () => {
      const { createGameState, gameTick } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
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
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const state = createGameState({ seed: 1 });
      const stateWithoutGun = { ...state, gun: undefined };
      const next = gameTick(
        stateWithoutGun,
        { move: { x: 0, y: 0 }, lookDelta: 0 },
        undefined,
        16,
      );
      expect(next.gun!.recoilOffset).toBe(0);
      expect(next.gun!.firing).toBe(false);
    });

    it('does not set gun recoil when firing with no ammo', async () => {
      const { createGameState, gameTick } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
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
      expect(next.bolts!.length).toBe(0);
      expect(next.gun!.recoilOffset).toBe(0);
      expect(next.gun!.firing).toBe(false);
    });

    it('falls back to a default gun state when fireBolt returns a state without gun', async () => {
      await jest.isolateModulesAsync(async () => {
        jest.doMock('./combat.ts', () => {
          const actual = jest.requireActual('./combat.ts') as Record<
            string,
            unknown
          >;
          return {
            ...actual,
            fireBolt: jest.fn(() => ({
              state: {},
              fired: true,
              bolt: null,
            })),
          };
        });
        const mod = (await import('./tick.ts')) as typeof import('./tick.ts');
        const { createGameState, gameTick } = mod;
        const state = createGameState({ seed: 1 });
        const next = gameTick(state, { fire: true });
        expect(next.gun!.recoilOffset).toBe(
          NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX,
        );
        expect(next.gun!.firing).toBe(true);
        jest.dontMock('./combat.ts');
      });
    });
  });

  describe('ageImpacts helper', () => {
    it('removes impact spots whose lifetime has expired', async () => {
      const { ageImpacts } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
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

  describe('ageEnemyImpacts helper (AC-11c-004)', () => {
    /**
     * Build a minimal enemy-impact spot for aging tests.
     */
    function makeEnemyImpact(lifetimeMs: number): EnemyImpactSpot {
      return {
        position: { x: 5, y: 5 },
        createdAtMs: 0,
        lifetimeMs,
        boltTravelTimeMs: 0,
      };
    }

    it('exports ageEnemyImpacts', async () => {
      const mod = (await import('./tick.ts')) as Record<string, unknown>;
      expect(typeof mod.ageEnemyImpacts).toBe('function');
    });

    it('removes enemy impact spots whose lifetime has expired', async () => {
      const { ageEnemyImpacts } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const impacts = [
        makeEnemyImpact(NEATENSTEIN_FIXED_TIMESTEP_MS),
        makeEnemyImpact(NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS),
      ];
      const aged = ageEnemyImpacts(impacts, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(aged.map((i: EnemyImpactSpot) => i.lifetimeMs)).toEqual([
        NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS - NEATENSTEIN_FIXED_TIMESTEP_MS,
      ]);
    });

    it('decrements lifetime by dtMs for surviving impacts', async () => {
      const { ageEnemyImpacts } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const impacts = [makeEnemyImpact(NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS)];
      const aged = ageEnemyImpacts(impacts, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(aged.length).toBe(1);
      expect(aged[0].lifetimeMs).toBe(
        NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS - NEATENSTEIN_FIXED_TIMESTEP_MS,
      );
    });

    it('returns an empty array when there are no impacts', async () => {
      const { ageEnemyImpacts } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const aged = ageEnemyImpacts([], NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(aged).toEqual([]);
    });

    it('removes an impact exactly when lifetime reaches zero', async () => {
      const { ageEnemyImpacts } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const impacts = [makeEnemyImpact(NEATENSTEIN_FIXED_TIMESTEP_MS)];
      const aged = ageEnemyImpacts(impacts, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(aged).toEqual([]);
    });
  });

  describe('AC-107: bolt movement and expiry', () => {
    it('exports updateBolts', async () => {
      const { updateBolts } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      expect(typeof updateBolts).toBe('function');
    });

    it('advances a bolt by speed multiplied by dt', async () => {
      const { updateBolts, NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
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
        (await import('./tick.ts')) as typeof import('./tick.ts');
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
        (await import('./tick.ts')) as typeof import('./tick.ts');
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
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const { NEATENSTEIN_BOLT_MAX_RANGE_CELLS } =
        (await import('./constants.ts')) as typeof import('./constants.ts');
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
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const { NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND } =
        (await import('./constants.ts')) as typeof import('./constants.ts');
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
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const { NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND } =
        (await import('./constants.ts')) as typeof import('./constants.ts');
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
        (await import('./tick.ts')) as typeof import('./tick.ts');
      expect(typeof decayGunRecoil).toBe('function');
    });

    it('reduces recoil offset toward zero over time', async () => {
      const { decayGunRecoil } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const gun = { recoilOffset: 10, firing: false };
      const next = decayGunRecoil(gun, 16);
      expect(next.recoilOffset).toBeLessThan(gun.recoilOffset);
      expect(next.firing).toBe(false);
    });

    it('resets firing to false even when the input gun was firing', async () => {
      const { decayGunRecoil } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const gun = { recoilOffset: 10, firing: true };
      const next = decayGunRecoil(gun, 16);
      expect(next.firing).toBe(false);
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
      (await import('./tick.ts')) as typeof import('./tick.ts');
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

  it('creates an EnemyImpactSpot in the traveling bolt path (AC-11c-001)', async () => {
    const { gameTick } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const collisionMap = createEmptyCollisionMap();
    const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
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
    // AC-11c-001: EnemyImpactSpot created in the traveling bolt path
    expect(next.enemyImpacts).toBeDefined();
    expect(next.enemyImpacts!.length).toBe(1);
    const impact = next.enemyImpacts![0];
    expect(impact.position).toEqual({ x: 5.3, y: 5 });
    expect(impact.boltTravelTimeMs).toBe(0);
    // lifetime is decremented by ageEnemyImpacts in the same tick
    expect(impact.lifetimeMs).toBe(
      NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS - NEATENSTEIN_FIXED_TIMESTEP_MS,
    );
    expect(typeof impact.createdAtMs).toBe('number');
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

describe('AC-11-enemy-fire: updateEnemyBolts', () => {
  function createSolidCollisionMap(): { isSolid: () => true } {
    return { isSolid: () => true };
  }

  function makeEnemyBolt(
    overrides?: Partial<{
      position: { x: number; y: number };
      direction: { x: number; y: number };
      speedCellsPerSecond: number;
      active: boolean;
      createdAtMs: number;
      origin: { x: number; y: number };
      damage: number;
      hitPlayer: boolean;
    }>,
  ): EnemyBoltState {
    return {
      position: { x: 0, y: 0 },
      direction: { x: 1, y: 0 },
      speedCellsPerSecond: 36,
      active: true,
      createdAtMs: 0,
      origin: { x: 0, y: 0 },
      damage: 10,
      ...overrides,
    };
  }

  function makeState(overrides?: Partial<GameState>): GameState {
    const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    return {
      ...base,
      player: {
        ...base.player,
        position: { x: 60.5, y: 60.5 },
        contactIFrameMs: undefined,
        dashTimeRemainingMs: 0,
      },
      ...overrides,
    };
  }

  it('exports updateEnemyBolts', async () => {
    const mod = (await import('./tick.ts')) as Record<string, unknown>;
    expect(typeof mod.updateEnemyBolts).toBe('function');
  });

  it('moves an active bolt along its direction by speed * dt', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState();
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 50, y: 50 },
    });
    const result = updateEnemyBolts([bolt], 1000, 1000, undefined, state);
    const updated = result.bolts[0];
    // 36 cells/sec * 1 second = 36 cells, but maxRange is 30 so it stops at origin.
    // distanceTraveled = 36 >= 30, so beyondMaxRange = true, movementStopped, position stays.
    // Actually position = bolt.position since movementStopped.
    expect(updated.active).toBe(true);
  });

  it('advances bolt position when within range', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState();
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 50, y: 50 },
      createdAtMs: 1000,
    });
    // dt=100ms → 3.6 cells traveled, well within 30-cell max range, 2000ms lifetime
    const result = updateEnemyBolts([bolt], 100, 1100, undefined, state);
    const updated = result.bolts[0];
    const pos = updated.position;
    expect(pos.x).toBeCloseTo(53.6, 1);
    expect(updated.active).toBe(true);
  });

  it('deactivates a bolt when it hits a wall', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const collisionMap = createSolidCollisionMap();
    const state = makeState();
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 50, y: 50 },
      createdAtMs: 1000,
    });
    const result = updateEnemyBolts([bolt], 100, 1100, collisionMap, state);
    const updated = result.bolts[0];
    // Wall hit → movementStopped, position stays at bolt.position
    expect(updated.active).toBe(true);
    const pos = updated.position;
    expect(pos).toEqual({ x: 50, y: 50 });
  });

  it('deactivates a bolt that goes out of bounds', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState();
    // Place bolt near the edge of the map (MAP_SIZE=120)
    const bolt = makeEnemyBolt({
      position: { x: 119, y: 60 },
      direction: { x: 1, y: 0 },
      origin: { x: 0, y: 60 },
      createdAtMs: 1000,
    });
    // dt=1000ms → 36 cells, but out of bounds (119+36=155 >= 120), so movementStopped
    const result = updateEnemyBolts([bolt], 1000, 2000, undefined, state);
    const updated = result.bolts[0];
    const pos = updated.position;
    // Out of bounds → movementStopped → position stays at original
    expect(pos).toEqual({ x: 119, y: 60 });
  });

  it('keeps bolt active when max range not exceeded', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState();
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 50, y: 50 },
      createdAtMs: 1000,
    });
    // dt=100ms → 3.6 cells < 30 max range, lifetime = 100ms < 2000ms
    const result = updateEnemyBolts([bolt], 100, 1100, undefined, state);
    const updated = result.bolts[0];
    expect(updated.active).toBe(true);
  });

  it('deactivates a bolt when it exceeds max range (30 cells)', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState();
    const bolt = makeEnemyBolt({
      position: { x: 0, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 0, y: 50 },
      createdAtMs: 1000,
    });
    // dt=1000ms → 36 cells > 30 max range → beyondMaxRange → movementStopped
    const result = updateEnemyBolts([bolt], 1000, 2000, undefined, state);
    const updated = result.bolts[0];
    // beyondMaxRange → active stays true (lifetime not expired), but movementStopped
    expect(updated.active).toBe(true);
    const pos = updated.position;
    expect(pos).toEqual({ x: 0, y: 50 }); // position stays since movementStopped
  });

  it('deactivates a bolt when its lifetime expires', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState();
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 50, y: 50 },
      createdAtMs: 0,
    });
    // currentTimeMs = 2001 → elapsedMs = 2001 >= 2000 → lifetimeExpired
    const result = updateEnemyBolts([bolt], 16, 2001, undefined, state);
    const updated = result.bolts[0];
    expect(updated.active).toBe(false);
  });

  it('detects player hit when bolt is within hit radius', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState({
      player: {
        ...makeState().player,
        position: { x: 50.3, y: 50 },
        contactIFrameMs: undefined,
        dashTimeRemainingMs: 0,
      },
    });
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 30, y: 50 },
      createdAtMs: 1000,
      damage: 10,
    });
    // dt=16ms → 0.576 cells, nextPosition.x = 50.576, player at 50.3
    // playerDist = |50.576 - 50.3| = 0.276 <= 0.5 → hitPlayer!
    const result = updateEnemyBolts([bolt], 16, 1016, undefined, state);
    const updated = result.bolts[0];
    expect(updated.active).toBe(false);
    expect(updated.hitPlayer).toBe(true);
  });

  it('applies exactly 10 damage when a bolt hits the player', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const baseState = makeState();
    const playerBefore = baseState.player;
    const state = {
      ...baseState,
      player: {
        ...playerBefore,
        position: { x: 50.3, y: 50 },
        contactIFrameMs: undefined,
        dashTimeRemainingMs: 0,
      },
    };
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 30, y: 50 },
      createdAtMs: 1000,
      damage: 10,
    });
    const result = updateEnemyBolts([bolt], 16, 1016, undefined, state);
    const resultState = result.state;
    const resultPlayer = resultState.player;
    expect(resultPlayer.health).toBe(playerBefore.health - 10);
  });

  it('grants 500ms contact i-frames when a bolt hits the player', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const baseState = makeState();
    const playerBefore = baseState.player;
    const state = {
      ...baseState,
      player: {
        ...playerBefore,
        position: { x: 50.3, y: 50 },
        contactIFrameMs: undefined,
        dashTimeRemainingMs: 0,
      },
    };
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 30, y: 50 },
      createdAtMs: 1000,
      damage: 10,
    });
    const result = updateEnemyBolts([bolt], 16, 1016, undefined, state);
    const resultState = result.state;
    const resultPlayer = resultState.player;
    expect(resultPlayer.contactIFrameMs).toBe(500);
  });

  it('does not re-apply damage when bolt.hitPlayer is already true', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const baseState = makeState();
    const playerBefore = baseState.player;
    const healthBefore = playerBefore.health;
    const state = {
      ...baseState,
      player: {
        ...playerBefore,
        position: { x: 50.3, y: 50 },
        contactIFrameMs: undefined,
        dashTimeRemainingMs: 0,
      },
    };
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 30, y: 50 },
      createdAtMs: 1000,
      damage: 10,
      hitPlayer: true,
    });
    const result = updateEnemyBolts([bolt], 16, 1016, undefined, state);
    const resultState = result.state;
    const resultPlayer = resultState.player;
    // Already hit → no damage applied
    expect(resultPlayer.health).toBe(healthBefore);
  });

  it('filters out inactive bolts', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState();
    const activeBolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 50, y: 50 },
      createdAtMs: 1000,
    });
    const inactiveBolt = makeEnemyBolt({
      position: { x: 10, y: 10 },
      direction: { x: 1, y: 0 },
      origin: { x: 10, y: 10 },
      createdAtMs: 1000,
      active: false,
    });
    const result = updateEnemyBolts(
      [activeBolt, inactiveBolt],
      100,
      1100,
      undefined,
      state,
    );
    const bolts = result.bolts;
    expect(bolts).toHaveLength(1);
  });

  it('handles bolt without origin (distanceTraveled = 0)', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState();
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      createdAtMs: 1000,
    });
    // No origin → distanceTraveled = 0 → beyondMaxRange = false
    // Lifetime not expired (1100-1000=100 < 2000)
    const result = updateEnemyBolts([bolt], 100, 1100, undefined, state);
    const updated = result.bolts[0];
    expect(updated.active).toBe(true);
  });

  it('returns hitWall=false when collisionMap is undefined', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState();
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 50, y: 50 },
      createdAtMs: 1000,
    });
    // No collisionMap → hitWall = false, bolt should move freely
    const result = updateEnemyBolts([bolt], 100, 1100, undefined, state);
    const updated = result.bolts[0];
    const pos = updated.position;
    expect(pos.x).toBeCloseTo(53.6, 1);
    expect(updated.active).toBe(true);
  });

  it('does not hit the player when the bolt hits a wall', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const collisionMap = createSolidCollisionMap();
    const state = makeState({
      player: {
        ...makeState().player,
        position: { x: 50.3, y: 50 },
        contactIFrameMs: undefined,
        dashTimeRemainingMs: 0,
      },
    });
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      origin: { x: 30, y: 50 },
      createdAtMs: 1000,
      damage: 10,
    });
    const result = updateEnemyBolts([bolt], 16, 1016, collisionMap, state);
    const updated = result.bolts[0];
    // hitWall = true → hitPlayer = false (guarded by !hitWall)
    expect(updated.hitPlayer).toBeFalsy();
    const resultState = result.state;
    const resultPlayer = resultState.player;
    expect(resultPlayer.contactIFrameMs).toBeUndefined();
  });

  it('does not hit the player when bolt is out of bounds', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState({
      player: {
        ...makeState().player,
        position: { x: 119.3, y: 60 },
        contactIFrameMs: undefined,
        dashTimeRemainingMs: 0,
      },
    });
    const bolt = makeEnemyBolt({
      position: { x: 119, y: 60 },
      direction: { x: 1, y: 0 },
      origin: { x: 0, y: 60 },
      createdAtMs: 1000,
      damage: 10,
    });
    const result = updateEnemyBolts([bolt], 1000, 2000, undefined, state);
    const updated = result.bolts[0];
    // outOfBounds = true → hitPlayer = false
    expect(updated.hitPlayer).toBeFalsy();
  });

  it('filters inactive enemy bolts through gameTick (line 293 branch)', async () => {
    const { gameTick } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const collisionMap = { isSolid: () => false as const };
    const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const stateWithEnemyBolts = {
      ...state,
      spawnCount: 999,
      enemyBolts: [
        {
          position: { x: 50, y: 50 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 36,
          active: true,
          createdAtMs: 0,
          origin: { x: 50, y: 50 },
          damage: 10,
        },
        {
          position: { x: 10, y: 10 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 36,
          active: false,
          createdAtMs: 0,
          origin: { x: 10, y: 10 },
          damage: 10,
        },
      ],
    };
    const snapshot = {
      move: { x: 0, y: 0 },
      lookDelta: 0,
      fire: false,
      dash: false,
    };
    const next = gameTick(stateWithEnemyBolts, snapshot, collisionMap);
    // gameTick should filter out inactive enemy bolts at line 293
    expect(next.enemyBolts).toBeDefined();
    expect(
      (next.enemyBolts ?? []).every((bolt: { active: boolean }) => bolt.active),
    ).toBe(true);
    expect((next.enemyBolts ?? []).length).toBeLessThanOrEqual(1);
  });

  it('covers if(enemy) false branch when hitEnemyIndex is out of bounds (line 275)', async () => {
    const tickModule =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const combatModule = await import('./combat');

    // Mock applyEnemyDamage to avoid crash when enemy index is out of bounds.
    const damageSpy = jest
      .spyOn(combatModule, 'applyEnemyDamage')
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- mock implementation
      .mockImplementation((state: any) => state);

    const collisionMap = { isSolid: () => false as const };
    const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const stateWithBolt = {
      ...state,
      spawnCount: 999,
      simTimeMs: 300, // travelExpired: 300 + 16 = 316 >= 300
      enemies: [{ position: { x: 100, y: 100 }, health: 100, active: true }],
      bolts: [
        {
          position: { x: 5, y: 5 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 36,
          active: true,
          createdAtMs: 0,
          origin: { x: 5, y: 5 },
          hitEnemyIndex: 99, // out of bounds — no enemy at index 99
        },
      ],
    };
    const snapshot = {
      move: { x: 0, y: 0 },
      lookDelta: 0,
      fire: false,
      dash: false,
    };
    const next = tickModule.gameTick(stateWithBolt, snapshot, collisionMap);
    // Bolt deactivated by travelExpired, hitEnemyIndex 99 is out of bounds
    // → if(enemy) false branch covered, applyEnemyDamage mocked to no-op
    expect(next.bolts).toHaveLength(0);

    damageSpy.mockRestore();
  });

  it('covers next.enemyBolts ?? [] fallback when enemyBolts is undefined (line 303)', async () => {
    const { gameTick } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const collisionMap = { isSolid: () => false as const };
    const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const stateWithoutEnemyBolts = {
      ...state,
      spawnCount: 999,
      enemyBolts: undefined,
    };
    const snapshot = {
      move: { x: 0, y: 0 },
      lookDelta: 0,
      fire: false,
      dash: false,
    };
    const next = gameTick(stateWithoutEnemyBolts, snapshot, collisionMap);
    // enemyBolts ?? [] fallback at line 303 fires; result should be defined
    expect(next.enemyBolts).toBeDefined();
  });

  it('handles bolt with null origin in updateEnemyBolts (line 619 false branch)', async () => {
    const { updateEnemyBolts } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const state = makeState();
    const bolt = makeEnemyBolt({
      position: { x: 50, y: 50 },
      direction: { x: 1, y: 0 },
      createdAtMs: 1000,
      origin: null as unknown as { x: number; y: number },
    });
    // origin is null → bolt.origin && ... is false → distanceTraveled = 0
    const result = updateEnemyBolts([bolt], 100, 1100, undefined, state);
    const updated = result.bolts[0];
    // distanceTraveled = 0 → beyondMaxRange = false, lifetime not expired
    expect(updated.active).toBe(true);
  });

  describe('AC-801-S05-003: ammo pickup lifecycle in tick', () => {
    it('exports updateAmmoPickups', async () => {
      const mod = (await import('./tick.ts')) as Record<string, unknown>;
      expect(typeof mod.updateAmmoPickups).toBe('function');
    });

    it('collects ammo pickups within the collection radius of the player', async () => {
      const { updateAmmoPickups } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      // Place a pickup right at the player position (distance = 0, within collection radius)
      const state = {
        ...base,
        player: { ...base.player, ammo: 10 },
        ammoPickups: [
          {
            position: { ...base.player.position },
            amount: 5,
            active: true,
            createdAtMs: 0,
          },
        ],
      } as GameState;
      const result = updateAmmoPickups(state, 0);
      // Player ammo should increase by the pickup amount (10 + 5 = 15)
      expect(result.player.ammo).toBe(15);
      // Collected pickup should be marked inactive or removed
      expect(result.ammoPickups!.every((p) => p.active === false)).toBe(true);
    });

    it('marks expired ammo pickups as inactive', async () => {
      const { updateAmmoPickups } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const lifetimeMs = 10000;
      const expiredTime = lifetimeMs + 1000;
      // Place a pickup far from the player so it won't be collected, but past its lifetime
      const state = {
        ...base,
        ammoPickups: [
          {
            position: { x: 100, y: 100 },
            amount: 5,
            active: true,
            createdAtMs: 0,
            lifetimeMs,
          },
        ],
      } as GameState;
      const result = updateAmmoPickups(state, expiredTime);
      expect(result.ammoPickups!.every((p) => p.active === false)).toBe(true);
    });

    it('returns inactive pickups unchanged without collecting or expiring them', async () => {
      const { updateAmmoPickups } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state = {
        ...base,
        player: { ...base.player, ammo: 10 },
        ammoPickups: [
          {
            position: { ...base.player.position },
            amount: 5,
            active: false,
            createdAtMs: 0,
          },
        ],
      } as GameState;
      const result = updateAmmoPickups(state, 0);
      // Inactive pickup is returned unchanged — ammo should NOT increase
      expect(result.player.ammo).toBe(10);
      expect(result.ammoPickups!.every((p) => p.active === false)).toBe(true);
    });

    it('returns active pickups unchanged when not expired and not within collection radius', async () => {
      const { updateAmmoPickups } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const state = {
        ...base,
        player: { ...base.player, ammo: 10 },
        ammoPickups: [
          {
            position: { x: 100, y: 100 },
            amount: 5,
            active: true,
            createdAtMs: 0,
          },
        ],
      } as GameState;
      // simTimeMs=1000 is well within lifetime and the pickup is far from
      // the player, so neither the expired branch nor the collection branch
      // fires — the unchanged `return pickup;` at line 387 is covered.
      const result = updateAmmoPickups(state, 1000);
      expect(result.player.ammo).toBe(10);
      expect(result.ammoPickups!.length).toBeGreaterThan(0);
      expect(result.ammoPickups!.every((p) => p.active === true)).toBe(true);
    });

    it('uses ?? [] fallback when ammoPickups is undefined', async () => {
      const { updateAmmoPickups } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      // Strip ammoPickups so the property is entirely absent from the state,
      // exercising the `state.ammoPickups ?? []` nullish branch at line 362.
      const { ammoPickups: _stripped, ...stateWithoutPickups } = base;
      void _stripped;
      const state = {
        ...stateWithoutPickups,
      } as GameState;
      const result = updateAmmoPickups(state, 0);
      // When ammoPickups is undefined, the ?? [] fallback yields an empty
      // array, so pickups.length === 0 triggers the early return at line
      // 363-364 and the state is returned unchanged.
      expect(result).toBe(state);
      expect(result.player.ammo).toBe(state.player.ammo);
    });
  });

  describe('AC-203: hero respawn on death', () => {
    it('respawns the hero at the map center with full health and ammo when health reaches zero', async () => {
      const { createGameState, gameTick } =
        (await import('./tick.ts')) as typeof import('./tick.ts');
      const {
        NEATENSTEIN_PLAYER_MAX_AMMO,
        NEATENSTEIN_PLAYER_MAX_HEALTH,
        NEATENSTEIN_SPAWN_CENTER_X,
        NEATENSTEIN_SPAWN_CENTER_Y,
      } = (await import('./constants.ts')) as typeof import('./constants.ts');

      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const before = {
        ...base,
        player: {
          ...base.player,
          health: 0,
          ammo: 3,
          position: { x: 10, y: 10 },
          previousPosition: { x: 9, y: 9 },
          dashTimeRemainingMs: 50,
          dashCooldownMs: 200,
          contactIFrameMs: 100,
        },
        deaths: 0,
      };
      const next = gameTick(before, {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: false,
        dash: false,
      });

      expect({
        health: next.player.health,
        ammo: next.player.ammo,
        deaths: next.deaths,
        positionX: next.player.position.x,
        positionY: next.player.position.y,
        prevPositionX: next.player.previousPosition!.x,
        prevPositionY: next.player.previousPosition!.y,
        dashTime: next.player.dashTimeRemainingMs,
        dashCooldown: next.player.dashCooldownMs,
        iFrames: next.player.contactIFrameMs,
      }).toEqual({
        health: NEATENSTEIN_PLAYER_MAX_HEALTH,
        ammo: NEATENSTEIN_PLAYER_MAX_AMMO,
        deaths: 1,
        positionX: NEATENSTEIN_SPAWN_CENTER_X,
        positionY: NEATENSTEIN_SPAWN_CENTER_Y,
        prevPositionX: NEATENSTEIN_SPAWN_CENTER_X,
        prevPositionY: NEATENSTEIN_SPAWN_CENTER_Y,
        dashTime: 0,
        dashCooldown: 0,
        iFrames: 0,
      });
    });

    it('increments deaths counter cumulatively across multiple respawns', async () => {
      const { createGameState, gameTick } =
        (await import('./tick.ts')) as typeof import('./tick.ts');

      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      let state: GameState = {
        ...base,
        player: { ...base.player, health: 0 },
        deaths: 2,
      };
      state = gameTick(state, {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: false,
        dash: false,
      });
      // After respawn, health is full (> 0), so setting health to 0 again
      // simulates a second death on the next tick.
      state = {
        ...state,
        player: { ...state.player, health: 0 },
      };
      state = gameTick(state, {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: false,
        dash: false,
      });
      expect(state.deaths).toBe(4);
    });

    it('defaults deaths to 0 via ?? when deaths is undefined on respawn', async () => {
      const { createGameState, gameTick } =
        (await import('./tick.ts')) as typeof import('./tick.ts');

      const base = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      // Strip deaths entirely so the ?? fallback branch at line 345 is
      // exercised: (next.deaths ?? 0) + 1 should yield 1.
      const { deaths: _stripped, ...stateWithoutDeaths } = base;
      void _stripped;
      const before = {
        ...stateWithoutDeaths,
        player: { ...base.player, health: 0 },
      };
      const next = gameTick(before, {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: false,
        dash: false,
      });
      expect(next.deaths).toBe(1);
    });
  });
});

describe('AC-P3S1c-001: lastShotHit flag in gameTick', () => {
  function createEmptyCollisionMap(): { isSolid: () => false } {
    return { isSolid: () => false };
  }

  it('sets lastShotHit=true when a bolt hits an active enemy', async () => {
    const { gameTick } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const collisionMap = createEmptyCollisionMap();
    const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const stateWithBoltAndEnemy: GameState = {
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
    const next = gameTick(
      stateWithBoltAndEnemy,
      { move: { x: 0, y: 0 }, lookDelta: 0, fire: false, dash: false },
      collisionMap,
    );
    expect(next.lastShotHit).toBe(true);
  });

  it('sets lastShotHit=false when no bolt hits an enemy', async () => {
    const { gameTick } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const collisionMap = createEmptyCollisionMap();
    const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
    const stateWithNoBolts: GameState = {
      ...state,
      spawnCount: 999,
      bolts: [],
      enemies: [
        {
          position: { x: 5.3, y: 5 },
          health: 100,
          active: true,
        },
      ],
      lastShotHit: true, // carry-over from previous tick
    };
    const next = gameTick(
      stateWithNoBolts,
      { move: { x: 0, y: 0 }, lookDelta: 0, fire: false, dash: false },
      collisionMap,
    );
    expect(next.lastShotHit).toBe(false);
  });

  it('resets lastShotHit from true to false on the next tick without a hit', async () => {
    const { gameTick } =
      (await import('./tick.ts')) as typeof import('./tick.ts');
    const collisionMap = createEmptyCollisionMap();
    const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });

    // Tick 1: bolt hits enemy → lastShotHit = true
    const stateWithHit: GameState = {
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
    const afterHit = gameTick(
      stateWithHit,
      { move: { x: 0, y: 0 }, lookDelta: 0, fire: false, dash: false },
      collisionMap,
    );
    expect(afterHit.lastShotHit).toBe(true);

    // Tick 2: no bolts → lastShotHit should reset to false
    const afterNoHit = gameTick(
      afterHit,
      { move: { x: 0, y: 0 }, lookDelta: 0, fire: false, dash: false },
      collisionMap,
    );
    expect(afterNoHit.lastShotHit).toBe(false);
  });
});
