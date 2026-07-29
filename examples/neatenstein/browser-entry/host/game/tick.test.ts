import { describe, expect, it } from '@jest/globals';
import { NEATENSTEIN_IMPACT_SPOT_LIFETIME_MS } from '../../constants';
import { NEATENSTEIN_FIXED_TIMESTEP_MS } from './constants';
import type { ImpactSpot } from './types';

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
