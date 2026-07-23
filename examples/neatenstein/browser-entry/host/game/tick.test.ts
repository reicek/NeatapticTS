import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_BEAM_COLOR,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_TRACER_DURATION_MS,
} from './constants';
import type { TracerState } from './types';

/**
 * Minimal tracer factory for tests that only care about duration semantics.
 *
 * @param durationMs - Remaining visibility time for the tracer.
 * @returns A valid {@link TracerState} with placeholder geometry.
 */
function makeTracer(durationMs: number): TracerState {
  return {
    origin: { x: 0, y: 0 },
    direction: { x: 1, y: 0 },
    hit: { x: 1, y: 0 },
    distance: 1,
    hitType: 'wall',
    durationMs,
    color: NEATENSTEIN_BEAM_COLOR,
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
  });

  describe('ageTracers helper', () => {
    it('decrements durationMs by the timestep and removes expired tracers', async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
      const { ageTracers } = (await import('./tick.ts')) as Record<string, any>;
      const tracers = [
        makeTracer(NEATENSTEIN_FIXED_TIMESTEP_MS),
        makeTracer(NEATENSTEIN_TRACER_DURATION_MS),
      ];
      const aged = ageTracers(tracers, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(aged.map((t: TracerState) => t.durationMs)).toEqual([
        NEATENSTEIN_TRACER_DURATION_MS - NEATENSTEIN_FIXED_TIMESTEP_MS,
      ]);
    });

    it('keeps a newly fired tracer at full duration while older tracers age', async () => {
      const { createGameState, gameTick } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./tick.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      state.player.angleRad = 0;
      state.tracers = [makeTracer(NEATENSTEIN_TRACER_DURATION_MS)];
      const next = gameTick(state, { fire: true });
      expect(
        next.tracers
          .map((t: TracerState) => t.durationMs)
          .toSorted((a: number, b: number) => a - b),
      ).toEqual([
        NEATENSTEIN_TRACER_DURATION_MS - NEATENSTEIN_FIXED_TIMESTEP_MS,
        NEATENSTEIN_TRACER_DURATION_MS,
      ]);
    });
  });
});
