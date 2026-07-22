import { describe, expect, it } from '@jest/globals';

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
});
