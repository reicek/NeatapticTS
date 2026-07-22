import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/waves.ts.
 *
 * Covers AC-203: enemy waves spawn as a continuous trickle with at most one
 * new enemy per tick and no more than 8 concurrent enemies.
 */

describe('Neatenstein game waves', () => {
  describe('AC-203: continuous-trickle spawn contract', () => {
    it('exports spawnWaveTick', async () => {
      const mod = (await import('./waves.ts')) as Record<string, unknown>;
      expect(typeof mod.spawnWaveTick).toBe('function');
    });

    it('spawns at most one enemy per tick', async () => {
      const { createGameState, spawnWaveTick } =
        (await import('./waves.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const result = spawnWaveTick(before, 1000);
      expect(result.spawnedThisTick).toBeLessThanOrEqual(1);
    });

    it('does not spawn when the active enemy count is already at the cap', async () => {
      const { createGameState, spawnWaveTick } =
        (await import('./waves.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      while (state.enemies.length < 8) {
        state.enemies.push({ position: { x: 0, y: 0 }, health: 1 });
      }
      const result = spawnWaveTick(state, 1000);
      expect(result.spawnedThisTick).toBe(0);
    });

    it('keeps the active enemy count at or below 8 after spawning', async () => {
      const { createGameState, spawnWaveTick } =
        (await import('./waves.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const result = spawnWaveTick(state, 1000);
      expect(result.state.enemies.length).toBeLessThanOrEqual(8);
    });
  });
});
