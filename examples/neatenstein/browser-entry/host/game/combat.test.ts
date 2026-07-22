import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/combat.ts.
 *
 * Covers AC-204 (hitscan neon beam) and AC-210 (only the beam weapon exists).
 */

describe('Neatenstein game combat', () => {
  describe('AC-204 / AC-210: neon beam hitscan contract', () => {
    it('exports fireNeonBeam', async () => {
      const mod = (await import('./combat.ts')) as Record<string, unknown>;
      expect(typeof mod.fireNeonBeam).toBe('function');
    });

    it('returns a tracer object for the fired frame', async () => {
      const { createGameState, fireNeonBeam } =
        (await import('./combat.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const result = fireNeonBeam(state);
      expect(typeof result.tracer).toBe('object');
    });

    it('does not fire when ammo is zero', async () => {
      const { createGameState, fireNeonBeam } =
        (await import('./combat.ts')) as Record<string, any>;
      const state = {
        ...createGameState({ seed: 1 }),
        player: {
          ...createGameState({ seed: 1 }).player,
          ammo: 0,
        },
      };
      const result = fireNeonBeam(state);
      expect(result.fired).toBe(false);
    });
  });
});
