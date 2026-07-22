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

    it('consumes one unit of ammo when fired', async () => {
      const { createGameState, fireNeonBeam } =
        (await import('./combat.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const result = fireNeonBeam(state);
      expect(result.state.player.ammo).toBe(state.player.ammo - 1);
    });

    it('appends a single tracer to the new state when fired', async () => {
      const { createGameState, fireNeonBeam } =
        (await import('./combat.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const result = fireNeonBeam(state);
      expect(result.state.tracers.length).toBe(state.tracers.length + 1);
    });

    it('returns a tracer with beam origin, endpoint, and hit metadata', async () => {
      const { createGameState, fireNeonBeam } =
        (await import('./combat.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const result = fireNeonBeam(state);
      expect(result.tracer).toMatchObject({
        origin: state.player.position,
        hitType: expect.any(String),
        distance: expect.any(Number),
        color: expect.any(String),
      });
    });

    it('damages the first enemy in the beam path', async () => {
      const { createGameState, fireNeonBeam } =
        (await import('./combat.ts')) as Record<string, any>;
      const base = createGameState({ seed: 1 });
      const state = {
        ...base,
        player: { ...base.player, angleRad: 0, ammo: base.player.maxAmmo },
        enemies: [
          {
            position: {
              x: base.player.position.x + 2,
              y: base.player.position.y,
            },
            health: 100,
          },
        ],
      };
      const result = fireNeonBeam(state);
      expect(result.state.enemies[0].health).toBe(50);
    });
  });
});
