import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/movement.ts.
 *
 * Covers AC-205: WASD translation, wall collision, and normalized diagonal
 * movement speed.
 */

describe('Neatenstein game movement', () => {
  describe('AC-205: WASD movement and wall collision contract', () => {
    it('exports movePlayer', async () => {
      const mod = (await import('./movement.ts')) as Record<string, unknown>;
      expect(typeof mod.movePlayer).toBe('function');
    });

    it('exports normalizeMoveVector', async () => {
      const mod = (await import('./movement.ts')) as Record<string, unknown>;
      expect(typeof mod.normalizeMoveVector).toBe('function');
    });

    it('exports resolveWallCollision', async () => {
      const mod = (await import('./movement.ts')) as Record<string, unknown>;
      expect(typeof mod.resolveWallCollision).toBe('function');
    });

    it('normalizes a diagonal movement vector to length one', async () => {
      const { normalizeMoveVector } = (await import('./movement.ts')) as Record<
        string,
        any
      >;
      const vector = normalizeMoveVector({ x: 1, y: 1 });
      expect(Math.hypot(vector.x, vector.y)).toBeCloseTo(1, 10);
    });

    it('prevents the player from entering a solid map cell', async () => {
      const { createGameState, movePlayer, resolveWallCollision } =
        (await import('./movement.ts')) as Record<string, any>;
      const state = createGameState({ seed: 1 });
      const map = { isSolid: () => true };
      const moved = movePlayer(state, { x: 1, y: 0 });
      const resolved = resolveWallCollision(moved, map);
      expect(resolved.player.position).toEqual(state.player.position);
    });
  });
});
