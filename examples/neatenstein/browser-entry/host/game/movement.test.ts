import { describe, expect, it } from '@jest/globals';
import { NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND } from './constants';
import {
  createGameState,
  movePlayer,
  normalizeMoveVector,
  resolveWallCollision,
  updatePlayerMovement,
} from './movement';

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

    it('normalizes a diagonal movement vector to length one', () => {
      const vector = normalizeMoveVector({ x: 1, y: 1 });
      expect(Math.hypot(vector.x, vector.y)).toBeCloseTo(1, 10);
    });

    it('prevents the player from entering a solid map cell', () => {
      const state = createGameState({ seed: 1 });
      const map = { isSolid: () => true };
      const moved = movePlayer(state, { x: 1, y: 0 });
      const resolved = resolveWallCollision(moved, map);
      expect(resolved.player.position).toEqual(state.player.position);
    });

    it('exports updatePlayerMovement', async () => {
      const mod = (await import('./movement.ts')) as Record<string, unknown>;
      expect(typeof mod.updatePlayerMovement).toBe('function');
    });

    it('records the previous position when the player moves', () => {
      const before = createGameState({ seed: 1 });
      const after = movePlayer(before, { x: 1, y: 0 });
      expect(after.player.previousPosition).toEqual(before.player.position);
    });

    it('moves the player forward along the look angle on an open map', () => {
      const state = createGameState({ seed: 1 });
      state.player.angleRad = 0;
      const map = { isSolid: () => false };
      const after = updatePlayerMovement(
        state,
        { forward: true, backward: false, left: false, right: false },
        map,
        1_000,
      );
      expect(after.player.position.x).toBeCloseTo(
        state.player.position.x + NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND,
        10,
      );
    });

    it('keeps diagonal movement at the same speed as cardinal movement', () => {
      const state = createGameState({ seed: 1 });
      state.player.angleRad = 0;
      const map = { isSolid: () => false };
      const after = updatePlayerMovement(
        state,
        { forward: true, backward: false, left: false, right: true },
        map,
        1_000,
      );
      const displacement = Math.hypot(
        after.player.position.x - state.player.position.x,
        after.player.position.y - state.player.position.y,
      );
      expect(displacement).toBeCloseTo(
        NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND,
        10,
      );
    });
  });
});
