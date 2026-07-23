import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_MS_PER_SECOND,
  NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND,
  NEATENSTEIN_TEST_SEED,
} from './constants';
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
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
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
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const after = movePlayer(before, { x: 1, y: 0 });
      expect(after.player.previousPosition).toEqual(before.player.position);
    });

    it('moves the player forward along the look angle on an open map', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      state.player.angleRad = 0;
      const map = { isSolid: () => false };
      const after = updatePlayerMovement(
        state,
        { forward: true, backward: false, left: false, right: false },
        map,
        NEATENSTEIN_MS_PER_SECOND,
      );
      expect(after.player.position.x).toBeCloseTo(
        state.player.position.x + NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND,
        10,
      );
    });

    it('keeps diagonal movement at the same speed as cardinal movement', () => {
      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      state.player.angleRad = 0;
      const map = { isSolid: () => false };
      const after = updatePlayerMovement(
        state,
        { forward: true, backward: false, left: false, right: true },
        map,
        NEATENSTEIN_MS_PER_SECOND,
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

    it('returns a zero vector when normalizing a zero-length input', () => {
      expect(normalizeMoveVector({ x: 0, y: 0 })).toEqual({ x: 0, y: 0 });
    });

    it('slides along Y when only the new X position is blocked', () => {
      const PREV_X = 2.5;
      const PREV_Y = 2.5;
      const BLOCKED_X = 6.5;
      const ADVANCED_Y = 6.5;
      const SOLID_COLUMN_X = 6;
      const SOLID_COLUMN_Y_MIN = 2;
      const SOLID_COLUMN_Y_MAX = 7;

      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      state.player.position = { x: BLOCKED_X, y: ADVANCED_Y };
      state.player.previousPosition = { x: PREV_X, y: PREV_Y };

      const solidCells = new Set(
        Array.from(
          { length: SOLID_COLUMN_Y_MAX - SOLID_COLUMN_Y_MIN + 1 },
          (_, i) => ({
            x: SOLID_COLUMN_X,
            y: SOLID_COLUMN_Y_MIN + i,
          }),
        ),
      );
      const map = {
        isSolid: (x: number, y: number) =>
          Array.from(solidCells).some((cell) => cell.x === x && cell.y === y),
      };
      const resolved = resolveWallCollision(state, map);
      expect(resolved.player.position).toEqual({ x: PREV_X, y: ADVANCED_Y });
    });

    it('slides along X when only the new Y position is blocked', () => {
      const PREV_X = 2.5;
      const PREV_Y = 2.5;
      const ADVANCED_X = 6.5;
      const BLOCKED_Y = 6.5;
      const SOLID_ROW_Y = 6;
      const SOLID_ROW_X_MIN = 2;
      const SOLID_ROW_X_MAX = 7;

      const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      state.player.position = { x: ADVANCED_X, y: BLOCKED_Y };
      state.player.previousPosition = { x: PREV_X, y: PREV_Y };

      const solidCells = new Set(
        Array.from(
          { length: SOLID_ROW_X_MAX - SOLID_ROW_X_MIN + 1 },
          (_, i) => ({
            x: SOLID_ROW_X_MIN + i,
            y: SOLID_ROW_Y,
          }),
        ),
      );
      const map = {
        isSolid: (x: number, y: number) =>
          Array.from(solidCells).some((cell) => cell.x === x && cell.y === y),
      };
      const resolved = resolveWallCollision(state, map);
      expect(resolved.player.position).toEqual({ x: ADVANCED_X, y: PREV_Y });
    });
  });
});
