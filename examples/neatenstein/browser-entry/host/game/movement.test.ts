import {
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_MS_PER_SECOND,
  NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND,
  NEATENSTEIN_TEST_SEED,
} from './constants';
import {
  createGameState,
  isPositionBlocked,
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

  describe('AC-10.2c-001: isPositionBlocked uses synced controller positions', () => {
    const stepDistance = NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND;

    it('blocks movement using the synced controller position, not the stale spawn position', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.player.angleRad = 0;
      const map = { isSolid: () => false };
      const targetX = before.player.position.x + stepDistance;

      before.enemies = [
        {
          // Stale spawn point is far away and would not block the path.
          position: { x: targetX + 50, y: before.player.position.y },
          // Synced controller position sits on the player's target cell.
          controllerPosition: { x: targetX, y: before.player.position.y },
          health: 100,
          active: true,
        },
      ];

      const after = updatePlayerMovement(
        before,
        { forward: true, backward: false, left: false, right: false },
        map,
        NEATENSTEIN_MS_PER_SECOND,
      );

      // Currently isPositionBlocked reads enemy.position, so the player moves.
      // After the fix it must read enemy.controllerPosition and stay blocked.
      expect(after.player.position.x).toBe(before.player.position.x);
    });
  });

  describe('AC-10.2c-002: dead/inactive enemies do not block movement', () => {
    const stepDistance = NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND;

    it('lets the player walk through a dead enemy overlapping the path', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.player.angleRad = 0;
      const map = { isSolid: () => false };
      before.enemies = [
        {
          position: {
            x: before.player.position.x + stepDistance,
            y: before.player.position.y,
          },
          health: 0,
        },
      ];

      const after = updatePlayerMovement(
        before,
        { forward: true, backward: false, left: false, right: false },
        map,
        NEATENSTEIN_MS_PER_SECOND,
      );

      expect(after.player.position.x).toBeGreaterThan(before.player.position.x);
    });

    it('lets the player walk through an inactive enemy overlapping the path', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.player.angleRad = 0;
      const map = { isSolid: () => false };
      before.enemies = [
        {
          position: {
            x: before.player.position.x + stepDistance,
            y: before.player.position.y,
          },
          health: 100,
          active: false,
        },
      ];

      const after = updatePlayerMovement(
        before,
        { forward: true, backward: false, left: false, right: false },
        map,
        NEATENSTEIN_MS_PER_SECOND,
      );

      expect(after.player.position.x).toBeGreaterThan(before.player.position.x);
    });

    describe('Coverage: edge cases', () => {
      it('returns a zero vector when normalizing a NaN input', () => {
        expect(normalizeMoveVector({ x: NaN, y: 1 })).toEqual({ x: 0, y: 0 });
      });

      it('returns a zero vector when normalizing an Infinity input', () => {
        expect(normalizeMoveVector({ x: Infinity, y: -Infinity })).toEqual({
          x: 0,
          y: 0,
        });
      });

      it('returns the same state when movePlayer receives a zero delta', () => {
        const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        const after = movePlayer(state, { x: 0, y: 0 });
        expect(after).toBe(state);
      });

      it('returns the same state when updatePlayerMovement receives no input', () => {
        const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        const map = { isSolid: () => false };
        const after = updatePlayerMovement(
          state,
          {
            forward: false,
            backward: false,
            left: false,
            right: false,
          },
          map,
        );
        expect(after).toBe(state);
      });

      it('returns blocked for a non-finite player position', () => {
        const map = { isSolid: () => false };
        expect(isPositionBlocked({ x: NaN, y: 0 }, map)).toBe(true);
        expect(isPositionBlocked({ x: Infinity, y: 0 }, map)).toBe(true);
      });

      it('falls back to enemy.position when controllerPosition is absent', () => {
        const map = { isSolid: () => false };
        const enemyPosition = { x: 10, y: 10 };
        const enemies = [
          {
            position: enemyPosition,
            health: 100,
          },
        ];
        expect(isPositionBlocked(enemyPosition, map, enemies)).toBe(true);
      });

      it('falls back to the fixed timestep when movePlayer receives a non-finite dtMs', () => {
        const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        state.player.angleRad = 0;
        const after = movePlayer(state, { x: 1, y: 0 }, NaN);
        const expectedDistance =
          NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND *
          (NEATENSTEIN_FIXED_TIMESTEP_MS / NEATENSTEIN_MS_PER_SECOND);
        expect(after.player.position.x).toBeCloseTo(
          state.player.position.x + expectedDistance,
          10,
        );
      });

      it('resolves wall collision when previousPosition is missing', () => {
        const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        delete (state.player as { previousPosition?: unknown })
          .previousPosition;
        const map = { isSolid: () => true };
        const after = resolveWallCollision(state, map);
        expect(after.player.position).toEqual(state.player.position);
      });

      it('treats a non-finite player angle as zero during movement', () => {
        const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        state.player.angleRad = NaN;
        const map = { isSolid: () => false };
        const after = updatePlayerMovement(
          state,
          { forward: true, backward: false, left: false, right: false },
          map,
          NEATENSTEIN_MS_PER_SECOND,
        );
        expect(after.player.position.x).toBeGreaterThan(
          state.player.position.x,
        );
      });

      it('covers backward and left movement intent flags', () => {
        const state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
        state.player.angleRad = 0;
        const map = { isSolid: () => false };
        const after = updatePlayerMovement(
          state,
          {
            forward: false,
            backward: true,
            left: true,
            right: false,
          },
          map,
          NEATENSTEIN_MS_PER_SECOND,
        );
        expect(after.player.position).not.toEqual(state.player.position);
      });
    });
  });
});
