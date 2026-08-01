import { describe, expect, it } from '@jest/globals';

import type { CollisionMap } from '../browser-entry/renderer/map';
import {
  buildNeatensteinMap,
  createCollisionMap,
} from '../browser-entry/renderer/map';
import { NEATENSTEIN_MAP_SIZE } from '../browser-entry/constants';
import { createGameState } from '../browser-entry/host/game/state';
import type {
  EnemyState,
  GameState,
  Vector2,
} from '../browser-entry/host/game/types';
import {
  createEnemyControllerState,
  ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
  ENEMY_CONTROLLER_FIRE_COOLDOWN_MS,
  ENEMY_CONTROLLER_FIRE_RANGE_CELLS,
  ENEMY_CONTROLLER_HITSCAN_DAMAGE,
  ENEMY_CONTROLLER_RADIUS_CELLS,
  ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND,
  ENEMY_CONTROLLER_STARTING_AMMO,
  ENEMY_CONTROLLER_STOP_DISTANCE_CELLS,
  updateEnemyController,
} from './enemy-controller';

/**
 * Red/green contract tests for examples/neatenstein/scripts/enemy-controller.ts.
 *
 * Covers AC-702I: movement, collision, hitscan fire, and ammo-depletion de-rez.
 */

/** Build a collision map where every cell is open floor. */
function createEmptyCollisionMap(): CollisionMap {
  return {
    isSolid: () => false,
  };
}

/** Build a collision map with solid cells at the supplied grid coordinates. */
function createWallCollisionMap(
  walls: ReadonlyArray<{ readonly x: number; readonly y: number }>,
): CollisionMap {
  const set = new Set(walls.map((w) => `${w.x},${w.y}`));
  return {
    isSolid: (x: number, y: number) => set.has(`${x},${y}`),
  };
}

/** Build a game state with a single enemy placed at the requested offset. */
function stateWithEnemy(
  base: GameState,
  offset: Vector2,
  health = 100,
): GameState {
  const enemy: EnemyState = {
    position: {
      x: base.player.position.x + offset.x,
      y: base.player.position.y + offset.y,
    },
    health,
  };

  return {
    ...base,
    enemies: [enemy],
  };
}

/** Build a game state with a single enemy at an absolute world position. */
function stateWithEnemyAt(
  base: GameState,
  position: Vector2,
  health = 100,
): GameState {
  const enemy: EnemyState = { position: { ...position }, health };
  return { ...base, enemies: [enemy] };
}

describe('enemy controller (AC-702I contracts)', () => {
  it('exports createEnemyControllerState', () => {
    expect(typeof createEnemyControllerState).toBe('function');
  });

  it('exports updateEnemyController', () => {
    expect(typeof updateEnemyController).toBe('function');
  });

  it('exports positive tuning constants', () => {
    expect(ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND).toBeGreaterThan(0);
    expect(ENEMY_CONTROLLER_RADIUS_CELLS).toBeGreaterThan(0);
    expect(ENEMY_CONTROLLER_FIRE_RANGE_CELLS).toBeGreaterThan(0);
    expect(ENEMY_CONTROLLER_FIRE_COOLDOWN_MS).toBeGreaterThan(0);
    expect(ENEMY_CONTROLLER_HITSCAN_DAMAGE).toBeGreaterThan(0);
    expect(ENEMY_CONTROLLER_STARTING_AMMO).toBeGreaterThan(0);
    expect(ENEMY_CONTROLLER_DE_REZ_DURATION_MS).toBeGreaterThan(0);
    expect(ENEMY_CONTROLLER_STOP_DISTANCE_CELLS).toBeGreaterThanOrEqual(0);
  });

  it('initializes a controlled enemy from GameState', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: -5, y: 0 });
    const controller = createEnemyControllerState(state);

    expect(controller.enemies).toHaveLength(1);
    expect(controller.enemies[0].index).toBe(0);
    expect(controller.enemies[0].health).toBe(100);
    expect(controller.enemies[0].ammo).toBe(ENEMY_CONTROLLER_STARTING_AMMO);
    expect(controller.enemies[0].active).toBe(true);
    expect(controller.enemies[0].animationState).toBe('idle');
    expect(controller.hitscanEvents).toHaveLength(0);
  });

  it('initializes an empty controller when the snapshot has no enemies', () => {
    const base = createGameState({ seed: 1 });
    const empty: GameState = { ...base, enemies: [] };
    const controller = createEnemyControllerState(empty);

    expect(controller.enemies).toHaveLength(0);
    expect(controller.hitscanEvents).toHaveLength(0);
  });

  it('moves a living enemy toward the player', () => {
    const base = createGameState({ seed: 1 });
    // Place enemy beyond fire range so the tick is purely movement.
    const state = stateWithEnemy(base, {
      x: -ENEMY_CONTROLLER_FIRE_RANGE_CELLS - 2,
      y: 0,
    });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    const startX = controller.enemies[0].position.x;

    controller = updateEnemyController(controller, state, emptyMap, 1000);

    expect(controller.enemies[0].position.x).toBeGreaterThan(startX);
    expect(controller.enemies[0].position.y).toBeCloseTo(
      state.enemies[0].position.y,
    );
    expect(controller.enemies[0].animationState).toBe('move');
    expect(controller.hitscanEvents).toHaveLength(0);
  });

  it('falls back to the fixed timestep for non-positive dt values', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, {
      x: -ENEMY_CONTROLLER_FIRE_RANGE_CELLS - 2,
      y: 0,
    });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    const startX = controller.enemies[0].position.x;

    // Negative dt must not crash and must advance by the canonical fixed step.
    controller = updateEnemyController(controller, state, emptyMap, -1);

    expect(controller.enemies[0].position.x).toBeGreaterThan(startX);
    expect(controller.enemies[0].active).toBe(true);
  });

  it('faces the player while moving', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 0, y: -3 });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 1000);

    expect(controller.enemies[0].yawRad).toBeCloseTo(Math.PI / 2, 2);
  });

  it('does not move a dead enemy', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: -3, y: 0 }, 0);
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    const startX = controller.enemies[0].position.x;

    controller = updateEnemyController(controller, state, emptyMap, 1000);

    expect(controller.enemies[0].position.x).toBe(startX);
    expect(controller.enemies[0].animationState).toBe('death');
  });

  it('stops at walls instead of passing through them', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: -3, y: 0 });
    // Vertical wall directly between enemy and player.
    const wallMap = createWallCollisionMap([
      {
        x: Math.floor(base.player.position.x) - 1,
        y: Math.floor(base.player.position.y),
      },
    ]);

    let controller = createEnemyControllerState(state);

    for (let i = 0; i < 20; i += 1) {
      controller = updateEnemyController(controller, state, wallMap, 100);
    }

    const wallX = Math.floor(base.player.position.x) - 1;
    expect(controller.enemies[0].position.x).toBeLessThan(wallX);
    expect(controller.enemies[0].active).toBe(true);
  });

  it('fires a hitscan shot when the player is in range and line-of-sight', () => {
    const base = createGameState({ seed: 1 });
    // Enemy to the left of the player; it should fire toward +X.
    const state = stateWithEnemy(base, { x: -3, y: 0 });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 1000);

    expect(controller.hitscanEvents).toHaveLength(1);
    expect(controller.hitscanEvents[0].enemyIndex).toBe(0);
    expect(controller.hitscanEvents[0].damage).toBe(
      ENEMY_CONTROLLER_HITSCAN_DAMAGE,
    );
    expect(controller.hitscanEvents[0].direction.x).toBeCloseTo(1, 5);
    expect(controller.enemies[0].animationState).toBe('fire');
    expect(controller.enemies[0].ammo).toBe(ENEMY_CONTROLLER_STARTING_AMMO - 1);
    expect(controller.enemies[0].fireCooldownMs).toBe(
      ENEMY_CONTROLLER_FIRE_COOLDOWN_MS,
    );
  });

  it('does not fire when the player is out of range', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, {
      x: ENEMY_CONTROLLER_FIRE_RANGE_CELLS + 1,
      y: 0,
    });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 1000);

    expect(controller.hitscanEvents).toHaveLength(0);
    expect(controller.enemies[0].animationState).not.toBe('fire');
  });

  it('does not fire when a wall blocks line-of-sight', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 3, y: 0 });
    const wallMap = createWallCollisionMap([
      {
        x: Math.floor(base.player.position.x) + 1,
        y: Math.floor(base.player.position.y),
      },
    ]);

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 1000);

    expect(controller.hitscanEvents).toHaveLength(0);
  });

  it('respects the fire cooldown between shots', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 3, y: 0 });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);

    // First tick: fires and starts cooldown.
    controller = updateEnemyController(controller, state, emptyMap, 100);
    expect(controller.hitscanEvents).toHaveLength(1);

    // Second tick: still on cooldown, should not fire again.
    controller = updateEnemyController(controller, state, emptyMap, 100);
    expect(controller.hitscanEvents).toHaveLength(0);
  });

  it('enters de-rez when ammunition is depleted', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 3, y: 0 });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);

    // Fire until ammo is exhausted. Use a dt larger than the cooldown.
    const fireDt = ENEMY_CONTROLLER_FIRE_COOLDOWN_MS + 100;
    for (let i = 0; i < ENEMY_CONTROLLER_STARTING_AMMO; i += 1) {
      controller = updateEnemyController(controller, state, emptyMap, fireDt);
    }

    expect(controller.enemies[0].ammo).toBe(0);
    expect(controller.enemies[0].animationState).toBe('fire');

    // Next tick: ammo-depletion triggers de-rez.
    controller = updateEnemyController(controller, state, emptyMap, fireDt);
    expect(controller.enemies[0].animationState).toBe('death');
    expect(controller.enemies[0].active).toBe(true);
  });

  it('handles ammo-depletion de-rez while sharing the player cell', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { ...base.player.position });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);

    // Fire until the magazine is empty while occupying the same cell as the
    // player. The death-path facing branch must still handle distance zero.
    const fireDt = ENEMY_CONTROLLER_FIRE_COOLDOWN_MS + 100;
    for (let i = 0; i < ENEMY_CONTROLLER_STARTING_AMMO + 1; i += 1) {
      controller = updateEnemyController(controller, state, emptyMap, fireDt);
    }

    expect(controller.enemies[0].ammo).toBe(0);
    expect(controller.enemies[0].animationState).toBe('death');
    expect(controller.enemies[0].active).toBe(true);
  });

  it('fully de-rezzes after the configured duration', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 3, y: 0 });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);

    // Deplete ammo.
    const fireDt = ENEMY_CONTROLLER_FIRE_COOLDOWN_MS + 100;
    for (let i = 0; i < ENEMY_CONTROLLER_STARTING_AMMO + 1; i += 1) {
      controller = updateEnemyController(controller, state, emptyMap, fireDt);
    }

    expect(controller.enemies[0].animationState).toBe('death');

    // Advance through the de-rez window.
    controller = updateEnemyController(
      controller,
      state,
      emptyMap,
      ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
    );

    expect(controller.enemies[0].active).toBe(false);
    expect(controller.enemies[0].animationState).toBe('death');
  });

  it('enters de-rez immediately when health is zero', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 3, y: 0 }, 0);
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 100);

    expect(controller.enemies[0].animationState).toBe('death');
    expect(controller.enemies[0].deRezElapsedMs).toBeGreaterThan(0);
  });

  it('does not fire while de-rezzing', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 3, y: 0 }, 0);
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 1000);

    expect(controller.hitscanEvents).toHaveLength(0);
  });

  it('recovers a missing controlled enemy entry on the fly', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 3, y: 0 });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    // Simulate a truncated controller state (e.g. a respawn added a new index).
    controller.enemies = [];
    controller = updateEnemyController(controller, state, emptyMap, 100);

    expect(controller.enemies).toHaveLength(1);
    expect(controller.enemies[0].active).toBe(true);
    // The recovered entry fires one shot during its first tick.
    expect(controller.enemies[0].ammo).toBe(ENEMY_CONTROLLER_STARTING_AMMO - 1);
  });

  it('remains deterministic for the same inputs', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: -3, y: 2 });
    const emptyMap = createEmptyCollisionMap();

    let a = createEnemyControllerState(state);
    let b = createEnemyControllerState(state);

    for (let i = 0; i < 5; i += 1) {
      a = updateEnemyController(a, state, emptyMap, 200);
      b = updateEnemyController(b, state, emptyMap, 200);
    }

    expect(a.enemies[0].position.x).toBe(b.enemies[0].position.x);
    expect(a.enemies[0].position.y).toBe(b.enemies[0].position.y);
    expect(a.enemies[0].ammo).toBe(b.enemies[0].ammo);
    expect(a.hitscanEvents.length).toBe(b.hitscanEvents.length);
  });

  it('handles an enemy sharing the player position without crashing', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { ...base.player.position });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 1000);

    expect(controller.enemies[0].position.x).toBeCloseTo(
      base.player.position.x,
      5,
    );
    expect(controller.enemies[0].position.y).toBeCloseTo(
      base.player.position.y,
      5,
    );
    expect(controller.hitscanEvents.length).toBeGreaterThanOrEqual(0);
  });

  it('slides along a wall when only one axis is blocked', () => {
    const base = createGameState({ seed: 1 });
    const enemyPos = { x: 10.5, y: 10.5 };
    const state = stateWithEnemyAt(base, enemyPos);
    // Single wall cell on the diagonal path; X-only and Y-only recovery are
    // both open, so the enemy slides horizontally (first successful recovery).
    const wallMap = createWallCollisionMap([{ x: 11, y: 11 }]);

    let controller = createEnemyControllerState(state);
    // Step far enough to hit the diagonal wall cell.
    controller = updateEnemyController(controller, state, wallMap, 283);

    expect(controller.enemies[0].position.x).toBeGreaterThan(enemyPos.x);
    expect(controller.enemies[0].position.y).toBe(enemyPos.y);
  });

  it('reverts to the previous position when both recovery axes are blocked', () => {
    const base = createGameState({ seed: 1 });
    const enemyPos = { x: 10.5, y: 10.5 };
    const state = stateWithEnemyAt(base, enemyPos);
    // Walls block both the horizontal-only and vertical-only recovery
    // positions, so the enemy cannot move this tick.
    const wallMap = createWallCollisionMap([
      { x: 11, y: 10 },
      { x: 10, y: 11 },
    ]);

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 283);

    expect(controller.enemies[0].position.x).toBeCloseTo(enemyPos.x, 5);
    expect(controller.enemies[0].position.y).toBeCloseTo(enemyPos.y, 5);
  });

  it('does not crash when the source position is non-finite', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: Number.NaN, y: Number.NaN });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 100);

    expect(Number.isNaN(controller.enemies[0].position.x)).toBe(true);
    expect(Number.isNaN(controller.enemies[0].position.y)).toBe(true);
    expect(controller.enemies[0].active).toBe(true);
  });

  it('works with the real seeded map collision data', () => {
    const base = createGameState({ seed: 42 });
    const mapGrid = buildNeatensteinMap(base.seed);
    const collisionMap = createCollisionMap(mapGrid, NEATENSTEIN_MAP_SIZE);

    // Spawn enemy a few cells away from the central spawn point.
    const state = stateWithEnemy(base, { x: 4, y: 0 });

    let controller = createEnemyControllerState(state);
    for (let i = 0; i < 10; i += 1) {
      controller = updateEnemyController(controller, state, collisionMap, 100);
    }

    expect(controller.enemies[0].active).toBe(true);
    expect(Number.isFinite(controller.enemies[0].position.x)).toBe(true);
    expect(Number.isFinite(controller.enemies[0].position.y)).toBe(true);
  });

  it('resets controller state for a respawned enemy at the same index', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 3, y: 0 });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);

    // Deplete ammo and de-rez fully.
    const fireDt = ENEMY_CONTROLLER_FIRE_COOLDOWN_MS + 100;
    for (let i = 0; i < ENEMY_CONTROLLER_STARTING_AMMO + 1; i += 1) {
      controller = updateEnemyController(controller, state, emptyMap, fireDt);
    }
    controller = updateEnemyController(
      controller,
      state,
      emptyMap,
      ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
    );

    expect(controller.enemies[0].active).toBe(false);
    expect(controller.enemies[0].ammo).toBe(0);

    // Respawn a fresh enemy at index 0, out of fire range so ammo is not
    // immediately consumed by a shot.
    const respawnedState = stateWithEnemy(base, {
      x: -ENEMY_CONTROLLER_FIRE_RANGE_CELLS - 2,
      y: 0,
    });
    controller = updateEnemyController(
      controller,
      respawnedState,
      emptyMap,
      100,
    );

    expect(controller.enemies[0].active).toBe(true);
    expect(controller.enemies[0].ammo).toBe(ENEMY_CONTROLLER_STARTING_AMMO);
    expect(controller.enemies[0].deRezElapsedMs).toBe(0);
  });
});
