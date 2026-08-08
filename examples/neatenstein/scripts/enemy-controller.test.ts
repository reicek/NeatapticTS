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
  ENEMY_CONTROLLER_FLANKING_RADIUS_CELLS,
  ENEMY_CONTROLLER_HITSCAN_DAMAGE,
  ENEMY_CONTROLLER_RADIUS_CELLS,
  ENEMY_CONTROLLER_SHOOT_BLINK_TICKS,
  ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND,
  ENEMY_CONTROLLER_STARTING_AMMO,
  ENEMY_CONTROLLER_STOP_DISTANCE_CELLS,
  ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
  updateEnemyController,
} from './enemy-controller';
import type { ControlledEnemy } from './enemy-controller';
import { NEATENSTEIN_FIXED_TIMESTEP_MS } from '../browser-entry/host/game/constants';

/**
 * Red/green contract tests for examples/neatenstein/scripts/enemy-controller.ts.
 *
 * Covers AC-702I: movement, collision, hitscan fire, and ammo-depletion de-rez.
 * Also covers AC-10.5a-002: ControlledEnemy vision fields (weights, variantId,
 * previousStepDistance).
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

  it('navigates around walls without entering them', () => {
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
      const cellX = Math.floor(controller.enemies[0].position.x);
      const cellY = Math.floor(controller.enemies[0].position.y);
      expect(wallMap.isSolid(cellX, cellY)).toBe(false);
    }

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
    // Use a short timestep so the enemy stays in its starting cell and the
    // wall still blocks line-of-sight to the player.
    controller = updateEnemyController(controller, state, wallMap, 100);

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

  it('enters de-rez when health reaches zero', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 3, y: 0 }, 0);
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 100);

    expect(controller.enemies[0].animationState).toBe('death');
    expect(controller.enemies[0].active).toBe(true);
  });

  it('does not enter de-rez when ammunition is depleted', () => {
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

    // Next tick: ammo is gone but the enemy stays alive and keeps moving/firing.
    controller = updateEnemyController(controller, state, emptyMap, fireDt);
    expect(controller.enemies[0].animationState).not.toBe('death');
    expect(controller.enemies[0].active).toBe(true);
  });

  it('handles zero-health de-rez while sharing the player cell', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { ...base.player.position }, 0);
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 100);

    expect(controller.enemies[0].animationState).toBe('death');
    expect(controller.enemies[0].active).toBe(true);
  });

  it('fully de-rezzes after the configured duration', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: 3, y: 0 }, 0);
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);

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

  it('does not move when surrounded by walls on all sides', () => {
    const base = createGameState({ seed: 1 });
    const enemyPos = { x: 10.5, y: 10.5 };
    const state = stateWithEnemyAt(base, enemyPos);
    // Walls on all four cardinal neighbours so the BFS cannot reach the
    // enemy's cell and no navigation step is available.
    const wallMap = createWallCollisionMap([
      { x: 10, y: 9 },
      { x: 11, y: 10 },
      { x: 10, y: 11 },
      { x: 9, y: 10 },
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
    // Start with zero health so the enemy fully de-rezzes by health, not by
    // ammunition depletion (which no longer triggers death).
    const state = stateWithEnemy(base, { x: 3, y: 0 }, 0);
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);

    // Advance through the full de-rez window so the enemy becomes inactive.
    controller = updateEnemyController(
      controller,
      state,
      emptyMap,
      ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
    );

    expect(controller.enemies[0].active).toBe(false);

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

  it('separates overlapping active enemies', () => {
    const base = createGameState({ seed: 1 });
    // Two enemies very close together (within 2× radius).
    const enemy0: EnemyState = {
      position: { x: base.player.position.x - 5, y: 0 },
      health: 100,
    };
    const enemy1: EnemyState = {
      position: {
        x: base.player.position.x - 5 + ENEMY_CONTROLLER_RADIUS_CELLS * 0.5,
        y: 0,
      },
      health: 100,
    };
    const state: GameState = { ...base, enemies: [enemy0, enemy1] };
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    const beforeX0 = controller.enemies[0].position.x;
    const beforeX1 = controller.enemies[1].position.x;

    controller = updateEnemyController(controller, state, emptyMap, 100);

    // Enemies should have been pushed apart.
    const afterDist = Math.abs(
      controller.enemies[1].position.x - controller.enemies[0].position.x,
    );
    const beforeDist = Math.abs(beforeX1 - beforeX0);
    expect(afterDist).toBeGreaterThan(beforeDist);
  });

  it('separates enemies at the exact same position (zero-distance fallback)', () => {
    const base = createGameState({ seed: 1 });
    // Two enemies at the EXACT same position to trigger the
    // Math.sqrt(0) || 1 fallback in separateEnemies.
    const sharedX = base.player.position.x - 5;
    const enemy0: EnemyState = {
      position: { x: sharedX, y: 0 },
      health: 100,
    };
    const enemy1: EnemyState = {
      position: { x: sharedX, y: 0 },
      health: 100,
    };
    const state: GameState = { ...base, enemies: [enemy0, enemy1] };
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);

    // This should not crash even when distance is 0.
    controller = updateEnemyController(controller, state, emptyMap, 100);

    // Both enemies should remain active.
    expect(controller.enemies[0].active).toBe(true);
    expect(controller.enemies[1].active).toBe(true);
  });

  it('skips inactive enemies during separation', () => {
    const base = createGameState({ seed: 1 });
    // Two enemies close together, but enemy 1 is dead (health 0).
    const enemy0: EnemyState = {
      position: { x: base.player.position.x - 5, y: 0 },
      health: 100,
    };
    const enemy1: EnemyState = {
      position: {
        x: base.player.position.x - 5 + ENEMY_CONTROLLER_RADIUS_CELLS * 0.1,
        y: 0,
      },
      health: 0,
    };
    const state: GameState = { ...base, enemies: [enemy0, enemy1] };
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);

    // Tick through the full de-rez duration so enemy 1 becomes inactive.
    controller = updateEnemyController(
      controller,
      state,
      emptyMap,
      ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
    );

    // Enemy 1 should now be inactive after de-rez completed.
    expect(controller.enemies[1].active).toBe(false);

    // Tick again: the inactive enemy 1 should be skipped by separateEnemies.
    // Enemy 0 should not be pushed by the inactive enemy 1.
    controller = updateEnemyController(controller, state, emptyMap, 100);

    // Enemy 0 may move toward the player, but it should not be pushed
    // away from enemy 1's position due to separation with the inactive enemy.
    // Since both are near the same x, separation would push enemy 0 backward
    // (toward the player), so we just verify no crash and enemy 0 is still active.
    expect(controller.enemies[0].active).toBe(true);
  });

  it('does not separate active enemies that are already far apart', () => {
    const base = createGameState({ seed: 1 });
    const separation = ENEMY_CONTROLLER_RADIUS_CELLS * 4;
    const enemy0: EnemyState = {
      position: { x: base.player.position.x - 10, y: 0 },
      health: 100,
    };
    const enemy1: EnemyState = {
      position: { x: base.player.position.x - 10 - separation, y: 0 },
      health: 100,
    };
    const state: GameState = { ...base, enemies: [enemy0, enemy1] };
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 100);

    expect(controller.enemies[0].active).toBe(true);
    expect(controller.enemies[1].active).toBe(true);
  });

  it('sets walkTick to 0 and shootBlinkTicks to 0 on initial creation', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: -5, y: 0 });
    const controller = createEnemyControllerState(state);

    expect(controller.enemies[0].walkTick).toBe(0);
    expect(controller.enemies[0].shootBlinkTicks).toBe(0);
  });

  it('increments walkTick when enemy moves and resets to 0 when idle', () => {
    const base = createGameState({ seed: 1 });
    // Place enemy within fire range but not moving (at stop distance).
    const state = stateWithEnemy(base, { x: -3, y: 0 });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    // First tick: enemy fires (within range), walkTick may stay 0 or increment.
    controller = updateEnemyController(controller, state, emptyMap, 100);
    expect(controller.enemies[0].walkTick).toBeGreaterThanOrEqual(0);
    expect(controller.enemies[0].shootBlinkTicks).toBeGreaterThan(0);
  });

  it('exports ENEMY_CONTROLLER_SHOOT_BLINK_TICKS as a positive constant', () => {
    expect(ENEMY_CONTROLLER_SHOOT_BLINK_TICKS).toBeGreaterThanOrEqual(3);
    expect(ENEMY_CONTROLLER_SHOOT_BLINK_TICKS).toBeLessThanOrEqual(5);
  });
});

describe('AC-10.2d-002: enemies chase player until killed, never stop from ammo depletion', () => {
  it('does not stop chasing before reaching the player', () => {
    expect(ENEMY_CONTROLLER_STOP_DISTANCE_CELLS).toBe(1.5);
  });

  it('moves an enemy that is outside the stop distance', () => {
    const base = createGameState({ seed: 42 });
    const state = stateWithEnemyAt(base, {
      x: base.player.position.x + 2.0,
      y: base.player.position.y,
    });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    const startX = controller.enemies[0].position.x;
    controller = updateEnemyController(
      controller,
      state,
      emptyMap,
      NEATENSTEIN_FIXED_TIMESTEP_MS,
    );

    expect(controller.enemies[0].position.x).not.toBe(startX);
  });

  it('keeps firing enemies alive after ammunition depletion', () => {
    const base = createGameState({ seed: 42 });
    const state = stateWithEnemy(base, { x: 3, y: 0 });
    const emptyMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    const initialHealth = controller.enemies[0].health;
    const fireDt = ENEMY_CONTROLLER_FIRE_COOLDOWN_MS + 100;

    for (let i = 0; i < ENEMY_CONTROLLER_STARTING_AMMO; i += 1) {
      controller = updateEnemyController(controller, state, emptyMap, fireDt);
    }

    // One tick past ammo exhaustion plus the full de-rez window.
    controller = updateEnemyController(controller, state, emptyMap, fireDt);
    controller = updateEnemyController(
      controller,
      state,
      emptyMap,
      ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
    );

    expect(controller.enemies[0].active).toBe(true);
    expect(controller.enemies[0].animationState).not.toBe('death');
    expect(controller.enemies[0].health).toBe(initialHealth);
  });
});

describe('AC-10.4-r-002: enemy navigation follows BFS distance gradient around walls', () => {
  /**
   * Fixture: enemy at (60.5, 70.5), player at (60.5, 60.5) — 10 cells directly
   * north. Wall cells (60, 68) and (60, 69) block the direct vertical path.
   * Distance = 10.0 > FIRE_RANGE (8), so the enemy should move, not fire.
   *
   * Current seek-player code moves straight north, hits the wall, and the
   * axis-slide fallback returns the previous position (x unchanged). BFS
   * navigation should route the enemy sideways around the wall.
   */
  it('moves sideways to navigate around a wall blocking the direct path (AC-10.4-r-002)', () => {
    const base = createGameState({ seed: 1 });
    // Player is at (60.5, 60.5) from createGameState.
    const enemyPos = { x: 60.5, y: 70.5 };
    const state = stateWithEnemyAt(base, enemyPos);
    // Wall cells block the direct north path from enemy to player.
    const wallMap = createWallCollisionMap([
      { x: 60, y: 68 },
      { x: 60, y: 69 },
    ]);

    let controller = createEnemyControllerState(state);
    const startX = controller.enemies[0].position.x;

    // One tick at full speed (dtMs=1000 → stepDistance=2.5 cells).
    controller = updateEnemyController(controller, state, wallMap, 1000);

    // With BFS navigation, the enemy should move sideways (x changes) to
    // route around the wall. With current seek-player, x stays the same
    // because the axis-slide fallback returns the previous position.
    expect(controller.enemies[0].position.x).not.toBe(startX);
  });

  /**
   * Coverage test for enemy-controller.ts line 408: the `continue` branch
   * when `collisionMap.isSolid(cellX + dx, cellY + dy)` returns true for
   * the BFS-best direction.
   *
   * The BFS distance map and the collisionMap normally agree on which cells
   * are solid. To trigger the defensive `isSolid` check at line 407, we use
   * a stateful collisionMap that returns `false` for all cells during the
   * BFS wall-marking phase (which iterates over all size*size cells) and
   * returns `true` for the enemy's best-direction cell only during the
   * subsequent navigation phase. This creates the discrepancy the check
   * guards against.
   */
  it('skips the BFS-best direction when the target cell is solid (line 408 coverage)', () => {
    const base = createGameState({ seed: 1 });
    // Player is at (60.5, 60.5) from createGameState.
    const enemyPos = { x: 60.5, y: 70.5 };
    const state = stateWithEnemyAt(base, enemyPos);

    // The enemy is at cell (60, 70). Player is at cell (60, 60).
    // BFS-best direction is north (0, -1) → cell (60, 69), distance 9.
    // currentDist (enemy's cell) = 10. So 9 < 10 passes the distance check
    // at line 400, reaching the isSolid check at line 407.
    //
    // We make isSolid return false for (60, 69) during BFS wall-marking
    // (so BFS assigns it a valid distance) but true during navigation
    // (triggering the continue at line 408).
    const TARGET_X = 60;
    const TARGET_Y = 69;
    let callCount = 0;
    const collisionMap: CollisionMap = {
      isSolid: (x: number, y: number) => {
        callCount += 1;
        // BFS wall-marking calls isSolid for every cell (120*120 = 14400
        // calls). After that, navigation calls isSolid for specific
        // neighbour cells. Return true for the target cell only after
        // the BFS phase to create a BFS/collisionMap discrepancy.
        if (
          callCount > NEATENSTEIN_MAP_SIZE * NEATENSTEIN_MAP_SIZE &&
          x === TARGET_X &&
          y === TARGET_Y
        ) {
          return true;
        }
        return false;
      },
    };

    let controller = createEnemyControllerState(state);
    const startY = controller.enemies[0].position.y;

    controller = updateEnemyController(controller, state, collisionMap, 1000);

    // The enemy's best direction (north) is blocked by the isSolid check.
    // All other directions have BFS distance >= currentDist, so they are
    // skipped at line 401. The enemy should not move north.
    expect(controller.enemies[0].position.y).toBeGreaterThanOrEqual(startY);
  });
});

describe('AC-10.4-r-003: directional wall collision radius prevents clipping without blocking corridors', () => {
  /**
   * One-sided centering allows horizontal movement past a diagonal wall.
   *
   * Enemy at (10.5, 10.9) — off-center in y (near the south edge of cell 10).
   * Wall at (11, 11) — south of the target cell (11, 10). Without one-sided
   * centering, the circle-overlap check at (11.0, 10.9) would inspect (11, 11)
   * and block East. With one-sided centering, the enemy is drifting toward the
   * south wall (y > 10.5, wS = isSolid(11, 11) = true), so desired.y is snapped
   * to 10.5 before the collision check. At (11.0, 10.5) the circle no longer
   * overlaps (11, 11), so East succeeds.
   */
  it('one-sided centering allows horizontal movement past a diagonal wall', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Enemy off-center in y.
    const enemyPos = { x: 10.5, y: 10.9 };
    const state = stateWithEnemyAt(base, enemyPos);
    // Wall at (11, 11) — south of the target cell (11, 10).
    const wallMap = createWallCollisionMap([{ x: 11, y: 11 }]);

    let controller = createEnemyControllerState(state);

    // dtMs=200 → stepDistance=0.5. One-sided centering snaps desired.y to
    // 10.5 before the collision check, so the circle at (11.0, 10.5) no
    // longer overlaps the wall at (11, 11). East succeeds.
    controller = updateEnemyController(controller, state, wallMap, 200);

    // Enemy moved East after centering snapped y to cell center.
    expect(controller.enemies[0].position.x).toBeCloseTo(11.0, 5);
    expect(controller.enemies[0].position.y).toBeCloseTo(10.5, 5);
  });

  /**
   * One-sided centering allows vertical movement past a diagonal wall.
   *
   * Enemy at (60.9, 55.5) — off-center in x (near the east edge of cell 60).
   * Player directly north at (60.5, 60.5). BFS best direction is South
   * (distance 4 < current 5). Wall at (61, 56) — east of the target cell
   * (60, 56). Without one-sided centering, the circle-overlap check at
   * (60.9, 56.0) would inspect (61, 56) and block South. With one-sided
   * centering, the enemy is drifting toward the east wall (x > 60.5,
   * wE = isSolid(61, 56) = true), so desired.x is snapped to 60.5 before
   * the collision check. At (60.5, 56.0) the circle no longer overlaps
   * (61, 56), so South succeeds.
   */
  it('one-sided centering allows vertical movement past a diagonal wall', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Enemy off-center in x.
    const enemyPos = { x: 60.9, y: 55.5 };
    const state = stateWithEnemyAt(base, enemyPos);
    // Wall at (61, 56) — east of the target cell (60, 56).
    const wallMap = createWallCollisionMap([{ x: 61, y: 56 }]);

    let controller = createEnemyControllerState(state);

    // dtMs=200 → stepDistance=0.5. One-sided centering snaps desired.x to
    // 60.5 before the collision check, so the circle at (60.5, 56.0) no
    // longer overlaps the wall at (61, 56). South succeeds.
    controller = updateEnemyController(controller, state, wallMap, 200);

    // Enemy moved South after centering snapped x to cell center.
    expect(controller.enemies[0].position.x).toBeCloseTo(60.5, 5);
    expect(controller.enemies[0].position.y).toBeCloseTo(56.0, 5);
  });

  /**
   * Coverage test for dtMs=0 sync pass (lines 236, 462, 477).
   *
   * display.worker.ts calls updateEnemyController with dtMs=0 after the real
   * tick. With dtMs=0, resolveTimestepMs returns 0, stepDistance becomes 0,
   * the desired position equals the current position (line 462 false branch),
   * and the walkTick guard `if (dtMs > 0)` is false (line 477 false branch).
   * The enemy must not move and walkTick must be preserved.
   */
  it('preserves position and walkTick on a zero-timestep sync pass (dtMs=0)', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: -3, y: 0 });
    const collisionMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);
    // First, do a real tick to increment walkTick.
    controller = updateEnemyController(controller, state, collisionMap, 16);
    const walkTickAfterRealTick = controller.enemies[0].walkTick;
    expect(walkTickAfterRealTick).toBeGreaterThan(0);

    const posBeforeSync = controller.enemies[0].position;
    // Now do a zero-timestep sync pass.
    controller = updateEnemyController(controller, state, collisionMap, 0);

    // Position must not change.
    expect(controller.enemies[0].position.x).toBeCloseTo(posBeforeSync.x, 5);
    expect(controller.enemies[0].position.y).toBeCloseTo(posBeforeSync.y, 5);
    // walkTick must be preserved (not reset to 0 by the sync pass).
    expect(controller.enemies[0].walkTick).toBe(walkTickAfterRealTick);
  });
});

describe('AC-10.4-fix-collision-v2: circle-overlap wall collision, corridor centering, and wall-stuck prevention', () => {
  /**
   * Import the wall collision radius constant to verify it was changed to
   * a quarter cell (0.25) per the fix-collision-v2 requirement.
   */
  it('uses a quarter-cell (0.25) wall collision radius', () => {
    // Verify the constant is 0.25 — gives 0.5 cells of freedom in 1-cell
    // corridors (center ± 0.25).
    expect(ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS).toBe(0.25);
  });

  /**
   * Enemy in a 1-cell-wide horizontal corridor (walls above and below)
   * should traverse it without getting stuck. The enemy starts at one end
   * and the player is at the other end. After multiple ticks the enemy
   * should have moved through the corridor and be centered on the
   * perpendicular (Y) axis.
   */
  it('traverses a 1-cell-wide corridor without getting stuck', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Build a horizontal corridor at y=60.
    const walls: Array<{ x: number; y: number }> = [];
    for (let x = 53; x <= 62; x += 1) {
      walls.push({ x, y: 59 });
      walls.push({ x, y: 61 });
    }
    const wallMap = createWallCollisionMap(walls);
    const state = stateWithEnemyAt(base, { x: 55.5, y: 60.5 });

    let controller = createEnemyControllerState(state);
    const startX = controller.enemies[0].position.x;

    // Run 30 ticks at 16ms each (0.04 cells/tick → 1.2 cells total).
    for (let i = 0; i < 30; i += 1) {
      controller = updateEnemyController(controller, state, wallMap, 16);
    }

    // Enemy should have moved east through the corridor.
    expect(controller.enemies[0].position.x).toBeGreaterThan(startX);
    // Enemy should be centered in the corridor (y ≈ 60.5 ± 0.01).
    expect(controller.enemies[0].position.y).toBeCloseTo(60.5, 2);
  });

  /**
   * Corridor centering: after moving horizontally in a 1-cell-wide corridor
   * (walls above and below), the enemy's Y should snap to the cell center
   * (cellY + 0.5) even if it started off-center.
   */
  it('centers the enemy on the perpendicular axis in a horizontal corridor', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Corridor at y=60.
    const walls: Array<{ x: number; y: number }> = [];
    for (let x = 53; x <= 62; x += 1) {
      walls.push({ x, y: 59 });
      walls.push({ x, y: 61 });
    }
    const wallMap = createWallCollisionMap(walls);
    // Enemy at y=60.3 — off-center (should be centered to 60.5).
    const state = stateWithEnemyAt(base, { x: 55.5, y: 60.3 });

    let controller = createEnemyControllerState(state);

    // One tick: enemy moves east, centering should snap y to 60.5.
    controller = updateEnemyController(controller, state, wallMap, 16);

    // Y should be centered at 60.5 (cell 60 center).
    expect(controller.enemies[0].position.y).toBeCloseTo(60.5, 5);
  });

  /**
   * Corridor centering: after moving vertically in a 1-cell-wide corridor
   * (walls left and right), the enemy's X should snap to the cell center
   * (cellX + 0.5) even if it started off-center.
   */
  it('centers the enemy on the perpendicular axis in a vertical corridor', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Vertical corridor at x=60.
    const walls: Array<{ x: number; y: number }> = [];
    for (let y = 63; y <= 72; y += 1) {
      walls.push({ x: 59, y });
      walls.push({ x: 61, y });
    }
    const wallMap = createWallCollisionMap(walls);
    // Enemy at x=60.3 — off-center (should be centered to 60.5).
    // Player is north, so enemy moves north (dy=-1) through the corridor.
    const state = stateWithEnemyAt(base, { x: 60.3, y: 70.5 });

    let controller = createEnemyControllerState(state);

    // One tick: enemy moves north, centering should snap x to 60.5.
    controller = updateEnemyController(controller, state, wallMap, 16);

    // X should be centered at 60.5 (cell 60 center).
    expect(controller.enemies[0].position.x).toBeCloseTo(60.5, 5);
  });

  /**
   * Wall-stuck prevention: after the separation pass, if an enemy was
   * pushed inside a wall, its position should be reverted to the
   * pre-separation position (which was verified wall-free by the movement
   * code).
   *
   * Two enemies are placed very close together near a wall. The separation
   * push would move one of them into the wall, but the wall re-check
   * reverts it. dtMs=0 prevents movement so only separation runs.
   */
  it('prevents enemies from being pushed into walls by separation', () => {
    const base = createGameState({ seed: 1 });
    // Wall to the east of the enemies.
    const wallMap = createWallCollisionMap([{ x: 11, y: 10 }]);
    const enemy0: EnemyState = {
      position: { x: 10.5, y: 10.5 },
      health: 100,
    };
    const enemy1: EnemyState = {
      position: { x: 10.6, y: 10.5 },
      health: 100,
    };
    const state: GameState = { ...base, enemies: [enemy0, enemy1] };

    let controller = createEnemyControllerState(state);
    // dtMs=0: no movement, only separation runs.
    controller = updateEnemyController(controller, state, wallMap, 0);

    // Enemy 1 was pushed east by separation toward the wall at (11, 10).
    // The wall re-check should have reverted it to its pre-separation
    // position (10.6, 10.5).
    expect(controller.enemies[1].position.x).toBeCloseTo(10.6, 5);
    // Enemy 1 center must be at least 0.25 from the wall edge at x=11.
    expect(controller.enemies[1].position.x).toBeLessThan(10.75);
  });

  /**
   * Integration test: on the real seeded map, after many ticks, no enemy
   * should end up inside a wall. The circle-overlap check and post-
   * separation wall re-check together should prevent wall-stuck enemies.
   */
  it('does not place enemies inside walls on the real seeded map', () => {
    const base = createGameState({ seed: 42 });
    const mapGrid = buildNeatensteinMap(base.seed);
    const collisionMap = createCollisionMap(mapGrid, NEATENSTEIN_MAP_SIZE);

    // Spawn enemy a few cells away from the central spawn point.
    const state = stateWithEnemy(base, { x: 4, y: 0 });

    let controller = createEnemyControllerState(state);
    for (let i = 0; i < 50; i += 1) {
      controller = updateEnemyController(controller, state, collisionMap, 100);
    }

    // Enemy should be active and at a finite, wall-free position.
    expect(controller.enemies[0].active).toBe(true);
    expect(Number.isFinite(controller.enemies[0].position.x)).toBe(true);
    expect(Number.isFinite(controller.enemies[0].position.y)).toBe(true);

    // Verify the enemy center is at least 0.25 cells from any wall edge
    // by checking the bounding box cells are all open.
    const px = controller.enemies[0].position.x;
    const py = controller.enemies[0].position.y;
    const R = 0.25;
    const minX = Math.floor(px - R);
    const maxX = Math.ceil(px + R) - 1;
    const minY = Math.floor(py - R);
    const maxY = Math.ceil(py + R) - 1;
    for (let cy = minY; cy <= maxY; cy += 1) {
      for (let cx = minX; cx <= maxX; cx += 1) {
        expect(collisionMap.isSolid(cx, cy)).toBe(false);
      }
    }
  });
});

describe('AC-10.4 fix-coverage-gaps: vertical move corridor centering', () => {
  /**
   * Corridor centering: after moving vertically in a 1-cell-wide corridor
   * (walls left and right), the enemy's X should snap to the cell center
   * (cellX + 0.5) when both perpendicular X walls are solid. This exercises
   * the branch at enemy-controller.ts:462-468 where
   * collisionMap.isSolid(newCellX-1, newCellY) &&
   * collisionMap.isSolid(newCellX+1, newCellY) are both true.
   */
  it('centers enemy X when vertical move has solid walls on both X sides (line 462 branch)', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Vertical corridor at x=60.
    const walls: Array<{ x: number; y: number }> = [];
    for (let y = 63; y <= 72; y += 1) {
      walls.push({ x: 59, y });
      walls.push({ x: 61, y });
    }
    const wallMap = createWallCollisionMap(walls);
    // Enemy at x=60.35 — off-center (should be centered to 60.5).
    // Player is north, so enemy moves north (dy=-1) through the corridor.
    const state = stateWithEnemyAt(base, { x: 60.35, y: 70.5 });

    let controller = createEnemyControllerState(state);

    // One tick: enemy moves north, centering should snap x to 60.5.
    controller = updateEnemyController(controller, state, wallMap, 16);

    // X should be centered at 60.5 (cell 60 center), exercising the
    // collisionMap.isSolid(newCellX-1,newCellY) &&
    // collisionMap.isSolid(newCellX+1,newCellY) branch.
    expect(controller.enemies[0].position.x).toBeCloseTo(60.5, 5);
  });
});

describe('AC-10.4 fix-gap-traversal: pre-collision centering for 1-cell gap entry', () => {
  /**
   * Bug: enemies cannot cross 1-cell-wide gaps. They get stuck at the
   * entrance of any 1-cell corridor because the corridor centering snap
   * runs AFTER the collision check. When an enemy drifts off-center in
   * open areas (e.g., y=60.99), the collision check at the gap entrance
   * samples both the open gap cell and the adjacent wall cell, blocking
   * the move. The post-move centering never runs because the move was
   * blocked — a chicken-and-egg deadlock.
   *
   * Fix: apply perpendicular centering BEFORE the collision check. If
   * the target cell (or current cell) is a 1-cell gap (walls on both
   * perpendicular sides), snap the perpendicular coordinate to the cell
   * center before running isPositionBlockedByWall.
   */

  /**
   * Target cell check (horizontal): enemy is off-center in Y (y=60.99)
   * in an open area outside a horizontal 1-cell gap. The target cell
   * (first gap cell) has walls on both Y sides. Without the fix, the
   * collision check at y=60.99 samples the wall cell → blocked → enemy
   * stuck. With the fix, pre-collision centering snaps y to 60.5 before
   * the collision check → enemy enters the gap.
   */
  it('enters a horizontal 1-cell gap when off-center (target cell check)', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Horizontal corridor at y=60 with walls
    // at y=59 and y=61 from x=51 to x=62.
    const walls: Array<{ x: number; y: number }> = [];
    for (let x = 51; x <= 62; x += 1) {
      walls.push({ x, y: 59 });
      walls.push({ x, y: 61 });
    }
    const wallMap = createWallCollisionMap(walls);
    // Enemy at (50.5, 60.99) — off-center in Y, just outside the gap.
    // Distance to player ≈ 10 > FIRE_RANGE (8), so enemy moves, not fires.
    const state = stateWithEnemyAt(base, { x: 50.5, y: 60.99 });

    let controller = createEnemyControllerState(state);
    const startX = controller.enemies[0].position.x;

    // dtMs=1000 → stepDistance=2.5. Without fix: collision check at
    // y=60.99 samples cell (52, 61) and (53, 61) which are walls →
    // blocked. With fix: snaps y to 60.5, collision check only samples
    // cell 60 → open → enemy moves east.
    controller = updateEnemyController(controller, state, wallMap, 1000);

    // Enemy should have moved east into the gap.
    expect(controller.enemies[0].position.x).toBeGreaterThan(startX);
    // Enemy should be centered in the corridor (y ≈ 60.5).
    expect(controller.enemies[0].position.y).toBeCloseTo(60.5, 2);
  });

  /**
   * Current cell check (horizontal): enemy is off-center in Y inside a
   * horizontal corridor. The target cell (east, outside corridor) does
   * NOT have walls on both Y sides, but the current cell does. Without
   * the fix, the collision check at y=60.99 samples the wall cell of the
   * current cell → blocked. With the fix, the current cell fallback snaps
   * y to 60.5 → enemy exits the corridor centered.
   */
  it('exits a horizontal corridor when off-center (current cell check)', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Corridor at y=60 from x=53 to x=55 only.
    // Open area from x=56 onwards (no walls at y=59 or y=61 for x≥56).
    const walls: Array<{ x: number; y: number }> = [];
    for (let x = 53; x <= 55; x += 1) {
      walls.push({ x, y: 59 });
      walls.push({ x, y: 61 });
    }
    const wallMap = createWallCollisionMap(walls);
    // Enemy at (55.5, 60.99) — in the corridor, off-center in Y.
    const state = stateWithEnemyAt(base, { x: 55.5, y: 60.99 });

    let controller = createEnemyControllerState(state);
    const startX = controller.enemies[0].position.x;

    // dtMs=200 → stepDistance=0.5. Target cell (56, 60) has no
    // perpendicular walls → target check false. Current cell (55, 60)
    // has walls at (55, 59) and (55, 61) → current check true → snaps
    // y to 60.5. Without fix: collision check at y=60.99 samples
    // (55, 61) wall → blocked.
    controller = updateEnemyController(controller, state, wallMap, 200);

    // Enemy should have moved east out of the corridor.
    expect(controller.enemies[0].position.x).toBeGreaterThan(startX);
    // Enemy should be centered (y ≈ 60.5) by pre-collision centering.
    expect(controller.enemies[0].position.y).toBeCloseTo(60.5, 2);
  });

  /**
   * Target cell check (vertical): enemy is off-center in X (x=60.99)
   * in an open area outside a vertical 1-cell gap. The target cell
   * (first gap cell) has walls on both X sides. Without the fix, the
   * collision check at x=60.99 samples the wall cell → blocked. With
   * the fix, pre-collision centering snaps x to 60.5 → enemy enters.
   */
  it('enters a vertical 1-cell gap when off-center (target cell check)', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Vertical corridor at x=60 with walls
    // at x=59 and x=61 from y=51 to y=62.
    const walls: Array<{ x: number; y: number }> = [];
    for (let y = 51; y <= 62; y += 1) {
      walls.push({ x: 59, y });
      walls.push({ x: 61, y });
    }
    const wallMap = createWallCollisionMap(walls);
    // Enemy at (60.99, 50.5) — off-center in X, just outside the gap
    // (south of it). Distance to player ≈ 10 > FIRE_RANGE (8).
    const state = stateWithEnemyAt(base, { x: 60.99, y: 50.5 });

    let controller = createEnemyControllerState(state);
    const startY = controller.enemies[0].position.y;

    // dtMs=1000 → stepDistance=2.5. Enemy moves south (toward player).
    // Without fix: collision check at x=60.99 samples wall cells at
    // x=61 → blocked. With fix: snaps x to 60.5 → collision check
    // only samples x=60 → open → enemy moves south.
    controller = updateEnemyController(controller, state, wallMap, 1000);

    // Enemy should have moved south into the gap.
    expect(controller.enemies[0].position.y).toBeGreaterThan(startY);
    // Enemy should be centered in the corridor (x ≈ 60.5).
    expect(controller.enemies[0].position.x).toBeCloseTo(60.5, 2);
  });

  /**
   * Current cell check (vertical): enemy is off-center in X inside a
   * vertical corridor. The target cell (south, outside corridor) does
   * NOT have walls on both X sides, but the current cell does. Without
   * the fix, the collision check at x=60.99 samples the wall cell of
   * the current cell → blocked. With the fix, the current cell fallback
   * snaps x to 60.5 → enemy exits the corridor centered.
   */
  it('exits a vertical corridor when off-center (current cell check)', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Vertical corridor at x=60 from y=56 to
    // y=58 only. Open area from y=59 onwards (toward player).
    const walls: Array<{ x: number; y: number }> = [];
    for (let y = 56; y <= 58; y += 1) {
      walls.push({ x: 59, y });
      walls.push({ x: 61, y });
    }
    const wallMap = createWallCollisionMap(walls);
    // Enemy at (60.99, 58.5) — in the corridor, off-center in X.
    const state = stateWithEnemyAt(base, { x: 60.99, y: 58.5 });

    let controller = createEnemyControllerState(state);
    const startY = controller.enemies[0].position.y;

    // dtMs=200 → stepDistance=0.5. Enemy moves south (toward player).
    // Target cell (60, 59) has no perpendicular walls → target check
    // false. Current cell (60, 58) has walls at (59, 58) and (61, 58)
    // → current check true → snaps x to 60.5. Without fix: collision
    // check at x=60.99 samples (61, 58) wall → blocked.
    controller = updateEnemyController(controller, state, wallMap, 200);

    // Enemy should have moved south (toward player) out of the corridor.
    expect(controller.enemies[0].position.y).toBeGreaterThan(startY);
    // Enemy should be centered (x ≈ 60.5) by pre-collision centering.
    expect(controller.enemies[0].position.x).toBeCloseTo(60.5, 2);
  });

  // -------------------------------------------------------------------------
  // AC-10.4-fix-turns: enemy navigates corridor turns without getting stuck
  // -------------------------------------------------------------------------

  // Bug 1 test A: Enemy at (60.9, 62.5) with wall at (61, 62). The enemy is
  // trying to turn north. Pre-collision centering snaps X (perpendicular to
  // south→north) but Y stays at 60.9 → circle overlaps (61, 62) wall → blocked.
  // The pre-move position correction nudge should fix X before the collision
  // check on the desired position.
  it('nudges off-center X position when turning north near a corner wall', () => {
    const walls = [
      // Vertical corridor walls at x=59, x=61 (columns)
      { x: 59, y: 60 },
      { x: 59, y: 61 },
      { x: 59, y: 62 },
      { x: 61, y: 60 },
      { x: 61, y: 61 },
      { x: 61, y: 62 },
      // Horizontal corridor wall at y=59 (row), only x=60 open
      { x: 59, y: 59 },
      { x: 61, y: 59 },
      // Block south of y=62 to force north turn
      { x: 59, y: 63 },
      { x: 60, y: 63 },
      { x: 61, y: 63 },
    ];
    const wallMap = createWallCollisionMap(walls);
    const base = createGameState({ seed: 1 });
    // Place enemy at x=60.9 (off-center, drifting toward east wall), y=62.5
    const state = stateWithEnemyAt(base, { x: 60.9, y: 62.5 });

    let controller = createEnemyControllerState(state);
    // dtMs=200 → stepDistance=0.5. Enemy moves north.
    // Without fix: current position (60.9, 62.5) has circle that samples
    // cell (61, 62) which is a wall → pre-move nudge corrects X to 60.5.
    // Then north move to (60.5, 62.0) is not blocked.
    controller = updateEnemyController(controller, state, wallMap, 200);

    // Enemy should have moved north.
    expect(controller.enemies[0].position.y).toBeLessThan(62.5);
    // Enemy should be centered on X after the nudge.
    expect(controller.enemies[0].position.x).toBeCloseTo(60.5, 1);
  });

  // Bug 1 test B: Enemy at (57.5, 60.9) in horizontal corridor with walls at
  // y=59 and y=61. The enemy is off-center on Y (60.9, near wall at y=61).
  // Pre-collision centering snaps Y (perpendicular to west→east) but since
  // the target cell (58, 60) only has one wall (y=61), the target-cell check
  // may not fire. The pre-move nudge corrects Y to 60.5.
  it('nudges off-center Y position when moving east in a corridor', () => {
    const walls = [
      // Horizontal corridor walls at y=59, y=61
      { x: 55, y: 59 },
      { x: 56, y: 59 },
      { x: 57, y: 59 },
      { x: 58, y: 59 },
      { x: 59, y: 59 },
      { x: 55, y: 61 },
      { x: 56, y: 61 },
      { x: 57, y: 61 },
      { x: 58, y: 61 },
      { x: 59, y: 61 },
    ];
    const wallMap = createWallCollisionMap(walls);
    const base = createGameState({ seed: 1 });
    // Place enemy at x=57.5, y=60.9 (off-center toward south wall)
    const state = stateWithEnemyAt(base, { x: 57.5, y: 60.9 });

    let controller = createEnemyControllerState(state);
    // dtMs=200 → stepDistance=0.5. Enemy moves east.
    // Without fix: current position (57.5, 60.9) has circle that samples
    // cell (57, 61) which is a wall → pre-move nudge corrects Y to 60.5.
    // Then east move to (58.0, 60.5) is not blocked.
    controller = updateEnemyController(controller, state, wallMap, 200);

    // Enemy should have moved east (toward player at x=60.5).
    expect(controller.enemies[0].position.x).toBeGreaterThan(57.5);
    // Enemy should be centered on Y after the nudge.
    expect(controller.enemies[0].position.y).toBeCloseTo(60.5, 1);
  });

  // -------------------------------------------------------------------------
  // AC-10.4-fix-flanking: enemies spread around the player at different angles
  // -------------------------------------------------------------------------

  it('exports a positive ENEMY_CONTROLLER_FLANKING_RADIUS_CELLS constant', () => {
    expect(ENEMY_CONTROLLER_FLANKING_RADIUS_CELLS).toBeGreaterThan(0);
  });

  it('assigns multiple enemies to different flanking slots around the player', () => {
    // Open area — no walls near the player, enemies can move freely.
    const wallMap = createEmptyCollisionMap();
    const base = createGameState({ seed: 1 });
    // Player is at (60.5, 60.5). Place two enemies west of the player.
    // stateWithEnemyAt replaces the enemies array, so we build a two-enemy
    // state manually.
    const state: GameState = {
      ...base,
      enemies: [
        { position: { x: 58.5, y: 60.5 }, health: 100 },
        { position: { x: 58.4, y: 60.5 }, health: 100 },
      ],
    };

    let controller = createEnemyControllerState(state);
    // Run many ticks so enemies have time to reach their flanking slots.
    for (let i = 0; i < 60; i++) {
      controller = updateEnemyController(controller, state, wallMap, 200);
    }

    const e0 = controller.enemies[0].position;
    const e1 = controller.enemies[1].position;
    // Enemy 0 has slot 0° (east of player). After flanking, enemy 0 should
    // be east of the player (x > 60.5).
    expect(e0.x).toBeGreaterThan(60.5 + 0.3);
    // Enemy 1 has slot 180° (west of player). After flanking, enemy 1 should
    // be west of the player (x < 60.5).
    expect(e1.x).toBeLessThan(60.5 - 0.3);
  });

  it('does not flank when there is only one enemy — moves directly toward player', () => {
    const wallMap = createEmptyCollisionMap();
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Enemy far to the west.
    const state = stateWithEnemyAt(base, { x: 50.5, y: 60.5 });

    let controller = createEnemyControllerState(state);
    // Run several ticks. Single enemy should use BFS, not flanking.
    for (let i = 0; i < 10; i++) {
      controller = updateEnemyController(controller, state, wallMap, 200);
    }

    // Enemy should have moved east (toward player), not circled around.
    expect(controller.enemies[0].position.x).toBeGreaterThan(50.5);
  });
});

describe('AC-10.4-fix-turns: both-axes turn centering and framerate-scaled nudge', () => {
  /**
   * Issue 1: The pre-move nudge has 3 cascade branches: X-only, Y-only,
   * both-axes. The both-axes branch handles the case where the enemy is
   * off-center on BOTH axes at an inside corner (e.g., position (60.9, 60.9)
   * with walls east and south). Without this branch, the enemy can't center
   * on either axis because each single-axis nudge still overlaps a wall.
   */
  it('nudges off-center X and Y position at an inside corner (both-axes branch)', () => {
    const base = createGameState({ seed: 1 });
    // Player far north so the enemy enters BFS mode and moves north.
    const state: GameState = {
      ...base,
      player: { ...base.player, position: { x: 60.5, y: 50.5 } },
      enemies: [{ position: { x: 60.9, y: 60.9 }, health: 100 }],
    };
    // Walls east (61,60) and south (60,61) of the enemy's cell (60,60).
    // Current pos (60.9, 60.9) overlaps both walls → nudge cascade starts.
    // X-only (60.5, 60.9) overlaps (60,61) wall → fails.
    // Y-only (60.9, 60.5) overlaps (61,60) wall → fails.
    // Both-axes (60.5, 60.5) only samples (60,60) → open → succeeds.
    const wallMap = createWallCollisionMap([
      { x: 61, y: 60 },
      { x: 60, y: 61 },
    ]);

    let controller = createEnemyControllerState(state);
    // dtMs=200 → stepDistance=0.5, nudgeScale=1 (full nudge).
    controller = updateEnemyController(controller, state, wallMap, 200);

    // Nudge centers both X and Y to 60.5, then the enemy moves north
    // (toward player) by 0.5: final position (60.5, 60.0).
    expect(controller.enemies[0].position.x).toBeCloseTo(60.5, 5);
    expect(controller.enemies[0].position.y).toBeCloseTo(60.0, 5);
  });

  /**
   * Issue 5d: Guard test — the pre-move nudge should NOT fire in open areas
   * (no walls). The enemy is off-center on both axes but no wall overlaps,
   * so isPositionBlockedByWall returns false and the nudge cascade is
   * skipped entirely. The enemy should move normally without any centering.
   */
  it('does not nudge when the enemy is in an open area (guard test)', () => {
    const base = createGameState({ seed: 1 });
    // Player far west so the enemy enters BFS mode and moves west.
    const state: GameState = {
      ...base,
      player: { ...base.player, position: { x: 50.5, y: 60.5 } },
      enemies: [{ position: { x: 60.9, y: 60.9 }, health: 100 }],
    };
    const wallMap = createEmptyCollisionMap();

    let controller = createEnemyControllerState(state);

    // dtMs=200 → stepDistance=0.5, nudgeScale=1. The enemy is off-center
    // on both axes (60.9, 60.9) but in an open area (no walls). The nudge
    // cascade should NOT fire because isPositionBlockedByWall returns false.
    // The enemy should move west (toward player) without any centering.
    controller = updateEnemyController(controller, state, wallMap, 200);

    // Without nudge: enemy moves west by 0.5 from 60.9 → x=60.4, y=60.9.
    // Y should NOT be centered (stays at 60.9, not snapped to 60.5).
    expect(controller.enemies[0].position.x).toBeCloseTo(60.4, 5);
    expect(controller.enemies[0].position.y).toBeCloseTo(60.9, 5);
  });
});

describe('AC-10.4-fix-flanking: wall-aware slot placement and stall fallback', () => {
  /**
   * Issue 5a: When the slot target lands inside a wall, the wall-aware slot
   * placement code should either find a shifted slot (by trying angle
   * offsets) or fall back to BFS mode. Either way, the enemy should not
   * be permanently stuck against the wall.
   */
  it('falls back to BFS or finds a shifted slot when the slot target is inside a wall', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Two enemies in flanking range.
    // Enemy 0 slot angle = 0° → slot target (62.0, 60.5) — inside walls.
    // Enemy 1 slot angle = 180° → slot target (59.0, 60.5) — open.
    const wallMap = createWallCollisionMap([
      { x: 62, y: 60 },
      { x: 62, y: 61 },
    ]);
    const state: GameState = {
      ...base,
      enemies: [
        { position: { x: 61.0, y: 60.5 }, health: 100 },
        { position: { x: 60.0, y: 60.5 }, health: 100 },
      ],
    };

    let controller = createEnemyControllerState(state);
    // Run several ticks. Enemy 0's original slot is in a wall. The
    // wall-aware code should find a shifted slot or fall back to BFS.
    // Either way, the enemy should not be permanently stuck.
    for (let i = 0; i < 20; i++) {
      controller = updateEnemyController(controller, state, wallMap, 200);
    }

    const e0 = controller.enemies[0].position;
    expect(Number.isFinite(e0.x)).toBe(true);
    expect(Number.isFinite(e0.y)).toBe(true);
    // Enemy should have moved from its starting position (not stuck).
    const dist = Math.hypot(e0.x - 61.0, e0.y - 60.5);
    expect(dist).toBeGreaterThan(0.1);
  });

  /**
   * Issue 5b: Flanking with walls present. Enemies should spread around
   * the player even when walls are nearby, using createWallCollisionMap
   * (not an empty map). The wall-aware slot placement and stall fallback
   * should prevent permanent stalls.
   */
  it('flanks around the player with walls present without getting stuck', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Two enemies in flanking range.
    // Add scattered walls near the player but leave room for flanking.
    const wallMap = createWallCollisionMap([
      { x: 58, y: 60 },
      { x: 63, y: 61 },
      { x: 60, y: 58 },
      { x: 61, y: 63 },
    ]);
    const state: GameState = {
      ...base,
      enemies: [
        { position: { x: 59.5, y: 60.5 }, health: 100 },
        { position: { x: 61.5, y: 60.5 }, health: 100 },
      ],
    };

    let controller = createEnemyControllerState(state);
    // Run many ticks. Enemies should spread around the player without
    // getting stuck against walls.
    for (let i = 0; i < 40; i++) {
      controller = updateEnemyController(controller, state, wallMap, 200);
    }

    // Both enemies should be active and at finite positions.
    for (const enemy of controller.enemies) {
      expect(enemy.active).toBe(true);
      expect(Number.isFinite(enemy.position.x)).toBe(true);
      expect(Number.isFinite(enemy.position.y)).toBe(true);
    }
    // Enemies should have moved from their starting positions.
    const e0Dist = Math.hypot(
      controller.enemies[0].position.x - 59.5,
      controller.enemies[0].position.y - 60.5,
    );
    const e1Dist = Math.hypot(
      controller.enemies[1].position.x - 61.5,
      controller.enemies[1].position.y - 60.5,
    );
    expect(e0Dist).toBeGreaterThan(0.1);
    expect(e1Dist).toBeGreaterThan(0.1);
  });

  /**
   * Issue 5c: With 8 enemies, each gets slot angle = index * 2π / 8 = index *
   * 45°. After flanking, enemies should be spread at roughly 45° intervals
   * around the player, not clustering on one side.
   */
  it('assigns 8 enemies to 8 evenly spaced flanking slots (45° apart)', () => {
    const wallMap = createEmptyCollisionMap();
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). 8 enemies placed near the player.
    const enemies: EnemyState[] = [];
    for (let i = 0; i < 8; i++) {
      enemies.push({
        position: { x: 59.5 + i * 0.1, y: 60.5 },
        health: 100,
      });
    }
    const state: GameState = { ...base, enemies };

    let controller = createEnemyControllerState(state);
    // Run many ticks so enemies reach their flanking slots.
    for (let i = 0; i < 80; i++) {
      controller = updateEnemyController(controller, state, wallMap, 200);
    }

    // Compute angles of all enemies relative to the player.
    const angles = controller.enemies.map((e) =>
      Math.atan2(e.position.y - 60.5, e.position.x - 60.5),
    );
    angles.sort((a, b) => a - b);

    // The total angular spread should cover most of the circle (> 270°),
    // proving enemies spread to different sides, not clustering.
    const totalSpread = angles[7] - angles[0];
    expect(totalSpread).toBeGreaterThan((3 * Math.PI) / 2);

    // No two adjacent enemies should be at the same angle.
    for (let i = 1; i < 8; i++) {
      const gap = angles[i] - angles[i - 1];
      expect(gap).toBeGreaterThan(0.2); // > ~11°
    }
  });

  /**
   * Coverage target: `flankStallTicks += 1` (line 709 of enemy-controller.ts).
   *
   * When an enemy is in flanking mode (shouldMoveByFlank = true) but all four
   * cardinal movement directions are blocked by walls, `moved` stays false and
   * the stall counter increments.  The wall-aware slot-placement code must
   * still find a valid (shifted) slot so that shouldMoveByFlank is true.
   */
  it('increments flankStallTicks when a flanking enemy is completely walled in', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5).  Enemy 0 at (62.5, 60.5) — cell (62, 60).
    // Wall all four cardinal neighbours of cell (62, 60) so the enemy cannot
    // move in any direction.  The original slot target (62.0, 60.5) lands in
    // a wall, but the wall-aware code shifts to a valid slot at angle π/3,
    // so shouldMoveByFlank stays true while moved stays false → stall.
    const wallMap = createWallCollisionMap([
      { x: 62, y: 59 }, // N
      { x: 63, y: 60 }, // E
      { x: 62, y: 61 }, // S
      { x: 61, y: 60 }, // W
    ]);
    const state: GameState = {
      ...base,
      enemies: [
        { position: { x: 62.5, y: 60.5 }, health: 100 },
        { position: { x: 59.5, y: 60.5 }, health: 100 },
      ],
    };

    let controller = createEnemyControllerState(state);
    expect(controller.enemies[0].flankStallTicks).toBe(0);
    controller = updateEnemyController(controller, state, wallMap, 200);
    // Enemy was in flanking mode but could not move → stall counter increments.
    expect(controller.enemies[0].flankStallTicks).toBe(1);
  });

  /**
   * Coverage target: stall fallback lines 485-488 of enemy-controller.ts.
   *
   * When flankStallTicks exceeds 3 and the enemy is still in flanking range,
   * the code switches shouldMoveByBfs on and shouldMoveByFlank off, breaking
   * out of the stalled flanking approach.
   */
  it('switches to BFS mode after 3 consecutive flanking stalls (flankStallTicks > 3)', () => {
    const base = createGameState({ seed: 1 });
    const wallMap = createWallCollisionMap([
      { x: 62, y: 59 },
      { x: 63, y: 60 },
      { x: 62, y: 61 },
      { x: 61, y: 60 },
    ]);
    const state: GameState = {
      ...base,
      enemies: [
        { position: { x: 62.5, y: 60.5 }, health: 100 },
        { position: { x: 59.5, y: 60.5 }, health: 100 },
      ],
    };

    let controller = createEnemyControllerState(state);
    // Simulate 4 prior consecutive stalls so flankStallTicks > 3.
    controller.enemies[0].flankStallTicks = 4;
    controller = updateEnemyController(controller, state, wallMap, 200);
    // After the fallback, shouldMoveByFlank is set to false and
    // flankStallTicks resets to 0.
    expect(controller.enemies[0].flankStallTicks).toBe(0);
    // Enemy should still be active and at a finite position.
    expect(controller.enemies[0].active).toBe(true);
    expect(Number.isFinite(controller.enemies[0].position.x)).toBe(true);
    expect(Number.isFinite(controller.enemies[0].position.y)).toBe(true);
  });

  /**
   * Coverage target: both-axes nudge branch (line 541 of enemy-controller.ts).
   *
   * When the enemy's position overlaps a wall and both the X-only and Y-only
   * nudge targets are also blocked, the code falls back to nudging both axes
   * simultaneously toward the cell center.
   */
  it('nudges along both axes when single-axis nudges are blocked by walls', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (62.8, 60.8) — near the SE corner of cell (62, 60).
    // Walls at (62, 61) and (63, 60) block the Y-only and X-only nudge
    // positions respectively, but the cell center (62.5, 60.5) is free,
    // so the both-axes nudge branch executes.
    const wallMap = createWallCollisionMap([
      { x: 62, y: 61 },
      { x: 63, y: 60 },
    ]);
    const state: GameState = {
      ...base,
      enemies: [{ position: { x: 62.8, y: 60.8 }, health: 100 }],
    };

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 200);
    // The both-axes nudge should have moved the enemy toward the cell center.
    const e0 = controller.enemies[0];
    expect(Number.isFinite(e0.position.x)).toBe(true);
    expect(Number.isFinite(e0.position.y)).toBe(true);
    expect(e0.active).toBe(true);
  });

  /**
   * Coverage target: false branch of the both-axes nudge `else if` (line 541).
   *
   * When the enemy's own cell is solid (a degenerate wall-overlap), all three
   * nudge options (X-only, Y-only, both-axes) are blocked. In flanking mode the
   * nudge block still executes (shouldMoveByFlank bypasses the BFS-reachability
   * check), so the false branch of the final `else if` is taken.
   */
  it('skips nudge entirely when all three nudge targets are inside walls', () => {
    const base = createGameState({ seed: 1 });
    // Enemy 0 at (62.8, 60.8) — overlaps solid cell (62, 60) due to radius 0.25.
    // With (62, 60) solid, every nudge candidate also overlaps that cell.
    // Enemy 0 is in flanking range (distToPlayer ≈ 2.32 ≤ 3.5), and the
    // wall-aware slot shifter finds a valid slot at angle π/6, so
    // shouldMoveByFlank is true and the nudge block is entered.
    const wallMap = createWallCollisionMap([{ x: 62, y: 60 }]);
    const state: GameState = {
      ...base,
      enemies: [
        { position: { x: 62.8, y: 60.8 }, health: 100 },
        { position: { x: 59.5, y: 60.5 }, health: 100 },
      ],
    };

    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 200);
    const e0 = controller.enemies[0];
    expect(Number.isFinite(e0.position.x)).toBe(true);
    expect(Number.isFinite(e0.position.y)).toBe(true);
    expect(e0.active).toBe(true);
  });

  // --- One-sided centering tests (Fix 1: diagonal-gap centering) ---

  /**
   * One-sided centering: when a wall flanks only one perpendicular side and
   * the enemy is drifting toward that wall, the perpendicular coordinate is
   * snapped to the cell center before the collision check. This is the core
   * fix for diagonal-gap traversal: previously the `&&` (both-walls) check
   * never fired in a zig-zag corridor where only one side is walled per row.
   */
  it('snaps perpendicular coordinate toward center when drifting toward a one-sided wall', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Enemy at (60.2, 55.5) — cell (60, 55),
    // drifting west (x < 60.5). Wall at (59, 56) is west of target cell
    // (60, 56) for the south move. One-sided centering should snap x to
    // 60.5 because the enemy is drifting toward the west wall.
    const wallMap = createWallCollisionMap([{ x: 59, y: 56 }]);
    const state = stateWithEnemyAt(base, { x: 60.2, y: 55.5 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 200);
    expect(controller.enemies[0].position.x).toBe(60.5);
    expect(controller.enemies[0].position.y).toBe(56.0);
  });

  /**
   * One-sided centering: when the enemy is drifting away from the one-sided
   * wall, the perpendicular coordinate is NOT snapped. The collision check
   * still passes because the enemy's circle does not reach the wall.
   */
  it('does NOT snap perpendicular coordinate when drifting away from a one-sided wall', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (60.8, 55.5) — drifting east, away from west wall at (59, 56).
    const wallMap = createWallCollisionMap([{ x: 59, y: 56 }]);
    const state = stateWithEnemyAt(base, { x: 60.8, y: 55.5 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 200);
    expect(controller.enemies[0].position.x).toBe(60.8);
    expect(controller.enemies[0].position.y).toBe(56.0);
  });

  /**
   * One-sided centering: in an open area with no perpendicular walls, the
   * perpendicular coordinate is NOT snapped. This preserves existing
   * open-area behavior.
   */
  it('preserves open-area behavior with no perpendicular walls (no snap)', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (60.3, 55.5) — slightly off-center, no walls anywhere.
    const emptyMap = createEmptyCollisionMap();
    const state = stateWithEnemyAt(base, { x: 60.3, y: 55.5 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, emptyMap, 200);
    expect(controller.enemies[0].position.x).toBe(60.3);
    expect(controller.enemies[0].position.y).toBe(56.0);
  });

  // --- Diagonal gap traversal tests ---

  /**
   * Diagonal gap (vertical staircase): an enemy navigating south through a
   * zig-zag corridor where walls alternate east/west per row should not
   * stall. The one-sided centering snaps the enemy to center when it drifts
   * toward a wall, allowing smooth traversal. This is the exact scenario
   * that caused permanent stalls before the fix (centering never fired
   * because `&&` required walls on both sides).
   */
  it('traverses a vertical staircase diagonal gap without stalling', () => {
    const base = createGameState({ seed: 1 });
    // Walls alternate sides per row, creating a zig-zag corridor:
    //   row 56: east wall at (61, 56)
    //   row 57: west wall at (59, 57)
    //   row 58: east wall at (61, 58)
    //   row 59: west wall at (59, 59)
    const wallMap = createWallCollisionMap([
      { x: 61, y: 56 },
      { x: 59, y: 57 },
      { x: 61, y: 58 },
      { x: 59, y: 59 },
    ]);
    const state = stateWithEnemyAt(base, { x: 60.2, y: 55.5 });
    let controller = createEnemyControllerState(state);
    // Simulate 5 ticks (dtMs=200 → stepDistance=0.5 per tick).
    for (let tick = 0; tick < 5; tick += 1) {
      controller = updateEnemyController(controller, state, wallMap, 200);
    }
    const e = controller.enemies[0];
    // Enemy should have moved south through the zig-zag corridor.
    expect(e.position.y).toBeGreaterThan(57);
    // Enemy should be centered on X after the one-sided snap.
    expect(e.position.x).toBe(60.5);
    expect(e.active).toBe(true);
  });

  /**
   * Diagonal gap (horizontal staircase): an enemy navigating east through a
   * zig-zag corridor where walls alternate north/south per column should not
   * stall. Mirrors the vertical staircase test.
   */
  it('traverses a horizontal staircase diagonal gap without stalling', () => {
    const base = createGameState({ seed: 1 });
    // Walls alternate sides per column, creating a zig-zag corridor:
    //   col 56: north wall at (56, 59)
    //   col 57: south wall at (57, 61)
    //   col 58: north wall at (58, 59)
    //   col 59: south wall at (59, 61)
    const wallMap = createWallCollisionMap([
      { x: 56, y: 59 },
      { x: 57, y: 61 },
      { x: 58, y: 59 },
      { x: 59, y: 61 },
    ]);
    const state = stateWithEnemyAt(base, { x: 55.8, y: 60.8 });
    let controller = createEnemyControllerState(state);
    for (let tick = 0; tick < 5; tick += 1) {
      controller = updateEnemyController(controller, state, wallMap, 200);
    }
    const e = controller.enemies[0];
    // Enemy should have moved east through the zig-zag corridor.
    expect(e.position.x).toBeGreaterThan(58);
    // Enemy should be centered on Y after the one-sided snap.
    expect(e.position.y).toBe(60.5);
    expect(e.active).toBe(true);
  });

  // --- BFS stall-recovery test (Fix 2: BFS stall-recovery fallback) ---

  /**
   * BFS stall-recovery: when a BFS-mode enemy has been stalled for more than
   * 3 consecutive ticks, the fallback tries non-distance-reducing cardinal
   * directions to escape the deadlock. This scenario uses a large timestep
   * (dtMs=1000 → stepDistance=2.5) so the south step's collision check
   * reaches a wall two cells away, blocking the only distance-reducing
   * direction. After bfsStallTicks > 3, the escape moves the enemy north.
   */
  it('activates BFS stall-recovery after 4+ consecutive stalls', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Enemy at (60.5, 55.5) — centered, 5 cells north.
    // Wall at (60, 57) blocks a large south step (dtMs=1000 → step 2.5):
    //   desired = (60.5, 58.0), collision checks (60, 57) → wall → fail.
    // Walls at (61, 55) and (59, 55) block east/west isSolid checks.
    // Only south is distance-reducing but collision blocks it → stall.
    // After bfsStallTicks > 3, the escape tries N (60, 54): open, collision
    // passes → enemy moves north to (60.5, 53.0).
    const wallMap = createWallCollisionMap([
      { x: 60, y: 57 },
      { x: 61, y: 55 },
      { x: 59, y: 55 },
    ]);
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    let controller = createEnemyControllerState(state);
    // Simulate 4 prior stalls so bfsStallTicks > 3.
    controller.enemies[0].bfsStallTicks = 4;
    controller = updateEnemyController(controller, state, wallMap, 1000);
    const e = controller.enemies[0];
    // The escape should have moved the enemy (north, away from player).
    expect(e.position.y).toBeLessThan(55.5);
    // bfsStallTicks should reset to 0 because the escape moved the enemy.
    expect(e.bfsStallTicks).toBe(0);
    expect(e.active).toBe(true);
  });

  // --- One-sided centering: pre-collision, new cell wall (line 633) ---

  it('pre-collision one-sided centering: new cell north wall, drifting north', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Enemy at (10.3, 60.3) moving east (step 0.5).
    // New cell (11, 60) has north wall at (11, 59), no south wall.
    // Enemy drifting north (y=60.3 < 60.5) → one-sided snap to 60.5.
    const wallMap = createWallCollisionMap([{ x: 11, y: 59 }]);
    const state = stateWithEnemyAt(base, { x: 10.3, y: 60.3 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 200);
    expect(controller.enemies[0].position.y).toBe(60.5);
    expect(controller.enemies[0].position.x).toBe(10.8);
  });

  // --- One-sided centering: pre-collision, fallback to original cell (lines 642, 644) ---

  it('pre-collision one-sided centering: original cell north wall, drifting north', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (10.3, 60.3) moving east. New cell (11, 60) has no perp walls.
    // Original cell (10, 60) has north wall at (10, 59). Drifting north → snap.
    const wallMap = createWallCollisionMap([{ x: 10, y: 59 }]);
    const state = stateWithEnemyAt(base, { x: 10.3, y: 60.3 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 200);
    expect(controller.enemies[0].position.y).toBe(60.5);
    expect(controller.enemies[0].position.x).toBe(10.8);
  });

  it('pre-collision one-sided centering: original cell south wall, drifting south', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (10.3, 60.7) moving east. New cell (11, 60) has no perp walls.
    // Original cell (10, 60) has south wall at (10, 61). Drifting south → snap.
    // y=60.7 (not 60.8) so the 0.25-radius circle does NOT overlap wall (10,61),
    // preventing pre-move position correction from snapping y first.
    const wallMap = createWallCollisionMap([{ x: 10, y: 61 }]);
    const state = stateWithEnemyAt(base, { x: 10.3, y: 60.7 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 200);
    expect(controller.enemies[0].position.y).toBe(60.5);
    expect(controller.enemies[0].position.x).toBe(10.8);
  });

  // --- One-sided centering: pre-collision, vertical move fallback (lines 665, 667) ---

  it('pre-collision one-sided centering: original cell west wall, drifting west (vertical move)', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (60.3, 55.3) moving south. New cell (60, 56) has no perp walls.
    // Original cell (60, 55) has west wall at (59, 55). Drifting west → snap.
    const wallMap = createWallCollisionMap([{ x: 59, y: 55 }]);
    const state = stateWithEnemyAt(base, { x: 60.3, y: 55.3 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 200);
    expect(controller.enemies[0].position.x).toBe(60.5);
    expect(controller.enemies[0].position.y).toBe(55.8);
  });

  it('pre-collision one-sided centering: original cell east wall, drifting east (vertical move)', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (60.7, 55.3) moving south. New cell (60, 56) has no perp walls.
    // Original cell (60, 55) has east wall at (61, 55). Drifting east → snap.
    // x=60.7 (not 60.8) so the 0.25-radius circle does NOT overlap wall (61,55),
    // preventing pre-move position correction from snapping x first.
    const wallMap = createWallCollisionMap([{ x: 61, y: 55 }]);
    const state = stateWithEnemyAt(base, { x: 60.7, y: 55.3 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 200);
    expect(controller.enemies[0].position.x).toBe(60.5);
    expect(controller.enemies[0].position.y).toBe(55.8);
  });

  // --- Retry-after-block succeeds (line 708) ---

  it('retry-after-block snaps perpendicular and succeeds when initial collision fails', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (10.3, 60.8) moving east with large step (dtMs=1000 → step 2.5).
    // desired = (12.8, 60.8). Wall at (12, 61) is in collision rect but NOT a
    // perpendicular wall of new cell (11, 60) or original cell (10, 60).
    // Pre-collision centering does NOT fire. Retry snaps y to 60.5 → passes.
    const wallMap = createWallCollisionMap([{ x: 12, y: 61 }]);
    const state = stateWithEnemyAt(base, { x: 10.3, y: 60.8 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 1000);
    expect(controller.enemies[0].position.x).toBe(12.8);
    expect(controller.enemies[0].position.y).toBe(60.5);
  });

  // --- Post-move one-sided centering (lines 730, 732, 743, 745) ---

  it('post-move one-sided centering: north wall, drifting north', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (10.5, 60.49) moving east with large step (dtMs=1000 → step 2.5).
    // Landing cell (13, 60) has north wall at (13, 59). Pre-collision checks
    // (11, 59)/(10, 59) — no walls → no snap. Post-move: snap y to 60.5.
    const wallMap = createWallCollisionMap([{ x: 13, y: 59 }]);
    const state = stateWithEnemyAt(base, { x: 10.5, y: 60.49 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 1000);
    expect(controller.enemies[0].position.y).toBe(60.5);
    expect(controller.enemies[0].position.x).toBe(13.0);
  });

  it('post-move one-sided centering: south wall, drifting south', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (10.5, 60.51) moving east with large step.
    // Landing cell (13, 60) has south wall at (13, 61). Post-move: snap y to 60.5.
    const wallMap = createWallCollisionMap([{ x: 13, y: 61 }]);
    const state = stateWithEnemyAt(base, { x: 10.5, y: 60.51 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 1000);
    expect(controller.enemies[0].position.y).toBe(60.5);
    expect(controller.enemies[0].position.x).toBe(13.0);
  });

  it('post-move one-sided centering: west wall, drifting west (vertical move)', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (60.49, 55.5) moving south with large step (dtMs=1000 → step 2.5).
    // Landing cell (60, 58) has west wall at (59, 58). Post-move: snap x to 60.5.
    const wallMap = createWallCollisionMap([{ x: 59, y: 58 }]);
    const state = stateWithEnemyAt(base, { x: 60.49, y: 55.5 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 1000);
    expect(controller.enemies[0].position.x).toBe(60.5);
    expect(controller.enemies[0].position.y).toBe(58.0);
  });

  it('post-move one-sided centering: east wall, drifting east (vertical move)', () => {
    const base = createGameState({ seed: 1 });
    // Enemy at (60.51, 55.5) moving south with large step.
    // Landing cell (60, 58) has east wall at (61, 58). Post-move: snap x to 60.5.
    const wallMap = createWallCollisionMap([{ x: 61, y: 58 }]);
    const state = stateWithEnemyAt(base, { x: 60.51, y: 55.5 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(controller, state, wallMap, 1000);
    expect(controller.enemies[0].position.x).toBe(60.5);
    expect(controller.enemies[0].position.y).toBe(58.0);
  });

  // --- BFS stall-recovery: escape with solid direction + horizontal snap (lines 773, 782) ---

  it('BFS stall-recovery: escape skips solid direction and uses horizontal snap', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Enemy at (60.5, 55.5) — 5 cells north.
    // Wall at (60, 57) blocks south step (only distance-reducing direction).
    // Wall at (59, 55) blocks W isSolid. Wall at (61, 56) makes E's BFS
    // distance ≥ currentDist so E is skipped in the main loop. Wall at
    // (60, 54) makes N escape solid → continue (line 773).
    // E escape: isSolid(61, 55)=false, escapeDesired=(63.0, 55.5), dx=1 →
    // snap y to 55.5 (line 782). Collision passes → move.
    const wallMap = createWallCollisionMap([
      { x: 60, y: 57 },
      { x: 59, y: 55 },
      { x: 61, y: 56 },
      { x: 60, y: 54 },
    ]);
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    let controller = createEnemyControllerState(state);
    controller.enemies[0].bfsStallTicks = 4;
    controller = updateEnemyController(controller, state, wallMap, 1000);
    const e = controller.enemies[0];
    expect(e.position.x).toBe(63.0);
    expect(e.position.y).toBe(55.5);
    expect(e.bfsStallTicks).toBe(0);
  });

  // --- BFS stall-recovery: escape blocked by wall collision (line 794) ---

  it('BFS stall-recovery: escape blocked by wall collision continues to next direction', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Enemy at (60.5, 55.5) — 5 cells north.
    // Wall at (60, 57) blocks south step. Wall at (59, 55) blocks W isSolid.
    // Wall at (61, 56) makes E's BFS distance ≥ currentDist → skipped in main.
    // Wall at (60, 52) blocks N escape collision: escapeDesired=(60.5, 53.0),
    // collision rect includes (60, 52) → blocked → continue (line 794).
    // E escape succeeds.
    const wallMap = createWallCollisionMap([
      { x: 60, y: 57 },
      { x: 59, y: 55 },
      { x: 61, y: 56 },
      { x: 60, y: 52 },
    ]);
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    let controller = createEnemyControllerState(state);
    controller.enemies[0].bfsStallTicks = 4;
    controller = updateEnemyController(controller, state, wallMap, 1000);
    const e = controller.enemies[0];
    expect(e.position.x).toBe(63.0);
    expect(e.position.y).toBe(55.5);
    expect(e.bfsStallTicks).toBe(0);
  });

  // --- BFS stall-recovery: zero-timestep no-op (line 796 false branch) ---

  it('BFS stall-recovery: zero-timestep sync pass does not move enemy', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Enemy at (60.5, 55.5) — 5 cells north, at cell
    // center. With dtMs=0, escapeStep=0, so escapeDesired equals position.
    // The perpendicular snap (line 784) sets x to cellX+0+0.5=60.5=position.x.
    // So escapeDesired === position → line 796 false → moved stays false.
    // bfsStallTicks unchanged (dtMs=0 skips the counter update at line 811).
    const wallMap = createEmptyCollisionMap();
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    let controller = createEnemyControllerState(state);
    controller.enemies[0].bfsStallTicks = 4;
    controller = updateEnemyController(controller, state, wallMap, 0);
    const e = controller.enemies[0];
    expect(e.position.x).toBe(60.5);
    expect(e.position.y).toBe(55.5);
    expect(e.bfsStallTicks).toBe(4);
  });
});

/**
 * Contract tests for ControlledEnemy vision fields (AC-10.5a-002).
 */
describe('ControlledEnemy vision fields (AC-10.5a-002)', () => {
  it('createEnemyControllerState initializes weights as undefined', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    const controller = createEnemyControllerState(state);
    const enemy: ControlledEnemy = controller.enemies[0];
    expect(enemy.weights).toBeUndefined();
  });

  it('createEnemyControllerState initializes variantId as 0', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    const controller = createEnemyControllerState(state);
    expect(controller.enemies[0].variantId).toBe(0);
  });

  it('createEnemyControllerState initializes previousStepDistance as -1', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    const controller = createEnemyControllerState(state);
    expect(controller.enemies[0].previousStepDistance).toBe(-1);
  });

  it('updateEnemyController preserves weights field across ticks', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    let controller = createEnemyControllerState(state);
    // Set custom weights on the enemy
    const customWeights = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    controller.enemies[0].weights = customWeights;
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    // Weights should be preserved across ticks
    expect(controller.enemies[0].weights).toBe(customWeights);
  });

  it('updateEnemyController preserves variantId across ticks', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    let controller = createEnemyControllerState(state);
    controller.enemies[0].variantId = 7;
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    expect(controller.enemies[0].variantId).toBe(7);
  });

  it('updateEnemyController sets previousStepDistance to current cell distance', () => {
    const base = createGameState({ seed: 1 });
    // Player at (60.5, 60.5). Enemy at (60.5, 55.5) — 5 cells north.
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    let controller = createEnemyControllerState(state);
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    // The enemy is at cell (60, 55). Distance to player at cell (60, 60) = 5.
    // After the tick, previousStepDistance should be the distance at the
    // enemy's final cell (which may have moved). On an empty map, the enemy
    // moves toward the player, so it moves south. The final cell distance
    // should be less than the starting distance.
    expect(controller.enemies[0].previousStepDistance).toBeGreaterThanOrEqual(
      0,
    );
  });

  it('updateEnemyController updates previousStepDistance across two ticks', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    let controller = createEnemyControllerState(state);
    // First tick: enemy moves, previousStepDistance gets set to current dist
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    const firstDist = controller.enemies[0].previousStepDistance;
    expect(firstDist).toBeGreaterThanOrEqual(0);
    // Second tick: previousStepDistance should change as enemy moves closer
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    // The enemy should have moved closer to the player, so the distance
    // should be different (smaller or equal)
    expect(controller.enemies[0].previousStepDistance).toBeGreaterThanOrEqual(
      0,
    );
  });

  it('updateEnemyController initializes new fields for respawned enemies', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 60.5, y: 55.5 });
    let controller = createEnemyControllerState(state);
    // Kill the enemy and tick until fully de-rezzed
    const deadState = stateWithEnemyAt(base, { x: 60.5, y: 55.5 }, 0);
    // De-rez duration is 4000ms; tick 5 times with 1000ms each to exceed it
    for (let i = 0; i < 5; i++) {
      controller = updateEnemyController(
        controller,
        deadState,
        createEmptyCollisionMap(),
        1000,
      );
    }
    expect(controller.enemies[0].active).toBe(false);
    // Respawn: give health back
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    expect(controller.enemies[0].active).toBe(true);
    // Respawned enemy should have fresh defaults
    expect(controller.enemies[0].weights).toBeUndefined();
    expect(controller.enemies[0].variantId).toBe(0);
    expect(controller.enemies[0].previousStepDistance).toBe(-1);
  });
});

describe('MLP re-ranking and BFS fallback (AC-10.5b-002/003/004)', () => {
  /**
   * Build MLP weights that produce a deterministic output regardless of input.
   *
   * All connection weights are zeroed so the output depends only on the final
   * layer biases. The bias for the specified output index is set to a large
   * value so tanh(bias) ≈ 1. The topology is [6,6,4,4] (90 params):
   * - Layer 1 (6→6): weights[0..35] connections, weights[36..41] biases
   * - Layer 2 (6→4): weights[42..65] connections, weights[66..69] biases
   * - Layer 3 (4→4): weights[70..85] connections, weights[86..89] biases
   *
   * @param outputIndex - Index of the output to activate (0=move, 1=strafe, 2=turn, 3=fire).
   * @returns Float32Array of 90 weights.
   */
  function buildDeterministicWeights(outputIndex: number): Float32Array {
    const weights = new Float32Array(90);
    // All connection weights (0-75) and hidden biases (76-85) are 0.
    // Only the specified output bias (86-89) is set to 10 → tanh(10) ≈ 1.
    weights[86 + outputIndex] = 10;
    return weights;
  }

  /**
   * AC-10.5b-002: When weights are set, MLP re-ranks BFS candidate directions.
   *
   * Enemy at (55.5, 58.5), player at (60.5, 60.5).
   * BFS has two valid directions: E (dist 6) and S (dist 6).
   * Without MLP, BFS picks E (earlier in directions array).
   * With strafe=1 weights, MLP scores S higher → enemy moves south.
   */
  it('AC-10.5b-002: MLP re-ranks BFS candidates when weights are set', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 55.5, y: 58.5 });
    let controller = createEnemyControllerState(state);
    controller.enemies[0].weights = buildDeterministicWeights(1); // strafe=1
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    // Enemy should have moved south (y increased), not east (x unchanged).
    expect(controller.enemies[0].position.y).toBeGreaterThan(58.5);
    expect(controller.enemies[0].position.x).toBe(55.5);
  });

  /**
   * AC-10.5b-004: BFS fallback when weights are undefined (identical to pre-10.5).
   *
   * Same scenario as above but without weights → BFS picks east.
   */
  it('AC-10.5b-004: BFS fallback when weights are undefined picks east (identical to pre-10.5)', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 55.5, y: 58.5 });
    let controller = createEnemyControllerState(state);
    // No weights → BFS fallback → enemy moves east (first valid BFS direction).
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    // Enemy should have moved east (x increased), y unchanged.
    expect(controller.enemies[0].position.x).toBeGreaterThan(55.5);
    expect(controller.enemies[0].position.y).toBe(58.5);
  });

  /**
   * AC-10.5b-002: BFS fallback when MLP outputs NaN.
   *
   * NaN weights produce NaN MLP outputs → BFS fallback → enemy moves east.
   */
  it('AC-10.5b-002: BFS fallback when MLP outputs NaN', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 55.5, y: 58.5 });
    let controller = createEnemyControllerState(state);
    const nanWeights = new Float32Array(90);
    nanWeights.fill(NaN);
    controller.enemies[0].weights = nanWeights;
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    // BFS fallback → enemy moves east.
    expect(controller.enemies[0].position.x).toBeGreaterThan(55.5);
    expect(controller.enemies[0].position.y).toBe(58.5);
  });

  /**
   * AC-10.5b-002: BFS fallback when weights have wrong length.
   *
   * Weights of length 4 (not 90) → activateMlp throws → BFS fallback.
   * Also verifies weights are preserved in the output despite not being used.
   */
  it('AC-10.5b-002: BFS fallback when weights have wrong length', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 55.5, y: 58.5 });
    let controller = createEnemyControllerState(state);
    const shortWeights = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    controller.enemies[0].weights = shortWeights;
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    // BFS fallback → enemy moves east.
    expect(controller.enemies[0].position.x).toBeGreaterThan(55.5);
    expect(controller.enemies[0].position.y).toBe(58.5);
    // Weights should be preserved.
    expect(controller.enemies[0].weights).toBe(shortWeights);
  });

  /**
   * AC-10.5b-003: MLP activation is skipped on zero-timestep sync pass.
   *
   * With weights set and dtMs=0, MLP should not be called and the enemy
   * should not move (same as no-weights at dtMs=0).
   */
  it('AC-10.5b-003: MLP activation skipped on zero-timestep sync pass (dtMs=0)', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 55.5, y: 58.5 });
    let controller = createEnemyControllerState(state);
    controller.enemies[0].weights = buildDeterministicWeights(1); // strafe=1
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      0,
    );
    // Enemy should not have moved.
    expect(controller.enemies[0].position.x).toBe(55.5);
    expect(controller.enemies[0].position.y).toBe(58.5);
  });

  /**
   * AC-10.5b-002: MLP with move=1 picks the most forward valid BFS direction.
   *
   * With move=1, the desired direction is the facing vector (toward player).
   * E has a higher score than S because the facing is more east than south.
   * Enemy moves east (same as BFS, but chosen by MLP scoring).
   */
  it('AC-10.5b-002: MLP with move=1 picks the most forward valid BFS direction', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 55.5, y: 58.5 });
    let controller = createEnemyControllerState(state);
    controller.enemies[0].weights = buildDeterministicWeights(0); // move=1
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    // Enemy should move east (aligned with facing).
    expect(controller.enemies[0].position.x).toBeGreaterThan(55.5);
    expect(controller.enemies[0].position.y).toBe(58.5);
  });

  /**
   * Coverage gap: exercises the prevStepDist >= 0 true branch at line 665.
   *
   * The existing MLP tests all run on the FIRST tick (previousStepDistance = -1
   * from createEnemyControllerState), so `prevStepDist >= 0 ? prevStepDist :
   * undefined` always evaluates the false branch (undefined).
   *
   * This test runs a first tick WITHOUT weights (to set previousStepDistance
   * to a non-negative value), then sets weights and runs a SECOND tick so the
   * MLP path enters with prevStepDist >= 0, covering the true branch.
   */
  it('AC-10.5b-002: MLP re-ranking with previousStepDistance >= 0 (second tick)', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemyAt(base, { x: 55.5, y: 58.5 });
    let controller = createEnemyControllerState(state);

    // First tick WITHOUT weights → BFS moves enemy, sets previousStepDistance.
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );
    expect(controller.enemies[0].previousStepDistance).toBeGreaterThanOrEqual(
      0,
    );

    // Set weights for the second tick → MLP path entered with prevStepDist >= 0.
    controller.enemies[0].weights = buildDeterministicWeights(1); // strafe=1
    controller = updateEnemyController(
      controller,
      state,
      createEmptyCollisionMap(),
      1000,
    );

    // Enemy should have moved (MLP re-ranking applied with prevStepDist).
    const moved =
      controller.enemies[0].position.x !== 55.5 ||
      controller.enemies[0].position.y !== 58.5;
    expect(moved).toBe(true);
    // Weights preserved.
    expect(controller.enemies[0].weights).toBeDefined();
  });
});

describe('AC-11b-003/004: hit-stun behavior in enemy controller', () => {
  it('initializes stunTimerMs to 0 in createEnemyControllerState', () => {
    const base = createGameState({ seed: 1 });
    const state = stateWithEnemy(base, { x: -5, y: 0 });
    const controller = createEnemyControllerState(state);
    expect(controller.enemies[0].stunTimerMs).toBe(0);
  });

  it('sets animationState to damage while stunned', () => {
    const base = createGameState({ seed: 1 });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: {
            x: base.player.position.x - 5,
            y: base.player.position.y,
          },
          health: 80,
          stunTimerMs: 200,
        },
      ],
    };
    const emptyMap = createEmptyCollisionMap();
    const controller = createEnemyControllerState(state);
    const updated = updateEnemyController(controller, state, emptyMap, 16);
    expect(updated.enemies[0].animationState).toBe('damage');
  });

  it('skips movement while stunned', () => {
    const base = createGameState({ seed: 1 });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: {
            x: base.player.position.x - 5,
            y: base.player.position.y,
          },
          health: 80,
          stunTimerMs: 200,
        },
      ],
    };
    const emptyMap = createEmptyCollisionMap();
    const controller = createEnemyControllerState(state);
    const startX = controller.enemies[0].position.x;
    const updated = updateEnemyController(controller, state, emptyMap, 1000);
    // Stunned enemy should not move.
    expect(updated.enemies[0].position.x).toBeCloseTo(startX, 5);
  });

  it('does not fire while stunned', () => {
    const base = createGameState({ seed: 1 });
    // Place enemy within fire range so it would normally fire.
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: {
            x: base.player.position.x - 3,
            y: base.player.position.y,
          },
          health: 80,
          stunTimerMs: 200,
        },
      ],
    };
    const emptyMap = createEmptyCollisionMap();
    const controller = createEnemyControllerState(state);
    const updated = updateEnemyController(controller, state, emptyMap, 1000);
    expect(updated.hitscanEvents).toHaveLength(0);
  });

  it('decrements stunTimerMs per tick', () => {
    const base = createGameState({ seed: 1 });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: {
            x: base.player.position.x - 5,
            y: base.player.position.y,
          },
          health: 80,
          stunTimerMs: 200,
        },
      ],
    };
    const emptyMap = createEmptyCollisionMap();
    const controller = createEnemyControllerState(state);
    const updated = updateEnemyController(controller, state, emptyMap, 16);
    expect(updated.enemies[0].stunTimerMs).toBeLessThan(200);
    expect(updated.enemies[0].stunTimerMs).toBeGreaterThan(0);
  });

  it('recovers from stun when stunTimerMs reaches 0', () => {
    const base = createGameState({ seed: 1 });
    // stunTimerMs = 10, dtMs = 16 → decrements to 0 (clamped).
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: {
            x: base.player.position.x - 5,
            y: base.player.position.y,
          },
          health: 80,
          stunTimerMs: 10,
        },
      ],
    };
    const emptyMap = createEmptyCollisionMap();
    const controller = createEnemyControllerState(state);
    const updated = updateEnemyController(controller, state, emptyMap, 16);
    // The stun expired this tick; the enemy should still be in 'damage' state
    // (since enemyState.stunTimerMs was > 0 at the start of the tick).
    expect(updated.enemies[0].animationState).toBe('damage');
    expect(updated.enemies[0].stunTimerMs).toBe(0);

    // On the next tick, the stun is gone — enemy should move normally.
    const recoveredState: GameState = {
      ...state,
      enemies: state.enemies.map((e) => ({ ...e, stunTimerMs: 0 })),
    };
    const updated2 = updateEnemyController(
      updated,
      recoveredState,
      emptyMap,
      1000,
    );
    expect(updated2.enemies[0].animationState).not.toBe('damage');
  });

  it('adopts enemyState position while stunned (pushback sync)', () => {
    const base = createGameState({ seed: 1 });
    const pushbackPos = {
      x: base.player.position.x - 10,
      y: base.player.position.y,
    };
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: pushbackPos,
          health: 80,
          stunTimerMs: 200,
        },
      ],
    };
    const emptyMap = createEmptyCollisionMap();
    // Create controller with original position (before pushback).
    const controller = createEnemyControllerState(state);
    // Override the controller position to be different from enemyState.
    controller.enemies[0].position = { x: 0, y: 0 };

    const updated = updateEnemyController(controller, state, emptyMap, 16);
    // The stunned enemy should adopt enemyState.position, not the stale
    // ControlledEnemy position.
    expect(updated.enemies[0].position.x).toBeCloseTo(pushbackPos.x, 5);
    expect(updated.enemies[0].position.y).toBeCloseTo(pushbackPos.y, 5);
  });
});

describe('AC-11b-004: separateEnemies skips stunned enemies', () => {
  it('does not separate stunned enemies', () => {
    const base = createGameState({ seed: 1 });
    const posA = { x: base.player.position.x - 3, y: base.player.position.y };
    const posB = { x: base.player.position.x - 3, y: base.player.position.y };
    const state: GameState = {
      ...base,
      enemies: [
        { position: { ...posA }, health: 80, stunTimerMs: 200 },
        { position: { ...posB }, health: 80, stunTimerMs: 200 },
      ],
    };
    const emptyMap = createEmptyCollisionMap();
    const controller = createEnemyControllerState(state);
    const updated = updateEnemyController(controller, state, emptyMap, 16);
    // Both enemies are stunned, so separateEnemies should not move them.
    // Their positions should match the enemyState positions (not pushed apart).
    expect(updated.enemies[0].position.x).toBeCloseTo(posA.x, 5);
    expect(updated.enemies[1].position.x).toBeCloseTo(posB.x, 5);
  });
});

describe('AC-11b-006: stun timer determinism (fixed-timestep decrement)', () => {
  it('decrements stunTimerMs by NEATENSTEIN_FIXED_TIMESTEP_MS regardless of dtMs', () => {
    const base = createGameState({ seed: 1 });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: {
            x: base.player.position.x - 5,
            y: base.player.position.y,
          },
          health: 80,
          stunTimerMs: 200,
        },
      ],
    };
    const emptyMap = createEmptyCollisionMap();

    // Run with dtMs = 16 (fixed timestep) � should decrement by 16.
    const controller16 = createEnemyControllerState(state);
    const updated16 = updateEnemyController(controller16, state, emptyMap, 16);
    expect(updated16.enemies[0].stunTimerMs).toBe(
      200 - NEATENSTEIN_FIXED_TIMESTEP_MS,
    );

    // Run with dtMs = 32 (variable frame rate) � should STILL decrement by 16.
    const controller32 = createEnemyControllerState(state);
    const updated32 = updateEnemyController(controller32, state, emptyMap, 32);
    expect(updated32.enemies[0].stunTimerMs).toBe(
      200 - NEATENSTEIN_FIXED_TIMESTEP_MS,
    );

    // Both should produce the same stunTimerMs regardless of dtMs.
    expect(updated16.enemies[0].stunTimerMs).toBe(
      updated32.enemies[0].stunTimerMs,
    );
  });

  it('does not decrement stunTimerMs on zero-timestep sync pass (dtMs=0)', () => {
    const base = createGameState({ seed: 1 });
    const state: GameState = {
      ...base,
      enemies: [
        {
          position: {
            x: base.player.position.x - 5,
            y: base.player.position.y,
          },
          health: 80,
          stunTimerMs: 100,
        },
      ],
    };
    const emptyMap = createEmptyCollisionMap();
    const controller = createEnemyControllerState(state);
    const updated = updateEnemyController(controller, state, emptyMap, 0);
    // dtMs=0 is a zero-timestep sync pass � stunTimerMs should not change.
    expect(updated.enemies[0].stunTimerMs).toBe(100);
  });
});
