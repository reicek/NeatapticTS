/**
 * Enemy AI controller for the Neatenstein NGE demo.
 *
 * Produces deterministic per-enemy animation/AI descriptors from a
 * {@link GameState} snapshot. Each controlled enemy navigates toward the
 * player using a BFS distance map (recycled from asciiMaze patterns),
 * moves with wall-block-then-stay behavior, fires hitscan shots when the
 * player is in range and line-of-sight, and plays a 700 ms de-rez death
 * animation when its health reaches zero or its hitscan ammunition is depleted.
 *
 * @module
 */

import type { CollisionMap } from '../browser-entry/renderer/map';
import { NEATENSTEIN_MAP_SIZE } from '../browser-entry/constants';
import type { EnemyState, GameState } from '../browser-entry/host/game/types';
import {
  buildEnemyDistanceMap,
  type DistanceMap,
  getDistance,
} from './enemy-navigation';
import { separateEnemies } from './enemy-controller.collision.utils';
import {
  isFiniteNumber,
  resolveTimestepMs,
} from './enemy-controller.state.utils';
import { resolveRespawnState } from './enemy-controller.spawn.utils';
import { handleDeath } from './enemy-controller.death.utils';
import { handleStun } from './enemy-controller.stun.utils';
import { resolveFlankState } from './enemy-controller.flank.utils';
import { computeMovement } from './enemy-controller.move.utils';
import { resolveFire } from './enemy-controller.fire.utils';

// Re-export extracted types and constants so existing imports stay valid.
export type {
  ControlledEnemy,
  HitscanEvent,
  EnemyControllerState,
  EnemyUpdateContext,
} from './enemy-controller.types';
export {
  PREVIOUS_STEP_DISTANCE_SENTINEL,
  CELL_CENTER_OFFSET,
  DIRECTIONS,
  NUM_DIRECTIONS,
  DIR_N,
  DIR_NW,
  DIR_W,
  DIR_SW,
  DIR_S,
  DIR_SE,
  DIR_E,
  DIR_NE,
  ENEMY_CONTROLLER_SHOOT_BLINK_TICKS,
  ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND,
  ENEMY_CONTROLLER_RADIUS_CELLS,
  ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
  ENEMY_CONTROLLER_FIRE_RANGE_CELLS,
  ENEMY_CONTROLLER_STOP_DISTANCE_CELLS,
  ENEMY_CONTROLLER_FLANKING_RADIUS_CELLS,
  ENEMY_CONTROLLER_FIRE_COOLDOWN_MS,
  ENEMY_CONTROLLER_HITSCAN_DAMAGE,
  ENEMY_CONTROLLER_STARTING_AMMO,
  ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
} from './enemy-controller.constants';

// Re-export public symbols that moved to sibling util files.
export { createEnemyControllerState } from './enemy-controller.state.utils';

import type {
  ControlledEnemy,
  EnemyControllerState,
  EnemyUpdateContext,
  HitscanEvent,
} from './enemy-controller.types';
import { PREVIOUS_STEP_DISTANCE_SENTINEL } from './enemy-controller.constants';

/**
 * Update one controlled enemy for a single tick via a declarative pipeline.
 *
 * Each branch (respawn, death, stun, movement, fire) is delegated to an
 * imported executor that operates on a shared {@link EnemyUpdateContext}.
 * The orchestrator threads the context through the steps and assembles the
 * final {@link ControlledEnemy}.
 *
 * @param index - Index in the source enemy array.
 * @param enemyState - Source enemy snapshot.
 * @param previous - Previous controlled state, if any.
 * @param gameState - Current game snapshot.
 * @param collisionMap - Map queried for solid cells.
 * @param distanceMap - BFS distance map from the player position.
 * @param dtMs - Tick duration in milliseconds.
 * @param hitscanEvents - Array to append any fire event to.
 * @param injectedWeights - Champion MLP weights from the current population
 *   snapshot. When provided, overrides the per-enemy weights for this tick,
 *   including respawns (which previously forced `undefined`). When omitted,
 *   the previous-or-default weights are preserved (backward-compatible).
 * @returns New controlled enemy descriptor.
 */
function updateControlledEnemy(
  index: number,
  enemyState: EnemyState,
  previous: ControlledEnemy | undefined,
  gameState: GameState,
  collisionMap: CollisionMap,
  distanceMap: DistanceMap,
  dtMs: number,
  hitscanEvents: HitscanEvent[],
  injectedWeights?: Float32Array | undefined,
): ControlledEnemy {
  const ctx: EnemyUpdateContext = {
    index,
    enemyState,
    previous,
    gameState,
    collisionMap,
    distanceMap,
    dtMs,
    hitscanEvents,
    injectedWeights,
    isRespawn: false,
    previousOrDefault: previous as ControlledEnemy,
    ammo: 0,
    fireCooldownMs: 0,
    deRezElapsedMs: 0,
    walkTick: 0,
    shootBlinkTicks: 0,
    flankStallTicks: 0,
    bfsStallTicks: 0,
    weights: undefined,
    variantId: 0,
    position: { x: 0, y: 0 },
    yawRad: 0,
    stunTimerMs: 0,
    distToPlayer: 0,
    moved: false,
    animationState: 'idle',
    shouldMoveByBfs: false,
    shouldMoveByFlank: false,
    slotTarget: { x: 0, y: 0 },
    isFiring: false,
  };

  // Step 1: Resolve respawn and initialize mutable state.
  resolveRespawnState(ctx);

  // Step 2: Handle death / de-rez (early return).
  const deathResult = handleDeath(ctx);
  if (deathResult) return deathResult;

  // Step 3: Handle hit-stun (early return).
  const stunResult = handleStun(ctx);
  if (stunResult) return stunResult;

  // Step 4: Resolve yaw toward player.
  const dxToPlayer = gameState.player.position.x - ctx.position.x;
  const dyToPlayer = gameState.player.position.y - ctx.position.y;
  ctx.distToPlayer = Math.hypot(dxToPlayer, dyToPlayer);
  if (isFiniteNumber(ctx.distToPlayer) && ctx.distToPlayer > 0) {
    ctx.yawRad = Math.atan2(dyToPlayer, dxToPlayer);
  }

  // Step 5: Resolve flanking slot and movement mode.
  resolveFlankState(ctx);

  // Step 6: Compute movement (BFS / flanking / MLP / stall-recovery).
  computeMovement(ctx);

  // Step 7: Update walk tick and stall counters.
  if (dtMs > 0) {
    ctx.walkTick = ctx.moved ? ctx.walkTick + 1 : 0;
    if (ctx.shouldMoveByFlank && !ctx.moved) {
      ctx.flankStallTicks += 1;
    } else {
      ctx.flankStallTicks = 0;
    }
    if (ctx.shouldMoveByBfs && !ctx.shouldMoveByFlank && !ctx.moved) {
      ctx.bfsStallTicks += 1;
    } else {
      ctx.bfsStallTicks = 0;
    }
  }

  // Step 8: Resolve fire decision, shoot blink, and animation state.
  resolveFire(ctx);

  // Step 9: Compute final BFS distance for progress tracking.
  const finalCellX = Math.floor(ctx.position.x);
  const finalCellY = Math.floor(ctx.position.y);
  const finalDist = ctx.isRespawn
    ? PREVIOUS_STEP_DISTANCE_SENTINEL
    : getDistance(distanceMap, finalCellX, finalCellY);

  // Step 10: Assemble and return.
  return {
    index,
    position: ctx.position,
    health: enemyState.health,
    yawRad: ctx.yawRad,
    animationState: ctx.animationState,
    ammo: ctx.ammo,
    fireCooldownMs: ctx.fireCooldownMs,
    deRezElapsedMs: ctx.deRezElapsedMs,
    active: true,
    walkTick: ctx.walkTick,
    shootBlinkTicks: ctx.shootBlinkTicks,
    flankStallTicks: ctx.flankStallTicks,
    bfsStallTicks: ctx.bfsStallTicks,
    weights: ctx.weights,
    variantId: ctx.variantId,
    previousStepDistance:
      finalDist >= 0 ? finalDist : PREVIOUS_STEP_DISTANCE_SENTINEL,
    stunTimerMs: 0,
  };
}

/**
 * Advance the enemy AI controller by one simulation tick, producing new enemy
 * descriptors and any hitscan fire events produced this tick.
 *
 * The controller matches enemies by array index so it can preserve per-enemy
 * ammunition, fire cooldowns, and de-rez timing across calls. If an enemy is
 * replaced at the same index while the previous one was fully de-rezzed, the
 * new enemy receives fresh controller state.
 *
 * When `weights` is provided, the champion MLP weights are injected into every
 * enemy (including respawns), activating the MLP re-ranking branch. When
 * omitted, the previous-or-default weights are preserved (backward-compatible).
 *
 * @param controller - Previous controller state.
 * @param state - Current game state.
 * @param collisionMap - Map queried for solid cells.
 * @param dtMs - Tick duration in milliseconds.
 * @param weights - Optional champion MLP weights from the current population
 *   snapshot. Activates MLP re-ranking when provided.
 * @returns New controller state with updated enemy descriptors and hitscan
 *   events.
 */
export function updateEnemyController(
  controller: EnemyControllerState,
  state: GameState,
  collisionMap: CollisionMap,
  dtMs: number,
  weights?: Float32Array | undefined,
): EnemyControllerState {
  const resolvedDtMs = resolveTimestepMs(dtMs);
  const previousByIndex = new Map(
    controller.enemies.map((enemy) => [enemy.index, enemy]),
  );

  // Build BFS distance map from the player cell once per tick.
  const playerCellX = Math.floor(state.player.position.x);
  const playerCellY = Math.floor(state.player.position.y);
  const distanceMap = buildEnemyDistanceMap(
    collisionMap,
    playerCellX,
    playerCellY,
    NEATENSTEIN_MAP_SIZE,
  );

  const enemies: ControlledEnemy[] = [];
  const hitscanEvents: HitscanEvent[] = [];

  for (let i = 0; i < state.enemies.length; i += 1) {
    const previous = previousByIndex.get(i);
    enemies.push(
      updateControlledEnemy(
        i,
        state.enemies[i],
        previous,
        state,
        collisionMap,
        distanceMap,
        resolvedDtMs,
        hitscanEvents,
        weights,
      ),
    );
  }

  separateEnemies(enemies, collisionMap);

  return {
    enemies,
    hitscanEvents,
  };
}
