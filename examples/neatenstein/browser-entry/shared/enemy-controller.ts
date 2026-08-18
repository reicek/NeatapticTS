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

import type { CollisionMap } from '../renderer/map';
import type { EnemyState, GameState } from '../host/game/types';
import {
  getOrBuildCachedDistanceMap,
  type DistanceMap,
  getDistance,
} from './enemy-navigation';
import { separateEnemies } from './enemy-controller.collision.utils';
import { isFiniteNumber } from './math-guards.utils';
import {
  resolveTimestepMs,
} from './enemy-controller.state.utils';
import { resolveRespawnState } from './enemy-controller.spawn.utils';
import { handleDeath } from './enemy-controller.death.utils';
import { handleStun } from './enemy-controller.stun.utils';
import { resolveFlankState } from './enemy-controller.flank.utils';
import { computeMovementFlat, resetMlpDebugGuard } from './enemy-controller.move.utils';
import { resolveFire } from './enemy-controller.fire.utils';

// Re-export parallel inference infrastructure (B1).
export {
  ENEMY_INFERENCE_POOL_SIZE,
  createEnemyInferencePool,
  loadEnemyWeightSlots,
  resolveInferenceStrategy,
  dispatchParallelInference,
  awaitInferenceBarrier,
} from './enemy-controller.parallel.utils';
export type {
  InferenceStrategy,
  InferenceResult,
  InferenceBarrierState,
  EnemyInferencePool,
} from './enemy-controller.parallel.utils';

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
export { computeMovementFlat } from './enemy-controller.move.utils';

// --- Pooled context slots and reusable Map (A2 Fix 8) ---

/** Number of enemy context slots to preallocate (B1: raised for 16-32 enemies). */
const POOLED_CONTEXT_SLOT_COUNT = 16;

/**
 * Writable variant of {@link EnemyUpdateContext} with all `readonly`
 * modifiers stripped. Used internally by the pool to reset slots between
 * ticks; callers always receive the readonly `EnemyUpdateContext` view.
 */
type WritableEnemyUpdateContext = {
  -readonly [K in keyof EnemyUpdateContext]: EnemyUpdateContext[K];
};

/** Preallocated EnemyUpdateContext slots — reused across ticks. */
const pooledContextSlots: WritableEnemyUpdateContext[] = Array.from(
  { length: POOLED_CONTEXT_SLOT_COUNT },
  () => ({
    index: 0,
    enemyState: undefined as unknown as EnemyState,
    previous: undefined,
    gameState: undefined as unknown as GameState,
    collisionMap: undefined as unknown as CollisionMap,
    distanceMap: undefined as unknown as DistanceMap,
    dtMs: 0,
    hitscanEvents: [],
    injectedWeights: undefined,
    isRespawn: false,
    previousOrDefault: undefined as unknown as ControlledEnemy,
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
  }),
);

/** Counter for reusable Map/array allocations (test-only diagnostic). */
let reusableMapAllocCount = 0;

/**
 * Reusable plain array indexed by enemy index, replacing the per-tick
 * `previousByIndex` allocation (A2 Fix 8). Allocated once at module load;
 * `.length = 0` resets it each tick.
 */
const reusablePreviousByIndex: (ControlledEnemy | undefined)[] = [];
reusableMapAllocCount += 1; // Track the reusable array allocation.

/**
 * Return the pooled EnemyUpdateContext slots (test-only).
 *
 * @returns The array of 8 preallocated context slots.
 */
export function __testOnlyGetPooledContextSlots(): EnemyUpdateContext[] {
  return pooledContextSlots;
}

/**
 * Return the number of Map allocations that have occurred (test-only).
 *
 * @returns The cumulative count of Map allocations.
 */
export function __testOnlyGetReusableMapAllocCount(): number {
  return reusableMapAllocCount;
}

import type {
  ControlledEnemy,
  EnemyControllerState,
  EnemyUpdateContext,
  HitscanEvent,
} from './enemy-controller.types';
import { PREVIOUS_STEP_DISTANCE_SENTINEL } from './enemy-controller.constants';

/**
 * Detect whether the weights argument is a per-enemy weight array
 * (`Float32Array[]`) rather than a single shared `Float32Array`.
 *
 * At runtime, `Array.isArray` returns `false` for `Float32Array` instances,
 * so this correctly distinguishes the two cases.
 *
 * @param weights - The weights argument (typed as Float32Array for backward
 *   compatibility, but may be Float32Array[] at runtime).
 * @returns `true` when weights is an array of Float32Array (one per enemy).
 */
function isPerEnemyWeights(weights: unknown): weights is Float32Array[] {
  return (
    Array.isArray(weights) &&
    weights.length > 0 &&
    weights[0] instanceof Float32Array
  );
}

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
/** Next pool slot index for round-robin allocation. */
let nextPoolSlotIndex = 0;

/**
 * Acquire a pooled {@link EnemyUpdateContext} slot and reset it for a new
 * enemy update. The slot's scalar fields are reset; `position` and
 * `slotTarget` are mutated in place (their x/y are zeroed) so no new
 * objects are allocated per enemy per tick (A2 Fix 8).
 *
 * @param index - Enemy index in the source array.
 * @param enemyState - Source enemy snapshot.
 * @param previous - Previous controlled state, if any.
 * @param gameState - Current game snapshot.
 * @param collisionMap - Map queried for solid cells.
 * @param distanceMap - BFS distance map from the player position.
 * @param dtMs - Tick duration in milliseconds.
 * @param hitscanEvents - Array to append any fire event to.
 * @param injectedWeights - Champion MLP weights, if provided.
 * @returns The acquired (and reset) pooled context slot.
 */
function acquirePooledContext(
  index: number,
  enemyState: EnemyState,
  previous: ControlledEnemy | undefined,
  gameState: GameState,
  collisionMap: CollisionMap,
  distanceMap: DistanceMap,
  dtMs: number,
  hitscanEvents: HitscanEvent[],
  injectedWeights: Float32Array | undefined,
): WritableEnemyUpdateContext {
  const slot = pooledContextSlots[nextPoolSlotIndex % POOLED_CONTEXT_SLOT_COUNT];
  nextPoolSlotIndex += 1;

  // Reset all fields on the pooled slot — no new object allocation.
  slot.index = index;
  slot.enemyState = enemyState;
  slot.previous = previous;
  slot.gameState = gameState;
  slot.collisionMap = collisionMap;
  slot.distanceMap = distanceMap;
  slot.dtMs = dtMs;
  slot.hitscanEvents = hitscanEvents;
  slot.injectedWeights = injectedWeights;
  slot.isRespawn = false;
  slot.previousOrDefault = previous as ControlledEnemy;
  slot.ammo = 0;
  slot.fireCooldownMs = 0;
  slot.deRezElapsedMs = 0;
  slot.walkTick = 0;
  slot.shootBlinkTicks = 0;
  slot.flankStallTicks = 0;
  slot.bfsStallTicks = 0;
  slot.weights = undefined;
  slot.variantId = 0;
  // Mutate position in place — do NOT create a new {x, y} object.
  slot.position.x = 0;
  slot.position.y = 0;
  slot.yawRad = 0;
  slot.stunTimerMs = 0;
  slot.distToPlayer = 0;
  slot.moved = false;
  slot.animationState = 'idle';
  slot.shouldMoveByBfs = false;
  slot.shouldMoveByFlank = false;
  // Mutate slotTarget in place.
  slot.slotTarget.x = 0;
  slot.slotTarget.y = 0;
  slot.isFiring = false;

  return slot;
}

/**
 * Release a pooled context slot back to the pool. Clears object references
 * to avoid retaining large structures between ticks (A2 Fix 8).
 *
 * @param ctx - The pooled context slot to release.
 */
function releasePooledContext(ctx: WritableEnemyUpdateContext): void {
  ctx.enemyState = undefined as unknown as EnemyState;
  ctx.gameState = undefined as unknown as GameState;
  ctx.collisionMap = undefined as unknown as CollisionMap;
  ctx.distanceMap = undefined as unknown as DistanceMap;
  ctx.previous = undefined;
  ctx.previousOrDefault = undefined as unknown as ControlledEnemy;
  ctx.injectedWeights = undefined;
  ctx.weights = undefined;
  ctx.hitscanEvents = [];
}

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
  // Acquire a pooled context slot — no fresh ctx object allocated (A2 Fix 8).
  const ctx = acquirePooledContext(
    index,
    enemyState,
    previous,
    gameState,
    collisionMap,
    distanceMap,
    dtMs,
    hitscanEvents,
    injectedWeights,
  );

  // Step 1: Resolve respawn and initialize mutable state.
  resolveRespawnState(ctx);

  // Step 2: Handle death / de-rez (early return).
  const deathResult = handleDeath(ctx);
  if (deathResult) {
    releasePooledContext(ctx);
    return deathResult;
  }

  // Step 3: Handle hit-stun (early return).
  const stunResult = handleStun(ctx);
  if (stunResult) {
    releasePooledContext(ctx);
    return stunResult;
  }

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
  computeMovementFlat(ctx);

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

  // Step 10: Assemble and return. Create a fresh position object for the
  // returned ControlledEnemy so it is not affected by the next pool reuse.
  const result: ControlledEnemy = {
    index,
    position: { x: ctx.position.x, y: ctx.position.y },
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

  // Release the pooled context slot back to the pool (A2 Fix 8).
  releasePooledContext(ctx);

  return result;
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
 * When `weights` is a `Float32Array[]` (per-enemy weight vectors from A4's
 * `selectVariant`), each enemy at index `i` receives its own weight slot
 * (`weights[i]`) instead of sharing a single reference. This is the B1
 * per-enemy genome contract.
 *
 * @param controller - Previous controller state.
 * @param state - Current game state.
 * @param collisionMap - Map queried for solid cells.
 * @param dtMs - Tick duration in milliseconds.
 * @param weights - Optional champion MLP weights from the current population
 *   snapshot. Activates MLP re-ranking when provided. May be a single
 *   `Float32Array` (shared by all enemies) or a `Float32Array[]` (per-enemy
 *   weight vectors from A4's `selectVariant`); the latter assigns each enemy
 *   its own weight slot.
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

  // Reset the once-per-tick MLP debug guard so the first activation
  // failure in this tick is logged.
  resetMlpDebugGuard();

  // Use a plain array indexed by enemy index instead of allocating a new Map
  // every tick. Enemies are indexed 0..7 so a small array suffices.
  // Reuse the module-level array (A2 Fix 8: no per-tick allocation).
  reusablePreviousByIndex.length = 0;
  const previousByIndex = reusablePreviousByIndex;
  for (const enemy of controller.enemies) {
    previousByIndex[enemy.index] = enemy;
  }

  // Build BFS distance map from the player cell once per tick.
  // Use the cached distance map to avoid ~57 KB allocation per tick (A2 Fix 3).
  const distanceMap = getOrBuildCachedDistanceMap(
    collisionMap,
    state.player.position.x,
    state.player.position.y,
    state.seed ?? 0,
  );

  const enemies: ControlledEnemy[] = [];
  const hitscanEvents: HitscanEvent[] = [];

  // B1: Detect per-enemy weights (Float32Array[] from A4's selectVariant).
  // When provided, each enemy gets its own weight slot instead of sharing
  // a single Float32Array reference.
  const perEnemy = isPerEnemyWeights(weights);
  const perEnemyArr = perEnemy ? (weights as unknown as Float32Array[]) : null;

  for (let i = 0; i < state.enemies.length; i += 1) {
    const previous = previousByIndex[i];
    const enemyWeights = perEnemyArr
      ? perEnemyArr[i]
      : weights;
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
        enemyWeights,
      ),
    );
  }

  separateEnemies(enemies, collisionMap);

  return {
    enemies,
    hitscanEvents,
  };
}
