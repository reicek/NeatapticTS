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
import {
  NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
} from '../browser-entry/host/game/constants';
import type {
  EnemyState,
  GameState,
  Vector2,
} from '../browser-entry/host/game/types';
import {
  buildEnemyDistanceMap,
  buildVisionVector,
  type DistanceMap,
  getDistance,
} from './enemy-navigation';
import { activateMlp } from '../browser-entry/harness/enemy-mlp';

/**
 * Number of sim ticks the muzzle-flash shoot blink lasts on the upper body.
 *
 * When an enemy fires, the upper body shows the shoot frame for this many
 * ticks before reverting to the walk frame. Must be in the 3–5 range per
 * AC-10f-005.
 */
export const ENEMY_CONTROLLER_SHOOT_BLINK_TICKS = 4;

/**
 * Animation and AI state for a single controlled enemy, produced by the
 * controller each tick and consumed by the sprite renderer.
 *
 * This type is intentionally local to the controller so the core
 * {@link EnemyState} can stay minimal while the renderer gets richer,
 * per-enemy behavior data.
 */
export interface ControlledEnemy {
  /** Index of this enemy in the source {@link GameState.enemies} array. */
  index: number;
  /** Current world position after movement and collision resolution. */
  position: Vector2;
  /** Current hit points, mirrored from the source {@link EnemyState}. */
  health: number;
  /** Facing angle in radians; 0 = +X axis. */
  yawRad: number;
  /** Animation state consumed by the sprite renderer. */
  animationState: 'idle' | 'move' | 'fire' | 'death' | 'damage';
  /** Remaining hitscan ammunition. Depletes to trigger de-rez. */
  ammo: number;
  /** Milliseconds until the enemy may fire again. */
  fireCooldownMs: number;
  /** Milliseconds spent in the death de-rez animation. */
  deRezElapsedMs: number;
  /** `true` while the enemy is still active (not fully de-rezzed). */
  active: boolean;
  /** Sim tick counter driving the walk cycle (stand → walk1 → stand → walk2). Increments each tick the enemy moves; resets to 0 when idle. */
  walkTick: number;
  /** Remaining sim ticks for the muzzle-flash shoot blink on the upper body. When > 0 the renderer composites the shoot upper body over the walk lower body. */
  shootBlinkTicks: number;
  /** Number of consecutive ticks the enemy has been stalled in flanking mode. When this exceeds 3, the enemy temporarily switches to BFS mode to avoid permanent stalls against walls. */
  flankStallTicks: number;
  /** Number of consecutive ticks the enemy has been stalled in BFS mode. When this exceeds 3, the enemy tries non-distance-reducing cardinal directions for one tick to escape diagonal-gap deadlocks. */
  bfsStallTicks: number;
  /**
   * Neural network weights for this enemy's MLP controller, or `undefined`
   * when the enemy uses the default BFS navigation fallback.
   */
  weights: Float32Array | undefined;
  /** Variant identifier for this enemy (0 = default/champion). */
  variantId: number;
  /**
   * BFS distance at the enemy's cell from the previous tick, used to compute
   * the progress delta in the vision vector. `-1` when no previous data is
   * available (first tick or respawn).
   */
  previousStepDistance: number;
  /** Remaining hit-stun time in milliseconds (0 when not stunned). */
  stunTimerMs: number;
}

/**
 * One hitscan fire event emitted by an enemy this tick.
 */
export interface HitscanEvent {
  /** Source enemy index in {@link GameState.enemies}. */
  enemyIndex: number;
  /** World-space origin of the hitscan ray. */
  origin: Vector2;
  /** Normalized direction toward the player. */
  direction: Vector2;
  /** Hit points removed from the player on a confirmed hit. */
  damage: number;
}

/**
 * Output of a single controller tick, bundling per-enemy AI/animation
 * descriptors with any hitscan fire events produced this tick.
 */
export interface EnemyControllerState {
  /** Per-enemy AI/animation descriptors. */
  enemies: ControlledEnemy[];
  /** Hitscan fire events produced this tick. */
  hitscanEvents: HitscanEvent[];
}

/**
 * Enemy movement speed in world cells per second, tuned so evolved enemies
 * close distance quickly without overshooting the player at short range.
 */
export const ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND = 2.5;

/**
 * Collision radius in world cells.
 *
 * Enemies have a 192×192 block footprint and each floor cell spans 252 blocks,
 * giving a radius of 96 / 252 ≈ 0.381 cells.
 */
export const ENEMY_CONTROLLER_RADIUS_CELLS =
  NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS;

/**
 * Wall collision radius for BFS navigation movement, in world cells.
 *
 * Set to a quarter cell (0.25) so the center of an agent always stays at
 * least 0.25 cells away from any wall edge. In a 1-cell-wide corridor this
 * gives 0.5 cells of freedom (center ± 0.25), allowing enemies to pass
 * through comfortably. Used by {@link isPositionBlockedByWall} as a
 * bounding-box filter after the target-cell `isSolid` check: the target
 * cell is checked first for a fast rejection, then all cells overlapping
 * the circle of this radius around the desired position are checked for
 * solidity.
 */
export const ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS = 0.25;

/** Maximum cell distance at which an enemy will attempt to fire. */
export const ENEMY_CONTROLLER_FIRE_RANGE_CELLS = 8;

/**
 * Minimum cell distance the enemy tries to maintain from the player.
 *
 * Enemies chase the player but stop at this distance to prevent overlapping
 * the player's position. Set to 1.5 cells so enemies keep a visible gap and
 * do not clip into the player's sprite while remaining within firing range.
 */
export const ENEMY_CONTROLLER_STOP_DISTANCE_CELLS = 1.5;

/**
 * Radius within which enemies switch from BFS approach to flanking
 * (circling toward an assigned slot angle around the player).
 *
 * When an enemy is within this distance of the player, it stops using the
 * shared BFS distance map and instead navigates directly toward its assigned
 * slot position, causing enemies to spread out and surround the player
 * rather than clustering on one side.
 */
export const ENEMY_CONTROLLER_FLANKING_RADIUS_CELLS = 3.5;

/**
 * Cooldown between hitscan shots in milliseconds, preventing enemies from
 * firing continuously and giving the player a predictable rhythm.
 */
export const ENEMY_CONTROLLER_FIRE_COOLDOWN_MS = 1000;

/** Hit points removed from the player by a single enemy hitscan shot. */
export const ENEMY_CONTROLLER_HITSCAN_DAMAGE = 10;

/**
 * Starting ammunition for each freshly tracked enemy, depleted by hitscan
 * shots and used to trigger the de-rez death animation.
 */
export const ENEMY_CONTROLLER_STARTING_AMMO = 3;

/**
 * Duration of the death de-rez animation in milliseconds, kept identical to
 * the sprite-side timing so visual and AI states stay synchronized.
 *
 * The 700 ms window delivers a fast Tron-style pixel scatter rather than the
 * original 4-second fade, keeping combat feedback snappy.
 */
export const ENEMY_CONTROLLER_DE_REZ_DURATION_MS = 700;

/** Line-of-sight sampling step in world cells. */
const LINE_OF_SIGHT_STEP_CELLS = 0.5;

/**
 * Return whether a value is a finite number.
 *
 * @param value - Candidate value.
 * @returns Whether the value is finite.
 */
function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

/**
 * Check whether a circle at the given position overlaps any solid cell.
 *
 * For a circle of radius R centered at (x, y), this checks every grid cell
 * whose bounding box could overlap the circle. The cell range is computed
 * as floor(x − R) to ceil(x + R) − 1 on each axis, then each cell in that
 * bounding box is tested for solidity. This catches all wall cells the
 * enemy circle could overlap, including diagonal cells, preventing enemies
 * from clipping into walls or getting stuck inside them.
 *
 * @param collisionMap - Map queried for solid cells.
 * @param x - Desired center X in world cells.
 * @param y - Desired center Y in world cells.
 * @param radius - Collision radius in world cells.
 * @returns Whether any solid cell overlaps the circle's bounding box.
 */
function isPositionBlockedByWall(
  collisionMap: CollisionMap,
  x: number,
  y: number,
  radius: number,
): boolean {
  const minX = Math.floor(x - radius);
  const maxX = Math.ceil(x + radius) - 1;
  const minY = Math.floor(y - radius);
  const maxY = Math.ceil(y + radius) - 1;
  for (let cy = minY; cy <= maxY; cy += 1) {
    for (let cx = minX; cx <= maxX; cx += 1) {
      if (collisionMap.isSolid(cx, cy)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Clamp a timestep to a positive finite value.
 *
 * Invalid or non-positive timesteps fall back to the canonical fixed timestep
 * so malformed caller input cannot poison enemy behavior.
 *
 * @param dtMs - Candidate timestep in milliseconds.
 * @returns Positive finite timestep.
 */
function resolveTimestepMs(dtMs: number): number {
  if (dtMs === 0) return 0;
  return isFiniteNumber(dtMs) && dtMs > 0
    ? dtMs
    : NEATENSTEIN_FIXED_TIMESTEP_MS;
}

/**
 * Return whether a straight line from `from` to `to` is free of solid cells.
 *
 * Samples the grid at regular intervals. The start cell is skipped because the
 * caller is expected to be in an open cell.
 *
 * @param from - Ray origin in world cells.
 * @param to - Ray target in world cells.
 * @param collisionMap - Map queried for solid cells.
 * @returns Whether the target is visible from the origin.
 */
function hasLineOfSight(
  from: Vector2,
  to: Vector2,
  collisionMap: CollisionMap,
): boolean {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);

  if (!isFiniteNumber(distance) || distance === 0) {
    return true;
  }

  const steps = Math.max(1, Math.ceil(distance / LINE_OF_SIGHT_STEP_CELLS));
  const stepX = dx / steps;
  const stepY = dy / steps;

  for (let i = 1; i <= steps; i += 1) {
    const x = from.x + stepX * i;
    const y = from.y + stepY * i;

    if (collisionMap.isSolid(Math.floor(x), Math.floor(y))) {
      return false;
    }
  }

  return true;
}

/**
 * Build a fresh controller state from a {@link GameState} snapshot.
 *
 * Each source enemy is initialized with default ammunition, no fire cooldown,
 * and the idle animation state. Callers must hold the returned state across
 * ticks so fire cooldowns, ammunition, and de-rez timing advance correctly.
 *
 * @param state - Source game snapshot.
 * @returns Fresh controller state.
 */
export function createEnemyControllerState(
  state: GameState,
): EnemyControllerState {
  return {
    enemies: state.enemies.map((enemy, index) => ({
      index,
      position: { ...enemy.position },
      health: enemy.health,
      yawRad: 0,
      animationState: enemy.health <= 0 ? 'death' : 'idle',
      ammo: ENEMY_CONTROLLER_STARTING_AMMO,
      fireCooldownMs: 0,
      deRezElapsedMs: 0,
      active: true,
      walkTick: 0,
      shootBlinkTicks: 0,
      flankStallTicks: 0,
      bfsStallTicks: 0,
      weights: undefined,
      variantId: 0,
      previousStepDistance: -1,
      stunTimerMs: 0,
    })),
    hitscanEvents: [],
  };
}

/**
 * Update one controlled enemy for a single tick.
 *
 * @param index - Index in the source enemy array.
 * @param enemyState - Source enemy snapshot.
 * @param previous - Previous controlled state, if any.
 * @param gameState - Current game snapshot.
 * @param collisionMap - Map queried for solid cells.
 * @param distanceMap - BFS distance map from the player position.
 * @param dtMs - Tick duration in milliseconds.
 * @param hitscanEvents - Array to append any fire event to.
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
): ControlledEnemy {
  const isRespawn =
    previous !== undefined && !previous.active && enemyState.health > 0;

  const previousOrDefault: ControlledEnemy =
    previous ??
    ({
      index,
      position: { ...enemyState.position },
      health: enemyState.health,
      yawRad: 0,
      animationState: 'idle',
      ammo: ENEMY_CONTROLLER_STARTING_AMMO,
      fireCooldownMs: 0,
      deRezElapsedMs: 0,
      active: true,
      walkTick: 0,
      shootBlinkTicks: 0,
      flankStallTicks: 0,
      bfsStallTicks: 0,
      weights: undefined,
      variantId: 0,
      previousStepDistance: -1,
      stunTimerMs: 0,
    } as ControlledEnemy);

  const ammo = isRespawn
    ? ENEMY_CONTROLLER_STARTING_AMMO
    : previousOrDefault.ammo;
  const fireCooldownMs = Math.max(
    0,
    (isRespawn ? 0 : previousOrDefault.fireCooldownMs) - dtMs,
  );
  let deRezElapsedMs = isRespawn ? 0 : previousOrDefault.deRezElapsedMs;
  let walkTick = isRespawn ? 0 : previousOrDefault.walkTick;
  let shootBlinkTicks = isRespawn ? 0 : previousOrDefault.shootBlinkTicks;
  let flankStallTicks = isRespawn ? 0 : previousOrDefault.flankStallTicks;
  let bfsStallTicks = isRespawn ? 0 : previousOrDefault.bfsStallTicks;
  const weights = isRespawn ? undefined : previousOrDefault.weights;
  const variantId = isRespawn ? 0 : previousOrDefault.variantId;
  let position = isRespawn
    ? { ...enemyState.position }
    : { ...previousOrDefault.position };
  let yawRad = isRespawn ? 0 : previousOrDefault.yawRad;

  // Death / de-rez path: only health depletion can kill an enemy.
  if (enemyState.health <= 0) {
    deRezElapsedMs += dtMs;
    const active = deRezElapsedMs < ENEMY_CONTROLLER_DE_REZ_DURATION_MS;

    return {
      index,
      position,
      health: enemyState.health,
      yawRad,
      animationState: 'death',
      ammo: Math.max(0, ammo),
      fireCooldownMs,
      deRezElapsedMs,
      active,
      walkTick: 0,
      shootBlinkTicks: 0,
      flankStallTicks: 0,
      bfsStallTicks: 0,
      weights,
      variantId,
      previousStepDistance: -1,
      stunTimerMs: 0,
    };
  }

  // Hit-stun path: while stunTimerMs > 0 the enemy skips movement, MLP, and
  // fire. The position is adopted from EnemyState (which may include the
  // pushback offset applied by applyEnemyDamage). The animation state is set
  // to 'damage' so the renderer shows the hit-flash overlay.
  // Decrement by the FIXED simulation timestep (not the variable rAF dtMs)
  // so stun duration is deterministic across different frame timings.
  // On zero-timestep sync passes (dtMs === 0) the timer is not decremented.
  const stunTimerMs = Math.max(
    0,
    (enemyState.stunTimerMs ?? 0) -
      (dtMs > 0 ? NEATENSTEIN_FIXED_TIMESTEP_MS : 0),
  );

  if (stunTimerMs > 0 || (enemyState.stunTimerMs ?? 0) > 0) {
    // Adopt the pushed-back position from EnemyState, not the stale
    // ControlledEnemy position.
    const stunPosition = { ...enemyState.position };

    return {
      index,
      position: stunPosition,
      health: enemyState.health,
      yawRad,
      animationState: 'damage',
      ammo: Math.max(0, ammo),
      fireCooldownMs,
      deRezElapsedMs,
      active: true,
      walkTick: 0,
      shootBlinkTicks: 0,
      flankStallTicks: 0,
      bfsStallTicks: 0,
      weights,
      variantId,
      previousStepDistance: -1,
      stunTimerMs,
    };
  }

  // Alive path: navigate via BFS distance map and fire when able.
  const dxToPlayer = gameState.player.position.x - position.x;
  const dyToPlayer = gameState.player.position.y - position.y;
  const distToPlayer = Math.hypot(dxToPlayer, dyToPlayer);

  if (isFiniteNumber(distToPlayer) && distToPlayer > 0) {
    yawRad = Math.atan2(dyToPlayer, dxToPlayer);
  }

  let moved = false;
  let animationState: ControlledEnemy['animationState'] = 'idle';

  // Flanking: assign each enemy a slot angle around the player so they
  // spread out and approach from different directions instead of all
  // clustering on one side. Each enemy gets an evenly-spaced angle; with
  // one enemy the slot is at the player's position (no flanking).
  const numEnemies = gameState.enemies.length;
  const slotAngle = numEnemies > 1 ? (index * 2 * Math.PI) / numEnemies : 0;
  const slotTarget = {
    x:
      gameState.player.position.x +
      Math.cos(slotAngle) * ENEMY_CONTROLLER_STOP_DISTANCE_CELLS,
    y:
      gameState.player.position.y +
      Math.sin(slotAngle) * ENEMY_CONTROLLER_STOP_DISTANCE_CELLS,
  };

  // Wall-aware slot placement: if the slot target lands inside a wall,
  // try shifting the angle by small increments to find a nearby non-solid
  // slot. If no valid slot is found after all offsets, fall back to BFS
  // mode for this enemy (shouldMoveByBfs = true, shouldMoveByFlank = false).
  let slotInWall = isPositionBlockedByWall(
    collisionMap,
    slotTarget.x,
    slotTarget.y,
    ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
  );
  if (slotInWall) {
    const angleOffsets = [
      Math.PI / 12,
      -Math.PI / 12,
      Math.PI / 6,
      -Math.PI / 6,
      Math.PI / 4,
      -Math.PI / 4,
      Math.PI / 3,
      -Math.PI / 3,
      Math.PI / 2,
      -Math.PI / 2,
    ];
    for (const offset of angleOffsets) {
      const shiftedAngle = slotAngle + offset;
      const candidateX =
        gameState.player.position.x +
        Math.cos(shiftedAngle) * ENEMY_CONTROLLER_STOP_DISTANCE_CELLS;
      const candidateY =
        gameState.player.position.y +
        Math.sin(shiftedAngle) * ENEMY_CONTROLLER_STOP_DISTANCE_CELLS;
      if (
        !isPositionBlockedByWall(
          collisionMap,
          candidateX,
          candidateY,
          ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
        )
      ) {
        slotTarget.x = candidateX;
        slotTarget.y = candidateY;
        slotInWall = false;
        break;
      }
    }
  }
  const dxToSlot = slotTarget.x - position.x;
  const dyToSlot = slotTarget.y - position.y;
  const distToSlot = Math.hypot(dxToSlot, dyToSlot);
  const inFlankingRange =
    numEnemies > 1 &&
    isFiniteNumber(distToPlayer) &&
    distToPlayer <= ENEMY_CONTROLLER_FLANKING_RADIUS_CELLS;

  let shouldMoveByBfs =
    (!inFlankingRange || slotInWall) &&
    isFiniteNumber(distToPlayer) &&
    distToPlayer > ENEMY_CONTROLLER_STOP_DISTANCE_CELLS;
  let shouldMoveByFlank =
    inFlankingRange &&
    !slotInWall &&
    isFiniteNumber(distToSlot) &&
    distToSlot > 0.15;

  // Greedy descent stall fallback: if the enemy has been stalled in
  // flanking mode for too many consecutive ticks, temporarily switch to
  // BFS mode to prevent permanent stalls against walls.
  if (inFlankingRange && flankStallTicks > 3) {
    shouldMoveByBfs =
      isFiniteNumber(distToPlayer) &&
      distToPlayer > ENEMY_CONTROLLER_STOP_DISTANCE_CELLS;
    shouldMoveByFlank = false;
  }

  if (shouldMoveByBfs || shouldMoveByFlank) {
    const cellX = Math.floor(position.x);
    const cellY = Math.floor(position.y);
    const currentDist = getDistance(distanceMap, cellX, cellY);

    // BFS mode requires a valid distance map entry. Flanking mode
    // navigates by direct vector to the slot target and does not need BFS.
    if (shouldMoveByFlank || currentDist >= 0) {
      const stepDistance =
        ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND * (dtMs / 1000);

      // Scale nudge magnitude by stepDistance so the correction is visually
      // continuous at high frame rates. At 60fps a normal step is ~0.04 cells
      // but the nudge could otherwise jump ~0.5 cells — a visible teleport.
      const nudgeScale = Math.min(1, stepDistance / 0.5);

      // Pre-move position correction (turn centering): if the enemy's
      // current position circle overlaps a wall (e.g., off-center on the
      // axis parallel to the upcoming move from a previous corridor leg),
      // nudge toward the cell center before computing desired positions.
      // This prevents turn deadlocks where the enemy can't center without
      // moving and can't move without centering.
      if (
        isPositionBlockedByWall(
          collisionMap,
          position.x,
          position.y,
          ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
        )
      ) {
        const cellCenterX = Math.floor(position.x) + 0.5;
        const cellCenterY = Math.floor(position.y) + 0.5;
        if (
          !isPositionBlockedByWall(
            collisionMap,
            cellCenterX,
            position.y,
            ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
          )
        ) {
          position.x += (cellCenterX - position.x) * nudgeScale;
        } else if (
          !isPositionBlockedByWall(
            collisionMap,
            position.x,
            cellCenterY,
            ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
          )
        ) {
          position.y += (cellCenterY - position.y) * nudgeScale;
        } else if (
          !isPositionBlockedByWall(
            collisionMap,
            cellCenterX,
            cellCenterY,
            ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
          )
        ) {
          position.x += (cellCenterX - position.x) * nudgeScale;
          position.y += (cellCenterY - position.y) * nudgeScale;
        }
      }

      // Try all 4 cardinal directions sorted by priority. In BFS mode,
      // directions are sorted by BFS distance (lowest first). In flanking
      // mode, directions are sorted by distance to the slot target.
      // When MLP weights are assigned and dtMs > 0, BFS directions are
      // re-ranked by MLP score (highest first) instead of pure BFS distance.
      const directions: Array<[number, number]> = [
        [0, -1], // N
        [1, 0], // E
        [0, 1], // S
        [-1, 0], // W
      ];

      // MLP re-ranking: if weights are valid and dtMs > 0, compute MLP
      // outputs from the vision vector and re-rank BFS candidates by MLP
      // score. Falls back to standard BFS sort on any error or NaN output.
      let mlpDesiredX = 0;
      let mlpDesiredY = 0;
      let useMlpReRank = false;
      if (
        shouldMoveByBfs &&
        !shouldMoveByFlank &&
        dtMs > 0 &&
        weights !== undefined
      ) {
        try {
          /* istanbul ignore next -- dead code: isRespawn clears weights before MLP guard at line 655 */
          const prevStepDist = isRespawn
            ? -1
            : previousOrDefault.previousStepDistance;
          const visionVector = buildVisionVector(
            distanceMap,
            cellX,
            cellY,
            prevStepDist >= 0 ? prevStepDist : undefined,
          );
          const outputs = activateMlp(weights, visionVector);
          if (Number.isFinite(outputs[0]) && Number.isFinite(outputs[1])) {
            const move = outputs[0];
            const strafe = outputs[1];
            const facingX = Math.cos(yawRad);
            const facingY = Math.sin(yawRad);
            const perpX = -facingY;
            const perpY = facingX;
            mlpDesiredX = move * facingX + strafe * perpX;
            mlpDesiredY = move * facingY + strafe * perpY;
            useMlpReRank = true;
          }
        } catch {
          // Wrong weight length or mismatched input → BFS fallback.
        }
      }

      if (shouldMoveByFlank) {
        directions.sort(
          (a, b) =>
            Math.hypot(
              position.x + a[0] * stepDistance - slotTarget.x,
              position.y + a[1] * stepDistance - slotTarget.y,
            ) -
            Math.hypot(
              position.x + b[0] * stepDistance - slotTarget.x,
              position.y + b[1] * stepDistance - slotTarget.y,
            ),
        );
      } else if (useMlpReRank) {
        // Re-rank BFS candidates by MLP score (highest first).
        directions.sort(
          (a, b) =>
            b[0] * mlpDesiredX +
            b[1] * mlpDesiredY -
            (a[0] * mlpDesiredX + a[1] * mlpDesiredY),
        );
      } else {
        directions.sort(
          (a, b) =>
            getDistance(distanceMap, cellX + a[0], cellY + a[1]) -
            getDistance(distanceMap, cellX + b[0], cellY + b[1]),
        );
      }

      for (const [dx, dy] of directions) {
        // In BFS mode, skip directions that don't reduce BFS distance.
        // In flanking mode, all cardinal directions are candidates.
        if (!shouldMoveByFlank) {
          const dist = getDistance(distanceMap, cellX + dx, cellY + dy);
          if (dist < 0 || dist >= currentDist) {
            continue; // skip walls, unreachable, and increasing distance
          }
        }
        // Fast first filter: check only the target cell for solidity (like
        // asciiMaze moveAgent). This rejects moves into wall cells quickly.
        if (collisionMap.isSolid(cellX + dx, cellY + dy)) {
          continue;
        }
        let desired = {
          x: position.x + dx * stepDistance,
          y: position.y + dy * stepDistance,
        };

        // Pre-collision centering: if the target or current cell is a
        // 1-cell gap (walls on both perpendicular sides), snap the
        // perpendicular coordinate to the cell center BEFORE the
        // collision check. This prevents a chicken-and-egg deadlock
        // where an off-center enemy cannot enter a 1-cell gap because
        // the collision check blocks the move, and the post-move
        // centering never runs because the move was blocked. Gated by
        // stepDistance > 0 so that zero-timestep sync passes (dtMs=0)
        // do not snap the perpendicular axis and accidentally "move" the
        // enemy via centering alone.
        const newCellX = cellX + dx;
        const newCellY = cellY + dy;
        if (stepDistance > 0 && dx !== 0) {
          // Horizontal move: perpendicular axis is Y. Check the target
          // cell first, then the current cell as a fallback. Both snap
          // to the same value (cellY + 0.5) since dy === 0 for horizontal
          // moves, so newCellY === cellY. One-sided centering: when only
          // one perpendicular side is walled, snap toward center only if
          // the enemy is drifting toward that wall. This rescues diagonal
          // gaps where walls alternate sides per row.
          const wN = collisionMap.isSolid(newCellX, newCellY - 1);
          const wS = collisionMap.isSolid(newCellX, newCellY + 1);
          if (wN && wS) {
            desired.y = newCellY + 0.5;
          } else if (wN && desired.y < newCellY + 0.5) {
            desired.y = newCellY + 0.5;
          } else if (wS && desired.y > newCellY + 0.5) {
            desired.y = newCellY + 0.5;
          } else {
            const wN0 = collisionMap.isSolid(cellX, cellY - 1);
            const wS0 = collisionMap.isSolid(cellX, cellY + 1);
            if (wN0 && wS0) {
              desired.y = cellY + 0.5;
            } else if (wN0 && desired.y < cellY + 0.5) {
              desired.y = cellY + 0.5;
            } else if (wS0 && desired.y > cellY + 0.5) {
              desired.y = cellY + 0.5;
            }
          }
        } else if (stepDistance > 0) {
          // Vertical move: perpendicular axis is X.
          /* istanbul ignore else -- cardinal directions always have exactly one of dx/dy nonzero, so when dx===0, dy must be nonzero */
          if (dy !== 0) {
            const wW = collisionMap.isSolid(newCellX - 1, newCellY);
            const wE = collisionMap.isSolid(newCellX + 1, newCellY);
            if (wW && wE) {
              desired.x = newCellX + 0.5;
            } else if (wW && desired.x < newCellX + 0.5) {
              desired.x = newCellX + 0.5;
            } else if (wE && desired.x > newCellX + 0.5) {
              desired.x = newCellX + 0.5;
            } else {
              const wW0 = collisionMap.isSolid(cellX - 1, cellY);
              const wE0 = collisionMap.isSolid(cellX + 1, cellY);
              if (wW0 && wE0) {
                desired.x = cellX + 0.5;
              } else if (wW0 && desired.x < cellX + 0.5) {
                desired.x = cellX + 0.5;
              } else if (wE0 && desired.x > cellX + 0.5) {
                desired.x = cellX + 0.5;
              }
            }
          }
        }

        // Second filter: circle-overlap wall check. Check all cells that
        // the enemy circle (center + radius) could overlap for solidity.
        // This prevents enemies from clipping into walls, including
        // diagonal cells, while allowing traversal of 1-cell-wide corridors
        // (center ± 0.25 gives 0.5 cells of freedom in a 1-cell corridor).
        // Enemy-enemy separation is handled separately by separateEnemies.
        if (
          isPositionBlockedByWall(
            collisionMap,
            desired.x,
            desired.y,
            ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
          )
        ) {
          // Retry-after-block: snap the perpendicular coordinate to the
          // target cell center and re-check once. This rescues moves that
          // fail collision only because of off-center drift in a diagonal
          // gap, where the pre-collision one-sided snap did not fire
          // because the enemy was drifting away from the single wall.
          const retryDesired = { ...desired };
          if (dx !== 0) {
            retryDesired.y = newCellY + 0.5;
          } else {
            retryDesired.x = newCellX + 0.5;
          }
          if (
            isPositionBlockedByWall(
              collisionMap,
              retryDesired.x,
              retryDesired.y,
              ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
            )
          ) {
            continue;
          }
          desired = retryDesired;
        }
        if (desired.x !== position.x || desired.y !== position.y) {
          moved = true;
          position = desired;

          // Corridor centering: after moving in a cardinal direction, if
          // the perpendicular axis has walls on both sides (1-cell-wide
          // corridor), nudge the enemy toward the cell center on the
          // perpendicular axis. This prevents drift that causes enemies to
          // get stuck in narrow corridors over many ticks.
          const newCellX = Math.floor(position.x);
          const newCellY = Math.floor(position.y);
          if (dx !== 0) {
            // Horizontal move: check perpendicular (Y) walls. One-sided
            // centering: snap toward center when only one side is walled
            // and the enemy is drifting toward that wall.
            const wN = collisionMap.isSolid(newCellX, newCellY - 1);
            const wS = collisionMap.isSolid(newCellX, newCellY + 1);
            if (wN && wS) {
              position.y = newCellY + 0.5;
            } else if (wN && position.y < newCellY + 0.5) {
              position.y = newCellY + 0.5;
            } else if (wS && position.y > newCellY + 0.5) {
              position.y = newCellY + 0.5;
            }
          } else {
            // Vertical move: check perpendicular (X) walls.
            /* istanbul ignore else -- cardinal directions always have exactly one of dx/dy nonzero, so when dx===0, dy must be nonzero */
            if (dy !== 0) {
              const wW = collisionMap.isSolid(newCellX - 1, newCellY);
              const wE = collisionMap.isSolid(newCellX + 1, newCellY);
              if (wW && wE) {
                position.x = newCellX + 0.5;
              } else if (wW && position.x < newCellX + 0.5) {
                position.x = newCellX + 0.5;
              } else if (wE && position.x > newCellX + 0.5) {
                position.x = newCellX + 0.5;
              }
            }
          }
        }
        break;
      }
    }

    // BFS stall-recovery fallback: if the enemy has been stalled in BFS
    // mode for too many consecutive ticks (e.g., diagonal-gap deadlock
    // where circle-overlap collision blocks the only BFS-decreasing
    // direction, or the BFS distance map cannot reach the enemy's cell),
    // try all 4 cardinal directions allowing non-distance-reducing moves
    // for one tick to escape the deadlock. This runs outside the
    // currentDist >= 0 guard so it can also rescue enemies in
    // BFS-unreachable cells.
    if (shouldMoveByBfs && !shouldMoveByFlank && !moved && bfsStallTicks > 3) {
      const escapeStep =
        ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND * (dtMs / 1000);
      const escapeDirections: Array<[number, number]> = [
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, 0],
      ];
      for (const [dx, dy] of escapeDirections) {
        if (collisionMap.isSolid(cellX + dx, cellY + dy)) {
          continue;
        }
        const escapeDesired = {
          x: position.x + dx * escapeStep,
          y: position.y + dy * escapeStep,
        };
        // Snap perpendicular to target cell center to avoid circle-overlap
        // collision with diagonal walls.
        if (dx !== 0) {
          escapeDesired.y = cellY + dy + 0.5;
        } else {
          escapeDesired.x = cellX + dx + 0.5;
        }
        if (
          isPositionBlockedByWall(
            collisionMap,
            escapeDesired.x,
            escapeDesired.y,
            ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
          )
        ) {
          continue;
        }
        if (escapeDesired.x !== position.x || escapeDesired.y !== position.y) {
          moved = true;
          position = escapeDesired;
        }
        break;
      }
    }
  }

  // Update walk tick: increment each tick the enemy moves, reset when idle.
  // Skip walk tick updates on zero-timestep sync passes (display.worker.ts
  // calls updateEnemyController with dtMs=0 after the real tick) to preserve
  // the walkTick incremented by the preceding real-timestep pass. Without
  // this guard, the sync pass resets walkTick to 0 every frame, causing
  // enemies to glide instead of playing the walk cycle.
  if (dtMs > 0) {
    walkTick = moved ? walkTick + 1 : 0;
    // Track consecutive flanking stalls. If the enemy was in flanking mode
    // and didn't move, increment the counter. Otherwise reset it.
    if (shouldMoveByFlank && !moved) {
      flankStallTicks += 1;
    } else {
      flankStallTicks = 0;
    }
    // Track consecutive BFS stalls. If the enemy was in BFS mode (not
    // flanking) and didn't move, increment the counter. Otherwise reset
    // it. When the counter exceeds 3, the BFS stall-recovery fallback
    // tries non-distance-reducing cardinal directions to escape.
    if (shouldMoveByBfs && !shouldMoveByFlank && !moved) {
      bfsStallTicks += 1;
    } else {
      bfsStallTicks = 0;
    }
  }

  let isFiring = false;
  if (
    fireCooldownMs <= 0 &&
    ammo > 0 &&
    distToPlayer <= ENEMY_CONTROLLER_FIRE_RANGE_CELLS &&
    hasLineOfSight(position, gameState.player.position, collisionMap)
  ) {
    isFiring = true;
    hitscanEvents.push({
      enemyIndex: index,
      origin: { ...position },
      direction: { x: Math.cos(yawRad), y: Math.sin(yawRad) },
      damage: ENEMY_CONTROLLER_HITSCAN_DAMAGE,
    });
  }

  // Decrement shoot blink from the previous tick, then refresh if firing.
  if (shootBlinkTicks > 0) {
    shootBlinkTicks -= 1;
  }
  if (isFiring) {
    shootBlinkTicks = ENEMY_CONTROLLER_SHOOT_BLINK_TICKS;
  }

  if (isFiring) {
    animationState = 'fire';
  } else if (moved) {
    animationState = 'move';
  }

  // Compute the BFS distance at the enemy's final cell for progress tracking.
  // This becomes previousStepDistance on the next tick, enabling the vision
  // vector's progressDelta signal. On respawn, set to -1 (no previous data).
  const finalCellX = Math.floor(position.x);
  const finalCellY = Math.floor(position.y);
  const finalDist = isRespawn
    ? -1
    : getDistance(distanceMap, finalCellX, finalCellY);

  return {
    index,
    position,
    health: enemyState.health,
    yawRad,
    animationState,
    ammo: isFiring ? Math.max(0, ammo - 1) : ammo,
    fireCooldownMs: isFiring
      ? ENEMY_CONTROLLER_FIRE_COOLDOWN_MS
      : fireCooldownMs,
    deRezElapsedMs,
    active: true,
    walkTick,
    shootBlinkTicks,
    flankStallTicks,
    bfsStallTicks,
    weights,
    variantId,
    previousStepDistance: finalDist >= 0 ? finalDist : -1,
    stunTimerMs: 0,
  };
}

/**
 * Push active enemies apart so their 192×192-block footprints do not overlap.
 *
 * A single pairwise pass is sufficient because the AI moves slowly and the
 * collision radius is small. The separation is applied symmetrically, so two
 * overlapping enemies each move half the overlap distance.
 *
 * After the separation pass, each active enemy's position is re-checked
 * against the wall collision map. If separation pushed an enemy inside a
 * wall, its position is reverted to the pre-separation position (which was
 * already verified wall-free by the movement code).
 *
 * @param enemies - Resolved enemy descriptors produced by
 *   {@link updateControlledEnemy} this tick.
 * @param collisionMap - Map queried for solid cells, used for the
 *   post-separation wall re-check.
 */
function separateEnemies(
  enemies: ControlledEnemy[],
  collisionMap: CollisionMap,
): void {
  const combinedDiameter = ENEMY_CONTROLLER_RADIUS_CELLS * 2;

  // Save pre-separation positions for the post-separation wall re-check.
  const prePositions = enemies.map((e) => ({
    x: e.position.x,
    y: e.position.y,
  }));

  for (let i = 0; i < enemies.length; i += 1) {
    const a = enemies[i];
    if (!a.active || a.stunTimerMs > 0) {
      continue;
    }

    for (let j = i + 1; j < enemies.length; j += 1) {
      const b = enemies[j];
      if (!b.active || b.stunTimerMs > 0) {
        continue;
      }

      const dx = b.position.x - a.position.x;
      const dy = b.position.y - a.position.y;
      const distanceSquared = dx * dx + dy * dy;
      const combinedRadius = combinedDiameter;
      const combinedRadiusSquared = combinedRadius * combinedRadius;

      if (distanceSquared >= combinedRadiusSquared) {
        continue;
      }

      const distance = Math.sqrt(distanceSquared) || 1;
      const overlap = combinedRadius - distance;
      const offsetX = (dx / distance) * (overlap * 0.5);
      const offsetY = (dy / distance) * (overlap * 0.5);

      a.position.x -= offsetX;
      a.position.y -= offsetY;
      b.position.x += offsetX;
      b.position.y += offsetY;
    }
  }

  // Wall re-check: if separation pushed an enemy inside a wall, revert to
  // the pre-separation position (already verified wall-free by the move).
  for (let i = 0; i < enemies.length; i += 1) {
    const e = enemies[i];
    if (!e.active || e.stunTimerMs > 0) {
      continue;
    }
    if (
      isPositionBlockedByWall(
        collisionMap,
        e.position.x,
        e.position.y,
        ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
      )
    ) {
      e.position.x = prePositions[i].x;
      e.position.y = prePositions[i].y;
    }
  }
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
 * @param controller - Previous controller state.
 * @param state - Current game snapshot.
 * @param collisionMap - Map queried for solid cells.
 * @param dtMs - Tick duration in milliseconds.
 * @returns New controller state and any hitscan events produced this tick.
 */
export function updateEnemyController(
  controller: EnemyControllerState,
  state: GameState,
  collisionMap: CollisionMap,
  dtMs: number,
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
      ),
    );
  }

  separateEnemies(enemies, collisionMap);

  return {
    enemies,
    hitscanEvents,
  };
}
