/**
 * BFS / flanking / MLP movement executor for the enemy controller pipeline.
 *
 * Extracted from {@link enemy-controller.ts}. Contains the per-cardinal-
 * direction movement loop with pre-move centering, MLP re-ranking, collision
 * checks, post-move corridor centering, and BFS stall-recovery fallback.
 *
 * @module
 */

import {
  ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND,
  ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
  CELL_CENTER_OFFSET,
  DIRECTIONS,
  PREVIOUS_STEP_DISTANCE_SENTINEL,
} from './enemy-controller.constants';
import { NEATENSTEIN_MS_PER_SECOND } from '../constants';
import type { EnemyUpdateContext } from './enemy-controller.types';
import { isPositionBlockedByWall } from './enemy-controller.collision.utils';
import { buildVisionVector, getDistance } from './enemy-navigation';
import { activateMlpPooled } from '../harness/enemy-mlp';
import type { CollisionMap } from '../renderer/map';

/**
 * Once-per-tick guard for MLP activation failure debug logging.
 *
 * Set to `true` after the first `console.debug` call within a tick and
 * reset to `false` by {@link resetMlpDebugGuard} at the start of each
 * `updateEnemyController` call. Prevents spamming `console.debug` when
 * multiple enemies fail MLP activation in the same tick.
 */
let mlpDebugLoggedThisTick = false;

/**
 * Reset the once-per-tick MLP debug guard.
 *
 * Called at the start of each `updateEnemyController` tick so that the
 * first MLP activation failure in the new tick is logged, but subsequent
 * failures within the same tick are suppressed.
 */
export function resetMlpDebugGuard(): void {
  mlpDebugLoggedThisTick = false;
}

/**
 * Compute enemy movement for one tick via BFS, flanking, or MLP re-ranking.
 *
 * Reads `shouldMoveByBfs`, `shouldMoveByFlank`, `position`, and other state
 * from the context, then attempts to move the enemy in the best cardinal
 * direction. Updates `ctx.position` and `ctx.moved` on success.
 *
 * @param ctx - Mutable pipeline context.
 */
export function computeMovement(ctx: EnemyUpdateContext): void {
  const {
    shouldMoveByBfs,
    shouldMoveByFlank,
    collisionMap,
    distanceMap,
    dtMs,
    weights,
    yawRad,
    isRespawn,
    previousOrDefault,
    bfsStallTicks,
    slotTarget,
  } = ctx;

  if (!shouldMoveByBfs && !shouldMoveByFlank) return;

  let position = ctx.position;
  const cellX = Math.floor(position.x);
  const cellY = Math.floor(position.y);
  const currentDist = getDistance(distanceMap, cellX, cellY);

  // MLP re-ranking: if weights are valid and dtMs > 0, compute MLP
  // outputs from the vision vector and re-rank BFS candidates by MLP
  // score. Falls back to standard BFS sort on any error or NaN output.
  //
  // This block runs before the BFS distance early-return so that MLP
  // activation failures are logged via console.debug even when the enemy
  // occupies an unreachable cell (e.g. inside a wall).
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
      const prevStepDist = isRespawn
        ? PREVIOUS_STEP_DISTANCE_SENTINEL
        : previousOrDefault.previousStepDistance;
      const visionVector = buildVisionVector(
        distanceMap,
        cellX,
        cellY,
        prevStepDist >= 0 ? prevStepDist : undefined,
      );
      const outputs = activateMlpPooled(weights, visionVector);
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
      if (!mlpDebugLoggedThisTick) {
        console.debug('computeMovement: MLP activation failed, using BFS fallback');
        mlpDebugLoggedThisTick = true;
      }
    }
  }

  // BFS mode requires a valid distance map entry. Flanking mode
  // navigates by direct vector to the slot target and does not need BFS.
  if (!shouldMoveByFlank && currentDist < 0) return;

  const stepDistance =
    ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND *
    (dtMs / NEATENSTEIN_MS_PER_SECOND);

  // Scale nudge magnitude by stepDistance so the correction is visually
  // continuous at high frame rates.
  const nudgeScale = Math.min(1, stepDistance / CELL_CENTER_OFFSET);

  // Pre-move position correction (turn centering).
  position = applyPreMoveCentering(position, collisionMap, nudgeScale);

  const directions: Array<[number, number]> = [...DIRECTIONS].map(
    (d) => [...d] as [number, number],
  );
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

  let moved = false;

  for (const [dx, dy] of directions) {
    // In BFS mode, skip directions that don't reduce BFS distance.
    if (!shouldMoveByFlank) {
      const dist = getDistance(distanceMap, cellX + dx, cellY + dy);
      if (dist < 0 || dist >= currentDist) {
        continue;
      }
    }
    // Fast first filter: check only the target cell for solidity.
    if (collisionMap.isSolid(cellX + dx, cellY + dy)) {
      continue;
    }
    let desired = {
      x: position.x + dx * stepDistance,
      y: position.y + dy * stepDistance,
    };

    // Pre-collision centering for 1-cell gaps.
    const newCellX = cellX + dx;
    const newCellY = cellY + dy;
    if (stepDistance > 0 && dx !== 0) {
      desired = applyHorizontalCentering(
        desired,
        collisionMap,
        cellX,
        cellY,
        newCellX,
        newCellY,
      );
    } else if (stepDistance > 0) {
      /* istanbul ignore else -- cardinal directions always have exactly one of dx/dy nonzero, so when dx===0, dy must be nonzero */
      if (dy !== 0) {
        desired = applyVerticalCentering(
          desired,
          collisionMap,
          cellX,
          cellY,
          newCellX,
          newCellY,
        );
      }
    }

    // Second filter: circle-overlap wall check.
    if (
      isPositionBlockedByWall(
        collisionMap,
        desired.x,
        desired.y,
        ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
      )
    ) {
      // Retry-after-block: snap the perpendicular coordinate to the
      // target cell center and re-check once.
      const retryDesired = { ...desired };
      if (dx !== 0) {
        retryDesired.y = newCellY + CELL_CENTER_OFFSET;
      } else {
        retryDesired.x = newCellX + CELL_CENTER_OFFSET;
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
      position = applyPostMoveCentering(
        position,
        desired,
        collisionMap,
        dx,
        dy,
      );
    }
    break;
  }

  // BFS stall-recovery fallback.
  if (shouldMoveByBfs && !shouldMoveByFlank && !moved && bfsStallTicks > 3) {
    const escapeResult = applyBfsStallRecovery(
      position,
      collisionMap,
      cellX,
      cellY,
      dtMs,
    );
    if (escapeResult.moved) {
      moved = true;
      position = escapeResult.position;
    }
  }

  // Mutate ctx.position in place (A2 Fix 8: no new {x,y} allocation).
  ctx.position.x = position.x;
  ctx.position.y = position.y;
  ctx.moved = moved;
}

/**
 * Pre-move position correction: nudge toward cell center when the enemy's
 * position circle overlaps a wall.
 *
 * @param position - Current position (may be mutated and returned).
 * @param collisionMap - Map queried for solid cells.
 * @param nudgeScale - Scale factor for the nudge magnitude.
 * @returns Corrected position.
 */
function applyPreMoveCentering(
  position: { x: number; y: number },
  collisionMap: CollisionMap,
  nudgeScale: number,
): { x: number; y: number } {
  if (
    !isPositionBlockedByWall(
      collisionMap,
      position.x,
      position.y,
      ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
    )
  ) {
    return position;
  }
  const cellCenterX = Math.floor(position.x) + CELL_CENTER_OFFSET;
  const cellCenterY = Math.floor(position.y) + CELL_CENTER_OFFSET;
  if (
    !isPositionBlockedByWall(
      collisionMap,
      cellCenterX,
      position.y,
      ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
    )
  ) {
    return {
      x: position.x + (cellCenterX - position.x) * nudgeScale,
      y: position.y,
    };
  }
  if (
    !isPositionBlockedByWall(
      collisionMap,
      position.x,
      cellCenterY,
      ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
    )
  ) {
    return {
      x: position.x,
      y: position.y + (cellCenterY - position.y) * nudgeScale,
    };
  }
  if (
    !isPositionBlockedByWall(
      collisionMap,
      cellCenterX,
      cellCenterY,
      ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
    )
  ) {
    return {
      x: position.x + (cellCenterX - position.x) * nudgeScale,
      y: position.y + (cellCenterY - position.y) * nudgeScale,
    };
  }
  return position;
}

/**
 * Apply pre-collision horizontal centering for 1-cell gaps.
 *
 * When moving horizontally, checks perpendicular (Y) walls on both the
 * target and current cells, snapping the Y coordinate toward cell center
 * when walls are present.
 *
 * @param desired - Desired position before collision check.
 * @param collisionMap - Map queried for solid cells.
 * @param cellX - Current cell X.
 * @param cellY - Current cell Y.
 * @param newCellX - Target cell X.
 * @param newCellY - Target cell Y.
 * @returns Centered desired position.
 */
function applyHorizontalCentering(
  desired: { x: number; y: number },
  collisionMap: CollisionMap,
  cellX: number,
  cellY: number,
  newCellX: number,
  newCellY: number,
): { x: number; y: number } {
  const wN = collisionMap.isSolid(newCellX, newCellY - 1);
  const wS = collisionMap.isSolid(newCellX, newCellY + 1);
  if (wN && wS) {
    return { ...desired, y: newCellY + CELL_CENTER_OFFSET };
  }
  if (wN && desired.y < newCellY + CELL_CENTER_OFFSET) {
    return { ...desired, y: newCellY + CELL_CENTER_OFFSET };
  }
  if (wS && desired.y > newCellY + CELL_CENTER_OFFSET) {
    return { ...desired, y: newCellY + CELL_CENTER_OFFSET };
  }
  const wN0 = collisionMap.isSolid(cellX, cellY - 1);
  const wS0 = collisionMap.isSolid(cellX, cellY + 1);
  if (wN0 && wS0) {
    return { ...desired, y: cellY + CELL_CENTER_OFFSET };
  }
  if (wN0 && desired.y < cellY + CELL_CENTER_OFFSET) {
    return { ...desired, y: cellY + CELL_CENTER_OFFSET };
  }
  if (wS0 && desired.y > cellY + CELL_CENTER_OFFSET) {
    return { ...desired, y: cellY + CELL_CENTER_OFFSET };
  }
  return desired;
}

/**
 * Apply pre-collision vertical centering for 1-cell gaps.
 *
 * When moving vertically, checks perpendicular (X) walls on both the
 * target and current cells, snapping the X coordinate toward cell center
 * when walls are present.
 *
 * @param desired - Desired position before collision check.
 * @param collisionMap - Map queried for solid cells.
 * @param cellX - Current cell X.
 * @param cellY - Current cell Y.
 * @param newCellX - Target cell X.
 * @param newCellY - Target cell Y.
 * @returns Centered desired position.
 */
function applyVerticalCentering(
  desired: { x: number; y: number },
  collisionMap: CollisionMap,
  cellX: number,
  cellY: number,
  newCellX: number,
  newCellY: number,
): { x: number; y: number } {
  const wW = collisionMap.isSolid(newCellX - 1, newCellY);
  const wE = collisionMap.isSolid(newCellX + 1, newCellY);
  if (wW && wE) {
    return { ...desired, x: newCellX + CELL_CENTER_OFFSET };
  }
  if (wW && desired.x < newCellX + CELL_CENTER_OFFSET) {
    return { ...desired, x: newCellX + CELL_CENTER_OFFSET };
  }
  if (wE && desired.x > newCellX + CELL_CENTER_OFFSET) {
    return { ...desired, x: newCellX + CELL_CENTER_OFFSET };
  }
  const wW0 = collisionMap.isSolid(cellX - 1, cellY);
  const wE0 = collisionMap.isSolid(cellX + 1, cellY);
  if (wW0 && wE0) {
    return { ...desired, x: cellX + CELL_CENTER_OFFSET };
  }
  if (wW0 && desired.x < cellX + CELL_CENTER_OFFSET) {
    return { ...desired, x: cellX + CELL_CENTER_OFFSET };
  }
  if (wE0 && desired.x > cellX + CELL_CENTER_OFFSET) {
    return { ...desired, x: cellX + CELL_CENTER_OFFSET };
  }
  return desired;
}

/**
 * Apply post-move corridor centering after a successful move.
 *
 * After moving in a cardinal direction, if the perpendicular axis has walls
 * on both sides (1-cell-wide corridor), nudge the enemy toward the cell
 * center on the perpendicular axis. One-sided centering is also applied.
 *
 * @param oldPosition - Position before the move (unused, for API symmetry).
 * @param newPosition - Position after the move.
 * @param collisionMap - Map queried for solid cells.
 * @param dx - Movement direction X.
 * @param dy - Movement direction Y.
 * @returns Centered position after move.
 */
function applyPostMoveCentering(
  oldPosition: { x: number; y: number },
  newPosition: { x: number; y: number },
  collisionMap: CollisionMap,
  dx: number,
  dy: number,
): { x: number; y: number } {
  const position = { ...newPosition };
  const newCellX = Math.floor(position.x);
  const newCellY = Math.floor(position.y);
  if (dx !== 0) {
    const wN = collisionMap.isSolid(newCellX, newCellY - 1);
    const wS = collisionMap.isSolid(newCellX, newCellY + 1);
    if (wN && wS) {
      position.y = newCellY + CELL_CENTER_OFFSET;
    } else if (wN && position.y < newCellY + CELL_CENTER_OFFSET) {
      position.y = newCellY + CELL_CENTER_OFFSET;
    } else if (wS && position.y > newCellY + CELL_CENTER_OFFSET) {
      position.y = newCellY + CELL_CENTER_OFFSET;
    }
  } else {
    /* istanbul ignore else -- cardinal directions always have exactly one of dx/dy nonzero, so when dx===0, dy must be nonzero */
    if (dy !== 0) {
      const wW = collisionMap.isSolid(newCellX - 1, newCellY);
      const wE = collisionMap.isSolid(newCellX + 1, newCellY);
      if (wW && wE) {
        position.x = newCellX + CELL_CENTER_OFFSET;
      } else if (wW && position.x < newCellX + CELL_CENTER_OFFSET) {
        position.x = newCellX + CELL_CENTER_OFFSET;
      } else if (wE && position.x > newCellX + CELL_CENTER_OFFSET) {
        position.x = newCellX + CELL_CENTER_OFFSET;
      }
    }
  }
  return position;
}

/**
 * BFS stall-recovery fallback: try all 4 cardinal directions allowing
 * non-distance-reducing moves for one tick to escape a deadlock.
 *
 * @param position - Current position.
 * @param collisionMap - Map queried for solid cells.
 * @param cellX - Current cell X.
 * @param cellY - Current cell Y.
 * @param dtMs - Tick duration in milliseconds.
 * @returns Result with updated position and moved flag.
 */
function applyBfsStallRecovery(
  position: { x: number; y: number },
  collisionMap: CollisionMap,
  cellX: number,
  cellY: number,
  dtMs: number,
): { position: { x: number; y: number }; moved: boolean } {
  const escapeStep =
    ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND *
    (dtMs / NEATENSTEIN_MS_PER_SECOND);
  for (const [dx, dy] of DIRECTIONS) {
    if (collisionMap.isSolid(cellX + dx, cellY + dy)) {
      continue;
    }
    const escapeDesired = {
      x: position.x + dx * escapeStep,
      y: position.y + dy * escapeStep,
    };
    if (dx !== 0) {
      escapeDesired.y = cellY + dy + CELL_CENTER_OFFSET;
    } else {
      escapeDesired.x = cellX + dx + CELL_CENTER_OFFSET;
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
      return { position: escapeDesired, moved: true };
    }
    break;
  }
  return { position, moved: false };
}

// --- Flat typed-array direction data (A2 Fix 8) ---

/**
 * Flat Int32Array of dx,dy pairs for the 4 cardinal directions.
 *
 * Avoids cloning direction tuples per enemy by providing a pre-allocated
 * typed array that consumers can index directly.
 */
export const FLAT_DIRECTIONS: Int32Array = new Int32Array([
  0, -1, // N
  1, 0, // E
  0, 1, // S
  -1, 0, // W
]);

/** Reusable Int8Array index buffer for flat direction sorting (A2 Fix 8). */
const flatDirIndexBuf: Int8Array = new Int8Array(4);

/** Reusable score buffer for flat direction sorting (A2 Fix 8). */
const flatDirScoreBuf: Float64Array = new Float64Array(4);

/**
 * Compute movement using flat typed arrays for direction data.
 *
 * Mirrors the core logic of {@link computeMovement} but uses the flat
 * {@link FLAT_DIRECTIONS} Int32Array and a pooled {@link Int8Array} index
 * buffer for sorting, avoiding per-enemy direction tuple clones.
 *
 * @param ctx - Mutable pipeline context.
 */
export function computeMovementFlat(ctx: EnemyUpdateContext): void {
  const {
    shouldMoveByBfs,
    shouldMoveByFlank,
    collisionMap,
    distanceMap,
    dtMs,
    weights,
    yawRad,
    isRespawn,
    previousOrDefault,
    bfsStallTicks,
    slotTarget,
  } = ctx;

  if (!shouldMoveByBfs && !shouldMoveByFlank) return;

  let position = ctx.position;
  const cellX = Math.floor(position.x);
  const cellY = Math.floor(position.y);
  const currentDist = getDistance(distanceMap, cellX, cellY);

  // MLP re-ranking: if weights are valid and dtMs > 0, compute MLP
  // outputs from the vision vector and re-rank BFS candidates by MLP
  // score. Falls back to standard BFS sort on any error or NaN output.
  //
  // This block runs before the BFS distance early-return so that MLP
  // activation failures are logged via console.debug even when the enemy
  // occupies an unreachable cell (e.g. inside a wall).
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
      const prevStepDist = isRespawn
        ? PREVIOUS_STEP_DISTANCE_SENTINEL
        : previousOrDefault.previousStepDistance;
      const visionVector = buildVisionVector(
        distanceMap,
        cellX,
        cellY,
        prevStepDist >= 0 ? prevStepDist : undefined,
      );
      const outputs = activateMlpPooled(weights, visionVector);
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
      if (!mlpDebugLoggedThisTick) {
        console.debug('computeMovementFlat: MLP activation failed, using BFS fallback');
        mlpDebugLoggedThisTick = true;
      }
    }
  }

  if (!shouldMoveByFlank && currentDist < 0) return;

  const stepDistance =
    ENEMY_CONTROLLER_SPEED_CELLS_PER_SECOND *
    (dtMs / NEATENSTEIN_MS_PER_SECOND);

  const nudgeScale = Math.min(1, stepDistance / CELL_CENTER_OFFSET);

  position = applyPreMoveCentering(position, collisionMap, nudgeScale);

  // buffer — no direction tuple clones allocated (A2 Fix 8).
  const indices = flatDirIndexBuf;
  indices[0] = 0;
  indices[1] = 1;
  indices[2] = 2;
  indices[3] = 3;

  const scores = flatDirScoreBuf;
  for (let d = 0; d < 4; d += 1) {
    const fdx = FLAT_DIRECTIONS[d * 2];
    const fdy = FLAT_DIRECTIONS[d * 2 + 1];
    if (shouldMoveByFlank) {
      scores[d] = Math.hypot(
        position.x + fdx * stepDistance - slotTarget.x,
        position.y + fdy * stepDistance - slotTarget.y,
      );
    } else if (useMlpReRank) {
      scores[d] = fdx * mlpDesiredX + fdy * mlpDesiredY;
    } else {
      scores[d] = getDistance(distanceMap, cellX + fdx, cellY + fdy);
    }
  }

  // Insertion sort on the Int8Array index buffer (4 elements — trivial).
  // Flanking and BFS: lower score is better (ascending).
  // MLP: higher score is better (descending).
  const ascending = shouldMoveByFlank || !useMlpReRank;
  for (let i = 1; i < 4; i += 1) {
    const keyIdx = indices[i];
    const keyScore = scores[keyIdx];
    let j = i - 1;
    while (j >= 0) {
      const cmp = ascending
        ? scores[indices[j]] - keyScore
        : keyScore - scores[indices[j]];
      if (cmp > 0) {
        indices[j + 1] = indices[j];
        j -= 1;
      } else {
        break;
      }
    }
    indices[j + 1] = keyIdx;
  }

  let moved = false;

  for (let s = 0; s < 4; s += 1) {
    const d = indices[s];
    const dx = FLAT_DIRECTIONS[d * 2];
    const dy = FLAT_DIRECTIONS[d * 2 + 1];

    // In BFS mode, skip directions that don't reduce BFS distance.
    if (!shouldMoveByFlank) {
      const dist = getDistance(distanceMap, cellX + dx, cellY + dy);
      if (dist < 0 || dist >= currentDist) {
        continue;
      }
    }
    // Fast first filter: check only the target cell for solidity.
    if (collisionMap.isSolid(cellX + dx, cellY + dy)) {
      continue;
    }
    let desired = {
      x: position.x + dx * stepDistance,
      y: position.y + dy * stepDistance,
    };

    // Pre-collision centering for 1-cell gaps.
    const newCellX = cellX + dx;
    const newCellY = cellY + dy;
    if (stepDistance > 0 && dx !== 0) {
      desired = applyHorizontalCentering(
        desired,
        collisionMap,
        cellX,
        cellY,
        newCellX,
        newCellY,
      );
    } else if (stepDistance > 0) {
      /* istanbul ignore else -- cardinal directions always have exactly one of dx/dy nonzero */
      if (dy !== 0) {
        desired = applyVerticalCentering(
          desired,
          collisionMap,
          cellX,
          cellY,
          newCellX,
          newCellY,
        );
      }
    }

    // Second filter: circle-overlap wall check.
    if (
      isPositionBlockedByWall(
        collisionMap,
        desired.x,
        desired.y,
        ENEMY_CONTROLLER_WALL_COLLISION_RADIUS_CELLS,
      )
    ) {
      // Retry-after-block: snap the perpendicular coordinate to the
      // target cell center and re-check once.
      const retryDesired = { ...desired };
      if (dx !== 0) {
        retryDesired.y = newCellY + CELL_CENTER_OFFSET;
      } else {
        retryDesired.x = newCellX + CELL_CENTER_OFFSET;
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
      position = applyPostMoveCentering(
        position,
        desired,
        collisionMap,
        dx,
        dy,
      );
    }
    break;
  }

  // BFS stall-recovery fallback.
  if (shouldMoveByBfs && !shouldMoveByFlank && !moved && bfsStallTicks > 3) {
    const escapeResult = applyBfsStallRecovery(
      position,
      collisionMap,
      cellX,
      cellY,
      dtMs,
    );
    if (escapeResult.moved) {
      moved = true;
      position = escapeResult.position;
    }
  }

  // Mutate ctx.position in place (A2 Fix 8: no new {x,y} allocation).
  ctx.position.x = position.x;
  ctx.position.y = position.y;
  ctx.moved = moved;
}
