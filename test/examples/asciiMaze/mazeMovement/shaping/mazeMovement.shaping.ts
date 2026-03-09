/**
 * Reward-shaping helpers for the dedicated mazeMovement module.
 *
 * This file owns movement reward shaping, stagnation penalties, entropy-guided
 * bonuses, and other post-action fitness adjustments.
 */

import { MAZE_MOVEMENT_CONSTANTS } from '../mazeMovement.constants';
import {
  getMazeMovementRunServiceState,
  requireMazeMovementBufferPools,
} from '../mazeMovement.services';
import type { SimulationState } from '../mazeMovement.types';
import { sumVisionGroup } from '../mazeMovement.utils';
import {
  getMazeMovementDistance,
  isMazeMovementCellOpen,
} from '../runtime/mazeMovement.runtime';
import { MazeUtils } from '../../mazeUtils';

const C = MAZE_MOVEMENT_CONSTANTS;
const SHAPING_VISION_SUM_SCRATCH = new Float64Array(4);

/**
 * Execute the chosen move and apply the shaping terms tied to that move.
 *
 * @param state - Mutable simulation state for the active run.
 * @param encodedMaze - Maze grid used for move validity and distance lookup.
 * @param distanceMap - Optional precomputed distance map.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function executeMazeMovementAndRewards(
  state: SimulationState,
  encodedMaze: number[][],
  distanceMap: number[][] | undefined,
  coordinateScratch: Int32Array,
): void {
  if (state.earlyTerminate) return;

  const previousDistance = getMazeMovementDistance(
    encodedMaze,
    state.position,
    distanceMap,
  );
  state.prevDistance = previousDistance;
  state.moved = false;

  const chosenAction = state.direction;
  if (chosenAction >= 0 && chosenAction < C.ACTION_DIM) {
    const [deltaX, deltaY] = C.DIRECTION_DELTAS[chosenAction];
    const candidateX = (state.position[0] + deltaX) | 0;
    const candidateY = (state.position[1] + deltaY) | 0;
    coordinateScratch[0] = candidateX;
    coordinateScratch[1] = candidateY;

    if (
      isMazeMovementCellOpen(
        encodedMaze,
        candidateX,
        candidateY,
        coordinateScratch,
      )
    ) {
      state.position[0] = candidateX;
      state.position[1] = candidateY;
      state.moved = true;
    }
  }

  const rewardScale = C.REWARD_SCALE;
  const bufferPools = requireMazeMovementBufferPools();
  if (state.moved) {
    const writeIndex = state.pathLength | 0;
    bufferPools.pathX[writeIndex] = state.position[0];
    bufferPools.pathY[writeIndex] = state.position[1];
    state.pathLength = writeIndex + 1;

    MazeUtils.pushHistory(
      state.recentPositions,
      [state.position[0], state.position[1]],
      C.LOCAL_WINDOW,
    );

    applyMazeMovementLocalAreaPenalty(state, rewardScale, coordinateScratch);

    const currentDistance = state.hasDistanceMap
      ? (state.distanceMap?.[state.position[1]]?.[state.position[0]] ??
        Infinity)
      : getMazeMovementDistance(encodedMaze, state.position, state.distanceMap);
    const distanceDelta = previousDistance - currentDistance;
    const improved = distanceDelta > 0;
    const worsened = !improved && currentDistance > previousDistance;
    applyMazeMovementProgressShaping(
      state,
      distanceDelta,
      improved,
      worsened,
      rewardScale,
    );
    applyMazeMovementExplorationVisitAdjustment(
      state,
      rewardScale,
      coordinateScratch,
    );

    if (state.direction >= 0) state.directionCounts[state.direction]++;
    state.minDistanceToExit = Math.min(
      state.minDistanceToExit,
      currentDistance,
    );
  } else {
    state.invalidMovePenalty -= C.INVALID_MOVE_PENALTY_MILD * rewardScale;
  }

  applyMazeMovementGlobalDistanceImprovementBonus(
    state,
    encodedMaze,
    rewardScale,
    coordinateScratch,
  );
}

/**
 * Apply the post-action shaping and penalty aggregation phase.
 *
 * @param state - Mutable simulation state for the active run.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function applyMazeMovementPostActionPenalties(
  state: SimulationState,
  coordinateScratch: Int32Array,
): void {
  if (state.earlyTerminate) return;

  const rewardScale = C.REWARD_SCALE;
  applyMazeMovementRepetitionAndBacktrackPenalties(
    state,
    rewardScale,
    coordinateScratch,
  );
  if (state.moved) state.prevAction = state.direction;
  applyMazeMovementEntropyGuidanceShaping(
    state,
    rewardScale,
    coordinateScratch,
  );
  applyMazeMovementSaturationPenaltyCycle(
    state,
    rewardScale,
    coordinateScratch,
  );

  coordinateScratch[0] =
    (state.loopPenalty || 0) +
    (state.memoryPenalty || 0) +
    (state.revisitPenalty || 0);
  state.invalidMovePenalty += coordinateScratch[0];
}

/**
 * Apply a local-area stagnation penalty when the run oscillates in a tight window.
 *
 * @param state - Mutable simulation state for the active run.
 * @param rewardScale - Global reward scale used for the penalty magnitude.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function applyMazeMovementLocalAreaPenalty(
  state: SimulationState,
  rewardScale: number,
  coordinateScratch: Int32Array,
): void {
  const recentWindow = state.recentPositions;
  if (recentWindow.length !== C.LOCAL_WINDOW) return;

  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (let recentIndex = 0; recentIndex < recentWindow.length; recentIndex++) {
    const [positionX, positionY] = recentWindow[recentIndex];
    const coercedX = positionX | 0;
    const coercedY = positionY | 0;
    if (coercedX < minX) minX = coercedX;
    if (coercedX > maxX) maxX = coercedX;
    if (coercedY < minY) minY = coercedY;
    if (coercedY > maxY) maxY = coercedY;
  }

  coordinateScratch[0] = minX;
  coordinateScratch[1] = minY;
  const span = maxX - minX + (maxY - minY);
  if (
    span <= C.LOCAL_AREA_SPAN_THRESHOLD &&
    state.stepsSinceImprovement > C.LOCAL_AREA_STAGNATION_STEPS
  ) {
    state.localAreaPenalty -= C.LOCAL_AREA_PENALTY_AMOUNT * rewardScale;
  }
}

/**
 * Apply progress and away-from-goal shaping after a move.
 *
 * @param state - Mutable simulation state for the active run.
 * @param distanceDelta - Positive when the agent moved closer to the goal.
 * @param improved - True when the move improved distance to the goal.
 * @param worsened - True when the move increased distance to the goal.
 * @param rewardScale - Global reward scale used by the shaping terms.
 */
export function applyMazeMovementProgressShaping(
  state: SimulationState,
  distanceDelta: number,
  improved: boolean,
  worsened: boolean,
  rewardScale: number,
): void {
  const currentConfidence = state.actionStats?.maxProb ?? (improved ? 1 : 0.5);
  if (improved) {
    const confidenceScaledBase =
      (C.PROGRESS_REWARD_BASE +
        C.PROGRESS_REWARD_CONF_SCALE * currentConfidence) *
      rewardScale;
    if (state.stepsSinceImprovement > 0) {
      const stepBonus = Math.min(
        state.stepsSinceImprovement * C.PROGRESS_STEPS_MULT * rewardScale,
        C.PROGRESS_STEPS_MAX * rewardScale,
      );
      state.progressReward += stepBonus;
    }
    state.progressReward += confidenceScaledBase;
    state.stepsSinceImprovement = 0;
    const distanceContribution =
      distanceDelta *
      C.DISTANCE_DELTA_SCALE *
      (C.DISTANCE_DELTA_CONF_BASE +
        C.DISTANCE_DELTA_CONF_SCALE * currentConfidence);
    state.progressReward += distanceContribution;
    return;
  }

  if (worsened) {
    const awayPenalty =
      (C.PROGRESS_AWAY_BASE_PENALTY +
        C.PROGRESS_AWAY_CONF_SCALE * currentConfidence) *
      rewardScale;
    state.progressReward -= awayPenalty;
    state.stepsSinceImprovement++;
    return;
  }

  state.stepsSinceImprovement++;
}

/**
 * Apply the per-cell exploration bonus or revisit penalty.
 *
 * @param state - Mutable simulation state for the active run.
 * @param rewardScale - Global reward scale used by the adjustment.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function applyMazeMovementExplorationVisitAdjustment(
  state: SimulationState,
  rewardScale: number,
  coordinateScratch: Int32Array,
): void {
  const visitsAtThisCell = state.visitsAtCurrent | 0;
  const positiveBonus = C.NEW_CELL_EXPLORATION_BONUS * rewardScale;
  const revisitPenalty = C.REVISIT_PENALTY_STRONG * rewardScale;
  coordinateScratch[0] =
    visitsAtThisCell === 1 ? positiveBonus : -revisitPenalty;
  state.newCellExplorationBonus += coordinateScratch[0];
}

/**
 * Apply the long-horizon global-distance improvement bonus.
 *
 * @param state - Mutable simulation state for the active run.
 * @param encodedMaze - Maze grid used for distance lookup.
 * @param rewardScale - Global reward scale used by the bonus magnitude.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function applyMazeMovementGlobalDistanceImprovementBonus(
  state: SimulationState,
  encodedMaze: number[][],
  rewardScale: number,
  coordinateScratch: Int32Array,
): void {
  const positionX = state.position[0] | 0;
  const positionY = state.position[1] | 0;
  const currentGlobalDistance = state.hasDistanceMap
    ? (state.distanceMap?.[positionY]?.[positionX] ?? Infinity)
    : getMazeMovementDistance(encodedMaze, state.position, state.distanceMap);
  coordinateScratch[0] = currentGlobalDistance as number;

  const previousGlobalDistance = state.lastDistanceGlobal ?? Infinity;
  if (currentGlobalDistance < previousGlobalDistance) {
    const stagnationSteps = state.stepsSinceImprovement | 0;
    if (stagnationSteps > C.GLOBAL_BREAK_BONUS_START) {
      const bonusSteps = stagnationSteps - C.GLOBAL_BREAK_BONUS_START;
      const uncappedBonus =
        bonusSteps * C.GLOBAL_BREAK_BONUS_PER_STEP * rewardScale;
      const cappedBonus = Math.min(
        uncappedBonus,
        C.GLOBAL_BREAK_BONUS_CAP * rewardScale,
      );
      state.progressReward += cappedBonus;
    }
    state.stepsSinceImprovement = 0;
  }

  state.lastDistanceGlobal = currentGlobalDistance;
}

/**
 * Apply repetition and immediate-backtrack penalties.
 *
 * @param state - Mutable simulation state for the active run.
 * @param rewardScale - Global reward scale used by the penalties.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function applyMazeMovementRepetitionAndBacktrackPenalties(
  state: SimulationState,
  rewardScale: number,
  coordinateScratch: Int32Array,
): void {
  if (state.earlyTerminate) return;

  const previousAction = state.prevAction;
  const currentAction = state.direction;
  const stagnationSteps = state.stepsSinceImprovement | 0;

  if (
    previousAction === currentAction &&
    stagnationSteps > C.REPETITION_PENALTY_START
  ) {
    const repetitionMultiplier = stagnationSteps - C.REPETITION_PENALTY_START;
    const computedRepetitionPenalty =
      C.REPETITION_PENALTY_BASE * repetitionMultiplier * rewardScale;
    coordinateScratch[0] = -computedRepetitionPenalty;
    state.invalidMovePenalty += coordinateScratch[0];
  }

  if (
    previousAction >= 0 &&
    currentAction >= 0 &&
    stagnationSteps > 0 &&
    currentAction === C.OPPOSITE_DIR[previousAction]
  ) {
    coordinateScratch[1] = -C.BACK_MOVE_PENALTY * rewardScale;
    state.invalidMovePenalty += coordinateScratch[1];
  }
}

/**
 * Apply entropy-guided shaping based on confidence and perceptual guidance.
 *
 * @param state - Mutable simulation state for the active run.
 * @param rewardScale - Global reward scale used by the penalties and bonuses.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function applyMazeMovementEntropyGuidanceShaping(
  state: SimulationState,
  rewardScale: number,
  coordinateScratch: Int32Array,
): void {
  if (state.earlyTerminate || !state.actionStats) return;

  const { entropy, maxProb, secondProb } = state.actionStats;
  const hasLineOfSightGuidance =
    sumVisionGroup(
      state.vision,
      C.VISION_LOS_START,
      C.VISION_GROUP_LEN,
      SHAPING_VISION_SUM_SCRATCH,
    ) > 0;
  const hasGradientGuidance =
    sumVisionGroup(
      state.vision,
      C.VISION_GRAD_START,
      C.VISION_GROUP_LEN,
      SHAPING_VISION_SUM_SCRATCH,
    ) > 0;

  if (entropy > C.ENTROPY_HIGH_THRESHOLD) {
    coordinateScratch[0] = -C.ENTROPY_PENALTY * rewardScale;
    state.invalidMovePenalty += coordinateScratch[0];
    return;
  }

  const maxMinusSecond = (maxProb ?? 0) - (secondProb ?? 0);
  if (
    (hasLineOfSightGuidance || hasGradientGuidance) &&
    entropy < C.ENTROPY_CONFIDENT_THRESHOLD &&
    maxMinusSecond > C.ENTROPY_CONFIDENT_DIFF
  ) {
    coordinateScratch[0] = C.EXPLORATION_BONUS_SMALL * rewardScale;
    state.newCellExplorationBonus += coordinateScratch[0];
  }
}

/**
 * Apply the periodic saturation penalty cycle.
 *
 * @param state - Mutable simulation state for the active run.
 * @param rewardScale - Global reward scale used by the penalties.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function applyMazeMovementSaturationPenaltyCycle(
  state: SimulationState,
  rewardScale: number,
  coordinateScratch: Int32Array,
): void {
  const saturations = getMazeMovementRunServiceState().saturations;
  if (saturations < C.SATURATION_PENALTY_TRIGGER) return;

  coordinateScratch[0] = -C.SATURATION_PENALTY_BASE * rewardScale;
  state.invalidMovePenalty += coordinateScratch[0];

  const period = C.SATURATION_PENALTY_PERIOD;
  if (period > 0 && saturations % period === 0) {
    coordinateScratch[1] = -C.SATURATION_PENALTY_ESCALATE * rewardScale;
    state.invalidMovePenalty += coordinateScratch[1];
  }
}

/**
 * Apply the deep-stagnation termination penalty when appropriate.
 *
 * @param state - Mutable simulation state for the active run.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 * @returns True when the run should terminate early.
 */
export function maybeTerminateMazeMovementDeepStagnation(
  state: SimulationState,
  coordinateScratch: Int32Array,
): boolean {
  const stagnationSteps = state.stepsSinceImprovement | 0;
  if (stagnationSteps <= C.DEEP_STAGNATION_THRESHOLD) {
    return state.earlyTerminate;
  }

  const rewardScale = C.REWARD_SCALE;
  try {
    const runningOutsideBrowser = typeof window === 'undefined';
    if (runningOutsideBrowser) {
      coordinateScratch[0] = -C.DEEP_STAGNATION_PENALTY * rewardScale;
      state.invalidMovePenalty += coordinateScratch[0];
      return true;
    }
  } catch {
    coordinateScratch[0] = -C.DEEP_STAGNATION_PENALTY * rewardScale;
    state.invalidMovePenalty += coordinateScratch[0];
    return true;
  }

  return state.earlyTerminate;
}

export {};
