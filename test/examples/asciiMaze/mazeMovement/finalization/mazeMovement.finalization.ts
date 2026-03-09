/**
 * Result-finalization helpers for the dedicated mazeMovement module.
 *
 * These helpers assemble the final simulation payload once the orchestration
 * facade has finished stepping the run. Keeping this logic here lets the main
 * facade stay focused on the episode loop rather than on score shaping math.
 */

import { MazeUtils } from '../../mazeUtils';
import { MAZE_MOVEMENT_CONSTANTS } from '../mazeMovement.constants';
import {
  materializeMazeMovementPath,
  randomMazeMovementUnit,
} from '../mazeMovement.services';
import type {
  MazeMovementSimulationResult,
  SimulationState,
} from '../mazeMovement.types';
import { computeActionEntropyFromCounts } from '../mazeMovement.utils';

const C = MAZE_MOVEMENT_CONSTANTS;
const ACTION_ENTROPY_SCRATCH = new Float64Array(1);

/**
 * Build the finalized payload for a successful maze run.
 *
 * @param state - Completed simulation state for the successful run.
 * @param maxSteps - Maximum allowed step budget for the run.
 * @returns Success result with fitness, path, and diagnostic summaries.
 */
export function finalizeSuccessfulMazeMovementRun(
  state: SimulationState,
  maxSteps: number,
): MazeMovementSimulationResult {
  const stepsTaken = state.steps | 0;
  const stepEfficiency = (maxSteps | 0) - stepsTaken;
  const actionEntropy = computeMazeMovementActionEntropy(state.directionCounts);
  const baseFitness =
    C.SUCCESS_BASE_FITNESS +
    stepEfficiency * C.STEP_EFFICIENCY_SCALE +
    state.progressReward +
    state.newCellExplorationBonus +
    state.invalidMovePenalty;
  const totalFitness =
    baseFitness + actionEntropy * C.SUCCESS_ACTION_ENTROPY_SCALE;
  const pathSnapshot = materializeMazeMovementPath(state.pathLength);
  const saturationFraction =
    stepsTaken > 0 ? state.saturatedSteps / stepsTaken : 0;

  return {
    success: true,
    steps: stepsTaken,
    path: pathSnapshot,
    fitness: Math.max(C.MIN_SUCCESS_FITNESS, totalFitness),
    progress: 100,
    saturationFraction,
    actionEntropy,
  };
}

/**
 * Build the finalized payload for a failed maze run.
 *
 * @param state - Completed simulation state for the failed run.
 * @param encodedMaze - Maze grid used to compute fallback geometric progress.
 * @param startPos - Start coordinate for the current episode.
 * @param exitPos - Exit coordinate for the current episode.
 * @param distanceMap - Optional precomputed distance map aligned to the maze.
 * @returns Failure result with shaped fitness, path, and diagnostic summaries.
 */
export function finalizeFailedMazeMovementRun(
  state: SimulationState,
  encodedMaze: number[][],
  startPos: readonly [number, number],
  exitPos: readonly [number, number],
  distanceMap?: number[][],
): MazeMovementSimulationResult {
  const progress = distanceMap
    ? MazeUtils.calculateProgressFromDistanceMap(
        distanceMap,
        state.position,
        startPos,
      )
    : MazeUtils.calculateProgress(
        encodedMaze,
        state.position,
        startPos,
        exitPos,
      );
  const progressFraction = progress / 100;
  const shapedProgress =
    Math.pow(progressFraction, C.PROGRESS_POWER) * C.PROGRESS_SCALE;
  const explorationScore = state.visitedUniqueCount;
  const actionEntropy = computeMazeMovementActionEntropy(state.directionCounts);
  const entropyBonus = actionEntropy * C.ENTROPY_BONUS_WEIGHT;
  const baseFitness =
    shapedProgress +
    explorationScore +
    state.progressReward +
    state.newCellExplorationBonus +
    state.invalidMovePenalty +
    entropyBonus +
    state.localAreaPenalty;
  const randomizedFitness =
    baseFitness + randomMazeMovementUnit() * C.FITNESS_RANDOMNESS;
  const stabilizedFitness =
    randomizedFitness >= 0
      ? randomizedFitness
      : -Math.log1p(1 - randomizedFitness);
  const pathSnapshot = materializeMazeMovementPath(state.pathLength);
  const stepsTaken = state.steps | 0;
  const saturationFraction =
    stepsTaken > 0 ? state.saturatedSteps / stepsTaken : 0;

  return {
    success: false,
    steps: stepsTaken,
    path: pathSnapshot,
    fitness: stabilizedFitness,
    progress,
    saturationFraction,
    actionEntropy,
  };
}

/**
 * Compute the normalized action-entropy summary for a finished run.
 *
 * @param directionCounts - Per-direction action counts recorded during the run.
 * @returns Normalized entropy in the range `[0, 1]`.
 */
function computeMazeMovementActionEntropy(directionCounts: number[]): number {
  return computeActionEntropyFromCounts(
    directionCounts,
    C.LOG_ACTIONS,
    ACTION_ENTROPY_SCRATCH,
  );
}
