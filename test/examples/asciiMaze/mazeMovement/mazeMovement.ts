/**
 * Public MazeMovement facade for the dedicated mazeMovement module boundary.
 *
 * The folder now owns runtime helpers, policy helpers, shaping helpers, and
 * finalization logic. This file keeps the user-facing API in one place while
 * the implementation stays split into focused helpers.
 */

import type { INetwork } from '../interfaces';
import type {
  DirectionSelectionStats,
  MazeMovementSimulationResult,
  SimulationState,
} from './mazeMovement.types';
import { MAZE_MOVEMENT_CONSTANTS } from './mazeMovement.constants';
import {
  decideMazeMovementDirection,
  selectMazeMovementDirection,
  applyMazeMovementEpsilonExploration,
  applyMazeMovementForcedExploration,
  applyMazeMovementProximityGreedy,
} from './policy/mazeMovement.policy';
import {
  applyMazeMovementPostActionPenalties,
  executeMazeMovementAndRewards,
  maybeTerminateMazeMovementDeepStagnation,
} from './shaping/mazeMovement.shaping';
import {
  finalizeFailedMazeMovementRun,
  finalizeSuccessfulMazeMovementRun,
} from './finalization/mazeMovement.finalization';
import {
  buildMazeMovementVisionAndDistance,
  createMazeMovementRunState,
  isMazeMovementCellOpen,
  recordMazeMovementVisitAndPenalties,
} from './runtime/mazeMovement.runtime';

const C = MAZE_MOVEMENT_CONSTANTS;

/**
 * Maze movement entry surface used by fitness evaluation and evolution runs.
 *
 * The public API intentionally remains class-based so existing example imports
 * do not change while the implementation lives under the dedicated module.
 */
export class MazeMovement {
  /** Reused integer coordinate scratch for hot-path movement helpers. */
  static #COORDINATE_SCRATCH = new Int32Array(2);

  /**
   * Determine whether a candidate target cell is inside bounds and not a wall.
   *
   * @param encodedMaze - Maze grid to inspect.
   * @param position - Candidate `[x, y]` position.
   * @returns True when the target cell can be entered.
   */
  static isValidMove(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    position: readonly [number, number],
  ): boolean;

  /**
   * Determine whether a candidate target cell is inside bounds and not a wall.
   *
   * @param encodedMaze - Maze grid to inspect.
   * @param x - Candidate maze column.
   * @param y - Candidate maze row.
   * @returns True when the target cell can be entered.
   */
  static isValidMove(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    x: number,
    y: number,
  ): boolean;

  static isValidMove(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    positionOrX: readonly [number, number] | number,
    yMaybe?: number,
  ): boolean {
    if (typeof positionOrX === 'number') {
      const candidateColumn = positionOrX | 0;
      const candidateRow = (yMaybe ?? 0) | 0;
      return isMazeMovementCellOpen(
        encodedMaze,
        candidateColumn,
        candidateRow,
        MazeMovement.#COORDINATE_SCRATCH,
      );
    }

    if (!Array.isArray(positionOrX) || positionOrX.length !== 2) return false;

    const [candidateColumn, candidateRow] = positionOrX;
    return isMazeMovementCellOpen(
      encodedMaze,
      candidateColumn,
      candidateRow,
      MazeMovement.#COORDINATE_SCRATCH,
    );
  }

  /**
   * Move the agent one step in the requested direction when the target cell is open.
   *
   * @param encodedMaze - Maze grid used for collision checks.
   * @param position - Current agent position.
   * @param direction - Direction index in the action space.
   * @returns New position when the move is valid, otherwise the original position.
   * @example
   * const moved = MazeMovement.moveAgent(encodedMaze, [3, 2], 1);
   */
  static moveAgent(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    position: readonly [number, number],
    direction: number,
  ): [number, number] {
    if (direction === C.NO_MOVE) {
      return [position[0], position[1]];
    }

    const nextPosition: [number, number] = [position[0], position[1]];
    if (direction >= 0 && direction < C.ACTION_DIM) {
      const [deltaColumn, deltaRow] = C.DIRECTION_DELTAS[direction];
      nextPosition[0] += deltaColumn;
      nextPosition[1] += deltaRow;
    }

    return MazeMovement.isValidMove(encodedMaze, nextPosition)
      ? nextPosition
      : [position[0], position[1]];
  }

  /**
   * Convert raw network outputs into a chosen action plus diagnostics.
   *
   * @param outputs - Raw action logits for the four maze directions.
   * @returns Chosen direction plus softmax and entropy diagnostics.
   * @example
   * const stats = MazeMovement.selectDirection([0.2, 1.4, -0.1, 0]);
   */
  static selectDirection(outputs: number[]): DirectionSelectionStats {
    return selectMazeMovementDirection(outputs);
  }

  /**
   * Simulate one full maze episode for a network-controlled agent.
   *
   * @param network - Policy network used to choose actions.
   * @param encodedMaze - Maze grid for the active episode.
   * @param startPos - Start coordinate.
   * @param exitPos - Exit coordinate.
   * @param distanceMap - Optional precomputed distance map.
   * @param maxSteps - Maximum allowed step count before termination.
   * @returns Final simulation result including path, fitness, and progress.
   */
  static simulateAgent(
    network: INetwork,
    encodedMaze: number[][],
    startPos: readonly [number, number],
    exitPos: readonly [number, number],
    distanceMap?: number[][],
    maxSteps = C.DEFAULT_MAX_STEPS,
  ): MazeMovementSimulationResult {
    const simulationState = createMazeMovementRunState(
      encodedMaze,
      startPos,
      distanceMap,
      maxSteps,
    );

    while (simulationState.steps < maxSteps) {
      simulationState.steps++;

      // Step 1: refresh visit bookkeeping, perception, and policy choice.
      MazeMovement.#processPerceptionAndPolicy(
        simulationState,
        network,
        encodedMaze,
        exitPos,
        distanceMap,
      );

      // Step 2: execute the chosen action, apply shaping, and stop if needed.
      const shouldStop = MazeMovement.#processMovementAndShaping(
        simulationState,
        encodedMaze,
        distanceMap,
      );
      if (shouldStop) break;

      // Step 3: finalize immediately when the exit is reached.
      if (MazeMovement.#hasReachedExit(simulationState, exitPos)) {
        return finalizeSuccessfulMazeMovementRun(simulationState, maxSteps);
      }
    }

    return finalizeFailedMazeMovementRun(
      simulationState,
      encodedMaze,
      startPos,
      exitPos,
      distanceMap,
    );
  }

  /**
   * Refresh visit bookkeeping, perception state, and direction policy.
   *
   * @param simulationState - Mutable run state for the active episode.
   * @param network - Policy network used for action selection.
   * @param encodedMaze - Maze grid used for the run.
   * @param exitPos - Exit coordinate for the run.
   * @param distanceMap - Optional precomputed distance map.
   */
  static #processPerceptionAndPolicy(
    simulationState: SimulationState,
    network: INetwork,
    encodedMaze: number[][],
    exitPos: readonly [number, number],
    distanceMap?: number[][],
  ): void {
    recordMazeMovementVisitAndPenalties(simulationState);
    buildMazeMovementVisionAndDistance(
      simulationState,
      encodedMaze,
      exitPos,
      distanceMap,
    );
    decideMazeMovementDirection(
      simulationState,
      network,
      MazeMovement.#COORDINATE_SCRATCH,
    );
    applyMazeMovementProximityGreedy(
      simulationState,
      encodedMaze,
      distanceMap,
      MazeMovement.#COORDINATE_SCRATCH,
    );
    applyMazeMovementEpsilonExploration(
      simulationState,
      encodedMaze,
      MazeMovement.#COORDINATE_SCRATCH,
    );
    applyMazeMovementForcedExploration(
      simulationState,
      encodedMaze,
      MazeMovement.#COORDINATE_SCRATCH,
    );
  }

  /**
   * Execute the selected move, apply post-action shaping, and evaluate stop rules.
   *
   * @param simulationState - Mutable run state for the active episode.
   * @param encodedMaze - Maze grid used for movement and distance lookup.
   * @param distanceMap - Optional precomputed distance map.
   * @returns True when the episode should stop after this step.
   */
  static #processMovementAndShaping(
    simulationState: SimulationState,
    encodedMaze: number[][],
    distanceMap?: number[][],
  ): boolean {
    executeMazeMovementAndRewards(
      simulationState,
      encodedMaze,
      distanceMap,
      MazeMovement.#COORDINATE_SCRATCH,
    );
    applyMazeMovementPostActionPenalties(
      simulationState,
      MazeMovement.#COORDINATE_SCRATCH,
    );
    return maybeTerminateMazeMovementDeepStagnation(
      simulationState,
      MazeMovement.#COORDINATE_SCRATCH,
    );
  }

  /**
   * Determine whether the current state has reached the maze exit.
   *
   * @param simulationState - Mutable run state for the active episode.
   * @param exitPos - Exit coordinate for the run.
   * @returns True when the agent position matches the exit coordinate.
   */
  static #hasReachedExit(
    simulationState: SimulationState,
    exitPos: readonly [number, number],
  ): boolean {
    return (
      simulationState.position[0] === exitPos[0] &&
      simulationState.position[1] === exitPos[1]
    );
  }
}
