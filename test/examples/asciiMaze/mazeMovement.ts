/**
 * Maze Movement - Handles agent movement and simulation logic (Simplified)
 *
 * This module contains functions for agent movement and simulation in the maze environment,
 * focusing on simple navigation based primarily on neural network decisions.
 *
 * The agent movement system demonstrates:
 * - Decision making based on neural network outputs
 * - Basic reward calculations for reinforcement learning
 * - Simple goal-seeking behavior
 * - Simulation of movement with collision detection
 */

import { INetwork } from './interfaces';
import { MazeUtils } from './mazeUtils';
import { MazeVision } from './mazeVision';

/**
 * MazeMovement provides static methods for agent movement and simulation.
 */
export class MazeMovement {
  /**
   * Checks if a move is valid (within bounds and not a wall).
   *
   * @param encodedMaze - 2D array representation of the maze.
   * @param [x, y] - Coordinates to check.
   * @returns Boolean indicating if the position is valid for movement.
   */
  static isValidMove(encodedMaze: number[][], [x, y]: [number, number]): boolean {
    return x >= 0 &&
           y >= 0 &&
           y < encodedMaze.length &&
           x < encodedMaze[0].length &&
           encodedMaze[y][x] !== -1;
  }

  /**
   * Moves the agent in the given direction if possible, otherwise stays in place.
   *
   * Handles collision detection with walls and maze boundaries,
   * preventing the agent from making invalid moves.
   *
   * @param encodedMaze - 2D array representation of the maze.
   * @param position - Current [x,y] position of the agent.
   * @param direction - Direction index (0=North, 1=East, 2=South, 3=West).
   * @returns New position after movement, or original position if move was invalid.
   */
  static moveAgent(encodedMaze: number[][], position: [number, number], direction: number): [number, number] {
    // Don't move if direction === -1
    if (direction === -1) {
      return [...position] as [number, number];
    }
    const nextPosition: [number, number] = [...position] as [number, number];
    switch (direction) {
      case 0: // North
        nextPosition[1] -= 1;
        break;
      case 1: // East
        nextPosition[0] += 1;
        break;
      case 2: // South
        nextPosition[1] += 1;
        break;
      case 3: // West
        nextPosition[0] -= 1;
        break;
    }
    if (MazeMovement.isValidMove(encodedMaze, nextPosition)) {
      return nextPosition;
    } else {
      return position;
    }
  }

  /**
   * Selects the direction with the highest output value from the neural network.
   * Applies softmax to interpret outputs as probabilities, then uses argmax.
   *
   * @param outputs - Array of output values from the neural network (length 4).
   * @returns Index of the highest output value (0=N, 1=E, 2=S, 3=W), or -1 for no movement.
   */
  static selectDirection(outputs: number[]): number {
    if (!outputs || outputs.length !== 4) {
      return -1;
    }
    // Apply softmax for numerical stability
    const max = Math.max(...outputs);
    const exps = outputs.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    const softmax = exps.map(e => e / sum);
    // Find the index of the highest probability
    let maxVal = -Infinity, maxIdx = -1;
    for (let i = 0; i < softmax.length; i++) {
      if (!isNaN(softmax[i]) && softmax[i] > maxVal) {
        maxVal = softmax[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  /**
   * Simulates the agent navigating the maze using its neural network.
   *
   * Runs a complete simulation of an agent traversing a maze,
   * using its neural network for decision making. This implementation focuses
   * on a minimalist approach, putting more responsibility on the neural network.
   *
   * @param network - Neural network controlling the agent.
   * @param encodedMaze - 2D array representation of the maze.
   * @param startPos - Starting position [x,y] of the agent.
   * @param exitPos - Exit/goal position [x,y] of the maze.
   * @param maxSteps - Maximum steps allowed before terminating (default 3000).
   * @returns Object containing:
   *   - success: Boolean indicating if exit was reached.
   *   - steps: Number of steps taken.
   *   - path: Array of positions visited.
   *   - fitness: Calculated fitness score for evolution.
   *   - progress: Percentage progress toward exit (0-100).
   */
  static simulateAgent(
    network: INetwork,
    encodedMaze: number[][],
    startPos: [number, number],
    exitPos: [number, number],
    maxSteps = 3000
  ): {
    success: boolean;
    steps: number;
    path: [number, number][];
    fitness: number;
    progress: number;
  } {
    // Initialize agent state and tracking variables
    let position = [...startPos] as [number, number];
    let steps = 0;
    let path = [position.slice() as [number, number]];
    let visitedPositions = new Set<string>();
    let visitCounts = new Map<string, number>(); // Track number of visits per cell
    let moveHistory: string[] = []; // Short-term memory for last N positions
    const MOVE_HISTORY_LENGTH = 6; // Tune as needed
    let minDistanceToExit = MazeUtils.bfsDistance(encodedMaze, position, exitPos);

    // Reward scaling factor for all reward/penalty calculations
    const rewardScale = 0.5;

    // Initialize reward tracking variables
    let progressReward = 0;
    let newCellExplorationBonus = 0;
    let invalidMovePenalty = 0;

    // Main simulation loop: agent moves until maxSteps or exit is reached
    while (steps < maxSteps) {
      steps++;

      // Record current position as visited
      const currentPosKey = `${position[0]},${position[1]}`;
      visitedPositions.add(currentPosKey);
      visitCounts.set(currentPosKey, (visitCounts.get(currentPosKey) || 0) + 1);
      moveHistory.push(currentPosKey);
      if (moveHistory.length > MOVE_HISTORY_LENGTH) moveHistory.shift();

      // Calculate percent of maze explored so far
      const percentExplored = visitedPositions.size / (encodedMaze.length * encodedMaze[0].length);

      // Oscillation/loop detection: if the last 4 moves form a 2-step loop (A->B->A->B)
      let loopPenalty = 0;
      if (
        moveHistory.length >= 4 &&
        moveHistory[moveHistory.length - 1] === moveHistory[moveHistory.length - 3] &&
        moveHistory[moveHistory.length - 2] === moveHistory[moveHistory.length - 4]
      ) {
        loopPenalty -= 10 * rewardScale; // Strong penalty for 2-step loop
      }
      // Penalty for returning to any cell in recent history (short-term memory)
      let memoryPenalty = 0;
      if (
        moveHistory.length > 1 &&
        moveHistory.slice(0, -1).includes(currentPosKey)
      ) {
        memoryPenalty -= 2 * rewardScale;
      }
      // Dynamic penalty for multiple visits
      let revisitPenalty = 0;
      const visits = visitCounts.get(currentPosKey) || 1;
      if (visits > 1) {
        revisitPenalty -= (0.2 * (visits - 1)) * rewardScale; // Penalty increases with each revisit
      }
      // Early termination if a cell is visited too many times (hard oscillation)
      if (visits > 10) {
        // Penalize and break out of the loop
        invalidMovePenalty -= 1000 * rewardScale;
        break;
      }

      // --- PERCEPTION: Get vision inputs using enhanced vision system
      const vision = MazeVision.getEnhancedVision(
        encodedMaze,
        position,
        visitedPositions,
        exitPos,
        percentExplored,
        maxSteps // Pass maxSteps as visionRange
      );

      // --- DECISION MAKING: Let the neural network decide
      let direction;
      try {
        const outputs = network.activate(vision);
        direction = MazeMovement.selectDirection(outputs);
      } catch (error) {
        console.error("Error activating network:", error);
        direction = -1; // Fallback: don't move
      }

      // Save previous state for reward calculation
      const prevPosition = [...position] as [number, number];
      const prevDistance = MazeUtils.bfsDistance(encodedMaze, position, exitPos);

      // --- ACTION: Move based on network decision
      position = MazeMovement.moveAgent(encodedMaze, position, direction);
      const moved = prevPosition[0] !== position[0] || prevPosition[1] !== position[1];

      // Record movement and update rewards/penalties
      if (moved) {
        path.push(position.slice() as [number, number]);

        // Calculate current distance to exit
        const currentDistance = MazeUtils.bfsDistance(encodedMaze, position, exitPos);

        // Reward for getting closer to exit, penalty for moving away
        if (currentDistance < prevDistance) {
          progressReward += 0.5 * rewardScale;
        } else if (currentDistance > prevDistance) {
          progressReward -= 0.05 * rewardScale;
        }

        // Bonus for exploring new cells, penalty for revisiting
        if (visits === 1) {
          newCellExplorationBonus += 0.3 * rewardScale;
        } else {
          newCellExplorationBonus -= 0.5 * rewardScale; // Stronger penalty for revisiting
        }

        // Track closest approach to exit
        minDistanceToExit = Math.min(minDistanceToExit, currentDistance);
      } else {
        // Penalty for invalid move (collision or out of bounds)
        invalidMovePenalty -= 1000 * rewardScale;
        // No tolerance for invalid moves; break if needed
        steps === maxSteps;
      }

      // Apply oscillation/loop/memory/revisit penalties
      invalidMovePenalty += loopPenalty + memoryPenalty + revisitPenalty;

      // --- SUCCESS CHECK: Exit reached
      if (position[0] === exitPos[0] && position[1] === exitPos[1]) {
        // Calculate fitness for successful completion
        const stepEfficiency = maxSteps - steps;
        const fitness = 500 + // Base success reward
          stepEfficiency * 0.2 + // Reward for efficiency
          progressReward +
          newCellExplorationBonus +
          invalidMovePenalty;

        return {
          success: true,
          steps,
          path,
          fitness: Math.max(150, fitness),
          progress: 100
        };
      }
    }

    // --- FAILURE CASE: Did not reach exit
    const progress = MazeUtils.calculateProgress(encodedMaze, path[path.length - 1], startPos, exitPos);

    // Fitness for unsuccessful attempts: focus on progress and exploration
    const fitness = progress * 2.0 +
      progressReward +
      newCellExplorationBonus +
      invalidMovePenalty +
      (visitedPositions.size * 0.1);

    return { success: false, steps, path, fitness, progress };
  }
}
