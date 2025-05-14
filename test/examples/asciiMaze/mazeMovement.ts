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

import { Network } from '../../../src/neataptic';
import { manhattanDistance, calculateProgress } from './mazeUtils';
import { getEnhancedVision } from './mazeVision';

/**
 * Checks if a move is valid (within bounds and not a wall).
 * 
 * @param encodedMaze - 2D array representation of the maze
 * @param [x, y] - Coordinates to check
 * @returns Boolean indicating if the position is valid for movement
 */
export function isValidMove(encodedMaze: number[][], [x, y]: [number, number]): boolean {
  return x >= 0 && 
         y >= 0 && 
         y < encodedMaze.length && 
         x < encodedMaze[0].length && 
         encodedMaze[y][x] !== -1;
}

/**
 * Moves the agent in the given direction if possible, otherwise stays in place.
 * 
 * This function handles collision detection with walls and maze boundaries,
 * preventing the agent from making invalid moves.
 * 
 * @param encodedMaze - 2D array representation of the maze
 * @param position - Current [x,y] position of the agent
 * @param direction - Direction index (0=North, 1=East, 2=South, 3=West)
 * @returns New position after movement, or original position if move was invalid
 */
export function moveAgent(encodedMaze: number[][], position: [number, number], direction: number): [number, number] {
  // Don't move if direction is -1
  if (direction === -1) {
    return [...position] as [number, number];
  }

  // Initialize next position as current position
  const nextPosition: [number, number] = [...position] as [number, number];
  
  // Update next position based on direction
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
  
  // Check if the move is valid
  if (isValidMove(encodedMaze, nextPosition)) {
    return nextPosition;
  } else {
    return position; // Return original position if the move is invalid
  }
}

/**
 * Selects the direction with the highest output value from the neural network.
 * 
 * This implements the "winner takes all" approach for action selection from 
 * neural network outputs, converting continuous values to discrete actions.
 * 
 * @param outputs - Array of output values from the neural network
 * @returns Index of the highest output value (0-3 for N/E/S/W)
 */
export function selectDirection(outputs: number[]): number {
  // Basic safety check - make sure we have valid outputs
  if (!outputs || outputs.length === 0) {
    return -1; // Return -1 to indicate "don't move" instead of random choice
  }

  // Find the index with the maximum value
  let maxVal = -Infinity, maxIdx = -1;
  for (let i = 0; i < outputs.length; i++) {
    if (!isNaN(outputs[i]) && outputs[i] > maxVal) {
      maxVal = outputs[i];
      maxIdx = i;
    }
  }
  
  // If all values were NaN or below activation threshold, don't move
  if (maxVal === -Infinity || maxVal < 0.2) {
    return -1; // Don't move if network doesn't have a strong preference
  }
  
  return maxIdx;
}

/**
 * Simulates the agent navigating the maze using its neural network.
 * 
 * This function runs a complete simulation of an agent traversing a maze,
 * using its neural network for decision making. This implementation focuses
 * on minimalist approach, putting more responsibility on the neural network.
 *
 * @param network - Neural network controlling the agent
 * @param encodedMaze - 2D array representation of the maze
 * @param startPos - Starting position [x,y] of the agent
 * @param exitPos - Exit/goal position [x,y] of the maze
 * @param maxSteps - Maximum steps allowed before terminating (default 3000)
 * @returns Object containing:
 *   - success: Boolean indicating if exit was reached
 *   - steps: Number of steps taken
 *   - path: Array of positions visited
 *   - fitness: Calculated fitness score for evolution
 *   - progress: Percentage progress toward exit (0-100)
 */
export function simulateAgent(
  network: Network,
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
  // Initialize agent state
  let position = [...startPos] as [number, number];
  let steps = 0;
  let path = [position.slice() as [number, number]];
  let visitedPositions = new Set<string>();
  let minDistanceToExit = manhattanDistance(position, exitPos);
  
  // Basic reward scale - simplified
  const rewardScale = 0.5;
  
  // Initialize reward tracking
  let progressReward = 0;
  let newCellExplorationBonus = 0;
  let invalidMovePenalty = 0;

  // Main simulation loop
  while (steps < maxSteps) {
    steps++;
    
    // Record position
    const currentPosKey = `${position[0]},${position[1]}`;
    visitedPositions.add(currentPosKey);
    
    // Calculate percent explored
    const percentExplored = visitedPositions.size / (encodedMaze.length * encodedMaze[0].length);
    
    // --- PERCEPTION: Get vision inputs using DFS-based system
    const vision = getEnhancedVision(
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
      direction = selectDirection(outputs);
    } catch (error) {
      console.error("Error activating network:", error);
      direction = Math.floor(Math.random() * 4); // Random fallback
    }
    
    // Save previous state
    const prevPosition = [...position] as [number, number];
    const prevDistance = manhattanDistance(position, exitPos);
    
    // --- ACTION: Move based on network decision
    position = moveAgent(encodedMaze, position, direction);
    const moved = prevPosition[0] !== position[0] || prevPosition[1] !== position[1];
    
    // Record movement
    if (moved) {
      path.push(position.slice() as [number, number]);
      
      // Calculate current distance to exit
      const currentDistance = manhattanDistance(position, exitPos);
      
      // Simple reward for getting closer to exit
      if (currentDistance < prevDistance) {
        progressReward += 0.5 * rewardScale;
      } else if (currentDistance > prevDistance) {
        progressReward -= 0.05 * rewardScale;
      }
      
      // Small exploration bonus
      if (!visitedPositions.has(`${position[0]},${position[1]}`)) {
        newCellExplorationBonus += 0.3 * rewardScale;
      }
      
      // Update closest approach
      minDistanceToExit = Math.min(minDistanceToExit, currentDistance);
    } else {
      invalidMovePenalty -= 1000 * rewardScale; // Penalty for invalid move
      // No tolerance for invalid moves
      steps === maxSteps;
    }

    // --- SUCCESS CHECK
    if (position[0] === exitPos[0] && position[1] === exitPos[1]) {
      // Success fitness calculation - simplified
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
  
  // --- FAILURE CASE
  const progress = calculateProgress(path[path.length - 1], startPos, exitPos);
  
  // Simple fitness for unsuccessful attempts - focus on progress toward exit
  const fitness = progress * 2.0 + 
    progressReward + 
    newCellExplorationBonus + 
    invalidMovePenalty + 
    (visitedPositions.size * 0.1);
    
  return { success: false, steps, path, fitness, progress };
}
