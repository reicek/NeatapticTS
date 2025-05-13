/**
 * Maze Movement - Handles agent movement and simulation logic
 * 
 * This module contains functions for agent movement and simulation in the maze environment,
 * including path finding, reward calculations, and the reinforcement learning incentive system.
 * 
 * The agent movement system demonstrates:
 * - Decision making based on neural network outputs
 * - Reward and penalty calculations for reinforcement learning
 * - Various heuristics to encourage exploration and goal-seeking behavior
 * - Simulation of movement with collision detection
 */

import { Network } from '../../../src/neataptic';
import { manhattanDistance, calculateProgress } from './mazeUtils';
import { getEnhancedVision } from './mazeVision';
import { MazeAgentEnhancements } from './mazeEnhancements';

// Create a global enhancement controller
const enhancements = new MazeAgentEnhancements();

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
  const [x, y] = position;
  // Direction vectors: North, East, South, West
  const moves = [ [0, -1], [1, 0], [0, 1], [-1, 0] ];
  
  // Validate and normalize the direction input, defaulting to random if invalid
  const validDirection = Number.isInteger(direction) && direction >= 0 && direction < moves.length 
    ? direction 
    : Math.floor(Math.random() * moves.length);
  
  // Get the direction vector for the chosen direction
  const [dx, dy] = moves[validDirection];
  const newX = x + dx;
  const newY = y + dy;
  
  // Return new position if valid, otherwise return original position
  return isValidMove(encodedMaze, [newX, newY]) ? [newX, newY] : [x, y];
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
  // Apply momentum to enhance continuation in the same direction
  const enhancedOutputs = enhancements.applyMomentum(outputs);
  
  // Find the index with the maximum value
  let maxVal = -Infinity, maxIdx = 0;
  for (let i = 0; i < enhancedOutputs.length; i++) {
    if (!isNaN(enhancedOutputs[i]) && enhancedOutputs[i] > maxVal) {
      maxVal = enhancedOutputs[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

/**
 * Detects which side of the agent is next to a wall (if any).
 * 
 * Used for wall-hugging detection in the reward system to discourage
 * behaviors like following walls without exploring.
 * 
 * @param encodedMaze - 2D array representation of the maze
 * @param position - Current [x,y] position of the agent
 * @returns Direction character ('L','R','U','D') if wall present, or null if no adjacent walls
 */
function detectWallSide(encodedMaze: number[][], position: [number, number]): string | null {
  const [x, y] = position;
  if (!isValidMove(encodedMaze, [x - 1, y])) return 'L';
  if (!isValidMove(encodedMaze, [x + 1, y])) return 'R';
  if (!isValidMove(encodedMaze, [x, y - 1])) return 'U';
  if (!isValidMove(encodedMaze, [x, y + 1])) return 'D';
  return null;
}

/**
 * Simulates the agent navigating the maze using its neural network.
 * 
 * This function runs a complete simulation of an agent traversing a maze,
 * using its neural network for decision making. The simulation includes:
 * 
 * 1. Sensory input gathering through vision system
 * 2. Neural network activation to produce movement decisions
 * 3. Position updates with collision detection
 * 4. Comprehensive reward calculation system
 * 5. Tracking of various agent behaviors
 * 
 * The reward system uses multiple heuristics to encourage:
 * - Moving closer to the exit
 * - Exploration of new cells
 * - Efficient route finding
 * - Avoiding repetitive behaviors (oscillation, wall-hugging)
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
  let revisitCount = 0;
  let minDistanceToExit = manhattanDistance(position, exitPos);
  let invalidMoves = 0;
  
  // Increase max vision range to 999 steps as requested
  const MAX_VISION_RANGE = 999;
  
  // Scale factor to balance vision vs. reinforcement learning incentives
  // Lower value makes the agent rely more on visual information than rewards
  const NON_VISION_SCALE = 0.015; // Reduced from 0.02 to prioritize vision more
  
  // Initialize reward/penalty tracking variables
  let progressReward = 0;
  let consecutiveStays = 0;
  let wallHugStreak = 0;
  let lastWallSide: string | null = null;
  let noProgressSteps = 0;
  let bestDistance = manhattanDistance(position, exitPos);
  let revisitPenalty = 0;
  let ditheringPenalty = 0;
  let noProgressPenalty = 0;
  let revisitMap = new Map<string, number>();
  let oscillationPenalty = 0;
  let lastReward = 0;
  let newCellExplorationBonus = 0;
  let stuckPenaltyFactor = 1;
  
  // Apply dynamic scaling for rewards based on maze complexity
  const dynamicRewardScale = enhancements.getRewardScale(encodedMaze) * NON_VISION_SCALE;

  // Main simulation loop
  while (steps < maxSteps) {
    steps++;
    
    // --- PERCEPTION: Gather sensory inputs for the neural network ---
    const percentExplored = visitedPositions.size / (encodedMaze.length * encodedMaze[0].length);
    
    // Get anti-revisit signal from enhancements - increased weight
    const antiRevisitSignal = enhancements.getAntiRevisitSignal(position) * 1.25;
    
    // Request adaptive vision range up to MAX_VISION_RANGE
    const visionRange = Math.min(MAX_VISION_RANGE, enhancements.getVisionRange(encodedMaze, position));
    
    const vision = getEnhancedVision(
      encodedMaze,
      position,
      visitedPositions,
      exitPos,
      lastReward,
      percentExplored,
      antiRevisitSignal
    );
    
    // Validate input vector size
    const expectedInputSize = 3 + 1 + 1 + 4; // Direction to exit (3) + Explored % + Last reward + Open paths (4)
    if (vision.length !== expectedInputSize) {
      throw new Error(`simulateAgent: Input vector length is ${vision.length}, expected ${expectedInputSize}`);
    }
    
    // --- DECISION MAKING: Use neural network to choose movement direction ---
    const outputs = network.activate(vision);
    
    // Apply stronger anti-revisit bias to the outputs (priority #2: already visited)
    const enhancedOutputs = enhancements.applyAntiRevisitBias(outputs, position, encodedMaze);
    
    let direction = (enhancedOutputs && enhancedOutputs.length > 0) 
      ? selectDirection(enhancedOutputs)
      : Math.floor(Math.random() * 4);
    
    // Save previous state for comparison
    const prevPosition = [...position] as [number, number];
    const prevDistance = manhattanDistance(position, exitPos);
    
    lastReward = 0; // Reset reward for current step's outcome evaluation

    // --- ACTION: Move the agent based on the network's decision ---
    position = moveAgent(encodedMaze, position, direction);
    const moved = prevPosition[0] !== position[0] || prevPosition[1] !== position[1];
    const currentPosKey = `${position[0]},${position[1]}`;

    // --- REWARD SYSTEM: Calculate rewards/penalties based on agent's actions ---

    if (moved) { // Agent successfully moved to a new position
      path.push(position.slice() as [number, number]);
      consecutiveStays = 0;
      const currentDistance = manhattanDistance(position, exitPos);

      // Reward for discovering new cells (exploration incentive)
      if (!visitedPositions.has(currentPosKey)) {
        // Strong incentive for exploring new cells
        newCellExplorationBonus += 0.6 * dynamicRewardScale * stuckPenaltyFactor;
        lastReward = Math.max(lastReward, 0.3 * dynamicRewardScale);
        stuckPenaltyFactor = 1;
      }

      // Reward/penalty based on distance change to exit (priority #3: minDistanceToExit)
      if (currentDistance < prevDistance) {
        // Increased reward for moving closer to exit
        progressReward += 0.3 * dynamicRewardScale;
        lastReward = Math.max(lastReward, 0.15 * dynamicRewardScale);
        noProgressSteps = 0;
      } else if (currentDistance > prevDistance) {
        // Small penalty for moving away from exit
        progressReward -= 0.04 * dynamicRewardScale;
        lastReward = Math.min(lastReward, -0.03 * dynamicRewardScale);
        noProgressSteps++;
      } else {
        // Neutral for lateral movement (same distance)
        noProgressSteps += 0.5; // Half-count for lateral movement
      }

      // Track best position reached
      if (currentDistance < bestDistance) {
        bestDistance = currentDistance;
        noProgressSteps = 0;
        lastReward = Math.max(lastReward, 0.1 * dynamicRewardScale);
      }

      // Oscillation Detection with enhanced system
      const enhancedOscillationPenalty = enhancements.getOscillationPenalty();
      if (enhancedOscillationPenalty < 0) {
        oscillationPenalty += enhancedOscillationPenalty * dynamicRewardScale * 1.5;
        lastReward = Math.min(lastReward, -1.5 * dynamicRewardScale);
      }

      // Wall-Hugging Detection
      const currentWallSide = detectWallSide(encodedMaze, position);
      if (currentWallSide && currentWallSide === lastWallSide) {
        wallHugStreak++;
      } else {
        wallHugStreak = 0;
      }
      lastWallSide = currentWallSide;
      
      // Penalty for wall hugging
      if (wallHugStreak >= 8) {
        progressReward -= 2.5 * dynamicRewardScale;
        lastReward = Math.min(lastReward, -0.8 * dynamicRewardScale);
      }

    } else { // Agent did not move (invalid move or "dithering")
      invalidMoves++;
      consecutiveStays++;
      
      // Stronger penalty for repeatedly failing to move
      if (consecutiveStays >= 3) {
        const stayPenalty = Math.pow(1.3, consecutiveStays) * dynamicRewardScale;
        ditheringPenalty -= stayPenalty;
        lastReward = Math.min(lastReward, -0.5 * dynamicRewardScale * consecutiveStays);
      } else {
        ditheringPenalty -= 0.7 * dynamicRewardScale;
        lastReward = Math.min(lastReward, -0.15 * dynamicRewardScale);
      }
      noProgressSteps++;
      
      // Slowly increase stuck penalty factor
      stuckPenaltyFactor *= 1.04;
    }

    // No Progress Penalty
    if (noProgressSteps >= 8) {
      noProgressPenalty -= noProgressSteps * 0.12 * dynamicRewardScale;
      lastReward = Math.min(lastReward, -0.15 * noProgressSteps * dynamicRewardScale);
    }

    // Revisit Penalties (priority #2: already visited)
    if (moved) {
      if (visitedPositions.has(currentPosKey)) {
        revisitCount++;
        const cellRevisits = revisitMap.get(currentPosKey) || 0;
        
        // Stronger exponential penalty for revisits
        const revisitFactor = Math.pow(1.25, Math.min(6, cellRevisits));
        revisitPenalty -= 0.35 * revisitFactor * dynamicRewardScale;
        lastReward = Math.min(lastReward, -0.2 * revisitFactor * dynamicRewardScale);
      }
      visitedPositions.add(currentPosKey);
      revisitMap.set(currentPosKey, (revisitMap.get(currentPosKey) || 0) + 1);
    }

    // Track closest approach to exit (priority #3: minDistanceToExit)
    const currentDistanceToExit = manhattanDistance(position, exitPos);
    minDistanceToExit = Math.min(minDistanceToExit, currentDistanceToExit);

    // Distance-based incentive - subtle reward for being closer to exit
    progressReward -= 0.015 * currentDistanceToExit * dynamicRewardScale;
    
    // Record position and direction for enhanced behaviors
    enhancements.recordPosition(position, direction, encodedMaze);

    // More balanced early termination conditions
    // Only terminate early if highly stuck or highly inefficient
    if ((revisitCount > 150 && visitedPositions.size < encodedMaze.length * encodedMaze[0].length * 0.3) || 
        noProgressSteps > 120) {
      
      // Try finding a backtrack target
      const backtrackTarget = enhancements.findBacktrackTarget(position, path, encodedMaze);
      if (backtrackTarget) {
        if (backtrackTarget[0] !== position[0] || backtrackTarget[1] !== position[1]) {
          const targetKey = `${backtrackTarget[0]},${backtrackTarget[1]}`;
          // Reduce penalties for revisiting the backtrack target
          revisitMap.set(targetKey, Math.max(0, (revisitMap.get(targetKey) || 0) - 5));
        }
      } else {
        // If no backtrack target found, terminate
        break;
      }
    }

    // --- SUCCESS CHECK: Has the agent reached the exit? ---
    if (position[0] === exitPos[0] && position[1] === exitPos[1]) {
      // Calculate fitness with efficiency bonuses
      const efficiencyBonus = Math.max(0, 75 - (steps * 0.04) * dynamicRewardScale);
      const stepPenalty = steps > (minDistanceToExit * 1.2) 
        ? (steps - minDistanceToExit * 1.2) * 0.08 * dynamicRewardScale 
        : 0;
        
      // Final fitness calculation for successful run
      let fitness = 500 + 
        efficiencyBonus -
        stepPenalty +
        (maxSteps - steps) * 0.25 * dynamicRewardScale +
        progressReward +
        revisitPenalty +
        ditheringPenalty +
        noProgressPenalty +
        oscillationPenalty +
        (visitedPositions.size * 0.6 * dynamicRewardScale);
        
      fitness = Math.max(150, fitness);
        
      const progress = 100;
      return { success: true, steps, path, fitness, progress };
    }
  }
  
  // --- FAILURE HANDLING: If exit not reached within max steps ---
  const progress = calculateProgress(path[path.length - 1], startPos, exitPos);
  
  // Calculate fitness for unsuccessful attempt
  // Base score primarily on progress toward exit and exploration
  const baseScore = progress * 2.0 + (visitedPositions.size * 0.15);
  
  // Final fitness combining base score with all rewards/penalties
  const fitness = baseScore + 
    progressReward + 
    revisitPenalty + 
    ditheringPenalty + 
    noProgressPenalty + 
    oscillationPenalty + 
    newCellExplorationBonus;
    
  return { success: false, steps, path, fitness, progress };
}