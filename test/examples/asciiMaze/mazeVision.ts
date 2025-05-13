/**
 * Maze Vision - Handles agent perception and vision-related functions
 * 
 * This module contains functions for agent perception in the maze environment,
 * implementing various sensory inputs for the neural network-controlled agent.
 * 
 * The agent's perception system is designed to mimic simplified "sensors" that provide
 * information about the environment, helping the neural network make navigation decisions.
 * 
 * Key features:
 * - Directional sensing toward exit (similar to a compass)
 * - Detection of open paths in cardinal directions (like distance sensors)
 * - Awareness of exploration progress (memory of visited locations)
 * - Reward feedback from previous actions
 */

import { manhattanDistance } from './mazeUtils';
import { MazeAgentEnhancements } from './mazeEnhancements';

// Global instance of agent enhancements
const agentEnhancements = new MazeAgentEnhancements();

/**
 * Encodes the normalized direction and distance from the agent to the exit.
 * 
 * This function acts like a "compass" for the agent, providing normalized
 * vector components pointing toward the exit, which helps the neural network
 * learn directional navigation.
 * 
 * @param agentX - X-coordinate of the agent's current position
 * @param agentY - Y-coordinate of the agent's current position
 * @param exitPos - [x,y] coordinates of the maze exit, or undefined if unknown
 * @param width - Width of the maze for normalization
 * @param height - Height of the maze for normalization
 * @param encodedMaze - Optional encoded maze for enhanced direction sensing
 * @returns Array of [dx, dy, normalizedDistance] values:
 *   - dx: X-component of direction vector to exit (-1 to 1, normalized)
 *   - dy: Y-component of direction vector to exit (-1 to 1, normalized)
 *   - normalizedDistance: Distance to exit (0 to 1, normalized)
 */
export function encodeDirectionToExit(
  agentX: number,
  agentY: number,
  exitPos: [number, number] | undefined,
  width: number,
  height: number,
  encodedMaze?: number[][]
): number[] {
  // Handle case where exit position is unknown
  if (!exitPos) {
    // Return neutral values when exit is unknown
    return [0, 0, 0];
  }
  
  const [exitX, exitY] = exitPos;
  
  // For vision priority, we want to properly balance the directional information
  // Use a dynamic scaling factor based on maze complexity (if available)
  let scaleFactor = 0.5;  
  if (encodedMaze) {
    // Get maze complexity to adjust directional influence
    const complexity = agentEnhancements.getRewardScale(encodedMaze);
    
    // In more complex mazes, directional information becomes less reliable
    // So we slightly reduce the scaling factor in complex mazes
    scaleFactor = Math.max(0.4, 0.6 - (complexity * 0.2));
  }
  
  // Calculate normalized direction components
  const dx = ((exitX - agentX) / width) * scaleFactor;
  const dy = ((exitY - agentY) / height) * scaleFactor;
  
  // Calculate distance with proper scaling for minDistanceToExit priority
  const maxDistance = width + height;
  const distance = manhattanDistance([agentX, agentY], exitPos);
  
  // Non-linear scaling for distance gives better gradient as agent approaches exit
  // Square root makes distance changes more significant when closer to exit
  const normalizedDistance = Math.sqrt(1 - (distance / maxDistance)) * scaleFactor;
  
  // If we have an encoded maze, use enhanced direction sensing
  if (encodedMaze) {
    return agentEnhancements.enhanceDirectionSense(dx, dy, normalizedDistance, encodedMaze);
  }
  
  return [dx, dy, normalizedDistance];
}

/**
 * Combines all agent perception features into a single input vector for the neural network.
 *
 * This function creates a comprehensive "sensory input" vector for the neural network,
 * simulating different types of environmental perception an agent might have.
 * 
 * Input vector layout:
 * - Direction to exit: 3 (dx, dy, normalized distance) - like a compass
 * - Percent explored: 1 - memory of how much of the maze has been seen
 * - Last reward: 1 - feedback from previous action
 * - Open distance: 4 (N, S, E, W) - distance sensors in cardinal directions
 *
 * Total: 3 + 1 + 1 + 4 = 9 inputs to the neural network
 * 
 * @param encodedMaze - 2D array representation of the maze
 * @param [agentX, agentY] - Current position of the agent
 * @param visitedPositions - Set of positions the agent has already visited
 * @param exitPos - Position of the exit/goal
 * @param lastReward - Reward from the previous action
 * @param globalProgress - Overall progress toward goal (if known)
 * @param antiRevisitSignal - Signal indicating how much the agent should avoid revisiting (0-1)
 * @returns Array of sensory inputs for the neural network
 */
export function getEnhancedVision(
  encodedMaze: number[][],
  [agentX, agentY]: [number, number],
  visitedPositions: Set<string>,
  exitPos?: [number, number],
  lastReward?: number,
  globalProgress?: number,
  antiRevisitSignal?: number
): number[] {
  const width = encodedMaze[0].length;
  const height = encodedMaze.length;
  const percentExplored = globalProgress !== undefined ? globalProgress : visitedPositions.size / (width * height);
  
  // Get adaptive vision range based on maze complexity
  const visionRange = Math.min(999, agentEnhancements.getVisionRange(encodedMaze, [agentX, agentY]));
  
  // Cardinal directions: North, South, East, West
  const dirs: [number, number][] = [[0, -1], [0, 1], [1, 0], [-1, 0]];

  /**
   * Checks if a given position is the exit
   * 
   * @param x - X coordinate to check
   * @param y - Y coordinate to check
   * @returns True if position is the exit, false otherwise
   */
  function isExit(x: number, y: number): boolean {
    if (!exitPos) return false;
    return x === exitPos[0] && y === exitPos[1];
  }

  /**
   * Scans a given direction to determine openness and path quality.
   * 
   * @param dx - X-direction to scan (-1, 0, 1)
   * @param dy - Y-direction to scan (-1, 0, 1)
   * @returns Value between 0-1 indicating path quality
   */
  function scanDirection(dx: number, dy: number): number {
    const visited = new Set<string>();
    let foundExit = false;

    // Get dynamic vision range based on maze complexity (up to the maximum 999)
    const visionRange = Math.min(999, agentEnhancements.getVisionRange(encodedMaze, [agentX, agentY]));

    /**
     * Depth-first search algorithm to find paths in a given direction
     * This is a recursive path-finding algorithm that explores as far as
     * possible along a branch before backtracking.
     * 
     * @param x - Current X-coordinate being explored
     * @param y - Current Y-coordinate being explored
     * @param px - Parent X-coordinate (where we came from)
     * @param py - Parent Y-coordinate (where we came from)
     * @param steps - Number of steps taken so far
     * @returns Value between 0-1 indicating path quality
     */
    function dfs(x: number, y: number, px: number, py: number, steps: number): number {
      // Stop if we've gone too far (limit search depth to the dynamic visionRange)
      if (steps > visionRange) return 0;
      
      const key = `${x},${y}`;
      if (visited.has(key)) return 0;
      visited.add(key);
      
      // If we found the exit, mark it and return max value
      if (isExit(x, y)) {
        foundExit = true;
        return 1;
      }
      
      // Check if this is a previously visited cell by the agent
      const isVisitedByAgent = visitedPositions.has(key);
      
      // Find all valid neighboring cells we can move to
      const open: [number, number][] = [];
      for (const [ndx, ndy] of dirs) {
        const nx = x + ndx;
        const ny = y + ndy;
        if (nx === px && ny === py) continue; // Don't go back where we came from
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue; // Stay in bounds
        if (encodedMaze[ny][nx] === -1) continue; // Can't move through walls
        open.push([ndx, ndy]);
      }
      
      // Dead end - nowhere to go
      if (open.length === 0) {
        return 0;
      }
      
      // Check for wall pattern to detect special features
      const wallPattern = agentEnhancements.detectWallPattern(encodedMaze, [x, y]);
      
      // Corridor - only one way to go, keep following it
      if (open.length === 1) {
        const [ndx, ndy] = open[0];
        // Apply small penalty for previously visited corridors to encourage exploration
        const baseValue = dfs(x + ndx, y + ndy, x, y, steps + 1);
        return isVisitedByAgent ? baseValue * 0.95 : baseValue;
      }
      
      // Junction - multiple paths to explore
      let splitValue = 0;
      for (const [ndx, ndy] of open) {
        splitValue += dfs(x + ndx, y + ndy, x, y, steps + 1);
      }
      
      // Add extra value for junctions with more options (better exploration potential)
      // This helps distinguish between simple corridors and complex junctions
      if (!foundExit && splitValue > 0) {
        // Add a subtle random factor (0-0.001) to break ties when paths look identical
        const tieBreaker = Math.random() * 0.001;
        
        // Add bonus for junctions - more exits = better exploration potential
        const junctionBonus = open.length * 0.002;
        
        // Slightly prefer unexplored junctions over revisited ones
        const explorationFactor = isVisitedByAgent ? 0.98 : 1.02;
        
        return Math.min(0.999, (open.length * 0.001 + splitValue + tieBreaker + junctionBonus) * explorationFactor);
      }
      
      return splitValue;
    }

    // Begin scan from the adjacent cell in the specified direction
    const sx = agentX + dx;
    const sy = agentY + dy;
    
    // Return 0 if we can't move in this direction (wall or edge)
    if (sx < 0 || sx >= width || sy < 0 || sy >= height) return 0;
    if (encodedMaze[sy][sx] === -1) return 0;
    
    // Check if this is a known dead end from memory
    const posKey = `${sx},${sy}`;
    const isDeadEnd = agentEnhancements.memory && 
                     agentEnhancements.memory.isDeadEnd && 
                     agentEnhancements.memory.isDeadEnd(posKey);
    
    if (isDeadEnd) {
      // Return low value for dead ends to discourage exploration
      return 0.02;
    }
    
    // Perform the DFS search
    const value = dfs(sx, sy, agentX, agentY, 1);
    
    // If we found the exit during search, return max value
    if (foundExit) return 1;
    
    // Otherwise return a value representing the "openness" in that direction
    // Add a small random factor to break ties between similar paths
    const tieBreaker = Math.random() * 0.001;
    
    // Apply enhanced scanning with memory
    const baseValue = Math.min(0.999, value + tieBreaker);
    return agentEnhancements.enhanceScan(
      baseValue, 
      dx, 
      dy, 
      agentX, 
      agentY, 
      encodedMaze
    );
  }

  // Calculate openness values in all four directions with enhanced vision
  const openDistances = dirs.map(([dx, dy]) => scanDirection(dx, dy));
  
  // Get direction information with stronger emphasis on exit sensing
  // This improves the "compass" to make the agent better at finding the exit
  const directionInfo = encodeDirectionToExit(agentX, agentY, exitPos, width, height, encodedMaze);
  
  // Create an enhanced anti-revisit signal that gets stronger with each revisit
  // This helps the agent avoid getting stuck in loops
  let enhancedAntiRevisitSignal = antiRevisitSignal;
  if (antiRevisitSignal === undefined) {
    // If no signal provided, generate one based on position history
    const posKey = `${agentX},${agentY}`;
    const revisitCount = agentEnhancements.memory?.visitCounts?.get(posKey) || 0;
    
    // Calculate exponential penalty for revisits
    enhancedAntiRevisitSignal = Math.min(1.0, revisitCount * 0.15);
  }
  
  // Blend the anti-revisit signal with the last reward
  // This makes the agent more sensitive to areas it has visited before (priority #2)
  const revisitAwareReward = (lastReward || 0) - ((enhancedAntiRevisitSignal || 0) * 0.6);
  
  // Wall pattern detection to help with navigation decisions
  // This gives the agent the ability to recognize maze patterns
  const wallPattern = agentEnhancements.detectWallPattern(encodedMaze, [agentX, agentY]);
  
  // Calculate distance-based exit gradient (priority #3)
  let distanceToExitSignal = 0;
  if (exitPos) {
    const distance = manhattanDistance([agentX, agentY], exitPos);
    const maxDistance = width + height;
    distanceToExitSignal = 1 - (distance / maxDistance);
  }
  
  // Create the complete vision vector with all sensory inputs
  const vision = [
    ...directionInfo,                 // Direction to exit (3) - Priority #1
    percentExplored,                  // Percentage of maze explored (1)
    revisitAwareReward,               // Reward from previous step with anti-revisit (1) - Priority #2
    ...openDistances                  // Openness in cardinal directions (4) - Part of vision priority #1
  ];
  
  // Record vision for temporal processing
  agentEnhancements.recordVision(vision);
  
  // Return the final vision vector
  return vision;
}