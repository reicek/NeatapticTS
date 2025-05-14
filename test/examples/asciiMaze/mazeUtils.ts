/**
 * Utility functions for maze logic and encoding.
 * 
 * This module provides essential helpers for working with ASCII mazes, including:
 * - Converting between ASCII and numeric representations
 * - Finding specific positions in the maze
 * - Calculating distances and measuring progress
 * 
 * These utilities form the foundation of the maze navigation system,
 * enabling both the agent's perception and the evaluation of its performance.
 * 
 * @module mazeUtils
 */

/**
 * Converts an ASCII/Unicode maze (array of strings) into a 2D numeric array for processing by the agent.
 * 
 * This function translates the human-readable representation into a numeric format
 * that's more suitable for neural network processing. The encoding follows this scheme:
 * 
 * Encoding:
 *   '#' = -1 (wall/obstacle)
 *   Box drawing characters (═,║,╔,╗,╚,╝,╠,╣,╦,╩,╬) = -1 (wall/obstacle)
 *   '.' = 0 (open path)
 *   'E' = 1 (exit/goal)
 *   'S' = 2 (start position)
 *   any other character = 0 (treated as open path)
 *
 * @param asciiMaze - Array of strings representing the maze
 * @returns 2D array of numbers encoding the maze elements
 */
export function encodeMaze(asciiMaze: string[]): number[][] {
  // Unicode box drawing characters that should be treated as walls
  const wallChars = new Set(['#', '═', '║', '╔', '╗', '╚', '╝', '╠', '╣', '╦', '╩', '╬']);
  
  // Map each row and cell to its numeric encoding
  return asciiMaze.map(row => 
    [...row].map(cell => {
      // Check for wall characters first
      if (wallChars.has(cell)) return -1; // Wall
      
      // Check for special cells
      switch(cell) {
        case '.': return 0;  // Open path
        case 'E': return 1;  // Exit
        case 'S': return 2;  // Start
        default: return 0;   // Treat unknown as open path
      }
    })
  );
}

/**
 * Finds the (x, y) position of a given character in the ASCII maze.
 * 
 * This function searches through the entire maze to locate a specific character,
 * such as the start ('S') or exit ('E') position. It returns the coordinates
 * as an [x, y] tuple where x is the column and y is the row.
 *
 * @param asciiMaze - Array of strings representing the maze
 * @param char - Character to find (e.g., 'S' for start, 'E' for exit)
 * @returns [x, y] coordinates of the character
 * @throws Error if the character is not found in the maze
 */
export function findPosition(asciiMaze: string[], char: string): [number, number] {
  for (let y = 0; y < asciiMaze.length; y++) {
    const x = asciiMaze[y].indexOf(char);
    if (x !== -1) return [x, y]; // Return as soon as found
  }
  throw new Error(`Character ${char} not found in maze`);
}

/**
 * Calculates the Manhattan distance between two points in the maze.
 * 
 * Manhattan distance (or "taxicab geometry") is the sum of the absolute differences
 * of their Cartesian coordinates. In the context of a grid-based maze where
 * movement is restricted to horizontal and vertical directions, this represents
 * the minimum number of steps required to move from one point to another.
 *
 * This metric is used for:
 * - Measuring proximity to the exit
 * - Evaluating path efficiency
 * - Calculating progress
 *
 * @param a - [x1, y1] first position
 * @param b - [x2, y2] second position
 * @returns Manhattan distance between the two points
 */
export function manhattanDistance([x1, y1]: [number, number], [x2, y2]: [number, number]): number {
  return Math.abs(x1 - x2) + Math.abs(y1 - y2);
}

/**
 * Calculates the agent's progress toward the exit as a percentage.
 * 
 * This function estimates how much progress the agent has made by comparing
 * its current position relative to both the start and exit positions.
 * The formula used is:
 *   ((totalDistance - remainingDistance) / totalDistance) * 100
 * 
 * Where:
 * - totalDistance: Manhattan distance from start to exit
 * - remainingDistance: Manhattan distance from current position to exit
 * 
 * This creates a percentage scale where:
 * - 0% means the agent is at the start or has moved away from the exit
 * - 100% means the agent has reached the exit
 * - Values in between represent partial progress toward the goal
 *
 * @param currentPos - [x, y] current agent position
 * @param startPos - [x, y] start position
 * @param exitPos - [x, y] exit position
 * @returns Progress percentage (0-100)
 */
export function calculateProgress(currentPos: [number, number], startPos: [number, number], exitPos: [number, number]): number {
  const totalDistance = manhattanDistance(startPos, exitPos);
  if (totalDistance === 0) return 100; // Handle case where start and exit positions are the same
  
  const remainingDistance = manhattanDistance(currentPos, exitPos);
  
  // Clamp result between 0 and 100 for safety
  return Math.min(100, Math.max(0, Math.round(((totalDistance - remainingDistance) / totalDistance) * 100)));
}
