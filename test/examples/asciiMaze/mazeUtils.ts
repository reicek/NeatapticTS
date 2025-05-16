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
 * Computes the shortest path distance between two points in the maze using BFS.
 * Returns Infinity if no path exists.
 *
 * This metric is used for:
 * - Measuring proximity to the exit
 * - Evaluating path efficiency
 * - Calculating progress
 *
 * @param encodedMaze - 2D array representation of the maze
 * @param start - [x, y] start position
 * @param goal - [x, y] goal position
 * @returns Shortest path length (number of steps), or Infinity if unreachable
 */
export function bfsDistance(encodedMaze: number[][], start: [number, number], goal: [number, number]): number {
  const [gx, gy] = goal;
  if (encodedMaze[gy][gx] === -1) return Infinity;
  const queue: Array<[[number, number], number]> = [[start, 0]];
  const visited = new Set<string>();
  const key = ([x, y]: [number, number]) => `${x},${y}`;
  visited.add(key(start));
  const directions = [
    [0, -1], // North
    [1, 0],  // East
    [0, 1],  // South
    [-1, 0], // West
  ];
  while (queue.length > 0) {
    const [[x, y], dist] = queue.shift()!;
    if (x === gx && y === gy) return dist;
    for (const [dx, dy] of directions) {
      const nx = x + dx, ny = y + dy;
      if (
        nx >= 0 && ny >= 0 &&
        ny < encodedMaze.length && nx < encodedMaze[0].length &&
        encodedMaze[ny][nx] !== -1 &&
        !visited.has(key([nx, ny]))
      ) {
        visited.add(key([nx, ny]));
        queue.push([[nx, ny], dist + 1]);
      }
    }
  }
  return Infinity;
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
 * - totalDistance: BFS distance from start to exit
 * - remainingDistance: BFS distance from current position to exit
 * 
 * This creates a percentage scale where:
 * - 0% means the agent is at the start or has moved away from the exit
 * - 100% means the agent has reached the exit
 * - Values in between represent partial progress toward the goal
 *
 * @param encodedMaze - 2D array representation of the maze
 * @param currentPos - [x, y] current agent position
 * @param startPos - [x, y] start position
 * @param exitPos - [x, y] exit position
 * @returns Progress percentage (0-100)
 */
export function calculateProgress(encodedMaze: number[][], currentPos: [number, number], startPos: [number, number], exitPos: [number, number]): number {
  const totalDistance = bfsDistance(encodedMaze, startPos, exitPos);
  if (totalDistance === 0) return 100; // Handle case where start and exit positions are the same
  const remainingDistance = bfsDistance(encodedMaze, currentPos, exitPos);
  // Clamp result between 0 and 100 for safety
  return Math.min(100, Math.max(0, Math.round(((totalDistance - remainingDistance) / totalDistance) * 100)));
}
