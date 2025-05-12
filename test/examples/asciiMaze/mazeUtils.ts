/**
 * Utility functions for maze logic and encoding.
 * Provides helpers for encoding ASCII mazes, finding positions, and calculating distances/progress.
 * @module mazeUtils
 */

/**
 * Converts an ASCII maze (array of strings) into a 2D numeric array for processing by the agent.
 * Encoding:
 *   '#' = -1 (wall)
 *   '.' = 0 (open path)
 *   'E' = 1 (exit/goal)
 *   'S' = 2 (start)
 *   any other character = 0 (open path)
 *
 * @param asciiMaze - Array of strings representing the maze
 * @returns 2D array of numbers encoding the maze
 */
export function encodeMaze(asciiMaze: string[]): number[][] {
  // Map each row and cell to its numeric encoding
  return asciiMaze.map(row => 
    [...row].map(cell => {
      switch(cell) {
        case '#': return -1; // Wall
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
 * @param asciiMaze - Array of strings representing the maze
 * @param char - Character to find (e.g., 'S' for start, 'E' for exit)
 * @returns [x, y] coordinates of the character
 * @throws If the character is not found in the maze
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
 * Manhattan distance is the sum of the absolute differences of their coordinates.
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
 * Progress is defined as:
 *   ((totalDistance - remainingDistance) / totalDistance) * 100
 * Where totalDistance is the Manhattan distance from start to exit,
 * and remainingDistance is the Manhattan distance from current position to exit.
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
