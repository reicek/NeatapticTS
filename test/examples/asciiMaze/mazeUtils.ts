/**
 * Utility class for maze logic and encoding.
 * Provides static methods for encoding, position finding, BFS, and progress calculation.
 */
export class MazeUtils {
  /**
   * Converts an ASCII/Unicode maze (array of strings) into a 2D numeric array for processing by the agent.
   *
   * Encoding:
   *   '#' = -1 (wall/obstacle)
   *   Box drawing characters (═,║,╔,╗,╚,╝,╠,╣,╦,╩,╬) = -1 (wall/obstacle)
   *   '.' = 0 (open path)
   *   'E' = 1 (exit/goal)
   *   'S' = 2 (start position)
   *   any other character = 0 (treated as open path)
   *
   * @param asciiMaze - Array of strings representing the maze.
   * @returns 2D array of numbers encoding the maze elements.
   */
  static encodeMaze(asciiMaze: string[]): number[][] {
    const wallChars = new Set([
      '#',
      '═',
      '║',
      '╔',
      '╗',
      '╚',
      '╝',
      '╠',
      '╣',
      '╦',
      '╩',
      '╬',
    ]);
    return asciiMaze.map((row) =>
      [...row].map((cell) => {
        if (wallChars.has(cell)) return -1;
        switch (cell) {
          case '.':
            return 0;
          case 'E':
            return 1;
          case 'S':
            return 2;
          default:
            return 0;
        }
      })
    );
  }

  /**
   * Finds the (x, y) position of a given character in the ASCII maze.
   * @param asciiMaze - Array of strings representing the maze.
   * @param char - Character to find (e.g., 'S' for start, 'E' for exit).
   * @returns [x, y] coordinates of the character.
   * @throws Error if the character is not found in the maze.
   */
  static findPosition(asciiMaze: string[], char: string): [number, number] {
    for (let y = 0; y < asciiMaze.length; y++) {
      const x = asciiMaze[y].indexOf(char);
      if (x !== -1) return [x, y];
    }
    throw new Error(`Character ${char} not found in maze`);
  }

  /**
   * Computes the shortest path distance between two points in the maze using BFS.
   * Returns Infinity if no path exists.
   * @param encodedMaze - 2D array representation of the maze.
   * @param start - [x, y] start position.
   * @param goal - [x, y] goal position.
   * @returns Shortest path length (number of steps), or Infinity if unreachable.
   */
  static bfsDistance(
    encodedMaze: number[][],
    start: [number, number],
    goal: [number, number]
  ): number {
    const [gx, gy] = goal;
    if (encodedMaze[gy][gx] === -1) return Infinity;
    const queue: Array<[[number, number], number]> = [[start, 0]];
    const visited = new Set<string>();
    const key = ([x, y]: [number, number]) => `${x},${y}`;
    visited.add(key(start));
    const directions = [
      [0, -1],
      [1, 0],
      [0, 1],
      [-1, 0],
    ];
    while (queue.length > 0) {
      const [[x, y], dist] = queue.shift()!;
      if (x === gx && y === gy) return dist;
      for (const [dx, dy] of directions) {
        const nx = x + dx,
          ny = y + dy;
        if (
          nx >= 0 &&
          ny >= 0 &&
          ny < encodedMaze.length &&
          nx < encodedMaze[0].length &&
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
   * Progress is measured as the proportion of the shortest path covered from start to exit.
   * @param encodedMaze - 2D array representation of the maze.
   * @param currentPos - [x, y] current agent position.
   * @param startPos - [x, y] start position.
   * @param exitPos - [x, y] exit position.
   * @returns Progress percentage (0-100).
   */
  static calculateProgress(
    encodedMaze: number[][],
    currentPos: [number, number],
    startPos: [number, number],
    exitPos: [number, number]
  ): number {
    const totalDistance = MazeUtils.bfsDistance(encodedMaze, startPos, exitPos);
    if (totalDistance === 0) return 100;
    const remainingDistance = MazeUtils.bfsDistance(
      encodedMaze,
      currentPos,
      exitPos
    );
    return Math.min(
      100,
      Math.max(
        0,
        Math.round(((totalDistance - remainingDistance) / totalDistance) * 100)
      )
    );
  }
}
