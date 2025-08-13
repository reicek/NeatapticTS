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
    /**
     * Set of characters representing walls in the maze.
     * Includes box-drawing and hash characters.
     */
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
    // Map each row and cell to its numeric encoding
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
    // Search each row for the target character
    for (let y = 0; y < asciiMaze.length; y++) {
      /**
       * Index of the character in the current row, or -1 if not found.
       */
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
    /**
     * Goal coordinates
     */
    const [gx, gy] = goal;
    // If the goal is a wall, return Infinity
    if (encodedMaze[gy][gx] === -1) return Infinity;
    /**
     * BFS queue: each entry is [[x, y], distance]
     */
    const queue: Array<[[number, number], number]> = [[start, 0]];
    /**
     * Set of visited positions (as string keys)
     */
    const visited = new Set<string>();
    /**
     * Helper to create a unique key for a position
     */
    const key = ([x, y]: [number, number]) => `${x},${y}`;
    visited.add(key(start));
    /**
     * Possible movement directions (N, E, S, W)
     */
    const directions = [
      [0, -1],
      [1, 0],
      [0, 1],
      [-1, 0],
    ];
    // BFS loop
    while (queue.length > 0) {
      const [[x, y], dist] = queue.shift()!;
      if (x === gx && y === gy) return dist;
      for (const [dx, dy] of directions) {
        const nx = x + dx;
        const ny = y + dy;
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
    /**
     * Total shortest path distance from start to exit
     */
    const totalDistance = MazeUtils.bfsDistance(encodedMaze, startPos, exitPos);
    if (totalDistance === 0) return 100;
    /**
     * Remaining shortest path distance from current position to exit
     */
    const remainingDistance = MazeUtils.bfsDistance(
      encodedMaze,
      currentPos,
      exitPos
    );
    // Calculate progress as a percentage
    return Math.min(
      100,
      Math.max(
        0,
        Math.round(((totalDistance - remainingDistance) / totalDistance) * 100)
      )
    );
  }

  /**
   * Calculates progress using a precomputed distance map (goal-centric BFS distances).
   * Faster alternative to repeated BFS calls. Distance map holds distance from each cell TO the exit (goal).
   * @param distanceMap - 2D array of distances (Infinity for walls/unreachable)
   * @param currentPos - Agent current position [x,y]
   * @param startPos - Start position [x,y]
   * @returns Progress percentage (0-100)
   */
  static calculateProgressFromDistanceMap(
    distanceMap: number[][],
    currentPos: [number, number],
    startPos: [number, number]
  ): number {
    /**
     * Start and current coordinates
     */
    const [sx, sy] = startPos;
    const [cx, cy] = currentPos;
    /**
     * Total distance from start to goal (from distance map)
     */
    const totalDistance = distanceMap[sy]?.[sx];
    /**
     * Remaining distance from current position to goal (from distance map)
     */
    const remaining = distanceMap[cy]?.[cx];
    if (
      totalDistance == null ||
      remaining == null ||
      !isFinite(totalDistance) ||
      totalDistance <= 0
    )
      return 0;
    // Calculate progress as a percentage
    const prog = ((totalDistance - remaining) / totalDistance) * 100;
    return Math.min(100, Math.max(0, Math.round(prog)));
  }

  /**
   * Builds a full distance map (Manhattan shortest path lengths via BFS) from a goal cell to every reachable cell.
   * Walls are marked as Infinity. Unreachable cells remain Infinity.
   * @param encodedMaze - 2D maze encoding
   * @param goal - [x,y] goal position (typically exit)
   */
  static buildDistanceMap(
    encodedMaze: number[][],
    goal: [number, number]
  ): number[][] {
    /**
     * Maze height and width
     */
    const height = encodedMaze.length;
    const width = encodedMaze[0].length;
    /**
     * Distance map (initialized to Infinity)
     */
    const dist: number[][] = Array.from({ length: height }, () =>
      Array(width).fill(Infinity)
    );
    /**
     * Goal coordinates
     */
    const [gx, gy] = goal;
    if (encodedMaze[gy][gx] === -1) return dist;
    /**
     * BFS queue for distance propagation
     */
    const q: Array<[number, number]> = [[gx, gy]];
    dist[gy][gx] = 0;
    /**
     * Possible movement directions (N, E, S, W)
     */
    const dirs = [
      [0, -1],
      [1, 0],
      [0, 1],
      [-1, 0],
    ];
    // BFS loop to fill distance map
    while (q.length) {
      const [x, y] = q.shift()!;
      const d = dist[y][x];
      for (const [dx, dy] of dirs) {
        const nx = x + dx;
        const ny = y + dy;
        if (
          nx >= 0 &&
          ny >= 0 &&
          ny < height &&
          nx < width &&
          encodedMaze[ny][nx] !== -1 &&
          dist[ny][nx] === Infinity
        ) {
          dist[ny][nx] = d + 1;
          q.push([nx, ny]);
        }
      }
    }
    return dist;
  }
}
