/**
 * Utility class for maze logic and encoding.
 * Provides static methods for encoding, position finding, BFS, and progress calculation.
 */
export class MazeUtils {
  /** Shared set of wall characters to avoid re-allocating on every call */
  // Private static set of wall characters (box-drawing + hash).
  // Kept private (#) because it's an implementation detail; use `encodeMaze`.
  static #WALL_CHARS = new Set([
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

  // Movement vectors used by BFS (N, E, S, W). Private to avoid accidental external use.
  // Marked readonly to signal these vectors are constant and must not be mutated.
  static #DIRECTIONS: readonly (readonly [number, number])[] = [
    [0, -1],
    [1, 0],
    [0, 1],
    [-1, 0],
  ];

  /** Convert [x,y] into canonical key string 'x,y' */
  static posKey([x, y]: readonly [number, number]): string {
    return `${x},${y}`;
  }
  /**
   * Return up to the last `n` items from `arr` as a new array.
   * Public helper for extracting a small tail window without allocating a full slice
   * in hot code paths.
   */
  static tail<T>(arr: T[] | undefined, n: number): T[] {
    if (!Array.isArray(arr) || n <= 0) return [];
    const out: T[] = [];
    const start = Math.max(0, arr.length - n);
    for (let i = start; i < arr.length; i++) out.push(arr[i]!);
    return out;
  }
  /**
   * Return the last element of an array or undefined when empty. Public helper.
   * @param arr - array to read from
   * @returns last element or undefined
   */
  static safeLast<T>(arr?: T[] | null): T | undefined {
    // Prefer Array.prototype.at where available (ES2022+). Fallback to classic checks.
    if (!Array.isArray(arr) || arr.length === 0) return undefined;
    // Use at(-1) for clarity and expressiveness.
    // Note: `arr.at` is available in modern runtimes; guard defensively.
    // Keep behavior identical (undefined when empty).
    return (arr as any).at ? (arr as any).at(-1) : arr[arr.length - 1];
  }

  /**
   * Push a value onto a bounded history buffer and trim the head if needed.
   * Returns the (possibly new) array. Made public so many modules can reuse it.
   * @param buf - existing buffer (may be undefined)
   * @param v - value to push
   * @param maxLen - maximum length to retain
   */
  static pushHistory<T>(buf: T[] | undefined, v: T, maxLen: number): T[] {
    // Keep the same in-place semantics: callers may hold references to the buffer.
    if (!Array.isArray(buf)) {
      return [v];
    }
    buf.push(v);
    if (buf.length > maxLen) {
      // Remove any excess items from the head in one operation to avoid repeated O(n)
      // shifts when the buffer somehow overflows by more than one element.
      const excess = buf.length - maxLen;
      if (excess === 1) {
        // Common case: remove single oldest element.
        buf.shift();
      } else {
        // Remove multiple at once.
        buf.splice(0, excess);
      }
    }
    return buf;
  }
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
  static encodeMaze(asciiMaze: ReadonlyArray<string>): number[][] {
    /**
     * Set of characters representing walls in the maze.
     * Includes box-drawing and hash characters.
     */
    const wallChars = MazeUtils.#WALL_CHARS;
    // Map each row and cell to its numeric encoding
    const codeMap: Record<string, number> = { '.': 0, E: 1, S: 2 };
    return asciiMaze.map((row) =>
      [...row].map((cell) => {
        if (wallChars.has(cell)) return -1;
        return codeMap[cell] ?? 0;
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
  static findPosition(
    asciiMaze: ReadonlyArray<string>,
    char: string
  ): readonly [number, number] {
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
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    start: readonly [number, number],
    goal: readonly [number, number]
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
    const queue: Array<[readonly [number, number], number]> = [[start, 0]];
    /**
     * Set of visited positions (as string keys)
     */
    const visited = new Set<string>();
    visited.add(MazeUtils.posKey(start));
    /**
     * Possible movement directions (N, E, S, W)
     */
    // Use shared private DIRECTIONS vector for clarity and to avoid duplication.
    const directions = MazeUtils.#DIRECTIONS;
    // BFS loop — use index pointer instead of shift() to avoid O(n^2) behavior
    // when the queue grows large. This keeps the algorithm O(n).
    let qIndex = 0;
    while (qIndex < queue.length) {
      const [[x, y], dist] = queue[qIndex++];
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
          !visited.has(MazeUtils.posKey([nx, ny]))
        ) {
          visited.add(MazeUtils.posKey([nx, ny]));
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
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    currentPos: readonly [number, number],
    startPos: readonly [number, number],
    exitPos: readonly [number, number]
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
    distanceMap: ReadonlyArray<ReadonlyArray<number>>,
    currentPos: readonly [number, number],
    startPos: readonly [number, number]
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
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    goal: readonly [number, number]
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
    const dirs = MazeUtils.#DIRECTIONS;
    // BFS loop to fill distance map. Use index pointer to avoid shift() overhead.
    let qi = 0;
    while (qi < q.length) {
      const [x, y] = q[qi++];
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
