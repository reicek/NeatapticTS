/**
 * Utility class for maze logic and encoding.
 *
 * Provides helpers used by the ASCII maze examples and tests such as
 * encoding, position lookup, BFS distance computations and progress
 * calculations. Methods are intentionally small and well-documented to
 * support educational consumption.
 *
 * @example
 * const encoded = MazeUtils.encodeMaze(['S..','.#E','...']);
 */
export class MazeUtils {
  /**
   * Shared set of wall characters (box-drawing + hash). Kept private to avoid
   * reallocation in hot code paths. Use `encodeMaze` to convert ASCII mazes.
   */
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
  /** Movement vectors used by BFS (North, East, South, West). */
  static #DIRECTIONS: readonly (readonly [number, number])[] = [
    [0, -1],
    [1, 0],
    [0, 1],
    [-1, 0],
  ];

  // Shared, resizable queue buffer used by BFS to avoid allocating a new
  // Int32Array on each invocation. This is a simple pooling strategy: the
  // buffer grows as needed and is reused for subsequent calls.
  static #QUEUE_BUFFER: Int32Array = new Int32Array(0);

  /**
   * Ensure the internal shared queue buffer has at least `minLength` capacity.
   * Returns the buffer (may be newly allocated). Maintaining a shared buffer
   * reduces GC churn in hot code paths that repeatedly run BFS on many mazes.
   */
  static #getQueueBuffer(minLength: number): Int32Array {
    if (MazeUtils.#QUEUE_BUFFER.length < minLength) {
      // Grow to the next power-of-two for exponential growth behaviour.
      let grownCapacity = MazeUtils.#QUEUE_BUFFER.length || 1;
      while (grownCapacity < minLength) grownCapacity <<= 1;
      MazeUtils.#QUEUE_BUFFER = new Int32Array(grownCapacity);
    }
    return MazeUtils.#QUEUE_BUFFER;
  }

  /**
   * Convert a coordinate pair into the canonical key string `"x,y"`.
   *
   * @param coord - Coordinate pair `[x, y]` to stringify.
   * @returns Canonical string key suitable for Map/Set storage.
   * @example MazeUtils.posKey([2,3]) // -> '2,3'
   */
  static posKey(coord: readonly [number, number]): string {
    const [x, y] = coord;
    return `${x},${y}`;
  }

  /**
   * Return up to the last `n` items from `arr` as a new array.
   * This implementation is allocation-friendly for hot paths: it preallocates
   * the output array and copies only the required suffix.
   *
   * @param arr - Source array (may be undefined).
   * @param n - Maximum number of trailing elements to return.
   * @returns New array containing up to the last `n` items of `arr`.
   * @example
   * MazeUtils.tail([1,2,3,4], 2) // -> [3,4]
   */
  static tail<T>(arr: T[] | undefined, n: number): T[] {
    // Step 0: handle invalid inputs fast.
    if (!Array.isArray(arr) || n <= 0) return [];

    // Step 1: compute bounds for the suffix to copy.
    const length = arr.length;
    const startIndex = Math.max(0, length - n);
    const resultLength = length - startIndex;

    // Step 2: preallocate result and copy elements (avoids push churn).
    const result: T[] = new Array(resultLength);
    let writeIndex = 0;
    for (let readIndex = startIndex; readIndex < length; readIndex++) {
      result[writeIndex++] = arr[readIndex]!;
    }
    return result;
  }

  /**
   * Return the last element of an array or undefined when empty.
   *
   * This helper prefers `Array.prototype.at` when available (ES2022+), but
   * falls back gracefully for older runtimes.
   *
   * @param arr - Array to read from.
   * @returns The last item or undefined.
   */
  static safeLast<T>(arr?: T[] | null): T | undefined {
    if (!Array.isArray(arr) || arr.length === 0) return undefined;
    return (arr as any).at ? (arr as any).at(-1) : arr[arr.length - 1];
  }

  /**
   * Push a value onto a bounded history buffer and trim the head if needed.
   * This helper preserves in-place semantics (the original array reference is
   * returned and mutated) which callers may rely on for performance.
   *
   * @param buffer - Existing buffer (may be undefined). If undefined a new
   *  single-element array containing `value` is returned.
   * @param value - Value to push onto the buffer.
   * @param maxLen - Maximum length to retain. When the buffer exceeds this
   *  length the oldest entries are removed from the head.
   * @returns The updated buffer containing the new value and trimmed to maxLen.
   * @example
   * const buf = [1,2]; MazeUtils.pushHistory(buf, 3, 3) // -> [1,2,3]
   */
  static pushHistory<T>(
    buffer: T[] | undefined,
    value: T,
    maxLen: number
  ): T[] {
    // Fast-path: if no existing buffer, return a new one containing the value.
    if (!Array.isArray(buffer)) return [value];

    // Keep in-place semantics: append the new value onto the provided buffer.
    buffer.push(value);

    // If we exceed the allowed length, remove excess items from the head.
    const excessCount = buffer.length - maxLen;
    if (excessCount > 0) {
      if (excessCount === 1) {
        // Common-case optimization: remove single oldest element.
        buffer.shift();
      } else {
        // Remove multiple oldest elements in one splice operation.
        buffer.splice(0, excessCount);
      }
    }

    return buffer;
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
     * Characters treated as walls. Kept as a private Set for fast `has()` lookups.
     */
    const wallChars = MazeUtils.#WALL_CHARS;

    // Small lookup for non-wall special characters. Map is explicit and avoids
    // accidental coercion when checking membership.
    const codeMap = new Map<string, number>([
      ['.', 0],
      ['E', 1],
      ['S', 2],
    ]);

    // Step: convert each ASCII row into a numeric row. Preallocate the numeric
    // row to avoid intermediate arrays and reduce allocations in hot code paths.
    return asciiMaze.map((rowString, rowIndex) => {
      const rowLength = rowString.length;
      const encodedRow: number[] = new Array(rowLength);
      for (let colIndex = 0; colIndex < rowLength; colIndex++) {
        const cellChar = rowString[colIndex];
        // Wall characters take precedence.
        if (wallChars.has(cellChar)) {
          encodedRow[colIndex] = -1;
          continue;
        }
        // Lookup special encodings, default to 0 (open path).
        encodedRow[colIndex] = codeMap.get(cellChar) ?? 0;
      }
      return encodedRow;
    });
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
    // Fast, low-allocation search:
    // - Use an index-based loop to avoid iterator allocations.
    // - Cache the row count for one-time lookup.
    // - Use `String#indexOf` which is the fastest way to locate a substring/char in V8.
    const rowCount = asciiMaze.length;
    for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
      const rowString = asciiMaze[rowIndex];
      // Skip empty rows quickly.
      if (!rowString) continue;

      const columnIndex = rowString.indexOf(char);
      if (columnIndex !== -1) {
        // Return coordinates as a readonly tuple to match existing contract.
        return [columnIndex, rowIndex] as const;
      }
    }

    // Not found: preserve original behavior and raise an explicit error.
    throw new Error(`Character ${char} not found in maze`);
  }

  /**
   * Compute the Manhattan shortest-path length between two cells using a
   * breadth-first search (BFS).
   *
   * This implementation is allocation-friendly and optimized for hot paths:
   * - Flattens the maze into a typed Int8Array for O(1) cell checks.
   * - Uses an Int32Array distances buffer with sentinel -1 for unvisited cells.
   * - Reuses a shared Int32Array queue buffer (pool) to avoid per-call allocations.
   *
   * @example
   * const encoded = MazeUtils.encodeMaze(['S..','.#E','...']);
   * const start = MazeUtils.findPosition(['S..','.#E','...'], 'S');
   * const exit = MazeUtils.findPosition(['S..','.#E','...'], 'E');
   * const steps = MazeUtils.bfsDistance(encoded, start, exit); // => number | Infinity
   *
   * @param encodedMaze - 2D numeric maze encoding where -1 are walls and other
   *  values are walkable. Array shape is [rows][cols].
   * @param start - Start coordinates as [x, y].
   * @param goal - Goal coordinates as [x, y].
   * @returns Number of steps in the shortest path, or Infinity when unreachable.
   */
  static bfsDistance(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    start: readonly [number, number],
    goal: readonly [number, number]
  ): number {
    // --- Step 1: validate inputs and derive grid metadata ---
    const [startX, startY] = start;
    const [goalX, goalY] = goal;
    const rowCount = encodedMaze.length;
    const colCount = encodedMaze[0].length;

    // Ensure start and goal are within bounds; return Infinity for invalid coords.
    if (
      startY < 0 ||
      startY >= rowCount ||
      startX < 0 ||
      startX >= colCount ||
      goalY < 0 ||
      goalY >= rowCount ||
      goalX < 0 ||
      goalX >= colCount
    ) {
      return Infinity;
    }

    // If either start or goal is a wall, there is no path.
    if (
      encodedMaze[startY][startX] === -1 ||
      encodedMaze[goalY][goalX] === -1
    ) {
      return Infinity;
    }

    // Trivial same-cell case.
    if (startX === goalX && startY === goalY) return 0;

    // --- Step 2: flatten maze into an Int8Array for fast neighbor checks ---
    const cellCount = rowCount * colCount;
    const flatWalkable = new Int8Array(cellCount);
    for (let rowIndex = 0, flatWrite = 0; rowIndex < rowCount; rowIndex++) {
      const row = encodedMaze[rowIndex];
      for (let colIndex = 0; colIndex < colCount; colIndex++, flatWrite++) {
        // -1 denotes wall; anything else is walkable (0).
        flatWalkable[flatWrite] = row[colIndex] === -1 ? -1 : 0;
      }
    }

    // --- Step 3: distances buffer (-1 = unvisited) ---
    const distances = new Int32Array(cellCount);
    for (let fillIndex = 0; fillIndex < cellCount; fillIndex++)
      distances[fillIndex] = -1;

    const startFlatIndex = startY * colCount + startX;
    const goalFlatIndex = goalY * colCount + goalX;
    distances[startFlatIndex] = 0;

    // --- Step 4: queue (shared buffer) and BFS core loop ---
    // Borrow a shared Int32Array buffer large enough for the grid.
    const queue = MazeUtils.#getQueueBuffer(cellCount);
    let queueHead = 0;
    let queueTail = 0;
    queue[queueTail++] = startFlatIndex;

    // Precompute vertical offsets for neighbors on the flattened grid.
    const northOffset = -colCount;
    const southOffset = colCount;

    // BFS: each cell is visited at most once, so this loop is O(cellCount).
    while (queueHead < queueTail) {
      const currentFlatIndex = queue[queueHead++];
      const currentDistance = distances[currentFlatIndex];

      // Early exit: if we reached the goal return its distance now.
      if (currentFlatIndex === goalFlatIndex) return currentDistance;

      // Compute 2D coordinates from the flattened index.
      const currentRow = (currentFlatIndex / colCount) | 0; // fast floor
      const currentCol = currentFlatIndex - currentRow * colCount;

      // Examine cardinal neighbors using a small switch to centralize bounds
      // checks and enqueue logic (keeps code generation compact and JIT-friendly).
      for (let direction = 0; direction < 4; direction++) {
        let neighborFlatIndex: number;
        switch (direction) {
          case 0: // North
            if (currentRow === 0) continue;
            neighborFlatIndex = currentFlatIndex + northOffset;
            break;
          case 1: // East
            if (currentCol + 1 >= colCount) continue;
            neighborFlatIndex = currentFlatIndex + 1;
            break;
          case 2: // South
            if (currentRow + 1 >= rowCount) continue;
            neighborFlatIndex = currentFlatIndex + southOffset;
            break;
          default:
            // West
            if (currentCol === 0) continue;
            neighborFlatIndex = currentFlatIndex - 1;
        }

        // If the neighbor is walkable and unvisited, mark distance and enqueue.
        if (
          flatWalkable[neighborFlatIndex] !== -1 &&
          distances[neighborFlatIndex] === -1
        ) {
          const neighborDistance = currentDistance + 1;
          distances[neighborFlatIndex] = neighborDistance;
          if (neighborFlatIndex === goalFlatIndex) return neighborDistance;
          queue[queueTail++] = neighborFlatIndex;
        }
      }
    }

    // Goal not reachable from start.
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
    const [startX, startY] = startPos;
    const [currentX, currentY] = currentPos;
    /**
     * Total distance from start to goal (from distance map)
     */
    const totalDistance = distanceMap[startY]?.[startX];
    /**
     * Remaining distance from current position to goal (from distance map)
     */
    const remaining = distanceMap[currentY]?.[currentX];
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
    // Use the fast flat variant internally, then convert to the legacy number[][] shape.
    const {
      width,
      height,
      distances,
      WALL_VALUE,
      UNREACHABLE_VALUE,
    } = MazeUtils.buildDistanceMapFlat(encodedMaze, goal);

    const result: number[][] = Array.from(
      { length: height },
      () => new Array<number>(width)
    );

    // Convert flat typed-array distances back into legacy 2D shape.
    for (let rowIndex = 0, flatIndex = 0; rowIndex < height; rowIndex++) {
      const resultRow = result[rowIndex];
      for (let colIndex = 0; colIndex < width; colIndex++, flatIndex++) {
        const cellDistance = distances[flatIndex];
        // Walls and unreachable cells map to Infinity to preserve the original API.
        resultRow[colIndex] =
          cellDistance === WALL_VALUE || cellDistance === UNREACHABLE_VALUE
            ? Infinity
            : cellDistance;
      }
    }

    return result;
  }

  /**
   * High-performance variant of `buildDistanceMap` that returns a flat Int32Array
   * distance buffer with metadata. This minimizes allocations and GC pressure for
   * large mazes and is the recommended API for performance-sensitive code.
   *
   * Encoding in the returned `distances` buffer:
   *  - wall cells => WALL_VALUE (number, -2)
   *  - unreachable cells => UNREACHABLE_VALUE (number, -1)
   *  - reachable cells => non-negative distance (0 ..)
   *
   * The returned object contains the flat buffer plus `width` and `height` so
   * callers can translate between (x,y) and flat indices: index = y*width + x.
   *
   * Note: this API intentionally avoids converting the typed buffer back into
   * nested `number[][]` to keep allocation minimal; use `buildDistanceMap` if
   * you require the legacy `number[][]` shape.
   *
   * @param encodedMaze - 2D maze encoding
   * @param goal - [x,y] goal position (typically exit)
   * @returns Object with `width`, `height`, and `distances` (Int32Array).
   * @example
   * const flat = MazeUtils.buildDistanceMapFlat(encoded, [5,3]);
   * const idx = 3 * flat.width + 5; // y*width + x
   * console.log(flat.distances[idx]); // -2 wall, -1 unreachable, >=0 distance
   */
  static buildDistanceMapFlat(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    goal: readonly [number, number]
  ): {
    width: number;
    height: number;
    distances: Int32Array;
    readonly WALL_VALUE: number;
    readonly UNREACHABLE_VALUE: number;
  } {
    const height = encodedMaze.length;
    const width = encodedMaze[0].length;

    const WALL_VALUE = -2; // distinct sentinel for walls
    const UNREACHABLE_VALUE = -1; // sentinel for unvisited / unreachable

    const cellCount = width * height;
    const distances = new Int32Array(cellCount);
    // initialize all cells to UNREACHABLE
    for (let flatInitIndex = 0; flatInitIndex < cellCount; flatInitIndex++)
      distances[flatInitIndex] = UNREACHABLE_VALUE;

    const [goalX, goalY] = goal;
    // if goal is out of bounds or a wall, return early with walls marked
    if (
      goalY < 0 ||
      goalY >= height ||
      goalX < 0 ||
      goalX >= width ||
      encodedMaze[goalY][goalX] === -1
    ) {
      // mark walls and return
      for (let rowIndex = 0, flatDest = 0; rowIndex < height; rowIndex++) {
        const row = encodedMaze[rowIndex];
        for (let colIndex = 0; colIndex < width; colIndex++, flatDest++) {
          if (row[colIndex] === -1) distances[flatDest] = WALL_VALUE;
        }
      }
      return { width, height, distances, WALL_VALUE, UNREACHABLE_VALUE };
    }

    // mark walls in distances buffer
    for (let rowIndex = 0, flatDest = 0; rowIndex < height; rowIndex++) {
      const row = encodedMaze[rowIndex];
      for (let colIndex = 0; colIndex < width; colIndex++, flatDest++) {
        if (row[colIndex] === -1) distances[flatDest] = WALL_VALUE;
      }
    }

    const goalIndex = goalY * width + goalX;
    distances[goalIndex] = 0;

    // typed queue
    const queue = new Int32Array(cellCount);
    let queueHead = 0;
    let queueTail = 0;
    queue[queueTail++] = goalIndex;

    const northOffset = -width;
    const southOffset = width;

    while (queueHead < queueTail) {
      const currentIndex = queue[queueHead++];
      const currentDistance = distances[currentIndex];

      const currentRow = (currentIndex / width) | 0;
      const currentCol = currentIndex - currentRow * width;

      // Process four cardinal neighbors via a small switch to centralize logic
      for (let dir = 0; dir < 4; dir++) {
        let neighborFlatIndex: number;
        switch (dir) {
          case 0: // North
            if (currentRow === 0) continue;
            neighborFlatIndex = currentIndex + northOffset;
            break;
          case 1: // East
            if (currentCol + 1 >= width) continue;
            neighborFlatIndex = currentIndex + 1;
            break;
          case 2: // South
            if (currentRow + 1 >= height) continue;
            neighborFlatIndex = currentIndex + southOffset;
            break;
          default:
            // West
            if (currentCol === 0) continue;
            neighborFlatIndex = currentIndex - 1;
        }

        if (distances[neighborFlatIndex] === UNREACHABLE_VALUE) {
          distances[neighborFlatIndex] = currentDistance + 1;
          queue[queueTail++] = neighborFlatIndex;
        }
      }
    }

    return { width, height, distances, WALL_VALUE, UNREACHABLE_VALUE };
  }
}
