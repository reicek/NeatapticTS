/**
 * Maze Visualization - Handles rendering and visualization of mazes
 *
 * This module contains functions for visualizing mazes in the terminal,
 * including colored cell rendering, path visualization, and progress indicators.
 * It provides an intuitive way to observe the agent's behavior and solution paths.
 *
 * The visualization uses ANSI color codes to create a rich terminal interface
 * showing different maze elements (walls, paths, start/exit) and the agent's
 * current position and traversal history.
 */

import { MazeUtils } from './mazeUtils';
import { colors } from './colors';
import { NetworkVisualization } from './networkVisualization';

/**
 * MazeVisualization provides static methods for rendering mazes and agent progress.
 */
export class MazeVisualization {
  // Shared set of wall characters (private implementation detail).
  // Provide a public getter for backward compatibility.
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

  static get WALL_CHARS() {
    return MazeVisualization.#WALL_CHARS;
  }

  /** Return the last element of an array or undefined when empty. */
  static #last<T>(arr?: readonly T[] | null): T | undefined {
    return MazeUtils.safeLast(arr as any) as T | undefined;
  }

  /** Convert a [x,y] pair to the canonical 'x,y' key. */
  static #posKey([x, y]: readonly [number, number]): string {
    return `${x},${y}`;
  }

  // Shared scratch buffer used for temporary per-call computations such as
  // counting walkable cells. Reusing a single Int8Array reduces allocations
  // when `printMazeStats` is invoked repeatedly during benchmarks.
  static #SCRATCH_INT8 = new Int8Array(0);

  static #getScratchInt8(minLength: number): Int8Array {
    if (MazeVisualization.#SCRATCH_INT8.length < minLength) {
      let newCapacity = MazeVisualization.#SCRATCH_INT8.length || 1;
      while (newCapacity < minLength) newCapacity <<= 1;
      MazeVisualization.#SCRATCH_INT8 = new Int8Array(newCapacity);
    }
    return MazeVisualization.#SCRATCH_INT8;
  }
  /**
   * Renders a single maze cell with proper coloring based on its content and agent location.
   *
   * Applies appropriate colors and styling to each cell in the maze:
   * - Different colors for walls, open paths, start and exit positions
   * - Highlights the agent's current position
   * - Marks cells that are part of the agent's path
   * - Renders box drawing characters as walls with proper styling
   *
   * @param cell - The character representing the cell ('S', 'E', '#', '.' etc.)
   * @param x - X-coordinate of the cell
   * @param y - Y-coordinate of the cell
   * @param agentX - X-coordinate of the agent's current position
   * @param agentY - Y-coordinate of the agent's current position
   * @param path - Optional set of visited coordinates in "x,y" format
   * @returns Colorized string representing the cell
   */
  static renderCell(
    cell: string,
    x: number,
    y: number,
    agentX: number,
    agentY: number,
    path: ReadonlySet<string> | undefined
  ): string {
    /**
     * renderCell: Render a single maze character with ANSI styling.
     * - Agent position takes precedence.
     * - Start/Exit are highlighted.
     * - Visited path cells are rendered as small bullets to give breadcrumb context.
     *
     * The function avoids allocations on the hot path: it formats and returns
     * the colorized string immediately and does not mutate shared state.
     */
    // Use shared WALL_CHARS to avoid allocating repeatedly
    const wallChars = MazeVisualization.WALL_CHARS;

    // Agent's current position takes precedence in visualization
    if (x === agentX && y === agentY) {
      if (cell === 'S')
        return `${colors.bgBlack}${colors.orangeNeon}S${colors.reset}`;
      if (cell === 'E')
        return `${colors.bgBlack}${colors.orangeNeon}E${colors.reset}`;
      return `${colors.bgBlack}${colors.orangeNeon}A${colors.reset}`; // 'A' for Agent - TRON cyan
    }

    // Render other cell types with explicit conditionals (avoids string switch)
    if (cell === 'S')
      return `${colors.bgBlack}${colors.orangeNeon}S${colors.reset}`;
    if (cell === 'E')
      return `${colors.bgBlack}${colors.orangeNeon}E${colors.reset}`;
    if (cell === '.') {
      if (path && path.has(`${x},${y}`))
        return `${colors.floorBg}${colors.orangeNeon}•${colors.reset}`;
      return `${colors.floorBg}${colors.gridLineText}.${colors.reset}`;
    }
    // For box drawing characters and # - render as wall
    if (wallChars.has(cell)) {
      return `${colors.bgBlack}${colors.blueNeon}${cell}${colors.reset}`;
    }
    return cell; // Any other character
  }

  /**
   * Renders the entire maze as a colored ASCII string, showing the agent and its path.
   *
   * Converts the maze data structure into a human-readable, colorized representation showing:
   * - The maze layout with walls and open paths
   * - The start and exit positions
   * - The agent's current position
   * - The path the agent has taken (if provided)
   *
   * @param asciiMaze - Array of strings representing the maze layout
   * @param [agentX, agentY] - Current position of the agent
   * @param path - Optional array of positions representing the agent's path
   * @returns A multi-line string with the visualized maze
   */
  static visualizeMaze(
    asciiMaze: string[],
    [agentX, agentY]: readonly [number, number],
    path?: readonly [number, number][]
  ): string {
    /**
     * visualizeMaze: Convert a maze to a colored ASCII representation.
     *
     * For quick membership checks, we convert the optional `path` array into
     * a Set of "x,y" strings. This reduces repeated O(n) scans when rendering
     * large mazes and keeps the rendering loop tight.
     */
    // Convert path array to a set of "x,y" strings for quick lookup
    let visitedPositions: Set<string> | undefined = undefined;
    if (path) {
      visitedPositions = new Set<string>();
      for (const p of path) visitedPositions.add(MazeVisualization.#posKey(p));
    }

    // Process each row and cell
    return asciiMaze
      .map((row, y) =>
        [...row]
          .map((cell, x) =>
            this.renderCell(cell, x, y, agentX, agentY, visitedPositions)
          )
          .join('')
      )
      .join('\n');
  }

  /**
   * Print a concise, colorized summary of the agent's attempt.
   *
   * This function aggregates key run metrics and prints them using `forceLog`.
   * It is intended for human-friendly terminal output and is optimized to
   * minimize intermediate allocations (reusing a typed scratch buffer).
   *
   * @example
   * MazeVisualization.printMazeStats(currentBest, asciiMaze, console.log);
   *
   * @param currentBest - Object containing the run `result`, `network` and `generation`.
   * @param maze - Array of strings representing the ASCII maze layout.
   * @param forceLog - Logging function used for emitting formatted lines.
   */
  static printMazeStats(
    currentBest: {
      result: any;
      network: any;
      generation: number;
    },
    maze: string[],
    forceLog: (...args: any[]) => void
  ): void {
    // --- Step 0: unpack inputs and derive colors ---
    const { result, generation } = currentBest;
    const successColor = result.success ? colors.cyanNeon : colors.neonRed;

    // --- Step 1: locate important maze positions and compute optimal length ---
    const startPos = MazeUtils.findPosition(maze, 'S');
    const exitPos = MazeUtils.findPosition(maze, 'E');
    const optimalLength = MazeUtils.bfsDistance(
      MazeUtils.encodeMaze(maze),
      startPos,
      exitPos
    );

    // Layout constants (keep in sync with DashboardManager framing)
    const FRAME_WIDTH = 148;
    const LEFT_PAD = 7;
    const RIGHT_PAD = 1;
    const CONTENT_WIDTH = FRAME_WIDTH - LEFT_PAD - RIGHT_PAD;

    forceLog(
      `${colors.blueCore}║${NetworkVisualization.pad(' ', FRAME_WIDTH, ' ')}${
        colors.blueCore
      }║${colors.reset}`
    );
    forceLog(
      `${colors.blueCore}║${NetworkVisualization.pad(' ', FRAME_WIDTH, ' ')}${
        colors.blueCore
      }║${colors.reset}`
    );
    forceLog(
      `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
        `${colors.neonSilver}Success:${colors.neonIndigo} ${successColor}${
          result.success ? 'YES' : 'NO'
        }`,
        CONTENT_WIDTH,
        ' ',
        'left'
      )}${' '.repeat(RIGHT_PAD)}${colors.blueCore}║${colors.reset}`
    );
    // Print generation number with color and padding
    forceLog(
      `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
        `${colors.neonSilver}Generation:${colors.neonIndigo} ${successColor}${generation}`,
        CONTENT_WIDTH,
        ' ',
        'left'
      )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
    );
    // Print fitness score
    forceLog(
      `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
        `${colors.neonSilver}Fitness:${
          colors.neonOrange
        } ${result.fitness.toFixed(2)}`,
        CONTENT_WIDTH,
        ' ',
        'left'
      )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
    );
    // Print steps taken
    forceLog(
      `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
        `${colors.neonSilver}Steps taken:${colors.neonIndigo} ${result.steps}`,
        CONTENT_WIDTH,
        ' ',
        'left'
      )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
    );
    // Print path length
    forceLog(
      `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
        `${colors.neonSilver}Path length:${colors.neonIndigo} ${result.path.length}${colors.blueCore}`,
        CONTENT_WIDTH,
        ' ',
        'left'
      )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
    );
    // Print optimal distance to exit
    forceLog(
      `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
        `${colors.neonSilver}Optimal distance to exit:${colors.neonYellow} ${optimalLength}`,
        CONTENT_WIDTH,
        ' ',
        'left'
      )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
    );
    // Print a blank padded line for spacing
    forceLog(
      `${colors.blueCore}║${NetworkVisualization.pad(' ', FRAME_WIDTH, ' ')}${
        colors.blueCore
      }║${colors.reset}`
    );

    if (result.success) {
      // --- Step 2: basic path metrics ---
      const pathLength = result.path.length - 1;

      // Efficiency: ratio of optimal path to actual path, capped at 100%.
      const efficiency = Math.min(
        100,
        Math.round((optimalLength / pathLength) * 100)
      ).toFixed(1);

      // Overhead: percent longer than optimal (positive = worse).
      const overhead = ((pathLength / optimalLength) * 100 - 100).toFixed(1);

      // --- Step 3: analyze the path for unique cells, revisits and direction changes ---
      const uniqueCells = new Set<string>();
      let revisitedCells = 0;
      let directionChanges = 0;
      let lastDirection: string | null = null;

      for (let stepIndex = 0; stepIndex < result.path.length; stepIndex++) {
        const [cellX, cellY] = result.path[stepIndex];
        const cellKey = `${cellX},${cellY}`;

        // Track revisits
        if (uniqueCells.has(cellKey)) revisitedCells++;
        else uniqueCells.add(cellKey);

        // Count direction changes (skip first step)
        if (stepIndex > 0) {
          const [prevX, prevY] = result.path[stepIndex - 1];
          const dx = cellX - prevX;
          const dy = cellY - prevY;

          // Replace chained if/else with a switch for clarity and JIT-friendliness.
          let currentDirection = '';
          switch (true) {
            case dx > 0:
              currentDirection = 'E';
              break;
            case dx < 0:
              currentDirection = 'W';
              break;
            case dy > 0:
              currentDirection = 'S';
              break;
            case dy < 0:
              currentDirection = 'N';
              break;
            default:
              currentDirection = '';
          }

          if (lastDirection !== null && currentDirection !== lastDirection)
            directionChanges++;
          lastDirection = currentDirection;
        }
      }

      const mazeWidth = maze[0].length;
      const mazeHeight = maze.length;

      // Encode the maze and count walkable cells using a reusable Int8Array
      const encodedMaze = MazeUtils.encodeMaze(maze);

      // Use scratch Int8Array: 1 => walkable, 0 => wall. Then count ones.
      const flatCellCount = mazeWidth * mazeHeight;
      const scratch = MazeVisualization.#getScratchInt8(flatCellCount);
      let scratchIndex = 0;
      for (let rowY = 0; rowY < mazeHeight; rowY++) {
        const row = encodedMaze[rowY];
        for (let colX = 0; colX < mazeWidth; colX++, scratchIndex++) {
          scratch[scratchIndex] = row[colX] === -1 ? 0 : 1;
        }
      }

      let walkableCells = 0;
      for (let i = 0; i < flatCellCount; i++) walkableCells += scratch[i];

      const coveragePercent = (
        (uniqueCells.size / walkableCells) *
        100
      ).toFixed(1);

      // Display detailed statistics
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Path efficiency:      ${colors.neonIndigo} ${optimalLength}/${pathLength} (${efficiency}%)`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Optimal steps:        ${colors.neonIndigo} ${optimalLength}`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Path overhead:        ${colors.neonIndigo} ${overhead}% longer than optimal`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Direction changes:    ${colors.neonIndigo} ${directionChanges}`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Unique cells visited: ${colors.neonIndigo} ${uniqueCells.size} (${coveragePercent}% of maze)`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Cells revisited:      ${colors.neonIndigo} ${revisitedCells} times`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Decisions per cell:   ${colors.neonIndigo} ${(
            directionChanges / uniqueCells.size
          ).toFixed(2)}`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonOrange}Agent successfully navigated the maze!`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
    } else {
      // If the agent did not succeed, display progress toward the exit and unique cells visited.
      const lastPos =
        MazeVisualization.#last(result.path as readonly [number, number][]) ??
        startPos;
      const bestProgress = MazeUtils.calculateProgress(
        MazeUtils.encodeMaze(maze),
        lastPos,
        startPos,
        exitPos
      );

      const uniqueCells = new Set<string>();
      for (const [x, y] of result.path) uniqueCells.add(`${x},${y}`);

      // Display partial progress statistics
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Best progress toward exit:      ${colors.neonIndigo} ${bestProgress}%`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Shortest possible steps:        ${colors.neonIndigo} ${optimalLength}`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Unique cells visited:           ${colors.neonIndigo} ${uniqueCells.size}`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
      forceLog(
        `${colors.blueCore}║${' '.repeat(LEFT_PAD)}${NetworkVisualization.pad(
          `${colors.neonSilver}Agent trying to reach the exit. ${colors.neonIndigo}`,
          CONTENT_WIDTH,
          ' ',
          'left'
        )}${' '.repeat(RIGHT_PAD)}║${colors.reset}`
      );
    }
  }

  /**
   * Displays a colored progress bar for agent progress.
   *
   * Creates a visual representation of the agent's progress toward the exit
   * as a horizontal bar with appropriate coloring based on percentage.
   *
   * @example
   * // Render a 40% progress bar of default length
   * MazeVisualization.displayProgressBar(40);
   *
   * @param progress - Progress percentage (0-100). Values outside the range
   *  will be clamped to [0,100].
   * @param length - Total length of the bar in characters (defaults to 60).
   * @returns A colorized string containing the formatted progress bar.
   */
  static displayProgressBar(progress: number, length: number = 60): string {
    // --- Step 1: normalize inputs and compute filled/empty counts ---
    const clampedProgress = Math.max(0, Math.min(100, Math.round(progress)));
    const filledCount = Math.max(
      0,
      Math.min(length, Math.floor((length * clampedProgress) / 100))
    );

    // Characters for the progress bar visuals
    const startCap = `${colors.blueCore}|>|`;
    const endCap = `${colors.blueCore}|<|`;
    const fillSegment = `${colors.neonOrange}═`;
    const emptySegment = `${colors.neonIndigo}:`;
    const pointerGlyph = `${colors.neonOrange}▶`; // Indicates the current progress point

    // Build the progress bar using clearer variable names and fewer branches
    let bar = '';
    bar += startCap;

    if (filledCount > 0) {
      // Fill all but the last filled position with the fill segment, then
      // place a pointer glyph at the current progress location.
      if (filledCount > 1) bar += fillSegment.repeat(filledCount - 1);
      bar += pointerGlyph;
    }

    // Remaining (empty) slots
    const remainingCount = length - filledCount;
    if (remainingCount > 0) bar += emptySegment.repeat(remainingCount);

    bar += endCap;

    // --- Step 3: choose color based on progress using a switch for clarity ---
    let barColor = colors.cyanNeon;
    switch (true) {
      case clampedProgress < 30:
        barColor = colors.neonYellow;
        break;
      case clampedProgress < 70:
        barColor = colors.orangeNeon;
        break;
      default:
        barColor = colors.cyanNeon;
    }

    return `${barColor}${bar}${colors.reset} ${clampedProgress}%`;
  }

  /**
   * Formats elapsed time in a human-readable way.
   *
   * Converts seconds into appropriate units (seconds, minutes, hours)
   * for more intuitive display of time durations.
   *
   * @param seconds - Time in seconds
   * @returns Formatted string (e.g., "5.3s", "2m 30s", "1h 15m")
   */
  static(seconds: number): string {
    // If less than a minute, show seconds with one decimal
    if (seconds < 60) return `${seconds.toFixed(1)}s`;

    // If less than an hour, show minutes and seconds
    if (seconds < 3600) {
      /**
       * Number of whole minutes in the input seconds.
       */
      const minutes = Math.floor(seconds / 60);
      /**
       * Remaining seconds after extracting minutes.
       */
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    }

    /**
     * Number of whole hours in the input seconds.
     */
    const hours = Math.floor(seconds / 3600);
    /**
     * Number of whole minutes after extracting hours.
     */
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
}
