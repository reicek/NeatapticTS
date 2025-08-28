/**
 * Maze Definitions - Collection of test mazes with varying complexity
 *
 * This module contains predefined ASCII mazes used for testing and training
 * the neural network agents. The mazes range from simple to complex and demonstrate
 * different challenges for pathfinding algorithms.
 *
 * Maze symbols:
 * - '║' = Wall (obstacle the agent cannot pass through)
 * - '.' = Open path
 * - 'S' = Start position
 * - 'E' = Exit/Goal position
 *
 * Each maze contains exactly one start ('S') and one exit ('E') position.
 */

/** Very small maze for quick testing with minimal complexity */
export const tiny = [
  '╔════════════════════╗',
  '║S...................║',
  '╠═══════════════════.║',
  '║....................║',
  '║.═══════════════════╣',
  '║....................║',
  '╚══════════════════╗E║',
];

/** Small spiral maze - tests the agent's ability to follow a continuous path */
export const spiralSmall = [
  '╔═══════════╗',
  '║...........║',
  '║.╔═══════╗.║',
  '║.║.......║.║',
  '║.║.╔═══╗.║.║',
  '║.║.║...║.║.║',
  '║.║.║S║.║.║.║',
  '║.║.╚═╝.║.║.║',
  '║.║.....║.║.║',
  '║.╚═════╝.║.║',
  '║.........║.║',
  '╚═════════╣E║',
];

/** Medium spiral maze - more challenging version requiring longer path following */
export const spiral = [
  '╔═══════════════╗',
  '║...............║',
  '║.║.╔═══════════╣',
  '║.║.║...........║',
  '║.║.║.╔═══════╗.║',
  '║.║.║.║.......║.║',
  '║.║.║.║.╔═══╗.║.║',
  '║.║.║.║.║...║.║.║',
  '║.║.║.║.║S║.║.║.║',
  '║.║.║.║.╚═╝.║.║.║',
  '║.║.║.║.....║.║.║',
  '║.║.║.╚═════╝.║.║',
  '║.║.║.........║.║',
  '║.║.╚═════════╝.║',
  '║.║.............║',
  '║E╠═════════════╝',
];

/** Small maze with multiple path options and dead ends */
export const small = [
  '╔═══════╦══════════╗',
  '║S......║..........║',
  '╠══.╔══.║.╔══.║.║..║',
  '║...║...║.║...║.║..║',
  '║.║.║.══╝.╚═══╝.╚══╣',
  '║.║.║..............║',
  '║.║.╚══════.══╦═╗..║',
  '║.║...........║.║..║',
  '║.╚═════╗.════╣.║..║',
  '║.......║.....║.║..║',
  '╠══════.╚═══╗.║.╚══╣',
  '║...........║.║....║',
  '║.═══.╔════.║.╚═══.║',
  '║.....║.....║......║',
  '╚═════╣E╔═══╩══════╝',
];

/** Medium-sized maze with branching paths and decision points */
export const medium = [
  '╔═╦═══════╦═════════════╗',
  '║S║.......║.............║',
  '║.║.════╗.╠═════╗.════╗.║',
  '║.║.....║.║.....║.....║.║',
  '║.╚═══╗.║.╚═══╗.╚═╗.║.║.║',
  '║.....║.║.....║...║.║.║.║',
  '╠═══╗.║.╚═══╗.╠══.║.║.║.║',
  '║...║.║.....║.║...║.║.║.║',
  '║.║.║.╚═══╗.║.║.══╝.║.║.║',
  '║.║.║.....║.║.║.....║.║.║',
  '║.║.╚═══╗.║.║.╚═════╝.║.║',
  '║.║.....║.║.║.........║.║',
  '║.╚═══╗.║.║.╚═════════╣.║',
  '║.....║.║.║...........║.║',
  '╠════.║.║.╚═════╦═══╗.║.║',
  '║.....║.........║...║.║.║',
  '║.════╩═════════╝.║.║.║.║',
  '║.................║.║.║.║',
  '╠══════════════.║.║.║.║.║',
  '║...............║.║...║.║',
  '╠═══════════════╩═╩═══╝.║',
  '║.......................║',
  '║E╔═════════════════════╝',
];

/** Medium-sized maze with branching paths and decision points */
export const medium2 = [
  '╔═╦═══════╦═════════════════════╦═════════════╗',
  '║S║.......║.....................║.............║',
  '║.║.════╗.╠═════╗.════╗.║.════╗.║.════╗.════╗.║',
  '║.║.....║.║.....║.....║.║.....║.║.....║.....║.║',
  '║.╚═══╗.║.╚═══╗.╚═╗.║.║.║.╚═══╣.║.╚═══╣.╚═╗.║.║',
  '║.....║.║.....║...║.║.║.║.....║.║.....║...║.║.║',
  '╠═══╗.║.╚═══╗.╠══.║.║.║.╠═══╗.║.╚═══╗.╠══.║.║.║',
  '║...║.║.....║.║...║.║.║.║...║.║.....║.║...║.║.║',
  '║.║.║.╚═══╗.║.║.══╝.║.║.║.║.║.╚═══╗.║.║.══╝.║.║',
  '║.║.║.....║.║.║.....║.║.║.║.║.....║.║.║.....║.║',
  '║.║.╚═══╗.║.║.╚═════╝.║.║.║.╚═══╗.║.║.╚═════╝.║',
  '║.║.....║.║.║.........║.║.║.....║.║.║.........║',
  '║.╚═══╗.║.║.╚═════════╣.║.╚═══╗.║.║.╚═════════╣',
  '║.....║.║.║...........║.║.....║.║.║...........║',
  '╠════.║.║.╚═════╦═══╗.║.║═════╣.║.╚═════╦═══╗.║',
  '║.....║.........║...║.║.║.....║.........║...║.║',
  '║.════╩═════════╝.║.║.║.║.════╩═════════╝.║.║.║',
  '║.................║.║.║.║.................║.║.║',
  '╠══════════════.║.║.║.║.╠══════════════.║.║.║.║',
  '║...............║.║...║.║...............║.║...║',
  '╠═══════════════╩═╩═══╩═╩═══════════════╝.╚═══╣',
  '║.............................................║',
  '║E╔═══════════════════════════════════════════╝',
];

/** Large, complex maze with many intersections and potential paths */
export const large = [
  '╔════════════════════════════════════════╦════════════════╗',
  '║S.......................................║................║',
  '╠═════════.╔═════.═════════════╦═════════╩════════════╦══.║',
  '║..........║...................║......................║...║',
  '║.╔══════.╔╩═.════╗.╔══════════╩═══════════╗.╔═════╗.═╩══.║',
  '║.║.......║.......║.║......................║.║.....║......║',
  '║.╠═══════╝.══════╝.║.║.╔══════════════════╣.║.════╬══.═══╣',
  '║.║.................║.║.║..................║.║.....║......║',
  '║.║.════╦═══════════╣.║.║.║.══════════════.║.╠════.║.═════╣',
  '║.║....╔╩╗..........╚╦╝.║.║..............║.║.║.....║......║',
  '╠═╝..║.║.╠══.║.║.║...║..╚═╩╦═╗.═════════.║.║.║.════╩══════╣',
  '║....║.║.║...║.║.║.║.║.....║.║.......║...║.║..............║',
  '╠═.══╩═╝.╚═══╩═╝.╚═╩╦╩════╦╝.║.═══╦══╝.══╩╦╩════════════.═╣',
  '║...................║.....║..║....║.......║...............║',
  '╠══════════════════.║.║.══╣.╔═════╩════.╔═╝.══════════════╣',
  '║...................║.║...║.║...........║.................║',
  '║.╔═════════════════╝.╚═══╝.║.╔══.══════╩═════╦═.═════════╣',
  '║.║.....................║...║.║...............║...........║',
  '║.╚═╗.╔═══════╦═══╦════.║.╔═╝.╚═╗..═════════╗.╠═════════.═╣',
  '║...║.║.......║...║.....║.║.....║...........║.║...........║',
  '╠═╗.║.╚═══╗.║.║.║.║.║.══╝.╔══.╔═╩════.╔═╦══.║.║.═════════.║',
  '║.║.║.....║.║...║.║.║.....║...║.......║.║...║.............║',
  '║.║.╚═════╣.║.╔═╣.╚═╝.╔══.║.╔═╩══.════╝.║.══╩═════════════╣',
  '║.║.......║...║.║.....║...║.║...........║.................║',
  '║.╚══════.╚══.║.╠════.╚═══╝.╚══.═╦═╦════╩════════════════.║',
  '║.............║.║................║.║......................║',
  '╠════════════.║.║.════╦══════.═══╝.║.╔══════════.═════════╣',
  '║.............║.......║............║.║....................║',
  '║.╔══════════.╚═══╦═══╣.╔═╦═╦══════╩═╩╗.╔══════.║.════════╣',
  '║.║...............║...║.║.║.║.........║.║.......║.........║',
  '║.╠═╗.════════════╝.══╝.║.║.║.╔═══════╝.║.══════╩═════════╣',
  '║.║.║.....................║.║.║.........║.................║',
  '║.║.╚═══════════════.═════╝.║.║.═══════════════════════...║',
  '║.║.......................................................║',
  '╠═╩═╦═══╦═══╦═══╦═══╦═══╦═══╦═╗.══════════════════════════╣',
  '║...║...║...║...║...║...║...║.║...........................║',
  '║.║.║.║.║.║.║.║.║.║.║.║.║.║.║.╚══════════════════════════.║',
  '║.║...║...║...║...║...║...║...............................║',
  '║E╔═══╩═══╩═══╩═══╩═══╩═══╩═══════════════════════════════╝',
];

/**
 * Extremely large and challenging "Minotaur's Labyrinth" maze
 * This maze represents the most complex challenge with many parallel paths,
 * symmetrical corridors, and a very long optimal solution path.
 */
export const minotaur = [
  '╔══════════════════════════════════════════════════════════════════════════════╗',
  '║..............................................................................║',
  '║.╔════════════╗.╔═════════════════════════════════════════╗.╔══════════════╗..║',
  '║.║............║.║.........................................║.║..............║..║',
  '║.║.╔════════╗.║.║.╔═════════════════════════════════════╗.║.║.╔════════╗.║.║..║',
  '║.║.║........║.║.║.║.....................................║.║.║.║........║.║.║..║',
  '║.║.║.╔════╗.║.║.║.║.╔═════════════════════════════════╗.║.║.║.║.╔════╗.║.║.║..║',
  '║.║.║.║....║.║.║.║.║.║.................................║.║.║.║.║.║....║.║.║.║..║',
  '║.║.║.║.╔═.║.║.║.║.║.║.╔═════════════════════════════╗.║.║.║.║.║.║.╔═.║.║.║.║..║',
  '║.║.║.║.║..║.║.║.║.║.║.║.............................║.║.║.║.║.║.║.║..║.║.║.║..║',
  '║.║.║.║.║.═╝.║.║.║.║.║.║.╔═════════════════════════╗.║.║.║.║.║.║.║.║.═╣.║.║.║..║',
  '║.║.║.║.║....║.║.║.║.║.║.║.........................║.║.║.║.║.║.║.║....║.║.║.║..║',
  '║.║.║.║.╚════╝.║.║.║.║.║.║.╔═════════════════════╗.║.║.║.║.║.║.║.╚════╝.║.║.║..║',
  '║.║.║.║........║.║.║.║.║.║.║.....................║.║.║.║.║.║.║.║........║.║.║..║',
  '║.║.║.╚════════╝.║.║.║.║.║.║.╔═════════════════╗.║.║.║.║.║.║.║.╚════════╝.║.║..║',
  '║.║.║............║.║.║.║.║.║.║.................║.║.║.║.║.║.║.║............║.║..║',
  '║.║.╚════════════╝.║.║.║.║.║.║.╔═════════════╗.║.║.║.║.║.║.║.╚════════════╝.║..║',
  '║.║................║.║.║.║.║.║.║.............║.║.║.║.║.║.║.║................║..║',
  '║.╠════════════════╝.║.║.║.║.║.║.╔═════════╗.║.║.║.║.║.║.║.╚════════════════╣..║',
  '║.║..................║.║.║.║.║.║.║.........║.║.║.║.║.║.║.║..................║..║',
  '║.║.╔════════════════╝.║.║.║.║.║.║.╔═════╗.║.║.║.║.║.║.║.╚════════════════╗.║..║',
  '║.║.║..................║.║.║.║.║.║.║.....║.║.║.║.║.║.║.║..................║.║..║',
  '║.║.║.╔════════════════╝.║.║.║.║.║.║.╔═╗.║.║.║.║.║.║.║.╚════════════════╗.║.║..║',
  '║.║.║.║..................║.║.║.║.║.║.║S║.║.║.║.║.║.║.║..................║.║.║..║',
  '║.║.║.║.╔════════════════╝.║.║.║.║.║.║.║.║.║.║.║.║.║.╚════════════════╗.║.║.║..║',
  '║.║.║.║.║..................║.║.║.║.║.║...║.║.║.║.║.║..................║.║.║.║..║',
  '║.║.║.║.║.╔════════════════╝.║.║.║.║.╠═══╝.║.║.║.║.║.═════════════════╣.║.║.║..║',
  '║.║.║.║.║.║..................║.║.║.║.║.....║.║.║.║.║..................║.║.║.║..║',
  '║.║.║.║.║.║.╔════════════════╝.║.║.║.╚═════╝.║.║.║.╚════════════════╗.║.║.║.║..║',
  '║.║.║.║.║.║.║..................║.║.║.........║.║.║..................║.║.║.║.║..║',
  '║.║.║.║.║.║.║.╔════════════════╝.║.╚═════════╝.║.╚════════════════╗.║.║.║.║.║..║',
  '║.║.║.║.║.║.║.║..................║................................║.║.║.║.╠═╝..║',
  '║.║.║.║.║.║.║.║.╔════════════════╩═════════════════════════════╗.╔╩╦╩╦╩╦╩╦╩╗...║',
  '║.║.║.║.║.║.║.║.║..............................................║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.╔══════════╗.╔════════════════════════════╗..║.║...║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.║..........║.║............................║..║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.║.╔══════╗.║.║.╔════════════════════════╗.╚╗.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.║.║......║.║.║.║........................║..║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.║.╠══╗.║.║.║.║.╠════════════════════╗..╔╩╗.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.║.║..║.║.║.║.║.║....................║..║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.║.║.═╝.║.║.║.║.║.╔════════════════╗.╚╗.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.║.║....║.║.║.║.║.║................║..║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.║.╚════╝.║.║.║.║.║.╔════════════╗.╚╗.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.║........║.║.║.║.║.║............║..║.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║.╚════════╝.║.║.║.║.║.╔═════════.╠═.║.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.║............║.║.║.║.║.║..........║..║.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.║.║.╚════════════╝.║.║.║.║.║.╔══════╗.╚╗.║.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║....................║.║.║.║.║.║......║..║.║.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║.═══════════════════╝.║.║.║.║.║.╔══╗.╚╗.║.║.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.║.║......................║.║.║.║.║.║..║..║.║.║.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║.╚═╩══.═══════════════════╝.║.║.║.║.║.═╝.╔╝.║.║.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.║............................║.║.║.║.║....║..║.║.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.║.╚═.════════════════════════════╝.║.║.║.═══╩══╩╦╝.║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║....................................║.║.║........║..║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.║.═══════════════════════════════════╝.║.║.═══════╩══╣.║.║.║.║.║.║.║.║.║.║.║.║',
  '║........................................║.║...........║.║.║.║.║.║.║.║.║.║.║.║.║',
  '║.═══════════════════════════════════════╝.╚═══════════╝.║.║.║.║.║.║.║.║.║.║.║.║',
  '║..........................................................║.....║.║.........║.║',
  '╚══════════════════════════════════════════════════════════╩═════╩═╩═════════╣E║',
];

/**
 * Procedurally generate a complex perfect (loop-free) maze with many dead ends.
 *
 * Uses an iterative recursive-backtracker (depth-first search) on a logical cell
 * grid (carving only at odd coordinates) to produce a spanning tree. This yields a
 * maze where every open cell has exactly one simple path to any other (a "perfect"
 * maze). The start 'S' is placed at the central cell; the exit 'E' is chosen as the
 * farthest carved cell from the start (by DFS depth) to maximize solution length.
 *
 * Rendering rules:
 * - Walkable cells become '.' except start 'S' and exit 'E'.
 * - All remaining cells become box-drawing walls synthesized from neighbor walls
 *   for continuous line aesthetics (matching existing static mazes' style).
 * - Outer border is kept solid (no accidental openings) aside from the exit cell
 *   if it lies on the border; otherwise exit will be inside but still reachable.
 *
 * @param width  Overall maze character width (forced to odd, min 5, default 200).
 * @param height Overall maze character height (forced to odd, min 5, default 200).
 * @returns Array of strings representing the maze rows.
 * @example
 * const m = procedural(41, 21);
 * console.log(m.join("\n"));
 */
export class MazeGenerator {
  #width: number;
  #height: number;
  #grid: number[][] = [];
  #startX = 0;
  #startY = 0;
  #farthest: { x: number; y: number; depth: number } = { x: 0, y: 0, depth: 0 };

  // Cell markers
  static readonly WALL = 1;
  static readonly PATH = 0;
  static readonly START = 2;
  static readonly EXIT = 3;

  constructor(rawWidth: number, rawHeight: number) {
    this.#width = rawWidth;
    this.#height = rawHeight;
    this.#normalizeDimensions();
    this.#initializeGrid();
    this.#carvePerfectMaze();
    this.#markStartAndExit();
  }

  /**
   * STEP 1: Normalize requested dimensions.
   * - Floors values.
   * - Enforces minimum size 5.
   * - Forces odd numbers so corridors are one cell thick with surrounding walls.
   */
  #normalizeDimensions(): void {
    this.#width = Math.max(5, Math.floor(this.#width));
    this.#height = Math.max(5, Math.floor(this.#height));
    if (this.#width % 2 === 0) this.#width -= 1;
    if (this.#height % 2 === 0) this.#height -= 1;
  }

  /**
   * STEP 2: Initialize full wall grid and compute central starting cell (odd coordinates).
   */
  #initializeGrid(): void {
    this.#grid = Array.from({ length: this.#height }, () =>
      Array.from({ length: this.#width }, () => MazeGenerator.WALL)
    );
    const toOdd = (value: number) => (value % 2 === 0 ? value - 1 : value);
    this.#startX = toOdd(Math.floor(this.#width / 2));
    this.#startY = toOdd(Math.floor(this.#height / 2));
    this.#grid[this.#startY][this.#startX] = MazeGenerator.PATH;
    this.#farthest = { x: this.#startX, y: this.#startY, depth: 0 };
  }

  /**
   * Helper: bounds check for safe coordinate access.
   */
  #inBounds(column: number, row: number): boolean {
    return (
      column >= 0 && row >= 0 && column < this.#width && row < this.#height
    );
  }

  /**
   * STEP 3: Carve a perfect maze using an iterative recursive-backtracker (depth-first search).
   *
   * Implementation notes:
   * - Uses separate typed-array stacks for x/y/depth to reduce per-iteration
   *   object allocations for large mazes.
   * - Reuses a simple static scratch buffer (grown on demand) to avoid frequent
   *   GC pressure when generating many mazes in a single process.
   * - Shuffles neighbor offsets with Fisher-Yates to ensure unbiased random order.
   *
   * Example:
   * // internal usage only - no external arguments
   * // this.#carvePerfectMaze();
   *
   * Steps:
   * 1. Prepare a typed-array stack with the central starting cell.
   * 2. While stack not empty: inspect top, collect reachable neighbours two cells away.
   * 3. If neighbours exist: pick the first (directions were shuffled), carve wall+cell and push.
   * 4. If no neighbours: backtrack by popping the stack.
   */
  #carvePerfectMaze(): void {
    // --- Tiny typed-array scratch-pool (grow-only) used across instances ---
    // Stored as a regular private static member to keep TS happy with older emit targets.
    if (!(MazeGenerator as any)._scratchBuffers) {
      (MazeGenerator as any)._scratchBuffers = {
        capacity: 0,
        stackX: new Int32Array(0),
        stackY: new Int32Array(0),
        stackDepth: new Int32Array(0),
      };
    }
    const scratch = (MazeGenerator as any)._scratchBuffers as {
      capacity: number;
      stackX: Int32Array;
      stackY: Int32Array;
      stackDepth: Int32Array;
    };

    // Estimate a safe stack capacity: quarter of cells (every second cell both axes), minimum 1024
    const estimatedCapacity = Math.max(
      1024,
      ((this.#width * this.#height) >> 2) + 1
    );
    if (scratch.capacity < estimatedCapacity) {
      scratch.capacity = estimatedCapacity;
      scratch.stackX = new Int32Array(estimatedCapacity);
      scratch.stackY = new Int32Array(estimatedCapacity);
      scratch.stackDepth = new Int32Array(estimatedCapacity);
    }

    // Initialize stack with the central start cell.
    let stackSize = 0;
    scratch.stackX[stackSize] = this.#startX;
    scratch.stackY[stackSize] = this.#startY;
    scratch.stackDepth[stackSize] = 0;
    stackSize++;

    // Reusable neighbor offset table (cardinal directions two cells away).
    const neighborOffsets = [
      { deltaX: 0, deltaY: -2 },
      { deltaX: 2, deltaY: 0 },
      { deltaX: 0, deltaY: 2 },
      { deltaX: -2, deltaY: 0 },
    ];

    // Local helpers for clarity
    const shuffleInPlace = (
      array: Array<{ deltaX: number; deltaY: number }>
    ) => {
      for (let i = array.length - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        if (i !== j) {
          const tmp = array[i];
          array[i] = array[j];
          array[j] = tmp;
        }
      }
    };

    // Candidate buffers (small fixed-size ephemeral arrays to avoid allocations)
    const candidateNextX = new Int32Array(4);
    const candidateNextY = new Int32Array(4);
    const candidateWallX = new Int32Array(4);
    const candidateWallY = new Int32Array(4);

    // Main iterative carve loop
    while (stackSize > 0) {
      const topIndex = stackSize - 1;
      const currentX = scratch.stackX[topIndex];
      const currentY = scratch.stackY[topIndex];
      const currentDepth = scratch.stackDepth[topIndex];

      // Shuffle direction order each iteration for randomized carving patterns.
      shuffleInPlace(neighborOffsets);

      // Collect unvisited neighbors (two cells away) into the small candidate buffers.
      let candidateCount = 0;
      for (let dirIndex = 0; dirIndex < neighborOffsets.length; dirIndex++) {
        const offset = neighborOffsets[dirIndex];
        const nextX = currentX + offset.deltaX;
        const nextY = currentY + offset.deltaY;

        // Skip out-of-bounds or border-adjacent coordinates to keep outer frame intact.
        if (!this.#inBounds(nextX, nextY)) continue;
        if (
          nextX <= 0 ||
          nextY <= 0 ||
          nextX >= this.#width - 1 ||
          nextY >= this.#height - 1
        )
          continue;
        if (this.#grid[nextY][nextX] !== MazeGenerator.WALL) continue; // already carved

        candidateNextX[candidateCount] = nextX;
        candidateNextY[candidateCount] = nextY;
        candidateWallX[candidateCount] = currentX + offset.deltaX / 2;
        candidateWallY[candidateCount] = currentY + offset.deltaY / 2;
        candidateCount++;
      }

      if (candidateCount === 0) {
        // Backtrack when no unvisited neighbours exist.
        stackSize--;
        continue;
      }

      // Choose the first candidate (directions were shuffled), carve corridor and push new cell.
      const chosenIndex = 0;
      const chosenNextX = candidateNextX[chosenIndex];
      const chosenNextY = candidateNextY[chosenIndex];
      const chosenWallX = candidateWallX[chosenIndex];
      const chosenWallY = candidateWallY[chosenIndex];

      // Carve the intermediary wall cell and the destination cell.
      this.#grid[chosenWallY][chosenWallX] = MazeGenerator.PATH;
      this.#grid[chosenNextY][chosenNextX] = MazeGenerator.PATH;

      const newDepth = currentDepth + 1;

      // Push chosen cell onto the typed-array stack.
      scratch.stackX[stackSize] = chosenNextX;
      scratch.stackY[stackSize] = chosenNextY;
      scratch.stackDepth[stackSize] = newDepth;
      stackSize++;

      // Track farthest carved cell by DFS depth to place the exit later.
      if (newDepth > this.#farthest.depth) {
        this.#farthest = { x: chosenNextX, y: chosenNextY, depth: newDepth };
      }
    }
  }

  /**
   * STEP 4: Mark start and farthest cell (exit) on the grid.
   */
  #markStartAndExit(): void {
    // 1) Mark start at central carved cell.
    this.#grid[this.#startY][this.#startX] = MazeGenerator.START;

    // 2) Carve an exit on the outer frame. We pick the interior path cell adjacent
    //    to the border that is farthest (shortest-path distance) from the start.
    //    Then we replace the bordering wall cell with EXIT to create the literal opening.
    this.#placeEdgeExit();
  }

  /**
   * Compute shortest-path distances (BFS) from the start across carved PATH cells.
   *
   * Implementation details:
   * - Uses a flat Int32Array as the distance grid to reduce nested-array allocations.
   * - Reuses a small process-global scratch allocation for the distance buffer and
   *   BFS queue (grow-on-demand) to minimize GC pressure when generating many mazes.
   * - Converts the flat buffer back to a 2D number[][] before returning to preserve
   *   the original public shape.
   *
   * @returns 2D array of distances (or -1 if unreachable).
   * @example
   * // inside class usage
   * const distances = this.#computeDistances();
   * console.log(distances[this.#startY][this.#startX]); // 0
   */
  #computeDistances(): number[][] {
    // --- Setup / scratch buffers reuse ---
    if (!(MazeGenerator as any)._scratchBuffers) {
      (MazeGenerator as any)._scratchBuffers = {
        capacity: 0,
        stackX: new Int32Array(0),
        stackY: new Int32Array(0),
        stackDepth: new Int32Array(0),
        // BFS-specific buffers
        distancesFlat: new Int32Array(0),
        queueX: new Int32Array(0),
        queueY: new Int32Array(0),
      };
    }
    const scratch = (MazeGenerator as any)._scratchBuffers as {
      capacity: number;
      stackX: Int32Array;
      stackY: Int32Array;
      stackDepth: Int32Array;
      distancesFlat: Int32Array;
      queueX: Int32Array;
      queueY: Int32Array;
    };

    // Defensive: previous code paths may have initialized a partial
    // `_scratchBuffers` object (e.g., carving stage). Ensure BFS-specific
    // typed arrays exist so `.length` checks below are safe.
    if (!('distancesFlat' in scratch) || !scratch.distancesFlat) {
      (scratch as any).distancesFlat = new Int32Array(0);
    }
    if (!('queueX' in scratch) || !scratch.queueX) {
      (scratch as any).queueX = new Int32Array(0);
    }
    if (!('queueY' in scratch) || !scratch.queueY) {
      (scratch as any).queueY = new Int32Array(0);
    }

    const totalCells = this.#width * this.#height;
    // Grow flat buffers if needed (grow-only to keep reuse simple)
    if (scratch.distancesFlat.length < totalCells) {
      scratch.distancesFlat = new Int32Array(totalCells);
    }
    if (scratch.queueX.length < totalCells) {
      scratch.queueX = new Int32Array(totalCells);
      scratch.queueY = new Int32Array(totalCells);
    }

    const distancesFlat = scratch.distancesFlat;

    // STEP 1: Initialize distances to -1 (unvisited). Use typed-array fill for speed.
    distancesFlat.fill(-1);

    // STEP 2: BFS queue implemented with two parallel typed arrays (x/y) and read/write indices.
    let writeIndex = 0;
    let readIndex = 0;
    scratch.queueX[writeIndex] = this.#startX;
    scratch.queueY[writeIndex] = this.#startY;
    distancesFlat[this.#startY * this.#width + this.#startX] = 0;
    writeIndex++;

    // STEP 3: BFS loop - visit neighbors in 4 directions and set distances.
    while (readIndex < writeIndex) {
      const currentX = scratch.queueX[readIndex];
      const currentY = scratch.queueY[readIndex];
      readIndex++;
      const currentIndex = currentY * this.#width + currentX;
      const baseDistance = distancesFlat[currentIndex];

      // Explore 4-neighbors in cardinal order.
      // Each neighbor: check bounds, check cell type, check unvisited, then enqueue.
      // North
      const nx0 = currentX;
      const ny0 = currentY - 1;
      if (this.#inBounds(nx0, ny0)) {
        const ni = ny0 * this.#width + nx0;
        const cellValue = this.#grid[ny0][nx0];
        if (
          (cellValue === MazeGenerator.PATH ||
            cellValue === MazeGenerator.START) &&
          distancesFlat[ni] === -1
        ) {
          distancesFlat[ni] = baseDistance + 1;
          scratch.queueX[writeIndex] = nx0;
          scratch.queueY[writeIndex] = ny0;
          writeIndex++;
        }
      }

      // East
      const nx1 = currentX + 1;
      const ny1 = currentY;
      if (this.#inBounds(nx1, ny1)) {
        const ni = ny1 * this.#width + nx1;
        const cellValue = this.#grid[ny1][nx1];
        if (
          (cellValue === MazeGenerator.PATH ||
            cellValue === MazeGenerator.START) &&
          distancesFlat[ni] === -1
        ) {
          distancesFlat[ni] = baseDistance + 1;
          scratch.queueX[writeIndex] = nx1;
          scratch.queueY[writeIndex] = ny1;
          writeIndex++;
        }
      }

      // South
      const nx2 = currentX;
      const ny2 = currentY + 1;
      if (this.#inBounds(nx2, ny2)) {
        const ni = ny2 * this.#width + nx2;
        const cellValue = this.#grid[ny2][nx2];
        if (
          (cellValue === MazeGenerator.PATH ||
            cellValue === MazeGenerator.START) &&
          distancesFlat[ni] === -1
        ) {
          distancesFlat[ni] = baseDistance + 1;
          scratch.queueX[writeIndex] = nx2;
          scratch.queueY[writeIndex] = ny2;
          writeIndex++;
        }
      }

      // West
      const nx3 = currentX - 1;
      const ny3 = currentY;
      if (this.#inBounds(nx3, ny3)) {
        const ni = ny3 * this.#width + nx3;
        const cellValue = this.#grid[ny3][nx3];
        if (
          (cellValue === MazeGenerator.PATH ||
            cellValue === MazeGenerator.START) &&
          distancesFlat[ni] === -1
        ) {
          distancesFlat[ni] = baseDistance + 1;
          scratch.queueX[writeIndex] = nx3;
          scratch.queueY[writeIndex] = ny3;
          writeIndex++;
        }
      }
    }

    // STEP 4: Convert flat Int32Array distances back to number[][] shape for compatibility.
    const distances2D: number[][] = new Array(this.#height);
    for (let row = 0; row < this.#height; row++) {
      const rowStart = row * this.#width;
      const rowArray: number[] = new Array(this.#width);
      for (let col = 0; col < this.#width; col++) {
        rowArray[col] = distancesFlat[rowStart + col];
      }
      distances2D[row] = rowArray;
    }
    return distances2D;
  }

  /**
   * Place an exit on the outer frame adjacent to the carved path cell that has
   * the maximum BFS distance from the start.
   *
   * Implementation notes / props:
   * - Reuses the flat Int32Array produced in #computeDistances (scratch.distancesFlat)
   *   to avoid creating a full list of candidate objects.
   * - Scans interior cells adjacent to the border in a single pass and selects the
   *   candidate with the largest distance value. Tie-breaking is stable (first seen).
   * - If no valid border-adjacent interior path is found, falls back to the internal
   *   farthest cell previously recorded by the maze-carving DFS.
   *
   * @example
   * // internal usage (no args): this.#placeEdgeExit();
   */
  #placeEdgeExit(): void {
    // Ensure distances are computed and available on the shared scratch buffer.
    const distances2D = this.#computeDistances();
    // Access the flat distances buffer directly for scanning (avoid extra allocations).
    const scratch = (MazeGenerator as any)._scratchBuffers as
      | { distancesFlat: Int32Array }
      | undefined;
    const distancesFlat =
      scratch && scratch.distancesFlat ? scratch.distancesFlat : null;

    // Track the best candidate found during a single pass.
    let bestDistance = -1;
    let bestInteriorX = -1;
    let bestInteriorY = -1;
    let bestBorderX = -1;
    let bestBorderY = -1;

    // Helper: read distance either from flat buffer (fast) or from the 2D array fallback.
    const readDistance = (ix: number, iy: number): number => {
      if (distancesFlat) return distancesFlat[iy * this.#width + ix];
      return distances2D[iy][ix];
    };

    // STEP 1: scan interior cells adjacent to the outer border and pick max-distance.
    for (let interiorY = 1; interiorY < this.#height - 1; interiorY++) {
      for (let interiorX = 1; interiorX < this.#width - 1; interiorX++) {
        // Skip unreachable cells and the start cell.
        const currentDistance = readDistance(interiorX, interiorY);
        if (currentDistance < 0) continue;
        if (this.#grid[interiorY][interiorX] === MazeGenerator.START) continue;

        // For each border-adjacent direction, consider opening the border cell.
        // Check north-adjacent to top border
        if (interiorY === 1) {
          const borderX = interiorX;
          const borderY = 0;
          if (
            this.#grid[borderY][borderX] === MazeGenerator.WALL &&
            currentDistance > bestDistance
          ) {
            bestDistance = currentDistance;
            bestInteriorX = interiorX;
            bestInteriorY = interiorY;
            bestBorderX = borderX;
            bestBorderY = borderY;
          }
        }

        // Check south-adjacent to bottom border
        if (interiorY === this.#height - 2) {
          const borderX = interiorX;
          const borderY = this.#height - 1;
          if (
            this.#grid[borderY][borderX] === MazeGenerator.WALL &&
            currentDistance > bestDistance
          ) {
            bestDistance = currentDistance;
            bestInteriorX = interiorX;
            bestInteriorY = interiorY;
            bestBorderX = borderX;
            bestBorderY = borderY;
          }
        }

        // Check west-adjacent to left border
        if (interiorX === 1) {
          const borderX = 0;
          const borderY = interiorY;
          if (
            this.#grid[borderY][borderX] === MazeGenerator.WALL &&
            currentDistance > bestDistance
          ) {
            bestDistance = currentDistance;
            bestInteriorX = interiorX;
            bestInteriorY = interiorY;
            bestBorderX = borderX;
            bestBorderY = borderY;
          }
        }

        // Check east-adjacent to right border
        if (interiorX === this.#width - 2) {
          const borderX = this.#width - 1;
          const borderY = interiorY;
          if (
            this.#grid[borderY][borderX] === MazeGenerator.WALL &&
            currentDistance > bestDistance
          ) {
            bestDistance = currentDistance;
            bestInteriorX = interiorX;
            bestInteriorY = interiorY;
            bestBorderX = borderX;
            bestBorderY = borderY;
          }
        }
      }
    }

    // STEP 2: Apply choice or fallback.
    if (bestDistance < 0) {
      // No border-adjacent candidate found; use previously recorded farthest internal cell.
      this.#grid[this.#farthest.y][this.#farthest.x] = MazeGenerator.EXIT;
      return;
    }

    // Ensure interior cell is a PATH (avoid turning an existing EXIT into EXIT twice).
    if (this.#grid[bestInteriorY][bestInteriorX] === MazeGenerator.EXIT) {
      this.#grid[bestInteriorY][bestInteriorX] = MazeGenerator.PATH;
    }

    // Open the border cell as the maze exit.
    this.#grid[bestBorderY][bestBorderX] = MazeGenerator.EXIT;
  }

  /**
   * STEP 5: Derive box drawing wall character from neighboring wall continuity.
   * Considers the four cardinal neighbors as wall/not-wall and maps bitmask to glyph.
   * Falls back to a solid block if an unexpected isolated pattern appears.
   */
  /**
   * Determine the box-drawing glyph for a wall cell by inspecting adjacent
   * wall continuity in the four cardinal directions. Uses a 4-bit mask where
   * bit 0 = north, bit 1 = east, bit 2 = south, bit 3 = west.
   *
   * Props:
   * - column: x coordinate (0-based) of the wall cell to evaluate.
   * - row: y coordinate (0-based) of the wall cell to evaluate.
   *
   * Performance:
   * - Reuses a small pooled Int8Array from the class-level scratch buffers to
   *   avoid allocating a fresh array on each call, which matters when rendering
   *   large mazes in tight loops.
   *
   * Example:
   * const glyph = generator.#wallGlyph(5, 3);
   * // glyph === '╬' // four-way junction
   *
   * @param column - column index of the cell to evaluate
   * @param row - row index of the cell to evaluate
   * @returns single-character box-drawing glyph representing the wall shape
   */
  #wallGlyph(column: number, row: number): string {
    // --- STEP 1: Prepare a tiny pooled buffer for neighbor flags ---
    if (!(MazeGenerator as any)._scratchBuffers) {
      (MazeGenerator as any)._scratchBuffers = {} as any;
    }
    const scratch = (MazeGenerator as any)._scratchBuffers as {
      maskFlags?: Int8Array;
    };
    if (!scratch.maskFlags) scratch.maskFlags = new Int8Array(4);

    // Helper: treat any non-PATH/START/EXIT as a wall for rendering.
    const isNeighborWall = (col: number, rw: number): boolean =>
      this.#inBounds(col, rw) &&
      ![MazeGenerator.PATH, MazeGenerator.START, MazeGenerator.EXIT].includes(
        this.#grid[rw][col]
      );

    // --- STEP 2: Compute neighbor wall presence (explicit descriptive names) ---
    // North
    const hasNorthWall = isNeighborWall(column, row - 1);
    // East
    const hasEastWall = isNeighborWall(column + 1, row);
    // South
    const hasSouthWall = isNeighborWall(column, row + 1);
    // West
    const hasWestWall = isNeighborWall(column - 1, row);

    // Store boolean flags into the pooled Int8Array to avoid ephemeral allocations
    scratch.maskFlags[0] = hasNorthWall ? 1 : 0;
    scratch.maskFlags[1] = hasEastWall ? 1 : 0;
    scratch.maskFlags[2] = hasSouthWall ? 1 : 0;
    scratch.maskFlags[3] = hasWestWall ? 1 : 0;

    // --- STEP 3: Build the 4-bit mask (bit order: N=1, E=2, S=4, W=8) ---
    const maskValue =
      (scratch.maskFlags[0] ? 1 : 0) |
      (scratch.maskFlags[1] ? 2 : 0) |
      (scratch.maskFlags[2] ? 4 : 0) |
      (scratch.maskFlags[3] ? 8 : 0);

    // --- STEP 4: Map mask to Unicode box-drawing glyphs ---
    switch (maskValue) {
      // Vertical line
      case 0b0101:
      case 0b0001:
      case 0b0100:
        return '║';

      // Horizontal line
      case 0b1010:
      case 0b0010:
      case 0b1000:
        return '═';

      // Corners
      case 0b0011:
        return '╚';
      case 0b1001:
        return '╝';
      case 0b0110:
        return '╔';
      case 0b1100:
        return '╗';

      // T-junctions and cross
      case 0b1110:
        return '╦';
      case 0b1011:
        return '╩';
      case 0b0111:
        return '╠';
      case 0b1101:
        return '╣';
      case 0b1111:
        return '╬';

      // Fallback: isolated/irregular wall -> solid block
      default:
        return '█';
    }
  }

  /**
   * STEP 6: Render final ASCII maze lines.
   * Preserves a continuous rectangular outer frame using the standard corner/edge glyphs.
   */
  /**
   * Render the internal numeric grid into ASCII maze rows.
   *
   * Steps (high-level):
   * 1) Iterate rows and columns of the internal grid.
   * 2) For each cell, choose a glyph based on the cell marker or outer-frame
   *    position. We prefer a small switch-based decision tree instead of a
   *    long else-if cascade for readability and clearer intent.
   * 3) For interior wall cells, delegate to `#wallGlyph` which already
   *    reuses a pooled scratch buffer for neighbor inspection.
   *
   * Performance note: This method focuses on string assembly. The most
   * expensive per-cell work (neighbor inspection and mask creation) is
   * performed inside `#wallGlyph` and already benefits from typed-array
   * pooling; adding additional pooling here gives negligible benefit and
   * would complicate the implementation.
   *
   * Example:
   * const rows = generator.generate();
   * console.log(rows.join('\n'));
   *
   * @returns array of rendered ASCII maze rows (strings)
   */
  #render(): string[] {
    const renderedLines: string[] = [];

    // Iterate each row and column, assembling one string per row.
    for (let row = 0; row < this.#height; row++) {
      let line = '';
      for (let col = 0; col < this.#width; col++) {
        const cellValue = this.#grid[row][col];

        // Use a switch(true) pattern to replace the previous else-if chain.
        // Each case is a clear, self-documenting condition.
        switch (true) {
          // Open path
          case cellValue === MazeGenerator.PATH:
            line += '.';
            break;

          // Start / Exit markers
          case cellValue === MazeGenerator.START:
            line += 'S';
            break;
          case cellValue === MazeGenerator.EXIT:
            line += 'E';
            break;

          // Outer frame corners
          case row === 0 && col === 0:
            line += '╔';
            break;
          case row === 0 && col === this.#width - 1:
            line += '╗';
            break;
          case row === this.#height - 1 && col === 0:
            line += '╚';
            break;
          case row === this.#height - 1 && col === this.#width - 1:
            line += '╝';
            break;

          // Outer horizontal edges (top/bottom)
          case row === 0 || row === this.#height - 1:
            line += '═';
            break;

          // Outer vertical edges (left/right)
          case col === 0 || col === this.#width - 1:
            line += '║';
            break;

          // Interior walls: delegate to wall glyph generator (pooled internally)
          default:
            line += this.#wallGlyph(col, row);
        }
      }
      renderedLines.push(line);
    }

    return renderedLines;
  }

  /**
   * Public entry: returns the rendered maze as string lines.
   */
  generate(): string[] {
    return this.#render();
  }
}
