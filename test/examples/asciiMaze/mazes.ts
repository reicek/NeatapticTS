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
   * STEP 3: Carve a perfect maze using an iterative depth-first search (recursive backtracker).
   * Maintains a stack of frontier cells; at each step chooses a random unvisited neighbor
   * two cells away, carving both the intermediary wall cell and the target cell. Tracks
   * the farthest cell (by depth) from the start for later exit placement.
   */
  #carvePerfectMaze(): void {
    interface PendingCell {
      x: number;
      y: number;
      depth: number;
    }
    const stack: PendingCell[] = [
      { x: this.#startX, y: this.#startY, depth: 0 },
    ];

    while (stack.length) {
      const current = stack[stack.length - 1];
      // Gather candidate neighbors two cells away.
      const neighborDirections: Array<{ deltaX: number; deltaY: number }> = [
        { deltaX: 0, deltaY: -2 },
        { deltaX: 2, deltaY: 0 },
        { deltaX: 0, deltaY: 2 },
        { deltaX: -2, deltaY: 0 },
      ];
      // Shuffle directions using an in-place Fisher-Yates to avoid sort comparator bias.
      // NOTE: The previous implementation used Array.sort(() => Math.random() - 0.5) which
      // yields heavily biased permutations (particularly with only 4 elements) and in practice
      // was producing an alternating pattern of two mazes between page reloads in some browsers.
      for (
        let remaining = neighborDirections.length - 1;
        remaining > 0;
        remaining--
      ) {
        const randomFloat = Math.random();
        const swapIndex = (randomFloat * (remaining + 1)) | 0; // floor
        if (swapIndex !== remaining) {
          const tmp = neighborDirections[remaining];
          neighborDirections[remaining] = neighborDirections[swapIndex];
          neighborDirections[swapIndex] = tmp;
        }
      }

      const unvisited: Array<{
        nextX: number;
        nextY: number;
        wallX: number;
        wallY: number;
      }> = [];
      for (const direction of neighborDirections) {
        const nextX = current.x + direction.deltaX;
        const nextY = current.y + direction.deltaY;
        if (!this.#inBounds(nextX, nextY)) continue;
        if (
          nextX <= 0 ||
          nextY <= 0 ||
          nextX >= this.#width - 1 ||
          nextY >= this.#height - 1
        )
          continue; // preserve outer frame
        if (this.#grid[nextY][nextX] !== MazeGenerator.WALL) continue;
        unvisited.push({
          nextX,
          nextY,
          wallX: current.x + direction.deltaX / 2,
          wallY: current.y + direction.deltaY / 2,
        });
      }

      if (unvisited.length === 0) {
        stack.pop();
        continue;
      }

      // Pick first candidate (already randomized), carve corridor between.
      const chosen = unvisited[0];
      this.#grid[chosen.wallY][chosen.wallX] = MazeGenerator.PATH;
      this.#grid[chosen.nextY][chosen.nextX] = MazeGenerator.PATH;
      const depth = current.depth + 1;
      stack.push({ x: chosen.nextX, y: chosen.nextY, depth });
      if (depth > this.#farthest.depth)
        this.#farthest = { x: chosen.nextX, y: chosen.nextY, depth };
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
   * @returns 2D array of distances (or -1 if unreachable).
   */
  #computeDistances(): number[][] {
    const distances: number[][] = Array.from({ length: this.#height }, () =>
      Array.from({ length: this.#width }, () => -1)
    );
    const queue: Array<{ x: number; y: number } | undefined> = [];
    queue.push({ x: this.#startX, y: this.#startY });
    distances[this.#startY][this.#startX] = 0;
    let readIndex = 0; // efficient queue (avoid shift)
    while (readIndex < queue.length) {
      const current = queue[readIndex++]!;
      const baseDistance = distances[current.y][current.x];
      // Explore 4-neighbors.
      const neighbors = [
        { x: current.x, y: current.y - 1 },
        { x: current.x + 1, y: current.y },
        { x: current.x, y: current.y + 1 },
        { x: current.x - 1, y: current.y },
      ];
      for (const neighbor of neighbors) {
        if (!this.#inBounds(neighbor.x, neighbor.y)) continue;
        const cell = this.#grid[neighbor.y][neighbor.x];
        if (
          ![MazeGenerator.PATH, MazeGenerator.START].includes(cell) ||
          distances[neighbor.y][neighbor.x] !== -1
        )
          continue;
        distances[neighbor.y][neighbor.x] = baseDistance + 1;
        queue.push(neighbor);
      }
    }
    return distances;
  }

  /**
   * Select a border-adjacent interior path cell maximizing distance from the start
   * and open an actual edge (outer frame) cell, marking that outer cell as EXIT.
   * If no interior candidate is found (degenerate tiny maze), fallback to previous
   * farthest internal cell marking.
   */
  #placeEdgeExit(): void {
    const distances = this.#computeDistances();
    interface Candidate {
      interiorX: number;
      interiorY: number;
      borderX: number;
      borderY: number;
      distance: number;
    }
    const candidates: Candidate[] = [];
    // Scan interior cells adjacent to outer border.
    for (let y = 1; y < this.#height - 1; y++) {
      for (let x = 1; x < this.#width - 1; x++) {
        if (distances[y][x] < 0) continue; // not reachable path
        if (this.#grid[y][x] === MazeGenerator.START) continue; // skip start
        // Check adjacency to each border direction and ensure border cell currently wall.
        if (y === 1) {
          candidates.push({
            interiorX: x,
            interiorY: y,
            borderX: x,
            borderY: 0,
            distance: distances[y][x],
          });
        }
        if (y === this.#height - 2) {
          candidates.push({
            interiorX: x,
            interiorY: y,
            borderX: x,
            borderY: this.#height - 1,
            distance: distances[y][x],
          });
        }
        if (x === 1) {
          candidates.push({
            interiorX: x,
            interiorY: y,
            borderX: 0,
            borderY: y,
            distance: distances[y][x],
          });
        }
        if (x === this.#width - 2) {
          candidates.push({
            interiorX: x,
            interiorY: y,
            borderX: this.#width - 1,
            borderY: y,
            distance: distances[y][x],
          });
        }
      }
    }
    if (candidates.length === 0) {
      // Fallback: mark previously stored farthest cell (internal) if edge impossible.
      this.#grid[this.#farthest.y][this.#farthest.x] = MazeGenerator.EXIT;
      return;
    }
    // Choose candidate with max distance (tie-breaker: deterministic order stable)
    candidates.sort((a, b) => b.distance - a.distance);
    const chosen = candidates[0];
    // Mark interior cell as path (ensure not spuriously EXIT) and open border cell as EXIT.
    if (this.#grid[chosen.interiorY][chosen.interiorX] === MazeGenerator.EXIT)
      this.#grid[chosen.interiorY][chosen.interiorX] = MazeGenerator.PATH;
    this.#grid[chosen.borderY][chosen.borderX] = MazeGenerator.EXIT;
  }

  /**
   * STEP 5: Derive box drawing wall character from neighboring wall continuity.
   * Considers the four cardinal neighbors as wall/not-wall and maps bitmask to glyph.
   * Falls back to a solid block if an unexpected isolated pattern appears.
   */
  #wallGlyph(column: number, row: number): string {
    const isWall = (c: number, r: number) =>
      this.#inBounds(c, r) &&
      ![MazeGenerator.PATH, MazeGenerator.START, MazeGenerator.EXIT].includes(
        this.#grid[r][c]
      );
    const north = isWall(column, row - 1);
    const east = isWall(column + 1, row);
    const south = isWall(column, row + 1);
    const west = isWall(column - 1, row);
    const mask =
      (north ? 1 : 0) | (east ? 2 : 0) | (south ? 4 : 0) | (west ? 8 : 0);
    switch (mask) {
      case 0b0101:
      case 0b0001:
      case 0b0100:
        return '║';
      case 0b1010:
      case 0b0010:
      case 0b1000:
        return '═';
      case 0b0011:
        return '╚';
      case 0b1001:
        return '╝';
      case 0b0110:
        return '╔';
      case 0b1100:
        return '╗';
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
      default:
        return '█';
    }
  }

  /**
   * STEP 6: Render final ASCII maze lines.
   * Preserves a continuous rectangular outer frame using the standard corner/edge glyphs.
   */
  #render(): string[] {
    const lines: string[] = [];
    for (let rowIndex = 0; rowIndex < this.#height; rowIndex++) {
      let rendered = '';
      for (let columnIndex = 0; columnIndex < this.#width; columnIndex++) {
        const cellValue = this.#grid[rowIndex][columnIndex];
        if (cellValue === MazeGenerator.PATH) rendered += '.';
        else if (cellValue === MazeGenerator.START) rendered += 'S';
        else if (cellValue === MazeGenerator.EXIT) rendered += 'E';
        else if (rowIndex === 0 && columnIndex === 0) rendered += '╔';
        else if (rowIndex === 0 && columnIndex === this.#width - 1)
          rendered += '╗';
        else if (rowIndex === this.#height - 1 && columnIndex === 0)
          rendered += '╚';
        else if (
          rowIndex === this.#height - 1 &&
          columnIndex === this.#width - 1
        )
          rendered += '╝';
        else if (rowIndex === 0 || rowIndex === this.#height - 1)
          rendered += '═';
        else if (columnIndex === 0 || columnIndex === this.#width - 1)
          rendered += '║';
        else rendered += this.#wallGlyph(columnIndex, rowIndex);
      }
      lines.push(rendered);
    }
    return lines;
  }

  /**
   * Public entry: returns the rendered maze as string lines.
   */
  generate(): string[] {
    return this.#render();
  }
}
