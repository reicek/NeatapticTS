/**
 * Shared constants for the browser-hosted ASCII Maze demo lifecycle.
 *
 * These values keep the browser entry facade declarative while the host,
 * resize, and curriculum services consume a single named configuration table.
 * Read them as three teaching-focused families rather than as an arbitrary bag
 * of numbers:
 *
 * - curriculum ladder knobs such as `INITIAL_MAZE_DIMENSION`,
 *   `MAX_MAZE_DIMENSION`, and `MAZE_DIMENSION_INCREMENT`,
 * - per-phase runtime budgets such as `AGENT_MAX_STEPS`,
 *   `DEFAULT_MAX_GENERATIONS`, and stagnation limits,
 * - browser-host pacing settings such as auto-start delay and redraw debounce.
 *
 * If you want the hosted demo to begin with smaller mazes, push farther into
 * larger ones, or spend more time inside each maze before a phase is judged,
 * this table is the intended first stop.
 */
/**
 * Browser-hosted ASCII Maze curriculum knobs and host pacing defaults.
 *
 * The most frequently tuned values are the maze-size ladder and the movement
 * budget. `INITIAL_MAZE_DIMENSION` decides where the browser curriculum begins,
 * `MAX_MAZE_DIMENSION` decides how far it can grow, `MAZE_DIMENSION_INCREMENT`
 * decides how abruptly solved phases get harder, and `AGENT_MAX_STEPS` decides
 * how much room each candidate gets to explore one maze.
 *
 * @example
 * ```ts
 * const {
 *   INITIAL_MAZE_DIMENSION,
 *   MAX_MAZE_DIMENSION,
 *   MAZE_DIMENSION_INCREMENT,
 *   AGENT_MAX_STEPS,
 * } = BROWSER_ENTRY_CONSTANTS;
 * ```
 */
export const BROWSER_ENTRY_CONSTANTS = {
  DEFAULT_CONTAINER_ID: 'ascii-maze-output',
  ALLOW_RECURRENT: false,
  RESIZE_WIDTH_THRESHOLD: 8,
  RESIZE_DEBOUNCE_MS: 120,
  AUTO_START_DELAY_MS: 20,
  MIN_PROGRESS_TO_PASS: 90,
  DEFAULT_MAX_STAGNANT_GENERATIONS: 50,
  DEFAULT_MAX_GENERATIONS: 100,
  PER_GENERATION_LOG_FREQUENCY: 1,
  INITIAL_MAZE_DIMENSION: 8,
  MAX_MAZE_DIMENSION: 40,
  MAZE_DIMENSION_INCREMENT: 4,
  AGENT_MAX_STEPS: 600,
  POPULATION_SIZE: 100,
  FIRST_PHASE_POPULATION_SIZE: 100,
  FIRST_PHASE_ADAPTIVE_MUTATION: {
    enabled: true,
    strategy: 'twoTier',
    adaptEvery: 5,
    sigma: 0.1,
    minRate: 0.001,
  },
  LAMARCKIAN_ITERATIONS: 4,
  FIRST_PHASE_LAMARCKIAN_ITERATIONS: 8,
  LAMARCKIAN_SAMPLE_SIZE: 12,
  FIRST_PHASE_LAMARCKIAN_SAMPLE_SIZE: 20,
  FIRST_PHASE_MAX_GENERATIONS: 60,
} as const;
