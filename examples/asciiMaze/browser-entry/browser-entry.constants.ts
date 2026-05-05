/**
 * Shared constants for the browser-hosted ASCII Maze demo lifecycle.
 *
 * These values keep the browser entry facade declarative while the host,
 * resize, and curriculum services consume a single named configuration table.
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
