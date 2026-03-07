import type { SharedObservationFeatures } from '../flappy.simulation.shared.utils';

/**
 * Pipe obstacle definition.
 *
 * Pipes move from right to left. The bird scores once per pipe when the pipe
 * completely crosses the bird x-position.
 */
export interface FlappyPipe {
  /** Horizontal position of the left edge (pixels). */
  xPx: number;

  /** Vertical center of the opening/gap (pixels). */
  gapCenterYPx: number;

  /** Vertical gap size used by this specific pipe (pixels). */
  gapSizePx: number;

  /** Whether the bird has already been credited for passing this pipe. */
  passed: boolean;
}

/**
 * Bird kinematic state for one simulation frame.
 */
export interface FlappyBird {
  /** Vertical position (pixels). */
  yPx: number;

  /** Vertical velocity (pixels/frame). */
  velocityYPxPerFrame: number;
}

/**
 * Full simulation state for one Flappy episode.
 */
export interface FlappyGameState {
  /** Current frame counter (0-based). */
  frameIndex: number;

  /** Bird state. */
  bird: FlappyBird;

  /** Pipe obstacles ordered by increasing x (left to right). */
  pipes: FlappyPipe[];

  /** Gap size of the most recently spawned pipe (pixels). */
  lastSpawnedPipeGapPx: number;

  /** Gap-center y value of the most recently spawned pipe (pixels). */
  lastSpawnedPipeGapCenterYPx: number;

  /** Spawn interval used for the most recently spawned pipe (frames). */
  lastSpawnedPipeSpawnIntervalFrames: number;

  /** Countdown until the next pipe spawn (frames). */
  framesUntilNextPipeSpawn: number;

  /** Total number of pipes passed. */
  pipesPassed: number;

  /** Whether the episode has terminated. */
  done: boolean;

  /**
   * Termination reason (diagnostic only).
   * - `collision`: hit a pipe
   * - `out_of_bounds`: hit floor/ceiling
   * - `timeout`: exceeded max frame limit
   */
  doneReason?: 'collision' | 'out_of_bounds' | 'timeout';
}

/**
 * Structured observation features used to build the neural-network input vector.
 *
 * Re-exported from shared simulation utilities so trainer and browser paths
 * stay synchronized as the observation schema evolves.
 */
export type FlappyObservationFeatures = SharedObservationFeatures;

/**
 * Difficulty scale used by the curriculum scheduler.
 *
 * - `0` means easiest profile (wide gaps, slower pipes).
 * - `1` means fully adaptive profile based on passed pipes.
 */
export type FlappyDifficultyScale = number;
