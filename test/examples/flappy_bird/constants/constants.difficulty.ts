import { FLAPPY_BIRD_HEIGHT_PX } from './constants.world';
import {
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_TARGET_FLAP_INTERVAL_FRAMES,
} from './constants.physics';
import { FLAPPY_PIPE_WIDTH_PX } from './constants.pipes';

/**
 * Adaptive-difficulty constants for Flappy runs.
 *
 * This module defines how spacing, speed, and gap variability tighten as
 * agents become more capable. Keeping it separate helps tune curriculum
 * behavior without mixing with core physics or rendering.
 */

/** Small geometric buffer so "barely possible" remains physically solvable. */
export const FLAPPY_MIN_CLEARANCE_MARGIN_PX = 120;

/**
 * Hard floor on time between pipes at max speed so controllers can recover.
 *
 * This prevents endgame spacing from becoming too tight for realistic policy
 * reaction and vertical correction.
 */
export const FLAPPY_MIN_PIPE_RECOVERY_FRAMES = 50;

/**
 * Minimum edge-to-edge spacing needed to recover between consecutive pipes.
 *
 * Derived from bird size + expected control drop budget + a small margin.
 */
export const FLAPPY_MIN_EDGE_TO_EDGE_PIPE_SPACING_PX =
  FLAPPY_BIRD_HEIGHT_PX +
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME * FLAPPY_TARGET_FLAP_INTERVAL_FRAMES +
  FLAPPY_MIN_CLEARANCE_MARGIN_PX;

/** Minimum pipe gap used at peak adaptive difficulty. */
export const FLAPPY_PIPE_GAP_MIN_PX =
  FLAPPY_BIRD_HEIGHT_PX +
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME * FLAPPY_TARGET_FLAP_INTERVAL_FRAMES +
  FLAPPY_MIN_CLEARANCE_MARGIN_PX;

/** Initial spawn gap multiplier relative to the current hardest gap target. */
export const FLAPPY_PIPE_GAP_START_MULTIPLIER = 2.15;

/** Per-pipe gap shrink step toward the current hardest target gap (pixels). */
export const FLAPPY_PIPE_GAP_SHRINK_PER_PIPE_PX = 10;

/** Random jitter range applied to each spawned pipe gap (pixels). */
export const FLAPPY_PIPE_GAP_RANDOM_JITTER_PX = 10;

/**
 * Maximum allowed vertical jump between consecutive pipe gap centers (pixels).
 *
 * This reduces abrupt zig-zag transitions that are often unrecoverable once
 * spacing tightens at higher difficulty.
 */
export const FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX = 100;

/** Maximum pipe speed used at peak adaptive difficulty. */
export const FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME = 3;

/** Minimum spawn interval used at peak adaptive difficulty. */
export const FLAPPY_PIPE_SPAWN_INTERVAL_MIN_FRAMES = Math.max(
  FLAPPY_MIN_PIPE_RECOVERY_FRAMES,
  Math.ceil(
    (FLAPPY_PIPE_WIDTH_PX + FLAPPY_MIN_EDGE_TO_EDGE_PIPE_SPACING_PX) /
      FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME,
  ),
);

/** Initial spawn-interval multiplier relative to the current hardest interval target. */
export const FLAPPY_PIPE_SPAWN_INTERVAL_START_MULTIPLIER = 2.35;

/** Per-pipe spawn-interval shrink step toward the current hardest interval target (frames). */
export const FLAPPY_PIPE_SPAWN_INTERVAL_SHRINK_PER_PIPE_FRAMES = 2;

/**
 * Pipe-pass count needed to reach maximum adaptive difficulty.
 *
 * After this point, spacing does not tighten further, so proficient agents can
 * sustain long runs without additional spacing compression.
 */
export const FLAPPY_DIFFICULTY_RAMP_PIPES = 25;
