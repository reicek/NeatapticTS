import { FLAPPY_WORLD_HEIGHT_PX } from './constants.world';

/**
 * Pipe geometry and baseline pacing constants.
 *
 * These values describe the canonical pipe body size, opening geometry, visual
 * outline offsets, and default leftward movement cadence.
 *
 * Educational note:
 * Pipe constants do double duty: some values describe actual gameplay geometry,
 * while others are chosen so the visible pipe outline and the collision envelope
 * feel aligned to a human viewer.
 */

/** Pipe width (pixels). */
export const FLAPPY_PIPE_WIDTH_PX = 60;

/** Visual gap between the pipe body and its outline on the sides (pixels). */
export const FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX = 2;

/** Visual gap between the pipe body and its outline at the pipe entrance rim (pixels). */
export const FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX = 5;

/** Stroke width used for the pipe outline (pixels). */
export const FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX = 2;

/**
 * Effective side expansion used for pipe collision checks (pixels).
 *
 * Includes the visual side gap and half the outline stroke so collision
 * matches the visible outer line thickness.
 */
export const FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX = Math.round(
  FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX + FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX * 0.5,
);

/**
 * Effective gap-rim expansion used for pipe collision checks (pixels).
 *
 * Includes the visual entrance gap and half the outline stroke so collision
 * matches the visible outer line around the gap opening.
 */
export const FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX = Math.round(
  FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX +
    FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX * 0.5,
);

/**
 * Vertical opening size of each pipe gap (pixels).
 *
 * This is the nominal baseline gap before adaptive difficulty narrows it.
 */
export const FLAPPY_PIPE_GAP_PX = 150;

/**
 * Pipe horizontal speed (pixels/frame).
 *
 * Higher values shrink reaction time, which makes the same gap geometry much
 * harder even before the adaptive curriculum starts tightening gaps.
 */
export const FLAPPY_PIPE_SPEED_PX_PER_FRAME = 5;

/**
 * Frames between spawning new pipes.
 *
 * Together with pipe speed, this controls horizontal pacing and how much time a
 * policy has to recover between obstacles.
 */
export const FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES = 50;

/** Minimum allowed gap center height (pixels). */
export const FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX = 100;

/** Maximum allowed gap center height (pixels). */
export const FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX = FLAPPY_WORLD_HEIGHT_PX - 100;

/**
 * Minimum fraction of world height that must remain as solid pipe above and
 * below the gap opening.
 *
 * At the world height of 512 px this resolves to ≈26 px of visible pipe cap on
 * each side, which prevents the opening from clipping into or touching the
 * canvas edge — especially important when the initial wide gap is active.
 */
export const FLAPPY_PIPE_GAP_EDGE_MARGIN_RATIO = 0.05;
