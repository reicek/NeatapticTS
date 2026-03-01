import { FLAPPY_WORLD_HEIGHT_PX } from './constants.world';

/**
 * Pipe geometry and baseline pacing constants.
 *
 * These values describe the canonical pipe body size, opening geometry, visual
 * outline offsets, and default leftward movement cadence.
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

/** Vertical opening size of each pipe gap (pixels). */
export const FLAPPY_PIPE_GAP_PX = 150;

/** Pipe horizontal speed (pixels/frame). */
export const FLAPPY_PIPE_SPEED_PX_PER_FRAME = 5;

/** Frames between spawning new pipes. */
export const FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES = 50;

/** Minimum allowed gap center height (pixels). */
export const FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX = 100;

/** Maximum allowed gap center height (pixels). */
export const FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX = FLAPPY_WORLD_HEIGHT_PX - 100;
