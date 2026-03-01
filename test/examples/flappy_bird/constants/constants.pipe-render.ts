import {
  FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX as SHARED_FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX,
  FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX as SHARED_FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX,
  FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX as SHARED_FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX,
} from './constants';
import { FLAPPY_BIRD_BODY_GLOW_BLUR_PX } from './constants.birds';

/**
 * Browser-only pipe rendering constants.
 *
 * This module bridges simulation pipe geometry with visual glow/outline passes
 * used by the canvas renderer.
 */

/** Visual gap between the pipe body and its outline on the sides (pixels). */
export const FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX =
  SHARED_FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX;

/** Visual gap between the pipe body and its outline at the pipe entrance rim (pixels). */
export const FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX =
  SHARED_FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX;

/** Stroke width used for the pipe outline (pixels). */
export const FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX =
  SHARED_FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX;

/** Opacity used for the soft pipe glow stroke pass. */
export const FLAPPY_PIPE_OUTLINE_GLOW_ALPHA = 0.68;

/** Stroke width used for the soft pipe glow stroke pass (pixels). */
export const FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX = 8;

/**
 * Cyan neon glow used for pipe outline shadow.
 *
 * This intentionally matches the asciiMaze `neonCyan` ANSI color (`\x1b[38;5;87m`)
 * which maps to xterm color 87 ~= rgb(95, 255, 255) / hex `#5fffff`.
 */
export const FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR = 'rgba(95, 255, 255, 0.95)';

/** Blur radius used for the cyan pipe outline glow (pixels). */
export const FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX = Math.max(
  24,
  Math.round(FLAPPY_BIRD_BODY_GLOW_BLUR_PX * 3.2),
);
