/**
 * Synthwave floor renderer for the Neatenstein neon raycasting demo.
 *
 * This module directly reuses the Flappy Bird ground-grid math helpers for the
 * depth curve, alpha, blur, and thickness. The visual adaptation is in the
 * perspective layout: Flappy keeps the vertical-ray vanishing point centered,
 * while Neatenstein rotates the vanishing point left/right with camera yaw so
 * the floor follows the player's mouse-look direction.
 *
 * @module
 */

import { FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT } from '../../../flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.constants';
import { FLAPPY_NEON_PALETTE } from '../../../flappy_bird/constants/constants.palette';
import {
  resolvePlaybackGroundGridDepthCurve,
  resolvePlaybackGroundGridLineAlpha,
  resolvePlaybackGroundGridLineBlur,
  resolvePlaybackGroundGridLineThickness,
} from '../../../flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.math.utils';

/**
 * Default fallback canvas width used when the caller does not supply a canvas.
 *
 * The renderer must be testable with a lightweight mock context that has no
 * `canvas` property, so a fixed size is declared locally.
 */
export const NEATENSTEIN_FLOOR_DEFAULT_WIDTH = 320;

/**
 * Default fallback canvas height used when the caller does not supply a canvas.
 */
export const NEATENSTEIN_FLOOR_DEFAULT_HEIGHT = 240;

/**
 * Fraction of the canvas width that the vanishing point shifts under full yaw.
 */
export const NEATENSTEIN_FLOOR_YAW_SHIFT_RATIO = 0.25;

/**
 * Number of vertical perspective rays drawn across the floor.
 */
export const NEATENSTEIN_FLOOR_VERTICAL_RAY_COUNT = 12;

/**
 * Camera state consumed by the floor renderer.
 *
 * `yaw` is the horizontal look angle. `x` and `y` are the player position and
 * are reserved for future parallax offsets; they do not affect the current
 * acceptance contract.
 */
export interface NeatensteinFloorCamera {
  /** Horizontal look angle in radians. */
  yaw: number;
  /** Player world x-position (reserved for future parallax). */
  x: number;
  /** Player world y-position (reserved for future parallax). */
  y: number;
}

/**
 * Minimal canvas-like context consumed by the floor renderer.
 *
 * The interface is intentionally narrow so the renderer can be unit-tested
 * with a lightweight mock and still accept a real `CanvasRenderingContext2D` at
 * runtime through structural typing.
 */
export interface NeatensteinFloorRenderContext {
  /** Start a new path. */
  beginPath(): void;
  /** Move the path cursor to `(x, y)`. */
  moveTo(x: number, y: number): void;
  /** Add a line segment to `(x, y)`. */
  lineTo(x: number, y: number): void;
  /** Stroke the current path. */
  stroke(): void;
  /** Optional stroke color. */
  strokeStyle?: string;
  /** Optional line width in pixels. */
  lineWidth?: number;
  /** Optional glow blur radius in pixels. */
  shadowBlur?: number;
  /** Optional glow color. */
  shadowColor?: string;
}

/**
 * Reused Flappy Bird depth curve: maps a normalized depth ratio into a
 * perspective spacing curve that bunches lines near the horizon.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Curved depth ratio used for line placement.
 */
export const resolveNeatensteinFloorDepthCurve =
  resolvePlaybackGroundGridDepthCurve;

/**
 * Reused Flappy Bird alpha curve: depth lines brighten as they approach the
 * viewer.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Opacity for the rendered line.
 */
export const resolveNeatensteinFloorAlpha = resolvePlaybackGroundGridLineAlpha;

/**
 * Reused Flappy Bird blur curve: depth lines sharpen as they approach the
 * viewer.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Blur radius for the rendered line.
 */
export const resolveNeatensteinFloorBlur = resolvePlaybackGroundGridLineBlur;

/**
 * Reused Flappy Bird thickness curve: depth lines thicken as they approach the
 * viewer.
 *
 * @param depthRatio - Normalized 0..1 depth where 0 is far and 1 is near.
 * @returns Stroke width in pixels.
 */
export const resolveNeatensteinFloorThickness =
  resolvePlaybackGroundGridLineThickness;

/**
 * Render the neon synthwave floor grid below the horizon.
 *
 * Horizontal depth bands are spaced with the Flappy power curve so they bunch
 * up near the horizon. Vertical perspective rays converge on a vanishing
 * point that shifts left/right with {@link camera.yaw}, producing a
 * mouse-look floor effect.
 *
 * @param ctx - Canvas-like context with path and stroke methods.
 * @param camera - Current camera look state.
 * @returns void
 *
 * @example
 * ```ts
 * renderNeatensteinFloor(ctx, { yaw: 0, x: 0, y: 0 });
 * ```
 */
export function renderNeatensteinFloor(
  ctx: NeatensteinFloorRenderContext,
  camera: NeatensteinFloorCamera,
): void {
  const width = NEATENSTEIN_FLOOR_DEFAULT_WIDTH;
  const height = NEATENSTEIN_FLOOR_DEFAULT_HEIGHT;
  const horizonY = height * 0.5;
  const vanishingPointXPx =
    width * 0.5 +
    Math.sin(camera.yaw) * width * NEATENSTEIN_FLOOR_YAW_SHIFT_RATIO;

  // Step 1: horizontal depth bands below the horizon.
  for (
    let index = 0;
    index < FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT;
    index++
  ) {
    const depthRatio = (index + 1) / FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT;
    const curvedRatio = resolveNeatensteinFloorDepthCurve(depthRatio);
    const lineY = horizonY + curvedRatio * (height - horizonY);

    ctx.strokeStyle = FLAPPY_NEON_PALETTE.groundGridLine;
    ctx.lineWidth = resolveNeatensteinFloorThickness(depthRatio);
    ctx.shadowBlur = resolveNeatensteinFloorBlur(depthRatio);
    ctx.shadowColor = FLAPPY_NEON_PALETTE.groundGridGlow;

    ctx.beginPath();
    ctx.moveTo(0, lineY);
    ctx.lineTo(width, lineY);
    ctx.stroke();
  }

  // Step 2: vertical perspective rays converging on the yaw-shifted vanishing
  // point.
  const rayCount = NEATENSTEIN_FLOOR_VERTICAL_RAY_COUNT;
  for (let rayIndex = 0; rayIndex < rayCount; rayIndex++) {
    const bottomXPx = (width * rayIndex) / Math.max(1, rayCount - 1);

    ctx.strokeStyle = FLAPPY_NEON_PALETTE.groundGridLine;
    ctx.lineWidth = resolveNeatensteinFloorThickness(1);
    ctx.shadowBlur = resolveNeatensteinFloorBlur(1);
    ctx.shadowColor = FLAPPY_NEON_PALETTE.groundGridGlow;

    ctx.beginPath();
    ctx.moveTo(bottomXPx, height);
    ctx.lineTo(vanishingPointXPx, horizonY);
    ctx.stroke();
  }
}
