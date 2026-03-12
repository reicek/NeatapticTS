import {
  FLAPPY_BIRD_BODY_GLOW_BLUR_PX,
  FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX,
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_NON_CHAMPION_BODY_GLOW_BLUR_PX,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX,
  FLAPPY_PIPE_OUTLINE_GLOW_ALPHA,
  FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX,
  FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX,
} from '../../../constants/constants';
import { resolveBirdRenderStyle } from '../playback.render.utils';
import type { PlaybackBirdGeometry } from './playback.frame-render.types';

/**
 * Detailed bird-painting helpers for playback frames.
 *
 * The bird renderer keeps the body intentionally simple and geometric: a neon
 * square with optional champion glow passes. That makes the population readable
 * at a glance even when many birds overlap.
 */

/**
 * Draws one active bird body and champion-only highlight passes.
 *
 * The champion receives an extra additive-outline treatment so viewers can spot
 * the leading bird quickly during playback.
 *
 * @param context - Canvas 2D drawing context.
 * @param birdYPx - Bird vertical position in world pixels.
 * @param birdIndex - Index of the bird being rendered.
 * @param championBirdIndex - Champion index for the current frame.
 * @returns Nothing.
 */
export function renderPlaybackBird(
  context: CanvasRenderingContext2D,
  birdYPx: number,
  birdIndex: number,
  championBirdIndex: number,
): void {
  const birdGeometry = resolvePlaybackBirdGeometry(birdYPx);
  const birdRenderStyle = resolveBirdRenderStyle(birdIndex, championBirdIndex);

  drawChampionPlaybackBirdGlow(context, birdGeometry, birdRenderStyle);
  drawPlaybackBirdBody(context, birdGeometry, birdRenderStyle);
}

/**
 * Resolves the fixed bird geometry used by all body rendering passes.
 *
 * Geometry is snapped to integer pixels so the square body stays crisp instead
 * of blurring across subpixel boundaries.
 *
 * @param birdYPx - Bird vertical position in world pixels.
 * @returns Pixel-aligned square geometry for the bird body.
 */
export function resolvePlaybackBirdGeometry(
  birdYPx: number,
): PlaybackBirdGeometry {
  const birdSideLengthPx = Math.max(1, Math.round(FLAPPY_BIRD_RADIUS_PX * 2));

  return {
    birdSideLengthPx,
    birdLeftPx: Math.round(FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX),
    birdTopPx: Math.round(birdYPx - FLAPPY_BIRD_RADIUS_PX),
  };
}

/**
 * Draws the square bird body with its base neon glow.
 *
 * This is the standard body pass shared by champion and non-champion birds.
 *
 * @param context - Canvas 2D drawing context.
 * @param birdGeometry - Pixel-aligned bird geometry.
 * @param birdRenderStyle - Resolved bird style payload.
 * @returns Nothing.
 */
export function drawPlaybackBirdBody(
  context: CanvasRenderingContext2D,
  birdGeometry: PlaybackBirdGeometry,
  birdRenderStyle: ReturnType<typeof resolveBirdRenderStyle>,
): void {
  context.globalAlpha = birdRenderStyle.birdOpacity;
  context.fillStyle = birdRenderStyle.birdRenderColor;
  context.shadowColor = birdRenderStyle.birdRenderColor;
  context.shadowBlur = resolvePlaybackBirdBodyGlowBlur(birdRenderStyle);
  context.fillRect(
    birdGeometry.birdLeftPx,
    birdGeometry.birdTopPx,
    birdGeometry.birdSideLengthPx,
    birdGeometry.birdSideLengthPx,
  );
}

/**
 * Draws the champion bird using the same two-pass additive outline glow method as pipes.
 *
 * Reusing the pipe-style glow language helps the whole playback scene feel like
 * one visual system instead of unrelated rendering effects.
 *
 * @param context - Canvas 2D drawing context.
 * @param birdGeometry - Pixel-aligned bird geometry.
 * @param birdRenderStyle - Resolved bird style payload.
 * @returns Nothing.
 */
export function drawChampionPlaybackBirdGlow(
  context: CanvasRenderingContext2D,
  birdGeometry: PlaybackBirdGeometry,
  birdRenderStyle: ReturnType<typeof resolveBirdRenderStyle>,
): void {
  if (!birdRenderStyle.isChampionBird) {
    return;
  }

  const previousCompositeOperation = context.globalCompositeOperation;
  const previousGlobalAlpha = context.globalAlpha;
  const previousStrokeStyle = context.strokeStyle;
  const previousShadowColor = context.shadowColor;
  const previousShadowBlur = context.shadowBlur;
  const previousLineWidth = context.lineWidth;
  const previousShadowOffsetX = context.shadowOffsetX;
  const previousShadowOffsetY = context.shadowOffsetY;

  context.globalCompositeOperation = 'lighter';
  context.strokeStyle = birdRenderStyle.birdRenderColor;
  context.globalAlpha = FLAPPY_PIPE_OUTLINE_GLOW_ALPHA;
  context.shadowColor = birdRenderStyle.birdRenderColor;
  context.shadowBlur = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX;
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;
  context.lineWidth = FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX;
  context.strokeRect(
    birdGeometry.birdLeftPx,
    birdGeometry.birdTopPx,
    birdGeometry.birdSideLengthPx,
    birdGeometry.birdSideLengthPx,
  );

  context.globalAlpha = 1;
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  context.lineWidth = FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX;
  context.strokeRect(
    birdGeometry.birdLeftPx,
    birdGeometry.birdTopPx,
    birdGeometry.birdSideLengthPx,
    birdGeometry.birdSideLengthPx,
  );

  context.globalCompositeOperation = previousCompositeOperation;
  context.globalAlpha = previousGlobalAlpha;
  context.strokeStyle = previousStrokeStyle;
  context.shadowColor = previousShadowColor;
  context.shadowBlur = previousShadowBlur;
  context.lineWidth = previousLineWidth;
  context.shadowOffsetX = previousShadowOffsetX;
  context.shadowOffsetY = previousShadowOffsetY;
}

/**
 * Resolves the body glow blur for one bird render pass.
 *
 * Champion birds receive a slightly stronger body glow than the rest of the
 * population.
 *
 * @param birdRenderStyle - Resolved bird style payload.
 * @returns Blur radius used behind the square bird body.
 */
export function resolvePlaybackBirdBodyGlowBlur(
  birdRenderStyle: ReturnType<typeof resolveBirdRenderStyle>,
): number {
  if (!birdRenderStyle.isChampionBird) {
    return FLAPPY_NON_CHAMPION_BODY_GLOW_BLUR_PX;
  }

  return (
    FLAPPY_BIRD_BODY_GLOW_BLUR_PX + FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX
  );
}
