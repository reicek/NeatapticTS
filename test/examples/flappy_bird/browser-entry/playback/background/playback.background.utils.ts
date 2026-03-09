import {
  FLAPPY_BACKGROUND_HORIZON_STYLE,
  FLAPPY_BACKGROUND_HORIZON_HALF_THICKNESS_MULTIPLIER,
  FLAPPY_BACKGROUND_HORIZON_LINE_THICKNESS_PX,
  FLAPPY_BACKGROUND_MIN_VIEWPORT_DIMENSION_PX,
  FLAPPY_BACKGROUND_ODD_STROKE_ALIGNMENT_OFFSET_PX,
  FLAPPY_BACKGROUND_ODD_STROKE_DIVISOR,
  FLAPPY_BACKGROUND_SKY_HEIGHT_RATIO,
} from './playback.background.constants';
import type {
  PlaybackBackgroundLayout,
  PlaybackHorizonStyle,
} from './playback.background.types';

/**
 * Resolves the vertical split between the starfield sky and the future ground.
 *
 * @param visibleWorldHeightPx - Current visible world height in pixels.
 * @returns Stable scene layout for the current frame.
 */
export function resolvePlaybackBackgroundLayout(
  visibleWorldHeightPx: number,
): PlaybackBackgroundLayout {
  // Step 1: Clamp the viewport height so follow-up math stays render-safe.
  const safeVisibleWorldHeightPx =
    resolveSafeBackgroundDimension(visibleWorldHeightPx);

  // Step 2: Reserve the top segment for the current starfield parallax.
  const skyHeightPx = Math.max(
    FLAPPY_BACKGROUND_MIN_VIEWPORT_DIMENSION_PX,
    Math.round(safeVisibleWorldHeightPx * FLAPPY_BACKGROUND_SKY_HEIGHT_RATIO),
  );
  const horizonThicknessPx = FLAPPY_BACKGROUND_HORIZON_LINE_THICKNESS_PX;
  const lowerBandTopYPx = skyHeightPx;
  const lowerBandBottomYPx = safeVisibleWorldHeightPx;
  const lowerBandHeightPx = Math.max(
    FLAPPY_BACKGROUND_MIN_VIEWPORT_DIMENSION_PX,
    lowerBandBottomYPx - lowerBandTopYPx,
  );
  const horizonYPx = Math.max(
    0,
    lowerBandTopYPx -
      horizonThicknessPx * FLAPPY_BACKGROUND_HORIZON_HALF_THICKNESS_MULTIPLIER,
  );

  // Step 3: Return an explicit layout object so render code stays declarative.
  return {
    skyHeightPx,
    lowerBandTopYPx,
    lowerBandHeightPx,
    lowerBandBottomYPx,
    horizonYPx,
    horizonThicknessPx,
  };
}

/**
 * Resolves the neon paint settings for the horizon divider.
 *
 * @returns Reusable draw style for both the glow and crisp line passes.
 */
export function resolvePlaybackHorizonStyle(): PlaybackHorizonStyle {
  // Step 1: Reuse one frozen style object so horizon drawing stays allocation-free.
  return FLAPPY_BACKGROUND_HORIZON_STYLE;
}

/**
 * Resolves pixel-snapped horizon positioning for crisp canvas strokes.
 *
 * @param horizonYPx - Logical horizon centerline in pixels.
 * @param lineThicknessPx - Stroke thickness in pixels.
 * @returns Pixel-snapped y-position for the stroke.
 */
export function resolveAlignedHorizonYPx(
  horizonYPx: number,
  lineThicknessPx: number,
): number {
  // Step 1: Offset odd-width strokes onto half pixels for crisp rasterization.
  const oddStrokeAlignmentOffsetPx =
    lineThicknessPx % FLAPPY_BACKGROUND_ODD_STROKE_DIVISOR === 1
      ? FLAPPY_BACKGROUND_ODD_STROKE_ALIGNMENT_OFFSET_PX
      : 0;

  // Step 2: Return the aligned horizon position.
  return horizonYPx + oddStrokeAlignmentOffsetPx;
}

/**
 * Clamps a background dimension into a render-safe positive integer.
 *
 * @param dimensionPx - Candidate viewport dimension in pixels.
 * @returns Positive integer dimension suitable for canvas math.
 */
export function resolveSafeBackgroundDimension(dimensionPx: number): number {
  // Step 1: Round sub-pixel measurements and enforce a positive floor.
  return Math.max(
    FLAPPY_BACKGROUND_MIN_VIEWPORT_DIMENSION_PX,
    Math.round(dimensionPx),
  );
}
