import {
  FLAPPY_BIRD_AURA_ALPHA,
  FLAPPY_BIRD_AURA_BLUR_MULTIPLIER,
  FLAPPY_BIRD_AURA_EXPAND_PX,
  FLAPPY_BIRD_BODY_GLOW_BLUR_PX,
  FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX,
  FLAPPY_BIRD_CHAMPION_RED_GLOW_ALPHA,
  FLAPPY_BIRD_CHAMPION_RED_GLOW_EXPAND_PX,
  FLAPPY_BIRD_CHAMPION_SHINE_FILL_STYLE,
  FLAPPY_BIRD_CHAMPION_SHINE_GLOW_COLOR,
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_SHINE_FILL_STYLE,
  FLAPPY_BIRD_SHINE_INSET_RATIO,
  FLAPPY_BIRD_SHINE_SIZE_RATIO,
  FLAPPY_BIRD_WHITE_SHINE_GLOW_BLUR_PX,
  FLAPPY_BIRD_WHITE_SHINE_GLOW_COLOR,
  FLAPPY_BIRD_X_PX,
  FLAPPY_LEADER_RING_GLOW_BLUR_PX,
  FLAPPY_LEADER_RING_LINE_WIDTH_PX,
  FLAPPY_LEADER_RING_RADIUS_OFFSET_PX,
  FLAPPY_NEON_PALETTE,
  FLAPPY_NON_CHAMPION_OPACITY,
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_TRAIL_LINE_WIDTH_PX,
  FLAPPY_TRAIL_MIN_HORIZONTAL_SEGMENT_PX,
  FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX,
  FLAPPY_TRAIL_OPACITY_FACTOR,
} from '../../../constants/constants';
import type { TrailPoint } from '../../browser-entry.types';
import { resolveBirdRenderStyle } from '../playback.render.utils';
import {
  resolveEdgeOpacityFactor,
  resolveTrailLifetimeOpacityFactor,
} from '../playback.trail.utils';
import type { PlaybackEdgeBounds } from '../playback.types';
import type {
  PlaybackBirdGeometry,
  PlaybackTrailRenderStyle,
} from './playback.frame-render.types';

/**
 * Draws one active bird body, glow, shine, and leader ring.
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
  // Step 1: Resolve geometry and style contracts for the current bird.
  const birdGeometry = resolvePlaybackBirdGeometry(birdYPx);
  const birdRenderStyle = resolveBirdRenderStyle(birdIndex, championBirdIndex);

  // Step 2: Draw champion-only glow passes behind the bird body.
  drawPlaybackBirdChampionAura(context, birdGeometry, birdRenderStyle);
  drawPlaybackBirdChampionGlowPlate(context, birdGeometry, birdRenderStyle);

  // Step 3: Draw the bird body, shine highlight, and champion ring.
  drawPlaybackBirdBody(context, birdGeometry, birdRenderStyle);
  drawPlaybackBirdShine(context, birdGeometry, birdRenderStyle.isChampionBird);
  drawPlaybackBirdLeaderRing(
    context,
    birdGeometry,
    birdRenderStyle.isChampionBird,
  );
}

/**
 * Resolves the trail style used for one bird's stepped trail.
 *
 * @param birdIndex - Index of the bird being rendered.
 * @param championBirdIndex - Champion index for the current frame.
 * @returns Base opacity and color for the bird trail.
 */
export function resolvePlaybackTrailStyle(
  birdIndex: number,
  championBirdIndex: number,
): PlaybackTrailRenderStyle {
  // Step 1: Resolve champion-aware opacity and trail color.
  const isChampionBird = birdIndex === championBirdIndex;
  return {
    baseOpacity:
      (isChampionBird ? 1 : FLAPPY_NON_CHAMPION_OPACITY) *
      FLAPPY_TRAIL_OPACITY_FACTOR,
    trailColor: isChampionBird
      ? FLAPPY_NEON_PALETTE.trail
      : FLAPPY_NEON_PALETTE.nonChampionBird,
  };
}

/**
 * Draws the stepped trail history for one active bird.
 *
 * @param context - Canvas 2D drawing context.
 * @param trailPoints - Cached per-frame trail points for one bird.
 * @param color - Stroke color for the trail.
 * @param anchorX - Bird anchor x-position in world space.
 * @param baseOpacity - Base opacity before edge and lifetime fading.
 * @param edgeBounds - Visible world bounds used for edge fading.
 * @returns Nothing.
 */
export function drawTrail(
  context: CanvasRenderingContext2D,
  trailPoints: TrailPoint[],
  color: string,
  anchorX: number,
  baseOpacity: number,
  edgeBounds: PlaybackEdgeBounds,
): void {
  // Step 1: Guard empty trails.
  if (trailPoints.length === 0) {
    return;
  }

  const latestTrailFrameIndex = trailPoints.at(-1)?.frameIndex ?? 0;

  const firstTrailPoint = trailPoints[0];
  const firstTrailFrameOffset = Math.max(
    0,
    latestTrailFrameIndex - firstTrailPoint.frameIndex,
  );
  let previousXPosition =
    anchorX - firstTrailFrameOffset * FLAPPY_PIPE_SPEED_PX_PER_FRAME;
  let previousYPosition = firstTrailPoint.yPx;
  let previousFrameOffset = firstTrailFrameOffset;
  const maximumTrailFrameOffset = Math.max(1, firstTrailFrameOffset);

  // Step 2: Configure trail stroke style.
  const previousGlobalAlpha = context.globalAlpha;
  context.strokeStyle = color;
  context.lineWidth = FLAPPY_TRAIL_LINE_WIDTH_PX;

  // Step 3: Render stepped segments with edge-proximity alpha fading.
  trailPoints.slice(1).forEach((trailPoint) => {
    const frameOffset = Math.max(
      0,
      latestTrailFrameIndex - trailPoint.frameIndex,
    );
    const nextXPosition =
      anchorX - frameOffset * FLAPPY_PIPE_SPEED_PX_PER_FRAME;
    const nextYPosition = trailPoint.yPx;

    const horizontalDeltaPx = nextXPosition - previousXPosition;
    const horizontalDirection = Math.sign(horizontalDeltaPx) || 1;
    const steppedHorizontalLengthPx = Math.max(
      Math.abs(horizontalDeltaPx),
      FLAPPY_TRAIL_MIN_HORIZONTAL_SEGMENT_PX,
    );
    const steppedHorizontalXPosition =
      previousXPosition + horizontalDirection * steppedHorizontalLengthPx;
    drawTrailSegmentWithEdgeFade(
      context,
      previousXPosition,
      previousYPosition,
      steppedHorizontalXPosition,
      previousYPosition,
      baseOpacity,
      edgeBounds,
      previousFrameOffset,
      frameOffset,
      maximumTrailFrameOffset,
    );
    previousXPosition = steppedHorizontalXPosition;

    const verticalDeltaPx = nextYPosition - previousYPosition;
    if (verticalDeltaPx !== 0) {
      const verticalDirection = Math.sign(verticalDeltaPx);
      const steppedVerticalLengthPx = Math.max(
        Math.abs(verticalDeltaPx),
        FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX,
      );
      const steppedVerticalYPosition =
        previousYPosition + verticalDirection * steppedVerticalLengthPx;
      drawTrailSegmentWithEdgeFade(
        context,
        previousXPosition,
        previousYPosition,
        steppedHorizontalXPosition,
        steppedVerticalYPosition,
        baseOpacity,
        edgeBounds,
        previousFrameOffset,
        frameOffset,
        maximumTrailFrameOffset,
      );
      previousYPosition = steppedVerticalYPosition;
    }

    drawTrailSegmentWithEdgeFade(
      context,
      previousXPosition,
      previousYPosition,
      steppedHorizontalXPosition,
      nextYPosition,
      baseOpacity,
      edgeBounds,
      previousFrameOffset,
      frameOffset,
      maximumTrailFrameOffset,
    );
    previousYPosition = nextYPosition;

    drawTrailSegmentWithEdgeFade(
      context,
      previousXPosition,
      previousYPosition,
      nextXPosition,
      nextYPosition,
      baseOpacity,
      edgeBounds,
      previousFrameOffset,
      frameOffset,
      maximumTrailFrameOffset,
    );
    previousXPosition = nextXPosition;
    previousYPosition = nextYPosition;
    previousFrameOffset = frameOffset;
  });

  // Step 4: Restore caller alpha state.
  context.globalAlpha = previousGlobalAlpha;
}

/**
 * Resolves the fixed bird geometry used by all body rendering passes.
 *
 * @param birdYPx - Bird vertical position in world pixels.
 * @returns Pixel-aligned square geometry for the bird body.
 */
function resolvePlaybackBirdGeometry(birdYPx: number): PlaybackBirdGeometry {
  // Step 1: Resolve a fixed square body size from the collision radius.
  const birdSideLengthPx = Math.max(1, Math.round(FLAPPY_BIRD_RADIUS_PX * 2));

  // Step 2: Return pixel-aligned body bounds for the current y-position.
  return {
    birdSideLengthPx,
    birdLeftPx: Math.round(FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX),
    birdTopPx: Math.round(birdYPx - FLAPPY_BIRD_RADIUS_PX),
  };
}

/**
 * Draws the soft champion aura plate behind the bird body.
 *
 * @param context - Canvas 2D drawing context.
 * @param birdGeometry - Pixel-aligned bird geometry.
 * @param birdRenderStyle - Resolved bird style payload.
 * @returns Nothing.
 */
function drawPlaybackBirdChampionAura(
  context: CanvasRenderingContext2D,
  birdGeometry: PlaybackBirdGeometry,
  birdRenderStyle: ReturnType<typeof resolveBirdRenderStyle>,
): void {
  // Step 1: Skip the aura pass for non-champion birds.
  if (!birdRenderStyle.isChampionBird) {
    return;
  }

  // Step 2: Draw the additive aura plate behind the champion body.
  const auraExpandPx = Math.round(FLAPPY_BIRD_AURA_EXPAND_PX);
  const previousCompositeOperation = context.globalCompositeOperation;
  context.globalCompositeOperation = 'lighter';
  context.globalAlpha = birdRenderStyle.birdOpacity * FLAPPY_BIRD_AURA_ALPHA;
  context.fillStyle = birdRenderStyle.birdRenderColor;
  context.shadowColor = birdRenderStyle.birdRenderColor;
  context.shadowBlur = Math.round(
    FLAPPY_BIRD_BODY_GLOW_BLUR_PX * FLAPPY_BIRD_AURA_BLUR_MULTIPLIER,
  );
  context.fillRect(
    birdGeometry.birdLeftPx - auraExpandPx,
    birdGeometry.birdTopPx - auraExpandPx,
    birdGeometry.birdSideLengthPx + auraExpandPx * 2,
    birdGeometry.birdSideLengthPx + auraExpandPx * 2,
  );
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  context.globalCompositeOperation = previousCompositeOperation;
}

/**
 * Draws the champion-only red glow plate beneath the bird body.
 *
 * @param context - Canvas 2D drawing context.
 * @param birdGeometry - Pixel-aligned bird geometry.
 * @param birdRenderStyle - Resolved bird style payload.
 * @returns Nothing.
 */
function drawPlaybackBirdChampionGlowPlate(
  context: CanvasRenderingContext2D,
  birdGeometry: PlaybackBirdGeometry,
  birdRenderStyle: ReturnType<typeof resolveBirdRenderStyle>,
): void {
  // Step 1: Skip the red glow plate for non-champion birds.
  if (!birdRenderStyle.isChampionBird) {
    return;
  }

  // Step 2: Draw the expanded red glow plate behind the champion body.
  const expandedGlowInsetPx = Math.round(
    FLAPPY_BIRD_CHAMPION_RED_GLOW_EXPAND_PX,
  );
  context.globalAlpha =
    birdRenderStyle.birdOpacity * FLAPPY_BIRD_CHAMPION_RED_GLOW_ALPHA;
  context.fillStyle = FLAPPY_NEON_PALETTE.championBird;
  context.shadowColor = FLAPPY_NEON_PALETTE.championBird;
  context.shadowBlur =
    FLAPPY_BIRD_BODY_GLOW_BLUR_PX + FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX;
  context.fillRect(
    birdGeometry.birdLeftPx - expandedGlowInsetPx,
    birdGeometry.birdTopPx - expandedGlowInsetPx,
    birdGeometry.birdSideLengthPx + expandedGlowInsetPx * 2,
    birdGeometry.birdSideLengthPx + expandedGlowInsetPx * 2,
  );
}

/**
 * Draws the square bird body with its base neon glow.
 *
 * @param context - Canvas 2D drawing context.
 * @param birdGeometry - Pixel-aligned bird geometry.
 * @param birdRenderStyle - Resolved bird style payload.
 * @returns Nothing.
 */
function drawPlaybackBirdBody(
  context: CanvasRenderingContext2D,
  birdGeometry: PlaybackBirdGeometry,
  birdRenderStyle: ReturnType<typeof resolveBirdRenderStyle>,
): void {
  // Step 1: Draw the main square body using the resolved bird color.
  context.globalAlpha = birdRenderStyle.birdOpacity;
  context.fillStyle = birdRenderStyle.birdRenderColor;
  context.shadowColor = birdRenderStyle.birdRenderColor;
  context.shadowBlur =
    FLAPPY_BIRD_BODY_GLOW_BLUR_PX +
    (birdRenderStyle.isChampionBird
      ? FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX
      : 0);
  context.fillRect(
    birdGeometry.birdLeftPx,
    birdGeometry.birdTopPx,
    birdGeometry.birdSideLengthPx,
    birdGeometry.birdSideLengthPx,
  );
}

/**
 * Draws the reflective shine highlight for one bird body.
 *
 * @param context - Canvas 2D drawing context.
 * @param birdGeometry - Pixel-aligned bird geometry.
 * @param isChampionBird - Whether the current bird is the champion.
 * @returns Nothing.
 */
function drawPlaybackBirdShine(
  context: CanvasRenderingContext2D,
  birdGeometry: PlaybackBirdGeometry,
  isChampionBird: boolean,
): void {
  // Step 1: Resolve shine geometry inside the square bird body.
  const shineInsetPx =
    birdGeometry.birdSideLengthPx * FLAPPY_BIRD_SHINE_INSET_RATIO;
  const shineSideLengthPx = Math.max(
    1,
    birdGeometry.birdSideLengthPx * FLAPPY_BIRD_SHINE_SIZE_RATIO,
  );

  // Step 2: Draw the inner shine highlight using champion-aware colors.
  context.fillStyle = isChampionBird
    ? FLAPPY_BIRD_CHAMPION_SHINE_FILL_STYLE
    : FLAPPY_BIRD_SHINE_FILL_STYLE;
  context.shadowColor = isChampionBird
    ? FLAPPY_BIRD_CHAMPION_SHINE_GLOW_COLOR
    : FLAPPY_BIRD_WHITE_SHINE_GLOW_COLOR;
  context.shadowBlur = FLAPPY_BIRD_WHITE_SHINE_GLOW_BLUR_PX;
  context.fillRect(
    Math.round(birdGeometry.birdLeftPx + shineInsetPx),
    Math.round(birdGeometry.birdTopPx + shineInsetPx),
    Math.round(shineSideLengthPx),
    Math.round(shineSideLengthPx),
  );
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
}

/**
 * Draws the leader ring around the champion bird.
 *
 * @param context - Canvas 2D drawing context.
 * @param birdGeometry - Pixel-aligned bird geometry.
 * @param isChampionBird - Whether the current bird is the champion.
 * @returns Nothing.
 */
function drawPlaybackBirdLeaderRing(
  context: CanvasRenderingContext2D,
  birdGeometry: PlaybackBirdGeometry,
  isChampionBird: boolean,
): void {
  // Step 1: Skip the leader ring for non-champion birds.
  if (!isChampionBird) {
    return;
  }

  // Step 2: Draw the glowing leader ring around the champion body.
  const leaderRingInsetPx = Math.round(FLAPPY_LEADER_RING_RADIUS_OFFSET_PX);
  context.strokeStyle = FLAPPY_NEON_PALETTE.leaderRing;
  context.lineWidth = FLAPPY_LEADER_RING_LINE_WIDTH_PX;
  context.shadowColor = FLAPPY_NEON_PALETTE.leaderRing;
  context.shadowBlur = FLAPPY_LEADER_RING_GLOW_BLUR_PX;
  context.strokeRect(
    birdGeometry.birdLeftPx - leaderRingInsetPx,
    birdGeometry.birdTopPx - leaderRingInsetPx,
    birdGeometry.birdSideLengthPx + leaderRingInsetPx * 2,
    birdGeometry.birdSideLengthPx + leaderRingInsetPx * 2,
  );
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
}

/**
 * Draws one trail segment with combined edge and lifetime fading.
 *
 * @param context - Canvas 2D drawing context.
 * @param startXPx - Segment start x-position.
 * @param startYPx - Segment start y-position.
 * @param endXPx - Segment end x-position.
 * @param endYPx - Segment end y-position.
 * @param baseOpacity - Base opacity before fade factors.
 * @param edgeBounds - Visible world bounds used for edge fading.
 * @param startFrameOffset - Relative age of the segment start.
 * @param endFrameOffset - Relative age of the segment end.
 * @param maximumTrailFrameOffset - Oldest visible trail age.
 * @returns Nothing.
 */
function drawTrailSegmentWithEdgeFade(
  context: CanvasRenderingContext2D,
  startXPx: number,
  startYPx: number,
  endXPx: number,
  endYPx: number,
  baseOpacity: number,
  edgeBounds: PlaybackEdgeBounds,
  startFrameOffset: number,
  endFrameOffset: number,
  maximumTrailFrameOffset: number,
): void {
  const segmentLengthPx = Math.hypot(endXPx - startXPx, endYPx - startYPx);
  if (segmentLengthPx === 0 || baseOpacity <= 0) {
    return;
  }

  // Step 1: Resolve edge-fade factor from both segment endpoints.
  const startOpacityFactor = resolveEdgeOpacityFactor(
    startXPx,
    startYPx,
    edgeBounds,
  );
  const endOpacityFactor = resolveEdgeOpacityFactor(endXPx, endYPx, edgeBounds);
  const edgeOpacityFactor = Math.min(startOpacityFactor, endOpacityFactor);

  // Step 2: Resolve lifetime fade so older trail history fades near cutoff.
  const startLifetimeOpacityFactor = resolveTrailLifetimeOpacityFactor(
    startFrameOffset,
    maximumTrailFrameOffset,
  );
  const endLifetimeOpacityFactor = resolveTrailLifetimeOpacityFactor(
    endFrameOffset,
    maximumTrailFrameOffset,
  );
  const lifetimeOpacityFactor = Math.min(
    startLifetimeOpacityFactor,
    endLifetimeOpacityFactor,
  );

  const segmentOpacity =
    baseOpacity * edgeOpacityFactor * lifetimeOpacityFactor;
  if (segmentOpacity <= 0) {
    return;
  }

  // Step 3: Draw the segment with resolved opacity.
  context.globalAlpha = segmentOpacity;
  context.beginPath();
  context.moveTo(startXPx, startYPx);
  context.lineTo(endXPx, endYPx);
  context.stroke();
}
