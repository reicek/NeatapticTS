import {
  FLAPPY_NEON_PALETTE,
  FLAPPY_NON_CHAMPION_OPACITY,
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_TRAIL_LINE_WIDTH_PX,
  FLAPPY_TRAIL_MIN_HORIZONTAL_SEGMENT_PX,
  FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX,
  FLAPPY_TRAIL_OPACITY_FACTOR,
} from '../../../constants/constants';
import type { TrailPoint } from '../../browser-entry.types';
import {
  resolveEdgeOpacityFactor,
  resolveTrailLifetimeOpacityFactor,
} from '../playback.trail.utils';
import type { PlaybackEdgeBounds } from '../playback.types';
import type { PlaybackTrailRenderStyle } from './playback.frame-render.types';

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

  const previousGlobalAlpha = context.globalAlpha;
  context.strokeStyle = color;
  context.lineWidth = FLAPPY_TRAIL_LINE_WIDTH_PX;

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

  context.globalAlpha = previousGlobalAlpha;
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

  const startOpacityFactor = resolveEdgeOpacityFactor(
    startXPx,
    startYPx,
    edgeBounds,
  );
  const endOpacityFactor = resolveEdgeOpacityFactor(endXPx, endYPx, edgeBounds);
  const edgeOpacityFactor = Math.min(startOpacityFactor, endOpacityFactor);

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

  context.globalAlpha = segmentOpacity;
  context.beginPath();
  context.moveTo(startXPx, startYPx);
  context.lineTo(endXPx, endYPx);
  context.stroke();
}