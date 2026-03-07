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
  FLAPPY_BIRD_VIEWPORT_X_RATIO,
  FLAPPY_BIRD_WHITE_SHINE_GLOW_BLUR_PX,
  FLAPPY_BIRD_WHITE_SHINE_GLOW_COLOR,
  FLAPPY_BIRD_X_PX,
  FLAPPY_LEADER_RING_GLOW_BLUR_PX,
  FLAPPY_LEADER_RING_LINE_WIDTH_PX,
  FLAPPY_LEADER_RING_RADIUS_OFFSET_PX,
  FLAPPY_NEON_PALETTE,
  FLAPPY_NON_CHAMPION_OPACITY,
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_TRAIL_LINE_WIDTH_PX,
  FLAPPY_TRAIL_MIN_HORIZONTAL_SEGMENT_PX,
  FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX,
  FLAPPY_TRAIL_OPACITY_FACTOR,
} from '../../constants/constants';
import type {
  PopulationRenderState,
  TrailPoint,
  TrailState,
} from '../browser-entry.types';
import { resolveWorldViewport } from '../browser-entry.viewport.utils';
import { drawPipeNeonOutline } from './playback.render.service';
import {
  resolveBirdRenderStyle,
  resolveChampionBirdIndex,
} from './playback.render.utils';
import { resolveStarfieldTiles } from './playback.starfield.service';
import { positiveModulo } from './playback.starfield.utils';
import {
  pushTrailPoint,
  resolveEdgeOpacityFactor,
  resolveTrailLifetimeOpacityFactor,
} from './playback.trail.utils';
import type { PlaybackEdgeBounds } from './playback.types';

/**
 * Draws one simulation frame for the current population state.
 *
 * @param context - Canvas 2D drawing context.
 * @param renderState - Mutable simulation state snapshot.
 * @param trailState - Leader trail render cache.
 * @returns Nothing.
 */
export function renderPopulationFrame(
  context: CanvasRenderingContext2D,
  renderState: PopulationRenderState,
  trailState: TrailState,
): void {
  // Step 1: Resolve viewport transform and clear target canvas.
  const viewport = resolveWorldViewport(context.canvas);
  const visibleWorldWidthPx = Math.max(1, renderState.visibleWorldWidthPx);
  const visibleWorldHeightPx = Math.max(1, renderState.visibleWorldHeightPx);
  const desiredBirdScreenXPx =
    visibleWorldWidthPx * FLAPPY_BIRD_VIEWPORT_X_RATIO;
  const cameraLeftPx = FLAPPY_BIRD_X_PX - desiredBirdScreenXPx;

  context.globalAlpha = 1;
  context.globalCompositeOperation = 'source-over';
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;

  context.clearRect(0, 0, context.canvas.width, context.canvas.height);

  context.save();
  context.translate(viewport.offsetXPx, viewport.offsetYPx);
  context.scale(viewport.scale, viewport.scale);
  context.translate(-cameraLeftPx, 0);

  // Step 2: Draw parallax background.
  drawParallaxBackground(context, renderState);

  // Step 3: Draw pipes with neon outlines.
  for (const pipe of renderState.pipes) {
    const gapHalf = pipe.gapSizePx * 0.5;
    const pipeLeftPx = pipe.xPx;
    const gapTopPx = pipe.gapCenterYPx - gapHalf;
    const gapBottomPx = pipe.gapCenterYPx + gapHalf;

    context.fillStyle = FLAPPY_NEON_PALETTE.pipeFill;
    context.fillRect(pipeLeftPx, 0, FLAPPY_PIPE_WIDTH_PX, gapTopPx);
    drawPipeNeonOutline(context, pipeLeftPx, 0, FLAPPY_PIPE_WIDTH_PX, gapTopPx);
    context.fillRect(
      pipeLeftPx,
      gapBottomPx,
      FLAPPY_PIPE_WIDTH_PX,
      visibleWorldHeightPx - gapBottomPx,
    );
    drawPipeNeonOutline(
      context,
      pipeLeftPx,
      gapBottomPx,
      FLAPPY_PIPE_WIDTH_PX,
      visibleWorldHeightPx - gapBottomPx,
    );
  }

  // Step 4: Resolve champion bird and render bird bodies, shines, and rings.
  const championBirdIndex = resolveChampionBirdIndex(renderState);

  renderState.birds.forEach((bird, birdIndex) => {
    if (bird.done) {
      return;
    }

    const birdSideLengthPx = Math.max(1, Math.round(FLAPPY_BIRD_RADIUS_PX * 2));
    const birdLeftPx = Math.round(FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX);
    const birdTopPx = Math.round(bird.yPx - FLAPPY_BIRD_RADIUS_PX);
    const { birdOpacity, birdRenderColor, isChampionBird } =
      resolveBirdRenderStyle(birdIndex, championBirdIndex);

    // Step 4.1: Add a soft aura plate behind the champion bird.
    if (isChampionBird) {
      const auraExpandPx = Math.round(FLAPPY_BIRD_AURA_EXPAND_PX);
      const previousCompositeOperation = context.globalCompositeOperation;
      context.globalCompositeOperation = 'lighter';
      context.globalAlpha = birdOpacity * FLAPPY_BIRD_AURA_ALPHA;
      context.fillStyle = birdRenderColor;
      context.shadowColor = birdRenderColor;
      context.shadowBlur = Math.round(
        FLAPPY_BIRD_BODY_GLOW_BLUR_PX * FLAPPY_BIRD_AURA_BLUR_MULTIPLIER,
      );
      context.fillRect(
        birdLeftPx - auraExpandPx,
        birdTopPx - auraExpandPx,
        birdSideLengthPx + auraExpandPx * 2,
        birdSideLengthPx + auraExpandPx * 2,
      );
      context.shadowBlur = 0;
      context.shadowColor = 'transparent';
      context.globalCompositeOperation = previousCompositeOperation;
    }

    if (isChampionBird) {
      const expandedGlowInsetPx = Math.round(
        FLAPPY_BIRD_CHAMPION_RED_GLOW_EXPAND_PX,
      );
      context.globalAlpha = birdOpacity * FLAPPY_BIRD_CHAMPION_RED_GLOW_ALPHA;
      context.fillStyle = FLAPPY_NEON_PALETTE.championBird;
      context.shadowColor = FLAPPY_NEON_PALETTE.championBird;
      context.shadowBlur =
        FLAPPY_BIRD_BODY_GLOW_BLUR_PX + FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX;
      context.fillRect(
        birdLeftPx - expandedGlowInsetPx,
        birdTopPx - expandedGlowInsetPx,
        birdSideLengthPx + expandedGlowInsetPx * 2,
        birdSideLengthPx + expandedGlowInsetPx * 2,
      );
    }

    context.globalAlpha = birdOpacity;
    context.fillStyle = birdRenderColor;
    context.shadowColor = birdRenderColor;
    context.shadowBlur =
      FLAPPY_BIRD_BODY_GLOW_BLUR_PX +
      (isChampionBird ? FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX : 0);
    context.fillRect(birdLeftPx, birdTopPx, birdSideLengthPx, birdSideLengthPx);

    const shineInsetPx = birdSideLengthPx * FLAPPY_BIRD_SHINE_INSET_RATIO;
    const shineSideLengthPx = Math.max(
      1,
      birdSideLengthPx * FLAPPY_BIRD_SHINE_SIZE_RATIO,
    );
    context.fillStyle = isChampionBird
      ? FLAPPY_BIRD_CHAMPION_SHINE_FILL_STYLE
      : FLAPPY_BIRD_SHINE_FILL_STYLE;
    context.shadowColor = isChampionBird
      ? FLAPPY_BIRD_CHAMPION_SHINE_GLOW_COLOR
      : FLAPPY_BIRD_WHITE_SHINE_GLOW_COLOR;
    context.shadowBlur = FLAPPY_BIRD_WHITE_SHINE_GLOW_BLUR_PX;
    context.fillRect(
      Math.round(birdLeftPx + shineInsetPx),
      Math.round(birdTopPx + shineInsetPx),
      Math.round(shineSideLengthPx),
      Math.round(shineSideLengthPx),
    );
    context.shadowBlur = 0;
    context.shadowColor = 'transparent';

    if (isChampionBird) {
      context.strokeStyle = FLAPPY_NEON_PALETTE.leaderRing;
      context.lineWidth = FLAPPY_LEADER_RING_LINE_WIDTH_PX;
      context.shadowColor = FLAPPY_NEON_PALETTE.leaderRing;
      context.shadowBlur = FLAPPY_LEADER_RING_GLOW_BLUR_PX;
      const leaderRingInsetPx = Math.round(FLAPPY_LEADER_RING_RADIUS_OFFSET_PX);
      context.strokeRect(
        birdLeftPx - leaderRingInsetPx,
        birdTopPx - leaderRingInsetPx,
        birdSideLengthPx + leaderRingInsetPx * 2,
        birdSideLengthPx + leaderRingInsetPx * 2,
      );
      context.shadowBlur = 0;
      context.shadowColor = 'transparent';
    }
  });

  // Step 5: Draw stepped trails for active birds.
  renderState.birds.forEach((bird, birdIndex) => {
    if (bird.done) {
      return;
    }

    const birdTrailPoints = trailState.birdTrailsY[birdIndex];
    if (!birdTrailPoints || birdTrailPoints.length === 0) {
      return;
    }

    const parentBirdOpacity =
      birdIndex === championBirdIndex ? 1 : FLAPPY_NON_CHAMPION_OPACITY;
    const birdTrailColor =
      birdIndex === championBirdIndex
        ? FLAPPY_NEON_PALETTE.trail
        : FLAPPY_NEON_PALETTE.nonChampionBird;

    drawTrail(
      context,
      birdTrailPoints,
      birdTrailColor,
      FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX,
      parentBirdOpacity * FLAPPY_TRAIL_OPACITY_FACTOR,
      {
        leftXPx: cameraLeftPx,
        rightXPx: cameraLeftPx + visibleWorldWidthPx,
        topYPx: 0,
        bottomYPx: visibleWorldHeightPx,
      },
    );
  });

  context.globalAlpha = 1;
  // Step 6: Restore context to pre-viewport transform state.
  context.restore();
}

/**
 * Updates the trail cache from the latest frame snapshot.
 *
 * @param trailState - Mutable trail state.
 * @param renderState - Current render state.
 * @returns Nothing.
 */
export function updateTrailState(
  trailState: TrailState,
  renderState: PopulationRenderState,
): void {
  renderState.birds.forEach((bird, birdIndex) => {
    if (!trailState.birdTrailsY[birdIndex]) {
      trailState.birdTrailsY[birdIndex] = [];
    }
    const birdTrail = trailState.birdTrailsY[birdIndex];

    if (bird.done) {
      birdTrail.length = 0;
      return;
    }

    pushTrailPoint(birdTrail, renderState.frameIndex, bird.yPx);
  });
}

function drawParallaxBackground(
  context: CanvasRenderingContext2D,
  renderState: PopulationRenderState,
): void {
  const visibleWorldWidthPx = Math.max(
    1,
    Math.round(renderState.visibleWorldWidthPx),
  );
  const scrollBasePx = renderState.frameIndex * FLAPPY_PIPE_SPEED_PX_PER_FRAME;

  // Step 1: Paint the background fill.
  context.globalAlpha = 1;
  context.globalCompositeOperation = 'source-over';
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  context.fillStyle = FLAPPY_NEON_PALETTE.background;
  const visibleWorldHeightPx = Math.max(
    1,
    Math.round(renderState.visibleWorldHeightPx),
  );
  context.fillRect(0, 0, visibleWorldWidthPx, visibleWorldHeightPx);

  // Step 2: Draw cached starfield layers with subtle parallax.
  context.globalCompositeOperation = 'lighter';
  const starfieldTiles = resolveStarfieldTiles(visibleWorldHeightPx);
  for (const starfieldTile of starfieldTiles) {
    const scrollOffsetPx = scrollBasePx * starfieldTile.scrollRatio;
    drawTiledImageRow(context, {
      tile: starfieldTile.image,
      tileWidthPx: starfieldTile.tileWidthPx,
      visibleWidthPx: visibleWorldWidthPx,
      offsetPx: scrollOffsetPx,
    });
  }
  context.globalCompositeOperation = 'source-over';
}

function drawTiledImageRow(
  context: CanvasRenderingContext2D,
  layer: {
    tile: CanvasImageSource;
    tileWidthPx: number;
    visibleWidthPx: number;
    offsetPx: number;
  },
): void {
  const normalizedOffsetPx = positiveModulo(layer.offsetPx, layer.tileWidthPx);
  const maximumTileIndex =
    Math.ceil(layer.visibleWidthPx / layer.tileWidthPx) + 1;

  for (let tileIndex = -1; tileIndex <= maximumTileIndex; tileIndex += 1) {
    const tileLeftPx = tileIndex * layer.tileWidthPx - normalizedOffsetPx;
    context.drawImage(layer.tile, tileLeftPx, 0);
  }
}

function drawTrail(
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
