import { FLAPPY_NEON_PALETTE } from '../../../constants/constants';
import { resolveStarfieldTiles } from '../playback.starfield.service';
import { positiveModulo } from '../playback.starfield.utils';
import {
  FLAPPY_BACKGROUND_COMPOSITE_LIGHTER,
  FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER,
  FLAPPY_BACKGROUND_TILE_ROW_BUFFER_COUNT,
  FLAPPY_BACKGROUND_TILE_ROW_START_INDEX,
  FLAPPY_BACKGROUND_TRANSPARENT_SHADOW_COLOR,
} from './playback.background.constants';
import type {
  PlaybackBackgroundRequest,
  PlaybackBackgroundSceneContext,
  PlaybackHorizonLineRequest,
  PlaybackTiledImageRowRequest,
} from './playback.background.types';
import {
  resolveAlignedHorizonYPx,
  resolvePlaybackBackgroundLayout,
  resolvePlaybackHorizonStyle,
  resolveSafeBackgroundDimension,
} from './playback.background.utils';

/**
 * Resolves the derived scene contract required by the background passes.
 *
 * @param request - Narrow render input required for background composition.
 * @returns Immutable scene context shared by the private render helpers.
 */
export function resolvePlaybackBackgroundSceneContext(
  request: PlaybackBackgroundRequest,
): PlaybackBackgroundSceneContext {
  // Step 1: Clamp viewport dimensions into render-safe pixel values.
  const visibleWorldWidthPx = resolveSafeBackgroundDimension(
    request.visibleWorldWidthPx,
  );
  const visibleWorldHeightPx = resolveSafeBackgroundDimension(
    request.visibleWorldHeightPx,
  );

  // Step 2: Resolve the vertical sky-ground split and horizon style.
  const backgroundLayout =
    resolvePlaybackBackgroundLayout(visibleWorldHeightPx);
  const horizonStyle = resolvePlaybackHorizonStyle();
  const alignedHorizonYPx = resolveAlignedHorizonYPx(
    backgroundLayout.horizonYPx,
    horizonStyle.lineThicknessPx,
  );

  // Step 3: Return a compact context object for the remaining passes.
  return {
    viewportLeftXPx: request.viewportLeftXPx,
    visibleWorldWidthPx,
    visibleWorldHeightPx,
    skyHeightPx: backgroundLayout.skyHeightPx,
    alignedHorizonYPx,
    horizonStyle,
  };
}

/**
 * Paints the base background fill for the currently visible viewport.
 *
 * @param context - Canvas 2D drawing context.
 * @param sceneContext - Derived scene geometry and style contract.
 * @returns Nothing.
 */
export function paintPlaybackBackgroundBase(
  context: CanvasRenderingContext2D,
  sceneContext: PlaybackBackgroundSceneContext,
): void {
  // Step 1: Reset canvas paint state to a predictable base pass.
  context.globalAlpha = 1;
  context.globalCompositeOperation = FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER;
  context.shadowBlur = 0;
  context.shadowColor = FLAPPY_BACKGROUND_TRANSPARENT_SHADOW_COLOR;
  context.fillStyle = FLAPPY_NEON_PALETTE.background;

  // Step 2: Fill only the currently visible world-space viewport.
  context.fillRect(
    sceneContext.viewportLeftXPx,
    0,
    sceneContext.visibleWorldWidthPx,
    sceneContext.visibleWorldHeightPx,
  );
}

/**
 * Draws the starfield parallax clipped to the upper sky band.
 *
 * @param context - Canvas 2D drawing context.
 * @param sceneContext - Derived scene geometry and style contract.
 * @param request - Narrow render input required for background composition.
 * @returns Nothing.
 */
export function drawPlaybackBackgroundSky(
  context: CanvasRenderingContext2D,
  sceneContext: PlaybackBackgroundSceneContext,
  request: PlaybackBackgroundRequest,
): void {
  // Step 1: Constrain star drawing to the sky band above the horizon.
  context.save();
  context.beginPath();
  context.rect(
    sceneContext.viewportLeftXPx,
    0,
    sceneContext.visibleWorldWidthPx,
    sceneContext.skyHeightPx,
  );
  context.clip();
  context.globalCompositeOperation = FLAPPY_BACKGROUND_COMPOSITE_LIGHTER;

  // Step 2: Draw each cached starfield layer with its own parallax offset.
  const starfieldTiles = resolveStarfieldTiles(sceneContext.skyHeightPx);
  for (const starfieldTile of starfieldTiles) {
    const scrollOffsetPx = request.scrollBasePx * starfieldTile.scrollRatio;
    drawPlaybackTiledImageRow(context, {
      startXPx: sceneContext.viewportLeftXPx,
      tile: starfieldTile.image,
      tileWidthPx: starfieldTile.tileWidthPx,
      visibleWidthPx: sceneContext.visibleWorldWidthPx,
      offsetPx: scrollOffsetPx,
    });
  }

  // Step 3: Restore the caller's unclipped canvas state.
  context.restore();
}

/**
 * Draws the glowing horizon divider across the visible viewport.
 *
 * @param context - Canvas 2D drawing context.
 * @param sceneContext - Derived scene geometry and style contract.
 * @returns Nothing.
 */
export function drawPlaybackBackgroundHorizon(
  context: CanvasRenderingContext2D,
  sceneContext: PlaybackBackgroundSceneContext,
): void {
  // Step 1: Delegate the divider rendering to the dedicated horizon helper.
  drawPlaybackHorizonLine(context, {
    viewportLeftXPx: sceneContext.viewportLeftXPx,
    visibleWorldWidthPx: sceneContext.visibleWorldWidthPx,
    alignedHorizonYPx: sceneContext.alignedHorizonYPx,
    horizonStyle: sceneContext.horizonStyle,
  });
}

/**
 * Draws a horizontally tiled image strip across the visible width.
 *
 * @param context - Canvas 2D drawing context.
 * @param row - Tile image and wrap parameters.
 * @returns Nothing.
 */
function drawPlaybackTiledImageRow(
  context: CanvasRenderingContext2D,
  row: PlaybackTiledImageRowRequest,
): void {
  // Step 1: Normalize the scroll offset so tile placement stays bounded.
  const normalizedOffsetPx = positiveModulo(row.offsetPx, row.tileWidthPx);
  const maximumTileIndex =
    Math.ceil(row.visibleWidthPx / row.tileWidthPx) +
    FLAPPY_BACKGROUND_TILE_ROW_BUFFER_COUNT;

  // Step 2: Draw enough repeated tiles to cover the visible strip.
  for (
    let tileIndex = FLAPPY_BACKGROUND_TILE_ROW_START_INDEX;
    tileIndex <= maximumTileIndex;
    tileIndex += 1
  ) {
    const tileLeftPx =
      row.startXPx + tileIndex * row.tileWidthPx - normalizedOffsetPx;
    context.drawImage(row.tile, tileLeftPx, 0);
  }
}

/**
 * Draws the glowing horizon divider using the provided neon style.
 *
 * @param context - Canvas 2D drawing context.
 * @param request - Width, aligned y-position, and style for the divider.
 * @returns Nothing.
 */
function drawPlaybackHorizonLine(
  context: CanvasRenderingContext2D,
  request: PlaybackHorizonLineRequest,
): void {
  // Step 1: Draw the outer glow pass to establish the neon bloom.
  context.globalCompositeOperation = FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER;
  context.strokeStyle = request.horizonStyle.lineColor;
  context.lineWidth = request.horizonStyle.lineThicknessPx;
  context.shadowColor = request.horizonStyle.glowColor;
  context.shadowBlur = request.horizonStyle.glowBlurPx;
  context.globalAlpha = request.horizonStyle.glowAlpha;
  context.beginPath();
  context.moveTo(request.viewportLeftXPx, request.alignedHorizonYPx);
  context.lineTo(
    request.viewportLeftXPx + request.visibleWorldWidthPx,
    request.alignedHorizonYPx,
  );
  context.stroke();

  // Step 2: Restore a crisp core line after the glow pass.
  context.shadowBlur = 0;
  context.shadowColor = FLAPPY_BACKGROUND_TRANSPARENT_SHADOW_COLOR;
  context.globalAlpha = 1;
  context.beginPath();
  context.moveTo(request.viewportLeftXPx, request.alignedHorizonYPx);
  context.lineTo(
    request.viewportLeftXPx + request.visibleWorldWidthPx,
    request.alignedHorizonYPx,
  );
  context.stroke();
}
