import {
  FLAPPY_GROUND_GRID_FOG_ALPHA,
  FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO,
} from './playback.background.ground-grid.constants';
import { FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER } from '../playback.background.constants';
import {
  resolveCachedGroundGridFogGradient,
  resolveGroundGridSceneCacheKey,
} from './playback.background.ground-grid.cache.services';
import type {
  PlaybackBackgroundGroundGridResolvedScene,
  PlaybackGroundGridPulse,
} from './playback.background.ground-grid.types';

/**
 * Draws the lower-band atmospheric wash behind the neon line work.
 *
 * @param context - Canvas 2D drawing context.
 * @param resolvedScene - Geometry and style for the current viewport.
 * @returns Nothing.
 */
export function drawGroundGridFog(
  context: CanvasRenderingContext2D,
  resolvedScene: PlaybackBackgroundGroundGridResolvedScene,
): void {
  const fogHeightPx =
    resolvedScene.sceneContext.lowerBandHeightPx *
    FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO;
  const sceneCacheKey = resolveGroundGridSceneCacheKey(
    resolvedScene.sceneContext,
  );
  const fogGradient = resolveCachedGroundGridFogGradient(
    context,
    sceneCacheKey,
    resolvedScene.sceneContext,
    resolvedScene.style.fogColor,
  );

  context.save();
  context.globalAlpha = FLAPPY_GROUND_GRID_FOG_ALPHA;
  context.fillStyle = fogGradient;
  context.fillRect(
    0,
    resolvedScene.sceneContext.lowerBandTopYPx,
    resolvedScene.sceneContext.visibleWorldWidthPx,
    fogHeightPx,
  );
  context.restore();
}

/**
 * Draws one pulse square above the grid lines and below gameplay entities.
 *
 * @param context - Canvas 2D drawing context.
 * @param pulse - Visible pulse square for the current frame.
 * @param fillColor - Core neon fill color.
 * @returns Nothing.
 */
export function drawGroundGridPulse(
  context: CanvasRenderingContext2D,
  pulse: PlaybackGroundGridPulse | null,
  fillColor: string,
): void {
  if (!pulse) {
    return;
  }

  const halfSizePx = pulse.sizePx * 0.5;

  context.save();
  context.globalCompositeOperation = FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER;
  context.fillStyle = fillColor;
  context.globalAlpha = pulse.alpha;
  context.fillRect(
    pulse.centerXPx - halfSizePx,
    pulse.centerYPx - halfSizePx,
    pulse.sizePx,
    pulse.sizePx,
  );
  context.restore();
}