import {
  ensurePlaybackBackgroundViewportCacheValidity,
  resolveCachedPlaybackBackgroundLayout,
} from './playback.background.cache.services';
import type {
  PlaybackBackgroundRequest,
  PlaybackBackgroundSceneContext,
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
  ensurePlaybackBackgroundViewportCacheValidity(
    visibleWorldWidthPx,
    visibleWorldHeightPx,
  );

  // Step 2: Resolve the vertical sky-ground split and horizon style.
  const backgroundLayout = resolveCachedPlaybackBackgroundLayout(
    visibleWorldHeightPx,
    () => resolvePlaybackBackgroundLayout(visibleWorldHeightPx),
  );
  const horizonStyle = resolvePlaybackHorizonStyle();
  const alignedHorizonYPx = resolveAlignedHorizonYPx(
    backgroundLayout.horizonYPx,
    horizonStyle.lineThicknessPx,
  );
  const vanishingPointXPx = request.viewportLeftXPx + visibleWorldWidthPx * 0.5;
  const vanishingPointYPx = visibleWorldHeightPx * 0.5;

  // Step 3: Return a compact context object for the remaining passes.
  return {
    viewportLeftXPx: request.viewportLeftXPx,
    visibleWorldWidthPx,
    visibleWorldHeightPx,
    skyHeightPx: backgroundLayout.skyHeightPx,
    lowerBandTopYPx: backgroundLayout.lowerBandTopYPx,
    lowerBandHeightPx: backgroundLayout.lowerBandHeightPx,
    lowerBandBottomYPx: backgroundLayout.lowerBandBottomYPx,
    alignedHorizonYPx,
    vanishingPointXPx,
    vanishingPointYPx,
    horizonStyle,
  };
}