import type {
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackBackgroundGroundGridSourceScene,
} from './playback.background.ground-grid.types';

/**
 * Resolves the shared scene context used by the ground-grid renderer.
 *
 * @param sceneContext - Lower-band geometry provided by the background module.
 * @returns Narrow scene contract consumed by grid-specific helpers.
 */
export function resolvePlaybackGroundGridSceneContext(
  sceneContext: PlaybackBackgroundGroundGridSourceScene,
): PlaybackBackgroundGroundGridSceneContext {
  return {
    viewportOffsetXPx: sceneContext.viewportLeftXPx,
    visibleWorldWidthPx: sceneContext.visibleWorldWidthPx,
    alignedHorizonYPx: sceneContext.alignedHorizonYPx,
    lowerBandTopYPx: sceneContext.lowerBandTopYPx,
    lowerBandHeightPx: sceneContext.lowerBandHeightPx,
    lowerBandBottomYPx: sceneContext.lowerBandBottomYPx,
    vanishingPointXPx:
      sceneContext.vanishingPointXPx - sceneContext.viewportLeftXPx,
    vanishingPointYPx: sceneContext.vanishingPointYPx,
  };
}