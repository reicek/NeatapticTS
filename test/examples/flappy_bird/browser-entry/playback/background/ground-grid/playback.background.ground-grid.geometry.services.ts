import {
  resolvePlaybackGroundGridHorizontalGeometry,
  resolvePlaybackGroundGridVerticalGeometry,
} from './playback.background.ground-grid.geometry.utils';
import { resolvePlaybackGroundGridPulse } from './playback.background.ground-grid.pulse.utils';
import type {
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridGeometry,
  PlaybackGroundGridHorizontalGeometry,
  PlaybackGroundGridVerticalGeometry,
} from './playback.background.ground-grid.types';

/**
 * Builds the line geometry for the neon ground grid.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @param frameIndex - Current deterministic playback frame index.
 * @param scrollBasePx - Shared world scroll used for parallax motion.
 * @returns Horizontal depth bands and perspective rays for the current frame.
 */
export function resolvePlaybackGroundGridGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  frameIndex: number,
  scrollBasePx: number,
): PlaybackGroundGridGeometry {
  // Step 1: Resolve and cache the fixed horizontal depth bands for the lower plane.
  const horizontalGeometry: PlaybackGroundGridHorizontalGeometry =
    resolvePlaybackGroundGridHorizontalGeometry(sceneContext);

  // Step 2: Resolve and cache the moving perspective rays within one wrapped cycle.
  const verticalGeometry: PlaybackGroundGridVerticalGeometry =
    resolvePlaybackGroundGridVerticalGeometry(sceneContext, scrollBasePx);

  // Step 3: Resolve one visible pulse above the grid lines.
  const pulse = resolvePlaybackGroundGridPulse({
    frameIndex,
    horizontalPulsePaths: horizontalGeometry.preferredHorizontalPulsePaths,
    sceneContext,
    verticalPulsePaths: verticalGeometry.verticalPulsePaths,
    visibleVerticalPulsePaths: verticalGeometry.visibleVerticalPulsePaths,
  });

  // Step 4: Return the pure geometry bundle used by the canvas renderer.
  return {
    horizontalLineBatches: horizontalGeometry.horizontalLineBatches,
    pulse,
    verticalLineBatches: verticalGeometry.verticalLineBatches,
  };
}