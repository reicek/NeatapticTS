import {
  resolvePlaybackGroundGridHorizontalLines,
  resolvePlaybackGroundGridVerticalLines,
} from './playback.background.ground-grid.geometry.utils';
import { resolvePlaybackGroundGridPulse } from './playback.background.ground-grid.pulse.utils';
import type {
  PlaybackBackgroundGroundGridSceneContext,
  PlaybackGroundGridGeometry,
} from './playback.background.ground-grid.types';

export {
  interpolatePlaybackGroundGridPoint,
  resolvePlaybackGroundGridDepthCurve,
  resolvePlaybackGroundGridDepthFromHorizonDistance,
  resolvePlaybackGroundGridLineAlpha,
  resolvePlaybackGroundGridLineBlur,
  resolvePlaybackGroundGridLineThickness,
} from './playback.background.ground-grid.math.utils';

/**
 * Resolves the shared scene context used by the ground-grid renderer.
 *
 * @param sceneContext - Lower-band geometry provided by the background module.
 * @returns Narrow scene contract consumed by grid-specific helpers.
 */
export function resolvePlaybackGroundGridSceneContext(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): PlaybackBackgroundGroundGridSceneContext {
  return sceneContext;
}

/**
 * Builds the line geometry for the neon ground grid.
 *
 * @param sceneContext - Lower-band geometry for the current viewport.
 * @param scrollBasePx - Shared world scroll used for parallax motion.
 * @returns Horizontal depth bands and perspective rays for the current frame.
 */
export function resolvePlaybackGroundGridGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  frameIndex: number,
  scrollBasePx: number,
): PlaybackGroundGridGeometry {
  // Step 1: Resolve the fixed horizontal depth bands for the lower plane.
  const horizontalLines =
    resolvePlaybackGroundGridHorizontalLines(sceneContext);

  // Step 2: Resolve the moving perspective rays with wrapped anchor spacing.
  const { verticalLines, verticalPulsePaths } =
    resolvePlaybackGroundGridVerticalLines(sceneContext, scrollBasePx);

  // Step 3: Resolve one visible pulse above the grid lines.
  const pulse = resolvePlaybackGroundGridPulse({
    frameIndex,
    horizontalLines,
    sceneContext,
    verticalPulsePaths,
  });

  // Step 4: Return the pure geometry bundle used by the canvas renderer.
  return {
    horizontalLines,
    pulse,
    verticalLines,
  };
}
