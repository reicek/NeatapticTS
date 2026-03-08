import { resolveWorldViewport } from '../../browser-entry.viewport.utils';
import type { PlaybackEdgeBounds } from '../playback.types';

/**
 * Local type contracts for playback frame rendering.
 *
 * These types are extracted from the broader frame renderer so scene state,
 * bird geometry, and trail styling can evolve behind a dedicated module
 * boundary.
 */

/**
 * Shared scene contract used by one playback frame render pass.
 */
export type PlaybackFrameSceneContext = {
  viewport: ReturnType<typeof resolveWorldViewport>;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  cameraLeftPx: number;
  championBirdIndex: number;
  edgeBounds: PlaybackEdgeBounds;
};

/**
 * Pixel-aligned square geometry used by bird paint helpers.
 */
export type PlaybackBirdGeometry = {
  birdSideLengthPx: number;
  birdLeftPx: number;
  birdTopPx: number;
};

/**
 * Resolved opacity and color for one bird trail render pass.
 */
export type PlaybackTrailRenderStyle = {
  baseOpacity: number;
  trailColor: string;
};
