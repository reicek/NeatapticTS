import { resolveWorldViewport } from '../../browser-entry.viewport.utils';
import type {
  TrailPoint,
} from '../../browser-entry.types';
import type { PlaybackEdgeBounds } from '../playback.types';
import type { PlaybackBirdRenderStyle } from '../playback.render.utils';

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

/**
 * Bird body renderer contract used by frame-render orchestration helpers.
 */
export type PlaybackBirdRenderer = (
  context: CanvasRenderingContext2D,
  birdYPx: number,
  birdIndex: number,
  championBirdIndex: number,
) => void;

/**
 * Trail style resolver used by frame-render orchestration helpers.
 */
export type PlaybackTrailStyleResolver = (
  birdIndex: number,
  championBirdIndex: number,
) => PlaybackTrailRenderStyle;

/**
 * Trail segment renderer used by frame-render orchestration helpers.
 */
export type PlaybackTrailRenderer = (
  context: CanvasRenderingContext2D,
  trailPoints: TrailPoint[],
  color: string,
  anchorX: number,
  baseOpacity: number,
  edgeBounds: PlaybackEdgeBounds,
) => void;

/**
 * Bird-paint input shared by the bird render helper module.
 */
export type PlaybackBirdPaintInput = {
  birdGeometry: PlaybackBirdGeometry;
  birdRenderStyle: PlaybackBirdRenderStyle;
};
