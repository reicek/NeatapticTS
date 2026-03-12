import { resolveWorldViewport } from '../../browser-entry.viewport.utils';
import type { TrailPoint } from '../../browser-entry.types';
import type { PlaybackEdgeBounds } from '../playback.types';
import type { PlaybackBirdRenderStyle } from '../playback.render.utils';

/**
 * Local type contracts for playback frame rendering.
 *
 * These types are extracted from the broader frame renderer so scene state,
 * bird geometry, and trail styling can evolve behind a dedicated module
 * boundary.
 *
 * Together they describe the inputs and intermediate state needed to paint one
 * world-space frame on the browser canvas.
 */

/**
 * Shared scene contract used by one playback frame render pass.
 *
 * This collects the camera, viewport, and champion metadata that multiple frame
 * render helpers need during the same paint pass.
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
 *
 * Bird geometry is resolved once so the detailed bird renderer can reuse a
 * stable square body box across glow and fill passes.
 */
export type PlaybackBirdGeometry = {
  birdSideLengthPx: number;
  birdLeftPx: number;
  birdTopPx: number;
};

/**
 * Resolved opacity and color for one bird trail render pass.
 *
 * Trails use a lighter-weight style record than birds because they only need a
 * base opacity plus stroke color.
 */
export type PlaybackTrailRenderStyle = {
  baseOpacity: number;
  trailColor: string;
};

/**
 * Bird body renderer contract used by frame-render orchestration helpers.
 *
 * Keeping the renderer as an injected function makes the high-level frame pass
 * independent from the detailed bird-paint implementation.
 */
export type PlaybackBirdRenderer = (
  context: CanvasRenderingContext2D,
  birdYPx: number,
  birdIndex: number,
  championBirdIndex: number,
) => void;

/**
 * Trail style resolver used by frame-render orchestration helpers.
 *
 * This lets orchestration ask for champion-vs-non-champion trail styling
 * without embedding color policy directly in the frame pass.
 */
export type PlaybackTrailStyleResolver = (
  birdIndex: number,
  championBirdIndex: number,
) => PlaybackTrailRenderStyle;

/**
 * Trail segment renderer used by frame-render orchestration helpers.
 *
 * The detailed trail painter owns edge fading and stepped segments; the frame
 * orchestration only decides when to call it.
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
 *
 * This combines geometry and style into one small input object for lower-level
 * drawing helpers.
 */
export type PlaybackBirdPaintInput = {
  birdGeometry: PlaybackBirdGeometry;
  birdRenderStyle: PlaybackBirdRenderStyle;
};
