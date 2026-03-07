/**
 * Shared type contract for starfield tile rendering layers.
 *
 * A tile is pre-rendered and repeated horizontally to draw efficient
 * parallax backgrounds during playback.
 */
export type StarTile = {
  image: CanvasImageSource;
  tileWidthPx: number;
  tileHeightPx: number;
  scrollRatio: number;
};

/**
 * Axis-aligned visible world bounds used for edge-aware trail fading.
 */
export type PlaybackEdgeBounds = {
  leftXPx: number;
  rightXPx: number;
  topYPx: number;
  bottomYPx: number;
};
