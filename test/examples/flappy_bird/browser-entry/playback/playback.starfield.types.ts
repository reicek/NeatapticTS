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
 * Declarative recipe for building one cached starfield parallax layer.
 */
export type PlaybackStarfieldLayerSpec = {
  seed: number;
  scrollRatio: number;
  starCount: number;
  minSizePx: number;
  maxSizePx: number;
  minAlpha: number;
  maxAlpha: number;
  blurPx: number;
};

/**
 * Input contract for pre-rendering one deterministic starfield tile.
 *
 * The generated tile is cached and repeated horizontally during playback,
 * so every field here affects both the visual look and the parallax cost.
 */
export type CreateStarTileCanvasOptions = {
  seed: number;
  tileWidthPx: number;
  tileHeightPx: number;
  starCount: number;
  minSizePx: number;
  maxSizePx: number;
  minAlpha: number;
  maxAlpha: number;
  blurPx: number;
};

/**
 * Normalized canvas dimensions used by browser and offscreen tile creation.
 */
export type StarfieldCanvasDimensions = {
  width: number;
  height: number;
};

/**
 * Deterministic placement and appearance for one rendered star sprite.
 */
export type StarPlacement = {
  xPx: number;
  yPx: number;
  sizePx: number;
  alpha: number;
};
