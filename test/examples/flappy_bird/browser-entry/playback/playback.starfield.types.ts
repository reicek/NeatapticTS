/**
 * Starfield and parallax rendering contracts for playback backgrounds.
 *
 * The playback view uses a cached layered starfield to add depth without paying
 * a large per-frame rendering cost. These types define the tile, layer, and
 * deterministic placement data needed for that effect.
 */

/**
 * Shared type contract for starfield tile rendering layers.
 *
 * A tile is pre-rendered and repeated horizontally to draw efficient
 * parallax backgrounds during playback.
 */
export type StarTileImage = HTMLCanvasElement | OffscreenCanvas;

/**
 * Shared type contract for starfield tile rendering layers.
 *
 * A tile is pre-rendered and repeated horizontally to draw efficient
 * parallax backgrounds during playback.
 *
 * Separating the image type from the tile record lets the same starfield logic
 * work with ordinary canvases and `OffscreenCanvas` when available.
 */
export type StarTile = {
  image: StarTileImage;
  tileWidthPx: number;
  tileHeightPx: number;
  scrollRatio: number;
};

/**
 * Declarative recipe for building one cached starfield parallax layer.
 *
 * Each layer spec describes how dense, bright, blurred, and fast one visual
 * depth plane should feel.
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
 *
 * The creation path works with both `HTMLCanvasElement` and `OffscreenCanvas`,
 * so dimensions are stored in a narrow shared shape rather than tied to one DOM
 * type.
 */
export type StarfieldCanvasDimensions = {
  width: number;
  height: number;
};

/**
 * Deterministic placement and appearance for one rendered star sprite.
 *
 * Determinism matters here because cached starfield tiles should remain stable
 * across redraws instead of sparkling randomly every frame.
 */
export type StarPlacement = {
  xPx: number;
  yPx: number;
  sizePx: number;
  alpha: number;
};
