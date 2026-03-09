/**
 * Minimal render input required to draw the playback background.
 *
 * Keeping this contract narrow prevents the background renderer from reaching
 * into unrelated playback state such as birds, pipes, or trail caches.
 */
export type PlaybackBackgroundRequest = {
  viewportLeftXPx: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  frameIndex: number;
  scrollBasePx: number;
};

/**
 * Derived scene contract shared by the playback background render passes.
 */
export type PlaybackBackgroundSceneContext = {
  viewportLeftXPx: number;
  visibleWorldWidthPx: number;
  visibleWorldHeightPx: number;
  skyHeightPx: number;
  lowerBandTopYPx: number;
  lowerBandHeightPx: number;
  lowerBandBottomYPx: number;
  alignedHorizonYPx: number;
  vanishingPointXPx: number;
  vanishingPointYPx: number;
  horizonStyle: PlaybackHorizonStyle;
};

/**
 * Resolved vertical scene split used by playback background composition.
 */
export type PlaybackBackgroundLayout = {
  skyHeightPx: number;
  lowerBandTopYPx: number;
  lowerBandHeightPx: number;
  lowerBandBottomYPx: number;
  horizonYPx: number;
  horizonThicknessPx: number;
};

/**
 * Neon styling contract for the horizon divider line.
 */
export type PlaybackHorizonStyle = {
  lineColor: string;
  glowColor: string;
  glowAlpha: number;
  glowBlurPx: number;
  lineThicknessPx: number;
};

/**
 * Draw request for the horizon divider line.
 */
export type PlaybackHorizonLineRequest = {
  viewportLeftXPx: number;
  visibleWorldWidthPx: number;
  alignedHorizonYPx: number;
  horizonStyle: PlaybackHorizonStyle;
};

/**
 * Horizontal tiled-image draw request.
 */
export type PlaybackTiledImageRowRequest = {
  startXPx: number;
  tile: CanvasImageSource;
  tileWidthPx: number;
  visibleWidthPx: number;
  offsetPx: number;
};
