/**
 * Minimal render input required to draw the playback background.
 *
 * The background is intentionally treated as a deterministic camera effect
 * rather than a gameplay-aware renderer. By restricting the contract to
 * viewport geometry, frame index, and scroll position, the module can create a
 * stable neon sky-ground composition without coupling itself to bird state,
 * pipe arrays, or trail caches.
 *
 * @example
 * ```ts
 * const request: PlaybackBackgroundRequest = {
 *   viewportLeftXPx: cameraLeftPx,
 *   visibleWorldWidthPx: 288,
 *   visibleWorldHeightPx: 512,
 *   frameIndex,
 *   scrollBasePx: frameIndex * pipeSpeedPxPerFrame,
 * };
 * ```
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
 *
 * This is the background module's precomputed staging area. The scene service
 * resolves the sky/lower-band split, vanishing point, and horizon styling once
 * so the draw passes can stay orchestration-first and avoid repeating geometry
 * math every frame.
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
 *
 * The layout fixes the classic synthwave composition used by this demo: a tall
 * sky band for layered starfield parallax and a compressed lower strip for the
 * perspective grid. Caching this structure by viewport size keeps redraws cheap
 * when the scene is otherwise stable.
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
 * Zero-argument builder used to lazily construct one cached background layout.
 *
 * The cache service accepts a factory instead of raw data so callers can defer
 * the slightly more expensive layout computation until a viewport-size cache
 * miss actually occurs.
 */
export type PlaybackBackgroundLayoutFactory = () => PlaybackBackgroundLayout;

/**
 * Neon styling contract for the horizon divider line.
 *
 * The horizon is rendered as both a crisp divider and a glow source, much like
 * the luminous skyline separator common in synthwave and TRON-inspired poster
 * art. Keeping those paint properties bundled makes it easier to reason about
 * the horizon as one semantic effect instead of a pile of canvas state.
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
 *
 * This narrow contract is the final handoff from layout math to the canvas
 * stroke helper: world-space x extents, the pixel-snapped y position, and the
 * resolved glow style needed for both line passes.
 */
export type PlaybackHorizonLineRequest = {
  viewportLeftXPx: number;
  visibleWorldWidthPx: number;
  alignedHorizonYPx: number;
  horizonStyle: PlaybackHorizonStyle;
};
