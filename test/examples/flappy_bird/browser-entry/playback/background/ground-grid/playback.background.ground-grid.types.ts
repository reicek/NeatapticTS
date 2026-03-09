import type { PlaybackBackgroundSceneContext } from '../playback.background.types';

/**
 * Narrow request required to render the playback ground grid.
 */
export type PlaybackBackgroundGroundGridRequest = {
  frameIndex: number;
  scrollBasePx: number;
};

/**
 * Immutable scene context resolved for one lower-band ground-grid pass.
 */
export type PlaybackBackgroundGroundGridSceneContext = {
  viewportLeftXPx: number;
  visibleWorldWidthPx: number;
  alignedHorizonYPx: number;
  lowerBandTopYPx: number;
  lowerBandHeightPx: number;
  lowerBandBottomYPx: number;
  vanishingPointXPx: number;
  vanishingPointYPx: number;
};

/**
 * Theme-owned style contract for the neon ground grid.
 */
export type PlaybackBackgroundGroundGridStyle = {
  lineColor: string;
  glowColor: string;
  fogColor: string;
  pulseFillColor: string;
  pulseGlowColor: string;
};

/** Travel orientation used by lightweight pulse overlays. */
export type PlaybackGroundGridPulseOrientation = 'horizontal' | 'vertical';

/**
 * Geometry and style package resolved before drawing the ground grid.
 */
export type PlaybackBackgroundGroundGridResolvedScene = {
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
  style: PlaybackBackgroundGroundGridStyle;
};

/**
 * Declarative line segment model used by the ground-grid renderer.
 */
export type PlaybackGroundGridLineSegment = {
  startXPx: number;
  startYPx: number;
  endXPx: number;
  endYPx: number;
  alpha: number;
  blurPx: number;
  thicknessPx: number;
};

/** Simplified path used by one visible pulse event. */
export type PlaybackGroundGridPulsePath = {
  orientation: PlaybackGroundGridPulseOrientation;
  startXPx: number;
  startYPx: number;
  endXPx: number;
  endYPx: number;
  thicknessPx: number;
};

/** One visible pulse square rendered above the grid lines. */
export type PlaybackGroundGridPulse = {
  centerXPx: number;
  centerYPx: number;
  sizePx: number;
  alpha: number;
  glowBlurPx: number;
};

/**
 * Pure geometry bundle generated before canvas drawing begins.
 */
export type PlaybackGroundGridGeometry = {
  horizontalLines: readonly PlaybackGroundGridLineSegment[];
  pulse: PlaybackGroundGridPulse | null;
  verticalLines: readonly PlaybackGroundGridLineSegment[];
};

/**
 * Helper alias used when adapting the shared background scene context.
 */
export type PlaybackBackgroundGroundGridSourceScene = Pick<
  PlaybackBackgroundSceneContext,
  | 'viewportLeftXPx'
  | 'visibleWorldWidthPx'
  | 'alignedHorizonYPx'
  | 'lowerBandTopYPx'
  | 'lowerBandHeightPx'
  | 'lowerBandBottomYPx'
  | 'vanishingPointXPx'
  | 'vanishingPointYPx'
>;
