import type { PlaybackBackgroundSceneContext } from '../playback.background.types';

/**
 * Narrow request required to render the playback ground grid.
 */
export type PlaybackBackgroundGroundGridRequest = {
  frameIndex: number;
  scrollBasePx: number;
};

/**
 * Lazy builder used when one horizontal geometry cache entry is missing.
 */
export type PlaybackGroundGridHorizontalGeometryFactory =
  () => PlaybackGroundGridHorizontalGeometry;

/**
 * Lazy builder used when one vertical geometry cache entry is missing.
 */
export type PlaybackGroundGridVerticalGeometryFactory =
  () => PlaybackGroundGridVerticalGeometry;

/**
 * Immutable scene context resolved for one lower-band ground-grid pass.
 */
export type PlaybackBackgroundGroundGridSceneContext = {
  viewportOffsetXPx: number;
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
  fogColor: string;
  pulseFillColor: string;
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

/** Ordered batch of line segments that share one render style. */
export type PlaybackGroundGridSegmentBatch = {
  alpha: number;
  blurPx: number;
  path: Path2D | null;
  thicknessPx: number;
  segments: readonly PlaybackGroundGridLineSegment[];
};

/**
 * Internal helper contract used while generating vertical-ray sub-segments.
 */
export type PlaybackGroundGridVerticalRayInput = PlaybackGroundGridPulsePath & {
  lineDepthRatio: number;
  maximumDistanceToHorizonPx: number;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
  verticalSegmentCount: number;
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
};

/**
 * Pure geometry bundle generated before canvas drawing begins.
 */
export type PlaybackGroundGridGeometry = {
  horizontalLineBatches: readonly PlaybackGroundGridSegmentBatch[];
  pulse: PlaybackGroundGridPulse | null;
  verticalLineBatches: readonly PlaybackGroundGridSegmentBatch[];
};

/**
 * Cached horizontal geometry bundle reused across matching scene sizes.
 */
export type PlaybackGroundGridHorizontalGeometry = {
  horizontalLineBatches: readonly PlaybackGroundGridSegmentBatch[];
  horizontalLines: readonly PlaybackGroundGridLineSegment[];
  preferredHorizontalPulsePaths: readonly PlaybackGroundGridPulsePath[];
};

/**
 * Cached vertical geometry bundle reused across one wrapped scroll cycle.
 */
export type PlaybackGroundGridVerticalGeometry = {
  verticalLineBatches: readonly PlaybackGroundGridSegmentBatch[];
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
  visibleVerticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
};

/**
 * Visible horizon bounds projected onto the bottom anchor line.
 */
export type PlaybackGroundGridAnchorBounds = {
  leftAnchorXPx: number;
  rightAnchorXPx: number;
  anchorSpanPx: number;
};

/**
 * Input used when projecting a visible horizon span onto anchor space.
 */
export type PlaybackGroundGridAnchorBoundsInput = {
  horizonLeftXPx: number;
  horizonRightXPx: number;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
};

/**
 * Input used when projecting one horizon x-position onto the floor anchor line.
 */
export type PlaybackGroundGridAnchorProjectionInput = {
  horizonXPx: number;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
};

/**
 * Wrapped vertical-cycle state derived from scroll for one frame.
 */
export type PlaybackGroundGridVerticalCycleContext = {
  quantizedWrappedOffsetPx: number;
  safeLaneSpacingPx: number;
};

/**
 * Input contract used while resolving one deterministic pulse event.
 */
export type PlaybackGroundGridPulseInput = {
  frameIndex: number;
  horizontalPulsePaths: readonly PlaybackGroundGridPulsePath[];
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
  visibleVerticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
};

/**
 * Direction and timing state used when resolving pulse travel progress.
 */
export type PlaybackGroundGridPulseTravelRatioInput = {
  directionIsForward: boolean;
  lifetimeProgressRatio: number;
  orientation: PlaybackGroundGridPulseOrientation;
};

/**
 * Input used when adapting a pulse position into a local track thickness.
 */
export type PlaybackGroundGridPulseTrackThicknessInput = {
  pulseCenterYPx: number;
  pulsePath: PlaybackGroundGridPulsePath;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
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
