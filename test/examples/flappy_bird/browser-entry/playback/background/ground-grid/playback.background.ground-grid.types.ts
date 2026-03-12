import type { PlaybackBackgroundSceneContext } from '../playback.background.types';

/**
 * Narrow request required to render the playback ground grid.
 *
 * The ground grid only needs time and world-scroll context. That separation is
 * deliberate: the lower-band renderer should respond to camera motion like a
 * piece of scenic lighting, not reach into gameplay entities or playback HUD
 * state.
 */
export type PlaybackBackgroundGroundGridRequest = {
  frameIndex: number;
  scrollBasePx: number;
};

/**
 * Lazy builder used when one horizontal geometry cache entry is missing.
 *
 * Horizontal depth bands depend only on stable scene dimensions, so callers can
 * defer their construction until a cache miss proves the work is actually
 * needed.
 */
export type PlaybackGroundGridHorizontalGeometryFactory =
  () => PlaybackGroundGridHorizontalGeometry;

/**
 * Lazy builder used when one vertical geometry cache entry is missing.
 *
 * Vertical rays additionally depend on wrapped scroll state, so the cache keeps
 * a compact builder hook rather than eagerly storing every possible cycle.
 */
export type PlaybackGroundGridVerticalGeometryFactory =
  () => PlaybackGroundGridVerticalGeometry;

/**
 * Lazy builder used when one vertical scene-metrics cache entry is missing.
 *
 * Scene metrics such as projected anchor bounds are pure functions of the
 * viewport, which makes them ideal cache inputs for the performance-sensitive
 * lower-band renderer.
 */
export type PlaybackGroundGridVerticalSceneMetricsFactory =
  () => PlaybackGroundGridVerticalSceneMetrics;

/**
 * Immutable scene context resolved for one lower-band ground-grid pass.
 *
 * This is the adapted subset of the shared background scene that the ground
 * grid cares about: horizon position, lower-band bounds, and the centered
 * vanishing point that gives the grid its forced-perspective look.
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
 *
 * The ground grid uses three coordinated colors: the structural line work, a
 * fog wash that softens the lower band, and small pulse markers that briefly
 * travel along eligible tracks.
 */
export type PlaybackBackgroundGroundGridStyle = {
  lineColor: string;
  fogColor: string;
  pulseFillColor: string;
};

/**
 * Travel orientation used by lightweight pulse overlays.
 *
 * Horizontal pulses skim along depth bands, while vertical pulses ride the
 * perspective rays toward or away from the horizon.
 */
export type PlaybackGroundGridPulseOrientation = 'horizontal' | 'vertical';

/**
 * Geometry and style package resolved before drawing the ground grid.
 *
 * Splitting scene resolution from drawing keeps canvas code simple: by the time
 * the renderer runs, every geometric and palette decision has already been made
 * and packed into one immutable object graph.
 */
export type PlaybackBackgroundGroundGridResolvedScene = {
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
  style: PlaybackBackgroundGroundGridStyle;
};

/**
 * Declarative line segment model used by the ground-grid renderer.
 *
 * The grid is built as plain data first so batch builders can group segments by
 * shared style before any canvas path or glow work happens.
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

/**
 * Ordered batch of line segments that share one render style.
 *
 * Grouping segments by alpha, blur, and thickness reduces canvas state churn
 * and gives the renderer a natural place to cache `Path2D` instances when the
 * environment supports them.
 */
export type PlaybackGroundGridSegmentBatch = {
  alpha: number;
  blurPx: number;
  path: Path2D | null;
  thicknessPx: number;
  segments: readonly PlaybackGroundGridLineSegment[];
};

/**
 * Internal helper contract used while generating vertical-ray sub-segments.
 *
 * Each vertical ray is split into depth-aware pieces so stroke thickness and
 * glow can evolve as the ray approaches the viewer instead of staying uniform.
 */
export type PlaybackGroundGridVerticalRayInput = PlaybackGroundGridPulsePath & {
  lineDepthRatio: number;
  maximumDistanceToHorizonPx: number;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
  verticalSegmentCount: number;
};

/**
 * Simplified path used by one visible pulse event.
 *
 * Pulses do not need the full batch geometry; they only need a single lane to
 * travel along, represented here as a start-end segment plus local thickness.
 */
export type PlaybackGroundGridPulsePath = {
  orientation: PlaybackGroundGridPulseOrientation;
  startXPx: number;
  startYPx: number;
  endXPx: number;
  endYPx: number;
  thicknessPx: number;
};

/**
 * One visible pulse square rendered above the grid lines.
 *
 * These pulses are tiny accent lights, not gameplay markers. Their job is to
 * give the lower band a hint of moving circuitry without competing with birds,
 * pipes, or the active network HUD.
 */
export type PlaybackGroundGridPulse = {
  centerXPx: number;
  centerYPx: number;
  sizePx: number;
  alpha: number;
};

/**
 * Pure geometry bundle generated before canvas drawing begins.
 *
 * Horizontal bands, vertical rays, and the optional pulse are resolved into one
 * package so the draw layer can stay strictly about paint order.
 */
export type PlaybackGroundGridGeometry = {
  horizontalLineBatches: readonly PlaybackGroundGridSegmentBatch[];
  pulse: PlaybackGroundGridPulse | null;
  verticalLineBatches: readonly PlaybackGroundGridSegmentBatch[];
};

/**
 * Cached horizontal geometry bundle reused across matching scene sizes.
 *
 * Horizontal depth lines are stable for a given viewport, which makes them the
 * cheapest part of the grid to cache aggressively.
 */
export type PlaybackGroundGridHorizontalGeometry = {
  horizontalLineBatches: readonly PlaybackGroundGridSegmentBatch[];
  horizontalLines: readonly PlaybackGroundGridLineSegment[];
  preferredHorizontalPulsePaths: readonly PlaybackGroundGridPulsePath[];
};

/**
 * Cached vertical geometry bundle reused across one wrapped scroll cycle.
 *
 * Vertical rays move with scroll, but only within a repeating wrapped cycle.
 * Caching at that granularity captures most of the reuse without pretending the
 * rays are globally static.
 */
export type PlaybackGroundGridVerticalGeometry = {
  verticalLineBatches: readonly PlaybackGroundGridSegmentBatch[];
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
  visibleVerticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
};

/**
 * Cached scene metrics reused across matching vertical-grid frames.
 *
 * These metrics answer the expensive geometric questions once per viewport,
 * such as how wide the visible anchor span is and how many perspective lanes
 * can fit while preserving the intended spacing.
 */
export type PlaybackGroundGridVerticalSceneMetrics = {
  visibleAnchorBounds: PlaybackGroundGridAnchorBounds;
  totalVisibleLaneCount: number;
  safeLaneSpacingPx: number;
};

/**
 * Visible horizon bounds projected onto the bottom anchor line.
 *
 * Perspective rays begin conceptually at the horizon and terminate on the floor
 * anchor line, so this structure captures the visible lane span after the
 * horizon segment has been projected downward.
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
 *
 * The perspective rays repeat on a fixed cycle. Wrapping the scroll state lets
 * the renderer reuse cached geometry while still appearing to drift sideways.
 */
export type PlaybackGroundGridVerticalCycleContext = {
  wrappedOffsetPx: number;
  safeLaneSpacingPx: number;
};

/**
 * Input contract used while resolving one deterministic pulse event.
 *
 * Pulses are deterministic decoration: given the same frame and scene, the same
 * lane should light up. This input bundle gathers the candidate paths needed to
 * make that choice without consulting external state.
 */
export type PlaybackGroundGridPulseInput = {
  frameIndex: number;
  horizontalPulsePaths: readonly PlaybackGroundGridPulsePath[];
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
  visibleVerticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[];
};

/**
 * Timing state resolved for one deterministic ground-grid pulse slot.
 *
 * The slot model keeps pulse timing legible: each pulse has a start bucket, an
 * elapsed time inside that bucket, and a normalized progress value used by the
 * position and fade helpers.
 */
export type PlaybackGroundGridPulseTimingState = {
  pulseSlotIndex: number;
  pulseElapsedMs: number;
  lifetimeProgressRatio: number;
};

/**
 * Direction and timing state used when resolving pulse travel progress.
 *
 * Some pulse lanes feel better moving away from the viewer and others toward
 * it, so travel resolution keeps orientation and forward/reverse intent paired
 * with the current lifetime progress.
 */
export type PlaybackGroundGridPulseTravelRatioInput = {
  directionIsForward: boolean;
  lifetimeProgressRatio: number;
  orientation: PlaybackGroundGridPulseOrientation;
};

/**
 * Input used when adapting a pulse position into a local track thickness.
 *
 * Because the grid uses perspective-weighted stroke widths, the pulse size must
 * be adjusted to the local lane thickness rather than using one fixed square.
 */
export type PlaybackGroundGridPulseTrackThicknessInput = {
  pulseCenterYPx: number;
  pulsePath: PlaybackGroundGridPulsePath;
  sceneContext: PlaybackBackgroundGroundGridSceneContext;
};

/**
 * Cached continuation state used to keep one vertical pulse on the same ray.
 *
 * Without this continuity state a vertical pulse could jitter between adjacent
 * rays across frames, which looks like noise instead of a deliberate light.
 */
export type PlaybackGroundGridVerticalPulseContinuationState = {
  centerXPx: number;
  centerYPx: number;
  frameIndex: number;
};

/**
 * Helper alias used when adapting the shared background scene context.
 *
 * The parent background module owns the full sky-plus-ground scene contract;
 * the grid renderer narrows that shape to only the fields needed for the lower
 * band so the dependency direction stays clear.
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
