/**
 * Canvas 2D renderer for the Tier 0 racing curriculum demo.
 *
 * Rendering is intentionally flat and stateless relative to game logic —
 * the renderer consumes a frozen `TrackSpec`, the current `EnvironmentState`,
 * and a small mutable `RacingRenderState` (tire marks) and produces one frame.
 *
 * Visual style: neon-retro-arcade — dark background, cyan/blue structure,
 * square-outline car, fading tire marks, yellow heading indicator.
 *
 * The world coordinate system is math-convention (Y increases upward).
 * Canvas pixels use Y-down convention. The affine `WorldTransform` absorbs
 * both the scale and the canvas centre offset; no explicit Y-flip is needed
 * because positive world-Y maps to positive canvas-Y (lower on screen) and
 * both the car heading render and the physics movement are consistent in the
 * same coordinate frame.
 */

import type { EnvironmentState, TireStateTuple } from '../environment/environment.types';
import type { TrackSpec } from '../track/track.generator.types';
import type { RacingRenderFrame } from '../workers/simulation-worker/simulation-worker.types';

// ── Color palette ────────────────────────────────────────────────────────────

const COLOR_BACKGROUND = '#060b14';
const COLOR_TRACK_SURFACE = '#0c1e35';
const COLOR_TRACK_EDGE = '#00d4f5';
const COLOR_TRACK_GLOW_ALPHA = 0.15;
const COLOR_CENTERLINE = 'rgba(0,180,220,0.30)';
const COLOR_GUIDANCE_LINE_RGB = '255,209,102';
const COLOR_CAR_BODY = '#00e5ff';
const COLOR_HEADING_INDICATOR = '#ffcc00';
const COLOR_TIRE_MARK_MAX_ALPHA = 0.52;
const COLOR_TIRE_GOOD = '#22c55e';
const COLOR_TIRE_WARN = '#facc15';
const COLOR_TIRE_ALERT = '#fb923c';
const COLOR_TIRE_CRITICAL = '#ef4444';
const COLOR_PIT_TEAM_A = 'rgba(0, 229, 255, 0.38)';
const COLOR_PIT_TEAM_B = 'rgba(255, 122, 69, 0.38)';
const COLOR_PIT_OCCUPIED = 'rgba(255, 204, 0, 0.16)';

// ── Geometry constants ───────────────────────────────────────────────────────

/** Padding multiplier applied around the track bounding box before scaling. */
const WORLD_PADDING_RATIO = 1.16;
/** Half-length of the car rectangle in world units (front/back). */
const CAR_HALF_LENGTH_WORLD = 3.8;
/** Half-width of the car rectangle in world units (left/right). */
const CAR_HALF_WIDTH_WORLD = 2.2;
/** Length of the forward heading indicator in world units. */
const HEADING_INDICATOR_WORLD_LENGTH = 5.5;
/** Stroke width in CSS pixels used for the cyan boundary lines. */
const TRACK_EDGE_LINE_WIDTH_PX = 2.4;
/** Blur radius in CSS pixels used for the boundary glow. */
const TRACK_GLOW_BLUR_PX = 18;

// ── Tire-mark accumulation constants ────────────────────────────────────────

/** Maximum age (in ticks) before a tire mark is discarded. */
const TIRE_MARK_MAX_AGE_TICKS = 260;
/** Minimum ticks between successive tire-mark samples. */
const TIRE_MARK_SAMPLE_INTERVAL_TICKS = 3;

// ── Types ────────────────────────────────────────────────────────────────────

/** Canvas-space coordinate pair produced by the world transform. */
type CanvasPoint = { readonly x: number; readonly y: number };
/** World-space coordinate pair used while building cached track geometry. */
type WorldPoint = { readonly x: number; readonly y: number };
/** One sampled centerline point with its interpolated track width. */
type TrackSamplePoint = WorldPoint & { readonly width: number };
/** Cached spline-derived geometry used by transform and draw helpers. */
type TrackRenderGeometry = {
  readonly centerlinePoints: readonly TrackSamplePoint[];
  readonly leftBoundaryPoints: readonly WorldPoint[];
  readonly rightBoundaryPoints: readonly WorldPoint[];
  readonly minX: number;
  readonly maxX: number;
  readonly minY: number;
  readonly maxY: number;
};

const trackRenderGeometryCache = new WeakMap<TrackSpec, TrackRenderGeometry>();

/**
 * Affine transform mapping world units to canvas pixels.
 *
 * Computed once per episode from the track bounding box and canvas size.
 * Recompute if the canvas is resized.
 */
export interface WorldTransform {
  /** Uniform scale factor: canvas pixels per world unit. */
  readonly scale: number;
  /** Horizontal canvas offset so the track centre is at the canvas centre. */
  readonly offsetX: number;
  /** Vertical canvas offset so the track centre is at the canvas centre. */
  readonly offsetY: number;
}

/** One sampled point in the fading tire-mark trail. */
export interface TireMark {
  readonly worldX: number;
  readonly worldY: number;
  /** Ticks elapsed since this mark was recorded. Mutated each frame. */
  age: number;
}

/**
 * Mutable render state owned by the animation loop.
 *
 * Isolated from the physics `EnvironmentState` so that rendering artefacts
 * (trail length, mark density) can be tuned without touching the simulation.
 */
export interface RacingRenderState {
  /** Ordered list of sampled car positions forming the tire-mark trail. */
  readonly tireMarks: TireMark[];
  /** Ticks elapsed since the last mark was appended. */
  ticksSinceLastMark: number;
}

/** Narrow worker-frame fields consumed by the Tier 4 renderer overlays. */
type RacingRenderOverlayFrame = Pick<RacingRenderFrame, 'pitStatus' | 'tireState'>;

/**
 * Optional overlays layered on top of the base track render.
 *
 * Tier 4 uses `frame.tireState` for live corner colors and `frame.pitStatus`
 * for pit occupancy overlays. `pitStatus` follows the packed tuple
 * `[teamA_car, teamA_ticks, teamB_car, teamB_ticks]`.
 */
export interface RacingRenderOptions {
  /** Faded optimal-line guidance alpha in [0, 1]. */
  readonly guidanceAlpha?: number;
  /** Optional packed worker frame used for Tier 4 tire/pit overlays. */
  readonly frame?: RacingRenderOverlayFrame;
  /** Focused car index inside `frame.tireState`; defaults to `0`. */
  readonly focusCarIndex?: number;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Creates a zeroed `RacingRenderState` ready for first use.
 *
 * @returns Fresh render state with an empty tire-mark list.
 *
 * @example
 * ```ts
 * const renderState = createRacingRenderState();
 * renderRacingFrame(canvas, spec, envState, renderState, transform);
 * ```
 */
export function createRacingRenderState(): RacingRenderState {
  return { tireMarks: [], ticksSinceLastMark: 0 };
}

/**
 * Computes a world-to-canvas affine transform that fits the entire track
 * inside the canvas with uniform padding on all sides.
 *
 * @param canvas - Target canvas (uses `width` and `height` attributes).
 * @param spec - Frozen track spec whose bounding box determines the scale.
 * @returns Affine transform for use in `renderRacingFrame`.
 *
 * @example
 * ```ts
 * const transform = computeWorldTransform(canvasElement, trackSpec);
 * ```
 */
export function computeWorldTransform(
  canvas: HTMLCanvasElement,
  spec: TrackSpec,
): WorldTransform {
  const trackRenderGeometry = getTrackRenderGeometry(spec);
  const paddedWorldSpanX =
    (trackRenderGeometry.maxX - trackRenderGeometry.minX) * WORLD_PADDING_RATIO;
  const paddedWorldSpanY =
    (trackRenderGeometry.maxY - trackRenderGeometry.minY) * WORLD_PADDING_RATIO;
  const scale = Math.min(
    canvas.width / paddedWorldSpanX,
    canvas.height / paddedWorldSpanY,
  );
  const trackCentreWorldX =
    (trackRenderGeometry.minX + trackRenderGeometry.maxX) / 2;
  const trackCentreWorldY =
    (trackRenderGeometry.minY + trackRenderGeometry.maxY) / 2;

  return {
    scale,
    offsetX: canvas.width / 2 - trackCentreWorldX * scale,
    offsetY: canvas.height / 2 - trackCentreWorldY * scale,
  };
}

/**
 * Renders one animation frame onto the canvas.
 *
 * Rendering order: background → track glow → track surface → track edges →
 * centerline dashes → tire marks → car body + heading indicator.
 *
 * When `renderOptions.frame` is present, the renderer also colors the four tire
 * corners from the packed Tier 4 tire tuple and draws pit entrance/stall
 * overlays from the packed pit-status tuple.
 *
 * Mutates `renderState.tireMarks` and `renderState.ticksSinceLastMark`.
 *
 * @param canvas - Target canvas element.
 * @param spec - Frozen track spec (geometry only).
 * @param envState - Current physics state from the simulation.
 * @param renderState - Mutable tire-mark accumulator.
 * @param transform - World-to-canvas affine transform.
 * @param renderOptions - Optional overlay configuration; defaults to no guidance overlay.
 */
export function renderRacingFrame(
  canvas: HTMLCanvasElement,
  spec: TrackSpec,
  envState: EnvironmentState,
  renderState: RacingRenderState,
  transform: WorldTransform,
  renderOptions: RacingRenderOptions = {},
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Step 1: Advance tire marks for this tick.
  advanceTireMarks(renderState, envState);

  // Step 2: Background fill.
  ctx.fillStyle = COLOR_BACKGROUND;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Step 3: Track layers (glow, surface, edges, centerline).
  drawTrack(
    ctx,
    spec,
    transform,
    renderOptions.guidanceAlpha ?? 0,
    renderOptions.frame?.pitStatus,
  );

  // Step 4: Fading tire-mark trail.
  drawTireMarks(ctx, renderState.tireMarks, transform);

  // Step 5: Car body and heading indicator.
  drawCar(
    ctx,
    envState,
    transform,
    resolveRenderTireState(
      envState,
      renderOptions.frame?.tireState,
      renderOptions.focusCarIndex ?? 0,
    ),
  );
}

// ── Private helpers ───────────────────────────────────────────────────────────

/**
 * Appends a new tire mark at the car's current position and ages all marks.
 * Marks older than `TIRE_MARK_MAX_AGE_TICKS` are evicted from the front.
 *
 * @param renderState - Mutable render state (mutated in-place).
 * @param envState - Current car position source.
 */
function advanceTireMarks(
  renderState: RacingRenderState,
  envState: EnvironmentState,
): void {
  renderState.ticksSinceLastMark++;

  if (renderState.ticksSinceLastMark >= TIRE_MARK_SAMPLE_INTERVAL_TICKS) {
    renderState.tireMarks.push({
      worldX: envState.carX,
      worldY: envState.carY,
      age: 0,
    });
    renderState.ticksSinceLastMark = 0;
  }

  for (const mark of renderState.tireMarks) {
    mark.age++;
  }

  while (
    renderState.tireMarks.length > 0 &&
    (renderState.tireMarks[0]?.age ?? 0) > TIRE_MARK_MAX_AGE_TICKS
  ) {
    renderState.tireMarks.shift();
  }
}

/**
 * Draws all track layers onto the canvas context.
 *
 * Layers (back to front): glow halo, asphalt surface, left/right edge lines,
 * dashed centerline.
 *
 * @param ctx - 2D rendering context.
 * @param spec - Frozen track geometry.
 * @param transform - World-to-canvas affine transform.
 */
function drawTrack(
  ctx: CanvasRenderingContext2D,
  spec: TrackSpec,
  transform: WorldTransform,
  guidanceAlpha: number,
  pitStatus?: Uint8Array | Uint16Array | Int16Array,
): void {
  const trackRenderGeometry = getTrackRenderGeometry(spec);
  drawTrackGlowLayer(ctx, trackRenderGeometry, transform);
  drawTrackSurfaceLayer(ctx, trackRenderGeometry, transform);
  drawTrackEdgeLines(ctx, trackRenderGeometry, transform);
  drawTrackCenterline(ctx, trackRenderGeometry, transform);
  drawOptimalLineGuidance(ctx, trackRenderGeometry, transform, guidanceAlpha);
  drawPitOverlays(ctx, spec, transform, pitStatus);
}

/**
 * Draws a wide semi-transparent glow behind each track segment.
 *
 * @param ctx - 2D rendering context.
 * @param spec - Track geometry.
 * @param transform - World-to-canvas transform.
 */
function drawTrackGlowLayer(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void {
  ctx.save();
  ctx.fillStyle = `rgba(0, 212, 245, ${COLOR_TRACK_GLOW_ALPHA})`;
  ctx.shadowColor = COLOR_TRACK_EDGE;
  ctx.shadowBlur = TRACK_GLOW_BLUR_PX;
  traceTrackRibbonPath(ctx, trackRenderGeometry, transform);
  ctx.fill();

  ctx.restore();
}

/**
 * Fills the track surface with the dark asphalt colour.
 *
 * @param ctx - 2D rendering context.
 * @param spec - Track geometry.
 * @param transform - World-to-canvas transform.
 */
function drawTrackSurfaceLayer(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void {
  ctx.strokeStyle = COLOR_TRACK_SURFACE;
  ctx.fillStyle = COLOR_TRACK_SURFACE;
  traceTrackRibbonPath(ctx, trackRenderGeometry, transform);
  ctx.fill();
}

/**
 * Draws the bright cyan edge lines on both sides of each segment using
 * perpendicular offset geometry.
 *
 * @param ctx - 2D rendering context.
 * @param spec - Track geometry.
 * @param transform - World-to-canvas transform.
 */
function drawTrackEdgeLines(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void {
  ctx.strokeStyle = COLOR_TRACK_EDGE;
  ctx.lineWidth = TRACK_EDGE_LINE_WIDTH_PX;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  traceClosedWorldPath(ctx, trackRenderGeometry.leftBoundaryPoints, transform);
  ctx.stroke();

  traceClosedWorldPath(ctx, trackRenderGeometry.rightBoundaryPoints, transform);
  ctx.stroke();
}

/**
 * Draws a dashed centerline along each segment.
 *
 * @param ctx - 2D rendering context.
 * @param spec - Track geometry.
 * @param transform - World-to-canvas transform.
 */
function drawTrackCenterline(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void {
  ctx.save();
  ctx.setLineDash([10, 12]);
  ctx.strokeStyle = COLOR_CENTERLINE;
  ctx.lineWidth = 1.2;
  traceClosedSamplePath(ctx, trackRenderGeometry.centerlinePoints, transform);
  ctx.stroke();

  ctx.setLineDash([]);
  ctx.restore();
}

/**
 * Draws the faded optimal-line overlay used by the Tier 1 browser harness.
 *
 * @param ctx - 2D rendering context.
 * @param trackRenderGeometry - Cached spline-derived track geometry.
 * @param transform - World-to-canvas transform.
 * @param guidanceAlpha - Overlay alpha in [0, 1].
 */
function drawOptimalLineGuidance(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
  guidanceAlpha: number,
): void {
  const clampedGuidanceAlpha = Math.max(0, Math.min(1, guidanceAlpha));

  if (clampedGuidanceAlpha <= 0) {
    return;
  }

  ctx.save();
  ctx.setLineDash([18, 10]);
  ctx.strokeStyle = `rgba(${COLOR_GUIDANCE_LINE_RGB}, ${clampedGuidanceAlpha.toFixed(3)})`;
  ctx.lineWidth = 2.1;
  ctx.shadowColor = `rgba(${COLOR_GUIDANCE_LINE_RGB}, ${Math.min(0.75, clampedGuidanceAlpha + 0.1).toFixed(3)})`;
  ctx.shadowBlur = 12;
  traceClosedSamplePath(ctx, trackRenderGeometry.centerlinePoints, transform);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

/**
 * Resolves cached spline-derived geometry for the given `TrackSpec`.
 *
 * @param spec - Frozen track geometry.
 * @returns Cached centerline/boundary geometry.
 */
function getTrackRenderGeometry(spec: TrackSpec): TrackRenderGeometry {
  const cachedGeometry = trackRenderGeometryCache.get(spec);
  if (cachedGeometry !== undefined) {
    return cachedGeometry;
  }

  const centerlinePoints = spec.splineSamples;
  const leftBoundaryPoints: WorldPoint[] = [];
  const rightBoundaryPoints: WorldPoint[] = [];
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const [sampleIndex, samplePoint] of centerlinePoints.entries()) {
    const previousPoint = centerlinePoints.at(
      (sampleIndex - 1 + centerlinePoints.length) % centerlinePoints.length,
    )!;
    const nextPoint =
      centerlinePoints[(sampleIndex + 1) % centerlinePoints.length]!;
    const tangentDeltaX = nextPoint.x - previousPoint.x;
    const tangentDeltaY = nextPoint.y - previousPoint.y;
    const tangentLength = Math.hypot(tangentDeltaX, tangentDeltaY) || 1;
    const normalX = -tangentDeltaY / tangentLength;
    const normalY = tangentDeltaX / tangentLength;
    const halfWidth = samplePoint.width / 2;
    const leftPoint = {
      x: samplePoint.x + normalX * halfWidth,
      y: samplePoint.y + normalY * halfWidth,
    };
    const rightPoint = {
      x: samplePoint.x - normalX * halfWidth,
      y: samplePoint.y - normalY * halfWidth,
    };

    leftBoundaryPoints.push(leftPoint);
    rightBoundaryPoints.push(rightPoint);

    minX = Math.min(minX, leftPoint.x, rightPoint.x);
    maxX = Math.max(maxX, leftPoint.x, rightPoint.x);
    minY = Math.min(minY, leftPoint.y, rightPoint.y);
    maxY = Math.max(maxY, leftPoint.y, rightPoint.y);
  }

  const resolvedGeometry = {
    centerlinePoints,
    leftBoundaryPoints,
    rightBoundaryPoints,
    minX,
    maxX,
    minY,
    maxY,
  };
  trackRenderGeometryCache.set(spec, resolvedGeometry);
  return resolvedGeometry;
}

/**
 * Traces the filled ribbon path bounded by the left and right track edges.
 *
 * @param ctx - 2D rendering context.
 * @param trackRenderGeometry - Cached world-space track geometry.
 * @param transform - World-to-canvas affine transform.
 */
function traceTrackRibbonPath(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void {
  const [firstLeftPoint] = trackRenderGeometry.leftBoundaryPoints;
  if (firstLeftPoint === undefined) {
    return;
  }

  const firstCanvasPoint = toCanvas(
    firstLeftPoint.x,
    firstLeftPoint.y,
    transform,
  );
  ctx.beginPath();
  ctx.moveTo(firstCanvasPoint.x, firstCanvasPoint.y);

  for (const boundaryPoint of trackRenderGeometry.leftBoundaryPoints.slice(1)) {
    const canvasPoint = toCanvas(boundaryPoint.x, boundaryPoint.y, transform);
    ctx.lineTo(canvasPoint.x, canvasPoint.y);
  }

  for (const boundaryPoint of trackRenderGeometry.rightBoundaryPoints.toReversed()) {
    const canvasPoint = toCanvas(boundaryPoint.x, boundaryPoint.y, transform);
    ctx.lineTo(canvasPoint.x, canvasPoint.y);
  }

  ctx.closePath();
}

/**
 * Traces one closed polyline derived from world-space points.
 *
 * @param ctx - 2D rendering context.
 * @param worldPoints - Ordered world-space points.
 * @param transform - World-to-canvas affine transform.
 */
function traceClosedWorldPath(
  ctx: CanvasRenderingContext2D,
  worldPoints: readonly WorldPoint[],
  transform: WorldTransform,
): void {
  const [firstPoint] = worldPoints;
  if (firstPoint === undefined) {
    return;
  }

  const firstCanvasPoint = toCanvas(firstPoint.x, firstPoint.y, transform);
  ctx.beginPath();
  ctx.moveTo(firstCanvasPoint.x, firstCanvasPoint.y);

  for (const worldPoint of worldPoints.slice(1)) {
    const canvasPoint = toCanvas(worldPoint.x, worldPoint.y, transform);
    ctx.lineTo(canvasPoint.x, canvasPoint.y);
  }

  ctx.closePath();
}

/**
 * Traces the sampled centerline path.
 *
 * @param ctx - 2D rendering context.
 * @param centerlinePoints - Ordered sampled centerline points.
 * @param transform - World-to-canvas affine transform.
 */
function traceClosedSamplePath(
  ctx: CanvasRenderingContext2D,
  centerlinePoints: readonly TrackSamplePoint[],
  transform: WorldTransform,
): void {
  const [firstPoint] = centerlinePoints;
  if (firstPoint === undefined) {
    return;
  }

  const firstCanvasPoint = toCanvas(firstPoint.x, firstPoint.y, transform);
  ctx.beginPath();
  ctx.moveTo(firstCanvasPoint.x, firstCanvasPoint.y);

  for (const samplePoint of centerlinePoints.slice(1)) {
    const canvasPoint = toCanvas(samplePoint.x, samplePoint.y, transform);
    ctx.lineTo(canvasPoint.x, canvasPoint.y);
  }

  ctx.closePath();
}

/**
 * Draws the fading tire-mark trail behind the car.
 *
 * Each mark fades from `COLOR_TIRE_MARK_MAX_ALPHA` to fully transparent as
 * its age increases toward `TIRE_MARK_MAX_AGE_TICKS`.
 *
 * @param ctx - 2D rendering context.
 * @param marks - Tire mark list from the render state.
 * @param transform - World-to-canvas transform.
 */
function drawTireMarks(
  ctx: CanvasRenderingContext2D,
  marks: readonly TireMark[],
  transform: WorldTransform,
): void {
  for (const mark of marks) {
    const remainingRatio = 1 - mark.age / TIRE_MARK_MAX_AGE_TICKS;
    const alpha = remainingRatio * COLOR_TIRE_MARK_MAX_ALPHA;
    if (alpha <= 0) continue;

    const canvasPos = toCanvas(mark.worldX, mark.worldY, transform);
    ctx.beginPath();
    ctx.arc(canvasPos.x, canvasPos.y, 1.5, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(0,185,225,${alpha.toFixed(3)})`;
    ctx.fill();
  }
}

/**
 * Draws the car as a square outline with a yellow heading indicator line
 * extending from the front face.
 *
 * The car rectangle is drawn in local space (car centre at origin, facing
 * positive local X), then rotated and translated to world position via the
 * canvas transform stack.
 *
 * @param ctx - 2D rendering context.
 * @param state - Current physics state (position + heading).
 * @param transform - World-to-canvas affine transform.
 */
function drawCar(
  ctx: CanvasRenderingContext2D,
  state: EnvironmentState,
  transform: WorldTransform,
  tireState: TireStateTuple,
): void {
  const canvasPos = toCanvas(state.carX, state.carY, transform);
  const halfLengthCanvas = CAR_HALF_LENGTH_WORLD * transform.scale;
  const halfWidthCanvas = CAR_HALF_WIDTH_WORLD * transform.scale;
  const headingLineLengthCanvas =
    HEADING_INDICATOR_WORLD_LENGTH * transform.scale;

  ctx.save();
  ctx.translate(canvasPos.x, canvasPos.y);
  ctx.rotate(state.carHeading);

  // Glow behind car body.
  ctx.shadowColor = COLOR_CAR_BODY;
  ctx.shadowBlur = 10;

  // Car body — cyan square outline.
  ctx.beginPath();
  ctx.rect(
    -halfLengthCanvas,
    -halfWidthCanvas,
    halfLengthCanvas * 2,
    halfWidthCanvas * 2,
  );
  ctx.strokeStyle = COLOR_CAR_BODY;
  ctx.lineWidth = 1.8;
  ctx.stroke();
  drawCarTireCorners(ctx, halfLengthCanvas, halfWidthCanvas, tireState);

  // Heading indicator — yellow line extending from the front face.
  ctx.beginPath();
  ctx.moveTo(halfLengthCanvas, 0);
  ctx.lineTo(halfLengthCanvas + headingLineLengthCanvas, 0);
  ctx.strokeStyle = COLOR_HEADING_INDICATOR;
  ctx.lineWidth = 1.8;
  ctx.stroke();

  ctx.restore();
}

/**
 * Converts a world-coordinate point to canvas pixels using the affine transform.
 *
 * @param worldX - World X coordinate.
 * @param worldY - World Y coordinate.
 * @param transform - Affine transform to apply.
 * @returns Canvas pixel coordinates.
 */
/**
 * Draws the Tier 4 pit entrance and stall overlays.
 *
 * `pitStatus` uses the packed tuple `[teamA_car, teamA_ticks, teamB_car,
 * teamB_ticks]`. A positive tick count marks that team's pit as occupied and
 * causes both the stall and entrance corridor AABB to render with the occupied
 * fill overlay.
 *
 * @param ctx - 2D rendering context.
 * @param spec - Frozen track geometry.
 * @param transform - World-to-canvas affine transform.
 * @param pitStatus - Optional packed pit-status tuple.
 */
function drawPitOverlays(
  ctx: CanvasRenderingContext2D,
  spec: TrackSpec,
  transform: WorldTransform,
  pitStatus?: Uint8Array | Uint16Array | Int16Array,
): void {
  for (const pitBox of spec.pitBoxes ?? []) {
    const teamColor = pitBox.teamIndex === 0 ? COLOR_PIT_TEAM_A : COLOR_PIT_TEAM_B;
    const occupiedTicks = pitStatus?.[pitBox.teamIndex * 2 + 1] ?? 0;
    const pitBoxCenter = pitBox.boxCenter ?? {
      x: pitBox.entranceCorridor.x + pitBox.entranceCorridor.width / 2,
      y: pitBox.entranceCorridor.y + pitBox.entranceCorridor.height / 2,
    };
    const renderedPitBox = pitBox.pitBox ?? {
        x: pitBoxCenter.x - 9,
        y: pitBoxCenter.y - 6,
        width: 18,
        height: 12,
      };

    drawAxisAlignedOverlay(ctx, renderedPitBox, transform, teamColor, occupiedTicks > 0);
    drawAxisAlignedOverlay(
      ctx,
      pitBox.entranceCorridor,
      transform,
      teamColor,
      occupiedTicks > 0,
      [8, 6],
    );
  }
}

/**
 * Draws one axis-aligned overlay rectangle in world space.
 *
 * @param ctx - 2D rendering context.
 * @param worldBox - World-space rectangle.
 * @param transform - World-to-canvas affine transform.
 * @param strokeColor - Outline/fill color.
 * @param occupied - Whether the box is currently occupied.
 * @param dashPattern - Optional dashed outline pattern.
 */
function drawAxisAlignedOverlay(
  ctx: CanvasRenderingContext2D,
  worldBox: { readonly x: number; readonly y: number; readonly width: number; readonly height: number },
  transform: WorldTransform,
  strokeColor: string,
  occupied: boolean,
  dashPattern: readonly number[] = [],
): void {
  const topLeft = toCanvas(worldBox.x, worldBox.y, transform);
  const canvasWidth = worldBox.width * transform.scale;
  const canvasHeight = worldBox.height * transform.scale;

  ctx.save();
  ctx.setLineDash([...dashPattern]);
  ctx.lineWidth = 1.4;
  ctx.fillStyle = occupied ? COLOR_PIT_OCCUPIED : 'transparent';
  ctx.strokeStyle = strokeColor;
  ctx.beginPath();
  ctx.rect(topLeft.x, topLeft.y, canvasWidth, canvasHeight);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

/**
 * Draws live tire-health corner markers on the car body.
 *
 * @param ctx - 2D rendering context.
 * @param halfLengthCanvas - Half car length in canvas pixels.
 * @param halfWidthCanvas - Half car width in canvas pixels.
 * @param tireState - Ordered tire-health tuple `[FL, FR, RL, RR]`.
 */
function drawCarTireCorners(
  ctx: CanvasRenderingContext2D,
  halfLengthCanvas: number,
  halfWidthCanvas: number,
  tireState: TireStateTuple,
): void {
  const tireCornerRadius = 1.3;
  const tireCornerOffsets = [
    { x: halfLengthCanvas, y: -halfWidthCanvas, tireHealth: tireState[0] },
    { x: halfLengthCanvas, y: halfWidthCanvas, tireHealth: tireState[1] },
    { x: -halfLengthCanvas, y: -halfWidthCanvas, tireHealth: tireState[2] },
    { x: -halfLengthCanvas, y: halfWidthCanvas, tireHealth: tireState[3] },
  ];

  for (const tireCorner of tireCornerOffsets) {
    ctx.beginPath();
    ctx.arc(tireCorner.x, tireCorner.y, tireCornerRadius, 0, Math.PI * 2);
    ctx.fillStyle = resolveTireHealthColor(tireCorner.tireHealth);
    ctx.fill();
  }
}

/**
 * Resolves the rendered tire-health tuple for the focused car.
 *
 * @param envState - Current environment snapshot.
 * @param packedTireState - Optional packed worker tire array.
 * @param focusCarIndex - Focused car row inside the packed worker frame.
 * @returns Ordered tire-health tuple for renderer use.
 */
function resolveRenderTireState(
  envState: EnvironmentState,
  packedTireState: Float32Array | undefined,
  focusCarIndex: number,
): TireStateTuple {
  const packedOffset = focusCarIndex * 4;

  if (packedTireState !== undefined && packedTireState.length >= packedOffset + 4) {
    return [
      packedTireState[packedOffset],
      packedTireState[packedOffset + 1],
      packedTireState[packedOffset + 2],
      packedTireState[packedOffset + 3],
    ];
  }

  return envState.tireState ?? [1, 1, 1, 1];
}

/**
 * Resolves the live tire marker color from tire health.
 *
 * @param tireHealth - Normalized tire-health value.
 * @returns CSS color string for the tire marker.
 */
function resolveTireHealthColor(tireHealth: number): string {
  if (tireHealth >= 0.75) {
    return COLOR_TIRE_GOOD;
  }

  if (tireHealth >= 0.5) {
    return COLOR_TIRE_WARN;
  }

  if (tireHealth >= 0.25) {
    return COLOR_TIRE_ALERT;
  }

  return COLOR_TIRE_CRITICAL;
}

function toCanvas(
  worldX: number,
  worldY: number,
  transform: WorldTransform,
): CanvasPoint {
  return {
    x: worldX * transform.scale + transform.offsetX,
    y: worldY * transform.scale + transform.offsetY,
  };
}
