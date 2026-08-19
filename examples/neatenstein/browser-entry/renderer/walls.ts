/**
 * Neon wall column renderer for the Neatenstein CPU-tier raycaster.
 *
 * This module implements the CPU ImageData path for wall columns. Columns are
 * written directly into a shared `Uint8ClampedArray` framebuffer, avoiding
 * per-column `fillRect`, `globalAlpha` mutations, and other canvas state churn.
 *
 * For optimal performance, callers should use {@link writeNeonWallColumn} for
 * each wall stripe and then flush the finished framebuffer once with
 * `ctx.putImageData(...)`.
 *
 * @module
 */

import {
  NEATENSTEIN_BACKGROUND_RGB,
  resolveNeatensteinFogFactor,
} from './framebuffer';
import { RGBA_CHANNELS, RGBA_OPAQUE_ALPHA } from './renderer.wall.constants';
import type { ParsedRgb } from './renderer.wall.types';

// Re-export constants and types for external consumers.
export { RGBA_CHANNELS, RGBA_OPAQUE_ALPHA } from './renderer.wall.constants';
export type {
  NeatensteinWallRenderContext,
  ParsedRgb,
} from './renderer.wall.types';

/**
 * Circuit trace constants for procedural wall surface decoration.
 *
 * The lower 42% of each wall column is decorated with a deterministic circuit
 * board pattern: horizontal and vertical traces with pad nodes at intersections,
 * rendered in a dark neon blue that complements the existing wall palette.
 *
 * The pattern uses 4 lanes with ~55% existence probability each (~2.2 lanes
 * average) for a sparse motherboard look, 45° diagonal connectors at ~10%
 * density, and circular pads (radius 8) at intersections, trace ends, and
 * tile edges. Traces are 6px wide (half-width 3) for clear visibility.
 */
const CIRCUIT_WALL_VIRTUAL_WIDTH = 256;
const CIRCUIT_WALL_VIRTUAL_HEIGHT = 256;
const CIRCUIT_LANE_COUNT = 4;
const CIRCUIT_LOWER_START = Math.round(CIRCUIT_WALL_VIRTUAL_HEIGHT * 0.58);
const CIRCUIT_LANE_SPACING = Math.round(
  (CIRCUIT_WALL_VIRTUAL_HEIGHT * 0.42) / CIRCUIT_LANE_COUNT,
);
const CIRCUIT_CONNECTOR_SPACING = 48;
const CIRCUIT_TRACE_HALF_WIDTH = 3;
const CIRCUIT_PAD_RADIUS = 8;
const CIRCUIT_PAD_RADIUS_SQ = CIRCUIT_PAD_RADIUS * CIRCUIT_PAD_RADIUS;
const CIRCUIT_LANE_EXIST_THRESH = 140;
const CIRCUIT_CONNECTOR_THRESH = 38;
const CIRCUIT_DIAGONAL_THRESH = 26;
const CIRCUIT_TRACE_RGB: ParsedRgb = { r: 0, g: 58, b: 92 };
const CIRCUIT_PAD_RGB: ParsedRgb = { r: 0, g: 102, b: 153 };

/**
 * Precomputed per-column circuit trace data.
 *
 * All hash calls and lane existence checks are hoisted out of the per-pixel
 * loop into this structure, computed once per column by
 * {@link precomputeCircuitColumn}. The per-pixel resolver
 * {@link resolveCircuitTraceTypeFast} then performs only simple arithmetic
 * comparisons against the precomputed values.
 *
 * @property wallX - Virtual wall X coordinate for this column.
 * @property existingLaneYs - Y centers of all existing horizontal lanes.
 * @property hasVertical - Whether this column has a vertical connector.
 * @property vertOnConnX - Whether `wallX` is on the vertical connector center.
 * @property vertTopY - Top Y of the vertical connector span.
 * @property vertBotY - Bottom Y of the vertical connector span.
 * @property hasDiagonal - Whether this column has a 45° diagonal connector.
 * @property diagSlope - Diagonal slope: `1` for `/`, `-1` for `\`.
 * @property diagTopY - Top Y of the diagonal connector span.
 * @property diagBotY - Bottom Y of the diagonal connector span.
 * @property diagStartX - Starting X of the diagonal at `diagTopY`.
 * @property padAnchors - Precomputed pad anchors within X range of this column,
 *   each as `{ dxSq, y }` where `dxSq` is the squared horizontal distance from
 *   `wallX` to the pad center.
 */
interface CircuitColumnData {
  wallX: number;
  existingLaneYs: number[];
  hasVertical: boolean;
  vertOnConnX: boolean;
  vertTopY: number;
  vertBotY: number;
  hasDiagonal: boolean;
  diagSlope: number;
  diagTopY: number;
  diagBotY: number;
  diagStartX: number;
  padAnchors: { dxSq: number; y: number }[];
}

/**
 * Cache of parsed wall colors.
 *
 * Wall colors are reused across many columns and frames, so caching avoids
 * repeated string parsing in the hot path.
 */
const WALL_COLOR_CACHE = new Map<string, ParsedRgb>();

/**
 * Return whether a number is a positive integer dimension.
 *
 * @param value - Candidate dimension.
 * @returns Whether the value is usable as a framebuffer dimension.
 */
function isPositiveIntegerDimension(value: number): boolean {
  return Number.isInteger(value) && value > 0;
}

/**
 * Parse a strict `#rrggbb` hex color string into RGB channels.
 *
 * @param hex - Color string in `#rrggbb` format.
 * @returns Parsed RGB triplet.
 * @throws {Error} When the string is not a valid `#rrggbb` color.
 */
function parseHexColor(hex: string): ParsedRgb {
  const cached = WALL_COLOR_CACHE.get(hex);
  if (cached !== undefined) {
    return cached;
  }

  const match = /^#([0-9a-fA-F]{6})$/.exec(hex);
  if (match === null) {
    throw new Error(`Expected #rrggbb hex color, got "${hex}"`);
  }

  const digits = match[1];
  const rgb = {
    r: Number.parseInt(digits.slice(0, 2), 16),
    g: Number.parseInt(digits.slice(2, 4), 16),
    b: Number.parseInt(digits.slice(4, 6), 16),
  };

  WALL_COLOR_CACHE.set(hex, rgb);
  return rgb;
}

/**
 * Resolve the wall fog factor for a perpendicular wall distance.
 *
 * Delegates to the shared smoothstep fog function so walls, floor, ceiling,
 * and sprites all use the same distance-based fog curve.
 *
 * @param perpWallDist - Perpendicular wall distance.
 * @returns Fog interpolation factor where `0` is near and `1` is fully fogged.
 */
function resolveWallFogFactor(perpWallDist: number): number {
  return resolveNeatensteinFogFactor(perpWallDist);
}

/**
 * Blend a base wall color toward the background color using distance fog.
 *
 * @param base - Base neon wall color.
 * @param fogT - Fog interpolation factor in `[0, 1]`.
 * @returns Fogged RGB color.
 */
function resolveFoggedWallColor(base: ParsedRgb, fogT: number): ParsedRgb {
  const { r: bgR, g: bgG, b: bgB } = NEATENSTEIN_BACKGROUND_RGB;
  const invFog = 1 - fogT;

  return {
    r: Math.round(base.r * invFog + bgR * fogT),
    g: Math.round(base.g * invFog + bgG * fogT),
    b: Math.round(base.b * invFog + bgB * fogT),
  };
}

/**
 * Fast integer hash for deterministic circuit trace generation.
 *
 * @param a - First hash input (wall seed).
 * @param b - Second hash input (cell X or derived value).
 * @param c - Third hash input (cell Y or derived value).
 * @returns Unsigned 32-bit hash value.
 */
function circuitHash(a: number, b: number, c: number): number {
  let h = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791);
  h = (h ^ (h >>> 13)) * 1274126177;
  h = h ^ (h >>> 16);
  return h >>> 0;
}

/**
 * Precompute all circuit trace data for a single wall column.
 *
 * This function runs ONCE per column, hoisting all hash calls and lane
 * existence checks out of the per-pixel loop. The returned
 * {@link CircuitColumnData} is then passed to
 * {@link resolveCircuitTraceTypeFast} for each pixel, which performs only
 * simple arithmetic comparisons.
 *
 * The precomputation covers:
 *
 * - **Lane existence:** 4 lanes, each with ~55% existence probability
 *   (threshold {@link CIRCUIT_LANE_EXIST_THRESH} = 140/255), producing ~2.2
 *   lanes on average for a sparse motherboard look.
 * - **Vertical connectors:** At fixed X intervals
 *   ({@link CIRCUIT_CONNECTOR_SPACING} = 48px), ~15% of connector positions
 *   have a vertical trace linking the first adjacent existing lane pair.
 * - **45° diagonal connectors:** When no vertical connector is present, ~10%
 *   of positions get a 45° diagonal connector (both `/` and `\` slopes),
 *   confined to a single tile width.
 * - **Pad anchors:** Circular pads (radius {@link CIRCUIT_PAD_RADIUS} = 8px)
 *   are precomputed for all pad positions within X range of this column:
 *   intersection pads, diagonal-end pads, connector-end pads, and tile-edge
 *   pads. Each anchor stores the squared horizontal distance (`dxSq`) and
 *   the pad Y center so the per-pixel check is a single `dxSq + dy*dy`
 *   comparison.
 *
 * @param wallSeed - Deterministic seed for this wall tile.
 * @param wallX - Horizontal position on the virtual wall texture.
 * @returns Precomputed circuit data for this column.
 */
function precomputeCircuitColumn(
  wallSeed: number,
  wallX: number,
): CircuitColumnData {
  // Lane existence (hash once per lane, not per pixel).
  // A minimum of 2 lanes is enforced so every wall cell has visible circuitry.
  const laneExists: boolean[] = new Array(CIRCUIT_LANE_COUNT);
  const existingLaneYs: number[] = [];
  for (let lane = 0; lane < CIRCUIT_LANE_COUNT; lane += 1) {
    const ly = CIRCUIT_LOWER_START + CIRCUIT_LANE_SPACING * (lane + 0.5);
    const laneHash = circuitHash(wallSeed, 999, lane * 101);
    const exists = (laneHash & 0xff) < CIRCUIT_LANE_EXIST_THRESH;
    laneExists[lane] = exists;
    if (exists) {
      existingLaneYs.push(ly);
    }
  }
  // Enforce minimum 2 lanes: force on lanes with strongest hash values.
  if (existingLaneYs.length < 2) {
    const candidates: { lane: number; hash: number }[] = [];
    for (let lane = 0; lane < CIRCUIT_LANE_COUNT; lane += 1) {
      if (!laneExists[lane]) {
        candidates.push({
          lane,
          hash: circuitHash(wallSeed, 999, lane * 101) & 0xff,
        });
      }
    }
    candidates.sort((a, b) => a.hash - b.hash);
    const needed = 2 - existingLaneYs.length;
    for (let i = 0; i < needed && i < candidates.length; i += 1) {
      const lane = candidates[i].lane;
      laneExists[lane] = true;
      existingLaneYs.push(
        CIRCUIT_LOWER_START + CIRCUIT_LANE_SPACING * (lane + 0.5),
      );
    }
  }

  // Connector slot
  const connectorIdx = Math.floor(wallX / CIRCUIT_CONNECTOR_SPACING);
  const inConnectorX = wallX - connectorIdx * CIRCUIT_CONNECTOR_SPACING;
  const connectorCenterX = CIRCUIT_CONNECTOR_SPACING / 2;
  const connectorAbsX =
    connectorIdx * CIRCUIT_CONNECTOR_SPACING + connectorCenterX;

  const connHash = circuitHash(wallSeed, connectorIdx * 47, 777);
  const hasVertical = (connHash & 0xff) < CIRCUIT_CONNECTOR_THRESH;
  const hasDiagonal =
    !hasVertical && ((connHash >>> 8) & 0xff) < CIRCUIT_DIAGONAL_THRESH;
  const diagSlope = ((connHash >>> 16) & 1) === 0 ? 1 : -1;

  // Find first adjacent existing lane pair for connector
  let connectorLaneTop = -1;
  let connectorLaneBot = -1;
  for (let lane = 0; lane < CIRCUIT_LANE_COUNT - 1; lane += 1) {
    if (laneExists[lane] && laneExists[lane + 1]) {
      connectorLaneTop = lane;
      connectorLaneBot = lane + 1;
      break;
    }
  }

  const vertTopY =
    connectorLaneTop >= 0
      ? CIRCUIT_LOWER_START + CIRCUIT_LANE_SPACING * (connectorLaneTop + 0.5)
      : 0;
  const vertBotY =
    connectorLaneBot >= 0
      ? CIRCUIT_LOWER_START + CIRCUIT_LANE_SPACING * (connectorLaneBot + 0.5)
      : 0;
  const vertOnConnX =
    Math.abs(inConnectorX - connectorCenterX) <= CIRCUIT_TRACE_HALF_WIDTH;

  // Pad anchors: precompute (dxSq, padY) for all pads within X range of this column
  const padAnchors: { dxSq: number; y: number }[] = [];
  const addPad = (padAbsX: number, padY: number): void => {
    const dx = wallX - padAbsX;
    const dxSq = dx * dx;
    if (dxSq <= CIRCUIT_PAD_RADIUS_SQ) {
      padAnchors.push({ dxSq, y: padY });
    }
  };

  // (a) Intersection pads: connector meets existing lane
  if ((hasVertical || hasDiagonal) && connectorLaneTop >= 0) {
    addPad(connectorAbsX, vertTopY);
    addPad(connectorAbsX, vertBotY);
  }

  // (b) Diagonal end pad (bottom lane at offset X)
  if (hasDiagonal && connectorLaneTop >= 0) {
    const diagEndX = connectorAbsX + diagSlope * CIRCUIT_LANE_SPACING;
    addPad(diagEndX, vertBotY);
  }

  // (c) Connector-end pads: where connector meets a lane whose neighbor doesn't exist
  if (hasVertical) {
    for (let lane = 0; lane < CIRCUIT_LANE_COUNT - 1; lane += 1) {
      const yThis = CIRCUIT_LOWER_START + CIRCUIT_LANE_SPACING * (lane + 0.5);
      const yNext = CIRCUIT_LOWER_START + CIRCUIT_LANE_SPACING * (lane + 1.5);
      if (laneExists[lane] && !laneExists[lane + 1]) {
        addPad(connectorAbsX, yThis);
      }
      if (!laneExists[lane] && laneExists[lane + 1]) {
        addPad(connectorAbsX, yNext);
      }
    }
  }

  // (d) Tile-edge pads for existing lanes
  for (let lane = 0; lane < CIRCUIT_LANE_COUNT; lane += 1) {
    if (!laneExists[lane]) continue;
    const ly = CIRCUIT_LOWER_START + CIRCUIT_LANE_SPACING * (lane + 0.5);
    addPad(0, ly);
    addPad(CIRCUIT_WALL_VIRTUAL_WIDTH, ly);
  }

  return {
    wallX,
    existingLaneYs,
    hasVertical: hasVertical && connectorLaneTop >= 0,
    vertOnConnX,
    vertTopY,
    vertBotY,
    hasDiagonal: hasDiagonal && connectorLaneTop >= 0,
    diagSlope,
    diagTopY: vertTopY,
    diagBotY: vertBotY,
    diagStartX: connectorAbsX,
    padAnchors,
  };
}

/**
 * Resolve the circuit trace type for a wall pixel using precomputed column data.
 *
 * This is the per-pixel resolver that runs inside the interior row loop. It
 * performs only simple arithmetic comparisons against the precomputed
 * {@link CircuitColumnData} — no hash calls, no lane existence checks.
 *
 * Features resolved:
 *
 * - **Horizontal lanes:** Scans ALL existing lanes (not just the first match)
 *   so every lane is drawn. Each lane is 6px wide
 *   ({@link CIRCUIT_TRACE_HALF_WIDTH} = 3).
 * - **Vertical connectors:** Checked against the precomputed connector span.
 * - **45° diagonal connectors:** Checked against the precomputed diagonal line
 *   with a `Math.SQRT2` width factor so the perpendicular width matches
 *   horizontal traces.
 * - **Circular pads:** Checked INDEPENDENTLY of trace membership (fixes the
 *   square-pad bug where pads were gated behind `onHorizontal && onVertical`).
 *   Pads are radius 8 ({@link CIRCUIT_PAD_RADIUS}) and rendered as true
 *   circles using `dxSq + dy*dy <= CIRCUIT_PAD_RADIUS_SQ`.
 *
 * @param virtualY - Stable vertical position on the virtual wall surface.
 * @param data - Precomputed column data from {@link precomputeCircuitColumn}.
 * @returns `'trace'` for a trace pixel, `'pad'` for a pad pixel, or `null` if
 *   the pixel is not on any circuit feature.
 */
function resolveCircuitTraceTypeFast(
  virtualY: number,
  data: CircuitColumnData,
): 'trace' | 'pad' | null {
  if (virtualY < CIRCUIT_LOWER_START) {
    return null;
  }

  // Horizontal lanes: scan ALL existing lanes (fixes break-on-first-lane bug)
  let onHorizontal = false;
  for (const laneY of data.existingLaneYs) {
    if (Math.abs(virtualY - laneY) <= CIRCUIT_TRACE_HALF_WIDTH) {
      onHorizontal = true;
      break;
    }
  }

  // Vertical connector
  let onVertical = false;
  if (data.hasVertical && data.vertOnConnX) {
    onVertical =
      virtualY >= data.vertTopY - CIRCUIT_TRACE_HALF_WIDTH &&
      virtualY <= data.vertBotY + CIRCUIT_TRACE_HALF_WIDTH;
  }

  // Diagonal connector (45°)
  let onDiagonal = false;
  if (data.hasDiagonal) {
    if (
      virtualY >= data.diagTopY - CIRCUIT_TRACE_HALF_WIDTH &&
      virtualY <= data.diagBotY + CIRCUIT_TRACE_HALF_WIDTH
    ) {
      const diagLineX =
        data.diagSlope > 0
          ? data.diagStartX + (virtualY - data.diagTopY)
          : data.diagStartX - (virtualY - data.diagTopY);
      onDiagonal =
        Math.abs(data.wallX - diagLineX) <=
        CIRCUIT_TRACE_HALF_WIDTH * Math.SQRT2;
    }
  }

  // Pads: INDEPENDENT of trace membership (fixes square-pad bug)
  for (const pad of data.padAnchors) {
    const dy = virtualY - pad.y;
    if (pad.dxSq + dy * dy <= CIRCUIT_PAD_RADIUS_SQ) {
      return 'pad';
    }
  }

  if (onHorizontal || onVertical || onDiagonal) {
    return 'trace';
  }
  return null;
}

/**
 * Compute the wall texture coordinate for a DDA wall hit.
 *
 * Returns the fractional part of the wall intersection point along the wall
 * surface, in `[0, 1)`. For an X-side hit (`side === 0`) the wall coordinate is
 * `posY + perpWallDist * dirY`; for a Y-side hit it is
 * `posX + perpWallDist * dirX`. This function computes the X-side variant
 * directly; callers pass the appropriate fixed coordinate and ray-perpendicular
 * component.
 *
 * @param fixedCoord - Camera coordinate perpendicular to the ray direction on
 *   the hit side (posY for X-side, posX for Y-side).
 * @param perpWallDist - Perpendicular wall distance from the DDA hit.
 * @param rayPerp - Ray direction component perpendicular to the hit side
 *   (dirY for X-side, dirX for Y-side).
 * @returns Texture coordinate in `[0, 1)` via `fract(wallX)`.
 */
export function computeWallTexcoord(
  fixedCoord: number,
  perpWallDist: number,
  rayPerp: number,
): number {
  const wallX = fixedCoord + perpWallDist * rayPerp;
  return wallX - Math.floor(wallX);
}

/**
 * Write a single neon wall column into the CPU ImageData framebuffer.
 *
 * This function performs no canvas flush. It is the preferred hot-path helper
 * for renderers that draw many columns and call `putImageData` once after the
 * framebuffer is complete.
 *
 * Distance fog linearly interpolates the wall color toward
 * {@link NEATENSTEIN_BACKGROUND_RGB} as `perpWallDist` approaches
 * {@link NEATENSTEIN_RENDER_DISTANCE_CAP}. Non-finite distances are treated as
 * fully fogged.
 *
 * When a `texcoord` in `[0, 1)` is provided (computed via
 * {@link computeWallTexcoord}), the column is shaded with a subtle vertical
 * stripe pattern derived from the texcoord. This breaks up the flat colour
 * without altering the neon hue, preserving the Invariant §3 aesthetic. When
 * no texcoord is supplied (or the value is outside `[0, 1)`), the column is
 * flat-shaded as before.
 *
 * Wall top and bottom edges are subpixel anti-aliased: `drawStart` and
 * `drawEnd` are kept as floats and the fractional coverage of each edge row is
 * alpha-composited over the existing framebuffer pixel so upscaling does not
 * produce serrated integer boundaries.
 *
 * When a `wallSeed` is provided alongside a valid `texcoord`, the lower 42% of
 * the wall column is decorated with a procedural circuit board pattern:
 * 4 horizontal lanes (~55% existence each), vertical and 45° diagonal
 * connectors, and circular pad nodes (radius 8) at intersections, trace ends,
 * and tile edges — rendered in a dark neon blue. The pattern is deterministic
 * per wall tile so each map cell produces a stable, recognizable circuit
 * layout. Circuit traces are not applied to edge-blend rows (subpixel AA rows)
 * or when no texcoord is supplied. All hash calls are precomputed once per
 * column for performance.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param framebufferWidth - Framebuffer width in pixels.
 * @param framebufferHeight - Framebuffer height in pixels.
 * @param column - Horizontal column index to write.
 * @param drawStart - Top row of the wall stripe, inclusive.
 * @param drawEnd - Bottom row of the wall stripe, exclusive.
 * @param hexColor - Wall color as `#rrggbb`.
 * @param perpWallDist - Perpendicular wall distance.
 * @param texcoord - Optional wall texture coordinate in `[0, 1)` from
 *   {@link computeWallTexcoord}. Drives vertical stripe shading and circuit
 *   trace generation.
 * @param wallSeed - Deterministic seed for the wall tile, derived from map
 *   grid coordinates. Used to generate procedural circuit traces in the lower
 *   42% of the wall surface. Defaults to `0`.
 * @throws {Error} When `hexColor` is not a valid `#rrggbb` string.
 *
 * @example
 * ```ts
 * writeNeonWallColumn(framebuffer, 640, 480, 12, 120, 340, '#00bfff', 4.5, 0.37, 12345);
 * ```
 */
export function writeNeonWallColumn(
  framebuffer: Uint8ClampedArray,
  framebufferWidth: number,
  framebufferHeight: number,
  column: number,
  drawStart: number,
  drawEnd: number,
  hexColor: string,
  perpWallDist: number,
  texcoord: number = -1,
  wallSeed: number = 0,
): void {
  if (
    !isPositiveIntegerDimension(framebufferWidth) ||
    !isPositiveIntegerDimension(framebufferHeight)
  ) {
    return;
  }

  // Do not clamp invalid columns to the edge; skip them instead.
  if (!Number.isFinite(column)) {
    return;
  }

  const x = Math.trunc(column);
  if (x < 0 || x >= framebufferWidth) {
    return;
  }

  const baseColor = parseHexColor(hexColor);
  const fogT = resolveWallFogFactor(perpWallDist);
  const foggedColor = resolveFoggedWallColor(baseColor, fogT);

  // Subtle texcoord-driven sinusoidal striping at 50% subtler amplitude than
  // the original 0.1. The narrower 0.05 range keeps the neon hue intact while
  // breaking up flat columns without "indented border" artifacts (Invariant §3).
  const hasTexcoord = texcoord >= 0 && texcoord < 1;
  const brightness = hasTexcoord
    ? 1.0 - 0.05 * (1.0 - Math.cos(texcoord * Math.PI * 4))
    : 1.0;

  const finalColor = hasTexcoord
    ? {
        r: Math.round(foggedColor.r * brightness),
        g: Math.round(foggedColor.g * brightness),
        b: Math.round(foggedColor.b * brightness),
      }
    : foggedColor;

  // Subpixel anti-aliasing: keep drawStart/drawEnd as floats and compute
  // fractional coverage at the top and bottom edge rows so upscaling does not
  // produce serrated integer boundaries.
  if (!Number.isFinite(drawStart) || !Number.isFinite(drawEnd)) {
    return;
  }
  if (drawStart >= drawEnd) {
    return;
  }
  // Wall is entirely outside the framebuffer vertically.
  if (drawEnd <= 0 || drawStart >= framebufferHeight) {
    return;
  }

  const startRow = Math.floor(drawStart);
  const endRow = Math.floor(drawEnd);

  // Fractional coverage of each edge row by the wall.
  let startCoverage = 1 - (drawStart - startRow);
  let endCoverage = drawEnd - endRow;

  // When the wall extends past a canvas bound, the clamped edge row is fully
  // covered by wall, not fractional.
  if (drawStart <= 0) {
    startCoverage = 1.0;
  }
  if (drawEnd >= framebufferHeight) {
    endCoverage = 1.0;
  }

  if (startRow === endRow) {
    // Wall is thinner than one pixel row: blend both edges in a single row.
    const coverage = Math.min(1, Math.max(0, drawEnd - drawStart));
    blendWallEdgeRow(
      framebuffer,
      framebufferWidth,
      framebufferHeight,
      x,
      startRow,
      coverage,
      finalColor,
    );
    return;
  }

  // Start edge row (partial coverage at the top).
  if (startCoverage > 0 && startRow >= 0 && startRow < framebufferHeight) {
    blendWallEdgeRow(
      framebuffer,
      framebufferWidth,
      framebufferHeight,
      x,
      startRow,
      startCoverage,
      finalColor,
    );
  }

  // Circuit trace precomputation for the lower 42% of the wall surface.
  // Traces are only applied when a texcoord is available; edge-blend rows are
  // excluded (handled above and below the interior loop).
  const wallHeight = drawEnd - drawStart;
  const circuitStartY =
    drawStart +
    (CIRCUIT_LOWER_START / CIRCUIT_WALL_VIRTUAL_HEIGHT) * wallHeight;
  const circuitWallX = hasTexcoord ? texcoord * CIRCUIT_WALL_VIRTUAL_WIDTH : 0;
  const circuitData = hasTexcoord
    ? precomputeCircuitColumn(wallSeed, circuitWallX)
    : null;
  const foggedTraceBase = resolveFoggedWallColor(CIRCUIT_TRACE_RGB, fogT);
  const foggedPadBase = resolveFoggedWallColor(CIRCUIT_PAD_RGB, fogT);
  const foggedTraceColor = hasTexcoord
    ? {
        r: Math.round(foggedTraceBase.r * brightness),
        g: Math.round(foggedTraceBase.g * brightness),
        b: Math.round(foggedTraceBase.b * brightness),
      }
    : foggedTraceBase;
  const foggedPadColor = hasTexcoord
    ? {
        r: Math.round(foggedPadBase.r * brightness),
        g: Math.round(foggedPadBase.g * brightness),
        b: Math.round(foggedPadBase.b * brightness),
      }
    : foggedPadBase;

  // Interior rows (fully covered by the wall).
  const interiorStart = Math.max(0, startRow + 1);
  const interiorEnd = Math.min(framebufferHeight, endRow);
  for (let row = interiorStart; row < interiorEnd; row += 1) {
    const offset = (row * framebufferWidth + x) * RGBA_CHANNELS;

    // Defensive guard for mismatched framebuffer dimensions.
    if (offset + 3 >= framebuffer.length) {
      break;
    }

    // Circuit traces: procedural circuit board pattern in the lower 42% of
    // the wall surface. Only applied when a texcoord is available.
    let pixelColor = finalColor;
    if (circuitData !== null && row >= circuitStartY) {
      const virtualY =
        ((row - drawStart) / wallHeight) * CIRCUIT_WALL_VIRTUAL_HEIGHT;
      const traceType = resolveCircuitTraceTypeFast(virtualY, circuitData);
      if (traceType === 'pad') {
        pixelColor = foggedPadColor;
      } else if (traceType === 'trace') {
        pixelColor = foggedTraceColor;
      }
    }

    framebuffer[offset] = pixelColor.r;
    framebuffer[offset + 1] = pixelColor.g;
    framebuffer[offset + 2] = pixelColor.b;
    framebuffer[offset + 3] = RGBA_OPAQUE_ALPHA;
  }

  // End edge row (partial coverage at the bottom).
  if (
    endCoverage > 0 &&
    endRow !== startRow &&
    endRow >= 0 &&
    endRow < framebufferHeight
  ) {
    blendWallEdgeRow(
      framebuffer,
      framebufferWidth,
      framebufferHeight,
      x,
      endRow,
      endCoverage,
      finalColor,
    );
  }
}

/**
 * Blend a wall edge row into the framebuffer with fractional coverage.
 *
 * Reads the existing framebuffer pixel and alpha-composites the wall color over
 * it using the supplied coverage factor, producing subpixel anti-aliasing at
 * wall boundaries. Rows outside the framebuffer or non-positive coverage are
 * skipped. A coverage of `1` (or greater) writes a fully opaque wall pixel
 * without reading back the framebuffer.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param framebufferWidth - Framebuffer width in pixels.
 * @param framebufferHeight - Framebuffer height in pixels.
 * @param x - Column index of the wall stripe.
 * @param row - Row index to blend.
 * @param coverage - Wall coverage of the row in `[0, 1]`.
 * @param wallColor - Fully-covered wall color to blend over the background.
 */
function blendWallEdgeRow(
  framebuffer: Uint8ClampedArray,
  framebufferWidth: number,
  framebufferHeight: number,
  x: number,
  row: number,
  coverage: number,
  wallColor: ParsedRgb,
): void {
  if (row < 0 || row >= framebufferHeight || coverage <= 0) {
    return;
  }

  const offset = (row * framebufferWidth + x) * RGBA_CHANNELS;

  // Defensive guard for mismatched framebuffer dimensions.
  if (offset + 3 >= framebuffer.length) {
    return;
  }

  if (coverage >= 1) {
    framebuffer[offset] = wallColor.r;
    framebuffer[offset + 1] = wallColor.g;
    framebuffer[offset + 2] = wallColor.b;
    framebuffer[offset + 3] = RGBA_OPAQUE_ALPHA;
    return;
  }

  // Alpha-composite the wall color over the existing framebuffer pixel. The
  // Uint8ClampedArray assignment rounds and clamps the blended float result.
  const inv = 1 - coverage;
  framebuffer[offset] = wallColor.r * coverage + framebuffer[offset] * inv;
  framebuffer[offset + 1] =
    wallColor.g * coverage + framebuffer[offset + 1] * inv;
  framebuffer[offset + 2] =
    wallColor.b * coverage + framebuffer[offset + 2] * inv;
  framebuffer[offset + 3] = RGBA_OPAQUE_ALPHA;
}
