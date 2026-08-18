/**
 * Floor-band calculation executors extracted from the floor renderer.
 *
 * Contains pure leaf functions that compute which alpha band a depth ratio
 * maps to and create the depth-banded segment buffers used by the grid
 * renderer.
 *
 * @module
 */

import {
  projectNeatensteinGridPointInto,
  type NeatensteinGridProjectionContext,
  type ProjectedNeatensteinGridPoint,
} from './floor.projection.utils';
import type { NeatensteinFloorRenderContext } from './renderer.floor.types';
import {
  NEATENSTEIN_FLOOR_ALPHA_BANDS,
  NEATENSTEIN_FLOOR_LINE_SAMPLES,
} from './renderer.floor.constants';
import type { NeatensteinFloorSegmentBuffer } from './renderer.floor.types';
import { clamp } from '../shared/math-guards.utils';

// Re-export previously-public symbols that moved to dedicated files.
export type { NeatensteinFloorSegmentBuffer } from './renderer.floor.types';
export {
  NEATENSTEIN_FLOOR_ALPHA_BANDS,
  NEATENSTEIN_FLOOR_LINE_SAMPLES,
} from './renderer.floor.constants';

/**
 * Map a normalized depth ratio to an alpha-band index.
 *
 * @param depthRatio - Normalized depth ratio where `0` is far and `1` is near.
 * @returns Band index in the range `0..NEATENSTEIN_FLOOR_ALPHA_BANDS - 1`.
 */
export function bandIndexForDepthRatio(depthRatio: number): number {
  const clampedRatio = clamp(depthRatio, 0, 1);

  return Math.min(
    NEATENSTEIN_FLOOR_ALPHA_BANDS - 1,
    Math.floor(clampedRatio * NEATENSTEIN_FLOOR_ALPHA_BANDS),
  );
}

/**
 * Create one flat segment buffer per alpha band.
 *
 * @returns Empty depth-banded segment buffers.
 */
export function createNeatensteinFloorSegmentBands(): NeatensteinFloorSegmentBuffer[] {
  return Array.from(
    { length: NEATENSTEIN_FLOOR_ALPHA_BANDS },
    () => [] as NeatensteinFloorSegmentBuffer,
  );
}

/**
 * Reusable scratch slots for ping-pong projection in
 * {@link appendNeatensteinGridLine}.
 *
 * Kept at module level to avoid per-call allocation (called 240+ times per
 * frame for the 120×120 grid). Fields are always overwritten by
 * {@link projectNeatensteinGridPointInto} before being read, so no explicit
 * reset is needed between calls.
 */
const gridLineScratchA: ProjectedNeatensteinGridPoint = {
  x: 0,
  y: 0,
  depthRatio: 0,
  distance: 0,
};
const gridLineScratchB: ProjectedNeatensteinGridPoint = {
  x: 0,
  y: 0,
  depthRatio: 0,
  distance: 0,
};

/**
 * Append one sampled world-space grid line to the depth-banded segment buffers.
 *
 * The line is sampled at evenly spaced points. Consecutive visible projected
 * samples become one screen-space segment. If a sample is behind the camera,
 * the visible run is broken so the next valid sample starts a new segment.
 *
 * @param bands - Depth-banded flat segment buffers.
 * @param projection - Shared projection constants for this frame.
 * @param fixedCoord - Integer world coordinate for the fixed axis.
 * @param isXLine - `true` for constant-X lines, `false` for constant-Y lines.
 * @param startCoord - Start value for the varying axis.
 * @param endCoord - End value for the varying axis.
 * @param forCeiling - Whether to mirror the projection above the horizon.
 */
export function appendNeatensteinGridLine(
  bands: NeatensteinFloorSegmentBuffer[],
  projection: NeatensteinGridProjectionContext,
  fixedCoord: number,
  isXLine: boolean,
  startCoord: number,
  endCoord: number,
  forCeiling: boolean,
): void {
  // Ping-pong scratch slots so that `previous` retains values from the
  // prior iteration while `current` is written into the other slot.
  // Using a single shared scratch would alias `previous` and `current`
  // to the same object, producing zero-length segments.
  // Module-level to avoid 480+ object allocations per frame (A2 Fix 1).
  const scratchA = gridLineScratchA;
  const scratchB = gridLineScratchB;
  let previous: ProjectedNeatensteinGridPoint | null = null;
  let slotFlip = false;

  for (let i = 0; i <= NEATENSTEIN_FLOOR_LINE_SAMPLES; i += 1) {
    const t = i / NEATENSTEIN_FLOOR_LINE_SAMPLES;
    const coord = startCoord + (endCoord - startCoord) * t;

    // Constant-X lines vary Y; constant-Y lines vary X.
    const worldX = isXLine ? fixedCoord : coord;
    const worldY = isXLine ? coord : fixedCoord;

    const scratch = slotFlip ? scratchB : scratchA;
    slotFlip = !slotFlip;

    const projected = projectNeatensteinGridPointInto(
      worldX,
      worldY,
      projection,
      forCeiling,
      scratch,
    );

    if (projected === null) {
      previous = null;
      continue;
    }

    if (previous !== null) {
      // C1.2: Split the segment at band boundaries so no segment straddles
      // two alpha bands. This replaces the single-band direct push with
      // per-boundary splitting for correct fog/alpha assignment.
      splitNeatensteinFloorSegmentAtBandBoundaries(
        bands,
        previous.x,
        previous.y,
        projected.x,
        projected.y,
        previous.depthRatio,
        projected.depthRatio,
      );
    }

    previous = projected;
  }
}

/**
 * Split a floor segment at alpha-band boundaries so no segment straddles two
 * bands (C1.2).
 *
 * If the segment's start and end depth ratios fall in different bands, the
 * segment is split at each band boundary crossing. Each piece is pushed into
 * the appropriate band's flat buffer as a 4-float tuple (`x1, y1, x2, y2`).
 * If both endpoints are in the same band, the segment is pushed whole.
 *
 * @param bands - Depth-banded flat segment buffers (mutated in place).
 * @param x1 - Screen-space X of the segment start.
 * @param y1 - Screen-space Y of the segment start.
 * @param x2 - Screen-space X of the segment end.
 * @param y2 - Screen-space Y of the segment end.
 * @param depthRatioStart - Normalized depth ratio at the segment start.
 * @param depthRatioEnd - Normalized depth ratio at the segment end.
 */
export function splitNeatensteinFloorSegmentAtBandBoundaries(
  bands: NeatensteinFloorSegmentBuffer[],
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  depthRatioStart: number,
  depthRatioEnd: number,
): void {
  const startBand = bandIndexForDepthRatio(depthRatioStart);
  const endBand = bandIndexForDepthRatio(depthRatioEnd);

  // Same band — push the whole segment.
  if (startBand === endBand) {
    bands[startBand].push(x1, y1, x2, y2);
    return;
  }

  // Walk through each band boundary between start and end.
  const lo = Math.min(startBand, endBand);
  const hi = Math.max(startBand, endBand);
  const ratioSpan = depthRatioEnd - depthRatioStart;
  const absSpan = Math.abs(ratioSpan);

  // Track the current split point as we cross boundaries.
  let curX = x1;
  let curY = y1;
  let curBand = startBand;

  const bandCount = NEATENSTEIN_FLOOR_ALPHA_BANDS;

  if (ratioSpan > 0) {
    // Positive span: iterate boundaries ascending (from lo+1 to hi).
    // Crossing boundary b upward means leaving band b-1 and entering band b.
    for (let b = lo + 1; b <= hi; b += 1) {
      const boundaryRatio = b / bandCount;
      if (boundaryRatio > depthRatioEnd) break;
      if (absSpan < 1e-12) break;

      const t = (boundaryRatio - depthRatioStart) / ratioSpan;
      if (t < 0 || t > 1) continue;

      const splitX = x1 + (x2 - x1) * t;
      const splitY = y1 + (y2 - y1) * t;

      bands[curBand].push(curX, curY, splitX, splitY);

      curX = splitX;
      curY = splitY;
      curBand = b;
    }
  } else {
    // Negative span: iterate boundaries descending (from hi down to lo+1).
    // Crossing boundary b downward means leaving band b and entering band b-1,
    // so the segment from cur to the split point is in band b.
    for (let b = hi; b >= lo + 1; b -= 1) {
      const boundaryRatio = b / bandCount;
      if (boundaryRatio < depthRatioEnd) break;
      if (absSpan < 1e-12) break;

      const t = (boundaryRatio - depthRatioStart) / ratioSpan;
      if (t < 0 || t > 1) continue;

      const splitX = x1 + (x2 - x1) * t;
      const splitY = y1 + (y2 - y1) * t;

      curBand = b;
      bands[curBand].push(curX, curY, splitX, splitY);

      curX = splitX;
      curY = splitY;
    }
  }

  // Push the remaining tail segment.
  bands[endBand].push(curX, curY, x2, y2);
}

/**
 * Stroke one flat segment buffer.
 *
 * Values are read in groups of four: `x1, y1, x2, y2`.
 *
 * @param ctx - Canvas-like rendering context.
 * @param segments - Flat screen-space segment buffer.
 */
export function strokeNeatensteinFloorBand(
  ctx: NeatensteinFloorRenderContext,
  segments: NeatensteinFloorSegmentBuffer,
): void {
  ctx.beginPath();

  for (let i = 0; i < segments.length; i += 4) {
    ctx.moveTo(segments[i], segments[i + 1]);
    ctx.lineTo(segments[i + 2], segments[i + 3]);
  }

  ctx.stroke();
}