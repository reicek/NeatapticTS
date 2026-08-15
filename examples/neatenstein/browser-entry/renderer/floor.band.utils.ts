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
  projectNeatensteinGridPoint,
  type NeatensteinGridProjectionContext,
  type ProjectedNeatensteinGridPoint,
} from './floor.projection.utils';
import type { NeatensteinFloorRenderContext } from './renderer.floor.types';
import {
  NEATENSTEIN_FLOOR_ALPHA_BANDS,
  NEATENSTEIN_FLOOR_LINE_SAMPLES,
} from './renderer.floor.constants';
import type { NeatensteinFloorSegmentBuffer } from './renderer.floor.types';

// Re-export previously-public symbols that moved to dedicated files.
export type { NeatensteinFloorSegmentBuffer } from './renderer.floor.types';
export {
  NEATENSTEIN_FLOOR_ALPHA_BANDS,
  NEATENSTEIN_FLOOR_LINE_SAMPLES,
} from './renderer.floor.constants';

/**
 * Clamp a number into the inclusive range `[min, max]`.
 *
 * @param value - Value to clamp.
 * @param min - Lower bound.
 * @param max - Upper bound.
 * @returns Clamped value.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

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
  let previous: ProjectedNeatensteinGridPoint | null = null;

  for (let i = 0; i <= NEATENSTEIN_FLOOR_LINE_SAMPLES; i += 1) {
    const t = i / NEATENSTEIN_FLOOR_LINE_SAMPLES;
    const coord = startCoord + (endCoord - startCoord) * t;

    // Constant-X lines vary Y; constant-Y lines vary X.
    const worldX = isXLine ? fixedCoord : coord;
    const worldY = isXLine ? coord : fixedCoord;

    const projected = projectNeatensteinGridPoint(
      worldX,
      worldY,
      projection,
      forCeiling,
    );

    if (projected === null) {
      previous = null;
      continue;
    }

    if (previous !== null) {
      const averageDepthRatio =
        (previous.depthRatio + projected.depthRatio) / 2;
      const band = bandIndexForDepthRatio(averageDepthRatio);

      // Store as a flat tuple to avoid allocating one array/object per segment.
      bands[band].push(previous.x, previous.y, projected.x, projected.y);
    }

    previous = projected;
  }
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
