/**
 * Decoration LOD/distance projection for the neatRacing right-world pass
 * (slice P1-decoration-scale, Step 14D, AC-051).
 *
 * Decides how many right-world decorations to draw, how they scale with
 * camera distance, and whether a 2D billboard or voxel fallback should be
 * used. The P1 decision record keeps 2D sprites as the preferred path and
 * reserves the voxel branch for future phases.
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - Exports {@link NEATRACING_DECORATION_LOD_BUDGET} as the per-frame cap.
 * - Exports {@link computeDecorationScale} returning a larger positive scale
 *   for near distances and a smaller positive scale for far distances.
 * - Exports {@link resolveDecorationRenderMode} returning '2d' for the P1
 *   decision at all distances, and 'voxel' only when an explicit fallback
 *   threshold is supplied and the distance exceeds it.
 * - Exports {@link computeDecorationLod} exposing visibility, scale, and mode.
 * - Exports {@link selectDecorationsForCamera} culling distant decorations and
 *   keeping only the nearest ones up to the budget.
 */

/** Maximum number of visible right-world decorations drawn per frame. */
export const NEATRACING_DECORATION_LOD_BUDGET = 128;

/** Distance beyond which decorations are culled as too far to matter. */
export const NEATRACING_DECORATION_CULL_DISTANCE_M = 320;

/** Reference distance where decoration scale equals 1.0. */
export const NEATRACING_DECORATION_NEAR_REFERENCE_M = 10;

/** Floor value so distant decorations remain visible instead of vanishing. */
export const NEATRACING_DECORATION_MIN_SCALE = 0.01;

/** Decision record documenting the 2D-preferred / voxel-fallback choice. */
export interface DecorationRenderDecision {
  /** Preferred representation when both are viable. */
  readonly preferredMode: '2d' | 'voxel';
  /**
   * Distance threshold in meters at which the voxel fallback takes over.
   * `null` means the fallback is disabled and 2D is always used.
   */
  readonly voxelFallbackThresholdM: number | null;
  /** Human-readable rationale for the decision. */
  readonly rationale: string;
}

/** LOD tier returned for a decoration at a given distance. */
export interface DecorationLod {
  /** Selected render representation. */
  readonly mode: '2d' | 'voxel';
  /** Distance-aware scale (unitless, positive). */
  readonly scale: number;
  /** Distance from the camera in meters. */
  readonly distance: number;
  /** Whether the decoration is close enough to be drawn. */
  readonly visible: boolean;
}

/**
 * P1 decision record: 2D billboards are preferred for all visible right-world
 * decorations. The voxel fallback branch is explicit but disabled in P1; it
 * is reserved for future phases where 2D sprites might be unsuitable.
 */
export const NEATRACING_DECORATION_P1_DECISION: DecorationRenderDecision = {
  preferredMode: '2d',
  voxelFallbackThresholdM: null,
  rationale:
    'P1 uses 2D sprite billboards for all visible right-world decorations. ' +
    'Voxel fallback is an explicit future-phase branch, not active in P1.',
};

/**
 * Computes a distance-aware scale for a decoration.
 *
 * The anchor is the near reference distance, where the returned scale is 1.0.
 * Distant decorations shrink inversely with distance and are floored at the
 * configured minimum so they remain visible rather than disappearing.
 *
 * @param distance - Distance from the camera in meters.
 * @returns Positive scale that is larger near the camera and smaller far away.
 */
export function computeDecorationScale(distance: number): number {
  const clampedDistance = Math.max(distance, NEATRACING_DECORATION_NEAR_REFERENCE_M / 100);
  const rawScale = NEATRACING_DECORATION_NEAR_REFERENCE_M / clampedDistance;

  return Math.max(NEATRACING_DECORATION_MIN_SCALE, rawScale);
}

/**
 * Resolves the 2D-vs-voxel render mode for a decoration at the given distance.
 *
 * P1 always prefers 2D. When an explicit voxel fallback threshold is supplied,
 * distances strictly greater than the threshold switch to the voxel branch;
 * all other distances stay in 2D.
 *
 * @param distance - Distance from the camera in meters.
 * @param decision - Render decision; defaults to the P1 2D-preferred record.
 * @returns '2d' or 'voxel' depending on the decision threshold.
 */
export function resolveDecorationRenderMode(
  distance: number,
  decision: DecorationRenderDecision = NEATRACING_DECORATION_P1_DECISION,
): '2d' | 'voxel' {
  if (decision.voxelFallbackThresholdM === null) {
    return decision.preferredMode;
  }

  return distance > decision.voxelFallbackThresholdM ? 'voxel' : '2d';
}

/**
 * Computes the full LOD record for a decoration at a given distance.
 *
 * @param distance - Distance from the camera in meters.
 * @param decision - Render decision; defaults to the P1 2D-preferred record.
 * @returns LOD tier with visibility, scale, and mode.
 */
export function computeDecorationLod(
  distance: number,
  decision: DecorationRenderDecision = NEATRACING_DECORATION_P1_DECISION,
): DecorationLod {
  return {
    mode: resolveDecorationRenderMode(distance, decision),
    scale: computeDecorationScale(distance),
    distance,
    visible: distance <= NEATRACING_DECORATION_CULL_DISTANCE_M,
  };
}

/** Decoration with a world-space position. */
export interface PositionedDecoration {
  readonly position: readonly [number, number, number];
}

/**
 * Sorts decorations by distance from the camera and culls those beyond the
 * budget or the far cull distance.
 *
 * The returned array is front-to-back (nearest first) and capped at the
 * budget. It is built without mutating the input list.
 *
 * This helper is convenient for scene-build or test paths; the per-frame
 * render path uses {@link selectVisibleDecorationsForCamera} with a
 * preallocated scratch buffer to preserve the zero per-frame allocation
 * contract (P9).
 *
 * @param decorations - Unsorted candidate decorations.
 * @param cameraPosition - Camera world position in meters.
 * @param budget - Maximum decorations to keep; defaults to the LOD budget.
 * @returns Nearest decorations, front-to-back, capped at the budget.
 */
export function selectDecorationsForCamera<T extends PositionedDecoration>(
  decorations: readonly T[],
  cameraPosition: readonly [number, number, number],
  budget: number = NEATRACING_DECORATION_LOD_BUDGET,
): readonly T[] {
  const [cx, cy, cz] = cameraPosition;

  const visible = decorations
    .map((decoration) => ({
      decoration,
      distance: Math.hypot(
        decoration.position[0] - cx,
        decoration.position[1] - cy,
        decoration.position[2] - cz,
      ),
    }))
    .filter(({ distance }) => distance <= NEATRACING_DECORATION_CULL_DISTANCE_M)
    .toSorted((a, b) => a.distance - b.distance);

  return visible.slice(0, budget).map(({ decoration }) => decoration);
}

/**
 * Preallocated scratch buffer used by
 * {@link selectVisibleDecorationsForCamera}. Reusing one scratch object per
 * pass keeps the steady-state path free of per-frame JavaScript allocations.
 */
export interface DecorationSelectionScratch {
  /** Per-slot distance from the camera in meters. */
  distances: Float32Array;
  /** Per-slot index into the source decoration array. */
  indices: Uint16Array;
  /** Maximum number of decorations the scratch can hold. */
  capacity: number;
}

/**
 * Creates a preallocated scratch buffer for zero-allocation per-frame
 * decoration selection.
 *
 * @param capacity - Maximum number of decorations that will be selected.
 * @returns Scratch buffer ready for {@link selectVisibleDecorationsForCamera}.
 */
export function createDecorationSelectionScratch(
  capacity: number,
): DecorationSelectionScratch {
  return {
    distances: new Float32Array(capacity),
    indices: new Uint16Array(capacity),
    capacity,
  };
}

/**
 * Fills the scratch buffer with decorations visible from the runtime camera,
 * sorted nearest-first, without allocating any new objects.
 *
 * The returned count is the number of valid entries in the scratch arrays.
 * Callers can iterate `scratch.indices[0..count)` for near-to-far drawing, or
 * iterate in reverse for far-to-near translucent sorting.
 *
 * @param decorations - Candidate decorations.
 * @param cameraPosition - Runtime camera world position in meters.
 * @param cullDistanceM - Distance beyond which decorations are dropped.
 * @param budget - Maximum decorations to keep.
 * @param scratch - Preallocated scratch buffer.
 * @returns Number of visible decorations written to the scratch buffer.
 */
export function selectVisibleDecorationsForCamera<T extends PositionedDecoration>(
  decorations: readonly T[],
  cameraPosition: readonly [number, number, number],
  cullDistanceM: number,
  budget: number,
  scratch: DecorationSelectionScratch,
): number {
  const [cx, cy, cz] = cameraPosition;
  const limit = Math.min(decorations.length, scratch.capacity, budget);
  let count = 0;

  for (let index = 0; index < decorations.length; index += 1) {
    if (count >= limit) {
      break;
    }

    const decoration = decorations[index]!;
    const distance = Math.hypot(
      decoration.position[0] - cx,
      decoration.position[1] - cy,
      decoration.position[2] - cz,
    );

    if (distance <= cullDistanceM) {
      scratch.distances[count] = distance;
      scratch.indices[count] = index;
      count += 1;
    }
  }

  // Insertion sort by ascending distance; indices travel with distances.
  for (let right = 1; right < count; right += 1) {
    const keyDistance = scratch.distances[right];
    const keyIndex = scratch.indices[right];
    let left = right - 1;

    while (left >= 0 && scratch.distances[left] > keyDistance) {
      scratch.distances[left + 1] = scratch.distances[left];
      scratch.indices[left + 1] = scratch.indices[left];
      left -= 1;
    }

    scratch.distances[left + 1] = keyDistance;
    scratch.indices[left + 1] = keyIndex;
  }

  return count;
}
