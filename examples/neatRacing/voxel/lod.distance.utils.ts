/**
 * LOD distance swap for Neat Racing voxel car tiers (spec §7.3.4; slice
 * P1-S3-E, AC-012).
 *
 * The car renders in two tiers: the fine `lod128` tier (full 128³ grid) and
 * the coarse `lod64` tier (2×2×2 merge, §7.3.3). The tier is selected from
 * the camera distance so that the worst-case geometric error of a
 * coarse-vs-fine surface projects below a fixed pixel budget: the coarse
 * voxel is 2× the fine voxel, so the worst-case surface error is one fine
 * voxel (`NEATRACING_VOXEL_WORLD_SIZE`), and the swap happens when that
 * error projects below 0.5 px at the fixed render height and 60° FOV
 * (TP-06). The swap distance is therefore **computed** from car dimensions,
 * the fixed render height, and the FOV — the derived ~44 m value is a
 * consequence, not an input (LD-20).
 *
 * Hysteresis (P4): near and far thresholds sit ±10% (TP-06) around the
 * computed swap distance so overtaking cars do not rapid-toggle tiers at
 * the boundary. The tier only changes when the distance crosses the edge on
 * the opposite side of the current tier; distances inside the band hold the
 * current tier. The selector is a pure function with zero per-frame
 * allocation (P9) — it returns existing tier constants, never a fresh
 * object.
 *
 * @module
 */

import { NEATRACING_FIXED_RENDER_HEIGHT } from '../browser-entry/constants';

/**
 * The fine car voxel tier: the full 128³ grid (spec §7.3.3; tier names are
 * locked to the spec wording).
 */
export const LOD_TIER_FINE = 'lod128';

/** The coarse car voxel tier: the 2×2×2-merged 64³ grid (spec §7.3.3). */
export const LOD_TIER_COARSE = 'lod64';

/**
 * Car voxel LOD tiers, named per spec §7.3.3 (`lod128` fine, `lod64`
 * coarse).
 */
export type LodTier = typeof LOD_TIER_FINE | typeof LOD_TIER_COARSE;

/**
 * Longest car dimension in meters, assumed to fill the 128 voxel axis
 * (spec §7.3.4 G1 cubic-voxel assumption: length ≈ 4.5 m).
 */
export const NEATRACING_VOXEL_CAR_LENGTH_M = 4.5;

/** Voxel axis resolution of the fine (lod128) car tier. */
export const NEATRACING_LOD_FINE_GRID_SIZE = 128;

/**
 * World size of one fine voxel in meters: the car's longest dimension
 * fills the 128 axis, so `carLength / 128` is the per-axis voxel size
 * (spec §7.3.4 named computed constant, AC-012 — not a magic number).
 */
export const NEATRACING_VOXEL_WORLD_SIZE =
  NEATRACING_VOXEL_CAR_LENGTH_M / NEATRACING_LOD_FINE_GRID_SIZE;

/**
 * Vertical field of view in degrees of the projection basis used for the
 * swap decision (spec §7.3.4: 60° FOV at the fixed render height).
 */
export const NEATRACING_LOD_SWAP_FOV_DEG = 60;

/**
 * Projected pixel error at which the coarse tier becomes indistinguishable
 * from the fine tier (TP-06: 0.5 px projected error).
 */
export const NEATRACING_LOD_SWAP_PIXEL_ERROR_PX = 0.5;

/**
 * Hysteresis band around the computed swap distance as a fraction
 * (TP-06/P4: ±10% separate near/far thresholds so overtaking cars do not
 * rapid-toggle tiers).
 */
export const NEATRACING_LOD_HYSTERESIS_BAND_FRACTION = 0.1;

/** Divisor mapping the full render height to the projection half-height. */
const FOCAL_HALF_HEIGHT_DIVISOR = 2;

/** Divisor mapping the full vertical FOV to its half angle. */
const FOCAL_HALF_FOV_DIVISOR = 2;

/** Degrees-to-radians conversion factor for the trigonometric focal length. */
const DEGREES_TO_RADIANS = Math.PI / 180;

/**
 * Focal length in pixels of the §7.3.4 projection basis, computed from the
 * fixed render height and the swap FOV:
 * `focalPx = (height / 2) / tan(fov / 2) ≈ (720 / 2) / tan(30°) ≈ 623 px`.
 */
export const NEATRACING_LOD_SWAP_FOCAL_PX =
  NEATRACING_FIXED_RENDER_HEIGHT /
  FOCAL_HALF_HEIGHT_DIVISOR /
  Math.tan(
    (NEATRACING_LOD_SWAP_FOV_DEG / FOCAL_HALF_FOV_DIVISOR) * DEGREES_TO_RADIANS,
  );

/**
 * Camera distance in meters at which one fine voxel of coarse-vs-fine
 * geometric error projects exactly the 0.5 px budget:
 * `swapDist = errWorld × focalPx / 0.5 ≈ 44 m` (spec §7.3.4; the value is
 * derived, TP-06 locks only the formula and the pixel threshold).
 */
export const NEATRACING_LOD_SWAP_DISTANCE_M =
  (NEATRACING_VOXEL_WORLD_SIZE * NEATRACING_LOD_SWAP_FOCAL_PX) /
  NEATRACING_LOD_SWAP_PIXEL_ERROR_PX;

/**
 * Near hysteresis edge in meters (swap distance − 10%): a coarse-tier car
 * returns to the fine tier only when the distance drops to this edge or
 * below.
 */
export const NEATRACING_LOD_SWAP_DISTANCE_NEAR_M =
  NEATRACING_LOD_SWAP_DISTANCE_M *
  (1 - NEATRACING_LOD_HYSTERESIS_BAND_FRACTION);

/**
 * Far hysteresis edge in meters (swap distance + 10%): a fine-tier car
 * downgrades to the coarse tier only when the distance reaches this edge or
 * beyond.
 */
export const NEATRACING_LOD_SWAP_DISTANCE_FAR_M =
  NEATRACING_LOD_SWAP_DISTANCE_M *
  (1 + NEATRACING_LOD_HYSTERESIS_BAND_FRACTION);

/**
 * Selects the car voxel LOD tier for a camera distance with hysteresis
 * (spec §7.3.4, AC-012).
 *
 * The swap decision depends on the tier currently rendered, which is what
 * produces the hysteresis band: a fine (`lod128`) car stays fine until the
 * distance reaches the far edge (+10%), and a coarse (`lod64`) car stays
 * coarse until the distance drops to the near edge (−10%). Inside the band
 * — and at the base swap distance itself — the current tier is held, so an
 * overtaking car oscillating around the boundary never rapid-toggles
 * (mitigation for risk R-03).
 *
 * The function is total and deterministic for any finite `distanceM`
 * (expected range is ≥ 0): a non-finite distance matches neither edge and
 * returns the current tier unchanged, which is the safe no-op fallback.
 *
 * @param distanceM - Camera distance to the car in meters (expected ≥ 0).
 * @param currentTier - The tier currently rendered for the car.
 * @returns The tier to render after this frame: `lod128` or `lod64`.
 *
 * @example
 * ```ts
 * let tier: LodTier = LOD_TIER_FINE;
 * // Per frame, at the observed camera distance:
 * tier = selectLodTier(cameraDistanceM, tier);
 * ```
 */
export function selectLodTier(
  distanceM: number,
  currentTier: LodTier,
): LodTier {
  if (currentTier === LOD_TIER_FINE) {
    return distanceM >= NEATRACING_LOD_SWAP_DISTANCE_FAR_M
      ? LOD_TIER_COARSE
      : LOD_TIER_FINE;
  }
  return distanceM <= NEATRACING_LOD_SWAP_DISTANCE_NEAR_M
    ? LOD_TIER_FINE
    : LOD_TIER_COARSE;
}
