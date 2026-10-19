/**
 * Track grid builder for the neatRacing example (slice P1-grid-footprint,
 * AC-045).
 *
 * The Step 14B composition used a 1.0 m engine world-space ground grid as a
 * baseline (cell footprint area 1 m²). User visual feedback (2026-10-12)
 * requested larger cells, so the track grid now uses a 25 m² footprint,
 * giving a linear spacing of 5.0 m. The builder owns the constants and pure
 * math; the pass in `grid.pass.ts` consumes these values when configuring
 * the flat-horizon fullscreen draw.
 */

/** Baseline cell footprint area from the engine ground grid, in m². */
export const NEATRACING_TRACK_GRID_BASELINE_CELL_FOOTPRINT_AREA_M2 = 1;

/** Multiplier applied to the baseline footprint area for the track grid. */
export const NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_MULTIPLIER = 25;

/**
 * Target cell footprint area for the track grid, in m².
 * Equal to the baseline area times the multiplier.
 */
export const NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_M2 =
  NEATRACING_TRACK_GRID_BASELINE_CELL_FOOTPRINT_AREA_M2 *
  NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_MULTIPLIER;

/**
 * Linear spacing of the track grid lines, derived from the 25 m² cell
 * footprint area. For a square grid cell, `area = spacing²`, so
 * `spacing = sqrt(area)`.
 */
export const NEATRACING_TRACK_GRID_SPACING_M = Math.sqrt(
  NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_M2,
);

/**
 * Returns the track grid line spacing in world meters.
 *
 * This is a thin helper so tests can assert the same value the pass will
 * feed to the `uSpacing` uniform, without hard-coding `5.0` in multiple
 * places.
 *
 * @returns The track grid spacing in meters.
 *
 * @example
 * ```ts
 * const spacing = buildTrackGridSpacing(); // 5.0
 * ```
 */
export function buildTrackGridSpacing(): number {
  return NEATRACING_TRACK_GRID_SPACING_M;
}

/**
 * Counts how many grid lines fit into a one-axis world extent at the given
 * spacing. Used by tests to prove that the larger footprint area scales
 * the line density versus the 1.0 m baseline.
 *
 * @param extentM - World extent along one axis, in meters.
 * @param spacingM - Distance between adjacent grid lines, in meters.
 * @returns The number of whole grid intervals that fit in the extent.
 *
 * @example
 * ```ts
 * const count = trackGridLineCountForExtent(100, buildTrackGridSpacing());
 * ```
 */
export function trackGridLineCountForExtent(
  extentM: number,
  spacingM: number,
): number {
  if (spacingM <= 0) {
    throw new Error('track grid spacing must be positive');
  }
  return Math.round(extentM / spacingM);
}
