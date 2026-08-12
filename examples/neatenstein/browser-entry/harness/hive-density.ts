/**
 * HIVE DENSITY computation for the Neatenstein SWARM enemy backend.
 *
 * The density metric is a normalized 0–1 coordination budget that measures how
 * tightly the swarm's stigmergic coordinates are ordered. It is sensitive to the
 * coordinate order (a reversed or shuffled order yields a different value) so
 * the metric cannot be hardcoded from a static snapshot.
 *
 * @module
 */

import type { SwarmSnapshot } from './types';

/** Coordinate bounds used by the SWARM backend; coordinates are sampled in [-1, 1]^2. */
const HIVE_COORDINATE_BOUND = 1;

/**
 * Squared maximum distance between two coordinates in the bounded swarm space.
 *
 * The farthest two points can be in the [-1, 1]^2 box is the diagonal, whose
 * squared length is (2)^2 + (2)^2 = 8.
 */
const HIVE_MAX_SEGMENT_SQUARED =
  (2 * HIVE_COORDINATE_BOUND) ** 2 + (2 * HIVE_COORDINATE_BOUND) ** 2;

/**
 * Tiny positive offset added to the density denominator so that degenerate
 * snapshots (fewer than two coordinates) return 0 instead of dividing by zero.
 */
const HIVE_DENSITY_EPSILON = 1e-12;

/**
 * Density thresholds that partition the 0–1 coordination budget into behavior
 * bands.
 */
export const HIVE_DENSITY_THRESHOLDS: readonly number[] = [
  0.25, 0.5, 0.75, 1.0,
];

/**
 * Human-readable behavior label for each density threshold band.
 *
 * The label order matches {@link HIVE_DENSITY_THRESHOLDS}.
 */
export const HIVE_DENSITY_BEHAVIOR_LABELS: readonly string[] = [
  'Scattered',
  'Loose',
  'Coordinated',
  'Hive',
];

/**
 * Compute a deterministic, normalized HIVE DENSITY for a SWARM snapshot.
 *
 * The metric weights later coordinate transitions more heavily than earlier
 * ones, so shuffling the coordinate order changes the density even though the
 * underlying set of points is unchanged. The result is clamped to [0, 1].
 *
 * @param snapshot - Frozen SWARM snapshot.
 * @returns Normalized density in the closed interval [0, 1].
 *
 * @example
 * ```ts
 * const density = computeHiveDensity({
 *   kind: 'swarm',
 *   dna: 'swarm:1:abcdefgh',
 *   coordinates: [
 *     { x: 0.1, y: 0.2 },
 *     { x: -0.3, y: 0.4 },
 *   ],
 * });
 * console.log(density); // 0..1
 * ```
 */
export function computeHiveDensity(snapshot: SwarmSnapshot): number {
  const coordinates = snapshot.coordinates;
  const maxWeightSum = (coordinates.length * (coordinates.length - 1)) / 2;
  const maxRaw = HIVE_MAX_SEGMENT_SQUARED * maxWeightSum + HIVE_DENSITY_EPSILON;

  let weightedSquaredDistance = 0;
  for (let i = 0; i < coordinates.length - 1; i++) {
    const dx = coordinates[i].x - coordinates[i + 1].x;
    const dy = coordinates[i].y - coordinates[i + 1].y;
    const weight = i + 1;
    weightedSquaredDistance += weight * (dx * dx + dy * dy);
  }

  return weightedSquaredDistance / maxRaw;
}
