/**
 * @module enemy-navigation.types
 *
 * Extracted types for the enemy navigation module.
 */

/**
 * BFS distance map from a goal cell to every reachable cell.
 *
 * The `distances` buffer is a flat `Int32Array` of length `size * size`,
 * indexed as `y * size + x`. Wall cells are `wallValue`, unreachable cells
 * are `unreachableValue`, and reachable cells hold a non-negative distance
 * (0 at the goal, increasing by 1 per BFS step).
 */
export interface DistanceMap {
  /** Grid width and height (square grid). */
  readonly size: number;
  /** Flat distance buffer indexed as `y * size + x`. */
  readonly distances: Int32Array;
  /** Sentinel for wall cells. */
  readonly wallValue: number;
  /** Sentinel for unreachable cells. */
  readonly unreachableValue: number;
}