/**
 * Type definitions for the map module extracted from {@link module:./map}.
 *
 * @module
 */

/**
 * Read-only interface used by movement, AI, and collision systems to test
 * whether a grid cell blocks movement.
 *
 * The interface deliberately hides the underlying storage format so callers do
 * not depend on row-major indexing or typed-array details.
 */
export interface CollisionMap {
  /**
   * Return whether the cell at integer grid coordinate `(x, y)` is solid.
   *
   * Coordinates outside the map are treated as solid by the default
   * implementation, preventing entities from leaving the arena.
   *
   * @param x - Integer grid X coordinate.
   * @param y - Integer grid Y coordinate.
   * @returns Whether the requested cell blocks movement.
   */
  isSolid(x: number, y: number): boolean;
}