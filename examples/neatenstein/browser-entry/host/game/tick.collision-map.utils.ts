/**
 * Collision-map resolution executor for the Neatenstein tick pipeline.
 *
 * Owns the one-entry collision-map cache used when callers omit an explicit
 * map.
 *
 * @module
 */

import {
  buildNeatensteinMap,
  createCollisionMap,
  type CollisionMap,
} from '../../renderer/map';
import { NEATENSTEIN_MAP_SIZE } from './constants';
import type { GameState } from './types';

/**
 * One-entry collision-map cache used only when callers omit an explicit map.
 *
 * Runtime callers should still pass a prebuilt {@link CollisionMap}; this cache
 * prevents accidental per-tick map rebuilding in tests or fallback paths while
 * keeping the module simple and deterministic.
 */
let cachedCollisionSeed: number | null = null;
let cachedCollisionMap: CollisionMap | null = null;

/**
 * Resolve the collision map for a tick.
 *
 * If the caller provides a map, it is used directly. Otherwise, a deterministic
 * map is built from the state seed and cached for later calls with the same
 * seed.
 *
 * @param state - Current game state.
 * @param collisionMap - Optional caller-provided collision map.
 * @returns Collision map for this tick.
 */
export function resolveCollisionMap(
  state: GameState,
  collisionMap?: CollisionMap,
): CollisionMap {
  if (collisionMap) {
    return collisionMap;
  }

  if (cachedCollisionMap && cachedCollisionSeed === state.seed) {
    return cachedCollisionMap;
  }

  const flatMap = buildNeatensteinMap(state.seed);
  const nextCollisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);

  cachedCollisionSeed = state.seed;
  cachedCollisionMap = nextCollisionMap;

  return nextCollisionMap;
}
