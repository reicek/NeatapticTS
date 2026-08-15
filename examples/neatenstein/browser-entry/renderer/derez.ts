/**
 * Tron-style pixel-by-pixel derez death animation helpers.
 *
 * When an enemy dies its voxel sprite dissolves pixel-by-pixel in a scattered
 * pattern driven by a deterministic hash. Each pixel is independently tested
 * against the animation progress `t = elapsedMs / durationMs`: when the
 * per-pixel noise value falls below `t`, the pixel is removed. Surviving
 * pixels are tinted toward {@link NEATENSTEIN_ENEMY_DEATH_COLOR} so the
 * crumbling silhouette shifts to a cool gray as it disintegrates.
 *
 * The hash operates on **logical** sprite-grid coordinates (0–47), not raw
 * voxel coordinates (0–191), so the dissolution pattern is stable regardless
 * of the voxel upscale factor. A per-enemy seed (enemy index) ensures every
 * enemy shatters differently while remaining fully deterministic and replay-
 * stable.
 *
 * @module
 */

import { ROBOT_SPRITE_SCALE } from '../../robot-sprite-data.js';
import {
  DEREZ_HASH_MODULUS,
  DEREZ_HASH_PRIME_1,
  DEREZ_HASH_PRIME_2,
  DEREZ_HASH_PRIME_3,
} from './renderer.rng.constants';

/**
 * Prime-mixing integer hash producing a deterministic `[0, 1)` value for
 * logical sprite-grid coordinates and a per-enemy seed.
 *
 * The hash combines three large primes via integer multiply and XOR, then
 * normalises the unsigned 32-bit result to `[0, 1)` by dividing by `2^32`.
 * This guarantees the output is strictly less than 1 even when the raw hash
 * is `0xffffffff`.
 *
 * @param x - Logical X coordinate (0–47 after {@link voxelToLogical}).
 * @param y - Logical Y coordinate (0–47 after {@link voxelToLogical}).
 * @param seed - Per-enemy deterministic seed (typically `enemy.index`).
 * @returns Float in `[0, 1)` — deterministic for identical inputs.
 */
export function derezHash(x: number, y: number, seed: number): number {
  return (
    (((x * DEREZ_HASH_PRIME_1 + y * DEREZ_HASH_PRIME_2) ^ (seed * DEREZ_HASH_PRIME_3)) >>> 0) /
    DEREZ_HASH_MODULUS
  );
}

/**
 * Convert a raw voxel coordinate to a logical sprite-grid coordinate by
 * dividing by {@link ROBOT_SPRITE_SCALE} and flooring.
 *
 * With `ROBOT_SPRITE_SCALE = 4`, voxels 0–3 map to logical 0, voxels 4–7 to
 * logical 1, and so on up to voxel 191 → logical 47.
 *
 * @param voxelCoord - Raw voxel coordinate (0–191).
 * @returns Logical sprite-grid coordinate (0–47).
 */
export function voxelToLogical(voxelCoord: number): number {
  return Math.floor(voxelCoord / ROBOT_SPRITE_SCALE);
}

/**
 * Determine whether a single voxel pixel should be dissolved at the given
 * animation progress.
 *
 * Converts the voxel coordinates to logical coordinates, computes a
 * deterministic noise hash, and dissolves the pixel when `noise < t` where
 * `t` is the normalised animation progress (`elapsedMs / durationMs`).
 *
 * At `t = 0` no pixels dissolve (all noise values are `≥ 0`). At `t = 1`
 * every pixel dissolves (all noise values are `< 1`). At intermediate values
 * the pattern is spatially scattered because the hash mixes both axes
 * independently.
 *
 * @param voxelX - Raw voxel X coordinate (0–191).
 * @param voxelY - Raw voxel Y coordinate (0–191).
 * @param seed - Per-enemy deterministic seed.
 * @param elapsedMs - Elapsed milliseconds since death began.
 * @param durationMs - Total de-rez animation duration in milliseconds.
 * @returns `true` when the pixel should be dissolved (skipped) this frame.
 */
export function shouldDissolvePixel(
  voxelX: number,
  voxelY: number,
  seed: number,
  elapsedMs: number,
  durationMs: number,
): boolean {
  const logicalX = voxelToLogical(voxelX);
  const logicalY = voxelToLogical(voxelY);
  const noise = derezHash(logicalX, logicalY, seed);
  const t = elapsedMs / durationMs;
  return noise < t;
}
