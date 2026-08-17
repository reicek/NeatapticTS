/**
 * 8-direction voxel snapshot renderer for the Neatenstein NGE demo.
 *
 * Projects a sparse N×192×M voxel grid into a canvas-sized RGBA buffer for
 * one of eight equally spaced yaw angles. The renderer uses orthographic
 * projection with a weak-perspective scale, a camera-left directional light,
 * emissive self-illumination for neon/accent voxels, and edge darkening so
 * silhouettes stay readable at 128×128 runtime resolution.
 */

import type { Voxel, VoxelGrid } from './voxel-enemy';

// Re-export types and constants from extracted modules so existing imports
// from snapshot-renderer.ts continue to work.
export type { VoxelSnapshot, SnapshotOptions } from './snapshot-renderer.types';
export {
  ENEMY_FRAME_SIZE_PX,
  NUM_DIRECTIONS,
  RGBA_OPAQUE_ALPHA,
  YAW_STEP_DEGREES,
  PROJECTION_MARGIN,
  AMBIENT_LIGHT,
  MAX_DIFFUSE,
  EMISSIVE_BOOST,
  EDGE_DARKENING_MAX,
  SNAPSHOT_EDGE_DARKENING_PER_FACE,
  LIGHT_DIRECTION,
  LIGHT_NORM,
} from './snapshot-renderer.constants';

import type { VoxelSnapshot, SnapshotOptions } from './snapshot-renderer.types';
import {
  ENEMY_FRAME_SIZE_PX,
  NUM_DIRECTIONS,
  RGBA_OPAQUE_ALPHA,
  YAW_STEP_DEGREES,
  PROJECTION_MARGIN,
  AMBIENT_LIGHT,
  MAX_DIFFUSE,
  EMISSIVE_BOOST,
  EDGE_DARKENING_MAX,
  SNAPSHOT_EDGE_DARKENING_PER_FACE,
  LIGHT_DIRECTION,
  LIGHT_NORM,
} from './snapshot-renderer.constants';

/**
 * Render an orthographic snapshot of a voxel enemy facing one of eight
 * equally spaced yaw directions.
 *
 * @param voxelGrid - The N×192×M voxel grid to render.
 * @param yawIndex - Camera yaw angle index in the range 0–7.
 *   0 = front, 2 = right, 4 = back, 6 = left.
 * @param options - Optional output dimensions.
 * @returns An RGBA snapshot buffer sized to the requested dimensions.
 * @throws {Error} when `yawIndex` is outside the 0–7 range.
 *
 * @example
 * ```ts
 * const grid = buildVoxelEnemy();
 * const front = renderVoxelSnapshot(grid, 0, { width: 192, height: 192 });
 * console.log(front.width, front.height); // 192 192
 * ```
 */
export function renderVoxelSnapshot(
  voxelGrid: VoxelGrid,
  yawIndex: number,
  options?: SnapshotOptions,
): VoxelSnapshot {
  validateYawIndex(yawIndex);

  const width = options?.width ?? ENEMY_FRAME_SIZE_PX;
  const height = options?.height ?? ENEMY_FRAME_SIZE_PX;
  const frame = createClearFrame(width, height);
  const zBuffer = new Float32Array(width * height).fill(-Infinity);

  const angle = (yawIndex * YAW_STEP_DEGREES * Math.PI) / 180;
  const cosYaw = Math.cos(angle);
  const sinYaw = Math.sin(angle);

  const centerX = voxelGrid.width / 2;
  const centerZ = voxelGrid.depth / 2;
  const scale = (height - 2 * PROJECTION_MARGIN) / voxelGrid.height;
  const offsetX = width / 2;
  const offsetY = height / 2;

  const occupancy =
    options?.occupancy ??
    buildOccupancyGrid(
      voxelGrid.width,
      voxelGrid.height,
      voxelGrid.depth,
      voxelGrid.voxels,
    );

  for (const voxel of voxelGrid.voxels) {
    const rx = (voxel.x - centerX) * cosYaw - (voxel.z - centerZ) * sinYaw;
    const rz = (voxel.x - centerX) * sinYaw + (voxel.z - centerZ) * cosYaw;
    const ry = voxel.y - voxelGrid.height / 2;

    const px = Math.round(offsetX + rx * scale);
    const py = Math.round(offsetY - ry * scale);

    if (px < 0 || px >= width || py < 0 || py >= height) {
      continue;
    }

    const index = py * width + px;
    if (rz <= zBuffer[index]) {
      continue;
    }

    zBuffer[index] = rz;
    const shade = computeShade(
      voxel,
      occupancy,
      voxelGrid.width,
      voxelGrid.height,
      voxelGrid.depth,
      cosYaw,
      sinYaw,
    );
    writePixel(frame.data, index * 4, voxel, shade);
  }

  return frame;
}

/**
 * Validate that the yaw index is an integer in the supported 0–7 range.
 */
function validateYawIndex(yawIndex: number): void {
  if (
    !Number.isInteger(yawIndex) ||
    yawIndex < 0 ||
    yawIndex >= NUM_DIRECTIONS
  ) {
    throw new Error(
      `yawIndex must be an integer in [0, ${NUM_DIRECTIONS - 1}]`,
    );
  }
}

/**
 * Create a transparent RGBA frame buffer of the requested size.
 */
function createClearFrame(width: number, height: number): VoxelSnapshot {
  return {
    width,
    height,
    data: new Uint8ClampedArray(width * height * 4),
  };
}

/**
 * Build a dense occupancy grid for fast neighbor lookups.
 *
 * The returned array has length `width * height * depth` and stores `1` for
 * occupied cells, `0` otherwise. Indexing follows `z * (height * width) +
 * y * width + x` so all six neighbor checks become constant-time reads instead
 * of string-based `Set` lookups.
 */
function buildOccupancyGrid(
  width: number,
  height: number,
  depth: number,
  voxels: readonly Voxel[],
): Uint8Array {
  const grid = new Uint8Array(width * height * depth);
  for (const voxel of voxels) {
    if (
      voxel.x >= 0 &&
      voxel.x < width &&
      voxel.y >= 0 &&
      voxel.y < height &&
      voxel.z >= 0 &&
      voxel.z < depth
    ) {
      grid[voxel.z * (height * width) + voxel.y * width + voxel.x] = 1;
    }
  }
  return grid;
}

/**
 * Return whether an integer voxel coordinate is occupied in the grid.
 */
function isOccupied(
  grid: Uint8Array,
  width: number,
  height: number,
  depth: number,
  x: number,
  y: number,
  z: number,
): boolean {
  if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
    return false;
  }
  return grid[z * (height * width) + y * width + x] === 1;
}

/**
 * Compute a combined shading multiplier for a voxel.
 *
 * Combines ambient light, a camera-left directional light, emissive
 * self-illumination, and edge darkening for isolated silhouette voxels.
 */
function computeShade(
  voxel: Voxel,
  occupancy: Uint8Array,
  gridWidth: number,
  gridHeight: number,
  gridDepth: number,
  cosYaw: number,
  sinYaw: number,
): number {
  const normal = computeNormal(
    voxel,
    occupancy,
    gridWidth,
    gridHeight,
    gridDepth,
  );
  const cameraNormal = rotateY(normal, cosYaw, sinYaw);

  const diffuse = Math.max(
    0,
    (cameraNormal.x * LIGHT_DIRECTION.x +
      cameraNormal.y * LIGHT_DIRECTION.y +
      cameraNormal.z * LIGHT_DIRECTION.z) /
      LIGHT_NORM,
  );

  const emissive = voxel.emissive ? EMISSIVE_BOOST : 0;
  const edgeFactor = computeEdgeFactor(
    voxel,
    occupancy,
    gridWidth,
    gridHeight,
    gridDepth,
  );

  return (AMBIENT_LIGHT + MAX_DIFFUSE * diffuse + emissive) * edgeFactor;
}

/**
 * Compute an outward-pointing normal from the occupied neighbors of a voxel.
 */
function computeNormal(
  voxel: Voxel,
  occupancy: Uint8Array,
  width: number,
  height: number,
  depth: number,
): { x: number; y: number; z: number } {
  const normal = { x: 0, y: 0, z: 0 };
  if (
    !isOccupied(occupancy, width, height, depth, voxel.x + 1, voxel.y, voxel.z)
  ) {
    normal.x += 1;
  }
  if (
    !isOccupied(occupancy, width, height, depth, voxel.x - 1, voxel.y, voxel.z)
  ) {
    normal.x -= 1;
  }
  if (
    !isOccupied(occupancy, width, height, depth, voxel.x, voxel.y + 1, voxel.z)
  ) {
    normal.y += 1;
  }
  if (
    !isOccupied(occupancy, width, height, depth, voxel.x, voxel.y - 1, voxel.z)
  ) {
    normal.y -= 1;
  }
  if (
    !isOccupied(occupancy, width, height, depth, voxel.x, voxel.y, voxel.z + 1)
  ) {
    normal.z += 1;
  }
  if (
    !isOccupied(occupancy, width, height, depth, voxel.x, voxel.y, voxel.z - 1)
  ) {
    normal.z -= 1;
  }

  const length = Math.hypot(normal.x, normal.y, normal.z);
  if (length > 0) {
    return {
      x: normal.x / length,
      y: normal.y / length,
      z: normal.z / length,
    };
  }

  return { x: 0, y: 0, z: 1 };
}

/**
 * Rotate a vector around the world Y axis by the current yaw angle.
 */
function rotateY(
  vector: { x: number; y: number; z: number },
  cosYaw: number,
  sinYaw: number,
): { x: number; y: number; z: number } {
  return {
    x: vector.x * cosYaw - vector.z * sinYaw,
    y: vector.y,
    z: vector.x * sinYaw + vector.z * cosYaw,
  };
}

/**
 * Darken voxels that sit on the silhouette edge.
 *
 * Voxels with few occupied neighbors are presumed to be on an exposed edge
 * and receive a small darkening penalty so outlines read clearly.
 */
function computeEdgeFactor(
  voxel: Voxel,
  occupancy: Uint8Array,
  width: number,
  height: number,
  depth: number,
): number {
  let neighborCount = 0;
  if (
    isOccupied(occupancy, width, height, depth, voxel.x + 1, voxel.y, voxel.z)
  ) {
    neighborCount += 1;
  }
  if (
    isOccupied(occupancy, width, height, depth, voxel.x - 1, voxel.y, voxel.z)
  ) {
    neighborCount += 1;
  }
  if (
    isOccupied(occupancy, width, height, depth, voxel.x, voxel.y + 1, voxel.z)
  ) {
    neighborCount += 1;
  }
  if (
    isOccupied(occupancy, width, height, depth, voxel.x, voxel.y - 1, voxel.z)
  ) {
    neighborCount += 1;
  }
  if (
    isOccupied(occupancy, width, height, depth, voxel.x, voxel.y, voxel.z + 1)
  ) {
    neighborCount += 1;
  }
  if (
    isOccupied(occupancy, width, height, depth, voxel.x, voxel.y, voxel.z - 1)
  ) {
    neighborCount += 1;
  }

  const exposedFaces = 6 - neighborCount;
  return Math.max(
    1 - EDGE_DARKENING_MAX,
    1 - exposedFaces * SNAPSHOT_EDGE_DARKENING_PER_FACE,
  );
}

/**
 * Write a shaded voxel color into the RGBA frame buffer.
 */
function writePixel(
  data: Uint8ClampedArray,
  offset: number,
  voxel: Voxel,
  shade: number,
): void {
  data[offset] = clampByte(voxel.r * shade);
  data[offset + 1] = clampByte(voxel.g * shade);
  data[offset + 2] = clampByte(voxel.b * shade);
  data[offset + 3] = RGBA_OPAQUE_ALPHA;
}

/**
 * Clamp a floating-point color value to an 8-bit unsigned byte.
 */
function clampByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}
