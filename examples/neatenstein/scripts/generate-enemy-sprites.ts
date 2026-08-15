/**
 * Sprite-sheet and reference-snapshot generator for the Neatenstein enemy.
 *
 * Procedurally builds an 8-direction × 4-state voxel sprite atlas and four
 * 192×192 reference snapshots (front/back/left/right). All output is written
 * to `examples/neatenstein/generated/` as PNGs using a small, dependency-free
 * PNG encoder/decoder backed by Node's built-in `zlib` module.
 */

import { mkdirSync, writeFileSync } from 'fs';
import { resolve } from 'path';
import { buildVoxelEnemy } from './voxel-enemy';
import { renderVoxelSnapshot } from './snapshot-renderer';
import { ENEMY_ANIMATION_FRAME_COUNTS } from './enemy-animator';
import { encodePng } from './generate-enemy-sprites.png.utils';
import {
  buildAnimatedVoxelEnemy,
  buildOccupancyGrid,
} from './generate-enemy-sprites.voxel.utils';

// Re-export public symbols that moved to sibling util files.
export {
  encodePng,
  decodePng,
  makePngChunk,
} from './generate-enemy-sprites.png.utils';
export { compareSnapshotBuffers } from './generate-enemy-sprites.compare.utils';

// Re-export types from the extracted types module.
export type {
  DecodedPng,
  SpriteSheetOptions,
  SpriteSheetResult,
  SpriteFrameDescriptor,
  ReferenceSnapshotResult,
  SnapshotComparison,
} from './generate-enemy-sprites.types';

// Re-export constants from the extracted constants module.
export {
  ENEMY_FRAME_SIZE_PX,
  ENEMY_REFERENCE_SIZE_PX,
  ENEMY_SPRITE_STATES,
  ENEMY_SPRITE_DIRECTIONS,
  NUM_DIRECTIONS,
  DEFAULT_GENERATED_DIR,
} from './generate-enemy-sprites.constants';

// Backward-compatible aliases for the old constant names.
export {
  ENEMY_FRAME_SIZE_PX as ENEMY_SPRITE_FRAME_SIZE,
  ENEMY_REFERENCE_SIZE_PX as ENEMY_REFERENCE_SIZE,
} from './enemy-animator.constants';

// Import types and constants for internal use.
import type {
  SpriteSheetOptions,
  SpriteSheetResult,
  SpriteFrameDescriptor,
  ReferenceSnapshotResult,
} from './generate-enemy-sprites.types';
import {
  ENEMY_FRAME_SIZE_PX,
  ENEMY_REFERENCE_SIZE_PX,
  ENEMY_SPRITE_STATES,
  ENEMY_SPRITE_DIRECTIONS,
  DEFAULT_GENERATED_DIR,
  RGBA_CHANNELS,
} from './generate-enemy-sprites.constants';
import {
  ENEMY_SPRITE_ATLAS_FILENAME,
  ENEMY_SPRITE_MANIFEST_FILENAME,
} from './enemy-sprite.constants';

/**
 * Generate the enemy runtime sprite sheet, JSON manifest, and write both to disk.
 *
 * Produces a single combined atlas PNG containing every 128×128 frame:
 * 8 directions × 4 states × the approved per-state frame count. A JSON
 * manifest records the cell coordinates.
 *
 * @param accentColor - Optional accent override; defaults to Ares Red.
 * @param options - Optional output directory override.
 * @returns Metadata describing the generated atlas and manifest.
 *
 * @example
 * ```ts
 * const result = generateEnemySpriteSheet('#FFAA00', {
 *   outputDir: 'examples/neatenstein/generated',
 * });
 * console.log(result.atlasPath, result.frames.length);
 * ```
 */
export function generateEnemySpriteSheet(
  accentColor?: string,
  options?: SpriteSheetOptions,
): SpriteSheetResult {
  const outputDir = resolve(options?.outputDir ?? DEFAULT_GENERATED_DIR);
  mkdirSync(outputDir, { recursive: true });

  const maxFrameCount = Math.max(
    ...ENEMY_SPRITE_STATES.map((state) => ENEMY_ANIMATION_FRAME_COUNTS[state]),
  );
  const atlasWidth = maxFrameCount * ENEMY_FRAME_SIZE_PX;
  const atlasHeight =
    ENEMY_SPRITE_DIRECTIONS * ENEMY_SPRITE_STATES.length * ENEMY_FRAME_SIZE_PX;
  const atlas = Buffer.alloc(atlasWidth * atlasHeight * RGBA_CHANNELS);

  const frames: SpriteFrameDescriptor[] = [];

  for (let direction = 0; direction < ENEMY_SPRITE_DIRECTIONS; direction++) {
    for (
      let stateIndex = 0;
      stateIndex < ENEMY_SPRITE_STATES.length;
      stateIndex++
    ) {
      const state = ENEMY_SPRITE_STATES[stateIndex];
      const frameCount = ENEMY_ANIMATION_FRAME_COUNTS[state];

      for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        const grid = buildAnimatedVoxelEnemy(accentColor, state, frameIndex);
        const occupancy = buildOccupancyGrid(grid, grid.voxels);
        const snapshot = renderVoxelSnapshot(grid, direction, {
          width: ENEMY_FRAME_SIZE_PX,
          height: ENEMY_FRAME_SIZE_PX,
          occupancy,
        });

        const x = frameIndex * ENEMY_FRAME_SIZE_PX;
        const y =
          (direction * ENEMY_SPRITE_STATES.length + stateIndex) *
          ENEMY_FRAME_SIZE_PX;

        blitRgba(
          atlas,
          atlasWidth,
          snapshot.data,
          ENEMY_FRAME_SIZE_PX,
          ENEMY_FRAME_SIZE_PX,
          x,
          y,
        );

        frames.push({
          direction,
          state,
          frameIndex,
          x,
          y,
          width: ENEMY_FRAME_SIZE_PX,
          height: ENEMY_FRAME_SIZE_PX,
        });
      }
    }
  }

  const atlasPath = resolve(outputDir, ENEMY_SPRITE_ATLAS_FILENAME);
  const manifestPath = resolve(outputDir, ENEMY_SPRITE_MANIFEST_FILENAME);

  writeFileSync(atlasPath, encodePng(atlasWidth, atlasHeight, atlas));
  writeFileSync(
    manifestPath,
    JSON.stringify(
      {
        frameSize: ENEMY_FRAME_SIZE_PX,
        directions: ENEMY_SPRITE_DIRECTIONS,
        states: Object.fromEntries(
          ENEMY_SPRITE_STATES.map((state) => [
            state,
            ENEMY_ANIMATION_FRAME_COUNTS[state],
          ]),
        ),
        width: atlasWidth,
        height: atlasHeight,
        frames,
      },
      null,
      2,
    ),
  );

  return {
    atlasPath,
    manifestPath,
    width: atlasWidth,
    height: atlasHeight,
    frames,
  };
}

/**
 * Generate four 192×192 reference snapshots for the enemy and write them to disk.
 *
 * Writes front (yaw 0), right (yaw 2), back (yaw 4), and left (yaw 6)
 * orthographic snapshots. The back view naturally mirrors the right-arm
 * cannon to the viewer's left because the camera is positioned behind the
 * robot.
 *
 * @param accentColor - Optional accent override; defaults to Ares Red.
 * @returns Absolute paths to the four generated PNGs.
 *
 * @example
 * ```ts
 * const paths = generateEnemyReferenceSnapshots('#FFAA00');
 * console.log(paths.front, paths.back);
 * ```
 */
export function generateEnemyReferenceSnapshots(
  accentColor?: string,
): ReferenceSnapshotResult {
  const outputDir = resolve(DEFAULT_GENERATED_DIR);
  mkdirSync(outputDir, { recursive: true });

  const grid = buildVoxelEnemy(accentColor);
  const views: Record<keyof ReferenceSnapshotResult, number> = {
    front: 0,
    right: 2,
    back: 4,
    left: 6,
  };

  const paths = {} as Record<keyof ReferenceSnapshotResult, string>;

  for (const [name, yawIndex] of Object.entries(views) as [
    keyof ReferenceSnapshotResult,
    number,
  ][]) {
    const snapshot = renderVoxelSnapshot(grid, yawIndex, {
      width: ENEMY_REFERENCE_SIZE_PX,
      height: ENEMY_REFERENCE_SIZE_PX,
    });
    const filePath = resolve(outputDir, `enemy-${name}.png`);
    writeFileSync(
      filePath,
      encodePng(
        ENEMY_REFERENCE_SIZE_PX,
        ENEMY_REFERENCE_SIZE_PX,
        snapshot.data,
      ),
    );
    paths[name] = filePath;
  }

  return paths;
}

/**
 * Copy a small RGBA image into a larger destination buffer.
 */
function blitRgba(
  dest: Buffer,
  destWidth: number,
  src: Uint8ClampedArray,
  srcWidth: number,
  srcHeight: number,
  offsetX: number,
  offsetY: number,
): void {
  for (let y = 0; y < srcHeight; y++) {
    for (let x = 0; x < srcWidth; x++) {
      const srcIndex = (y * srcWidth + x) * RGBA_CHANNELS;
      const destIndex =
        ((offsetY + y) * destWidth + (offsetX + x)) * RGBA_CHANNELS;
      dest[destIndex] = src[srcIndex];
      dest[destIndex + 1] = src[srcIndex + 1];
      dest[destIndex + 2] = src[srcIndex + 2];
      dest[destIndex + 3] = src[srcIndex + 3];
    }
  }
}
