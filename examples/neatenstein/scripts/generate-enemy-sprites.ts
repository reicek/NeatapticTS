/**
 * Sprite-sheet and reference-snapshot generator for the Neatenstein enemy.
 *
 * Procedurally builds an 8-direction × 4-state voxel sprite atlas and four
 * 192×192 reference snapshots (front/back/left/right). All output is written
 * to `examples/neatenstein/generated/` as PNGs using a small, dependency-free
 * PNG encoder/decoder backed by Node's built-in `zlib` module.
 */

import { deflateSync, inflateSync } from 'zlib';
import { mkdirSync, writeFileSync } from 'fs';
import { resolve } from 'path';
import { buildVoxelEnemy, type Voxel, type VoxelGrid } from './voxel-enemy';
import { renderVoxelSnapshot } from './snapshot-renderer';
import {
  ENEMY_ANIMATION_FRAME_COUNTS,
  type EnemyAnimationState,
} from './enemy-animator';

/**
 * Pixel size of each runtime sprite frame in the generated atlas.
 */
export const ENEMY_SPRITE_FRAME_SIZE = 128;

/**
 * Width and height in pixels of each generated 192×192 reference snapshot.
 */
export const ENEMY_REFERENCE_SIZE = 192;

/**
 * Default filesystem directory where generated enemy sprite assets are written.
 */
export const DEFAULT_GENERATED_DIR = 'examples/neatenstein/generated';

/**
 * Deterministic animation states included in the runtime enemy sprite atlas.
 */
export const ENEMY_SPRITE_STATES: EnemyAnimationState[] = [
  'idle',
  'move',
  'fire',
  'death',
];

/**
 * Number of equally spaced yaw directions rendered in the sprite sheet.
 */
export const ENEMY_SPRITE_DIRECTIONS = 8;

/**
 * Decoded RGBA image extracted from an 8-bit PNG file by `decodePng`.
 */
export interface DecodedPng {
  /** Image width in pixels. */
  width: number;
  /** Image height in pixels. */
  height: number;
  /** Flat RGBA buffer in row-major order. */
  data: Buffer;
}

/**
 * Configuration options accepted by the enemy sprite sheet generator function.
 */
export interface SpriteSheetOptions {
  /** Override the output directory. */
  outputDir?: string;
}

/**
 * Metadata describing a generated enemy sprite atlas and its JSON manifest.
 */
export interface SpriteSheetResult {
  /** Absolute path to the combined atlas PNG. */
  atlasPath: string;
  /** Absolute path to the JSON manifest. */
  manifestPath: string;
  /** Atlas width in pixels. */
  width: number;
  /** Atlas height in pixels. */
  height: number;
  /** Per-cell frame descriptors. */
  frames: SpriteFrameDescriptor[];
}

/**
 * Describes one 128×128 frame cell inside the combined enemy sprite atlas.
 */
export interface SpriteFrameDescriptor {
  /** Yaw direction index (0–7). */
  direction: number;
  /** Animation state. */
  state: EnemyAnimationState;
  /** Frame index within the state. */
  frameIndex: number;
  /** Horizontal offset in the atlas. */
  x: number;
  /** Vertical offset in the atlas. */
  y: number;
  /** Cell width in pixels. */
  width: number;
  /** Cell height in pixels. */
  height: number;
}

/**
 * Absolute paths to the four generated 192×192 orthographic reference PNGs.
 */
export interface ReferenceSnapshotResult {
  /** Absolute path to the front-view PNG. */
  front: string;
  /** Absolute path to the back-view PNG. */
  back: string;
  /** Absolute path to the left-view PNG. */
  left: string;
  /** Absolute path to the right-view PNG. */
  right: string;
}

/**
 * Perceptual parity scores and opaque-pixel counts returned by the snapshot
 * comparator.
 */
export interface SnapshotComparison {
  /** Silhouette intersection-over-union in [0, 1]. */
  iou: number;
  /** Color-class overlap in [0, 1]. */
  colorSimilarity: number;
  /** Number of opaque pixels in the generated image. */
  generatedOpaque: number;
  /** Number of opaque pixels in the reference image. */
  referenceOpaque: number;
}

const PNG_SIGNATURE = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);

const CRC_TABLE = buildCrcTable();

/**
 * Build a standard PNG CRC-32 lookup table so the encoder stays
 * self-contained and does not depend on newer Node.js `zlib.crc32`.
 */
function buildCrcTable(): number[] {
  const table = new Array<number>(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) !== 0 ? (0xedb88320 ^ (c >>> 1)) >>> 0 : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
}

/**
 * Update a running CRC-32 with a byte buffer.
 */
function updateCrc(crc: number, data: Uint8Array): number {
  let c = (crc ^ 0xffffffff) >>> 0;
  for (const b of data) {
    c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

/**
 * Encode a flat RGBA buffer as an 8-bit PNG image using zlib deflate.
 *
 * Uses 8-bit RGBA (`colorType=6`) with per-row filter byte `0` and zlib
 * deflate compression. The resulting buffer is deterministic for the same
 * input dimensions and pixel data.
 *
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @param rgba - Flat RGBA buffer in row-major order.
 * @returns A complete PNG file as a Node.js Buffer.
 * @see https://www.w3.org/TR/PNG/ — PNG Specification (ISO/IEC 15948)
 */
export function encodePng(
  width: number,
  height: number,
  rgba: Uint8Array | Uint8ClampedArray | Buffer,
): Buffer {
  if (rgba.length !== width * height * 4) {
    throw new Error(
      `RGBA buffer length ${rgba.length} does not match ${width}x${height}x4`,
    );
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA color type
  ihdr[10] = 0; // deflate compression
  ihdr[11] = 0; // adaptive filtering
  ihdr[12] = 0; // no interlace

  const stride = width * 4;
  const raw = Buffer.alloc(height * (stride + 1));
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0; // filter byte: none
    const src = y * stride;
    for (let x = 0; x < stride; x++) {
      raw[y * (stride + 1) + 1 + x] = rgba[src + x];
    }
  }

  const compressed = deflateSync(raw, { level: 9 });
  return Buffer.concat([
    PNG_SIGNATURE,
    makePngChunk('IHDR', ihdr),
    makePngChunk('IDAT', compressed),
    makePngChunk('IEND', Buffer.alloc(0)),
  ]);
}

/**
 * Assemble a single PNG chunk with length, type, data, and CRC-32 fields.
 *
 * Exported so tests can craft synthetic PNGs for filter and error-path
 * coverage without relying on an external PNG library.
 *
 * @internal
 * @see https://www.w3.org/TR/PNG/ — PNG Specification (ISO/IEC 15948)
 */
export function makePngChunk(type: string, data: Buffer): Buffer {
  const typeBuf = Buffer.from(type, 'ascii');
  const chunk = Buffer.concat([typeBuf, data]);
  const crc = updateCrc(0, chunk);
  const out = Buffer.alloc(4 + chunk.length + 4);
  out.writeUInt32BE(data.length, 0);
  chunk.copy(out, 4);
  out.writeUInt32BE(crc, 4 + chunk.length);
  return out;
}

/**
 * Decode an 8-bit RGBA PNG image to a flat, unfiltered RGBA buffer.
 *
 * Supports 8-bit RGBA (`colorType=6`) PNGs with all five filter types and
 * no interlacing. Used by the reference-parity tests to load the approved
 * `plans/robot-proposal-192-*.png` files.
 *
 * @param buffer - PNG file contents.
 * @returns Decoded width, height, and RGBA pixel data.
 * @throws Error when the file is not a supported PNG.
 * @see https://www.w3.org/TR/PNG/ — PNG Specification (ISO/IEC 15948)
 */
export function decodePng(buffer: Buffer): DecodedPng {
  if (buffer.length < 8 || !buffer.subarray(0, 8).equals(PNG_SIGNATURE)) {
    throw new Error('Input is not a PNG file');
  }

  let pos = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  const idatChunks: Buffer[] = [];

  while (pos < buffer.length) {
    const length = buffer.readUInt32BE(pos);
    pos += 4;
    const type = buffer.toString('ascii', pos, pos + 4);
    pos += 4;
    const data = buffer.subarray(pos, pos + length);
    pos += length;
    pos += 4; // CRC

    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
    } else if (type === 'IDAT') {
      idatChunks.push(data);
    } else if (type === 'IEND') {
      break;
    }
  }

  if (idatChunks.length === 0) {
    throw new Error('PNG has no IDAT chunks');
  }
  if (bitDepth !== 8 || colorType !== 6) {
    throw new Error(
      `Unsupported PNG format: bitDepth=${bitDepth} colorType=${colorType}`,
    );
  }

  const compressed = Buffer.concat(idatChunks);
  const inflated = inflateSync(compressed);
  const bpp = 4;
  const stride = width * bpp;
  const out = Buffer.alloc(width * height * bpp);

  for (let y = 0; y < height; y++) {
    const filter = inflated[y * (stride + 1)];
    const row = inflated.subarray(
      y * (stride + 1) + 1,
      y * (stride + 1) + 1 + stride,
    );

    for (let x = 0; x < stride; x++) {
      let raw = row[x];
      if (filter === 1) {
        // Sub
        const left = x >= bpp ? out[y * stride + x - bpp] : 0;
        raw = (raw + left) & 0xff;
      } else if (filter === 2) {
        // Up
        const up = y > 0 ? out[(y - 1) * stride + x] : 0;
        raw = (raw + up) & 0xff;
      } else if (filter === 3) {
        // Average
        const left = x >= bpp ? out[y * stride + x - bpp] : 0;
        const up = y > 0 ? out[(y - 1) * stride + x] : 0;
        raw = (raw + Math.floor((left + up) / 2)) & 0xff;
      } else if (filter === 4) {
        // Paeth
        const left = x >= bpp ? out[y * stride + x - bpp] : 0;
        const up = y > 0 ? out[(y - 1) * stride + x] : 0;
        const upLeft = y > 0 && x >= bpp ? out[(y - 1) * stride + x - bpp] : 0;
        raw = (raw + paethPredictor(left, up, upLeft)) & 0xff;
      }
      out[y * stride + x] = raw;
    }
  }

  return { width, height, data: out };
}

/**
 * Paeth predictor used by PNG filter type 4.
 */
function paethPredictor(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

/**
 * Build a voxel grid for a specific animation state and frame.
 *
 * Deformations are deterministic and based only on `(state, frameIndex)`:
 * - `idle`: subtle whole-body vertical bob and slight breathing.
 * - `move`: alternating leg stride and counter-swinging arms.
 * - `fire`: cannon recoil and a short-lived muzzle flash.
 * - `death`: collapse, darkening, and back-disk flicker.
 *
 * @param accentColor - Optional accent override; defaults to Ares Red.
 * @param state - Animation state.
 * @param frameIndex - Frame index within the state.
 * @returns A deformed voxel grid ready for rendering.
 */
// Cache the heavy base voxel mesh per accent color. This preserves full
// determinism because each animation frame still receives an independent clone
// of the cached base voxels; only the immutable dimensions/palette are reused.
const baseVoxelCache = new Map<string, { grid: VoxelGrid; voxels: Voxel[] }>();

function buildOccupancyGrid(
  grid: VoxelGrid,
  voxels: readonly Voxel[],
): Uint8Array {
  const { width, height, depth } = grid;
  const occupancy = new Uint8Array(width * height * depth);
  // The animated grid is clamped to its bounds before rendering, so every
  // voxel here is guaranteed to be in-bounds.
  for (const voxel of voxels) {
    occupancy[voxel.z * (height * width) + voxel.y * width + voxel.x] = 1;
  }
  return occupancy;
}

function getBaseVoxelEnemy(accentColor: string | undefined): {
  grid: VoxelGrid;
  voxels: Voxel[];
} {
  const key = accentColor ?? 'default';
  let entry = baseVoxelCache.get(key);
  if (!entry) {
    const grid = buildVoxelEnemy(accentColor);
    entry = { grid, voxels: grid.voxels.map((voxel) => ({ ...voxel })) };
    baseVoxelCache.set(key, entry);
  }
  return {
    grid: entry.grid,
    voxels: entry.voxels.map((voxel) => ({ ...voxel })),
  };
}

function buildAnimatedVoxelEnemy(
  accentColor: string | undefined,
  state: EnemyAnimationState,
  frameIndex: number,
): VoxelGrid {
  const base = getBaseVoxelEnemy(accentColor);
  const voxels: Voxel[] = base.voxels.map((voxel) => ({ ...voxel }));

  switch (state) {
    case 'idle': {
      const bob = Math.round(
        2 *
          Math.sin(
            (frameIndex / ENEMY_ANIMATION_FRAME_COUNTS.idle) * 2 * Math.PI,
          ),
      );
      for (const voxel of voxels) {
        voxel.y += bob;
      }
      break;
    }

    case 'move': {
      const stride = Math.sin(
        (frameIndex / ENEMY_ANIMATION_FRAME_COUNTS.move) * 2 * Math.PI,
      );
      for (const voxel of voxels) {
        if (voxel.part === 'legs') {
          voxel.y += Math.round(
            stride * (voxel.x < base.grid.width / 2 ? 2 : -2),
          );
        } else if (voxel.part === 'arms') {
          voxel.y += Math.round(-stride * 2);
        }
      }
      break;
    }

    case 'fire': {
      const recoil =
        1 - frameIndex / Math.max(1, ENEMY_ANIMATION_FRAME_COUNTS.fire - 1);
      for (const voxel of voxels) {
        if (voxel.part === 'cannon') {
          voxel.z -= Math.round(2 * recoil);
          voxel.y += Math.round(1 * recoil);
        }
      }

      if (frameIndex === 0) {
        const cannonVoxels = voxels.filter((voxel) => voxel.part === 'cannon');
        const maxZ = Math.max(...cannonVoxels.map((voxel) => voxel.z));
        const tipVoxels = cannonVoxels.filter((voxel) => voxel.z === maxZ);
        for (const tip of tipVoxels) {
          for (let dz = 1; dz <= 2; dz++) {
            voxels.push({
              ...tip,
              z: tip.z + dz,
              material: 'damage',
              emissive: true,
            });
          }
        }
      }
      break;
    }

    case 'death': {
      const progress =
        frameIndex / Math.max(1, ENEMY_ANIMATION_FRAME_COUNTS.death - 1);
      for (const voxel of voxels) {
        voxel.y = Math.round(voxel.y * (1 - progress * 0.5));
        voxel.y -= Math.round(progress * 20);
        voxel.r = Math.round(voxel.r * (1 - progress * 0.6));
        voxel.g = Math.round(voxel.g * (1 - progress * 0.6));
        voxel.b = Math.round(voxel.b * (1 - progress * 0.6));

        if (voxel.part === 'back disk' && frameIndex % 3 === 0) {
          voxel.emissive = !voxel.emissive;
        }
      }
      break;
    }
  }

  const clamped = voxels.filter(
    (voxel) =>
      voxel.x >= 0 &&
      voxel.x < base.grid.width &&
      voxel.y >= 0 &&
      voxel.y < base.grid.height &&
      voxel.z >= 0 &&
      voxel.z < base.grid.depth,
  );

  return { ...base.grid, voxels: clamped };
}

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
  const atlasWidth = maxFrameCount * ENEMY_SPRITE_FRAME_SIZE;
  const atlasHeight =
    ENEMY_SPRITE_DIRECTIONS *
    ENEMY_SPRITE_STATES.length *
    ENEMY_SPRITE_FRAME_SIZE;
  const atlas = Buffer.alloc(atlasWidth * atlasHeight * 4);

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
          width: ENEMY_SPRITE_FRAME_SIZE,
          height: ENEMY_SPRITE_FRAME_SIZE,
          occupancy,
        });

        const x = frameIndex * ENEMY_SPRITE_FRAME_SIZE;
        const y =
          (direction * ENEMY_SPRITE_STATES.length + stateIndex) *
          ENEMY_SPRITE_FRAME_SIZE;

        blitRgba(
          atlas,
          atlasWidth,
          snapshot.data,
          ENEMY_SPRITE_FRAME_SIZE,
          ENEMY_SPRITE_FRAME_SIZE,
          x,
          y,
        );

        frames.push({
          direction,
          state,
          frameIndex,
          x,
          y,
          width: ENEMY_SPRITE_FRAME_SIZE,
          height: ENEMY_SPRITE_FRAME_SIZE,
        });
      }
    }
  }

  const atlasPath = resolve(outputDir, 'enemy-sprite-atlas.png');
  const manifestPath = resolve(outputDir, 'enemy-sprite-manifest.json');

  writeFileSync(atlasPath, encodePng(atlasWidth, atlasHeight, atlas));
  writeFileSync(
    manifestPath,
    JSON.stringify(
      {
        frameSize: ENEMY_SPRITE_FRAME_SIZE,
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
      width: ENEMY_REFERENCE_SIZE,
      height: ENEMY_REFERENCE_SIZE,
    });
    const filePath = resolve(outputDir, `enemy-${name}.png`);
    writeFileSync(
      filePath,
      encodePng(ENEMY_REFERENCE_SIZE, ENEMY_REFERENCE_SIZE, snapshot.data),
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
      const srcIndex = (y * srcWidth + x) * 4;
      const destIndex = ((offsetY + y) * destWidth + (offsetX + x)) * 4;
      dest[destIndex] = src[srcIndex];
      dest[destIndex + 1] = src[srcIndex + 1];
      dest[destIndex + 2] = src[srcIndex + 2];
      dest[destIndex + 3] = src[srcIndex + 3];
    }
  }
}

/**
 * Compare a generated snapshot PNG against an approved reference PNG.
 *
 * The metric is intentionally loose because the reference images are stylized
 * 192×192 art targets while the runtime pipeline renders a detailed voxel
 * model with shading. Two scores are returned:
 * - **Silhouette IoU**: overlap of opaque pixels.
 * - **Color-class overlap**: each pixel is classified to the nearest dominant
 *   color from the reference, then the per-class counts are compared with a
 *   Jaccard-like `sum(min) / sum(max)` ratio.
 *
 * The current implementation documents the thresholds in the test suite, not
 * in this helper, so callers can choose appropriate passing criteria.
 *
 * @param generatedPng - Generated PNG file contents.
 * @param referencePng - Approved reference PNG file contents.
 * @returns Comparison scores and opaque-pixel counts.
 */
export function compareSnapshotBuffers(
  generatedPng: Buffer,
  referencePng: Buffer,
): SnapshotComparison {
  const generated = decodePng(generatedPng);
  const reference = decodePng(referencePng);

  if (
    generated.width !== reference.width ||
    generated.height !== reference.height
  ) {
    throw new Error(
      `Size mismatch: generated ${generated.width}x${generated.height} vs reference ${reference.width}x${reference.height}`,
    );
  }

  const palette = extractReferencePalette(reference.data);
  const genClasses = classifyImage(generated.data, palette);
  const refClasses = classifyImage(reference.data, palette);

  let bothOpaque = 0;
  let genOpaque = 0;
  let refOpaque = 0;
  let sumMin = 0;
  let sumMax = 0;

  for (let i = 0; i < palette.length; i++) {
    const genCount = genClasses[i];
    const refCount = refClasses[i];
    sumMin += Math.min(genCount, refCount);
    sumMax += Math.max(genCount, refCount);
  }

  for (let i = 0; i < generated.data.length; i += 4) {
    const genO = generated.data[i + 3] > 0;
    const refO = reference.data[i + 3] > 0;
    if (genO) genOpaque++;
    if (refO) refOpaque++;
    if (genO && refO) bothOpaque++;
  }

  const union = genOpaque + refOpaque - bothOpaque;
  const iou = union > 0 ? bothOpaque / union : 0;
  const colorSimilarity = sumMax > 0 ? sumMin / sumMax : 0;

  return {
    iou,
    colorSimilarity,
    generatedOpaque: genOpaque,
    referenceOpaque: refOpaque,
  };
}

/**
 * Extract the dominant opaque RGB colors from a reference image.
 *
 * Returns up to 6 colors sorted by descending frequency.
 */
function extractReferencePalette(
  data: Buffer,
): Array<[number, number, number]> {
  const counts = new Map<string, number>();
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    const key = `${data[i]},${data[i + 1]},${data[i + 2]}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([key]) => {
      const [r, g, b] = key.split(',').map(Number);
      return [r, g, b] as [number, number, number];
    });
}

/**
 * Classify every opaque pixel of an image to the nearest reference-palette
 * color and return per-class counts.
 */
function classifyImage(
  data: Buffer,
  palette: Array<[number, number, number]>,
): number[] {
  const counts = new Array(palette.length).fill(0);
  if (palette.length === 0) return counts;

  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    let best = 0;
    let bestDist = Infinity;
    for (let c = 0; c < palette.length; c++) {
      const [pr, pg, pb] = palette[c];
      const dist =
        (data[i] - pr) ** 2 + (data[i + 1] - pg) ** 2 + (data[i + 2] - pb) ** 2;
      if (dist < bestDist) {
        bestDist = dist;
        best = c;
      }
    }
    counts[best]++;
  }

  return counts;
}
