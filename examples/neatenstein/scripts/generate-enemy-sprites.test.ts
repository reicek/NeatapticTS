/**
 * Green-phase tests for the enemy sprite-sheet and reference-snapshot
 * generator (Phase 3 Step 06 slice `06-green`).
 *
 * Validates:
 * - PNG encode/decode round-trip.
 * - 8-direction × 4-state sprite atlas generation and manifest.
 * - Deterministic output for identical generation parameters.
 * - 192×192 front/back/left/right reference snapshots.
 * - Perceptual parity with the approved `plans/robot-proposal-192-*.png`
 *   art targets.
 * - Back-view mirror rule: right-arm cannon appears on the viewer's left.
 */

import { existsSync, mkdirSync, readFileSync, rmSync } from 'fs';
import { resolve } from 'path';
import { deflateSync } from 'zlib';
import {
  compareSnapshotBuffers,
  decodePng,
  DEFAULT_GENERATED_DIR,
  ENEMY_REFERENCE_SIZE,
  ENEMY_SPRITE_DIRECTIONS,
  ENEMY_SPRITE_FRAME_SIZE,
  ENEMY_SPRITE_STATES,
  encodePng,
  generateEnemyReferenceSnapshots,
  generateEnemySpriteSheet,
  makePngChunk,
  type SpriteSheetOptions,
} from './generate-enemy-sprites';
import { ENEMY_ANIMATION_FRAME_COUNTS } from './enemy-animator';

const PLANS_DIR = 'plans';
const REFERENCE_FILES: Record<string, string> = {
  front: 'robot-proposal-192.png',
  back: 'robot-proposal-192-back.png',
  left: 'robot-proposal-192-left.png',
  right: 'robot-proposal-192-right.png',
};

/**
 * Clean a directory and recreate it so generator tests always exercise the
 * `mkdirSync` branch.
 */
function resetGeneratedDir(): void {
  const dir = resolve(DEFAULT_GENERATED_DIR);
  if (existsSync(dir)) {
    rmSync(dir, { recursive: true, force: true });
  }
  mkdirSync(dir, { recursive: true });
}

const PNG_SIGNATURE = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);

/**
 * Build a minimal valid PNG with per-row filter bytes so we can exercise
 * every PNG filter path (Sub, Up, Average, Paeth) without an external lib.
 */
function buildFilteredPng(
  width: number,
  height: number,
  pixels: number[][][],
  rowFilters: number[],
): Buffer {
  const bpp = 4;
  const stride = width * bpp;
  const raw = Buffer.alloc(height * (stride + 1));
  const reconstructed = Buffer.alloc(width * height * bpp);

  function paeth(a: number, b: number, c: number): number {
    const p = a + b - c;
    const pa = Math.abs(p - a);
    const pb = Math.abs(p - b);
    const pc = Math.abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
  }

  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = rowFilters[y];
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < bpp; c++) {
        const actual = pixels[y][x][c];
        const idx = y * stride + x * bpp + c;
        const left = x > 0 ? reconstructed[idx - bpp] : 0;
        const up = y > 0 ? reconstructed[idx - stride] : 0;
        const upLeft = y > 0 && x > 0 ? reconstructed[idx - stride - bpp] : 0;

        let prediction = 0;
        const filter = rowFilters[y];
        if (filter === 1) {
          prediction = left;
        } else if (filter === 2) {
          prediction = up;
        } else if (filter === 3) {
          prediction = Math.floor((left + up) / 2);
        } else if (filter === 4) {
          prediction = paeth(left, up, upLeft);
        }

        raw[y * (stride + 1) + 1 + x * bpp + c] = (actual - prediction) & 0xff;
        reconstructed[idx] = actual;
      }
    }
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  return Buffer.concat([
    PNG_SIGNATURE,
    makePngChunk('IHDR', ihdr),
    makePngChunk('IDAT', deflateSync(raw, { level: 9 })),
    makePngChunk('IEND', Buffer.alloc(0)),
  ]);
}

describe('PNG encoder/decoder', () => {
  it('round-trips a small RGBA image losslessly', () => {
    const width = 4;
    const height = 3;
    const rgba = new Uint8Array(width * height * 4);
    for (let i = 0; i < rgba.length; i += 4) {
      rgba[i] = (i * 7) & 0xff;
      rgba[i + 1] = (i * 13) & 0xff;
      rgba[i + 2] = (i * 23) & 0xff;
      rgba[i + 3] = i % 8 === 0 ? 0 : 255;
    }

    const png = encodePng(width, height, rgba);
    const decoded = decodePng(Buffer.from(png));

    expect(decoded.width).toBe(width);
    expect(decoded.height).toBe(height);
    for (let i = 0; i < rgba.length; i++) {
      expect(decoded.data[i]).toBe(rgba[i]);
    }
  });

  it('throws when the input is not a PNG', () => {
    expect(() => decodePng(Buffer.from('not a png'))).toThrow('not a PNG');
  });

  it('throws when the RGBA buffer size does not match dimensions', () => {
    expect(() => encodePng(2, 2, new Uint8Array(7))).toThrow(
      'RGBA buffer length 7 does not match 2x2x4',
    );
  });

  it('decodes PNGs that use every filter type', () => {
    // 7 rows × 3 columns exercising all filters and both sides of every
    // intra-row branch (first/left pixel, first/current row).
    const pixels = Array.from({ length: 8 }, (_, y) =>
      Array.from({ length: 3 }, (_, x) => {
        const base = (y * 3 + x) * 7 + 10;
        return [base & 0xff, (base + 50) & 0xff, (base + 100) & 0xff, 255];
      }),
    );
    const filters = [3, 4, 2, 2, 3, 4, 1, 0];
    const png = buildFilteredPng(3, 8, pixels, filters);
    const decoded = decodePng(png);

    expect(decoded.width).toBe(3);
    expect(decoded.height).toBe(8);
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 3; x++) {
        const i = (y * 3 + x) * 4;
        expect(decoded.data[i]).toBe(pixels[y][x][0]);
        expect(decoded.data[i + 1]).toBe(pixels[y][x][1]);
        expect(decoded.data[i + 2]).toBe(pixels[y][x][2]);
        expect(decoded.data[i + 3]).toBe(255);
      }
    }
  });

  it('throws when a PNG has no IDAT chunks', () => {
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(1, 0);
    ihdr.writeUInt32BE(1, 4);
    ihdr[8] = 8;
    ihdr[9] = 6;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    const png = Buffer.concat([
      PNG_SIGNATURE,
      makePngChunk('IHDR', ihdr),
      makePngChunk('IEND', Buffer.alloc(0)),
    ]);
    expect(() => decodePng(png)).toThrow('PNG has no IDAT chunks');
  });

  it('throws for unsupported PNG bit depths or color types', () => {
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(1, 0);
    ihdr.writeUInt32BE(1, 4);
    ihdr[8] = 16; // unsupported bit depth
    ihdr[9] = 6;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    const idat = deflateSync(Buffer.from([0, 0, 0, 0, 0, 0]), { level: 9 });
    const png = Buffer.concat([
      PNG_SIGNATURE,
      makePngChunk('IHDR', ihdr),
      makePngChunk('IDAT', idat),
      makePngChunk('IEND', Buffer.alloc(0)),
    ]);
    expect(() => decodePng(png)).toThrow(
      'Unsupported PNG format: bitDepth=16 colorType=6',
    );
  });

  it('decodes an Up-filtered top row', () => {
    const pixels = [
      [
        [11, 21, 31, 255],
        [41, 51, 61, 255],
        [71, 81, 91, 255],
      ],
    ];
    const png = buildFilteredPng(3, 1, pixels, [2]);
    const decoded = decodePng(png);
    expect(decoded.width).toBe(3);
    expect(decoded.height).toBe(1);
    for (let i = 0; i < 12; i++) {
      expect(decoded.data[i]).toBe(pixels[0][Math.floor(i / 4)][i % 4]);
    }
  });

  it('decodes a Paeth-filtered top row', () => {
    const pixels = [
      [
        [10, 20, 30, 255],
        [40, 50, 60, 255],
        [70, 80, 90, 255],
      ],
    ];
    const png = buildFilteredPng(3, 1, pixels, [4]);
    const decoded = decodePng(png);
    expect(decoded.width).toBe(3);
    expect(decoded.height).toBe(1);
    for (let i = 0; i < 12; i++) {
      expect(decoded.data[i]).toBe(pixels[0][Math.floor(i / 4)][i % 4]);
    }
  });

  it('ignores ancillary chunks that appear before IEND', () => {
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(1, 0);
    ihdr.writeUInt32BE(1, 4);
    ihdr[8] = 8;
    ihdr[9] = 6;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    const idat = deflateSync(Buffer.from([0, 0, 0, 0, 0, 0]), { level: 9 });
    const png = Buffer.concat([
      PNG_SIGNATURE,
      makePngChunk('IHDR', ihdr),
      makePngChunk('IDAT', idat),
      makePngChunk('tEXt', Buffer.from('hello', 'ascii')),
      makePngChunk('IEND', Buffer.alloc(0)),
    ]);
    const decoded = decodePng(png);
    expect(decoded.width).toBe(1);
    expect(decoded.height).toBe(1);
  });
});

describe('compareSnapshotBuffers', () => {
  it('throws when the generated and reference images differ in size', () => {
    const small = encodePng(2, 2, new Uint8Array(2 * 2 * 4).fill(255));
    const large = encodePng(4, 4, new Uint8Array(4 * 4 * 4).fill(255));
    expect(() => compareSnapshotBuffers(small, large)).toThrow(
      'Size mismatch: generated 2x2 vs reference 4x4',
    );
  });

  it('returns zero scores for fully transparent images', () => {
    const transparent = encodePng(4, 4, new Uint8Array(4 * 4 * 4).fill(0));
    const result = compareSnapshotBuffers(transparent, transparent);
    expect(result.iou).toBe(0);
    expect(result.colorSimilarity).toBe(0);
    expect(result.generatedOpaque).toBe(0);
    expect(result.referenceOpaque).toBe(0);
  });
});

describe('generateEnemySpriteSheet', () => {
  beforeAll(resetGeneratedDir);
  afterAll(() => {
    const dir = resolve(DEFAULT_GENERATED_DIR);
    if (existsSync(dir)) {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('writes a combined atlas and manifest with all 8×4×N frames', () => {
    const result = generateEnemySpriteSheet();
    const atlasBuf = readFileSync(result.atlasPath);
    const decoded = decodePng(atlasBuf);

    const expectedFrames =
      ENEMY_SPRITE_DIRECTIONS *
      ENEMY_SPRITE_STATES.reduce(
        (sum, state) => sum + ENEMY_ANIMATION_FRAME_COUNTS[state],
        0,
      );
    expect(result.frames.length).toBe(expectedFrames);

    const maxFrameCount = Math.max(
      ...ENEMY_SPRITE_STATES.map(
        (state) => ENEMY_ANIMATION_FRAME_COUNTS[state],
      ),
    );
    expect(decoded.width).toBe(maxFrameCount * ENEMY_SPRITE_FRAME_SIZE);
    expect(decoded.height).toBe(
      ENEMY_SPRITE_DIRECTIONS *
        ENEMY_SPRITE_STATES.length *
        ENEMY_SPRITE_FRAME_SIZE,
    );
    expect(result.width).toBe(decoded.width);
    expect(result.height).toBe(decoded.height);

    const manifest = JSON.parse(readFileSync(result.manifestPath, 'utf8'));
    expect(manifest.frameSize).toBe(ENEMY_SPRITE_FRAME_SIZE);
    expect(manifest.directions).toBe(ENEMY_SPRITE_DIRECTIONS);
    expect(manifest.states.idle).toBe(6);
    expect(manifest.states.move).toBe(12);
    expect(manifest.states.fire).toBe(3);
    expect(manifest.states.death).toBe(12);
    expect(manifest.frames.length).toBe(expectedFrames);

    let nonTransparent = 0;
    for (let i = 3; i < decoded.data.length; i += 4) {
      if (decoded.data[i] > 0) nonTransparent++;
    }
    expect(nonTransparent).toBeGreaterThan(0);
  });

  it('produces deterministic atlas bytes across runs', () => {
    const first = generateEnemySpriteSheet('#DD2200');
    const firstBytes = readFileSync(first.atlasPath);

    const second = generateEnemySpriteSheet('#DD2200');
    const secondBytes = readFileSync(second.atlasPath);

    expect(secondBytes.equals(firstBytes)).toBe(true);
  });

  it('honors a custom output directory', () => {
    const tempDir = resolve(DEFAULT_GENERATED_DIR, 'custom-sheet');
    rmSync(tempDir, { recursive: true, force: true });

    const options: SpriteSheetOptions = { outputDir: tempDir };
    const result = generateEnemySpriteSheet(undefined, options);

    expect(result.atlasPath.startsWith(tempDir)).toBe(true);
    expect(existsSync(result.atlasPath)).toBe(true);
    expect(existsSync(result.manifestPath)).toBe(true);

    rmSync(tempDir, { recursive: true, force: true });
  });
});

describe('generateEnemyReferenceSnapshots', () => {
  beforeAll(resetGeneratedDir);
  afterAll(() => {
    const dir = resolve(DEFAULT_GENERATED_DIR);
    if (existsSync(dir)) {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('writes four 192×192 PNGs for front/back/left/right', () => {
    const paths = generateEnemyReferenceSnapshots();
    for (const [, filePath] of Object.entries(paths)) {
      expect(existsSync(filePath)).toBe(true);
      const decoded = decodePng(readFileSync(filePath));
      expect(decoded.width).toBe(ENEMY_REFERENCE_SIZE);
      expect(decoded.height).toBe(ENEMY_REFERENCE_SIZE);

      let opaque = 0;
      for (let i = 3; i < decoded.data.length; i += 4) {
        if (decoded.data[i] > 0) opaque++;
      }
      expect(opaque).toBeGreaterThan(0);
    }
  });

  it('produces deterministic reference bytes across runs', () => {
    const first = generateEnemyReferenceSnapshots('#DD2200');
    const firstBytes = {
      front: readFileSync(first.front),
      back: readFileSync(first.back),
      left: readFileSync(first.left),
      right: readFileSync(first.right),
    };

    const second = generateEnemyReferenceSnapshots('#DD2200');
    expect(readFileSync(second.front).equals(firstBytes.front)).toBe(true);
    expect(readFileSync(second.back).equals(firstBytes.back)).toBe(true);
    expect(readFileSync(second.left).equals(firstBytes.left)).toBe(true);
    expect(readFileSync(second.right).equals(firstBytes.right)).toBe(true);
  });
});

describe('reference parity and back-view mirror rule', () => {
  beforeAll(() => {
    resetGeneratedDir();
    generateEnemyReferenceSnapshots();
  });

  afterAll(() => {
    const dir = resolve(DEFAULT_GENERATED_DIR);
    if (existsSync(dir)) {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('front/back/left/right snapshots resemble the approved references', () => {
    // Thresholds are deliberately loose. The approved `plans/robot-proposal-192-*.png`
    // files are stylized 6-color art targets, while the generator renders a shaded
    // voxel silhouette. We therefore measure silhouette overlap (IoU) and a
    // color-class overlap after quantizing both images to the reference palette.
    const iouThreshold = 0.15;
    const colorThreshold = 0.2;

    const generatedPaths = {
      front: resolve(DEFAULT_GENERATED_DIR, 'enemy-front.png'),
      back: resolve(DEFAULT_GENERATED_DIR, 'enemy-back.png'),
      left: resolve(DEFAULT_GENERATED_DIR, 'enemy-left.png'),
      right: resolve(DEFAULT_GENERATED_DIR, 'enemy-right.png'),
    };

    for (const [view, generatedPath] of Object.entries(generatedPaths)) {
      const referencePath = resolve(PLANS_DIR, REFERENCE_FILES[view]);
      const comparison = compareSnapshotBuffers(
        readFileSync(generatedPath),
        readFileSync(referencePath),
      );

      expect(comparison.iou).toBeGreaterThanOrEqual(iouThreshold);
      expect(comparison.colorSimilarity).toBeGreaterThanOrEqual(colorThreshold);
      expect(comparison.generatedOpaque).toBeGreaterThan(0);
      expect(comparison.referenceOpaque).toBeGreaterThan(0);
    }
  });

  it("places the cannon on the viewer's left in the back view", () => {
    const backPath = resolve(DEFAULT_GENERATED_DIR, 'enemy-back.png');
    const back = decodePng(readFileSync(backPath));
    const { width, data } = back;

    let sumX = 0;
    let count = 0;
    for (let y = 0; y < back.height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        // Accent/damage voxels are reddish; use a broad red gate to catch the
        // cannon regardless of shading.
        if (data[i + 3] > 0 && data[i] > 180 && data[i + 1] < 120) {
          sumX += x;
          count++;
        }
      }
    }

    expect(count).toBeGreaterThan(0);
    const centroidX = sumX / count;
    expect(centroidX).toBeLessThan(width / 2);
  });
});
