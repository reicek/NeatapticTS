/* global Buffer:readonly, console:readonly */

/**
 * Generate robot sprite PNGs from robot-sprite-data.js arrays.
 *
 * Renders every direction × pose as a 192×192 PNG (48×48 logical grid ×
 * ROBOT_SPRITE_SCALE, nearest-neighbor scaling) plus "shoot composite" frames
 * that combine the upper body of the shoot pose with the lower body of each
 * walk-cycle pose, cropped at row 35 — matching the preview HTML's tick()
 * animation logic.
 *
 * Uses a dependency-free zlib-based PNG encoder (same pattern as
 * generate-enemy-sprites.ts) so no native or third-party image libraries are
 * required.
 *
 * Usage:  node examples/neatenstein/scripts/generate-robot-pngs.mjs
 *
 * @author 04-implementing
 * @see     examples/neatenstein/robot-sprite-preview.html — reference renderer
 * @see     examples/neatenstein/scripts/generate-enemy-sprites.ts — PNG encoder pattern
 */

import { deflateSync } from 'node:zlib';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  ROBOT_SPRITE_SCALE,
  ROBOT_SPRITE_PALETTE,
  ROBOT_SPRITE_FRAMES,
} from '../robot-sprite-data.js';

// ── Constants ────────────────────────────────────────────────────────────

const DIRECTIONS = [
  'front',
  'frontRight',
  'right',
  'backRight',
  'back',
  'backLeft',
  'left',
  'frontLeft',
];

const POSES = ['stand', 'walk1', 'walk2', 'shoot'];

/** Walk-cycle order used by the preview HTML's tick() animation. */
const WALK_CYCLE = ['stand', 'walk1', 'stand', 'walk2'];

/** Row at which the upper/lower body split occurs for shoot composites. */
const SPLIT_ROW = 35;

/** Default team accent color (Ares Red). */
const DEFAULT_ACCENT = '#DD2200';

// ── PNG Encoder (dependency-free, zlib deflate) ───────────────────────────

const PNG_SIGNATURE = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);

/** CRC-32 lookup table for self-contained PNG chunk encoding. */
const CRC_TABLE = (() => {
  const table = new Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) !== 0 ? (0xedb88320 ^ (c >>> 1)) >>> 0 : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

/**
 * Update a running CRC-32 with a byte buffer.
 *
 * @param {number} crc - Initial CRC value (use 0 for a fresh chunk).
 * @param {Uint8Array} data - Bytes to fold into the CRC.
 * @returns {number} Updated CRC-32 as an unsigned 32-bit integer.
 */
function updateCrc(crc, data) {
  let c = (crc ^ 0xffffffff) >>> 0;
  for (const b of data) {
    c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

/**
 * Assemble a single PNG chunk with length, type, data, and CRC-32 fields.
 *
 * @param {string} type - 4-character chunk type (e.g. 'IHDR').
 * @param {Buffer} data - Chunk payload.
 * @returns {Buffer} Complete chunk buffer.
 */
function makePngChunk(type, data) {
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
 * Encode a flat RGBA buffer as an 8-bit PNG image using zlib deflate.
 *
 * Uses 8-bit RGBA (`colorType=6`) with per-row filter byte `0` and zlib
 * deflate compression. The resulting buffer is deterministic for the same
 * input dimensions and pixel data.
 *
 * @param {number} width - Image width in pixels.
 * @param {number} height - Image height in pixels.
 * @param {Uint8Array} rgba - Flat RGBA buffer in row-major order.
 * @returns {Buffer} A complete PNG file as a Node.js Buffer.
 */
function encodePng(width, height, rgba) {
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

// ── Color & Palette Helpers ──────────────────────────────────────────────

/**
 * Parse a hex color string (#RRGGBB) into an [R, G, B, A] tuple.
 *
 * @param {string} hex - Hex color string (e.g. '#DD2200').
 * @returns {number[]} Array of [r, g, b, a] values (a is always 255).
 */
function parseColor(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 255];
}

/**
 * Build a remapped palette where accent indices (5, 6, 7) use the team
 * color's RGB values but retain their original palette alpha. All other
 * indices use the palette as-is.
 *
 * @param {string} [accentHex] - Team accent color as a hex string.
 * @returns {number[][]} Remapped palette array of [R, G, B, A] tuples.
 */
function buildRemappedPalette(accentHex = DEFAULT_ACCENT) {
  const team = parseColor(accentHex);
  return ROBOT_SPRITE_PALETTE.map((rgba, i) => {
    if (i === 5 || i === 6 || i === 7) {
      return [team[0], team[1], team[2], rgba[3]];
    }
    return rgba;
  });
}

// ── Frame Rendering ──────────────────────────────────────────────────────

/**
 * Render a 48×48 logical frame (array of palette indices) into a 192×192
 * RGBA buffer using nearest-neighbor scaling.
 *
 * Palette index 0 is transparent and is skipped so those pixels remain
 * fully transparent (the buffer is zero-initialized).
 *
 * @param {number[][]} frame - 48×48 array of palette indices.
 * @param {number[][]} palette - Remapped palette of [R, G, B, A] tuples.
 * @returns {Uint8Array} Flat RGBA buffer (192 × 192 × 4 bytes).
 */
function renderFrameToRgba(frame, palette) {
  const scaled = ROBOT_SPRITE_SCALE;
  const outWidth = 48 * scaled; // 192
  const outHeight = 48 * scaled; // 192
  const rgba = new Uint8Array(outWidth * outHeight * 4);

  for (let y = 0; y < frame.length; y++) {
    const row = frame[y];
    for (let x = 0; x < row.length; x++) {
      const idx = row[x];
      if (idx === 0) continue; // transparent
      const [r, g, b, a] = palette[idx];
      // Fill the scale×scale block for this logical pixel
      for (let dy = 0; dy < scaled; dy++) {
        for (let dx = 0; dx < scaled; dx++) {
          const px = (x * scaled + dx) * 4;
          const py = (y * scaled + dy) * 4;
          const offset = py * outWidth + px;
          rgba[offset] = r;
          rgba[offset + 1] = g;
          rgba[offset + 2] = b;
          rgba[offset + 3] = a;
        }
      }
    }
  }

  return rgba;
}

/**
 * Build a composite 48×48 frame combining the upper body (rows 0–34) of the
 * shoot pose with the lower body (rows 35–47) of a walk pose, matching the
 * preview HTML's tick() logic.
 *
 * @param {number[][]} shootFrame - 48×48 shoot pose frame.
 * @param {number[][]} walkFrame - 48×48 walk pose frame.
 * @returns {number[][]} Composite 48×48 frame.
 */
function buildShootComposite(shootFrame, walkFrame) {
  const composite = [];
  for (let y = 0; y < 48; y++) {
    if (y < SPLIT_ROW) {
      composite.push(shootFrame[y]);
    } else {
      composite.push(walkFrame[y]);
    }
  }
  return composite;
}

// ── Main ─────────────────────────────────────────────────────────────────

/**
 * Generate all robot sprite PNGs and write them to the output directory.
 */
function main() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = dirname(__filename);
  const outDir = resolve(__dirname, '..', 'generated');
  mkdirSync(outDir, { recursive: true });

  const palette = buildRemappedPalette(DEFAULT_ACCENT);
  let count = 0;

  // Render all direction × pose frames
  for (const dir of DIRECTIONS) {
    const dirFrames = ROBOT_SPRITE_FRAMES[dir];
    if (!dirFrames) {
      throw new Error(`Direction "${dir}" not found in ROBOT_SPRITE_FRAMES`);
    }
    for (const pose of POSES) {
      const frame = dirFrames[pose];
      if (!frame) {
        throw new Error(`Pose "${pose}" not found for direction "${dir}"`);
      }
      const rgba = renderFrameToRgba(frame, palette);
      const png = encodePng(192, 192, rgba);
      const filename = `robot-${dir}-${pose}.png`;
      writeFileSync(resolve(outDir, filename), png);
      count++;
    }
  }

  // Render shoot composite frames (shoot upper + walk lower)
  // Unique lower-body poses from the walk cycle: stand, walk1, walk2
  const compositePoses = [...new Set(WALK_CYCLE)];
  for (const dir of DIRECTIONS) {
    const dirFrames = ROBOT_SPRITE_FRAMES[dir];
    const shootFrame = dirFrames.shoot;
    for (const lowerPose of compositePoses) {
      const walkFrame = dirFrames[lowerPose];
      const composite = buildShootComposite(shootFrame, walkFrame);
      const rgba = renderFrameToRgba(composite, palette);
      const png = encodePng(192, 192, rgba);
      const filename = `robot-${dir}-shoot-${lowerPose}.png`;
      writeFileSync(resolve(outDir, filename), png);
      count++;
    }
  }

  console.log(
    `Generated ${count} robot sprite PNGs in ${outDir} ` +
      `(accent: ${DEFAULT_ACCENT}, scale: ${ROBOT_SPRITE_SCALE}x)`,
  );
}

main();