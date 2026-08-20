/**
 * @module generate-enemy-sprites.png.utils
 *
 * Dependency-free PNG codec executors for the enemy sprite generator.
 *
 * Implements a minimal 8-bit RGBA PNG encoder and decoder backed by Node's
 * built-in `zlib` module. Used by the sprite-sheet orchestrator to write atlas
 * PNGs and by the reference-parity tests to load approved snapshot files.
 */

import { deflateSync, inflateSync } from 'zlib';

import type { DecodedPng } from './generate-enemy-sprites.types';
import {
  PNG_CHUNK_IHDR,
  PNG_CHUNK_IDAT,
  PNG_CHUNK_IEND,
  RGBA_CHANNELS,
} from './generate-enemy-sprites.constants';
import {
  PNG_SIGNATURE,
  PNG_CRC_POLYNOMIAL,
  PNG_HASH_TABLE_SIZE,
  PNG_INITIAL_HASH,
  PNG_HASH_MASK,
  PNG_BYTE_MASK,
} from './png.utils.constants';

// Re-export PNG_SIGNATURE so tests that imported it from this module still work.
export { PNG_SIGNATURE } from './png.utils.constants';

/** CRC-32 lookup table, initialised once at module load. */
const CRC_TABLE = buildCrcTable();

/**
 * Build a standard PNG CRC-32 lookup table so the encoder stays
 * self-contained and does not depend on newer Node.js `zlib.crc32`.
 *
 * @returns A 256-entry CRC-32 lookup table.
 */
function buildCrcTable(): number[] {
  const table = new Array<number>(PNG_HASH_TABLE_SIZE);
  for (let n = 0; n < PNG_HASH_TABLE_SIZE; n++) {
    let c = n;
    for (let k = 0; k < PNG_INITIAL_HASH; k++) {
      c = (c & 1) !== 0 ? (PNG_CRC_POLYNOMIAL ^ (c >>> 1)) >>> 0 : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
}

/**
 * Update a running CRC-32 with a byte buffer.
 *
 * @param crc - Starting CRC value.
 * @param data - Bytes to fold into the CRC.
 * @returns Updated CRC-32 value.
 */
function updateCrc(crc: number, data: Uint8Array): number {
  let c = (crc ^ PNG_HASH_MASK) >>> 0;
  for (const b of data) {
    c = CRC_TABLE[(c ^ b) & PNG_BYTE_MASK] ^ (c >>> 8);
  }
  return (c ^ PNG_HASH_MASK) >>> 0;
}

/**
 * Paeth predictor used by PNG filter type 4.
 *
 * @param a - Left pixel.
 * @param b - Above pixel.
 * @param c - Above-left pixel.
 * @returns Predicted value.
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
  if (rgba.length !== width * height * RGBA_CHANNELS) {
    throw new Error(
      `RGBA buffer length ${rgba.length} does not match ${width}x${height}x${RGBA_CHANNELS}`,
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

  const stride = width * RGBA_CHANNELS;
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
    makePngChunk(PNG_CHUNK_IHDR, ihdr),
    makePngChunk(PNG_CHUNK_IDAT, compressed),
    makePngChunk(PNG_CHUNK_IEND, Buffer.alloc(0)),
  ]);
}

/**
 * Assemble a single PNG chunk with length, type, data, and CRC-32 fields.
 *
 * Exported so tests can craft synthetic PNGs for filter and error-path
 * coverage without relying on an external PNG library.
 *
 * @param type - Four-byte chunk type string (e.g. `'IHDR'`).
 * @param data - Chunk payload bytes.
 * @returns A complete PNG chunk buffer.
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
 * `examples/neatenstein/robot-proposal-192-*.png` files.
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

    if (type === PNG_CHUNK_IHDR) {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
    } else if (type === PNG_CHUNK_IDAT) {
      idatChunks.push(data);
    } else if (type === PNG_CHUNK_IEND) {
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
  const bpp = RGBA_CHANNELS;
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
        raw = (raw + left) & PNG_BYTE_MASK;
      } else if (filter === 2) {
        // Up
        const up = y > 0 ? out[(y - 1) * stride + x] : 0;
        raw = (raw + up) & PNG_BYTE_MASK;
      } else if (filter === 3) {
        // Average
        const left = x >= bpp ? out[y * stride + x - bpp] : 0;
        const up = y > 0 ? out[(y - 1) * stride + x] : 0;
        raw = (raw + Math.floor((left + up) / 2)) & PNG_BYTE_MASK;
      } else if (filter === 4) {
        // Paeth
        const left = x >= bpp ? out[y * stride + x - bpp] : 0;
        const up = y > 0 ? out[(y - 1) * stride + x] : 0;
        const upLeft = y > 0 && x >= bpp ? out[(y - 1) * stride + x - bpp] : 0;
        raw = (raw + paethPredictor(left, up, upLeft)) & PNG_BYTE_MASK;
      }
      out[y * stride + x] = raw;
    }
  }

  return { width, height, data: out };
}
