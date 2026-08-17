/**
 * Browser-compatible shim for `node:crypto` providing a synchronous SHA-256
 * implementation.
 *
 * The Neatenstein worker bundle pulls in `src/neat/nge-dna/neat.nge-dna.utils.ts`
 * through a static import chain. That file imports `createHash` from
 * `node:crypto`, which crashes at module-load time in the browser because
 * `require("node:crypto")` does not exist. This shim is aliased to `node:crypto`
 * in the esbuild config (`scripts/build-neatenstein.mjs`) so the worker bundle
 * gets a pure-JS SHA-256 implementation instead of a runtime `require` call.
 *
 * Only the subset used by `neat.nge-dna.utils.ts` is implemented:
 * `createHash('sha256')` returns a chainable hash object with `.update(data)`
 * and `.digest('hex')`.
 *
 * The implementation is a compact, self-contained SHA-256 written from the
 * FIPS 180-4 specification. It is synchronous and has no dependencies.
 */

// SHA-256 round constants (first 32 bits of fractional parts of cube roots of first 64 primes).
const ROUND_CONSTANTS: readonly number[] = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes).
const INITIAL_HASH: readonly number[] = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
  0x1f83d9ab, 0x5be0cd19,
];

/** Mask to keep operations within 32-bit unsigned range. */
const MASK_32 = 0xffffffff;

/** Rotate a 32-bit value right by the given number of bits. */
function rotateRight(value: number, bits: number): number {
  return ((value >>> bits) | (value << (32 - bits))) & MASK_32;
}

/**
 * Encode a UTF-8 string into a byte array.
 *
 * @param text - UTF-8 string to encode.
 * @returns Array of byte values (0–255).
 */
function encodeUtf8(text: string): number[] {
  const bytes: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const code = text.codePointAt(i) as number;
    if (code < 0x80) {
      bytes.push(code);
    } else if (code < 0x800) {
      bytes.push(0xc0 | (code >> 6), 0x80 | (code & 0x3f));
    } else if (code < 0x10000) {
      bytes.push(
        0xe0 | (code >> 12),
        0x80 | ((code >> 6) & 0x3f),
        0x80 | (code & 0x3f),
      );
    } else {
      bytes.push(
        0xf0 | (code >> 18),
        0x80 | ((code >> 12) & 0x3f),
        0x80 | ((code >> 6) & 0x3f),
        0x80 | (code & 0x3f),
      );
      i++; // Skip the low surrogate — codePointAt consumed both units.
    }
  }
  return bytes;
}

/**
 * Compute the SHA-256 digest of a byte array and return it as a lowercase
 * hexadecimal string.
 *
 * @param bytes - Input bytes to hash.
 * @returns Lowercase hex SHA-256 digest.
 */
function sha256Hex(bytes: number[]): string {
  // Pre-processing: append padding bits and length.
  const padded: number[] = [...bytes];
  const originalBitLength = bytes.length * 8;
  padded.push(0x80);
  while (padded.length % 64 !== 56) {
    padded.push(0x00);
  }
  // Append 64-bit big-endian length.
  // JavaScript bitwise ops are 32-bit (shift amounts are taken mod 32),
  // so `>>> 32` is equivalent to `>>> 0`, which would corrupt the high
  // 32 bits. Push literal zeros for the high word, then shift the low word.
  padded.push(0x00, 0x00, 0x00, 0x00);
  for (let i = 3; i >= 0; i--) {
    padded.push((originalBitLength >>> (i * 8)) & 0xff);
  }

  let h0 = INITIAL_HASH[0];
  let h1 = INITIAL_HASH[1];
  let h2 = INITIAL_HASH[2];
  let h3 = INITIAL_HASH[3];
  let h4 = INITIAL_HASH[4];
  let h5 = INITIAL_HASH[5];
  let h6 = INITIAL_HASH[6];
  let h7 = INITIAL_HASH[7];

  const schedule = new Array<number>(64);

  for (let chunk = 0; chunk < padded.length; chunk += 64) {
    // Build the message schedule.
    for (let i = 0; i < 16; i++) {
      const offset = chunk + i * 4;
      schedule[i] =
        ((padded[offset] << 24) |
          (padded[offset + 1] << 16) |
          (padded[offset + 2] << 8) |
          padded[offset + 3]) >>>
        0;
    }
    for (let i = 16; i < 64; i++) {
      const s0 =
        rotateRight(schedule[i - 15], 7) ^
        rotateRight(schedule[i - 15], 18) ^
        (schedule[i - 15] >>> 3);
      const s1 =
        rotateRight(schedule[i - 2], 17) ^
        rotateRight(schedule[i - 2], 19) ^
        (schedule[i - 2] >>> 10);
      schedule[i] =
        ((schedule[i - 16] + s0 + schedule[i - 7] + s1) & MASK_32) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i++) {
      const s1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 =
        ((h + s1 + ch + ROUND_CONSTANTS[i] + schedule[i]) & MASK_32) >>> 0;
      const s0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = ((s0 + maj) & MASK_32) >>> 0;

      h = g;
      g = f;
      f = e;
      e = ((d + temp1) & MASK_32) >>> 0;
      d = c;
      c = b;
      b = a;
      a = ((temp1 + temp2) & MASK_32) >>> 0;
    }

    h0 = ((h0 + a) & MASK_32) >>> 0;
    h1 = ((h1 + b) & MASK_32) >>> 0;
    h2 = ((h2 + c) & MASK_32) >>> 0;
    h3 = ((h3 + d) & MASK_32) >>> 0;
    h4 = ((h4 + e) & MASK_32) >>> 0;
    h5 = ((h5 + f) & MASK_32) >>> 0;
    h6 = ((h6 + g) & MASK_32) >>> 0;
    h7 = ((h7 + h) & MASK_32) >>> 0;
  }

  const hashValues = [h0, h1, h2, h3, h4, h5, h6, h7];
  let hex = '';
  for (const value of hashValues) {
    hex += value.toString(16).padStart(8, '0');
  }
  return hex;
}

import { SHA256_ALGORITHM, SHA256_ENCODING } from './constants';
import type { ShimHash } from './browser-entry.types';

/**
 * Hash object interface matching the Node.js `crypto.Hash` subset.
 *
 * @deprecated Import from `./browser-entry.types` instead. This re-export
 *   preserves the public API for existing consumers.
 */
export type { ShimHash } from './browser-entry.types';

/**
 * Create a SHA-256 hash object compatible with the Node.js `crypto.createHash`
 * subset used by `neat.nge-dna.utils.ts`.
 *
 * @param algorithm - Must be `'sha256'`.
 * @returns A chainable hash object with `.update()` and `.digest('hex')`.
 * @throws Error if an unsupported algorithm is requested.
 */
export function createHash(algorithm: string): ShimHash {
  if (algorithm !== SHA256_ALGORITHM) {
    throw new Error(
      `node-crypto-shim only supports '${SHA256_ALGORITHM}', got '${algorithm}'.`,
    );
  }

  const bytes: number[] = [];

  return {
    update(data: string): ShimHash {
      bytes.push(...encodeUtf8(data));
      return this;
    },
    digest(encoding: 'hex'): string {
      if (encoding !== SHA256_ENCODING) {
        throw new Error(
          `node-crypto-shim only supports '${SHA256_ENCODING}' encoding, got '${encoding}'.`,
        );
      }
      return sha256Hex(bytes);
    },
  };
}
