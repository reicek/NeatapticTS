/* eslint-disable @typescript-eslint/no-explicit-any -- dynamic import test helper */
import { describe, expect, it } from '@jest/globals';

/**
 * Red-phase contract tests for the palette-indexed chaingun sprite asset and
 * decoder.
 *
 * Part 1 (raw asset tests) verifies the gun-sprite-data.js exports, grid
 * dimensions, material distribution, angular stepped profile, and firing-frame
 * muzzle-flash burst. These PASS on the existing asset file.
 *
 * Part 2 (decoder tests) verifies the gun-sprite-decode.ts module that does
 * not exist yet. These FAIL with a module-not-found error until 04c-impl-decode
 * creates the decoder.
 */

describe('Neatenstein gun-sprite-data asset', () => {
  describe('AC-04c-001: palette-indexed sprite exports', () => {
    it('exports GUN_SPRITE_SCALE as a positive integer', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      expect(mod.GUN_SPRITE_SCALE).toBeGreaterThan(0);
      expect(Number.isInteger(mod.GUN_SPRITE_SCALE)).toBe(true);
    });

    it('exports GUN_SPRITE_PALETTE as an array of RGBA tuples', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      expect(Array.isArray(mod.GUN_SPRITE_PALETTE)).toBe(true);
      expect(mod.GUN_SPRITE_PALETTE.length).toBeGreaterThanOrEqual(9);
      for (const color of mod.GUN_SPRITE_PALETTE) {
        expect(color).toHaveLength(4);
      }
    });

    it('exports GUN_SPRITE_FRAMES with at least idle and fire keys', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      expect(mod.GUN_SPRITE_FRAMES).toBeDefined();
      expect(mod.GUN_SPRITE_FRAMES.idle).toBeDefined();
      expect(mod.GUN_SPRITE_FRAMES.fire).toBeDefined();
    });
  });

  describe('AC-04c-001: grid dimensions', () => {
    it('idle frame is a 40x24 grid (wide horizontal chaingun)', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const idle = mod.GUN_SPRITE_FRAMES.idle as number[][];
      expect(idle.length).toBe(24);
      for (const row of idle) {
        expect(row.length).toBe(40);
      }
    });

    it('fire frame is a 40x24 grid', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const fire = mod.GUN_SPRITE_FRAMES.fire as number[][];
      expect(fire.length).toBe(24);
      for (const row of fire) {
        expect(row.length).toBe(40);
      }
    });

    it('decoded bounds produce a ~1.6 aspect ratio (width/height)', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const idle = mod.GUN_SPRITE_FRAMES.idle as number[][];
      let minX = 40;
      let maxX = -1;
      let minY = 24;
      let maxY = -1;
      for (let y = 0; y < idle.length; y++) {
        for (let x = 0; x < idle[y].length; x++) {
          if (idle[y][x] !== 0) {
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
          }
        }
      }
      const width = maxX - minX + 1;
      const height = maxY - minY + 1;
      expect(width / height).toBeCloseTo(1.6, 0);
    });
  });

  describe('AC-04c-003: material distribution from palette indices', () => {
    it('upper half (barrel rows 0-11) uses majority metallic/neon-white (index 4)', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const idle = mod.GUN_SPRITE_FRAMES.idle as number[][];
      const upperRows = idle.slice(0, 12);
      let total = 0;
      let metallic = 0;
      for (const row of upperRows) {
        for (const idx of row) {
          if (idx !== 0) {
            total++;
            if (idx === 4 || idx === 8) metallic++;
          }
        }
      }
      expect(total).toBeGreaterThan(0);
      expect(metallic / total).toBeGreaterThan(0.5);
    });

    it('lower half (receiver rows 12-23) uses dark palette indices (1, 2, 3)', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const idle = mod.GUN_SPRITE_FRAMES.idle as number[][];
      const lowerRows = idle.slice(12);
      let darkCount = 0;
      for (const row of lowerRows) {
        for (const idx of row) {
          if (idx === 1 || idx === 2 || idx === 3) darkCount++;
        }
      }
      expect(darkCount).toBeGreaterThan(0);
    });

    it('topmost row has teal muzzle ring (index 5 or 6)', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const idle = mod.GUN_SPRITE_FRAMES.idle as number[][];
      const topRow = idle[0] as number[];
      const hasTeal = topRow.some((idx) => idx === 5 || idx === 6);
      expect(hasTeal).toBe(true);
    });
  });

  describe('AC-04c-004: angular stepped profile', () => {
    it('horizontal half-widths include at least 3 distinct values', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const idle = mod.GUN_SPRITE_FRAMES.idle as number[][];
      const halfWidths: number[] = [];
      for (const row of idle) {
        let minX = 40;
        let maxX = -1;
        for (let x = 0; x < row.length; x++) {
          if (row[x] !== 0) {
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
          }
        }
        if (maxX >= 0) {
          halfWidths.push((maxX - minX) / 2);
        }
      }
      const distinct = new Set(halfWidths);
      expect(distinct.size).toBeGreaterThanOrEqual(3);
    });

    it('has a sharp drop of >=2 cells at a barrel/receiver boundary', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const idle = mod.GUN_SPRITE_FRAMES.idle as number[][];
      const halfWidths: number[] = [];
      for (const row of idle) {
        let minX = 40;
        let maxX = -1;
        for (let x = 0; x < row.length; x++) {
          if (row[x] !== 0) {
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
          }
        }
        if (maxX >= 0) {
          halfWidths.push((maxX - minX) / 2);
        }
      }
      let maxDrop = 0;
      for (let i = 1; i < halfWidths.length; i++) {
        const drop = halfWidths[i - 1] - halfWidths[i];
        if (drop > maxDrop) maxDrop = drop;
      }
      expect(maxDrop).toBeGreaterThanOrEqual(2);
    });
  });

  describe('AC-04c-005: firing-frame muzzle-flash burst', () => {
    it('fire frame has muzzle-flash pixels (index 7) at the top rows', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const fire = mod.GUN_SPRITE_FRAMES.fire as number[][];
      const topRows = fire.slice(0, 4);
      let hasFlash = false;
      for (const row of topRows) {
        if (row.some((idx) => idx === 7)) {
          hasFlash = true;
          break;
        }
      }
      expect(hasFlash).toBe(true);
    });

    it('fire frame has more non-transparent pixels than idle frame', async () => {
      const mod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const idle = mod.GUN_SPRITE_FRAMES.idle as number[][];
      const fire = mod.GUN_SPRITE_FRAMES.fire as number[][];
      let idleCount = 0;
      let fireCount = 0;
      for (const row of idle) {
        for (const idx of row) {
          if (idx !== 0) idleCount++;
        }
      }
      for (const row of fire) {
        for (const idx of row) {
          if (idx !== 0) fireCount++;
        }
      }
      expect(fireCount).toBeGreaterThan(idleCount);
    });
  });
});

/**
 * Decoder contract tests.
 *
 * These tests FAIL because examples/neatenstein/browser-entry/renderer/
 * gun-sprite-decode.ts does not exist yet. Slice 04c-impl-decode will create
 * the module with a decodeGunSpriteFrame function that converts palette-indexed
 * grid rows into a scaled RGBA VoxelSnapshot, with optional palette-swap
 * support for tinting.
 *
 * The dynamic-import path is stored in a variable so TypeScript does not
 * statically resolve the not-yet-existing module (avoids TS2307), following
 * the same pattern as gun-voxel.test.ts.
 */
const DECODE_MODULE = './gun-sprite-decode.ts';

describe('Neatenstein gun sprite decoder', () => {
  describe('AC-04c-011: decodeGunSpriteFrame function', () => {
    it('exports decodeGunSpriteFrame as a function', async () => {
      const mod = (await import(DECODE_MODULE)) as Record<string, any>;
      expect(typeof mod.decodeGunSpriteFrame).toBe('function');
    });
  });

  describe('AC-04c-011: decoded frame dimensions and data', () => {
    it('decoded idle frame has dimensions 40*scale x 24*scale', async () => {
      const decodeMod = (await import(DECODE_MODULE)) as Record<string, any>;
      const dataMod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const scale = dataMod.GUN_SPRITE_SCALE as number;
      const idle = dataMod.GUN_SPRITE_FRAMES.idle as number[][];
      const decoded = decodeMod.decodeGunSpriteFrame(
        idle,
        dataMod.GUN_SPRITE_PALETTE,
      );
      expect(decoded.width).toBe(40 * scale);
      expect(decoded.height).toBe(24 * scale);
    });

    it('decoded frame data is a Uint8ClampedArray of correct length', async () => {
      const decodeMod = (await import(DECODE_MODULE)) as Record<string, any>;
      const dataMod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const scale = dataMod.GUN_SPRITE_SCALE as number;
      const idle = dataMod.GUN_SPRITE_FRAMES.idle as number[][];
      const decoded = decodeMod.decodeGunSpriteFrame(
        idle,
        dataMod.GUN_SPRITE_PALETTE,
      );
      expect(decoded.data).toBeInstanceOf(Uint8ClampedArray);
      expect(decoded.data.length).toBe(40 * scale * 24 * scale * 4);
    });

    it('decoded idle frame maps neon-white palette index 4 to pure white RGBA', async () => {
      const decodeMod = (await import(DECODE_MODULE)) as Record<string, any>;
      const dataMod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const scale = dataMod.GUN_SPRITE_SCALE as number;
      const idle = dataMod.GUN_SPRITE_FRAMES.idle as number[][];
      const decoded = decodeMod.decodeGunSpriteFrame(
        idle,
        dataMod.GUN_SPRITE_PALETTE,
      );
      // Index 4 in the palette is [255, 255, 255, 255] (pure neon white).
      // Find a pixel in the barrel area (row 4, col 19 in the idle grid)
      // that uses index 4 and verify the decoded RGBA matches.
      const pixelX = 19 * scale;
      const pixelY = 4 * scale;
      const offset = (pixelY * decoded.width + pixelX) * 4;
      expect(decoded.data[offset]).toBe(255);
      expect(decoded.data[offset + 1]).toBe(255);
      expect(decoded.data[offset + 2]).toBe(255);
      expect(decoded.data[offset + 3]).toBe(255);
    });
  });

  describe('AC-04c-012: palette swap support', () => {
    it('palette swap produces different RGBA values for swapped indices', async () => {
      const decodeMod = (await import(DECODE_MODULE)) as Record<string, any>;
      const dataMod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const idle = dataMod.GUN_SPRITE_FRAMES.idle as number[][];
      const originalPalette = dataMod.GUN_SPRITE_PALETTE as number[][];
      const swappedPalette = originalPalette.map((color, i) =>
        i === 4 ? ([255, 0, 0, 255] as number[]) : color,
      );
      const originalDecoded = decodeMod.decodeGunSpriteFrame(
        idle,
        originalPalette,
      );
      const swappedDecoded = decodeMod.decodeGunSpriteFrame(
        idle,
        swappedPalette,
      );
      let differs = false;
      for (let i = 0; i < originalDecoded.data.length; i += 4) {
        if (
          originalDecoded.data[i] !== swappedDecoded.data[i] ||
          originalDecoded.data[i + 1] !== swappedDecoded.data[i + 1] ||
          originalDecoded.data[i + 2] !== swappedDecoded.data[i + 2]
        ) {
          differs = true;
          break;
        }
      }
      expect(differs).toBe(true);
    });
  });

  describe('AC-04c-013: buildGunAccentPalette accent color swap', () => {
    it('exports buildGunAccentPalette as a function', async () => {
      const mod = (await import(DECODE_MODULE)) as Record<string, any>;
      expect(typeof mod.buildGunAccentPalette).toBe('function');
    });

    it('replaces RGB of indices 5 and 6 with accent color, preserving alpha', async () => {
      const decodeMod = (await import(DECODE_MODULE)) as Record<string, any>;
      const accentPalette = decodeMod.buildGunAccentPalette([
        255, 0, 128,
      ]) as number[][];

      // Index 5: original [0, 240, 255, 255] → accent [255, 0, 128, 255]
      expect(accentPalette[5]).toEqual([255, 0, 128, 255]);
      // Index 6: original [0, 200, 220, 255] → accent [255, 0, 128, 255]
      expect(accentPalette[6]).toEqual([255, 0, 128, 255]);
    });

    it('leaves all non-accent palette entries unchanged (incl. index 7 alpha)', async () => {
      const decodeMod = (await import(DECODE_MODULE)) as Record<string, any>;
      const dataMod = (await import('../../gun-sprite-data.js')) as Record<
        string,
        any
      >;
      const originalPalette = dataMod.GUN_SPRITE_PALETTE as number[][];
      const accentPalette = decodeMod.buildGunAccentPalette([
        100, 200, 50,
      ]) as number[][];

      // Exclude indices 5 and 6 (intentionally swapped); verify all others match
      const originalFiltered = originalPalette.filter(
        (_, i) => i !== 5 && i !== 6,
      );
      const accentFiltered = accentPalette.filter((_, i) => i !== 5 && i !== 6);
      expect(accentFiltered).toEqual(originalFiltered);
    });
  });
});
