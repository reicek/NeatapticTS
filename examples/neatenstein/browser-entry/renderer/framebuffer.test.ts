import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_FRAMEBUFFER_CHANNELS,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
  clampInt,
  hasExactNeatensteinFramebufferByteLength,
  isValidNeatensteinFramebufferSize,
  resolveNeatensteinFogFactor,
  resolveNeatensteinFoggedColor,
} from './framebuffer';

/** Canonical valid framebuffer dimensions used across these tests. */
const TEST_FRAMEBUFFER_WIDTH = 4;
/** Canonical valid framebuffer height used across these tests. */
const TEST_FRAMEBUFFER_HEIGHT = 3;
/** Expected byte count for the canonical dimensions at four bytes per pixel. */
const EXPECTED_FRAMEBUFFER_BYTE_LENGTH =
  TEST_FRAMEBUFFER_WIDTH *
  TEST_FRAMEBUFFER_HEIGHT *
  NEATENSTEIN_FRAMEBUFFER_CHANNELS;

describe('Neatenstein framebuffer utilities', () => {
  describe('isValidNeatensteinFramebufferSize', () => {
    it('returns true for positive integer dimensions', () => {
      expect(
        isValidNeatensteinFramebufferSize(
          TEST_FRAMEBUFFER_WIDTH,
          TEST_FRAMEBUFFER_HEIGHT,
        ),
      ).toBe(true);
    });

    it('returns false for zero width', () => {
      expect(
        isValidNeatensteinFramebufferSize(0, TEST_FRAMEBUFFER_HEIGHT),
      ).toBe(false);
    });

    it('returns false for zero height', () => {
      expect(isValidNeatensteinFramebufferSize(TEST_FRAMEBUFFER_WIDTH, 0)).toBe(
        false,
      );
    });

    it('returns false for negative width', () => {
      expect(
        isValidNeatensteinFramebufferSize(
          -TEST_FRAMEBUFFER_WIDTH,
          TEST_FRAMEBUFFER_HEIGHT,
        ),
      ).toBe(false);
    });

    it('returns false for negative height', () => {
      expect(
        isValidNeatensteinFramebufferSize(
          TEST_FRAMEBUFFER_WIDTH,
          -TEST_FRAMEBUFFER_HEIGHT,
        ),
      ).toBe(false);
    });

    it('returns false for non-integer width', () => {
      expect(
        isValidNeatensteinFramebufferSize(4.5, TEST_FRAMEBUFFER_HEIGHT),
      ).toBe(false);
    });

    it('returns false for non-integer height', () => {
      expect(
        isValidNeatensteinFramebufferSize(TEST_FRAMEBUFFER_WIDTH, 3.7),
      ).toBe(false);
    });
  });

  describe('hasExactNeatensteinFramebufferByteLength', () => {
    it('returns true when byte length matches width * height * channels', () => {
      const framebuffer = new Uint8ClampedArray(
        EXPECTED_FRAMEBUFFER_BYTE_LENGTH,
      );

      expect(
        hasExactNeatensteinFramebufferByteLength(
          framebuffer,
          TEST_FRAMEBUFFER_WIDTH,
          TEST_FRAMEBUFFER_HEIGHT,
        ),
      ).toBe(true);
    });

    it('returns false when byte length is too small', () => {
      const framebuffer = new Uint8ClampedArray(
        EXPECTED_FRAMEBUFFER_BYTE_LENGTH - 1,
      );

      expect(
        hasExactNeatensteinFramebufferByteLength(
          framebuffer,
          TEST_FRAMEBUFFER_WIDTH,
          TEST_FRAMEBUFFER_HEIGHT,
        ),
      ).toBe(false);
    });

    it('returns false when byte length is too large', () => {
      const framebuffer = new Uint8ClampedArray(
        EXPECTED_FRAMEBUFFER_BYTE_LENGTH + 1,
      );

      expect(
        hasExactNeatensteinFramebufferByteLength(
          framebuffer,
          TEST_FRAMEBUFFER_WIDTH,
          TEST_FRAMEBUFFER_HEIGHT,
        ),
      ).toBe(false);
    });

    it('returns false for invalid dimensions regardless of byte length', () => {
      const framebuffer = new Uint8ClampedArray(0);

      expect(hasExactNeatensteinFramebufferByteLength(framebuffer, 0, 0)).toBe(
        false,
      );
    });
  });

  describe('clampInt', () => {
    it('clamps values below the minimum to the minimum', () => {
      expect(clampInt(-5, 0, 10)).toBe(0);
    });

    it('clamps values above the maximum to the maximum', () => {
      expect(clampInt(15, 0, 10)).toBe(10);
    });

    it('truncates in-range floating-point values toward zero', () => {
      expect(clampInt(5.9, 0, 10)).toBe(5);
    });

    it('returns already-integer in-range values unchanged', () => {
      expect(clampInt(5, 0, 10)).toBe(5);
    });

    it('clamps NaN to the lower bound', () => {
      expect(clampInt(Number.NaN, 0, 10)).toBe(0);
    });

    it('clamps negative infinity to the lower bound', () => {
      expect(clampInt(Number.NEGATIVE_INFINITY, 0, 10)).toBe(0);
    });

    it('clamps positive infinity to the upper bound', () => {
      expect(clampInt(Number.POSITIVE_INFINITY, 0, 10)).toBe(10);
    });

    it('normalizes reversed bounds', () => {
      expect(clampInt(12, 10, 0)).toBe(10);
    });

    it('treats non-finite minimum as zero when maximum is finite', () => {
      expect(clampInt(-3, Number.NaN, 10)).toBe(0);
    });

    it('treats non-finite maximum as the truncated minimum', () => {
      expect(clampInt(5, 0, Number.POSITIVE_INFINITY)).toBe(0);
    });
  });

  describe('removal contract', () => {
    it('no longer exports resolveNeatensteinFramebufferSize', async () => {
      const module = await import('./framebuffer.ts');

      expect('resolveNeatensteinFramebufferSize' in module).toBe(false);
    });
  });

  describe('resolveNeatensteinFogFactor', () => {
    it('returns 0 at distance 0', () => {
      expect(resolveNeatensteinFogFactor(0)).toBe(0);
    });

    it('returns 1 at the render distance cap', () => {
      expect(resolveNeatensteinFogFactor(NEATENSTEIN_RENDER_DISTANCE_CAP)).toBe(
        1,
      );
    });

    it('returns 1 beyond the render distance cap', () => {
      expect(resolveNeatensteinFogFactor(100)).toBe(1);
    });

    it('returns 1 for non-finite distances', () => {
      expect(resolveNeatensteinFogFactor(Number.NaN)).toBe(1);
      expect(resolveNeatensteinFogFactor(Number.POSITIVE_INFINITY)).toBe(1);
    });

    it('returns 0 below the cap (step function)', () => {
      expect(
        resolveNeatensteinFogFactor(NEATENSTEIN_RENDER_DISTANCE_CAP / 2),
      ).toBe(0);
    });
  });

  describe('resolveNeatensteinFoggedColor', () => {
    it('returns the base color when fog factor is 0', () => {
      const base = { r: 100, g: 200, b: 50 };
      expect(resolveNeatensteinFoggedColor(base, 0)).toEqual(base);
    });

    it('returns the background color when fog factor is 1', () => {
      const result = resolveNeatensteinFoggedColor({ r: 255, g: 0, b: 0 }, 1);
      expect(result).toEqual(NEATENSTEIN_BACKGROUND_RGB);
    });

    it('linearly interpolates at fog factor 0.5', () => {
      const base = { r: 100, g: 200, b: 50 };
      const result = resolveNeatensteinFoggedColor(base, 0.5);
      expect(result.r).toBe(
        Math.round((100 + NEATENSTEIN_BACKGROUND_RGB.r) / 2),
      );
      expect(result.g).toBe(
        Math.round((200 + NEATENSTEIN_BACKGROUND_RGB.g) / 2),
      );
      expect(result.b).toBe(
        Math.round((50 + NEATENSTEIN_BACKGROUND_RGB.b) / 2),
      );
    });
  });
});
