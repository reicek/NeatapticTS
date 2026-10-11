/**
 * @file Red-first tests for the neatRacing fixed-canvas strategy.
 *
 * Locks the Phase 1 canvas contract (AC-001 / spec §6.2): a fixed 720-pixel
 * backing-store height, a proportional backing width clamped at 1920 pixels
 * for ultrawide displays, viewport fallbacks when client dimensions are zero,
 * and letterbox math that keeps the rendered box inside the viewport bounds
 * with centered, non-negative offsets (no page scroll) at odd viewport sizes.
 *
 * The tests exercise pure functions with injectable dimensions so they run in
 * the default node test environment without DOM globals.
 */
import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_FALLBACK_CANVAS_WIDTH,
  NEATRACING_FIXED_RENDER_HEIGHT,
  NEATRACING_MAX_BACKING_WIDTH,
} from './constants';
import {
  applyCanvasBackingStore,
  resolveCanvasBackingDimensions,
  resolveLetterboxLayout,
  updateRendererSize,
} from './canvas-dimensions.utils';

describe('neatRacing canvas constants', () => {
  it('exports the fixed 720-pixel backing-store render height', () => {
    expect(NEATRACING_FIXED_RENDER_HEIGHT).toBe(720);
  });

  it('exports the 1920-pixel maximum backing-store width clamp', () => {
    expect(NEATRACING_MAX_BACKING_WIDTH).toBe(1920);
  });

  it('exports the 1280-pixel fallback canvas width for the final ladder step', () => {
    expect(NEATRACING_FALLBACK_CANVAS_WIDTH).toBe(1280);
  });
});

describe('resolveCanvasBackingDimensions', () => {
  it('resolves a proportional backing width from client aspect at the fixed height', () => {
    expect(
      resolveCanvasBackingDimensions({
        clientWidth: 1920,
        clientHeight: 1080,
        viewportWidth: 0,
        viewportHeight: 0,
      }),
    ).toEqual({ width: 1280, height: 720 });
  });

  it('rounds odd client aspect ratios without exceeding the fixed height', () => {
    expect(
      resolveCanvasBackingDimensions({
        clientWidth: 1281,
        clientHeight: 721,
        viewportWidth: 0,
        viewportHeight: 0,
      }),
    ).toEqual({ width: 1279, height: 720 });
  });

  it('caps ultrawide backing width at the 1920-pixel constant', () => {
    expect(
      resolveCanvasBackingDimensions({
        clientWidth: 3840,
        clientHeight: 1080,
        viewportWidth: 0,
        viewportHeight: 0,
      }),
    ).toEqual({
      width: NEATRACING_MAX_BACKING_WIDTH,
      height: NEATRACING_FIXED_RENDER_HEIGHT,
    });
  });

  it('falls back to viewport aspect when client dimensions are zero', () => {
    expect(
      resolveCanvasBackingDimensions({
        clientWidth: 0,
        clientHeight: 0,
        viewportWidth: 1920,
        viewportHeight: 1080,
      }),
    ).toEqual({ width: 1280, height: 720 });
  });

  it('uses the fallback constant when client and viewport dimensions are unavailable', () => {
    expect(
      resolveCanvasBackingDimensions({
        clientWidth: 0,
        clientHeight: 0,
        viewportWidth: 0,
        viewportHeight: 0,
      }),
    ).toEqual({
      width: NEATRACING_FALLBACK_CANVAS_WIDTH,
      height: NEATRACING_FIXED_RENDER_HEIGHT,
    });
  });
});

describe('resolveLetterboxLayout', () => {
  it('fills a 16:9 viewport exactly with a 1280x720 backing store', () => {
    expect(
      resolveLetterboxLayout({
        viewportWidth: 1920,
        viewportHeight: 1080,
        contentWidth: 1280,
        contentHeight: 720,
      }),
    ).toEqual({ width: 1920, height: 1080, offsetX: 0, offsetY: 0 });
  });

  it('letterboxes horizontally when the backing width is clamped on an ultrawide viewport', () => {
    expect(
      resolveLetterboxLayout({
        viewportWidth: 2560,
        viewportHeight: 720,
        contentWidth: 1920,
        contentHeight: 720,
      }),
    ).toEqual({ width: 1920, height: 720, offsetX: 320, offsetY: 0 });
  });

  it('keeps the letterbox box within the viewport bounds at odd viewport sizes', () => {
    const layout = resolveLetterboxLayout({
      viewportWidth: 1281,
      viewportHeight: 721,
      contentWidth: 1279,
      contentHeight: 720,
    });
    expect(layout.width).toBeLessThanOrEqual(1281);
    expect(layout.height).toBeLessThanOrEqual(721);
  });

  it('centers the letterbox box with non-negative offsets at odd viewport sizes', () => {
    const layout = resolveLetterboxLayout({
      viewportWidth: 1281,
      viewportHeight: 721,
      contentWidth: 1279,
      contentHeight: 720,
    });
    expect(layout.offsetX).toBeGreaterThanOrEqual(0);
    expect(layout.offsetY).toBeGreaterThanOrEqual(0);
  });

  it('collapses to a zero-size box when the viewport has no measurable area', () => {
    expect(
      resolveLetterboxLayout({
        viewportWidth: 0,
        viewportHeight: 0,
        contentWidth: 1280,
        contentHeight: 720,
      }),
    ).toEqual({ width: 0, height: 0, offsetX: 0, offsetY: 0 });
  });

  it('collapses to a zero-size box when the content has no measurable area', () => {
    expect(
      resolveLetterboxLayout({
        viewportWidth: 1920,
        viewportHeight: 1080,
        contentWidth: 0,
        contentHeight: 0,
      }),
    ).toEqual({ width: 0, height: 0, offsetX: 0, offsetY: 0 });
  });
});

describe('applyCanvasBackingStore', () => {
  it('writes computed dimensions onto the canvas backing store', () => {
    const surface = { clientWidth: 0, clientHeight: 0, width: 0, height: 0 };
    applyCanvasBackingStore(surface, { width: 1280, height: 720 });
    expect(surface.width).toBe(1280);
    expect(surface.height).toBe(720);
  });
});

describe('updateRendererSize', () => {
  it('applies the backing store and returns the letterbox layout for the viewport', () => {
    const surface = {
      clientWidth: 1920,
      clientHeight: 1080,
      width: 0,
      height: 0,
    };
    const update = updateRendererSize({
      canvas: surface,
      viewportWidth: 1920,
      viewportHeight: 1080,
    });
    expect({ width: surface.width, height: surface.height }).toEqual({
      width: 1280,
      height: 720,
    });
    expect(update.dimensions).toEqual({ width: 1280, height: 720 });
    expect(update.letterbox).toEqual({
      width: 1920,
      height: 1080,
      offsetX: 0,
      offsetY: 0,
    });
  });
});
