import { describe, expect, it } from '@jest/globals';

const loadModule = (path: string): Promise<any> => import(path);

const BACKGROUND_RGB = { r: 6, g: 11, b: 20 };
const MAX_VIEW_DIST = 20;

function createMockCanvasContext() {
  const globalAlphaSets: number[] = [];
  const createImageData = jest.fn((w: number, h: number) => ({
    data: new Uint8ClampedArray(w * h * 4),
    height: h,
    width: w,
  }));
  const putImageData = jest.fn();
  const fillRect = jest.fn();

  return {
    createImageData,
    putImageData,
    fillRect,
    get globalAlpha() {
      return 1;
    },
    set globalAlpha(value: number) {
      globalAlphaSets.push(value);
    },
    _globalAlphaSets: globalAlphaSets,
  };
}

function rgbDistanceToBackground(
  red: number,
  green: number,
  blue: number,
): number {
  const dr = red - BACKGROUND_RGB.r;
  const dg = green - BACKGROUND_RGB.g;
  const db = blue - BACKGROUND_RGB.b;
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

function sampleColumnPixel(framebuffer: Uint8ClampedArray, column: number) {
  const width = Math.floor(Math.sqrt(framebuffer.length / 4));
  const row = 2; // matches drawStart used in the render calls
  const offset = (row * width + column) * 4;
  return {
    r: framebuffer[offset],
    g: framebuffer[offset + 1],
    b: framebuffer[offset + 2],
  };
}

describe('Neatenstein neon wall column renderer', () => {
  it('writes into the framebuffer without invoking ctx.fillRect', async () => {
    const { renderNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    const ctx = createMockCanvasContext();
    renderNeonWallColumn(framebuffer, 0, 2, 5, '#00bfff', 1, ctx);
    expect(ctx.fillRect).not.toHaveBeenCalled();
  });

  it('flushes the CPU path with a single ImageData putImageData call', async () => {
    const { renderNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    const ctx = createMockCanvasContext();
    renderNeonWallColumn(framebuffer, 0, 2, 5, '#00bfff', 1, ctx);
    expect({
      putImageDataCalls: ctx.putImageData.mock.calls.length,
      fillRectCalls: ctx.fillRect.mock.calls.length,
    }).toEqual({
      putImageDataCalls: 1,
      fillRectCalls: 0,
    });
  });

  it('darkens far wall columns toward the background color', async () => {
    const { renderNeonWallColumn } = await loadModule('./walls.ts');
    const nearBuffer = new Uint8ClampedArray(8 * 8 * 4);
    const farBuffer = new Uint8ClampedArray(8 * 8 * 4);
    const ctx = createMockCanvasContext();
    renderNeonWallColumn(nearBuffer, 0, 2, 5, '#00bfff', 1, ctx);
    renderNeonWallColumn(farBuffer, 0, 2, 5, '#00bfff', MAX_VIEW_DIST, ctx);

    const nearPixel = sampleColumnPixel(nearBuffer, 0);
    const farPixel = sampleColumnPixel(farBuffer, 0);
    const nearDistance = rgbDistanceToBackground(
      nearPixel.r,
      nearPixel.g,
      nearPixel.b,
    );
    const farDistance = rgbDistanceToBackground(
      farPixel.r,
      farPixel.g,
      farPixel.b,
    );
    expect(farDistance).toBeLessThan(nearDistance);
  });

  it('does not mutate ctx.globalAlpha during rendering', async () => {
    const { renderNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    const ctx = createMockCanvasContext();
    renderNeonWallColumn(framebuffer, 0, 2, 5, '#00bfff', 1, ctx);
    expect(ctx._globalAlphaSets.length).toBe(0);
  });
});
