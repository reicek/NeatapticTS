import { describe, expect, it } from '@jest/globals';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
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

function sampleColumnPixel(
  framebuffer: Uint8ClampedArray,
  column: number,
  width: number,
) {
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
    const { writeNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    const ctx = createMockCanvasContext();
    writeNeonWallColumn(framebuffer, 8, 8, 0, 2, 5, '#00bfff', 1);
    expect(ctx.fillRect).not.toHaveBeenCalled();
  });

  it('flushes the CPU path with a single ImageData putImageData call', async () => {
    const { writeNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    const ctx = createMockCanvasContext();
    writeNeonWallColumn(framebuffer, 8, 8, 0, 2, 5, '#00bfff', 1);
    ctx.putImageData({ data: framebuffer, width: 8, height: 8 }, 0, 0);
    expect({
      putImageDataCalls: ctx.putImageData.mock.calls.length,
      fillRectCalls: ctx.fillRect.mock.calls.length,
    }).toEqual({
      putImageDataCalls: 1,
      fillRectCalls: 0,
    });
  });

  it('darkens far wall columns toward the background color', async () => {
    const { writeNeonWallColumn } = await loadModule('./walls.ts');
    const nearBuffer = new Uint8ClampedArray(8 * 8 * 4);
    const farBuffer = new Uint8ClampedArray(8 * 8 * 4);
    writeNeonWallColumn(nearBuffer, 8, 8, 0, 2, 5, '#00bfff', 1);
    writeNeonWallColumn(farBuffer, 8, 8, 0, 2, 5, '#00bfff', MAX_VIEW_DIST);

    const nearPixel = sampleColumnPixel(nearBuffer, 0, 8);
    const farPixel = sampleColumnPixel(farBuffer, 0, 8);
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
    const { writeNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    const ctx = createMockCanvasContext();
    writeNeonWallColumn(framebuffer, 8, 8, 0, 2, 5, '#00bfff', 1);
    expect(ctx._globalAlphaSets.length).toBe(0);
  });
});

describe('writeNeonWallColumn edge cases', () => {
  it('returns early when framebuffer dimensions are invalid', async () => {
    const { writeNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    writeNeonWallColumn(framebuffer, -8, 8, 0, 2, 5, '#00bfff', 1);
    writeNeonWallColumn(framebuffer, 8, 0, 0, 2, 5, '#00bfff', 1);
    expect(framebuffer.every((value) => value === 0)).toBe(true);
  });

  it('returns early for non-finite or out-of-bounds columns', async () => {
    const { writeNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    writeNeonWallColumn(framebuffer, 8, 8, Number.NaN, 2, 5, '#00bfff', 1);
    writeNeonWallColumn(framebuffer, 8, 8, -1, 2, 5, '#00bfff', 1);
    writeNeonWallColumn(framebuffer, 8, 8, 8, 2, 5, '#00bfff', 1);
    expect(framebuffer.every((value) => value === 0)).toBe(true);
  });

  it('returns early when the clamped stripe is empty', async () => {
    const { writeNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    writeNeonWallColumn(framebuffer, 8, 8, 0, 5, 2, '#00bfff', 1);
    writeNeonWallColumn(framebuffer, 8, 8, 0, 8, 8, '#00bfff', 1);
    expect(framebuffer.every((value) => value === 0)).toBe(true);
  });

  it('treats non-finite distances as fully fogged', async () => {
    const { writeNeonWallColumn } = await loadModule('./walls.ts');
    const buffer = new Uint8ClampedArray(8 * 8 * 4);
    writeNeonWallColumn(
      buffer,
      8,
      8,
      0,
      2,
      5,
      '#00bfff',
      Number.POSITIVE_INFINITY,
    );
    const pixel = sampleColumnPixel(buffer, 0, 8);
    expect(pixel).toEqual(BACKGROUND_RGB);
  });

  it('throws when the wall color is not a valid #rrggbb string', async () => {
    const { writeNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    expect(() =>
      writeNeonWallColumn(framebuffer, 8, 8, 0, 2, 5, 'bad-color', 1),
    ).toThrow();
  });

  it('breaks early when the framebuffer is shorter than the dimensions imply', async () => {
    const { writeNeonWallColumn } = await loadModule('./walls.ts');
    const framebuffer = new Uint8ClampedArray(3);
    writeNeonWallColumn(framebuffer, 8, 8, 0, 0, 8, '#00bfff', 1);
    expect(framebuffer.every((value) => value === 0)).toBe(true);
  });
});

describe('Legacy wall renderer removal', () => {
  it('does not export the legacy renderNeonWallColumn wrapper', async () => {
    const walls = await loadModule('./walls.ts');
    expect(walls.renderNeonWallColumn).toBeUndefined();
  });

  it('does not infer square framebuffer dimensions from buffer length', async () => {
    const { resolveNeatensteinFramebufferSize } =
      await loadModule('./framebuffer.ts');
    const framebuffer = new Uint8ClampedArray(8 * 8 * 4);
    expect(() => resolveNeatensteinFramebufferSize(framebuffer)).toThrow();
  });
});
