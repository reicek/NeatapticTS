/** @jest-environment jsdom */

import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';

/**
 * Contract tests for the Neatenstein robot mugshot overlay.
 *
 * Covers AC-008:
 * - Head-crop pixel counts for front/frontLeft/frontRight
 * - Mouse-look direction selection from yawDelta (left/right/front)
 * - Healthy eye-stripe tint (neon teal at full health)
 *
 * The implementation module `hud-mugshot.ts` consumes the per-frame
 * `InputSnapshot.look.yawDelta` value to pick the facing mugshot crop.
 */

/** Logical head-crop region: rows 0–14, cols 19–29 of the 48×48 grid. */
const MUGSHOT_CROP_COLS = 11;
const MUGSHOT_CROP_ROWS = 15;
const ROBOT_SPRITE_SCALE = 4;
const MUGSHOT_CROP_WIDTH = MUGSHOT_CROP_COLS * ROBOT_SPRITE_SCALE; // 44
const MUGSHOT_CROP_HEIGHT = MUGSHOT_CROP_ROWS * ROBOT_SPRITE_SCALE; // 60

/** Neon teal — healthy eye-stripe color at full health. */
const NEON_TEAL: readonly [number, number, number, number] = [0, 240, 255, 255];

/** Neon gray — dead eye-stripe color at zero health. */
const NEON_GRAY: readonly [number, number, number, number] = [
  180, 190, 210, 255,
];

interface MugshotHeadCrop {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

interface MugshotLook {
  yawDelta: number;
}

interface MugshotOverlay {
  canvas: HTMLCanvasElement;
  update: (
    direction: 'front' | 'frontLeft' | 'frontRight',
    healthRatio?: number,
  ) => void;
}

interface Mock2DContext {
  createImageData: jest.Mock;
  putImageData: jest.Mock;
  drawImage: jest.Mock;
}

interface MugshotModule {
  decodeMugshotHeadCrop: (
    direction: 'front' | 'frontLeft' | 'frontRight',
    healthRatio?: number,
  ) => MugshotHeadCrop;
  selectMugshotDirection: (
    look: MugshotLook,
  ) => 'front' | 'frontLeft' | 'frontRight';
  createMugshotOverlay: () => MugshotOverlay;
}

/**
 * Count non-transparent pixels (alpha > 0) in an RGBA buffer.
 */
function countNonTransparentPixels(data: Uint8ClampedArray): number {
  let count = 0;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] > 0) count += 1;
  }
  return count;
}

/**
 * Count physical pixels whose RGBA exactly matches `target`.
 */
function countMatchingPixels(
  data: Uint8ClampedArray,
  target: readonly [number, number, number, number],
): number {
  let count = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (
      data[i] === target[0] &&
      data[i + 1] === target[1] &&
      data[i + 2] === target[2] &&
      data[i + 3] === target[3]
    ) {
      count += 1;
    }
  }
  return count;
}

describe('Neatenstein robot mugshot overlay', () => {
  describe('AC-008: head-crop pixel counts', () => {
    it('front head crop is 44×60 with 1888 non-transparent pixels', async () => {
      const { decodeMugshotHeadCrop } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const crop = decodeMugshotHeadCrop('front');

      expect(crop.width).toBe(MUGSHOT_CROP_WIDTH);
      expect(crop.height).toBe(MUGSHOT_CROP_HEIGHT);
      expect(countNonTransparentPixels(crop.data)).toBe(1888);
    });

    it('frontLeft head crop is 44×60 with 1648 non-transparent pixels', async () => {
      const { decodeMugshotHeadCrop } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const crop = decodeMugshotHeadCrop('frontLeft');

      expect(crop.width).toBe(MUGSHOT_CROP_WIDTH);
      expect(crop.height).toBe(MUGSHOT_CROP_HEIGHT);
      expect(countNonTransparentPixels(crop.data)).toBe(1648);
    });

    it('frontRight head crop is 44×60 with 1648 non-transparent pixels', async () => {
      const { decodeMugshotHeadCrop } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const crop = decodeMugshotHeadCrop('frontRight');

      expect(crop.width).toBe(MUGSHOT_CROP_WIDTH);
      expect(crop.height).toBe(MUGSHOT_CROP_HEIGHT);
      expect(countNonTransparentPixels(crop.data)).toBe(1648);
    });
  });

  describe('AC-008: mouse-look direction selection from yawDelta', () => {
    it('returns front when yawDelta is zero (mouse still)', async () => {
      const { selectMugshotDirection } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;

      expect(selectMugshotDirection({ yawDelta: 0 })).toBe('front');
    });

    it('returns frontLeft when yawDelta is negative (turning left)', async () => {
      const { selectMugshotDirection } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;

      expect(selectMugshotDirection({ yawDelta: -0.05 })).toBe('frontLeft');
    });

    it('returns frontRight when yawDelta is positive (turning right)', async () => {
      const { selectMugshotDirection } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;

      expect(selectMugshotDirection({ yawDelta: 0.05 })).toBe('frontRight');
    });

    it('returns frontLeft for a large negative yawDelta', async () => {
      const { selectMugshotDirection } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;

      expect(selectMugshotDirection({ yawDelta: -2.5 })).toBe('frontLeft');
    });

    it('returns frontRight for a large positive yawDelta', async () => {
      const { selectMugshotDirection } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;

      expect(selectMugshotDirection({ yawDelta: 3.0 })).toBe('frontRight');
    });
  });

  describe('AC-008: healthy eye-stripe tint', () => {
    it('tints front eye-stripe to neon teal rgb(0,240,255) at full health (128 physical pixels)', async () => {
      const { decodeMugshotHeadCrop } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const crop = decodeMugshotHeadCrop('front', 1.0);

      // Front head crop has 8 logical pixels at palette index 5 (eye-stripe),
      // each expanded 4×4 by the sprite scale → 128 physical teal pixels.
      expect(countMatchingPixels(crop.data, NEON_TEAL)).toBe(128);
    });

    it('tints frontLeft eye-stripe to neon teal at full health (80 physical pixels)', async () => {
      const { decodeMugshotHeadCrop } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const crop = decodeMugshotHeadCrop('frontLeft', 1.0);

      expect(countMatchingPixels(crop.data, NEON_TEAL)).toBe(80);
    });

    it('tints frontRight eye-stripe to neon teal at full health (80 physical pixels)', async () => {
      const { decodeMugshotHeadCrop } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const crop = decodeMugshotHeadCrop('frontRight', 1.0);

      expect(countMatchingPixels(crop.data, NEON_TEAL)).toBe(80);
    });

    it('tints front eye-stripe to neon gray rgb(180,190,210) at zero health (128 physical pixels)', async () => {
      const { decodeMugshotHeadCrop } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const crop = decodeMugshotHeadCrop('front', 0.0);

      expect(countMatchingPixels(crop.data, NEON_GRAY)).toBe(128);
    });
  });

  describe('AC-008: createMugshotOverlay canvas factory (stubbed 2D context)', () => {
    let mockCtx: Mock2DContext;
    let originalGetContext: typeof HTMLCanvasElement.prototype.getContext;

    beforeEach(() => {
      mockCtx = {
        createImageData: jest.fn((width: number, height: number) => ({
          data: new Uint8ClampedArray(width * height * 4),
        })) as unknown as jest.Mock,
        putImageData: jest.fn(),
        drawImage: jest.fn(),
      };
      originalGetContext = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = jest.fn(
        () => mockCtx,
      ) as unknown as typeof HTMLCanvasElement.prototype.getContext;
    });

    afterEach(() => {
      HTMLCanvasElement.prototype.getContext = originalGetContext;
    });

    it('createMugshotOverlay returns {canvas, update}', async () => {
      const { createMugshotOverlay } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const overlay = createMugshotOverlay();

      expect(overlay).toBeDefined();
      expect(overlay.canvas).toBeInstanceOf(HTMLCanvasElement);
      expect(typeof overlay.update).toBe('function');
    });

    it('canvas has correct physical dimensions (44×60)', async () => {
      const { createMugshotOverlay } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const overlay = createMugshotOverlay();

      expect(overlay.canvas.width).toBe(MUGSHOT_CROP_WIDTH);
      expect(overlay.canvas.height).toBe(MUGSHOT_CROP_HEIGHT);
    });

    it('initial update("front", 1.0) is called during construction (putImageData invoked once)', async () => {
      const { createMugshotOverlay } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      createMugshotOverlay();

      expect(mockCtx.putImageData).toHaveBeenCalledTimes(1);
    });

    it('update() calls putImageData with correct dimensions and position (0,0)', async () => {
      const { createMugshotOverlay } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const overlay = createMugshotOverlay();

      // Clear calls from the initial update during construction.
      mockCtx.putImageData.mockClear();

      overlay.update('frontLeft', 0.5);

      expect(mockCtx.createImageData).toHaveBeenCalledWith(
        MUGSHOT_CROP_WIDTH,
        MUGSHOT_CROP_HEIGHT,
      );
      expect(mockCtx.putImageData).toHaveBeenCalledTimes(1);
      const [imageData, x, y] = mockCtx.putImageData.mock.calls[0] as [
        { data: Uint8ClampedArray },
        number,
        number,
      ];
      expect(x).toBe(0);
      expect(y).toBe(0);
      expect(imageData.data.length).toBe(
        MUGSHOT_CROP_WIDTH * MUGSHOT_CROP_HEIGHT * 4,
      );
    });

    it('update() is a safe no-op when 2D context is unavailable (null)', async () => {
      // Override the stub to return null (simulating jsdom without canvas package)
      HTMLCanvasElement.prototype.getContext = jest.fn(
        () => null,
      ) as unknown as typeof HTMLCanvasElement.prototype.getContext;

      const { createMugshotOverlay } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;

      // Should not throw — the draw is a safe no-op
      const overlay = createMugshotOverlay();
      expect(overlay.canvas).toBeInstanceOf(HTMLCanvasElement);
      expect(() => overlay.update('front', 0.5)).not.toThrow();
    });

    it('update() sets crop pixel data into the ImageData buffer', async () => {
      const { createMugshotOverlay } =
        (await import('./hud-mugshot.ts')) as unknown as MugshotModule;
      const overlay = createMugshotOverlay();

      mockCtx.putImageData.mockClear();

      overlay.update('frontRight', 0.25);

      expect(mockCtx.createImageData).toHaveBeenCalledWith(
        MUGSHOT_CROP_WIDTH,
        MUGSHOT_CROP_HEIGHT,
      );
      // ImageData.data.set(crop.data) was called — the mock's data array
      // should have been populated with non-zero pixel data.
      const imageData = mockCtx.putImageData.mock.calls[0]![0] as {
        data: Uint8ClampedArray;
      };
      // The decoded frontRight crop has 1648 non-transparent pixels.
      let nonTransparent = 0;
      for (let i = 3; i < imageData.data.length; i += 4) {
        if (imageData.data[i]! > 0) nonTransparent += 1;
      }
      expect(nonTransparent).toBe(1648);
    });
  });
});
