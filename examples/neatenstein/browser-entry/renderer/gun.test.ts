import { describe, expect, it, jest } from '@jest/globals';

/**
 * Minimal mock 2D canvas context for verifying gun render side effects.
 *
 * Only the methods the gun renderer is expected to call are stubbed.
 */
function createMockCanvasContext(): CanvasRenderingContext2D {
  const gradient = { addColorStop: jest.fn() };
  return {
    fillRect: jest.fn(),
    fillStyle: '',
    strokeStyle: '',
    globalAlpha: 1,
    lineWidth: 1,
    shadowColor: '',
    shadowBlur: 0,
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    stroke: jest.fn(),
    fill: jest.fn(),
    arc: jest.fn(),
    ellipse: jest.fn(),
    closePath: jest.fn(),
    save: jest.fn(),
    restore: jest.fn(),
    translate: jest.fn(),
    createLinearGradient: jest.fn(() => gradient),
  } as unknown as CanvasRenderingContext2D;
}

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/renderer/gun.ts.
 *
 * Covers AC-101: the center-screen DOOM-style plasma cannon must be renderable
 * as a pure function of GunState, with Neon White body, teal accents, and a
 * recoil offset applied before drawing.
 */

describe('Neatenstein gun overlay renderer', () => {
  describe('AC-101: required exports', () => {
    it('exports renderGunOverlay', async () => {
      const { renderGunOverlay } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      expect(typeof renderGunOverlay).toBe('function');
    });

    it('exports createInitialGunState', async () => {
      const { createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      expect(typeof createInitialGunState).toBe('function');
    });
  });

  describe('AC-101: geometry and side effects', () => {
    it('does not draw a filled rectangle covering the entire gun body', async () => {
      const { renderGunOverlay, createInitialGunState, GUN_BODY_ASPECT_RATIO } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const ctx = createMockCanvasContext();
      const width = 640;
      const height = 360;
      renderGunOverlay(ctx, createInitialGunState(), width, height);

      const gunHeight = height * 0.22;
      const gunWidth = gunHeight * GUN_BODY_ASPECT_RATIO;
      const centerX = width / 2;
      const bodyLeft = centerX - gunWidth / 2;
      const bodyRight = centerX + gunWidth / 2;
      const gunTop = height - gunHeight;
      const eps = 1;

      const fillRectCalls = (
        ctx.fillRect as unknown as {
          mock: { calls: [number, number, number, number][] };
        }
      ).mock.calls;
      for (const [x, y, w, h] of fillRectCalls) {
        const coversFullWidth = x <= bodyLeft + eps && x + w >= bodyRight - eps;
        const coversFullHeight = y <= gunTop + eps && y + h >= height - eps;
        expect(coversFullWidth && coversFullHeight).toBe(false);
      }
    });

    it('saves the canvas state before applying recoil', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const ctx = createMockCanvasContext();
      const gun = { ...createInitialGunState(), recoilOffset: 5 };
      renderGunOverlay(ctx, gun, 640, 360);
      expect(ctx.save).toHaveBeenCalled();
    });

    it('translates the canvas by a numeric recoil vector', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const ctx = createMockCanvasContext();
      const gun = { ...createInitialGunState(), recoilOffset: 5 };
      renderGunOverlay(ctx, gun, 640, 360);
      expect(ctx.translate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.any(Number),
      );
    });

    it('restores the canvas state after applying recoil', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const ctx = createMockCanvasContext();
      const gun = { ...createInitialGunState(), recoilOffset: 5 };
      renderGunOverlay(ctx, gun, 640, 360);
      expect(ctx.restore).toHaveBeenCalled();
    });
  });

  describe('AC-101: color configuration', () => {
    it('uses Neon White (#FBFFFF) for the gun body color constant', async () => {
      const { NEATENSTEIN_GUN_BODY_COLOR } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      expect(NEATENSTEIN_GUN_BODY_COLOR).toBe('#FBFFFF');
    });

    it('uses teal (#00f0ff) for the gun accent color constant', async () => {
      const { NEATENSTEIN_GUN_ACCENT_COLOR } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      expect(NEATENSTEIN_GUN_ACCENT_COLOR).toBe('#00f0ff');
    });
  });

  describe('AC-403R: no elliptical shadow bar', () => {
    it('does not draw an elliptical shadow under the cannon', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const ctx = createMockCanvasContext();
      renderGunOverlay(ctx, createInitialGunState(), 640, 360);
      expect(ctx.ellipse).not.toHaveBeenCalled();
    });
  });

  describe('AC-11a: aspect-correct sizing and detail', () => {
    it('exposes GUN_BODY_ASPECT_RATIO and keeps body width proportional to body height across aspect ratios', async () => {
      const { renderGunOverlay, createInitialGunState, GUN_BODY_ASPECT_RATIO } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      expect(GUN_BODY_ASPECT_RATIO).toBeDefined();

      const gun = createInitialGunState();
      for (const [viewportWidth, viewportHeight] of [
        [640, 360],
        [2560, 1080],
      ]) {
        const ctx = createMockCanvasContext();
        renderGunOverlay(ctx, gun, viewportWidth, viewportHeight);

        const lineToCalls = (
          ctx.lineTo as unknown as { mock: { calls: number[][] } }
        ).mock.calls;
        const moveToCalls = (
          ctx.moveTo as unknown as { mock: { calls: number[][] } }
        ).mock.calls;

        const xs = [
          moveToCalls[0][0],
          lineToCalls[0][0],
          lineToCalls[1][0],
          lineToCalls[2][0],
        ];
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const gunTop = lineToCalls[0][1];
        const gunHeight = viewportHeight - gunTop;

        expect((maxX - minX) / gunHeight).toBeCloseTo(GUN_BODY_ASPECT_RATIO);
      }
    });

    it('draws a barrel-band detail with fillRect', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const ctx = createMockCanvasContext();
      renderGunOverlay(ctx, createInitialGunState(), 640, 360);
      expect(ctx.fillRect).toHaveBeenCalled();
    });
  });
});
