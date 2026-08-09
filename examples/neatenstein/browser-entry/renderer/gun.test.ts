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

  describe('AC-04c-002: wide horizontally-elongated chaingun aspect ratio', () => {
    it('exposes GUN_BODY_ASPECT_RATIO near 1.6 (width/height) for the wide chaingun silhouette', async () => {
      const { GUN_BODY_ASPECT_RATIO } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      expect(GUN_BODY_ASPECT_RATIO).toBeDefined();
      // AC-04c-002: the wide Wolfenstein chaingun targets a 1.6 width/height
      // ratio, replacing the old 0.75 square-column ratio. The current 0.75
      // constant must fail this assertion.
      expect(GUN_BODY_ASPECT_RATIO).toBeCloseTo(1.6, 1);
    });

    it('measures a wide body silhouette (~1.6 width/height) from fillRect calls across aspect ratios', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const gun = createInitialGunState();
      for (const [viewportWidth, viewportHeight] of [
        [640, 360],
        [2560, 1080],
      ]) {
        const ctx = createMockCanvasContext();
        renderGunOverlay(ctx, gun, viewportWidth, viewportHeight);

        // With the unified-sprite contract, the gun body is drawn without
        // vector-path commands. We measure the body from sprite (fillRect)
        // calls anchored in the lower portion of the viewport.
        const fillRectCalls = (
          ctx.fillRect as unknown as {
            mock: { calls: [number, number, number, number][] };
          }
        ).mock.calls;

        const bodyRects = fillRectCalls.filter(
          ([, y, , h]) => y + h >= viewportHeight * 0.65,
        );
        expect(bodyRects.length).toBeGreaterThan(0);

        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        for (const [x, y, w, h] of bodyRects) {
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x + w);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y + h);
        }

        const bodyWidth = maxX - minX;
        const bodyHeight = maxY - minY;
        // The measured body ratio must match the new wide-chaingun target
        // (~1.6), not the old square-column ratio (0.75).
        expect(bodyWidth / bodyHeight).toBeCloseTo(1.6, 0);
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

  describe('AC-018b: unified sprite, no vector body', () => {
    it('does not use vector path commands for the cannon body', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const ctx = createMockCanvasContext();
      renderGunOverlay(ctx, createInitialGunState(), 640, 360);
      expect(ctx.beginPath).not.toHaveBeenCalled();
      expect(ctx.moveTo).not.toHaveBeenCalled();
      expect(ctx.lineTo).not.toHaveBeenCalled();
      expect(ctx.closePath).not.toHaveBeenCalled();
    });

    it('does not create linear gradients for the gun body', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const ctx = createMockCanvasContext();
      renderGunOverlay(ctx, createInitialGunState(), 640, 360);
      expect(ctx.createLinearGradient).not.toHaveBeenCalled();
    });

    it('does not mix vector-body fills and voxel fills for the same overlay', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const ctx = createMockCanvasContext();
      renderGunOverlay(ctx, createInitialGunState(), 640, 360);

      const vectorPathUsed =
        (ctx.beginPath as jest.Mock).mock.calls.length > 0 ||
        (ctx.moveTo as jest.Mock).mock.calls.length > 0 ||
        (ctx.lineTo as jest.Mock).mock.calls.length > 0;
      const voxelFillUsed = (ctx.fillRect as jest.Mock).mock.calls.length > 0;

      expect(vectorPathUsed && voxelFillUsed).toBe(false);
    });
  });

  describe('AC-04c-008: palette-indexed sprite rendering contract', () => {
    /**
     * Extract the R channel from a CSS color string. Returns -1 if parsing
     * fails. Used to distinguish palette-indexed sprite colors (which include
     * entries like rgb(10,10,12) from GUN_SPRITE_PALETTE index 1) from the
     * voxel-gun palette (which never uses R=10).
     */
    function extractR(color: string): number {
      // rgb(r, g, b) or rgba(r, g, b, a)
      const rgbMatch = color.match(/rgba?\(\s*(\d+)/i);
      if (rgbMatch) return parseInt(rgbMatch[1], 10);
      // #rrggbb hex
      const hexMatch = color.match(/^#([0-9a-f]{2})/i);
      if (hexMatch) return parseInt(hexMatch[1], 16);
      return -1;
    }

    it('renders at least one fillStyle color from GUN_SPRITE_PALETTE not in the voxel-gun palette', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;

      // Track every fillStyle assignment via a setter spy
      const loggedStyles: string[] = [];
      const ctx = createMockCanvasContext();
      let currentFillStyle = '';
      Object.defineProperty(ctx, 'fillStyle', {
        get() {
          return currentFillStyle;
        },
        set(v: string) {
          currentFillStyle = v;
          loggedStyles.push(v);
        },
        configurable: true,
      });

      renderGunOverlay(ctx, createInitialGunState(), 640, 360);

      // GUN_SPRITE_PALETTE index 1 is [10, 10, 12, 255] (dark outline).
      // The voxel-gun palette uses #121418 (18,20,24) for dark — never
      // produces R=10. After 04c-impl-renderer, the decoded sprite will
      // include index-1 pixels from the idle grid, producing a fillStyle
      // with R=10.
      const hasPaletteColor = loggedStyles.some(
        (style) => extractR(style) === 10,
      );
      expect(hasPaletteColor).toBe(true);
    });

    it('renders the decoded sprite frame (per-pixel fillRect calls matching grid bounds)', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      const { GUN_SPRITE_SCALE } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('../../gun-sprite-data.js')) as Record<string, any>;

      const ctx = createMockCanvasContext();
      renderGunOverlay(ctx, createInitialGunState(), 640, 360);

      const fillRectCalls = (
        ctx.fillRect as unknown as {
          mock: { calls: [number, number, number, number][] };
        }
      ).mock.calls;

      // The decoded sprite at GUN_SPRITE_SCALE produces fillRect calls whose
      // widths and heights are multiples of GUN_SPRITE_SCALE. The voxel-gun
      // projector produces variable-size voxels that are NOT all equal to
      // GUN_SPRITE_SCALE. After 04c-impl-renderer, every fillRect call
      // representing a sprite pixel should have width=height=GUN_SPRITE_SCALE.
      const scaleUniformCalls = fillRectCalls.filter(
        ([, , w, h]) => w === GUN_SPRITE_SCALE && h === GUN_SPRITE_SCALE,
      );

      // The idle grid has well over 100 non-transparent pixels, so after
      // decoding at GUN_SPRITE_SCALE there should be many uniform fillRect
      // calls. The voxel-gun projector produces far fewer.
      expect(scaleUniformCalls.length).toBeGreaterThan(50);
    });

    it('when firing, renders muzzle-flash pixels with semi-transparent alpha from GUN_SPRITE_PALETTE index 7', async () => {
      const { renderGunOverlay, createInitialGunState } =
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        (await import('./gun.ts')) as Record<string, any>;
      // Track fillStyle assignments with alpha
      const loggedStyles: string[] = [];
      const ctx = createMockCanvasContext();
      let currentFillStyle = '';
      Object.defineProperty(ctx, 'fillStyle', {
        get() {
          return currentFillStyle;
        },
        set(v: string) {
          currentFillStyle = v;
          loggedStyles.push(v);
        },
        configurable: true,
      });

      const firingGun = { ...createInitialGunState(), firing: true };
      renderGunOverlay(ctx, firingGun, 640, 360);

      // GUN_SPRITE_PALETTE index 7 is [255, 230, 120, 200] — semi-transparent
      // (alpha = 200/255 ≈ 0.78). The current voxel-gun muzzle flash uses
      // 'rgb(255,230,120)' (fully opaque, no alpha). After 04c-impl-renderer,
      // the decoded sprite should produce 'rgba(255,230,120,0.78...)' for
      // muzzle-flash pixels.
      const hasAlphaMuzzleFlash = loggedStyles.some(
        (style) =>
          style.includes('rgba') &&
          style.includes('255') &&
          style.includes('230') &&
          style.includes('120'),
      );
      expect(hasAlphaMuzzleFlash).toBe(true);
    });
  });
});
