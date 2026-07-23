/**
 * @jest-environment node
 */

import { describe, expect, it } from '@jest/globals';

describe('neatenstein sprites', () => {
  it('projects a centered sprite when directly in front of the camera', async () => {
    const { projectNeatensteinSprite } = await import('./sprites');

    const camera = {
      posX: 1,
      posY: 1,
      dirX: 1,
      dirY: 0,
      planeX: 0,
      planeY: 0.66,
    };
    const projection = projectNeatensteinSprite(
      { worldX: 3, worldY: 1 },
      camera,
      64,
      64,
    );

    expect(projection.screenX).toBeCloseTo(32, 0);
  });

  it('marks a sprite behind the camera plane as invisible', async () => {
    const { projectNeatensteinSprite } = await import('./sprites');

    const camera = {
      posX: 1,
      posY: 1,
      dirX: 1,
      dirY: 0,
      planeX: 0,
      planeY: 0.66,
    };
    const projection = projectNeatensteinSprite(
      { worldX: 0, worldY: 1 },
      camera,
      64,
      64,
    );

    expect(projection.visible).toBe(false);
  });

  it('produces a larger screen scale for a closer sprite', async () => {
    const { projectNeatensteinSprite } = await import('./sprites');

    const camera = {
      posX: 1,
      posY: 1,
      dirX: 1,
      dirY: 0,
      planeX: 0,
      planeY: 0.66,
    };
    const near = projectNeatensteinSprite(
      { worldX: 2, worldY: 1 },
      camera,
      64,
      64,
    );
    const far = projectNeatensteinSprite(
      { worldX: 5, worldY: 1 },
      camera,
      64,
      64,
    );

    expect(near.scale).toBeGreaterThan(far.scale);
  });

  it('clips a sprite against the z-buffer and returns only visible columns', async () => {
    const { buildNeatensteinZBuffer } = await import('./zbuffer');
    const { clipNeatensteinSprite } = await import('./sprites');

    const camera = {
      posX: 1,
      posY: 1,
      dirX: 1,
      dirY: 0,
      planeX: 0,
      planeY: 0.66,
    };

    const zBuffer = buildNeatensteinZBuffer(64);
    for (let i = 0; i < 64; i++) {
      zBuffer[i] = i === 32 ? 0.5 : 5;
    }

    const clip = clipNeatensteinSprite(
      { worldX: 3, worldY: 1 },
      camera,
      64,
      64,
      zBuffer,
    );

    expect(clip.visibleColumns).toEqual([28, 29, 30, 31, 33, 34, 35]);
  });

  it('draws sprite columns where the sprite is closer than the wall', async () => {
    const { renderNeatensteinSprite } = await import('./sprites');
    const { buildNeatensteinZBuffer } = await import('./zbuffer');

    const framebuffer = new Uint8ClampedArray(8 * 8 * 4).fill(0);
    const zBuffer = buildNeatensteinZBuffer(8);
    for (let i = 0; i < 8; i++) {
      zBuffer[i] = i < 4 ? 5 : 1;
    }

    const ctx: {
      calls: unknown[][];
      putImageData: (...args: unknown[]) => void;
    } = {
      calls: [],
      putImageData(...args: unknown[]) {
        this.calls.push(args);
      },
    };

    renderNeatensteinSprite(
      framebuffer,
      zBuffer,
      {
        screenX: 4,
        scale: 6,
        perpDist: 2,
        left: 1,
        right: 6,
        visible: true,
      },
      '#ff0040',
      ctx,
    );

    expect(ctx.calls.length).toBe(1);
  });

  it('does not flush the canvas when every sprite column is occluded', async () => {
    const { renderNeatensteinSprite } = await import('./sprites');
    const { buildNeatensteinZBuffer } = await import('./zbuffer');

    const framebuffer = new Uint8ClampedArray(8 * 8 * 4).fill(0);
    const zBuffer = buildNeatensteinZBuffer(8);
    for (let i = 0; i < 8; i++) {
      zBuffer[i] = 0.5;
    }

    const ctx: {
      calls: unknown[][];
      putImageData: (...args: unknown[]) => void;
    } = {
      calls: [],
      putImageData(...args: unknown[]) {
        this.calls.push(args);
      },
    };

    renderNeatensteinSprite(
      framebuffer,
      zBuffer,
      {
        screenX: 4,
        scale: 6,
        perpDist: 2,
        left: 1,
        right: 6,
        visible: true,
      },
      '#ff0040',
      ctx,
    );

    expect(ctx.calls.length).toBe(0);
  });

  it('writes the requested neon color into visible framebuffer pixels', async () => {
    const { renderNeatensteinSprite } = await import('./sprites');
    const { buildNeatensteinZBuffer } = await import('./zbuffer');

    const framebuffer = new Uint8ClampedArray(8 * 8 * 4).fill(0);
    const zBuffer = buildNeatensteinZBuffer(8);
    for (let i = 0; i < 8; i++) {
      zBuffer[i] = 5;
    }

    const ctx: { putImageData: () => void } = {
      putImageData() {
        /* no-op for CPU test */
      },
    };

    renderNeatensteinSprite(
      framebuffer,
      zBuffer,
      {
        screenX: 2,
        scale: 4,
        perpDist: 2,
        left: 1.5,
        right: 2.5,
        visible: true,
      },
      '#00bfff',
      ctx,
    );

    expect(Array.from(framebuffer.subarray(72, 76))).toEqual([
      0, 191, 255, 255,
    ]);
  });

  it('returns an empty visible span when the sprite is behind the camera', async () => {
    const { clipNeatensteinSprite } = await import('./sprites');
    const { buildNeatensteinZBuffer } = await import('./zbuffer');

    const camera = {
      posX: 1,
      posY: 1,
      dirX: 1,
      dirY: 0,
      planeX: 0,
      planeY: 0.66,
    };

    const clip = clipNeatensteinSprite(
      { worldX: 0, worldY: 1 },
      camera,
      64,
      64,
      buildNeatensteinZBuffer(64),
    );

    expect(clip).toMatchObject({ visible: false, visibleColumns: [] });
  });

  it('throws when the sprite color is not a #rrggbb hex string', async () => {
    const { renderNeatensteinSprite } = await import('./sprites');
    const { buildNeatensteinZBuffer } = await import('./zbuffer');

    const framebuffer = new Uint8ClampedArray(8 * 8 * 4).fill(0);
    const zBuffer = buildNeatensteinZBuffer(8).fill(5);
    const ctx: { putImageData: () => void } = {
      putImageData() {},
    };

    expect(() =>
      renderNeatensteinSprite(
        framebuffer,
        zBuffer,
        {
          screenX: 2,
          scale: 4,
          perpDist: 2,
          left: 1.5,
          right: 2.5,
          visible: true,
        },
        'red',
        ctx,
      ),
    ).toThrow('Expected #rrggbb hex color, got "red"');
  });

  it('throws when the sprite color has invalid hex components', async () => {
    const { renderNeatensteinSprite } = await import('./sprites');
    const { buildNeatensteinZBuffer } = await import('./zbuffer');

    const framebuffer = new Uint8ClampedArray(8 * 8 * 4).fill(0);
    const zBuffer = buildNeatensteinZBuffer(8).fill(5);
    const ctx: { putImageData: () => void } = {
      putImageData() {},
    };

    expect(() =>
      renderNeatensteinSprite(
        framebuffer,
        zBuffer,
        {
          screenX: 2,
          scale: 4,
          perpDist: 2,
          left: 1.5,
          right: 2.5,
          visible: true,
        },
        '#gg0000',
        ctx,
      ),
    ).toThrow('Invalid hex color components in "#gg0000"');
  });

  it('skips a sprite column that is fully off-screen vertically', async () => {
    const { renderNeatensteinSpriteColumn } = await import('./sprites');

    const framebuffer = new Uint8ClampedArray(8 * 8 * 4).fill(0);
    renderNeatensteinSpriteColumn(framebuffer, 2, 8, 4, '#00bfff');

    expect(framebuffer.every((v) => v === 0)).toBe(true);
  });
});
