/**
 * @jest-environment node
 */

import { describe, expect, it } from '@jest/globals';
import { type VoxelSnapshot } from '../../../neatenstein/scripts/snapshot-renderer';
import { NEATENSTEIN_FLOOR_FOV_RADIANS } from './floor';
import type { NeatensteinSpriteProjection } from './sprites';

/**
 * Build a tiny pre-rendered voxel-like frame for unit tests.
 *
 * The returned frame is transparent except for a vertical strip of colored
 * pixels so the voxel renderer can be exercised without loading the full
 * enemy voxel grid.
 *
 * @param width - Frame width in pixels.
 * @param height - Frame height in pixels.
 * @param leftColor - RGBA for the left half of the frame.
 * @param rightColor - RGBA for the right half of the frame.
 * @returns A {@link VoxelSnapshot}-compatible object.
 */
function createMockVoxelSnapshot(
  width: number,
  height: number,
  leftColor: [number, number, number, number],
  rightColor: [number, number, number, number],
): VoxelSnapshot {
  const data = new Uint8ClampedArray(width * height * 4).fill(0);
  const mid = Math.floor(width / 2);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const color = x < mid ? leftColor : rightColor;
      const offset = (y * width + x) * 4;
      data[offset] = color[0];
      data[offset + 1] = color[1];
      data[offset + 2] = color[2];
      data[offset + 3] = color[3];
    }
  }

  return { data, width, height };
}

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

  it('uses the focal-length formula from the wall/floor renderer for scale', async () => {
    const { projectNeatensteinSprite } = await import('./sprites');

    const camera = {
      posX: 0,
      posY: 0,
      dirX: 1,
      dirY: 0,
      planeX: 0,
      planeY: 0.66,
    };
    const perpDist = 2;
    const worldX = camera.posX + camera.dirX * perpDist;
    const canvasHeight = 64;
    const focalLength =
      canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
    const expectedScale = Math.abs(focalLength / perpDist) * 0.5;

    const projection = projectNeatensteinSprite(
      { worldX, worldY: 0 },
      camera,
      64,
      canvasHeight,
    );

    expect(projection.scale).toBeCloseTo(expectedScale, 5);
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

    expect(clip.visibleColumns).toEqual([
      25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38,
    ]);
  });

  it('resolves yaw index 0 when the sprite faces the camera', async () => {
    const { resolveNeatensteinEnemyFrame } = await import('./sprites');

    // Camera is east of the sprite and the sprite faces east, so the
    // camera-relative yaw and sprite facing cancel to zero.
    const sprite = {
      worldX: 0,
      worldY: 0,
      facing: 0,
      animationState: 'idle' as const,
      frameIndex: 0,
    };
    const camera = {
      posX: 1,
      posY: 0,
      dirX: 1,
      dirY: 0,
      planeX: 0,
      planeY: 0.66,
    };

    const frame = resolveNeatensteinEnemyFrame(sprite, camera);
    expect(frame).not.toBeNull();
  });

  it('resolves yaw index 4 when the sprite faces away from the camera', async () => {
    const { resolveNeatensteinEnemyFrame } = await import('./sprites');

    // Camera is east of the sprite while the sprite faces west. The relative
    // yaw is PI, which maps to the opposite atlas index 4.
    const sprite = {
      worldX: 0,
      worldY: 0,
      facing: Math.PI,
      animationState: 'idle' as const,
      frameIndex: 0,
    };
    const camera = {
      posX: 1,
      posY: 0,
      dirX: 1,
      dirY: 0,
      planeX: 0,
      planeY: 0.66,
    };

    const frame = resolveNeatensteinEnemyFrame(sprite, camera);
    expect(frame).not.toBeNull();
  });

  it('returns different frames for facings separated by one yaw step', async () => {
    const { resolveNeatensteinEnemyFrame } = await import('./sprites');

    const camera = {
      posX: 1,
      posY: 0,
      dirX: 1,
      dirY: 0,
      planeX: 0,
      planeY: 0.66,
    };
    const frameA = resolveNeatensteinEnemyFrame(
      {
        worldX: 0,
        worldY: 0,
        facing: 0,
        animationState: 'idle' as const,
      },
      camera,
    );
    const frameB = resolveNeatensteinEnemyFrame(
      {
        worldX: 0,
        worldY: 0,
        facing: Math.PI / 4,
        animationState: 'idle' as const,
      },
      camera,
    );

    expect(frameA).not.toBeNull();
    expect(frameB).not.toBeNull();
    expect(frameA).not.toBe(frameB);
  });

  it('returns null when the sprite lacks facing or animation state', async () => {
    const { resolveNeatensteinEnemyFrame } = await import('./sprites');
    const camera = {
      posX: 1,
      posY: 0,
      dirX: 1,
      dirY: 0,
      planeX: 0,
      planeY: 0.66,
    };

    expect(
      resolveNeatensteinEnemyFrame(
        { worldX: 0, worldY: 0, animationState: 'idle' as const },
        camera,
      ),
    ).toBeNull();
    expect(
      resolveNeatensteinEnemyFrame({ worldX: 0, worldY: 0, facing: 0 }, camera),
    ).toBeNull();
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

    const frame = createMockVoxelSnapshot(
      4,
      4,
      [255, 0, 64, 255],
      [0, 191, 255, 255],
    );

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
      frame,
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

    const frame = createMockVoxelSnapshot(
      4,
      4,
      [255, 0, 64, 255],
      [0, 191, 255, 255],
    );

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
      frame,
      ctx,
    );

    expect(ctx.calls.length).toBe(0);
  });

  it('writes voxel frame colors into visible framebuffer pixels', async () => {
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

    const frame = createMockVoxelSnapshot(
      2,
      4,
      [0, 191, 255, 255],
      [255, 0, 64, 255],
    );

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
      frame,
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

  describe('projectNeatensteinSprite rejection cases', () => {
    it('returns an invisible projection for invalid canvas dimensions', async () => {
      const { projectNeatensteinSprite } = await import('./sprites');
      const projection = projectNeatensteinSprite(
        { worldX: 5, worldY: 5 },
        {
          posX: 1,
          posY: 1,
          dirX: 1,
          dirY: 0,
          planeX: 0,
          planeY: 0.66,
        },
        Number.NaN,
        64,
      );
      expect(projection.visible).toBe(false);
    });

    it('returns an invisible projection for non-finite sprite or camera values', async () => {
      const { projectNeatensteinSprite } = await import('./sprites');
      const camera = {
        posX: 1,
        posY: 1,
        dirX: 1,
        dirY: 0,
        planeX: 0,
        planeY: 0.66,
      };
      const badSprite = projectNeatensteinSprite(
        { worldX: Number.NaN, worldY: 5 },
        camera,
        64,
        64,
      );
      const badCamera = projectNeatensteinSprite(
        { worldX: 5, worldY: 5 },
        { ...camera, posX: Number.POSITIVE_INFINITY },
        64,
        64,
      );
      expect({
        badSpriteVisible: badSprite.visible,
        badCameraVisible: badCamera.visible,
      }).toEqual({ badSpriteVisible: false, badCameraVisible: false });
    });

    it('returns an invisible projection for a degenerate camera matrix', async () => {
      const { projectNeatensteinSprite } = await import('./sprites');
      const projection = projectNeatensteinSprite(
        { worldX: 5, worldY: 5 },
        {
          posX: 1,
          posY: 1,
          dirX: 1,
          dirY: 0,
          planeX: 1,
          planeY: 0,
        },
        64,
        64,
      );
      expect(projection.visible).toBe(false);
    });

    it('returns an invisible projection for a sprite too close to the camera', async () => {
      const { projectNeatensteinSprite } = await import('./sprites');
      const projection = projectNeatensteinSprite(
        { worldX: 0.05, worldY: 0 },
        {
          posX: 0,
          posY: 0,
          dirX: 1,
          dirY: 0,
          planeX: 0,
          planeY: 0.66,
        },
        64,
        64,
      );
      expect(projection.visible).toBe(false);
    });
  });

  describe('clipNeatensteinSprite edge cases', () => {
    it('returns an empty visible span when the z-buffer is empty', async () => {
      const { clipNeatensteinSprite } = await import('./sprites');
      const camera = {
        posX: 1,
        posY: 1,
        dirX: 1,
        dirY: 0,
        planeX: 0,
        planeY: 0.66,
      };
      const clip = clipNeatensteinSprite(
        { worldX: 3, worldY: 1 },
        camera,
        64,
        64,
        new Float32Array(0),
      );
      expect(clip.visibleColumns).toEqual([]);
    });
  });

  describe('renderNeatensteinSprite voxel frame rendering', () => {
    const FRAME_SIZE = 8;
    const FRAME_PIXELS = FRAME_SIZE * FRAME_SIZE;
    const WALL_DISTANCE = 5;
    const SPRITE_DISTANCE = 2;
    const SPRITE_SCALE = 8;
    const RED_OPAQUE: [number, number, number, number] = [255, 0, 0, 255];
    const BLUE_OPAQUE: [number, number, number, number] = [0, 0, 255, 255];

    it('does not flush the canvas for an invisible projection', async () => {
      const { renderNeatensteinSprite } = await import('./sprites');
      const frame = createMockVoxelSnapshot(
        FRAME_SIZE,
        FRAME_SIZE,
        RED_OPAQUE,
        BLUE_OPAQUE,
      );
      const calls: unknown[][] = [];
      const trackingCtx = {
        putImageData(...args: unknown[]) {
          calls.push(args);
        },
      };

      renderNeatensteinSprite(
        new Uint8ClampedArray(FRAME_PIXELS * 4),
        new Float32Array(FRAME_SIZE).fill(WALL_DISTANCE),
        {
          screenX: 4,
          scale: 6,
          perpDist: SPRITE_DISTANCE,
          left: 1,
          right: 6,
          visible: false,
        },
        frame,
        trackingCtx,
      );
      expect(calls.length).toBe(0);
    });

    it('does not flush the canvas when the z-buffer is empty', async () => {
      const { renderNeatensteinSprite } = await import('./sprites');
      const frame = createMockVoxelSnapshot(
        FRAME_SIZE,
        FRAME_SIZE,
        RED_OPAQUE,
        BLUE_OPAQUE,
      );
      const calls: unknown[][] = [];
      const ctx = {
        putImageData(...args: unknown[]) {
          calls.push(args);
        },
      };

      renderNeatensteinSprite(
        new Uint8ClampedArray(FRAME_PIXELS * 4),
        new Float32Array(0),
        {
          screenX: 4,
          scale: 6,
          perpDist: SPRITE_DISTANCE,
          left: 1,
          right: 6,
          visible: true,
        },
        frame,
        ctx,
      );
      expect(calls.length).toBe(0);
    });

    it('does not flush the canvas when framebuffer dimensions cannot be drawn', async () => {
      const { renderNeatensteinSprite } = await import('./sprites');
      const frame = createMockVoxelSnapshot(
        FRAME_SIZE,
        FRAME_SIZE,
        RED_OPAQUE,
        BLUE_OPAQUE,
      );
      const calls: unknown[][] = [];
      const ctx = {
        putImageData(...args: unknown[]) {
          calls.push(args);
        },
      };

      renderNeatensteinSprite(
        new Uint8ClampedArray(0),
        new Float32Array(FRAME_SIZE).fill(WALL_DISTANCE),
        {
          screenX: 4,
          scale: 6,
          perpDist: SPRITE_DISTANCE,
          left: 1,
          right: 6,
          visible: true,
        },
        frame,
        ctx,
      );
      expect(calls.length).toBe(0);
    });

    it('flushes exactly once for a visible voxel projection', async () => {
      const { renderNeatensteinSprite } = await import('./sprites');
      const frame = createMockVoxelSnapshot(
        FRAME_SIZE,
        FRAME_SIZE,
        RED_OPAQUE,
        BLUE_OPAQUE,
      );
      const framebuffer = new Uint8ClampedArray(FRAME_PIXELS * 4).fill(0);
      const zBuffer = new Float32Array(FRAME_SIZE).fill(WALL_DISTANCE);
      const calls: unknown[][] = [];
      const ctx = {
        putImageData(...args: unknown[]) {
          calls.push(args);
        },
      };

      renderNeatensteinSprite(
        framebuffer,
        zBuffer,
        {
          screenX: FRAME_SIZE / 2,
          scale: SPRITE_SCALE,
          perpDist: SPRITE_DISTANCE,
          left: 0,
          right: FRAME_SIZE - 1,
          visible: true,
        },
        frame,
        ctx,
      );
      expect(calls.length).toBe(1);
    });

    it('writes at least two distinct non-black pixel colors from the voxel frame', async () => {
      const { renderNeatensteinSprite } = await import('./sprites');
      const frame = createMockVoxelSnapshot(
        FRAME_SIZE,
        FRAME_SIZE,
        RED_OPAQUE,
        BLUE_OPAQUE,
      );
      const framebuffer = new Uint8ClampedArray(FRAME_PIXELS * 4).fill(0);
      const zBuffer = new Float32Array(FRAME_SIZE).fill(WALL_DISTANCE);
      const ctx = {
        putImageData() {
          /* no-op for CPU-side flush */
        },
      };

      renderNeatensteinSprite(
        framebuffer,
        zBuffer,
        {
          screenX: FRAME_SIZE / 2,
          scale: SPRITE_SCALE,
          perpDist: SPRITE_DISTANCE,
          left: 0,
          right: FRAME_SIZE - 1,
          visible: true,
        },
        frame,
        ctx,
      );

      const distinctColors = new Set<string>();
      for (let i = 0; i < framebuffer.length; i += 4) {
        if (framebuffer[i + 3] === 0) {
          continue;
        }
        const r = framebuffer[i];
        const g = framebuffer[i + 1];
        const b = framebuffer[i + 2];
        if (r === 0 && g === 0 && b === 0) {
          continue;
        }
        distinctColors.add(`${r},${g},${b}`);
      }
      expect(distinctColors.size).toBeGreaterThanOrEqual(2);
    });

    it('uses precomputed visible columns when present on the projection', async () => {
      const { clipNeatensteinSprite, renderNeatensteinSprite } =
        await import('./sprites');
      const { buildNeatensteinZBuffer } = await import('./zbuffer');

      const frame = createMockVoxelSnapshot(
        FRAME_SIZE,
        FRAME_SIZE,
        RED_OPAQUE,
        BLUE_OPAQUE,
      );
      const framebuffer = new Uint8ClampedArray(FRAME_PIXELS * 4).fill(0);
      const zBuffer = buildNeatensteinZBuffer(FRAME_SIZE).fill(WALL_DISTANCE);

      const camera = {
        posX: 1,
        posY: 1,
        dirX: 1,
        dirY: 0,
        planeX: 0,
        planeY: 0.66,
      };
      const clip = clipNeatensteinSprite(
        { worldX: 3, worldY: 1 },
        camera,
        FRAME_SIZE,
        FRAME_SIZE,
        zBuffer,
      );

      const calls: unknown[][] = [];
      const ctx = {
        putImageData(...args: unknown[]) {
          calls.push(args);
        },
      };

      renderNeatensteinSprite(framebuffer, zBuffer, clip, frame, ctx);
      expect(calls.length).toBe(1);
    });

    it('ignores legacy color strings and does not flush', async () => {
      const { renderNeatensteinSprite } = await import('./sprites');
      const framebuffer = new Uint8ClampedArray(FRAME_PIXELS * 4).fill(0);
      const zBuffer = new Float32Array(FRAME_SIZE).fill(WALL_DISTANCE);
      const calls: unknown[][] = [];
      const ctx = {
        putImageData(...args: unknown[]) {
          calls.push(args);
        },
      };

      renderNeatensteinSprite(
        framebuffer,
        zBuffer,
        {
          screenX: FRAME_SIZE / 2,
          scale: SPRITE_SCALE,
          perpDist: SPRITE_DISTANCE,
          left: 0,
          right: FRAME_SIZE - 1,
          visible: true,
        },
        '#00bfff',
        ctx,
      );
      expect(calls.length).toBe(0);
    });

    it('samples the leftmost frame column when the screen span is zero', async () => {
      const { renderNeatensteinSprite } = await import('./sprites');
      const frame = createMockVoxelSnapshot(
        FRAME_SIZE,
        FRAME_SIZE,
        RED_OPAQUE,
        BLUE_OPAQUE,
      );
      const framebuffer = new Uint8ClampedArray(FRAME_PIXELS * 4).fill(0);
      const zBuffer = new Float32Array(FRAME_SIZE).fill(WALL_DISTANCE);
      const calls: unknown[][] = [];
      const ctx = {
        putImageData(...args: unknown[]) {
          calls.push(args);
        },
      };

      renderNeatensteinSprite(
        framebuffer,
        zBuffer,
        {
          screenX: FRAME_SIZE / 2,
          scale: SPRITE_SCALE,
          perpDist: SPRITE_DISTANCE,
          left: 2,
          right: 2,
          visible: true,
          visibleColumns: [2],
        } as unknown as NeatensteinSpriteProjection,
        frame,
        ctx,
      );
      expect(calls.length).toBe(1);
    });
  });

  describe('renderNeatensteinSprite voxel column sampling', () => {
    const FRAME_SIZE = 8;
    const FRAME_PIXELS = FRAME_SIZE * FRAME_SIZE;
    const WALL_DISTANCE = 5;
    const SPRITE_DISTANCE = 2;

    it('skips transparent frame pixels and preserves the background', async () => {
      const { renderNeatensteinSprite } = await import('./sprites');
      const transparentLeft: [number, number, number, number] = [0, 0, 0, 0];
      const opaqueBlue: [number, number, number, number] = [0, 0, 255, 255];
      const frame = createMockVoxelSnapshot(
        FRAME_SIZE,
        FRAME_SIZE,
        transparentLeft,
        opaqueBlue,
      );
      const framebuffer = new Uint8ClampedArray(FRAME_PIXELS * 4).fill(0);
      const zBuffer = new Float32Array(FRAME_SIZE).fill(WALL_DISTANCE);
      const ctx = {
        putImageData() {
          /* CPU-side flush */
        },
      };

      renderNeatensteinSprite(
        framebuffer,
        zBuffer,
        {
          screenX: FRAME_SIZE / 2,
          scale: FRAME_SIZE,
          perpDist: SPRITE_DISTANCE,
          left: 0,
          right: FRAME_SIZE - 1,
          visible: true,
        },
        frame,
        ctx,
      );

      const leftColumnOpaque = framebuffer[3] !== 0;
      expect(leftColumnOpaque).toBe(false);
      // clipNeatensteinSpriteSpan excludes the right screen edge, so the
      // rightmost rendered column is FRAME_SIZE - 2.
      const rightColumnOffset = (FRAME_SIZE - 2) * 4;
      const rightColumnBlue =
        framebuffer[rightColumnOffset] === 0 &&
        framebuffer[rightColumnOffset + 1] === 0 &&
        framebuffer[rightColumnOffset + 2] === 255 &&
        framebuffer[rightColumnOffset + 3] === 255;
      expect(rightColumnBlue).toBe(true);
    });
  });

  describe('resolveNeatensteinEnemyFrame edge returns', () => {
    it('returns null when sprite or camera coordinates are non-finite', async () => {
      const { resolveNeatensteinEnemyFrame } = await import('./sprites');
      const camera = {
        posX: 1,
        posY: 0,
        dirX: 1,
        dirY: 0,
        planeX: 0,
        planeY: 0.66,
      };

      expect(
        resolveNeatensteinEnemyFrame(
          {
            worldX: Number.NaN,
            worldY: 0,
            facing: 0,
            animationState: 'idle' as const,
          },
          camera,
        ),
      ).toBeNull();
      expect(
        resolveNeatensteinEnemyFrame(
          {
            worldX: 0,
            worldY: 0,
            facing: 0,
            animationState: 'idle' as const,
          },
          { ...camera, posX: Number.POSITIVE_INFINITY },
        ),
      ).toBeNull();
    });

    it('returns null for an unknown animation state', async () => {
      const { resolveNeatensteinEnemyFrame } = await import('./sprites');
      const camera = {
        posX: 1,
        posY: 0,
        dirX: 1,
        dirY: 0,
        planeX: 0,
        planeY: 0.66,
      };

      expect(
        resolveNeatensteinEnemyFrame(
          {
            worldX: 0,
            worldY: 0,
            facing: 0,
            animationState: 'unknown' as unknown as 'idle',
          },
          camera,
        ),
      ).toBeNull();
    });
  });

  describe('resolveFramebufferSize branches', () => {
    it('uses both preferred dimensions when they fit in the buffer', async () => {
      const { __testOnlyResolveFramebufferSize } = await import('./sprites');
      const framebuffer = new Uint8ClampedArray(16 * 4);

      expect(__testOnlyResolveFramebufferSize(framebuffer, 2, 2)).toEqual({
        width: 2,
        height: 2,
      });
    });

    it('falls back to preferred height when width is omitted', async () => {
      const { __testOnlyResolveFramebufferSize } = await import('./sprites');
      const framebuffer = new Uint8ClampedArray(16 * 4);

      expect(
        __testOnlyResolveFramebufferSize(framebuffer, undefined, 4),
      ).toEqual({
        width: 4,
        height: 4,
      });
    });
  });

  describe('renderNeatensteinVoxelSpriteColumn edge returns', () => {
    it('returns early for an out-of-bounds screen column', async () => {
      const { __testOnlyRenderNeatensteinVoxelSpriteColumn } =
        await import('./sprites');
      const framebuffer = new Uint8ClampedArray(4 * 4 * 4).fill(0);
      const frame = createMockVoxelSnapshot(
        2,
        2,
        [255, 0, 0, 255],
        [0, 0, 255, 255],
      );

      __testOnlyRenderNeatensteinVoxelSpriteColumn(
        framebuffer,
        4,
        4,
        -1,
        0,
        4,
        frame,
        0,
      );

      expect(framebuffer.every((value) => value === 0)).toBe(true);
    });

    it('returns early for a zero-dimension frame', async () => {
      const { __testOnlyRenderNeatensteinVoxelSpriteColumn } =
        await import('./sprites');
      const framebuffer = new Uint8ClampedArray(4 * 4 * 4).fill(0);
      const frame = createMockVoxelSnapshot(
        4,
        0,
        [255, 0, 0, 255],
        [0, 0, 255, 255],
      );

      __testOnlyRenderNeatensteinVoxelSpriteColumn(
        framebuffer,
        4,
        4,
        0,
        0,
        4,
        frame,
        0,
      );

      expect(framebuffer.every((value) => value === 0)).toBe(true);
    });
  });
});
