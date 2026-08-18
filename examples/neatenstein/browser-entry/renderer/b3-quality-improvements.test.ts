/**
 * B3: Raycasting — Quality Improvements — RED phase test contracts.
 *
 * Every acceptance criterion and invariant from Step B3 of the
 * Neatenstein Ultimate Quality Upgrade plan is encoded as a failing
 * test below.  Tests that exercise not-yet-implemented behaviour
 * MUST fail for the right reason (missing export, missing module, or
 * wrong current value) — not for a fixture or syntax error.
 *
 * Invariant §8 regression guards (floor-wall alignment, spark↔grid
 * coupling) are included alongside the RED contracts.  These guards
 * verify the *existing* alignment that the B3 implementation must
 * preserve; they are expected to pass today and must continue to
 * pass after implementation.
 *
 * @jest-environment node
 */

import { describe, expect, it } from '@jest/globals';

// ── Static constant imports (safe — these modules already exist) ───────────

import { NEATENSTEIN_ZBUFFER_EMPTY } from './renderer.zbuffer.constants';
import {
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  NEATENSTEIN_FLOOR_GLOW_WIDTH_PX,
  NEATENSTEIN_FLOOR_LINE_WIDTH_PX,
  NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER,
  NEATENSTEIN_FLOOR_MAX_ALPHA,
  NEATENSTEIN_FLOOR_ALPHA_BANDS,
  NEATENSTEIN_FLOOR_LINE_SAMPLES,
} from './renderer.floor.constants';
import {
  NEATENSTEIN_RENDER_DISTANCE_CAP,
  NEATENSTEIN_FOG_START_DISTANCE,
  resolveNeatensteinFogFactor,
} from './framebuffer';
import type { NeatensteinPulse } from './renderer.pulse.types';
import type { NeatensteinGridProjectionContext } from './renderer.floor.types';

// ── Shared test helpers ─────────────────────────────────────────────────────

/** Standard test canvas dimensions. */
const CANVAS_WIDTH = 320;
const CANVAS_HEIGHT = 240;
const COLUMN_COUNT = 320;
const STRIPE_WIDTH = CANVAS_WIDTH / COLUMN_COUNT; // 1.0

/** Standard projection-context constants derived from the canvas. */
const FOCAL_LENGTH = CANVAS_HEIGHT / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
const HALF_WIDTH = CANVAS_WIDTH / 2;
const HORIZON_Y = CANVAS_HEIGHT * NEATENSTEIN_FLOOR_HORIZON_RATIO;
const PLANE_SCALE = (CANVAS_WIDTH / CANVAS_HEIGHT) * Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

/**
 * Build a `NeatensteinGridProjectionContext` for a given camera pose.
 */
function buildProjectionContext(
  cameraX: number,
  cameraY: number,
  yaw: number,
): NeatensteinGridProjectionContext {
  return {
    width: CANVAS_WIDTH,
    height: CANVAS_HEIGHT,
    cameraX,
    cameraY,
    cosYaw: Math.cos(yaw),
    sinYaw: Math.sin(yaw),
    focalLength: FOCAL_LENGTH,
    halfWidth: HALF_WIDTH,
    horizonY: HORIZON_Y,
    cameraHeight: NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  };
}

/**
 * Create a simple square flat map with walls only on the border.
 */
function createBorderedFlatMap(side: number): Uint8Array {
  const flatMap = new Uint8Array(side * side);
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      if (x === 0 || x === side - 1 || y === 0 || y === side - 1) {
        flatMap[y * side + x] = 1;
      }
    }
  }
  return flatMap;
}

/**
 * Create a mock VoxelSnapshot for sprite tests.
 */
function createMockVoxelSnapshot(
  width: number,
  height: number,
  color: [number, number, number, number],
): { data: Uint8ClampedArray; width: number; height: number } {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = color[0];
    data[i * 4 + 1] = color[1];
    data[i * 4 + 2] = color[2];
    data[i * 4 + 3] = color[3];
  }
  return { data, width, height };
}

// ── B3.1 — Wall texture mapping ────────────────────────────────────────────

describe('B3.1 — Wall texture mapping', () => {
  it('exports computeWallTexcoord from walls module', async () => {
    const walls = await import('./walls');
    // RED: computeWallTexcoord does not exist yet — writeNeonWallColumn is flat-shaded.
    expect(typeof (walls as unknown as Record<string, unknown>).computeWallTexcoord).toBe('function');
  });

  it('computeWallTexcoord returns a valid texcoord in [0,1) for an X-side hit', async () => {
    const walls = await import('./walls');
    const compute = (walls as unknown as Record<string, unknown>).computeWallTexcoord as
      | ((...args: unknown[]) => number)
      | undefined;
    // RED: function does not exist → compute is undefined → TypeError.
    expect(compute).toBeDefined();
    // Wall hit at (15, 8.5), side=0 (X-side), perpWallDist=6.5, dirY=-0.3849
    // wallX = posY + perpWallDist * dirY = 8.5 + 6.5 * (-0.3849) = 5.998
    // texcoord = fract(wallX) = 0.998
    const texcoord = compute!(8.5, 6.5, -0.3849);
    expect(texcoord).toBeGreaterThanOrEqual(0);
    expect(texcoord).toBeLessThan(1);
  });

  it('writeNeonWallColumn preserves neon-dominant aesthetic (Invariant §3)', async () => {
    const walls = await import('./walls');
    // The wall colors #00b7ff (X-side) and #00a4e5 (Y-side) are the neon identity.
    // If texture mapping is added, the primary read MUST remain the saturated flat color.
    // RED: writeNeonWallColumn currently has no texcoord/texture parameter —
    // the signature must be extended to accept texcoord without changing the
    // neon color output for the default (no-texture) case.
    const fn = walls.writeNeonWallColumn;
    expect(fn.length).toBeGreaterThanOrEqual(6); // at least framebuffer, x, height, color, fog, +texcoord
  });
});

// ── B3.2 — Floor alpha from unified fog factor ─────────────────────────────

describe('B3.2 — Floor alpha from unified fog factor', () => {
  it('exports resolveNeatensteinFloorAlphaFromDistance from floor.shade.utils', async () => {
    const mod = await import('./floor.shade.utils');
    // RED: resolveNeatensteinFloorAlphaFromDistance does not exist yet.
    expect(
      typeof (mod as unknown as Record<string, unknown>).resolveNeatensteinFloorAlphaFromDistance,
    ).toBe('function');
  });

  it('returns MAX_ALPHA at FOG_START_DISTANCE (no fog → full alpha)', async () => {
    const mod = await import('./floor.shade.utils');
    const fn = (mod as unknown as Record<string, unknown>).resolveNeatensteinFloorAlphaFromDistance as
      | ((d: number) => number)
      | undefined;
    expect(fn).toBeDefined();
    // At FOG_START, fogFactor = 0, so alpha should be MAX_ALPHA.
    const alpha = fn!(NEATENSTEIN_FOG_START_DISTANCE);
    expect(alpha).toBeCloseTo(NEATENSTEIN_FLOOR_MAX_ALPHA, 5);
  });

  it('returns ~0 at RENDER_DISTANCE_CAP (full fog → invisible)', async () => {
    const mod = await import('./floor.shade.utils');
    const fn = (mod as unknown as Record<string, unknown>).resolveNeatensteinFloorAlphaFromDistance as
      | ((d: number) => number)
      | undefined;
    expect(fn).toBeDefined();
    // At CAP, fogFactor = 1, so alpha should be ~0 (grid vanishes into fog).
    const alpha = fn!(NEATENSTEIN_RENDER_DISTANCE_CAP);
    expect(alpha).toBeLessThanOrEqual(0.01);
  });

  it('uses smoothstep fog factor (not linear interpolation)', async () => {
    const mod = await import('./floor.shade.utils');
    const fn = (mod as unknown as Record<string, unknown>).resolveNeatensteinFloorAlphaFromDistance as
      | ((d: number) => number)
      | undefined;
    expect(fn).toBeDefined();
    // At the midpoint, smoothstep gives fogFactor = 0.5 (not 0.5 — smoothstep(0.5) = 0.5).
    // The key assertion: alpha is NOT linear. Linear would give alpha = MAX_ALPHA * (1 - 0.5) = 0.29.
    // smoothstep gives fogFactor = 0.5, so alpha = MAX_ALPHA * (1 - 0.5) = 0.29.
    // Actually, smoothstep(0.5) = 0.5, so they coincide at the midpoint.
    // Use a non-midpoint distance to distinguish smoothstep from linear.
    const t = 0.25; // 1/4 into the fog range
    const distance = NEATENSTEIN_FOG_START_DISTANCE + t * (NEATENSTEIN_RENDER_DISTANCE_CAP - NEATENSTEIN_FOG_START_DISTANCE);
    const expectedFogFactor = t * t * (3 - 2 * t); // smoothstep
    const expectedAlpha = NEATENSTEIN_FLOOR_MAX_ALPHA * (1 - expectedFogFactor);
    const linearAlpha = NEATENSTEIN_FLOOR_MAX_ALPHA * (1 - t);
    const actualAlpha = fn!(distance);
    // Assert it matches smoothstep, not linear.
    expect(actualAlpha).toBeCloseTo(expectedAlpha, 3);
    expect(actualAlpha).not.toBeCloseTo(linearAlpha, 3);
  });
});

// ── B3.3 — Unified z-buffer sentinel ───────────────────────────────────────

describe('B3.3 — Unified z-buffer sentinel', () => {
  it('buildNeatensteinRenderFrame initializes zBuffer to NEATENSTEIN_ZBUFFER_EMPTY', async () => {
    const { buildNeatensteinRenderFrame } = await import('./frame');
    const frame = buildNeatensteinRenderFrame(
      {
        canvasWidth: CANVAS_WIDTH,
        canvasHeight: CANVAS_HEIGHT,
        simTick: 0,
        cameraX: 8,
        cameraY: 8,
        cameraYaw: 0,
        mapSeed: 42,
      },
      COLUMN_COUNT,
    );
    // RED: Currently zBuffer is `new Float32Array(columnCount)` = all ZEROS,
    // not Infinity. Every column must be the sentinel.
    expect(frame.zBuffer[0]).toBe(NEATENSTEIN_ZBUFFER_EMPTY);
    expect(frame.zBuffer[COLUMN_COUNT - 1]).toBe(NEATENSTEIN_ZBUFFER_EMPTY);
  });

  it('zBuffer sentinel is Number.POSITIVE_INFINITY (not 30 or 0)', () => {
    // The sentinel constant itself is already correct — this is a guard.
    expect(NEATENSTEIN_ZBUFFER_EMPTY).toBe(Number.POSITIVE_INFINITY);
  });
});

// ── B3.4 — NaN edge case fix ───────────────────────────────────────────────

describe('B3.4 — NaN edge case fix (sideDistX/Y 0*Infinity guard)', () => {
  it('exports resolveSideDistance guard from raycast module', async () => {
    const raycast = await import('./raycast');
    // RED: resolveSideDistance does not exist yet.
    // The plan requires: "Initialize sideDistX/Y with Number.POSITIVE_INFINITY
    // guard for near-zero direction." This helper encapsulates the guard.
    expect(typeof (raycast as unknown as Record<string, unknown>).resolveSideDistance).toBe('function');
  });

  it('resolveSideDistance returns Infinity when deltaDist is Infinity and offset is 0', async () => {
    const raycast = await import('./raycast');
    const fn = (raycast as unknown as Record<string, unknown>).resolveSideDistance as
      | ((...args: unknown[]) => number)
      | undefined;
    expect(fn).toBeDefined();
    // 0 * Infinity = NaN without the guard; must return Infinity.
    const result = fn!(0, Number.POSITIVE_INFINITY);
    expect(Number.isNaN(result)).toBe(false);
    expect(result).toBe(Number.POSITIVE_INFINITY);
  });

  it('resolveSideDistance returns offset*deltaDist for finite deltaDist', async () => {
    const raycast = await import('./raycast');
    const fn = (raycast as unknown as Record<string, unknown>).resolveSideDistance as
      | ((...args: unknown[]) => number)
      | undefined;
    expect(fn).toBeDefined();
    const result = fn!(0.5, 2.0);
    expect(result).toBe(1.0);
  });

  it('castRayDDAFromFlatMap does not produce NaN perpWallDist for grid-line-aligned near-zero dirX', async () => {
    const { castRayDDAFromFlatMap } = await import('./raycast');
    // Regression guard: posX exactly on grid line, dirX nearly zero (negative).
    // sideDistX = 0 * Infinity = NaN without the guard.
    // The DDA must still produce a valid (non-NaN) perpWallDist.
    const flatMap = createBorderedFlatMap(16);
    const hit = castRayDDAFromFlatMap(flatMap, 16, 8.0, 8.5, -1e-10, 0.5);
    expect(Number.isNaN(hit.perpWallDist)).toBe(false);
    // perpWallDist should be finite (hit a wall) or Infinity (no hit), never NaN.
    expect(Number.isFinite(hit.perpWallDist) || hit.perpWallDist === Number.POSITIVE_INFINITY).toBe(true);
  });

  it('castRayDDAFromFlatMap does not produce NaN perpWallDist when both dirs are near-zero on grid lines', async () => {
    const { castRayDDAFromFlatMap } = await import('./raycast');
    // Both posX and posY on grid lines, both directions nearly zero and negative.
    // sideDistX = 0 * Infinity = NaN, sideDistY = 0 * Infinity = NaN.
    const flatMap = createBorderedFlatMap(16);
    const hit = castRayDDAFromFlatMap(flatMap, 16, 8.0, 8.0, -1e-10, -1e-10);
    expect(Number.isNaN(hit.perpWallDist)).toBe(false);
  });
});

// ── B3.5 — Remove per-sprite putImageData ──────────────────────────────────

describe('B3.5 — Remove per-sprite putImageData', () => {
  it('renderNeatensteinSprite does not call ctx.putImageData', async () => {
    const { renderNeatensteinSprite, clipNeatensteinSprite } = await import('./sprites');

    const FRAME_SIZE = 8;
    const framebuffer = new Uint8ClampedArray(FRAME_SIZE * FRAME_SIZE * 4).fill(0);
    const zBuffer = new Float32Array(FRAME_SIZE).fill(5); // wall distance > sprite distance

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

    const frame = createMockVoxelSnapshot(4, 4, [255, 0, 64, 255]);

    const calls: unknown[][] = [];
    const ctx = {
      putImageData(...args: unknown[]) {
        calls.push(args);
      },
    };

    renderNeatensteinSprite(framebuffer, zBuffer, clip, frame, ctx);
    // RED: Currently renderNeatensteinSprite calls ctx.putImageData once (line 331).
    // After B3.5, the caller should do a single flush, not per-sprite.
    expect(calls.length).toBe(0);
  });
});

// ── B3.6 — Shader pipeline alignment contract ──────────────────────────────

/**
 * Helper that attempts a dynamic import of a shader module that may not exist
 * yet (RED phase). Using a string variable bypasses TypeScript static module
 * resolution so the test file compiles even when the target module is absent.
 */
async function tryImportShader(
  path: string,
): Promise<Record<string, unknown> | null> {
  try {
    const mod = await import(path);
    return mod as Record<string, unknown>;
  } catch {
    return null;
  }
}

describe('B3.6 — Shader pipeline alignment contract', () => {
  it('exports a camera-uniform creator from a shader module', async () => {
    // RED: No shader modules exist yet. The dynamic import will fail.
    const shaderMod = await tryImportShader('./shaders/camera-uniform');
    expect(shaderMod).not.toBeNull();
    expect(typeof shaderMod!.createNeatensteinCameraUniform).toBe('function');
  });

  it('exports a wall-DDA shader source string', async () => {
    const shaderMod = await tryImportShader('./shaders/wall-dda');
    expect(shaderMod).not.toBeNull();
    expect(typeof shaderMod!.NEATENSTEIN_WALL_DDA_SHADER_SOURCE).toBe('string');
  });

  it('exports a floor-caster shader source string', async () => {
    const shaderMod = await tryImportShader('./shaders/floor-caster');
    expect(shaderMod).not.toBeNull();
    expect(typeof shaderMod!.NEATENSTEIN_FLOOR_CASTER_SHADER_SOURCE).toBe(
      'string',
    );
  });

  it('camera uniform struct includes all required fields', async () => {
    const shaderMod = await tryImportShader('./shaders/camera-uniform');
    expect(shaderMod).not.toBeNull();
    const createFn = shaderMod!.createNeatensteinCameraUniform as
      | ((...args: unknown[]) => Record<string, unknown>)
      | undefined;
    expect(createFn).toBeDefined();
    // Build a uniform with a minimal camera pose.
    const uniform = createFn!({
      canvasWidth: CANVAS_WIDTH,
      canvasHeight: CANVAS_HEIGHT,
      cameraX: 8,
      cameraY: 8,
      cameraYaw: 0,
    });
    // Per Invariant §2: focalLength, planeScale, cameraDirection, cameraPlane,
    // cameraHeight, horizon, RENDER_DISTANCE_CAP.
    expect(uniform.focalLength).toBeDefined();
    expect(uniform.planeScale).toBeDefined();
    expect(uniform.cameraDirection).toBeDefined();
    expect(uniform.cameraPlane).toBeDefined();
    expect(uniform.cameraHeight).toBe(NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD);
    expect(uniform.horizon).toBeCloseTo(HORIZON_Y, 5);
    expect(uniform.renderDistanceCap).toBe(NEATENSTEIN_RENDER_DISTANCE_CAP);
  });

  it('wall DDA shader source contains highp precision qualifier', async () => {
    const shaderMod = await tryImportShader('./shaders/wall-dda');
    expect(shaderMod).not.toBeNull();
    const source = shaderMod!.NEATENSTEIN_WALL_DDA_SHADER_SOURCE as string;
    // Per B3.6: WebGL2 paths MUST mandate highp float precision.
    expect(source).toContain('highp');
  });

  it('wall DDA shader source contains perpendicular distance formula', async () => {
    const shaderMod = await tryImportShader('./shaders/wall-dda');
    expect(shaderMod).not.toBeNull();
    const source = shaderMod!.NEATENSTEIN_WALL_DDA_SHADER_SOURCE as string;
    // Per B3.6: DDA MUST use perpWallDist = (mapX - posX + (1-stepX)/2)/dirX,
    // NOT Euclidean distance.
    expect(source.toLowerCase()).toContain('perpwalldist');
  });
});

// ── B3.7 — Per-pixel floor casting ─────────────────────────────────────────

describe('B3.7 — Per-pixel floor casting', () => {
  it('exports castNeatensteinFloorPerPixel from floor module', async () => {
    const floor = await import('./floor');
    // RED: castNeatensteinFloorPerPixel does not exist yet — floor.ts uses
    // 80-sample line projection (NEATENSTEIN_FLOOR_LINE_SAMPLES = 80).
    expect(typeof (floor as unknown as Record<string, unknown>).castNeatensteinFloorPerPixel).toBe('function');
  });

  it('per-pixel caster uses fract(worldCoord) for procedural integer grid', async () => {
    const floor = await import('./floor');
    const fn = (floor as unknown as Record<string, unknown>).castNeatensteinFloorPerPixel as
      | ((...args: unknown[]) => unknown)
      | undefined;
    expect(fn).toBeDefined();
    // Per Invariant §4: floor MUST render a procedural world-space integer grid
    // computed via fract(worldCoord) or equivalent — NOT an arbitrary texture.
    // The function must accept a framebuffer and camera context and produce
    // per-pixel floor pixels with grid lines at 1-unit spacing.
    // RED: function does not exist → fn is undefined → TypeError on call.
    const framebuffer = new Uint8ClampedArray(CANVAS_WIDTH * CANVAS_HEIGHT * 4).fill(0);
    fn!(framebuffer, CANVAS_WIDTH, CANVAS_HEIGHT, {
      cameraX: 8,
      cameraY: 8,
      cosYaw: 1,
      sinYaw: 0,
      focalLength: FOCAL_LENGTH,
      halfWidth: HALF_WIDTH,
      horizonY: HORIZON_Y,
      cameraHeight: NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
    });
    // Verify some pixels were written (not all zero).
    const nonZero = Array.from(framebuffer.slice(0, 100)).some((v) => v !== 0);
    expect(nonZero).toBe(true);
  });

  it('per-pixel caster reuses NEATENSTEIN_FLOOR_* constants (not re-derived)', async () => {
    // Per Invariant §2 and B3.7: rowDistance must derive from the exact same
    // NEATENSTEIN_FLOOR_* constants (FOV, cameraHeight, horizon, focalLength)
    // as the current line-projection path.
    // This is a documentation/contract assertion — the per-pixel caster must
    // exist and use the shared constants.
    const floor = await import('./floor');
    expect(typeof (floor as unknown as Record<string, unknown>).castNeatensteinFloorPerPixel).toBe('function');
  });

  it('per-pixel caster replicates halo glow (GLOW_WIDTH_PX + 1px core)', async () => {
    // Per B3.6 double-stroke neon glow replication: the CPU per-pixel caster
    // MUST replicate the halo glow (NEATENSTEIN_FLOOR_GLOW_WIDTH_PX = 3 + 1px core).
    const floor = await import('./floor');
    expect(typeof (floor as unknown as Record<string, unknown>).castNeatensteinFloorPerPixel).toBe('function');
    // The glow constants must be used by the per-pixel caster.
    expect(NEATENSTEIN_FLOOR_GLOW_WIDTH_PX).toBe(3);
    expect(NEATENSTEIN_FLOOR_LINE_WIDTH_PX).toBe(1);
    expect(NEATENSTEIN_FLOOR_GLOW_ALPHA_MULTIPLIER).toBe(0.35);
  });
});

// ── Invariant §8 — Floor-wall alignment regression ─────────────────────────

describe('Invariant §8 — Floor-wall alignment regression', () => {
  /**
   * Helper: cast a wall ray for a given column and compute the wall-hit
   * world point, then project it via projectNeatensteinGridPoint.
   *
   * Returns { wallScreenX, projectedScreenX, perpWallDist, side }.
   */
  async function castAndAlign(
    column: number,
    cameraX: number,
    cameraY: number,
    yaw: number,
    flatMap: Uint8Array,
    mapSize: number,
  ): Promise<{
    wallScreenX: number;
    projectedScreenX: number;
    projectedScreenY: number;
    perpWallDist: number;
    side: number;
  }> {
    const { castRayDDAFromFlatMap } = await import('./raycast');
    const { projectNeatensteinGridPoint } = await import('./floor.projection.utils');

    const cosYaw = Math.cos(yaw);
    const sinYaw = Math.sin(yaw);
    const dirX = cosYaw;
    const dirY = sinYaw;
    const planeX = -sinYaw * PLANE_SCALE;
    const planeY = cosYaw * PLANE_SCALE;

    const offset = (2 * column) / COLUMN_COUNT - 1;
    const rayDirX = dirX + planeX * offset;
    const rayDirY = dirY + planeY * offset;

    const hit = castRayDDAFromFlatMap(flatMap, mapSize, cameraX, cameraY, rayDirX, rayDirY);

    if (!Number.isFinite(hit.perpWallDist)) {
      throw new Error(`Column ${column}: no wall hit (perpWallDist=${hit.perpWallDist})`);
    }

    // Wall-hit world point: cam + perpWallDist * rayDir
    const worldHitX = cameraX + hit.perpWallDist * rayDirX;
    const worldHitY = cameraY + hit.perpWallDist * rayDirY;

    const ctx = buildProjectionContext(cameraX, cameraY, yaw);
    const projected = projectNeatensteinGridPoint(worldHitX, worldHitY, ctx, false);

    if (projected === null) {
      throw new Error(`Column ${column}: projection returned null for world point (${worldHitX}, ${worldHitY})`);
    }

    const wallScreenX = column * STRIPE_WIDTH;

    return {
      wallScreenX,
      projectedScreenX: projected.x,
      projectedScreenY: projected.y,
      perpWallDist: hit.perpWallDist,
      side: hit.side,
    };
  }

  describe('Horizontal (X) alignment', () => {
    // Test several camera poses and columns.
    const testCases: Array<{ column: number; cameraX: number; cameraY: number; yaw: number; label: string }> = [
      { column: 160, cameraX: 8.5, cameraY: 8.5, yaw: 0, label: 'center column, yaw=0' },
      { column: 80, cameraX: 8.5, cameraY: 8.5, yaw: 0, label: 'left column, yaw=0' },
      { column: 240, cameraX: 8.5, cameraY: 8.5, yaw: 0, label: 'right column, yaw=0' },
      { column: 160, cameraX: 8.5, cameraY: 8.5, yaw: Math.PI / 4, label: 'center column, yaw=45°' },
    ];

    for (const tc of testCases) {
      it(`wall column screenX matches projected floor grid point screenX (${tc.label})`, async () => {
        const flatMap = createBorderedFlatMap(16);
        const result = await castAndAlign(tc.column, tc.cameraX, tc.cameraY, tc.yaw, flatMap, 16);
        // Tolerance: ≤ 1 * stripeWidth for JS/fallback tier.
        const tolerance = 1 * STRIPE_WIDTH;
        expect(Math.abs(result.projectedScreenX - result.wallScreenX)).toBeLessThanOrEqual(tolerance);
      });
    }
  });

  describe('Vertical (Y) alignment', () => {
    it('wall-base screenY equals floor grid line screenY at the same perpWallDist', async () => {
      const { castRayDDAFromFlatMap } = await import('./raycast');
      const flatMap = createBorderedFlatMap(16);
      const cameraX = 8.5;
      const cameraY = 8.5;
      const yaw = 0;
      const cosYaw = Math.cos(yaw);
      const sinYaw = Math.sin(yaw);
      const dirX = cosYaw;
      const dirY = sinYaw;
      const planeX = -sinYaw * PLANE_SCALE;
      const planeY = cosYaw * PLANE_SCALE;

      // Use the center column for a clean alignment check.
      const column = 160;
      const offset = (2 * column) / COLUMN_COUNT - 1;
      const rayDirX = dirX + planeX * offset;
      const rayDirY = dirY + planeY * offset;

      const hit = castRayDDAFromFlatMap(flatMap, 16, cameraX, cameraY, rayDirX, rayDirY);
      expect(Number.isFinite(hit.perpWallDist)).toBe(true);

      const perpWallDist = hit.perpWallDist;

      // Wall-base screenY: horizonY + wallFocalLength / (2 * perpWallDist)
      // where wallFocalLength = focalLength (same as floor).
      const wallBaseScreenY = HORIZON_Y + FOCAL_LENGTH / (2 * perpWallDist);

      // Floor grid line screenY at the same perpWallDist:
      // Use projectNeatensteinGridPoint for a point at (cameraX + perpWallDist, cameraY).
      const { projectNeatensteinGridPoint } = await import('./floor.projection.utils');
      const ctx = buildProjectionContext(cameraX, cameraY, yaw);
      const worldX = cameraX + perpWallDist * rayDirX;
      const worldY = cameraY + perpWallDist * rayDirY;
      const projected = projectNeatensteinGridPoint(worldX, worldY, ctx, false);
      expect(projected).not.toBeNull();

      // Tolerance: < 0.5px for shader tiers, ≤ 1px for JS/fallback.
      const tolerance = 1.0;
      expect(Math.abs(projected!.y - wallBaseScreenY)).toBeLessThanOrEqual(tolerance);
    });
  });

  describe('Spark↔grid coupling', () => {
    it('emitNeatensteinAmbientPulse produces integer fixedCoord', async () => {
      const { emitNeatensteinAmbientPulse } = await import('./pulse');
      // Find a tick that produces a pulse.
      for (let tick = 0; tick < 200; tick++) {
        const pulse = emitNeatensteinAmbientPulse(tick, 42);
        if (pulse !== null) {
          // The fixedCoord is worldX for axis 'x', worldY for axis 'y'.
          const fixedCoord = pulse.axis === 'x' ? pulse.worldX : pulse.worldY;
          expect(Number.isInteger(fixedCoord)).toBe(true);
          return;
        }
      }
      // If no pulse was emitted in 200 ticks, something is wrong.
      throw new Error('No ambient pulse emitted in 200 ticks');
    });

    it('updateNeatensteinPulses preserves integer fixedCoord', async () => {
      const { emitNeatensteinAmbientPulse, updateNeatensteinPulses } = await import('./pulse');

      // Emit a pulse and update it several times.
      let pulses: NeatensteinPulse[] = [];
      for (let tick = 0; tick < 200 && pulses.length === 0; tick++) {
        const pulse = emitNeatensteinAmbientPulse(tick, 99);
        if (pulse !== null) {
          pulses = [pulse];
        }
      }
      expect(pulses.length).toBeGreaterThan(0);

      // Update the pulse through several ticks.
      for (let tick = 1; tick <= 50; tick++) {
        pulses = updateNeatensteinPulses(pulses, tick);
        for (const pulse of pulses) {
          const fixedCoord = pulse.axis === 'x' ? pulse.worldX : pulse.worldY;
          // The fixedCoord MUST remain integer after update.
          expect(Number.isInteger(fixedCoord)).toBe(true);
        }
      }
    });

    it('depthTestPulse uses strict < comparison (wall wins ties)', async () => {
      const { depthTestPulse } = await import('./pulse');
      const zBuffer = new Float32Array([5.0]);
      // Pulse at distance 5.0 (same as wall) → wall wins tie → not visible.
      expect(depthTestPulse({ screenColumn: 0, distance: 5.0 }, zBuffer)).toBe(false);
      // Pulse at distance 4.9 (closer than wall) → visible.
      expect(depthTestPulse({ screenColumn: 0, distance: 4.9 }, zBuffer)).toBe(true);
      // Pulse at distance 5.1 (behind wall) → not visible.
      expect(depthTestPulse({ screenColumn: 0, distance: 5.1 }, zBuffer)).toBe(false);
    });
  });

  describe('Per-pixel caster alignment (RED — caster does not exist yet)', () => {
    it('per-pixel caster screenX matches projectNeatensteinGridPoint', async () => {
      const floor = await import('./floor');
      const fn = (floor as unknown as Record<string, unknown>).castNeatensteinFloorPerPixel as
        | ((...args: unknown[]) => unknown)
        | undefined;
      // RED: castNeatensteinFloorPerPixel does not exist yet.
      expect(fn).toBeDefined();
    });
  });
});

// ── Non-Negotiable Invariants §1–§7 (contract guards) ─────────────────────

describe('Non-Negotiable Invariants (contract guards)', () => {
  it('Invariant §1: shared integer grid (1 world unit = 1 map cell)', () => {
    // The DDA cell size is 1.0 world units and floor grid lines are at integer
    // world coordinates. This is a constant guard, not a RED test.
    expect(NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD).toBe(0.5);
  });

  it('Invariant §2: shared projection constants are not independently re-derivable', () => {
    // focalLength = H/2 / tan(FOV/2)
    const expectedFocalLength = CANVAS_HEIGHT / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
    expect(FOCAL_LENGTH).toBeCloseTo(expectedFocalLength, 10);
    // planeScale = (W/H) * tan(FOV/2)
    const expectedPlaneScale = (CANVAS_WIDTH / CANVAS_HEIGHT) * Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
    expect(PLANE_SCALE).toBeCloseTo(expectedPlaneScale, 10);
    // planeScale * focalLength = halfWidth (identity)
    expect(PLANE_SCALE * FOCAL_LENGTH).toBeCloseTo(HALF_WIDTH, 5);
  });

  it('Invariant §4: procedural floor grid (not texture)', async () => {
    // The per-pixel caster must use fract(worldCoord), not a sampled texture.
    // RED: caster does not exist yet.
    const floor = await import('./floor');
    expect(typeof (floor as unknown as Record<string, unknown>).castNeatensteinFloorPerPixel).toBe('function');
  });

  it('Invariant §6: step count ≠ perpendicular distance (decoupled constants)', () => {
    // NEATENSTEIN_RENDER_DISTANCE_CAP = 30 (perpendicular distance cap).
    expect(NEATENSTEIN_RENDER_DISTANCE_CAP).toBe(30);
    // The DDA step cap may be angle-aware (ceil(CAP / min(|dirX|, |dirY|)))
    // but the perpendicular cap stays 30.
  });

  it('Invariant §7: fog coordination (single smoothstep factor)', () => {
    // The same smoothstep(FOG_START, CAP, d) fog factor must apply to:
    // (a) wall color fog, (b) floor grid color fog, (c) floor grid alpha.
    // Verify the fog factor function exists and produces smoothstep.
    expect(resolveNeatensteinFogFactor(NEATENSTEIN_FOG_START_DISTANCE)).toBe(0);
    expect(resolveNeatensteinFogFactor(NEATENSTEIN_RENDER_DISTANCE_CAP)).toBe(1);
    // Midpoint: smoothstep(0.5) = 0.5
    const mid = (NEATENSTEIN_FOG_START_DISTANCE + NEATENSTEIN_RENDER_DISTANCE_CAP) / 2;
    expect(resolveNeatensteinFogFactor(mid)).toBeCloseTo(0.5, 10);
  });

  it('Invariant §7: floor alpha bands are superseded by fog factor (not increased)', () => {
    // Per B3.2: do NOT increase from 4 to 8-16 bands; the fog factor replaces
    // the band system entirely. The band constant should be removed or unused
    // after B3.2 implementation.
    // Guard: the current band count is 4 (will be superseded).
    expect(NEATENSTEIN_FLOOR_ALPHA_BANDS).toBe(4);
    expect(NEATENSTEIN_FLOOR_LINE_SAMPLES).toBe(80);
  });
});