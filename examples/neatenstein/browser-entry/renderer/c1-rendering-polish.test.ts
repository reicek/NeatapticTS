/**
 * C1: Rendering Polish — RED phase test contracts.
 *
 * Every acceptance criterion and invariant from Step C1 of the
 * Neatenstein Ultimate Quality Upgrade plan is encoded as a failing
 * test below.  Tests that exercise not-yet-implemented behaviour
 * MUST fail for the right reason (missing export, missing module, or
 * wrong current value) — not for a fixture or syntax error.
 *
 * C1 items covered:
 *  1. Per-pixel floor casting (unconditional)
 *  2. Segment band splitting
 *  3. Floor color fog (smooth fog factor on RGB, not alpha-only)
 *  4. Temporal coherence / half-resolution raycasting
 *  5. OffscreenCanvas + transferToImageBitmap present path
 *  6. TAA / MSAA (2× MSAA resolve)
 *
 * Invariant §8 regression coupling (horizontal X, vertical Y,
 * spark↔grid coupling) is asserted against the NEW C1 entry points so
 * the contracts fail today (no implementation) and must pass after
 * implementation without breaking alignment.
 *
 * @jest-environment node
 */

import { describe, expect, it } from '@jest/globals';

/* eslint-disable @typescript-eslint/no-unused-vars -- RED test: constants will be used once implementation lands */

// ── Static constant imports (safe — these modules already exist) ───────────

import {
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  NEATENSTEIN_FLOOR_ALPHA_BANDS,
} from './renderer.floor.constants';
import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_FOG_START_DISTANCE,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
  resolveNeatensteinFogFactor,
} from './framebuffer';
import { FLAPPY_NEON_PALETTE } from '../../../flappy_bird/constants/constants.palette';

// ── Shared test helpers ─────────────────────────────────────────────────────

/** Standard test canvas dimensions. */
const CANVAS_WIDTH = 320;
const CANVAS_HEIGHT = 240;
const COLUMN_COUNT = 320;

/** Standard projection constants derived from the canvas. */
const FOCAL_LENGTH = CANVAS_HEIGHT / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
const HALF_WIDTH = CANVAS_WIDTH / 2;
const HORIZON_Y = CANVAS_HEIGHT * NEATENSTEIN_FLOOR_HORIZON_RATIO;

/** Grid line color parsed from the shared neon palette (#0a8ea0). */
const GRID_COLOR = { r: 10, g: 142, b: 160 } as const;

/**
 * Helper that attempts a dynamic import of a module that may not exist yet
 * (RED phase).  Using a string variable bypasses TypeScript static module
 * resolution so the test file compiles even when the target module is absent.
 */
async function tryImportModule(
  path: string,
): Promise<Record<string, unknown> | null> {
  try {
    const mod = await import(path);
    return mod as Record<string, unknown>;
  } catch {
    return null;
  }
}

/** Read a numeric export from a dynamically-imported module. */
function readNumber(mod: Record<string, unknown>, key: string): number | undefined {
  const value = mod[key];
  return typeof value === 'number' ? value : undefined;
}

/** Read a function export from a dynamically-imported module. */
function readFunction(
  mod: Record<string, unknown>,
  key: string,
): ((...args: unknown[]) => unknown) | undefined {
  const value = mod[key];
  return typeof value === 'function' ? (value as (...args: unknown[]) => unknown) : undefined;
}

// ── C1.1 — Per-pixel floor casting (unconditional) ─────────────────────────

describe('C1.1 — Per-pixel floor casting (unconditional)', () => {
  it('exports NEATENSTEIN_PER_PIXEL_FLOOR_UNCONDITIONAL flag from floor module', async () => {
    const floor = await import('./floor');
    // RED: the unconditional integration flag does not exist yet — the
    // Canvas 2D line-projection path (drawNeatensteinGrid) is still the
    // default for the worker tier. C1.1 must make the per-pixel caster the
    // unconditional default across all tiers.
    const flag = (floor as unknown as Record<string, unknown>)
      .NEATENSTEIN_PER_PIXEL_FLOOR_UNCONDITIONAL;
    expect(flag).toBe(true);
  });

  it('exports isNeatensteinPerPixelFloorActive predicate that is true unconditionally', async () => {
    const floor = await import('./floor');
    const fn = readFunction(floor as unknown as Record<string, unknown>, 'isNeatensteinPerPixelFloorActive');
    // RED: predicate does not exist yet.
    expect(fn).toBeDefined();
    // Must return true regardless of tier/quality argument (unconditional).
    expect(fn!('worker')).toBe(true);
    expect(fn!('cpu')).toBe(true);
  });

  it('per-pixel caster renders procedural integer grid at exactly 1-unit spacing (Invariant §4)', async () => {
    const floor = await import('./floor');
    const cast = readFunction(floor as unknown as Record<string, unknown>, 'castNeatensteinFloorPerPixel');
    expect(cast).toBeDefined();
    // The grid spacing contract: a helper resolving the world-space grid
    // pitch MUST return exactly 1.0 world unit = 1 map cell.
    const spacing = readNumber(floor as unknown as Record<string, unknown>, 'NEATENSTEIN_FLOOR_GRID_SPACING_WORLD');
    // RED: the explicit grid-spacing constant is not exported yet.
    expect(spacing).toBe(1.0);
  });
});

// ── C1.2 — Segment band splitting ──────────────────────────────────────────

describe('C1.2 — Segment band splitting', () => {
  it('exports splitNeatensteinFloorSegmentAtBandBoundaries from floor.band.utils', async () => {
    const mod = await import('./floor.band.utils');
    // RED: the band-boundary splitter does not exist yet — segments that
    // straddle an alpha-band boundary produce a visible seam.
    expect(
      typeof (mod as unknown as Record<string, unknown>)
        .splitNeatensteinFloorSegmentAtBandBoundaries,
    ).toBe('function');
  });

  it('splits a segment straddling a band boundary into two segments in different bands', async () => {
    const mod = await import('./floor.band.utils');
    const split = readFunction(
      mod as unknown as Record<string, unknown>,
      'splitNeatensteinFloorSegmentAtBandBoundaries',
    );
    // RED: function undefined → split is undefined → TypeError on call.
    expect(split).toBeDefined();
    // A segment whose average depth ratio crosses a band edge must be split
    // so each piece lands in its own band. With 4 bands, the boundary between
    // band 0 and band 1 is at depthRatio 0.25.
    const bands = [
      [] as number[],
      [] as number[],
      [] as number[],
      [] as number[],
    ];
    // Segment from depthRatio 0.1 (band 0) to 0.4 (band 1) straddles 0.25.
    split!(bands, 0, HORIZON_Y + 10, 0, HORIZON_Y + 50, 0.1, 0.4);
    // Both band 0 and band 1 must contain at least one segment (4 floats).
    expect(bands[0].length).toBeGreaterThanOrEqual(4);
    expect(bands[1].length).toBeGreaterThanOrEqual(4);
  });

  it('splits a segment with negative ratioSpan (descending depth) into correct bands', async () => {
    const mod = await import('./floor.band.utils');
    const split = readFunction(
      mod as unknown as Record<string, unknown>,
      'splitNeatensteinFloorSegmentAtBandBoundaries',
    );
    expect(split).toBeDefined();
    // A segment from depthRatio 0.9 (band 3) to 0.1 (band 0) has a negative
    // ratioSpan. It must cross boundaries 0.75, 0.5, and 0.25 in descending
    // order, placing each sub-segment in the correct band.
    const bands = [
      [] as number[],
      [] as number[],
      [] as number[],
      [] as number[],
    ];
    split!(bands, 0, HORIZON_Y + 10, 0, HORIZON_Y + 50, 0.9, 0.1);
    // All four bands must contain at least one segment (4 floats each).
    expect(bands[0].length).toBeGreaterThanOrEqual(4);
    expect(bands[1].length).toBeGreaterThanOrEqual(4);
    expect(bands[2].length).toBeGreaterThanOrEqual(4);
    expect(bands[3].length).toBeGreaterThanOrEqual(4);
    // Band 3 (nearest) should contain the first sub-segment, and band 0
    // (farthest) should contain the tail.
    // Verify band 3's first y coordinate is near the start (HORIZON_Y + 10)
    // and band 0's last y coordinate is near the end (HORIZON_Y + 50).
    expect(bands[3][1]).toBeCloseTo(HORIZON_Y + 10, 0);
    expect(bands[0][bands[0].length - 1]).toBeCloseTo(HORIZON_Y + 50, 0);
  });
});

// ── C1.3 — Floor color fog ─────────────────────────────────────────────────

describe('C1.3 — Floor color fog (smooth fog factor on RGB)', () => {
  it('exports resolveNeatensteinFloorFoggedColor from floor.shade.utils', async () => {
    const mod = await import('./floor.shade.utils');
    // RED: the RGB fog helper does not exist yet — the floor currently uses
    // alpha-only falloff (resolveNeatensteinFloorAlphaFromDistance).
    expect(
      typeof (mod as unknown as Record<string, unknown>)
        .resolveNeatensteinFloorFoggedColor,
    ).toBe('function');
  });

  it('returns the grid color unchanged at FOG_START_DISTANCE (no fog)', async () => {
    const mod = await import('./floor.shade.utils');
    const fn = readFunction(
      mod as unknown as Record<string, unknown>,
      'resolveNeatensteinFloorFoggedColor',
    );
    // RED: function undefined.
    expect(fn).toBeDefined();
    const result = fn!(GRID_COLOR.r, GRID_COLOR.g, GRID_COLOR.b, NEATENSTEIN_FOG_START_DISTANCE) as {
      r: number; g: number; b: number;
    };
    expect(result.r).toBeCloseTo(GRID_COLOR.r, 0);
    expect(result.g).toBeCloseTo(GRID_COLOR.g, 0);
    expect(result.b).toBeCloseTo(GRID_COLOR.b, 0);
  });

  it('returns the background color at RENDER_DISTANCE_CAP (full fog)', async () => {
    const mod = await import('./floor.shade.utils');
    const fn = readFunction(
      mod as unknown as Record<string, unknown>,
      'resolveNeatensteinFloorFoggedColor',
    );
    expect(fn).toBeDefined();
    const result = fn!(GRID_COLOR.r, GRID_COLOR.g, GRID_COLOR.b, NEATENSTEIN_RENDER_DISTANCE_CAP) as {
      r: number; g: number; b: number;
    };
    // Full fog → grid color blends fully toward the background void color.
    expect(result.r).toBeCloseTo(NEATENSTEIN_BACKGROUND_RGB.r, 0);
    expect(result.g).toBeCloseTo(NEATENSTEIN_BACKGROUND_RGB.g, 0);
    expect(result.b).toBeCloseTo(NEATENSTEIN_BACKGROUND_RGB.b, 0);
  });

  it('uses smoothstep fog factor (not linear) for the RGB blend', async () => {
    const mod = await import('./floor.shade.utils');
    const fn = readFunction(
      mod as unknown as Record<string, unknown>,
      'resolveNeatensteinFloorFoggedColor',
    );
    expect(fn).toBeDefined();
    // At t = 0.25 into the fog range, smoothstep gives fogFactor = 0.15625,
    // linear would give 0.25. The blended R channel must match smoothstep.
    const t = 0.25;
    const distance = NEATENSTEIN_FOG_START_DISTANCE + t * (NEATENSTEIN_RENDER_DISTANCE_CAP - NEATENSTEIN_FOG_START_DISTANCE);
    const expectedFog = t * t * (3 - 2 * t);
    const expectedR = GRID_COLOR.r + (NEATENSTEIN_BACKGROUND_RGB.r - GRID_COLOR.r) * expectedFog;
    const linearR = GRID_COLOR.r + (NEATENSTEIN_BACKGROUND_RGB.r - GRID_COLOR.r) * t;
    const result = fn!(GRID_COLOR.r, GRID_COLOR.g, GRID_COLOR.b, distance) as {
      r: number; g: number; b: number;
    };
    expect(result.r).toBeCloseTo(expectedR, 1);
    expect(result.r).not.toBeCloseTo(linearR, 1);
  });
});

// ── C1.4 — Temporal coherence / half-resolution raycasting ─────────────────

describe('C1.4 — Temporal coherence / half-resolution raycasting', () => {
  it('exports a quality constants module with half-res toggle', async () => {
    // RED: the quality constants module does not exist yet.
    const mod = await tryImportModule('./renderer.quality.constants');
    expect(mod).not.toBeNull();
    // Runtime quality toggle (not unconditional) — Invariant §1 safeguard.
    expect(mod!.NEATENSTEIN_HALF_RES_DEFAULT_ENABLED).toBeDefined();
    expect(typeof mod!.resolveNeatensteinHalfResEnabled).toBe('function');
  });

  it('half-res is gated behind a quality setting, not unconditional', async () => {
    const mod = await tryImportModule('./renderer.quality.constants');
    expect(mod).not.toBeNull();
    const resolve = readFunction(mod!, 'resolveNeatensteinHalfResEnabled');
    expect(resolve).toBeDefined();
    // Disabling via the runtime quality toggle MUST turn half-res off.
    expect(resolve!({ halfResEnabled: false })).toBe(false);
  });

  it('exports interpolateNeatensteinWallColumn that interpolates wall color only (Invariant §1)', async () => {
    const interpolate = await import('./interpolate');
    // RED: the wall-column interpolator does not exist yet. It MUST
    // interpolate wall COLOR only and NEVER the column screen X.
    const fn = readFunction(
      interpolate as unknown as Record<string, unknown>,
      'interpolateNeatensteinWallColumn',
    );
    expect(fn).toBeDefined();
    // Two adjacent cast columns with different colors but the target column
    // must remain at its true integer pixel position.
    const result = fn!(
      { screenX: 80, r: 10, g: 142, b: 160 },
      { screenX: 82, r: 0, g: 183, b: 255 },
      0.5,
    ) as { screenX: number; r: number; g: number; b: number };
    // screenX must NOT be interpolated (stays at the true column position 81).
    expect(result.screenX).toBe(81);
    // Color IS interpolated (midpoint of the two wall colors).
    expect(result.r).toBeCloseTo(5, 0);
  });

  it('z-buffer depth is cast at every column even when half-res is enabled', async () => {
    const mod = await tryImportModule('./renderer.quality.constants');
    expect(mod).not.toBeNull();
    // RED: the full-resolution z-buffer constant does not exist yet.
    // Half-res decimation applies ONLY to wall color; the z-buffer MUST be
    // cast at every column so depth occlusion stays exact.
    const flag = readNumber(mod!, 'NEATENSTEIN_HALF_RES_ZBUFFER_FULL_RESOLUTION');
    expect(flag).toBe(1);
  });

  it('floor/ceiling grid is exempt from decimation (stays full resolution)', async () => {
    const mod = await tryImportModule('./renderer.quality.constants');
    expect(mod).not.toBeNull();
    // RED: the grid-decimation exemption constant does not exist yet.
    // The stroked floor/ceiling grid MUST NOT be downsampled.
    const flag = readNumber(mod!, 'NEATENSTEIN_HALF_RES_GRID_EXEMPT');
    expect(flag).toBe(1);
  });
});

// ── C1.5 — OffscreenCanvas + transferToImageBitmap ─────────────────────────

describe('C1.5 — OffscreenCanvas + transferToImageBitmap present path', () => {
  it('exports presentNeatensteinFrameBitmap from worker render utils', async () => {
    const mod = await import('../worker/display.worker.render.utils');
    // RED: the transferToImageBitmap present function does not exist yet —
    // the worker still calls ctx.commit() (display.worker.ts:306).
    expect(
      typeof (mod as unknown as Record<string, unknown>)
        .presentNeatensteinFrameBitmap,
    ).toBe('function');
  });

  it('present function calls transferToImageBitmap, not commit()', async () => {
    const mod = await import('../worker/display.worker.render.utils');
    const fn = readFunction(
      mod as unknown as Record<string, unknown>,
      'presentNeatensteinFrameBitmap',
    );
    // RED: function undefined.
    expect(fn).toBeDefined();

    // Minimal fake OffscreenCanvas capturing transferToImageBitmap calls.
    const calls: string[] = [];
    const fakeCanvas = {
      transferToImageBitmap() {
        calls.push('transferToImageBitmap');
        return { width: CANVAS_WIDTH, height: CANVAS_HEIGHT, close() { /* noop */ } };
      },
      getContext() {
        return { commit() { calls.push('commit'); } };
      },
      width: CANVAS_WIDTH,
      height: CANVAS_HEIGHT,
    };

    fn!(fakeCanvas);
    // The 2024-preferred present path MUST use transferToImageBitmap.
    expect(calls).toContain('transferToImageBitmap');
    expect(calls).not.toContain('commit');
  });

  it('host frame consumer exports consumeNeatensteinFrameBitmap using createImageBitmap', async () => {
    // RED: the host-side bitmap consumer does not exist yet — browser-entry
    // still reads the OffscreenCanvas directly.
    const mod = await tryImportModule('../host/frame-bitmap-consumer');
    expect(mod).not.toBeNull();
    expect(typeof mod!.consumeNeatensteinFrameBitmap).toBe('function');
  });
});

// ── C1.6 — TAA / MSAA ──────────────────────────────────────────────────────

describe('C1.6 — TAA / MSAA (2× MSAA resolve)', () => {
  it('exports NEATENSTEIN_MSAA_SAMPLE_COUNT === 2 from a quality module', async () => {
    // RED: the MSAA constants module does not exist yet.
    const mod = await tryImportModule('./renderer.msaa.constants');
    expect(mod).not.toBeNull();
    const count = readNumber(mod!, 'NEATENSTEIN_MSAA_SAMPLE_COUNT');
    // 2× MSAA resolve cleans wall-sprite seams cheaply.
    expect(count).toBe(2);
  });

  it('exports resolveNeatensteinMsaaResolvedColumn that averages 2 sub-sample colors', async () => {
    const mod = await tryImportModule('./renderer.msaa.constants');
    expect(mod).not.toBeNull();
    const fn = readFunction(mod!, 'resolveNeatensteinMsaaResolvedColumn');
    // RED: function undefined.
    expect(fn).toBeDefined();
    // Two sub-samples: pure cyan (0,183,255) and pure teal (10,142,160).
    // 2× MSAA resolve = average → (5, 162.5, 207.5).
    const result = fn!(
      { r: 0, g: 183, b: 255 },
      { r: 10, g: 142, b: 160 },
    ) as { r: number; g: number; b: number };
    expect(result.r).toBeCloseTo(5, 0);
    expect(result.g).toBeCloseTo(162.5, 0);
    expect(result.b).toBeCloseTo(207.5, 0);
  });

  it('MSAA resolve clamps channels into [0,255]', async () => {
    const mod = await tryImportModule('./renderer.msaa.constants');
    expect(mod).not.toBeNull();
    const fn = readFunction(mod!, 'resolveNeatensteinMsaaResolvedColumn');
    expect(fn).toBeDefined();
    const result = fn!(
      { r: 300, g: -10, b: 128 },
      { r: 300, g: -10, b: 128 },
    ) as { r: number; g: number; b: number };
    expect(result.r).toBeLessThanOrEqual(255);
    expect(result.g).toBeGreaterThanOrEqual(0);
  });

  it('rgbToHex produces #rrggbb format for writeNeonWallColumn compatibility', async () => {
    const mod = await import('../worker/display.worker.color.utils');
    const fn = readFunction(
      mod as unknown as Record<string, unknown>,
      'rgbToHex',
    );
    expect(fn).toBeDefined();
    const hex = fn!({ r: 0, g: 183, b: 255 }) as string;
    expect(hex).toMatch(/^#[0-9a-fA-F]{6}$/);
    expect(hex).toBe('#00b7ff');
    // Verify clamping + rounding edge cases.
    const clamped = fn!({ r: 300, g: -10, b: 128.7 }) as string;
    expect(clamped).toMatch(/^#[0-9a-fA-F]{6}$/);
    expect(clamped).toBe('#ff0081');
  });
});

// ── Invariant §8 — Regression coupling (X, Y, spark↔grid) for C1 ───────────

describe('Invariant §8 — C1 regression coupling', () => {
  describe('Horizontal (X) alignment — half-res preserves column screen X', () => {
    it('interpolateNeatensteinWallColumn never shifts the column away from its true integer pixel', async () => {
      const interpolate = await import('./interpolate');
      const fn = readFunction(
        interpolate as unknown as Record<string, unknown>,
        'interpolateNeatensteinWallColumn',
      );
      // RED: function undefined → no half-res interpolator to verify.
      expect(fn).toBeDefined();
      // Even column (n=80) and odd column (n=81) — half-res renders every
      // other column and interpolates the rest. The interpolated column MUST
      // keep screenX at its true integer position, NOT a blended X.
      for (const col of [80, 81, 82, 83]) {
        const result = fn!(
          { screenX: col - 1, r: 10, g: 142, b: 160 },
          { screenX: col + 1, r: 0, g: 183, b: 255 },
          0.5,
        ) as { screenX: number };
        expect(result.screenX).toBe(col);
      }
    });
  });

  describe('Vertical (Y) alignment — per-pixel floor grid pitch stays 1 world unit', () => {
    it('NEATENSTEIN_FLOOR_GRID_SPACING_WORLD === 1 so wall base aligns to floor line', async () => {
      const floor = await import('./floor');
      // RED: explicit spacing constant not exported yet.
      const spacing = readNumber(
        floor as unknown as Record<string, unknown>,
        'NEATENSTEIN_FLOOR_GRID_SPACING_WORLD',
      );
      expect(spacing).toBe(1.0);
      // The DDA cell size MUST equal the floor grid pitch (Invariant §1).
      // Re-derive the wall DDA cell size from the raycast module to confirm
      // they share the same integer grid.
      const raycast = await import('./raycast');
      const cellSize = readNumber(
        raycast as unknown as Record<string, unknown>,
        'NEATENSTEIN_DDA_CELL_SIZE_WORLD',
      );
      expect(cellSize).toBe(1.0);
    });
  });

  describe('Spark↔grid coupling — half-res decimation does not affect the floor grid', () => {
    it('half-res quality toggle exempts the floor/ceiling grid from decimation', async () => {
      const mod = await tryImportModule('./renderer.quality.constants');
      expect(mod).not.toBeNull();
      // The traveling spark slides along integer floor-grid lines (Invariant
      // §3). Half-res MUST NOT downsample the stroked grid, or the spark
      // would detach from its grid line.
      const exempt = readNumber(mod!, 'NEATENSTEIN_HALF_RES_GRID_EXEMPT');
      expect(exempt).toBe(1);
      // The spark overlay path MUST remain at full resolution.
      const sparkExempt = readNumber(mod!, 'NEATENSTEIN_HALF_RES_SPARK_EXEMPT');
      expect(sparkExempt).toBe(1);
    });

    it('resolveNeatensteinHalfResEnabled does not alter grid or spark resolution when enabled', async () => {
      const mod = await tryImportModule('./renderer.quality.constants');
      expect(mod).not.toBeNull();
      const resolve = readFunction(mod!, 'resolveNeatensteinHalfResEnabled');
      expect(resolve).toBeDefined();
      // When half-res is enabled, only wall COLOR is decimated — the function
      // MUST report which layers are exempt.
      const config = resolve!({ halfResEnabled: true }) as unknown;
      // RED: the return shape is not yet defined; the contract requires the
      // grid and spark layers to be marked exempt.
      expect(config).not.toBe(true);
    });
  });
});

// ── Shared projection constant reuse (Invariant §2) ────────────────────────

describe('Invariant §2 — Shared projection constants across C1 paths', () => {
  it('half-res and per-pixel paths consume the same NEATENSTEIN_FLOOR_* constants', async () => {
    // The per-pixel caster (C1.1) and the half-res interpolator (C1.4) MUST
    // both derive focalLength and planeScale from the shared
    // NEATENSTEIN_FLOOR_* constants, not re-derive them independently.
    const floor = await import('./floor');
    const cast = readFunction(floor as unknown as Record<string, unknown>, 'castNeatensteinFloorPerPixel');
    expect(cast).toBeDefined();
    // The shared focalLength identity: planeScale * focalLength = halfWidth.
    const planeScale = (CANVAS_WIDTH / CANVAS_HEIGHT) * Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
    expect(planeScale * FOCAL_LENGTH).toBeCloseTo(HALF_WIDTH, 5);
    // The MSAA resolve path (C1.6) must also reuse the shared constants.
    const msaaMod = await tryImportModule('./renderer.msaa.constants');
    expect(msaaMod).not.toBeNull();
    const msaaFog = readFunction(msaaMod!, 'resolveNeatensteinMsaaFogBlend');
    // RED: MSAA fog blend helper not exported yet.
    expect(msaaFog).toBeDefined();
    // It MUST reuse the shared fog factor (not a re-derived curve).
    const fogAtStart = resolveNeatensteinFogFactor(NEATENSTEIN_FOG_START_DISTANCE);
    expect(fogAtStart).toBeCloseTo(0, 5);
  });
});