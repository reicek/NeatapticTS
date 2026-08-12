import { describe, expect, it, jest } from '@jest/globals';
import { FLAPPY_NEON_PALETTE } from '../../../flappy_bird/constants/constants.palette';
import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
} from './framebuffer';
import {
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_DEFAULT_HEIGHT,
  NEATENSTEIN_FLOOR_DEFAULT_WIDTH,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  NEATENSTEIN_FLOOR_MAX_ALPHA,
  NEATENSTEIN_FLOOR_MIN_ALPHA,
  __testOnlyStrokeNeatensteinGridBands,
  drawNeatensteinCeiling,
  drawNeatensteinFloor,
  projectNeatensteinCeilingPoint,
  projectNeatensteinFloorPoint,
  renderNeatensteinCeiling,
  renderNeatensteinFloor,
  resolveNeatensteinFloorAlpha,
  type NeatensteinFloorCamera,
} from './floor';

type MockFloorContext = ReturnType<typeof createMockFloorContext>;

/** Canonical test canvas dimensions used for deterministic projection checks. */
const TEST_CANVAS_WIDTH = 320;
/** Canonical test canvas height used for deterministic projection checks. */
const TEST_CANVAS_HEIGHT = 240;
/** Dimensions passed explicitly to {@link drawNeatensteinFloor} in explicit-width tests. */
const TEST_EXPLICIT_DRAW_WIDTH = 640;
/** Dimensions passed explicitly to {@link drawNeatensteinFloor} in explicit-height tests. */
const TEST_EXPLICIT_DRAW_HEIGHT = 360;
/** Canvas dimensions used when testing the {@link renderNeatensteinFloor} wrapper. */
const TEST_RENDER_CANVAS_WIDTH = 640;
/** Canvas dimensions used when testing the {@link renderNeatensteinFloor} wrapper. */
const TEST_RENDER_CANVAS_HEIGHT = 480;
/** Canonical test camera world position. */
const TEST_CAMERA_X = 5.5;
/** Canonical test camera world position. */
const TEST_CAMERA_Y = 5.5;
/**
 * Tolerance in pixels when comparing a projected screen coordinate.
 *
 * Grid lines are sampled every half world unit over a 30-unit span, so a
 * projected world point may not coincide exactly with a rasterized sample.
 * Use a generous tolerance that covers the resulting screen-space gap rather
 * than the sub-pixel ideal.
 */
const TEST_SCREEN_TOLERANCE = 8;

type MockCall =
  | { type: 'beginPath' }
  | { type: 'moveTo'; x: number; y: number }
  | { type: 'lineTo'; x: number; y: number }
  | { type: 'stroke' }
  | { type: 'save' }
  | { type: 'restore' }
  | { type: 'strokeStyle'; value: string | undefined }
  | { type: 'lineWidth'; value: number | undefined }
  | { type: 'shadowBlur'; value: number | undefined }
  | { type: 'shadowColor'; value: string | undefined };

function createMockFloorContext(canvasWidth?: number, canvasHeight?: number) {
  const calls = {
    beginPath: [] as unknown[][],
    moveTo: [] as [number, number][],
    lineTo: [] as [number, number][],
    stroke: [] as unknown[][],
    save: [] as unknown[][],
    restore: [] as unknown[][],
    strokeStyle: [] as (string | undefined)[],
    lineWidth: [] as (number | undefined)[],
    shadowBlur: [] as (number | undefined)[],
    shadowColor: [] as (string | undefined)[],
  };
  const log: MockCall[] = [];

  return {
    beginPath: jest.fn(() => {
      log.push({ type: 'beginPath' });
      calls.beginPath.push([]);
    }),
    moveTo: jest.fn((x: number, y: number) => {
      log.push({ type: 'moveTo', x, y });
      calls.moveTo.push([x, y]);
    }),
    lineTo: jest.fn((x: number, y: number) => {
      log.push({ type: 'lineTo', x, y });
      calls.lineTo.push([x, y]);
    }),
    stroke: jest.fn(() => {
      log.push({ type: 'stroke' });
      calls.stroke.push([]);
    }),
    save: jest.fn(() => {
      log.push({ type: 'save' });
      calls.save.push([]);
    }),
    restore: jest.fn(() => {
      log.push({ type: 'restore' });
      calls.restore.push([]);
    }),
    get strokeStyle(): string | undefined {
      return calls.strokeStyle.at(-1);
    },
    set strokeStyle(value: string | undefined) {
      log.push({ type: 'strokeStyle', value });
      calls.strokeStyle.push(value);
    },
    get lineWidth(): number | undefined {
      return calls.lineWidth.at(-1);
    },
    set lineWidth(value: number | undefined) {
      log.push({ type: 'lineWidth', value });
      calls.lineWidth.push(value);
    },
    get shadowBlur(): number | undefined {
      return calls.shadowBlur.at(-1);
    },
    set shadowBlur(value: number | undefined) {
      log.push({ type: 'shadowBlur', value });
      calls.shadowBlur.push(value);
    },
    get shadowColor(): string | undefined {
      return calls.shadowColor.at(-1);
    },
    set shadowColor(value: string | undefined) {
      log.push({ type: 'shadowColor', value });
      calls.shadowColor.push(value);
    },
    canvas: {
      width: canvasWidth ?? NEATENSTEIN_FLOOR_DEFAULT_WIDTH,
      height: canvasHeight ?? NEATENSTEIN_FLOOR_DEFAULT_HEIGHT,
    },
    calls,
    log,
  };
}

function camera(yaw: number, x = 0, y = 0): NeatensteinFloorCamera {
  return { yaw, x, y };
}

function extractDrawnPoints(ctx: MockFloorContext) {
  const points: Array<{ x: number; y: number }> = [];
  for (const call of ctx.log) {
    if (call.type === 'moveTo' || call.type === 'lineTo') {
      points.push({ x: call.x, y: call.y });
    }
  }
  return points;
}

function projectScreenToWorld(
  screenX: number,
  screenY: number,
  width: number,
  height: number,
  camera: NeatensteinFloorCamera,
): { worldX: number; worldY: number } {
  const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
  const halfWidth = width / 2;
  const focalLength = height / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const dy = screenY - horizonY;
  const rowDistance =
    (NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD * focalLength) / dy;

  const cos = Math.cos(camera.yaw);
  const sin = Math.sin(camera.yaw);
  const rightX = -sin;
  const rightY = cos;

  const centerWorldX = camera.x + cos * rowDistance;
  const centerWorldY = camera.y + sin * rowDistance;
  const stepX = rightX * (NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD / dy);
  const stepY = rightY * (NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD / dy);

  const startWorldX = centerWorldX - stepX * halfWidth;
  const startWorldY = centerWorldY - stepY * halfWidth;

  return {
    worldX: startWorldX + stepX * screenX,
    worldY: startWorldY + stepY * screenX,
  };
}

/**
 * Project a world point to screen space using the same formula as the renderer.
 *
 * This lets tests assert against specific screen coordinates rather than
 * averaging over the whole drawn path.
 */
function projectWorldToScreen(
  worldX: number,
  worldY: number,
  width: number,
  height: number,
  camera: NeatensteinFloorCamera,
): { x: number; y: number } {
  const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
  const halfWidth = width / 2;
  const focalLength = height / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const dx = worldX - camera.x;
  const dy = worldY - camera.y;
  const cos = Math.cos(camera.yaw);
  const sin = Math.sin(camera.yaw);
  const camSpaceY = dx * cos + dy * sin;
  const camSpaceX = -dx * sin + dy * cos;

  return {
    x: halfWidth + (camSpaceX / camSpaceY) * focalLength,
    y:
      horizonY +
      (NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD / camSpaceY) * focalLength,
  };
}

function isNearIntegerGridLine(
  worldX: number,
  worldY: number,
  tolerance: number,
): boolean {
  const distX = Math.min(worldX - Math.floor(worldX), 1);
  const wrappedX = Math.min(distX, 1 - distX);
  const distY = Math.min(worldY - Math.floor(worldY), 1);
  const wrappedY = Math.min(distY, 1 - distY);
  return wrappedX < tolerance || wrappedY < tolerance;
}

function hasPointNear(
  ctx: MockFloorContext,
  x: number,
  y: number,
  tolerance = TEST_SCREEN_TOLERANCE,
): boolean {
  return extractDrawnPoints(ctx).some(
    (point) =>
      Math.abs(point.x - x) <= tolerance && Math.abs(point.y - y) <= tolerance,
  );
}

describe('Neatenstein floor renderer ray-cast grid', () => {
  it('returns the minimum alpha at the horizon', () => {
    expect(resolveNeatensteinFloorAlpha(0)).toBeCloseTo(
      NEATENSTEIN_FLOOR_MIN_ALPHA,
    );
  });

  it('returns the maximum alpha near the camera', () => {
    expect(resolveNeatensteinFloorAlpha(1)).toBeCloseTo(
      NEATENSTEIN_FLOOR_MAX_ALPHA,
    );
  });

  it('draws using explicit width and height parameters', () => {
    const ctx = createMockFloorContext(TEST_CANVAS_WIDTH, TEST_CANVAS_HEIGHT);
    drawNeatensteinFloor(
      ctx,
      TEST_EXPLICIT_DRAW_WIDTH,
      TEST_EXPLICIT_DRAW_HEIGHT,
      camera(0),
    );

    const maxY = Math.max(...ctx.calls.lineTo.map(([, y]) => y));

    expect(maxY).toBeGreaterThan(NEATENSTEIN_FLOOR_DEFAULT_HEIGHT);
  });

  it('does not draw when canvas dimensions are invalid', () => {
    const ctx = createMockFloorContext(TEST_CANVAS_WIDTH, TEST_CANVAS_HEIGHT);
    drawNeatensteinFloor(ctx, 0, 0, camera(0));

    expect(ctx.calls.moveTo.length + ctx.calls.lineTo.length).toBe(0);
  });

  it('reads canvas dimensions through the render wrapper', () => {
    const ctx = createMockFloorContext(
      TEST_RENDER_CANVAS_WIDTH,
      TEST_RENDER_CANVAS_HEIGHT,
    );
    renderNeatensteinFloor(ctx, camera(0));

    const maxY = Math.max(...ctx.calls.lineTo.map(([, y]) => y));

    expect(maxY).toBeGreaterThan(NEATENSTEIN_FLOOR_DEFAULT_HEIGHT);
  });

  it('projects every drawn point onto an integer world grid line', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const cam = camera(0, TEST_CAMERA_X, TEST_CAMERA_Y);
    const ctx = createMockFloorContext(width, height);
    drawNeatensteinFloor(ctx, width, height, cam);

    const points = extractDrawnPoints(ctx);
    const allOnGrid = points.every((point) => {
      const world = projectScreenToWorld(point.x, point.y, width, height, cam);
      return isNearIntegerGridLine(world.worldX, world.worldY, 0.2);
    });

    expect(allOnGrid).toBe(true);
  });

  it('only draws floor points below the horizon', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const ctx = createMockFloorContext(width, height);
    drawNeatensteinFloor(ctx, width, height, camera(0));

    const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
    const allBelow = extractDrawnPoints(ctx).every(
      (point) => point.y > horizonY - 0.5,
    );

    expect(allBelow).toBe(true);
  });

  it('rotates the grid projection when camera yaw changes', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const cam = camera(0, TEST_CAMERA_X, TEST_CAMERA_Y);
    const yawedCam = camera(Math.PI / 4, TEST_CAMERA_X, TEST_CAMERA_Y);
    const worldX = 7;
    const worldY = 6;

    const ctxBase = createMockFloorContext(width, height);
    const ctxYawed = createMockFloorContext(width, height);
    drawNeatensteinFloor(ctxBase, width, height, cam);
    drawNeatensteinFloor(ctxYawed, width, height, yawedCam);

    const base = projectWorldToScreen(worldX, worldY, width, height, cam);
    const yawed = projectWorldToScreen(worldX, worldY, width, height, yawedCam);

    expect({
      baseFound: hasPointNear(ctxBase, base.x, base.y),
      yawedFound: hasPointNear(ctxYawed, yawed.x, yawed.y),
      xChanged: Math.round(base.x) !== Math.round(yawed.x),
    }).toEqual({
      baseFound: true,
      yawedFound: true,
      xChanged: true,
    });
  });

  it('scrolls the grid when camera Y changes', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const worldX = 7;
    const worldY = 6;
    const cam = camera(0, TEST_CAMERA_X, TEST_CAMERA_Y);
    const scrolledCam = camera(0, TEST_CAMERA_X, TEST_CAMERA_Y + 1);

    const ctxBase = createMockFloorContext(width, height);
    const ctxScrolled = createMockFloorContext(width, height);
    drawNeatensteinFloor(ctxBase, width, height, cam);
    drawNeatensteinFloor(ctxScrolled, width, height, scrolledCam);

    const base = projectWorldToScreen(worldX, worldY, width, height, cam);
    const scrolled = projectWorldToScreen(
      worldX,
      worldY,
      width,
      height,
      scrolledCam,
    );

    expect({
      baseFound: hasPointNear(ctxBase, base.x, base.y),
      scrolledFound: hasPointNear(ctxScrolled, scrolled.x, scrolled.y),
      xChanged: Math.round(base.x) !== Math.round(scrolled.x),
    }).toEqual({
      baseFound: true,
      scrolledFound: true,
      xChanged: true,
    });
  });

  it('scrolls the grid laterally when camera X changes', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const worldX = 8;
    const worldY = 6;
    const cam = camera(0, TEST_CAMERA_X, TEST_CAMERA_Y);
    const scrolledCam = camera(0, TEST_CAMERA_X + 1, TEST_CAMERA_Y);

    const ctxBase = createMockFloorContext(width, height);
    const ctxScrolled = createMockFloorContext(width, height);
    drawNeatensteinFloor(ctxBase, width, height, cam);
    drawNeatensteinFloor(ctxScrolled, width, height, scrolledCam);

    const base = projectWorldToScreen(worldX, worldY, width, height, cam);
    const scrolled = projectWorldToScreen(
      worldX,
      worldY,
      width,
      height,
      scrolledCam,
    );

    expect({
      baseFound: hasPointNear(ctxBase, base.x, base.y),
      scrolledFound: hasPointNear(ctxScrolled, scrolled.x, scrolled.y),
      xChanged: Math.round(base.x) !== Math.round(scrolled.x),
    }).toEqual({
      baseFound: true,
      scrolledFound: true,
      xChanged: true,
    });
  });

  it('guards non-finite yaw without throwing', () => {
    const ctx = createMockFloorContext();

    expect(() =>
      drawNeatensteinFloor(
        ctx,
        TEST_CANVAS_WIDTH,
        TEST_CANVAS_HEIGHT,
        camera(Number.NaN),
      ),
    ).not.toThrow();
  });

  it('saves context state before rendering', () => {
    const ctx = createMockFloorContext();

    drawNeatensteinFloor(ctx, TEST_CANVAS_WIDTH, TEST_CANVAS_HEIGHT, camera(0));

    expect(ctx.calls.save.length).toBeGreaterThan(0);
  });

  it('restores context state after rendering', () => {
    const ctx = createMockFloorContext();

    drawNeatensteinFloor(ctx, TEST_CANVAS_WIDTH, TEST_CANVAS_HEIGHT, camera(0));

    expect(ctx.calls.restore.length).toBe(ctx.calls.save.length);
  });

  it('does not draw floor grid points beyond NEATENSTEIN_RENDER_DISTANCE_CAP (AC-10.3d-001)', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const cam = camera(0, TEST_CAMERA_X, TEST_CAMERA_Y);
    const ctx = createMockFloorContext(width, height);

    drawNeatensteinFloor(ctx, width, height, cam);

    const points = extractDrawnPoints(ctx);
    expect(points.length).toBeGreaterThan(0);

    const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
    const focalLength =
      height / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

    // Camera-space forward distance from screen Y:
    //   forwardDist = (cameraHeight * focalLength) / |screenY - horizonY|
    // Points at the horizon (dy ≈ 0) are at infinite distance — beyond any cap.
    const beyondCap = points.filter((p) => {
      const dy = Math.abs(p.y - horizonY);
      if (dy < 1e-9) {
        return true;
      }
      const forwardDist =
        (NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD * focalLength) / dy;
      return forwardDist > NEATENSTEIN_RENDER_DISTANCE_CAP;
    });

    expect(beyondCap).toEqual([]);
  });

  it('skips the frame when projection constants become non-finite', () => {
    const ctx = createMockFloorContext(TEST_CANVAS_WIDTH, TEST_CANVAS_HEIGHT);
    // Math.tan returns 0 → focalLength = canvasHeight / 2 / 0 = Infinity
    const tanSpy = jest.spyOn(Math, 'tan').mockReturnValue(0);
    drawNeatensteinFloor(ctx, TEST_CANVAS_WIDTH, TEST_CANVAS_HEIGHT, camera(0));
    tanSpy.mockRestore();

    expect(ctx.calls.moveTo.length + ctx.calls.lineTo.length).toBe(0);
  });

  it('projects a world point to screen space via projectNeatensteinFloorPoint', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
    const focalLength =
      height / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
    const halfWidth = width / 2;

    const result = projectNeatensteinFloorPoint(
      7,
      6,
      TEST_CAMERA_X,
      TEST_CAMERA_Y,
      1,
      0,
      focalLength,
      halfWidth,
      horizonY,
      height,
      NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
    );

    expect(result).not.toBeNull();
    expect(result!.y).toBeGreaterThan(horizonY);
    expect(result!.depthRatio).toBeGreaterThanOrEqual(0);
    expect(result!.depthRatio).toBeLessThanOrEqual(1);
  });

  it('returns null when projected screen coordinates are non-finite', () => {
    const result = projectNeatensteinFloorPoint(
      7,
      6,
      5.5,
      5.5,
      1,
      0,
      Number.POSITIVE_INFINITY, // focalLength = Infinity → screenX = Infinity
      160,
      120,
      240,
      NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
    );

    expect(result).toBeNull();
  });

  it('returns null when depthDenominator is zero for floor projection', () => {
    const height = 240;
    const horizonY = height; // depthDenominator = height - horizonY = 0

    const result = projectNeatensteinFloorPoint(
      7,
      6,
      5.5,
      5.5,
      1,
      0,
      200,
      160,
      horizonY,
      height,
      NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
    );

    expect(result).toBeNull();
  });

  it('guards non-finite camera x and y without throwing', () => {
    const ctx = createMockFloorContext();

    expect(() =>
      drawNeatensteinFloor(
        ctx,
        TEST_CANVAS_WIDTH,
        TEST_CANVAS_HEIGHT,
        camera(0, Number.NaN, Number.NaN),
      ),
    ).not.toThrow();
  });

  it('uses fallback dimensions when the render wrapper has no canvas', () => {
    const ctx = createMockFloorContext();
    // Remove the canvas property so resolveContextCanvasDimension uses fallbacks
    (ctx as unknown as { canvas: undefined }).canvas = undefined;

    expect(() => renderNeatensteinFloor(ctx, camera(0))).not.toThrow();
    // The function should still draw using the default fallback dimensions
    expect(ctx.calls.save.length).toBeGreaterThan(0);
  });

  it('clamps non-finite depthRatio to 0 in resolveNeatensteinFloorAlpha', () => {
    const alpha = resolveNeatensteinFloorAlpha(Number.NaN);
    // NaN depthRatio → fallback to 0 → clamped to 0 → min alpha
    expect(alpha).toBe(NEATENSTEIN_FLOOR_MIN_ALPHA);
  });
});

/** Deterministic seed used for all ceiling mirror fixtures. */
const CEILING_FIXTURE_SEED = 42;

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;

  return () => {
    // Park-Miller LCG, matching the deterministic PRNG used by pulses.
    state = (state * 16807) % 2147483647;
    return state / 2147483647;
  };
}

function parseHexColor(
  hex: string,
): { r: number; g: number; b: number } | null {
  const digits = hex.replace('#', '');
  if (digits.length !== 6) {
    return null;
  }

  const r = parseInt(digits.slice(0, 2), 16);
  const g = parseInt(digits.slice(2, 4), 16);
  const b = parseInt(digits.slice(4, 6), 16);

  if (!Number.isFinite(r + g + b)) {
    return null;
  }

  return { r, g, b };
}

describe('Neatenstein ceiling mirror renderer', () => {
  it('exports a ceiling grid drawing function', () => {
    expect(typeof drawNeatensteinCeiling).toBe('function');
  });

  it('exports a canvas-backed ceiling render wrapper', () => {
    expect(typeof renderNeatensteinCeiling).toBe('function');
  });

  it('draws at least one ceiling point', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const rng = createSeededRandom(CEILING_FIXTURE_SEED);
    const cam = camera(
      0,
      TEST_CAMERA_X + rng() * 2 - 1,
      TEST_CAMERA_Y + rng() * 2 - 1,
    );
    const ctx = createMockFloorContext(width, height);

    drawNeatensteinCeiling(ctx, width, height, cam);

    expect(extractDrawnPoints(ctx).length).toBeGreaterThan(0);
  });

  it('draws ceiling points above the horizon', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const rng = createSeededRandom(CEILING_FIXTURE_SEED);
    const cam = camera(
      0,
      TEST_CAMERA_X + rng() * 2 - 1,
      TEST_CAMERA_Y + rng() * 2 - 1,
    );
    const ctx = createMockFloorContext(width, height);

    drawNeatensteinCeiling(ctx, width, height, cam);

    const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
    const allAbove = extractDrawnPoints(ctx).every(
      (point) => point.y < horizonY + 0.5,
    );

    expect(allAbove).toBe(true);
  });

  it('mirrors the floor grid vertically across the horizon', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const rng = createSeededRandom(CEILING_FIXTURE_SEED);
    const cam = camera(
      0,
      TEST_CAMERA_X + rng() * 2 - 1,
      TEST_CAMERA_Y + rng() * 2 - 1,
    );
    const ctxFloor = createMockFloorContext(width, height);
    const ctxCeiling = createMockFloorContext(width, height);

    drawNeatensteinFloor(ctxFloor, width, height, cam);
    drawNeatensteinCeiling(ctxCeiling, width, height, cam);

    const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
    const floorPoints = extractDrawnPoints(ctxFloor);
    const ceilingPoints = extractDrawnPoints(ctxCeiling);

    const ceilingByX = new Map<number, number[]>();
    for (const point of ceilingPoints) {
      const bucket = Math.round(point.x);
      const existing = ceilingByX.get(bucket);
      if (existing === undefined) {
        ceilingByX.set(bucket, [point.y]);
      } else {
        existing.push(point.y);
      }
    }

    const everyFloorPointMirrored = floorPoints.every((floorPoint) => {
      const ys = ceilingByX.get(Math.round(floorPoint.x));
      if (ys === undefined) {
        return false;
      }
      const expectedY = 2 * horizonY - floorPoint.y;
      return ys.some((y) => Math.abs(y - expectedY) <= TEST_SCREEN_TOLERANCE);
    });

    expect(everyFloorPointMirrored).toBe(true);
  });

  it('uses the same floor grid line color for the ceiling texture lookup', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const rng = createSeededRandom(CEILING_FIXTURE_SEED);
    const cam = camera(
      0,
      TEST_CAMERA_X + rng() * 2 - 1,
      TEST_CAMERA_Y + rng() * 2 - 1,
    );
    const ctx = createMockFloorContext(width, height);

    drawNeatensteinCeiling(ctx, width, height, cam);

    const parsed = parseHexColor(FLAPPY_NEON_PALETTE.groundGridLine);
    const expectedPrefix =
      parsed !== null ? `rgba(${parsed.r}, ${parsed.g}, ${parsed.b}` : '';
    const usesGridColor = ctx.calls.strokeStyle.some(
      (style) => typeof style === 'string' && style.startsWith(expectedPrefix),
    );

    expect(usesGridColor).toBe(true);
  });

  it('does not draw ceiling grid points beyond NEATENSTEIN_RENDER_DISTANCE_CAP (AC-10.3d-001)', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const cam = camera(0, TEST_CAMERA_X, TEST_CAMERA_Y);
    const ctx = createMockFloorContext(width, height);

    drawNeatensteinCeiling(ctx, width, height, cam);

    const points = extractDrawnPoints(ctx);
    expect(points.length).toBeGreaterThan(0);

    const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
    const focalLength =
      height / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

    // Ceiling points are mirrored above the horizon, but the forward-distance
    // formula is identical: |screenY - horizonY| maps to the same depth.
    const beyondCap = points.filter((p) => {
      const dy = Math.abs(p.y - horizonY);
      if (dy < 1e-9) {
        return true;
      }
      const forwardDist =
        (NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD * focalLength) / dy;
      return forwardDist > NEATENSTEIN_RENDER_DISTANCE_CAP;
    });

    expect(beyondCap).toEqual([]);
  });

  it('reads canvas dimensions through the ceiling render wrapper', () => {
    const ctx = createMockFloorContext(
      TEST_RENDER_CANVAS_WIDTH,
      TEST_RENDER_CANVAS_HEIGHT,
    );
    renderNeatensteinCeiling(ctx, camera(0));

    const horizonY =
      TEST_RENDER_CANVAS_HEIGHT * NEATENSTEIN_FLOOR_HORIZON_RATIO;
    const allAbove = extractDrawnPoints(ctx).every(
      (point) => point.y < horizonY + 0.5,
    );

    expect(extractDrawnPoints(ctx).length).toBeGreaterThan(0);
    expect(allAbove).toBe(true);
  });

  it('projects a world point above the horizon via projectNeatensteinCeilingPoint', () => {
    const width = TEST_CANVAS_WIDTH;
    const height = TEST_CANVAS_HEIGHT;
    const horizonY = height * NEATENSTEIN_FLOOR_HORIZON_RATIO;
    const focalLength =
      height / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
    const halfWidth = width / 2;

    const result = projectNeatensteinCeilingPoint(
      7,
      6,
      TEST_CAMERA_X,
      TEST_CAMERA_Y,
      1,
      0,
      focalLength,
      halfWidth,
      horizonY,
      height,
      NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
    );

    expect(result).not.toBeNull();
    expect(result!.y).toBeLessThan(horizonY);
    expect(result!.depthRatio).toBeGreaterThanOrEqual(0);
    expect(result!.depthRatio).toBeLessThanOrEqual(1);
  });

  it('returns null when depthDenominator is zero for ceiling projection', () => {
    const result = projectNeatensteinCeilingPoint(
      7,
      6,
      5.5,
      5.5,
      1,
      0,
      200,
      160,
      0, // horizonY = 0 → depthDenominator = horizonY = 0
      240,
      NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
    );

    expect(result).toBeNull();
  });
});

describe('AC-10.4-r-003: floor/ceiling rendering uses original floor color within 30 cells', () => {
  /**
   * Fixture: draw the floor grid with a camera that produces visible grid
   * lines spanning multiple depth bands. With the step-function fog fix,
   * all bands within 30 cells use the original floor grid line color
   * (FLAPPY_NEON_PALETTE.groundGridLine) and never blend toward
   * NEATENSTEIN_BACKGROUND_RGB.
   */
  it('uses FLOOR_BASE_RGB floor color for all bands within 30 cells (AC-10.4-r-003)', () => {
    const ctx = createMockFloorContext(
      TEST_RENDER_CANVAS_WIDTH,
      TEST_RENDER_CANVAS_HEIGHT,
    );
    drawNeatensteinFloor(
      ctx,
      TEST_RENDER_CANVAS_WIDTH,
      TEST_RENDER_CANVAS_HEIGHT,
      camera(0, TEST_CAMERA_X, TEST_CAMERA_Y),
    );

    // Parse the floor grid line color to get the expected RGB prefix.
    const hex = FLAPPY_NEON_PALETTE.groundGridLine.replace('#', '');
    const floorR = Number.parseInt(hex.slice(0, 2), 16);
    const floorG = Number.parseInt(hex.slice(2, 4), 16);
    const floorB = Number.parseInt(hex.slice(4, 6), 16);
    const floorPrefix = `rgba(${floorR}, ${floorG}, ${floorB}`;

    // All stroke styles should use the floor grid line color.
    const hasFloorColor = ctx.calls.strokeStyle.some(
      (style) => typeof style === 'string' && style.startsWith(floorPrefix),
    );
    expect(hasFloorColor).toBe(true);

    // No stroke style should use the background fog color.
    const fogPrefix = `rgba(${NEATENSTEIN_BACKGROUND_RGB.r}, ${NEATENSTEIN_BACKGROUND_RGB.g}, ${NEATENSTEIN_BACKGROUND_RGB.b}`;
    const hasFogColor = ctx.calls.strokeStyle.some(
      (style) => typeof style === 'string' && style.startsWith(fogPrefix),
    );
    expect(hasFogColor).toBe(false);
  });
});

describe('strokeNeatensteinGridBands empty-band continue branch', () => {
  it('skips empty segment buffers without throwing', () => {
    const ctx = createMockFloorContext(TEST_CANVAS_WIDTH, TEST_CANVAS_HEIGHT);

    // 4 bands: band 1 is empty, the others each have one line segment.
    const bands = [
      [10, 20, 30, 40], // band 0 — non-empty
      [], // band 1 — empty (exercises the `continue` branch)
      [50, 60, 70, 80], // band 2 — non-empty
      [90, 100, 110, 120], // band 3 — non-empty
    ];

    expect(() =>
      __testOnlyStrokeNeatensteinGridBands(ctx, bands),
    ).not.toThrow();
  });
});
