import { describe, expect, it, jest } from '@jest/globals';
import {
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_DEFAULT_HEIGHT,
  NEATENSTEIN_FLOOR_DEFAULT_WIDTH,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  NEATENSTEIN_FLOOR_MAX_ALPHA,
  NEATENSTEIN_FLOOR_MIN_ALPHA,
  drawNeatensteinFloor,
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
 * Grid lines are sampled every half world unit over a 40-unit span, so a
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
  const focalLength = halfWidth / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
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
  const focalLength = halfWidth / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
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
});
