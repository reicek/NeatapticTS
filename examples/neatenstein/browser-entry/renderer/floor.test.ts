import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_FLOOR_DEFAULT_HEIGHT,
  NEATENSTEIN_FLOOR_DEFAULT_WIDTH,
  NEATENSTEIN_FLOOR_MAX_ALPHA,
  NEATENSTEIN_FLOOR_MIN_ALPHA,
  NEATENSTEIN_FLOOR_ROW_COUNT,
  renderNeatensteinFloor,
  resolveNeatensteinFloorAlpha,
  type NeatensteinFloorCamera,
} from './floor';

type MockFloorContext = ReturnType<typeof createMockFloorContext>;

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

function parseAlphaFromStyle(style: string | undefined): number {
  if (style === undefined) {
    return 0;
  }
  const channels = style.split(',');
  const alpha = channels.at(-1)?.replace(')', '').trim();
  return Number.parseFloat(alpha ?? '0');
}

function averageLineToX(ctx: MockFloorContext) {
  const lineToCalls = ctx.calls.lineTo;
  if (lineToCalls.length === 0) return 0;
  const sum = lineToCalls.reduce((acc, [x]) => acc + x, 0);
  return sum / lineToCalls.length;
}

function extractHorizontalLines(ctx: MockFloorContext, width: number) {
  const lines: Array<{ y: number; alpha: number }> = [];
  let currentStyle: string | undefined;

  for (let index = 0; index < ctx.log.length; index += 1) {
    const call = ctx.log[index];

    if (call.type === 'strokeStyle') {
      currentStyle = call.value;
    }

    if (call.type !== 'beginPath') {
      continue;
    }

    const moveTo = ctx.log[index + 1];
    const lineTo = ctx.log[index + 2];
    const afterLine = ctx.log[index + 3];

    if (
      moveTo?.type === 'moveTo' &&
      lineTo?.type === 'lineTo' &&
      afterLine?.type === 'stroke' &&
      moveTo.x === 0 &&
      lineTo.x === width &&
      moveTo.y === lineTo.y
    ) {
      lines.push({ y: moveTo.y, alpha: parseAlphaFromStyle(currentStyle) });

      // The renderer draws the transverse depth rows first, followed by
      // longitudinal world-axis grid lines. Some of those grid lines can also
      // appear as full-width horizontal strokes when they are parallel to the
      // screen plane, so stop after collecting the depth rows.
      if (lines.length >= NEATENSTEIN_FLOOR_ROW_COUNT) {
        break;
      }
    }
  }

  return lines;
}

function extractVerticalSegments(ctx: MockFloorContext, width: number) {
  const segments: Array<{
    startX: number;
    startY: number;
    endX: number;
    endY: number;
  }> = [];

  for (let index = 0; index < ctx.log.length; index += 1) {
    if (ctx.log[index].type !== 'beginPath') {
      continue;
    }

    const points: Array<{ x: number; y: number }> = [];
    for (
      let innerIndex = index + 1;
      innerIndex < ctx.log.length;
      innerIndex += 1
    ) {
      const innerCall = ctx.log[innerIndex];
      if (innerCall.type === 'stroke') {
        break;
      }
      if (innerCall.type === 'moveTo' || innerCall.type === 'lineTo') {
        points.push({ x: innerCall.x, y: innerCall.y });
      }
    }

    for (let pointIndex = 0; pointIndex < points.length - 1; pointIndex += 1) {
      const start = points[pointIndex];
      const end = points[pointIndex + 1];
      const isFullWidthHorizontal =
        start.x === 0 && end.x === width && start.y === end.y;
      if (!isFullWidthHorizontal) {
        segments.push({
          startX: start.x,
          startY: start.y,
          endX: end.x,
          endY: end.y,
        });
      }
    }
  }

  return segments;
}

function camera(yaw: number): NeatensteinFloorCamera {
  return { yaw, x: 0, y: 0 };
}

describe('Neatenstein floor renderer perspective grid', () => {
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

  it('renders horizontal depth bands below the horizon', () => {
    const ctx = createMockFloorContext();
    renderNeatensteinFloor(ctx, camera(0));

    const horizonY = ctx.canvas.height * 0.5;
    const horizontalLines = extractHorizontalLines(ctx, ctx.canvas.width);
    const belowHorizon = horizontalLines.filter((line) => line.y > horizonY);

    expect(belowHorizon.length).toBeGreaterThan(0);
  });

  it('renders vertical grid segments between adjacent depth rows', () => {
    const ctx = createMockFloorContext();
    renderNeatensteinFloor(ctx, camera(0));

    const segments = extractVerticalSegments(ctx, ctx.canvas.width);

    expect(segments.length).toBeGreaterThan(0);
  });

  it('spacing between horizontal rows grows toward the camera', () => {
    const ctx = createMockFloorContext();
    renderNeatensteinFloor(ctx, camera(0));

    const lines = extractHorizontalLines(ctx, ctx.canvas.width);
    const gaps = lines.slice(1).map((line, index) => line.y - lines[index].y);

    expect(gaps.at(-1)).toBeGreaterThan(gaps[0] ?? 0);
  });

  it('does not collapse the entire floor into a single point', () => {
    const ctx = createMockFloorContext();
    renderNeatensteinFloor(ctx, camera(0));

    const uniqueYs = new Set(ctx.calls.lineTo.map(([, y]) => y));

    expect(uniqueYs.size).toBeGreaterThan(NEATENSTEIN_FLOOR_ROW_COUNT / 2);
  });

  it('shifts the projection horizontally with camera yaw', () => {
    const ctxZero = createMockFloorContext();
    const ctxYawed = createMockFloorContext();

    renderNeatensteinFloor(ctxZero, camera(0));
    renderNeatensteinFloor(ctxYawed, camera(Math.PI / 4));

    expect(averageLineToX(ctxZero)).not.toBe(averageLineToX(ctxYawed));
  });

  it('uses the runtime canvas dimensions', () => {
    const ctx = createMockFloorContext(640, 480);
    renderNeatensteinFloor(ctx, camera(0));

    const maxY = Math.max(...ctx.calls.lineTo.map(([, y]) => y));

    expect(maxY).toBeGreaterThan(NEATENSTEIN_FLOOR_DEFAULT_HEIGHT);
  });

  it('guards non-finite yaw without throwing', () => {
    const ctx = createMockFloorContext();

    expect(() => renderNeatensteinFloor(ctx, camera(Number.NaN))).not.toThrow();
  });

  it('wraps yaw outside [-π, π] to an equivalent angle', () => {
    const ctxWrapped = createMockFloorContext();
    const ctxUnwrapped = createMockFloorContext();

    renderNeatensteinFloor(ctxWrapped, camera(Math.PI));
    renderNeatensteinFloor(ctxUnwrapped, camera(3 * Math.PI));

    expect(averageLineToX(ctxWrapped)).toBeCloseTo(
      averageLineToX(ctxUnwrapped),
    );
  });

  it('wraps context state mutations in save and restore', () => {
    const ctx = createMockFloorContext();

    renderNeatensteinFloor(ctx, camera(0));

    expect(ctx.calls.save.length).toBeGreaterThan(0);
    expect(ctx.calls.restore.length).toBe(ctx.calls.save.length);
  });
});
