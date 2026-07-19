import { describe, expect, it } from '@jest/globals';
import {
  resolvePlaybackGroundGridDepthCurve,
  resolvePlaybackGroundGridLineAlpha,
  resolvePlaybackGroundGridLineBlur,
  resolvePlaybackGroundGridLineThickness,
} from '../../../flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.math.utils';

const loadModule = (path: string): Promise<any> => import(path);

function createMockFloorContext() {
  const path: Array<{ x: number; y: number }> = [];
  return {
    beginPath: jest.fn(),
    moveTo: jest.fn((x: number, y: number) => {
      path.push({ x, y });
    }),
    lineTo: jest.fn((x: number, y: number) => {
      path.push({ x, y });
    }),
    stroke: jest.fn(),
    path,
  };
}

function averageLineToX(ctx: ReturnType<typeof createMockFloorContext>) {
  const lineToCalls = ctx.lineTo.mock.calls as Array<[number, number]>;
  if (lineToCalls.length === 0) return 0;
  const sum = lineToCalls.reduce((acc, [x]) => acc + x, 0);
  return sum / lineToCalls.length;
}

describe('Neatenstein floor renderer reuses Flappy ground-grid curves', () => {
  it('reuses the Flappy depth power curve exponent', async () => {
    const { resolveNeatensteinFloorDepthCurve } =
      await loadModule('./floor.ts');
    const ratios = [0, 0.25, 0.5, 0.75, 1];
    const neatensteinValues = ratios.map(resolveNeatensteinFloorDepthCurve);
    const flappyValues = ratios.map(resolvePlaybackGroundGridDepthCurve);
    expect(neatensteinValues).toEqual(flappyValues);
  });

  it('reuses the Flappy alpha range', async () => {
    const { resolveNeatensteinFloorAlpha } = await loadModule('./floor.ts');
    const ratios = [0, 0.5, 1];
    expect(ratios.map(resolveNeatensteinFloorAlpha)).toEqual(
      ratios.map(resolvePlaybackGroundGridLineAlpha),
    );
  });

  it('reuses the Flappy blur range', async () => {
    const { resolveNeatensteinFloorBlur } = await loadModule('./floor.ts');
    const ratios = [0, 0.5, 1];
    expect(ratios.map(resolveNeatensteinFloorBlur)).toEqual(
      ratios.map(resolvePlaybackGroundGridLineBlur),
    );
  });

  it('reuses the Flappy thickness range', async () => {
    const { resolveNeatensteinFloorThickness } = await loadModule('./floor.ts');
    const ratios = [0, 0.5, 1];
    expect(ratios.map(resolveNeatensteinFloorThickness)).toEqual(
      ratios.map(resolvePlaybackGroundGridLineThickness),
    );
  });

  it('shifts the perspective-ray vanishing point with camera yaw', async () => {
    const { renderNeatensteinFloor } = await loadModule('./floor.ts');
    const ctxZero = createMockFloorContext();
    const ctxYawed = createMockFloorContext();

    renderNeatensteinFloor(ctxZero, { yaw: 0, x: 1, y: 1 });
    renderNeatensteinFloor(ctxYawed, { yaw: Math.PI / 4, x: 1, y: 1 });

    expect(averageLineToX(ctxZero)).not.toBe(averageLineToX(ctxYawed));
  });
});
