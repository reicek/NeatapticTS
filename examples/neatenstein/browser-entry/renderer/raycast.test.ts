import { describe, expect, it } from '@jest/globals';

const loadModule = (path: string): Promise<any> => import(path);

const GRID_WIDTH = 8;
const GRID_HEIGHT = 8;

function createTestGrid(): number[][] {
  const grid: number[][] = [];
  for (let x = 0; x < GRID_WIDTH; x++) {
    grid[x] = [];
    for (let y = 0; y < GRID_HEIGHT; y++) {
      grid[x][y] = 0;
    }
  }
  grid[2][1] = 1;
  grid[1][0] = 1;
  return grid;
}

describe('Neatenstein raycast helpers', () => {
  it('returns perpWallDist, side, and hit cell coordinates', async () => {
    const { castRayDDA } = await loadModule('./raycast.ts');
    const result = castRayDDA(
      createTestGrid(),
      GRID_WIDTH,
      GRID_HEIGHT,
      1.5,
      1.5,
      1,
      0,
    );
    expect({
      hasPerpWallDist: typeof result.perpWallDist === 'number',
      hasSide: typeof result.side === 'number',
      hasMapX: typeof result.mapX === 'number',
      hasMapY: typeof result.mapY === 'number',
    }).toEqual({
      hasPerpWallDist: true,
      hasSide: true,
      hasMapX: true,
      hasMapY: true,
    });
  });

  it('hits the east wall cell with side 0', async () => {
    const { castRayDDA } = await loadModule('./raycast.ts');
    const result = castRayDDA(
      createTestGrid(),
      GRID_WIDTH,
      GRID_HEIGHT,
      1.5,
      1.5,
      1,
      0,
    );
    expect({ mapX: result.mapX, mapY: result.mapY, side: result.side }).toEqual(
      {
        mapX: 2,
        mapY: 1,
        side: 0,
      },
    );
  });

  it('returns side 1 for a north-facing hit', async () => {
    const { castRayDDA } = await loadModule('./raycast.ts');
    const result = castRayDDA(
      createTestGrid(),
      GRID_WIDTH,
      GRID_HEIGHT,
      1.5,
      1.5,
      0,
      -1,
    );
    expect(result.side).toBe(1);
  });

  it('builds a map of the fixed 24x24 size', async () => {
    const { buildNeatensteinMap } = await loadModule('./raycast.ts');
    const map = buildNeatensteinMap(12345);
    expect(map).toBeInstanceOf(Uint8Array);
    expect(map.length).toBe(24 * 24);
  });

  it('produces deterministic maps for the same seed', async () => {
    const { buildNeatensteinMap } = await loadModule('./raycast.ts');
    const first = buildNeatensteinMap(12345);
    const second = buildNeatensteinMap(12345);
    expect(Array.from(second)).toEqual(Array.from(first));
  });
});
