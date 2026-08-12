import { describe, expect, it } from '@jest/globals';
import {
  buildNeatensteinMap,
  castRayDDAFromFlatMap,
  hasLineOfSight,
} from './raycast';

const SIDE = 8;

function createClosedFlatGrid(): Uint8Array {
  const flatMap = new Uint8Array(SIDE * SIDE);
  for (let x = 0; x < SIDE; x++) {
    for (let y = 0; y < SIDE; y++) {
      if (x === 0 || x === SIDE - 1 || y === 0 || y === SIDE - 1) {
        flatMap[y * SIDE + x] = 1;
      }
    }
  }
  return flatMap;
}

function createOpenPerimeterGrid(side: number): Uint8Array {
  const flatMap = new Uint8Array(side * side);
  for (let x = 0; x < side; x += 1) {
    for (let y = 0; y < side; y += 1) {
      if (x === 0 || x === side - 1 || y === 0 || y === side - 1) {
        flatMap[y * side + x] = 1;
      }
    }
  }
  return flatMap;
}

describe('Neatenstein raycast helpers', () => {
  it('returns perpWallDist, side, and hit cell coordinates', () => {
    const result = castRayDDAFromFlatMap(
      createClosedFlatGrid(),
      SIDE,
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

  it('hits the east wall cell with side 0', () => {
    const result = castRayDDAFromFlatMap(
      createClosedFlatGrid(),
      SIDE,
      1.5,
      1.5,
      1,
      0,
    );
    expect({ mapX: result.mapX, mapY: result.mapY, side: result.side }).toEqual(
      {
        mapX: SIDE - 1,
        mapY: 1,
        side: 0,
      },
    );
  });

  it('returns side 1 for a north-facing hit', () => {
    const result = castRayDDAFromFlatMap(
      createClosedFlatGrid(),
      SIDE,
      1.5,
      1.5,
      0,
      -1,
    );
    expect(result.side).toBe(1);
  });

  it('builds a Uint8Array map', () => {
    const map = buildNeatensteinMap(12345);
    expect(map).toBeInstanceOf(Uint8Array);
  });

  it('builds a map of the fixed 120x120 size', () => {
    const map = buildNeatensteinMap(12345);
    expect(map.length).toBe(120 * 120);
  });

  it('produces deterministic maps for the same seed', () => {
    const first = buildNeatensteinMap(12345);
    const second = buildNeatensteinMap(12345);
    expect(Array.from(second)).toEqual(Array.from(first));
  });

  it('hits the perimeter of the generated 120x120 map', () => {
    const flatMap = buildNeatensteinMap(12345);
    const result = castRayDDAFromFlatMap(flatMap, 120, 60.5, 60.5, 1, 0);
    expect({
      positive: result.perpWallDist > 0,
      finite: Number.isFinite(result.perpWallDist),
    }).toEqual({ positive: true, finite: true });
  });

  it('caps DDA traversal at 30 cells and returns Infinity for long open sight lines', () => {
    const side = 64;
    const result = castRayDDAFromFlatMap(
      createOpenPerimeterGrid(side),
      side,
      32.5,
      32.5,
      1,
      0,
    );
    expect(Number.isFinite(result.perpWallDist)).toBe(false);
    expect(result.perpWallDist).toBe(Infinity);
  });
});

describe('hasLineOfSight', () => {
  it('returns true when no wall cell intersects the DDA ray between from and to', () => {
    // 8×8 grid with a closed perimeter; interior cells are open.
    const grid = createClosedFlatGrid();
    // Two interior points on the same row with no wall between them.
    expect(
      hasLineOfSight(grid, SIDE, { x: 1.5, y: 1.5 }, { x: 6.5, y: 1.5 }),
    ).toBe(true);
  });

  it('returns true for a diagonal path through open interior cells', () => {
    const grid = createClosedFlatGrid();
    expect(
      hasLineOfSight(grid, SIDE, { x: 1.5, y: 1.5 }, { x: 6.5, y: 6.5 }),
    ).toBe(true);
  });

  it('returns false when a wall cell occludes the path', () => {
    // Build an 8×8 grid with a closed perimeter and a wall at column 4,
    // rows 1–6 (interior wall splitting the arena).
    const grid = new Uint8Array(SIDE * SIDE);
    for (let x = 0; x < SIDE; x++) {
      for (let y = 0; y < SIDE; y++) {
        if (x === 0 || x === SIDE - 1 || y === 0 || y === SIDE - 1) {
          grid[y * SIDE + x] = 1;
        }
      }
    }
    // Add an interior wall at x=4, y=1..6
    for (let y = 1; y < SIDE - 1; y++) {
      grid[y * SIDE + 4] = 1;
    }
    // From left of the wall to right of the wall — wall at x=4 occludes.
    expect(
      hasLineOfSight(grid, SIDE, { x: 1.5, y: 3.5 }, { x: 6.5, y: 3.5 }),
    ).toBe(false);
  });

  it('returns true when from and to are the same position', () => {
    const grid = createClosedFlatGrid();
    expect(
      hasLineOfSight(grid, SIDE, { x: 3.5, y: 3.5 }, { x: 3.5, y: 3.5 }),
    ).toBe(true);
  });

  it('returns false when the destination is behind a perimeter wall', () => {
    const grid = createClosedFlatGrid();
    // From inside the arena toward a point outside the perimeter wall.
    // The ray hits the east perimeter wall (x=7) before reaching x=10.
    expect(
      hasLineOfSight(grid, SIDE, { x: 1.5, y: 1.5 }, { x: 10, y: 1.5 }),
    ).toBe(false);
  });
});
