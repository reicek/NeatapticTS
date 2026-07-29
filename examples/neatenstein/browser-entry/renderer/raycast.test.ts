import { describe, expect, it } from '@jest/globals';
import { buildNeatensteinMap, castRayDDAFromFlatMap } from './raycast';

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
});
