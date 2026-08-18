import { describe, expect, it } from '@jest/globals';
import { buildNeatensteinMap, createCollisionMap } from './map';
import * as mapConstants from './renderer.map.constants';

/** Expected square map dimension for the 120×120 target behavior. */
const EXPECTED_MAP_SIZE = 120;

/** Deterministic seed used for repeatable map fixtures. */
const TEST_SEED = 12345;

/**
 * Count reachable interior floor cells from the center of a generated map.
 *
 * Returns both the total number of interior floor cells and the number that
 * are reachable via 4-directional movement from the center.
 *
 * @param map - Flat row-major map grid.
 * @param side - Width and height of the square grid.
 * @returns Total and reachable interior floor cell counts.
 */
function countReachableInteriorFloorCells(
  map: Uint8Array,
  side: number,
): { total: number; reachable: number } {
  const center = Math.floor(side / 2);
  const startIndex = center * side + center;

  const visited = new Uint8Array(side * side);
  const queue: number[] = [];
  let queueHead = 0;
  let reachable = 0;

  if (map[startIndex] === 0) {
    visited[startIndex] = 1;
    queue.push(startIndex);
  }

  while (queueHead < queue.length) {
    const index = queue[queueHead++];
    reachable++;
    const x = index % side;
    const y = Math.floor(index / side);

    const directions = [
      [0, -1],
      [0, 1],
      [-1, 0],
      [1, 0],
    ];

    for (const [dx, dy] of directions) {
      const nx = x + dx;
      const ny = y + dy;
      const neighborIndex = ny * side + nx;

      if (
        nx < 1 ||
        nx >= side - 1 ||
        ny < 1 ||
        ny >= side - 1 ||
        visited[neighborIndex] ||
        map[neighborIndex] !== 0
      ) {
        continue;
      }

      visited[neighborIndex] = 1;
      queue.push(neighborIndex);
    }
  }

  let total = 0;
  for (let x = 1; x < side - 1; x++) {
    for (let y = 1; y < side - 1; y++) {
      if (map[y * side + x] === 0) {
        total++;
      }
    }
  }

  return { total, reachable };
}

describe('Neatenstein map generation', () => {
  it('returns a Uint8Array of length 120*120', () => {
    const map = buildNeatensteinMap(TEST_SEED);
    expect(map.length).toBe(EXPECTED_MAP_SIZE * EXPECTED_MAP_SIZE);
  });

  it('walls the full outer perimeter for every seed', () => {
    const map = buildNeatensteinMap(TEST_SEED);
    const side = EXPECTED_MAP_SIZE;

    const topWalled = Array.from({ length: side }, (_, x) => x).every(
      (x) => map[x] === 1,
    );
    const bottomWalled = Array.from({ length: side }, (_, x) => x).every(
      (x) => map[(side - 1) * side + x] === 1,
    );
    const leftWalled = Array.from({ length: side }, (_, y) => y).every(
      (y) => map[y * side] === 1,
    );
    const rightWalled = Array.from({ length: side }, (_, y) => y).every(
      (y) => map[y * side + (side - 1)] === 1,
    );

    expect({ topWalled, bottomWalled, leftWalled, rightWalled }).toEqual({
      topWalled: true,
      bottomWalled: true,
      leftWalled: true,
      rightWalled: true,
    });
  });

  it('keeps the central 4-cell clearance neighborhood open', () => {
    const map = buildNeatensteinMap(TEST_SEED);
    const side = EXPECTED_MAP_SIZE;
    const center = Math.floor(side / 2);
    const clearance = 4;

    let allOpen = true;
    for (let x = center - clearance; x <= center + clearance; x++) {
      for (let y = center - clearance; y <= center + clearance; y++) {
        if (map[y * side + x] !== 0) {
          allOpen = false;
        }
      }
    }

    expect(allOpen).toBe(true);
  });

  it('makes every interior floor cell reachable from the center', () => {
    const map = buildNeatensteinMap(TEST_SEED);
    const { total, reachable } = countReachableInteriorFloorCells(
      map,
      EXPECTED_MAP_SIZE,
    );
    expect(total - reachable).toBe(0);
  });

  it('produces identical output for the same seed', () => {
    const first = buildNeatensteinMap(TEST_SEED);
    const second = buildNeatensteinMap(TEST_SEED);
    expect(second).toEqual(first);
  });

  it('normalizes a zero seed into a map of the expected length', () => {
    const map = buildNeatensteinMap(0);
    expect(map.length).toBe(EXPECTED_MAP_SIZE * EXPECTED_MAP_SIZE);
  });

  it('normalizes a zero seed into a Uint8Array', () => {
    const map = buildNeatensteinMap(0);
    expect(map).toBeInstanceOf(Uint8Array);
  });

  it('normalizes a seed equal to the modulus into a map of the expected length', () => {
    const map = buildNeatensteinMap(2_147_483_647);
    expect(map.length).toBe(EXPECTED_MAP_SIZE * EXPECTED_MAP_SIZE);
  });

  it('normalizes a seed equal to the modulus into a Uint8Array', () => {
    const map = buildNeatensteinMap(2_147_483_647);
    expect(map).toBeInstanceOf(Uint8Array);
  });
});

describe('createCollisionMap', () => {
  it('reports out-of-bounds cells as solid', () => {
    const flatMap = buildNeatensteinMap(TEST_SEED);
    const collision = createCollisionMap(flatMap, EXPECTED_MAP_SIZE);

    expect({
      negativeX: collision.isSolid(-1, 5),
      negativeY: collision.isSolid(5, -1),
      oversizedX: collision.isSolid(EXPECTED_MAP_SIZE, 5),
      oversizedY: collision.isSolid(5, EXPECTED_MAP_SIZE),
    }).toEqual({
      negativeX: true,
      negativeY: true,
      oversizedX: true,
      oversizedY: true,
    });
  });

  it('reports a perimeter cell as solid and a central floor cell as open', () => {
    const flatMap = buildNeatensteinMap(TEST_SEED);
    const collision = createCollisionMap(flatMap, EXPECTED_MAP_SIZE);
    const center = Math.floor(EXPECTED_MAP_SIZE / 2);

    expect({
      cornerSolid: collision.isSolid(0, 0),
      centerOpen: collision.isSolid(center, center),
    }).toEqual({ cornerSolid: true, centerOpen: false });
  });

  it('throws when the side is a non-integer', () => {
    const flatMap = new Uint8Array(4);

    expect(() => createCollisionMap(flatMap, 1.5)).toThrow(
      'Invalid map dimensions: expected a square Uint8Array.',
    );
  });

  it('throws when the side is zero', () => {
    const flatMap = new Uint8Array(4);

    expect(() => createCollisionMap(flatMap, 0)).toThrow(
      'Invalid map dimensions: expected a square Uint8Array.',
    );
  });
});

describe('renderer.map.constants', () => {
  it('exposes MAZE_CORRIDOR_WIDTH equal to 3', () => {
    expect(mapConstants).toHaveProperty('MAZE_CORRIDOR_WIDTH', 3);
  });

  it('exposes MAZE_COARSE_GRID_DIVISOR equal to 4', () => {
    expect(mapConstants).toHaveProperty('MAZE_COARSE_GRID_DIVISOR', 4);
  });

  it('exposes MAZE_LOOP_REMOVAL_RATE equal to 0.03', () => {
    expect(mapConstants).toHaveProperty('MAZE_LOOP_REMOVAL_RATE', 0.03);
  });

  it('no longer exports INTERIOR_WALL_DENSITY', () => {
    expect(mapConstants).not.toHaveProperty('INTERIOR_WALL_DENSITY');
  });
});
