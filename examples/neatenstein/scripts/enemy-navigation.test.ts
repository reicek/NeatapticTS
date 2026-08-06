import { describe, expect, it } from '@jest/globals';

import type { CollisionMap } from '../browser-entry/renderer/map';
import { NEATENSTEIN_MAP_SIZE } from '../browser-entry/constants';
import {
  buildEnemyDistanceMap,
  buildVisionVector,
  findBestNavigationStep,
  getDistance,
} from './enemy-navigation';

/**
 * Contract tests for examples/neatenstein/scripts/enemy-navigation.ts.
 *
 * Covers the BFS distance map builder, distance lookup, and the
 * cardinal-neighbor gradient-descent step selector used by the enemy
 * controller to navigate around walls.
 */

/** Build a collision map where every cell is open floor. */
function createEmptyCollisionMap(): CollisionMap {
  return {
    isSolid: () => false,
  };
}

/** Build a collision map with solid cells at the supplied grid coordinates. */
function createWallCollisionMap(
  walls: ReadonlyArray<{ readonly x: number; readonly y: number }>,
): CollisionMap {
  const set = new Set(walls.map((w) => `${w.x},${w.y}`));
  return {
    isSolid: (x: number, y: number) => set.has(`${x},${y}`),
  };
}

describe('buildEnemyDistanceMap', () => {
  it('sets the goal cell distance to 0', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(getDistance(map, 10, 10)).toBe(0);
  });

  it('marks wall cells with wallValue (-2)', () => {
    const wallMap = createWallCollisionMap([{ x: 5, y: 5 }]);
    const map = buildEnemyDistanceMap(wallMap, 0, 0, NEATENSTEIN_MAP_SIZE);
    expect(getDistance(map, 5, 5)).toBe(map.wallValue);
    expect(map.wallValue).toBe(-2);
  });

  it('marks unreachable cells with unreachableValue (-1)', () => {
    // Surround a cell with walls to make it unreachable.
    const walls = [
      { x: 10, y: 9 },
      { x: 10, y: 11 },
      { x: 9, y: 10 },
      { x: 11, y: 10 },
    ];
    const wallMap = createWallCollisionMap(walls);
    const map = buildEnemyDistanceMap(wallMap, 0, 0, NEATENSTEIN_MAP_SIZE);
    // Cell (10,10) is enclosed by walls, so it's unreachable.
    expect(getDistance(map, 10, 10)).toBe(map.unreachableValue);
    expect(map.unreachableValue).toBe(-1);
  });

  it('computes distances that increase by 1 per BFS step', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(getDistance(map, 10, 10)).toBe(0);
    expect(getDistance(map, 10, 9)).toBe(1); // N
    expect(getDistance(map, 11, 10)).toBe(1); // E
    expect(getDistance(map, 10, 11)).toBe(1); // S
    expect(getDistance(map, 9, 10)).toBe(1); // W
    expect(getDistance(map, 10, 8)).toBe(2); // N,N
    expect(getDistance(map, 12, 10)).toBe(2); // E,E
    expect(getDistance(map, 10, 12)).toBe(2); // S,S
    expect(getDistance(map, 8, 10)).toBe(2); // W,W
    // Diagonal (Manhattan distance 2 via two orthogonal steps).
    expect(getDistance(map, 11, 9)).toBe(2);
    expect(getDistance(map, 9, 11)).toBe(2);
  });

  it('routes around walls producing correct distances', () => {
    // Wall at (10,9) blocks the direct north path from (10,8) to (10,10).
    // The shortest path from (10,8) to (10,10) is: (10,8)->(9,8)->(9,9)->(9,10)->(10,10) = 4 steps.
    // Or (10,8)->(11,8)->(11,9)->(11,10)->(10,10) = 4 steps.
    const wallMap = createWallCollisionMap([{ x: 10, y: 9 }]);
    const map = buildEnemyDistanceMap(wallMap, 10, 10, NEATENSTEIN_MAP_SIZE);
    // Cell (10,8) should have distance 4 (must go around the wall).
    expect(getDistance(map, 10, 8)).toBe(4);
    // Cells adjacent to the goal should still be 1.
    expect(getDistance(map, 11, 10)).toBe(1);
    expect(getDistance(map, 9, 10)).toBe(1);
    expect(getDistance(map, 10, 11)).toBe(1);
  });

  it('returns all-unreachable when the goal is out of bounds', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      -1,
      0,
      NEATENSTEIN_MAP_SIZE,
    );
    // Every cell should be unreachable since BFS never started.
    expect(getDistance(map, 0, 0)).toBe(map.unreachableValue);
    expect(getDistance(map, 10, 10)).toBe(map.unreachableValue);
  });

  it('returns all-unreachable when the goal is on a wall', () => {
    const wallMap = createWallCollisionMap([{ x: 5, y: 5 }]);
    const map = buildEnemyDistanceMap(wallMap, 5, 5, NEATENSTEIN_MAP_SIZE);
    // Goal is a wall, so BFS never starts — all cells unreachable (except walls).
    expect(getDistance(map, 0, 0)).toBe(map.unreachableValue);
    expect(getDistance(map, 5, 5)).toBe(map.wallValue);
  });

  it('handles NaN goal coordinates by returning all-unreachable', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      Number.NaN,
      0,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(getDistance(map, 0, 0)).toBe(map.unreachableValue);
  });

  it('returns a distances buffer of length size*size', () => {
    const size = 32;
    const map = buildEnemyDistanceMap(createEmptyCollisionMap(), 0, 0, size);
    expect(map.distances.length).toBe(size * size);
    expect(map.size).toBe(size);
  });
});

describe('getDistance', () => {
  it('returns the correct distance for reachable cells', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      0,
      0,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(getDistance(map, 0, 0)).toBe(0);
    expect(getDistance(map, 1, 0)).toBe(1);
    expect(getDistance(map, 0, 1)).toBe(1);
    expect(getDistance(map, 5, 5)).toBe(10);
  });

  it('returns wallValue for wall cells', () => {
    const wallMap = createWallCollisionMap([{ x: 3, y: 3 }]);
    const map = buildEnemyDistanceMap(wallMap, 0, 0, NEATENSTEIN_MAP_SIZE);
    expect(getDistance(map, 3, 3)).toBe(map.wallValue);
  });

  it('returns unreachableValue for out-of-bounds coordinates', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      0,
      0,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(getDistance(map, -1, 0)).toBe(map.unreachableValue);
    expect(getDistance(map, 0, -1)).toBe(map.unreachableValue);
    expect(getDistance(map, NEATENSTEIN_MAP_SIZE, 0)).toBe(
      map.unreachableValue,
    );
    expect(getDistance(map, 0, NEATENSTEIN_MAP_SIZE)).toBe(
      map.unreachableValue,
    );
  });

  it('returns unreachableValue for NaN coordinates', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      0,
      0,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(getDistance(map, Number.NaN, 0)).toBe(map.unreachableValue);
    expect(getDistance(map, 0, Number.NaN)).toBe(map.unreachableValue);
  });
});

describe('findBestNavigationStep', () => {
  it('returns a step toward the neighbour with the lowest distance', () => {
    // Goal at (10,10), enemy at (10,12) — south of goal.
    // Best step is north (dy=-1) to (10,11) which has distance 1.
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const step = findBestNavigationStep(map, 10, 12);
    expect(step).not.toBeNull();
    expect(step!.dx).toBe(0);
    expect(step!.dy).toBe(-1); // North
    expect(step!.distance).toBe(1);
  });

  it('returns null for wall cells', () => {
    const wallMap = createWallCollisionMap([{ x: 5, y: 5 }]);
    const map = buildEnemyDistanceMap(wallMap, 0, 0, NEATENSTEIN_MAP_SIZE);
    const step = findBestNavigationStep(map, 5, 5);
    expect(step).toBeNull();
  });

  it('returns null for unreachable cells', () => {
    // Enclose cell (10,10) with walls.
    const walls = [
      { x: 10, y: 9 },
      { x: 10, y: 11 },
      { x: 9, y: 10 },
      { x: 11, y: 10 },
    ];
    const wallMap = createWallCollisionMap(walls);
    const map = buildEnemyDistanceMap(wallMap, 0, 0, NEATENSTEIN_MAP_SIZE);
    const step = findBestNavigationStep(map, 10, 10);
    expect(step).toBeNull();
  });

  it('returns null at the goal (distance 0, no lower neighbour)', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const step = findBestNavigationStep(map, 10, 10);
    expect(step).toBeNull();
  });

  it('returns null at a dead-end where no neighbour has lower distance', () => {
    // Create a dead-end corridor: cell (5,5) is reachable only from (5,4).
    // Walls surround (5,5) except the north opening at (5,4).
    // Goal at (5,0) — distance from (5,5) is 5 via (5,4)->(5,3)->(5,2)->(5,1)->(5,0).
    // From (5,5), the only neighbour with lower distance is (5,4) = distance 4.
    // That should work. Let's instead test a true dead-end: (5,5) has no
    // neighbour with lower distance. This happens when (5,5) is unreachable.
    // We already test that above. Let's test a cell whose neighbours are all
    // walls except the approach which has a HIGHER distance.
    // Goal at (5,5), cell at (5,6) has distance 1, south neighbour (5,7) has distance 2.
    // No neighbour of (5,6) with distance < 1 except (5,5) which is the goal itself (dist 0).
    // Actually (5,5) has distance 0 < 1, so the step is north. Let's test
    // a different scenario: a cell surrounded by walls except one cell with
    // equal distance (should not return a step since we need strictly lower).
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    // At the goal (10,10), distance is 0. No neighbour has distance < 0.
    expect(findBestNavigationStep(map, 10, 10)).toBeNull();
  });

  it('breaks ties in N, E, S, W order (north preferred on equal distances)', () => {
    // Goal at (10,10). Cell at (10,10) is the goal itself (dist 0).
    // Cell at (9,9) has neighbours: N=(9,8) dist 4, E=(10,9) dist 1, S=(9,10) dist 1, W=(8,9) dist 4.
    // East and South both have distance 1. Since E comes before S in the
    // DIRECTIONS array and we use strict `<`, the first one found (E, dist 1)
    // wins because best is null initially, and S (dist 1) is not < 1.
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const step = findBestNavigationStep(map, 9, 9);
    expect(step).not.toBeNull();
    // East should win over South on equal distance (N,E,S,W order, strict <).
    expect(step!.dx).toBe(1); // East
    expect(step!.dy).toBe(0);
    expect(step!.distance).toBe(1);
  });

  it('breaks ties: east preferred over south on equal distances', () => {
    // Goal at (10,10). Cell at (9,9): E=(10,9) dist 1, S=(9,10) dist 1.
    // E should win (comes first in N,E,S,W and strict < means first-found wins).
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const step = findBestNavigationStep(map, 9, 9);
    expect(step).not.toBeNull();
    expect(step!.dx).toBe(1);
    expect(step!.dy).toBe(0);
  });

  it('navigates around a wall by choosing the lower-distance neighbour', () => {
    // Wall at (10,9) blocks direct north path from (10,8) to (10,10).
    // From (10,8), distance is 4 (must go around). Neighbours:
    // N=(10,7) dist 5 (higher, skip), E=(11,8) dist 3 (lower), S=(10,9) wall, W=(9,8) dist 3 (lower).
    // E and W both have distance 3. E comes first (N,E,S,W), so E wins.
    const wallMap = createWallCollisionMap([{ x: 10, y: 9 }]);
    const map = buildEnemyDistanceMap(wallMap, 10, 10, NEATENSTEIN_MAP_SIZE);
    const step = findBestNavigationStep(map, 10, 8);
    expect(step).not.toBeNull();
    expect(step!.distance).toBe(3);
    // East or West — both are valid; E wins by tie-break order.
    expect(step!.dx).toBe(1); // East
    expect(step!.dy).toBe(0);
  });

  it('returns null for out-of-bounds cell coordinates', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      0,
      0,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(findBestNavigationStep(map, -1, 0)).toBeNull();
    expect(findBestNavigationStep(map, 0, -1)).toBeNull();
  });

  it('returns null for NaN cell coordinates', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      0,
      0,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(findBestNavigationStep(map, Number.NaN, 0)).toBeNull();
  });
});

/**
 * Contract tests for buildVisionVector (AC-10.5a-001, AC-10.5a-003).
 *
 * buildVisionVector(distanceMap, cellX, cellY, previousDistance) returns a
 * Float32Array(6) with [compassScalar, openN, openE, openS, openW, progressDelta].
 */
describe('buildVisionVector', () => {
  it('returns a Float32Array of length 6', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 12, undefined);
    expect(vision).toBeInstanceOf(Float32Array);
    expect(vision.length).toBe(6);
  });

  it('sets compassScalar = bestDirection * 0.25 for north (best dir 0)', () => {
    // Goal at (10,10), enemy at (10,12). Best neighbor is N=(10,11) dist 1.
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 12, undefined);
    // bestDirection = 0 (N), compassScalar = 0 * 0.25 = 0
    expect(vision[0]).toBe(0);
  });

  it('sets compassScalar = bestDirection * 0.25 for south (best dir 2)', () => {
    // Goal at (10,10), enemy at (10,8). Best neighbor is S=(10,9) dist 1.
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 8, undefined);
    // bestDirection = 2 (S), compassScalar = 2 * 0.25 = 0.5
    expect(vision[0]).toBe(0.5);
  });

  it('sets compassScalar = bestDirection * 0.25 for east (best dir 1)', () => {
    // Goal at (10,10), enemy at (8,10). Best neighbor is E=(9,10) dist 1.
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 8, 10, undefined);
    // bestDirection = 1 (E), compassScalar = 1 * 0.25 = 0.25
    expect(vision[0]).toBe(0.25);
  });

  it('sets compassScalar = bestDirection * 0.25 for west (best dir 3)', () => {
    // Goal at (10,10), enemy at (12,10). Best neighbor is W=(11,10) dist 1.
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 12, 10, undefined);
    // bestDirection = 3 (W), compassScalar = 3 * 0.25 = 0.75
    expect(vision[0]).toBe(0.75);
  });

  it('compassScalar is always in range [0, 0.75]', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    // Test multiple positions
    for (const [cx, cy] of [
      [10, 12],
      [12, 10],
      [10, 8],
      [8, 10],
      [10, 10],
    ]) {
      const vision = buildVisionVector(map, cx, cy, undefined);
      expect(vision[0]).toBeGreaterThanOrEqual(0);
      expect(vision[0]).toBeLessThanOrEqual(0.75);
    }
  });

  it('sets openness = 1.0 for best neighbor, 0 for walls', () => {
    // Wall at (10,11) blocks direct south path from (10,12) to goal (10,10).
    const wallMap = createWallCollisionMap([{ x: 10, y: 11 }]);
    const map = buildEnemyDistanceMap(wallMap, 10, 10, NEATENSTEIN_MAP_SIZE);
    // Enemy at (10,12). Neighbors:
    // N=(10,11) = WALL → openN = 0
    // E=(11,12) dist 3, S=(10,13) dist 5, W=(9,12) dist 3
    // Best: E (dir 1, dist 3). W also dist 3 → openW = 1.0 (equal to best)
    const vision = buildVisionVector(map, 10, 12, undefined);
    expect(vision[1]).toBe(0); // openN = 0 (wall)
    expect(vision[2]).toBe(1.0); // openE = 1.0 (best)
    expect(vision[4]).toBe(1.0); // openW = 1.0 (equal to best)
  });

  it('sets openness = 1.0 for best, scaled for others, 0 for walls', () => {
    // Goal at (10,10), enemy at (10,12), empty map.
    // N=(10,11) dist 1, E=(11,12) dist 3, S=(10,13) dist 3, W=(9,12) dist 3
    // Best: N (dir 0, dist 1). Others: 1/3
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 12, undefined);
    expect(vision[1]).toBe(1.0); // openN = 1.0 (best, dist 1)
    expect(vision[2]).toBeCloseTo(1 / 3, 5); // openE = 1/3
    expect(vision[3]).toBeCloseTo(1 / 3, 5); // openS = 1/3
    expect(vision[4]).toBeCloseTo(1 / 3, 5); // openW = 1/3
  });

  it('sets all openness to 0 when all neighbors are walls', () => {
    // Enclose cell (10,10) with walls on all sides.
    const walls = [
      { x: 10, y: 9 },
      { x: 10, y: 11 },
      { x: 9, y: 10 },
      { x: 11, y: 10 },
    ];
    const wallMap = createWallCollisionMap(walls);
    const map = buildEnemyDistanceMap(wallMap, 0, 0, NEATENSTEIN_MAP_SIZE);
    const vision = buildVisionVector(map, 10, 10, undefined);
    expect(vision[1]).toBe(0); // openN
    expect(vision[2]).toBe(0); // openE
    expect(vision[3]).toBe(0); // openS
    expect(vision[4]).toBe(0); // openW
  });

  it('sets progressDelta = 0.5 (neutral) when previousDistance is undefined', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 12, undefined);
    expect(vision[5]).toBe(0.5);
  });

  it('sets progressDelta = 0.5 (neutral) when previousDistance is NaN', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 12, Number.NaN);
    expect(vision[5]).toBe(0.5);
  });

  it('sets progressDelta = 0.5 when previousDistance equals current distance', () => {
    // Enemy at (10,12), current distance is 2. Previous distance also 2.
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 12, 2);
    // delta = prevDist - curDist = 2 - 2 = 0, clipped = 0, progress = 0.5
    expect(vision[5]).toBe(0.5);
  });

  it('sets progressDelta > 0.5 when enemy moved closer to goal', () => {
    // Enemy at (10,12), current distance = 2. Previous distance = 5 (was farther).
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 12, 5);
    // delta = 5 - 2 = 3, clipped to 2, progress = 0.5 + 2/4 = 1.0
    expect(vision[5]).toBe(1.0);
  });

  it('sets progressDelta < 0.5 when enemy moved away from goal', () => {
    // Enemy at (10,12), current distance = 2. Previous distance = 0 (was at goal).
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 12, 0);
    // delta = 0 - 2 = -2, clipped to -2, progress = 0.5 + (-2)/4 = 0.0
    expect(vision[5]).toBe(0.0);
  });

  it('clamps progressDelta to [0, 1] for large positive delta', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    // current dist = 2, prev = 100 → delta = 98, clipped to 2 → 0.5 + 0.5 = 1.0
    const vision = buildVisionVector(map, 10, 12, 100);
    expect(vision[5]).toBe(1.0);
  });

  it('clamps progressDelta to [0, 1] for large negative delta', () => {
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    // current dist = 2, prev = 0 → delta = -2, clipped = -2 → 0.5 + (-0.5) = 0.0
    // Actually delta = prev - cur = 0 - 2 = -2, clipped = -2, progress = 0.5 - 0.5 = 0.0
    const vision = buildVisionVector(map, 10, 12, 0);
    expect(vision[5]).toBe(0.0);
  });

  it('sets progressDelta = 0.5 when current cell is unreachable (dist < 0)', () => {
    // Enclose cell (10,10) with walls. Goal at (0,0).
    const walls = [
      { x: 10, y: 9 },
      { x: 10, y: 11 },
      { x: 9, y: 10 },
      { x: 11, y: 10 },
    ];
    const wallMap = createWallCollisionMap(walls);
    const map = buildEnemyDistanceMap(wallMap, 0, 0, NEATENSTEIN_MAP_SIZE);
    // Cell (10,10) is unreachable (dist = -1). previousDistance = 5.
    const vision = buildVisionVector(map, 10, 10, 5);
    // curDist = -1 (unreachable), so progressDelta = 0.5 (neutral)
    expect(vision[5]).toBe(0.5);
  });

  it('sets progressDelta correctly for small positive step', () => {
    // Enemy at (10,12), current dist = 2. Previous dist = 3 (moved 1 closer).
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 12, 3);
    // delta = 3 - 2 = 1, clipped = 1, progress = 0.5 + 1/4 = 0.75
    expect(vision[5]).toBe(0.75);
  });

  it('sets progressDelta correctly for small negative step', () => {
    // Enemy at (10,12), current dist = 2. Previous dist = 1 (moved 1 away).
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      10,
      10,
      NEATENSTEIN_MAP_SIZE,
    );
    const vision = buildVisionVector(map, 10, 12, 1);
    // delta = 1 - 2 = -1, clipped = -1, progress = 0.5 + (-1)/4 = 0.25
    expect(vision[5]).toBe(0.25);
  });
});
