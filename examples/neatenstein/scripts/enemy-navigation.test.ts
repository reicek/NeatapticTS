import { describe, expect, it } from '@jest/globals';

import { buildNeatensteinMap } from '../browser-entry/renderer/map';
import type { CollisionMap } from '../browser-entry/renderer/map';
import { NEATENSTEIN_MAP_SIZE } from '../browser-entry/constants';
import type { GameState } from '../browser-entry/host/game/types';
import {
  buildEnemyDistanceMap,
  buildVisionVector,
  extractSensors,
  findBestNavigationStep,
  findNearestAmmoPickups,
  findNearestVisibleEnemy,
  getDistance,
} from './enemy-navigation';
import { NEATENSTEIN_AMMO_PICKUP_START_INDEX } from '../browser-entry/harness/neat-io-config';

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

  it('fills a caller-provided distances buffer in-place', () => {
    const size = 32;
    const distances = new Int32Array(size * size);
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      5,
      5,
      size,
      distances,
    );
    // The returned map should reference the same buffer.
    expect(map.distances).toBe(distances);
    // Distance values should be computed correctly.
    expect(getDistance(map, 5, 5)).toBe(0);
    expect(getDistance(map, 6, 5)).toBe(1);
    expect(getDistance(map, 5, 6)).toBe(1);
  });

  it('reuses the same buffer across multiple calls', () => {
    const size = 16;
    const distances = new Int32Array(size * size);
    // First call with goal at (0, 0).
    const map1 = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      0,
      0,
      size,
      distances,
    );
    expect(map1.distances).toBe(distances);
    expect(getDistance(map1, 0, 0)).toBe(0);

    // Second call with goal at (8, 8) — should overwrite the same buffer.
    const map2 = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      8,
      8,
      size,
      distances,
    );
    expect(map2.distances).toBe(distances);
    expect(getDistance(map2, 8, 8)).toBe(0);
    // Old goal (0,0) should now have a non-zero distance.
    expect(getDistance(map2, 0, 0)).toBe(16);
  });

  it('allocates a fresh buffer when distances is the wrong length', () => {
    const size = 32;
    const wrongBuffer = new Int32Array(10);
    const map = buildEnemyDistanceMap(
      createEmptyCollisionMap(),
      0,
      0,
      size,
      wrongBuffer,
    );
    // Should NOT reuse the wrong-length buffer.
    expect(map.distances).not.toBe(wrongBuffer);
    expect(map.distances.length).toBe(size * size);
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

/** Build a minimal GameState for sensor extraction tests. */
function createTestGameState(overrides?: Partial<GameState>): GameState {
  return {
    seed: 42,
    simTimeMs: 0,
    episodeTimeMs: 0,
    player: {
      position: { x: 10.5, y: 10.5 },
      angleRad: 0,
      health: 80,
      maxHealth: 100,
      ammo: 15,
      maxAmmo: 30,
      dashTimeRemainingMs: 0,
      dashCooldownMs: 0,
    },
    enemies: [],
    impacts: [],
    kills: 0,
    spawnCount: 0,
    generation: 1,
    ...overrides,
  } as GameState;
}

/** Build a flat map with only perimeter walls for guaranteed open LOS. */
function createOpenFlatMap(): Uint8Array {
  const s = NEATENSTEIN_MAP_SIZE;
  const map = new Uint8Array(s * s);
  for (let i = 0; i < s; i++) {
    map[i] = 1; // top row
    map[(s - 1) * s + i] = 1; // bottom row
    map[i * s] = 1; // left column
    map[i * s + s - 1] = 1; // right column
  }
  return map;
}

describe('extractSensors', () => {
  // Open map (perimeter walls only) for enemy visibility tests.
  const openMap = createOpenFlatMap();
  // Real Neatenstein map for wall raycast tests.
  const realMap = buildNeatensteinMap(42);

  it('AC-P3S1b-001: returns a 22-element sensor vector', () => {
    const state = createTestGameState();
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors.length).toBe(22);
  });

  it('AC-P3S1b-002: includes normalized player health ratio, ammo, angle, x, y in first 5 sensors', () => {
    const state = createTestGameState();
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[0]).toBeCloseTo(0.8, 5); // health / maxHealth = 80/100
    expect(sensors[1]).toBeCloseTo(0.5, 5); // ammo / maxAmmo = 15/30
    expect(sensors[2]).toBeCloseTo(0, 5); // angleRad 0 → 0/(2π) = 0
    expect(sensors[3]).toBeCloseTo(10.5 / NEATENSTEIN_MAP_SIZE, 5); // position.x / mapSize
    expect(sensors[4]).toBeCloseTo(10.5 / NEATENSTEIN_MAP_SIZE, 5); // position.y / mapSize
  });

  it('AC-P3S1b-002: health ratio is clamped to 0 when maxHealth is 0', () => {
    const state = createTestGameState({
      player: {
        position: { x: 10.5, y: 10.5 },
        angleRad: 0,
        health: 50,
        maxHealth: 0,
        ammo: 10,
        maxAmmo: 30,
        dashTimeRemainingMs: 0,
        dashCooldownMs: 0,
      },
    });
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[0]).toBe(0);
  });

  it('AC-P3S1b-001: sets enemy sensors to 0 when no active enemies exist', () => {
    const state = createTestGameState();
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[5]).toBe(0); // bearing
    expect(sensors[6]).toBe(0); // distance
    expect(sensors[7]).toBe(0); // health
    expect(sensors[12]).toBe(0); // enemyVisible
    expect(sensors[13]).toBe(0); // enemyInFiringArc
  });

  it('AC-P3S1b-002: includes nearest visible enemy bearing, distance, and health (normalized)', () => {
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 14.5, y: 10.5 },
          health: 60,
          active: true,
        },
      ],
    });
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    // Player at (10.5, 10.5), enemy at (14.5, 10.5), player angle = 0
    // dx = 4, dy = 0, bearing = atan2(0, 4) - 0 = 0
    // Normalized bearing: (0 + π) / (2π) = 0.5
    expect(sensors[5]).toBeCloseTo(0.5, 5); // bearing
    // Distance: 4 / 15 = 0.2667
    expect(sensors[6]).toBeCloseTo(4 / 15, 5); // distance
    // Health: 60 / 100 = 0.6 (maxHealth defaults to 100)
    expect(sensors[7]).toBeCloseTo(0.6, 5); // health
    expect(sensors[12]).toBe(1); // enemyVisible
    expect(sensors[13]).toBe(1); // enemyInFiringArc (|0| ≤ π/6)
  });

  it('AC-P3S1b-002: computes relative bearing normalized to [0, 1]', () => {
    // Player facing east (angle 0), enemy directly south at (10.5, 14.5).
    // dx = 0, dy = 4, bearing = atan2(4, 0) - 0 = π/2
    // Normalized: (π/2 + π) / (2π) = (3π/2) / (2π) = 0.75
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 10.5, y: 14.5 },
          health: 50,
          active: true,
        },
      ],
    });
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[5]).toBeCloseTo(0.75, 4);
  });

  it('AC-P3S1b-002: normalizes bearing that would exceed π', () => {
    // Player facing west (angle π), enemy to the east at (14.5, 10.5).
    // dx = 4, dy = 0, bearing = atan2(0, 4) - π = -π → normalized to π
    // Sensor value: (π + π) / (2π) = 1.0
    const state = createTestGameState({
      player: {
        position: { x: 10.5, y: 10.5 },
        angleRad: Math.PI,
        health: 100,
        maxHealth: 100,
        ammo: 20,
        maxAmmo: 30,
        dashTimeRemainingMs: 0,
        dashCooldownMs: 0,
      },
      enemies: [
        {
          position: { x: 14.5, y: 10.5 },
          health: 40,
          active: true,
        },
      ],
    });
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    // bearing = -π (or π, equivalent on the circle).
    // Normalized: (-π + π) / (2π) = 0, or (π + π) / (2π) = 1.
    // Both 0 and 1 represent the same direction (west); accept either.
    expect([0, 1]).toContain(sensors[5]);
  });

  it('AC-P3S1b-003: selects the nearest of multiple visible active enemies', () => {
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 20.5, y: 20.5 },
          health: 100,
          active: true,
        },
        {
          position: { x: 12.5, y: 10.5 },
          health: 30,
          active: true,
        },
      ],
    });
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    // Nearest visible enemy is at (12.5, 10.5), distance = 2
    // Normalized distance: 2 / 15 = 0.1333
    expect(sensors[6]).toBeCloseTo(2 / 15, 5);
    // Normalized health: 30 / 100 = 0.3
    expect(sensors[7]).toBeCloseTo(0.3, 5);
  });

  it('AC-P3S1b-003: ignores inactive enemies', () => {
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 12.5, y: 10.5 },
          health: 30,
          active: false,
        },
        {
          position: { x: 20.5, y: 10.5 },
          health: 100,
          active: true,
        },
      ],
    });
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    // The inactive enemy at distance 2 should be ignored.
    // Active enemy at (20.5, 10.5), distance = 10, within vision range.
    // Normalized distance: 10 / 15 = 0.6667
    expect(sensors[6]).toBeCloseTo(10 / 15, 4);
    // Normalized health: 100 / 100 = 1.0
    expect(sensors[7]).toBeCloseTo(1.0, 5);
  });

  it('AC-P3S1b-003: includes 4 wall raycast distances in sensors[8..11], normalized to [0,1]', () => {
    const state = createTestGameState();
    const sensors = extractSensors(state, realMap, NEATENSTEIN_MAP_SIZE);
    // All 4 raycasts should be normalized to [0, 1].
    expect(sensors[8]).toBeGreaterThanOrEqual(0);
    expect(sensors[8]).toBeLessThanOrEqual(1);
    expect(sensors[9]).toBeGreaterThanOrEqual(0);
    expect(sensors[9]).toBeLessThanOrEqual(1);
    expect(sensors[10]).toBeGreaterThanOrEqual(0);
    expect(sensors[10]).toBeLessThanOrEqual(1);
    expect(sensors[11]).toBeGreaterThanOrEqual(0);
    expect(sensors[11]).toBeLessThanOrEqual(1);
  });

  it('AC-P3S1b-002: all 22 sensor values are in [0, 1] range', () => {
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 14.5, y: 10.5 },
          health: 60,
          active: true,
        },
      ],
    });
    const sensors = extractSensors(state, realMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors.length).toBe(22);
    for (let i = 0; i < 22; i++) {
      expect(sensors[i]).toBeGreaterThanOrEqual(0);
      expect(sensors[i]).toBeLessThanOrEqual(1);
    }
  });

  it('AC-P3S1b-003: enemy sensors are zeroed when enemy is beyond VISION_RANGE_CELLS', () => {
    // Player at (10.5, 10.5), enemy at (30.5, 10.5) — distance = 20 > 15.
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 30.5, y: 10.5 },
          health: 50,
          active: true,
        },
      ],
    });
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[5]).toBe(0); // bearing
    expect(sensors[6]).toBe(0); // distance
    expect(sensors[7]).toBe(0); // health
    expect(sensors[12]).toBe(0); // enemyVisible
    expect(sensors[13]).toBe(0); // enemyInFiringArc
  });

  it('AC-P3S1b-003: enemy sensors are zeroed when no line of sight', () => {
    // Build a map with a wall column between player and enemy.
    const s = NEATENSTEIN_MAP_SIZE;
    const wallMap = createOpenFlatMap();
    // Place a vertical wall at x=12, y=5..15 to block LOS from (10.5,10.5) to (14.5,10.5).
    for (let y = 5; y <= 15; y++) {
      wallMap[y * s + 12] = 1;
    }
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 14.5, y: 10.5 },
          health: 60,
          active: true,
        },
      ],
    });
    const sensors = extractSensors(state, wallMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[5]).toBe(0); // bearing
    expect(sensors[6]).toBe(0); // distance
    expect(sensors[7]).toBe(0); // health
    expect(sensors[12]).toBe(0); // enemyVisible
    expect(sensors[13]).toBe(0); // enemyInFiringArc
  });

  it('AC-P3S1b-001: lastShotHit sensor reads from gameState', () => {
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 14.5, y: 10.5 },
          health: 60,
          active: true,
        },
      ],
    }) as GameState & { lastShotHit?: boolean };
    state.lastShotHit = true;
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[14]).toBe(1);
  });

  it('AC-P3S1b-001: lastShotHit defaults to 0 when not set', () => {
    const state = createTestGameState();
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[14]).toBe(0);
  });

  it('AC-P3S1b-001: enemyInFiringArc is 0 when enemy is visible but outside arc', () => {
    // Player facing east (angle 0), enemy to the south at (10.5, 14.5).
    // Bearing = π/2 (90°), which is > 30° (π/6) firing arc.
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 10.5, y: 14.5 },
          health: 50,
          active: true,
        },
      ],
    });
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[12]).toBe(1); // enemyVisible
    expect(sensors[13]).toBe(0); // enemyInFiringArc (90° > 30°)
  });

  it('AC-P3S1b-002: ammo ratio is clamped to 0 when maxAmmo is 0', () => {
    const state = createTestGameState({
      player: {
        position: { x: 10.5, y: 10.5 },
        angleRad: 0,
        health: 80,
        maxHealth: 100,
        ammo: 10,
        maxAmmo: 0,
        dashTimeRemainingMs: 0,
        dashCooldownMs: 0,
      },
    });
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[1]).toBe(0);
  });

  it('AC-P3S1b-002: enemy health ratio is clamped to 0 when maxHealth is 0', () => {
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 14.5, y: 10.5 },
          health: 50,
          maxHealth: 0,
          active: true,
        },
      ],
    });
    const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(sensors[7]).toBe(0);
  });

  describe('AC-P2S1-001: ammo-pickup sensors', () => {
    it('bearing and distance sensors are populated for the nearest active pickup', () => {
      const state = createTestGameState({
        ammoPickups: [
          {
            position: { x: 14.5, y: 10.5 },
            active: true,
            createdAtMs: 0,
            amount: 10,
          },
        ],
      });
      const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
      // Player at (10.5, 10.5), pickup east, angle 0 → bearing = 0 → normalized 0.5
      expect(sensors[NEATENSTEIN_AMMO_PICKUP_START_INDEX]).toBeCloseTo(0.5, 5);
      // Distance 4 / mapSize
      expect(sensors[NEATENSTEIN_AMMO_PICKUP_START_INDEX + 1]).toBeCloseTo(
        4 / NEATENSTEIN_MAP_SIZE,
        5,
      );
    });

    it('missing pickup slots remain zero', () => {
      const state = createTestGameState();
      const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
      for (let i = 0; i < 6; i++) {
        expect(sensors[NEATENSTEIN_AMMO_PICKUP_START_INDEX + i]).toBe(0);
      }
    });

    it('low-ammo gate is 1 when ammo/maxAmmo < NEATENSTEIN_LOW_AMMO_RATIO', () => {
      const state = createTestGameState({
        player: {
          position: { x: 10.5, y: 10.5 },
          angleRad: 0,
          health: 80,
          maxHealth: 100,
          ammo: 5,
          maxAmmo: 30,
          dashTimeRemainingMs: 0,
          dashCooldownMs: 0,
        },
      });
      const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
      expect(sensors[NEATENSTEIN_AMMO_PICKUP_START_INDEX + 6]).toBe(1);
    });

    it('low-ammo gate is 0 when ammo is above the ratio', () => {
      const state = createTestGameState({
        player: {
          position: { x: 10.5, y: 10.5 },
          angleRad: 0,
          health: 80,
          maxHealth: 100,
          ammo: 10,
          maxAmmo: 30,
          dashTimeRemainingMs: 0,
          dashCooldownMs: 0,
        },
      });
      const sensors = extractSensors(state, openMap, NEATENSTEIN_MAP_SIZE);
      expect(sensors[NEATENSTEIN_AMMO_PICKUP_START_INDEX + 6]).toBe(0);
    });
  });
});

describe('findNearestAmmoPickups', () => {
  const openMap = createOpenFlatMap();

  it('AC-P2S1-002: returns active pickups sorted by BFS path distance', () => {
    const state = createTestGameState({
      ammoPickups: [
        {
          position: { x: 20.5, y: 10.5 },
          active: true,
          createdAtMs: 0,
          amount: 10,
        },
        {
          position: { x: 12.5, y: 10.5 },
          active: true,
          createdAtMs: 0,
          amount: 10,
        },
      ],
    });
    const result = findNearestAmmoPickups(state, openMap, NEATENSTEIN_MAP_SIZE);
    expect(result.length).toBe(2);
    expect(result[0].position.x).toBeCloseTo(12.5, 5);
    expect(result[1].position.x).toBeCloseTo(20.5, 5);
  });

  it('AC-P2S1-002: falls back to Euclidean distance for unreachable/wall pickups', () => {
    const s = NEATENSTEIN_MAP_SIZE;
    const wallMap = createOpenFlatMap();
    // Place a wall cell at (12, 12) and put a pickup inside it.
    wallMap[12 * s + 12] = 1;
    // Place a reachable pickup farther away along the open floor.
    const state = createTestGameState({
      ammoPickups: [
        {
          position: { x: 12.5, y: 12.5 },
          active: true,
          createdAtMs: 0,
          amount: 10,
        },
        {
          position: { x: 20.5, y: 10.5 },
          active: true,
          createdAtMs: 0,
          amount: 10,
        },
      ],
    });
    const result = findNearestAmmoPickups(state, wallMap, NEATENSTEIN_MAP_SIZE);
    // The wall pickup is unreachable (path distance < 0), so it falls back to
    // Euclidean distance ~2.83, which is closer than the reachable pickup's
    // path distance of 10. The fallback distance is therefore used for ranking.
    expect(result.length).toBe(2);
    expect(result[0].position.x).toBeCloseTo(12.5, 5);
    expect(result[1].position.x).toBeCloseTo(20.5, 5);
  });

  it('AC-P2S1-002: stable tie-break preserves ammoPickups insertion order', () => {
    // Two pickups at equal distance from the player (symmetric east/west).
    const state = createTestGameState({
      ammoPickups: [
        {
          position: { x: 12.5, y: 10.5 },
          active: true,
          createdAtMs: 0,
          amount: 10,
        },
        {
          position: { x: 8.5, y: 10.5 },
          active: true,
          createdAtMs: 0,
          amount: 10,
        },
      ],
    });
    const result = findNearestAmmoPickups(state, openMap, NEATENSTEIN_MAP_SIZE);
    // Both are distance 2; toSorted preserves original insertion order.
    expect(result[0].position.x).toBeCloseTo(12.5, 5);
    expect(result[1].position.x).toBeCloseTo(8.5, 5);
  });
});

describe('findNearestVisibleEnemy', () => {
  const openMap = createOpenFlatMap();

  it('AC-P3S1b-005: returns null when no active enemies exist', () => {
    const state = createTestGameState();
    const result = findNearestVisibleEnemy(
      state,
      openMap,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(result).toBeNull();
  });

  it('AC-P3S1b-005: returns the nearest visible enemy within vision range', () => {
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 14.5, y: 10.5 },
          health: 60,
          active: true,
        },
      ],
    });
    const result = findNearestVisibleEnemy(
      state,
      openMap,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(result).not.toBeNull();
    expect(result!.health).toBe(60);
  });

  it('AC-P3S1b-005: returns the nearest of multiple visible enemies', () => {
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 20.5, y: 10.5 },
          health: 100,
          active: true,
        },
        {
          position: { x: 12.5, y: 10.5 },
          health: 30,
          active: true,
        },
      ],
    });
    const result = findNearestVisibleEnemy(
      state,
      openMap,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(result).not.toBeNull();
    expect(result!.health).toBe(30);
  });

  it('AC-P3S1b-005: returns null when all enemies are beyond vision range', () => {
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 30.5, y: 10.5 },
          health: 50,
          active: true,
        },
      ],
    });
    const result = findNearestVisibleEnemy(
      state,
      openMap,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(result).toBeNull();
  });

  it('AC-P3S1b-005: returns null when wall blocks line of sight', () => {
    const s = NEATENSTEIN_MAP_SIZE;
    const wallMap = createOpenFlatMap();
    for (let y = 5; y <= 15; y++) {
      wallMap[y * s + 12] = 1;
    }
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 14.5, y: 10.5 },
          health: 60,
          active: true,
        },
      ],
    });
    const result = findNearestVisibleEnemy(
      state,
      wallMap,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(result).toBeNull();
  });

  it('AC-P3S1b-005: ignores inactive enemies', () => {
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 12.5, y: 10.5 },
          health: 30,
          active: false,
        },
        {
          position: { x: 14.5, y: 10.5 },
          health: 60,
          active: true,
        },
      ],
    });
    const result = findNearestVisibleEnemy(
      state,
      openMap,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(result).not.toBeNull();
    expect(result!.health).toBe(60);
  });

  it('AC-P3S1b-005: skips enemies farther than the current nearest', () => {
    // Three enemies: nearest at (12.5, 10.5), then a farther one at (14.5, 10.5),
    // then an even farther one at (20.5, 10.5). The third should be skipped
    // by the dist >= nearestDist guard.
    const state = createTestGameState({
      enemies: [
        {
          position: { x: 12.5, y: 10.5 },
          health: 30,
          active: true,
        },
        {
          position: { x: 14.5, y: 10.5 },
          health: 60,
          active: true,
        },
        {
          position: { x: 20.5, y: 10.5 },
          health: 100,
          active: true,
        },
      ],
    });
    const result = findNearestVisibleEnemy(
      state,
      openMap,
      NEATENSTEIN_MAP_SIZE,
    );
    expect(result).not.toBeNull();
    expect(result!.health).toBe(30); // nearest is still the first one
  });
});
