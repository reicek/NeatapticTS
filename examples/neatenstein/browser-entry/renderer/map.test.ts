import { describe, expect, it } from '@jest/globals';
import { buildNeatensteinMap } from './map';

/** Expected square map dimension for the 60×60 target behavior. */
const EXPECTED_MAP_SIZE = 60;

/** Deterministic seed used for repeatable map fixtures. */
const TEST_SEED = 12345;

describe('Neatenstein map generation', () => {
  it('returns a Uint8Array of length 60*60', () => {
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

  it('produces identical output for the same seed', () => {
    const first = buildNeatensteinMap(TEST_SEED);
    const second = buildNeatensteinMap(TEST_SEED);
    expect(second).toEqual(first);
  });
});
