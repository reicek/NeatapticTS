import { describe, expect, it } from '@jest/globals';

const loadModule = <T>(path: string): Promise<T> => import(path) as Promise<T>;

function createStateFixture() {
  return {
    posX: 1,
    posY: 2,
    yaw: 0.5,
    health: 100,
  };
}

describe('Neatenstein render-side state interpolation', () => {
  it('interpolates scalar fields by alpha', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = createStateFixture();
    const curr = { ...prev, posX: 3, posY: 4, yaw: 1.5, health: 80 };
    const result = lerpNeatensteinState(prev, curr, 0.25);
    expect(result).toEqual({
      posX: 1.5,
      posY: 2.5,
      yaw: 0.75,
      health: 95,
    });
  });

  it('returns the previous state when alpha is 0', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = createStateFixture();
    const curr = { ...prev, posX: 99 };
    expect(lerpNeatensteinState(prev, curr, 0)).toEqual(prev);
  });

  it('returns the current state when alpha is 1', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = createStateFixture();
    const curr = { ...prev, posX: 99 };
    expect(lerpNeatensteinState(prev, curr, 1)).toEqual(curr);
  });

  it('returns the midpoint at alpha 0.5', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = { posX: 0, posY: 10 };
    const curr = { posX: 10, posY: 20 };
    expect(lerpNeatensteinState(prev, curr, 0.5)).toEqual({
      posX: 5,
      posY: 15,
    });
  });

  it('throws when alpha is not finite', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = createStateFixture();
    const curr = createStateFixture();
    expect(() => lerpNeatensteinState(prev, curr, NaN)).toThrow(
      'alpha must be a finite number',
    );
    expect(() => lerpNeatensteinState(prev, curr, Infinity)).toThrow(
      'alpha must be a finite number',
    );
  });

  it('clamps alpha outside [0, 1]', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = createStateFixture();
    const curr = { ...prev, posX: 99 };
    expect(lerpNeatensteinState(prev, curr, -0.5)).toEqual(prev);
    expect(lerpNeatensteinState(prev, curr, 1.5)).toEqual(curr);
  });

  it('copies new fields from the current snapshot', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = { posX: 1 };
    const curr = { posX: 3, posY: 5 };
    expect(lerpNeatensteinState(prev, curr, 0.5)).toEqual({
      posX: 2,
      posY: 5,
    });
  });

  it('carries over fields removed in the current snapshot', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = { posX: 1, posY: 2 };
    const curr = { posX: 3 };
    expect(lerpNeatensteinState(prev, curr, 0.5)).toEqual({
      posX: 2,
      posY: 2,
    });
  });

  // NOTE: The previous test asserting TypeError for non-finite snapshot
  // fields was removed in C4 — the behavior is now log-and-clamp instead of
  // throwing. See the "non-finite snapshot field safety (C4)" describe block
  // below for the updated expectations.
});

// ---------------------------------------------------------------------------
// Slice C4 — Interpolation Safety (RED phase)
//
// These tests define the expected behavior for three C4 sub-tasks:
//   1. lerpNeatensteinAngle — shortest-arc angle interpolation sibling
//   2. Non-finite snapshot fields — log and clamp instead of TypeError
//
// All tests are expected to FAIL until the implementation phase (Step 04).
// ---------------------------------------------------------------------------

describe('lerpNeatensteinAngle', () => {
  it('returns the from angle when alpha is 0', async () => {
    const { lerpNeatensteinAngle } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    expect(lerpNeatensteinAngle(1.0, 2.0, 0)).toBeCloseTo(1.0);
  });

  it('returns the to angle when alpha is 1', async () => {
    const { lerpNeatensteinAngle } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    expect(lerpNeatensteinAngle(1.0, 2.0, 1)).toBeCloseTo(2.0);
  });

  it('interpolates linearly for angles within half a turn', async () => {
    const { lerpNeatensteinAngle } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    // 0 to π — midpoint should be π/2 (short arc is the direct path)
    expect(lerpNeatensteinAngle(0, Math.PI, 0.5)).toBeCloseTo(Math.PI / 2);
  });

  it('takes the shortest arc across the 2π wrap-around boundary', async () => {
    const { lerpNeatensteinAngle } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    // 350° ≈ 6.108 rad → 10° ≈ 0.175 rad.
    // Shortest arc crosses 0°/2π — midpoint should be ≈ 0°, NOT 180° (≈ π).
    const result = lerpNeatensteinAngle(6.108, 0.175, 0.5);
    const normalized = ((result % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    expect(normalized).toBeCloseTo(0, 1);
  });

  it('handles negative angles via shortest arc', async () => {
    const { lerpNeatensteinAngle } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    // -0.1 to 0.1 — midpoint should be 0
    expect(lerpNeatensteinAngle(-0.1, 0.1, 0.5)).toBeCloseTo(0);
  });
});

describe('non-finite snapshot field safety (C4)', () => {
  it('clamps NaN snapshot fields to finite values instead of throwing', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = { posX: NaN };
    const curr = { posX: 1 };
    // Expected: log warning and clamp NaN to a finite value, NOT throw TypeError.
    expect(() => lerpNeatensteinState(prev, curr, 0.5)).not.toThrow();
    const result = lerpNeatensteinState(prev, curr, 0.5);
    expect(Number.isFinite(result.posX)).toBe(true);
  });

  it('clamps Infinity snapshot fields to finite values instead of throwing', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = { posX: 0 };
    const curr = { posX: Infinity };
    expect(() => lerpNeatensteinState(prev, curr, 0.5)).not.toThrow();
    const result = lerpNeatensteinState(prev, curr, 0.5);
    expect(Number.isFinite(result.posX)).toBe(true);
  });

  it('clamps -Infinity snapshot fields to finite values instead of throwing', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = { posX: -Infinity };
    const curr = { posX: 1 };
    expect(() => lerpNeatensteinState(prev, curr, 0.5)).not.toThrow();
    const result = lerpNeatensteinState(prev, curr, 0.5);
    expect(Number.isFinite(result.posX)).toBe(true);
  });
});
