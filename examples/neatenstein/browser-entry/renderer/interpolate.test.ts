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

  it('throws when a snapshot field is not finite', async () => {
    const { lerpNeatensteinState } =
      await loadModule<typeof import('./interpolate.ts')>('./interpolate.ts');
    const prev = { posX: NaN };
    const curr = { posX: 1 };
    expect(() => lerpNeatensteinState(prev, curr, 0.5)).toThrow(
      'previous.posX must be a finite number',
    );
  });
});
