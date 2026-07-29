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
});
