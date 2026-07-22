/**
 * @jest-environment node
 */

import { describe, expect, it } from '@jest/globals';

describe('neatenstein zbuffer', () => {
  it('builds an empty z-buffer filled with infinity', async () => {
    const { buildNeatensteinZBuffer, NEATENSTEIN_ZBUFFER_EMPTY } =
      await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(8);

    expect(zBuffer).toEqual(
      new Float32Array(8).fill(NEATENSTEIN_ZBUFFER_EMPTY),
    );
  });

  it('fills wall distances and replaces non-positive with empty sentinel', async () => {
    const {
      buildNeatensteinZBuffer,
      fillNeatensteinZBuffer,
      NEATENSTEIN_ZBUFFER_EMPTY,
    } = await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(4);
    const wallDistances = new Float32Array([1, 0, -3, 2.5]);

    fillNeatensteinZBuffer(zBuffer, wallDistances);

    expect(zBuffer).toEqual(
      new Float32Array([
        1,
        NEATENSTEIN_ZBUFFER_EMPTY,
        NEATENSTEIN_ZBUFFER_EMPTY,
        2.5,
      ]),
    );
  });

  it('clips a sprite span to visible z-buffer columns only', async () => {
    const {
      buildNeatensteinZBuffer,
      clipNeatensteinSpriteSpan,
      fillNeatensteinZBuffer,
    } = await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(8);
    fillNeatensteinZBuffer(zBuffer, new Float32Array([3, 3, 3, 3, 1, 1, 1, 1]));

    const clip = clipNeatensteinSpriteSpan(zBuffer, 0, 7, 2);

    expect(clip.visibleColumns).toEqual([0, 1, 2, 3]);
  });

  it('ignores columns outside canvas bounds when clipping', async () => {
    const { buildNeatensteinZBuffer, clipNeatensteinSpriteSpan } =
      await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(4);

    const clip = clipNeatensteinSpriteSpan(zBuffer, -10, 10, 1);

    expect(clip.visibleColumns).toEqual([0, 1, 2, 3]);
  });

  it('reports no visible columns when the sprite is behind every wall', async () => {
    const {
      buildNeatensteinZBuffer,
      clipNeatensteinSpriteSpan,
      fillNeatensteinZBuffer,
    } = await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(6);
    fillNeatensteinZBuffer(zBuffer, new Float32Array([1, 1, 1, 1, 1, 1]));

    const clip = clipNeatensteinSpriteSpan(zBuffer, 1, 4, 2);

    expect(clip.visibleColumns).toEqual([]);
  });

  it('is visible when sprite distance is strictly less than stored wall', async () => {
    const {
      buildNeatensteinZBuffer,
      isNeatensteinSpriteColumnVisible,
      fillNeatensteinZBuffer,
    } = await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(3);
    fillNeatensteinZBuffer(zBuffer, new Float32Array([2, 2, 2]));

    const visible = isNeatensteinSpriteColumnVisible(zBuffer, 1, 1.9);

    expect(visible).toBe(true);
  });

  it('is occluded when sprite distance equals stored wall', async () => {
    const {
      buildNeatensteinZBuffer,
      isNeatensteinSpriteColumnVisible,
      fillNeatensteinZBuffer,
    } = await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(3);
    fillNeatensteinZBuffer(zBuffer, new Float32Array([2, 2, 2]));

    const visible = isNeatensteinSpriteColumnVisible(zBuffer, 1, 2);

    expect(visible).toBe(false);
  });

  it('returns false for out-of-bounds columns', async () => {
    const { buildNeatensteinZBuffer, isNeatensteinSpriteColumnVisible } =
      await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(2);

    expect(isNeatensteinSpriteColumnVisible(zBuffer, -1, 1)).toBe(false);
  });
});
