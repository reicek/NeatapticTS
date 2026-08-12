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

  it('returns an empty buffer for invalid column counts', async () => {
    const { buildNeatensteinZBuffer } = await import('./zbuffer');

    expect(buildNeatensteinZBuffer(NaN).length).toBe(0);
    expect(buildNeatensteinZBuffer(-4).length).toBe(0);
    expect(buildNeatensteinZBuffer(0).length).toBe(0);
  });

  it('fills missing wall distances with the empty sentinel', async () => {
    const {
      buildNeatensteinZBuffer,
      fillNeatensteinZBuffer,
      NEATENSTEIN_ZBUFFER_EMPTY,
    } = await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(4);
    const wallDistances = new Float32Array([1, 2]);

    fillNeatensteinZBuffer(zBuffer, wallDistances);

    expect(zBuffer).toEqual(
      new Float32Array([
        1,
        2,
        NEATENSTEIN_ZBUFFER_EMPTY,
        NEATENSTEIN_ZBUFFER_EMPTY,
      ]),
    );
  });

  it('returns an empty clip for invalid sprite distances', async () => {
    const { buildNeatensteinZBuffer, clipNeatensteinSpriteSpan } =
      await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(8);

    expect(
      clipNeatensteinSpriteSpan(zBuffer, 0, 4, NaN).visibleColumns,
    ).toEqual([]);
    expect(clipNeatensteinSpriteSpan(zBuffer, 0, 4, 0).visibleColumns).toEqual(
      [],
    );
    expect(
      clipNeatensteinSpriteSpan(zBuffer, 0, 4, Infinity).visibleColumns,
    ).toEqual([]);
  });

  it('invokes the callback for each visible column', async () => {
    const {
      buildNeatensteinZBuffer,
      fillNeatensteinZBuffer,
      forEachVisibleNeatensteinSpriteColumn,
    } = await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(6);
    fillNeatensteinZBuffer(zBuffer, new Float32Array([2, 2, 2, 2, 2, 2]));

    const seen: number[] = [];
    forEachVisibleNeatensteinSpriteColumn(zBuffer, 0, 6, 1.5, (column) =>
      seen.push(column),
    );

    expect(seen).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('does not invoke the callback for invalid spans', async () => {
    const {
      buildNeatensteinZBuffer,
      fillNeatensteinZBuffer,
      forEachVisibleNeatensteinSpriteColumn,
    } = await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(4);
    fillNeatensteinZBuffer(zBuffer, new Float32Array([1, 1, 1, 1]));

    const seen: number[] = [];
    const collect = (column: number) => seen.push(column);

    forEachVisibleNeatensteinSpriteColumn(zBuffer, NaN, 3, 0.5, collect);
    forEachVisibleNeatensteinSpriteColumn(zBuffer, 0, Infinity, 0.5, collect);
    forEachVisibleNeatensteinSpriteColumn(zBuffer, 3, 0, 0.5, collect);
    forEachVisibleNeatensteinSpriteColumn(zBuffer, 0, 3, NaN, collect);

    expect(seen).toEqual([]);
  });

  it('does not invoke the callback when every column is occluded', async () => {
    const {
      buildNeatensteinZBuffer,
      fillNeatensteinZBuffer,
      forEachVisibleNeatensteinSpriteColumn,
    } = await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(4);
    fillNeatensteinZBuffer(zBuffer, new Float32Array([1, 1, 1, 1]));

    const seen: number[] = [];
    forEachVisibleNeatensteinSpriteColumn(zBuffer, 0, 3, 1, (column) =>
      seen.push(column),
    );

    expect(seen).toEqual([]);
  });

  it('returns early when the resolved span has left greater than right', async () => {
    const {
      buildNeatensteinZBuffer,
      fillNeatensteinZBuffer,
      forEachVisibleNeatensteinSpriteColumn,
    } = await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(4);
    fillNeatensteinZBuffer(zBuffer, new Float32Array([2, 2, 2, 2]));

    const seen: number[] = [];
    forEachVisibleNeatensteinSpriteColumn(zBuffer, 2, 2, 1.5, (column) =>
      seen.push(column),
    );

    expect(seen).toEqual([]);
  });

  it('returns an empty clip when the resolved span has left greater than right', async () => {
    const { buildNeatensteinZBuffer, clipNeatensteinSpriteSpan } =
      await import('./zbuffer');

    const zBuffer = buildNeatensteinZBuffer(4);

    const clip = clipNeatensteinSpriteSpan(zBuffer, 2, 2, 1);

    expect(clip.visibleColumns).toEqual([]);
  });
});
