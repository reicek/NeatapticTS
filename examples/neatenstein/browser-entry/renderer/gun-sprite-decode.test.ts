/**
 * Sibling test file for
 * {@link module:./gun-sprite-decode} — exercises the palette-indexed gun
 * sprite decoder and accent palette builder so the folder quality gate sees
 * a sibling test file for every source module.
 */

import { describe, expect, it } from '@jest/globals';

import {
  buildGunAccentPalette,
  decodeGunSpriteFrame,
  RGBA_CHANNELS,
} from './gun-sprite-decode';
import { GUN_SPRITE_PALETTE } from '../../gun-sprite-data.js';

describe('decodeGunSpriteFrame', () => {
  it('decodes a 1×1 frame into a scaled RGBA snapshot', () => {
    const frame = [[0]] as const;
    const snapshot = decodeGunSpriteFrame(frame);

    expect(snapshot.width).toBeGreaterThan(0);
    expect(snapshot.height).toBeGreaterThan(0);
    expect(snapshot.data.length).toBe(
      snapshot.width * snapshot.height * RGBA_CHANNELS,
    );
  });

  it('preserves palette colors in the decoded output', () => {
    const frame = [[0]] as const;
    const snapshot = decodeGunSpriteFrame(frame);
    const [r, g, b, a] = GUN_SPRITE_PALETTE[0] as readonly number[];

    expect(snapshot.data[0]).toBe(r);
    expect(snapshot.data[1]).toBe(g);
    expect(snapshot.data[2]).toBe(b);
    expect(snapshot.data[3]).toBe(a);
  });
});

describe('buildGunAccentPalette', () => {
  it('replaces RGB channels of palette entries 5 and 6 while preserving alpha', () => {
    const accent: readonly [number, number, number] = [10, 20, 30];
    const palette = buildGunAccentPalette(accent);

    const entry5 = palette[5] as readonly number[];
    const entry6 = palette[6] as readonly number[];
    const original5 = GUN_SPRITE_PALETTE[5] as readonly number[];
    const original6 = GUN_SPRITE_PALETTE[6] as readonly number[];

    expect(entry5[0]).toBe(10);
    expect(entry5[1]).toBe(20);
    expect(entry5[2]).toBe(30);
    expect(entry5[3]).toBe(original5[3]);

    expect(entry6[0]).toBe(10);
    expect(entry6[1]).toBe(20);
    expect(entry6[2]).toBe(30);
    expect(entry6[3]).toBe(original6[3]);
  });

  it('leaves non-accent palette entries unchanged', () => {
    const accent: readonly [number, number, number] = [99, 99, 99];
    const palette = buildGunAccentPalette(accent);

    for (let i = 0; i < GUN_SPRITE_PALETTE.length; i += 1) {
      if (i === 5 || i === 6) continue;
      expect(palette[i]).toEqual(GUN_SPRITE_PALETTE[i]);
    }
  });
});
