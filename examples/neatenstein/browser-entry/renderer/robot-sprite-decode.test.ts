/**
 * Sibling test file for
 * {@link module:./robot-sprite-decode} — exercises the palette-indexed robot
 * sprite decoder and team color palette builder so the folder quality gate
 * sees a sibling test file for every source module.
 */

import { describe, expect, it } from '@jest/globals';

import {
  buildTeamColorPalette,
  decodeRobotSpriteFrame,
  RGBA_CHANNELS,
} from './robot-sprite-decode';
import { ROBOT_SPRITE_PALETTE } from '../../robot-sprite-data.js';

describe('decodeRobotSpriteFrame', () => {
  it('decodes a 1×1 frame into a scaled RGBA snapshot', () => {
    const frame = [[0]] as const;
    const snapshot = decodeRobotSpriteFrame(frame);

    expect(snapshot.width).toBeGreaterThan(0);
    expect(snapshot.height).toBeGreaterThan(0);
    expect(snapshot.data.length).toBe(
      snapshot.width * snapshot.height * RGBA_CHANNELS,
    );
  });

  it('preserves palette colors in the decoded output', () => {
    const frame = [[0]] as const;
    const snapshot = decodeRobotSpriteFrame(frame);
    const [r, g, b, a] = ROBOT_SPRITE_PALETTE[0] as readonly number[];

    expect(snapshot.data[0]).toBe(r);
    expect(snapshot.data[1]).toBe(g);
    expect(snapshot.data[2]).toBe(b);
    expect(snapshot.data[3]).toBe(a);
  });
});

describe('buildTeamColorPalette', () => {
  it('replaces RGB channels of palette entries 5, 6, and 7 while preserving alpha', () => {
    const teamColor: readonly [number, number, number] = [10, 20, 30];
    const palette = buildTeamColorPalette(teamColor);

    for (const idx of [5, 6, 7]) {
      const entry = palette[idx] as readonly number[];
      const original = ROBOT_SPRITE_PALETTE[idx] as readonly number[];

      expect(entry[0]).toBe(10);
      expect(entry[1]).toBe(20);
      expect(entry[2]).toBe(30);
      expect(entry[3]).toBe(original[3]);
    }
  });

  it('leaves non-team-color palette entries unchanged', () => {
    const teamColor: readonly [number, number, number] = [99, 99, 99];
    const palette = buildTeamColorPalette(teamColor);

    for (let i = 0; i < ROBOT_SPRITE_PALETTE.length; i += 1) {
      if (i === 5 || i === 6 || i === 7) continue;
      expect(palette[i]).toEqual(ROBOT_SPRITE_PALETTE[i]);
    }
  });
});