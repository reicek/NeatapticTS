/**
 * @jest-environment node
 */

import { describe, expect, it } from '@jest/globals';
import * as robotSpriteData from '../../robot-sprite-data.js';
import type { EncodedRobotSpriteFrame } from './renderer.sprite.types';

const ENCODED_FRAME = robotSpriteData.ROBOT_SPRITE_FRAMES.front
  .stand as unknown as EncodedRobotSpriteFrame;

describe('team-color cache key correctness', () => {
  it('returns the same cached reference for the same team color', async () => {
    const { resolveDecodedRobotSpriteFrameWithTeamColor } = await import(
      './sprites.atlas.utils'
    );
    const first = resolveDecodedRobotSpriteFrameWithTeamColor(ENCODED_FRAME, [
      10, 20, 30,
    ]);
    const second = resolveDecodedRobotSpriteFrameWithTeamColor(ENCODED_FRAME, [
      10, 20, 30,
    ]);
    expect(first).toBe(second);
  });

  it('returns distinct references for different team colors', async () => {
    const { resolveDecodedRobotSpriteFrameWithTeamColor } = await import(
      './sprites.atlas.utils'
    );
    const first = resolveDecodedRobotSpriteFrameWithTeamColor(ENCODED_FRAME, [
      10, 20, 30,
    ]);
    const second = resolveDecodedRobotSpriteFrameWithTeamColor(ENCODED_FRAME, [
      10, 20, 31,
    ]);
    expect(first).not.toBe(second);
  });

  it('distinguishes all three channels independently', async () => {
    const { resolveDecodedRobotSpriteFrameWithTeamColor } = await import(
      './sprites.atlas.utils'
    );
    const r = resolveDecodedRobotSpriteFrameWithTeamColor(ENCODED_FRAME, [
      1, 0, 0,
    ]);
    const g = resolveDecodedRobotSpriteFrameWithTeamColor(ENCODED_FRAME, [
      0, 1, 0,
    ]);
    const b = resolveDecodedRobotSpriteFrameWithTeamColor(ENCODED_FRAME, [
      0, 0, 1,
    ]);
    expect(r).not.toBe(g);
    expect(g).not.toBe(b);
    expect(r).not.toBe(b);
  });

  it('distinguishes edge colors at byte boundaries', async () => {
    const { resolveDecodedRobotSpriteFrameWithTeamColor } = await import(
      './sprites.atlas.utils'
    );
    const white = resolveDecodedRobotSpriteFrameWithTeamColor(ENCODED_FRAME, [
      255, 255, 255,
    ]);
    const black = resolveDecodedRobotSpriteFrameWithTeamColor(ENCODED_FRAME, [
      0, 0, 0,
    ]);
    expect(white).not.toBe(black);
  });

  // Guard test: catches a buggy numeric key that does not mask channels to 8 bits.
  // [0, 1, 256] and [0, 2, 0] must produce distinct cache entries.
  // With an unmasked numeric key (r << 16) | (g << 8) | b:
  //   [0, 1, 256] -> (0 << 16) | (1 << 8) | 256 = 256 + 256 = 512
  //   [0, 2, 0]   -> (0 << 16) | (2 << 8) | 0   = 512  -- COLLISION
  // With masking: [0, 1, 256] -> (0 << 16) | (1 << 8) | (256 & 0xff) = 256
  //               [0, 2, 0]   -> (0 << 16) | (2 << 8) | 0            = 512  -- correct
  it('distinguishes out-of-range blue from a different green value (guard for unmasked numeric key)', async () => {
    const { resolveDecodedRobotSpriteFrameWithTeamColor } = await import(
      './sprites.atlas.utils'
    );
    const overflowBlue = resolveDecodedRobotSpriteFrameWithTeamColor(
      ENCODED_FRAME,
      [0, 1, 256],
    );
    const differentGreen = resolveDecodedRobotSpriteFrameWithTeamColor(
      ENCODED_FRAME,
      [0, 2, 0],
    );
    expect(overflowBlue).not.toBe(differentGreen);
  });
});