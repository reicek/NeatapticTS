/**
 * Atlas decode and composite-frame helpers extracted from the sprite renderer.
 *
 * Contains the lazily-decoded robot sprite caches and the composite
 * shoot+walk frame builder, along with the shared direction lookup table
 * they depend on.
 *
 * @module
 */

import { ROBOT_SPRITE_FRAMES } from '../../robot-sprite-data.js';
import { type VoxelSnapshot } from '../../../neatenstein/scripts/snapshot-renderer';
import {
  type EncodedRobotSpriteFrame,
  decodeRobotSpriteFrame,
  buildTeamColorPalette,
} from './robot-sprite-decode';
import {
  NEATENSTEIN_ENCODED_DIRECTIONS,
  NEATENSTEIN_SPRITE_UPPER_BODY_SPLIT_ROW,
} from './renderer.sprite.constants';

// Re-export for backward compatibility with consumers that import from
// sprites.atlas.utils.
export { NEATENSTEIN_ENCODED_DIRECTIONS };

/**
 * Lazily decoded frame cache.
 *
 * Encoded frames are immutable, so reference identity is a stable cache key.
 */
const decodedRobotSpriteCache = new Map<
  EncodedRobotSpriteFrame,
  VoxelSnapshot
>();

/**
 * Return a decoded RGBA snapshot for an encoded robot frame, caching the
 * result so repeated renders of the same direction/pose are allocation-free.
 *
 * @param frame - Encoded robot sprite frame.
 * @returns Decoded RGBA {@link VoxelSnapshot}.
 */
export function resolveDecodedRobotSpriteFrame(
  frame: EncodedRobotSpriteFrame,
): VoxelSnapshot {
  const cached = decodedRobotSpriteCache.get(frame);
  if (cached !== undefined) {
    return cached;
  }
  const decoded = decodeRobotSpriteFrame(frame);
  decodedRobotSpriteCache.set(frame, decoded);
  return decoded;
}

/**
 * Lazily decoded team-color frame cache, keyed by encoded frame reference
 * then by color tuple string.
 */
const teamColorDecodedCache = new Map<
  EncodedRobotSpriteFrame,
  Map<string, VoxelSnapshot>
>();

/**
 * Return a decoded RGBA snapshot for an encoded robot frame with a team
 * color applied to palette indices 5/6/7, caching the result.
 *
 * @param frame - Encoded robot sprite frame.
 * @param teamColor - [r, g, b] team color.
 * @returns Decoded RGBA {@link VoxelSnapshot} with team color applied.
 */
export function resolveDecodedRobotSpriteFrameWithTeamColor(
  frame: EncodedRobotSpriteFrame,
  teamColor: readonly [number, number, number],
): VoxelSnapshot {
  let colorMap = teamColorDecodedCache.get(frame);
  if (colorMap === undefined) {
    colorMap = new Map();
    teamColorDecodedCache.set(frame, colorMap);
  }
  const colorKey = `${teamColor[0]},${teamColor[1]},${teamColor[2]}`;
  const cached = colorMap.get(colorKey);
  if (cached !== undefined) {
    return cached;
  }
  const modifiedPalette = buildTeamColorPalette(teamColor);
  const decoded = decodeRobotSpriteFrame(frame, modifiedPalette);
  colorMap.set(colorKey, decoded);
  return decoded;
}

/**
 * Cache for composite shoot+walk frames keyed by direction index and walk pose.
 *
 * The composite frame takes the upper body (rows 0 to split row − 1) from
 * the shoot pose and the lower body (split row and below) from the walk
 * pose, allowing the enemy to shoot while walking.
 */
const compositeShootWalkCache = new Map<string, EncodedRobotSpriteFrame>();

/**
 * Build a composite encoded frame: upper body from the shoot pose and
 * lower body from the walk pose.
 *
 * @param directionIndex - Yaw atlas direction index 0–7.
 * @param walkPoseName - Lower-body walk pose ('stand', 'walk1', or 'walk2').
 * @returns Composite encoded frame.
 */
export function resolveCompositeShootWalkFrame(
  directionIndex: number,
  walkPoseName: 'stand' | 'walk1' | 'walk2',
): EncodedRobotSpriteFrame {
  const key = `${directionIndex}:${walkPoseName}`;
  const cached = compositeShootWalkCache.get(key);
  if (cached !== undefined) {
    return cached;
  }

  const direction = NEATENSTEIN_ENCODED_DIRECTIONS[directionIndex];
  const directionFrames = ROBOT_SPRITE_FRAMES[direction];
  const shootFrame = directionFrames.shoot as EncodedRobotSpriteFrame;
  const walkFrame = directionFrames[walkPoseName] as EncodedRobotSpriteFrame;

  const upperRows = shootFrame.slice(
    0,
    NEATENSTEIN_SPRITE_UPPER_BODY_SPLIT_ROW,
  );
  const lowerRows = walkFrame.slice(NEATENSTEIN_SPRITE_UPPER_BODY_SPLIT_ROW);
  const composite = [...upperRows, ...lowerRows] as EncodedRobotSpriteFrame;

  compositeShootWalkCache.set(key, composite);
  return composite;
}
