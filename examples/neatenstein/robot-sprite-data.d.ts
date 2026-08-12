/**
 * Type declaration for the generated encoded robot sprite asset.
 *
 * The runtime asset is produced by `examples/neatenstein/generate-robot-sprites.py` and lives
 * next to this declaration file so TypeScript can resolve the explicit `.js`
 * import specifier used by the sprite renderer and its tests.
 */

export type RobotSpritePose = 'stand' | 'walk1' | 'walk2' | 'shoot';
export type RobotSpriteDirection =
  | 'front'
  | 'frontRight'
  | 'right'
  | 'backRight'
  | 'back'
  | 'backLeft'
  | 'left'
  | 'frontLeft';

export type EncodedRobotSpriteFrame = readonly (readonly number[])[];

export const ROBOT_SPRITE_SCALE: number;
export const ROBOT_SPRITE_PALETTE: readonly [number, number, number, number][];
export const ROBOT_SPRITE_FRAMES: Readonly<
  Record<
    RobotSpriteDirection,
    Readonly<Record<RobotSpritePose, EncodedRobotSpriteFrame>>
  >
>;
