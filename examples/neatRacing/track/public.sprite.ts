/**
 * Recycled Neatenstein public/agent billboard sprites for the neatRacing
 * right-world render pass.
 *
 * Loads a 2D sprite atlas derived from the examples/neatenstein public sprites
 * and builds screen-facing billboard quads placed on the right-hand side of
 * the track. Billboards keep a HALF world-space pixel scale, the named
 * {@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} lever; the right-world
 * render pass applies the same `focal / distance` perspective projection used
 * for 3D voxels, so distant crowds shrink while nearby ones stay readable.
 *
 * THREE IDEAS SHAPE THE MODULE:
 *
 * - A FOOT ANCHOR. A billboard's `position` is the SURFACE it stands on, not
 *   its geometric center. The render pass expands the quad upward from that
 *   point, so a sprite stays glued to the ground or to a prop top no matter
 *   how far the camera is from it.
 * - A CAMERA-RELATIVE FACING. Sprites are drawn from one shared atlas of
 *   eight directions, so the module has to answer "which way is this robot
 *   looking, relative to the camera?". The direction helpers below turn a
 *   world-space heading into one of the eight atlas directions by measuring
 *   the angle between the robot's heading and the camera's forward axis.
 * - A SHARED SCALE. A billboard is HALF the size of the same pixel count in
 *   the voxel world - the named {@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE}
 *   lever - so the same perspective anchor shrinks a distant sprite and a
 *   distant voxel in exactly the same way.
 *
 * The module exports FOUR GROUPS of public surface:
 *
 * - THE CROWD ROLE API. {@link PublicRuntimeRole} is the role type, the four
 *   role constants are {@link PUBLIC_SEATED_RUNTIME_ROLE},
 *   {@link PUBLIC_WALKER_RUNTIME_ROLE}, {@link PUBLIC_STANDING_RUNTIME_ROLE}
 *   and {@link PUBLIC_MECHANIC_RUNTIME_ROLE}, {@link PUBLIC_RUNTIME_ROLES}
 *   lists them in their stable order, and {@link resolvePublicRuntimeRole}
 *   applies the role-less default.
 * - THE CROWD SIZE API. {@link NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER}
 *   is the seated quota of ONE bleacher,
 *   {@link NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE} is the cap on the
 *   agents OUTSIDE the bleachers, and
 *   {@link NEATRACING_PUBLIC_BILLBOARD_DENSITY_BUDGET} is the total the two
 *   shares add up to.
 * - THE ATLAS AND FRAME API. {@link PublicSpriteFrame},
 *   {@link PublicSpriteAtlas}, {@link PublicCamera} and {@link PublicBillboard}
 *   are the data shapes; {@link loadPublicSpriteAtlas} returns the non-empty
 *   atlas whose source is 'neatenstein-public';
 *   {@link selectPublicSpriteFrame} returns a deterministic frame for a given
 *   (atlas, seed, index) tuple; {@link selectPublicWalkFrame} returns a
 *   deterministic walk frame; {@link computePublicRuntimeFrameIndex} resolves
 *   the live frame index of a billboard; {@link resolveDirectionId} maps a
 *   world yaw onto one of the eight atlas directions of
 *   {@link DIRECTION_ORDER}; the pose indices
 *   {@link PUBLIC_POSE_INDEX_STAND}, {@link PUBLIC_POSE_INDEX_WALK1} and
 *   {@link PUBLIC_POSE_INDEX_WALK2}, together with
 *   {@link PUBLIC_POSE_COUNT}, address the direction x pose lookup table;
 *   {@link PUBLIC_TEAM_ACCENT_COLOR_IDS} lists the accents of the team accent
 *   family; and {@link computeBillboardScale} returns the constant
 *   {@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} half world scale so
 *   sprites share the same perspective anchor as voxels.
 * - THE SCENE BUILDER. {@link buildPublicBillboards} returns the right-side
 *   track billboards with positive screen-space quads and perspective-correct
 *   scale.
 *
 * The billboard representation is 2D; it carries no voxel fallback fields.
 *
 * THE CROWD CARRIES FOUR ROLES: the SEATED spectators on the bleacher seats,
 * the STANDING spectators outside the ring, the WALKERS on the trackside and
 * the MECHANICS of the inner pit band. Every role is built by
 * {@link buildPublicBillboards}, and every built billboard carries an
 * explicit role, so the render pass can pick the right pose and accent.
 */

import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';
import { resolveSplineSampleFrame } from '../../racing_curriculum/track/track.spline.utils';
import type { SplineSampleFrame } from '../../racing_curriculum/track/track.spline.utils';
import {
  ROBOT_SPRITE_FRAMES,
  ROBOT_SPRITE_PALETTE,
  ROBOT_SPRITE_SCALE,
} from '../../neatenstein/robot-sprite-data';
import { NEATRACING_TEAM_COLORS } from '../voxel/team-remap';
import {
  NEATRACING_BLEACHER_LEVEL_COUNT,
  NEATRACING_BLEACHER_SEATS_PER_LEVEL,
  NEATRACING_RIGHT_WORLD_BLEACHER_COUNT,
  NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M,
  NEATRACING_ROBOT_PERSONAL_SPACE_M,
  buildRightWorldProps,
  filterRobotsByPersonalSpace,
  getBleacherAxisAlignedHalfSize,
  getBleacherSeatAnchors,
  getBleacherSeatTopY,
  resolveOuterRingSlotCount,
  selectBleacherSlotIndices,
  selectPeopleRingSampleIndices,
} from './right-world.builder';
import type { RightWorldProp } from './right-world.builder';
import { resolvePropHalfSize } from './right-world.prop-sizes';
import type { TrackSlabBuildInput } from './track.slabs';

/**
 * One recycled Neatenstein public/agent sprite frame.
 *
 * A frame is a decoded, ready-to-upload 2D image: the palette-encoded
 * Neatenstein robot sprite is expanded to its pixel size and stored as
 * straight RGBA bytes, with row 0 at the logical sprite TOP, so a frame can be
 * sampled by a texture upload without any further conversion.
 *
 * The frame id keeps the plain `direction-pose` form, and the atlas is
 * COLOR-MAJOR, so the frames of one accent color are contiguous. An index
 * that addresses one color block therefore addresses the same direction and
 * the same pose of every other color block once it is shifted by a whole
 * block.
 *
 * The pixel dimensions are the contract the world size is derived from: a
 * frame of `w x h` pixels covers `w * NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE
 * * NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M` meters of world width, and the
 * same factor of world height.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 * const frame = atlas.frames[0]!;
 *
 * // The id keeps the `direction-pose` form, whatever the accent color is.
 * console.log(frame.id); // 'front-stand'
 *
 * // The RGBA buffer is a straight, top-down image of the frame's pixels.
 * console.log(frame.rgba.length); // 147456 - frame.width * frame.height * 4
 * ```
 */
export interface PublicSpriteFrame {
  /** Stable frame identifier, in the `direction-pose` form. */
  readonly id: string;
  /** Frame width in pixels. */
  readonly width: number;
  /** Frame height in pixels. */
  readonly height: number;
  /** Raw RGBA pixel data, row 0 at the logical sprite top. */
  readonly rgba: Uint8ClampedArray;
}

/**
 * Sprite atlas loaded from the recycled Neatenstein public sprites.
 *
 * The atlas is the single texture source of the crowd: one packed image that
 * holds every accent color, every facing direction and every pose of the
 * public sprite. The render path therefore uploads ONE texture and samples it
 * with a per-frame UV rectangle instead of uploading a texture per sprite.
 *
 * The frame list is COLOR-MAJOR - the frames of one accent color are
 * contiguous - so a frame index can be shifted by a whole color block to
 * reach the same direction and pose in a different accent.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 *
 * // 6 accent colors x 8 directions x 3 poses = 144 frames.
 * console.log(atlas.source); // 'neatenstein-public'
 * console.log(atlas.frames.length); // 144
 * ```
 */
export interface PublicSpriteAtlas {
  /** Identifier of the sprite source; always 'neatenstein-public'. */
  readonly source: 'neatenstein-public';
  /** Decoded frames of the atlas, in color-major order. */
  readonly frames: readonly PublicSpriteFrame[];
}

/**
 * Minimal camera contract used to project world billboards into screen space.
 *
 * The crowd needs three things from a camera: where it is, which way it looks
 * and how wide its field of view is. The basis is expected to be orthonormal -
 * `forward` and `right` are unit vectors and the world-up axis is derived
 * from their cross product - because the billboard projection measures the
 * anchor against that basis.
 *
 * The near and far planes belong to the camera contract but are NOT read by
 * the billboard projection: the crowd quads are drawn in world space and
 * depth-tested by the renderer, so the sprite module never has to clip.
 *
 * @example
 * ```ts
 * const camera: PublicCamera = {
 *   position: [0, 2, 20],
 *   forward: [0, -0.1, -1],
 *   right: [1, 0, 0],
 *   near: 0.1,
 *   far: 200,
 *   fovDegrees: 60,
 * };
 * ```
 */
export interface PublicCamera {
  /** World-space camera position, in meters. */
  readonly position: readonly [number, number, number];
  /** World-space forward axis of the camera; a unit vector. */
  readonly forward: readonly [number, number, number];
  /** World-space right axis of the camera; a unit vector. */
  readonly right: readonly [number, number, number];
  /** Near-plane distance in meters; part of the camera contract, not read by the projection. */
  readonly near: number;
  /** Far-plane distance in meters; part of the camera contract, not read by the projection. */
  readonly far: number;
  /** Vertical field of view in degrees; its half-angle tangent is the focal length of the projection. */
  readonly fovDegrees: number;
}

/**
 * A 2D billboard instance placed in world space and projected to screen space.
 *
 * A billboard is a camera-facing SPRITE, not a voxel. It carries the world
 * anchor the sprite stands on, the atlas frame it was built with, a small set
 * of runtime hints that let the render pass re-derive the live pose of the
 * agent on every frame, and the projection the building camera produced.
 *
 * The shape carries FOUR kinds of information:
 *
 * - THE ANCHOR. {@link PublicBillboard.position} is the SURFACE the sprite
 *   stands on, and the render pass expands the world-space quad upward from
 *   it, so the sprite stays glued to the ground or to a prop top.
 * - THE FRAME. {@link PublicBillboard.frameId} is the atlas frame the builder
 *   selected; the pass recomputes the live frame index from the role, the
 *   camera and the clock.
 * - THE RUNTIME HINTS. The role, the seed, the index and the walk direction
 *   are the deterministic identity of the agent, and they are what makes the
 *   live pose reproducible from frame to frame.
 * - THE PROJECTION. The scale, the distance and the quad describe how the
 *   billboard looked to the camera that built it.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 * const camera: PublicCamera = {
 *   position: [0, 2, 20],
 *   forward: [0, -0.1, -1],
 *   right: [1, 0, 0],
 *   near: 0.1,
 *   far: 200,
 *   fovDegrees: 60,
 * };
 *
 * const billboards = buildPublicBillboards(
 *   { seed: 42, layoutVersion: 1, sizeBucket: 'large' },
 *   camera,
 *   atlas,
 * );
 *
 * const first = billboards[0]!;
 *
 * // The first billboard is a seated spectator on a bleacher seat, and its
 * // anchor is the seat SURFACE, not the geometric center of the sprite.
 * console.log(first.runtimeRole); // 'seated'
 * console.log(first.position[1].toFixed(1)); // '0.8' - the seat top of level one
 *
 * // The quad is a positive-extent [left, top, right, bottom] rectangle.
 * console.log(first.quad[2] > first.quad[0]); // true
 * console.log(first.quad[3] > first.quad[1]); // true
 * ```
 */
export interface PublicBillboard {
  /**
   * World-space foot/base anchor in meters.
   *
   * The y coordinate is the surface the billboard stands on (ground or prop top),
   * not the geometric center. The render pass expands the world-space quad
   * upward from this point so the sprite stays anchored to the surface
   * regardless of runtime camera distance.
   */
  readonly position: readonly [number, number, number];
  /** Selected atlas frame id (build-time hint; the pass recomputes it live). */
  readonly frameId: string;
  /**
   * Runtime role used by the pass to recompute direction/walk/gaze each frame.
   *
   * The field is optional in the type, and a MISSING role is read as
   * {@link PUBLIC_WALKER_RUNTIME_ROLE} by {@link resolvePublicRuntimeRole}.
   * Every billboard produced by {@link buildPublicBillboards} carries an
   * explicit role - the four crowd roles of the re-populated world - so the
   * role-less default never decides a built scene.
   */
  readonly runtimeRole?: PublicRuntimeRole;
  /** Deterministic seed carried through from the builder. */
  readonly seed?: number;
  /** Billboard index carried through from the builder. */
  readonly index?: number;
  /** Preselected walk direction index (0..7) for walkers. */
  readonly walkDirectionIndex?: number;
  /**
   * World-space scale factor relative to the atlas frame's pixel size.
   *
   * Kept at the named {@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} HALF
   * world scale so perspective shrink is handled by the shared
   * `focal / distance` projection in the right-world pass rather than by an
   * extra per-distance scale factor. The factor that turns the scaled pixels
   * into METERS is the separate {@link NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M}
   * constant, so the world size of a frame is
   * `frame.width * scale * NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M`.
   */
  readonly scale: number;
  /**
   * Distance from the camera to the foot anchor, in meters.
   *
   * The value is clamped to a small positive minimum so the perspective
   * division of the projection can never divide by zero.
   */
  readonly distance: number;
  /**
   * Projection-space quad of the billboard, as `[left, top, right, bottom]`.
   *
   * The CENTER of the quad is a projection-space coordinate: the foot anchor
   * measured against the camera basis, divided by its distance and scaled by
   * the focal length `1 / tan(fovDegrees / 2)`. A center value of `1`
   * therefore sits on the vertical half-field edge of the frustum, and the
   * horizontal axis is not aspect-corrected.
   *
   * The EXTENT of the quad is NOT in that unit. It is the pixel size of the
   * frame multiplied by the named {@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE}
   * lever and halved, so it is expressed in SPRITE-PIXEL x WORLD-SCALE units -
   * neither projection units nor meters. A 192 px crowd frame gives a half
   * extent of 48 units, while the WORLD size of the same sprite is
   * `192 * NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE *
   * NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M` = 4.8 m.
   *
   * The quad is a BUILD-TIME PREVIEW, not the rendered size. The right-world
   * render pass ignores it and rebuilds world-space camera-facing billboard
   * geometry every frame from {@link PublicBillboard.position} and the live
   * atlas frame, so the drawn sprite always measures the world size above.
   * Read the quad for the projected anchor it still carries and for its sign
   * convention, and read the rendered size from the frame and the two named
   * constants.
   */
  readonly quad: readonly [number, number, number, number];
}

/**
 * Runtime role of a billboard of the public crowd.
 *
 * The re-populated right world carries FOUR crowd roles: the SEATED
 * spectators on the bleacher seats, the WALKERS on the trackside, the
 * STANDING spectators outside the ring and the MECHANICS of the inner pit
 * band. The role drives the runtime pose and the runtime accent of the
 * billboard.
 *
 * @example
 * ```ts
 * const role: PublicRuntimeRole = 'mechanic';
 *
 * // The role list is the stable enum of the crowd.
 * console.log(PUBLIC_RUNTIME_ROLES.includes(role)); // true
 * console.log(PUBLIC_RUNTIME_ROLES); // ['seated', 'walker', 'standing', 'mechanic']
 * ```
 */
export type PublicRuntimeRole = 'seated' | 'walker' | 'standing' | 'mechanic';

/**
 * Runtime role of the SEATED spectators on the bleacher seats.
 *
 * The seated role is the ONLY role of a billboard that is anchored to a
 * bleacher seat: the render pass keeps it on the stand pose and never puts
 * it on the walk cycle.
 *
 * @example
 * ```ts
 * console.log(PUBLIC_SEATED_RUNTIME_ROLE); // 'seated'
 * ```
 */
export const PUBLIC_SEATED_RUNTIME_ROLE = 'seated';

/**
 * Runtime role of a billboard that walks instead of sitting.
 *
 * This is the SINGLE definition of the role-less default: a billboard that
 * carries no `runtimeRole` is read as this role by
 * {@link resolvePublicRuntimeRole}. Keeping one definition is what makes the
 * atlas frame-index helper and the right-world render pass AGREE.
 *
 * @example
 * ```ts
 * console.log(PUBLIC_WALKER_RUNTIME_ROLE); // 'walker'
 * ```
 */
export const PUBLIC_WALKER_RUNTIME_ROLE = 'walker';

/**
 * Runtime role of the STANDING spectators outside the ring.
 *
 * The standing role shares the stand pose with the seated spectators, but it
 * describes an agent that is NOT anchored to a bleacher: it stands on the
 * ground outside the ring and looks at the car.
 *
 * @example
 * ```ts
 * console.log(PUBLIC_STANDING_RUNTIME_ROLE); // 'standing'
 * ```
 */
export const PUBLIC_STANDING_RUNTIME_ROLE = 'standing';

/**
 * Runtime role of the MECHANICS of the inner pit band.
 *
 * The mechanic role is the only role that wears a TEAM accent, so the pit
 * band reads as the crew of the teams instead of the colour-diverse public
 * crowd.
 *
 * @example
 * ```ts
 * console.log(PUBLIC_MECHANIC_RUNTIME_ROLE); // 'mechanic'
 * ```
 */
export const PUBLIC_MECHANIC_RUNTIME_ROLE = 'mechanic';

/**
 * Every runtime role of the public crowd, in the stable order.
 *
 * The order is SEATED, WALKER, STANDING, MECHANIC and never changes, so a
 * consumer can index the list as a stable enum of the crowd.
 *
 * @example
 * ```ts
 * console.log(PUBLIC_RUNTIME_ROLES.length); // 4
 * console.log(PUBLIC_RUNTIME_ROLES[0]); // 'seated'
 * ```
 */
export const PUBLIC_RUNTIME_ROLES: readonly PublicRuntimeRole[] = [
  PUBLIC_SEATED_RUNTIME_ROLE,
  PUBLIC_WALKER_RUNTIME_ROLE,
  PUBLIC_STANDING_RUNTIME_ROLE,
  PUBLIC_MECHANIC_RUNTIME_ROLE,
];

/**
 * Resolves the runtime role of a billboard, applying the role-less default.
 *
 * The atlas frame-index helper and the right-world render pass read the role
 * through this resolver, so a billboard without an explicit role is never
 * rendered by one consumer and dropped by the other. A MISSING role is read
 * as the WALKER role - the only role of the crowd on the walk cycle.
 *
 * @param billboard - Billboard (or role carrier) to resolve.
 * @returns The explicit role, or {@link PUBLIC_WALKER_RUNTIME_ROLE} when the
 *   billboard carries none.
 *
 * @example
 * ```ts
 * console.log(resolvePublicRuntimeRole({ runtimeRole: 'seated' })); // 'seated'
 * console.log(resolvePublicRuntimeRole({})); // 'walker'
 * ```
 */
export function resolvePublicRuntimeRole(
  billboard: Pick<PublicBillboard, 'runtimeRole'>,
): PublicRuntimeRole {
  return billboard.runtimeRole ?? PUBLIC_WALKER_RUNTIME_ROLE;
}

/**
 * Number of seated spectators every bleacher of the ring carries.
 *
 * The quota is PER BLEACHER: every bleacher of the right world is populated
 * with its own share of seated spectators, so no bleacher of the ring is
 * left empty. The quota is DERIVED from the bleacher geometry - the seats
 * per level times the number of levels - so the seated crowd covers the FULL
 * platform of every stand instead of crowding one end of it, and a geometry
 * change moves the quota with it instead of drifting away from it.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER); // 12
 * ```
 */
export const NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER =
  NEATRACING_BLEACHER_SEATS_PER_LEVEL * NEATRACING_BLEACHER_LEVEL_COUNT;

/**
 * Number of public agents allowed OUTSIDE the bleachers.
 *
 * The allowance is the share of the density budget that is spent on the
 * agents that are NOT seated on a stand: the trackside walkers, the standing
 * spectators outside the ring and the mechanics of the inner pit band. It is
 * a CAP and not a target: the off-bleacher count never exceeds it, and it
 * never exceeds the share of the budget the seated spectators did not use.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE); // 34
 * ```
 */
export const NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE = 34;

/**
 * Density budget of the public/agent billboards per track.
 *
 * The budget is DERIVED from the world instead of being a magic number: the
 * seated spectators of every bleacher of the ring (20 bleachers times 12
 * seats = 240 billboards) PLUS the off-bleacher allowance (34 billboards).
 * The derivation keeps both shares in step - a world with more bleachers
 * seats more spectators - so the budget never silently truncates the crowd.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_PUBLIC_BILLBOARD_DENSITY_BUDGET); // 274
 * ```
 */
export const NEATRACING_PUBLIC_BILLBOARD_DENSITY_BUDGET =
  NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER *
    NEATRACING_RIGHT_WORLD_BLEACHER_COUNT +
  NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE;

/**
 * Trackside distance of the standing spectators, in meters beyond the road
 * half-width of their track sample.
 */
const NEATRACING_PUBLIC_STANDING_TRACKSIDE_OFFSET_M = 19;

/**
 * Random spread of the standing spectator trackside distance, in meters.
 */
const NEATRACING_PUBLIC_STANDING_OFFSET_JITTER_M = 1;

/**
 * Trackside distance of the first walker band, in meters beyond the road
 * half-width of their track sample.
 */
const NEATRACING_PUBLIC_WALKER_BAND_ONE_OFFSET_M = 11.5;

/**
 * Trackside distance of the second walker band, in meters beyond the road
 * half-width of their track sample.
 */
const NEATRACING_PUBLIC_WALKER_BAND_TWO_OFFSET_M = 15;

/**
 * Random spread of the walker trackside distance, in meters.
 */
const NEATRACING_PUBLIC_WALKER_OFFSET_JITTER_M = 1;

/**
 * Radial step of the off-bleacher clearance search, in meters.
 *
 * An off-bleacher agent that lands on a scenery prop or on the personal
 * space of another billboard is pushed further out in steps of this size,
 * until a clear spot is found or the agent is skipped.
 */
const NEATRACING_PUBLIC_CLEARANCE_STEP_M = 1.5;

/**
 * Number of candidate offsets the off-bleacher clearance search tries.
 *
 * The search covers the base offset plus four clearance steps, so an agent
 * can be pushed at most six meters further out than its base band.
 */
const NEATRACING_PUBLIC_CLEARANCE_STEP_COUNT = 5;

/**
 * Personal space an off-bleacher agent keeps to every other billboard.
 *
 * The clearance is the personal-space band MINUS its epsilon, so an agent
 * that sits exactly on the band edge still counts as clear.
 */
const NEATRACING_PUBLIC_OFF_BLEACHER_PERSONAL_SPACE_CLEARANCE_M =
  NEATRACING_ROBOT_PERSONAL_SPACE_M - NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M;

/**
 * Trackside distances of the two walker bands, in meters beyond the road
 * half-width of their track sample.
 */
const NEATRACING_PUBLIC_WALKER_BAND_OFFSETS_M: readonly number[] = [
  NEATRACING_PUBLIC_WALKER_BAND_ONE_OFFSET_M,
  NEATRACING_PUBLIC_WALKER_BAND_TWO_OFFSET_M,
];

/**
 * Meters per sprite pixel used to turn frame pixels into world-space size.
 *
 * The value is the SHARED meter scale of the crowd sprites. It is EXPORTED so
 * the right-world render pass CAN import it instead of keeping its own copy,
 * and so the placement footprint of a billboard and the rendered quad can never
 * disagree about the size of a sprite pixel.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M); // 0.05 - meters per sprite pixel
 * ```
 */
export const NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M = 0.05;

/** Small vertical clearance so a surface-aligned billboard is strictly above scenery. */
const NEATRACING_BILLBOARD_SURFACE_EPSILON_M = 0.001;

/** Minimum distance used to avoid division by zero in projection math. */
const BILLBOARD_MIN_DISTANCE_M = 0.1;

/**
 * World-space scale of the crowd billboards relative to the voxel world.
 *
 * The lever is NAMED and EXPORTED so the halving of the crowd sprite world
 * size has exactly ONE source of truth: the module, the placement footprint
 * and the right-world render pass all read this constant. With this value the
 * world width of a billboard is
 * `frame.width * NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE * NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M`,
 * which is HALF the width of the same pixel count in the voxel world. The
 * result is still perspective-projected by the right-world pass exactly like a
 * voxel of that world size, so distant sprites shrink only because of the
 * shared `focal / distance` projection and never because of an extra
 * per-distance scale factor.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE); // 0.5 - half the voxel world scale
 * ```
 */
export const NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE = 0.5;

/** RGBA channel count used when decoding Neatenstein palette data. */
const RGBA_CHANNELS = 4;

/**
 * Canonical 8-direction order used by the Neatenstein robot sprite data.
 *
 * The order is the shared index space of the atlas: a direction index is a
 * position in this list, and the runtime lookup table multiplies it by the
 * number of poses per direction.
 *
 * @example
 * ```ts
 * console.log(DIRECTION_ORDER.length); // 8
 * console.log(DIRECTION_ORDER[0]); // 'front'
 * console.log(DIRECTION_ORDER[4]); // 'back'
 * ```
 */
export const DIRECTION_ORDER = [
  'front',
  'frontRight',
  'right',
  'backRight',
  'back',
  'backLeft',
  'left',
  'frontLeft',
] as const;

/**
 * Angular width of one atlas direction step (45 degrees).
 *
 * The Neatenstein sprite atlas stores eight equally spaced facing directions,
 * so a yaw is quantized with this step.
 */
const PUBLIC_DIRECTION_STEP_RAD = Math.PI / 4;

/** Full turn in radians, used to normalize a relative yaw into [0, 2*PI). */
const PUBLIC_FULL_TURN_RAD = 2 * Math.PI;

/** Diverse neon accent palette used for spectator public sprites. */
const PUBLIC_NEON_ACCENT_PALETTE: readonly (readonly [number, number, number, number])[] = [
  [0, 255, 255, 255],
  [255, 0, 255, 255],
  [0, 255, 128, 255],
  [255, 255, 0, 255],
  [255, 0, 128, 255],
  [128, 0, 255, 255],
];

/** Number of palette colors available for public robots. */
const PUBLIC_COLOR_COUNT = PUBLIC_NEON_ACCENT_PALETTE.length;

/** Width of one hue bucket used to group colors into families, in degrees. */
const PUBLIC_HUE_BUCKET_DEGREES = 30;

/** Max hue distance of a palette accent to a team color, in degrees. */
const PUBLIC_TEAM_ACCENT_HUE_BAND_DEGREES = 30;

/**
 * Converts an RGB triple into a hue angle in degrees.
 *
 * The classic max-delta hue formula is used, so a grey pixel - the same max
 * and min channel - has no hue at all and yields NaN.
 *
 * @param red - Red channel in the 0..255 range.
 * @param green - Green channel in the 0..255 range.
 * @param blue - Blue channel in the 0..255 range.
 * @returns The hue angle in the 0..360 degree range, or NaN for a grey pixel.
 */
function computeRgbHueDegrees(red: number, green: number, blue: number): number {
  const max = Math.max(red, green, blue);
  const min = Math.min(red, green, blue);

  if (max === min) {
    return Number.NaN;
  }

  const delta = max - min;
  let sector: number;

  if (max === red) {
    sector = ((green - blue) / delta) % 6;
  } else if (max === green) {
    sector = (blue - red) / delta + 2;
  } else {
    sector = (red - green) / delta + 4;
  }

  let hue = sector * 60;
  if (hue < 0) {
    hue += 360;
  }

  return hue;
}

/**
 * Rounds a hue angle to its hue bucket.
 *
 * @param hueDegrees - Hue angle in degrees.
 * @returns The bucket centre in degrees, on the 30 degree grid.
 */
function computeHueBucket(hueDegrees: number): number {
  return (
    Math.round(hueDegrees / PUBLIC_HUE_BUCKET_DEGREES) * PUBLIC_HUE_BUCKET_DEGREES
  );
}

/**
 * Measures the distance between two hue buckets on the wrapped colour circle.
 *
 * The distance wraps around 360 degrees, so a bucket at 350 degrees is only
 * 10 degrees away from a bucket at 0 degrees.
 *
 * @param left - First hue bucket in degrees.
 * @param right - Second hue bucket in degrees.
 * @returns The wrapped distance between the two buckets, in degrees.
 */
function computeHueBucketDistance(left: number, right: number): number {
  const raw = Math.abs(left - right) % 360;

  return Math.min(raw, 360 - raw);
}

/**
 * Resolves the palette accent ids that belong to the TEAM accent family.
 *
 * An accent belongs to the family when its hue bucket is within one bucket
 * of the hue bucket of one of the team colors.
 *
 * @returns The accent ids of the team family, in palette order.
 */
function resolvePublicTeamAccentColorIds(): readonly number[] {
  const teamBuckets = NEATRACING_TEAM_COLORS.map((team) =>
    computeHueBucket(computeRgbHueDegrees(team.rgb[0], team.rgb[1], team.rgb[2])),
  );
  const colorIds: number[] = [];

  for (let index = 0; index < PUBLIC_NEON_ACCENT_PALETTE.length; index += 1) {
    const accent = PUBLIC_NEON_ACCENT_PALETTE[index];
    const accentBucket = computeHueBucket(
      computeRgbHueDegrees(accent[0], accent[1], accent[2]),
    );
    const isTeamFamily = teamBuckets.some(
      (teamBucket) =>
        computeHueBucketDistance(accentBucket, teamBucket) <=
        PUBLIC_TEAM_ACCENT_HUE_BAND_DEGREES,
    );

    if (isTeamFamily) {
      colorIds.push(index);
    }
  }

  return colorIds;
}

/**
 * Accent ids of the public palette that belong to the TEAM accent family.
 *
 * The family rule is the "same 30 degree hue band": an accent belongs to the
 * family when its hue bucket is within one bucket of the hue bucket of one
 * of the team colors. The ids are DERIVED from the accent palette and the
 * team colors at module load, so a palette or a team-color change moves the
 * family with it instead of drifting away from it.
 *
 * @example
 * ```ts
 * console.log(PUBLIC_TEAM_ACCENT_COLOR_IDS); // [0, 2, 4]
 * ```
 */
export const PUBLIC_TEAM_ACCENT_COLOR_IDS: readonly number[] =
  resolvePublicTeamAccentColorIds();

/** Stable per-robot accent color index derived from deterministic identity. */
function computePublicRobotColorId(seed: number, index: number): number {
  let hash = seed >>> 0;
  hash ^= (index * 0x9e3779b9) >>> 0;
  hash = Math.imul(hash ^ 0x85ebca6b, 0xc2b2ae35);
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x27d4eb2f);
  hash ^= hash >>> 15;

  return (hash >>> 0) % PUBLIC_COLOR_COUNT;
}

/**
 * Maps the stable robot accent identity into the TEAM accent family.
 *
 * The mapping is deterministic: the same seed and index always resolve to
 * the same team-family accent, so the mechanics of a frame keep the same
 * team-coloured accent across frames. The accent is a member of the TEAM
 * FAMILY - a hue neighbour of one of the named team colours - and not the
 * exact colour of a single named team, because the public accent palette
 * carries no entry for every named team colour.
 *
 * @param seed - Deterministic seed of the billboard.
 * @param index - Deterministic billboard index.
 * @returns The team-family accent id, or the plain robot accent id when the
 *   team family is empty.
 */
function resolvePublicTeamAccentColorId(seed: number, index: number): number {
  const colorId = computePublicRobotColorId(seed, index);

  if (PUBLIC_TEAM_ACCENT_COLOR_IDS.length === 0) {
    return colorId;
  }

  const familyIndex = colorId % PUBLIC_TEAM_ACCENT_COLOR_IDS.length;

  return PUBLIC_TEAM_ACCENT_COLOR_IDS[familyIndex];
}

/** Encoded robot sprite frame shape reused from Neatenstein. */
type EncodedFrame = readonly (readonly number[])[];

/**
 * Loads the recycled Neatenstein public/agent sprite atlas.
 *
 * @returns A non-empty 2D sprite atlas whose source is 'neatenstein-public'.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 * console.log(atlas.source); // 'neatenstein-public'
 * console.log(atlas.frames.length > 0); // true
 * ```
 */
export function loadPublicSpriteAtlas(): PublicSpriteAtlas {
  return {
    source: 'neatenstein-public',
    frames: buildPublicSpriteAtlasFrames(),
  };
}

/**
 * Selects a deterministic sprite frame from the atlas.
 *
 * The selection is a seeded draw over the frames of the atlas, so the same
 * (atlas, seed, index) tuple always picks the same frame. An atlas that
 * carries no frames has nothing to draw from, and the call returns
 * `undefined` in that case.
 *
 * @param atlas - Public sprite atlas.
 * @param seed - Deterministic seed for frame selection.
 * @param index - Billboard index used to vary the selected frame.
 * @returns The selected frame with positive dimensions, or `undefined` when
 *   the atlas carries no frames.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 * const frame = selectPublicSpriteFrame(atlas, 42, 0);
 * console.log(frame.width > 0); // true
 * console.log(frame.height > 0); // true
 * ```
 */
export function selectPublicSpriteFrame(
  atlas: PublicSpriteAtlas,
  seed: number,
  index: number,
): PublicSpriteFrame {
  const random = createDeterministicRandom(seed ^ (index * 0x9e3779b9));
  const frameIndex = Math.floor(random() * atlas.frames.length);

  return atlas.frames[frameIndex]!;
}

/**
 * Returns the billboard world-space scale relative to its raw sprite pixel size.
 *
 * The returned value is the named
 * {@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} HALF world scale, so a
 * crowd billboard is half the size of the same pixel count in the voxel
 * world. The scale stays DISTANCE-INDEPENDENT: the right-world pass applies
 * the same `focal / distance` perspective projection used for 3D voxels, so
 * distant sprites shrink at the same rate as geometry of the same world size.
 *
 * @param distance - Distance from the camera in meters (kept for API
 *   stability; the caller is responsible for perspective projection).
 * @returns The named half world scale, for every distance.
 *
 * @example
 * ```ts
 * const nearScale = computeBillboardScale(5);
 * const farScale = computeBillboardScale(50);
 * console.log(nearScale).toBeCloseTo(0.5, 6); // 1:2 proportion with the voxel world
 * console.log(farScale).toBeCloseTo(0.5, 6);  // perspective handles shrink
 * ```
 */
export function computeBillboardScale(distance: number): number {
  // Distance is intentionally ignored: the billboard world size stays at the
  // named half scale and the right-world render pass applies perspective
  // projection.
  void distance;

  return NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE;
}

/**
 * Resolves a world yaw angle in degrees to the nearest 8-direction sprite label.
 *
 * @param yawDegrees - World yaw in degrees. 0 faces +Z (front), positive
 *   values rotate toward +X (right).
 * @returns One of the eight Neatenstein direction labels.
 *
 * @example
 * ```ts
 * console.log(resolveDirectionId(0)); // 'front'
 * console.log(resolveDirectionId(90)); // 'right'
 * console.log(resolveDirectionId(180)); // 'back'
 * console.log(resolveDirectionId(-45)); // 'frontLeft'
 * ```
 */
export function resolveDirectionId(yawDegrees: number): string {
  const step = 360 / DIRECTION_ORDER.length;
  const normalized = ((yawDegrees % 360) + 360 + step / 2) % 360;
  const index = Math.floor(normalized / step) % DIRECTION_ORDER.length;

  return DIRECTION_ORDER[index]!;
}

/**
 * Resolves the camera-relative 8-direction index of a sprite facing a target.
 *
 * The conversion is expressed in the neatRacing world compass
 * (yaw = atan2(dx, dz), 0 rad = +Z = 'front'):
 *
 *   cameraYaw   = atan2(forwardX, forwardZ)
 *   facingYaw   = atan2(targetX - robotX, targetZ - robotZ)
 *   relativeYaw = facingYaw - cameraYaw + PI
 *
 * The canonical Neatenstein contract
 * (examples/neatenstein/browser-entry/renderer/sprites.ts:154-156) states the
 * same contract as `cameraYaw + PI - facingYaw`, but Neatenstein measures yaw
 * in the OPPOSITE (counter-clockwise, atan2(dy, dx)) compass. Expressed in
 * the neatRacing compass the same contract is the mirrored form used here,
 * so the atlas anchor is preserved:
 *
 *   index 0 ('front') - the sprite FACES the camera
 *   index 2 ('right') - the sprite nose points to the camera's SCREEN-RIGHT
 *   index 4 ('back')  - the sprite faces AWAY from the camera
 *   index 6 ('left')  - the sprite nose points to the camera's SCREEN-LEFT
 *
 * The neatRacing camera right axis is `(-forwardZ, 0, forwardX)` (the
 * `lookAtInto` xAxis in examples/neatRacing/browser-entry/browser-entry.ts),
 * i.e. the forward axis rotated by -90 degrees in the neatRacing compass,
 * and the public atlas is uploaded without a U flip, so atlas image-right
 * maps to camera screen-right. A seated spectator therefore has to use the
 * 'right' frame when its nose points to the camera's screen-right.
 *
 * The helper is allocation-free: it allocates no arrays, objects, or
 * strings, so it is safe on the per-frame draw path.
 *
 * @param cameraForward - Camera forward axis. When omitted, the legacy
 *   five-argument callers fall back to the camera-to-target bearing, which
 *   matches the neatRacing chase rig because it always looks at the car.
 * @param cameraWorld - Camera world position, consulted by the legacy
 *   fallback heading only.
 * @param robotPosition - Sprite world-space foot position.
 * @param targetPosition - World-space position the sprite faces.
 * @returns Index in {@link DIRECTION_ORDER} (0..7).
 *
 * @example
 * ```ts
 * // The camera looks down -Z from [0, 2, 10]. The robot stands at [5, 0, 0]
 * // and looks at the origin, so its nose points to the camera's screen-left.
 * const index = resolveCameraRelativeDirectionIndex(
 *   [0, 0, -1], // camera forward axis
 *   [0, 2, 10], // camera world position
 *   [5, 0, 0],  // robot foot position
 *   [0, 0, 0],  // point the robot faces
 * );
 *
 * console.log(index); // 6
 * console.log(DIRECTION_ORDER[index]); // 'left'
 * ```
 */
function resolveCameraRelativeDirectionIndex(
  cameraForward: readonly [number, number, number] | undefined,
  cameraWorld: readonly [number, number, number],
  robotPosition: readonly [number, number, number],
  targetPosition: readonly [number, number, number],
): number {
  const cameraYaw = cameraForward !== undefined
    ? Math.atan2(cameraForward[0], cameraForward[2])
    : Math.atan2(
      targetPosition[0] - cameraWorld[0],
      targetPosition[2] - cameraWorld[2],
    );
  const facingYaw = Math.atan2(
    targetPosition[0] - robotPosition[0],
    targetPosition[2] - robotPosition[2],
  );
  const relativeYaw = facingYaw - cameraYaw + Math.PI;
  const normalized =
    ((relativeYaw % PUBLIC_FULL_TURN_RAD) + PUBLIC_FULL_TURN_RAD) %
    PUBLIC_FULL_TURN_RAD;

  return (
    Math.round(normalized / PUBLIC_DIRECTION_STEP_RAD) % DIRECTION_ORDER.length
  );
}

/**
 * Returns the 8-direction stand frame id for a robot looking at a target point.
 *
 * The direction is CAMERA-RELATIVE: the selected frame faces the camera and
 * matches the frame the render pass recomputes at runtime with
 * {@link computePublicRuntimeFrameIndex}.
 *
 * @param atlas - Public sprite atlas (unused; kept for call-site symmetry).
 * @param robotPosition - Robot world-space foot position.
 * @param targetPosition - World-space position the robot faces. At build
 *   time this is the camera position, so it doubles as the camera world
 *   position of the legacy fallback heading.
 * @param cameraForward - Camera forward axis used as the camera heading.
 * @returns Stand frame id, e.g. `'front-stand'`.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 *
 * // The camera sits at [0, 2, 10] and looks down -Z. The robot stands at
 * // [5, 0, 0] and faces the camera, so it is drawn turned towards it.
 * const frameId = selectPublicLookAtCarFrameId(
 *   atlas,
 *   [5, 0, 0],  // robot foot position
 *   [0, 2, 10], // point the robot faces - the camera position
 *   [0, 0, -1], // camera forward axis
 * );
 *
 * console.log(frameId); // 'frontLeft-stand'
 * ```
 */
function selectPublicLookAtCarFrameId(
  atlas: PublicSpriteAtlas,
  robotPosition: readonly [number, number, number],
  targetPosition: readonly [number, number, number],
  cameraForward: readonly [number, number, number],
): string {
  void atlas;
  const directionIndex = resolveCameraRelativeDirectionIndex(
    cameraForward,
    targetPosition,
    robotPosition,
    targetPosition,
  );

  return `${DIRECTION_ORDER[directionIndex]!}-stand`;
}

/**
 * Numeric index of the stand pose inside a direction×pose lookup table.
 *
 * @example
 * ```ts
 * console.log(PUBLIC_POSE_INDEX_STAND); // 0
 * ```
 */
export const PUBLIC_POSE_INDEX_STAND = 0;
/**
 * Numeric index of the first walk pose inside a direction×pose lookup table.
 *
 * @example
 * ```ts
 * console.log(PUBLIC_POSE_INDEX_WALK1); // 1
 * ```
 */
export const PUBLIC_POSE_INDEX_WALK1 = 1;
/**
 * Numeric index of the second walk pose inside a direction×pose lookup table.
 *
 * @example
 * ```ts
 * console.log(PUBLIC_POSE_INDEX_WALK2); // 2
 * ```
 */
export const PUBLIC_POSE_INDEX_WALK2 = 2;
/**
 * Number of poses stored per direction in the runtime lookup table.
 *
 * @example
 * ```ts
 * console.log(PUBLIC_POSE_COUNT); // 3
 * ```
 */
export const PUBLIC_POSE_COUNT = 3;

/** Slow walk cycle period in seconds (must be >= 2.4s per directed fix). */
const PUBLIC_WALK_CYCLE_PERIOD_SECONDS = 2.4;

/**
 * Selects the current walk pose index for a walker at a given clock time.
 *
 * The cycle is slow (2.4s per pose) and deterministic for a fixed clock.
 *
 * @param timeSeconds - Monotonic clock time in seconds.
 * @returns 0 for walk1, 1 for walk2.
 */
function selectPublicWalkPoseIndex(timeSeconds: number): number {
  return Math.floor(timeSeconds / PUBLIC_WALK_CYCLE_PERIOD_SECONDS) % 2;
}

/**
 * Deterministic slow look-around pause for a subset of walkers.
 *
 * A walker glances at the car only during a short window every 16 seconds,
 * and only for ~20% of walkers chosen deterministically from seed+index.
 * The math is branch-free aside from the boolean result and allocates no
 * objects, keeping the per-frame draw path allocation-free.
 */
function isLookAroundPause(
  seed: number,
  index: number,
  timeSeconds: number,
): boolean {
  const period = 16.0;
  const inWindow = (timeSeconds % period) < PUBLIC_WALK_CYCLE_PERIOD_SECONDS;
  if (!inWindow) {
    return false;
  }
  const phase = Math.floor(timeSeconds / period);

  return (((seed ^ index ^ (phase * 0x9e3779b9)) >>> 0) % 5) === 0;
}

/**
 * Computes the runtime atlas frame index for a public/agent billboard.
 *
 * No strings or objects are allocated: the result is read directly from the
 * prebuilt direction×pose index table.
 *
 * The stand frame is CAMERA-RELATIVE: direction index 0 ('front') renders the
 * sprite facing the camera, which is derived from the camera heading and the
 * robot-to-car bearing. EVERY role but the walker role renders that frame:
 * the seated spectators, the standing spectators and the mechanics all show
 * the camera-relative look-at-car STAND pose, and the walker role is the
 * ONLY role of the crowd on the walk cycle.
 *
 * The accent of a billboard follows its role as well: the mechanics wear a
 * TEAM accent, and every other role keeps the colour-diverse accent of the
 * public palette.
 *
 * A billboard WITHOUT a role is read as {@link PUBLIC_WALKER_RUNTIME_ROLE},
 * which selects the walk poses. Every billboard produced by
 * {@link buildPublicBillboards} carries an explicit role, so the case does
 * not arise in a built scene.
 *
 * @param billboard - Billboard with runtime role metadata.
 * @param cameraWorld - Current camera world position. It is the camera
 *   position of the legacy fallback heading.
 * @param carPosition - Live car world position for gaze.
 * @param timeSeconds - Monotonic frame time in seconds.
 * @param indexTable - Prebuilt direction×pose atlas index table.
 * @param cameraForward - Optional camera forward axis. When supplied, the
 *   camera heading comes from that axis; when omitted, the legacy fallback
 *   derives the heading from the camera-to-car bearing (the neatRacing chase
 *   rig always looks at the car).
 * @returns Atlas frame index, or -1 if the frame is missing.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 *
 * // The lookup table maps a direction and a pose to the frame of the LAST
 * // colour block, which is how the render pass builds it.
 * const indexTable = new Int16Array(DIRECTION_ORDER.length * PUBLIC_POSE_COUNT)
 *   .fill(-1);
 * indexTable[PUBLIC_POSE_INDEX_STAND] = 120; // 'front-stand' of colour 5
 *
 * const billboard: PublicBillboard = {
 *   position: [0, 0, 0],
 *   frameId: 'front-stand',
 *   scale: 1,
 *   distance: 10,
 *   quad: [0, 0, 0, 0],
 *   runtimeRole: 'seated',
 *   seed: 42,
 *   index: 0,
 * };
 *
 * // The robot looks at the car, and the camera looks down -Z from behind it.
 * const frameIndex = computePublicRuntimeFrameIndex(
 *   billboard,
 *   [0, 2, 10], // camera world position
 *   [0, 0, 0],  // car position
 *   0,          // frame time in seconds
 *   indexTable,
 *   [0, 0, -1], // camera forward axis
 * );
 *
 * // The atlas is laid out color-major with 24 frames per color, and the
 * // accent color of this robot is color 4, so the frame lands in the
 * // color-4 block that starts at 4 * 24 = 96.
 * console.log(frameIndex); // 96 - the 'front-stand' frame of this robot's colour
 *
 * // A table without that cell has no frame to show.
 * console.log(
 *   computePublicRuntimeFrameIndex(
 *     billboard,
 *     [0, 2, 10],
 *     [0, 0, 0],
 *     0,
 *     new Int16Array(DIRECTION_ORDER.length * PUBLIC_POSE_COUNT).fill(-1),
 *     [0, 0, -1],
 *   ),
 * ); // -1
 * ```
 */
export function computePublicRuntimeFrameIndex(
  billboard: PublicBillboard,
  cameraWorld: readonly [number, number, number],
  carPosition: readonly [number, number, number],
  timeSeconds: number,
  indexTable: Int16Array,
  cameraForward?: readonly [number, number, number],
): number {
  const role = resolvePublicRuntimeRole(billboard);
  const seed = billboard.seed ?? 0;
  const index = billboard.index ?? 0;
  let directionIndex: number;
  let poseIndex: number;

  if (role !== PUBLIC_WALKER_RUNTIME_ROLE || isLookAroundPause(seed, index, timeSeconds)) {
    directionIndex = resolveCameraRelativeDirectionIndex(
      cameraForward,
      cameraWorld,
      billboard.position,
      carPosition,
    );
    poseIndex = PUBLIC_POSE_INDEX_STAND;
  } else {
    directionIndex = billboard.walkDirectionIndex ?? 0;
    poseIndex = selectPublicWalkPoseIndex(timeSeconds) === 0
      ? PUBLIC_POSE_INDEX_WALK1
      : PUBLIC_POSE_INDEX_WALK2;
  }

  const tableIndex = directionIndex * PUBLIC_POSE_COUNT + poseIndex;
  const defaultAtlasIndex = indexTable[tableIndex];

  if (defaultAtlasIndex === undefined || defaultAtlasIndex < 0) {
    return -1;
  }

  // The atlas is laid out color-major and the simple id-split table points at
  // the LAST color block. Offset back to this robot's stable identity color.
  const defaultColorId = PUBLIC_COLOR_COUNT - 1;
  const colorId =
    role === PUBLIC_MECHANIC_RUNTIME_ROLE
      ? resolvePublicTeamAccentColorId(seed, index)
      : computePublicRobotColorId(seed, index);
  const blockSize = DIRECTION_ORDER.length * PUBLIC_POSE_COUNT;
  const atlasIndex = defaultAtlasIndex + (colorId - defaultColorId) * blockSize;

  return atlasIndex;
}

/**
 * Selects a deterministic slow two-step walk frame.
 *
 * The direction comes from a seeded generator keyed by the robot's identity,
 * so the same robot always walks the same way, and the pose alternates with
 * the shared walk-cycle period.
 *
 * @param atlas - Public sprite atlas (unused; kept for call-site symmetry).
 * @param seed - Deterministic seed of the robot.
 * @param index - Deterministic index of the robot.
 * @param timeSeconds - Monotonic clock time in seconds.
 * @returns Walk frame id, e.g. `'front-walk1'`.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 *
 * // The first half of the 2.4 s cycle shows walk pose 1, the second half
 * // walk pose 2.
 * console.log(selectPublicWalkFrame(atlas, 42, 0, 0)); // 'front-walk1'
 * console.log(selectPublicWalkFrame(atlas, 42, 0, 3)); // 'front-walk2'
 * ```
 */
export function selectPublicWalkFrame(
  atlas: PublicSpriteAtlas,
  seed: number,
  index: number,
  timeSeconds: number,
): string {
  void atlas;
  const random = createDeterministicRandom(seed ^ (index * 0x9e3779b9));
  const directionIndex = Math.floor(random() * DIRECTION_ORDER.length) % DIRECTION_ORDER.length;
  const pose = selectPublicWalkPoseIndex(timeSeconds) === 0 ? 'walk1' : 'walk2';

  return `${DIRECTION_ORDER[directionIndex]!}-${pose}`;
}

/**
 * Builds the deterministic seat positions of the spectator rows.
 *
 * A bleacher offers exactly ONE sprite row per level, and the rows are read
 * from the real voxel geometry, so spectator billboards stand on actual seat
 * surfaces instead of on an arbitrary bounding-box top. The anchors come
 * back LEVEL-MAJOR - the lowest platform first, then the next platform, and
 * so on - and every bleacher carries its OWN quota of seated spectators, so
 * no bleacher of the ring is left empty.
 *
 * @param bleachers - Bleacher props emitted by the right-world builder.
 * @returns Seat anchors in world-space [x, topY, z] order, at most
 *   {@link NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER} per bleacher.
 */
function buildBleacherSeatPositions(
  bleachers: readonly RightWorldProp[],
): readonly [number, number, number][] {
  const seats: [number, number, number][] = [];

  for (const bleacher of bleachers) {
    const bleacherAnchors = getBleacherSeatAnchors(bleacher);

    // The per-bleacher quota is DERIVED from the bleacher geometry, so the
    // anchor count of a bleacher has to match it EXACTLY. A mismatch is a
    // geometry change, and it fails loudly here instead of silently
    // truncating the seated crowd of the affected bleacher (or silently
    // adding seats the quota does not cover).
    if (
      bleacherAnchors.length !== NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER
    ) {
      throw new Error(
        `bleacher seat anchor drift: expected ` +
          `${NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER} anchors, got ` +
          `${bleacherAnchors.length}`,
      );
    }

    for (const [x, topY, z] of bleacherAnchors) {
      seats.push([
        x,
        topY + NEATRACING_BILLBOARD_SURFACE_EPSILON_M,
        z,
      ]);
    }
  }

  return seats;
}

/**
 * Builds public/agent billboards for the right-world pass.
 *
 * EVERY bleacher of the ring carries its own quota of seated spectators, so
 * no bleacher of the right world is empty, and the share of the density
 * budget the seated spectators did not use is spent on the agents OUTSIDE
 * the bleachers - spread over the WHOLE track instead of over its tail. The
 * crowd size follows from the world itself: the seated share comes from the
 * bleacher geometry, and the off-bleacher share is a cap on top of it.
 *
 * The crowd carries FOUR roles, and they are built in THAT order:
 *
 * 1. The SEATED spectators on the bleacher seats. They come FIRST, because
 *    the personal-space filter keeps the FIRST billboard of every band, so a
 *    seated seat must never be starved by an off-bleacher agent.
 * 2. The MECHANICS of the inner pit band. They stand on their own pit
 *    pedestal and wear a team accent.
 * 3. The STANDING spectators OUTSIDE the ring, beyond the road edge.
 * 4. The WALKERS on the trackside, in two bands per people-ring sample.
 *
 * The STANDING spectators and the WALKERS are placed through a CLEARANCE
 * search: the agent is pushed further out in steps until its footprint is
 * clear of every scenery prop AND it keeps the 1 m personal space of every
 * billboard placed before it. An agent with no clear spot is SKIPPED - the
 * crowd is a cap, never a quota. The MECHANICS are NOT searched: they are
 * anchored to their own pit pedestal, so they keep their spot by
 * construction.
 *
 * The seated spectators, the standing spectators and the mechanics face the
 * camera with a stand frame so they look at the car, while the walkers use a
 * slow deterministic walk frame. All billboards keep the foot-anchor
 * convention and the named
 * {@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} half world scale.
 *
 * @param input - Determinism tuple for the track layout.
 * @param camera - Camera used for projection and 8-direction frame selection.
 * @param atlas - Public/agent sprite atlas.
 * @returns Ordered list of 2D billboards on the right side of the track.
 *
 * @throws Error - If a bleacher of the world carries a seat-anchor count that
 *   does not match the seated quota derived from the bleacher geometry
 *   (`bleacher seat anchor drift: expected N anchors, got M`), or if the
 *   bleacher geometry of the world defines a different number of levels than
 *   the named level count (`bleacher level count drift: ...`). Both messages
 *   name the drift, so a geometry change fails loudly instead of silently
 *   resizing the crowd.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 * const camera: PublicCamera = {
 *   position: [0, 2, 10],
 *   forward: [0, 0, -1],
 *   right: [1, 0, 0],
 *   near: 0.1,
 *   far: 1000,
 *   fovDegrees: 60,
 * };
 *
 * const billboards = buildPublicBillboards(
 *   { seed: 42, layoutVersion: 1, sizeBucket: 'large' },
 *   camera,
 *   atlas,
 * );
 *
 * console.log(billboards.length > 0); // true
 * console.log(billboards[0]?.runtimeRole); // 'seated' - the first bleacher seat
 * console.log(billboards[0]?.scale); // 0.5 - sprites share the half voxel world scale
 * ```
 */
export function buildPublicBillboards(
  input: TrackSlabBuildInput,
  camera: PublicCamera,
  atlas: PublicSpriteAtlas,
): readonly PublicBillboard[] {
  if (atlas.frames.length === 0) {
    return [];
  }

  const spec = generateTrack(input);
  const samples = spec.splineSamples;
  if (samples.length === 0) {
    return [];
  }

  const random = createDeterministicRandom(resolvePublicSpriteSeed(input, 0));
  const cameraUp = computeCameraUp(camera);
  const props = buildRightWorldProps(input);
  const bleacherSeats = buildBleacherSeatPositions(
    props.filter((prop) => prop.kind === 'bleacher'),
  );
  const seatedCount = bleacherSeats.length;
  const mechanicProps = props.filter((prop) => prop.kind === 'mechanic');
  const outerSlotCount = resolveOuterRingSlotCount(samples.length);
  const peopleSampleIndices = selectPeopleRingSampleIndices(
    samples,
    selectBleacherSlotIndices(outerSlotCount),
    outerSlotCount,
  );
  const offBleacherAllowance = Math.max(
    0,
    Math.min(
      NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE,
      NEATRACING_PUBLIC_BILLBOARD_DENSITY_BUDGET - seatedCount,
    ),
  );
  const mechanicCount = Math.min(mechanicProps.length, offBleacherAllowance);
  const standingCount = Math.min(
    peopleSampleIndices.length,
    offBleacherAllowance - mechanicCount,
  );
  const walkerCount = Math.min(
    peopleSampleIndices.length * NEATRACING_PUBLIC_WALKER_BAND_OFFSETS_M.length,
    offBleacherAllowance - mechanicCount - standingCount,
  );
  const billboards: PublicBillboard[] = [];

  // (a) The SEATED spectators of the bleacher seats come FIRST, so the
  // personal-space filter can never starve a bleacher seat.
  for (let index = 0; index < seatedCount; index += 1) {
    const [seatX, , seatZ] = bleacherSeats[index]!;
    const seatPosition: readonly [number, number, number] = [seatX, 0, seatZ];
    const standFrameId = selectPublicLookAtCarFrameId(
      atlas,
      seatPosition,
      camera.position,
      camera.forward,
    );
    const selectedFrame =
      atlas.frames.find((candidate) => candidate.id === standFrameId) ??
      atlas.frames[0]!;
    const placed = placeBillboardOnSurface(
      seatX,
      seatZ,
      selectedFrame,
      camera,
      cameraUp,
      props,
    );

    billboards.push({
      ...placed,
      runtimeRole: PUBLIC_SEATED_RUNTIME_ROLE,
      seed: input.seed,
      index,
      walkDirectionIndex: 0,
    });
  }

  // (b) The MECHANICS of the inner pit band stand on their own pedestal.
  for (let ordinal = 0; ordinal < mechanicCount; ordinal += 1) {
    const [propX, , propZ] = mechanicProps[ordinal]!.position;
    const standFrameId = selectPublicLookAtCarFrameId(
      atlas,
      [propX, 0, propZ],
      camera.position,
      camera.forward,
    );
    const selectedFrame =
      atlas.frames.find((candidate) => candidate.id === standFrameId) ??
      atlas.frames[0]!;
    const placed = placeBillboardOnSurface(
      propX,
      propZ,
      selectedFrame,
      camera,
      cameraUp,
      props,
    );

    billboards.push({
      ...placed,
      runtimeRole: PUBLIC_MECHANIC_RUNTIME_ROLE,
      seed: input.seed,
      index: billboards.length,
      walkDirectionIndex: 0,
    });
  }

  // (c) The STANDING spectators wait OUTSIDE the ring, beyond the road edge.
  for (let ordinal = 0; ordinal < standingCount; ordinal += 1) {
    const sample = samples[peopleSampleIndices[ordinal]!];
    const frame = resolveSplineSampleFrame(samples, sample.globalIndex);
    const baseOffsetM =
      NEATRACING_PUBLIC_STANDING_TRACKSIDE_OFFSET_M +
      random() * NEATRACING_PUBLIC_STANDING_OFFSET_JITTER_M;
    const placed = placeOffBleacherAgent({
      atlas,
      camera,
      cameraUp,
      props,
      placed: billboards,
      sample,
      frame,
      baseOffsetM,
      resolveFrameId: (x, z) =>
        selectPublicLookAtCarFrameId(
          atlas,
          [x, 0, z],
          camera.position,
          camera.forward,
        ),
    });

    if (placed) {
      billboards.push({
        ...placed,
        runtimeRole: PUBLIC_STANDING_RUNTIME_ROLE,
        seed: input.seed,
        index: billboards.length,
        walkDirectionIndex: 0,
      });
    }
  }

  // (d) The WALKERS stroll the trackside, in two bands per people-ring sample.
  let walkerBudget = walkerCount;
  for (
    let ordinal = 0;
    ordinal < peopleSampleIndices.length && walkerBudget > 0;
    ordinal += 1
  ) {
    const sample = samples[peopleSampleIndices[ordinal]!];
    const frame = resolveSplineSampleFrame(samples, sample.globalIndex);

    for (const bandOffsetM of NEATRACING_PUBLIC_WALKER_BAND_OFFSETS_M) {
      if (walkerBudget <= 0) {
        break;
      }

      const baseOffsetM =
        bandOffsetM + random() * NEATRACING_PUBLIC_WALKER_OFFSET_JITTER_M;
      const walkDirectionIndex =
        Math.floor(random() * DIRECTION_ORDER.length) % DIRECTION_ORDER.length;
      const walkFrameId = `${DIRECTION_ORDER[walkDirectionIndex]!}-walk1`;
      const placed = placeOffBleacherAgent({
        atlas,
        camera,
        cameraUp,
        props,
        placed: billboards,
        sample,
        frame,
        baseOffsetM,
        resolveFrameId: () => walkFrameId,
      });

      if (placed) {
        billboards.push({
          ...placed,
          runtimeRole: PUBLIC_WALKER_RUNTIME_ROLE,
          seed: input.seed,
          index: billboards.length,
          walkDirectionIndex,
        });
        walkerBudget -= 1;
      }
    }
  }

  // Thin the public crowd to the subset that fits the 1 m XZ personal-space
  // band. This is a deterministic build-time filter; the per-frame draw path still
  // allocates no objects.
  return filterRobotsByPersonalSpace(billboards);
}

/** Placement request of one off-bleacher crowd agent. */
interface OffBleacherAgentPlacement {
  /** Public sprite atlas the agent frame is selected from. */
  readonly atlas: PublicSpriteAtlas;
  /** Camera used for projection and frame selection. */
  readonly camera: PublicCamera;
  /** Normalized camera-up vector of the projection. */
  readonly cameraUp: readonly [number, number, number];
  /** Scenery props the agent footprint has to stay clear of. */
  readonly props: readonly RightWorldProp[];
  /** Billboards placed before the agent, for the personal-space test. */
  readonly placed: readonly PublicBillboard[];
  /** Track sample the agent belongs to. */
  readonly sample: SplineSample;
  /** Track-side frame of the sample. */
  readonly frame: SplineSampleFrame;
  /** Base distance beyond the road half-width, in meters. */
  readonly baseOffsetM: number;
  /** Resolves the frame id of a candidate position. */
  readonly resolveFrameId: (x: number, z: number) => string;
}

/**
 * Places ONE off-bleacher agent through the clearance candidate search.
 *
 * The candidate offsets are the base offset and the base offset plus the
 * clearance steps, and the FIRST candidate whose footprint is clear of every
 * scenery prop AND keeps the personal space of every billboard placed before
 * it wins. An agent with no clear candidate is SKIPPED: the crowd is a cap,
 * never a quota.
 *
 * @param request - Placement request of the agent.
 * @returns The placed billboard, or undefined when no candidate was clear.
 */
function placeOffBleacherAgent(
  request: OffBleacherAgentPlacement,
): PublicBillboard | undefined {
  const {
    atlas,
    camera,
    cameraUp,
    props,
    placed,
    sample,
    frame,
    baseOffsetM,
    resolveFrameId,
  } = request;

  for (let step = 0; step < NEATRACING_PUBLIC_CLEARANCE_STEP_COUNT; step += 1) {
    const offset =
      sample.width / 2 +
      baseOffsetM +
      step * NEATRACING_PUBLIC_CLEARANCE_STEP_M;
    const generatorX = sample.x - frame.normalX * offset;
    const generatorY = sample.y - frame.normalY * offset;
    const x = generatorX;
    const z = -generatorY;
    const frameId = resolveFrameId(x, z);
    const selectedFrame =
      atlas.frames.find((candidate) => candidate.id === frameId) ??
      atlas.frames[0]!;
    // The footprint reads the SAME named lever
    // (NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE) and the SAME shared
    // meters-per-pixel constant as the rendered quad, so the placement
    // footprint and the rendered quad can never disagree.
    const [halfWidth, halfDepth] = billboardWorldHalfSize(selectedFrame);

    if (!isOffBleacherSpotClear(x, z, halfWidth, halfDepth, props, placed)) {
      continue;
    }

    return placeBillboardOnSurface(
      x,
      z,
      selectedFrame,
      camera,
      cameraUp,
      props,
    );
  }

  return undefined;
}

/**
 * Reports whether an off-bleacher spot is clear.
 *
 * The spot is clear when the sprite footprint overlaps NO scenery prop and
 * when the agent keeps the personal space of every billboard placed before
 * it.
 *
 * @param x - World-space x of the candidate spot.
 * @param z - World-space z of the candidate spot.
 * @param halfWidth - World-space half-width of the sprite footprint.
 * @param halfDepth - World-space half-depth of the sprite footprint.
 * @param props - Scenery props of the scene.
 * @param placed - Billboards placed before the agent.
 * @returns True when the spot is clear of props and of the personal space.
 */
function isOffBleacherSpotClear(
  x: number,
  z: number,
  halfWidth: number,
  halfDepth: number,
  props: readonly RightWorldProp[],
  placed: readonly PublicBillboard[],
): boolean {
  for (const prop of props) {
    const [propX, , propZ] = prop.position;
    const [propHalfWidth, propHalfDepth] = resolvePropHalfExtent(prop);

    if (
      Math.abs(x - propX) < halfWidth + propHalfWidth &&
      Math.abs(z - propZ) < halfDepth + propHalfDepth
    ) {
      return false;
    }
  }

  for (const billboard of placed) {
    const deltaX = x - billboard.position[0];
    const deltaZ = z - billboard.position[2];

    if (
      Math.hypot(deltaX, deltaZ) <
      NEATRACING_PUBLIC_OFF_BLEACHER_PERSONAL_SPACE_CLEARANCE_M
    ) {
      return false;
    }
  }

  return true;
}

/**
 * Resolves the world-space half-width and half-depth of a scenery prop.
 *
 * A bleacher is a stepped voxel structure, so its half size comes from the
 * real voxel geometry. Every other prop uses the half size of its kind.
 *
 * @param prop - Scenery prop of the scene.
 * @returns The world-space half-width and half-depth of the prop, in meters.
 */
function resolvePropHalfExtent(
  prop: RightWorldProp,
): readonly [number, number] {
  if (prop.kind === 'bleacher' && prop.bleacherGeometry) {
    return getBleacherAxisAlignedHalfSize(prop);
  }

  const [halfWidth, , halfDepth] = resolvePropHalfSize(prop.kind);

  return [halfWidth, halfDepth];
}

/**
 * Resolves the top surface of a scenery prop a billboard can rest on.
 *
 * @param prop - Scenery prop of the scene.
 * @param x - World-space x of the billboard.
 * @param z - World-space z of the billboard.
 * @returns The world-space top surface of the prop, in meters.
 */
function resolvePropTopY(prop: RightWorldProp, x: number, z: number): number {
  if (prop.kind === 'bleacher' && prop.bleacherGeometry) {
    return getBleacherSeatTopY(prop, x, z);
  }

  const [, halfHeight] = resolvePropHalfSize(prop.kind);

  return prop.position[1] + 2 * halfHeight;
}

/**
 * Builds a stable frame list from the Neatenstein robot sprite atlas.
 *
 * Frames are emitted in COLOR-major order: for each palette color, every
 * direction×pose frame is recolored and appended. The frame id stays in the
 * base `direction-pose` form so existing id-split lookup tables keep working.
 * The last color block is therefore the one a simple direction×pose table maps
 * to; runtime helpers offset back to each robot's stable color identity.
 *
 * @returns Decoded 2D RGBA frames recycled from `ROBOT_SPRITE_FRAMES`.
 */
function buildPublicSpriteAtlasFrames(): readonly PublicSpriteFrame[] {
  const frames: PublicSpriteFrame[] = [];
  const poseOrder = ['stand', 'walk1', 'walk2'] as const;

  for (const accent of PUBLIC_NEON_ACCENT_PALETTE) {
    for (const direction of DIRECTION_ORDER) {
      const poseMap =
        (ROBOT_SPRITE_FRAMES as Record<string, Record<string, EncodedFrame>>)[
          direction
        ];
      if (!poseMap) {
        continue;
      }
      for (const pose of poseOrder) {
        const frame = poseMap[pose];
        if (!frame) {
          continue;
        }
        const id = `${direction}-${pose}`;
        frames.push(
          recolorFrameRgba(decodePublicFrame(frame as EncodedFrame, id), accent),
        );
      }
    }
  }

  return frames;
}

/**
 * Decodes one encoded Neatenstein robot frame into a 2D RGBA sprite.
 *
 * @param frame - Encoded palette-index rows.
 * @param id - Stable frame identifier.
 * @returns Decoded RGBA sprite frame with positive dimensions.
 */
function decodePublicFrame(frame: EncodedFrame, id: string): PublicSpriteFrame {
  const logicalHeight = frame.length;
  const logicalWidth = logicalHeight > 0 ? frame[0]!.length : 0;
  const width = logicalWidth * ROBOT_SPRITE_SCALE;
  const height = logicalHeight * ROBOT_SPRITE_SCALE;
  const rgba = new Uint8ClampedArray(width * height * RGBA_CHANNELS);

  for (let y = 0; y < height; y += 1) {
    const logicalY = Math.floor(y / ROBOT_SPRITE_SCALE);
    const row = frame[logicalY];
    if (row === undefined) {
      continue;
    }

    for (let x = 0; x < width; x += 1) {
      const logicalX = Math.floor(x / ROBOT_SPRITE_SCALE);
      const paletteIndex = row[logicalX] ?? 0;
      const color = ROBOT_SPRITE_PALETTE[paletteIndex] ?? [0, 0, 0, 0];
      const offset = (y * width + x) * RGBA_CHANNELS;
      rgba[offset] = color[0]!;
      rgba[offset + 1] = color[1]!;
      rgba[offset + 2] = color[2]!;
      rgba[offset + 3] = color[3]!;
    }
  }

  return { id, width, height, rgba };
}

/**
 * Recolors the saturated red/orange accent pixels of a decoded sprite to a neon
 * accent. This is applied once at atlas build time, so the render path stays
 * allocation-free.
 *
 * @param frame - Decoded sprite frame.
 * @param accent - RGBA accent color to apply.
 * @returns The same frame reference, with accent pixels recolored.
 */
function recolorFrameRgba(
  frame: PublicSpriteFrame,
  accent: readonly [number, number, number, number],
): PublicSpriteFrame {
  const rgba = frame.rgba;

  for (let offset = 0; offset < rgba.length; offset += RGBA_CHANNELS) {
    const alpha = rgba[offset + 3]!;
    if (alpha < 32) {
      continue;
    }

    const red = rgba[offset]!;
    const green = rgba[offset + 1]!;
    const blue = rgba[offset + 2]!;

    // Detect the original Neatenstein red/orange accent family (palette
    // indices 5/6/7). Body grays and whites fall outside this threshold.
    if (red > 150 && green < 120 && blue < 80) {
      rgba[offset] = accent[0];
      rgba[offset + 1] = accent[1];
      rgba[offset + 2] = accent[2];
      // Preserve the original alpha so transparent accents stay transparent.
    }
  }

  return frame;
}

/**
 * Resolves a deterministic unsigned 32-bit seed for the public billboard builder.
 *
 * @param input - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.
 * @param index - Index mixed into the seed, so index streams that start from
 *   the same base seed still draw independent sequences.
 * @returns Unsigned 32-bit seed for the local PRNG.
 */
function resolvePublicSpriteSeed(input: TrackSlabBuildInput, index: number): number {
  let combinedSeed = input.seed ^ (input.layoutVersion * 0x9e3779b9);

  for (const character of input.sizeBucket) {
    combinedSeed = Math.imul(combinedSeed ^ character.charCodeAt(0), 16777619);
  }

  return ((combinedSeed >>> 0) ^ index ^ 0x7075_626c) >>> 0;
}

/**
 * Creates a deterministic xorshift32 PRNG.
 *
 * @param initialSeed - Unsigned 32-bit seed.
 * @returns Stable pseudo-random number generator in the range [0, 1).
 */
function createDeterministicRandom(initialSeed: number): () => number {
  let state = initialSeed === 0 ? 0x6d2b79f5 : initialSeed >>> 0;

  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;

    return (state >>> 0) / 0x1_0000_0000;
  };
}

/**
 * Computes the camera-up vector from the forward and right basis vectors.
 *
 * @param camera - Camera whose forward/right basis is known.
 * @returns Normalized world-up vector perpendicular to forward and right.
 */
function computeCameraUp(camera: PublicCamera): readonly [number, number, number] {
  const upX = camera.forward[1] * camera.right[2] - camera.forward[2] * camera.right[1];
  const upY = camera.forward[2] * camera.right[0] - camera.forward[0] * camera.right[2];
  const upZ = camera.forward[0] * camera.right[1] - camera.forward[1] * camera.right[0];
  const length = Math.hypot(upX, upY, upZ) || 1;

  return [upX / length, upY / length, upZ / length];
}

/**
 * Projects one world-space billboard into screen space.
 *
 * @param position - World-space position in meters.
 * @param frame - Selected 2D sprite frame.
 * @param camera - Camera used for projection and distance scaling.
 * @param cameraUp - Normalized camera-up vector.
 * @returns Billboard with positive screen-space quad and perspective-correct scale.
 */
function buildBillboard(
  position: readonly [number, number, number],
  frame: PublicSpriteFrame,
  camera: PublicCamera,
  cameraUp: readonly [number, number, number],
): PublicBillboard {
  const relX = position[0] - camera.position[0];
  const relY = position[1] - camera.position[1];
  const relZ = position[2] - camera.position[2];

  const distance = Math.max(BILLBOARD_MIN_DISTANCE_M, Math.hypot(relX, relY, relZ));
  const scale = computeBillboardScale(distance);
  const focal = 1 / Math.tan((camera.fovDegrees * Math.PI / 180) / 2);
  const rightDot = relX * camera.right[0] + relY * camera.right[1] + relZ * camera.right[2];
  const upDot = relX * cameraUp[0] + relY * cameraUp[1] + relZ * cameraUp[2];
  const screenX = (focal * rightDot) / distance;
  const screenY = (focal * upDot) / distance;
  const halfWidth = (frame.width * scale) / 2;
  const halfHeight = (frame.height * scale) / 2;

  return {
    position,
    frameId: frame.id,
    scale,
    distance,
    quad: [screenX - halfWidth, screenY - halfHeight, screenX + halfWidth, screenY + halfHeight],
  };
}

/**
 * Returns the highest surface a billboard footprint must rest on.
 *
 * @param x - World-space x center of the billboard.
 * @param z - World-space z center of the billboard.
 * @param halfWidth - World-space half-width of the billboard footprint.
 * @param halfDepth - World-space half-depth of the billboard footprint.
 * @param props - Scenery props to test against.
 * @returns Y value of the highest overlapping prop top, or 0 for the ground plane.
 */
function resolveBillboardBaseY(
  x: number,
  z: number,
  halfWidth: number,
  halfDepth: number,
  props: readonly RightWorldProp[],
): number {
  let base = 0;

  for (const prop of props) {
    const [px, , pz] = prop.position;
    const [propHalfWidth, propHalfDepth] = resolvePropHalfExtent(prop);
    const propTop = resolvePropTopY(prop, x, z);

    if (
      Math.abs(x - px) < halfWidth + propHalfWidth &&
      Math.abs(z - pz) < halfDepth + propHalfDepth &&
      propTop > base
    ) {
      base = propTop;
    }
  }

  return base;
}

/**
 * Computes the world-space half-width and half-height of a billboard.
 *
 * The footprint reads the SAME named constants as the rendered quad - the
 * {@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} lever and the
 * {@link NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M} meter scale - so the
 * placement footprint and the rendered quad can never disagree.
 *
 * @param frame - Sprite frame whose pixel dimensions are known.
 * @returns Half-width and half-height in meters.
 */
function billboardWorldHalfSize(
  frame: PublicSpriteFrame,
): readonly [number, number] {
  const worldScale =
    NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE *
    NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M;

  return [(frame.width * worldScale) / 2, (frame.height * worldScale) / 2];
}

/**
 * Places a billboard so its base rests on the ground or on top of any scenery
 * prop its footprint overlaps. The returned position is the foot/base anchor
 * (y = surface + a small clearance), not the geometric center. The render pass
 * expands the quad upward from this anchor so the sprite remains surface-aligned
 * at any runtime camera distance.
 *
 * @param x - World-space x center.
 * @param z - World-space z center.
 * @param frame - Selected sprite frame.
 * @param camera - Camera used for projection.
 * @param cameraUp - Normalized camera-up vector.
 * @param props - Scenery props that can elevate the billboard.
 * @returns Billboard with a surface-aligned base anchor and consistent distance/scale.
 */
function placeBillboardOnSurface(
  x: number,
  z: number,
  frame: PublicSpriteFrame,
  camera: PublicCamera,
  cameraUp: readonly [number, number, number],
  props: readonly RightWorldProp[],
): PublicBillboard {
  let positionY = 0;

  for (let iteration = 0; iteration < 8; iteration += 1) {
    const [halfWidth, halfDepth] = billboardWorldHalfSize(frame);
    const baseY = resolveBillboardBaseY(x, z, halfWidth, halfDepth, props);
    const targetY = baseY + NEATRACING_BILLBOARD_SURFACE_EPSILON_M;

    if (Math.abs(positionY - targetY) <= 0.01) {
      return buildBillboard([x, targetY, z], frame, camera, cameraUp);
    }

    positionY = targetY;
  }

  return buildBillboard([x, positionY, z], frame, camera, cameraUp);
}

