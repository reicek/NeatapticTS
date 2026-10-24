/**
 * Red contract tests for the recycled Neatenstein public/agent billboard
 * sprites (slice P1-neatenstein-public, Step 14D, AC-049).
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - {@link loadPublicSpriteAtlas} returns a non-empty atlas whose source is
 *   'neatenstein-public'.
 * - {@link selectPublicSpriteFrame} returns a deterministic frame with positive
 *   dimensions for a given (seed, index).
 * - {@link buildPublicBillboards} returns billboards on the right side of the
 *   track with positive screen-space quads.
 * - {@link computeBillboardScale} returns the named
 *   NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE half world scale so perspective
 *   projection handles distance shrink (AC-054 ratio + AC-077).
 * - Billboards are 2D sprite quads, not voxel snapshots.
 *
 * RED-A (slice P1-sprite-scale-red-contracts, Step 14L, AC-077) adds the
 * named world-scale lever contract: the module exports the NAMED constant
 * NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE = 0.5, and computeBillboardScale
 * returns EXACTLY that value for every camera distance.
 *
 * RED-B (slice P1-sprite-scale-red-contracts, Step 14L, AC-077) adds the
 * placement-footprint contract: the shared meters-per-pixel constant
 * NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M is exported, and a 192 x 192 px
 * crowd frame reports a 4.8 m world footprint (a 2.4 m half-extent) instead
 * of the pre-14L 9.6 m.
 */

import { describe, expect, it } from '@jest/globals';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';
import { resolveSplineSampleFrame } from '../../racing_curriculum/track/track.spline.utils';
import {
  buildRightWorldProps,
  getBleacherAxisAlignedHalfSize,
  getBleacherSeatTopY,
} from './right-world.builder';
import { resolvePropHalfSize } from './right-world.prop-sizes';
import type { TrackSlabBuildInput } from './track.slabs';
import {
  buildPublicBillboards,
  computeBillboardScale,
  computePublicRuntimeFrameIndex,
  DIRECTION_ORDER,
  loadPublicSpriteAtlas,
  NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M,
  NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE,
  PUBLIC_POSE_COUNT,
  PUBLIC_POSE_INDEX_STAND,
  PUBLIC_POSE_INDEX_WALK1,
  PUBLIC_POSE_INDEX_WALK2,
  resolvePublicRuntimeRole,
  selectPublicSpriteFrame,
  selectPublicWalkFrame,
} from './public.sprite';
import * as publicSprite from './public.sprite';

/** Canonical determinism tuple reused from the track slab builder. */
const FIXTURE: TrackSlabBuildInput = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'small',
} as const;

/** Minimal camera used to exercise projection and distance scaling. */
const FIXTURE_CAMERA = {
  position: [0, 2, -10] as const,
  forward: [0, 0, 1] as const,
  right: [1, 0, 0] as const,
  near: 0.1,
  far: 1000,
  fovDegrees: 60,
} as const;

/**
 * Live car position used by the camera-relative gaze assertions.
 *
 * The car sits 10 m ahead of {@link FIXTURE_CAMERA} along the camera's +Z
 * forward axis, so the chase camera looks over the car's shoulder at the
 * crowd: a seated spectator that faces the car therefore faces the camera.
 */
const FIXTURE_CAR_POSITION = [0, 0, 0] as const;

/** Scene-world z mirrors generator y, so generator-space queries use -z. */
const sceneZToGeneratorY = (sceneZ: number): number => -sceneZ;

/** Finds the closest spline sample to a world-space point. */
function nearestSample(
  samples: readonly SplineSample[],
  sceneX: number,
  sceneZ: number,
): SplineSample {
  const generatorY = sceneZToGeneratorY(sceneZ);
  let best: SplineSample | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const sample of samples) {
    const distance = Math.hypot(sample.x - sceneX, sample.y - generatorY);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = sample;
    }
  }

  if (!best) {
    throw new Error('nearestSample called with an empty sample list');
  }

  return best;
}

/**
 * Computes the world-space half-width and half-height of a billboard.
 *
 * The meter scale is read from the SHARED module export instead of a local
 * copy, so the reported footprint follows the module and a future meter-scale
 * change cannot silently invalidate the contract.
 */
function billboardHalfSize(
  frame: ReturnType<typeof loadPublicSpriteAtlas>['frames'][number],
  distance: number,
): readonly [number, number] {
  const scale =
    computeBillboardScale(distance) * NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M;

  return [(frame.width * scale) / 2, (frame.height * scale) / 2];
}

/**
 * Reads a numeric module export by NAME.
 *
 * The RED-A and RED-B contracts of slice P1-sprite-scale-red-contracts
 * target the named world-scale lever
 * NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE and the shared meters-per-pixel
 * constant NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M. A named IMPORT of a
 * missing export would break the WHOLE suite, so the module namespace is
 * read through an index signature instead, and a missing export is reported
 * as a NAMED contract failure instead of an opaque undefined.
 *
 * @param name - Name of the exported constant.
 * @returns The exported numeric constant.
 */
function readNumericModuleExport(name: string): number {
  const value = (publicSprite as unknown as Record<string, unknown>)[name];

  if (typeof value !== 'number') {
    throw new Error(
      `public.sprite does not export the numeric constant ${name}`,
    );
  }

  return value;
}

/**
 * Computes the world-space half-width and half-height of a crowd billboard.
 *
 * The mirror of the module-private `billboardWorldHalfSize`, but the
 * meters-per-pixel constant is read from the SHARED module export instead of
 * a local copy, so the reported footprint follows the meter scale of the
 * module and a future meter-scale change cannot silently invalidate the
 * contract.
 *
 * @param frame - Crowd sprite frame whose pixel dimensions are known.
 * @param distance - Distance from the camera in meters.
 * @returns Half-width and half-height in meters.
 */
function crowdBillboardWorldHalfSize(
  frame: ReturnType<typeof loadPublicSpriteAtlas>['frames'][number],
  distance: number,
): readonly [number, number] {
  const metersPerPixel = readNumericModuleExport(
    'NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M',
  );
  const scale = computeBillboardScale(distance) * metersPerPixel;

  return [(frame.width * scale) / 2, (frame.height * scale) / 2];
}

/** Returns true when two axis-aligned rectangles overlap in the XZ plane. */
function aabbsOverlapXY(
  ax: number,
  az: number,
  aHx: number,
  aHz: number,
  bx: number,
  bz: number,
  bHx: number,
  bHz: number,
): boolean {
  return Math.abs(ax - bx) < aHx + bHx && Math.abs(az - bz) < aHz + bHz;
}

/** Canonical 8-direction order used by the Neatenstein robot sprite data. */
const ROBOT_DIRECTION_ORDER = [
  'front',
  'frontRight',
  'right',
  'backRight',
  'back',
  'backLeft',
  'left',
  'frontLeft',
] as const;

/** Parses the direction token out of a public sprite frame id. */
function directionFromPublicFrameId(frameId: string): string {
  return frameId.split('-')[0]!;
}

/**
 * Returns the direction label rendered by a runtime atlas frame index.
 *
 * The runtime frame index addresses the COLOR-major atlas, so the rendered
 * direction is recovered from the frame id instead of from the index itself.
 */
function directionOfRuntimeFrameIndex(
  atlas: ReturnType<typeof loadPublicSpriteAtlas>,
  frameIndex: number,
): string {
  const frame = atlas.frames[frameIndex];

  return frame ? directionFromPublicFrameId(frame.id) : 'out-of-range';
}

/**
 * Returns the minimal number of 45-degree atlas steps between two direction
 * labels (0 = identical, 4 = opposite).
 */
function directionStepDistance(a: string, b: string): number {
  const aIndex = (DIRECTION_ORDER as readonly string[]).indexOf(a);
  const bIndex = (DIRECTION_ORDER as readonly string[]).indexOf(b);
  if (aIndex < 0 || bIndex < 0) {
    return DIRECTION_ORDER.length;
  }
  const raw = Math.abs(aIndex - bIndex);

  return Math.min(raw, DIRECTION_ORDER.length - raw);
}

/**
 * Returns the CAMERA-RELATIVE direction label a spectator must use.
 *
 * AC-068 (slice P1-crowd-facing-relative) replaces the world-absolute
 * robot-to-car bearing with a camera-relative conversion. The expectation is
 * derived GEOMETRICALLY from the camera basis instead of re-implementing the
 * implementation algebra, so the test can never pass by mirroring the same
 * formula twice:
 *
 *   f        = normalize(cameraForward.xz)  // camera forward, horizontal part
 *   r        = (-fZ, 0, fX)                 // camera screen-right axis
 *   d        = normalize(gazeTarget.xz - robotPosition.xz)
 *   along    = d . f
 *   lateral  = d . r
 *   phi      = atan2(lateral, along)        // 0 = facing away, PI = facing the camera
 *   relative = ((PI - phi) mod 2PI)
 *   index    = round(relative / (PI/4)) mod 8
 *
 * The neatRacing camera right axis is `(-fZ, 0, fX)` (the `lookAtInto` xAxis
 * in examples/neatRacing/browser-entry/browser-entry.ts), so a camera looking
 * toward +Z sees world +X on its LEFT, and the atlas image-right maps to the
 * camera screen-right because the public atlas is uploaded without a U flip.
 * Direction index 0 ('front') therefore means "the sprite faces the camera",
 * index 2 ('right') means "the sprite nose points to the camera's
 * screen-right", index 4 ('back') means "the sprite faces away from the
 * camera" and index 6 ('left') means "the sprite nose points to the camera's
 * screen-left".
 */
function expectedCameraRelativeDirection(
  cameraForward: readonly [number, number, number],
  robotPosition: readonly [number, number, number],
  gazeTarget: readonly [number, number, number],
): string {
  const forwardLength = Math.hypot(cameraForward[0], cameraForward[2]) || 1;
  const forwardX = cameraForward[0] / forwardLength;
  const forwardZ = cameraForward[2] / forwardLength;
  const rightX = -forwardZ;
  const rightZ = forwardX;
  const gazeX = gazeTarget[0] - robotPosition[0];
  const gazeZ = gazeTarget[2] - robotPosition[2];
  const gazeLength = Math.hypot(gazeX, gazeZ) || 1;
  const directionX = gazeX / gazeLength;
  const directionZ = gazeZ / gazeLength;
  const along = directionX * forwardX + directionZ * forwardZ;
  const lateral = directionX * rightX + directionZ * rightZ;
  const phi = Math.atan2(lateral, along);
  const step = Math.PI / 4;
  const fullTurn = 2 * Math.PI;
  const relative = ((Math.PI - phi) % fullTurn + fullTurn) % fullTurn;

  return ROBOT_DIRECTION_ORDER[Math.round(relative / step) % 8]!;
}

/**
 * EXPECTED NEW API (AC-068, slice P1-crowd-facing-relative):
 * `computePublicRuntimeFrameIndex` must stop discarding the camera
 * (`void cameraWorld;` in public.sprite.ts) and derive the atlas frame index
 * from the camera HEADING. The minimal shape used by the RED tests below is
 * an OPTIONAL 6th parameter that carries the camera forward vector:
 *
 *   computePublicRuntimeFrameIndex(
 *     billboard, cameraWorld, carPosition, timeSeconds, indexTable,
 *     cameraForward,
 *   )
 *
 * The camera-relative conversion then uses the neatRacing world compass form
 * (yaw = atan2(dx, dz), 0 rad = +Z = 'front'):
 *
 *   cameraYaw      = atan2(forwardX, forwardZ)   // neatRacing world compass
 *   facingYaw      = atan2(carX - robotX, carZ - robotZ)
 *   relativeYaw    = facingYaw - cameraYaw + PI
 *   directionIndex = round(relativeYaw / 45deg) mod 8
 *
 * so direction index 0 ('front') is the frame that faces the camera, index 2
 * ('right') is the frame whose nose points to the camera's screen-right, and
 * index 6 ('left') is its screen-left mirror. The canonical Neatenstein rule
 * (examples/neatenstein/browser-entry/renderer/sprites.ts:154-156) is written
 * as `cameraYaw + PI - facingYaw` because Neatenstein measures yaw in the
 * OPPOSITE (counter-clockwise) compass; expressed in the neatRacing compass
 * the equivalent contract is the mirrored form used here. The cast below
 * keeps this test file COMPILING before that parameter exists, so the RED
 * failure is an assertion failure and not a type error.
 */
type CameraAwarePublicRuntimeFrameIndex = (
  billboard: Parameters<typeof computePublicRuntimeFrameIndex>[0],
  cameraWorld: readonly [number, number, number],
  carPosition: readonly [number, number, number],
  timeSeconds: number,
  indexTable: Int16Array,
  cameraForward?: readonly [number, number, number],
) => number;

const computeCameraAwareRuntimeFrameIndex =
  computePublicRuntimeFrameIndex as unknown as CameraAwarePublicRuntimeFrameIndex;

/**
 * Returns the direction×pose lookup slot encoded by a color-major atlas
 * frame index.
 *
 * The public atlas is color-major (6 accents x 8 directions x 3 poses = 144
 * frames), so `frameIndex % (8 * 3)` is exactly the slot the runtime lookup
 * table addresses: `directionIndex * PUBLIC_POSE_COUNT + poseIndex`.
 */
function directionPoseSlotOfRuntimeFrameIndex(frameIndex: number): number {
  return frameIndex % (DIRECTION_ORDER.length * PUBLIC_POSE_COUNT);
}

/**
 * Builds a controlled seated-spectator billboard at a fixed world position.
 *
 * A hand-built billboard keeps the robot-to-car bearing exact, so the
 * camera-relative contract can be asserted without scene geometry.
 */
function seatedSpectatorBillboard(
  position: readonly [number, number, number],
): Parameters<typeof computePublicRuntimeFrameIndex>[0] {
  return {
    position,
    runtimeRole: 'seated',
    seed: 1,
    index: 0,
    walkDirectionIndex: 0,
    frameId: 'front-stand',
    scale: 1,
    distance: 10,
    quad: [0, 0, 1, 1],
  };
}

/** Builds the public runtime direction×pose atlas-index lookup table. */
function buildPublicRuntimeIndexTable(
  atlas: ReturnType<typeof loadPublicSpriteAtlas>,
): Int16Array {
  const table = new Int16Array(
    DIRECTION_ORDER.length * PUBLIC_POSE_COUNT,
  ).fill(-1);

  for (let frameIndex = 0; frameIndex < atlas.frames.length; frameIndex += 1) {
    const frame = atlas.frames[frameIndex]!;
    const [direction, pose] = frame.id.split('-');
    const directionIndex = (DIRECTION_ORDER as readonly string[]).indexOf(direction);
    let poseIndex = -1;
    if (pose === 'stand') {
      poseIndex = PUBLIC_POSE_INDEX_STAND;
    } else if (pose === 'walk1') {
      poseIndex = PUBLIC_POSE_INDEX_WALK1;
    } else if (pose === 'walk2') {
      poseIndex = PUBLIC_POSE_INDEX_WALK2;
    }
    if (directionIndex >= 0 && poseIndex >= 0) {
      table[directionIndex * PUBLIC_POSE_COUNT + poseIndex] = frameIndex;
    }
  }

  return table;
}

/**
 * The public runtime role set of the re-populated crowd (AC-074 role set).
 *
 * The set carries the four crowd roles: the bleacher SEATED spectators, the
 * off-bleacher WALKERS, the off-bleacher STANDING spectators, and the
 * inner-ring MECHANICS.
 */
const PUBLIC_RUNTIME_ROLE_SET = ['seated', 'walker', 'standing', 'mechanic'] as const;

/** One role of the public runtime role set. */
type PublicRuntimeRole = (typeof PUBLIC_RUNTIME_ROLE_SET)[number];

/**
 * Builds a role carrier for one role of the public runtime role set.
 *
 * The carrier is cast through `unknown` because the resolver's role union
 * still declares only the two bleacher roles ('seated' | 'walker'). The cast
 * keeps THIS FILE COMPILING, so the RED failure is an assertion failure and
 * not a type error - the same convention the camera-aware frame-index cast
 * above uses.
 */
function publicRuntimeRoleCarrier(
  role: PublicRuntimeRole,
): Parameters<typeof resolvePublicRuntimeRole>[0] {
  return {
    runtimeRole: role,
  } as unknown as Parameters<typeof resolvePublicRuntimeRole>[0];
}

/**
 * Builds a controlled billboard that carries one public runtime role.
 *
 * The billboard stands 5 m north of the car, so the robot-to-car bearing is
 * exactly -Z and the camera-relative direction is fully determined. The
 * seed/index pair never triggers the deterministic look-around pause, so the
 * RESOLVED ROLE alone decides the rendered pose.
 */
function publicRuntimeRoleBillboard(
  role: PublicRuntimeRole,
): Parameters<typeof computePublicRuntimeFrameIndex>[0] {
  return {
    position: [0, 0, 5],
    runtimeRole: role,
    seed: 1,
    index: 0,
    walkDirectionIndex: 0,
    frameId: 'front-stand',
    scale: 1,
    distance: 10,
    quad: [0, 0, 1, 1],
  } as unknown as Parameters<typeof computePublicRuntimeFrameIndex>[0];
}

/**
 * Returns the pose index a runtime atlas frame index renders.
 *
 * The public atlas is color-major (6 accents x 8 directions x 3 poses), so
 * the pose survives the per-robot color offset in the low bits of the
 * direction x pose slot.
 */
function poseIndexOfRuntimeFrameIndex(frameIndex: number): number {
  return directionPoseSlotOfRuntimeFrameIndex(frameIndex) % PUBLIC_POSE_COUNT;
}

/** Computes a 0..360 hue angle, returning NaN for grayscale pixels. */
function rgbToHue(red: number, green: number, blue: number): number {
  const max = Math.max(red, green, blue);
  const min = Math.min(red, green, blue);

  if (max === min) {
    return Number.NaN;
  }

  const delta = max - min;
  let hue: number;

  if (max === red) {
    hue = ((green - blue) / delta) % 6;
  } else if (max === green) {
    hue = (blue - red) / delta + 2;
  } else {
    hue = (red - green) / delta + 4;
  }

  hue *= 60;
  if (hue < 0) {
    hue += 360;
  }

  return hue;
}

/** Collects distinct non-grayscale, non-transparent hue buckets from a frame. */
function distinctNeonHues(rgba: Uint8ClampedArray): Set<number> {
  const hues = new Set<number>();

  for (let index = 0; index < rgba.length; index += 4) {
    const red = rgba[index]!;
    const green = rgba[index + 1]!;
    const blue = rgba[index + 2]!;
    const alpha = rgba[index + 3]!;

    if (alpha < 32) {
      continue;
    }

    if (red === green && green === blue) {
      continue;
    }

    const max = Math.max(red, green, blue);
    const min = Math.min(red, green, blue);
    if (max > 0 && (max - min) / max < 0.5) {
      continue;
    }

    const hue = rgbToHue(red, green, blue);
    if (!Number.isNaN(hue)) {
      hues.add(Math.round(hue / 30) * 30);
    }
  }

  return hues;
}

/**
 * Returns the single most common saturated accent hue bucket in a frame,
 * or `null` when no accent pixels are found. Used to verify that a robot
 * keeps the same accent color even when its selected direction frame
 * changes at runtime.
 */
function dominantAccentHue(rgba: Uint8ClampedArray): number | null {
  const counts = new Map<number, number>();

  for (let index = 0; index < rgba.length; index += 4) {
    const red = rgba[index]!;
    const green = rgba[index + 1]!;
    const blue = rgba[index + 2]!;
    const alpha = rgba[index + 3]!;

    if (alpha < 32) {
      continue;
    }

    if (red === green && green === blue) {
      continue;
    }

    const max = Math.max(red, green, blue);
    const min = Math.min(red, green, blue);
    if (max > 0 && (max - min) / max < 0.5) {
      continue;
    }

    const hue = rgbToHue(red, green, blue);
    if (Number.isNaN(hue)) {
      continue;
    }

    const bucket = Math.round(hue / 30) * 30;
    counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
  }

  let bestHue: number | null = null;
  let bestCount = 0;
  for (const [hue, count] of counts) {
    if (count > bestCount) {
      bestCount = count;
      bestHue = hue;
    }
  }

  return bestHue;
}

describe('public.sprite (P1-neatenstein-public, AC-049)', () => {
  describe('sprite atlas load and selection', () => {
    it('loads a non-empty atlas recycled from Neatenstein public sprites', () => {
      const atlas = loadPublicSpriteAtlas();

      expect(atlas.source).toBe('neatenstein-public');
      expect(atlas.frames.length).toBeGreaterThan(0);
    });

    it('selects a deterministic sprite frame with positive dimensions', () => {
      const atlas = loadPublicSpriteAtlas();
      const frameA = selectPublicSpriteFrame(atlas, 42, 0);
      const frameB = selectPublicSpriteFrame(atlas, 42, 0);

      expect(frameA.id).toBe(frameB.id);
      expect(frameA.width).toBeGreaterThan(0);
      expect(frameA.height).toBeGreaterThan(0);
    });

        it('stores decoded rows top-to-bottom so UV v=0 maps to the sprite top (OBS-6)', () => {
          const atlas = loadPublicSpriteAtlas();
          const frame = atlas.frames[0]!;
          const rowHasOpaquePixel = (row: Uint8ClampedArray): boolean => {
            for (let index = 3; index < row.length; index += 4) {
              if (row[index]! >= 32) {
                return true;
              }
            }
            return false;
          };
          const topRow = frame.rgba.subarray(0, frame.width * 4);
          const bottomRow = frame.rgba.subarray(
            (frame.height - 1) * frame.width * 4,
            frame.height * frame.width * 4,
          );

          expect(rowHasOpaquePixel(topRow)).toBe(true);
          expect(rowHasOpaquePixel(bottomRow)).toBe(true);
        });
      });

      describe('billboard placement and scale', () => {
    it('places billboards on the right side with positive screen-space quads', () => {
      const atlas = loadPublicSpriteAtlas();
      const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
      const spec = generateTrack(FIXTURE);

      expect(billboards.length).toBeGreaterThan(0);

      const allRightAndUpright = billboards.every((billboard) => {
        const sample = nearestSample(
          spec.splineSamples,
          billboard.position[0],
          billboard.position[2],
        );
        const frame = resolveSplineSampleFrame(
          spec.splineSamples,
          sample.globalIndex,
        );
        const generatorY = sceneZToGeneratorY(billboard.position[2]);
        const deltaX = billboard.position[0] - sample.x;
        const deltaY = generatorY - sample.y;
        const lateral = deltaX * frame.normalX + deltaY * frame.normalY;

        return (
          lateral < 0 &&
          billboard.quad[2] > billboard.quad[0] &&
          billboard.quad[3] > billboard.quad[1]
        );
      });

      expect(allRightAndUpright).toBe(true);
    });

    it('exposes a positive distance-aware scale for every billboard', () => {
      const atlas = loadPublicSpriteAtlas();
      const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);

      const allScaled = billboards.every(
        (billboard) =>
          billboard.scale > 0 &&
          Number.isFinite(billboard.distance) &&
          billboard.distance > 0,
      );

      expect(allScaled).toBe(true);
    });

    it('keeps billboard world scale at 1:2 of the voxel world scale', () => {
      // AC-077: the world scale is the NAMED half-scale lever, and it stays
      // DISTANCE-INDEPENDENT: the perspective shrink is handled by the shared
      // focal/distance projection, not by a per-distance scale factor.
      const near = computeBillboardScale(10);
      const far = computeBillboardScale(200);

      expect(near).toBe(NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE);
      expect(far).toBe(NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE);
      expect(NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE * 2).toBe(1);
    });
  });

  describe('2D preferred over voxel fallback', () => {
    it('represents every billboard as a 2D sprite quad, not a voxel snapshot', () => {
      const atlas = loadPublicSpriteAtlas();
      const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);

      const all2D = billboards.every(
        (billboard) => !('voxels' in billboard) && !('voxelSnapshot' in billboard),
      );

      expect(all2D).toBe(true);
    });
  });
});

describe('public.sprite perspective scale (P1-sprite-perspective-scale, AC-054)', () => {
  it('projects billboard size with the same distance ratio as a same-world-size voxel', () => {
    const atlas = loadPublicSpriteAtlas();
    const frame = atlas.frames[0]!;
    const worldPixelScale = 0.05;
    const fovDegrees = 60;
    const focal = 1 / Math.tan((fovDegrees * Math.PI) / 360);

    const nearDistance = 20;
    const farDistance = 80;

    // A voxel of the same world width as the sprite follows pure perspective.
    const voxelWorldWidth = frame.width * worldPixelScale;
    const voxelScreenRatio =
      (voxelWorldWidth * focal / nearDistance) /
      (voxelWorldWidth * focal / farDistance);

    // The billboard is scaled by computeBillboardScale, then perspective-projected.
    const billboardWorldWidth = (distance: number) =>
      frame.width * worldPixelScale * computeBillboardScale(distance);
    const billboardScreenRatio =
      (billboardWorldWidth(nearDistance) * focal / nearDistance) /
      (billboardWorldWidth(farDistance) * focal / farDistance);

    expect(billboardScreenRatio).toBeCloseTo(voxelScreenRatio, 6);
  });

  it('renders at exactly half the pre-14L projected size of a same-world-size voxel', () => {
    // AC-054 keeps the RATIO contract (the test above), while AC-077
    // SUPERSEDES the old ABSOLUTE 1:1 clause: the crowd sprite is HALF the
    // world size of the same pixel count, so the projected size is exactly
    // HALF the pre-14L size. The factor is the NAMED lever, never a literal.
    const atlas = loadPublicSpriteAtlas();
    const frame = atlas.frames[0]!;
    const worldPixelScale = NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M;
    const fovDegrees = 60;
    const focal = 1 / Math.tan((fovDegrees * Math.PI) / 360);
    const distance = 25;

    const preHalvedWorldWidth = frame.width * worldPixelScale;
    const preHalvedScreenWidth = preHalvedWorldWidth * focal / distance;

    const billboardWorldWidth =
      frame.width * worldPixelScale * computeBillboardScale(distance);
    const billboardScreenWidth = billboardWorldWidth * focal / distance;

    expect(billboardScreenWidth).toBeCloseTo(
      preHalvedScreenWidth * NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE,
      6,
    );
  });
});

describe('public.sprite elevation anchor (P1-sprite-elevation, AC-052)', () => {
  it('places public/agent billboards with feet on the ground or seat plane', () => {
    const atlas = loadPublicSpriteAtlas();
    const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
    const props = buildRightWorldProps(FIXTURE);

    const allRestOnSurface = billboards.every((billboard) => {
      const frame = atlas.frames.find((candidate) => candidate.id === billboard.frameId);
      if (!frame) {
        return false;
      }
      const [halfWidth, halfDepth] = billboardHalfSize(frame, billboard.distance);
      const [bx, by, bz] = billboard.position;
      const billboardBottom = by;

      let expectedBase = 0;
      for (const prop of props) {
        const [px, py, pz] = prop.position;
        let pHx: number;
        let pHz: number;
        let propTop: number;
        if (prop.kind === 'bleacher' && prop.bleacherGeometry) {
          [pHx, pHz] = getBleacherAxisAlignedHalfSize(prop);
          propTop = getBleacherSeatTopY(prop, bx, bz);
        } else {
          const [phx, phy, phz] = resolvePropHalfSize(prop.kind);
          pHx = phx;
          pHz = phz;
          propTop = py + 2 * phy;
        }
        if (
          aabbsOverlapXY(bx, bz, halfWidth, halfDepth, px, pz, pHx, pHz) &&
          propTop > expectedBase
        ) {
          expectedBase = propTop;
        }
      }

      return Math.abs(billboardBottom - expectedBase) <= 0.1;
    });

    expect(allRestOnSurface).toBe(true);
  });

  it('keeps public/agent billboards above right-world voxel scenery', () => {
    const atlas = loadPublicSpriteAtlas();
    const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
    const props = buildRightWorldProps(FIXTURE);

    const intersects = billboards.some((billboard) => {
      const frame = atlas.frames.find((candidate) => candidate.id === billboard.frameId);
      if (!frame) {
        return false;
      }
      const [halfWidth, halfDepth] = billboardHalfSize(frame, billboard.distance);
      const [bx, by, bz] = billboard.position;
      const billboardYMin = by;
      const billboardYMax = by + 2 * halfDepth;

      return props.some((prop) => {
        // Billboards are intentionally anchored to bleacher stand cells, so the
        // coarse bleacher box is not the right collision proxy; the
        // "feet on seat plane" test already verifies they sit on the real seat
        // surface at that XZ location (OBS-B).
        if (prop.kind === 'bleacher' && prop.bleacherGeometry) {
          return false;
        }

        const [pHx, pHy, pHz] = resolvePropHalfSize(prop.kind);
        const [px, py, pz] = prop.position;
        const propYMin = py;
        const propYMax = py + 2 * pHy;

        return (
          aabbsOverlapXY(bx, bz, halfWidth, halfDepth, px, pz, pHx, pHz) &&
          billboardYMin < propYMax &&
          billboardYMax > propYMin
        );
      });
    });

    expect(intersects).toBe(false);
  });

  it('seats public billboards on real bleacher stand-cell tops (OBS-B)', () => {
    const atlas = loadPublicSpriteAtlas();
    const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
    const props = buildRightWorldProps(FIXTURE);
    const bleachers = props.filter((prop) => prop.kind === 'bleacher');

    expect(bleachers.length).toBeGreaterThan(0);

    const seatedCount = billboards.filter((billboard) => {
      const frame = atlas.frames.find((candidate) => candidate.id === billboard.frameId);
      if (!frame) {
        return false;
      }
      const [halfWidth, halfDepth] = billboardHalfSize(frame, billboard.distance);
      const [bx, by, bz] = billboard.position;

      return bleachers.some((bleacher) => {
        const [pHx, pHz] = getBleacherAxisAlignedHalfSize(bleacher);
        const [px, , pz] = bleacher.position;
        const bleacherTop = getBleacherSeatTopY(bleacher, bx, bz);

        return (
          aabbsOverlapXY(bx, bz, halfWidth, halfDepth, px, pz, pHx, pHz) &&
          Math.abs(by - bleacherTop) <= 0.1
        );
      });
    }).length;

    expect(seatedCount).toBeGreaterThan(0);
  });

  it('keeps every public billboard on the right side of the track', () => {
    const atlas = loadPublicSpriteAtlas();
    const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
    const spec = generateTrack(FIXTURE);

    const allRight = billboards.every((billboard) => {
      const sample = nearestSample(
        spec.splineSamples,
        billboard.position[0],
        billboard.position[2],
      );
      const frame = resolveSplineSampleFrame(
        spec.splineSamples,
        sample.globalIndex,
      );
      const generatorY = sceneZToGeneratorY(billboard.position[2]);
      const deltaX = billboard.position[0] - sample.x;
      const deltaY = generatorY - sample.y;
      const lateral = deltaX * frame.normalX + deltaY * frame.normalY;

      return lateral < 0;
    });

    expect(allRight).toBe(true);
  });
});

describe('public.sprite 8-direction frames and look-at-car gaze (P1-sprite-directional-walk-look, AC-055)', () => {
  it('exports a helper to map a world yaw to one of the 8 sprite directions', () => {
    expect(publicSprite).toHaveProperty('resolveDirectionId');
    expect(
      typeof (publicSprite as unknown as Record<string, unknown>).resolveDirectionId,
    ).toBe('function');
  });

  it('resolves world yaw angles to the nearest 8-direction label', () => {
    const helper = (publicSprite as unknown as Record<string, unknown>)
      .resolveDirectionId;
    expect(typeof helper).toBe('function');

    if (typeof helper === 'function') {
      const resolve = helper as (yawDegrees: number) => string;
      expect([resolve(0), resolve(90), resolve(-90)]).toEqual([
        'front',
        'right',
        'left',
      ]);
    }
  });

      it('recomputes seated gaze frame index when the live car moves (OBS-3)', () => {
      const atlas = loadPublicSpriteAtlas();
      const table = buildPublicRuntimeIndexTable(atlas);
      const billboard = {
        position: [0, 0, 0] as const,
        runtimeRole: 'seated' as const,
        seed: 1,
        index: 0,
        walkDirectionIndex: 0,
        frameId: 'front-stand',
        scale: 1,
        distance: 10,
        quad: [0, 0, 1, 1] as const,
      };

      const indexA = computePublicRuntimeFrameIndex(
        billboard,
        [0, 2, -10],
        [0, 0, 10],
        0,
        table,
      );
      const indexB = computePublicRuntimeFrameIndex(
        billboard,
        [0, 2, -10],
        [10, 0, 0],
        0,
        table,
      );

      expect(indexA).toBeGreaterThanOrEqual(0);
      expect(indexB).toBeGreaterThanOrEqual(0);
      expect(indexA).not.toBe(indexB);
      });

      it('walk frame changes with time and stays deterministic for a fixed clock (OBS-2)', () => {
      const atlas = loadPublicSpriteAtlas();
      const frame0 = selectPublicWalkFrame(atlas, FIXTURE.seed, 0, 0);
      const frameLater = selectPublicWalkFrame(atlas, FIXTURE.seed, 0, 2.5);
      const frame0Again = selectPublicWalkFrame(atlas, FIXTURE.seed, 0, 0);

      expect(frame0).toMatch(/walk/);
      expect(frameLater).toMatch(/walk/);
      expect(frame0).not.toBe(frameLater);
      expect(frame0Again).toBe(frame0);
      });

      it('recomputes the seated spectator frame when the camera heading changes', () => {
    const atlas = loadPublicSpriteAtlas();
    const table = buildPublicRuntimeIndexTable(atlas);
    const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
    const props = buildRightWorldProps(FIXTURE);
    const bleachers = props.filter((prop) => prop.kind === 'bleacher');

    expect(bleachers.length).toBeGreaterThan(0);

    const seatedBillboards = billboards.filter(
      (billboard) => billboard.runtimeRole === 'seated',
    );

    expect(seatedBillboards.length).toBeGreaterThan(0);

    // Same camera position and same car bearing, two different camera
    // headings: a camera-relative frame index MUST react to the heading.
    const spectator = seatedBillboards[0]!;
    const frameFacingNorth = computeCameraAwareRuntimeFrameIndex(
      spectator,
      FIXTURE_CAMERA.position,
      FIXTURE_CAR_POSITION,
      0,
      table,
      [0, 0, 1],
    );
    const frameFacingEast = computeCameraAwareRuntimeFrameIndex(
      spectator,
      FIXTURE_CAMERA.position,
      FIXTURE_CAR_POSITION,
      0,
      table,
      [1, 0, 0],
    );

    expect(frameFacingNorth).not.toBe(frameFacingEast);
  });

  it('renders a seated spectator facing the camera when the camera heading is north (AC-068, RED-A)', () => {
    const atlas = loadPublicSpriteAtlas();
    const table = buildPublicRuntimeIndexTable(atlas);
    // Hand-built fixture: the spectator sits 5 m north of the car, so the
    // robot-to-car bearing is exactly -Z (yaw = PI) and the expected
    // camera-relative direction is fully determined instead of incidental.
    const spectator = seatedSpectatorBillboard([0, 0, 5]);
    const cameraForwardNorth: readonly [number, number, number] = [0, 0, 1];
    const cameraForwardEast: readonly [number, number, number] = [1, 0, 0];

    const frameFacingNorth = computeCameraAwareRuntimeFrameIndex(
      spectator,
      FIXTURE_CAMERA.position,
      FIXTURE_CAR_POSITION,
      0,
      table,
      cameraForwardNorth,
    );
    const frameFacingEast = computeCameraAwareRuntimeFrameIndex(
      spectator,
      FIXTURE_CAMERA.position,
      FIXTURE_CAR_POSITION,
      0,
      table,
      cameraForwardEast,
    );

    // AC-068: the frame index is CAMERA-RELATIVE. Direction index 0
    // ('front') renders the sprite FACING the camera, so the conversion is
    // the neatRacing-compass form of the canonical Neatenstein rule
    // `relativeYaw = facingYaw - cameraYaw + PI`
    // (examples/neatenstein/browser-entry/renderer/sprites.ts:154-156 states
    // the mirrored `cameraYaw + PI - facingYaw` because Neatenstein measures
    // yaw in the opposite compass).
    // Here the spectator looks back at the camera along the camera axis, so
    // a north-looking camera MUST see direction index 0.
    expect(
      directionOfRuntimeFrameIndex(atlas, frameFacingNorth),
    ).toBe(
      expectedCameraRelativeDirection(
        cameraForwardNorth,
        spectator.position,
        FIXTURE_CAR_POSITION,
      ),
    );
    expect(directionOfRuntimeFrameIndex(atlas, frameFacingNorth)).toBe(
      'front',
    );
    // The atlas is color-major (6 accents x 8 directions x 3 poses), so the
    // direction x pose slot survives the per-robot color offset in the low
    // bits: slot 0 is direction 0 with the stand pose.
    expect(directionPoseSlotOfRuntimeFrameIndex(frameFacingNorth)).toBe(0);
    // Two camera headings over the SAME car bearing: the rendered direction
    // must differ, by exactly 2 atlas steps (90 deg) for a 90 deg camera
    // rotation, and the frame indices must differ too.
    expect(
      directionStepDistance(
        directionOfRuntimeFrameIndex(atlas, frameFacingNorth),
        directionOfRuntimeFrameIndex(atlas, frameFacingEast),
      ),
    ).toBe(2);
    expect(frameFacingNorth).not.toBe(frameFacingEast);
  });

  it('exports a helper to animate a slow two-step walk cycle', () => {
    expect(publicSprite).toHaveProperty('selectPublicWalkFrame');
    const helper = (publicSprite as unknown as Record<string, unknown>)
      .selectPublicWalkFrame;
    expect(typeof helper).toBe('function');

    if (typeof helper === 'function') {
      const atlas = loadPublicSpriteAtlas();
      const result = (helper as (
        atlas: unknown,
        seed: number,
        index: number,
        timeSeconds: number,
      ) => string | { id: string })(atlas, FIXTURE.seed, 0, 0);
      const frameId = typeof result === 'string' ? result : result.id;

      expect(frameId).toMatch(/walk/);
    }
  });
});

describe(
  'public.sprite camera-relative spectator facing (P1-crowd-facing-relative, AC-068)',
  () => {
    it('resolves the build-time seated frame to the front frame for a spectator directly in front of a north-looking camera', () => {
      const atlas = loadPublicSpriteAtlas();
      const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
      const spectator = billboards.find(
        (billboard) => billboard.runtimeRole === 'seated',
      );

      if (!spectator) {
        throw new Error('no seated spectator billboard was built');
      }

      const [spectatorX, , spectatorZ] = spectator.position;
      // The camera is placed due south of that spectator and looks north
      // (+Z), so the spectator looks straight back at the camera and must
      // resolve to the 'front' frame (direction index 0).
      const cameraBehindSpectator = {
        position: [spectatorX, 2, spectatorZ - 10] as const,
        forward: [0, 0, 1] as const,
        right: [1, 0, 0] as const,
        near: 0.1,
        far: 1000,
        fovDegrees: 60,
      } as const;
      const rebuilt = buildPublicBillboards(
        FIXTURE,
        cameraBehindSpectator,
        atlas,
      );
      const rebuiltSpectator = rebuilt.find(
        (billboard) =>
          billboard.runtimeRole === 'seated' &&
          billboard.position[0] === spectatorX &&
          billboard.position[2] === spectatorZ,
      );

      if (!rebuiltSpectator) {
        throw new Error('the rebuilt scene lost the seated spectator');
      }

      expect(rebuiltSpectator.frameId).toBe('front-stand');
    });

    it('rotates the build-time seated frame by 90 degrees when the camera heading rotates', () => {
      const atlas = loadPublicSpriteAtlas();
      const northBillboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
      const eastBillboards = buildPublicBillboards(
        FIXTURE,
        { ...FIXTURE_CAMERA, forward: [1, 0, 0] as const },
        atlas,
      );
      const northSpectator = northBillboards.find(
        (billboard) => billboard.runtimeRole === 'seated',
      );
      const eastSpectator = eastBillboards.find(
        (billboard) => billboard.runtimeRole === 'seated',
      );

      if (!northSpectator || !eastSpectator) {
        throw new Error('no seated spectator billboard was built');
      }

      // Same camera position and same car bearing, two camera headings: the
      // build-time frame must rotate with the heading, by exactly 2 atlas
      // steps (90 degrees) for a 90 degrees camera rotation.
      expect(
        directionStepDistance(
          directionFromPublicFrameId(northSpectator.frameId),
          directionFromPublicFrameId(eastSpectator.frameId),
        ),
      ).toBe(2);
    });

    it('agrees with the build-time frame id when the car sits at the camera position', () => {
      const atlas = loadPublicSpriteAtlas();
      const table = buildPublicRuntimeIndexTable(atlas);
      const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
      const spectator = billboards.find(
        (billboard) => billboard.runtimeRole === 'seated',
      );

      if (!spectator) {
        throw new Error('no seated spectator billboard was built');
      }

      // With the car at the camera position, the runtime camera-relative
      // conversion must reproduce the static build-time frame, so the
      // sprite never flips between the built frame and the drawn frame.
      const runtimeIndex = computeCameraAwareRuntimeFrameIndex(
        spectator,
        FIXTURE_CAMERA.position,
        FIXTURE_CAMERA.position,
        0,
        table,
        FIXTURE_CAMERA.forward,
      );

      expect(directionOfRuntimeFrameIndex(atlas, runtimeIndex)).toBe(
        directionFromPublicFrameId(spectator.frameId),
      );
    });

    it('falls back to the camera-to-car bearing when no camera heading is supplied', () => {
      const atlas = loadPublicSpriteAtlas();
      const table = buildPublicRuntimeIndexTable(atlas);
      // Legacy five-argument call: the camera heading falls back to the
      // camera->car bearing (+Z here), so the spectator 5 m north of the car
      // looks back at the camera and resolves to the 'front' frame.
      const frameIndex = computePublicRuntimeFrameIndex(
        seatedSpectatorBillboard([0, 0, 5]),
        FIXTURE_CAMERA.position,
        FIXTURE_CAR_POSITION,
        0,
        table,
      );

      expect(directionOfRuntimeFrameIndex(atlas, frameIndex)).toBe('front');
    });

    it('renders a spectator facing the camera screen-left as the left frame (AC-068, lateral handedness)', () => {
      const atlas = loadPublicSpriteAtlas();
      const table = buildPublicRuntimeIndexTable(atlas);
      // The spectator sits 5 m north of the car, so it looks back due south
      // (-Z). The neatRacing camera right axis is (-forwardZ, 0, forwardX),
      // so a camera looking toward +X has its screen-right on +Z and sees
      // the south-facing spectator on its screen-LEFT, while a camera
      // looking toward -X has its screen-right on -Z and sees the same
      // spectator on its screen-RIGHT.
      const spectator = seatedSpectatorBillboard([0, 0, 5]);
      const frameLookingEast = computeCameraAwareRuntimeFrameIndex(
        spectator,
        FIXTURE_CAMERA.position,
        FIXTURE_CAR_POSITION,
        0,
        table,
        [1, 0, 0],
      );
      const frameLookingWest = computeCameraAwareRuntimeFrameIndex(
        spectator,
        FIXTURE_CAMERA.position,
        FIXTURE_CAR_POSITION,
        0,
        table,
        [-1, 0, 0],
      );

      expect(directionOfRuntimeFrameIndex(atlas, frameLookingEast)).toBe(
        'left',
      );
      expect(directionOfRuntimeFrameIndex(atlas, frameLookingWest)).toBe(
        'right',
      );
    });

    it('resolves the build-time seated frame to the lateral stand frames (AC-068, lateral handedness)', () => {
      const atlas = loadPublicSpriteAtlas();
      const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
      const spectator = billboards.find(
        (billboard) => billboard.runtimeRole === 'seated',
      );

      if (!spectator) {
        throw new Error('no seated spectator billboard was built');
      }

      const [spectatorX, , spectatorZ] = spectator.position;
      // The camera is placed due south of that spectator, so the spectator
      // looks straight back at the camera (due south, -Z). The camera right
      // axis is (-forwardZ, 0, forwardX): a camera looking toward +X sees
      // the spectator on its screen-LEFT and a camera looking toward -X sees
      // the same spectator on its screen-RIGHT.
      const cameraLookingEast = {
        position: [spectatorX, 2, spectatorZ - 10] as const,
        forward: [1, 0, 0] as const,
        right: [0, 0, 1] as const,
        near: 0.1,
        far: 1000,
        fovDegrees: 60,
      } as const;
      const cameraLookingWest = {
        position: [spectatorX, 2, spectatorZ - 10] as const,
        forward: [-1, 0, 0] as const,
        right: [0, 0, -1] as const,
        near: 0.1,
        far: 1000,
        fovDegrees: 60,
      } as const;
      const eastSpectator = buildPublicBillboards(
        FIXTURE,
        cameraLookingEast,
        atlas,
      ).find(
        (billboard) =>
          billboard.runtimeRole === 'seated' &&
          billboard.position[0] === spectatorX &&
          billboard.position[2] === spectatorZ,
      );
      const westSpectator = buildPublicBillboards(
        FIXTURE,
        cameraLookingWest,
        atlas,
      ).find(
        (billboard) =>
          billboard.runtimeRole === 'seated' &&
          billboard.position[0] === spectatorX &&
          billboard.position[2] === spectatorZ,
      );

      if (!eastSpectator || !westSpectator) {
        throw new Error('the rebuilt scene lost the seated spectator');
      }

      expect(eastSpectator.frameId).toBe('left-stand');
      expect(westSpectator.frameId).toBe('right-stand');
    });
  },
);

describe('public.sprite neon palette color diversity (P1-sprite-directional-walk-look, AC-062)', () => {
  it('includes at least three distinct neon accent hues across the public atlas', () => {
    const atlas = loadPublicSpriteAtlas();
    const hues = new Set<number>();

    for (const frame of atlas.frames) {
      for (const hue of distinctNeonHues(frame.rgba)) {
        hues.add(hue);
      }
    }

    expect(hues.size).toBeGreaterThanOrEqual(3);
  });
});

describe(
  'public.sprite per-robot color identity (fix-packet-P1-bleacher-shape-iteration-3, OBS-2)',
  () => {
    it('keeps the same accent color for a seated robot when the car moves to a different direction', () => {
      const atlas = loadPublicSpriteAtlas();
      const table = buildPublicRuntimeIndexTable(atlas);
      const billboard = {
        position: [0, 0, 0] as const,
        runtimeRole: 'seated' as const,
        seed: 1,
        index: 0,
        walkDirectionIndex: 0,
        frameId: 'front-stand',
        scale: 1,
        distance: 10,
        quad: [0, 0, 1, 1] as const,
      };

      const indexA = computePublicRuntimeFrameIndex(
        billboard,
        [0, 2, -10],
        [0, 0, 10],
        0,
        table,
      );
      const indexB = computePublicRuntimeFrameIndex(
        billboard,
        [0, 2, -10],
        [10, 0, 0],
        0,
        table,
      );

      const frameA = atlas.frames[indexA];
      const frameB = atlas.frames[indexB];
      if (!frameA || !frameB) {
        throw new Error('runtime frame index out of bounds');
      }

      const hueA = dominantAccentHue(frameA.rgba);
      const hueB = dominantAccentHue(frameB.rgba);

      expect(hueA).not.toBeNull();
      expect(hueB).toBe(hueA);
    });

    it('keeps the same accent color for a walker across walk cycle time', () => {
      const atlas = loadPublicSpriteAtlas();
      const table = buildPublicRuntimeIndexTable(atlas);
      const billboard = {
        position: [0, 0, 0] as const,
        runtimeRole: 'walker' as const,
        seed: 1,
        index: 0,
        walkDirectionIndex: 0,
        frameId: 'front-walk1',
        scale: 1,
        distance: 10,
        quad: [0, 0, 1, 1] as const,
      };

      const indexA = computePublicRuntimeFrameIndex(
        billboard,
        [0, 2, -10],
        [0, 0, 10],
        0,
        table,
      );
      const indexB = computePublicRuntimeFrameIndex(
        billboard,
        [0, 2, -10],
        [0, 0, 10],
        2.5,
        table,
      );

      const frameA = atlas.frames[indexA];
      const frameB = atlas.frames[indexB];
      if (!frameA || !frameB) {
        throw new Error('runtime frame index out of bounds');
      }

      const hueA = dominantAccentHue(frameA.rgba);
      const hueB = dominantAccentHue(frameB.rgba);

      expect(hueA).not.toBeNull();
      expect(hueB).toBe(hueA);
    });

    it('keeps the same accent color across all 8 atlas directions for one robot', () => {
      const atlas = loadPublicSpriteAtlas();
      const table = buildPublicRuntimeIndexTable(atlas);
      const billboard = {
        position: [0, 0, 0] as const,
        runtimeRole: 'seated' as const,
        seed: 1,
        index: 0,
        walkDirectionIndex: 0,
        frameId: 'front-stand',
        scale: 1,
        distance: 10,
        quad: [0, 0, 1, 1] as const,
      };

      const carPositions: readonly (readonly [number, number, number])[] = [
        [0, 0, 10],
        [10, 0, 10],
        [10, 0, 0],
        [10, 0, -10],
        [0, 0, -10],
        [-10, 0, -10],
        [-10, 0, 0],
        [-10, 0, 10],
      ];

      const hues: (number | null)[] = [];
      for (const carPosition of carPositions) {
        const frameIndex = computePublicRuntimeFrameIndex(
          billboard,
          [0, 2, -10] as const,
          carPosition,
          0,
          table,
        );
        const frame = atlas.frames[frameIndex];
        if (!frame) {
          throw new Error('runtime frame index out of bounds');
        }
        hues.push(dominantAccentHue(frame.rgba));
      }

      const nonNullHues = hues.filter((hue): hue is number => hue !== null);
      expect(nonNullHues.length).toBe(8);
      expect(new Set(nonNullHues).size).toBe(1);
    });

    it('gives different robots different accent colors', () => {
      const atlas = loadPublicSpriteAtlas();
      const table = buildPublicRuntimeIndexTable(atlas);
      const hues: (number | null)[] = [];

      for (let index = 0; index < 8; index += 1) {
        const billboard = {
          position: [0, 0, 0] as const,
          runtimeRole: 'seated' as const,
          seed: 1,
          index,
          walkDirectionIndex: 0,
          frameId: 'front-stand',
          scale: 1,
          distance: 10,
          quad: [0, 0, 1, 1] as const,
        };

        const frameIndex = computePublicRuntimeFrameIndex(
          billboard,
          [0, 2, -10] as const,
          [0, 0, 10],
          0,
          table,
        );
        const frame = atlas.frames[frameIndex];
        if (!frame) {
          throw new Error('runtime frame index out of bounds');
        }
        hues.push(dominantAccentHue(frame.rgba));
      }

      const nonNullHues = hues.filter((hue): hue is number => hue !== null);
      expect(nonNullHues.length).toBeGreaterThan(0);
      expect(new Set(nonNullHues).size).toBeGreaterThanOrEqual(2);
    });
  },
);

describe(
  'public.sprite public runtime role set (P1-crowd2-red-contracts, AC-074 role set)',
  () => {
    it('resolves the seated, walker, standing and mechanic roles through the single role resolver', () => {
      const resolvedRoles = PUBLIC_RUNTIME_ROLE_SET.map((role) =>
        resolvePublicRuntimeRole(publicRuntimeRoleCarrier(role)),
      );

      expect(resolvedRoles).toEqual([...PUBLIC_RUNTIME_ROLE_SET]);
    });

    it('renders the seated, standing and mechanic roles on the stand pose', () => {
      const atlas = loadPublicSpriteAtlas();
      const table = buildPublicRuntimeIndexTable(atlas);
      const cameraForward: readonly [number, number, number] = [0, 0, 1];

      const seatedPose = poseIndexOfRuntimeFrameIndex(
        computePublicRuntimeFrameIndex(
          publicRuntimeRoleBillboard('seated'),
          FIXTURE_CAMERA.position,
          FIXTURE_CAR_POSITION,
          0,
          table,
          cameraForward,
        ),
      );
      const standingPose = poseIndexOfRuntimeFrameIndex(
        computePublicRuntimeFrameIndex(
          publicRuntimeRoleBillboard('standing'),
          FIXTURE_CAMERA.position,
          FIXTURE_CAR_POSITION,
          0,
          table,
          cameraForward,
        ),
      );
      const mechanicPose = poseIndexOfRuntimeFrameIndex(
        computePublicRuntimeFrameIndex(
          publicRuntimeRoleBillboard('mechanic'),
          FIXTURE_CAMERA.position,
          FIXTURE_CAR_POSITION,
          0,
          table,
          cameraForward,
        ),
      );

      // AC-074 (role set): the standing spectators and the inner-ring
      // mechanics are STANDING agents, so the frame-index helper must render
      // them on the public stand frames instead of the walk cycle. The
      // helper resolves the role through resolvePublicRuntimeRole, so the
      // resolver and the helper must agree on EVERY role of the set.
      expect(seatedPose).toBe(PUBLIC_POSE_INDEX_STAND);
      expect(standingPose).toBe(PUBLIC_POSE_INDEX_STAND);
      expect(mechanicPose).toBe(PUBLIC_POSE_INDEX_STAND);
    });

    it('renders the walker role on the slow walk-cycle pose', () => {
      const atlas = loadPublicSpriteAtlas();
      const table = buildPublicRuntimeIndexTable(atlas);
      const cameraForward: readonly [number, number, number] = [0, 0, 1];

      const walkerPose = poseIndexOfRuntimeFrameIndex(
        computePublicRuntimeFrameIndex(
          publicRuntimeRoleBillboard('walker'),
          FIXTURE_CAMERA.position,
          FIXTURE_CAR_POSITION,
          0,
          table,
          cameraForward,
        ),
      );

      // AC-074 (role set): the walkers keep their slow walk cycle, so the
      // walker role is the ONLY role of the set that renders a walk frame.
      expect(walkerPose).toBe(PUBLIC_POSE_INDEX_WALK1);
    });
  },
);

describe(
  'public.sprite crowd sprite world scale (P1-sprite-scale-red-contracts, AC-077, RED-A)',
  () => {
    it('exports the named NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE lever as 0.5 (RED-A)', () => {
      // AC-077: the halving has EXACTLY ONE source of truth - a NAMED
      // exported constant. The assertion reads the module namespace, so a
      // missing export fails THIS contract instead of breaking the import of
      // the whole test file.
      expect(publicSprite).toHaveProperty(
        'NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE',
        0.5,
      );
    });

    it('returns exactly the named world scale for every camera distance (RED-A)', () => {
      // The scale stays DISTANCE-INDEPENDENT: the perspective shrink is the
      // shared focal/distance projection, so the returned value IS the named
      // constant. The equality is EXACT (toBe, not toBeCloseTo): a
      // near-match on the pre-14L 1.0 must not pass.
      expect(computeBillboardScale(5)).toBe(0.5);
      expect(computeBillboardScale(50)).toBe(0.5);
      expect(computeBillboardScale(0.1)).toBe(0.5);
    });
  },
);

describe(
  'public.sprite crowd billboard placement footprint (P1-sprite-scale-red-contracts, AC-077, RED-B)',
  () => {
    it('exports the shared meters-per-pixel constant NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M (RED-B)', () => {
      // AC-077: the meters-per-pixel constant is the SHARED meter scale of
      // the crowd sprites. It is exported so the right-world pass can import
      // it instead of keeping its own copy, and so a footprint contract can
      // READ it instead of hard-coding 0.05.
      expect(publicSprite).toHaveProperty(
        'NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M',
        0.05,
      );
    });

    it('reports a 4.8 m world footprint (a 2.4 m half-extent) for a 192x192 crowd frame (RED-B)', () => {
      // The crowd frames are 192 x 192 px, so the PRE-14L footprint was
      // 192 * 0.05 = 9.6 m wide and tall. The halved footprint is
      // 192 * 0.5 * 0.05 = 4.8 m, which is a 2.4 m half-extent on each axis.
      // The square frame keeps its aspect ratio, so the world WIDTH follows
      // the SAME factor as the world HEIGHT.
      const atlas = loadPublicSpriteAtlas();
      const frame = atlas.frames[0]!;
      const [halfWidth, halfHeight] = billboardHalfSize(frame, 20);

      expect(frame.width).toBe(192);
      expect(halfWidth).toBeCloseTo(2.4, 6);
      expect(halfHeight).toBeCloseTo(2.4, 6);
    });

    it('derives the crowd footprint from the named lever and the shared meters-per-pixel constant (RED-B)', () => {
      // The footprint is read from the NAMED world-scale lever and the
      // SHARED meters-per-pixel constant, so the placement footprint and the
      // rendered quad can never disagree and a future meter-scale change
      // moves the reported footprint with it.
      const atlas = loadPublicSpriteAtlas();
      const frame = atlas.frames[0]!;
      const worldScale = readNumericModuleExport(
        'NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE',
      );
      const metersPerPixel = readNumericModuleExport(
        'NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M',
      );
      const [halfWidth, halfHeight] = crowdBillboardWorldHalfSize(frame, 20);

      expect(halfWidth).toBeCloseTo(
        (frame.width * worldScale * metersPerPixel) / 2,
        6,
      );
      expect(halfHeight).toBeCloseTo(
        (frame.height * worldScale * metersPerPixel) / 2,
        6,
      );
    });
  },
);
