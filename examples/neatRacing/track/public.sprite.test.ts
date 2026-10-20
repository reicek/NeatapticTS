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
 * - {@link computeBillboardScale} returns a constant 1.0 world scale so
 *   perspective projection handles distance shrink (AC-054).
 * - Billboards are 2D sprite quads, not voxel snapshots.
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
  PUBLIC_POSE_COUNT,
  PUBLIC_POSE_INDEX_STAND,
  PUBLIC_POSE_INDEX_WALK1,
  PUBLIC_POSE_INDEX_WALK2,
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

/** World-space meters per sprite pixel used by the right-world render pass. */
const BILLBOARD_WORLD_PIXEL_SCALE_M = 0.05;

/** Computes the world-space half-width and half-height of a billboard. */
function billboardHalfSize(
  frame: ReturnType<typeof loadPublicSpriteAtlas>['frames'][number],
  distance: number,
): readonly [number, number] {
  const scale = computeBillboardScale(distance) * BILLBOARD_WORLD_PIXEL_SCALE_M;

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
 * Returns the 8-direction frame a robot should use so that it faces a target
 * point in the XZ plane.
 */
function expectedLookDirection(
  target: readonly [number, number, number],
  robotPosition: readonly [number, number, number],
): string {
  const dx = target[0] - robotPosition[0];
  const dz = target[2] - robotPosition[2];
  const angle = Math.atan2(dx, dz);
  const step = Math.PI / 4;
  const index = (Math.round(angle / step) % 8 + 8) % 8;

  return ROBOT_DIRECTION_ORDER[index]!;
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

    it('keeps billboard world scale at 1:1 so perspective projection handles distance shrink', () => {
      const near = computeBillboardScale(10);
      const far = computeBillboardScale(200);

      expect(near).toBeCloseTo(1, 6);
      expect(far).toBeCloseTo(1, 6);
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

  it('matches the absolute projected size of a same-world-size voxel', () => {
    const atlas = loadPublicSpriteAtlas();
    const frame = atlas.frames[0]!;
    const worldPixelScale = 0.05;
    const fovDegrees = 60;
    const focal = 1 / Math.tan((fovDegrees * Math.PI) / 360);
    const distance = 25;

    const voxelWorldWidth = frame.width * worldPixelScale;
    const voxelScreenWidth = voxelWorldWidth * focal / distance;

    const billboardWorldWidth =
      frame.width * worldPixelScale * computeBillboardScale(distance);
    const billboardScreenWidth = billboardWorldWidth * focal / distance;

    expect(billboardScreenWidth).toBeCloseTo(voxelScreenWidth, 6);
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

      it('makes bleacher spectators face the camera/car', () => {
    const atlas = loadPublicSpriteAtlas();
    const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
    const props = buildRightWorldProps(FIXTURE);
    const bleachers = props.filter((prop) => prop.kind === 'bleacher');

    expect(bleachers.length).toBeGreaterThan(0);

    const seatedBillboards = billboards.filter((billboard) => {
      const frame = atlas.frames.find(
        (candidate) => candidate.id === billboard.frameId,
      );
      if (!frame) {
        return false;
      }
      const [halfWidth, halfDepth] = billboardHalfSize(frame, billboard.distance);
      const [bx, , bz] = billboard.position;

      return bleachers.some((bleacher) => {
        const [pHx, , pHz] = resolvePropHalfSize(bleacher.kind);
        const [px, , pz] = bleacher.position;

        return aabbsOverlapXY(bx, bz, halfWidth, halfDepth, px, pz, pHx, pHz);
      });
    });

    expect(seatedBillboards.length).toBeGreaterThan(0);

    const allFacingCamera = seatedBillboards.every((billboard) => {
      const actual = directionFromPublicFrameId(billboard.frameId);
      const expected = expectedLookDirection(
        FIXTURE_CAMERA.position,
        billboard.position,
      );

      return actual === expected;
    });

    expect(allFacingCamera).toBe(true);
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
