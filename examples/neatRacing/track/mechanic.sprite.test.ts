/**
 * Red contract tests for the recycled Neatenstein mechanic-crew billboard
 * sprites (slice P1-mechanic-crew, Step 14D, AC-050).
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - {@link loadMechanicSpriteAtlas} returns a non-empty atlas whose source is
 *   'neatenstein-mechanic-crew'.
 * - {@link selectMechanicSpriteFrame} returns a deterministic frame with positive
 *   dimensions for a given (seed, index).
 * - {@link buildMechanicBillboards} returns billboards on the right side near
 *   the start line with positive screen-space quads.
 * - {@link computeMechanicBillboardScale} returns a constant 1.0 world scale
 *   so sprites share the same perspective anchor as voxels (AC-054).
 * - Billboards are 2D sprite quads, not voxel snapshots.
 */

import { describe, expect, it } from '@jest/globals';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';
import { resolveSplineSampleFrame } from '../../racing_curriculum/track/track.spline.utils';
import { buildRightWorldProps } from './right-world.builder';
import { resolvePropHalfSize } from './right-world.prop-sizes';
import type { TrackSlabBuildInput } from './track.slabs';
import {
  buildMechanicBillboards,
  computeMechanicBillboardScale,
  computeMechanicRuntimeFrameIndex,
  loadMechanicSpriteAtlas,
  selectMechanicSpriteFrame,
} from './mechanic.sprite';
import * as mechanicSprite from './mechanic.sprite';
import {
  DIRECTION_ORDER,
  PUBLIC_POSE_COUNT,
  PUBLIC_POSE_INDEX_STAND,
} from './public.sprite';

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
  frame: ReturnType<typeof loadMechanicSpriteAtlas>['frames'][number],
  distance: number,
): readonly [number, number] {
  const scale = computeMechanicBillboardScale(distance) * BILLBOARD_WORLD_PIXEL_SCALE_M;

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

/** Parses the direction token out of a mechanic sprite frame id. */
function directionFromMechanicFrameId(frameId: string): string {
  return frameId.split('-')[1]!;
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

/** Builds the mechanic runtime direction×pose atlas-index lookup table. */
function buildMechanicRuntimeIndexTable(
  atlas: ReturnType<typeof loadMechanicSpriteAtlas>,
): Int16Array {
  const table = new Int16Array(
    DIRECTION_ORDER.length * PUBLIC_POSE_COUNT,
  ).fill(-1);

  for (let frameIndex = 0; frameIndex < atlas.frames.length; frameIndex += 1) {
    const frame = atlas.frames[frameIndex]!;
    const direction = directionFromMechanicFrameId(frame.id);
    const directionIndex = (DIRECTION_ORDER as readonly string[]).indexOf(direction);
    if (directionIndex >= 0) {
      table[directionIndex * PUBLIC_POSE_COUNT + PUBLIC_POSE_INDEX_STAND] =
        frameIndex;
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
 * or `null` when no accent pixels are found.
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

describe('mechanic.sprite (P1-mechanic-crew, AC-050)', () => {
  describe('sprite atlas load and selection', () => {
    it('loads a non-empty atlas recycled from Neatenstein mechanic-crew sprites', () => {
      const atlas = loadMechanicSpriteAtlas();

      expect(atlas.source).toBe('neatenstein-mechanic-crew');
      expect(atlas.frames.length).toBeGreaterThan(0);
    });

    it('selects a deterministic sprite frame with positive dimensions', () => {
      const atlas = loadMechanicSpriteAtlas();
      const frameA = selectMechanicSpriteFrame(atlas, 42, 0);
      const frameB = selectMechanicSpriteFrame(atlas, 42, 0);

      expect(frameA.id).toBe(frameB.id);
      expect(frameA.width).toBeGreaterThan(0);
      expect(frameA.height).toBeGreaterThan(0);
    });
  });

  describe('billboard placement and scale', () => {
    it('places mechanic billboards on the right side with positive screen-space quads', () => {
      const atlas = loadMechanicSpriteAtlas();
      const billboards = buildMechanicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
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
      const atlas = loadMechanicSpriteAtlas();
      const billboards = buildMechanicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);

      const allScaled = billboards.every(
        (billboard) =>
          billboard.scale > 0 &&
          Number.isFinite(billboard.distance) &&
          billboard.distance > 0,
      );

      expect(allScaled).toBe(true);
    });

    it('keeps mechanic billboard world scale constant at all distances', () => {
      const near = computeMechanicBillboardScale(10);
      const far = computeMechanicBillboardScale(200);

      expect(near).toBeCloseTo(1, 6);
      expect(far).toBeCloseTo(1, 6);
    });
  });

  describe('2D preferred over voxel fallback', () => {
    it('represents every billboard as a 2D sprite quad, not a voxel snapshot', () => {
      const atlas = loadMechanicSpriteAtlas();
      const billboards = buildMechanicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);

      const all2D = billboards.every(
        (billboard) => !('voxels' in billboard) && !('voxelSnapshot' in billboard),
      );

      expect(all2D).toBe(true);
    });
  });
});

describe('mechanic.sprite elevation anchor (P1-sprite-elevation, AC-052)', () => {
  it('places mechanic-crew billboards with feet on the ground or seat plane', () => {
    const atlas = loadMechanicSpriteAtlas();
    const billboards = buildMechanicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
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
        const [pHx, pHy, pHz] = resolvePropHalfSize(prop.kind);
        const [px, py, pz] = prop.position;
        const propTop = py + 2 * pHy;
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

  it('keeps mechanic-crew billboards above right-world voxel scenery', () => {
    const atlas = loadMechanicSpriteAtlas();
    const billboards = buildMechanicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
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

  it('anchors mechanic-crew billboards to inner-ring pedestal surfaces (OBS-2)', () => {
    const atlas = loadMechanicSpriteAtlas();
    const billboards = buildMechanicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
    const props = buildRightWorldProps(FIXTURE);
    const pedestals = props.filter(
      (prop) => prop.kind === 'mechanic' || prop.kind === 'crew',
    );

    expect(pedestals.length).toBeGreaterThan(0);
    expect(billboards.length).toBeLessThanOrEqual(pedestals.length);

    const allOnPedestals = billboards.every((billboard) => {
      const frame = atlas.frames.find((candidate) => candidate.id === billboard.frameId);
      if (!frame) {
        return false;
      }
      const [halfWidth, halfDepth] = billboardHalfSize(frame, billboard.distance);
      const [bx, by, bz] = billboard.position;

      return pedestals.some((pedestal) => {
        const [pHx, pHy, pHz] = resolvePropHalfSize(pedestal.kind);
        const [px, py, pz] = pedestal.position;
        const pedestalTop = py + 2 * pHy;

        return (
          aabbsOverlapXY(bx, bz, halfWidth, halfDepth, px, pz, pHx, pHz) &&
          by + 0.001 >= pedestalTop
        );
      });
    });

    expect(allOnPedestals).toBe(true);
  });

  it('keeps every mechanic-crew billboard on the right side of the track', () => {
    const atlas = loadMechanicSpriteAtlas();
    const billboards = buildMechanicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
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

describe('mechanic.sprite look-at-car gaze (P1-sprite-directional-walk-look, AC-055)', () => {
  it('exports a helper to pick the frame that looks at the car', () => {
    expect(mechanicSprite).toHaveProperty('selectMechanicLookAtCarFrame');
    expect(
      typeof (mechanicSprite as unknown as Record<string, unknown>)
        .selectMechanicLookAtCarFrame,
    ).toBe('function');
  });

  it('selects a direction frame that faces the car', () => {
    const helper = (mechanicSprite as unknown as Record<string, unknown>)
      .selectMechanicLookAtCarFrame;
    expect(typeof helper).toBe('function');

    if (typeof helper === 'function') {
      const atlas = loadMechanicSpriteAtlas();
      const result = (helper as (
        atlas: unknown,
        robotPosition: readonly [number, number, number],
        carPosition: readonly [number, number, number],
      ) => string | { id: string })(atlas, [0, 0, 0], [0, 0, 10]);
      const frameId = typeof result === 'string' ? result : result.id;

      expect(directionFromMechanicFrameId(frameId)).toBe('front');
    }
  });

      it('recomputes the runtime frame index when the live car moves (OBS-3)', () => {
      const atlas = loadMechanicSpriteAtlas();
      const table = buildMechanicRuntimeIndexTable(atlas);
      const billboard = {
        position: [0, 0, 0] as const,
        runtimeRole: 'mechanic' as const,
        seed: 1,
        index: 0,
        frameId: 'mechanic-front-stand',
        scale: 1,
        distance: 10,
        quad: [0, 0, 1, 1] as const,
      };

      const indexA = computeMechanicRuntimeFrameIndex(
        billboard,
        [0, 0, 10],
        table,
      );
      const indexB = computeMechanicRuntimeFrameIndex(
        billboard,
        [10, 0, 0],
        table,
      );

      expect(indexA).toBeGreaterThanOrEqual(0);
      expect(indexB).toBeGreaterThanOrEqual(0);
      expect(indexA).not.toBe(indexB);
      });

      it('makes every mechanic-crew billboard face the camera/car', () => {
    const atlas = loadMechanicSpriteAtlas();
    const billboards = buildMechanicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);

    expect(billboards.length).toBeGreaterThan(0);

    const allFacingCamera = billboards.every((billboard) => {
      const actual = directionFromMechanicFrameId(billboard.frameId);
      const expected = expectedLookDirection(
        FIXTURE_CAMERA.position,
        billboard.position,
      );

      return actual === expected;
    });

    expect(allFacingCamera).toBe(true);
  });
});

describe('mechanic.sprite team-color accents (P1-sprite-directional-walk-look, AC-062)', () => {
  it('keeps at least two distinct accent hues in the mechanic-crew atlas', () => {
    const atlas = loadMechanicSpriteAtlas();
    const hues = new Set<number>();

    for (const frame of atlas.frames) {
      for (const hue of distinctNeonHues(frame.rgba)) {
        hues.add(hue);
      }
    }

    expect(hues.size).toBeGreaterThanOrEqual(2);
  });
});

describe(
  'mechanic.sprite per-robot color identity (fix-packet-P1-bleacher-shape-iteration-3, OBS-2)',
  () => {
    it('keeps the same accent color when the car moves to a different direction', () => {
      const atlas = loadMechanicSpriteAtlas();
      const table = buildMechanicRuntimeIndexTable(atlas);
      const billboard = {
        position: [0, 0, 0] as const,
        runtimeRole: 'mechanic' as const,
        seed: 1,
        index: 0,
        frameId: 'mechanic-front-stand',
        scale: 1,
        distance: 10,
        quad: [0, 0, 1, 1] as const,
      };

      const indexA = computeMechanicRuntimeFrameIndex(
        billboard,
        [0, 0, 10],
        table,
      );
      // Move the car to +X +Z so the robot faces frontRight (direction index 1),
      // which currently carries a different team-color parity than front (index 0).
      const indexB = computeMechanicRuntimeFrameIndex(
        billboard,
        [10, 0, 10],
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

    it('keeps the same accent color across all 8 atlas directions for one mechanic', () => {
      const atlas = loadMechanicSpriteAtlas();
      const table = buildMechanicRuntimeIndexTable(atlas);
      const billboard = {
        position: [0, 0, 0] as const,
        runtimeRole: 'mechanic' as const,
        seed: 1,
        index: 0,
        frameId: 'mechanic-front-stand',
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
        const frameIndex = computeMechanicRuntimeFrameIndex(
          billboard,
          carPosition,
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

    it('gives different mechanics different accent colors', () => {
      const atlas = loadMechanicSpriteAtlas();
      const table = buildMechanicRuntimeIndexTable(atlas);
      const hues: (number | null)[] = [];

      for (let index = 0; index < 8; index += 1) {
        const billboard = {
          position: [0, 0, 0] as const,
          runtimeRole: 'mechanic' as const,
          seed: 1,
          index,
          frameId: 'mechanic-front-stand',
          scale: 1,
          distance: 10,
          quad: [0, 0, 1, 1] as const,
        };

        const frameIndex = computeMechanicRuntimeFrameIndex(
          billboard,
          [0, 0, 10],
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
