/**
 * Red contract tests for the neatRacing right-world robot billboard spacing
 * (slice P1-sprite-spacing, Step 14J, AC-065).
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - Every robot billboard (public crowd on bleachers/ground and mechanic crew
 *   near pedestals) keeps a personal-space radius >= 1 m equivalent in the XZ
 *   plane.
 * - No two robot sprites overlap in world space.
 */

import { describe, expect, it } from '@jest/globals';
import type { TrackSlabBuildInput } from './track.slabs';
import {
  buildMechanicBillboards,
  loadMechanicSpriteAtlas,
} from './mechanic.sprite';
import {
  buildPublicBillboards,
  loadPublicSpriteAtlas,
} from './public.sprite';
import { buildRightWorldSceneData } from './right-world.pass';

/**
 * Determinism tuple selected to expose current sprite overlap.
 *
 * Seed 61 / sizeBucket 'small' places multiple robot billboards within a
 * sub-meter XZ radius, violating AC-065. 04-implementing must introduce
 * spacing enforcement to make this contract pass.
 */
const FIXTURE: TrackSlabBuildInput = {
  seed: 61,
  layoutVersion: 1,
  sizeBucket: 'small',
} as const;

/** Minimal camera used to exercise billboard placement and projection. */
const FIXTURE_CAMERA = {
  position: [0, 2, -10] as const,
  forward: [0, 0, 1] as const,
  right: [1, 0, 0] as const,
  near: 0.1,
  far: 1000,
  fovDegrees: 60,
} as const;

/** Minimum XZ center-to-center separation enforced by AC-065 (meters). */
const NEATRACING_ROBOT_PERSONAL_SPACE_M = 1.0;

type Robot = {
  readonly position: readonly [number, number, number];
  readonly kind: 'public' | 'mechanic';
  readonly index: number;
};

/** Computes the smallest XZ distance between any two robot billboards. */
function minRobotXZDistance(robots: readonly Robot[]): number {
  let minDistance = Number.POSITIVE_INFINITY;

  for (let i = 0; i < robots.length; i += 1) {
    const a = robots[i]!;
    for (let j = i + 1; j < robots.length; j += 1) {
      const b = robots[j]!;
      const dx = a.position[0] - b.position[0];
      const dz = a.position[2] - b.position[2];
      const distance = Math.hypot(dx, dz);
      if (distance < minDistance) {
        minDistance = distance;
      }
    }
  }

  return minDistance;
}

/** Builds the combined public + mechanic robot list for spacing tests. */
function buildAllRobots(input: TrackSlabBuildInput): readonly Robot[] {
  // AC-065 spacing is enforced by the combined right-world scene assembly
  // (anchoring-capacity + personal-space filters). Test the assembled scene
  // rather than the raw sprite builders so cross-kind overlaps are removed.
  const scene = buildRightWorldSceneData(input, FIXTURE_CAMERA);

  return scene.billboards.map((billboard, index) => ({
    position: billboard.position,
    kind: billboard.runtimeRole === 'mechanic' ? ('mechanic' as const) : ('public' as const),
    index,
  }));
}

describe('right-world robot billboard spacing (P1-sprite-spacing, AC-065)', () => {
  it('maintains >= 1m XZ separation between every pair of robot billboards', () => {
    const robots = buildAllRobots(FIXTURE);

    expect(robots.length).toBeGreaterThan(1);

    const minDistance = minRobotXZDistance(robots);

    expect(minDistance).toBeGreaterThanOrEqual(NEATRACING_ROBOT_PERSONAL_SPACE_M);
  });

  it('keeps public crowd billboards at least 1m apart in the XZ plane', () => {
    const atlas = loadPublicSpriteAtlas();
    const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
    const publicRobots = billboards.map((billboard, index) => ({
      position: billboard.position,
      kind: 'public' as const,
      index,
    }));

    expect(publicRobots.length).toBeGreaterThan(0);

    const minDistance = minRobotXZDistance(publicRobots);

    expect(minDistance).toBeGreaterThanOrEqual(NEATRACING_ROBOT_PERSONAL_SPACE_M);
  });

  it('keeps mechanic-crew billboards at least 1m apart in the XZ plane', () => {
    const atlas = loadMechanicSpriteAtlas();
    const billboards = buildMechanicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
    const mechanicRobots = billboards.map((billboard, index) => ({
      position: billboard.position,
      kind: 'mechanic' as const,
      index,
    }));

    expect(mechanicRobots.length).toBeGreaterThan(0);

    const minDistance = minRobotXZDistance(mechanicRobots);

    expect(minDistance).toBeGreaterThanOrEqual(NEATRACING_ROBOT_PERSONAL_SPACE_M);
  });
});
