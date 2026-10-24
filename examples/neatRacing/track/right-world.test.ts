/**
 * Red contract tests for the neatRacing right-world robot billboard spacing
 * (slice P1-sprite-spacing, Step 14J, AC-065).
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - Every robot billboard (the public crowd on bleachers/ground) keeps a
 *   personal-space radius >= 1 m equivalent in the XZ plane.
 * - No two robot sprites overlap in world space.
 *
 * The AC-070 retirement removed the mechanic/crew people path, so the scene
 * carries the public crowd only.
 *
 * RED-E (slice P1-sprite-scale, Step 14L, AC-065 retained) adds the
 * NON-FAILING regression guard: the personal space is a FIXED constant and is
 * NOT re-derived from the halved (SPRITE-1) crowd sprite size.
 */

import { describe, expect, it } from '@jest/globals';
import type { TrackSlabBuildInput } from './track.slabs';
import {
  buildPublicBillboards,
  loadPublicSpriteAtlas,
  NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M,
  NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE,
} from './public.sprite';
import {
  filterRobotsByPersonalSpace,
  NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M,
  NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M,
  NEATRACING_ROBOT_PERSONAL_SPACE_M,
} from './right-world.builder';
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

/**
 * Effective AC-065 clearance: the nominal 1 m personal space minus the named
 * margin.
 *
 * The margin is the NAMED tolerance of the personal-space comparison: the
 * bleacher stand-cell lattice is mathematically 1 m wide, but the world-space
 * seat positions are produced through a rotated basis, so the realized
 * separation of two neighbouring cells measures a few 1e-14 m short of 1 m.
 * Asserting against the margin expresses the SAME 1 m contract without
 * weakening it.
 */
const AC_065_EFFECTIVE_CLEARANCE_M =
  NEATRACING_ROBOT_PERSONAL_SPACE_M - NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M;

/**
 * Meters per sprite pixel of the crowd atlas (the shared world pixel scale).
 *
 * The value is the SHARED export of `public.sprite.ts` (0.05 m/px), so the
 * RED-E guard measures the halved sprite against the meter scale the module
 * really uses instead of a local copy that could drift.
 */
const CROWD_METERS_PER_PIXEL_M = NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M;

/**
 * The SPRITE-1 halving factor of the crowd billboard world scale.
 *
 * The crowd frames are 192 x 192 px, so the halved sprite is
 * `192 * 0.5 * 0.05 = 4.8 m` wide instead of the pre-14L 9.6 m. The factor is
 * the NAMED export of `public.sprite.ts`, the single source of truth of the
 * halving.
 */
const HALVED_CROWD_WORLD_SCALE = NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE;

/**
 * XZ separation of the RED-E probe pair, in meters.
 *
 * The probe pair sits CLOSER than the halved sprite is wide, so a personal
 * space that was re-derived from the sprite size would drop it. The fixed
 * 1 m rule keeps it.
 */
const PERSONAL_SPACE_PROBE_SEPARATION_M = 1.5;

type Robot = {
  readonly position: readonly [number, number, number];
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

/** Builds the assembled right-world robot list for spacing tests. */
function buildAllRobots(input: TrackSlabBuildInput): readonly Robot[] {
  // AC-065 spacing is enforced by the combined right-world scene assembly
  // (anchoring-capacity + personal-space filters). Test the assembled scene
  // rather than the raw sprite builders so overlaps are removed.
  const scene = buildRightWorldSceneData(input, FIXTURE_CAMERA);

  return scene.billboards.map((billboard, index) => ({
    position: billboard.position,
    index,
  }));
}

describe('right-world robot billboard spacing (P1-sprite-spacing, AC-065)', () => {
  it('maintains >= 1m XZ separation between every pair of robot billboards', () => {
    const robots = buildAllRobots(FIXTURE);

    expect(robots.length).toBeGreaterThan(1);

    const minDistance = minRobotXZDistance(robots);

    // B4: the 1 m contract is asserted against the NAMED margin
    // (`NEATRACING_ROBOT_PERSONAL_SPACE_M -
    // NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M`), never against a bare 1 m.
    expect(minDistance).toBeGreaterThanOrEqual(AC_065_EFFECTIVE_CLEARANCE_M);
  });

  it('keeps public crowd billboards at least 1m apart in the XZ plane', () => {
    const atlas = loadPublicSpriteAtlas();
    const billboards = buildPublicBillboards(FIXTURE, FIXTURE_CAMERA, atlas);
    const publicRobots = billboards.map((billboard, index) => ({
      position: billboard.position,
      index,
    }));

    expect(publicRobots.length).toBeGreaterThan(0);

    const minDistance = minRobotXZDistance(publicRobots);

    // B4: the 1 m contract is asserted against the NAMED margin
    // (`NEATRACING_ROBOT_PERSONAL_SPACE_M -
    // NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M`), never against a bare 1 m.
    expect(minDistance).toBeGreaterThanOrEqual(AC_065_EFFECTIVE_CLEARANCE_M);
  });
});

/**
 * RED-E (AC-065 retained, P1-sprite-scale): the personal space is a FIXED
 * constant and must NOT be re-derived from the sprite size.
 *
 * The SPRITE-1 halving shrinks the crowd billboards from 9.6 m to 4.8 m of
 * world size. A 4.8 m wide sprite footprint could visually overlap at the
 * nominal 1 m personal space, but the personal space is a NAMED CONSTANT: it
 * stays exactly 1.0 m with its named 1e-12 m margin, and the >= 1 m XZ
 * separation of every crowd billboard pair keeps holding.
 *
 * These assertions are deliberately NON-FAILING regression guards: they pass
 * today and must keep passing after the halving. They fail loudly if a
 * future change re-derives the personal space from the sprite size.
 */
describe(
  'right-world personal space is a fixed constant, not sprite-derived (RED-E, AC-065 retained)',
  () => {
    it('keeps the personal-space threshold at exactly 1.0 m', () => {
      // The halving must not move the threshold: 1.0 m is the AC-065 value.
      expect(NEATRACING_ROBOT_PERSONAL_SPACE_M).toBe(1.0);
    });

    it('keeps the named margin on the personal-space threshold', () => {
      // The margin stays the named 1e-12 m tolerance, and the effective
      // clearance stays the nominal 1 m MINUS that margin - no sprite-derived
      // term may sneak into the effective clearance.
      expect(NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M).toBe(1e-12);
      expect(NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M).toBe(1.0 - 1e-12);
    });

    it('keeps the >= 1 m XZ separation of the assembled crowd with the halved sprite', () => {
      // The halving changes the rendered quad, not the placement, and the
      // personal-space filter thins the crowd independently of the sprite
      // size - so the assembled crowd keeps the 1 m separation contract.
      const robots = buildAllRobots(FIXTURE);

      expect(robots.length).toBeGreaterThan(1);

      const minDistance = minRobotXZDistance(robots);

      expect(minDistance).toBeGreaterThanOrEqual(AC_065_EFFECTIVE_CLEARANCE_M);
    });

    it('does not re-derive the personal space from the halved sprite footprint', () => {
      const atlas = loadPublicSpriteAtlas();
      const frame = atlas.frames[0]!;
      const halvedSpriteWidthM =
        frame.width * HALVED_CROWD_WORLD_SCALE * CROWD_METERS_PER_PIXEL_M;

      // The probe pair sits CLOSER than the halved sprite is wide, so a
      // sprite-derived personal space would drop the second billboard. The
      // fixed 1 m rule keeps it.
      const kept = filterRobotsByPersonalSpace([
        { position: [0, 0, 0] as const },
        { position: [PERSONAL_SPACE_PROBE_SEPARATION_M, 0, 0] as const },
      ]);

      expect(kept.length).toBe(2);
      expect(halvedSpriteWidthM).toBeGreaterThan(PERSONAL_SPACE_PROBE_SEPARATION_M);
    });

    it('still drops a crowd pair that sits closer than the fixed 1 m personal space', () => {
      const kept = filterRobotsByPersonalSpace([
        { position: [0, 0, 0] as const },
        { position: [0.5, 0, 0] as const },
      ]);

      expect(kept.length).toBe(1);
    });
  },
);
