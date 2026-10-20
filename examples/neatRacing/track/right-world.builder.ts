/**
 * TRON-style neon scenery prop builder for the neatRacing example
 * (plan slice P1-right-world-props, Step 14D, AC-048).
 *
 * Generates deterministic right-world decorations — pillars, gantries,
 * signs, spectator bleachers, and mechanic/crew props — placed on the
 * right-hand side of the track and kept within a LOD/density budget. The
 * builder honors the USER-CLAR-4 ring rule: spectators/bleachers sit on the
 * outer ring while mechanics/crew sit on the inner ring, both outside the
 * road + curb ribbon. The builder is a pure function of the same
 * `{ seed, layoutVersion, sizeBucket }` determinism tuple used by the track
 * slab builder.
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - Returns at least one prop and never exceeds the configured density budget.
 * - Every prop sits on the right-hand side of the track centerline (negative
 *   left normal in generator space) and outside the road + curb ribbon.
 * - Spectator bleachers carry `ringSide: 'outer'` and mechanic/crew props
 *   carry `ringSide: 'inner'` per the USER-CLAR-4 placement rule.
 * - Two calls with the same tuple produce byte-identical output on the same
 *   runtime.
 */

import { generateTrack } from '../../racing_curriculum/track/track.generator';
import {
  resolveSplineSampleFrame,
  type SplineSampleFrame,
} from '../../racing_curriculum/track/track.spline.utils';
import type { TrackSlabBuildInput } from './track.slabs';

/** Maximum number of right-world neon scenery props emitted per track. */
export const NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET = 64;

/** Outer-ring prop archetypes placed farther from the track centerline. */
const OUTER_RING_KINDS: readonly ('pillar' | 'gantry' | 'sign' | 'bleacher')[] = [
  'pillar',
  'gantry',
  'sign',
  'bleacher',
];

/** Inner-ring prop archetypes placed closer to the track, still off the pavement. */
const INNER_RING_KINDS: readonly ('mechanic' | 'crew' | 'team-banner')[] = [
  'mechanic',
  'crew',
  'team-banner',
];

/** TRON-style neon palette for right-world scenery props. */
const NEON_COLORS: readonly (readonly [number, number, number])[] = [
  [0, 255, 255],
  [255, 0, 128],
  [128, 0, 255],
  [0, 255, 128],
];

/**
 * Minimum XZ center-to-center separation enforced between robot billboards
 * (public crowd + mechanic crew) so sprites do not overlap in world space.
 *
 * A 1 m personal-space radius matches the apparent size of a standing robot
 * sprite at the shared 0.05 m/pixel world scale and prevents z-fighting or
 * visual clustering on small tracks.
 */
export const NEATRACING_ROBOT_PERSONAL_SPACE_M = 1.0;

/** Any object that carries a world-space `position` triplet. */
export interface PositionedInWorld {
  readonly position: readonly [number, number, number];
}

/**
 * Deterministically thins a robot billboard list so every retained robot keeps
 * at least {@link NEATRACING_ROBOT_PERSONAL_SPACE_M} meters of XZ clearance
 * from every other retained robot.
 *
 * Robots are evaluated in input order; the first robot is always kept, and each
 * subsequent robot is kept only when its XZ distance to all previously kept
 * robots is >= the personal-space threshold. This reduces density when the
 * track band cannot host the requested number of robots without overlap, while
 * staying pure, deterministic, and allocation-free on the per-frame draw path.
 *
 * @param robots - Candidate robot billboards in deterministic order.
 * @returns Subset of `robots` with pairwise XZ separation >= 1 m.
 */
export function filterRobotsByPersonalSpace<T extends PositionedInWorld>(
  robots: readonly T[],
): readonly T[] {
  const kept: T[] = [];

  for (const robot of robots) {
    const x = robot.position[0];
    const z = robot.position[2];
    let spaced = true;

    for (let index = 0; index < kept.length; index += 1) {
      const other = kept[index]!;
      const dx = x - other.position[0];
      const dz = z - other.position[2];

      if (Math.hypot(dx, dz) < NEATRACING_ROBOT_PERSONAL_SPACE_M) {
        spaced = false;
        break;
      }
    }

    if (spaced) {
      kept.push(robot);
    }
  }

  return kept;
}

/** Role of a single voxel inside an expanded bleacher prop. */
export type BleacherVoxelRole = 'seat' | 'walk' | 'stair' | 'guardrail';

/** One voxel emitted by a stepped bleacher prop. */
export interface BleacherVoxel {
  /** World-space voxel position in meters. */
  readonly position: readonly [number, number, number];
  /** 0-255 RGB neon color. */
  readonly color: readonly [number, number, number];
  /** Structural role inside the bleacher geometry. */
  readonly role: BleacherVoxelRole;
}

/** Geometry carried by a bleacher prop after P1-bleacher-shape. */
export interface BleacherGeometry {
  /** Ordered voxel set describing seats, walking space, stairs, and guardrails. */
  readonly voxels: readonly BleacherVoxel[];
}

/** Which ring-side band each archetype belongs to (USER-CLAR-4 placement rule). */
const RING_SIDE_BY_KIND: Record<RightWorldProp['kind'], 'inner' | 'outer'> = {
  pillar: 'outer',
  gantry: 'outer',
  sign: 'outer',
  bleacher: 'outer',
  mechanic: 'inner',
  crew: 'inner',
  'team-banner': 'inner',
};

/** One TRON-style neon scenery prop. */
export interface RightWorldProp {
  /** World-space position in meters. */
  readonly position: readonly [number, number, number];
  /** 0-255 RGB neon color. */
  readonly color: readonly [number, number, number];
  /** Prop archetype. */
  readonly kind: 'pillar' | 'gantry' | 'sign' | 'bleacher' | 'mechanic' | 'crew' | 'team-banner';
  /**
   * Placement band relative to the track ribbon.
   *
   * - `outer` — spectator/scenery props (bleachers, gantries, pillars, signs)
   *   placed farther from the centerline.
   * - `inner` — mechanic/crew props placed closer to the racing surface while
   *   still outside the road + curb band (USER-CLAR-4).
   */
  readonly ringSide: 'inner' | 'outer';
  /**
   * Optional stepped voxel geometry for spectator bleachers.
   *
   * Carries the seat, walk, stair, and guardrail voxels that describe the
   * prop's shape beyond the bounding box (P1-bleacher-shape, AC-066).
   */
  readonly bleacherGeometry?: BleacherGeometry;
}

/**
 * Builds the deterministic right-world neon scenery prop list for the given
 * track tuple.
 *
 * @param input - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.
 * @returns Ordered list of scenery props ready for the right-world render
 *   pass. Props are placed on the right-hand side of the track centerline,
 *   outside the road + curb ribbon, and never exceed the density budget.
 */
export function buildRightWorldProps(
  input: TrackSlabBuildInput,
): readonly RightWorldProp[] {
  const spec = generateTrack(input);
  const samples = spec.splineSamples;
  if (samples.length === 0) {
    return [];
  }

  const random = createDeterministicRandom(resolveRightWorldSeed(input));
  const requestedCount = Math.min(
    NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET,
    Math.max(1, Math.floor(samples.length / 10)),
  );
  // USER-CLAR-4: reserve slots for both the inner mechanic/crew ring and the
  // outer spectator/scenery ring so neither archetype band is empty.
  const innerTargetCount = Math.max(1, Math.floor(requestedCount * 0.25));
  const outerTargetCount = Math.max(1, requestedCount - innerTargetCount);
  const props: RightWorldProp[] = [];

  const emitRing = (
    count: number,
    kinds: readonly RightWorldProp['kind'][],
    baseOffset: number,
    offsetRange: number,
  ): void => {
    for (let index = 0; index < count; index += 1) {
      const sampleIndex = Math.floor((index * samples.length) / count);
      const sample = samples[sampleIndex]!;
      const frame = resolveSplineSampleFrame(samples, sample.globalIndex);
      const offset = sample.width / 2 + baseOffset + random() * offsetRange;
      const generatorX = sample.x - frame.normalX * offset;
      const generatorY = sample.y - frame.normalY * offset;

      const kind = kinds[Math.floor(random() * kinds.length)]!;
      const color = NEON_COLORS[Math.floor(random() * NEON_COLORS.length)]!;
      const prop: RightWorldProp = {
        position: [generatorX, 0, -generatorY],
        color,
        kind,
        ringSide: RING_SIDE_BY_KIND[kind],
        ...(kind === 'bleacher'
          ? {
              bleacherGeometry: buildBleacherGeometry(
                [generatorX, 0, -generatorY],
                color,
                frame,
              ),
            }
          : {}),
      };

      props.push(prop);
    }
  };

  // Inner ring: mechanic/crew props closer to the track, still off the pavement.
  emitRing(innerTargetCount, INNER_RING_KINDS, 2, 2);
  // Outer ring: spectator bleachers + neon scenery props farther out.
  emitRing(outerTargetCount, OUTER_RING_KINDS, 6, 4);

  return props;
}

/**
 * Builds the stepped voxel geometry for one spectator bleacher.
 *
 * The local frame is aligned to the spline tangent so the front of the
 * bleacher (guardrail side) faces the track and the back side (stairs) faces
 * away. Seats are arranged in ascending rows toward the back, walking space
 * covers the full horizontal footprint, and the guardrail band caps the front
 * edge above the top row.
 *
 * @param position - World-space center of the bleacher prop.
 * @param color - Neon color applied to every voxel.
 * @param frame - Spline sample frame at the nearest track sample.
 * @returns Bleacher voxel geometry carrying seats, walk, stairs, guardrail.
 */
function buildBleacherGeometry(
  position: readonly [number, number, number],
  color: readonly [number, number, number],
  frame: SplineSampleFrame,
): BleacherGeometry {
  const [px, py, pz] = position;
  // World-space basis derived from the spline frame. The left normal points
  // toward the track (higher lateral), which is the front/audience side of the
  // bleacher. The tangent runs along the track direction and spans the width of
  // the seating rows.
  const normalX = frame.normalX;
  const normalZ = -frame.normalY;
  const tangentX = -frame.normalY;
  const tangentZ = -frame.normalX;

  const voxels: BleacherVoxel[] = [];

  const addVoxel = (
    u: number,
    v: number,
    y: number,
    role: BleacherVoxelRole,
  ): void => {
    voxels.push({
      position: [
        px + u * normalX + v * tangentX,
        py + y,
        pz + u * normalZ + v * tangentZ,
      ],
      color,
      role,
    });
  };

  // USER-CLAR-8 (fix-packet-P1-bleacher-crowd-iteration-1, OBS-A): enlarged
  // flat-tier 3-level stadium for standing robots. Each level contributes four
  // stand-cell rows (twice the previous depth). Levels are flat horizontal
  // slabs; the whole level steps up toward the back of the bleacher, with a
  // stair column between consecutive levels pointing back (decreasing u).
  // Walking surface covers the full footprint with one free voxel row on each
  // outer side, and the track-facing guardrail band sits at the front free row.
  const LEVELS: readonly { readonly uStart: number; readonly y: number }[] = [
    // Level 1 (lowest, nearest the track)
    { uStart: 3, y: 0.7 },
    // Level 2
    { uStart: -1, y: 1.8 },
    // Level 3 (highest, farthest from the track)
    { uStart: -5, y: 2.9 },
  ];
  const ROWS_PER_LEVEL = 4;
  // Seven cells wide (span 6 m) plus one free voxel row on each outer side.
  const seatV = [-3, -2, -1, 0, 1, 2, 3];
  for (const level of LEVELS) {
    for (let row = 0; row < ROWS_PER_LEVEL; row += 1) {
      const u = level.uStart - row;
      for (const v of seatV) {
        addVoxel(u, v, level.y, 'seat');
      }
    }
  }

  // Walking surface across the full footprint, including the outer free rows.
  const walkU = [4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9];
  const walkV = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
  for (const u of walkU) {
    for (const v of walkV) {
      addVoxel(u, v, 0.1, 'walk');
    }
  }

  // Track-facing guardrail band at the front free row of the whole bleacher.
  // Rails are placed at each level's seat-top surface so the front row of each
  // level is protected without sharing x/z cells with the stand cells behind.
  const guardrailU = 4;
  const guardrailV = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
  const guardrailHeights = LEVELS.map(
    (level) => level.y + BLEACHER_VOXEL_HALF_HEIGHT_M,
  );
  for (const y of guardrailHeights) {
    for (const v of guardrailV) {
      addVoxel(guardrailU, v, y, 'guardrail');
    }
  }

  // Stairs between consecutive levels, on the back side (decreasing u). Each
  // stair column sits at the u midpoint between the back row of the lower
  // level and the front row of the upper level, at the midpoint height.
  const stairV = [-3, -2, -1, 0, 1, 2, 3];
  for (let index = 0; index < LEVELS.length - 1; index += 1) {
    const lower = LEVELS[index]!;
    const upper = LEVELS[index + 1]!;
    const backLowerU = lower.uStart - (ROWS_PER_LEVEL - 1);
    const frontUpperU = upper.uStart;
    const stairU = (backLowerU + frontUpperU) / 2;
    const stairY = (lower.y + upper.y) / 2;
    for (const v of stairV) {
      addVoxel(stairU, v, stairY, 'stair');
    }
  }

  return { voxels };
}

/** World-space horizontal half-extent used when drawing a bleacher voxel box. */
export const BLEACHER_VOXEL_HALF_EXTENT_M = 0.5;
/** World-space vertical half-height used when drawing a bleacher voxel box. */
export const BLEACHER_VOXEL_HALF_HEIGHT_M = 0.1;

/**
 * Returns the top Y of the seat surface at a given world-space XZ location.
 *
 * When `x` and `z` are omitted, this returns the highest seat row in the
 * bleacher (used for rendering bounding-box checks). When provided, it returns
 * the y of the nearest seat voxel so a billboard anchored to a lower level is
 * not hoisted to the top row (AC-066 + USER-CLAR-8).
 */
export function getBleacherSeatTopY(
  bleacher: RightWorldProp,
  x?: number,
  z?: number,
): number {
  const voxels = bleacher.bleacherGeometry?.voxels;
  if (!voxels) {
    return bleacher.position[1] + 1.5;
  }

  let top = -Infinity;
  if (x !== undefined && z !== undefined) {
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const voxel of voxels) {
      if (voxel.role !== 'seat') {
        continue;
      }
      const distance = Math.hypot(
        x - voxel.position[0],
        z - voxel.position[2],
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        top = voxel.position[1] + BLEACHER_VOXEL_HALF_HEIGHT_M;
      }
    }
  } else {
    for (const voxel of voxels) {
      if (voxel.role === 'seat') {
        const surfaceY = voxel.position[1] + BLEACHER_VOXEL_HALF_HEIGHT_M;
        if (surfaceY > top) {
          top = surfaceY;
        }
      }
    }
  }

  return Number.isFinite(top) ? top : bleacher.position[1] + 3.0;
}

/**
 * Returns the top Y of the bleacher walking surface.
 */
export function getBleacherWalkSurfaceY(bleacher: RightWorldProp): number {
  const voxels = bleacher.bleacherGeometry?.voxels;
  if (!voxels) {
    return bleacher.position[1] + 0.1;
  }

  let top = -Infinity;
  for (const voxel of voxels) {
    if (voxel.role === 'walk') {
      const surfaceY = voxel.position[1] + BLEACHER_VOXEL_HALF_HEIGHT_M;
      if (surfaceY > top) {
        top = surfaceY;
      }
    }
  }

  return Number.isFinite(top) ? top : bleacher.position[1] + 0.2;
}

/**
 * Returns the axis-aligned world-space half-width and half-depth that tightly
 * enclose all voxels of a bleacher. This is the real footprint used for host
 * matching and anchoring-capacity calculations (AC-066).
 */
export function getBleacherAxisAlignedHalfSize(
  bleacher: RightWorldProp,
): readonly [number, number] {
  const voxels = bleacher.bleacherGeometry?.voxels;
  if (!voxels || voxels.length === 0) {
    return [2, 1];
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const voxel of voxels) {
    const [x, , z] = voxel.position;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  return [(maxX - minX) / 2, (maxZ - minZ) / 2];
}

/**
 * Returns the horizontal hosting footprint area of a bleacher in square meters,
 * derived from the real voxel geometry instead of the coarse bounding box.
 */
export function getBleacherHostingFootprintArea(bleacher: RightWorldProp): number {
  const [halfWidth, halfDepth] = getBleacherAxisAlignedHalfSize(bleacher);
  return (2 * halfWidth) * (2 * halfDepth);
}

/**
 * Returns the world-space anchors for every stand-cell seat voxel in a
 * bleacher, covering all levels. The returned y coordinate is the rendered
 * seat-top surface for that cell; callers that place billboards should add
 * their own surface clearance epsilon.
 */
export function getBleacherSeatAnchors(
  bleacher: RightWorldProp,
): readonly [number, number, number][] {
  const voxels = bleacher.bleacherGeometry?.voxels;
  if (!voxels) {
    return [];
  }

  const anchors: [number, number, number][] = [];
  for (const voxel of voxels) {
    if (voxel.role === 'seat') {
      anchors.push([
        voxel.position[0],
        voxel.position[1] + BLEACHER_VOXEL_HALF_HEIGHT_M,
        voxel.position[2],
      ]);
    }
  }

  return anchors;
}

/**
 * Resolves a deterministic unsigned 32-bit seed for the right-world builder.
 *
 * @param input - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.
 * @returns Unsigned 32-bit seed for the local PRNG.
 */
function resolveRightWorldSeed(input: TrackSlabBuildInput): number {
  let combinedSeed = input.seed ^ (input.layoutVersion * 0x9e3779b9);

  for (const character of input.sizeBucket) {
    combinedSeed = Math.imul(combinedSeed ^ character.charCodeAt(0), 16777619);
  }

  return ((combinedSeed >>> 0) ^ 0x7269_6768) >>> 0;
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
