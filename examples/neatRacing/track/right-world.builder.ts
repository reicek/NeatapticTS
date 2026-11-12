/**
 * Deterministic TRON-style neon scenery prop builder for the neatRacing
 * example.
 *
 * The builder turns one determinism tuple - `{ seed, layoutVersion,
 * sizeBucket }` - into an ordered list of neon props that line the
 * right-hand side of the track: pillars, gantries, signs, and stepped
 * spectator bleachers. Every prop stands at least
 * {@link NEATRACING_PROP_CLEARANCE_FLOOR_M} beyond the SEAM EDGE, so the
 * driving line is never obstructed, and the total count stays inside a
 * density budget the render pass can afford.
 *
 * Four properties make the output usable by a renderer and by the crowd
 * builder that reads it:
 *
 * - DETERMINISM. The same tuple always produces the same prop list, byte for
 *   byte, on the same runtime. The builder owns its own seeded PRNG, so it
 *   never depends on ambient randomness, and every later prop stays exactly
 *   where it was when an earlier rule changed.
 * - A GUARANTEED CROWD. The outer ring reserves EXACTLY
 *   {@link NEATRACING_RIGHT_WORLD_BLEACHER_COUNT} bleacher slots BY
 *   CONSTRUCTION at evenly spaced slot indices, so every scene offers the
 *   same number of spectator stands whatever the seed and whatever the track
 *   size bucket is. The bleacher slots are reserved BY CONSTRUCTION rather
 *   than drawn at random, and the reserved slots are spread far enough
 *   apart that no two bleachers overlap in world space.
 * - A PIT BAND. The inner ring carries the mechanic, crew and team-banner
 *   props of the re-populated right world. The pit band sits closer to the
 *   road than the bleachers and at the midpoints between the bleacher
 *   blocks, so the two rings never share a track position.
 * - A SEAM-RELATIVE PLACEMENT. Every clearance is MEASURED against the
 *   SEAM EDGE - the published edge profile of the track, read through the
 *   road's own union reader - and NEVER against the retired
 *   spline-normal distance model, so the props and the road can never
 *   disagree about where the edge is.
 *
 * A bleacher is a stepped voxel structure rather than a box: seats, a walking
 * surface, stairs, and a track-facing guardrail, all expressed in the real
 * world-space geometry that the crowd builder and the renderer both read.
 */

import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type {
  SplineSample,
  TrackSegment,
} from '../../racing_curriculum/track/track.generator.types';
import {
  resolveSplineSampleFrame,
  type SplineSampleFrame,
} from '../../racing_curriculum/track/track.spline.utils';
import { resolveTrackEdgeProfile } from './track.edge';
import type { TrackEdgeProfile } from './track.edge';
import { classifyTrackSlabSurfaceAt } from './track.slabs';
import type { TrackSlabBuildInput } from './track.slabs';

/**
 * Maximum number of right-world neon scenery props emitted per track.
 *
 * The budget is the density ceiling of the right world and it covers BOTH
 * rings: the outer-ring slots AND the inner pit band. The outer ring
 * therefore reserves the pit band before it counts its own blocks, so the
 * emitted prop list never exceeds the budget. The budget is large enough for
 * the twenty reserved bleachers AND the twenty scenery slots beside them to
 * fit on every reachable track.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * console.log(props.length); // 50 - 20 bleachers, 20 scenery props, 10 pit props
 * console.log(props.length <= NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET); // true
 * ```
 */
export const NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET = 100;

/**
 * Number of spectator bleachers the right world carries.
 *
 * The count is a CONSTANT of the world, not a function of the seed or of
 * the track size bucket: the outer ring reserves this many bleacher slots BY
 * CONSTRUCTION at evenly spaced slot indices, so every scene offers the same
 * number of spectator stands. The reserved slots sit in the outer ring, so
 * they are placed farther from the centerline than the inner-ring pit props.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * console.log(
 *   props.filter((prop) => prop.kind === 'bleacher').length,
 * ); // 20 - the reserved slots, whatever the seed
 * ```
 */
export const NEATRACING_RIGHT_WORLD_BLEACHER_COUNT = 20;

/**
 * Number of inner-ring pit props the right world carries.
 *
 * The pit band places one prop at every midpoint between two reserved
 * bleacher slots, so the band holds one prop per bleacher pair - half the
 * bleacher count. The constant is part of the density budget arithmetic:
 * the budget covers the outer ring AND the band, so the outer ring reserves
 * the band before it counts its own blocks.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * console.log(props.filter((prop) => prop.ringSide === 'inner').length); // 10
 * ```
 */
export const NEATRACING_RIGHT_WORLD_PEOPLE_RING_COUNT = 10;

/**
 * Outer-ring prop archetypes placed farther from the track centerline.
 *
 * The bleacher archetype is NOT part of this draw table: the bleacher slots
 * are reserved BY CONSTRUCTION, so the random archetype draw can never
 * produce a bleacher.
 */
const OUTER_RING_KINDS: readonly ('pillar' | 'gantry' | 'sign')[] = [
  'pillar',
  'gantry',
  'sign',
];

/**
 * People prop kinds carried by the inner ring.
 *
 * The inner ring is the pit band of the re-populated right world: the
 * mechanics, the crew, and the team banners. The kinds are assigned BY
 * CONSTRUCTION in a fixed round robin, so every kind is present at every
 * seed and the scene never depends on a random draw to carry a people prop.
 */
const INNER_RING_PEOPLE_KINDS: readonly (
  'mechanic' | 'crew' | 'team-banner'
)[] = ['mechanic', 'crew', 'team-banner'];

/**
 * The USER-NAMED clearance floor of every right-world prop, in meters.
 *
 * The floor is the distance EVERY prop stands beyond the SEAM EDGE - the
 * one edge the road's own readers publish - and it is a USER directive, not
 * an author decision: it is never lowered, never widened by a tolerance and
 * never satisfied by dropping a prop. The placement clamps EVERY clearance
 * target to this floor, so the floor holds for every prop kind, at every
 * seed and for every track size bucket.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_PROP_CLEARANCE_FLOOR_M); // 10
 * ```
 */
export const NEATRACING_PROP_CLEARANCE_FLOOR_M = 10;

/**
 * Radial step of the seam-clearance search, in meters.
 *
 * The search walks the outward ray of a track sample in steps of this size
 * and stops at the FIRST candidate whose measured clearance beyond the seam
 * edge reaches the target. The step is the RESOLUTION of the measured
 * placement: a finer step resolves the clearance better and costs one extra
 * measurement per step.
 */
const NEATRACING_SEAM_CLEARANCE_STEP_M = 0.25;

/**
 * Largest outward offset the seam-clearance search walks, in meters.
 *
 * The bound keeps the search finite when the target clearance is not
 * reachable along the ray: the search then returns the candidate with the
 * BEST measured clearance, so the placement stays the best effort the world
 * can offer instead of an unbounded offset.
 */
const NEATRACING_SEAM_CLEARANCE_MAX_OFFSET_M = 64;

/**
 * Local depth of the front seat row of a bleacher, in meters.
 *
 * The value is the local `u` coordinate of the front row of the LOWEST
 * platform of a bleacher - the row closest to the track, and the row the
 * seated crowd of that platform stands on. It is the PROBE point of a
 * bleacher: the clearance search places the stand so that row clears the
 * floor, and the stand's own anchor then sits that many meters further out.
 */
const NEATRACING_BLEACHER_FRONT_SEAT_U_M = 3;

/**
 * Width of the bleacher footprint ALONG the track, in meters.
 *
 * The seats of a bleacher span seven cells plus the free outer rows, so the
 * structure is nine meters wide across the track direction.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_WIDTH_M); // 9
 * ```
 */
export const NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_WIDTH_M = 9;

/**
 * Depth of the bleacher footprint ACROSS the track, in meters.
 *
 * The walking surface runs from the track-facing guardrail row back to the
 * stair column, so the structure is fourteen meters deep in the lateral
 * direction.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_DEPTH_M); // 14
 * ```
 */
export const NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_DEPTH_M = 14;

/**
 * Minimum world-space center-to-center separation between two bleacher
 * FOOTPRINT CENTRES.
 *
 * Two bleachers whose footprint centres are closer than the sum of their
 * circumradii COULD overlap: the Minkowski difference of two rectangles of
 * the same shape is contained in the disk of that radius. The sum of the
 * two circumradii of a bleacher footprint is
 * `2 * hypot(footprintWidth / 2, footprintDepth / 2)` = 16.64 m, so the
 * named floor is that threshold rounded UP to the next tenth of a meter -
 * 16.7 m - which is a sufficient - and conservative - no-overlap condition
 * for the NOMINAL footprint rectangle. The rendered stand is the voxel
 * boxes, and every box adds a half extent per side, so the RENDERED
 * footprint is 10 m by 15 m and carries its own sufficient bound of
 * 18.03 m; the realized separation is far beyond both bounds.
 * The evenly spaced reserved slots realize a separation of the track length
 * divided by the bleacher count, far beyond this floor.
 *
 * The footprint centre is NOT the prop anchor: the anchor sits 4 m inboard
 * of the track-facing guardrail row, so the footprint centre lies 2.5 m
 * outward of it. The separation contract therefore measures the footprint
 * centres - the quantity this floor is derived from - and the prop anchors
 * as well.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_RIGHT_WORLD_BLEACHER_MIN_SEPARATION_M); // 16.7
 * ```
 */
export const NEATRACING_RIGHT_WORLD_BLEACHER_MIN_SEPARATION_M =
  Math.ceil(
    2 *
      Math.hypot(
        NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_WIDTH_M / 2,
        NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_DEPTH_M / 2,
      ) *
      10,
  ) / 10;

/**
 * Number of track samples one outer-ring slot needs.
 *
 * The outer ring is spread over the whole track, so a denser track can host
 * more slots. The constant is the DENSITY INPUT of the block rule in
 * {@link resolveOuterRingSlotCount}: one more block of twenty slots needs
 * `SAMPLES_PER_RING_SLOT * BLEACHER_COUNT` more track samples.
 *
 * On every REACHABLE track the constant does not bind: a reachable track
 * carries 108 to 144 samples, fewer than the 200 samples an extra block
 * needs, so the density term of the rule is zero and the floor of
 * {@link NEATRACING_RIGHT_WORLD_MIN_RING_BLOCKS} blocks governs the slot
 * count. The constant stays the named input of the rule, so a longer track
 * can widen the ring without the rule losing its density guard.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_RIGHT_WORLD_SAMPLES_PER_RING_SLOT); // 10
 * console.log(
 *   Math.floor(108 / NEATRACING_RIGHT_WORLD_SAMPLES_PER_RING_SLOT),
 * ); // 10 - fewer samples than one extra block of twenty slots needs
 * ```
 */
export const NEATRACING_RIGHT_WORLD_SAMPLES_PER_RING_SLOT = 10;

/**
 * Minimum number of bleacher blocks of the outer ring.
 *
 * The floor keeps the twenty reserved bleachers on every reachable track,
 * even a short one. The block count is never lowered to fit the track: a
 * track that is too short for twenty non-overlapping bleachers shows up as a
 * FAILING separation contract instead of a silently reduced crowd.
 *
 * @example
 * ```ts
 * console.log(
 *   resolveOuterRingSlotCount(108) / NEATRACING_RIGHT_WORLD_BLEACHER_COUNT,
 * ); // 2 - the floor governs a reachable track
 * ```
 */
export const NEATRACING_RIGHT_WORLD_MIN_RING_BLOCKS = 2;

/**
 * TRON-style neon palette for right-world scenery props. */
const NEON_COLORS: readonly (readonly [number, number, number])[] = [
  [0, 255, 255],
  [255, 0, 128],
  [128, 0, 255],
  [0, 255, 128],
];

/**
 * Minimum XZ center-to-center separation enforced between robot billboards
 * (the public crowd) so two spectators never occupy the same spot.
 *
 * The personal space is a FIXED world-space constant, and it is deliberately
 * NOT derived from the size of a crowd sprite. The world size of a crowd
 * billboard belongs to a different pair of levers - the shared
 * meters-per-pixel scale `NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M` and the
 * world-scale lever `NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE` of
 * `public.sprite.ts` - and a change to either of them must never move this
 * threshold. A crowd keeps its one-meter separation whatever the sprite
 * size is.
 *
 * The one-meter value is the PITCH of the bleacher stand lattice: a stand
 * cell is one 1 m voxel cell on the integer local lattice, so neighbouring
 * cells are one meter apart. A personal space of one meter therefore never
 * drops a spectator that stands on a cell of its own; it thins away the
 * billboards that would otherwise land on the same spot and z-fight there.
 *
 * The comparison runs against
 * {@link NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M} - the nominal value
 * minus the named {@link NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M} margin
 * - so every retained pair still keeps a full meter of separation.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_ROBOT_PERSONAL_SPACE_M); // 1 - one meter of space
 * ```
 */
export const NEATRACING_ROBOT_PERSONAL_SPACE_M = 1.0;

/**
 * Named tolerance applied to the personal-space comparison.
 *
 * The bleacher stand-cell lattice is mathematically 1 m wide, but the
 * world-space seat positions are produced through a rotated basis, so the
 * realized separation of two neighbouring cells measures a few 1e-14 m short
 * of 1 m. A strict `>= 1 m` comparison therefore rejects every second cell of
 * the lattice and truncates the crowd pick budget.
 *
 * The margin is 1e-12 m - roughly 30x the worst measured shortfall of
 * 2.99e-14 m - so the EFFECTIVE clearance stays 1 m and the personal-space
 * contract (no two robots closer than 1 m) is never relaxed; it is only
 * expressed against the named margin so the contract and the implementation
 * agree.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M); // 1e-12
 * console.log(
 *   NEATRACING_ROBOT_PERSONAL_SPACE_M -
 *     NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M,
 * ); // 0.999999999999 - the effective clearance
 * ```
 */
export const NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M = 1e-12;

/**
 * Effective personal-space clearance: the nominal 1 m minus the named margin.
 *
 * A robot is dropped when its XZ distance to an already kept robot is BELOW
 * this value, so every retained pair keeps at least
 * `NEATRACING_ROBOT_PERSONAL_SPACE_M` minus
 * `NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M` meters of separation.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M); // 0.999999999999
 * console.log(
 *   NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M <
 *     NEATRACING_ROBOT_PERSONAL_SPACE_M,
 * ); // true - the effective clearance carries the named floating-point margin
 * ```
 */
export const NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M =
  NEATRACING_ROBOT_PERSONAL_SPACE_M - NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M;

/**
 * Any object that carries a world-space `position` triplet.
 *
 * The personal-space filter only reads the XZ pair of the triplet, so any
 * billboard-shaped object with a position satisfies the contract.
 *
 * @example
 * ```ts
 * const billboard: PositionedInWorld = { position: [0, 0, 0] };
 *
 * console.log(billboard.position[2]); // 0 - the Z the filter reads
 * ```
 */
export interface PositionedInWorld {
  readonly position: readonly [number, number, number];
}

/**
 * Deterministically thins a robot billboard list so every retained robot keeps
 * at least {@link NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M} meters of XZ
 * clearance from every other retained robot.
 *
 * Robots are evaluated in input order; the first robot is always kept, and each
 * subsequent robot is kept only when its XZ distance to all previously kept
 * robots is >= the personal-space threshold. The threshold is the nominal
 * {@link NEATRACING_ROBOT_PERSONAL_SPACE_M} minus the named
 * {@link NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M} margin, so the
 * effective clearance is still 1 m while the comparison tolerates the
 * floating-point shortfall of the rotated seat lattice. This reduces density
 * when the track band cannot host the requested number of robots without
 * overlap, while staying pure, deterministic, and allocation-free on the
 * per-frame draw path.
 *
 * @param robots - Candidate robot billboards in deterministic order.
 * @returns Subset of `robots` whose pairwise XZ separation is at least
 * {@link NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M} - the nominal 1 m
 * minus the named 1e-12 m margin. A robot closer than that to an already
 * kept robot is dropped.
 *
 * @example
 * ```ts
 * const candidates = [
 *   { position: [0, 0, 0] },
 *   { position: [2, 0, 0] },
 *   { position: [0.5, 0, 0] },
 * ];
 *
 * const kept = filterRobotsByPersonalSpace(candidates);
 * console.log(kept.length); // 2 - the robot 0.5 m from the first one is dropped
 * console.log(kept[0]?.position); // [0, 0, 0]
 * console.log(kept[1]?.position); // [2, 0, 0]
 * ```
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

      if (Math.hypot(dx, dz) < NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M) {
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

/**
 * Role of a single voxel inside an expanded bleacher prop.
 *
 * The role tells the renderer and the crowd builder what a voxel is FOR:
 * a `seat` carries a spectator, a `walk` voxel is the standing surface of
 * the platform, a `stair` voxel climbs between two levels, and a
 * `guardrail` voxel caps the track-facing edge of a level.
 *
 * @example
 * ```ts
 * const role: BleacherVoxelRole = 'seat';
 *
 * console.log(role === 'seat'); // true
 * ```
 */
export type BleacherVoxelRole = 'seat' | 'walk' | 'stair' | 'guardrail';

/**
 * One voxel emitted by a stepped bleacher prop.
 *
 * A voxel carries BOTH frames of the stand: the world-space `position` the
 * renderer draws, and the local `localU` / `localV` coordinates the crowd
 * builder measures the stand in. The local coordinates are what makes a
 * rotated bleacher measurable - a stand that is turned relative to the
 * world axes still reports the same footprint area in its own frame.
 *
 * @example
 * ```ts
 * const voxel: BleacherVoxel = {
 *   position: [0, 0.8, 0],
 *   color: [0, 255, 255],
 *   role: 'seat',
 *   localU: 3,
 *   localV: 0,
 * };
 *
 * console.log(voxel.role); // 'seat'
 * ```
 */
export interface BleacherVoxel {
  /** World-space voxel position in meters. */
  readonly position: readonly [number, number, number];
  /** 0-255 RGB neon color. */
  readonly color: readonly [number, number, number];
  /** Structural role inside the bleacher geometry. */
  readonly role: BleacherVoxelRole;
  /**
   * Local depth coordinate of the voxel inside the bleacher frame.
   *
   * `u` grows toward the track (the front/guardrail side) and shrinks toward
   * the back (the stair side). The front row of a level is the row with the
   * largest `u` for that level.
   */
  readonly localU: number;
  /**
   * Local width coordinate of the voxel inside the bleacher frame.
   *
   * `v` runs along the track direction across the seating rows.
   */
  readonly localV: number;
}

/**
 * Geometry carried by a bleacher prop.
 *
 * The geometry is the STEP of the stand: seats, walking surface, stairs, and
 * guardrails as individual voxels. A bleacher without geometry is a plain
 * prop, so the geometry-aware helpers fall back to their documented default
 * shapes for it.
 *
 * @example
 * ```ts
 * const geometry: BleacherGeometry = {
 *   voxels: [
 *     {
 *       position: [0, 0.8, 0],
 *       color: [0, 255, 255],
 *       role: 'seat',
 *       localU: 3,
 *       localV: 0,
 *     },
 *   ],
 * };
 *
 * console.log(geometry.voxels.length); // 1
 * ```
 */
export interface BleacherGeometry {
  /** Ordered voxel set describing seats, walking space, stairs, and guardrails. */
  readonly voxels: readonly BleacherVoxel[];
}

/**
 * One TRON-style neon scenery prop.
 *
 * A prop is a POSITION, a COLOR, an ARCHETYPE, and the RING it belongs to.
 * The ring side is what separates the two bands of the right world: the
 * outer ring carries the neon scenery and the spectator bleachers, the
 * inner ring carries the pit props of the mechanic, crew, and team-banner
 * kinds.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * const bleacher: RightWorldProp | undefined = props.find(
 *   (prop) => prop.kind === 'bleacher',
 * );
 *
 * console.log(bleacher?.ringSide); // 'outer'
 * ```
 */
export interface RightWorldProp {
  /** World-space position in meters. */
  readonly position: readonly [number, number, number];
  /** 0-255 RGB neon color. */
  readonly color: readonly [number, number, number];
  /**
   * Prop archetype.
   *
   * The outer ring carries the neon scenery kinds and the spectator
   * bleachers; the inner ring carries the pit people kinds.
   */
  readonly kind:
    | 'pillar'
    | 'gantry'
    | 'sign'
    | 'bleacher'
    | 'mechanic'
    | 'crew'
    | 'team-banner';
  /**
   * Placement band relative to the SEAM edge of the track.
   *
   * `outer` props are the neon scenery and the spectator bleachers, placed
   * farther from the SEAM edge. `inner` props are the pit props - the
   * mechanic, crew, and team-banner kinds - placed closer to the road than
   * the bleachers. The field is kept because other contracts read it.
   */
  readonly ringSide: 'inner' | 'outer';
  /**
   * Optional stepped voxel geometry for spectator bleachers.
   *
   * Carries the seat, walk, stair, and guardrail voxels that describe the
   * prop's shape beyond the bounding box.
   */
  readonly bleacherGeometry?: BleacherGeometry;
}

/**
 * Placement parameters of one emitted right-world prop ring.
 *
 * The plan is the GEOMETRY of the ring, not its content: how many slots the
 * ring is divided into, which archetypes the slots may draw, how far the
 * ring sits beyond the SEAM edge, and which slot indices are reserved as
 * bleachers. The builder fills the plan, and the emitters only read it.
 *
 * @example
 * ```ts
 * const plan: RightWorldRingPlan = {
 *   slotCount: 40,
 *   kinds: ['pillar', 'gantry', 'sign'],
 *   baseOffset: 6,
 *   offsetRange: 4,
 *   reservedBleacherSlots: selectBleacherSlotIndices(40),
 * };
 *
 * console.log(plan.reservedBleacherSlots.length); // 20
 * ```
 */
export interface RightWorldRingPlan {
  /** Number of slots the ring is divided into. */
  readonly slotCount: number;
  /** Candidate archetypes of the ring, in draw order. */
  readonly kinds: readonly RightWorldProp['kind'][];
  /**
   * Base clearance of the ring beyond the SEAM EDGE, in meters.
   *
   * The value is a clearance, NOT a distance from the centreline: the
   * placement measures it against the published edge of the track. It is
   * clamped to {@link NEATRACING_PROP_CLEARANCE_FLOOR_M}, so a plan can
   * never ask for a band that breaks the USER-NAMED floor.
   */
  readonly baseOffset: number;
  /** Range in meters covered by the random clearance draw. */
  readonly offsetRange: number;
  /**
   * Slot indices reserved as bleachers BY CONSTRUCTION.
   *
   * The reserved slots are evenly spaced over the ring, so the bleacher
   * count is a property of the ring plan and never of a random draw.
   */
  readonly reservedBleacherSlots: readonly number[];
}

/**
 * Placement parameters of the inner-ring people band.
 *
 * The plan carries the two offsets of the band and the TRACK SAMPLE of every
 * people slot. The sample indices are the midpoints between the reserved
 * bleacher slots, so the pit band never shares a track position with a
 * bleacher.
 *
 * @example
 * ```ts
 * const plan: RightWorldPeopleRingPlan = {
 *   baseOffset: 3,
 *   offsetRange: 2,
 *   sampleIndices: [4, 9, 14],
 * };
 *
 * console.log(plan.sampleIndices.length); // 3
 * ```
 */
export interface RightWorldPeopleRingPlan {
  /**
   * Base clearance of the band beyond the SEAM EDGE, in meters.
   *
   * The value is a clearance, NOT a distance from the centreline: the
   * placement measures it against the published edge of the track. It is
   * clamped to {@link NEATRACING_PROP_CLEARANCE_FLOOR_M}, so a plan can
   * never ask for a band that breaks the USER-NAMED floor.
   */
  readonly baseOffset: number;
  /** Range in meters covered by the random clearance draw. */
  readonly offsetRange: number;
  /**
   * Track sample index of every people slot, in slot order.
   *
   * The indices are the midpoints between the reserved bleacher slots, so
   * the pit band never shares a track position with a bleacher.
   */
  readonly sampleIndices: readonly number[];
}

/**
 * The SEAM context of one track: the published edge profile and the raw
 * chord segments it was resolved from.
 *
 * The context is the ONE place where the right world reads the track edge.
 * Every clearance the builder measures - and every offset it derives from
 * that measurement - is read through the context, so the props and the road
 * can never disagree about where the edge is.
 */
export interface RightWorldSeamContext {
  /** The published edge profile of the track. */
  readonly profile: TrackEdgeProfile;
  /** The raw closed-loop segments the profile was resolved from. */
  readonly segments: readonly TrackSegment[];
}

/**
 * Resolves the SEAM context of a track from its raw chord segments.
 *
 * The profile is the seam's OWN publication for those segments, so the
 * context carries the same half-widths, corner measures and loop orientation
 * the road's own readers use.
 *
 * @param segments - Ordered closed-loop chord segments of the track.
 * @returns The seam context of the track.
 *
 * @example
 * ```ts
 * const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
 * const seam = resolveRightWorldSeam(spec.segments);
 *
 * console.log(seam.profile.chords.length); // 6 - one published edge per chord
 * ```
 */
export function resolveRightWorldSeam(
  segments: readonly TrackSegment[],
): RightWorldSeamContext {
  const profile = resolveTrackEdgeProfile(segments);

  return { profile, segments: profile.segments };
}

/**
 * Measures the clearance of a world point beyond the SEAM edge, in meters.
 *
 * The measurement uses the road's OWN union reader: the minimum past-edge
 * distance over EVERY published chord of the seam profile. A point that
 * clears the floor through this reader clears it through the nearest-chord
 * reader too, because the union is never larger than the nearest chord's
 * own reading.
 *
 * @param seam - The seam context of the track.
 * @param worldX - World-frame x of the measured point.
 * @param worldZ - World-frame z of the measured point.
 * @returns The signed clearance beyond the seam edge, in meters.
 *
 * @example
 * ```ts
 * const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
 * const seam = resolveRightWorldSeam(spec.segments);
 *
 * // The road centreline sits INSIDE the seam edge.
 * console.log(measureSeamClearanceM(seam, spec.splineSamples[0]!.x, 0) < 0); // true
 * ```
 */
export function measureSeamClearanceM(
  seam: RightWorldSeamContext,
  worldX: number,
  worldZ: number,
): number {
  return classifyTrackSlabSurfaceAt(worldX, worldZ, seam.segments, seam.profile)
    .distancePastEdgeM;
}

/**
 * Resolves the outward offset that puts a prop's PROBE point at the target
 * clearance beyond the SEAM edge.
 *
 * THE SEARCH: the ray starts at the track sample and walks OUTWARD - the
 * side the right world is built on - in steps of
 * {@link NEATRACING_SEAM_CLEARANCE_STEP_M}. The FIRST candidate whose
 * MEASURED clearance reaches the target wins, so the placement is the
 * closest one that clears the floor. The clearance along the ray is NOT
 * assumed to be monotone: the winning candidate is always a MEASURED one,
 * never an interpolated one, and a target the ray never reaches falls back
 * to the candidate with the best measured clearance.
 *
 * THE PROBE POINT: the clearance is measured at the probe - the prop's own
 * anchor pulled `probeInsetM` meters TOWARD the track - so a prop whose
 * body reaches closer to the road than its anchor (a bleacher with its
 * front seat row) is measured by the part that comes closest to the road.
 *
 * @param seam - The seam context of the track.
 * @param sample - Track sample the prop belongs to.
 * @param frame - Spline frame of the sample.
 * @param targetClearanceM - The clearance the probe has to reach, in meters.
 * @param probeInsetM - Distance from the anchor to the probe point, toward
 *   the track, in meters.
 * @returns The outward offset of the prop, in meters.
 *
 * @example
 * ```ts
 * const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
 * const seam = resolveRightWorldSeam(spec.segments);
 * const sample = spec.splineSamples[0]!;
 * const frame = resolveSplineSampleFrame(spec.splineSamples, sample.globalIndex);
 *
 * const offset = resolveSeamClearanceOffsetM(seam, sample, frame, 10, 0);
 * console.log(offset > 0); // true - the sample itself is inside the road
 * ```
 */
export function resolveSeamClearanceOffsetM(
  seam: RightWorldSeamContext,
  sample: SplineSample,
  frame: SplineSampleFrame,
  targetClearanceM: number,
  probeInsetM: number,
): number {
  let bestOffsetM = 0;
  let bestClearanceM = Number.NEGATIVE_INFINITY;

  for (
    let offsetM = 0;
    offsetM <= NEATRACING_SEAM_CLEARANCE_MAX_OFFSET_M;
    offsetM += NEATRACING_SEAM_CLEARANCE_STEP_M
  ) {
    // The outward ray of the sample: the right world is built along the
    // RIGHT normal of the spline frame, and the world frame mirrors the
    // generator plane as `worldZ = -generatorY`.
    const anchorX = sample.x - frame.normalX * offsetM;
    const anchorZ = -(sample.y - frame.normalY * offsetM);
    // The probe point: the anchor pulled toward the track along the LEFT
    // normal, which is the world direction `(normalX, -normalY)`.
    const probeX = anchorX + frame.normalX * probeInsetM;
    const probeZ = anchorZ - frame.normalY * probeInsetM;
    const clearanceM = measureSeamClearanceM(seam, probeX, probeZ);

    if (clearanceM > bestClearanceM) {
      bestClearanceM = clearanceM;
      bestOffsetM = offsetM;
    }

    if (clearanceM >= targetClearanceM) {
      return offsetM;
    }
  }

  return bestOffsetM;
}

/**
 * Emits one ring of right-world props.
 *
 * THE PLACEMENT: every slot is placed at the outward offset whose MEASURED
 * clearance beyond the SEAM edge reaches the clearance band of the slot, so
 * the ring never relies on the retired spline-normal distance model. A
 * bleacher is probed at its front seat row, so the row that carries the
 * seated crowd clears the floor and the stand's own anchor sits further
 * out.
 *
 * DRAW CONTRACT: every slot consumes EXACTLY three draws from `random` - the
 * clearance band, the archetype, and the colour - in that order. A reserved
 * bleacher slot consumes its archetype draw like every other slot and only
 * OVERRIDES the resolved archetype afterwards, so the reservation never
 * shifts the draw sequence and every later prop stays byte-identical to the
 * same draw sequence without the reservation.
 *
 * The function is exported so the draw contract can be exercised with a
 * reserved slot that is NOT the first outer slot. The reachable track
 * geometry reserves evenly spaced slots (see
 * {@link selectBleacherSlotIndices}), and because every reserved slot
 * consumes its archetype draw before the later slots, a skipped archetype
 * draw shifts EVERY later prop deterministically - the defect is directly
 * observable in the reachable space. A reservation further down the ring
 * exposes the same shift at a known interior position of the ring.
 *
 * @param samples - Spline samples of the track the ring is placed along.
 * @param random - Seeded PRNG the ring consumes its draws from.
 * @param plan - Slot count, archetypes, clearances, and reserved bleacher
 *   slots.
 * @param seam - Seam context the clearances are measured against. It is
 *   OPTIONAL: a caller that owns the track segments passes the seam it
 *   resolved for them, and a caller that only owns the spline samples gets
 *   the seam recovered from those samples.
 * @returns The props of the ring, in emission order.
 *
 * @throws Error - If the bleacher geometry and the named
 *   {@link NEATRACING_BLEACHER_LEVEL_COUNT} disagree, with the message
 *   `bleacher level count drift: the geometry defines <N> levels, the named
 *   constant is <M>`. The guard fires while the bleacher geometry of a
 *   reserved slot is built.
 *
 * @example
 * ```ts
 * const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
 *
 * // A tiny linear congruential generator, so the example needs no extra
 * // imports: the ring only ever asks for values in [0, 1).
 * const random = (() => {
 *   let state = 42;
 *   return () => {
 *     state = (state * 1664525 + 1013904223) % 4294967296;
 *     return state / 4294967296;
 *   };
 * })();
 *
 * const props = emitRightWorldRing(
 *   spec.splineSamples,
 *   random,
 *   {
 *     slotCount: 4,
 *     kinds: ['pillar', 'gantry', 'sign'],
 *     baseOffset: 6,
 *     offsetRange: 4,
 *     reservedBleacherSlots: [0, 2],
 *   },
 * );
 *
 * console.log(props[0]?.kind); // 'bleacher' - the reserved slot
 * ```
 */
export function emitRightWorldRing(
  samples: readonly SplineSample[],
  random: () => number,
  plan: RightWorldRingPlan,
  seam: RightWorldSeamContext = recoverRightWorldSeam(samples),
): readonly RightWorldProp[] {
  const props: RightWorldProp[] = [];
  const reservedBleacherSlots = new Set(plan.reservedBleacherSlots);

  for (let index = 0; index < plan.slotCount; index += 1) {
    const sampleIndex = Math.floor((index * samples.length) / plan.slotCount);
    const sample = samples[sampleIndex]!;
    const frame = resolveSplineSampleFrame(samples, sample.globalIndex);

    // The three draws of the slot are consumed in the SAME order as before:
    // the clearance band, the archetype, and the colour.
    const clearanceRoll = random();

    // The reserved slots are bleachers BY CONSTRUCTION, so the scene always
    // carries the same number of spectator stands. The archetype draw is
    // still consumed and only the RESOLVED archetype is overridden
    // afterwards, so the draw sequence - and therefore every later prop -
    // stays byte-identical to the same draw sequence without the
    // reservation.
    const kindRoll = random();
    const kind = reservedBleacherSlots.has(index)
      ? 'bleacher'
      : plan.kinds[Math.floor(kindRoll * plan.kinds.length)]!;
    const colorRoll = random();
    const color = NEON_COLORS[Math.floor(colorRoll * NEON_COLORS.length)]!;

    // The band of the slot is a CLEARANCE BEYOND THE SEAM EDGE, clamped to
    // the USER-NAMED floor: the search returns the closest outward offset
    // that reaches it. A bleacher is probed at its front seat row - the row
    // closest to the track, and the row the seated crowd stands on.
    const targetClearanceM = Math.max(
      NEATRACING_PROP_CLEARANCE_FLOOR_M,
      plan.baseOffset + clearanceRoll * plan.offsetRange,
    );
    const offset = resolveSeamClearanceOffsetM(
      seam,
      sample,
      frame,
      targetClearanceM,
      kind === 'bleacher' ? NEATRACING_BLEACHER_FRONT_SEAT_U_M : 0,
    );
    const generatorX = sample.x - frame.normalX * offset;
    const generatorY = sample.y - frame.normalY * offset;
    const prop: RightWorldProp = {
      position: [generatorX, 0, -generatorY],
      color,
      kind,
      ringSide: 'outer',
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

  return props;
}

/**
 * Emits the inner ring of pit props: the mechanic, crew, and team-banner
 * kinds of the re-populated right world.
 *
 * DRAW CONTRACT: every slot consumes EXACTLY three draws from `random` - the
 * offset, the archetype, and the colour - in that order, mirroring the
 * outer ring. The people kind is assigned BY CONSTRUCTION in a fixed round
 * robin, so the archetype draw is consumed and only the RESOLVED archetype
 * is overridden afterwards: the draw sequence never shifts, and every kind
 * is present at every seed.
 *
 * The three draws of a slot are consumed BEFORE the track sample of the slot
 * is looked up, so a missing track sample can never shift the draw sequence
 * of the following slots.
 *
 * The pit band sits closer to the road EDGE than the bleachers - the gap is
 * measured beyond the SEAM EDGE, so the two bands are compared on the same
 * scale - and its slots sit at the midpoints between the reserved bleacher
 * slots, so the two rings never share a track position. The rings are
 * separated ALONG the track: the distance of a prop from the centerline
 * alone does not order the two bands, because the road is wider on one part
 * of the track than on another.
 *
 * @param samples - Spline samples of the track the ring is placed along.
 * @param random - Seeded PRNG the ring consumes its draws from.
 * @param plan - Clearances and the track sample index of every people slot.
 * @param seam - Seam context the clearances are measured against. It is
 *   OPTIONAL: a caller that owns the track segments passes the seam it
 *   resolved for them, and a caller that only owns the spline samples gets
 *   the seam recovered from those samples.
 * @returns The props of the inner ring, in emission order.
 *
 * @example
 * ```ts
 * const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
 *
 * const random = (() => {
 *   let state = 42;
 *   return () => {
 *     state = (state * 1664525 + 1013904223) % 4294967296;
 *     return state / 4294967296;
 *   };
 * })();
 *
 * const props = emitRightWorldPeopleRing(spec.splineSamples, random, {
 *   baseOffset: 3,
 *   offsetRange: 2,
 *   sampleIndices: [4, 9, 14],
 * });
 *
 * console.log(props.map((prop) => prop.kind)); // ['mechanic', 'crew', 'team-banner']
 * ```
 */
export function emitRightWorldPeopleRing(
  samples: readonly SplineSample[],
  random: () => number,
  plan: RightWorldPeopleRingPlan,
  seam: RightWorldSeamContext = recoverRightWorldSeam(samples),
): readonly RightWorldProp[] {
  const props: RightWorldProp[] = [];

  for (let index = 0; index < plan.sampleIndices.length; index += 1) {
    // The three draws of the slot - offset, archetype, colour - are consumed
    // for EVERY slot and BEFORE the track sample is looked up, so a missing
    // track sample can never shift the draw sequence of the following slots.
    const offsetRoll = random();
    random();
    const colorRoll = random();

    const sample = samples[plan.sampleIndices[index]!];
    if (!sample) {
      continue;
    }
    const frame = resolveSplineSampleFrame(samples, sample.globalIndex);
    // The band of the slot is a CLEARANCE BEYOND THE SEAM EDGE, clamped to
    // the USER-NAMED floor: the search returns the closest outward offset
    // that reaches it.
    const targetClearanceM = Math.max(
      NEATRACING_PROP_CLEARANCE_FLOOR_M,
      plan.baseOffset + offsetRoll * plan.offsetRange,
    );
    const offset = resolveSeamClearanceOffsetM(
      seam,
      sample,
      frame,
      targetClearanceM,
      0,
    );
    const generatorX = sample.x - frame.normalX * offset;
    const generatorY = sample.y - frame.normalY * offset;

    // The people kinds are assigned BY CONSTRUCTION in a fixed round robin,
    // so every kind is present at every seed. The archetype draw is still
    // consumed and only the RESOLVED archetype is overridden afterwards, so
    // the three-draws-per-slot contract holds for the inner ring too.
    const kind =
      INNER_RING_PEOPLE_KINDS[index % INNER_RING_PEOPLE_KINDS.length]!;
    const color = NEON_COLORS[Math.floor(colorRoll * NEON_COLORS.length)]!;

    props.push({
      position: [generatorX, 0, -generatorY],
      color,
      kind,
      ringSide: 'inner',
    });
  }

  return props;
}

/**
 * Resolves the number of outer-ring slots for a track with the given number
 * of spline samples.
 *
 * The slot count is a whole multiple of
 * {@link NEATRACING_RIGHT_WORLD_BLEACHER_COUNT}, so the reserved bleacher
 * slots can be spread evenly over the ring. The multiple is bounded by the
 * density budget - the budget covers the outer ring AND the inner pit band,
 * so the band is reserved first - and by the track density, so a short track
 * never asks for more slots than it has samples to place them on. The floor
 * of two blocks keeps the twenty bleachers on every reachable track.
 *
 * On every REACHABLE track the density term of the rule is ZERO - it sits
 * BELOW the floor, not beside it - and the budget allows four blocks, so
 * the floor of two blocks governs: the resolved count is always forty
 * slots, twenty bleachers and twenty scenery props, plus the ten
 * inner-ring pit props of the band.
 *
 * @param sampleCount - Number of spline samples of the track.
 * @returns The number of outer-ring slots.
 *
 * @example
 * ```ts
 * console.log(resolveOuterRingSlotCount(108)); // 40
 * ```
 */
export function resolveOuterRingSlotCount(sampleCount: number): number {
  const budgetBlocks = Math.floor(
    (NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET -
      NEATRACING_RIGHT_WORLD_PEOPLE_RING_COUNT) /
      NEATRACING_RIGHT_WORLD_BLEACHER_COUNT,
  );
  const densityBlocks = Math.floor(
    Math.floor(sampleCount / NEATRACING_RIGHT_WORLD_SAMPLES_PER_RING_SLOT) /
      NEATRACING_RIGHT_WORLD_BLEACHER_COUNT,
  );
  const blocks = Math.min(
    budgetBlocks,
    Math.max(NEATRACING_RIGHT_WORLD_MIN_RING_BLOCKS, densityBlocks),
  );
  return blocks * NEATRACING_RIGHT_WORLD_BLEACHER_COUNT;
}

/**
 * Selects the outer-ring slot indices reserved as bleachers.
 *
 * The reserved slots are evenly spaced over the ring, so the bleacher count
 * is exactly {@link NEATRACING_RIGHT_WORLD_BLEACHER_COUNT} and the spacing
 * between two bleachers is the same everywhere along the track.
 *
 * PRECONDITION: `slotCount` is a whole multiple of
 * {@link NEATRACING_RIGHT_WORLD_BLEACHER_COUNT}, which
 * {@link resolveOuterRingSlotCount} guarantees. The stride itself is always
 * uniform; for a slot count that is NOT a whole multiple the leftover slots
 * all fall at the end of the ring, so the reserved slots stay inside the
 * ring but no longer cover it symmetrically.
 *
 * @param slotCount - Number of slots of the outer ring.
 * @returns The reserved slot indices, in ascending order.
 *
 * @example
 * ```ts
 * console.log(selectBleacherSlotIndices(40)); // [0, 2, 4, ..., 38]
 * ```
 */
export function selectBleacherSlotIndices(
  slotCount: number,
): readonly number[] {
  const stride = Math.max(
    1,
    Math.floor(slotCount / NEATRACING_RIGHT_WORLD_BLEACHER_COUNT),
  );
  const indices: number[] = [];
  for (
    let index = 0;
    index < NEATRACING_RIGHT_WORLD_BLEACHER_COUNT;
    index += 1
  ) {
    indices.push(index * stride);
  }
  return indices;
}

/**
 * Selects the track sample indices of the inner-ring people props.
 *
 * The people slots sit at the midpoints between the reserved bleacher
 * slots, so the pit band is spread evenly over the track and never shares a
 * track position with a bleacher.
 *
 * @param samples - Spline samples of the track.
 * @param bleacherSlotIndices - Reserved bleacher slots of the outer ring.
 * @param outerSlotCount - Number of slots of the outer ring.
 * @returns The track sample index of every people slot, in slot order.
 *
 * @example
 * ```ts
 * const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
 *
 * const indices = selectPeopleRingSampleIndices(
 *   spec.splineSamples,
 *   selectBleacherSlotIndices(40),
 *   40,
 * );
 *
 * console.log(indices.length); // 10 - one midpoint per bleacher pair
 * ```
 */
export function selectPeopleRingSampleIndices(
  samples: readonly SplineSample[],
  bleacherSlotIndices: readonly number[],
  outerSlotCount: number,
): readonly number[] {
  const indices: number[] = [];
  for (let index = 0; index + 1 < bleacherSlotIndices.length; index += 2) {
    const leftSample = Math.floor(
      (bleacherSlotIndices[index]! * samples.length) / outerSlotCount,
    );
    const rightSample = Math.floor(
      (bleacherSlotIndices[index + 1]! * samples.length) / outerSlotCount,
    );
    indices.push(Math.floor((leftSample + rightSample) / 2));
  }
  return indices;
}

/**
 * Builds the deterministic right-world neon scenery prop list for the given
 * track tuple.
 *
 * The scene carries TWO rings. The OUTER ring holds the neon scenery kinds
 * and exactly {@link NEATRACING_RIGHT_WORLD_BLEACHER_COUNT} bleachers
 * reserved BY CONSTRUCTION at evenly spaced slot indices. The INNER ring
 * holds the pit props - the mechanic, crew, and team-banner kinds - placed
 * closer to the road and at the midpoints between the bleacher blocks.
 *
 * @param input - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.
 * @returns Ordered list of scenery props ready for the right-world render
 *   pass. Props are placed on the right-hand side of the track centerline,
 *   at least {@link NEATRACING_PROP_CLEARANCE_FLOOR_M} beyond the SEAM edge
 *   of the road, and never exceed the density budget.
 *   The bleachers keep their reserved slot order, because every scene
 *   carries the same number of equally usable bleachers.
 *
 * @throws Error - If the bleacher geometry and the named
 *   {@link NEATRACING_BLEACHER_LEVEL_COUNT} disagree, with the message
 *   `bleacher level count drift: the geometry defines <N> levels, the named
 *   constant is <M>`.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * console.log(props.length > 0); // true - a track always carries props
 * console.log(props.filter((prop) => prop.kind === 'bleacher').length); // 20
 * ```
 */
export function buildRightWorldProps(
  input: TrackSlabBuildInput,
): readonly RightWorldProp[] {
  const spec = generateTrack(input);
  const samples = spec.splineSamples;
  if (samples.length === 0) {
    return [];
  }

  // The SEAM context of THIS track: every clearance of the rings is
  // measured against the edge the road's own readers publish, so the props
  // and the road can never disagree about where the edge is.
  const seam = resolveRightWorldSeam(spec.segments);
  const random = createDeterministicRandom(resolveRightWorldSeed(input));
  const outerSlotCount = resolveOuterRingSlotCount(samples.length);
  const bleacherSlotIndices = selectBleacherSlotIndices(outerSlotCount);

  // Outer ring: the neon scenery kinds plus the bleachers reserved BY
  // CONSTRUCTION at evenly spaced slot indices. Every slot consumes exactly
  // three draws, and the reserved slots only override the RESOLVED
  // archetype, so the bleacher count never depends on a random draw.
  const props: RightWorldProp[] = [
    ...emitRightWorldRing(
      samples,
      random,
      {
        slotCount: outerSlotCount,
        kinds: OUTER_RING_KINDS,
        // The band is a CLEARANCE beyond the SEAM edge, and it is RAISED
        // from the 12 m of the Step 14V anchor-level re-pin: the band is
        // measured at the bleacher PROBE ROW, so the parts of a prop that
        // reach closer to the road than the probe - the guardrail voxel
        // corner of a bleacher, and the 4.8 m placement footprint of a
        // seated spectator - were still BELOW the USER-NAMED 10 m floor.
        //
        // BASIS: the P-PC-R row of the Step 14V CLEARANCE GREEN-slice ledger
        // in `plans/neatracing-voxel-racer.plans.md` - the NAMED body-level
        // residual (9 of 400 prop bodies and 371 of 4384 billboard
        // footprints below the floor, worst 7.2994 m). The MEASURED
        // body-level reach of the band is 5.9630 m (a bleacher guardrail
        // corner against its own anchor) plus the 3.3941 m reach of the
        // seated footprint, so the raised band clears the floor at every
        // published seed with measured headroom.
        baseOffset: 15.5,
        offsetRange: 4,
        reservedBleacherSlots: bleacherSlotIndices,
      },
      seam,
    ),
  ];

  // Inner ring: the pit props - the mechanic, crew, and team-banner kinds -
  // closer to the road EDGE than the bleachers and at the midpoints between
  // the bleacher blocks, so the two rings never share a track position.
  props.push(
    ...emitRightWorldPeopleRing(
      samples,
      random,
      {
        // RAISED from the 10.5 m of the Step 14V anchor-level re-pin, for
        // the same P-PC-R body-level basis as the outer ring: the pit band
        // is measured at its own probe, while the mechanic billboard that
        // stands on a pit pedestal reaches 3.3935 m nearer than the
        // pedestal anchor, and the team banner body reaches 1.2035 m
        // nearer than its own anchor.
        //
        // BASIS: the P-PC-R row of the Step 14V CLEARANCE GREEN-slice ledger
        // in `plans/neatracing-voxel-racer.plans.md` (the NAMED body-level
        // residual), and the MEASURED body-level worst of the band raise.
        baseOffset: 14,
        offsetRange: 2,
        sampleIndices: selectPeopleRingSampleIndices(
          samples,
          bleacherSlotIndices,
          outerSlotCount,
        ),
      },
      seam,
    ),
  );

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
      localU: u,
      localV: v,
    });
  };

  // Enlarged flat-tier 3-level stadium for standing robots. Each level
  // contributes four stand-cell rows. Levels are flat horizontal slabs; the
  // whole level steps up toward the back of the bleacher, with a stair column
  // between consecutive levels pointing back (decreasing u). Walking surface
  // covers the full footprint with one free voxel row on each outer side, and
  // the track-facing guardrail band sits at the front free row.
  const LEVELS: readonly { readonly uStart: number; readonly y: number }[] = [
    // Level 1 (lowest, nearest the track) - its front row is the PROBE row
    // of the bleacher, so the depth is the named front-seat depth.
    { uStart: NEATRACING_BLEACHER_FRONT_SEAT_U_M, y: 0.7 },
    // Level 2
    { uStart: -1, y: 1.8 },
    // Level 3 (highest, farthest from the track)
    { uStart: -5, y: 2.9 },
  ];
  // The named level count and the geometry must agree: the per-bleacher seat
  // quota is DERIVED from the level count, so a geometry change that adds or
  // removes a platform fails loudly here instead of silently changing the
  // crowd density of every bleacher of the ring.
  if (LEVELS.length !== NEATRACING_BLEACHER_LEVEL_COUNT) {
    throw new Error(
      `bleacher level count drift: the geometry defines ${LEVELS.length} ` +
        `levels, the named constant is ${NEATRACING_BLEACHER_LEVEL_COUNT}`,
    );
  }
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

/**
 * World-space horizontal half-extent used when drawing a bleacher voxel box.
 *
 * The pair of voxel half-extents - this one and
 * {@link BLEACHER_VOXEL_HALF_HEIGHT_M} - turns the unit cell of a bleacher
 * voxel into a world-space box: the horizontal half-extent is the half of the
 * 1 m cell width, and the vertical half-height is the half of the 0.2 m cell
 * height.
 *
 * @example
 * ```ts
 * console.log(BLEACHER_VOXEL_HALF_EXTENT_M * 2); // 1 - one voxel wide
 * console.log(BLEACHER_VOXEL_HALF_HEIGHT_M * 2); // 0.2 - one voxel tall
 * ```
 */
export const BLEACHER_VOXEL_HALF_EXTENT_M = 0.5;

/**
 * World-space vertical half-height used when drawing a bleacher voxel box.
 *
 * The half-height turns the unit cell of a bleacher voxel into a world-space
 * box of 0.2 m, so the seat surface of a stand is the voxel height plus this
 * half-height.
 *
 * @example
 * ```ts
 * console.log(BLEACHER_VOXEL_HALF_HEIGHT_M * 2); // 0.2 - one voxel tall
 * ```
 */
export const BLEACHER_VOXEL_HALF_HEIGHT_M = 0.1;

/**
 * Returns the top Y of the seat surface at a given world-space XZ location.
 *
 * When `x` and `z` are omitted, this returns the highest seat row in the
 * bleacher (used for rendering bounding-box checks). When provided, it returns
 * the y of the nearest seat voxel so a billboard anchored to a lower level is
 * not hoisted to the top row.
 *
 * @param bleacher - Bleacher prop whose seat voxels are inspected.
 * @param x - Optional world-space X of the query point. It is read only
 *   together with `z`; when both are given, the seat voxel nearest to that
 *   XZ position decides the returned height.
 * @param z - Optional world-space Z of the query point, read only together
 *   with `x`.
 * @returns The world-space Y of the seat surface: the seat-top of the nearest
 *   seat voxel for a query point, the highest seat-top of the bleacher
 *   without one. A bleacher without voxel geometry falls back to
 *   `bleacher.position[1] + 1.5`, and a bleacher whose geometry carries no
 *   seat voxel falls back to `bleacher.position[1] + 3.0`.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * const bleacher = props.find((prop) => prop.kind === 'bleacher');
 * if (bleacher) {
 *   console.log(
 *     getBleacherSeatTopY(bleacher).toFixed(1),
 *   ); // '3.0' - the top seat row
 * }
 * ```
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
      const distance = Math.hypot(x - voxel.position[0], z - voxel.position[2]);
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
 *
 * @param bleacher - Bleacher prop whose walk voxels are inspected.
 * @returns The world-space Y of the topmost walking surface of the bleacher.
 *   A bleacher without voxel geometry falls back to
 *   `bleacher.position[1] + 0.1`, and a bleacher whose geometry carries no
 *   walk voxel falls back to `bleacher.position[1] + 0.2`.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * const bleacher = props.find((prop) => prop.kind === 'bleacher');
 * if (bleacher) {
 *   console.log(
 *     getBleacherWalkSurfaceY(bleacher).toFixed(1),
 *   ); // '0.2' - the walk surface
 * }
 * ```
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
 * enclose all voxels of a bleacher.
 *
 * The box is the world-space AABB of the stand, so it grows with the
 * rotation of the bleacher. It is used for the anchoring capacity and the
 * sprite placement; the crowd-host rule measures the stand in its OWN local
 * frame instead - see {@link getBleacherHostingFootprintArea}.
 *
 * @param bleacher - Bleacher prop whose voxels are measured.
 * @returns The half-width and half-depth in meters of the world-space
 *   axis-aligned box that tightly encloses the bleacher voxels, as
 *   `[halfWidth, halfDepth]`. The box follows the rotation of the stand, so
 *   the values differ from one bleacher to the next. A bleacher without voxel
 *   geometry falls back to `[2, 1]`.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * const bleacher = props.find((prop) => prop.kind === 'bleacher');
 * if (bleacher) {
 *   const [halfWidth, halfDepth] = getBleacherAxisAlignedHalfSize(bleacher);
 *
 *   // The stand is turned against the world axes, so the box is larger
 *   // than the local footprint of the stand.
 *   console.log(halfWidth.toFixed(1)); // '7.6'
 *   console.log(halfDepth.toFixed(1)); // '6.5'
 * }
 * ```
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
 * Returns the horizontal hosting footprint area of a bleacher in square
 * meters, derived from the real voxel geometry instead of the coarse bounding
 * box.
 *
 * The area is measured in the bleacher's OWN local frame - `localU` runs
 * along the track normal and `localV` along the tangent - so it describes the
 * STAND and not the axis-aligned bounding box of the stand in world space.
 * A rotated stand has a bounding box that grows with the rotation, so the
 * world-space box would rank the stands by how they happen to be turned.
 *
 * Every bleacher of the right world is the SAME stand, so every bleacher
 * measures the same hosting area and the named crowd-host rule falls back to
 * its documented tie-break (the lowest prop index, which is the bleacher of
 * the FIRST reserved slot).
 *
 * @param bleacher - Bleacher prop to measure.
 * @returns Footprint area in square meters.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * const host = props.find((prop) => prop.kind === 'bleacher');
 * if (host) {
 *   console.log(getBleacherHostingFootprintArea(host)); // 126
 * }
 * ```
 */
export function getBleacherHostingFootprintArea(
  bleacher: RightWorldProp,
): number {
  const voxels = bleacher.bleacherGeometry?.voxels;
  if (!voxels || voxels.length === 0) {
    return 0;
  }

  let minU = Number.POSITIVE_INFINITY;
  let maxU = Number.NEGATIVE_INFINITY;
  let minV = Number.POSITIVE_INFINITY;
  let maxV = Number.NEGATIVE_INFINITY;
  for (const voxel of voxels) {
    minU = Math.min(minU, voxel.localU);
    maxU = Math.max(maxU, voxel.localU);
    minV = Math.min(minV, voxel.localV);
    maxV = Math.max(maxV, voxel.localV);
  }

  // Every voxel covers one meter of local footprint, so the extent of the
  // stand is the cell span plus the half extent of the outer cells.
  const depth = maxU - minU + 2 * BLEACHER_VOXEL_HALF_EXTENT_M;
  const width = maxV - minV + 2 * BLEACHER_VOXEL_HALF_EXTENT_M;

  return depth * width;
}

/**
 * Tolerance (square meters) used when comparing bleacher hosting footprint
 * areas inside the named crowd-host rule.
 *
 * The tolerance turns the comparison into a TIE test: two bleachers whose
 * areas differ by less than the tolerance share the largest area, so a
 * floating-point wobble in the voxel geometry cannot break the tie.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_BLEACHER_FOOTPRINT_AREA_TOLERANCE_M2); // 1e-9
 * ```
 */
export const NEATRACING_BLEACHER_FOOTPRINT_AREA_TOLERANCE_M2 = 1e-9;

/**
 * Returns every bleacher that ties for the LARGEST hosting footprint area of
 * the scene, in prop order.
 *
 * Only bleachers that carry real voxel geometry take part in the rule; a
 * bleacher without geometry has no measurable hosting footprint. Bleachers
 * whose area is within
 * {@link NEATRACING_BLEACHER_FOOTPRINT_AREA_TOLERANCE_M2} of the largest area
 * are treated as a tie.
 *
 * The right world builds every bleacher as the SAME stand, so the whole
 * bleacher set ties and the returned tie-set is the bleacher set itself, in
 * the order of the reserved slots.
 *
 * The rule is a PURE function of the prop list: it selects on the hosting
 * footprint area alone and does not read where the seated crowd is placed.
 * The seated crowd covers EVERY bleacher of the ring, so a right-world scene
 * resolves to the whole bleacher set.
 *
 * @param props - Scene props in their final prop order.
 * @returns The tie-set of the largest-area bleachers, in prop order. An empty
 *   array when the scene carries no bleacher with voxel geometry.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * const largest = selectMaxAreaBleachers(props);
 * console.log(largest.length); // 20 - every bleacher of the ring is the same stand
 * console.log(largest[0]?.kind); // 'bleacher'
 * ```
 */
export function selectMaxAreaBleachers(
  props: readonly RightWorldProp[],
): readonly RightWorldProp[] {
  const bleachers = props.filter(
    (prop) => prop.kind === 'bleacher' && prop.bleacherGeometry !== undefined,
  );
  if (bleachers.length === 0) {
    return [];
  }

  let largestArea = Number.NEGATIVE_INFINITY;
  for (const bleacher of bleachers) {
    largestArea = Math.max(
      largestArea,
      getBleacherHostingFootprintArea(bleacher),
    );
  }

  return bleachers.filter(
    (bleacher) =>
      Math.abs(getBleacherHostingFootprintArea(bleacher) - largestArea) <=
      NEATRACING_BLEACHER_FOOTPRINT_AREA_TOLERANCE_M2,
  );
}

/**
 * Returns the bleacher selected by the named crowd-host rule: the one with
 * the LARGEST hosting footprint area, with ties broken by the LOWEST prop
 * index.
 *
 * The right world builds every bleacher as the SAME stand, so the rule falls
 * back to the tie-break and selects the bleacher of the FIRST reserved slot.
 *
 * The rule selects on the hosting footprint area alone, so it does not
 * single out the bleacher that carries the crowd: the seated spectators are
 * placed on EVERY bleacher of the ring, and the selection stays a
 * deterministic function of the prop list.
 *
 * @param props - Scene props in their final prop order.
 * @returns The rule-selected crowd host, or `undefined` without bleachers.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * const host = selectBleacherHostProp(props);
 * console.log(host?.kind); // 'bleacher'
 * ```
 */
export function selectBleacherHostProp(
  props: readonly RightWorldProp[],
): RightWorldProp | undefined {
  return selectMaxAreaBleachers(props)[0];
}

/**
 * Number of seated spectators placed on EVERY bleacher level.
 *
 * The constant is the seat density of ONE level of a bleacher: the anchors
 * of a level are the evenly spread subset of that level's front-row cell
 * lattice, so the seated crowd covers the FULL platform length instead of
 * bunching at one end of the row. Every bleacher of the right world carries
 * the same number of levels, so the constant also fixes the per-bleacher
 * anchor count.
 *
 * @example
 * ```ts
 * console.log(
 *   NEATRACING_BLEACHER_SEATS_PER_LEVEL * NEATRACING_BLEACHER_LEVEL_COUNT,
 * ); // 12 - the seat anchors of one bleacher
 *
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * const bleacher = props.find((prop) => prop.kind === 'bleacher');
 * if (bleacher) {
 *   console.log(getBleacherSeatAnchors(bleacher).length); // 12
 * }
 * ```
 */
export const NEATRACING_BLEACHER_SEATS_PER_LEVEL = 4;

/**
 * Number of flat seating levels of a bleacher.
 *
 * The constant is the level count of the stock bleacher stand: the levels
 * are ordered by ascending seat height, so level 0 is the lowest platform
 * (the one nearest the track). The per-bleacher seat quota is DERIVED from
 * this count and from {@link NEATRACING_BLEACHER_SEATS_PER_LEVEL}, so the
 * crowd density follows the geometry instead of a hand-written literal.
 *
 * @example
 * ```ts
 * console.log(NEATRACING_BLEACHER_LEVEL_COUNT); // 3
 * console.log(
 *   NEATRACING_BLEACHER_SEATS_PER_LEVEL * NEATRACING_BLEACHER_LEVEL_COUNT,
 * ); // 12 - the seat anchors of one bleacher
 * ```
 */
export const NEATRACING_BLEACHER_LEVEL_COUNT = 3;

/**
 * Selects the evenly spaced subset of the cells of one row.
 *
 * The selection spreads the requested seats over the FULL row: the first
 * selected cell is the FIRST cell of the row and the last selected cell is
 * the LAST cell of the row, and the cells in between are spread evenly. The
 * effective seat count never exceeds the number of available cells, so a row
 * that is shorter than the requested seat count yields every cell of the row
 * instead of a duplicated cell. The returned indices are ascending and
 * distinct, so a caller can map them straight onto the cells of the row.
 *
 * @param cellCount - Number of cells the row carries.
 * @param seatCount - Number of seats the caller wants to place on the row.
 * @returns The selected cell indices, ascending and distinct. A
 *   non-positive seat count selects nothing.
 *
 * @example
 * ```ts
 * console.log(selectEvenlySpacedCellIndices(7, 4)); // [0, 2, 4, 6]
 * console.log(selectEvenlySpacedCellIndices(3, 4)); // [0, 1, 2]
 * ```
 */
function selectEvenlySpacedCellIndices(
  cellCount: number,
  seatCount: number,
): number[] {
  if (cellCount <= 0) {
    return [];
  }

  const effectiveSeatCount = Math.min(seatCount, cellCount);

  if (effectiveSeatCount <= 0) {
    return [];
  }

  if (effectiveSeatCount === 1) {
    return [0];
  }

  const lastSeatIndex = effectiveSeatCount - 1;
  const lastCellIndex = cellCount - 1;
  const selectedIndices: number[] = [];

  for (let seatIndex = 0; seatIndex < effectiveSeatCount; seatIndex += 1) {
    const cellIndex = Math.round((seatIndex * lastCellIndex) / lastSeatIndex);
    const previousIndex = selectedIndices[selectedIndices.length - 1];

    if (previousIndex === cellIndex) {
      continue;
    }

    selectedIndices.push(cellIndex);
  }

  return selectedIndices;
}

/**
 * Returns the world-space anchors of the evenly spread front seat row of
 * every bleacher level, emitted level-major in ascending platform order.
 *
 * The candidate set is the front-most stand row of each level of the bleacher
 * - the row closest to the track - so a bleacher offers ONE standing row per
 * level rather than every row of every level. Levels and rows are both read
 * from the bleacher's own voxel geometry, so the helper stays generic: any
 * number of levels and any number of rows per level is handled, and the level
 * heights are taken from the geometry instead of being assumed.
 *
 * The anchors of a level are the EVENLY SPREAD subset of that level's front
 * row: exactly {@link NEATRACING_BLEACHER_SEATS_PER_LEVEL} anchors per level,
 * the first one on the first cell of the row and the last one on the last
 * cell, so the seated crowd covers the FULL platform length instead of
 * bunching at one end. The anchors are emitted LEVEL-MAJOR in ascending
 * platform order - the anchors of the lowest platform first, then the ones of
 * the next platform, and so on - so the stand is walked from the bottom up.
 * The returned y coordinate is the rendered seat-top surface for that cell;
 * callers that place billboards should add their own surface clearance
 * epsilon.
 *
 * @param bleacher - Bleacher prop whose seat voxels are inspected.
 * @returns The evenly spread front-row seat anchors, level-major in
 *   ascending platform order.
 *
 * @example
 * ```ts
 * const props = buildRightWorldProps({
 *   seed: 42,
 *   layoutVersion: 1,
 *   sizeBucket: 'large',
 * });
 *
 * const host = props.find((prop) => prop.kind === 'bleacher');
 * if (host) {
 *   const anchors = getBleacherSeatAnchors(host);
 *   console.log(anchors.length); // 12 - four cells per level, three levels
 *   // The three levels sit at three distinct heights.
 *   console.log(new Set(anchors.map((anchor) => anchor[1])).size); // 3
 *   console.log(anchors[0]?.[1]?.toFixed(1)); // '0.8' - the seat-top of level one
 * }
 * ```
 */
export function getBleacherSeatAnchors(
  bleacher: RightWorldProp,
): readonly [number, number, number][] {
  const voxels = bleacher.bleacherGeometry?.voxels;
  if (!voxels) {
    return [];
  }

  // Group the stand cells by level (every flat tier shares one world height)
  // and, inside a level, by row (the local depth coordinate).
  const levels = new Map<number, Map<number, BleacherVoxel[]>>();

  for (const voxel of voxels) {
    if (voxel.role !== 'seat') {
      continue;
    }

    const levelKey = voxel.position[1];
    const rows = levels.get(levelKey) ?? new Map<number, BleacherVoxel[]>();
    const row = rows.get(voxel.localU) ?? [];
    row.push(voxel);
    rows.set(voxel.localU, row);
    levels.set(levelKey, rows);
  }

  // Level-major emission: the anchors of the lowest platform come first,
  // then the anchors of the next platform, and so on.
  const anchors: [number, number, number][] = [];
  const levelKeys = [...levels.keys()].sort((left, right) => left - right);

  for (const levelKey of levelKeys) {
    const rows = levels.get(levelKey)!;
    let frontU = Number.NEGATIVE_INFINITY;
    for (const localU of rows.keys()) {
      if (localU > frontU) {
        frontU = localU;
      }
    }

    // Front row of the level: the row closest to the track (largest local u),
    // ordered by ascending local width.
    const frontRow = [...(rows.get(frontU) ?? [])].sort(
      (left, right) => left.localV - right.localV,
    );
    const selectedCellIndices = selectEvenlySpacedCellIndices(
      frontRow.length,
      NEATRACING_BLEACHER_SEATS_PER_LEVEL,
    );

    for (const cellIndex of selectedCellIndices) {
      const voxel = frontRow[cellIndex];
      if (!voxel) {
        continue;
      }

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
 * Recovers the SEAM context of a track from its spline samples.
 *
 * THE RECOVERY IS EXACT: the FIRST sample of every segment strip carries
 * the segment's own start point and the segment's own width, so the closed
 * loop of chords can be rebuilt from the samples themselves - the start of
 * a chord is the start point of its segment strip, and its end is the start
 * point of the next strip. The recovered context is the DEFAULT seam of the
 * emitters, so a caller that only owns the spline samples still measures
 * the clearances against the published edge; a caller that owns the track
 * segments passes the seam it resolved for them instead.
 *
 * @param samples - Spline samples of the track.
 * @returns The seam context recovered from the samples.
 */
function recoverRightWorldSeam(
  samples: readonly SplineSample[],
): RightWorldSeamContext {
  const segmentStarts: {
    readonly x: number;
    readonly y: number;
    readonly width: number;
  }[] = [];

  for (const sample of samples) {
    if (sample.sampleIndexWithinSegment === 0) {
      segmentStarts.push({
        x: sample.x,
        y: sample.y,
        width: sample.width,
      });
    }
  }

  const segments: TrackSegment[] = segmentStarts.map((start, index) => {
    const next = segmentStarts[(index + 1) % segmentStarts.length]!;

    return {
      startX: start.x,
      startY: start.y,
      endX: next.x,
      endY: next.y,
      width: start.width,
    };
  });

  return resolveRightWorldSeam(segments);
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
