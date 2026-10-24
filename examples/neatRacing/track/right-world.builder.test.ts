/**
 * Red contract tests for the neatRacing right-world TRON-style neon scenery
 * prop builder (slice P1-right-world-props, Step 14D, AC-048).
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - Exports a pure builder {@link buildRightWorldProps} taking the same
 *   `{ seed, layoutVersion, sizeBucket }` tuple used by the track slab builder.
 * - Returns a non-empty list of {@link RightWorldProp} values while honoring
 *   the configured density budget.
 * - Every prop is placed on the right-hand side of the track centerline
 *   (negative left normal in generator space) and outside the road + curb
 *   ribbon.
 * - The builder is deterministic: two calls with the same tuple produce
 *   byte-identical output on the same runtime.
 *
 * STEP 14K RED CONTRACTS (slice P1-crowd2-red-contracts, AC-072 + AC-073):
 * - RED-A: {@link getBleacherSeatAnchors} returns EXACTLY four seat anchors
 *   on EVERY bleacher level (three levels x four = twelve anchors for the
 *   stock bleacher), the four anchors of a level are the evenly spaced
 *   subset of that level's front-row cell lattice (the first anchor on the
 *   first cell, the last on the last cell, equal consecutive gaps within
 *   1e-9 m), and the anchors are emitted level-major in ascending platform
 *   order.
 * - RED-B: the built right world carries EXACTLY twenty bleacher props for
 *   every fixture seed and in every reachable track configuration, and the
 *   count comes from the named constant
 *   `NEATRACING_RIGHT_WORLD_BLEACHER_COUNT` - the bleacher slots are
 *   reserved BY CONSTRUCTION, so the retired random 1-in-4 archetype draw
 *   never produces a bleacher. No two bleachers overlap in world space, and
 *   the mechanic/crew/team-banner prop kinds are re-authored on the INNER
 *   ring, closer to the road than the bleachers.
 */

import { describe, expect, it } from '@jest/globals';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';
import {
  resolveSplineSampleFrame,
  type SplineSampleFrame,
} from '../../racing_curriculum/track/track.spline.utils';
import {
  NEATRACING_RIGHT_WORLD_BLEACHER_COUNT,
  NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_DEPTH_M,
  NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_WIDTH_M,
  NEATRACING_RIGHT_WORLD_BLEACHER_MIN_SEPARATION_M,
  NEATRACING_RIGHT_WORLD_MIN_RING_BLOCKS,
  NEATRACING_RIGHT_WORLD_PEOPLE_RING_COUNT,
  NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET,
  NEATRACING_RIGHT_WORLD_SAMPLES_PER_RING_SLOT,
  NEATRACING_ROBOT_PERSONAL_SPACE_M,
  buildRightWorldProps,
  emitRightWorldPeopleRing,
  emitRightWorldRing,
  filterRobotsByPersonalSpace,
  getBleacherHostingFootprintArea,
  getBleacherSeatAnchors,
  resolveOuterRingSlotCount,
  selectBleacherHostProp,
  selectBleacherSlotIndices,
  selectPeopleRingSampleIndices,
  type RightWorldProp,
} from './right-world.builder';
import * as rightWorldBuilderModule from './right-world.builder';
import { resolvePropHalfSize } from './right-world.prop-sizes';
import type { TrackSlabBuildInput } from './track.slabs';

/** Canonical determinism tuple reused from the track slab builder. */
const FIXTURE_TRACK_BUILD_INPUT = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'small',
} as const;

/**
 * Tolerance (meters) for the projection of a bleacher voxel back into the
 * local bleacher frame.
 *
 * The world-space voxel position is built from the local frame and the test
 * projects it back, so the round trip keeps a double-precision residue of
 * ~1e-15 m. The tolerance covers that residue only.
 */
const NEATRACING_BLEACHER_FRAME_TOLERANCE_M = 1e-9;

/**
 * Number of seated spectators every bleacher level has to offer (RED-A,
 * AC-072).
 *
 * The four anchors of a level are the evenly spaced subset of that level's
 * front-row cell lattice, so the stock three-level bleacher carries
 * 3 x 4 = 12 seat anchors.
 */
const NEATRACING_BLEACHER_SEATS_PER_LEVEL = 4;

/**
 * Number of flat seating levels of the stock bleacher (RED-A, AC-072).
 *
 * The levels are ordered by ascending seat height, so index 0 is the lowest
 * platform (the one nearest the track).
 */
const NEATRACING_BLEACHER_LEVEL_COUNT = 3;

/**
 * Number of bleacher props the right world has to carry (RED-B, AC-073).
 *
 * The bleacher slots are reserved BY CONSTRUCTION at evenly spaced
 * outer-ring slot indices, so the count never depends on the seed.
 */
const NEATRACING_RIGHT_WORLD_BLEACHER_COUNT_EXPECTED = 20;

/**
 * Deterministic seed sweep for the crowd-host contracts (AC-070).
 *
 * Mirrors the fixture seed set of the pass-side AC-070 contracts so the
 * builder and the pass are checked against the same seeds.
 */
const NEATRACING_CROWD_FIXTURE_SEEDS: readonly number[] = [
  0, 7, 42, 61, 123, 1337, 2777, 4242,
];

/** Determinism tuple fields shared by every seed of the crowd sweep. */
const NEATRACING_CROWD_BUILD_INPUT = {
  layoutVersion: 1,
  sizeBucket: 'large',
} as const;

/**
 * Determinism tuples used by the RNG-stream preservation contract (B5).
 *
 * The stream replay is the falsifiable form of the B5 mandate, so it runs
 * over more than one seed and over every size bucket. The tuple set also
 * covers all three REACHABLE ring configurations - 108/10/40/20,
 * 126/10/40/20, and 144/10/40/20 (sample count / inner pit props / outer
 * slots / bleachers) - so the reachable-configuration tripwire sees every
 * case the current track geometry can produce.
 */
const NEATRACING_STREAM_FIXTURE_INPUTS: readonly TrackSlabBuildInput[] = [
  { seed: 42, layoutVersion: 1, sizeBucket: 'small' },
  { seed: 42, layoutVersion: 1, sizeBucket: 'large' },
  { seed: 0, layoutVersion: 1, sizeBucket: 'large' },
  { seed: 1337, layoutVersion: 1, sizeBucket: 'small' },
  { seed: 7, layoutVersion: 1, sizeBucket: 'medium' },
];

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

/** Role of a single voxel inside an expanded bleacher prop. */
type BleacherVoxelRole = 'seat' | 'walk' | 'stair' | 'guardrail';

/** One voxel emitted by a stepped bleacher prop. */
interface BleacherVoxel {
  readonly position: readonly [number, number, number];
  readonly color: readonly [number, number, number];
  readonly role: BleacherVoxelRole;
  /**
   * Local depth coordinate of the voxel inside the bleacher frame.
   *
   * `u` grows toward the track, so the front row of a level is the row with
   * the largest `u`.
   */
  readonly localU: number;
  /** Local width coordinate of the voxel inside the bleacher frame. */
  readonly localV: number;
}

/** Geometry carried by a bleacher prop after P1-bleacher-shape. */
interface BleacherGeometry {
  readonly voxels: readonly BleacherVoxel[];
}

/** Filters the list down to props whose archetype is a spectator bleacher. */
function filterBleacherProps(
  props: readonly RightWorldProp[],
): readonly RightWorldProp[] {
  return props.filter(
    (prop) => (prop as unknown as { kind: string }).kind === 'bleacher',
  );
}

/**
 * Test-local mirror of the builder's TRON neon palette.
 *
 * The RNG-stream contract pins the emitted colour of every prop, so the
 * palette has to be known to the test. A palette change in the builder
 * surfaces as a loud failure of the stream test instead of a silent drift.
 */
const NEATRACING_TEST_NEON_COLORS: readonly (readonly [number, number, number])[] =
  [
    [0, 255, 255],
    [255, 0, 128],
    [128, 0, 255],
    [0, 255, 128],
  ];

/**
 * Test-local mirror of the builder's outer-ring archetype list.
 *
 * The bleacher archetype is NOT part of the list any more: the bleacher
 * slots are reserved BY CONSTRUCTION, so the retired random 1-in-4 bleacher
 * draw is gone from the stream.
 */
const NEATRACING_TEST_OUTER_RING_KINDS: readonly RightWorldProp['kind'][] = [
  'pillar',
  'gantry',
  'sign',
];

/**
 * Test-local mirror of the builder's inner-ring people kinds.
 *
 * The kinds are assigned BY CONSTRUCTION in a fixed round robin, so the
 * mirror is a list rather than a draw table.
 */
const NEATRACING_TEST_INNER_RING_PEOPLE_KINDS: readonly string[] = [
  'mechanic',
  'crew',
  'team-banner',
];

/**
 * Test-local mirror of the builder's inner-ring pit band size.
 *
 * The band places one people prop at every midpoint between two reserved
 * bleacher slots, so it holds one prop per bleacher PAIR - half the bleacher
 * count. The literal states the band size the contract expects, and the
 * contract compares it against the builder's own constant, so the two
 * independent declarations have to agree.
 */
const NEATRACING_TEST_PEOPLE_RING_COUNT = 10;

/**
 * Replays the builder's seed derivation for a determinism tuple.
 *
 * @param input - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.
 * @returns The unsigned 32-bit seed the builder starts its PRNG from.
 */
function replayRightWorldSeed(input: TrackSlabBuildInput): number {
  let combinedSeed = input.seed ^ (input.layoutVersion * 0x9e3779b9);

  for (const character of input.sizeBucket) {
    combinedSeed = Math.imul(combinedSeed ^ character.charCodeAt(0), 16777619);
  }

  return ((combinedSeed >>> 0) ^ 0x7269_6768) >>> 0;
}

/**
 * Replays the builder's xorshift32 PRNG.
 *
 * @param initialSeed - Unsigned 32-bit seed.
 * @returns Stable pseudo-random number generator in the range [0, 1).
 */
function replayDeterministicRandom(initialSeed: number): () => number {
  let state = initialSeed === 0 ? 0x6d2b79f5 : initialSeed >>> 0;

  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;

    return (state >>> 0) / 0x1_0000_0000;
  };
}

/** One prop of the replayed builder emission. */
interface ReplayedProp {
  readonly kind: string;
  readonly x: number;
  readonly z: number;
  readonly color: readonly [number, number, number];
}

/**
 * Placement parameters mirrored by the test-local ring replay.
 *
 * The mirror exists so the replay can also model the DEFECT the B5 contract
 * forbids: a slot whose archetype draw is skipped. That negative control
 * proves the stream assertions discriminate a skipped draw instead of
 * passing vacuously.
 */
interface ReplayedRingPlan {
  /** Number of slots the ring is divided into. */
  readonly slotCount: number;
  /** Candidate archetypes of the ring, in draw order. */
  readonly kinds: readonly string[];
  /** Distance in meters from the road edge to the ring's base offset. */
  readonly baseOffset: number;
  /** Range in meters covered by the random offset draw. */
  readonly offsetRange: number;
  /**
   * Slot indices reserved as bleachers BY CONSTRUCTION.
   *
   * The reserved slots are evenly spaced over the ring, so the bleacher
   * count is a property of the plan and never of a random draw.
   */
  readonly reservedBleacherSlots: readonly number[];
  /**
   * Slot index whose archetype draw is SKIPPED, used by the negative control
   * only. Defaults to `undefined`, which consumes every draw.
   */
  readonly skipArchetypeDrawAt?: number;
}

/**
 * Replays one ring of the builder emission.
 *
 * The replay mirrors the production draw order EXACTLY: every slot consumes
 * one offset draw, one kind draw, and one colour draw - including the
 * reserved bleacher slots, whose kind is overridden AFTER their draw was
 * consumed. That is the B5 contract: the reservation must not shift the
 * stream, so every later prop stays byte-identical to the pre-reservation
 * emission.
 *
 * @param samples - Spline samples of the track the ring is placed along.
 * @param random - Seeded PRNG the replay consumes its draws from.
 * @param plan - Slot count, archetypes, offsets, and reserved slots.
 * @returns The props the production ring emitter must have emitted, in
 *   emission order.
 */
function replayRingProps(
  samples: readonly SplineSample[],
  random: () => number,
  plan: ReplayedRingPlan,
): readonly ReplayedProp[] {
  const replayed: ReplayedProp[] = [];
  const reservedBleacherSlots = new Set(plan.reservedBleacherSlots);

  for (let index = 0; index < plan.slotCount; index += 1) {
    const sampleIndex = Math.floor((index * samples.length) / plan.slotCount);
    const sample = samples[sampleIndex]!;
    const frame = resolveSplineSampleFrame(samples, sample.globalIndex);
    const offset =
      sample.width / 2 + plan.baseOffset + random() * plan.offsetRange;
    const generatorX = sample.x - frame.normalX * offset;
    const generatorY = sample.y - frame.normalY * offset;
    const kindRoll = index === plan.skipArchetypeDrawAt ? 0 : random();
    const kind = reservedBleacherSlots.has(index)
      ? 'bleacher'
      : plan.kinds[Math.floor(kindRoll * plan.kinds.length)]!;
    const color =
      NEATRACING_TEST_NEON_COLORS[
        Math.floor(random() * NEATRACING_TEST_NEON_COLORS.length)
      ]!;

    replayed.push({ kind, x: generatorX, z: -generatorY, color });
  }

  return replayed;
}

/**
 * Replays the inner-ring people band of the builder emission.
 *
 * The mirror consumes the same three draws per slot as the outer ring - the
 * offset, the archetype, and the colour - and overrides the RESOLVED
 * archetype with the BY CONSTRUCTION round-robin people kind, so the stream
 * of the outer ring is never shifted by the inner band.
 *
 * @param samples - Spline samples of the track the band is placed along.
 * @param random - Seeded PRNG the replay consumes its draws from.
 * @param plan - Offsets and the track sample index of every people slot.
 * @returns The props the production inner-ring emitter must have emitted,
 *   in emission order.
 */
function replayPeopleRingProps(
  samples: readonly SplineSample[],
  random: () => number,
  plan: {
    readonly baseOffset: number;
    readonly offsetRange: number;
    readonly sampleIndices: readonly number[];
  },
): readonly ReplayedProp[] {
  const replayed: ReplayedProp[] = [];

  for (let index = 0; index < plan.sampleIndices.length; index += 1) {
    const sample = samples[plan.sampleIndices[index]!]!;
    const frame = resolveSplineSampleFrame(samples, sample.globalIndex);
    const offset =
      sample.width / 2 + plan.baseOffset + random() * plan.offsetRange;
    const generatorX = sample.x - frame.normalX * offset;
    const generatorY = sample.y - frame.normalY * offset;
    // The archetype draw is consumed and the resolved archetype is
    // overridden by the BY CONSTRUCTION round-robin people kind.
    random();
    const kind =
      NEATRACING_TEST_INNER_RING_PEOPLE_KINDS[
        index % NEATRACING_TEST_INNER_RING_PEOPLE_KINDS.length
      ]!;
    const color =
      NEATRACING_TEST_NEON_COLORS[
        Math.floor(random() * NEATRACING_TEST_NEON_COLORS.length)
      ]!;

    replayed.push({ kind, x: generatorX, z: -generatorY, color });
  }

  return replayed;
}

/**
 * Test-local mirror of the builder's outer-ring slot-count rule.
 *
 * The slot count is a whole multiple of the bleacher count, so the reserved
 * bleacher slots stay evenly spread. The multiple is bounded by the density
 * budget - the budget covers the outer ring AND the inner pit band, so the
 * band is reserved first - and by the track density, and the floor of two
 * blocks keeps the twenty bleachers on every reachable track.
 *
 * @param sampleCount - Number of spline samples of the track.
 * @returns The number of outer-ring slots the builder must use.
 */
function replayOuterRingSlotCount(sampleCount: number): number {
  const budgetBlocks = Math.floor(
    (NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET -
      NEATRACING_TEST_PEOPLE_RING_COUNT) /
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
 * Test-local mirror of the builder's reserved bleacher slot rule.
 *
 * @param slotCount - Number of slots of the outer ring.
 * @returns The reserved slot indices the builder must use.
 */
function replayBleacherSlotIndices(slotCount: number): readonly number[] {
  const stride = Math.max(
    1,
    Math.floor(slotCount / NEATRACING_RIGHT_WORLD_BLEACHER_COUNT),
  );

  return Array.from(
    { length: NEATRACING_RIGHT_WORLD_BLEACHER_COUNT },
    (_unused, index) => index * stride,
  );
}

/**
 * Test-local mirror of the builder's inner-band sample rule.
 *
 * The people slots sit at the midpoints between the reserved bleacher slots,
 * so the pit band never shares a track position with a bleacher.
 *
 * @param samples - Spline samples of the track.
 * @param bleacherSlotIndices - Reserved bleacher slots of the outer ring.
 * @param outerSlotCount - Number of slots of the outer ring.
 * @returns The track sample index of every people slot the builder must use.
 */
function replayPeopleRingSampleIndices(
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
 * Replays the whole builder emission from the determinism tuple.
 *
 * The emission carries TWO rings: the outer ring with the evenly spaced
 * reserved bleacher slots, and the inner people band placed at the
 * midpoints between the bleacher blocks. The replay mirrors both, including
 * the slot-count rule and the reserved-slot rule, so the stream contract
 * stays falsifiable.
 *
 * @param input - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.
 * @returns The props the production builder must have emitted, in emission
 *   order.
 */
function replayEmittedProps(
  input: TrackSlabBuildInput,
): readonly ReplayedProp[] {
  const spec = generateTrack(input);
  const samples = spec.splineSamples;
  const random = replayDeterministicRandom(replayRightWorldSeed(input));
  const outerSlotCount = replayOuterRingSlotCount(samples.length);
  const bleacherSlotIndices = replayBleacherSlotIndices(outerSlotCount);

  const outer = replayRingProps(samples, random, {
    slotCount: outerSlotCount,
    kinds: NEATRACING_TEST_OUTER_RING_KINDS,
    baseOffset: 6,
    offsetRange: 4,
    reservedBleacherSlots: bleacherSlotIndices,
  });

  const people = replayPeopleRingProps(samples, random, {
    baseOffset: 3,
    offsetRange: 2,
    sampleIndices: replayPeopleRingSampleIndices(
      samples,
      bleacherSlotIndices,
      outerSlotCount,
    ),
  });

  return [...outer, ...people];
}

/**
 * Builds the comparable key of a prop: archetype, world position, colour.
 *
 * @param prop - Prop to describe.
 * @returns A stable string key.
 */
function propStreamKey(prop: {
  readonly kind: string;
  readonly x: number;
  readonly z: number;
  readonly color: readonly [number, number, number];
}): string {
  return `${prop.kind}|${prop.x.toFixed(9)}|${prop.z.toFixed(9)}|${prop.color.join(',')}`;
}

/**
 * Builds the comparable stream keys of a production prop list.
 *
 * @param props - Production props to describe.
 * @returns One stable stream key per prop, in emission order.
 */
function propStreamKeys(props: readonly RightWorldProp[]): readonly string[] {
  return props.map((prop) =>
    propStreamKey({
      kind: prop.kind,
      x: prop.position[0],
      z: prop.position[2],
      color: prop.color,
    }),
  );
}

/** One reachable ring configuration derived from the production emission. */
interface ReachableRingConfiguration {
  /** Number of spline samples of the track. */
  readonly sampleCount: number;
  /** Number of emitted inner-ring props (the pit people band). */
  readonly innerPropCount: number;
  /** Number of emitted outer-ring props. */
  readonly outerPropCount: number;
  /** Number of bleacher props carried by the outer ring. */
  readonly bleacherCount: number;
}

/**
 * Derives the reachable ring configurations from the PRODUCTION emission.
 *
 * The prop counts are read back from the emitted props instead of being
 * mirrored from the builder source, so a silent change to the ring split, to
 * the density budget, or to the track generator is caught by the reachable
 * contracts instead of being encoded twice.
 *
 * @param inputs - Determinism tuples to derive the configurations from.
 * @returns The distinct reachable configurations, ordered by sample count.
 */
function deriveReachableRingConfigurations(
  inputs: readonly TrackSlabBuildInput[],
): readonly ReachableRingConfiguration[] {
  const byKey = new Map<string, ReachableRingConfiguration>();

  for (const input of inputs) {
    const sampleCount = generateTrack(input).splineSamples.length;
    const props = buildRightWorldProps(input);
    const innerPropCount = props.filter(
      (prop) => (prop as unknown as { ringSide: string }).ringSide === 'inner',
    ).length;
    const outerPropCount = props.filter(
      (prop) => (prop as unknown as { ringSide: string }).ringSide === 'outer',
    ).length;
    const bleacherCount = filterBleacherProps(props).length;

    byKey.set(
      `${sampleCount}/${innerPropCount}/${outerPropCount}/${bleacherCount}`,
      {
        sampleCount,
        innerPropCount,
        outerPropCount,
        bleacherCount,
      },
    );
  }

  return [...byKey.values()].toSorted(
    (left, right) => left.sampleCount - right.sampleCount,
  );
}

/** Reads the optional stepped geometry from a bleacher prop. */
function getBleacherGeometry(
  prop: RightWorldProp,
): BleacherGeometry | undefined {
  return (prop as unknown as { bleacherGeometry?: BleacherGeometry })
    .bleacherGeometry;
}

/**
 * Computes the world-space centre of a bleacher footprint.
 *
 * The separation floor of the bleachers is derived from the footprint
 * CENTRES - the Minkowski argument bounds the overlap of two rectangles by
 * the distance between their centres - so the separation contract has to
 * measure the footprint centre and not the prop anchor. The anchor sits at
 * the track-facing guardrail row of the stand, so the footprint centre lies
 * 2.5 m outward of it.
 *
 * The centre is read back from the EMITTED geometry: a bleacher footprint is
 * a rectangle, and a rectangle is centrally symmetric, so the centre of its
 * world-space bounding box is the centre of the rectangle itself. The
 * helper therefore needs no track frame and makes no assumption about the
 * shape of the stand.
 *
 * @param prop - Bleacher prop to measure.
 * @returns The world-space footprint centre as `[x, z]`.
 */
function bleacherFootprintCentre(
  prop: RightWorldProp,
): readonly [number, number] {
  const voxels = getBleacherGeometry(prop)?.voxels;
  if (!voxels || voxels.length === 0) {
    throw new Error('bleacher prop carries no bleacher geometry');
  }

  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (const voxel of voxels) {
    minX = Math.min(minX, voxel.position[0]);
    maxX = Math.max(maxX, voxel.position[0]);
    minZ = Math.min(minZ, voxel.position[2]);
    maxZ = Math.max(maxZ, voxel.position[2]);
  }

  return [(minX + maxX) / 2, (minZ + maxZ) / 2];
}

/**
 * Computes the signed lateral distance of `position` from the road centerline
 * using the nearest spline sample frame. Negative values are on the right-hand
 * side of the track (outer ring), matching the convention in the placement
 * tests above.
 */
function computeLateral(
  sample: SplineSample,
  frame: SplineSampleFrame,
  position: readonly [number, number, number],
): number {
  const generatorY = sceneZToGeneratorY(position[2]);
  const deltaX = position[0] - sample.x;
  const deltaY = generatorY - sample.y;

  return deltaX * frame.normalX + deltaY * frame.normalY;
}

/**
 * Computes how far a prop sits BEYOND the road + curb ribbon.
 *
 * The offset is the lateral distance of the prop from the track centerline
 * minus the half width of the ribbon at the nearest sample, so the outer
 * and the inner band are compared on the same scale.
 *
 * @param spec - Generated track the prop belongs to.
 * @param prop - Prop to measure.
 * @returns The offset beyond the road edge, in meters.
 */
function computeOffsetBeyondRoadEdge(
  spec: ReturnType<typeof generateTrack>,
  prop: RightWorldProp,
): number {
  const sample = nearestSample(
    spec.splineSamples,
    prop.position[0],
    prop.position[2],
  );
  const frame = resolveSplineSampleFrame(
    spec.splineSamples,
    sample.globalIndex,
  );
  const lateral = Math.abs(computeLateral(sample, frame, prop.position));

  return lateral - sample.width / 2;
}

describe('buildRightWorldProps (P1-right-world-props, AC-048)', () => {
  it('returns at least one prop and stays within the density budget', () => {
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);

    expect(props.length).toBeGreaterThan(0);
    expect(props.length).toBeLessThanOrEqual(
      NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET,
    );
  });

  it('places every prop on the right side of the track and outside the road ribbon', () => {
    const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);

    for (const prop of props) {
      const sample = nearestSample(
        spec.splineSamples,
        prop.position[0],
        prop.position[2],
      );
      const frame = resolveSplineSampleFrame(
        spec.splineSamples,
        sample.globalIndex,
      );

      const generatorY = sceneZToGeneratorY(prop.position[2]);
      const deltaX = prop.position[0] - sample.x;
      const deltaY = generatorY - sample.y;
      const lateral = deltaX * frame.normalX + deltaY * frame.normalY;

      // Negative lateral means the prop is on the right-hand side of the
      // centerline (opposite the left normal used by the inner lane).
      expect(lateral).toBeLessThan(0);
      // It must also sit beyond the road core + curb band, not on the pavement.
      const ribbonHalfWidth = sample.width / 2;
      expect(Math.abs(lateral)).toBeGreaterThan(ribbonHalfWidth + 1);
    }
  });

  it('produces byte-identical output for the same tuple (same runtime)', () => {
    const first = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const second = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);

    expect(JSON.stringify(second)).toBe(JSON.stringify(first));
  });
});

describe('bleacher ring-side classification (P1-tron-bleachers, AC-053 + AC-063)', () => {
  it('emits bleacher props and classifies them on the outer ring side', () => {
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const bleachers = props.filter((prop) => (prop as unknown as { kind: string }).kind === 'bleacher');

    // AC-053 + AC-063: the bleachers are the reserved outer-ring slots, so
    // the ring holds exactly the twenty reserved bleachers and every one of
    // them sits on the OUTER ring side.
    expect(bleachers.length).toBe(NEATRACING_RIGHT_WORLD_BLEACHER_COUNT_EXPECTED);
    expect(
      bleachers.every(
        (prop) => (prop as unknown as { ringSide: string }).ringSide === 'outer',
      ),
    ).toBe(true);
  });
});

describe('right-world prop half-size parity (P1-tron-bleachers, AC-052)', () => {
  it('provides a canonical half-size for every emitted prop kind', () => {
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const kinds = new Set(props.map((prop) => (prop as unknown as { kind: string }).kind));

    for (const kind of kinds) {
      const [hx, hy, hz] = resolvePropHalfSize(kind as Parameters<typeof resolvePropHalfSize>[0]);
      expect(hx).toBeGreaterThan(0);
      expect(hy).toBeGreaterThan(0);
      expect(hz).toBeGreaterThan(0);
    }
  });

  it('pins the re-authored mechanic/crew pedestal half-sizes so the billboards stand on top', () => {
    // AC-052 with the recorded 14F/14J half-sizes: the mechanic and the
    // crew are LOW PEDESTALS with a 0.25 m vertical half-size, so the
    // pedestal top sits at 0.5 m and the billboard anchored on it stays at
    // crowd height instead of being hoisted above the scene. A taller
    // pedestal silently moves the mechanic/crew anchor.
    //
    // The footprints are the recorded 14F pass-side values (a 0.5 m x 0.5 m
    // mechanic pedestal and a 0.4 m x 0.4 m crew pedestal) and the vertical
    // half-size is the 14F remedy that keeps the pit props LOW PEDESTALS of
    // at most 0.5 m tall. The table is pinned EXACTLY: a bound-only
    // assertion would let the pedestal footprint drift - or grow into a
    // 9 m x 9 m cube - through unnoticed and would silently move the
    // mechanic/crew billboard anchor.
    const [mechanicX, mechanicY, mechanicZ] = resolvePropHalfSize('mechanic');
    const [crewX, crewY, crewZ] = resolvePropHalfSize('crew');

    expect([mechanicX, mechanicY, mechanicZ]).toEqual([0.5, 0.25, 0.5]);
    expect([crewX, crewY, crewZ]).toEqual([0.4, 0.25, 0.4]);
  });

  it('keeps the recorded team-banner panel half-size', () => {
    // The team banner is the only tall pit prop: a thin vertical panel with
    // the recorded half-size, so the banner reads as a banner from the
    // track instead of a wall.
    expect(resolvePropHalfSize('team-banner')).toEqual([1.2, 1.5, 0.1]);
  });
});

describe('bleacher stepped geometry and support capacity (P1-bleacher-shape, AC-066 + AC-065b)', () => {
  it('expands each bleacher prop into voxel geometry', () => {
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const bleachers = filterBleacherProps(props);

    expect(bleachers.length).toBeGreaterThan(0);

    for (const bleacher of bleachers) {
      const geometry = getBleacherGeometry(bleacher);

      expect(geometry).toBeDefined();
      expect(geometry!.voxels.length).toBeGreaterThan(0);
    }
  });

  it('emits tiered seating rows with multiple distinct heights', () => {
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const bleachers = filterBleacherProps(props);

    for (const bleacher of bleachers) {
      const geometry = getBleacherGeometry(bleacher)!;
      const seatVoxels = geometry.voxels.filter(
        (voxel) => voxel.role === 'seat',
      );
      const seatHeights = new Set(
        seatVoxels.map((voxel) => voxel.position[1]),
      );

      expect(seatVoxels.length).toBeGreaterThan(0);
      expect(seatHeights.size).toBeGreaterThanOrEqual(2);
    }
  });

  it('includes walking space voxels between seating rows', () => {
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const bleachers = filterBleacherProps(props);

    for (const bleacher of bleachers) {
      const geometry = getBleacherGeometry(bleacher)!;
      const walkVoxels = geometry.voxels.filter(
        (voxel) => voxel.role === 'walk',
      );

      expect(walkVoxels.length).toBeGreaterThan(0);
    }
  });

  it('places stairs on the back side of the bleachers', () => {
    const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const bleachers = filterBleacherProps(props);

    for (const bleacher of bleachers) {
      const geometry = getBleacherGeometry(bleacher)!;
      const stairVoxels = geometry.voxels.filter(
        (voxel) => voxel.role === 'stair',
      );

      expect(stairVoxels.length).toBeGreaterThan(0);

      const sample = nearestSample(
        spec.splineSamples,
        bleacher.position[0],
        bleacher.position[2],
      );
      const frame = resolveSplineSampleFrame(
        spec.splineSamples,
        sample.globalIndex,
      );
      const baseLateral = computeLateral(sample, frame, bleacher.position);
      const stairLaterals = stairVoxels.map((voxel) =>
        computeLateral(sample, frame, voxel.position),
      );
      const farthestStairLateral = Math.min(...stairLaterals);

      expect(farthestStairLateral).toBeLessThan(baseLateral);
    }
  });

  it('places guardrails on the track-facing side of the bleachers', () => {
    const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const bleachers = filterBleacherProps(props);

    for (const bleacher of bleachers) {
      const geometry = getBleacherGeometry(bleacher)!;
      const guardrailVoxels = geometry.voxels.filter(
        (voxel) => voxel.role === 'guardrail',
      );

      expect(guardrailVoxels.length).toBeGreaterThan(0);

      const sample = nearestSample(
        spec.splineSamples,
        bleacher.position[0],
        bleacher.position[2],
      );
      const frame = resolveSplineSampleFrame(
        spec.splineSamples,
        sample.globalIndex,
      );
      const baseLateral = computeLateral(sample, frame, bleacher.position);
      const guardrailLaterals = guardrailVoxels.map((voxel) =>
        computeLateral(sample, frame, voxel.position),
      );
      const closestGuardrailLateral = Math.max(...guardrailLaterals);

      expect(closestGuardrailLateral).toBeGreaterThan(baseLateral);
    }
  });

  it('spreads stepped seating rows horizontally across the prop footprint', () => {
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const bleachers = filterBleacherProps(props);

    for (const bleacher of bleachers) {
      const geometry = getBleacherGeometry(bleacher)!;
      const xs = geometry.voxels.map((voxel) => voxel.position[0]);
      const zs = geometry.voxels.map((voxel) => voxel.position[2]);
      const xSpread = Math.max(...xs) - Math.min(...xs);
      const zSpread = Math.max(...zs) - Math.min(...zs);

      expect(xSpread).toBeGreaterThanOrEqual(2);
      expect(zSpread).toBeGreaterThanOrEqual(1);
    }
  });

  it('provides at least 1m of horizontal walking space per hosted robot', () => {
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const bleachers = filterBleacherProps(props);

    for (const bleacher of bleachers) {
      const geometry = getBleacherGeometry(bleacher)!;
      const seatVoxels = geometry.voxels.filter(
        (voxel) => voxel.role === 'seat',
      );
      const walkVoxels = geometry.voxels.filter(
        (voxel) => voxel.role === 'walk',
      );
      const walkXs = walkVoxels.map((voxel) => voxel.position[0]);
      const walkZs = walkVoxels.map((voxel) => voxel.position[2]);
      const walkWidth = Math.max(...walkXs) - Math.min(...walkXs);
      const walkDepth = Math.max(...walkZs) - Math.min(...walkZs);
      const walkArea = walkWidth * walkDepth;

      expect(seatVoxels.length).toBeGreaterThan(0);
      expect(walkVoxels.length).toBeGreaterThan(0);
      expect(walkArea).toBeGreaterThanOrEqual(seatVoxels.length);
    }
  });
});

/**
 * Groups seat voxels of a bleacher by their front-back row coordinate (u).
 * The u axis points toward the track, so the first group is the frontmost
 * row and the last group is the backmost row. Each returned row carries the
 * average seat height and the span in the tangent (side-to-side) direction.
 */
function computeSeatRows(
  bleacher: RightWorldProp,
  frame: SplineSampleFrame,
): { readonly u: number; readonly y: number; readonly depth: number }[] {
  const geometry = getBleacherGeometry(bleacher);
  if (!geometry) {
    return [];
  }

  const seats = geometry.voxels.filter((voxel) => voxel.role === 'seat');
  const groups = new Map<
    number,
    { readonly u: number; ys: number[]; vs: number[] }
  >();

  for (const seat of seats) {
    const dx = seat.position[0] - bleacher.position[0];
    const dz = seat.position[2] - bleacher.position[2];
    const u = dx * frame.normalX + dz * -frame.normalY;
    const v = dx * -frame.normalY + dz * -frame.normalX;
    // One-meter bins match the 1 m voxel-cell grid and keep a row from
    // splitting when the nearest spline frame differs slightly from the
    // frame used to emit the geometry.
    const key = Math.round(u);

    const existing = groups.get(key);
    if (existing) {
      existing.ys.push(seat.position[1]);
      existing.vs.push(v);
    } else {
      groups.set(key, { u, ys: [seat.position[1]], vs: [v] });
    }
  }

  return Array.from(groups.values())
    .map((group) => ({
      u: group.u,
      y: group.ys.reduce((sum, value) => sum + value, 0) / group.ys.length,
      depth: Math.max(...group.vs) - Math.min(...group.vs),
    }))
    .sort((a, b) => b.u - a.u);
}

/**
 * Groups seat rows of a bleacher into flat levels by rounded seat height.
 * Each returned level carries the average seat height and the count of
 * distinct front-to-back rows (rounded u) it contains. Used by the updated
 * legacy bleacher-shape assertions that now validate the flat-tier stadium
 * geometry (fix-packet-P1-bleacher-crowd-iteration-1, OBS-A).
 */
function computeSeatLevels(
  bleacher: RightWorldProp,
  frame: SplineSampleFrame,
): { readonly y: number; readonly rows: number }[] {
  const rows = computeSeatRows(bleacher, frame);
  const groups = new Map<number, { ySum: number; count: number }>();

  for (const row of rows) {
    const key = Math.round(row.y);
    const existing = groups.get(key);
    if (existing) {
      existing.ySum += row.y;
      existing.count += 1;
    } else {
      groups.set(key, { ySum: row.y, count: 1 });
    }
  }

  return Array.from(groups.values())
    .map((group) => ({
      y: group.ySum / group.count,
      rows: group.count,
    }))
    .sort((a, b) => a.y - b.y);
}

describe(
  'bleacher enlarged stadium geometry (fix-packet-P1-bleacher-shape-iteration-3, OBS-1)',
  () => {
    it('builds three flat bleacher levels with four stand-cell rows each and rising heights', () => {
      const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
      const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
      const bleachers = filterBleacherProps(props);

      expect(bleachers.length).toBeGreaterThan(0);

      for (const bleacher of bleachers) {
        const sample = nearestSample(
          spec.splineSamples,
          bleacher.position[0],
          bleacher.position[2],
        );
        const frame = resolveSplineSampleFrame(
          spec.splineSamples,
          sample.globalIndex,
        );
        const levels = computeSeatLevels(bleacher, frame);

        expect(levels.length).toBe(3);

        for (let index = 0; index < levels.length - 1; index += 1) {
          const lower = levels[index]!;
          const upper = levels[index + 1]!;
          expect(upper.rows).toBeGreaterThanOrEqual(4);
          expect(lower.rows).toBeGreaterThanOrEqual(4);
          expect(upper.y).toBeGreaterThan(lower.y);
        }
      }
    });

    it('places stairs between consecutive seat levels', () => {
      const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
      const bleachers = filterBleacherProps(props);

      for (const bleacher of bleachers) {
        const geometry = getBleacherGeometry(bleacher)!;
        const seatHeights = Array.from(
          new Set(
            geometry.voxels
              .filter((voxel) => voxel.role === 'seat')
              .map((voxel) => voxel.position[1]),
          ),
        ).sort((a, b) => a - b);

        expect(seatHeights.length).toBeGreaterThanOrEqual(3);

        for (let index = 0; index < seatHeights.length - 1; index += 1) {
          const lowHeight = seatHeights[index]!;
          const highHeight = seatHeights[index + 1]!;
          const hasStairBetween = geometry.voxels.some(
            (voxel) =>
              voxel.role === 'stair' &&
              voxel.position[1] > lowHeight &&
              voxel.position[1] < highHeight,
          );

          expect(hasStairBetween).toBe(true);
        }
      }
    });

    it('reserves at least one free voxel row on each outer side of the bleacher', () => {
      const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
      const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
      const bleachers = filterBleacherProps(props);

      for (const bleacher of bleachers) {
        const sample = nearestSample(
          spec.splineSamples,
          bleacher.position[0],
          bleacher.position[2],
        );
        const frame = resolveSplineSampleFrame(
          spec.splineSamples,
          sample.globalIndex,
        );
        const geometry = getBleacherGeometry(bleacher)!;
        const allVoxels = geometry.voxels;
        const seatVoxels = allVoxels.filter((voxel) => voxel.role === 'seat');

        const uOf = (voxel: BleacherVoxel): number => {
          const dx = voxel.position[0] - bleacher.position[0];
          const dz = voxel.position[2] - bleacher.position[2];
          return dx * frame.normalX + dz * -frame.normalY;
        };

        const minU = Math.min(...allVoxels.map(uOf));
        const maxU = Math.max(...allVoxels.map(uOf));
        const minSeatU = Math.min(...seatVoxels.map(uOf));
        const maxSeatU = Math.max(...seatVoxels.map(uOf));

        // The projection of a world position back into the local frame keeps
        // a double-precision residue of ~1e-15 m, so the free-row contract is
        // asserted with a named tolerance. The measured free row stays a full
        // meter - the tolerance only covers the projection residue.
        expect(maxSeatU).toBeLessThanOrEqual(
          maxU - 1.0 + NEATRACING_BLEACHER_FRAME_TOLERANCE_M,
        );
        expect(minSeatU).toBeGreaterThanOrEqual(
          minU + 1.0 - NEATRACING_BLEACHER_FRAME_TOLERANCE_M,
        );
      }
    });

    it('gives each seat row at least two cells of depth', () => {
      const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
      const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
      const bleachers = filterBleacherProps(props);

      for (const bleacher of bleachers) {
        const sample = nearestSample(
          spec.splineSamples,
          bleacher.position[0],
          bleacher.position[2],
        );
        const frame = resolveSplineSampleFrame(
          spec.splineSamples,
          sample.globalIndex,
        );
        const rows = computeSeatRows(bleacher, frame);

        expect(rows.length).toBeGreaterThan(0);

        for (const row of rows) {
          expect(row.depth).toBeGreaterThanOrEqual(2.0);
        }
      }
    });

    it('bleacher axis-aligned dimensions are visibly enlarged beyond the coarse [1.5,0.75,1] half-size and rise between flat levels', () => {
      const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
      const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
      const bleachers = filterBleacherProps(props);

      expect(bleachers.length).toBeGreaterThan(0);

      for (const bleacher of bleachers) {
        const sample = nearestSample(
          spec.splineSamples,
          bleacher.position[0],
          bleacher.position[2],
        );
        const frame = resolveSplineSampleFrame(
          spec.splineSamples,
          sample.globalIndex,
        );
        const geometry = getBleacherGeometry(bleacher)!;

        const uOf = (voxel: BleacherVoxel): number => {
          const dx = voxel.position[0] - bleacher.position[0];
          const dz = voxel.position[2] - bleacher.position[2];
          return dx * frame.normalX + dz * -frame.normalY;
        };
        const vOf = (voxel: BleacherVoxel): number => {
          const dx = voxel.position[0] - bleacher.position[0];
          const dz = voxel.position[2] - bleacher.position[2];
          return dx * -frame.normalY + dz * -frame.normalX;
        };

        const us = geometry.voxels.map(uOf);
        const vs = geometry.voxels.map(vOf);
        const ys = geometry.voxels.map((voxel) => voxel.position[1]);
        const uHalf = (Math.max(...us) - Math.min(...us)) / 2;
        const vHalf = (Math.max(...vs) - Math.min(...vs)) / 2;
        const yExtent = Math.max(...ys) - Math.min(...ys);

        expect(uHalf).toBeGreaterThan(1.5);
        expect(vHalf).toBeGreaterThan(1);
        expect(yExtent).toBeGreaterThan(2);

        const levels = computeSeatLevels(bleacher, frame);
        for (let index = 0; index < levels.length - 1; index += 1) {
          const rise = levels[index + 1]!.y - levels[index]!.y;
          expect(rise).toBeGreaterThan(0.4);
        }
      }
    });
  },
);

describe(
  'bleacher crowd geometry (fix-packet-P1-bleacher-crowd-iteration-1, OBS-A)',
  () => {
    it('doubles bleacher level depth to at least four stand-cell rows per level', () => {
      const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
      const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
      const bleachers = filterBleacherProps(props);

      expect(bleachers.length).toBeGreaterThan(0);

      for (const bleacher of bleachers) {
        const sample = nearestSample(
          spec.splineSamples,
          bleacher.position[0],
          bleacher.position[2],
        );
        const frame = resolveSplineSampleFrame(
          spec.splineSamples,
          sample.globalIndex,
        );
        const geometry = getBleacherGeometry(bleacher)!;
        const seats = geometry.voxels.filter(
          (voxel) => voxel.role === 'seat',
        );

        // Group stand cells by level (rounded seat height) and count distinct
        // front-to-back row positions (rounded u) inside each level. With the
        // agents standing, each level must be deep enough for four rows.
        const rowsPerLevel = new Map<number, Set<number>>();
        for (const seat of seats) {
          const dx = seat.position[0] - bleacher.position[0];
          const dz = seat.position[2] - bleacher.position[2];
          const u = dx * frame.normalX + dz * -frame.normalY;
          const levelKey = Math.round(seat.position[1]);
          const set = rowsPerLevel.get(levelKey) ?? new Set<number>();
          set.add(Math.round(u));
          rowsPerLevel.set(levelKey, set);
        }

        const rowCounts = Array.from(rowsPerLevel.values()).map(
          (set) => set.size,
        );

        expect(rowCounts.length).toBeGreaterThanOrEqual(3);
        expect(Math.min(...rowCounts)).toBeGreaterThanOrEqual(4);
      }
    });
  },
);

/**
 * Reads a named numeric export from a module namespace at RUNTIME.
 *
 * The named constants are read through the namespace instead of a static
 * import, so a contract can assert that the module really EXPORTS the
 * constant under that name: a static import would be erased by the compiler
 * and would prove nothing about the exported surface. The namespace lookup
 * keeps the suite compiling and reports a MISSING constant as a failed
 * assertion instead of a compile error.
 */
function readNamedNumericExport(
  moduleNamespace: Record<string, unknown>,
  exportName: string,
): number | undefined {
  const value = moduleNamespace[exportName];
  return typeof value === 'number' ? value : undefined;
}

/** Front-to-back (u) coordinate of a world point relative to a bleacher. */
function computeBleacherU(
  bleacher: RightWorldProp,
  frame: SplineSampleFrame,
  position: readonly [number, number, number],
): number {
  const dx = position[0] - bleacher.position[0];
  const dz = position[2] - bleacher.position[2];
  return dx * frame.normalX + dz * -frame.normalY;
}

/** Resolves the spline frame at the track sample nearest to a bleacher. */
function resolveBleacherFrame(
  spec: ReturnType<typeof generateTrack>,
  bleacher: RightWorldProp,
): SplineSampleFrame {
  const sample = nearestSample(
    spec.splineSamples,
    bleacher.position[0],
    bleacher.position[2],
  );
  return resolveSplineSampleFrame(spec.splineSamples, sample.globalIndex);
}

/**
 * Returns the seat positions of the FRONT-MOST seat row of every bleacher
 * level - the narrowed candidate set AC-069 targets (7 cells x 3 levels).
 *
 * The row is derived from the bleacher GEOMETRY, not from
 * `getBleacherSeatAnchors`, so the personal-space contract stays
 * independent of the anchor-narrowing contract.
 */
function computeFrontSeatRowPositions(
  bleacher: RightWorldProp,
  frame: SplineSampleFrame,
): readonly (readonly [number, number, number])[] {
  const geometry = getBleacherGeometry(bleacher);
  if (!geometry) {
    return [];
  }

  const seats = geometry.voxels.filter((voxel) => voxel.role === 'seat');
  const frontUByLevel = new Map<number, number>();

  for (const seat of seats) {
    const levelKey = Math.round(seat.position[1] - bleacher.position[1]);
    const u = Math.round(computeBleacherU(bleacher, frame, seat.position));
    const current = frontUByLevel.get(levelKey);
    if (current === undefined || u > current) {
      frontUByLevel.set(levelKey, u);
    }
  }

  const positions: (readonly [number, number, number])[] = [];
  for (const seat of seats) {
    const levelKey = Math.round(seat.position[1] - bleacher.position[1]);
    const u = Math.round(computeBleacherU(bleacher, frame, seat.position));
    if (u === frontUByLevel.get(levelKey)) {
      positions.push(seat.position);
    }
  }

  return positions;
}

/**
 * One bleacher level, read from the prop's own voxel geometry.
 *
 * The levels are ordered by ascending seat height, so index 0 is the lowest
 * platform (the one nearest the track) and the last index is the highest.
 */
interface BleacherSeatLevel {
  /** Ascending platform index of the level. */
  readonly levelIndex: number;
  /** World-space Y of the level's seat cells. */
  readonly levelY: number;
  /** Front-row cells of the level, ordered by ascending local width. */
  readonly frontRow: readonly BleacherVoxel[];
}

/** One seat anchor matched to the bleacher front-row cell it stands on. */
interface MatchedSeatAnchor {
  /** Ascending platform index of the owning bleacher level. */
  readonly levelIndex: number;
  /** Index of the owning cell inside the level's front row, or -1. */
  readonly cellIndex: number;
  /** Local width coordinate of the owning cell. */
  readonly localV: number;
  /** World-space position of the anchor. */
  readonly position: readonly [number, number, number];
}

/**
 * Reads the bleacher levels and their front-row cells from the prop's own
 * voxel geometry.
 *
 * The geometry carries the exact local frame coordinate of every cell, so
 * the level lattice is read directly instead of being re-derived from a
 * projected world position. The front row of a level is its row with the
 * largest local depth coordinate - the row closest to the track.
 *
 * @param bleacher - Bleacher prop whose voxel geometry is inspected.
 * @returns The bleacher levels in ascending platform order.
 */
function readBleacherSeatLevels(
  bleacher: RightWorldProp,
): BleacherSeatLevel[] {
  const geometry = getBleacherGeometry(bleacher);
  const seats = (geometry?.voxels ?? []).filter(
    (voxel) => voxel.role === 'seat',
  );
  const seatsByLevelY = new Map<number, BleacherVoxel[]>();

  for (const seat of seats) {
    const levelSeats = seatsByLevelY.get(seat.position[1]) ?? [];
    levelSeats.push(seat);
    seatsByLevelY.set(seat.position[1], levelSeats);
  }

  return [...seatsByLevelY.entries()]
    .sort(([leftY], [rightY]) => leftY - rightY)
    .map(([levelY, levelSeats], levelIndex) => {
      const frontU = levelSeats.reduce(
        (maximum, seat) => Math.max(maximum, seat.localU),
        Number.NEGATIVE_INFINITY,
      );
      const frontRow = levelSeats
        .filter((seat) => seat.localU === frontU)
        .sort((left, right) => left.localV - right.localV);

      return { levelIndex, levelY, frontRow };
    });
}

/**
 * Matches every seat anchor to the front-row cell it stands on.
 *
 * A cell is matched by its world-space XZ position, so the match is exact
 * and independent of the spline frame the caller happens to resolve. An
 * anchor that matches no front-row cell is reported with a cell index of
 * `-1` instead of being dropped, so a contract can flag it.
 *
 * @param levels - Bleacher levels read from the prop geometry.
 * @param anchors - Seat anchors to match.
 * @returns One matched entry per anchor, in anchor order.
 */
function matchAnchorsToSeatCells(
  levels: readonly BleacherSeatLevel[],
  anchors: readonly (readonly [number, number, number])[],
): MatchedSeatAnchor[] {
  return anchors.map((anchor) => {
    for (const level of levels) {
      const cellIndex = level.frontRow.findIndex(
        (cell) =>
          Math.abs(cell.position[0] - anchor[0]) <=
            NEATRACING_BLEACHER_FRAME_TOLERANCE_M &&
          Math.abs(cell.position[2] - anchor[2]) <=
            NEATRACING_BLEACHER_FRAME_TOLERANCE_M,
      );

      if (cellIndex >= 0) {
        return {
          levelIndex: level.levelIndex,
          cellIndex,
          localV: level.frontRow[cellIndex]!.localV,
          position: anchor,
        };
      }
    }

    return {
      levelIndex: nearestSeatLevelIndex(levels, anchor[1]),
      cellIndex: -1,
      localV: Number.NaN,
      position: anchor,
    };
  });
}

/**
 * Returns the index of the bleacher level whose seat height is the closest
 * to the given anchor height.
 *
 * @param levels - Bleacher levels read from the prop geometry.
 * @param anchorY - World-space Y of the anchor.
 * @returns The closest level index, or 0 without levels.
 */
function nearestSeatLevelIndex(
  levels: readonly BleacherSeatLevel[],
  anchorY: number,
): number {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const level of levels) {
    const distance = Math.abs(anchorY - level.levelY);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = level.levelIndex;
    }
  }

  return bestIndex;
}

/**
 * Returns the evenly spaced cell indices of a front row for the crowd
 * contract: the first anchor sits on the FIRST cell, the last anchor on the
 * LAST cell, and the anchors in between are spread evenly.
 *
 * @param cellCount - Number of cells in the front row.
 * @returns The selected cell indices, ascending.
 */
function evenlySpacedCellIndices(cellCount: number): number[] {
  return Array.from(
    { length: NEATRACING_BLEACHER_SEATS_PER_LEVEL },
    (_unused, index) =>
      Math.round(
        (index * (cellCount - 1)) / (NEATRACING_BLEACHER_SEATS_PER_LEVEL - 1),
      ),
  );
}

/** Smallest pairwise XZ distance between the given world positions. */
function computeMinPairwiseXzDistance(
  positions: readonly (readonly [number, number, number])[],
): number {
  let minimum = Number.POSITIVE_INFINITY;

  for (let i = 0; i < positions.length; i += 1) {
    for (let j = i + 1; j < positions.length; j += 1) {
      const a = positions[i]!;
      const b = positions[j]!;
      minimum = Math.min(minimum, Math.hypot(a[0] - b[0], a[2] - b[2]));
    }
  }

  return minimum;
}

describe(
  'bleacher seat anchor spread (RED-A, AC-072) and personal-space margin (AC-069)',
  () => {
    it('returns exactly four seat anchors per bleacher level and twelve anchors in total', () => {
      const bleachers = filterBleacherProps(
        buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT),
      );

      expect(bleachers.length).toBeGreaterThan(0);

      const levelCounts: number[] = [];
      const anchorsPerLevel: number[][] = [];
      const anchorTotals: number[] = [];

      for (const bleacher of bleachers) {
        const levels = readBleacherSeatLevels(bleacher);
        const seated = matchAnchorsToSeatCells(
          levels,
          getBleacherSeatAnchors(bleacher),
        );

        levelCounts.push(levels.length);
        anchorsPerLevel.push(
          levels.map(
            (level) =>
              seated.filter((entry) => entry.levelIndex === level.levelIndex)
                .length,
          ),
        );
        anchorTotals.push(seated.length);
      }

      // RED-A (AC-072): the stock bleacher keeps its three flat levels ...
      expect(levelCounts).toEqual(
        bleachers.map(() => NEATRACING_BLEACHER_LEVEL_COUNT),
      );
      // ... every level carries EXACTLY four seated spectators ...
      expect(anchorsPerLevel).toEqual(
        bleachers.map(() =>
          Array.from(
            { length: NEATRACING_BLEACHER_LEVEL_COUNT },
            () => NEATRACING_BLEACHER_SEATS_PER_LEVEL,
          ),
        ),
      );
      // ... so a bleacher emits 3 x 4 = 12 seat anchors. Today the anchor set
      // carries one anchor per front-row cell, so a level emits seven
      // anchors and the bleacher emits 21.
      expect(anchorTotals).toEqual(
        bleachers.map(
          () =>
            NEATRACING_BLEACHER_LEVEL_COUNT * NEATRACING_BLEACHER_SEATS_PER_LEVEL,
        ),
      );
    });

    it('keeps the named bleacher seat constants in step with the anchor contract', () => {
      // The RED-A expectations are pinned on the CONTRACT numbers - four
      // seats per level and three levels - so they stay a pin on the
      // contract instead of following the implementation. The production
      // exports carry the same numbers, and this cross-check fails LOUDLY
      // when a named constant drifts away from the contract, instead of
      // letting the implementation quietly redefine its own expectation.
      expect(
        rightWorldBuilderModule.NEATRACING_BLEACHER_SEATS_PER_LEVEL,
      ).toBe(NEATRACING_BLEACHER_SEATS_PER_LEVEL);
      expect(rightWorldBuilderModule.NEATRACING_BLEACHER_LEVEL_COUNT).toBe(
        NEATRACING_BLEACHER_LEVEL_COUNT,
      );
    });

    it('spreads the four anchors of a level evenly across the full front row', () => {
      const bleachers = filterBleacherProps(
        buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT),
      );

      expect(bleachers.length).toBeGreaterThan(0);

      const violations: string[] = [];

      for (const bleacher of bleachers) {
        const levels = readBleacherSeatLevels(bleacher);
        const seated = matchAnchorsToSeatCells(
          levels,
          getBleacherSeatAnchors(bleacher),
        );

        for (const level of levels) {
          const levelAnchors = seated.filter(
            (entry) => entry.levelIndex === level.levelIndex,
          );
          const cellIndices = levelAnchors.map((entry) => entry.cellIndex);
          const expectedCellIndices = evenlySpacedCellIndices(
            level.frontRow.length,
          );

          // RED-A (AC-072): the anchors of a level are the evenly spaced
          // subset of that level's front-row cell lattice - the first
          // anchor on the FIRST cell, the last on the LAST cell. Today the
          // anchor set carries EVERY front-row cell of the level.
          if (cellIndices.join(',') !== expectedCellIndices.join(',')) {
            violations.push(
              `level=${level.levelIndex} cells=[${cellIndices.join(',')}] expected=[${expectedCellIndices.join(',')}]`,
            );
          }

          // ... and the realized world-space gaps between them are equal.
          const gaps = levelAnchors.slice(1).map((entry, index) =>
            Math.hypot(
              entry.position[0] - levelAnchors[index]!.position[0],
              entry.position[1] - levelAnchors[index]!.position[1],
              entry.position[2] - levelAnchors[index]!.position[2],
            ),
          );

          if (gaps.length > 0) {
            const spread = Math.max(...gaps) - Math.min(...gaps);
            if (spread > NEATRACING_BLEACHER_FRAME_TOLERANCE_M) {
              violations.push(
                `level=${level.levelIndex} gaps=[${gaps.map((gap) => gap.toFixed(12)).join(',')}]`,
              );
            }
          }
        }
      }

      expect(violations).toEqual([]);
    });

    it('places every narrowed seat anchor in the front-most seat row of its bleacher level', () => {
      const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
      const bleachers = filterBleacherProps(
        buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT),
      );

      expect(bleachers.length).toBeGreaterThan(0);

      for (const bleacher of bleachers) {
        const frame = resolveBleacherFrame(spec, bleacher);
        const geometry = getBleacherGeometry(bleacher)!;
        const seats = geometry.voxels.filter(
          (voxel) => voxel.role === 'seat',
        );
        const frontUByLevel = new Map<number, number>();

        for (const seat of seats) {
          const levelKey = Math.round(
            seat.position[1] - bleacher.position[1],
          );
          const u = Math.round(
            computeBleacherU(bleacher, frame, seat.position),
          );
          const current = frontUByLevel.get(levelKey);
          if (current === undefined || u > current) {
            frontUByLevel.set(levelKey, u);
          }
        }

        const violations: string[] = [];
        for (const anchor of getBleacherSeatAnchors(bleacher)) {
          const levelKey = Math.round(
            anchor[1] - bleacher.position[1],
          );
          const u = Math.round(computeBleacherU(bleacher, frame, anchor));
          if (u !== frontUByLevel.get(levelKey)) {
            violations.push(`level=${levelKey} u=${u}`);
          }
        }

        // AC-069: the seat row of each level is the FRONT-MOST one, toward
        // the track. Today the anchor set also carries the three back rows
        // of every level.
        expect(violations).toEqual([]);
      }
    });

    it('emits the seat anchors level-major in ascending platform order', () => {
      const bleachers = filterBleacherProps(
        buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT),
      );

      expect(bleachers.length).toBeGreaterThan(0);

      const levelSequences: number[][] = [];

      for (const bleacher of bleachers) {
        const levels = readBleacherSeatLevels(bleacher);
        const seated = matchAnchorsToSeatCells(
          levels,
          getBleacherSeatAnchors(bleacher),
        );

        levelSequences.push(seated.map((entry) => entry.levelIndex));
      }

      // RED-A (AC-072): the anchors are emitted LEVEL-MAJOR in ascending
      // platform order, so a sequential consumer walks the lowest platform
      // first and then climbs the stand. Today they are emitted
      // level-interleaved (level 0, 1, 2, 0, 1, 2, ...), the order the
      // retired AC-069 clause demanded.
      const expectedSequence = Array.from(
        { length: NEATRACING_BLEACHER_LEVEL_COUNT },
        (_unused, levelIndex) =>
          Array.from(
            { length: NEATRACING_BLEACHER_SEATS_PER_LEVEL },
            () => levelIndex,
          ),
      ).flat();

      expect(levelSequences).toEqual(bleachers.map(() => expectedSequence));
    });

    it('applies the personal-space margin through the named NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M constant', () => {
      const moduleExports =
        rightWorldBuilderModule as unknown as Record<string, unknown>;
      const epsilon = readNamedNumericExport(
        moduleExports,
        'NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M',
      );

      // AC-069: the 1 m personal-space comparison must go through a NAMED
      // margin constant so the nominal lattice is not rejected by
      // double-precision noise. The builder exports that constant since
      // this slice, so the contract is live.
      expect(typeof epsilon).toBe('number');
      expect(epsilon).toBeGreaterThanOrEqual(1e-12);
    });

    it('retains every narrowed front-row seat anchor under the personal-space filter', () => {
      const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
      const bleachers = filterBleacherProps(
        buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT),
      );

      expect(bleachers.length).toBeGreaterThan(0);

      for (const bleacher of bleachers) {
        const frame = resolveBleacherFrame(spec, bleacher);
        const frontRowPositions = computeFrontSeatRowPositions(
          bleacher,
          frame,
        );
        const retained = filterRobotsByPersonalSpace(
          frontRowPositions.map((position) => ({ position })),
        );

        // AC-069: the personal-space comparison must not knife-edge-truncate
        // the narrowed front row. Today the strict `distance < 1.0`
        // comparison drops every second cell of the mathematically 1.0 m
        // lattice, so only 12 of the 21 front-row seats survive.
        expect(retained.length).toBe(frontRowPositions.length);
        // ... and the realized clearance must stay at the effective 1 m, so
        // AC-065 is not relaxed. The 1e-9 tolerance matches the AC-069 RED.
        expect(
          computeMinPairwiseXzDistance(
            retained.map((robot) => robot.position),
          ),
        ).toBeGreaterThanOrEqual(NEATRACING_ROBOT_PERSONAL_SPACE_M - 1e-9);
      }
    });
  },
);

describe(
  'bleacher count contract and named crowd-host rule (RED-B, AC-073)',
  () => {
    it('emits exactly twenty bleacher props for every fixture seed', () => {
      const counts = NEATRACING_CROWD_FIXTURE_SEEDS.map((seed) => {
        const props = buildRightWorldProps({
          seed,
          ...NEATRACING_CROWD_BUILD_INPUT,
        });

        return filterBleacherProps(props).length;
      });

      // RED-B (AC-073): the right world carries EXACTLY twenty bleacher props
      // and the count is SEED-INVARIANT, so the retired random 1-in-4
      // archetype draw never produces a bleacher - the bleacher slots are
      // reserved BY CONSTRUCTION at evenly spaced outer-ring slot indices.
      // The count no longer varies with the seed: every seed reserves the
      // same twenty slots.
      expect(counts).toEqual(
        NEATRACING_CROWD_FIXTURE_SEEDS.map(
          () => NEATRACING_RIGHT_WORLD_BLEACHER_COUNT_EXPECTED,
        ),
      );
    });

    it('derives the bleacher count from the named NEATRACING_RIGHT_WORLD_BLEACHER_COUNT constant', () => {
      const moduleExports =
        rightWorldBuilderModule as unknown as Record<string, unknown>;
      const count = readNamedNumericExport(
        moduleExports,
        'NEATRACING_RIGHT_WORLD_BLEACHER_COUNT',
      );
      const observed = filterBleacherProps(
        buildRightWorldProps({
          seed: 42,
          ...NEATRACING_CROWD_BUILD_INPUT,
        }),
      ).length;

      // RED-B (AC-073): the bleacher count comes from the NAMED constant
      // NEATRACING_RIGHT_WORLD_BLEACHER_COUNT = 20, and the emitted world
      // matches that constant.
      expect(count).toBe(NEATRACING_RIGHT_WORLD_BLEACHER_COUNT_EXPECTED);
      expect(count).toBe(observed);
    });

    it('keeps the reserved bleachers in their reserved slot order', () => {
      // The retired single-host ordering helper is GONE: the bleachers keep
      // the order of their reserved slots, so the rule-selected host is not
      // moved into the first bleacher slot any more. The replayed emission
      // pins the reserved slots, so a reordering breaks the comparison.
      const violations: string[] = [];

      for (const input of NEATRACING_STREAM_FIXTURE_INPUTS) {
        const production = buildRightWorldProps(input);
        const replayed = replayEmittedProps(input);

        const productionBleachers = production
          .filter((prop) => (prop as unknown as { kind: string }).kind === 'bleacher')
          .map((prop) =>
            propStreamKey({
              kind: (prop as unknown as { kind: string }).kind,
              x: prop.position[0],
              z: prop.position[2],
              color: prop.color,
            }),
          );
        const replayedBleachers = replayed
          .filter((prop) => prop.kind === 'bleacher')
          .map((prop) => propStreamKey(prop));

        if (productionBleachers.join('|') !== replayedBleachers.join('|')) {
          violations.push(
            `seed=${input.seed} bucket=${input.sizeBucket}`,
          );
        }
      }

      expect(violations).toEqual([]);
    });

    it('still selects the rule-selected crowd host among the bleachers', () => {
      const missing: string[] = [];

      for (const input of NEATRACING_STREAM_FIXTURE_INPUTS) {
        const production = buildRightWorldProps(input);
        const host = selectBleacherHostProp(production);

        // The named rule (largest hosting footprint area, ties broken by the
        // lowest prop index) is RETAINED; only the MOVE of the host into the
        // first bleacher slot is retired.
        if (!host || !production.includes(host)) {
          missing.push(`seed=${input.seed} bucket=${input.sizeBucket}`);
        }
      }

      expect(missing).toEqual([]);
    });

    it('selects the bleacher of the first reserved slot as the crowd host', () => {
      // The retired ordering helper moved the host into the first bleacher
      // slot. The replacement is the CONSTRUCTION itself: every bleacher is
      // the SAME stand, so the hosting footprint area ties for all of them
      // and the documented tie-break (the lowest prop index) picks the
      // bleacher of the FIRST reserved slot. The seat-anchor consumer fills
      // the seats in prop order, so the host is the bleacher whose seats are
      // consumed first and the host always receives its spectators.
      const violations: string[] = [];

      for (const input of NEATRACING_STREAM_FIXTURE_INPUTS) {
        const production = buildRightWorldProps(input);
        const bleachers = filterBleacherProps(production);
        const host = selectBleacherHostProp(production);
        const firstBleacher = bleachers[0];

        if (!host || !firstBleacher || host !== firstBleacher) {
          violations.push(`seed=${input.seed} bucket=${input.sizeBucket}`);
          continue;
        }

        const areas = new Set(
          bleachers.map((bleacher) =>
            getBleacherHostingFootprintArea(bleacher),
          ),
        );
        if (areas.size !== 1) {
          violations.push(
            `seed=${input.seed} bucket=${input.sizeBucket} areas=${areas.size}`,
          );
        }

        // The hosting area is the STAND footprint measured in the bleacher's
        // own local frame, so it equals the named footprint product and does
        // not change with the rotation of the stand.
        const area = getBleacherHostingFootprintArea(firstBleacher);
        if (
          area !==
          NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_WIDTH_M *
            NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_DEPTH_M
        ) {
          violations.push(
            `seed=${input.seed} bucket=${input.sizeBucket} area=${area}`,
          );
        }
      }

      expect(violations).toEqual([]);
    });

    it('pins the named outer-ring rule constants', () => {
      // The ring rule is driven by NAMED constants, so a silent change of a
      // rule input shows up here instead of quietly reshaping the world.
      expect(NEATRACING_RIGHT_WORLD_BLEACHER_COUNT).toBe(20);
      expect(NEATRACING_RIGHT_WORLD_PEOPLE_RING_COUNT).toBe(10);
      expect(NEATRACING_RIGHT_WORLD_SAMPLES_PER_RING_SLOT).toBe(10);
      expect(NEATRACING_RIGHT_WORLD_MIN_RING_BLOCKS).toBe(2);
      expect(NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_WIDTH_M).toBe(9);
      expect(NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_DEPTH_M).toBe(14);
      // The separation floor is the sum of the two circumradii of the
      // footprint, rounded UP to the next tenth of a meter.
      expect(NEATRACING_RIGHT_WORLD_BLEACHER_MIN_SEPARATION_M).toBe(16.7);
    });

    it('reserves the bleacher slots at one even stride across the whole ring', () => {
      // The reserved slots are spread over the ring at ONE even stride, so
      // the twenty bleachers sit evenly along the track. The rule is checked
      // against the PRODUCTION selector, so a silent change of the stride
      // shows up here instead of quietly bunching the stands.
      const slotCount = resolveOuterRingSlotCount(108);
      const reserved = selectBleacherSlotIndices(slotCount);

      // The reachable slot count is two blocks of twenty: twenty bleachers
      // and twenty scenery props.
      expect(slotCount).toBe(NEATRACING_RIGHT_WORLD_BLEACHER_COUNT * 2);
      expect(reserved).toHaveLength(NEATRACING_RIGHT_WORLD_BLEACHER_COUNT);
      // Every reserved slot is a distinct slot of the ring.
      expect(new Set(reserved).size).toBe(
        NEATRACING_RIGHT_WORLD_BLEACHER_COUNT,
      );
      // The reserved slots start at the first slot of the ring and stay
      // inside it.
      expect(reserved[0]).toBe(0);
      expect(Math.max(...reserved)).toBeLessThan(slotCount);
      // The stride between two reserved slots is the same everywhere.
      const strides = reserved
        .slice(1)
        .map((slot, index) => slot - reserved[index]!);
      expect(new Set(strides).size).toBe(1);
      expect(strides[0]).toBe(
        slotCount / NEATRACING_RIGHT_WORLD_BLEACHER_COUNT,
      );
    });

    it('keeps every pair of bleachers separated in world space', () => {
      // AC-073: no two bleachers overlap in world space. Two bleachers whose
      // FOOTPRINT CENTRES are closer than the sum of their circumradii COULD
      // overlap, so the named separation floor is a sufficient no-overlap
      // condition. The contract measures the footprint centres - the
      // quantity the floor is derived from - and the prop anchors as well:
      // the anchor sits 2.5 m in front of the footprint centre, so the
      // anchor distance is a proxy that is only sufficient 5 m above the
      // floor, while the centre distance IS the floor.
      // The bleacher count is NEVER lowered to fit a short track: the
      // reserved slots are fixed, and the realized separation is measured
      // per fixture below, so a track that is too short for twenty stands
      // shows up here as a violation instead of a silently reduced count.
      //
      // The sweep covers the AC seed set {0, 7, 42, 61, 123, 1337, 2777,
      // 4242} of the count contract, so the no-overlap clause is proven on
      // the SAME seeds the count is seed-invariant on, and it keeps the
      // reachable-configuration tuple set on top.
      const inputs: readonly TrackSlabBuildInput[] = [
        ...NEATRACING_STREAM_FIXTURE_INPUTS,
        ...NEATRACING_CROWD_FIXTURE_SEEDS.map((seed) => ({
          seed,
          ...NEATRACING_CROWD_BUILD_INPUT,
        })),
      ];
      const violations: string[] = [];

      for (const input of inputs) {
        const bleachers = filterBleacherProps(buildRightWorldProps(input));

        for (let i = 0; i < bleachers.length; i += 1) {
          for (let j = i + 1; j < bleachers.length; j += 1) {
            const left = bleachers[i]!;
            const right = bleachers[j]!;
            const anchorDistance = Math.hypot(
              left.position[0] - right.position[0],
              left.position[2] - right.position[2],
            );
            const leftCentre = bleacherFootprintCentre(left);
            const rightCentre = bleacherFootprintCentre(right);
            const centreDistance = Math.hypot(
              leftCentre[0] - rightCentre[0],
              leftCentre[1] - rightCentre[1],
            );

            if (
              anchorDistance < NEATRACING_RIGHT_WORLD_BLEACHER_MIN_SEPARATION_M
            ) {
              violations.push(
                `seed=${input.seed} bucket=${input.sizeBucket} pair=${i},${j} anchorDistance=${anchorDistance.toFixed(3)}`,
              );
            }
            if (
              centreDistance < NEATRACING_RIGHT_WORLD_BLEACHER_MIN_SEPARATION_M
            ) {
              violations.push(
                `seed=${input.seed} bucket=${input.sizeBucket} pair=${i},${j} centreDistance=${centreDistance.toFixed(3)}`,
              );
            }
          }
        }
      }

      expect(violations).toEqual([]);
    });

    it('emits exactly twenty bleacher props in every reachable track configuration', () => {
      const counts = NEATRACING_STREAM_FIXTURE_INPUTS.map(
        (input) => filterBleacherProps(buildRightWorldProps(input)).length,
      );

      // RED-B (AC-073): the bleacher slots are reserved BY CONSTRUCTION at
      // evenly spaced outer-ring slot indices, so EVERY reachable track
      // configuration carries the same twenty bleachers. The count does not
      // change with the reachable configuration any more.
      expect(counts).toEqual(
        NEATRACING_STREAM_FIXTURE_INPUTS.map(
          () => NEATRACING_RIGHT_WORLD_BLEACHER_COUNT_EXPECTED,
        ),
      );
    });
  },
);

describe(
  'inner-ring pit people band (AC-074, P1-crowd2-bleacher-count)',
  () => {
    it('carries the mechanic, crew and team-banner prop kinds on the inner ring', () => {
      const violations: string[] = [];

      for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
        const props = buildRightWorldProps({
          seed,
          ...NEATRACING_CROWD_BUILD_INPUT,
        });
        const innerKinds = new Set(
          props
            .filter(
              (prop) =>
                (prop as unknown as { ringSide: string }).ringSide === 'inner',
            )
            .map((prop) => (prop as unknown as { kind: string }).kind),
        );

        for (const required of NEATRACING_TEST_INNER_RING_PEOPLE_KINDS) {
          if (!innerKinds.has(required)) {
            violations.push(`seed=${seed} missing=${required}`);
          }
        }
      }

      // AC-074: the three people kinds are re-authored on the INNER ring, so
      // the assembled scene carries the mechanic, the crew, and the team
      // banner props at every seed.
      expect(violations).toEqual([]);
    });

    it('emits one people prop per bleacher pair and keeps the world inside the density budget', () => {
      const violations: string[] = [];

      for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
        const props = buildRightWorldProps({
          seed,
          ...NEATRACING_CROWD_BUILD_INPUT,
        });
        const innerCount = props.filter(
          (prop) => (prop as unknown as { ringSide: string }).ringSide === 'inner',
        ).length;
        const bleacherCount = props.filter(
          (prop) => (prop as unknown as { kind: string }).kind === 'bleacher',
        ).length;

        // The pit band places one prop at every midpoint between two
        // reserved bleacher slots, so the band holds one prop per bleacher
        // PAIR.
        if (innerCount !== Math.floor(bleacherCount / 2)) {
          violations.push(
            `seed=${seed} inner=${innerCount} bleachers=${bleacherCount}`,
          );
        }
        // The emitted band size also matches the PRODUCTION band constant,
        // so the two rings cannot drift apart silently.
        if (innerCount !== NEATRACING_RIGHT_WORLD_PEOPLE_RING_COUNT) {
          violations.push(
            `seed=${seed} inner=${innerCount} expected=${NEATRACING_RIGHT_WORLD_PEOPLE_RING_COUNT}`,
          );
        }
        // The named density budget covers BOTH rings, so the outer ring
        // reserves the pit band before it counts its own slots.
        if (props.length > NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET) {
          violations.push(`seed=${seed} props=${props.length}`);
        }
      }

      expect(violations).toEqual([]);
      // The band size is pinned against the PRODUCTION constant and not
      // against a literal that repeats the same arithmetic: the test-local
      // mirror states what the band must hold, the production constant
      // states what it does hold.
      expect(NEATRACING_TEST_PEOPLE_RING_COUNT).toBe(
        NEATRACING_RIGHT_WORLD_PEOPLE_RING_COUNT,
      );
    });

    it('keeps every bleacher on the outer ring', () => {
      const violations: string[] = [];

      for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
        const props = buildRightWorldProps({
          seed,
          ...NEATRACING_CROWD_BUILD_INPUT,
        });

        for (const prop of props) {
          if (
            (prop as unknown as { kind: string }).kind === 'bleacher' &&
            (prop as unknown as { ringSide: string }).ringSide !== 'outer'
          ) {
            violations.push(`seed=${seed}`);
          }
        }
      }

      expect(violations).toEqual([]);
    });

    it('measures the road-edge offset of every prop at the track sample it was placed on', () => {
      // The AC-074 road-edge metric is measured at the NEAREST track sample
      // of a prop. The metric is only exact when that nearest sample IS the
      // track sample the prop was placed on, so the identity is pinned here
      // instead of being left to a comment: a track shape that puts a prop
      // nearer a different sample fails this test loudly.
      const violations: string[] = [];
      const inputs: readonly TrackSlabBuildInput[] = [
        ...NEATRACING_STREAM_FIXTURE_INPUTS,
        ...NEATRACING_CROWD_FIXTURE_SEEDS.map((seed) => ({
          seed,
          ...NEATRACING_CROWD_BUILD_INPUT,
        })),
      ];

      for (const input of inputs) {
        const samples = generateTrack(input).splineSamples;
        const props = buildRightWorldProps(input);
        const outerSlotCount = resolveOuterRingSlotCount(samples.length);
        const bleacherSlotIndices = selectBleacherSlotIndices(outerSlotCount);
        const peopleSampleIndices = selectPeopleRingSampleIndices(
          samples,
          bleacherSlotIndices,
          outerSlotCount,
        );
        const outerProps = props.filter(
          (prop) =>
            (prop as unknown as { ringSide: string }).ringSide === 'outer',
        );
        const innerProps = props.filter(
          (prop) =>
            (prop as unknown as { ringSide: string }).ringSide === 'inner',
        );

        outerProps.forEach((prop, index) => {
          const placement = samples[
            Math.floor((index * samples.length) / outerSlotCount)
          ]!;

          if (
            nearestSample(samples, prop.position[0], prop.position[2]) !==
            placement
          ) {
            violations.push(
              `seed=${input.seed} bucket=${input.sizeBucket} outer=${index}`,
            );
          }
        });

        innerProps.forEach((prop, index) => {
          const placement = samples[peopleSampleIndices[index]!]!;

          if (
            nearestSample(samples, prop.position[0], prop.position[2]) !==
            placement
          ) {
            violations.push(
              `seed=${input.seed} bucket=${input.sizeBucket} inner=${index}`,
            );
          }
        });
      }

      expect(violations).toEqual([]);
    });

    it('places the inner-ring people props closer to the road than the bleachers', () => {
      const violations: string[] = [];

      for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
        const spec = generateTrack({
          seed,
          ...NEATRACING_CROWD_BUILD_INPUT,
        });
        const props = buildRightWorldProps({
          seed,
          ...NEATRACING_CROWD_BUILD_INPUT,
        });
        const innerOffsets = props
          .filter(
            (prop) =>
              (prop as unknown as { ringSide: string }).ringSide === 'inner',
          )
          .map((prop) => computeOffsetBeyondRoadEdge(spec, prop));
        const bleacherOffsets = props
          .filter(
            (prop) =>
              (prop as unknown as { kind: string }).kind === 'bleacher',
          )
          .map((prop) => computeOffsetBeyondRoadEdge(spec, prop));

        // AC-074: the pit band sits CLOSER to the road than the bleachers.
        // The offset is measured beyond the road + curb ribbon, so the two
        // bands are compared on the same scale.
        const innerMax = Math.max(...innerOffsets);
        const bleacherMin = Math.min(...bleacherOffsets);

        if (innerMax >= bleacherMin) {
          violations.push(
            `seed=${seed} innerMax=${innerMax.toFixed(2)} bleacherMin=${bleacherMin.toFixed(2)}`,
          );
        }

        // The two bands also sit in their OWN corridors: the pit props 3 m
        // to 5 m and the bleachers 6 m to 10 m beyond the road edge. The
        // measurement is exact on every reachable configuration, because
        // the nearest track sample of a prop IS the track sample it was
        // placed on - so this corridor check also proves that the
        // road-edge metric has not drifted onto a neighbouring sample.
        for (const offset of innerOffsets) {
          if (offset < 3 - 1e-9 || offset > 5 + 1e-9) {
            violations.push(`seed=${seed} innerOffset=${offset.toFixed(3)}`);
          }
        }
        for (const offset of bleacherOffsets) {
          if (offset < 6 - 1e-9 || offset > 10 + 1e-9) {
            violations.push(
              `seed=${seed} bleacherOffset=${offset.toFixed(3)}`,
            );
          }
        }
      }

      expect(violations).toEqual([]);
    });
  },
);

describe(
  'prop emission RNG stream preservation (AC-069/AC-070, B5)',
  () => {
    it('emits every prop exactly as the replayed xorshift32 stream predicts', () => {
      // B5: the reserved bleacher slot consumes its kind draw like every
      // other slot, so the stream never shifts. The replay below re-derives
      // the seed, the PRNG, the station mapping, and the per-slot draw order
      // (offset, kind, colour) and compares the result with the production
      // output. A reservation that skipped a draw, reordered a draw, or
      // changed a colour would shift every later slot and fail here.
      const mismatches: string[] = [];

      for (const input of NEATRACING_STREAM_FIXTURE_INPUTS) {
        const production = buildRightWorldProps(input);
        const replayed = replayEmittedProps(input);

        const productionKeys = production
          .map((prop) =>
            propStreamKey({
              kind: (prop as unknown as { kind: string }).kind,
              x: prop.position[0],
              z: prop.position[2],
              color: prop.color,
            }),
          )
          .sort();
        const replayedKeys = replayed.map((prop) => propStreamKey(prop)).sort();

        if (
          productionKeys.length !== replayedKeys.length ||
          productionKeys.some((key, index) => key !== replayedKeys[index])
        ) {
          mismatches.push(
            `seed=${input.seed} bucket=${input.sizeBucket} production=[${productionKeys.join(' ')}] replayed=[${replayedKeys.join(' ')}]`,
          );
        }
      }

      expect(mismatches).toEqual([]);
    });

    it('burns the three draws of an inner-ring slot even when its track sample is missing', () => {
      // The inner ring consumes EXACTLY three draws per slot - offset,
      // archetype, colour - so a missing track sample can never shift the
      // draw sequence of the following slots. The missing-sample branch is
      // unreachable on a reachable track, so the contract is pinned here
      // with a deliberately out-of-range sample index: the slot AFTER the
      // missing one has to consume the SAME draws as it does when the
      // skipped slot does have a track sample.
      const input = NEATRACING_STREAM_FIXTURE_INPUTS[0]!;
      const samples = generateTrack(input).splineSamples;
      const plan = {
        baseOffset: 3,
        offsetRange: 2,
      };
      const seeded = (): (() => number) =>
        replayDeterministicRandom(replayRightWorldSeed(input));

      const withMissingSample = emitRightWorldPeopleRing(samples, seeded(), {
        ...plan,
        sampleIndices: [0, samples.length + 1000, 1],
      });
      const withPresentSample = emitRightWorldPeopleRing(samples, seeded(), {
        ...plan,
        sampleIndices: [0, 2, 1],
      });

      // The skipped slot burns its three draws, so the third slot of both
      // runs sits on the same draw positions and comes out identical. A
      // skipped slot that burned NO draws would move the third slot onto
      // the draws of the second one and fail here.
      expect(withMissingSample).toHaveLength(2);
      expect(withPresentSample).toHaveLength(3);
      expect(withMissingSample[1]!.position).toEqual(
        withPresentSample[2]!.position,
      );
      expect(withMissingSample[1]!.color).toEqual(
        withPresentSample[2]!.color,
      );
      // ... and the compared slots really are DIFFERENT props, so the
      // comparison above is not vacuous.
      expect(withPresentSample[1]!.position).not.toEqual(
        withPresentSample[2]!.position,
      );
    });

    it('keeps the non-bleacher props in their emission order', () => {
      // The retired crowd-host reorder is GONE, so the non-bleacher props
      // are never moved at all: they keep their emission order, slot for
      // slot.
      const orderViolations: string[] = [];

      for (const input of NEATRACING_STREAM_FIXTURE_INPUTS) {
        const production = buildRightWorldProps(input);
        const replayed = replayEmittedProps(input);

        const productionKeys = production
          .filter((prop) => (prop as unknown as { kind: string }).kind !== 'bleacher')
          .map((prop) =>
            propStreamKey({
              kind: (prop as unknown as { kind: string }).kind,
              x: prop.position[0],
              z: prop.position[2],
              color: prop.color,
            }),
          );
        const replayedKeys = replayed
          .filter((prop) => prop.kind !== 'bleacher')
          .map((prop) => propStreamKey(prop));

        if (
          productionKeys.length !== replayedKeys.length ||
          productionKeys.some((key, index) => key !== replayedKeys[index])
        ) {
          orderViolations.push(
            `seed=${input.seed} bucket=${input.sizeBucket}`,
          );
        }
      }

      expect(orderViolations).toEqual([]);
    });

    it('keeps the stream stable when the reserved slot is not the first outer slot', () => {
      // The reachable track geometry reserves evenly spaced outer slots, so
      // a skipped archetype draw shifts EVERY later prop deterministically -
      // the stream defect is directly observable in the reachable space.
      // This contract drives the production ring emitter with a reserved
      // slot that is NOT the first one, so the same shift is asserted at a
      // known interior position of the ring.
      const input = NEATRACING_STREAM_FIXTURE_INPUTS[0]!;
      const samples = generateTrack(input).splineSamples;
      const reservedBleacherSlot = 1;
      const plan = {
        slotCount: 4,
        kinds: NEATRACING_TEST_OUTER_RING_KINDS,
        baseOffset: 6,
        offsetRange: 4,
        reservedBleacherSlots: [reservedBleacherSlot],
      };
      const seeded = (): (() => number) =>
        replayDeterministicRandom(replayRightWorldSeed(input));

      const production = propStreamKeys(
        emitRightWorldRing(samples, seeded(), plan),
      );
      const stable = replayRingProps(samples, seeded(), plan).map((prop) =>
        propStreamKey(prop),
      );
      const shifted = replayRingProps(samples, seeded(), {
        ...plan,
        skipArchetypeDrawAt: reservedBleacherSlot,
      }).map((prop) => propStreamKey(prop));

      // The reservation is not the terminal slot, so a skipped archetype
      // draw has to move the later slots - that is what gives the
      // comparison below its teeth.
      expect(reservedBleacherSlot).toBeLessThan(plan.slotCount - 1);
      expect(stable).toEqual(production);
      expect(shifted).not.toEqual(production);
    });

    it('pins the reserved bleachers of every reachable track configuration', () => {
      // Tripwire for the reachable space: the track generator emits 6..8
      // spline segments of 18 samples, so the reachable sample counts are
      // 108, 126, and 144. The outer ring is divided into whole bleacher
      // blocks, so every reachable configuration carries the same forty
      // outer slots - TWENTY reserved bleachers plus twenty scenery slots -
      // and the same ten inner-ring pit props. A change to the track
      // generator, to the density budget, or to the ring split fails this
      // test instead of silently weakening the bleacher-count contract.
      const reachable: readonly {
        readonly sampleCount: number;
        readonly innerPropCount: number;
        readonly outerPropCount: number;
        readonly bleacherCount: number;
      }[] = [
        { sampleCount: 108, innerPropCount: 10, outerPropCount: 40, bleacherCount: 20 },
        { sampleCount: 126, innerPropCount: 10, outerPropCount: 40, bleacherCount: 20 },
        { sampleCount: 144, innerPropCount: 10, outerPropCount: 40, bleacherCount: 20 },
      ];

      const derived = deriveReachableRingConfigurations(
        NEATRACING_STREAM_FIXTURE_INPUTS,
      );

      expect(
        derived.map(
          (entry) =>
            `${entry.sampleCount}/${entry.innerPropCount}/${entry.outerPropCount}/${entry.bleacherCount}`,
        ),
      ).toEqual(
        reachable.map(
          (entry) =>
            `${entry.sampleCount}/${entry.innerPropCount}/${entry.outerPropCount}/${entry.bleacherCount}`,
        ),
      );
      // The bleacher count is the NAMED constant in every reachable
      // configuration, independently of the seed and of the sample count.
      expect(derived.map((entry) => entry.bleacherCount)).toEqual(
        derived.map(() => NEATRACING_RIGHT_WORLD_BLEACHER_COUNT),
      );
    });

    it('derives the reachable ring configuration from the track generator', () => {
      // The pinned table above only holds while the track generator and the
      // density budget keep the reachable space at those three ring
      // configurations. This contract derives the reachable set from the
      // PRODUCTION emission - the prop counts are read back from the emitted
      // props, not re-derived here - so a silent change to the ring split,
      // to the density budget, or to the track generator is caught instead
      // of being mirrored by the test.
      const derived = deriveReachableRingConfigurations(
        NEATRACING_STREAM_FIXTURE_INPUTS,
      );

      expect(derived.length).toBeGreaterThan(0);
      // The inner pit band is back: it carries one people prop per bleacher
      // PAIR, so it holds half as many props as the outer ring holds
      // bleachers.
      expect(derived.map((entry) => entry.innerPropCount)).toEqual(
        derived.map(
          (entry) => Math.floor(entry.bleacherCount / 2),
        ),
      );
      // The outer ring is divided into whole bleacher blocks, so its slot
      // count is a whole multiple of the bleacher count and always offers
      // at least twenty bleacher slots.
      expect(
        derived.map((entry) => entry.outerPropCount % entry.bleacherCount),
      ).toEqual(derived.map(() => 0));
      expect(derived.map((entry) => entry.outerPropCount)).toEqual(
        derived.map((entry) => entry.bleacherCount * 2),
      );
      // The whole emission stays inside the named density budget.
      expect(
        derived.every(
          (entry) =>
            entry.innerPropCount + entry.outerPropCount <=
            NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET,
        ),
      ).toBe(true);
    });
  },
);

