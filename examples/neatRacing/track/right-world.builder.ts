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
import { resolveSplineSampleFrame } from '../../racing_curriculum/track/track.spline.utils';
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

      props.push({
        position: [generatorX, 0, -generatorY],
        color,
        kind,
        ringSide: RING_SIDE_BY_KIND[kind],
      });
    }
  };

  // Inner ring: mechanic/crew props closer to the track, still off the pavement.
  emitRing(innerTargetCount, INNER_RING_KINDS, 2, 2);
  // Outer ring: spectator bleachers + neon scenery props farther out.
  emitRing(outerTargetCount, OUTER_RING_KINDS, 6, 4);

  return props;
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
