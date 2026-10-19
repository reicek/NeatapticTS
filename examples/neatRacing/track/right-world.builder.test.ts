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
 */

import { describe, expect, it } from '@jest/globals';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';
import { resolveSplineSampleFrame } from '../../racing_curriculum/track/track.spline.utils';
import {
  NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET,
  buildRightWorldProps,
} from './right-world.builder';
import { resolvePropHalfSize } from './right-world.prop-sizes';

/** Canonical determinism tuple reused from the track slab builder. */
const FIXTURE_TRACK_BUILD_INPUT = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'small',
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

    expect(bleachers.length).toBeGreaterThan(0);
    expect(
      bleachers.every(
        (prop) => (prop as unknown as { ringSide: string }).ringSide === 'outer',
      ),
    ).toBe(true);
  });

  it('emits mechanic/crew props and classifies them on the inner ring side', () => {
    const props = buildRightWorldProps(FIXTURE_TRACK_BUILD_INPUT);
    const crew = props.filter(
      (prop) =>
        ['mechanic', 'crew', 'team-banner'].includes(
          (prop as unknown as { kind: string }).kind,
        ),
    );

    expect(crew.length).toBeGreaterThan(0);
    expect(
      crew.every(
        (prop) => (prop as unknown as { ringSide: string }).ringSide === 'inner',
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

  it('keeps inner-ring mechanic/crew pedestals short so billboards stand on top', () => {
    const [mX, mY, mZ] = resolvePropHalfSize('mechanic');
    const [cX, cY, cZ] = resolvePropHalfSize('crew');

    expect(mY).toBeLessThanOrEqual(0.25);
    expect(cY).toBeLessThanOrEqual(0.25);
    expect(Math.max(mX, mZ)).toBeGreaterThanOrEqual(0.4);
    expect(Math.max(cX, cZ)).toBeGreaterThanOrEqual(0.4);
  });
});
