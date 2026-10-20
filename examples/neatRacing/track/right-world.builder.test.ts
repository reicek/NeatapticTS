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
import {
  resolveSplineSampleFrame,
  type SplineSampleFrame,
} from '../../racing_curriculum/track/track.spline.utils';
import {
  NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET,
  buildRightWorldProps,
  type RightWorldProp,
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

/** Role of a single voxel inside an expanded bleacher prop. */
type BleacherVoxelRole = 'seat' | 'walk' | 'stair' | 'guardrail';

/** One voxel emitted by a stepped bleacher prop. */
interface BleacherVoxel {
  readonly position: readonly [number, number, number];
  readonly color: readonly [number, number, number];
  readonly role: BleacherVoxelRole;
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

/** Reads the optional stepped geometry from a bleacher prop. */
function getBleacherGeometry(
  prop: RightWorldProp,
): BleacherGeometry | undefined {
  return (prop as unknown as { bleacherGeometry?: BleacherGeometry })
    .bleacherGeometry;
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

        expect(maxSeatU).toBeLessThanOrEqual(maxU - 1.0);
        expect(minSeatU).toBeGreaterThanOrEqual(minU + 1.0);
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
