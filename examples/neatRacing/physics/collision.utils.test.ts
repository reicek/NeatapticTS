/**
 * Contract tests for the Neat Racing collision / boundary / off-track
 * surface utils and the keyboard input map (plan slice P1-S7-C; AC-024
 * partial, spec §4.3, §5.1 D5/D6, LD-04/LD-05, §5.2).
 *
 * Red phase: this suite must fail while './collision.utils' and
 * '../host/input' do not exist ("Cannot find module './collision.utils'" /
 * "Cannot find module '../host/input'" under ts-jest diagnostics); a green
 * suite during the red phase would be a contract defect. The plan red_test
 * note was reconciled against the spec source of truth: arcade elastic
 * bounce responses (car-car and car-boundary, LD-05) plus off-track
 * detection on the track ribbon lateral bounds (§5.2) and the D5 keyboard
 * map (sample once per rAF frame, replay into every catch-up tick).
 *
 * Determinism rung: same-runtime byte-identical via `JSON.stringify`
 * serialization; every util is pure — it returns new records, never
 * mutates its inputs, and holds no runtime-random state. Not claimed:
 * cross-environment byte equality.
 *
 * Fixture: a single straight ribbon segment (the plan red contract's
 * startX -50 / startY 0 / endX 50 / endY 0, width 8 sliver) plus seed-42
 * derived car states (the spawn baseline is overridable per behavior).
 * Geometry assertions are radius- or band-relative — computed from the
 * exported constants and the fixture width, never magic thresholds.
 *
 * Green contract for the implementer: './collision.utils' exports
 * `NEATRACING_CAR_COLLISION_RADIUS_M` (D6: a named fixed physics
 * constant, finite > 0, never derived from voxel models), the band widths
 * `NEATRACING_TRACK_CURB_WIDTH_M` (1.0) and
 * `NEATRACING_TRACK_GRASS_APRON_WIDTH_M` (3.0) mirroring the private
 * track.slabs band constants, the factory
 * `createTrackSurfaceResolver(segments)` returning a pure
 * `(car: PhysicsCarState) => PhysicsSurfaceType` whose nearest-segment
 * lateral bands mirror the slab rasterizer (road when
 * |lat| <= halfWidth - curb, curb when <= halfWidth, grass when
 * <= halfWidth + apron and beyond), `resolveCarCollisions(cars)` doing
 * pairwise circular overlap detection by the D6 radius with an
 * equal-mass frictionless arcade elastic exchange along the collision
 * normal (kinetic energy and normal momentum conserved, relative normal
 * velocity reversed, no deformation, no crash penalties or rewards per
 * LD-04) and centers separated to at least the sum of the radii, and
 * `resolveTrackBoundaryCar(car, segments)` using the OUTER grass-apron
 * edge (halfWidth + apron) as the bound — the untouched-at-lat-5 case
 * deliberately pins that interpretation (outer apron edge rather than the
 * road edge, because §5.2 keeps the grass apron reachable as the off-road
 * zone) for 04-implementing review: outside the bound the car is reseated
 * inside with its outward world-velocity component reflected elastically
 * at preserved magnitude; inside the bound the car is returned
 * byte-identical. '../host/input' exports the pure
 * `sampleKeyboardInput(presses)` reading a once-per-rAF-frame pressed-key
 * map `{ up, down, left, right }` (host-collected, transport neutral) and
 * returning the analog axes sample `NeatRacingKeyboardSample =
 * { throttle, brake, steer }` — up→throttle 1, down→brake 1, left→steer
 * +1 (counterclockwise), right→steer −1, neutral map→zero axes — a value
 * the host replayed unchanged into EVERY catch-up tick of its frame (D5).
 */

import { describe, expect, it } from '@jest/globals';
import {
  createInitialPhysicsCarState,
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  stepPhysics,
} from './physics.step.service';
import type {
  PhysicsCarState,
  PhysicsStepInput,
  PhysicsSurfaceType,
} from './physics.step.service';
import {
  createTrackSurfaceResolver,
  NEATRACING_CAR_COLLISION_RADIUS_M,
  NEATRACING_TRACK_CURB_WIDTH_M,
  NEATRACING_TRACK_GRASS_APRON_WIDTH_M,
  resolveCarCollisions,
  resolveTrackBoundaryCar,
} from './collision.utils';
import { sampleKeyboardInput } from '../host/input';
import type { NeatRacingKeyboardSample } from '../host/input';
import type { TrackSegment } from '../../racing_curriculum/track/track.generator.types';

/** The plan red contract's single straight ribbon fixture segment. */
const FIXTURE_SEGMENT: TrackSegment = {
  startX: -50,
  startY: 0,
  endX: 50,
  endY: 0,
  width: 8,
};

/** Fixture geometry list shape consumed by the resolver/boundary utils. */
const FIXTURE_TRACK_SEGMENTS: readonly TrackSegment[] = [FIXTURE_SEGMENT];

/** Driveable half width of the fixture ribbon, in meters. */
const FIXTURE_HALF_TRACK_WIDTH_M = FIXTURE_SEGMENT.width / 2;

/** Interior fixture-chord x (any x in (startX, endX) is on the chord). */
const FIXTURE_RIBBON_X_M = 25;

/**
 * The once-per-rAF-frame pressed-key map the host feeds to
 * sampleKeyboardInput (D5). Key-to-press binding is the host's DOM
 * concern; the map itself is plain boolean state.
 */
interface NeatRacingKeyboardPresses {
  readonly up: boolean;
  readonly down: boolean;
  readonly left: boolean;
  readonly right: boolean;
}

/** No key held: the all-neutral pressed-key fixture. */
const NEUTRAL_KEYS: NeatRacingKeyboardPresses = {
  up: false,
  down: false,
  left: false,
  right: false,
};

/** Throttle + left pressed-key fixture (determinism and burst blocks). */
const THROTTLE_LEFT_KEYS: NeatRacingKeyboardPresses = {
  ...NEUTRAL_KEYS,
  up: true,
  left: true,
};

/**
 * Spawns a seed-42 derived car state re-based onto the fixture ribbon:
 * the spawn baseline (0 speed, 0 yaw, 0 slip, heading 0) is overridable
 * per behavior so every test builds a self-contained state.
 */
const seededCarAt = (
  xM: number,
  yM: number,
  overrides: Partial<PhysicsCarState> = {},
): PhysicsCarState => ({
  ...createInitialPhysicsCarState(42),
  x: xM,
  y: yM,
  ...overrides,
});

/** A car standing on the fixture chord at the given lateral offset. */
const carOnRibbonAt = (lateralM: number): PhysicsCarState =>
  seededCarAt(FIXTURE_RIBBON_X_M, lateralM);

/**
 * Head-on closing fixture pair from the plan red contract: seed-42
 * states centered 1 cm apart on x, each closing at 10 m/s along the
 * shared collision normal +x (A heads +x from the left, B heads PI from
 * the right).
 */
const headOnOverlapPair = (): readonly [PhysicsCarState, PhysicsCarState] => {
  const overlappingGapM = 0.01;
  const closingSpeedPerS = 10;
  return [
    seededCarAt(-overlappingGapM / 2, 0, { speed: closingSpeedPerS }),
    seededCarAt(overlappingGapM / 2, 0, {
      heading: Math.PI,
      speed: closingSpeedPerS,
    }),
  ];
};

/**
 * World velocity [vx, vy] of a car state: the heading-longitudinal axis
 * plus the perpendicular lateral axis (plan red contract formula).
 */
const worldVelocityOf = (car: PhysicsCarState): readonly [number, number] => [
  Math.cos(car.heading + car.slipAngle) * car.speed -
    Math.sin(car.heading) * car.lateralVelocity,
  Math.sin(car.heading + car.slipAngle) * car.speed +
    Math.cos(car.heading) * car.lateralVelocity,
];

/** Byte serialization for byte-parity assertions over car-state lists. */
const serializeCars = (cars: readonly PhysicsCarState[]): string =>
  JSON.stringify(cars);

describe('collision.utils constants (D6, spec §5.1)', () => {
  it('exposes NEATRACING_CAR_COLLISION_RADIUS_M as a finite positive named constant', () => {
    expect(Number.isFinite(NEATRACING_CAR_COLLISION_RADIUS_M)).toBe(true);
    expect(NEATRACING_CAR_COLLISION_RADIUS_M).toBeGreaterThan(0);
  });

  it('mirrors the track slab curb and grass-apron band widths', () => {
    expect(NEATRACING_TRACK_CURB_WIDTH_M).toBe(1.0);
    expect(NEATRACING_TRACK_GRASS_APRON_WIDTH_M).toBe(3.0);
  });
});

describe('car-car elastic collision responses (LD-05, spec §4.3)', () => {
  it('exchanges velocities along the collision normal for a head-on closing pair', () => {
    const [carA, carB] = headOnOverlapPair();
    const [resolvedA, resolvedB] = resolveCarCollisions([carA, carB]);
    const [velocityA, velocityB] = [
      worldVelocityOf(resolvedA),
      worldVelocityOf(resolvedB),
    ];

    // Equal-mass frictionless exchange on the +x normal: the closing
    // relative normal velocity (+20) reverses to a separating (-20).
    expect(velocityA[0] - velocityB[0]).toBeCloseTo(-20, 6);
    // Normal momentum is conserved: zero total in, zero total out.
    expect(
      Math.hypot(velocityA[0] + velocityB[0], velocityA[1] + velocityB[1]),
    ).toBeCloseTo(0, 6);
    // Kinetic energy is conserved: half of 100 plus half of 100.
    expect(
      0.5 *
        (velocityA[0] ** 2 +
          velocityA[1] ** 2 +
          velocityB[0] ** 2 +
          velocityB[1] ** 2),
    ).toBeCloseTo(100, 6);
  });

  it('separates overlapping centers to at least the sum of collision radii', () => {
    const [carA, carB] = headOnOverlapPair();
    const [resolvedA, resolvedB] = resolveCarCollisions([carA, carB]);

    expect(
      Math.hypot(resolvedA.x - resolvedB.x, resolvedA.y - resolvedB.y),
    ).toBeGreaterThanOrEqual(2 * NEATRACING_CAR_COLLISION_RADIUS_M - 1e-6);
  });

  it('returns cars separated beyond the sum of radii byte-identically', () => {
    const nonOverlappingGapM = 2 * NEATRACING_CAR_COLLISION_RADIUS_M + 1;
    const carA = seededCarAt(-nonOverlappingGapM / 2, 0, { speed: 10 });
    const carB = seededCarAt(nonOverlappingGapM / 2, 0, {
      heading: Math.PI,
      speed: 10,
    });
    const resolved = resolveCarCollisions([carA, carB]);

    expect(serializeCars(resolved)).toBe(serializeCars([carA, carB]));
  });

  it('resolves repeated calls byte-identically without mutating the inputs', () => {
    const cars: PhysicsCarState[] = [...headOnOverlapPair()];
    const before = serializeCars(cars);
    const first = resolveCarCollisions(cars);
    const second = resolveCarCollisions(cars);

    expect(serializeCars(second)).toBe(serializeCars(first));
    expect(serializeCars(cars)).toBe(before);
    expect(first).not.toBe(cars);
  });
});

describe('track boundary bounce (spec §4.3, D6)', () => {
  /**
   * The OUTER grass-apron edge bound (halfWidth + apron) deliberately
   * pinned by the lat-5 untouched assertion below: §5.2 keeps the grass
   * apron reachable as the off-road zone inside this edge.
   */
  const outerApronBoundM =
    FIXTURE_HALF_TRACK_WIDTH_M + NEATRACING_TRACK_GRASS_APRON_WIDTH_M;

  it('reseats a car past the outer apron edge inside the bound and reflects the outward velocity', () => {
    const car = seededCarAt(FIXTURE_RIBBON_X_M, outerApronBoundM + 1, {
      heading: Math.PI / 2,
      speed: 10,
    });
    const bounced = resolveTrackBoundaryCar(car, FIXTURE_TRACK_SEGMENTS);
    const [velocityX, velocityY] = worldVelocityOf(bounced);

    expect(Math.abs(bounced.y)).toBeLessThanOrEqual(outerApronBoundM + 1e-6);
    // The outward world-velocity component (heading PI/2 → +y, outward)
    // reverses elastically while the magnitudes are preserved.
    expect(velocityY).toBeCloseTo(-10, 6);
    expect(Math.hypot(velocityX, velocityY)).toBeCloseTo(10, 6);
  });

  it('returns a car inside the bound byte-identically (lat 5 interpretation pin)', () => {
    const car = seededCarAt(FIXTURE_RIBBON_X_M, 5, { speed: 10 });
    const resolved = resolveTrackBoundaryCar(car, FIXTURE_TRACK_SEGMENTS);

    expect(serializeCars([resolved])).toBe(serializeCars([car]));
  });
});

describe('off-track surface resolver (§5.2 ribbon lateral bounds)', () => {
  const resolveSurfaceAt: (car: PhysicsCarState) => PhysicsSurfaceType =
    createTrackSurfaceResolver(FIXTURE_TRACK_SEGMENTS);

  it('resolves road inside the road core (curb band excluded)', () => {
    expect(resolveSurfaceAt(carOnRibbonAt(0))).toBe('road');
    expect(resolveSurfaceAt(carOnRibbonAt(2.5))).toBe('road');
  });

  it('resolves curb within the curb band', () => {
    expect(resolveSurfaceAt(carOnRibbonAt(3.5))).toBe('curb');
  });

  it('resolves grass off-track, including beyond the grass apron', () => {
    expect(resolveSurfaceAt(carOnRibbonAt(5))).toBe('grass');
    expect(resolveSurfaceAt(carOnRibbonAt(20))).toBe('grass');
  });
});

describe('physics wiring (resolver surfaced into stepPhysics catch-up ticks)', () => {
  it('decelerates an off-track car faster than an on-road car over idle catch-up ticks', () => {
    // One keyboard sample per frame, replayed into EVERY catch-up tick
    // (D5): the idle sample object is reused verbatim per tick.
    const idleInput: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };
    const surfaceResolver = createTrackSurfaceResolver(FIXTURE_TRACK_SEGMENTS);
    const startSpeedPerS = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 2;
    const catchUpTicks = 4;

    const runIdleTicks = (state: PhysicsCarState): PhysicsCarState => {
      let current = state;
      for (let tick = 0; tick < catchUpTicks; tick++) {
        current = stepPhysics(current, idleInput, surfaceResolver);
      }
      return current;
    };
    const roadEnd = runIdleTicks({
      ...carOnRibbonAt(2),
      speed: startSpeedPerS,
    });
    const grassEnd = runIdleTicks({
      ...carOnRibbonAt(20),
      speed: startSpeedPerS,
    });

    expect(grassEnd.speed).toBeLessThan(roadEnd.speed);
  });
});

describe('keyboard input map (D5, AC-024)', () => {
  it('maps the up key to a full positive throttle axis', () => {
    const sample: NeatRacingKeyboardSample = sampleKeyboardInput({
      ...NEUTRAL_KEYS,
      up: true,
    });

    expect(sample.throttle).toBe(1);
  });

  it('maps the down key to a full positive brake axis', () => {
    const sample = sampleKeyboardInput({ ...NEUTRAL_KEYS, down: true });

    expect(sample.brake).toBe(1);
  });

  it('maps the left key to steer +1 (counterclockwise per the steer axis contract)', () => {
    const sample = sampleKeyboardInput({ ...NEUTRAL_KEYS, left: true });

    expect(sample.steer).toBe(1);
  });

  it('maps the right key to steer -1', () => {
    const sample = sampleKeyboardInput({ ...NEUTRAL_KEYS, right: true });

    expect(sample.steer).toBe(-1);
  });

  it('maps the neutral key map to all-zero throttle, brake, and steer axes', () => {
    const sample = sampleKeyboardInput(NEUTRAL_KEYS);

    expect(sample.throttle).toBe(0);
    expect(sample.brake).toBe(0);
    expect(sample.steer).toBe(0);
  });

  it('produces byte-identical samples for identical pressed-key maps', () => {
    const sampledTwice: readonly NeatRacingKeyboardSample[] = [
      sampleKeyboardInput(THROTTLE_LEFT_KEYS),
      sampleKeyboardInput({ ...THROTTLE_LEFT_KEYS }),
    ];

    expect(JSON.stringify(sampledTwice[0])).toBe(
      JSON.stringify(sampledTwice[1]),
    );
  });

  it('applies one sampled object to every catch-up tick without consuming it', () => {
    const replayedSample = sampleKeyboardInput(THROTTLE_LEFT_KEYS);
    const start = { ...carOnRibbonAt(2), speed: 10 };
    const catchUpTicks = 4;

    // D5 burst: one sample object drives EVERY catch-up tick of a frame.
    let state = start;
    for (let tick = 0; tick < catchUpTicks; tick++) {
      state = stepPhysics(state, replayedSample);
    }
    // Fresh identical samples per tick must land on the same end state.
    let replayPerTick = start;
    for (let tick = 0; tick < catchUpTicks; tick++) {
      replayPerTick = stepPhysics(
        replayPerTick,
        sampleKeyboardInput(THROTTLE_LEFT_KEYS),
      );
    }

    expect(serializeCars([state])).toBe(serializeCars([replayPerTick]));
    expect(JSON.stringify(replayedSample)).toBe(
      JSON.stringify(sampleKeyboardInput(THROTTLE_LEFT_KEYS)),
    );
  });
});
