/**
 * ============================================================================
 * P1-outer-edge-red (plan step 14U) - the ASYMMETRIC EDGE MODEL - the
 * CONTAINMENT BOUND lane (RED-4).
 *
 * AC tags: AC-095 (the asymmetric edge - the named, curvature-derived
 * inner/outer half-width pair, the outer one growing with the local corner
 * measure), AC-081 (the RETAINED containment clauses - the pose-independence,
 * the ONE bound, the shipped tick and the anti-masking clauses are NOT
 * re-opened here; this lane only asserts that the bound MOVED with the
 * edge), AC-090 (the user gate is NOT claimed - this lane closes nothing
 * and records no verdict).
 *
 * RED-4 (DISCRIMINATING - the lane FAILS today on its own assertion): the
 * containment bound follows the SAME asymmetric OUTER edge - the bound is
 * measured against the widened outer edge, so the car is contained INSIDE
 * the widened road and never on the apron that the widening created - and
 * the observable is that the probe pose is NOT reseated. The contract opens
 * with a NON-VACUITY GUARD: the probe POSE MUST really be outside the
 * RETIRED symmetric bound and inside the widened bound, because a probe
 * that is outside both would satisfy the clause vacuously. The probe is
 * RUNTIME-ONLY: it reads the EXISTING containment bound, so the lane is red
 * BY ASSERTION and never by an import failure.
 *
 * The drift space (the measured outer extension) is a NAMED EXPORT owned by
 * the implementing slice: THIS contract NEVER carries its number - the probe
 * margin is expressed as a fraction of the EXISTING named curb-width
 * constant.
 *
 * Standing constraints (plan: zero per-frame allocation (P9) and the
 * capacity limits maxNodes 8000 / maxConnections 32000, AC-029): this RED
 * lane adds no per-frame allocation and no node/connection growth - it only
 * PROBES the existing readers.
 *
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
 * |lat| <= halfWidth - curb, curb when <= halfWidth, grass beyond
 * every band), the NAMED classification factories
 * `createTrackSurfaceClassifier(segments)` and
 * `classifyTrackSurface(car, segments)` exposing that ONE band law as a
 * `TrackSurfaceClassification` record (`surface`, `isRoadOnly`,
 * `isDrivable`) so a measurement lane can assert ROAD-ONLY instead of
 * re-deriving the bands at the call site, `resolveCarCollisions(cars)` doing
 * pairwise circular overlap detection by the D6 radius with an
 * equal-mass frictionless arcade elastic exchange along the collision
 * normal (kinetic energy and normal momentum conserved, relative normal
 * velocity reversed, no deformation, no crash penalties or rewards per
 * LD-04) and centers separated to at least the sum of the radii, and
 * `resolveTrackBoundaryCar(car, segments)` using the DRIVABLE-SURFACE
 * edge (the ribbon's driveable half width - the outer edge of the road
 * core plus the curb band) as the bound, which is the AC-081 contract of
 * Step 14I REUSED and never re-derived: the grass apron is a
 * NON-drivable off-road surface, so a car is reseated BEFORE it can come
 * to rest on it, and the bound is POSE-INDEPENDENT (it is derived from
 * the ribbon geometry only, never from the heading, the slip or the drift
 * pose) so the containment can never mask the AI steering defect owned
 * by Step 14M (AC-085): outside the bound the car is reseated on the
 * edge with its outward world-velocity component reflected elastically
 * at preserved magnitude; inside the bound the car is returned
 * byte-identical. The former untouched-at-lat-5 interpretation pin is
 * RE-BASELINED by the AC-081 contract: the inside-the-bound pin is
 * re-anchored at lat 2.5 (inside the drivable surface) and a car at
 * lat 5 is now PAST the bound, so it is reseated on the drivable edge.
 * '../host/input' exports the pure
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
  classifyTrackSurface,
  createTrackSurfaceClassifier,
  createTrackSurfaceResolver,
  NEATRACING_CAR_COLLISION_RADIUS_M,
  NEATRACING_TRACK_CURB_WIDTH_M,
  NEATRACING_TRACK_GRASS_APRON_WIDTH_M,
  resolveCarCollisions,
  resolveTrackBoundaryCar,
} from './collision.utils';
import type { TrackSurfaceClassification } from './collision.utils';
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

/**
 * The nested drift channel the tuned physics service will carry.
 * RED view: the field does not exist on `PhysicsCarState` yet, so the
 * channel is read through this typed view instead of the production
 * type — the view keeps this file compiling while the contract still
 * bites (plan slice P1-drift-feel-red, P0-3).
 */
interface RedDriftChannelView {
  readonly committed: {
    readonly slip: number;
    readonly turnSign: number;
    readonly reArmed: boolean;
  };
  readonly pose: {
    readonly pose: number;
    readonly poseRate: number;
    readonly articulation: number;
  };
}

/** Read the nested drift channel off a car state. */
const driftChannelOf = (car: PhysicsCarState): RedDriftChannelView =>
  (car as unknown as { drift: RedDriftChannelView }).drift;

/**
 * Attach a NON-ZERO drift channel to a car state: the RED fixture.
 * A bounce MUST wipe it, so the fixture starts dirty on purpose.
 */
const withDriftChannel = (
  car: PhysicsCarState,
  channel: RedDriftChannelView,
): PhysicsCarState =>
  ({ ...car, drift: channel }) as unknown as PhysicsCarState;

/** The dirty drift fixture every bounce contract starts from. */
const DIRTY_DRIFT_CHANNEL: RedDriftChannelView = {
  committed: { slip: 0.42, turnSign: -1, reArmed: false },
  pose: { pose: 0.31, poseRate: -0.8, articulation: 0.19 },
};

/**
 * The LATCHED dirty channel: the committed turn sign is set AND the
 * re-arm token is present. The bounce must clear BOTH, so the
 * "uncommitted" contract needs the token SET — a channel that arrives
 * un-armed would make the clearing unobservable.
 */
const LATCHED_DRIFT_CHANNEL: RedDriftChannelView = {
  committed: { slip: 0.42, turnSign: -1, reArmed: true },
  pose: { pose: 0.31, poseRate: -0.8, articulation: 0.19 },
};

describe('collision.utils constants (D6, spec §5.1)', () => {
  it('exposes NEATRACING_CAR_COLLISION_RADIUS_M as a finite positive named constant', () => {
    expect(Number.isFinite(NEATRACING_CAR_COLLISION_RADIUS_M)).toBe(true);
    expect(NEATRACING_CAR_COLLISION_RADIUS_M).toBeGreaterThan(0);
  });

  it('mirrors the track slab curb and grass-apron band widths', () => {
    expect(NEATRACING_TRACK_CURB_WIDTH_M).toBe(1.0);
    // The apron width mirrors the rasterized band geometry; it is NOT
    // the containment bound (the bound is the drivable-surface edge).
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

describe('track boundary bounce (spec §4.3, D6, AC-081 containment)', () => {
  /**
   * The DRIVABLE-SURFACE edge of the fixture ribbon: the ribbon's
   * driveable half width (the outer edge of the road core plus the curb
   * band). The grass apron lies BEYOND it and is a NON-drivable off-road
   * surface, so a car is reseated before it can come to rest there.
   */
  const drivableSurfaceEdgeM = FIXTURE_HALF_TRACK_WIDTH_M;

  it('reseats a car past the drivable surface edge onto the edge and reflects the outward velocity', () => {
    const car = seededCarAt(FIXTURE_RIBBON_X_M, drivableSurfaceEdgeM + 1, {
      heading: Math.PI / 2,
      speed: 10,
    });
    const bounced = resolveTrackBoundaryCar(car, FIXTURE_TRACK_SEGMENTS);
    const [velocityX, velocityY] = worldVelocityOf(bounced);

    // The car is reseated EXACTLY on the drivable surface edge, never
    // beyond it (the bound is the edge, not the grass-apron edge).
    expect(Math.abs(bounced.y)).toBeCloseTo(drivableSurfaceEdgeM, 6);
    expect(Math.abs(bounced.y)).toBeLessThanOrEqual(
      drivableSurfaceEdgeM + 1e-6,
    );
    // The outward world-velocity component (heading PI/2 → +y, outward)
    // reverses elastically while the magnitudes are preserved.
    expect(velocityY).toBeCloseTo(-10, 6);
    expect(Math.hypot(velocityX, velocityY)).toBeCloseTo(10, 6);
  });

  it('returns a car inside the bound byte-identically (re-baselined interpretation pin)', () => {
    // RE-BASELINED by the AC-081 contract: lat 2.5 is INSIDE the
    // drivable surface (the road core), so the car is untouched. The
    // former lat-5 pin asserted the outer grass-apron bound, which
    // AC-081 replaces with the drivable-surface edge.
    const car = seededCarAt(FIXTURE_RIBBON_X_M, 2.5, { speed: 10 });
    const resolved = resolveTrackBoundaryCar(car, FIXTURE_TRACK_SEGMENTS);

    expect(serializeCars([resolved])).toBe(serializeCars([car]));
  });

  it('reseats a car inside the grass apron onto the drivable surface edge (the former lat-5 case)', () => {
    // lat 5 is inside the grass apron (the drivable edge is 4.0), so
    // the car is PAST the bound and is reseated on the edge.
    const car = seededCarAt(FIXTURE_RIBBON_X_M, 5, { speed: 10 });
    const resolved = resolveTrackBoundaryCar(car, FIXTURE_TRACK_SEGMENTS);

    expect(Math.abs(resolved.y)).toBeCloseTo(drivableSurfaceEdgeM, 6);
    expect(Math.abs(resolved.y)).toBeLessThanOrEqual(
      drivableSurfaceEdgeM + 1e-6,
    );
  });

  it('reseats to the SAME position for every car pose (the bound is pose-independent)', () => {
    // AC-089: the containment bound is derived from the ribbon geometry
    // ONLY, so it can never MASK the AI steering defect owned by Step
    // 14M (AC-085) - the reseat position reads neither the heading, nor
    // the slip, nor the drift pose.
    const poses: readonly PhysicsCarState[] = [
      seededCarAt(FIXTURE_RIBBON_X_M, drivableSurfaceEdgeM + 1, {
        heading: Math.PI / 2,
        speed: 10,
      }),
      seededCarAt(FIXTURE_RIBBON_X_M, drivableSurfaceEdgeM + 1, {
        heading: -Math.PI / 2,
        speed: 10,
      }),
      seededCarAt(FIXTURE_RIBBON_X_M, drivableSurfaceEdgeM + 1, {
        heading: 0,
        speed: 10,
        slipAngle: 0.4,
        lateralVelocity: -3,
      }),
    ];

    for (const pose of poses) {
      const resolved = resolveTrackBoundaryCar(pose, FIXTURE_TRACK_SEGMENTS);

      expect(resolved.x).toBeCloseTo(FIXTURE_RIBBON_X_M, 6);
      expect(resolved.y).toBeCloseTo(drivableSurfaceEdgeM, 6);
    }
  });

  it('is the ONE bound the LIVE tick applies (the shipped tick reuses THIS bound)', () => {
    // AC-081-WIRE ("there is ONE bound in the codebase"): the live tick
    // must apply THIS bound and NOT a second, differently-edged one. The
    // lane therefore measures the LIVE published pose against the bound
    // applied to the very same unbounded tick output.
    const car = seededCarAt(FIXTURE_RIBBON_X_M, drivableSurfaceEdgeM + 1, {
      heading: Math.PI / 2,
      speed: 10,
    });
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    // The unbounded tick output: the pose the bound sees.
    const unbounded = stepPhysics(car, input);
    // The bound applied to that very pose.
    const expected = resolveTrackBoundaryCar(unbounded, FIXTURE_TRACK_SEGMENTS);
    // The LIVE tick WITH the ribbon geometry.
    const published = stepPhysics(
      car,
      input,
      undefined,
      FIXTURE_TRACK_SEGMENTS,
    );

    // The published pose IS the bound's pose - byte for byte.
    expect(published.x).toBe(expected.x);
    expect(published.y).toBe(expected.y);
    expect(published.heading).toBe(expected.heading);
    expect(published.speed).toBe(expected.speed);
    expect(published.slipAngle).toBe(expected.slipAngle);
    // The NON-VACUITY GUARD comes LAST: the unbounded tick really did
    // leave the drivable surface, so the equality above can never pass on
    // a pose that never moved.
    expect(unbounded.y).toBeGreaterThan(drivableSurfaceEdgeM);
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

  it('resolves grass off-track (the apron band and everything beyond it)', () => {
    expect(resolveSurfaceAt(carOnRibbonAt(5))).toBe('grass');
    expect(resolveSurfaceAt(carOnRibbonAt(20))).toBe('grass');
  });
});

describe('named surface classification (AC-087 - the ROAD-ONLY lane)', () => {
  const classify = createTrackSurfaceClassifier(FIXTURE_TRACK_SEGMENTS);

  it('names the road core as road - road-only and drivable', () => {
    const lane = classify(carOnRibbonAt(0));

    expect(lane.surface).toBe('road');
    expect(lane.isRoadOnly).toBe(true);
    expect(lane.isDrivable).toBe(true);
  });

  it('names the curb band as curb - drivable but NOT road-only', () => {
    const lane = classify(carOnRibbonAt(3.5));

    expect(lane.surface).toBe('curb');
    expect(lane.isRoadOnly).toBe(false);
    expect(lane.isDrivable).toBe(true);
  });

  it('names the grass apron and everything beyond it as grass - never drivable', () => {
    // The apron width is the rasterizer band mirror: the apron starts at
    // the drivable edge, so its middle is a named off-road sample.
    const apronLane = classify(
      carOnRibbonAt(
        FIXTURE_HALF_TRACK_WIDTH_M + NEATRACING_TRACK_GRASS_APRON_WIDTH_M / 2,
      ),
    );
    const farLane = classify(carOnRibbonAt(20));

    expect(apronLane.surface).toBe('grass');
    expect(apronLane.isRoadOnly).toBe(false);
    expect(apronLane.isDrivable).toBe(false);
    expect(farLane.surface).toBe('grass');
    expect(farLane.isRoadOnly).toBe(false);
    expect(farLane.isDrivable).toBe(false);
  });

  it('agrees with the bare resolver on every band sample (ONE band law)', () => {
    const resolve = createTrackSurfaceResolver(FIXTURE_TRACK_SEGMENTS);
    const lateralSamplesM = [0, 1, 2.5, 3, 3.5, 4, 4.5, 7, 20];

    for (const lateralM of lateralSamplesM) {
      const car = carOnRibbonAt(lateralM);

      expect(classify(car).surface).toBe(resolve(car));
    }
  });

  it('classifies the same car byte-identically on every call (pure)', () => {
    const car = carOnRibbonAt(3.5);
    const first = classifyTrackSurface(car, FIXTURE_TRACK_SEGMENTS);
    const second = classifyTrackSurface(car, FIXTURE_TRACK_SEGMENTS);

    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
    expect(first).toEqual(second);
  });

  it('never mutates the classified car state', () => {
    const car = carOnRibbonAt(3.5);
    const before = serializeCars([car]);

    classify(car);
    classifyTrackSurface(car, FIXTURE_TRACK_SEGMENTS);

    expect(serializeCars([car])).toBe(before);
  });

  it('keeps a curb-riding car contained yet NOT road-only (the containment never masks the road-only lane)', () => {
    // AC-089: the containment keeps the car on the drivable surface, but
    // a contained car that rides the curb is still NOT road-only, so a
    // contained-but-off-centre AI can never be reported as a pass by the
    // AC-085 measurement owned by Step 14M.
    const curbCar = carOnRibbonAt(3.5);
    const contained = resolveTrackBoundaryCar(curbCar, FIXTURE_TRACK_SEGMENTS);
    const lane: TrackSurfaceClassification = classify(contained);

    expect(serializeCars([contained])).toBe(serializeCars([curbCar]));
    expect(lane.surface).toBe('curb');
    expect(lane.isRoadOnly).toBe(false);
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

describe('drift channel reset on a collision bounce (P0-3, AC-079)', () => {
  /** The drivable-surface edge of the fixture ribbon, in meters. */
  const drivableSurfaceEdgeM = FIXTURE_HALF_TRACK_WIDTH_M;

  /**
   * A SLIDING car seated outside the drivable surface edge, heading
   * outward along +y so the elastic reflection actually fires: the
   * boundary branch only rebases an APPROACHING car, and the sliding
   * channels (slipAngle, lateralVelocity) are the ones the reset must
   * wipe.
   */
  const slidingCarOutsideDrivableSurface = (
    channel: RedDriftChannelView,
  ): PhysicsCarState =>
    withDriftChannel(
      seededCarAt(FIXTURE_RIBBON_X_M, drivableSurfaceEdgeM + 1, {
        heading: Math.PI / 2,
        speed: 10,
        slipAngle: 0.2,
        lateralVelocity: -3,
      }),
      channel,
    );

  /**
   * The file's head-on closing pair, both cars SLIDING (a non-zero slip
   * angle and a lateral velocity) and both carrying the dirty drift
   * channel: the car-car branch must wipe the channel of BOTH cars.
   */
  const dirtySlidingHeadOnPair = (): PhysicsCarState[] => {
    const [carA, carB] = headOnOverlapPair();
    return [
      withDriftChannel(
        { ...carA, slipAngle: 0.2, lateralVelocity: -3 },
        DIRTY_DRIFT_CHANNEL,
      ),
      withDriftChannel(
        { ...carB, slipAngle: 0.2, lateralVelocity: -3 },
        DIRTY_DRIFT_CHANNEL,
      ),
    ];
  };

  // The bounce recomposes the car around the exchanged world velocity
  // and zeroes the slip channels — the VISUAL pose must be zeroed with
  // them, or the car keeps the old slide's yaw while driving straight.
  // [RED-P0-3-CAR-CAR-ZERO]
  it('wipes the visible drift pose of both cars on a car-car bounce', () => {
    const [bouncedA, bouncedB] = resolveCarCollisions(dirtySlidingHeadOnPair());

    // The velocity-aligned recompose ran: the incoming slip channel of
    // the bounced car is wiped (0.2 -> 0), so the pair really bounced.
    expect(bouncedA.slipAngle).toBe(0);
    expect(driftChannelOf(bouncedA).pose.pose).toBe(0);
    expect(driftChannelOf(bouncedB).pose.pose).toBe(0);
  });

  // The channel reset covers the COMMITMENT as well as the pose: a
  // bounce that left the committed slip latched would re-enter the
  // committed chase after the bounce, with the car pointing somewhere
  // else — a phantom slide out of a wall.
  // [RED-P0-3-CAR-CAR-COMMITTED-ZERO]
  it('wipes the committed slip of both cars on a car-car bounce', () => {
    const [bouncedA, bouncedB] = resolveCarCollisions(dirtySlidingHeadOnPair());

    // The velocity-aligned recompose ran: the incoming slip channel of
    // the bounced car is wiped (0.2 -> 0), so the pair really bounced.
    expect(bouncedA.slipAngle).toBe(0);
    expect(driftChannelOf(bouncedA).committed.slip).toBe(0);
    expect(driftChannelOf(bouncedB).committed.slip).toBe(0);
  });

  // The pose spring is three coupled scalars; a bounce that zeroed the
  // offset but left the RATE or the articulation behind would keep
  // driving the spring after the car is already driving straight.
  // [RED-P0-3-CAR-CAR-POSE-RATE]
  it('wipes the pose rate and the articulation on a car-car bounce', () => {
    const [bouncedA] = resolveCarCollisions(dirtySlidingHeadOnPair());

    // The velocity-aligned recompose ran: the incoming slip channel of
    // the bounced car is wiped (0.2 -> 0), so the pair really bounced.
    expect(bouncedA.slipAngle).toBe(0);
    expect(driftChannelOf(bouncedA).pose.poseRate).toBe(0);
    expect(driftChannelOf(bouncedA).pose.articulation).toBe(0);
  });

  // The TRACK-BOUNDARY branch is the second bounce seam: the same wipe
  // must happen when the car is reseated on the drivable surface edge.
  // [RED-P0-3-BOUNDARY-ZERO]
  it('wipes the whole drift channel on a track-boundary bounce', () => {
    const bounced = resolveTrackBoundaryCar(
      slidingCarOutsideDrivableSurface(DIRTY_DRIFT_CHANNEL),
      FIXTURE_TRACK_SEGMENTS,
    );

    // The elastic reflection ran: the incoming slip channel of the
    // reseated car is wiped (0.2 -> 0).
    expect(bounced.slipAngle).toBe(0);
    expect(driftChannelOf(bounced).pose.pose).toBe(0);
    expect(driftChannelOf(bounced).committed.slip).toBe(0);
  });

  // The latent defect, asserted ABSENT: the reset zeroed slipAngle but
  // left the committed slip behind, so a bounced car kept the old
  // slide's commitment while its slip channel read zero.
  // [RED-P0-3-LATENT-SLIP-DEFECT]
  it('leaves no committed slip behind a bounce that zeroed the slip angle', () => {
    const bounced = resolveTrackBoundaryCar(
      slidingCarOutsideDrivableSurface(DIRTY_DRIFT_CHANNEL),
      FIXTURE_TRACK_SEGMENTS,
    );

    expect(bounced.slipAngle).toBe(0);
    expect(driftChannelOf(bounced).committed.slip).toBe(0);
  });

  // The retained guard: the bounce still zeroes the lateral velocity
  // exactly, so the channel wipe may not drop that pin.
  // [RED-P0-3-GUARD-LATERAL]
  it('keeps zeroing the lateral velocity on a bounce', () => {
    const bounced = resolveTrackBoundaryCar(
      slidingCarOutsideDrivableSurface(DIRTY_DRIFT_CHANNEL),
      FIXTURE_TRACK_SEGMENTS,
    );

    expect(bounced.lateralVelocity).toBe(0);
  });

  // The wipe leaves the channel UNCOMMITTED, not merely zeroed: the
  // latched turn sign is cleared and the re-arm token is cleared, so
  // the next legitimate drift has to re-latch from scratch.
  // [RED-P0-3-UNCOMMITTED]
  it('leaves the channel uncommitted after a bounce (turn sign and re-arm token cleared)', () => {
    const bounced = resolveTrackBoundaryCar(
      slidingCarOutsideDrivableSurface(LATCHED_DRIFT_CHANNEL),
      FIXTURE_TRACK_SEGMENTS,
    );

    // The elastic reflection ran: the incoming slip channel of the
    // reseated car is wiped (0.2 -> 0).
    expect(bounced.slipAngle).toBe(0);
    expect(driftChannelOf(bounced).committed.turnSign).toBe(0);
    expect(driftChannelOf(bounced).committed.reArmed).toBe(false);
  });
});

describe('the ASYMMETRIC EDGE MODEL (P1-outer-edge-red, plan step 14U) - the CONTAINMENT BOUND lane (RED-4)', () => {
  /**
   * The asymmetric-edge fixture track: a closed counter-clockwise loop with
   * FOUR 90 degree corners (P0, P3, P4, P7) and COLLINEAR vertices elsewhere,
   * so the long straight S1 (P1->P2) really has a ZERO turn at both of its
   * ends. The coordinates are the GENERATOR plane, which is also the physics
   * plane (the car state reads x/y directly).
   */
  const BOUND_FIXTURE_VERTICES: readonly (readonly [number, number])[] = [
    [-90, -30],
    [-60, -30],
    [60, -30],
    [90, -30],
    [90, 30],
    [60, 30],
    [-60, 30],
    [-90, 30],
  ];

  /** The fixture chords (generator plane: startX/startY/endX/endY + width). */
  const BOUND_FIXTURE_SEGMENTS: readonly TrackSegment[] =
    BOUND_FIXTURE_VERTICES.map((vertex, index) => {
      const next =
        BOUND_FIXTURE_VERTICES[(index + 1) % BOUND_FIXTURE_VERTICES.length]!;
      return {
        startX: vertex[0],
        startY: vertex[1],
        endX: next[0],
        endY: next[1],
        width: 8,
      };
    });

  /** The retired (symmetric) half width of the fixture chords, in meters. */
  const BOUND_RETIRED_HALF_WIDTH_M = 4;

  /** The probed corner vertex (P3), in the physics plane. */
  const BOUND_PROBE_CORNER_X = 90;
  const BOUND_PROBE_CORNER_Y = -30;

  /**
   * The probe distance from the corner vertex: the retired half width plus
   * HALF the named curb width. The margin is expressed through the EXISTING
   * curb-width constant (never a widening number), so the probe sits within
   * one curb width beyond the retired bound - the room the outer widening
   * absorbs - and therefore inside the widened bound.
   */
  const BOUND_PROBE_DISTANCE_M =
    BOUND_RETIRED_HALF_WIDTH_M + NEATRACING_TRACK_CURB_WIDTH_M / 2;

  /** How far along the corner chord the probe pose sits (meters). */
  const BOUND_PROBE_ALONG_CHORD_M = 4;

  /**
   * The probe pose: on the OUTER side of the probed corner CHORD - the chord
   * that STARTS at the probed corner (it heads north, so its OUTER side is
   * east) - at the retired half width plus HALF the named curb width, and
   * 4 m along that chord from the corner vertex.
   *
   * WHY it is NOT the corner's outer bisector: a bisector probe is
   * EQUIDISTANT from the two chords that meet at the corner, so the
   * nearest-chord reader (strict `<`, the first chord wins a tie)
   * tie-breaks the corner AWAY - to the incoming straight, whose corner
   * measure is 0 - and the lane could never go green. On the corner chord
   * the nearest chord is UNAMBIGUOUS.
   */
  const BOUND_PROBE_X = BOUND_PROBE_CORNER_X + BOUND_PROBE_DISTANCE_M;
  const BOUND_PROBE_Y = BOUND_PROBE_CORNER_Y + BOUND_PROBE_ALONG_CHORD_M;

  /**
   * The distance from a physics-plane point to a chord (the clamped
   * projection - the same measure the containment bound uses).
   */
  const distanceToChord = (
    pointX: number,
    pointY: number,
    segment: TrackSegment,
  ): number => {
    const chordX = segment.endX - segment.startX;
    const chordY = segment.endY - segment.startY;
    const chordLengthSquared = chordX * chordX + chordY * chordY;
    if (chordLengthSquared <= 0) {
      return Math.hypot(pointX - segment.startX, pointY - segment.startY);
    }
    const projection =
      ((pointX - segment.startX) * chordX +
        (pointY - segment.startY) * chordY) /
      chordLengthSquared;
    const clamped = Math.min(1, Math.max(0, projection));
    return Math.hypot(
      pointX - (segment.startX + clamped * chordX),
      pointY - (segment.startY + clamped * chordY),
    );
  };

  /**
   * The BOUND POSE of a resolved car: the position the containment bound
   * settled on, serialized. The pose-independence clause is about the BOUND
   * (the geometry-derived edge) and NOT about the whole car state: the
   * elastic reflection is POSE-DEPENDENT BY DESIGN (AC-081), so a
   * whole-state byte comparison would compare the reflection and never the
   * bound.
   */
  const boundPoseOf = (car: PhysicsCarState): string =>
    JSON.stringify({ x: car.x, y: car.y });

  it('RED-4 (DISCRIMINATING): the containment bound follows the SAME asymmetric OUTER edge (the probe pose on the widened outer side is NOT reseated)', () => {
    // WHY the guard: a probe that sits outside BOTH the retired symmetric
    // bound AND the widened bound would satisfy the RED-4 clause
    // VACUOUSLY (such a car is reseated by either bound, so "the pose is
    // NOT reseated" would prove nothing at all).
    const probeChordDistanceM = Math.min(
      ...BOUND_FIXTURE_SEGMENTS.map((segment) =>
        distanceToChord(BOUND_PROBE_X, BOUND_PROBE_Y, segment),
      ),
    );

    // The NON-VACUITY GUARD comes FIRST (the Step 14Q style): the probe
    // pose MUST really be outside the retired symmetric bound AND within
    // one named curb width beyond it (so it lies inside the widened bound -
    // a probe outside BOTH bounds would satisfy the clause VACUOUSLY), and
    // it MUST really be on the OUTER side of the corner.
    //
    // The probe really IS outside the retired symmetric bound (the retired
    // half width IS the retired drivable-surface edge).
    expect(probeChordDistanceM).toBeGreaterThan(BOUND_RETIRED_HALF_WIDTH_M);
    // ... and it lies WITHIN one named curb width beyond it: that band is
    // the room the outer widening absorbs, so the probe is inside the
    // widened bound (the contract never carries the drift-space number -
    // the margin is a FRACTION of the EXISTING curb-width constant).
    expect(probeChordDistanceM - BOUND_RETIRED_HALF_WIDTH_M).toBeLessThan(
      NEATRACING_TRACK_CURB_WIDTH_M,
    );
    // The probe really is on the OUTER side of the probed corner chord: the
    // cross product of the corner chord's heading with the offset from the
    // chord start is NEGATIVE (the outer side of a counter-clockwise loop
    // is the RIGHT of the heading).
    const cornerChord = BOUND_FIXTURE_SEGMENTS.find(
      (segment) =>
        segment.startX === BOUND_PROBE_CORNER_X &&
        segment.startY === BOUND_PROBE_CORNER_Y,
    )!;
    const cornerHeadingX = cornerChord.endX - cornerChord.startX;
    const cornerHeadingY = cornerChord.endY - cornerChord.startY;
    const cornerOffsetX = BOUND_PROBE_X - cornerChord.startX;
    const cornerOffsetY = BOUND_PROBE_Y - cornerChord.startY;

    expect(
      cornerHeadingX * cornerOffsetY - cornerHeadingY * cornerOffsetX,
    ).toBeLessThan(0);

    // The car is contained INSIDE the widened road and NEVER on the apron
    // that the widening created: the bound follows the SAME asymmetric
    // OUTER edge the road was widened to, so a pose that sits inside the
    // widened road is NOT reseated onto the retired (symmetric) edge.
    const car = seededCarAt(BOUND_PROBE_X, BOUND_PROBE_Y);
    const resolved = resolveTrackBoundaryCar(car, BOUND_FIXTURE_SEGMENTS);

    // The car state is untouched - byte for byte.
    expect(serializeCars([resolved])).toBe(serializeCars([car]));
    // The probe pose itself is NOT reseated.
    expect(resolved.x).toBeCloseTo(BOUND_PROBE_X, 9);
    expect(resolved.y).toBeCloseTo(BOUND_PROBE_Y, 9);
  });

  it('RED-4 (PRESERVATION - the bound is still resolved from the geometry only): the bound does not depend on the pose', () => {
    // AC-081's pose-independence stays GREEN and is only PINNED here: the
    // containment bound is derived from the ribbon geometry ONLY, so the
    // re-based (asymmetric) bound may never start reading the heading, the
    // slip or the drift pose of the car it bounds.
    const headingA = seededCarAt(BOUND_PROBE_X, BOUND_PROBE_Y, {
      heading: 0,
      speed: 10,
    });
    const headingB = seededCarAt(BOUND_PROBE_X, BOUND_PROBE_Y, {
      heading: Math.PI,
      speed: 10,
    });
    const slipped = seededCarAt(BOUND_PROBE_X, BOUND_PROBE_Y, {
      heading: 0,
      speed: 10,
      slipAngle: 0.4,
      lateralVelocity: -3,
    });

    const resolvedHeadingA = resolveTrackBoundaryCar(
      headingA,
      BOUND_FIXTURE_SEGMENTS,
    );
    const resolvedHeadingB = resolveTrackBoundaryCar(
      headingB,
      BOUND_FIXTURE_SEGMENTS,
    );
    const resolvedSlipped = resolveTrackBoundaryCar(
      slipped,
      BOUND_FIXTURE_SEGMENTS,
    );

    // The SAME probe pose with three different headings/slip states
    // settles on the SAME bound pose - byte for byte.
    expect(boundPoseOf(resolvedHeadingA)).toBe(boundPoseOf(resolvedHeadingB));
    expect(boundPoseOf(resolvedHeadingA)).toBe(boundPoseOf(resolvedSlipped));
  });
});
