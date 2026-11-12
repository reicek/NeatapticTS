/**
 * ============================================================================
 * P1-outer-edge-red (plan step 14U) - the ASYMMETRIC EDGE MODEL - the
 * CONTAINMENT BOUND lane (RED-4) - CLOSED BY THE RETIREMENT
 * (fix-packet-P1-sand-red-iteration-3, AC-090 VERDICT 8).
 *
 * AC tags: AC-095 (the asymmetric edge - the named, curvature-derived
 * inner/outer half-width pair, the outer one growing with the local corner
 * measure - SURVIVES as the BAND LAW: the classifier still reads the SAME
 * widened edge), AC-081 (the containment bound is RETIRED - the
 * pose-independence, the ONE-bound and the anti-masking clauses SURVIVE as
 * properties of the retired NO-OP seam and the band law, asserted in their
 * re-authored ABSENCE form), AC-090 (the user gate is NOT claimed - this
 * file closes nothing and records no verdict).
 *
 * RED-4 (CLOSED - VERDICT 8 retired the bound): there is no containment
 * bound left to follow the widened edge, so the probe pose is NEVER
 * reseated - the retirement makes the former "the bound MOVED with the
 * edge" clause true by construction. The guard the red lane opened with
 * survives in its re-authored form: the probe POSE MUST still really sit
 * beyond the RETIRED symmetric bound and within one named curb width of
 * it, and the re-anchored clause is that the probe stays a DRIVABLE
 * off-road pose under the band law (the widened edge survives as the
 * classifier's edge) while the retired seam leaves it untouched. The
 * probe is RUNTIME-ONLY: it reads the EXISTING readers, never the
 * import-time name of a gone lane.
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
 * bounce responses for car-vs-car (LD-05) plus off-track
 * detection on the track ribbon lateral bounds (§5.2) and the D5 keyboard
 * map (sample once per rAF frame, replay into every catch-up tick). THE
 * PLAN'S EARLIER car-vs-track-boundary NON-REFLECTING CLAMP CLAUSE IS
 * RETIRED (fix-packet-P1-sand-red-iteration-3, AC-090 VERDICT 8): the
 * road edge is an inner boundary and the ONE off-track limit is the
 * USER-named sand cap (AC-094).
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
 * `NEATRACING_TRACK_GRASS_APRON_WIDTH_M` (3.0) RE-EXPORTED from the
 * track.slabs band-geometry owner (never a mirrored value), the factory
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
 * LD-04) and centers separated to at least the sum of the radii. THE
 * RETIREMENT (fix-packet-P1-sand-red-iteration-3, AC-090 VERDICT 8):
 * `resolveTrackBoundaryCar` is REMOVED - there is NO track-boundary
 * containment at all (the road edge is an INNER boundary and the ONE
 * off-track limit is the USER-named sand cap, AC-094): the band
 * classification (`classifyTrackSurface` / the band law) survives
 * - the band BETWEEN the drivable edge and the band outer edge is a
 * DRIVABLE off-road surface, so a car that crossed the road edge really
 * ends up off the road and is never reseated on the ribbon - and the
 * LIVE tick applies the RETIRED seam: EVERY track-geometry input of its
 * 4-arg form is a byte-identical NO-OP whose published pose is the
 * reached pose (pose-independence TRUE BY CONSTRUCTION, so nothing can
 * mask the AI steering defect owned by Step 14M (AC-085)).
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

  it('re-exports the track slab curb and grass-apron band widths (never a mirrored value)', () => {
    expect(NEATRACING_TRACK_CURB_WIDTH_M).toBe(1.0);
    // The apron width is the PUBLISHED rasterized band geometry (the
    // band-geometry owner publishes it and this module RE-EXPORTS it);
    // its outer edge anchored the FORMER containment bound (the
    // drivable-surface edge PLUS this apron width, AC-094 (d)) that the
    // retirement has REMOVED.
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

describe('track boundary retirement (spec §4.3, D6, AC-090 VERDICT 8 - the AC-081 bound is RETIRED)', () => {
  /**
   * The DRIVABLE-SURFACE edge of the fixture ribbon: the ribbon's
   * driveable half width (the outer edge of the road core plus the curb
   * band). The off-track BAND lies BEYOND it and is a DRIVABLE off-road
   * surface (AC-094 (b)), so a car that crossed the drivable edge is NOT
   * reseated - it keeps driving on the band.
   */
  const drivableSurfaceEdgeM = FIXTURE_HALF_TRACK_WIDTH_M;

  /**
   * The OUTER edge of the off-track band (m): the drivable-surface edge
   * plus the EXISTING named grass-apron width (AC-094 (d) - the band
   * width is an EXISTING published constant, never a band-paint
   * constant). RETIREMENT ANCHOR (AC-090 VERDICT 8): the bound once
   * seated here is REMOVED - the road edge is an INNER boundary and
   * never a wall, so this number only anchors the lanes' probe poses
   * and their "kept rolling past it" guards.
   */
  const bandOuterEdgeM =
    drivableSurfaceEdgeM + NEATRACING_TRACK_GRASS_APRON_WIDTH_M;

  it('retirement: a car beyond the former bound keeps rolling - the outward velocity is never reflected or clamped', () => {
    const car = seededCarAt(FIXTURE_RIBBON_X_M, bandOuterEdgeM + 1, {
      heading: Math.PI / 2,
      speed: 10,
    });
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    // The RETIRED seam touches NOTHING: the LIVE tick with the ribbon
    // geometry is byte-identical to the free roll, so the car keeps
    // rolling outward instead of stopping on the band edge (able to
    // fail if a containment bound ever creeps back into the tick).
    const published = stepPhysics(
      car,
      input,
      undefined,
      FIXTURE_TRACK_SEGMENTS,
    );

    expect(published).toEqual(stepPhysics(car, input));
    // The published pose IS the reached pose: beyond the former band
    // outer edge, plain NON-drivable grass by the band law, with the
    // outward world-velocity component NOT reflected.
    const [velocityX, velocityY] = worldVelocityOf(published);
    expect(Math.abs(published.y)).toBeGreaterThan(bandOuterEdgeM);
    expect(
      classifyTrackSurface(published, FIXTURE_TRACK_SEGMENTS).surface,
    ).toBe('grass');
    expect(velocityY).toBeGreaterThan(0);
    expect(velocityX).toBeCloseTo(0, 6);
  });

  it('retirement: the tick is byte-identical with the ribbon geometry at lat 2.5 (re-baselined interpretation pin)', () => {
    // lat 2.5 is INSIDE the road core (the drivable edge is 4.0 and the
    // former band outer edge 7.0), and under the retirement EVERY pose
    // gets the same NO-OP seam - the LIVE tick is byte-identical with
    // and without the track segments.
    const car = seededCarAt(FIXTURE_RIBBON_X_M, 2.5, { speed: 10 });
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    const published = stepPhysics(
      car,
      input,
      undefined,
      FIXTURE_TRACK_SEGMENTS,
    );

    expect(published).toEqual(stepPhysics(car, input));
    expect(classifyTrackSurface(car, FIXTURE_TRACK_SEGMENTS).surface).toBe(
      'road',
    );
  });

  it('retirement: a car on the band at lat 5 keeps rolling - the road edge is an inner boundary, never a wall', () => {
    // lat 5 is on the off-track band (the drivable edge is 4.0 and the
    // band outer edge is 7.0). The retirement NEVER pulls the car back:
    // the LIVE tick keeps rolling it outward, byte-identical to the
    // free roll (able to fail if a containment bound creeps back).
    const car = seededCarAt(FIXTURE_RIBBON_X_M, 5, { speed: 10 });
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    const published = stepPhysics(
      car,
      input,
      undefined,
      FIXTURE_TRACK_SEGMENTS,
    );

    expect(published).toEqual(stepPhysics(car, input));
    // ...and the reached pose really stays off the road core, yet on
    // the DRIVABLE off-road band.
    expect(Math.abs(published.y)).toBeGreaterThan(FIXTURE_HALF_TRACK_WIDTH_M);
    expect(
      classifyTrackSurface(published, FIXTURE_TRACK_SEGMENTS).isDrivable,
    ).toBe(true);
  });

  it('retirement: the tick is byte-identical with the geometry for EVERY car pose (pose-independence by construction)', () => {
    // AC-089 anti-masking, re-authored under the retirement: the SEAM
    // reads no pose at all - a beyond-band car keeps its heading, slip
    // and drift channels, so nothing can mask the AI steering defect
    // owned by Step 14M (AC-085). Byte-identity of the LIVE tick with
    // and without the track segments is the strongest form of that.
    const poses: readonly PhysicsCarState[] = [
      seededCarAt(FIXTURE_RIBBON_X_M, bandOuterEdgeM + 1, {
        heading: Math.PI / 2,
        speed: 10,
      }),
      seededCarAt(FIXTURE_RIBBON_X_M, bandOuterEdgeM + 1, {
        heading: -Math.PI / 2,
        speed: 10,
      }),
      seededCarAt(FIXTURE_RIBBON_X_M, bandOuterEdgeM + 1, {
        heading: 0,
        speed: 10,
        slipAngle: 0.4,
        lateralVelocity: -3,
      }),
    ];
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    for (const pose of poses) {
      expect(
        stepPhysics(pose, input, undefined, FIXTURE_TRACK_SEGMENTS),
      ).toEqual(stepPhysics(pose, input));
    }
  });

  it('retirement: EVERY track-geometry input of the LIVE tick is a byte-identical NO-OP (the wire proves the absence)', () => {
    // The AC-081-WIRE re-authored under the retirement: there is ONE
    // named owner of the former containment - the retired seam - and
    // any bound re-landed OUTSIDE the seam would break the byte-identity
    // below. The probe is seated BEYOND the band outer edge, so the
    // compare can never pass on a pose that never moved (the former
    // non-vacuity guard).
    const car = seededCarAt(FIXTURE_RIBBON_X_M, bandOuterEdgeM + 1, {
      heading: Math.PI / 2,
      speed: 10,
    });
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    // The unbounded tick output: the pose the retired seam never even reads.
    const unbounded = stepPhysics(car, input);
    // The LIVE tick WITH the ribbon geometry: the RETIRED seam's output.
    const published = stepPhysics(
      car,
      input,
      undefined,
      FIXTURE_TRACK_SEGMENTS,
    );

    // The published pose IS the free pose - byte for byte.
    expect(published.x).toBe(unbounded.x);
    expect(published.y).toBe(unbounded.y);
    expect(published.heading).toBe(unbounded.heading);
    expect(published.speed).toBe(unbounded.speed);
    expect(published.slipAngle).toBe(unbounded.slipAngle);
    // The NON-VACUITY GUARD comes LAST: the unbounded tick really did
    // leave the band, so the equality above can never pass on a pose
    // that never moved.
    expect(unbounded.y).toBeGreaterThan(bandOuterEdgeM);
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

  it('names the off-track band as grass and DRIVABLE, and the grass beyond the band as grass and never drivable', () => {
    // The apron width is the PUBLISHED rasterized band geometry: the
    // apron starts at the drivable edge, so its middle is a named
    // off-road sample - and the off-track band is DRIVABLE (AC-094 (b)),
    // so the band sample reads isDrivable true. The grass BEYOND the
    // band outer edge is the grass the FORMER containment bound kept
    // the car out of (the bound is retired; the band law stays).
    const apronLane = classify(
      carOnRibbonAt(
        FIXTURE_HALF_TRACK_WIDTH_M + NEATRACING_TRACK_GRASS_APRON_WIDTH_M / 2,
      ),
    );
    const farLane = classify(carOnRibbonAt(20));

    expect(apronLane.surface).toBe('grass');
    expect(apronLane.isRoadOnly).toBe(false);
    expect(apronLane.isDrivable).toBe(true);
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

  it('keeps a curb-riding car NOT road-only (the retirement never masks the road-only lane)', () => {
    // AC-089, re-authored under the retirement: the SEAM keeps the car
    // exactly where it reaches, but a curb-riding car is still NOT
    // road-only, so an off-centre AI car can never be reported as a pass
    // by the AC-085 measurement owned by Step 14M.
    const curbCar = carOnRibbonAt(3.5);
    const lane: TrackSurfaceClassification = classify(curbCar);

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
   * The OUTER edge of the off-track band (m): the drivable-surface edge
   * plus the EXISTING named grass-apron width (AC-094 (d)). RETIREMENT
   * ANCHOR (AC-090 VERDICT 8): the ONE containment bound once seated
   * here is REMOVED - this number only seats the probes BEYOND it so the
   * lanes can prove the retired seam wipes nothing there.
   */
  const bandOuterEdgeM =
    drivableSurfaceEdgeM + NEATRACING_TRACK_GRASS_APRON_WIDTH_M;

  /**
   * A SLIDING car seated beyond the band outer edge, heading outward
   * along +y: the pose the former boundary-containment branch would have
   * scrubbed. UNDER THE RETIREMENT there is NO boundary branch - the
   * sliding channels (slipAngle, lateralVelocity) are exactly what the
   * re-authored lanes prove survive the live tick unstopped.
   */
  const slidingCarBeyondBandOuterEdge = (
    channel: RedDriftChannelView,
  ): PhysicsCarState =>
    withDriftChannel(
      seededCarAt(FIXTURE_RIBBON_X_M, bandOuterEdgeM + 1, {
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

  // The TRACK-BOUNDARY scrub is RETIRED (AC-090 VERDICT 8): there is no
  // boundary-contact branch any more, so a car seated beyond the band
  // outer edge keeps its WHOLE drift channel through the live tick - the
  // retired seam wipes NOTHING.
  // [RED-P0-3-BOUNDARY-ZERO - re-authored to the retirement]
  it('retirement: a boundary-past car keeps its whole drift channel (no boundary-contact scrub exists)', () => {
    const slidingCar = slidingCarBeyondBandOuterEdge(DIRTY_DRIFT_CHANNEL);
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    // The guard the former scrub lane opened with: the probe really was
    // a sliding car beyond the band, so the byte-identity below could
    // never pass on a pose the former law ignored.
    expect(Math.abs(slidingCar.y)).toBeGreaterThan(bandOuterEdgeM);
    expect(slidingCar.slipAngle).toBe(0.2);

    // The LIVE tick with the ribbon geometry is BYTE-IDENTICAL to the
    // free roll: the boundary contact that used to wipe the channel
    // (0.2 -> 0) is RETIRED, so the sliding car keeps rolling (able to
    // fail if a boundary scrub creeps back into the tick).
    expect(
      stepPhysics(slidingCar, input, undefined, FIXTURE_TRACK_SEGMENTS),
    ).toEqual(stepPhysics(slidingCar, input));
  });

  // The latent defect the red contract hunted - a contact that zeroed
  // the visible slip but left the committed slip behind - CANNOT EXIST
  // any more: there is no boundary contact left to hide a commitment
  // behind, and the seeded slide passes through the live tick UNWIPED.
  // [RED-P0-3-LATENT-SLIP-DEFECT - re-authored to the retirement]
  it('retirement: no boundary contact exists to hide a committed slip behind (the slide passes through unwiped)', () => {
    const slidingCar = slidingCarBeyondBandOuterEdge(DIRTY_DRIFT_CHANNEL);
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    // The visible slip and the committed slip BOTH survive the live
    // tick: the retired seam never zeroes the visible slip that used to
    // mask the latent commitment.
    expect(
      stepPhysics(slidingCar, input, undefined, FIXTURE_TRACK_SEGMENTS),
    ).toEqual(stepPhysics(slidingCar, input));
    // The input really carried BOTH a visible slip and a commitment (the
    // former discriminating pairing).
    expect(slidingCar.slipAngle).toBe(0.2);
    expect(driftChannelOf(slidingCar).committed.slip).toBe(
      DIRTY_DRIFT_CHANNEL.committed.slip,
    );
  });

  // The retained guard of the scrapped contact - the lateral velocity
  // zeroed on touch - is RETIRED WITH THE CONTACT: the lateral channel
  // survives the live tick untouched.
  // [RED-P0-3-GUARD-LATERAL - re-authored to the retirement]
  it('retirement: the lateral velocity is NOT zeroed - there is no boundary contact left to zero it', () => {
    const slidingCar = slidingCarBeyondBandOuterEdge(DIRTY_DRIFT_CHANNEL);
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    // The input really carried the lateral channel the former guard
    // zeroed on touch.
    expect(slidingCar.lateralVelocity).toBe(-3);

    // The LIVE tick leaves it intact - byte-identical to the free roll.
    expect(
      stepPhysics(slidingCar, input, undefined, FIXTURE_TRACK_SEGMENTS),
    ).toEqual(stepPhysics(slidingCar, input));
  });

  // The channel of the scrapped contact was left UNCOMMITTED - turn
  // sign and re-arm token cleared. THE CONTACT IS RETIRED, so a latched
  // channel passes through the live tick STILL LATCHED: nothing exists
  // to clear it.
  // [RED-P0-3-UNCOMMITTED - re-authored to the retirement]
  it('retirement: a latched channel stays latched (no boundary contact exists to clear the commitment)', () => {
    const slidingCar = slidingCarBeyondBandOuterEdge(LATCHED_DRIFT_CHANNEL);
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    // The LIVE tick is byte-identical to the free roll: the boundary
    // contact that used to clear the turn sign and the re-arm token is
    // RETIRED.
    expect(
      stepPhysics(slidingCar, input, undefined, FIXTURE_TRACK_SEGMENTS),
    ).toEqual(stepPhysics(slidingCar, input));
    // The input really carried the latched commitment the former contact
    // cleared.
    expect(slidingCar.slipAngle).toBe(0.2);
    expect(driftChannelOf(slidingCar).committed.turnSign).toBe(
      LATCHED_DRIFT_CHANNEL.committed.turnSign,
    );
    expect(driftChannelOf(slidingCar).committed.reArmed).toBe(
      LATCHED_DRIFT_CHANNEL.committed.reArmed,
    );
  });
});

/**
 * ============================================================================
 * P1-sand-red (plan step 14W) - the BOUNDARY RE-CONTRACT - CLOSED BY THE
 * RETIREMENT (fix-packet-P1-sand-red-iteration-3, AC-090 VERDICT 8): the
 * ONE containment bound is REMOVED and the boundary law lanes are
 * re-authored to the absence form.
 *
 * AC tags: AC-094 (a - RETIRED WITH THE BOUND: there is NO boundary law
 * left that could carry the car to the middle of the track; the lanes now
 * prove the LIVE tick reseats NOTHING), AC-094 (c/d - the off-road speed
 * law is the USER-named 20 km/h cap and the ONE-bound clause is retired),
 * AC-081 (RECONCILED - the POSE-INDEPENDENCE, the ONE-BOUND and the
 * SHIPPED-TICK clauses survive as properties of the retired NO-OP seam the
 * live tick exposes), AC-090 (the user gate is NOT claimed - this lane
 * closes nothing and records no verdict).
 *
 * THE CLASSIFICATION THIS SLICE RECORDS (before any production edit): the
 * user directive ("touching the edge teleports you to the middle of the
 * track") is a BOUNDARY/MODEL defect - the BOUNDARY LAW owned the pose -
 * and NEVER a VISUAL or a PRESENTATION defect. The retirement settles the
 * classification the packet resolved: the whole containment law is
 * REMOVED (the road edge is an inner boundary), so the observable is now
 * the LIVE tick's byte-identity with the free roll.
 *
 * RED-1 (RE-AUTHORED to the absence form): a car that crosses the
 * drivable edge keeps rolling - the published lateral distance stays
 * BEYOND the drivable edge and the longitudinal progress is never reset.
 * The NON-VACUITY GUARD in the Step 14Q style survives: the probe pose
 * MUST really cross the drivable edge (a pose that stays inside the
 * drivable edge would satisfy the clause vacuously) and the guard is
 * asserted BEFORE the relational clause. The former law reseated the car
 * EXACTLY on the drivable edge, so the re-authored clause would fail
 * against any re-landed reseat.
 *
 * RED-4 (RE-AUTHORED to the absence form): the retired seam touches
 * NOTHING - a band pose and a beyond-band pose are BOTH left where they
 * reached (each the byte-identical free roll). The NON-VACUITY GUARD
 * comes FIRST: the band probe MUST really be beyond the drivable edge,
 * the beyond-band probe really beyond the band (a probe inside BOTH
 * would satisfy the clause vacuously). The POSE-INDEPENDENCE rail (the
 * AC-081 clause the retired seam keeps) rides along: the seam reads the
 * ribbon geometry ONLY - the LIVE tick is byte-identical with and
 * without it for EVERY heading and slip. The ONE-BOUND and SHIPPED-TICK
 * clauses stay pinned by the relational lanes of the retirement block
 * above (they compare the LIVE published pose against its free roll, so
 * they can never pass while a second, differently-edged bound lives in
 * the tick).
 *
 * THE BAND WIDTH IS AN EXISTING PUBLISHED CONSTANT: the band reaches from
 * the drivable edge to the OUTER edge of the EXISTING named grass apron
 * (AC-094 (d)), so the probe margins are expressed through
 * `NEATRACING_TRACK_GRASS_APRON_WIDTH_M` and NEVER through a band-paint
 * constant (AC-080 keeps the tri-colour ladder a PAINT ladder).
 *
 * The lanes are RUNTIME-ONLY: they probe the EXISTING boundary law, so
 * every red is a NAMED RUNTIME failure and NEVER an import failure.
 *
 * Standing constraints (plan: zero per-frame allocation (P9) and the
 * capacity limits maxNodes 8000 / maxConnections 32000, AC-029): this RED
 * lane adds no per-frame allocation and no node/connection growth - it
 * only PROBES the existing readers.
 */
describe('P1-sand-red (plan step 14W) - the BOUNDARY RE-CONTRACT (the drivable off-track band)', () => {
  /**
   * The drivable-surface edge of the fixture ribbon, in meters: the
   * PUBLISHED side-specific half-width of the straight chord (a straight
   * publishes the SAME inner and outer half-width, so the edge is 4.0 m).
   */
  const drivableEdgeM = FIXTURE_HALF_TRACK_WIDTH_M;

  /**
   * The OUTER edge of the off-track band: the drivable edge plus the
   * EXISTING named grass-apron width (AC-094 (d) - the band width is an
   * EXISTING published constant, never a band-paint constant).
   */
  const bandOuterEdgeM = drivableEdgeM + NEATRACING_TRACK_GRASS_APRON_WIDTH_M;

  /**
   * The band probe's lateral offset: the MIDDLE of the off-track band
   * (the drivable edge plus HALF the named apron width), so the probe is
   * beyond the drivable edge AND inside the band.
   */
  const bandProbeLateralM =
    drivableEdgeM + NEATRACING_TRACK_GRASS_APRON_WIDTH_M / 2;

  /**
   * A lateral offset BEYOND the band: the pose the retirement had to
   * leave UNTOUCHED - the car keeps rolling past the former band outer
   * edge (never clamped to it).
   */
  const beyondBandLateralM = bandOuterEdgeM + 1;

  /** The probe's x: inside the fixture chord's span (a lateral crossing). */
  const probeX = FIXTURE_RIBBON_X_M;

  it('RED-1 (RE-AUTHORED to the absence form): the tick NEVER carries the car back onto the ribbon (the published lateral distance stays BEYOND the drivable edge) and NEVER resets the longitudinal progress', () => {
    // The NON-VACUITY GUARD comes FIRST (the Step 14Q style): the probe
    // pose MUST really cross the drivable edge (a pose that stays inside
    // the drivable edge would satisfy the clause VACUOUSLY) ...
    expect(bandProbeLateralM).toBeGreaterThan(drivableEdgeM);
    // ... and it MUST really be a LATERAL crossing: the probe sits inside
    // the chord's span, so the track direction is well defined and the
    // displacement along it is measurable.
    expect([
      probeX > FIXTURE_SEGMENT.startX,
      probeX < FIXTURE_SEGMENT.endX,
    ]).toEqual([true, true]);

    // Arrange: the band pose the retired seam receives - the car is
    // heading OUTWARD.
    const car = seededCarAt(probeX, bandProbeLateralM, {
      heading: Math.PI / 2,
      speed: 10,
    });
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    // Act: the LIVE tick with the ribbon geometry - the retired seam's
    // one remaining entry point.
    const published = stepPhysics(
      car,
      input,
      undefined,
      FIXTURE_TRACK_SEGMENTS,
    );

    // RELATIONAL CLAUSE (DISCRIMINATING, re-authored): the published
    // lateral distance from the centreline is STRICTLY GREATER than the
    // drivable half-width - the car kept rolling outward, past the
    // drivable edge. The former law reseated the car EXACTLY ONTO the
    // drivable edge, so the published distance would have EQUALLED the
    // drivable half-width and the re-landed reseat would fail here (by
    // assertion, never by an import failure).
    expect(Math.abs(published.y)).toBeGreaterThan(drivableEdgeM);

    // RAIL: the displacement along the TRACK direction is ZERO for a pose
    // that only crosses the edge laterally - the tick never resets the
    // longitudinal progress either.
    expect(published.x - car.x).toBeCloseTo(0, 9);
  });

  it('RED-4 (RE-AUTHORED to the absence form): the retired seam touches NOTHING (a band pose and a beyond-band pose are BOTH left where they reached)', () => {
    // The NON-VACUITY GUARD comes FIRST: the band probe MUST really be
    // beyond the drivable edge AND inside the band, and the beyond-band
    // probe really beyond the band (a probe inside BOTH would satisfy
    // the clause VACUOUSLY - a pose the former law left alone either
    // way).
    expect([
      bandProbeLateralM > drivableEdgeM,
      bandProbeLateralM < bandOuterEdgeM,
    ]).toEqual([true, true]);
    expect(beyondBandLateralM).toBeGreaterThan(bandOuterEdgeM);

    // Arrange + Act: both poses read through the LIVE tick's retired seam
    // (the ribbon geometry argument).
    const bandCar = seededCarAt(probeX, bandProbeLateralM, {
      heading: Math.PI / 2,
      speed: 10,
    });
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };
    const bandPublished = stepPhysics(
      bandCar,
      input,
      undefined,
      FIXTURE_TRACK_SEGMENTS,
    );
    const beyondBandCar = seededCarAt(probeX, beyondBandLateralM, {
      heading: Math.PI / 2,
      speed: 10,
    });
    const beyondBandPublished = stepPhysics(
      beyondBandCar,
      input,
      undefined,
      FIXTURE_TRACK_SEGMENTS,
    );

    // RELATIONAL CLAUSE 1 (DISCRIMINATING, re-authored): a pose beyond the
    // drivable edge but INSIDE the band is NEVER reseated - the published
    // pose is the free roll, byte for byte. The former drivable-edge
    // bound would have reseated it onto the edge.
    expect(bandPublished).toEqual(stepPhysics(bandCar, input));

    // RELATIONAL CLAUSE 2 (DISCRIMINATING, re-authored): a pose beyond the
    // band is never pulled back to the band outer edge - the former
    // band-edge clamp is RETIRED and the car keeps rolling outward.
    expect(beyondBandPublished).toEqual(stepPhysics(beyondBandCar, input));
    expect(beyondBandPublished.y).toBeGreaterThan(bandOuterEdgeM);

    // RAIL (the AC-081 POSE-INDEPENDENCE the retired seam keeps): the
    // seam reads the ribbon geometry ONLY - the LIVE tick is
    // byte-identical with and without it for EVERY heading, so no
    // re-landed pose-reading bound could ever hide behind the ribbon call.
    const headingList = [0, Math.PI / 2, Math.PI, -Math.PI / 2];
    for (const heading of headingList) {
      const probe = seededCarAt(probeX, beyondBandLateralM, {
        heading,
        speed: 10,
      });
      expect(
        stepPhysics(probe, input, undefined, FIXTURE_TRACK_SEGMENTS),
      ).toEqual(stepPhysics(probe, input));
    }
  });
});

describe('the ASYMMETRIC EDGE MODEL (P1-outer-edge-red, plan step 14U) - the band law after the RETIREMENT (AC-090 VERDICT 8)', () => {
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
   * The probe distance from the corner vertex: the retired (symmetric)
   * half width plus HALF the named curb width. The margin is expressed
   * through the EXISTING curb-width constant (never a widening number),
   * so the probe sits within one curb width beyond the retired bound -
   * the pose the retirement had to leave UNTOUCHED while the band law
   * (the asymmetric edge) still classifies the widened surface around
   * it.
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
   * projection - the same measure the band geometry and the former
   * containment bound both used).
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

  it('RED-4 (RE-AUTHORED to the absence form): the RETIRED bound reseats NOTHING (the probe pose on the widened outer side is left where it reached)', () => {
    // WHY the guard: a probe that sits far beyond BOTH the retired
    // symmetric bound AND the widened band-law surface would satisfy the
    // RED-4 clause VACUOUSLY (no seam ever rebases such a car, so "the
    // pose is NOT reseated" would prove nothing at all).
    const probeChordDistanceM = Math.min(
      ...BOUND_FIXTURE_SEGMENTS.map((segment) =>
        distanceToChord(BOUND_PROBE_X, BOUND_PROBE_Y, segment),
      ),
    );

    // The NON-VACUITY GUARD comes FIRST (the Step 14Q style): the probe
    // pose MUST really be outside the retired symmetric bound AND within
    // one named curb width beyond it (so it lies inside the widened band
    // the classifier still reads - a probe far beyond both would satisfy
    // the clause VACUOUSLY), and it MUST really be on the OUTER side of
    // the corner.
    //
    // The probe really IS outside the retired symmetric bound (the retired
    // half width IS the retired drivable-surface edge).
    expect(probeChordDistanceM).toBeGreaterThan(BOUND_RETIRED_HALF_WIDTH_M);
    // ... and it lies WITHIN one named curb width beyond it: that band is
    // the room the outer widening absorbs, so the probe sits inside the
    // widened band-law surface (the contract never carries the
    // drift-space number - the margin is a FRACTION of the EXISTING
    // curb-width constant).
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

    // The pose is NEVER reseated onto an edge - the widened road survives
    // as the band law's surface (the asymmetric edge classes this probe a
    // DRIVABLE off-road or widened-road pose), and the RETIRED seam
    // leaves the pose byte for byte: the published tick IS the free roll.
    const car = seededCarAt(BOUND_PROBE_X, BOUND_PROBE_Y);
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };
    const published = stepPhysics(
      car,
      input,
      undefined,
      BOUND_FIXTURE_SEGMENTS,
    );

    // The car state is untouched - byte for byte, no pose channel moved.
    expect(published).toEqual(stepPhysics(car, input));
    // The band law still reads the widened edge: the probed apron is a
    // DRIVABLE surface (never a hard off-track classification).
    expect(
      classifyTrackSurface(published, BOUND_FIXTURE_SEGMENTS).isDrivable,
    ).toBe(true);
    // The probe pose itself is NOT reseated.
    expect(published.x).toBeCloseTo(BOUND_PROBE_X, 9);
    expect(published.y).toBeCloseTo(BOUND_PROBE_Y, 9);
  });

  it('RED-4 (PRESERVATION - re-authored to the absence form): the retired seam does not depend on the pose (it reads NO pose at all)', () => {
    // AC-081's pose-independence, re-authored under the retirement: there
    // IS no bound left to rebase, and the seam proves it by NOT touching
    // ANY pose - the LIVE tick with the (asymmetric) ribbon fixture is
    // byte-identical to the free roll for EVERY heading and slip state,
    // so no re-landed pose-reading bound could pass here unnoticed.
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
    const input: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

    const publishedHeadingA = stepPhysics(
      headingA,
      input,
      undefined,
      BOUND_FIXTURE_SEGMENTS,
    );
    const publishedHeadingB = stepPhysics(
      headingB,
      input,
      undefined,
      BOUND_FIXTURE_SEGMENTS,
    );
    const publishedSlipped = stepPhysics(
      slipped,
      input,
      undefined,
      BOUND_FIXTURE_SEGMENTS,
    );

    // EVERY probe pose is left where it reached - byte for byte.
    expect(publishedHeadingA).toEqual(stepPhysics(headingA, input));
    expect(publishedHeadingB).toEqual(stepPhysics(headingB, input));
    expect(publishedSlipped).toEqual(stepPhysics(slipped, input));
  });
});
