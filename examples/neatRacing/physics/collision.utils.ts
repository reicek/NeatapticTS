/**
 * Collision / boundary / off-track surface utils for the neatRacing
 * example (plan slice P1-S7-C; AC-024 partial, spec sections 4.3, 5.1
 * D6, 5.2, LD-04/LD-05).
 *
 * Arcade collision model (LD-05): car-vs-car and car-vs-track-boundary
 * bounces are frictionless and perfectly elastic — kinetic energy and
 * normal momentum are conserved, the relative normal velocity reverses,
 * tangential velocity is preserved — with NO deformation and NO crash
 * penalties or rewards beyond the bounce itself (LD-04). Car bodies are
 * circles sized by the D6 fixed physics constant
 * {@link NEATRACING_CAR_COLLISION_RADIUS_M}, never derived from voxel
 * models, so `.bin` rebuilds stay gameplay neutral (renderer-only).
 *
 * Surface bands mirror the slab rasterizer (`../track/track.slabs`):
 * lateral bands measured from the segment chords as road
 * (`distance <= halfWidth - curb`), curb (`<= halfWidth`), and grass
 * (everything beyond the drivable surface — physics has no air
 * surface, so every point off the ribbon reads as off-track grass).
 * Junctions resolve by the rasterizer's priority rule (road before
 * curb). The same band law is exposed as a NAMED, testable
 * classification ({@link TrackSurfaceClassification}) so a traction or
 * slide measurement can be asserted ROAD-ONLY instead of re-deriving
 * the bands at the call site.
 *
 * THE EDGE IS NOT THIS MODULE'S TO DECIDE: the half-width of every
 * chord is the NAMED asymmetric half-width published by the edge seam
 * (`../track/track.edge`) - the inner half-width on the inner side of
 * a corner (the road never narrows) and the widened outer half-width on
 * the outer side - and the chord distance is the seam's canonical
 * measure. This module derives NO half-width of its own, so the
 * rasterizer, the tri-colour band paint and this bound can never
 * disagree about how WIDE the road is. The physics plane IS the
 * generator plane (the car state reads the generator x/y directly), so
 * the generator-plane entry points of the seam apply here.
 *
 * Containment (AC-081): {@link resolveTrackBoundaryCar} bounces a car
 * at the DRIVABLE surface edge (`halfWidth` — the outer edge of the
 * road core plus the curb band), so the grass apron stays a
 * NON-drivable off-road surface the car can never come to rest on. The
 * bound is POSE-INDEPENDENT (it is derived from the ribbon geometry
 * ONLY — never from the heading, the slip or the drift pose), so the
 * containment can never MASK the AI steering defect owned by Step 14M
 * (AC-085): a contained car may still wander off-centre, and a car
 * riding the curb is still not road-only.
 *
 * Determinism: every util is pure and state-free — inputs are never
 * mutated, resolved states are fresh records, and identical inputs on
 * the same runtime always produce byte-identical `JSON.stringify`
 * output (same-runtime reproducibility rung, ordered deterministic
 * arithmetic, no `Math.random`).
 */

import type {
  PhysicsCarState,
  PhysicsSurfaceType,
} from './physics.step.service';
import { createZeroDriftChannelState } from './drift.model';
import type { DriftChannelState } from './drift.model';
import type { TrackSegment } from '../../racing_curriculum/track/track.generator.types';
import {
  distanceToTrackChordM,
  resolveNearestTrackChord,
  resolveTrackEdgeHalfWidthM,
  resolveTrackEdgeProfile,
  resolveTrackEdgeSideAt,
} from '../track/track.edge';
import type { TrackEdgeProfile } from '../track/track.edge';

/**
 * Fixed arcade collision radius of every car body, in meters (D6): one
 * named fixed physics constant for the whole fleet, deliberately NOT
 * derived from voxel models (`.bin` rebuilds are renderer-only
 * artifacts with zero sim-side effect). The 1.0 m half-span stands in
 * for the ~2 m arcade car body diameter.
 */
export const NEATRACING_CAR_COLLISION_RADIUS_M = 1.0;

/**
 * Width of the curb band on each side of the road core, in meters —
 * mirrors the private `TRACK_SLAB_CURB_WIDTH_M` band constant of
 * `../track/track.slabs` so the driven surface bands and the
 * rasterized surface look agree exactly.
 */
export const NEATRACING_TRACK_CURB_WIDTH_M = 1.0;

/**
 * Width of the grass apron beyond the curb on each side, in meters —
 * mirrors the private `TRACK_SLAB_GRASS_APRON_WIDTH_M` band constant of
 * `../track/track.slabs`. The apron is a NON-drivable OFF-ROAD surface
 * (AC-081): the containment wall is the drivable-surface edge
 * ({@link resolveTrackBoundaryCar}), so a car is bounced BEFORE it can
 * sit on the apron. The width stays a named mirror of the rasterized
 * band geometry.
 */
export const NEATRACING_TRACK_GRASS_APRON_WIDTH_M = 3.0;

/** Pairwise collision diameter: circles overlap when centers sit closer. */
const CAR_COLLISION_DIAMETER_M = NEATRACING_CAR_COLLISION_RADIUS_M * 2;

/**
 * Deterministic collision-normal fallback for coincident centers: a
 * zero overlap distance has no geometric normal, so the exchange and
 * separation resolve along +x — an arbitrary but fixed axis; momentum
 * and kinetic energy stay conserved for any axis choice.
 */
const TIEBREAK_NORMAL_X = 1;

/** Y-half of the coincident-center collision-normal tiebreak (+x axis). */
const TIEBREAK_NORMAL_Y = 0;

/** Chord projection domain start (at the chord start point). */
const CHORD_PROJECTION_START = 0;

/** Chord projection domain end (at the chord end point). */
const CHORD_PROJECTION_END = 1;

/** Full-turn wrap period for heading normalization. */
const HEADING_WRAP_TAU_RADIANS = Math.PI * 2;

/**
 * Velocity magnitude below which a car counts as standing (its heading
 * stays frozen instead of snapping to the degenerate atan2 axis).
 */
const STANDING_VELOCITY_TOLERANCE_M_PER_S = 1e-6;

/** Road-core surface label for the surface resolver. */
const ROAD_SURFACE: PhysicsSurfaceType = 'road';

/** Curb-band surface label (shares the road longitudinal model here). */
const CURB_SURFACE: PhysicsSurfaceType = 'curb';

/** Off-track surface label (grass apron and everything beyond it). */
const OFF_TRACK_SURFACE: PhysicsSurfaceType = 'grass';

/**
 * The NAMED, testable surface classification of the ground under a
 * car: the surface label plus the two predicates a measurement lane
 * asserts (Step 14P, AC-087).
 *
 * The classification is the resolver's band law exposed as a RECORD, so
 * a slide, camera or traction measurement can assert that the surface
 * under the car is ROAD-ONLY instead of re-deriving the bands at the
 * call site. It is a PURE function of the track segments (the track
 * folder owns no off-track logic) and the existing resolver callers are
 * unchanged.
 */
export interface TrackSurfaceClassification {
  /**
   * The named surface label under the car: 'road' (the road core),
   * 'curb' (the curb band) or 'grass' (the non-drivable off-road
   * surface).
   */
  readonly surface: PhysicsSurfaceType;
  /**
   * TRUE only on the road core — the ROAD-ONLY measurement-lane
   * predicate (D_Q10): a slide, camera or traction measurement is
   * taken only where this reads TRUE.
   */
  readonly isRoadOnly: boolean;
  /**
   * TRUE on the DRIVABLE surface (the road core and the curb band) —
   * the surface the AC-081 containment keeps the car on. FALSE on the
   * grass apron, which is never drivable.
   */
  readonly isDrivable: boolean;
}

/** The road-core classification record (the ROAD-ONLY lane). */
const ROAD_CLASSIFICATION: TrackSurfaceClassification = Object.freeze({
  surface: ROAD_SURFACE,
  isRoadOnly: true,
  isDrivable: true,
});

/** The curb-band classification record (drivable, but NOT road-only). */
const CURB_CLASSIFICATION: TrackSurfaceClassification = Object.freeze({
  surface: CURB_SURFACE,
  isRoadOnly: false,
  isDrivable: true,
});

/** The off-track classification record (the non-drivable grass). */
const GRASS_CLASSIFICATION: TrackSurfaceClassification = Object.freeze({
  surface: OFF_TRACK_SURFACE,
  isRoadOnly: false,
  isDrivable: false,
});

/**
 * One resolved car-car pair: fresh states that replace the pair's
 * slots in the resolved cars list, in the same order.
 */
interface ResolvedCarPair {
  /** Replacement state for the first (lower-index) car of the pair. */
  readonly carA: PhysicsCarState;
  /** Replacement state for the second (higher-index) car of the pair. */
  readonly carB: PhysicsCarState;
}

/** Unit collision normal pointing from car A toward car B. */
interface CollisionNormal {
  readonly normalX: number;
  readonly normalY: number;
}

/**
 * Exchanged pair of world velocities after the normal-axis elastic
 * impulse (tangential components untouched).
 */
interface VelocityExchange {
  readonly velocityAX: number;
  readonly velocityAY: number;
  readonly velocityBX: number;
  readonly velocityBY: number;
}

/**
 * Nearest on-chord point of one segment relative to a world point.
 * The shared geometry step of the surface bands (distance from the
 * chord) and the boundary bounce (reseat point and outward normal).
 */
interface ChordPointSample {
  /** Nearest on-chord point x, in meters. */
  readonly pointX: number;
  /** Nearest on-chord point y, in meters. */
  readonly pointY: number;
  /**
   * The PUBLISHED half-width of the side the point sits on, in meters: the
   * inner half-width on the inner side of the road, the widened outer
   * half-width on the outer side (published by the edge seam).
   */
  readonly halfWidthM: number;
  /** Euclidean distance from the world point to the nearest chord point, in meters. */
  readonly distanceM: number;
}

/**
 * Creates a pure surface resolver for a fixed track geometry: the
 * returned probe maps a car state to the physics surface under it with
 * the same nearest-chord lateral bands the slab rasterizer paints
 * (junction priority: road before curb; beyond every band the car is
 * off-track grass).
 *
 * The resolver reads the SAME band law as the named classification
 * ({@link createTrackSurfaceClassifier}) — there is ONE band
 * implementation, never two — and the existing callers are unchanged:
 * the probe still answers with the bare surface label, so it stays
 * safe to pass straight into `stepPhysics` as its optional surface
 * resolver.
 *
 * @param segments - Closed-loop ribbon segments (ordered, read only).
 * @returns A pure `(car) => PhysicsSurfaceType` probe — the same
 *   segments always resolve to the same surface.
 *
 * @example
 * ```ts
 * const surfaceResolver = createTrackSurfaceResolver(spec.segments);
 * const surface = surfaceResolver(car); // 'road' | 'curb' | 'grass'
 * ```
 */
export function createTrackSurfaceResolver(
  segments: readonly TrackSegment[],
): (car: PhysicsCarState) => PhysicsSurfaceType {
  const classify = createTrackSurfaceClassifier(segments);
  return (car: PhysicsCarState) => classify(car).surface;
}

/**
 * Creates the pure NAMED surface classifier for a fixed track
 * geometry: the returned probe maps a car state to the
 * {@link TrackSurfaceClassification} of the ground under it, so a
 * traction or slide measurement can be asserted ROAD-ONLY (AC-087)
 * instead of re-deriving the bands at the call site.
 *
 * The classifier is a PURE function of the track segments (the track
 * folder owns no off-track logic), it never mutates its inputs, and
 * identical inputs always produce the identical record.
 *
 * @param segments - Closed-loop ribbon segments (ordered, read only).
 * @returns A pure `(car) => TrackSurfaceClassification` probe.
 *
 * @example
 * ```ts
 * const classify = createTrackSurfaceClassifier(spec.segments);
 * const lane = classify(car);
 * console.log(lane.surface, lane.isRoadOnly); // 'road' true
 * ```
 */
export function createTrackSurfaceClassifier(
  segments: readonly TrackSegment[],
): (car: PhysicsCarState) => TrackSurfaceClassification {
  // The edge profile is resolved ONCE for the whole probe: it is the
  // published width profile every probe measures the road through.
  const edgeProfile = resolveTrackEdgeProfile(segments.slice());
  return (car: PhysicsCarState) => classifySurfaceAt(car.x, car.y, edgeProfile);
}

/**
 * Classifies the ground under ONE car state: the single-shot form of
 * {@link createTrackSurfaceClassifier}, over the same band law and the
 * same pure function of the track segments.
 *
 * @param car - Car state, treated read-only; never mutated.
 * @param segments - Closed-loop ribbon segments (ordered, read only).
 * @returns The {@link TrackSurfaceClassification} of the ground under
 *   the car.
 *
 * @example
 * ```ts
 * const lane = classifyTrackSurface(car, spec.segments);
 * console.log(lane.isRoadOnly); // the ROAD-ONLY measurement lane
 * ```
 */
export function classifyTrackSurface(
  car: PhysicsCarState,
  segments: readonly TrackSegment[],
): TrackSurfaceClassification {
  return classifySurfaceAt(
    car.x,
    car.y,
    resolveTrackEdgeProfile(segments.slice()),
  );
}

/**
 * Resolves every pairwise car-car overlap with the LD-05 arcade
 * elastic bounce: overlapping circular bodies (D6 radius) exchange
 * world velocities along the collision normal (equal masses, no
 * friction — kinetic energy and normal momentum are conserved, the
 * relative normal velocity reverses, tangential velocity is
 * preserved) and are then pushed apart to exactly the sum of the
 * radii, half per car. Only an approaching overlap takes an impulse —
 * a separating overlap still separates positionally without adding
 * energy. No deformation or reward applies (LD-04).
 *
 * Pair resolution order follows car index order (A-B, A-C, B-C, ...),
 * so multi-car pile-ups resolve deterministically.
 *
 * @param cars - Car states, treated read-only; never mutated.
 * @returns Fresh car states in the same order; non-overlapping cars
 *   are returned byte-identical (same values, new records).
 *
 * @example
 * ```ts
 * const [resolvedA, resolvedB] = resolveCarCollisions([carA, carB]);
 * console.log(
 *   Math.hypot(resolvedA.x - resolvedB.x, resolvedA.y - resolvedB.y),
 * ); // at least 2 * NEATRACING_CAR_COLLISION_RADIUS_M
 * ```
 */
export function resolveCarCollisions(
  cars: readonly PhysicsCarState[],
): PhysicsCarState[] {
  const resolved = cars.map((car) => ({ ...car }));
  resolvePairImpulsesInOrder(resolved);
  return resolved;
}

/**
 * Bounces a car off the DRIVABLE surface edge of the nearest ribbon
 * segment (LD-05 arcade elastic, boundaries per spec section 5.2; the
 * AC-081 containment contract).
 *
 * The bound is the outer edge of the drivable band — the road core plus
 * the curb band (the PUBLISHED half-width of the side the car sits on:
 * the inner half-width on the inner side of the road, the widened outer
 * half-width on the outer side, both published by the edge seam). The
 * grass apron is a NON-drivable off-road surface, so a car is reseated
 * BEFORE it can come to rest on it. The bound is POSE-INDEPENDENT: it
 * is derived from the ribbon geometry ONLY (never from the heading, the
 * slip or the drift pose), so the containment can never MASK the AI
 * steering defect owned by Step 14M (AC-085) — a contained car may
 * still wander off-centre, and a car riding the curb is still not
 * road-only.
 *
 * Inside the bound (`distance <= halfWidthM`, the published side-specific
 * half-width) the car is returned byte-identical. Outside the bound the
 * car is reseated exactly on the drivable surface edge along its outward
 * chord normal, and only an approaching car (positive outward velocity
 * component) has that outward component negated elastically at preserved
 * magnitude (tangential velocity preserved); a car already moving inward
 * keeps every velocity-channel field untouched.
 *
 * @param car - Car state, treated read-only; never mutated.
 * @param segments - Closed-loop ribbon segments; the loop is non-empty
 *   and every segment carries a positive driveable width (the
 *   `TrackSegment` domain).
 * @returns A fresh car state inside the drivable-surface bound; an
 *   approaching car is bounced through the velocity-aligned
 *   recomposition (whose fields satisfy the `PhysicsCarState` contract:
 *   `speed >= 0`, zero slip/lateral drift), while a car already moving
 *   inward keeps its original velocity channels as-is.
 * @throws Error when `segments` is empty (no ribbon to bound against).
 *
 * @example
 * ```ts
 * const bounced = resolveTrackBoundaryCar(approachingCar, spec.segments);
 * console.log(bounced.lateralVelocity === 0); // recomposed, no drift
 * ```
 */
export function resolveTrackBoundaryCar(
  car: PhysicsCarState,
  segments: readonly TrackSegment[],
): PhysicsCarState {
  const edgeProfile = resolveTrackEdgeProfile(segments.slice());
  const nearestChord = resolveNearestChordPoint(car.x, car.y, edgeProfile);
  // The drivable-surface edge: the outer edge of the road core plus the
  // curb band. Geometry ONLY — the bound never reads the car pose.
  const drivableSurfaceEdgeM = nearestChord.halfWidthM;

  // Step 1: Inside the drivable surface — the car keeps everything.
  if (nearestChord.distanceM <= drivableSurfaceEdgeM) {
    return { ...car };
  }

  // Step 2: Reseat on the drivable surface edge; the helper reflects
  // only an approaching outward component.
  return reboundPastDrivableSurfaceEdge(
    car,
    nearestChord,
    drivableSurfaceEdgeM,
  );
}

/** Applies the pairwise elastic exchange and separation in index order. */
function resolvePairImpulsesInOrder(cars: PhysicsCarState[]): void {
  for (let indexA = 0; indexA < cars.length; indexA += 1) {
    for (let indexB = indexA + 1; indexB < cars.length; indexB += 1) {
      const pairResolution = resolveOverlappingPair(cars[indexA], cars[indexB]);
      if (pairResolution !== null) {
        cars[indexA] = pairResolution.carA;
        cars[indexB] = pairResolution.carB;
      }
    }
  }
}

/**
 * Resolves one overlapping car pair: geometry, normal elastic exchange,
 * and half-per-car re-separation. Returns `null` when the pair does not
 * overlap (kept untouched).
 */
function resolveOverlappingPair(
  carA: PhysicsCarState,
  carB: PhysicsCarState,
): ResolvedCarPair | null {
  const rawOffsetX = carB.x - carA.x;
  const rawOffsetY = carB.y - carA.y;
  const centerDistance = Math.hypot(rawOffsetX, rawOffsetY);

  if (centerDistance >= CAR_COLLISION_DIAMETER_M) {
    return null;
  }

  // Step 1: Exchange the world velocities along the collision normal.
  const collisionNormal = resolveCollisionNormal(
    rawOffsetX,
    rawOffsetY,
    centerDistance,
  );
  const exchange = resolveNormalElasticExchange(carA, carB, collisionNormal);

  // Step 2: Push the centers apart to exactly the sum of the radii.
  const halfOverlap = (CAR_COLLISION_DIAMETER_M - centerDistance) / 2;
  const displacedA = displaceCar(
    carA,
    -halfOverlap * collisionNormal.normalX,
    -halfOverlap * collisionNormal.normalY,
  );
  const displacedB = displaceCar(
    carB,
    halfOverlap * collisionNormal.normalX,
    halfOverlap * collisionNormal.normalY,
  );

  // Step 3: Recompose each displaced state around its bounced velocity.
  return {
    carA: recomposeCarWithVelocity(
      displacedA,
      exchange.velocityAX,
      exchange.velocityAY,
    ),
    carB: recomposeCarWithVelocity(
      displacedB,
      exchange.velocityBX,
      exchange.velocityBY,
    ),
  };
}

/**
 * Resolves the pair collision normal (points from car A toward car B),
 * falling back to the fixed +x tiebreak axis for coincident centers.
 */
function resolveCollisionNormal(
  rawOffsetX: number,
  rawOffsetY: number,
  centerDistance: number,
): CollisionNormal {
  if (centerDistance === 0) {
    return { normalX: TIEBREAK_NORMAL_X, normalY: TIEBREAK_NORMAL_Y };
  }
  return {
    normalX: rawOffsetX / centerDistance,
    normalY: rawOffsetY / centerDistance,
  };
}

/**
 * Exchanges the paired world velocities along the collision normal for
 * the equal-mass frictionless arcade bounce: only an approaching pair
 * (`(vA - vB) · n > 0`) takes an impulse; a separating pair keeps its
 * velocities (no added energy).
 */
function resolveNormalElasticExchange(
  carA: PhysicsCarState,
  carB: PhysicsCarState,
  collisionNormal: CollisionNormal,
): VelocityExchange {
  const [velocityAX, velocityAY] = resolveWorldVelocity(carA);
  const [velocityBX, velocityBY] = resolveWorldVelocity(carB);
  const relativeNormal =
    (velocityAX - velocityBX) * collisionNormal.normalX +
    (velocityAY - velocityBY) * collisionNormal.normalY;

  if (relativeNormal <= 0) {
    return { velocityAX, velocityAY, velocityBX, velocityBY };
  }

  const impulseX = relativeNormal * collisionNormal.normalX;
  const impulseY = relativeNormal * collisionNormal.normalY;
  return {
    velocityAX: velocityAX - impulseX,
    velocityAY: velocityAY - impulseY,
    velocityBX: velocityBX + impulseX,
    velocityBY: velocityBY + impulseY,
  };
}

/** Offsets a car state's world position (pure; fresh record). */
function displaceCar(
  car: PhysicsCarState,
  offsetX: number,
  offsetY: number,
): PhysicsCarState {
  return { ...car, x: car.x + offsetX, y: car.y + offsetY };
}

/**
 * World velocity `[vx, vy]` of a car state, in m/s: the heading-
 * longitudinal axis plus the perpendicular lateral axis — the exact
 * velocity algebra the red contract's `worldVelocityOf` helper (and
 * `stepPhysics` translation) shares, so bounces act on the same
 * velocity components the integrator moves.
 */
function resolveWorldVelocity(car: PhysicsCarState): readonly [number, number] {
  return [
    Math.cos(car.heading + car.slipAngle) * car.speed -
      Math.sin(car.heading) * car.lateralVelocity,
    Math.sin(car.heading + car.slipAngle) * car.speed +
      Math.cos(car.heading) * car.lateralVelocity,
  ];
}

/**
 * The fully ZEROED drift channel of a bounced car (P0-3, AC-079).
 *
 * A bounce is a RELEASE, not a spawn: the latch is cleared AND the
 * RE-ARM token is cleared, so the next legitimate drift has to re-latch
 * from scratch (the drift model re-latches the token when the wheel goes
 * neutral). The spawn factory arms the token TRUE - that is the SPAWN
 * contract - so the bounce reset derives its own zeroed channel from the
 * factory shape instead of reusing the armed spawn channel.
 *
 * @returns A zeroed drift channel (every numeric field 0, the token
 *   cleared).
 */
function zeroBouncedDriftChannel(): DriftChannelState {
  const spawned = createZeroDriftChannelState();
  return {
    committed: { ...spawned.committed, reArmed: false },
    pose: { ...spawned.pose },
  };
}

/**
 * Rebuilds a car state around a target world velocity: the velocity-
 * aligned heading representation (speed = world-velocity magnitude,
 * heading = its direction, drift channels zeroed because the
 * frictionless bounce carries no slip or lateral component). Every
 * recomposed field stays inside the `PhysicsCarState` contract
 * (`speed >= 0`, zero slip/lat) so the next `stepPhysics` tick
 * integrates the bounced state exactly as written. A standing target
 * (magnitude within tolerance) keeps the heading untouched instead of
 * snapping to a degenerate `atan2` axis. The drift channel is zeroed on
 * both branches (a bounce is a release, so the latch and the RE-ARM token
 * are cleared).
 */
function recomposeCarWithVelocity(
  car: PhysicsCarState,
  velocityX: number,
  velocityY: number,
): PhysicsCarState {
  const speedMagnitude = Math.hypot(velocityX, velocityY);

  if (speedMagnitude <= STANDING_VELOCITY_TOLERANCE_M_PER_S) {
    return {
      ...car,
      speed: 0,
      slipAngle: 0,
      lateralVelocity: 0,
      drift: zeroBouncedDriftChannel(),
    };
  }

  return {
    ...car,
    heading: wrapHeading(Math.atan2(velocityY, velocityX)),
    speed: speedMagnitude,
    slipAngle: 0,
    lateralVelocity: 0,
    drift: zeroBouncedDriftChannel(),
  };
}

/**
 * Reseats a car outside the drivable surface edge back onto the bound
 * along its outward chord normal. Only an approaching car — a positive
 * outward world-velocity component (`v · outward > 0`) — takes the
 * elastic reflection (`v - 2(v·n)n`, tangential velocity preserved)
 * followed by the velocity-aligned recomposition. A car already moving
 * inward along the normal (or tangent, `v · outward <= 0`) leaves the
 * bound by itself, so its reseated state keeps every velocity-channel
 * field untouched (no recompose, no added energy — the same
 * approaching-only impulse rule as the car-car exchange).
 *
 * The reseat POSITION depends on the ribbon geometry only (the chord
 * point and the bound), never on the car pose, so the AC-081
 * containment is POSE-INDEPENDENT.
 */
function reboundPastDrivableSurfaceEdge(
  car: PhysicsCarState,
  nearestChord: ChordPointSample,
  drivableSurfaceEdgeM: number,
): PhysicsCarState {
  const outwardNormalX = (car.x - nearestChord.pointX) / nearestChord.distanceM;
  const outwardNormalY = (car.y - nearestChord.pointY) / nearestChord.distanceM;
  const [velocityX, velocityY] = resolveWorldVelocity(car);
  const outwardComponent =
    velocityX * outwardNormalX + velocityY * outwardNormalY;
  const reseatedCar: PhysicsCarState = {
    ...car,
    x: nearestChord.pointX + outwardNormalX * drivableSurfaceEdgeM,
    y: nearestChord.pointY + outwardNormalY * drivableSurfaceEdgeM,
  };

  // Step 1: An inward- or tangent-moving car already leaves the bound,
  // so its velocity channels stay untouched (no recompose, no impulse).
  if (outwardComponent <= 0) {
    return reseatedCar;
  }

  // Step 2: Only the approaching (outward) component takes the elastic
  // reflection before the velocity-aligned recomposition.
  return recomposeCarWithVelocity(
    reseatedCar,
    velocityX - 2 * outwardComponent * outwardNormalX,
    velocityY - 2 * outwardComponent * outwardNormalY,
  );
}

/**
 * Resolves the nearest chord point across the whole ribbon (minimum
 * distance wins, the FIRST chord wins a tie - the seam's published
 * nearest-chord rule, which is the SAME policy this bound already used).
 * The ribbon is a closed loop, so at least one segment is required.
 *
 * The DISTANCE and the half-width both come from the edge seam: the
 * nearest chord is the seam's, the distance is the seam's canonical
 * chord measure and the half-width is the PUBLISHED half-width of the
 * side the point sits on. Only the nearest POINT on that chord is
 * resolved here, because the reseat needs the point the outward normal
 * is measured from.
 */
function resolveNearestChordPoint(
  pointX: number,
  pointY: number,
  edgeProfile: TrackEdgeProfile,
): ChordPointSample {
  if (edgeProfile.chords.length === 0) {
    throw new Error(
      'resolveTrackBoundaryCar requires at least one track segment',
    );
  }

  const nearest = resolveNearestTrackChord(edgeProfile, pointX, pointY);
  const chord = edgeProfile.chords[nearest.segmentIndex];
  const segment = edgeProfile.segments[nearest.segmentIndex];
  if (chord === undefined || segment === undefined) {
    throw new Error(
      'resolveTrackBoundaryCar requires at least one track segment',
    );
  }

  const nearestPoint = resolveNearestPointOnChord(pointX, pointY, segment);

  return {
    pointX: nearestPoint.x,
    pointY: nearestPoint.y,
    halfWidthM: resolveTrackEdgeHalfWidthM(
      chord,
      resolveTrackEdgeSideAt(edgeProfile, nearest.segmentIndex, pointX, pointY),
    ),
    distanceM: nearest.distanceM,
  };
}

/**
 * Classifies one ground-plane point with the slab rasterizer's band and
 * priority rules: road core, then curb, then grass (physics has no air
 * surface, so beyond every band the car is off-track grass). The bands
 * are the rasterizer's UNION rule (a point is road when ANY segment's
 * road core holds it), and the result is the NAMED classification
 * record — the ONE band law behind both the bare resolver and the
 * classifier.
 *
 * The half-width of every chord is the PUBLISHED half-width of the side
 * the point sits on, so the road is WIDER on the outer side of a corner
 * and never narrower on the inner side.
 */
function classifySurfaceAt(
  pointX: number,
  pointY: number,
  edgeProfile: TrackEdgeProfile,
): TrackSurfaceClassification {
  let inRoadCore = false;
  let inCurbBand = false;

  for (
    let segmentIndex = 0;
    segmentIndex < edgeProfile.chords.length;
    segmentIndex += 1
  ) {
    const chord = edgeProfile.chords[segmentIndex];
    const segment = edgeProfile.segments[segmentIndex];
    if (chord === undefined || segment === undefined) {
      continue;
    }

    const halfWidthM = resolveTrackEdgeHalfWidthM(
      chord,
      resolveTrackEdgeSideAt(edgeProfile, segmentIndex, pointX, pointY),
    );
    const distanceM = distanceToTrackChordM(segment, pointX, pointY);

    if (distanceM <= halfWidthM - NEATRACING_TRACK_CURB_WIDTH_M) {
      inRoadCore = true;
    } else if (distanceM <= halfWidthM) {
      inCurbBand = true;
    }
  }

  if (inRoadCore) {
    return ROAD_CLASSIFICATION;
  }
  if (inCurbBand) {
    return CURB_CLASSIFICATION;
  }
  return GRASS_CLASSIFICATION;
}

/**
 * Resolves the nearest ON-CHORD POINT of one chord (the clamped
 * projection the reseat needs, mirroring the slab rasterizer's clamped
 * chord projection). The DISTANCE to that chord is the seam's canonical
 * measure, so only the point itself is resolved here.
 */
function resolveNearestPointOnChord(
  pointX: number,
  pointY: number,
  segment: Readonly<TrackSegment>,
): { readonly x: number; readonly y: number } {
  const chordX = segment.endX - segment.startX;
  const chordY = segment.endY - segment.startY;
  const chordLengthSquared = chordX * chordX + chordY * chordY;

  if (chordLengthSquared <= 0) {
    // A degenerate point chord has no projection direction: its whole
    // geometry is the start point.
    return { x: segment.startX, y: segment.startY };
  }

  const projection =
    ((pointX - segment.startX) * chordX + (pointY - segment.startY) * chordY) /
    chordLengthSquared;
  const clampedProjection = Math.min(
    CHORD_PROJECTION_END,
    Math.max(CHORD_PROJECTION_START, projection),
  );

  return {
    x: segment.startX + clampedProjection * chordX,
    y: segment.startY + clampedProjection * chordY,
  };
}

/**
 * Wraps a heading into `[-π, π)` with one O(1) pass — the local mirror
 * of the private header wrap in `physics.step.service` (the P1-S7-B
 * slice boundary owns that module, so this file re-states the wrap).
 */
function wrapHeading(radians: number): number {
  return (
    radians -
    HEADING_WRAP_TAU_RADIANS *
      Math.floor((radians + Math.PI) / HEADING_WRAP_TAU_RADIANS)
  );
}
