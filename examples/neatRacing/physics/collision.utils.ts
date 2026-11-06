/**
 * Collision / boundary / off-track surface utils for the neatRacing
 * example (plan slice P1-S7-C; AC-024 partial, spec sections 4.3, 5.1
 * D6, 5.2, LD-04/LD-05).
 *
 * Arcade collision model (LD-05): the car-vs-car bounce is frictionless
 * and perfectly elastic — kinetic energy and normal momentum are
 * conserved, the relative normal velocity reverses, tangential velocity
 * is preserved — with NO deformation and NO crash penalties or rewards
 * beyond the bounce itself (LD-04). The car-vs-track-boundary law is
 * NOT a bounce and NOT a wall: there is NO track-boundary containment
 * at all (the AC-090 VERDICT-8 supersession), so a car that leaves the
 * ribbon really goes off track and keeps rolling. Car bodies are
 * circles sized by the D6 fixed physics constant
 * {@link NEATRACING_CAR_COLLISION_RADIUS_M}, never derived from voxel
 * models, so `.bin` rebuilds stay gameplay neutral (renderer-only).
 *
 * Surface bands ARE the slab rasterizer's bands (`../track/track.slabs`):
 * lateral bands measured from the segment chords as road
 * (`distance <= halfWidth - curb`), curb (`<= halfWidth`), the
 * DRIVABLE off-track band (`<= halfWidth + apron`) and the grass
 * beyond it (physics has no air surface, so every point off the
 * ribbon reads as off-track grass). Junctions resolve by the
 * rasterizer's priority rule (road before curb before grass). The
 * band WIDTHS are not mirrored here - they are READ from the
 * band-geometry owner ({@link NEATRACING_TRACK_GRASS_APRON_WIDTH_M}),
 * so the rasterizer and this classifier can never disagree about where
 * a band ends. The same band law is exposed as a NAMED, testable
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
 * rasterizer, the tri-colour band paint and this module can never
 * disagree about how WIDE the road is. The physics plane IS the
 * generator plane (the car state reads the generator x/y directly), so
 * the generator-plane entry points of the seam apply here.
 *
 * NO BOUND (AC-090 VERDICT 8, superseding AC-081/AC-094 (d)): the
 * track has NO invisible wall. The off-track band is DRIVABLE off-road
 * (the EXISTING 'grass' surface member) and the grass BEYOND the band
 * is drivable TOO - the car crosses the edge, keeps rolling, and is
 * limited only by the off-track SPEED cap (the NAMED 20 km/h off-track
 * limit owned by the physics step service), never by a geometric bound.
 * The road edge is therefore an INNER boundary only: a car that has
 * crossed the edge really ends up off the road, and no law ever reseats
 * it on the ribbon nor resets its longitudinal progress. Because there
 * is no bound at all, there is nothing that could MASK the AI steering
 * defect owned by Step 14M (AC-085) either: a car that has left the
 * ribbon is simply off the ribbon, and a car riding the curb is still
 * not road-only.
 *
 * Determinism: every util is pure and state-free — inputs are never
 * mutated, resolved states are fresh records, and identical inputs on
 * the same runtime always produce byte-identical `JSON.stringify`
 * output (same-runtime reproducibility rung, ordered deterministic
 * arithmetic, no `Math.random`). The ONE exception is the
 * CONTENT-GUARDED edge-profile memo
 * ({@link resolveEdgeProfileForSegments}): a pure cache that changes
 * only HOW OFTEN the published profile is resolved, NEVER the resolved
 * values themselves.
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
  resolveTrackEdgeHalfWidthM,
  resolveTrackEdgeProfile,
  resolveTrackEdgeSideAt,
} from '../track/track.edge';
import type { TrackEdgeProfile } from '../track/track.edge';
import {
  NEATRACING_TRACK_CURB_WIDTH_M,
  NEATRACING_TRACK_GRASS_APRON_WIDTH_M,
} from '../track/track.slabs';

/**
 * Fixed arcade collision radius of every car body, in meters (D6): one
 * named fixed physics constant for the whole fleet, deliberately NOT
 * derived from voxel models (`.bin` rebuilds are renderer-only
 * artifacts with zero sim-side effect). The 1.0 m half-span stands in
 * for the ~2 m arcade car body diameter.
 */
export const NEATRACING_CAR_COLLISION_RADIUS_M = 1.0;

/**
 * Width of the curb band on each side of the road core, in meters - the
 * PUBLISHED curb-band width, owned by the band-geometry owner
 * (`../track/track.slabs`) and RE-EXPORTED here, so the driven surface
 * bands and the rasterized surface agree exactly by construction instead
 * of by a mirrored value.
 */
export { NEATRACING_TRACK_CURB_WIDTH_M };

/**
 * Width of the grass apron beyond the curb on each side, in meters - the
 * PUBLISHED width of the OFF-TRACK BAND, owned by the band-geometry owner
 * (`../track/track.slabs`) and RE-EXPORTED here, so the rasterizer and the
 * physics classifier read ONE band boundary (AC-094 (e)).
 *
 * The apron IS the off-track band: it is DRIVABLE off-road (the EXISTING
 * 'grass' surface member). It is a SURFACE the car can drive on, not a
 * boundary that stops it - the car may roll past the band outer edge
 * onto the open grass (there is NO containment bound), and the only
 * limit out there is the off-track speed cap.
 */
export { NEATRACING_TRACK_GRASS_APRON_WIDTH_M };

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

/** Off-track surface label (the drivable band and the grass beyond it). */
const OFF_TRACK_SURFACE: PhysicsSurfaceType = 'grass';

/**
 * The CONTENT-GUARDED edge-profile memo: ONE resolved edge profile per
 * segments-ARRAY IDENTITY.
 *
 * WHY it exists: the edge profile is the PUBLISHED width profile every
 * reader measures the road through, and resolving it ALLOCATES (the
 * seam snapshots the segments and builds one chord per segment). The
 * surface classifier runs on the LIVE TICK path - once per car, per tick
 * - so resolving a FRESH profile there allocated four arrays plus two
 * objects per segment on EVERY frame, breaking the
 * resolve-the-profile-ONCE law this same file already follows and the
 * standing zero-per-frame-allocation constraint.
 *
 * WHY the ARRAY IDENTITY is the key: the shipped page hands ONE stable
 * `trackSpec.segments` reference to every tick, so the identity is the
 * ONE key that is stable across frames AND exact for the caller's array.
 * The memo is CONTENT-GUARDED ({@link isEdgeProfileCurrent}), so a
 * profile is NEVER served for an array whose content changed - the
 * changed content is re-resolved instead. A `WeakMap` lets an
 * unreferenced track be collected together with its profile.
 */
const EDGE_PROFILE_BY_SEGMENTS = new WeakMap<
  readonly TrackSegment[],
  TrackEdgeProfile
>();

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
   * 'curb' (the curb band) or 'grass' (the off-road surface: the
   * DRIVABLE off-track band and the grass beyond it - the surface union
   * stays road | curb | grass, so no new surface member was
   * introduced).
   */
  readonly surface: PhysicsSurfaceType;
  /**
   * TRUE only on the road core — the ROAD-ONLY measurement-lane
   * predicate (D_Q10): a slide, camera or traction measurement is
   * taken only where this reads TRUE.
   */
  readonly isRoadOnly: boolean;
  /**
   * TRUE on every DRIVABLE surface: the road core, the curb band AND
   * the off-track band (the grass apron between the drivable edge and
   * the band outer edge - the DRIVABLE off-road surface of AC-094
   * (b)). FALSE beyond the band outer edge, on the OPEN GRASS the car
   * may roll onto (there is NO containment bound any more, so nothing
   * holds the car there): the band itself stays the drivable off-road
   * surface.
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

/**
 * The DRIVABLE off-track band record (the off-road band between the
 * drivable edge and the band outer edge): the EXISTING 'grass' surface
 * member, so the surface union stays road | curb | grass (no new member
 * was introduced, so no exhaustive surface switch elsewhere can break)
 * and the band is the drivable off-road surface of AC-094 (b).
 */
const GRASS_BAND_CLASSIFICATION: TrackSurfaceClassification = Object.freeze({
  surface: OFF_TRACK_SURFACE,
  isRoadOnly: false,
  isDrivable: true,
});

/**
 * The off-track classification record BEYOND the band outer edge (the
 * open grass the car may roll onto - NO containment bound holds it
 * there).
 */
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
 * Creates a pure surface resolver for a fixed track geometry: the
 * returned probe maps a car state to the physics surface under it with
 * the same nearest-chord lateral bands the slab rasterizer paints
 * (junction priority: road before curb before the off-track band; beyond
 * every band the car is off-track grass).
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
  const edgeProfile = resolveEdgeProfileForSegments(segments);
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
    resolveEdgeProfileForSegments(segments),
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
 * Resolves the published edge profile for a segments array, REUSING the
 * profile already resolved for that exact array identity.
 *
 * The profile is resolved ONCE per segments-ARRAY IDENTITY and then
 * reused, so the live tick resolves NO profile and allocates NO array
 * and NO chord per car per frame. The reuse is CONTENT-GUARDED
 * ({@link isEdgeProfileCurrent}): a memoized profile is handed back only
 * while the caller's array still holds the content the profile was
 * resolved from, so a changed geometry is NEVER served a stale profile -
 * it is re-resolved instead.
 *
 * The returned profile is the SEAM'S OWN profile (never a copy of it), so
 * the published chords and the segment snapshot stay the ONE geometry
 * every reader measures the road through.
 *
 * @param segments - Closed-loop ribbon segments (ordered, read only).
 * @returns The published edge profile for these segments.
 */
function resolveEdgeProfileForSegments(
  segments: readonly TrackSegment[],
): TrackEdgeProfile {
  const memoized = EDGE_PROFILE_BY_SEGMENTS.get(segments);
  if (memoized !== undefined && isEdgeProfileCurrent(memoized, segments)) {
    return memoized;
  }

  const resolved = resolveTrackEdgeProfile(segments);
  EDGE_PROFILE_BY_SEGMENTS.set(segments, resolved);
  return resolved;
}

/**
 * Reports whether a memoized edge profile still describes the CONTENT of
 * the caller's segments array.
 *
 * The seam publishes a SNAPSHOT of the segments it resolved from, so a
 * caller that changes the geometry IN PLACE would otherwise keep reading
 * the profile resolved from the OLD content. The guard compares the live
 * segments against that snapshot - the five numeric fields of every
 * segment - and allocates NOTHING, so the hot path stays free of
 * per-frame garbage.
 *
 * @param edgeProfile - The memoized profile to check.
 * @param segments - The live segments array the caller passed.
 * @returns TRUE when the profile still describes the live array.
 */
function isEdgeProfileCurrent(
  edgeProfile: TrackEdgeProfile,
  segments: readonly TrackSegment[],
): boolean {
  const snapshot = edgeProfile.segments;
  if (snapshot.length !== segments.length) {
    return false;
  }

  for (
    let segmentIndex = 0;
    segmentIndex < segments.length;
    segmentIndex += 1
  ) {
    const live = segments[segmentIndex];
    const captured = snapshot[segmentIndex];
    if (
      live.startX !== captured.startX ||
      live.startY !== captured.startY ||
      live.endX !== captured.endX ||
      live.endY !== captured.endY ||
      live.width !== captured.width
    ) {
      return false;
    }
  }

  return true;
}

/**
 * Classifies one ground-plane point with the slab rasterizer's band and
 * priority rules: road core, then curb, then the DRIVABLE off-track band
 * (the grass apron), then the grass beyond it (physics has no air
 * surface, so beyond every band the car is off-track grass). The bands
 * are the rasterizer's UNION rule (a point is road when ANY segment's
 * road core holds it), and the result is the NAMED classification
 * record — the ONE band law behind both the bare resolver and the
 * classifier.
 *
 * THE BAND IS THE SAME BAND THE RASTERIZER PAINTS (AC-094 (e)): the band
 * test is the rasterizer's own apron test
 * (`distance <= halfWidth + apron`), with the apron width READ from the
 * band-geometry owner, so a point on the band is the SAME surface in both
 * models and the band boundary is ONE boundary.
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
  let inGrassBand = false;

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
    } else if (distanceM <= halfWidthM + NEATRACING_TRACK_GRASS_APRON_WIDTH_M) {
      inGrassBand = true;
    }
  }

  if (inRoadCore) {
    return ROAD_CLASSIFICATION;
  }
  if (inCurbBand) {
    return CURB_CLASSIFICATION;
  }
  if (inGrassBand) {
    return GRASS_BAND_CLASSIFICATION;
  }
  return GRASS_CLASSIFICATION;
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
