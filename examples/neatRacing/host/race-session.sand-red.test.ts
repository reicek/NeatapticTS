/**
 * RED -> GREEN contract tests for the SERVED sand-band wiring
 * (slice `P1-sand-red`, ITERATION 2; plan
 * `neatracing-voxel-racer.plans.md`, the 2026-11-04 RED MEASUREMENT ROUND
 * of the fix packet `fix-packet-P1-sand-red-iteration-2`).
 *
 * THE MEASURED DEFECT (the plan's RED MEASUREMENT ROUND - a
 * PRE-ITERATION-3 epoch; the mechanism below no longer exists): the
 * "teleport" that survived the iteration-1 band contract was the ELASTIC
 * REBOUND of the sand-band OUTER EDGE (`reboundPastBandOuterEdge` - a
 * symbol the ITERATION-3 RETIREMENT removed from `../physics/collision.utils`
 * entirely; AC-090 VERDICT 8): a car that crossed the band outer edge had
 * its OUTWARD velocity component REFLECTED and was RECOMPOSED around the
 * reflected velocity, so the published heading SNAPPED (the MEASURED
 * pre-retirement one-tick snap is 1.2870 rad) and the reflected velocity
 * carried the car BACK onto the ribbon (the MEASURED pre-retirement
 * lateral distance falls from 17.64 m to 13.59 m), which destroyed the
 * sand-like slow roll (the MEASURED pre-retirement off-road speed never
 * came near the then-22.25 m/s floor: it bottoms out at 57.18 m/s; the
 * floor itself was re-derived to the USER-named 20 km/h off-track speed
 * cap by fix-packet-P1-sand-red-iteration-3).
 *
 * THE OWNER FILE: this lane is the SERVED half of the iteration-2 RED
 * contracts (the physics half lives in
 * `../physics/physics.step.service.test.ts`). It drives the SAME defect
 * THROUGH the shipped host wiring - the real `createRaceSession` over the
 * REAL browser-entry dep shape - so the contract is proven on the path the
 * USER actually drove, not only on a hand-built physics fixture.
 *
 * THE SERVED WIRING (mirrors `browser-entry.ts` exactly):
 * - the track is the pinned demo fixture
 *   (`generateTrack({seed:42, layoutVersion:1, sizeBucket:'large'})`) and
 *   the physics closure is the REAL one
 *   (`stepPhysics(car, input, surfaceResolver, trackSpec.segments)`);
 * - the spawn pose is the REAL one (the first-segment start + 2 m along
 *   the first-segment heading, browser-entry.ts:804-813) through the
 *   `createInitialCarState` seam;
 * - the input is the REAL `sampleKeyboardInput` over a LIVE-MUTABLE
 *   presses record (the browser-entry presses shape), so the steer axis can
 *   be RELEASED mid-drive;
 * - the composer / HUD / mode selector are the structural spy doubles of
 *   the `race-session.test.ts` harness (no DOM, no GL, no timers, no RNG).
 *
 * THE DRIVE: the forward key is held for the whole drive; a steer key is
 * held ONLY to INITIATE the exit and is then RELEASED, so the reentry
 * check really runs with NO steer (nothing can carry the car back onto
 * the ribbon: the ITERATION-3 RETIREMENT removed the bound entirely, so
 * the check guards an ABSENCE, not a mechanism) and the yaw channel is
 * DISENGAGED (the published heading must then follow the yaw-rate channel
 * only).
 *
 * THE LANES (each one single-purpose, each one opening with the
 * NON-VACUITY GUARDS that must PASS, so a fixture that never leaves the
 * ribbon fails loudly instead of passing silently):
 * RED-I2-served-a (DISCRIMINATING): nothing SNAPS the yaw - for EVERY
 *   steer-free off-road tick the published heading delta stays inside
 *   the yaw-rate channel (`abs(yawRate) * dt + 1e-9`).
 * RED-I2-served-b (DISCRIMINATING): once the car is off the ribbon it
 *   NEVER reenters the drivable surface on its own (the retired rebound
 *   once carried it back - it is gone).
 * RED-I2-served-c (PRESERVATION GUARD - GREEN today, able to fail): the
 *   per-tick displacement stays inside ONE integration envelope on EVERY
 *   tick (the touch must never TELEPORT the position).
 *
 * THE LANES ARE RUNTIME-ONLY: they drive the EXISTING live tick through
 * the EXISTING host wiring, so every red is a NAMED RUNTIME failure and
 * NEVER an import failure.
 * ------------------------------------------------------------------------- */

import { describe, expect, it } from '@jest/globals';
import {
  createInitialPhysicsCarState,
  stepPhysics,
  type PhysicsCarSpawnPose,
  type PhysicsCarState,
  type PhysicsStepInput,
  type PhysicsSurfaceType,
} from '../physics/physics.step.service';
import {
  classifyTrackSurface,
  createTrackSurfaceResolver,
} from '../physics/collision.utils';
import { stepRaceRules } from '../physics/race-rules';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import { sampleKeyboardInput } from './input';
import {
  createRaceSession,
  type NeatRacingRaceSession,
  type NeatRacingRaceSessionDeps,
} from './race-session';
import {
  NEATRACING_CATCHUP_CEILING_MS,
  NEATRACING_FIXED_TIMESTEP_SECONDS,
} from './render-loop.utils';
import type { NeatRacingHudState } from './hud';
import type {
  NeatRacingRaceMode,
  NeatRacingRaceModeListener,
} from './mode-selector';
import type { NeatRacingFrameComposerFrame } from './frame-composer';

/**
 * The served demo-track fixture (the browser-entry generation input: the
 * REAL served geometry the defect was measured on).
 */
const SERVED_TRACK_FIXTURE = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
} as const;

/**
 * The number of saturated catch-up frames the steer key is held for: the
 * steer INITIATES the exit and is then released, so every off-road tick
 * runs with NO steer.
 */
const SERVED_STEER_FRAMES = 24;

/**
 * The number of saturated catch-up frames the drive continues after the
 * steer is released (the forward key stays held: the car keeps rolling
 * off-road). The PRE-RETIREMENT epoch measured the exit here (tick 133),
 * the rebound (tick 142) and the carry-back reentry (tick 154); under
 * the RETIREMENT (AC-090 VERDICT 8) none of those mechanisms exist and
 * the car simply keeps rolling off-road inside the drive window.
 */
const SERVED_ROLL_FRAMES = 24;

/** The live tick's fixed step (s). */
const SERVED_TICK_SECONDS = NEATRACING_FIXED_TIMESTEP_SECONDS;

/** The float64 epsilon of the heading arithmetic (rad). */
const SERVED_YAW_EPSILON_RAD = 1e-9;

/** The float64 epsilon of the displacement arithmetic (m). */
const SERVED_MOVE_EPSILON_M = 1e-6;

/**
 * The live-mutable keyboard presses (the browser-entry presses shape: the
 * host owns a stable record whose booleans the key handlers flip).
 */
interface MutablePresses {
  /** Forward key held down. */
  up: boolean;
  /** Brake key held down. */
  down: boolean;
  /** Counterclockwise (left) steer key held down. */
  left: boolean;
  /** Clockwise (right) steer key held down. */
  right: boolean;
}

/** One recorded served tick (the pre-step pose plus the replayed sample). */
interface ServedPhysicsRecord {
  /** The pose the tick started from (the pre-step state). */
  readonly state: PhysicsCarState;
  /** The sample the host replayed into the tick. */
  readonly input: PhysicsStepInput;
}

/** One recorded served tick with its MEASURED deltas. */
interface ServedTick {
  /** The tick index (0 is the first tick after the start pose). */
  readonly index: number;
  /** The pose the tick started from. */
  readonly pre: PhysicsCarState;
  /** The pose the tick published (the pose the world sees). */
  readonly post: PhysicsCarState;
  /** The sample the host replayed into the tick. */
  readonly input: PhysicsStepInput;
  /** The surface the PRODUCTION classifier resolved for the published pose. */
  readonly surface: PhysicsSurfaceType;
  /** The published displacement of the tick (m). */
  readonly moveM: number;
  /** The wrapped published heading delta of the tick (rad). */
  readonly yawDeltaRad: number;
}

/** The recorded served drive and its MEASURED summary. */
interface ServedLane {
  /** The recorded ticks. */
  readonly ticks: readonly ServedTick[];
  /** The index of the FIRST tick whose published pose is on the grass. */
  readonly firstGrassIndex: number;
  /** The index of the FIRST tick that ran with NO steer. */
  readonly firstSteerFreeIndex: number;
  /** The consecutive distinct surfaces the drive crossed. */
  readonly surfaceLadder: readonly PhysicsSurfaceType[];
  /** The largest heading delta OVER the yaw-rate envelope (rad). */
  readonly maxYawViolationRad: number;
  /** The index of the tick with the largest heading violation. */
  readonly maxYawViolationIndex: number;
  /** The tick indices (from the first grass tick on) back on the ribbon. */
  readonly reentryTickIndices: readonly number[];
  /** The smallest displacement of the drive (m). */
  readonly minMoveM: number;
  /** The largest displacement OVER the integration envelope (m). */
  readonly maxMoveViolationM: number;
  /** The smallest published speed of the drive (m/s). */
  readonly minSpeedMps: number;
}

/**
 * Wraps an angle into `[-pi, pi)` (the shortest signed difference between
 * two headings, so the wrap boundary never reads as a full turn).
 *
 * @param radians - The angle to wrap.
 * @returns The wrapped angle.
 */
const wrapAngle = (radians: number): number => {
  const tau = Math.PI * 2;
  return ((((radians + Math.PI) % tau) + tau) % tau) - Math.PI;
};

/**
 * Builds the served drive over the REAL shipped wiring: the real track
 * fixture, the real browser-entry physics closure, the real spawn pose,
 * the real keyboard sampler over live-mutable presses, and the real race
 * session.
 *
 * @returns The recorded drive and its measured summary.
 */
const buildServedLane = (): ServedLane => {
  const trackSpec = generateTrack(SERVED_TRACK_FIXTURE);
  const segments = trackSpec.segments;
  const surfaceResolver = createTrackSurfaceResolver(segments);

  // The REAL spawn pose (browser-entry.ts:804-813): the first-segment
  // start plus 2 m along the first-segment heading.
  const firstSegment = trackSpec.segments[0];
  const spawnHeading = firstSegment
    ? Math.atan2(
        firstSegment.endY - firstSegment.startY,
        firstSegment.endX - firstSegment.startX,
      )
    : 0;
  const spawnPose: PhysicsCarSpawnPose = {
    x: (firstSegment?.startX ?? 0) + Math.cos(spawnHeading) * 2,
    y: (firstSegment?.startY ?? 0) + Math.sin(spawnHeading) * 2,
    heading: spawnHeading,
  };

  // The LIVE-MUTABLE presses: the forward key is held for the whole drive
  // and the steer key is released after the exit is initiated.
  const presses: MutablePresses = {
    up: true,
    down: false,
    left: true,
    right: false,
  };
  const records: ServedPhysicsRecord[] = [];

  const deps: NeatRacingRaceSessionDeps = {
    composer: {
      drawFrame: (frame: NeatRacingFrameComposerFrame): void => {
        void frame;
      },
    },
    hud: {
      update: (state: NeatRacingHudState): void => {
        void state;
      },
    },
    modeSelector: {
      mode: 'manual' as NeatRacingRaceMode,
      onSwitch: (listener: NeatRacingRaceModeListener): void => {
        void listener;
      },
    },
    presses,
    sampleKeyboardInput,
    stepPhysics: (car: PhysicsCarState, input: PhysicsStepInput) => {
      records.push({ state: car, input });
      return stepPhysics(car, input, surfaceResolver, segments);
    },
    stepRaceRules,
    resolveForwardLapLineCrossing: (): boolean => false,
    createInitialCarState: () =>
      createInitialPhysicsCarState(trackSpec.seed, spawnPose),
  };

  const session: NeatRacingRaceSession = createRaceSession(deps);

  // The first frame releases no tick (the accumulator starts empty); every
  // saturated frame releases the catch-up cap of four simulation ticks.
  session.handleFrame(0);
  for (let frame = 0; frame < SERVED_STEER_FRAMES; frame += 1) {
    session.handleFrame((frame + 1) * NEATRACING_CATCHUP_CEILING_MS);
  }

  // THE STEER IS RELEASED: the reentry check must run with NO steer, so
  // the only thing that could ever carry the car back onto the ribbon was
  // the FORMER bound - RETIRED (AC-090 VERDICT 8), which is exactly why
  // nothing can carry the car back today.
  presses.left = false;
  presses.right = false;
  for (let frame = 0; frame < SERVED_ROLL_FRAMES; frame += 1) {
    session.handleFrame(
      (SERVED_STEER_FRAMES + frame + 1) * NEATRACING_CATCHUP_CEILING_MS,
    );
  }

  const ticks: ServedTick[] = [];
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    const post =
      index + 1 < records.length ? records[index + 1].state : session.state.car;
    ticks.push({
      index,
      pre: record.state,
      post,
      input: record.input,
      surface: classifyTrackSurface(post, segments).surface,
      moveM: Math.hypot(post.x - record.state.x, post.y - record.state.y),
      yawDeltaRad: wrapAngle(post.heading - record.state.heading),
    });
  }

  const firstGrassIndex = ticks.findIndex(
    (tick) => tick.surface === 'grass',
  );
  const firstSteerFreeIndex = ticks.findIndex(
    (tick) => tick.input.steer === 0,
  );

  // The consecutive distinct surfaces the drive crossed (the ladder).
  const surfaceLadder: PhysicsSurfaceType[] = [];
  for (const tick of ticks) {
    const last = surfaceLadder[surfaceLadder.length - 1];
    if (last !== tick.surface) surfaceLadder.push(tick.surface);
  }

  // The yaw violation is measured over the OFF-ROAD window (the post-touch
  // ticks): with the steer axis disengaged before the exit, the published
  // heading must follow the yaw-rate channel only.
  let maxYawViolationRad = 0;
  let maxYawViolationIndex = -1;
  let minMoveM = Number.POSITIVE_INFINITY;
  let maxMoveViolationM = Number.NEGATIVE_INFINITY;
  let minSpeedMps = Number.POSITIVE_INFINITY;
  const reentryTickIndices: number[] = [];
  for (const tick of ticks) {
    if (firstGrassIndex >= 0 && tick.index >= firstGrassIndex) {
      const violation =
        Math.abs(tick.yawDeltaRad) -
        Math.abs(tick.pre.yawRate) * SERVED_TICK_SECONDS;
      if (violation > maxYawViolationRad) {
        maxYawViolationRad = violation;
        maxYawViolationIndex = tick.index;
      }
      if (tick.surface === 'road' || tick.surface === 'curb') {
        reentryTickIndices.push(tick.index);
      }
    }
    if (tick.moveM < minMoveM) minMoveM = tick.moveM;
    const moveViolation = tick.moveM - tick.post.speed * SERVED_TICK_SECONDS;
    if (moveViolation > maxMoveViolationM) maxMoveViolationM = moveViolation;
    if (tick.post.speed < minSpeedMps) minSpeedMps = tick.post.speed;
  }

  return {
    ticks,
    firstGrassIndex,
    firstSteerFreeIndex,
    surfaceLadder,
    maxYawViolationRad,
    maxYawViolationIndex,
    reentryTickIndices,
    minMoveM,
    maxMoveViolationM,
    minSpeedMps,
  };
};

/**
 * The SERVED RED-I2 contracts (slice `P1-sand-red`, ITERATION 2; the plan's
 * `fix-packet-P1-sand-red-iteration-2`) - the discriminating clauses below
 * are now GUARANTEED by the ITERATION-3 RETIREMENT (AC-090 VERDICT 8,
 * fix-packet-P1-sand-red-iteration-3): the bound AND the rebound no longer
 * exist in the shipped tick, so a snap or an unsteered carry-back is
 * impossible BY CONSTRUCTION and every lane here is able to fail if either
 * mechanism creeps back.
 */
describe('the SERVED sand-band rebound (Step 14W iteration 2, slice P1-sand-red; CLOSED BY THE ITERATION-3 RETIREMENT)', () => {
  // [RED-I2-SERVED-A]
  it('RED-I2-served-a (DISCRIMINATING): the bound never SNAPS the yaw - for EVERY steer-free off-road tick the published heading delta stays inside the yaw-rate channel (the MEASURED served snap is 1.0566 rad)', () => {
    // Arrange: the served drive over the REAL shipped wiring.
    const lane = buildServedLane();

    // The NON-VACUITY GUARDS come FIRST (they PASS today and are able to
    // fail): the drive REALLY crosses the ribbon (road -> curb -> grass) ...
    expect(lane.surfaceLadder.slice(0, 3)).toEqual(['road', 'curb', 'grass']);
    // ... the steer axis REALLY was released BEFORE the exit (the reentry
    // and yaw checks must run with NO steer) ...
    expect(lane.firstSteerFreeIndex).toBeGreaterThanOrEqual(0);
    expect(lane.firstSteerFreeIndex).toBeLessThan(lane.firstGrassIndex);
    // ... and the car REALLY moved on EVERY tick (a standstill would
    // satisfy the yaw clause VACUOUSLY).
    expect(lane.minMoveM).toBeGreaterThan(0);

    // THE DISCRIMINATING CLAUSE: the published heading delta NEVER exceeds
    // the yaw-rate channel (`abs(yawRate) * dt + 1e-9`; the steer axis is
    // disengaged, so the yaw MUST follow that channel only). In the
    // PRE-RETIREMENT epoch the band rebound RECOMPOSED the car around the
    // reflected velocity, so the rebound tick published a 1.0566 rad
    // heading snap against a zero yaw rate - this lane FAILED here. The
    // RETIREMENT removed the rebound entirely (AC-090 VERDICT 8), so the
    // clause now holds by construction.
    expect(lane.maxYawViolationRad).toBeLessThanOrEqual(SERVED_YAW_EPSILON_RAD);
  });

  // [RED-I2-SERVED-B]
  it('RED-I2-served-b (DISCRIMINATING): once the car is off the ribbon it NEVER reenters the drivable surface on its own (the rebound carries it back: the MEASURED served reentry is the curb at tick 154)', () => {
    // Arrange: the served drive over the REAL shipped wiring.
    const lane = buildServedLane();

    // The NON-VACUITY GUARDS come FIRST (they PASS today and are able to
    // fail): the drive REALLY left the ribbon (a drive that never reached
    // the off-road band would satisfy the reentry clause VACUOUSLY) ...
    expect(lane.firstGrassIndex).toBeGreaterThanOrEqual(0);
    // ... the steer axis REALLY was released BEFORE the exit (nothing but
    // the bound itself may carry the car back) ...
    expect(lane.firstSteerFreeIndex).toBeLessThan(lane.firstGrassIndex);
    // ... and the car REALLY moved on EVERY tick (a standstill would
    // satisfy the reentry clause VACUOUSLY).
    expect(lane.minMoveM).toBeGreaterThan(0);

    // THE DISCRIMINATING CLAUSE: NO tick from the exit on may report a
    // drivable surface again - the off-road band is drivable sand, so the
    // car must NEVER be carried back onto the ribbon without user steer.
    // In the PRE-RETIREMENT epoch the rebound reflected the outward
    // velocity and the car was carried back over the drivable edge (the
    // MEASURED pre-retirement reentry was the curb at tick 154) - this
    // lane FAILED here. The RETIREMENT removed the bound entirely
    // (AC-090 VERDICT 8), so nothing carries the car back.
    expect(lane.reentryTickIndices).toEqual([]);
  });

  // [RED-I2-SERVED-C]
  it('RED-I2-served-c (PRESERVATION GUARD - GREEN today, able to fail): the per-tick displacement stays inside ONE integration envelope on EVERY tick (the touch must never TELEPORT the position)', () => {
    // Arrange: the served drive over the REAL shipped wiring.
    const lane = buildServedLane();

    // The NON-VACUITY GUARDS come FIRST (they PASS today and are able to
    // fail): the car REALLY moved (a standstill would satisfy the
    // envelope clause VACUOUSLY) ...
    expect(lane.minMoveM).toBeGreaterThan(0);
    // ... and the drive REALLY reached the off-road band (the envelope is
    // measured across the REAL retired-seam wiring, though the FORMER
    // bound event no longer exists under the RETIREMENT).
    expect(lane.firstGrassIndex).toBeGreaterThanOrEqual(0);

    // THE PRESERVATION CLAUSE: EVERY tick's displacement stays inside the
    // integration envelope of the tick that produced it
    // (`move_t <= speed_t * dt + 1e-6`; the MEASURED worst case today is
    // 1.49e-14 m, so this lane is GREEN today and FAILS the moment a
    // mechanism truly teleports the pose).
    expect(lane.maxMoveViolationM).toBeLessThanOrEqual(SERVED_MOVE_EPSILON_M);
  });
});
