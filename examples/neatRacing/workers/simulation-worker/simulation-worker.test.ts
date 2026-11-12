import { afterAll, beforeAll, describe, expect, it } from '@jest/globals';
import {
  createInitialPhysicsCarState,
  type PhysicsCarState,
  type PhysicsStepInput,
} from '../../physics/physics.step.service';
import { generateTrack } from '../../../racing_curriculum/track/track.generator';
import type {
  SplineSample,
  TrackSegment,
} from '../../../racing_curriculum/track/track.generator.types';
import {
  resolveTrackEdgeHalfWidthM,
  resolveTrackEdgeProfile,
  resolveTrackEdgeSideAtWorldZ,
  resolveTrackSplineEdges,
} from '../../track/track.edge';
import type { NeatRacingFleetControlSeam } from './nge.fleet.control';
import type {
  NeatRacingSimWorkerFramePayload,
  NeatRacingSimWorkerFramesPayload,
  NeatRacingSimWorkerInitPayload,
  NeatRacingSimWorkerRequest,
  NeatRacingSimWorkerResponse,
} from './simulation-worker';

/**
 * Green tests for the neatRacing in-band live simulation worker
 * (Phase 2, Step 01, slice P2-S1-A — tdd_sequence `green-only`).
 *
 * The red-test note for this slice is superseded by the green-only
 * sequence: there is no RED phase, and this suite asserts the slice's
 * acceptance surface — the worker boots (init → ready), answers paced
 * frames with per-frame `frames` replies carrying the released fixed-step
 * ticks, exposes the deterministic fleet spawn + controller resolution
 * the loop needs, and rejects malformed messages with `error` replies.
 *
 * Testing pattern: the flappy worker precedent — `globalThis.self` is
 * replaced with a fake worker scope BEFORE the worker module is dynamically
 * imported (TS's CommonJS emit hoists static requires, so only a dynamic
 * import runs after the fake scope binds; the module attaches its message
 * adapter at import time in a worker realm). The fake is restored in
 * `afterAll`, and every behavioral assertion drives the pure
 * `NEATRACING_SIMULATION_WORKER_INTERNALS` session state machine without a
 * real worker realm.
 *
 * THE PHASE-5 / STEP-27 RED RE-PIN (slice `P5-S27-1`, the RED-S27-2 lane):
 * the fleet spawn grid is RE-PINNED from the LINE-ASTERN grid (a 2 m lead
 * plus a 12 m astern pitch, no lateral term at all) to the SIDE-BY-SIDE
 * grid (ONE shared along-track station, one DISTINCT lateral offset per
 * car). The line-astern expectation is GONE (no dual path): the fleet-pose
 * pin below now measures the side-by-side law in the PUBLISHED path frames
 * - the worker's own chord frame on the legacy fixture, the published
 * closed spline polyline (arc length) on the generated layout - and the
 * generated layout's poses are held inside the PUBLISHED asymmetric
 * half-width (the `track.edge.ts` seam). Every measurement is
 * deterministic: no RNG, no clock.
 *
 * THE BOOT-CONTRACT PIN (slice `P5-S27-3`, the AC-105 re-pin): the boot
 * contract (the init-to-ready case) was the ONE case that still carried
 * the retired line-astern recipe - the +2 m lead past segment 0's start -
 * so the "line-astern expectation is GONE" claim above only became true at
 * FILE level when that pin was re-pinned to the SAME behind-the-line law
 * as the fleet-pose pins: the along-track component reads the upstream
 * station (-SIM_WORKER_TEST_SPAWN_BEHIND_CLEARANCE_M) and the lateral
 * component reads CAR 0's PUBLISHED lane offset, wired against the
 * published spawn grid (the lateral pitch is DERIVED from the spawn
 * station's room, so NO lateral literal is copied). The heading pin
 * SURVIVES UNCHANGED. No line-astern expectation survives in this file.
 *
 * THE BEHIND-THE-LINE RE-PIN (the 2026-11-12 USER clarification, AC-105):
 * the shared station is FULLY BEHIND the green start line - UPSTREAM, a
 * NEGATIVE along-track offset, NEVER 0 (on the line) and NEVER positive
 * (on/above it) - at the CLEARANCE of the ONE published car length (4.5 m,
 * measured to the car CENTRE). The two frames state the SAME law: the
 * legacy chord frame reads -4.5 m directly, while the closed-loop arc frame
 * runs FORWARD from the plane, so the upstream lap tail is read through
 * `resolveSimWorkerTestSignedArcOffsetM` (the closing span owns the tail).
 * The clearance is a TEST-LOCAL literal on purpose: a RED lane must NEVER
 * create an import failure, and the literal is the STRONGER pin.
 */

/** Lazily-loaded worker module type (import in `beforeAll` after the stub). */
type SimWorkerModule = typeof import('./simulation-worker');

/** Session handle type alias for helper signatures. */
type SimWorkerSession = ReturnType<
  SimWorkerModule['NEATRACING_SIMULATION_WORKER_INTERNALS']['createSimWorkerSession']
>;

/** Fake DedicatedWorkerGlobalScope the module's onmessage adapter binds to. */
const simWorkerScope: { onmessage: unknown; postMessage: jest.Mock } = {
  onmessage: undefined,
  postMessage: jest.fn(),
};
/** Original `self` property descriptor captured before the test fakes it. */
const originalSelfDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'self',
);

Object.defineProperty(globalThis, 'self', {
  configurable: true,
  value: simWorkerScope,
});

afterAll(() => {
  if (originalSelfDescriptor === undefined) {
    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: undefined,
    });
  } else {
    Object.defineProperty(globalThis, 'self', originalSelfDescriptor);
  }
});

/** Worker module loaded in `beforeAll` once the fake scope has bound. */
let simWorkerModule: SimWorkerModule;

beforeAll(async () => {
  simWorkerModule = await import('./simulation-worker');
});

/** Fixture fleet size: the AC-033 demo fleet (car 0 plus two agents). */
const SIM_WORKER_TEST_AGENT_COUNT = 3;

/** Fixture run seed for both the init payload and the fleet spawn tests. */
const SIM_WORKER_TEST_SEED = 42;

/** Fixture pack schema version (the packed-frame budget pinned in P2-S1-B). */
const SIM_WORKER_TEST_PACK_SCHEMA_VERSION = 'neatracing-packed-v1';

/** Fixture reproducibility track id for the AC-026 tuple. */
const SIM_WORKER_TEST_TRACK_ID = 'fixture-square-loop';

/** Width (meters) of every fixture segment: comfortably road-band wide. */
const SIM_WORKER_TEST_TRACK_WIDTH_M = 60;

/**
 * The lateral probe offset (meters) the spawn-station measurements are
 * self-checked against.
 *
 * Educational note - WHY A PROBE: a measurement that always returned zero
 * would make every side-by-side assertion pass VACUOUSLY (the fleet would
 * read as "one shared station" and as "distinct lateral offsets" for the
 * WRONG reason). Every measurement is therefore proven against a SYNTHETIC
 * pose a known distance to the side of the path, so a broken measurement
 * fails for the RIGHT reason instead of hiding the real one.
 */
const SIM_WORKER_TEST_PROBE_LATERAL_M = 4;

/**
 * The along-track station tolerance (meters) the side-by-side law allows:
 * every pose of one fleet grid shares ONE along-track station, up to
 * floating-point noise.
 */
const SIM_WORKER_TEST_STATION_TOLERANCE_M = 1e-6;

/**
 * The clearance (meters) the fleet grid must stand FULLY BEHIND the green
 * start line, measured to the car CENTRE.
 *
 * Educational note - WHY A TEST-LOCAL LITERAL: a RED lane must NEVER create
 * an import failure, and the literal is the STRONGER pin - a later drift of
 * the published car length is CAUGHT by this lane instead of being silently
 * followed. The value is the ONE published car length (4.5 m): the GREEN
 * lane derives its own constant from the published car length and must land
 * on this number.
 */
const SIM_WORKER_TEST_SPAWN_BEHIND_CLEARANCE_M = 4.5;

/**
 * One measured fleet-pose station: the ALONG-TRACK station and the SIGNED
 * lateral offset, both read in a PUBLISHED path frame.
 */
interface SimWorkerTestSpawnStation {
  /** The along-track station of the pose, in meters. */
  readonly alongM: number;
  /** The signed lateral offset of the pose from the path, in meters. */
  readonly lateralM: number;
}

/**
 * One frame of pacing in the tests: one 60 Hz tick plus a hair so the
 * accumulator releases exactly one step per frame (the small hair keeps the
 * floor at 1 without ever reaching 2 across a 20-frame run).
 */
const SIM_WORKER_TEST_FRAME_STEP_MS = 16.667;

/**
 * The replayed D5 human sample for car 0 (the human lane's control input).
 *
 * Educational note — car 0 is the HUMAN car in BOTH race modes (LD-12): the
 * host's once-per-frame sample drives it, and it is never handed to a
 * network.
 */
const SIM_WORKER_TEST_HUMAN_SAMPLE: PhysicsStepInput = {
  throttle: 1,
  brake: 0,
  steer: -0.5,
};

/**
 * The deterministic control answer the agent cars publish in these tests.
 *
 * Educational note — why a FIXED answer: the retired scripted lane is gone
 * (AC-029), so the AGENT cars take the CONTROLLER's answer. A frozen answer
 * keeps that assertion deterministic — no RNG, no network, no clock.
 */
const SIM_WORKER_TEST_AGENT_CONTROL: Readonly<PhysicsStepInput> = Object.freeze(
  {
    throttle: 1,
    brake: 0,
    steer: 0.25,
  },
);

/**
 * Creates the deterministic controller double the agent cars run.
 *
 * Educational note — the seam contract: `computeControl(fleet, carIndex)`
 * publishes the control input of ONE fleet car (or null when that car has no
 * adopted network yet). The double reads the ADDRESSED car's own heading and
 * publishes the frozen control with that heading folded onto the steer axis,
 * so the published control belongs to the car it addresses and stays
 * deterministic (the fixture cars start on the segment-0 heading 0, so the
 * published steer is exactly the frozen answer).
 *
 * @returns A deterministic fleet-control seam double.
 */
function createSimWorkerTestControllerDouble(): NeatRacingFleetControlSeam {
  return {
    computeControl(
      fleet: readonly PhysicsCarState[],
      carIndex: number,
    ): PhysicsStepInput {
      const addressedCar = fleet[carIndex];
      return {
        throttle: SIM_WORKER_TEST_AGENT_CONTROL.throttle,
        brake: SIM_WORKER_TEST_AGENT_CONTROL.brake,
        steer: SIM_WORKER_TEST_AGENT_CONTROL.steer + addressedCar.heading,
      };
    },
  };
}

/**
 * Builds the closed-loop square fixture track: four axis-aligned segments
 * whose endpoints close exactly ((0,0) → (800,0) → (800,800) → (0,800) →
 * (0,0)), with one spline sample per segment midpoint (the AC-026 replay
 * tuple carries them; the retired scripted lane no longer follows a racing
 * line).
 *
 * @returns A minimal deterministic track spec fixture.
 */
function createSimWorkerTestFixtureTrackSpec() {
  const segments = [
    { startX: 0, startY: 0, endX: 800, endY: 0 },
    { startX: 800, startY: 0, endX: 800, endY: 800 },
    { startX: 800, startY: 800, endX: 0, endY: 800 },
    { startX: 0, startY: 800, endX: 0, endY: 0 },
  ].map((segment) => ({
    ...segment,
    width: SIM_WORKER_TEST_TRACK_WIDTH_M,
  }));
  const splineSamples = [
    { x: 400, y: 0 },
    { x: 800, y: 400 },
    { x: 400, y: 800 },
    { x: 0, y: 400 },
  ].map((sample, globalIndex) => ({
    ...sample,
    width: SIM_WORKER_TEST_TRACK_WIDTH_M,
    segmentIndex: globalIndex,
    sampleIndexWithinSegment: 0,
    globalIndex,
  }));
  return {
    seed: SIM_WORKER_TEST_SEED,
    layoutVersion: 1,
    sizeBucket: 'fixture',
    segments,
    splineSamples,
    laneCount: 2,
  };
}

/**
 * Measures one fleet pose in the PUBLISHED CHORD frame: the plane through
 * segment 0's start point perpendicular to its chord.
 *
 * Educational note - WHY THIS FRAME: the worker-owned lap-line detector
 * (`resolveSimWorkerLapLineDetectorStep`) decomposes every pose against
 * segment 0's chord as `alongM = (relX*chordX + relY*chordZ)/chordLength`
 * and `lateralM = (chordX*relY - chordZ*relX)/chordLength`. This helper
 * applies the SAME published decomposition, so the side-by-side assertions
 * measure the fleet in the very frame the worker itself measures (on the
 * legacy square fixture the chord runs along +x, so `alongM` is the pose's
 * x and `lateralM` its y).
 *
 * @param pose - The measured spawn pose (physics coordinates).
 * @param firstSegment - Segment 0 of the owned track.
 * @returns The pose's along-track station and signed lateral offset.
 */
function measureSimWorkerTestChordStation(
  pose: { readonly x: number; readonly y: number },
  firstSegment: TrackSegment,
): SimWorkerTestSpawnStation {
  const chordX = firstSegment.endX - firstSegment.startX;
  const chordZ = firstSegment.endY - firstSegment.startY;
  const chordLengthM = Math.hypot(chordX, chordZ);
  const relativeX = pose.x - firstSegment.startX;
  const relativeY = pose.y - firstSegment.startY;
  return {
    alongM: (relativeX * chordX + relativeY * chordZ) / chordLengthM,
    lateralM: (chordX * relativeY - chordZ * relativeX) / chordLengthM,
  };
}

/**
 * Builds the SYNTHETIC probe pose that sits a known lateral distance to the
 * LEFT of segment 0's chord.
 *
 * Educational note - the probe's direction: the published lateral measure
 * reads positive along the chord's LEFT normal (`-chordZ, chordX`), so a
 * probe built on that normal measures a POSITIVE lateral offset.
 *
 * @param firstSegment - Segment 0 of the owned track.
 * @param lateralOffsetM - The probe's lateral offset, in meters.
 * @returns The synthetic probe pose.
 */
function createSimWorkerTestChordProbe(
  firstSegment: TrackSegment,
  lateralOffsetM: number,
): { readonly x: number; readonly y: number } {
  const chordX = firstSegment.endX - firstSegment.startX;
  const chordZ = firstSegment.endY - firstSegment.startY;
  const chordLengthM = Math.hypot(chordX, chordZ);
  return {
    x: firstSegment.startX + (-chordZ / chordLengthM) * lateralOffsetM,
    y: firstSegment.startY + (chordX / chordLengthM) * lateralOffsetM,
  };
}

/**
 * Measures one fleet pose against the PUBLISHED closed spline polyline: the
 * forward ARC LENGTH from sample 0 plus the SIGNED lateral offset.
 *
 * Educational note - WHY THE ARC (and not the chord): the generated layout
 * is a closed loop, so a chord-relative projection WRAPS (the same physical
 * station reads a different chord offset before and after the wrap). The
 * arc length along the published polyline is the loop-invariant station
 * measure, and the lateral offset is the component of the pose-to-polyline
 * gap along the span's LEFT normal (positive = left of the travel
 * direction).
 *
 * @param pose - The measured spawn pose (physics coordinates).
 * @param samples - The published spline samples of the closed loop.
 * @returns The pose's arc station and signed lateral offset.
 */
function measureSimWorkerTestSplineStation(
  pose: { readonly x: number; readonly y: number },
  samples: readonly SplineSample[],
): SimWorkerTestSpawnStation {
  let nearestDistanceM = Number.POSITIVE_INFINITY;
  let nearestAlongM = 0;
  let nearestLateralM = 0;
  let accumulatedM = 0;

  for (let index = 0; index < samples.length; index += 1) {
    const from = samples[index];
    const to = samples[(index + 1) % samples.length];
    const spanX = to.x - from.x;
    const spanY = to.y - from.y;
    const spanLengthM = Math.hypot(spanX, spanY);

    if (spanLengthM > 0) {
      const relativeX = pose.x - from.x;
      const relativeY = pose.y - from.y;
      const parameter = Math.min(
        1,
        Math.max(
          0,
          (relativeX * spanX + relativeY * spanY) / (spanLengthM * spanLengthM),
        ),
      );
      const offsetX = relativeX - spanX * parameter;
      const offsetY = relativeY - spanY * parameter;
      const distanceM = Math.hypot(offsetX, offsetY);

      // The strict comparison keeps the EARLIEST span on a tie, so a pose
      // sitting exactly on sample 0 reads arc 0 (not the wrapped lap end).
      if (distanceM < nearestDistanceM) {
        nearestDistanceM = distanceM;
        nearestAlongM = accumulatedM + spanLengthM * parameter;
        nearestLateralM = (spanX * offsetY - spanY * offsetX) / spanLengthM;
      }
    }
    accumulatedM += spanLengthM;
  }

  return { alongM: nearestAlongM, lateralM: nearestLateralM };
}

/**
 * Builds the SYNTHETIC probe pose that sits a known lateral distance to the
 * RIGHT of the published spline's FIRST span (measured at the span's
 * midpoint).
 *
 * Educational note - WHY THE RIGHT NORMAL: the generated layout is a
 * COUNTER-CLOCKWISE loop (the published edge profile says so), so a span's
 * RIGHT normal points AWAY from the loop. The probe therefore stands on the
 * OUTSIDE of the road, where the ONLY nearby polyline point is its own
 * perpendicular foot: the measured lateral offset is the probe offset
 * EXACTLY (a probe built on the INNER side would be measured short, because
 * the loop curves back towards it).
 *
 * @param samples - The published spline samples of the closed loop.
 * @param lateralOffsetM - The probe's lateral offset, in meters.
 * @returns The synthetic probe pose.
 */
function createSimWorkerTestSplineProbe(
  samples: readonly SplineSample[],
  lateralOffsetM: number,
): { readonly x: number; readonly y: number } {
  const from = samples[0];
  const to = samples[1 % samples.length];
  const spanX = to.x - from.x;
  const spanY = to.y - from.y;
  const spanLengthM = Math.hypot(spanX, spanY);
  const midpointX = from.x + spanX / 2;
  const midpointY = from.y + spanY / 2;
  return {
    x: midpointX + (spanY / spanLengthM) * lateralOffsetM,
    y: midpointY - (spanX / spanLengthM) * lateralOffsetM,
  };
}

/**
 * Resolves the forward ARC LENGTH (meters) of every spline sample, measured
 * from sample 0 along the published closed polyline.
 *
 * @param samples - The published spline samples of the closed loop.
 * @returns One arc station per sample, in sample order.
 */
function resolveSimWorkerTestSampleArcsM(
  samples: readonly SplineSample[],
): readonly number[] {
  const sampleArcsM: number[] = [0];
  let accumulatedM = 0;
  for (let index = 0; index < samples.length - 1; index += 1) {
    const from = samples[index];
    const to = samples[index + 1];
    accumulatedM += Math.hypot(to.x - from.x, to.y - from.y);
    sampleArcsM.push(accumulatedM);
  }
  return sampleArcsM;
}

/**
 * Resolves the arc station (meters) of the FIRST span's midpoint - the
 * station the synthetic spline probe stands on.
 *
 * @param samples - The published spline samples of the closed loop.
 * @returns The arc station of the first span's midpoint.
 */
function resolveSimWorkerTestFirstSpanMidpointArcM(
  samples: readonly SplineSample[],
): number {
  const from = samples[0];
  const to = samples[1 % samples.length];
  return Math.hypot(to.x - from.x, to.y - from.y) / 2;
}

/**
 * Resolves the TOTAL loop length (meters) of the published closed spline
 * polyline - every span, INCLUDING the closing span that runs from the last
 * sample back to sample 0.
 *
 * Educational note - WHY the closing span counts: the start/finish plane is
 * sample 0, so the arc just UPSTREAM of that plane (the lap tail) only
 * EXISTS when the closing span is counted - and the closing span is exactly
 * the span the start line closes. The sibling helper
 * `resolveSimWorkerTestSampleArcsM` deliberately EXCLUDES it, because it
 * publishes one arc per SAMPLE and the closing span owns no sample.
 *
 * @param samples - The published spline samples of the closed loop.
 * @returns The total loop length of the closed polyline, in meters.
 */
function resolveSimWorkerTestLoopLengthM(
  samples: readonly SplineSample[],
): number {
  let loopLengthM = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const from = samples[index];
    const to = samples[(index + 1) % samples.length];
    loopLengthM += Math.hypot(to.x - from.x, to.y - from.y);
  }
  return loopLengthM;
}

/**
 * Wraps a forward arc station onto the SIGNED along-track offset from the
 * start/finish plane: NEGATIVE = UPSTREAM (behind the line), POSITIVE =
 * FORWARD (past the line).
 *
 * Educational note - WHY the wrap: the closed-loop arc frame measures
 * FORWARD from the plane all the way around, so the upstream tail reads as
 * a LARGE positive arc (the loop length minus the clearance). The signed
 * offset turns that tail back into the negative offset the legacy chord
 * frame already reads, so BOTH frames state the SAME law in the SAME units:
 * the fleet waits BEHIND the line, never on it and never above it.
 *
 * @param alongM - The forward arc station, in meters.
 * @param loopLengthM - The total loop length, in meters.
 * @returns The signed along-track offset from the start/finish plane.
 */
function resolveSimWorkerTestSignedArcOffsetM(
  alongM: number,
  loopLengthM: number,
): number {
  return alongM > loopLengthM / 2 ? alongM - loopLengthM : alongM;
}

/**
 * Resolves the index of the spline sample that sits nearest to an arc
 * station (the sample whose published edge describes that station).
 * @param stationM - The arc station, in meters.
 * @param sampleArcsM - The forward arc of every sample, in sample order.
 * @returns The index of the nearest sample.
 */
function resolveSimWorkerTestNearestSampleIndex(
  stationM: number,
  sampleArcsM: readonly number[],
): number {
  let nearestIndex = 0;
  for (let index = 1; index < sampleArcsM.length; index += 1) {
    if (
      Math.abs(sampleArcsM[index] - stationM) <
      Math.abs(sampleArcsM[nearestIndex] - stationM)
    ) {
      nearestIndex = index;
    }
  }
  return nearestIndex;
}

/**
 * Builds the fixture init payload (the AC-026 tuple + owned geometry).
 *
 * Educational note — the injected controller: the legacy scripted lane is
 * retired (AC-029), so an agent car with NO injected controller has no
 * driver. The shared fixture therefore boots WITH the deterministic
 * controller double, and the fleet size stays 3 (the multi-car coverage
 * survives).
 *
 * @param overrides - Partial fields to override on the fixture payload.
 * @returns A complete init payload for one deterministic fixture boot.
 */
function createSimWorkerTestFixtureInitPayload(
  overrides: Partial<NeatRacingSimWorkerInitPayload> = {},
): NeatRacingSimWorkerInitPayload {
  return {
    seed: SIM_WORKER_TEST_SEED,
    agentCount: SIM_WORKER_TEST_AGENT_COUNT,
    packSchemaVersion: SIM_WORKER_TEST_PACK_SCHEMA_VERSION,
    opponentSnapshot: null,
    trackId: SIM_WORKER_TEST_TRACK_ID,
    featureFlags: { aiAutopilot: true },
    trackSpec: createSimWorkerTestFixtureTrackSpec(),
    raceMode: 'ai',
    controller: createSimWorkerTestControllerDouble(),
    ...overrides,
  };
}

/**
 * Narrowly-typed accessor for a `frames` reply's payload.
 *
 * @param responses - The single-response list from `handleRequest`.
 * @returns The `frames` payload of the sole response.
 */
function framePayloadOf(
  responses: ReturnType<SimWorkerSession['handleRequest']>,
): NeatRacingSimWorkerFramesPayload {
  expect(responses).toHaveLength(1);
  expect(responses[0].type).toBe('frames');
  return (responses[0] as { payload: NeatRacingSimWorkerFramesPayload })
    .payload;
}

/**
 * Narrowly-typed accessor for an `error` reply's reason message.
 *
 * @param responses - The single-response list from `handleRequest`.
 * @returns The error message of the sole response.
 */
function errorMessageOf(
  responses: ReturnType<SimWorkerSession['handleRequest']>,
): string {
  expect(responses).toHaveLength(1);
  expect(responses[0].type).toBe('error');
  return (responses[0] as { payload: { message: string } }).payload.message;
}

/**
 * Drives a booted `ai` session through N once-per-tick frames using the
 * fixture frame step, returning the reply of the last frame.
 *
 * @param session - Sim-worker session (already booted).
 * @param frameCount - Number of frames to pace.
 * @param startTimestampMs - First frame timestamp (accumulator anchor).
 * @returns The final `frames` reply.
 */
function driveSimWorkerTestFrames(
  session: SimWorkerSession,
  frameCount: number,
  startTimestampMs: number,
): NeatRacingSimWorkerFramesPayload {
  if (!simWorkerModule) {
    throw new Error('simulation-worker module not loaded in beforeAll');
  }
  let lastPayload: NeatRacingSimWorkerFramesPayload | undefined;
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const request: NeatRacingSimWorkerRequest = {
      type: 'frame',
      payload: {
        timestampMs:
          startTimestampMs + frameIndex * SIM_WORKER_TEST_FRAME_STEP_MS,
      } satisfies NeatRacingSimWorkerFramePayload,
    };
    lastPayload = framePayloadOf(session.handleRequest(request));
  }
  return lastPayload as NeatRacingSimWorkerFramesPayload;
}

describe('NEATRACING_SIMULATION_WORKER_INTERNALS', () => {
  describe('boot contract (init → ready)', () => {
    it('boots the session with a fleet, level-1 rules, and a ready echo of the AC-026 tuple', () => {
      // Arrange: an untouched idle session plus the fixture init payload.
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const session = internals.createSimWorkerSession();
      const payload = createSimWorkerTestFixtureInitPayload();

      // Act: boot.
      const responses = session.handleRequest({ type: 'init', payload });

      // Assert: exactly one ready reply echoing the tuple fields.
      expect(responses).toHaveLength(1);
      expect(responses[0]).toEqual({
        type: 'ready',
        payload: {
          seed: SIM_WORKER_TEST_SEED,
          agentCount: SIM_WORKER_TEST_AGENT_COUNT,
          packSchemaVersion: SIM_WORKER_TEST_PACK_SCHEMA_VERSION,
          trackId: SIM_WORKER_TEST_TRACK_ID,
          raceMode: 'ai',
        },
      });

      // Assert: the worker owns all three sim artifacts at boot.
      const snapshot = session.snapshot;
      expect(snapshot.phase).toBe('running');
      expect(snapshot.cars).toHaveLength(SIM_WORKER_TEST_AGENT_COUNT);
      expect(snapshot.rules.level).toBe(1);
      expect(snapshot.tick).toBe(0);

      // Assert: car 0 opens on the AC-105 START GRID - FULLY BEHIND the
      // green start line, on the PUBLISHED lane the grid publishes for it.
      //
      // THE ALONG-TRACK component is the UPSTREAM station of the legacy
      // chord frame, asserted against the TEST-LOCAL clearance literal
      // (the retired +2 m line-astern lead is GONE - never 0, never
      // positive). The LATERAL component is CAR 0's lane offset on the
      // side-by-side grid (NOT necessarily 0), pinned against the
      // PUBLISHED car-0 pose of the SAME seam the RED-S27-2 case measures
      // (the published spawn grid): the lateral pitch is DERIVED from the
      // spawn station's asymmetric room, so a copied literal would create a
      // SECOND source of truth. The HEADING pin SURVIVES UNCHANGED (the
      // forward road heading - the legacy chord heading is 0).
      const trackSpec = createSimWorkerTestFixtureTrackSpec();
      const firstSegment = trackSpec.segments[0];
      const publishedPoses = internals.resolveSimWorkerFleetSpawnPoses(
        trackSpec,
        SIM_WORKER_TEST_AGENT_COUNT,
      );
      expect(publishedPoses).toHaveLength(SIM_WORKER_TEST_AGENT_COUNT);
      const publishedCar0 = publishedPoses[0];
      const publishedStation = measureSimWorkerTestChordStation(
        publishedCar0,
        firstSegment,
      );
      const bootedStation = measureSimWorkerTestChordStation(
        snapshot.cars[0],
        firstSegment,
      );

      expect(bootedStation.alongM).toBeCloseTo(
        -SIM_WORKER_TEST_SPAWN_BEHIND_CLEARANCE_M,
        9,
      );
      expect(bootedStation.lateralM).toBeCloseTo(publishedStation.lateralM, 9);
      expect(snapshot.cars[0].x).toBeCloseTo(publishedCar0.x, 9);
      expect(snapshot.cars[0].y).toBeCloseTo(publishedCar0.y, 9);
      expect(snapshot.cars[0].heading).toBeCloseTo(0, 9);
    });

    it('rejects a second init with an error reply mentioning the phase', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const session = internals.createSimWorkerSession();
      session.handleRequest({
        type: 'init',
        payload: createSimWorkerTestFixtureInitPayload(),
      });

      const responses = session.handleRequest({
        type: 'init',
        payload: createSimWorkerTestFixtureInitPayload(),
      });

      expect(errorMessageOf(responses)).toContain('already running');
    });

    it('rejects out-of-range and non-integer agentCount with an error reply', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const invalidAgentCounts = [0, 33, 2.5];
      for (const agentCount of invalidAgentCounts) {
        const session = internals.createSimWorkerSession();
        const responses = session.handleRequest({
          type: 'init',
          payload: createSimWorkerTestFixtureInitPayload({ agentCount }),
        });
        expect(errorMessageOf(responses)).toContain('agentCount');
      }
    });

    it('rejects trackSpec geometry without segments or spline samples', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const emptySegments = createSimWorkerTestFixtureInitPayload({
        trackSpec: { ...createSimWorkerTestFixtureTrackSpec(), segments: [] },
      });
      const emptySplineSamples = createSimWorkerTestFixtureInitPayload({
        trackSpec: {
          ...createSimWorkerTestFixtureTrackSpec(),
          splineSamples: [],
        },
      });

      for (const payload of [emptySegments, emptySplineSamples]) {
        const session = internals.createSimWorkerSession();
        const responses = session.handleRequest({ type: 'init', payload });
        expect(errorMessageOf(responses)).toContain('trackSpec');
      }
    });

    it('rejects frames before init with an error reply', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const session = internals.createSimWorkerSession();

      const responses = session.handleRequest({
        type: 'frame',
        payload: { timestampMs: 0 },
      });

      expect(errorMessageOf(responses)).toContain('not initialized');
    });
  });

  describe('the named no-controller error reply (M1, slice P4-S2-rival-propagation)', () => {
    it('replies with a protocol error envelope instead of escaping handleRequest, carrying the same named message', () => {
      // Arrange: a session booted WITHOUT the injected controller seam (the
      // init payload's optional controller left absent) — the agent cars
      // have NO driver, so the first paced frame raises the NAMED
      // no-controller error inside the frame dispatch.
      //
      // Educational note — the M1 re-pin (slice `P4-S2-rival-propagation`):
      // the named error is the RIGHT runtime witness and stays reachable
      // (the `controller` field stays OPTIONAL by design), but the error
      // no longer ESCAPES `handleRequest` — the dispatch converts THAT ONE
      // named error into the protocol error envelope.
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const session = internals.createSimWorkerSession();
      const bootResponses = session.handleRequest({
        type: 'init',
        payload: createSimWorkerTestFixtureInitPayload({
          controller: undefined,
        }),
      });
      expect(bootResponses).toHaveLength(1);
      expect(bootResponses[0].type).toBe('ready');

      // Act: pace the frames. The dispatch MUST NOT raise. The FIRST frame
      // is the zero-delta anchor (a fresh accumulator releases no ticks —
      // the loop cannot pre-fill before real frames arrive), so the SECOND
      // frame is the one that releases a REAL tick and reaches the control
      // resolution.
      let responses: ReturnType<SimWorkerSession['handleRequest']> = [];
      expect(() => {
        session.handleRequest({
          type: 'frame',
          payload: { timestampMs: 0 },
        });
        responses = session.handleRequest({
          type: 'frame',
          payload: { timestampMs: SIM_WORKER_TEST_FRAME_STEP_MS },
        });
      }).not.toThrow();

      // Assert: exactly one error reply.
      expect(responses).toHaveLength(1);
      expect(responses[0].type).toBe('error');

      // Assert: the reply carries the SAME message text the resolver raises —
      // the NAMED marker (exported by the internals seam so a consumer can
      // match on the constant) plus the ` (race mode: <mode>)` suffix the
      // recorded M2 acceptance retains.
      const message = errorMessageOf(responses);
      expect(message).toContain(
        internals.NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR,
      );
      expect(message).toContain('(race mode: ai)');
    });

    it('keeps every OTHER worker error escaping as an uncaught error (only the named one converts)', () => {
      // Arrange: a booted session whose next frame request poisons the
      // timestamps read, so the DISPATCH raises a generic (NOT the named
      // no-controller) error.
      //
      // Educational note — the M1 constraint (ii): ONLY the named
      // no-controller error converts into a `{ type: 'error' }` reply. Every
      // OTHER worker error keeps its current behavior — it still escapes
      // `handleRequest` as an uncaught error — so no other protocol reply is
      // silently reshaped by the selective catch.
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const session = internals.createSimWorkerSession();
      session.handleRequest({
        type: 'init',
        payload: createSimWorkerTestFixtureInitPayload(),
      });
      const poisonedFramePayload = {
        get timestampMs(): number {
          throw new Error('sim worker frame failed: poisoned timestamp read');
        },
      } as unknown as NeatRacingSimWorkerFramePayload;

      // Act + assert: the generic error keeps escaping uncaught.
      expect(() =>
        session.handleRequest({ type: 'frame', payload: poisonedFramePayload }),
      ).toThrow('sim worker frame failed: poisoned timestamp read');
    });
  });

  describe('fixed-step frame loop', () => {
    it('releases exactly one tick per paced frame and snapshots the fleet and rules', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const session = internals.createSimWorkerSession();
      session.handleRequest({
        type: 'init',
        payload: createSimWorkerTestFixtureInitPayload(),
      });

      // Act: the first frame (t=0) contributes a zero delta, then one full
      // frame step releases exactly one tick.
      const zeroStepResponses = session.handleRequest({
        type: 'frame',
        payload: { timestampMs: 0 },
      });
      const oneStepPayload = framePayloadOf(
        session.handleRequest({
          type: 'frame',
          payload: { timestampMs: SIM_WORKER_TEST_FRAME_STEP_MS },
        }),
      );

      // Assert: the zero-delta anchor frame released 0 ticks.
      const zeroPayload = framePayloadOf(zeroStepResponses);
      expect(zeroPayload.fromTick).toBe(0);
      expect(zeroPayload.toTick).toBe(0);

      // Assert: exactly one tick on the second frame, tick counter visible
      // on the car record, snapshot copies instead of live aliases.
      expect(oneStepPayload.fromTick).toBe(0);
      expect(oneStepPayload.toTick).toBe(1);
      expect(oneStepPayload.cars).toHaveLength(SIM_WORKER_TEST_AGENT_COUNT);
      expect(oneStepPayload.cars[0].tick).toBe(1);
      expect(oneStepPayload.rules.level).toBe(1);
      oneStepPayload.cars[0].x = -12345;
      expect(session.snapshot.cars[0].x).not.toBe(-12345);
    });

    it('caps catch-up at the render-loop ceiling across oversized frame deltas', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const session = internals.createSimWorkerSession();
      session.handleRequest({
        type: 'init',
        payload: createSimWorkerTestFixtureInitPayload(),
      });
      session.handleRequest({ type: 'frame', payload: { timestampMs: 0 } });
      session.handleRequest({
        type: 'frame',
        payload: { timestampMs: SIM_WORKER_TEST_FRAME_STEP_MS },
      });

      // Act: a tab-switch-sized delta (100 ms) after one tick releases the
      // capped 4-tick ceiling, not the raw 6-tick backlog.
      const payload = framePayloadOf(
        session.handleRequest({
          type: 'frame',
          payload: { timestampMs: SIM_WORKER_TEST_FRAME_STEP_MS + 100 },
        }),
      );

      expect(payload.fromTick).toBe(1);
      expect(payload.toTick).toBe(5);
    });

    it('rejects a frame whose timestamp is not a finite number', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const session = internals.createSimWorkerSession();
      session.handleRequest({
        type: 'init',
        payload: createSimWorkerTestFixtureInitPayload(),
      });

      const responses = session.handleRequest({
        type: 'frame',
        payload: { timestampMs: Number.NaN },
      });

      expect(errorMessageOf(responses)).toContain('timestampMs');
    });

    it('keeps identical tuples and pacing byte-identical across independent sessions', () => {
      // Arrange: two sessions from the SAME tuple, driven identically.
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const sessionA = internals.createSimWorkerSession();
      const sessionB = internals.createSimWorkerSession();
      sessionA.handleRequest({
        type: 'init',
        payload: createSimWorkerTestFixtureInitPayload(),
      });
      sessionB.handleRequest({
        type: 'init',
        payload: createSimWorkerTestFixtureInitPayload(),
      });

      // Act: identical 20-frame pacing (20 ticks total, zero-step anchor).
      sessionA.handleRequest({ type: 'frame', payload: { timestampMs: 0 } });
      sessionB.handleRequest({ type: 'frame', payload: { timestampMs: 0 } });
      const repliesA = driveSimWorkerTestFrames(
        sessionA,
        20,
        SIM_WORKER_TEST_FRAME_STEP_MS,
      );
      const repliesB = driveSimWorkerTestFrames(
        sessionB,
        20,
        SIM_WORKER_TEST_FRAME_STEP_MS,
      );

      // Assert: Level-2 ordered determinism — the frame streams match.
      expect(repliesA.toTick).toBe(20);
      expect(repliesB.toTick).toBe(20);
      expect(repliesA.cars).toEqual(repliesB.cars);
      expect(repliesA.rules).toEqual(repliesB.rules);
    });
  });

  describe('D5 control sampling (manual mode)', () => {
    it('replays the latest frame sample across catch-up ticks and stays neutral before one arrives', () => {
      // Arrange: one throttled session and one never-sampled session,
      // both booted in manual mode so car 0 honors the D5 sample seam and
      // the agent cars run the injected controller.
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const driven = internals.createSimWorkerSession();
      const neutral = internals.createSimWorkerSession();
      const manualPayload = createSimWorkerTestFixtureInitPayload({
        raceMode: 'manual',
        controller: createSimWorkerTestControllerDouble(),
      });
      driven.handleRequest({ type: 'init', payload: manualPayload });
      neutral.handleRequest({ type: 'init', payload: manualPayload });

      // Act: anchor both sessions, then one throttled tick on `driven`.
      driven.handleRequest({ type: 'frame', payload: { timestampMs: 0 } });
      neutral.handleRequest({ type: 'frame', payload: { timestampMs: 0 } });
      const throttledPayload = framePayloadOf(
        driven.handleRequest({
          type: 'frame',
          payload: {
            timestampMs: SIM_WORKER_TEST_FRAME_STEP_MS,
            sample: { throttle: 1, brake: 0, steer: 0 },
          } satisfies NeatRacingSimWorkerFramePayload,
        }),
      );
      const neutralPayload = framePayloadOf(
        neutral.handleRequest({
          type: 'frame',
          payload: { timestampMs: SIM_WORKER_TEST_FRAME_STEP_MS },
        }),
      );

      // Assert: the sample reached physics — speed rose above standstill.
      expect(throttledPayload.cars[0].speed).toBeGreaterThan(0);
      // Assert: no sample ever sent means the neutral control sample applied.
      expect(neutralPayload.cars[0].speed).toBe(0);

      // Act + assert: the NEXT sampleless frame replays the HELD sample
      // (D5 catch-up replay), so car 0 keeps accelerating.
      const heldReplayPayload = framePayloadOf(
        driven.handleRequest({
          type: 'frame',
          payload: { timestampMs: 2 * SIM_WORKER_TEST_FRAME_STEP_MS },
        }),
      );
      expect(heldReplayPayload.cars[0].speed).toBeGreaterThan(
        throttledPayload.cars[0].speed,
      );
    });
  });

  describe('worker-owned controllers and spawn resolution', () => {
    it('routes car 0 to the human lane and the agent cars to the injected controller in ai mode', () => {
      // Arrange: the deterministic fixture fleet (car 0 human + car 1 agent),
      // the fixture spline samples, the replayed human sample and the
      // injected controller double.
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const cars = [
        createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 }),
        createInitialPhysicsCarState(43, { x: 14, y: 0, heading: 0 }),
      ];
      const trackSpec = createSimWorkerTestFixtureTrackSpec();
      const controller = createSimWorkerTestControllerDouble();

      // Act: resolve one tick's inputs in ai mode.
      const aiInputs = internals.resolveSimWorkerFleetTickInputs(
        cars,
        'ai',
        SIM_WORKER_TEST_HUMAN_SAMPLE,
        trackSpec.splineSamples,
        controller,
      );

      // Assert: car 0 keeps the HUMAN lane (never handed to a network) and
      // the agent car runs the injected controller — the retired scripted
      // lane has no fallback left.
      expect(aiInputs).toHaveLength(2);
      expect(aiInputs[0]).toEqual(SIM_WORKER_TEST_HUMAN_SAMPLE);
      expect(aiInputs[1]).toEqual(SIM_WORKER_TEST_AGENT_CONTROL);
    });

    it('routes car 0 to the replayed human sample and the agent cars to the injected controller in manual mode', () => {
      // Arrange: the deterministic fixture fleet (car 0 human + car 1 agent),
      // the fixture spline samples, the replayed human sample and the
      // injected controller double.
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const cars = [
        createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 }),
        createInitialPhysicsCarState(43, { x: 14, y: 0, heading: 0 }),
      ];
      const trackSpec = createSimWorkerTestFixtureTrackSpec();
      const controller = createSimWorkerTestControllerDouble();

      // Act: resolve one tick's inputs in manual mode.
      const manualInputs = internals.resolveSimWorkerFleetTickInputs(
        cars,
        'manual',
        SIM_WORKER_TEST_HUMAN_SAMPLE,
        trackSpec.splineSamples,
        controller,
      );

      // Assert: car 0 takes the replayed D5 sample while the agent car runs
      // the injected controller.
      expect(manualInputs).toHaveLength(2);
      expect(manualInputs[0]).toEqual(SIM_WORKER_TEST_HUMAN_SAMPLE);
      expect(manualInputs[1]).toEqual(SIM_WORKER_TEST_AGENT_CONTROL);
    });

    it('resolves the deterministic side-by-side spawn grid on segment 0', () => {
      // Arrange: the legacy square fixture (the LEGACY CHORD-WALK spawn
      // path) and the PUBLISHED chord frame it is measured in (the plane
      // through segment 0's start point perpendicular to its chord).
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const trackSpec = createSimWorkerTestFixtureTrackSpec();
      const firstSegment = trackSpec.segments[0];

      // Act: resolve the fleet spawn grid.
      const poses = internals.resolveSimWorkerFleetSpawnPoses(trackSpec, 3);

      // Assert (non-vacuity): the fixture yields the three fleet poses and
      // the chord frame reads a KNOWN lateral probe, so the side-by-side
      // assertions below can never pass on a broken measurement.
      expect(poses).toHaveLength(3);
      const probe = measureSimWorkerTestChordStation(
        createSimWorkerTestChordProbe(
          firstSegment,
          SIM_WORKER_TEST_PROBE_LATERAL_M,
        ),
        firstSegment,
      );
      expect(Math.abs(probe.lateralM)).toBeCloseTo(
        SIM_WORKER_TEST_PROBE_LATERAL_M,
        9,
      );

      // Assert: the three poses share ONE along-track station (the
      // side-by-side law). The retired line-astern grid measured THREE
      // stations (2 m, 14 m, 26 m), so this is the discriminating clause.
      const stations = poses.map((pose) =>
        measureSimWorkerTestChordStation(pose, firstSegment),
      );

      // Assert (the behind-the-line law): EVERY car - the player AND the
      // rivals - waits FULLY BEHIND the green start line, i.e. UPSTREAM of
      // the plane through segment 0's start point, and SIDE BY SIDE on that
      // one station. The retired line-astern grid published the fleet at
      // +2 / +14 / +26 m FORWARD of the plane, so the shared station reads
      // POSITIVE today and must read NEGATIVE once the law lands.
      expect(stations.map((station) => station.alongM)).toEqual([
        expect.closeTo(-SIM_WORKER_TEST_SPAWN_BEHIND_CLEARANCE_M, 3),
        expect.closeTo(-SIM_WORKER_TEST_SPAWN_BEHIND_CLEARANCE_M, 3),
        expect.closeTo(-SIM_WORKER_TEST_SPAWN_BEHIND_CLEARANCE_M, 3),
      ]);

      const stationSpreadM =
        Math.max(...stations.map((station) => station.alongM)) -
        Math.min(...stations.map((station) => station.alongM));
      expect(stationSpreadM).toBeLessThanOrEqual(
        SIM_WORKER_TEST_STATION_TOLERANCE_M,
      );

      // Assert: the lateral offsets are PAIRWISE DISTINCT (one lane each).
      const lateralOffsetsM = stations.map((station) => station.lateralM);
      expect(
        new Set(lateralOffsetsM.map((offsetM) => offsetM.toFixed(6))).size,
      ).toBe(lateralOffsetsM.length);
    });

    it('spawns the fleet side-by-side on the generated layout', () => {
      // Arrange: the deterministic generated layout (the ROAD-ANCHORED
      // spline spawn path) and its published closed spline polyline.
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const trackSpec = generateTrack({
        seed: SIM_WORKER_TEST_SEED,
        layoutVersion: 1,
        sizeBucket: 'small',
      });

      // Act: resolve the fleet spawn grid.
      const poses = internals.resolveSimWorkerFleetSpawnPoses(trackSpec, 3);

      // Assert (non-vacuity): the layout yields the three fleet poses and
      // the arc measurement reads a KNOWN lateral probe (and places it at
      // the probed station), so the side-by-side assertions below can
      // never pass on a broken measurement.
      expect(poses).toHaveLength(3);
      const edgeProfile = resolveTrackEdgeProfile(trackSpec.segments);
      expect(edgeProfile.loopOrientation).toBe('counter-clockwise');
      const probe = measureSimWorkerTestSplineStation(
        createSimWorkerTestSplineProbe(
          trackSpec.splineSamples,
          SIM_WORKER_TEST_PROBE_LATERAL_M,
        ),
        trackSpec.splineSamples,
      );
      expect(Math.abs(probe.lateralM)).toBeCloseTo(
        SIM_WORKER_TEST_PROBE_LATERAL_M,
        9,
      );
      expect(probe.alongM).toBeCloseTo(
        resolveSimWorkerTestFirstSpanMidpointArcM(trackSpec.splineSamples),
        9,
      );

      // Assert: the three poses share ONE along-track station, measured as
      // the ARC LENGTH along the published spline polyline. The retired
      // line-astern grid measured THREE stations (2 m, 14 m, 26 m).
      const stations = poses.map((pose) =>
        measureSimWorkerTestSplineStation(pose, trackSpec.splineSamples),
      );

      // Assert (the behind-the-line law): EVERY car - the player AND the
      // rivals - waits FULLY BEHIND the green start line. The closed-loop
      // arc frame runs FORWARD from the start/finish plane, so UPSTREAM is
      // the lap TAIL; the SIGNED offset turns that tail back into the same
      // NEGATIVE offset the chord frame reads. The retired line-astern grid
      // published the fleet at +2 / +14 / +26 m FORWARD of the plane, so the
      // shared station reads POSITIVE today and must read NEGATIVE once the
      // law lands.
      //
      // THE LATERAL FRAME (the GREEN lane's obligation): the measured
      // station is the CHORD projection of the pose, so the side-by-side
      // lateral offsets MUST be taken along the span CHORD's normal - the
      // frame `resolveSimWorkerSplineSpawnGrid` already uses for the pose
      // AND the heading. A lateral offset taken along the LOCAL TANGENT
      // normal slips the car d*sin(phi) ALONG the road (measured on the
      // generated 'small' layout: 0.27 m at d = 4.5 m), so the car centre
      // would stand only ~4.23 m upstream - a real breach of the law - and
      // this pin AND the shared-station spread pin would both fail.
      const loopLengthM = resolveSimWorkerTestLoopLengthM(
        trackSpec.splineSamples,
      );
      expect(
        stations.map((station) =>
          resolveSimWorkerTestSignedArcOffsetM(station.alongM, loopLengthM),
        ),
      ).toEqual([
        expect.closeTo(-SIM_WORKER_TEST_SPAWN_BEHIND_CLEARANCE_M, 3),
        expect.closeTo(-SIM_WORKER_TEST_SPAWN_BEHIND_CLEARANCE_M, 3),
        expect.closeTo(-SIM_WORKER_TEST_SPAWN_BEHIND_CLEARANCE_M, 3),
      ]);

      const stationSpreadM =
        Math.max(...stations.map((station) => station.alongM)) -
        Math.min(...stations.map((station) => station.alongM));
      expect(stationSpreadM).toBeLessThanOrEqual(
        SIM_WORKER_TEST_STATION_TOLERANCE_M,
      );

      // Assert: the lateral offsets are PAIRWISE DISTINCT (one lane each).
      const lateralOffsetsM = stations.map((station) => station.lateralM);
      expect(
        new Set(lateralOffsetsM.map((offsetM) => offsetM.toFixed(6))).size,
      ).toBe(lateralOffsetsM.length);
    });

    it('keeps every fleet pose inside the published asymmetric half-width', () => {
      // Arrange: the generated layout, its PUBLISHED edge profile and the
      // PUBLISHED per-sample edge (the half-width AT each station).
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const trackSpec = generateTrack({
        seed: SIM_WORKER_TEST_SEED,
        layoutVersion: 1,
        sizeBucket: 'small',
      });
      const edgeProfile = resolveTrackEdgeProfile(trackSpec.segments);
      const splineEdges = resolveTrackSplineEdges(
        trackSpec.splineSamples,
        edgeProfile,
      );
      const sampleArcsM = resolveSimWorkerTestSampleArcsM(
        trackSpec.splineSamples,
      );

      // Act: resolve the fleet spawn grid.
      const poses = internals.resolveSimWorkerFleetSpawnPoses(trackSpec, 3);

      // Assert (non-vacuity): the layout yields the three fleet poses and
      // the published profile publishes a POSITIVE half-width per sample,
      // so the containment bound below is a real bound.
      expect(poses).toHaveLength(3);
      expect(splineEdges).toHaveLength(trackSpec.splineSamples.length);
      expect(splineEdges.every((edge) => edge.innerHalfWidthM > 0)).toBe(true);

      // Assert: every pose stays inside the published half-width ON ITS OWN
      // SIDE (the road is asymmetric: the outer side of a corner is wider
      // than the inner one, so the bound is the side-resolved half-width).
      for (const pose of poses) {
        const station = measureSimWorkerTestSplineStation(
          pose,
          trackSpec.splineSamples,
        );
        const sampleIndex = resolveSimWorkerTestNearestSampleIndex(
          station.alongM,
          sampleArcsM,
        );
        const edge = splineEdges[sampleIndex];
        // The world frame MIRRORS the generator plane as `worldZ = -y`.
        const side = resolveTrackEdgeSideAtWorldZ(
          edgeProfile,
          edge.segmentIndex,
          pose.x,
          -pose.y,
        );
        const halfWidthM = resolveTrackEdgeHalfWidthM(edge, side);
        expect(Math.abs(station.lateralM)).toBeLessThanOrEqual(halfWidthM);
      }
    });

    it('keeps the fleet spawn grid a PURE function of (trackSpec, agentCount)', () => {
      // Arrange: the generated layout (the deterministic layout the fleet
      // grid serves) and the worker-owned spawn resolver.
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const trackSpec = generateTrack({
        seed: SIM_WORKER_TEST_SEED,
        layoutVersion: 1,
        sizeBucket: 'small',
      });

      // Act: resolve the same grid twice.
      const firstPoses = internals.resolveSimWorkerFleetSpawnPoses(
        trackSpec,
        3,
      );
      const secondPoses = internals.resolveSimWorkerFleetSpawnPoses(
        trackSpec,
        3,
      );

      // Assert: the two resolutions are deep-equal (NO RNG, NO clock - the
      // grid is a pure function of its inputs).
      expect(secondPoses).toEqual(firstPoses);
      expect(firstPoses).toHaveLength(3);
    });
  });

  describe('worker-owned lap-line detector', () => {
    it('ports the phase-1 detector semantics (gate, deadband, reset)', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const trackSpec = createSimWorkerTestFixtureTrackSpec();
      const firstSegment = trackSpec.segments[0];
      const resolveStep = internals.resolveSimWorkerLapLineDetectorStep;

      // Spawn-side sample: first qualifying sample latches +2, no crossing.
      const latch = resolveStep(
        createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 }),
        null,
        firstSegment,
      );
      expect(latch).toEqual({ crossing: false, nextSide: 2 });

      // Sign flip inside the gate: forward crossing counts.
      const crossing = resolveStep(
        createInitialPhysicsCarState(42, { x: -2, y: 0, heading: 0 }),
        2,
        firstSegment,
      );
      expect(crossing).toEqual({ crossing: true, nextSide: -2 });

      // Deadband pose (|along| < 0.5): no crossing, baseline NOT updated.
      const deadband = resolveStep(
        createInitialPhysicsCarState(42, { x: 0.2, y: 0, heading: 0 }),
        2,
        firstSegment,
      );
      expect(deadband).toEqual({ crossing: false, nextSide: 2 });

      // Outside the lateral gate: baseline resets to null.
      const gatedOut = resolveStep(
        createInitialPhysicsCarState(42, { x: 2, y: 10, heading: 0 }),
        2,
        firstSegment,
      );
      expect(gatedOut).toEqual({ crossing: false, nextSide: null });
    });
  });

  describe('lifecycle and malformed-message contract', () => {
    it('stops without a reply and rejects later frames, tolerating a pre-init stop', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const session = internals.createSimWorkerSession();
      session.handleRequest({
        type: 'init',
        payload: createSimWorkerTestFixtureInitPayload(),
      });

      // Act + assert: stop is reply-free by contract.
      expect(session.handleRequest({ type: 'stop' })).toHaveLength(0);
      expect(session.snapshot.phase).toBe('stopped');

      // Assert: frames after stop are rejected.
      expect(
        errorMessageOf(
          session.handleRequest({ type: 'frame', payload: { timestampMs: 0 } }),
        ),
      ).toContain('stopped');

      // Assert: stop-before-init is an idempotent no-op (still no reply).
      const idle = internals.createSimWorkerSession();
      expect(idle.handleRequest({ type: 'stop' })).toHaveLength(0);
      expect(idle.snapshot.phase).toBe('stopped');
    });

    it('answers unknown and malformed messages with an error reply', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const session = internals.createSimWorkerSession();

      const unknownResponses = session.handleRequest({
        type: 'explode',
      } as unknown as NeatRacingSimWorkerRequest);
      expect(unknownResponses).toHaveLength(1);
      expect(unknownResponses[0].type).toBe('error');
      expect(
        (unknownResponses[0] as { payload: { message: string } }).payload
          .message,
      ).toContain('unknown message type');

      const malformedResponses = session.handleRequest(
        null as unknown as NeatRacingSimWorkerRequest,
      );
      expect(malformedResponses).toHaveLength(1);
      expect(malformedResponses[0].type).toBe('error');
      expect(
        (malformedResponses[0] as { payload: { message: string } }).payload
          .message,
      ).toContain('malformed message');
    });
  });

  describe('worker-transport adapter', () => {
    it('wired onmessage posts exactly one reply per routed request', () => {
      expect(typeof simWorkerScope.onmessage).toBe('function');
      simWorkerScope.postMessage.mockClear();

      const handler = simWorkerScope.onmessage as (event: {
        data: unknown;
      }) => void;

      // Act: route boot, one zero-step frame, and one junk message.
      handler({
        data: {
          type: 'init',
          payload: createSimWorkerTestFixtureInitPayload(),
        },
      });
      handler({
        data: {
          type: 'frame',
          payload: { timestampMs: 0 },
        } satisfies NeatRacingSimWorkerRequest,
      });
      handler({ data: { type: 'nonsense' } });

      // Assert: the module-level session answered on the fake worker scope.
      const posts = simWorkerScope.postMessage.mock.calls.map(
        (call) => call[0] as NeatRacingSimWorkerResponse,
      );
      expect(posts).toHaveLength(3);
      expect(posts[0].type).toBe('ready');
      expect(posts[1].type).toBe('frames');
      expect(
        (posts[1] as { payload: NeatRacingSimWorkerFramesPayload }).payload
          .fromTick,
      ).toBe(0);
      expect(posts[2].type).toBe('error');
    });
  });
});
