import { afterAll, beforeAll, describe, expect, it } from '@jest/globals';
import { stepAutopilotInput } from '../../physics/ai.autopilot';
import { createInitialPhysicsCarState } from '../../physics/physics.step.service';
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
 * One frame of pacing in the tests: one 60 Hz tick plus a hair so the
 * accumulator releases exactly one step per frame (the small hair keeps the
 * floor at 1 without ever reaching 2 across a 20-frame run).
 */
const SIM_WORKER_TEST_FRAME_STEP_MS = 16.667;

/**
 * Builds the closed-loop square fixture track: four axis-aligned segments
 * whose endpoints close exactly ((0,0) → (800,0) → (800,800) → (0,800) →
 * (0,0)), with one spline sample per segment midpoint so the autopilot has
 * a frozen racing line to target.
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
 * Builds the fixture init payload (the AC-026 tuple + owned geometry).
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

      // Assert: car 0 opens on the phase-1 host spawn recipe (2 m past
      // segment 0's start point along its chord heading).
      expect(snapshot.cars[0].x).toBeCloseTo(2, 9);
      expect(snapshot.cars[0].y).toBeCloseTo(0, 9);
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
      // both booted in manual mode so car 0 honors the D5 sample seam.
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const driven = internals.createSimWorkerSession();
      const neutral = internals.createSimWorkerSession();
      const manualPayload = createSimWorkerTestFixtureInitPayload({
        raceMode: 'manual',
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
    it('routes car 0 to the autopilot in ai mode and to samples in manual mode', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const cars = [
        createInitialPhysicsCarState(42, { x: 2, y: 0, heading: 0 }),
        createInitialPhysicsCarState(43, { x: 14, y: 0, heading: 0 }),
      ];
      const trackSpec = createSimWorkerTestFixtureTrackSpec();

      // Act: resolve one tick's inputs in ai mode.
      const aiInputs = internals.resolveSimWorkerFleetTickInputs(
        cars,
        'ai',
        undefined,
        trackSpec.splineSamples,
      );

      // Assert: every slot runs the worker-owned autopilot step function.
      expect(aiInputs).toHaveLength(2);
      expect(aiInputs[0]).toEqual(
        stepAutopilotInput(cars[0], trackSpec.splineSamples),
      );
      expect(aiInputs[1]).toEqual(
        stepAutopilotInput(cars[1], trackSpec.splineSamples),
      );
      expect(typeof aiInputs[0].turnRatePercent).toBe('number');

      // Act + assert: in manual mode car 0 takes the replayed D5 sample
      // while agents keep their autopilot channel.
      const humanSample = { throttle: 1, brake: 0, steer: -0.5 };
      const manualInputs = internals.resolveSimWorkerFleetTickInputs(
        cars,
        'manual',
        humanSample,
        trackSpec.splineSamples,
      );
      expect(manualInputs[0]).toEqual(humanSample);
      expect(manualInputs[1]).toEqual(
        stepAutopilotInput(cars[1], trackSpec.splineSamples),
      );
    });

    it('resolves the deterministic line-astern spawn grid on segment 0', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const trackSpec = createSimWorkerTestFixtureTrackSpec();

      const poses = internals.resolveSimWorkerFleetSpawnPoses(trackSpec, 3);

      expect(poses).toHaveLength(3);
      expect(poses[0]).toEqual({ x: 2, y: 0, heading: 0 });
      expect(poses[1]).toEqual({ x: 14, y: 0, heading: 0 });
      expect(poses[2]).toEqual({ x: 26, y: 0, heading: 0 });
    });
  });

  describe('worker-owned lap-line detector', () => {
    it('ports the phase-1 detector semantics (gate, deadband, reset)', () => {
      const internals = simWorkerModule.NEATRACING_SIMULATION_WORKER_INTERNALS;
      const trackSpec = createSimWorkerTestFixtureTrackSpec();
      const firstSegment = trackSpec.segments[0];
      const resolveStep =
        internals.resolveSimWorkerLapLineDetectorStep;

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
      expect((posts[1] as { payload: NeatRacingSimWorkerFramesPayload }).payload.fromTick)
        .toBe(0);
      expect(posts[2].type).toBe('error');
    });
  });
});
