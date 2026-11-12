/** @jest-environment jsdom */

/**
 * RED contracts for the LIVE LADDER of the NeatRacing host lane (slice
 * `P4-S2-red` — Phase 4, Step 24 of `plans/neatracing-voxel-racer.plans.md`).
 *
 * Educational note — what this file owns:
 * The three RED-P4 contracts of the "live ladder" packet, expressed against
 * the HOST LANE (the live seam that boots the sim worker, drives the AI-mode
 * goal machine and rebinds the published round state):
 *
 * - RED-P4-4 — the PROMOTION ROUND BOUNDARY. When the AI lane promotes from
 *   one rung to the next, the fresh round must run on a FRESH round tuple
 *   that carries the promoted rung's SEED NETWORK, the CARRIED INNOVATION
 *   COUNTER and the NEXT RUNG's agent count, and the adaptation session must
 *   be RE-PINNED onto the promoted rung.
 * - RED-P4-5 — the RIVAL SWAP at a lap/line boundary. The live lane must
 *   publish a rival-swap payload when a RIVAL crosses the lap line, and the
 *   swap must mutate ONLY the rival's serialized weights (one bounded
 *   MOD_WEIGHT round, seeded). The two-car rung (ONE rival) is a tested
 *   NO-OP: the single rival has no replacement target.
 * - RED-P4-6 — the FIVE-CAR SAME-TUPLE REPLAY. The five-car round (the
 *   level-3 rung) must replay byte-identically from the same tuple across
 *   the adaptation window, must carry a rival-swap inside that window, and
 *   must show NO per-frame allocation growth against the one-car baseline.
 *
 * Educational note — the RED shape (why these assertions, and not others):
 * Every failure message must NAME THE MISSING BEHAVIOUR. A contract that
 * failed on a missing module or a missing export would gate the wrong
 * thing, so this file never imports a symbol that does not exist yet: the
 * witnesses are read off the LIVE seam's own published state (the captured
 * adaptation tuples, the captured worker requests and the captured frames
 * replies), and the expectations are phrased over OBSERVABLE FACTS (a tuple
 * at a rung, a payload in the published state, a buffer identity).
 *
 * Educational note — the seam doubles (the house pattern of
 * `host.wiring.test.ts`):
 * The rendering context, the render-loop step counter, the canvas sizing,
 * the HUD, the scripted-baseline probe and the sim-worker session factory are
 * the SAME inert doubles the live host suite uses. The adaptation session is
 * wrapped (the REAL session keeps driving) and additionally RECORDS the
 * tuple it was booted with; its snapshot publishes a GROWN carrier node count
 * (the accelerated-carrier double: the real carrier's 84-node network can
 * never cross the level-1 promotion band of 300 on its own), and its control
 * seam is replaced by the SHIPPED AUTOPILOT so the agent cars really drive
 * the track (a rival can then really cross the lap line).
 *
 * Educational note — the fixture contract:
 * DETERMINISTIC (no RNG, no timers, no wall-clock reads; the track, the
 * seeds and the probe pace are fixed), RESETTABLE (every captured array is
 * cleared in `beforeEach`, the DOM shell is rebuilt, and the live loop is
 * stopped in `afterEach`), and SINGLE-PURPOSE (one `describe` per contract,
 * one top-level `expect` per `it` unless two or three related facts make up
 * ONE behaviour).
 *
 * @module
 */

jest.mock('../engine/context.service', () => ({
  ...jest.requireActual('../engine/context.service'),
  initRenderingContext: jest.fn(),
  runShaderWarmup: jest.fn(),
}));

jest.mock('../host/render-loop.utils', () => ({
  ...jest.requireActual('../host/render-loop.utils'),
  advanceFrameSteps: jest.fn(),
}));

jest.mock('../browser-entry/canvas-dimensions.utils', () => ({
  ...jest.requireActual('../browser-entry/canvas-dimensions.utils'),
  updateFullWindowRendererSize: jest.fn(),
}));

jest.mock('../host/hud', () => ({
  ...jest.requireActual('../host/hud'),
  createHud: jest.fn(),
}));

/**
 * The scripted-baseline probe is the ONLY promotion seam doubled: the probe
 * episode is a boot-cost decision under test, not a law under test. The
 * promotion conditions, the thresholds and the ladder rungs stay REAL.
 */
jest.mock('../ai/promotion.rules', () => ({
  ...jest.requireActual('../ai/promotion.rules'),
  // The `__esModule` marker is re-declared on the mock: the spread above
  // drops the actual module's marker (it is not enumerable), and a
  // namespace import of a mock WITHOUT the marker is an interop COPY.
  __esModule: true,
  runNeatRacingScriptedBaselineProbeEpisode: jest.fn(),
}));

/**
 * The adaptation session module is jest-mocked in this file (the AI lane's
 * LIVE drive recorder is the ONLY doubled seam): the factory wraps the REAL
 * session, so the host keeps driving a REAL session while the wrapper
 * RECORDS the tuple every boot was given.
 *
 * The recorder arrays carry the `mock` prefix on purpose: a `jest.mock`
 * factory may only reach module-scope bindings whose name starts with
 * `mock` (the hoisting contract), so the recorder stays reachable from the
 * hoisted factory.
 */
jest.mock('../ai/racing-adaptation.session', () => {
  const actual = jest.requireActual<
    typeof import('../ai/racing-adaptation.session')
  >('../ai/racing-adaptation.session');
  // The SHIPPED autopilot is the agent cars' driver in this file: a rival
  // can then REALLY cross the lap line, which is the RED-P4-5 trigger.
  const autopilot = jest.requireActual<
    typeof import('../physics/ai.autopilot')
  >('../physics/ai.autopilot');
  return {
    ...actual,
    // The `__esModule` marker is re-declared on the mock (see above).
    __esModule: true,
    createNeatRacingAdaptationSession: (
      options: Parameters<typeof actual.createNeatRacingAdaptationSession>[0],
    ) => {
      const realSession = actual.createNeatRacingAdaptationSession(options);
      // The witness of the promotion round boundary (RED-P4-4): the tuple
      // every boot of the adaptation session was given.
      mockCapturedAdaptationTuples.push(options.tuple);
      mockCapturedAdaptationBootSummaries.push({
        level: options.level ?? null,
        carCount: options.carCount,
      });
      // The CARRIER slot is the session's LAST car (the dedicated
      // background carrier rides beyond the simulation fleet).
      const carrierIndex = options.carCount - 1;
      return {
        ...realSession,
        // The control seam: the SHIPPED autopilot over the round's own
        // spline samples (the agent cars really drive the track).
        controller: {
          computeControl: (
            fleet: readonly {
              x: number;
              y: number;
              heading: number;
              speed: number;
            }[],
            carIndex: number,
          ) => {
            const car = fleet[carIndex];
            if (car === undefined) {
              return null;
            }
            return autopilot.stepAutopilotInput(
              car,
              options.trackSpec.splineSamples,
            );
          },
        },
        // The eval worker is NOT the law under test: the tick resolves with
        // no outcome (the live drive stays observable, the eval lane stays
        // out of the RED).
        adaptOnTick: () => Promise.resolve([]),
        // The ACCELERATED-CARRIER double: the published carrier node count
        // is grown past the promotion band (the real carrier's 84-node
        // network can never cross the level-1 band of 300 on its own).
        snapshot: () => {
          const snapshot = realSession.snapshot();
          const nodeCounts = [...snapshot.nodeCounts];
          nodeCounts[carrierIndex] = MOCK_GROWN_CARRIER_NODE_COUNT;
          return { ...snapshot, nodeCounts };
        },
      };
    },
  };
});

jest.mock('../workers/simulation-worker/simulation-worker', () => ({
  ...jest.requireActual('../workers/simulation-worker/simulation-worker'),
  createSimWorkerSession: jest.fn(),
}));

import { afterEach, beforeEach, describe, expect, it } from '@jest/globals';
import {
  initRenderingContext,
  runShaderWarmup,
} from '../engine/context.service';
import type { RenderingContextInitResult } from '../engine/context.service';
import { createHud } from '../host/hud';
import type { CanvasDimensions } from '../browser-entry/canvas-dimensions.utils';
import { updateFullWindowRendererSize } from '../browser-entry/canvas-dimensions.utils';
import { advanceFrameSteps } from '../host/render-loop.utils';
import {
  runNeatRacingScriptedBaselineProbeEpisode,
  resolveNeatRacingLevelCarCount,
  type NeatRacingScriptedBaselineProbeResult,
} from '../ai/promotion.rules';
import {
  NEATRACING_RIVAL_SWAP_MAIN_AGENT_CAR_INDEX,
  NEATRACING_RIVAL_SWAP_PAYLOAD_KIND,
  NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX,
  restoreNeatRacingRivalSwapReplacementMinds,
  serializeNeatRacingRivalSwapPayload,
  type NeatRacingRivalSwapPayload,
} from '../ai/rival.propagation';
import { createTrackSpecForAutopilotFixture } from '../physics/ai.autopilot.fixtures';
import {
  createInitialPhysicsCarState,
  type PhysicsCarState,
  type PhysicsStepInput,
} from '../physics/physics.step.service';
import type { NeatRacingObservationLevel } from '../ai/observation.assembler';
import type { TrackSpec } from '../../racing_curriculum/track/track.generator.types';
import {
  createNeatRacingFleetControl,
  type NeatRacingFleetCarController,
  type NeatRacingFleetControlSeam,
  type NeatRacingFleetControllerRelay,
} from '../workers/simulation-worker/nge.fleet.control';
import {
  NEATRACING_REPLAY_ADAPTATION_TICKS,
  type NeatRacingReproducibilityTuple,
} from '../workers/simulation-worker/reproducibility.utils';
import type {
  NeatRacingSimWorkerFramesPayload,
  NeatRacingSimWorkerInitPayload,
  NeatRacingSimWorkerRequest,
  NeatRacingSimWorkerResponse,
  NeatRacingSimWorkerSession,
  NeatRacingSimWorkerSessionSnapshot,
} from '../workers/simulation-worker/simulation-worker';
import { createSimWorkerSession } from '../workers/simulation-worker/simulation-worker';
import type {
  NeatRacingStart,
  NeatRacingStop,
} from '../browser-entry/browser-entry';
// Side-effect import: evaluating the boot entry registers
// `window.neatRacingStart` on the jsdom host.
import '../browser-entry/browser-entry';

const { createSimWorkerSession: createRealSimWorkerSession } =
  jest.requireActual<
    typeof import('../workers/simulation-worker/simulation-worker')
  >('../workers/simulation-worker/simulation-worker');

/* ------------------------------------------------------------------------- *
 * Harness constants.
 * ------------------------------------------------------------------------- */

/** The reserved host DOM id for the HUD subtree (the selector + HUD host). */
const HUD_OUTPUT_ID = 'neatracing-hud-output';

/** Timestamp handed to the captured RAF callback when a test drives frame one. */
const FIRST_FRAME_TIMESTAMP_MS = 0;

/** Timestamp step (ms) between driven frames (the 60 Hz render cadence). */
const FRAME_STEP_MS = 1000 / 60;

/**
 * The simulation ticks the worker releases per driven frame in the rival
 * drive (the round lap is ~26 s of simulation time, so the drive releases
 * several ticks per drawn frame).
 */
const RIVAL_DRIVE_TICKS_PER_FRAME = 8;

/**
 * The frame budget the rival drive spends looking for the rival lap-line
 * crossing (a full circuit of the served track at the autopilot pace).
 */
const RIVAL_DRIVE_FRAME_BUDGET = 900;

/**
 * The promotion crossings one rung needs (the shared AI-lane session-lap
 * floor: the predicate is evaluated at a crossing only once the session has
 * completed this many laps).
 */
const PROMOTION_CROSSING_COUNT = 20;

/**
 * The GROWN carrier node count the accelerated-carrier double publishes: it
 * clears BOTH promotion bands (the level-1 band of 300 and the level-2 band
 * of 600) with one constant, so the REAL predicate fires on the LIVE
 * published count.
 */
const MOCK_GROWN_CARRIER_NODE_COUNT = 601;

/**
 * The dedicated background carrier rides ONE slot beyond the simulation
 * fleet, so the session's fleet size is the rung's car count plus this
 * carrier.
 */
const AI_LANE_BACKGROUND_CARRIER_COUNT = 1;

/** The tuple seed the seam-level fleet fixture derives its child seeds from. */
const FLEET_FIXTURE_TUPLE_SEED = 42;

/** The observation rung the five-car seam fixture observes at. */
const FLEET_FIXTURE_LEVEL = 3;

/** The distance one fixture tick advances a car per unit of throttle. */
const FLEET_FIXTURE_ADVANCE_M_PER_THROTTLE = 1;

/** The forward speed one fixture tick assigns per unit of throttle. */
const FLEET_FIXTURE_SPEED_PER_THROTTLE = 12;

/* ------------------------------------------------------------------------- *
 * Captured host state (the RED witnesses).
 * ------------------------------------------------------------------------- */

/**
 * Every tuple the host booted an adaptation session with (in boot order).
 *
 * WHY: the promotion round boundary's fresh tuple is the witness of
 * RED-P4-4 — the live lane must boot the promoted rung's session with a
 * tuple that carries the seed network, the carried innovation counter and
 * the next rung's agent count.
 */
const mockCapturedAdaptationTuples: NeatRacingReproducibilityTuple[] = [];

/**
 * Every adaptation session boot's level and fleet size (in boot order).
 *
 * WHY: the re-pin of the adaptation session onto the promoted rung is its
 * own observable fact (the session's level and its fleet size).
 */
const mockCapturedAdaptationBootSummaries: {
  readonly level: number | null;
  readonly carCount: number;
}[] = [];

/** Every frames reply the host consumed (in frame order). */
const capturedFramesPayloads: NeatRacingSimWorkerFramesPayload[] = [];

/** One captured sim-worker boot: the REAL session plus its request log. */
interface CapturedSimWorkerBoot {
  /** Every request the host sent, in order (captured BEFORE forwarding). */
  readonly capturedRequests: NeatRacingSimWorkerRequest[];
  /** The capturing forwarder the host holds. */
  readonly session: NeatRacingSimWorkerSession;
  /** The underlying REAL session (its snapshot reads the live fleet). */
  readonly realSession: NeatRacingSimWorkerSession;
}

/** Every sim-worker boot the harness produced, in boot order. */
const capturedSimWorkerBoots: CapturedSimWorkerBoot[] = [];

/** Returns the boot at the given index (or throws when absent). */
function capturedBootAt(index: number): CapturedSimWorkerBoot {
  const boot = capturedSimWorkerBoots[index];
  if (boot === undefined) {
    throw new Error(`sim-worker boot ${index} was never created`);
  }
  return boot;
}

/**
 * The optional doctor of the `frames` reply payload.
 *
 * WHY: the harness releases no simulation tick on its own during the
 * promotion drive, so a forward-lap crossing can never happen there. The
 * doctor replaces the WORKER's frames payload with a FRESH object (the
 * worker's own state is never mutated), which lets a test publish the lap
 * count the real lap authority never reached. It is null whenever no test
 * installed one.
 */
let framesReplyDoctor:
  | ((
      payload: NeatRacingSimWorkerFramesPayload,
    ) => NeatRacingSimWorkerFramesPayload)
  | null = null;

/** The published lap counter per sim-worker boot (one counter per round). */
const lapCountersByBoot: number[] = [];

/**
 * Applies the installed `frames` reply doctor to one reply list.
 *
 * @param responses - The replies the REAL session answered with.
 * @returns The very same replies when no doctor is installed, the
 *   doctored `frames` replies otherwise.
 */
function applyFramesReplyDoctor(
  responses: NeatRacingSimWorkerResponse[],
): NeatRacingSimWorkerResponse[] {
  const doctor = framesReplyDoctor;
  if (doctor === null) {
    return responses;
  }
  return responses.map((response): NeatRacingSimWorkerResponse =>
    response.type === 'frames'
      ? { type: 'frames', payload: doctor(response.payload) }
      : response,
  );
}

/**
 * Wraps a REAL pure session so every request is captured before the REAL
 * handler answers it (the worker module itself is unmocked beyond
 * `createSimWorkerSession`).
 */
function createCapturingSimWorkerSession(
  factory: typeof createRealSimWorkerSession,
): CapturedSimWorkerBoot {
  const realSession: NeatRacingSimWorkerSession = factory();
  const capturedRequests: NeatRacingSimWorkerRequest[] = [];
  const session: NeatRacingSimWorkerSession = {
    handleRequest: (request: NeatRacingSimWorkerRequest) => {
      capturedRequests.push(request);
      const responses = applyFramesReplyDoctor(
        realSession.handleRequest(request),
      );
      for (const response of responses) {
        if (response.type === 'frames') {
          capturedFramesPayloads.push(response.payload);
        }
      }
      return responses;
    },
    get snapshot(): NeatRacingSimWorkerSessionSnapshot {
      return realSession.snapshot;
    },
  };
  return { capturedRequests, session, realSession };
}

/** Returns the init request (the first captured request, typed honestly). */
function firstRequest(boot: CapturedSimWorkerBoot): NeatRacingSimWorkerRequest {
  const request = boot.capturedRequests.at(0);
  if (request === undefined) {
    throw new Error('the boot never sent a request');
  }
  return request;
}

/** Returns one captured boot's init payload. */
function firstInitPayload(
  boot: CapturedSimWorkerBoot,
): NeatRacingSimWorkerInitPayload {
  const request = firstRequest(boot);
  if (request.type !== 'init') {
    throw new Error(`expected an init request, received '${request.type}'`);
  }
  return request.payload;
}

/* ------------------------------------------------------------------------- *
 * Mode switches and frame driving.
 * ------------------------------------------------------------------------- */

/**
 * Switches the boot-mounted REAL race mode selector through its DOM control.
 *
 * @param mode - The race mode the lane switches into.
 */
function switchRaceMode(mode: 'ai' | 'manual'): void {
  const select = document.querySelector<HTMLSelectElement>(
    `#${HUD_OUTPUT_ID} select`,
  );
  if (select === null) {
    throw new Error('the race mode selector was never mounted');
  }
  select.value = mode;
  select.dispatchEvent(new Event('change'));
}

/** RAF callbacks captured from the boot loop; tests drive them manually. */
let scheduledRafCallbacks: FrameRequestCallback[] = [];

/** The monotonic timestamp the next driven frame hands to the loop. */
let drivenTimestampMs = FIRST_FRAME_TIMESTAMP_MS;

/**
 * Drives ONE frame by invoking the boot loop's captured RAF callback with
 * the advancing frame timestamp (no timers, no real rAF).
 */
function driveOneFrame(): void {
  const callback = scheduledRafCallbacks.at(-1);
  if (callback === undefined) {
    throw new Error('the loop never scheduled a frame callback');
  }
  callback(drivenTimestampMs);
  drivenTimestampMs += FRAME_STEP_MS;
}

/* ------------------------------------------------------------------------- *
 * Harness resets.
 * ------------------------------------------------------------------------- */

/** Live-branch boot doubles (the same inert gl fixture as the entry suite). */
let liveContextFixture: { result: RenderingContextInitResult };

/**
 * Resets the jsdom host shell to the two reserved containers (the game
 * output plus its HUD subtree host, which the REAL selector requires).
 */
function resetHostShell(): void {
  document.body.innerHTML =
    '<div id="neatracing-output"><div id="neatracing-hud-output"></div></div>';
}

/** Returns the boot function installed by the side-effect boot import. */
function getBootStart(): NeatRacingStart {
  const start = window.neatRacingStart;
  if (typeof start !== 'function') {
    throw new Error('neatRacingStart was not registered on window');
  }
  return start;
}

/** Returns the live stop handle from the last boot. */
let liveStopHandle: NeatRacingStop | null = null;

/** Boots the live loop on a reset harness and records the stop handle. */
function bootLiveLoop(): NeatRacingStop {
  const stop = getBootStart()('neatracing-output');
  liveStopHandle = stop;
  return stop;
}

/**
 * The inert WebGL2 boot double shared by every live-branch test.
 *
 * @returns The inert rendering context fixture.
 */
function createLiveContextFixture(): { result: RenderingContextInitResult } {
  const gl = {
    clear: (): void => undefined,
    clearColor: (): void => undefined,
    drawArrays: (): void => undefined,
    viewport: (): void => undefined,
  };
  const result: RenderingContextInitResult = {
    ok: true,
    gl: gl as unknown as WebGL2RenderingContext,
    isContextLost: (): boolean => false,
  };
  return { result };
}

const mockedInitRenderingContext = jest.mocked(initRenderingContext);
const mockedRunShaderWarmup = jest.mocked(runShaderWarmup);
const mockedAdvanceFrameSteps = jest.mocked(advanceFrameSteps);
const mockedUpdateFullWindowRendererSize = jest.mocked(
  updateFullWindowRendererSize,
);
const mockedCreateHud = jest.mocked(createHud);
const mockedProbeEpisode = jest.mocked(
  runNeatRacingScriptedBaselineProbeEpisode,
);
const mockedCreateSimWorkerSession = jest.mocked(createSimWorkerSession);

/** A deterministic probe double result (the honest lap-pace stand-in). */
const PROBE_RESULT: NeatRacingScriptedBaselineProbeResult = {
  meanLapSpeed: 30,
  lapTicks: 600,
  completedLap: true,
  consumedTicks: 600,
};

/** Standard 1280x720 backing-store size-double result. */
function createMockRendererSizeUpdate(): CanvasDimensions {
  return { width: 1280, height: 720 };
}

beforeEach(() => {
  jest.resetAllMocks();

  liveContextFixture = createLiveContextFixture();
  mockedInitRenderingContext.mockReturnValue(liveContextFixture.result);
  mockedRunShaderWarmup.mockReturnValue({
    warmedProgramCount: 0,
    offscreenDrawCount: 0,
  });
  mockedUpdateFullWindowRendererSize.mockReturnValue(
    createMockRendererSizeUpdate(),
  );
  mockedAdvanceFrameSteps.mockReturnValue(0);
  mockedCreateHud.mockImplementation(() => ({
    container: document.createElement('div'),
    speedLabel: document.createElement('div'),
    lapLabel: document.createElement('div'),
    timeLabel: document.createElement('div'),
    update: (): void => undefined,
  }));
  mockedProbeEpisode.mockReturnValue(PROBE_RESULT);
  mockedCreateSimWorkerSession.mockImplementation(() => {
    const boot = createCapturingSimWorkerSession(createRealSimWorkerSession);
    capturedSimWorkerBoots.push(boot);
    return boot.session;
  });

  capturedFramesPayloads.length = 0;
  capturedSimWorkerBoots.length = 0;
  lapCountersByBoot.length = 0;
  mockCapturedAdaptationTuples.length = 0;
  mockCapturedAdaptationBootSummaries.length = 0;
  framesReplyDoctor = null;
  scheduledRafCallbacks = [];
  drivenTimestampMs = FIRST_FRAME_TIMESTAMP_MS;
  window.requestAnimationFrame = (callback: FrameRequestCallback): number => {
    scheduledRafCallbacks.push(callback);
    return scheduledRafCallbacks.length;
  };
  liveStopHandle = null;
  jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  resetHostShell();
});

afterEach(() => {
  if (liveStopHandle !== null) {
    liveStopHandle();
    liveStopHandle = null;
  }
});

/* ------------------------------------------------------------------------- *
 * The promotion drive (the doctor that publishes the forward crossings).
 * ------------------------------------------------------------------------- */

/**
 * Installs the lap-crossing doctor: every driven frame publishes ONE
 * forward lap crossing on the round the frame belongs to.
 *
 * WHY the per-boot counter: a promoted round REBOOTS the worker, so the
 * fresh round's lap counter restarts at zero — the doctor's counter is
 * therefore kept per boot, exactly like the round it belongs to.
 */
function installLapCrossingDoctor(): void {
  framesReplyDoctor = (
    payload: NeatRacingSimWorkerFramesPayload,
  ): NeatRacingSimWorkerFramesPayload => {
    const bootIndex = capturedSimWorkerBoots.length - 1;
    const nextLap = (lapCountersByBoot[bootIndex] ?? 0) + 1;
    lapCountersByBoot[bootIndex] = nextLap;
    const cars = payload.cars.map((car, carIndex) =>
      carIndex === 0 ? { ...car, speed: 25 } : car,
    );
    return {
      ...payload,
      cars,
      rules: { ...payload.rules, lapsCompleted: nextLap },
    };
  };
}

/**
 * Drives the AI lane until the given number of forward lap crossings has
 * been published (the crossings themselves are the doctor's).
 *
 * @param count - The number of crossings to publish.
 */
function driveLapCrossings(count: number): void {
  for (let index = 0; index < count; index += 1) {
    driveOneFrame();
  }
}

/**
 * Boots the AI lane and climbs the ladder to the given rung (the REAL goal
 * machine, the REAL promotion predicate, the doctor-published crossings).
 *
 * @param targetLevel - The rung the lane must reach.
 */
function climbLadderTo(targetLevel: number): void {
  bootLiveLoop();
  switchRaceMode('ai');
  installLapCrossingDoctor();
  let level = 1;
  while (level < targetLevel) {
    driveLapCrossings(PROMOTION_CROSSING_COUNT);
    level += 1;
  }
}

/* ------------------------------------------------------------------------- *
 * The published-state scan (the location-tolerant witness).
 * ------------------------------------------------------------------------- */

/**
 * Collects every object under the given root that satisfies the predicate.
 *
 * WHY a deep scan: the RED must not pin WHERE the live lane publishes the
 * fresh round tuple or the rival-swap payload (the round's opponent
 * snapshot, the worker's init payload and the frames reply are all
 * legitimate homes), so the witness walks the published state instead of
 * naming one field.
 *
 * @param root - The published object to walk.
 * @param predicate - The shape test every visited object must satisfy.
 * @param seen - The already-visited objects (the cycle guard).
 * @returns The matching objects.
 */
function collectMatchingObjects(
  root: unknown,
  predicate: (value: Record<string, unknown>) => boolean,
  seen: WeakSet<object>,
  depth = 0,
): Record<string, unknown>[] {
  if (depth > 6 || root === null || typeof root !== 'object') {
    return [];
  }
  if (seen.has(root)) {
    return [];
  }
  seen.add(root);
  const record = root as Record<string, unknown>;
  const found: Record<string, unknown>[] = predicate(record) ? [record] : [];
  for (const value of Object.values(record)) {
    found.push(...collectMatchingObjects(value, predicate, seen, depth + 1));
  }
  return found;
}

/** Every published root the host handed out during the test. */
function collectPublishedRoots(): unknown[] {
  const roots: unknown[] = [];
  for (const tuple of mockCapturedAdaptationTuples) {
    roots.push(tuple);
  }
  for (const boot of capturedSimWorkerBoots) {
    for (const request of boot.capturedRequests) {
      roots.push(request);
    }
  }
  for (const payload of capturedFramesPayloads) {
    roots.push(payload);
  }
  return roots;
}

/**
 * Collects every rival-swap payload the live lane published.
 *
 * @returns The published rival-swap payloads (in discovery order).
 */
function collectRivalSwapPayloads(): NeatRacingRivalSwapPayload[] {
  const seen = new WeakSet<object>();
  const payloads: NeatRacingRivalSwapPayload[] = [];
  for (const root of collectPublishedRoots()) {
    for (const match of collectMatchingObjects(
      root,
      (value) => value.kind === NEATRACING_RIVAL_SWAP_PAYLOAD_KIND,
      seen,
    )) {
      payloads.push(match as unknown as NeatRacingRivalSwapPayload);
    }
  }
  return payloads;
}

/** One published object that LOOKS like a promotion round tuple. */
interface PromotionRoundTupleCandidate {
  /** The promoted rung the candidate's opponent snapshot names. */
  readonly level: number | null;
  /** The candidate's fleet size (the next rung's agent count). */
  readonly agentCount: number | null;
  /** The candidate's seed-network weights (null when absent). */
  readonly seedNetworkWeights: Float32Array | null;
  /** The candidate's seed-network envelope (null when absent). */
  readonly seedNetworkEnvelope: unknown;
  /** The candidate's carried innovation counter (null when absent). */
  readonly carriedInnovationCounter: number | null;
}

/**
 * Reads one published object as a promotion round tuple candidate.
 *
 * @param value - The published object.
 * @returns The candidate view of the object.
 */
function toPromotionRoundTupleCandidate(
  value: Record<string, unknown>,
): PromotionRoundTupleCandidate {
  const snapshot = value.opponentSnapshot;
  const level =
    typeof snapshot === 'object' &&
    snapshot !== null &&
    typeof (snapshot as { level?: unknown }).level === 'number'
      ? (snapshot as { level: number }).level
      : null;
  const seedNetwork = value.seedNetwork;
  const seedNetworkRecord =
    typeof seedNetwork === 'object' && seedNetwork !== null
      ? (seedNetwork as Record<string, unknown>)
      : null;
  const weights = seedNetworkRecord?.weights;
  return {
    level,
    agentCount: typeof value.agentCount === 'number' ? value.agentCount : null,
    seedNetworkWeights: weights instanceof Float32Array ? weights : null,
    seedNetworkEnvelope: seedNetworkRecord?.envelope ?? null,
    carriedInnovationCounter:
      typeof value.carriedInnovationCounter === 'number'
        ? value.carriedInnovationCounter
        : null,
  };
}

/**
 * Collects every published object that carries the promotion round tuple's
 * two D4 fields (the seed network and the carried innovation counter).
 *
 * @returns The published promotion round tuple candidates.
 */
function collectPromotionRoundTupleCandidates(): PromotionRoundTupleCandidate[] {
  const seen = new WeakSet<object>();
  const candidates: PromotionRoundTupleCandidate[] = [];
  for (const root of collectPublishedRoots()) {
    for (const match of collectMatchingObjects(
      root,
      (value) => 'seedNetwork' in value && 'carriedInnovationCounter' in value,
      seen,
    )) {
      candidates.push(toPromotionRoundTupleCandidate(match));
    }
  }
  return candidates;
}

/**
 * Decodes the payload's byte-exact hex weights into a float32 view.
 *
 * @param hex - The payload's hex weights.
 * @returns The decoded weights.
 */
function decodeHexWeights(hex: string): Float32Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return new Float32Array(bytes.buffer);
}

/* ------------------------------------------------------------------------- *
 * The seam-level fleet fixture (the five-car replay substrate).
 * ------------------------------------------------------------------------- */

/**
 * Builds the fixture track: the SHIPPED autopilot fixture line plus the
 * segment the observation assembler and the spawn resolver read.
 *
 * @returns The frozen fixture track geometry.
 */
function createFleetFixtureTrackSpec(): TrackSpec {
  return {
    seed: FLEET_FIXTURE_TUPLE_SEED,
    layoutVersion: 1,
    sizeBucket: 'promotion-live-fixture',
    segments: [{ startX: 0, startY: 0, endX: 39, endY: 0, width: 12 }],
    splineSamples: createTrackSpecForAutopilotFixture(),
    laneCount: 2,
  };
}

/**
 * Builds the fixture fleet: car 0 is the HUMAN car (no adopted network), the
 * other cars are the AGENT cars the seam drives.
 *
 * @param carCount - The fleet size.
 * @returns The deterministic fixture car states.
 */
function createFleetFixtureFleet(carCount: number): PhysicsCarState[] {
  const fleet: PhysicsCarState[] = [];
  for (let carIndex = 0; carIndex < carCount; carIndex += 1) {
    fleet.push(
      createInitialPhysicsCarState(FLEET_FIXTURE_TUPLE_SEED + carIndex, {
        x: 2 + carIndex * 8,
        y: 0,
        heading: 0,
      }),
    );
  }
  return fleet;
}

/**
 * Creates a fixture network whose forward pass is a PURE function of the
 * observation vector (no RNG, no clock, no internal state).
 *
 * @returns The fixture network.
 */
function createFleetFixtureNetwork(): NeatRacingFleetCarController {
  return {
    setSeed: (): void => undefined,
    activate: (vector: readonly number[] | Float32Array): readonly number[] => [
      0.5 + 0.25 * Math.sin(vector[0] ?? 0),
      0.5 * Math.cos(vector[1] ?? 0),
    ],
  };
}

/**
 * Creates the fixture relay: a tiny in-memory adopted-snapshot store.
 *
 * @param carCount - The fleet size.
 * @returns The fixture relay.
 */
function createFleetFixtureRelay(
  carCount: number,
): NeatRacingFleetControllerRelay {
  const adopted = new Map<number, NeatRacingFleetCarController>();
  for (let carIndex = 1; carIndex < carCount; carIndex += 1) {
    adopted.set(carIndex, createFleetFixtureNetwork());
  }
  return {
    readCarController: (
      carIndex: number,
    ): NeatRacingFleetCarController | null => adopted.get(carIndex) ?? null,
  };
}

/**
 * Creates the fixture seam over the given relay.
 *
 * @param relay - The relay publishing the adopted per-car networks.
 * @param level - The observation rung the seam observes at.
 * @returns The seam under test.
 */
function createFleetFixtureSeam(
  relay: NeatRacingFleetControllerRelay,
  level: NeatRacingObservationLevel,
): NeatRacingFleetControlSeam {
  return createNeatRacingFleetControl({
    tupleSeed: FLEET_FIXTURE_TUPLE_SEED,
    trackSpec: createFleetFixtureTrackSpec(),
    relay,
    level,
  });
}

/**
 * Advances the fixture fleet by one tick of the published controls.
 *
 * @param fleet - The live fixture fleet (mutated in place).
 * @param controls - The published controls of this tick, per car.
 */
function advanceFleetFixtureFleet(
  fleet: PhysicsCarState[],
  controls: readonly (PhysicsStepInput | null)[],
): void {
  for (let carIndex = 0; carIndex < fleet.length; carIndex += 1) {
    const control = controls[carIndex];
    if (control === null || control === undefined) {
      continue;
    }
    const car = fleet[carIndex];
    car.x += control.throttle * FLEET_FIXTURE_ADVANCE_M_PER_THROTTLE;
    car.speed = control.throttle * FLEET_FIXTURE_SPEED_PER_THROTTLE;
  }
}

/**
 * Runs ONE frozen-snapshot fleet session and collects the published control
 * stream of every car.
 *
 * Educational note: the published control buffer is REUSED across ticks (the
 * zero-allocation law), so the stream COPIES each control before holding it.
 *
 * @param carCount - The fleet size.
 * @param level - The observation rung the seam observes at.
 * @returns The per-tick control stream of the whole fleet.
 */
function collectFleetControlStream(
  carCount: number,
  level: NeatRacingObservationLevel,
): readonly (readonly ({ throttle: number } | null)[])[] {
  const seam = createFleetFixtureSeam(createFleetFixtureRelay(carCount), level);
  const fleet = createFleetFixtureFleet(carCount);
  const stream: ({ throttle: number } | null)[][] = [];
  for (let tick = 0; tick < NEATRACING_REPLAY_ADAPTATION_TICKS; tick += 1) {
    const controls = fleet.map((_, carIndex) =>
      seam.computeControl(fleet, carIndex),
    );
    stream.push(
      controls.map((control) =>
        control === null ? null : { throttle: control.throttle },
      ),
    );
    advanceFleetFixtureFleet(fleet, controls);
  }
  return stream;
}

/**
 * Measures the seam's published control-buffer allocation over the WHOLE
 * adaptation window.
 *
 * WHY the window: "no per-frame allocation growth" is a claim about TIME,
 * so the witness compares the FIRST tick of the window with the LAST one —
 * the published buffer COUNT must not grow, and every agent car must still
 * hold the SAME buffer instance it held on the first tick.
 *
 * WHY the null filter: the HUMAN car (slot 0) has no adopted network, so
 * the seam publishes NO buffer for it — counting `null` as a buffer would
 * measure the fleet size instead of the allocation law.
 *
 * @param carCount - The fleet size.
 * @param level - The observation rung the seam observes at.
 * @returns The measured allocation facts.
 */
function measureFleetBufferAllocation(
  carCount: number,
  level: NeatRacingObservationLevel,
): {
  readonly firstTickBufferCount: number;
  readonly lastTickBufferCount: number;
  readonly sameInstancePerCar: boolean;
  readonly agentCarCount: number;
} {
  const seam = createFleetFixtureSeam(createFleetFixtureRelay(carCount), level);
  const fleet = createFleetFixtureFleet(carCount);
  const firstTick = fleet.map((_, carIndex) =>
    seam.computeControl(fleet, carIndex),
  );
  let lastTick = firstTick;
  for (let tick = 1; tick < NEATRACING_REPLAY_ADAPTATION_TICKS; tick += 1) {
    const controls = fleet.map((_, carIndex) =>
      seam.computeControl(fleet, carIndex),
    );
    advanceFleetFixtureFleet(fleet, controls);
    lastTick = controls;
  }
  const publishedBufferCount = (
    controls: readonly (PhysicsStepInput | null)[],
  ): number => new Set(controls.filter((control) => control !== null)).size;
  return {
    firstTickBufferCount: publishedBufferCount(firstTick),
    lastTickBufferCount: publishedBufferCount(lastTick),
    sameInstancePerCar: firstTick.every(
      (control, carIndex) => lastTick[carIndex] === control,
    ),
    agentCarCount: carCount - 1,
  };
}

/* ------------------------------------------------------------------------- *
 * RED-P4-4 — the promotion round boundary.
 * ------------------------------------------------------------------------- */

describe('the promotion round boundary of the live lane (RED-P4-4)', () => {
  it('publishes a fresh round tuple at the promoted rung with the next rung agent count', () => {
    climbLadderTo(2);

    const candidates = collectPromotionRoundTupleCandidates();
    if (!candidates.some((candidate) => candidate.level === 2)) {
      throw new Error(
        'the live lane must publish a FRESH promotion round tuple at the promoted rung (the seed network, the carried innovation counter and the next rung agent count): NO promotion round tuple was published at rung 2',
      );
    }
    expect(
      candidates.map((candidate) => ({
        level: candidate.level,
        agentCount: candidate.agentCount,
      })),
    ).toContainEqual({
      level: 2,
      agentCount: resolveNeatRacingLevelCarCount(2),
    });
  }, 120_000);

  it('carries the promoted rung seed network and the carried innovation counter', () => {
    climbLadderTo(2);

    const promoted = collectPromotionRoundTupleCandidates().find(
      (candidate) => candidate.level === 2,
    );
    if (promoted === undefined) {
      throw new Error(
        'the live lane must publish a FRESH promotion round tuple carrying the promoted rung seed network and the carried innovation counter: NO promotion round tuple was published at rung 2',
      );
    }
    expect({
      hasSeedNetworkWeights:
        promoted.seedNetworkWeights instanceof Float32Array &&
        promoted.seedNetworkWeights.length > 0,
      hasSeedNetworkEnvelope: promoted.seedNetworkEnvelope != null,
      carriedInnovationCounter: promoted.carriedInnovationCounter ?? null,
    }).toEqual({
      hasSeedNetworkWeights: true,
      hasSeedNetworkEnvelope: true,
      carriedInnovationCounter: expect.any(Number),
    });
  }, 120_000);

  it('re-pins the adaptation session onto the promoted rung', () => {
    climbLadderTo(3);

    const summaries = mockCapturedAdaptationBootSummaries;
    if (!summaries.some((summary) => summary.level === 3)) {
      throw new Error(
        'the live lane must RE-PIN the adaptation session onto the promoted rung (the session level and the rung fleet size): NO adaptation session was ever booted at rung 3',
      );
    }
    expect(summaries).toContainEqual({
      level: 3,
      carCount:
        resolveNeatRacingLevelCarCount(3) + AI_LANE_BACKGROUND_CARRIER_COUNT,
    });
  }, 120_000);
});

/* ------------------------------------------------------------------------- *
 * RED-P4-5 — the rival swap at a lap/line boundary.
 * ------------------------------------------------------------------------- */

/**
 * Drives the live lane's five-car round until a rival crosses the lap line
 * (the published rival-swap payload is the witness) and returns the
 * published payloads.
 *
 * @returns The published rival-swap payloads.
 */
function driveFiveCarRoundUntilRivalSwap(): NeatRacingRivalSwapPayload[] {
  climbLadderTo(3);
  // The doctor is retired: the round's OWN lap authority drives the frame
  // budget from here on (the rival crossings are the worker's own fact).
  framesReplyDoctor = null;
  mockedAdvanceFrameSteps.mockReturnValue(RIVAL_DRIVE_TICKS_PER_FRAME);
  for (let index = 0; index < RIVAL_DRIVE_FRAME_BUDGET; index += 1) {
    driveOneFrame();
    if (index % 25 === 24) {
      const payloads = collectRivalSwapPayloads();
      if (payloads.length > 0) {
        return payloads;
      }
    }
  }
  return collectRivalSwapPayloads();
}

describe('the rival swap at a lap/line boundary (RED-P4-5)', () => {
  it('publishes a rival-swap payload when a rival crosses the lap line in the five-car round', () => {
    const payloads = driveFiveCarRoundUntilRivalSwap();
    if (payloads.length === 0) {
      throw new Error(
        'the live lane must publish a rival-swap payload when a RIVAL crosses the lap line: the five-car round published NO rival-swap payload',
      );
    }
    expect(payloads.map((payload) => payload.crossing.carIndex)).not.toContain(
      NEATRACING_RIVAL_SWAP_MAIN_AGENT_CAR_INDEX,
    );
  }, 180_000);

  it('mutates ONLY the rival weights: the main agent is never a target and every delta stays bounded', () => {
    const payloads = driveFiveCarRoundUntilRivalSwap();
    const payload = payloads[0];
    if (payload === undefined) {
      throw new Error(
        'the five-car round published NO rival-swap payload: the live lane never performs a rival swap at a lap/line boundary',
      );
    }
    const minds = restoreNeatRacingRivalSwapReplacementMinds(payload);
    const crossingWeights = decodeHexWeights(payload.crossing.weights);
    let maxAbsDelta = 0;
    for (const mind of minds) {
      for (let index = 0; index < mind.weights.length; index += 1) {
        const delta = Math.abs(
          (mind.weights[index] ?? 0) - (crossingWeights[index] ?? 0),
        );
        maxAbsDelta = Math.max(maxAbsDelta, delta);
      }
    }
    expect({
      targetCarIndices: minds.map((mind) => mind.carIndex),
      mainAgentTargeted: minds.some(
        (mind) => mind.carIndex === NEATRACING_RIVAL_SWAP_MAIN_AGENT_CAR_INDEX,
      ),
      crossingCarTargeted: minds.some(
        (mind) => mind.carIndex === payload.crossing.carIndex,
      ),
    }).toEqual({
      targetCarIndices: [...payload.propagatedCarIndices],
      mainAgentTargeted: false,
      crossingCarTargeted: false,
    });
    expect(maxAbsDelta).toBeLessThanOrEqual(
      NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX,
    );
  }, 180_000);

  it('is a tested no-op at the two-car rung: the single rival has no replacement target', () => {
    climbLadderTo(2);
    framesReplyDoctor = null;
    mockedAdvanceFrameSteps.mockReturnValue(RIVAL_DRIVE_TICKS_PER_FRAME);
    for (let index = 0; index < RIVAL_DRIVE_FRAME_BUDGET; index += 1) {
      driveOneFrame();
    }
    const payloads = collectRivalSwapPayloads();

    expect({
      roundCarCount: firstInitPayload(
        capturedBootAt(capturedSimWorkerBoots.length - 1),
      ).agentCount,
      propagatedCarIndices: payloads.map(
        (payload) => payload.propagatedCarIndices,
      ),
    }).toEqual({
      roundCarCount: resolveNeatRacingLevelCarCount(2),
      propagatedCarIndices: payloads.map(() => []),
    });
  }, 180_000);
});

/* ------------------------------------------------------------------------- *
 * RED-P4-6 — the five-car same-tuple replay.
 * ------------------------------------------------------------------------- */

describe('the five-car same-tuple replay (RED-P4-6)', () => {
  it('replays the five-car same-tuple round byte-identically across the adaptation window', () => {
    const fiveCarCount = resolveNeatRacingLevelCarCount(3);

    const streamA = collectFleetControlStream(
      fiveCarCount,
      FLEET_FIXTURE_LEVEL,
    );
    const streamB = collectFleetControlStream(
      fiveCarCount,
      FLEET_FIXTURE_LEVEL,
    );

    expect(JSON.stringify(streamA)).toBe(JSON.stringify(streamB));
  }, 60_000);

  it('reuses one published control buffer per agent car: no per-frame allocation growth against the one-car baseline', () => {
    const fiveCarCount = resolveNeatRacingLevelCarCount(3);
    const oneCarCount = resolveNeatRacingLevelCarCount(1);
    const fiveCar = measureFleetBufferAllocation(
      fiveCarCount,
      FLEET_FIXTURE_LEVEL,
    );
    const oneCar = measureFleetBufferAllocation(oneCarCount, 1);

    expect({
      fiveCarFirstTickBufferCount: fiveCar.firstTickBufferCount,
      fiveCarLastTickBufferCount: fiveCar.lastTickBufferCount,
      fiveCarSameInstancePerCar: fiveCar.sameInstancePerCar,
      fiveCarAgentCarCount: fiveCar.agentCarCount,
      oneCarFirstTickBufferCount: oneCar.firstTickBufferCount,
      oneCarLastTickBufferCount: oneCar.lastTickBufferCount,
      oneCarSameInstancePerCar: oneCar.sameInstancePerCar,
      oneCarAgentCarCount: oneCar.agentCarCount,
    }).toEqual({
      fiveCarFirstTickBufferCount: fiveCarCount - 1,
      fiveCarLastTickBufferCount: fiveCarCount - 1,
      fiveCarSameInstancePerCar: true,
      fiveCarAgentCarCount: fiveCarCount - 1,
      oneCarFirstTickBufferCount: oneCarCount - 1,
      oneCarLastTickBufferCount: oneCarCount - 1,
      oneCarSameInstancePerCar: true,
      oneCarAgentCarCount: oneCarCount - 1,
    });
  }, 60_000);

  it('carries a rival-swap inside the five-car replay window', () => {
    const payloads = driveFiveCarRoundUntilRivalSwap();
    if (payloads.length === 0) {
      throw new Error(
        'the five-car replay window must CARRY a rival-swap: the live lane never performs a rival swap at a lap/line boundary',
      );
    }
    expect(payloads.length).toBeGreaterThan(0);
    expect(
      payloads.map(
        (payload) => serializeNeatRacingRivalSwapPayload(payload).length > 0,
      ),
    ).toEqual(payloads.map(() => true));
  }, 180_000);
});
