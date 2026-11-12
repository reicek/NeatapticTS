/** @jest-environment jsdom */

/**
 * GREEN contract tests for the NeatRacing host sim-worker wiring (slice
 * P2-S4-C; AC-033 end-to-end wiring of the human car 0 racing
 * evolving-agent rivals through the simulation-worker protocol; spec
 * sections 13.1 AC-033, 10.2, 11.2; LD-10 / LD-12; E11(a)).
 *
 * GREEN-ONLY by dispatch: the wiring ships with its tests (the slice's
 * red_test note is superseded by the P2-S4-A/B precedent), so this suite
 * pins the live wiring contracts the re-plumb established instead of
 * gating a red phase:
 *
 * - The boot init contract: one `init` request on the demo tuple — the
 *   generated track spec, `resolveNeatRacingLevelCarCount(1)` = 1 (level
 *   1 is a lone car 0 with no rivals; rivals join from rung 2), the
 *   packed schema version, the honest opponent snapshot, the demo track
 *   id, `raceMode: 'manual'` ALWAYS (LD-12: the human takes car 0 in
 *   BOTH UI modes — the worker's `'ai'` race mode would steal car 0's
 *   controls and is never booted by the host) and the default feature
 *   flags.
 * - The frames rebinding: the LAST `frames` reply's rules clone reaches
 *   the HUD read model, the protocol clone's car-0 speed feeds both the
 *   HUD and the drift seam (one authority), and the level-1 fleet is a
 *   lone car 0 (per-rival pose bags exist only for cars 1..n — level 1
 *   has none, so the pin rides the panel's rivals-only surface).
 * - The lap-facts pipeline: `resolveNeatRacingLapFactAdvance` — the
 *   exported pure bookkeeping over the worker's own car-0 lap authority
 *   (the host detector was REMOVED dead in this slice) — plus the live
 *   frame consumption without invented crossings.
 * - The AI-mode goal lifecycle: the REAL `ai-mode.goals` machine behind
 *   the exported decision surfaces (`resolveNeatRacingAiModeGoalSessionForMode`,
 *   `resolveNeatRacingAiModeRebaseRung`); a synthetic promoted outcome is
 *   produced by the REAL machine (20 counted crossings over the honest
 *   promotion thresholds), never by a fabricated outcome object.
 * - The restart carrier in BOTH switch directions: `'stop'` + a fresh
 *   init, the panel mounted/unmounted, and the fresh round's HUD read
 *   model starting over.
 * - The most-forward network panel: mount/sizing/unmount, the level-1
 *   idle path without throw, the pure along-track progress helper (meters,
 *   forward growth, wrap-aware branches) over a synthetic chord index,
 *   and the REAL adapter's most-forward law.
 * - The stop handle: the terminal `stop` request, the panel uninstall and
 *   a symmetric teardown without throw.
 *
 * Seam doubles: the exact pattern of `../browser-entry/browser-entry.test.ts`
 * (context.service / render-loop.utils / canvas-dimensions.utils / hud
 * spread mocks) — EXCEPT mode-selector (kept REAL so a real selector
 * drives the DOM switches the carrier consumes) and the sim-worker
 * module, whose `createSimWorkerSession` is replaced by a capturing
 * wrapper over a REAL pure session. The worker module is used headless
 * like its own P2-S1-A seam tests (the bottom `self.onmessage`
 * assignment is harmless in jsdom). The scripted-baseline probe is an
 * inert double `runNeatRacingScriptedBaselineProbeEpisode` ONLY — every
 * other promotion law (conditions, thresholds, ladder rungs) stays REAL,
 * and the goal machine always runs REAL (`ai-mode.goals` is unmocked).
 *
 * Fixture rules: the per-file jsdom pragma turns on the DOM for this
 * file only; the host shell carries BOTH reserved containers
 * (`neatracing-output` + `neatracing-hud-output`) because the REAL
 * selector mount resolves the hud-output id or throws. Frames are driven
 * by the captured RAF callbacks with advancing timestamps (no timers).
 * The WebGL2 boot branches ride the same inert context doubles as the
 * browser-entry suite; the panel canvas' `getContext` returns null
 * (jsdom), which the panel honestly treats as an undrawable canvas.
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
 * The scripted-baseline probe is the ONLY promotion seam doubled: the
 * 12k-tick probe episode is a boot-cost decision under test, not a law
 * under test (the probe's own suite pins its law; the promotion
 * conditions it feeds stay REAL).
 */
jest.mock('../ai/promotion.rules', () => ({
  ...jest.requireActual('../ai/promotion.rules'),
  // The `__esModule` marker is re-declared on the mock: the spread above
  // drops the actual module's marker (it is not enumerable), and a
  // namespace import of a mock WITHOUT the marker is an interop COPY —
  // a spy installed on that copy is never observed by the goal machine.
  __esModule: true,
  runNeatRacingScriptedBaselineProbeEpisode: jest.fn(),
}));

/**
 * The adaptation session module is jest-mocked in this file (the AI
 * lane's LIVE drive recorder is the ONLY doubled seam): the factory
 * wraps the REAL session, so the host keeps driving a REAL session and
 * the seam injection still rides the REAL controller object, while the
 * wrapper RECORDS the simulation ticks the session was driven at.
 *
 * The recorder arrays carry the `mock` prefix on purpose: a `jest.mock`
 * factory may only reach module-scope bindings whose name starts with
 * `mock` (the hoisting contract), so the recorder stays reachable from
 * the hoisted factory.
 */
jest.mock('../ai/racing-adaptation.session', () => {
  const actual = jest.requireActual<
    typeof import('../ai/racing-adaptation.session')
  >('../ai/racing-adaptation.session');
  return {
    ...actual,
    // The `__esModule` marker is re-declared on the mock: the spread
    // above drops the actual module's marker (it is not enumerable), and
    // a namespace import of a mock WITHOUT the marker is an interop COPY.
    __esModule: true,
    createNeatRacingAdaptationSession: (
      options: Parameters<typeof actual.createNeatRacingAdaptationSession>[0],
    ) => {
      const realSession = actual.createNeatRacingAdaptationSession(options);
      mockCapturedAdaptationSessions.push(realSession);
      return {
        ...realSession,
        adaptOnTick: (
          input: Parameters<typeof realSession.adaptOnTick>[0],
        ): ReturnType<typeof realSession.adaptOnTick> => {
          mockCapturedAdaptationDriveTicks.push(input.tick);
          // The reference car's fed position: the boundary pins read the
          // drive's position term (one plus the rivals counted ahead) as
          // the honest stale-rival witness.
          const referenceCar = input.cars[0] ?? null;
          mockCapturedAdaptationTickCar0Positions.push(
            referenceCar === null ? null : referenceCar.position,
          );
          return realSession.adaptOnTick(input);
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
  NEATRACING_FALLBACK_MESSAGE,
  initRenderingContext,
  runShaderWarmup,
} from '../engine/context.service';
import type { RenderingContextInitResult } from '../engine/context.service';
import type { NeatRacingHud } from '../host/hud';
import { createHud } from '../host/hud';
import type { CanvasDimensions } from '../browser-entry/canvas-dimensions.utils';
import { updateFullWindowRendererSize } from '../browser-entry/canvas-dimensions.utils';
import { advanceFrameSteps } from '../host/render-loop.utils';
// The promotion rules module is jest-mocked in this file (the scripted
// baseline probe is the ONLY doubled seam), so the goal machine's
// predicate call is observed through a namespace spy: the spy stands in
// for the REAL predicate (it calls through) and records the node count
// the host fed it.
import * as promotionRules from '../ai/promotion.rules';
import {
  runNeatRacingScriptedBaselineProbeEpisode,
  resolveNeatRacingLevelCarCount,
  NEATRACING_AI_MIN_SESSION_LAPS,
  NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS,
  type NeatRacingScriptedBaselineProbeResult,
  type NeatRacingDrivabilityMeasurement,
  type NeatRacingPromotionRoundSnapshot,
} from '../ai/promotion.rules';
// The adaptation session module is jest-mocked in this file (the AI
// lane's LIVE drive recorder is the ONLY doubled seam): the host keeps
// driving a REAL session — the mock only RECORDS the simulation ticks
// the session was driven at.
import type { NeatRacingAdaptationSession } from '../ai/racing-adaptation.session';
// The module is jest-mocked above with a REAL-session wrapper, so this
// import resolves to the wrapped factory: creating a session here still
// builds a REAL adaptation session (the pins drive REAL relay laws).
import { createNeatRacingAdaptationSession } from '../ai/racing-adaptation.session';
import { resolveOscillationPromotionNodeThreshold } from '../ai/oscillation-penalty.utils';
import {
  createInitialPhysicsCarState,
  type PhysicsCarState,
} from '../physics/physics.step.service';
import { NEATRACING_FLEET_CONTROL_BRAKE } from '../workers/simulation-worker/nge.fleet.control';
// The carrier's LIVE network size at rung 1 is derived from the shipped
// laws (never a magic number): the level-1 observation width plus the
// controller's output count.
import {
  NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
  type NeatRacingObservationLevel,
} from '../ai/observation.assembler';
import { NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT } from '../ai/nge.controller';
import {
  createNeatRacingAiModeGoalSession,
  describeNeatRacingAiModeGoalPhase,
  NEATRACING_AI_MODE_START_LEVEL,
  recordNeatRacingAiModeGoalCrossing,
} from '../ai/ai-mode.goals';
import type {
  NeatRacingAiModeGoalCrossing,
  NeatRacingAiModeGoalCrossingOutcome,
  NeatRacingAiModeGoalSession,
} from '../ai/ai-mode.goals';
import {
  NEATRACING_DEFAULT_FEATURE_FLAGS,
  NEATRACING_PACK_SCHEMA_VERSION,
} from '../workers/simulation-worker/frame.pack.service';
import type { NeatRacingReproducibilityTuple } from '../workers/simulation-worker/reproducibility.utils';
import type {
  NeatRacingSimWorkerFramesPayload,
  NeatRacingSimWorkerInitPayload,
  NeatRacingSimWorkerRequest,
  NeatRacingSimWorkerResponse,
  NeatRacingSimWorkerSession,
  NeatRacingSimWorkerSessionSnapshot,
} from '../workers/simulation-worker/simulation-worker';
import { createSimWorkerSession } from '../workers/simulation-worker/simulation-worker';
import type { TrackSegment } from '../../racing_curriculum/track/track.generator.types';
// The shared most-forward law, consumed through the REAL visualizer
// adapter (one authority for the panel's winner).
import {
  buildNeatRacingTrackChordIndex,
  resolveNeatRacingAiModeGoalSessionForMode,
  resolveNeatRacingAiModeRebaseRung,
  resolveNeatRacingCarAlongProgressM,
  resolveNeatRacingLapFactAdvance,
} from '../browser-entry/browser-entry';
import type {
  NeatRacingStart,
  NeatRacingStop,
} from '../browser-entry/browser-entry';
import type { NeatRacingHudState } from '../host/hud';
import type { NeatRacingNetworkPanelCandidate } from '../visualizer/network-panel.adapter';
import { resolveMostForwardNeatRacingAgent } from '../visualizer/network-panel.adapter';
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

/** The panel canvas id the host mounts inside the reserved HUD container. */
const NETWORK_PANEL_CANVAS_ID = 'neatracing-network-panel';

/** The reserved host DOM id for the HUD subtree (the selector + HUD host). */
const HUD_OUTPUT_ID = 'neatracing-hud-output';

/** Timestamp handed to the captured RAF callback when a test drives frame one. */
const FIRST_FRAME_TIMESTAMP_MS = 0;

/** Timestamp step (ms) between driven frames (the 60 Hz render cadence). */
const FRAME_STEP_MS = 1000 / 60;

/** Frames the rebinding/restart contracts drive (a short honest window). */
const DRIVEN_FRAME_COUNT = 120;

/**
 * Frames the AI-lane tick-advance pin drives (the SAME simulation tick,
 * observed once per drawn frame — the condition that exposes a repeated
 * drive).
 */
const ADAPTATION_REPEAT_FRAME_COUNT = 5;

/** Chords the boundary pin's doctor places its rival clone ahead of car 0. */
const RIVAL_AHEAD_CHORD_COUNT = 12;

/* ------------------------------------------------------------------------- *
 * Captured HUD read model.
 * ------------------------------------------------------------------------- */

/** One captured HUD update call (the host's per-frame read model push). */
const hudUpdates: NeatRacingHudState[] = [];

/**
 * Capturing HUD double: the boot composition mounts a REAL-shaped `NeatRacingHud`
 * whose `update` records every pushed state for the rebinding assertions.
 */
function createCapturingHud(): NeatRacingHud {
  return {
    container: document.createElement('div'),
    speedLabel: document.createElement('div'),
    lapLabel: document.createElement('div'),
    timeLabel: document.createElement('div'),
    update: (state: NeatRacingHudState): void => {
      hudUpdates.push(state);
    },
  };
}

/** Returns the last captured HUD state (or throws when none was pushed). */
function lastHudUpdate(): NeatRacingHudState {
  const state = hudUpdates.at(-1);
  if (state === undefined) {
    throw new Error('the HUD read model was never updated');
  }
  return state;
}

/* ------------------------------------------------------------------------- *
 * Capturing sim-worker sessions over the REAL pure protocol machine.
 * ------------------------------------------------------------------------- */

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

/** Returns the last sim-worker boot (or throws when none exists). */
function lastCapturedBoot(): CapturedSimWorkerBoot {
  const boot = capturedSimWorkerBoots.at(-1);
  if (boot === undefined) {
    throw new Error('no sim-worker boot exists');
  }
  return boot;
}
/**
 * The optional doctor of the `frames` reply payload.
 *
 * WHY: the harness never advances the simulation (the frame-step double
 * releases zero ticks), so a forward-lap crossing can never happen on
 * its own. The doctor replaces the WORKER's frames payload with a FRESH
 * object (the worker's own state is never mutated), which lets a test
 * publish a lap count the real lap authority never reached. It is null
 * whenever no test installed one.
 */
let framesReplyDoctor:
  | ((
      payload: NeatRacingSimWorkerFramesPayload,
    ) => NeatRacingSimWorkerFramesPayload)
  | null = null;

/**
 * Applies the installed `frames` reply doctor to one reply list.
 *
 * @param responses - The replies the REAL session answered with.
 * @returns The very same replies when no doctor is installed, the
 *   doctored `frames` replies otherwise (every other reply passes
 *   through untouched).
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
 * Wraps a REAL pure session so every request is captured before the
 * REAL handler answers it (the worker module itself is unmocked beyond
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
      return applyFramesReplyDoctor(realSession.handleRequest(request));
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

/** Asserts the given boot's request log ends with the terminal stop. */
function isStoppedBoot(boot: CapturedSimWorkerBoot): boolean {
  const last = boot.capturedRequests.at(-1);
  return last?.type === 'stop';
}

/* ------------------------------------------------------------------------- *
 * Capturing the AI lane's LIVE adaptation drives.
 * ------------------------------------------------------------------------- */

/**
 * Every adaptation session the host created (in boot order).
 *
 * WHY: the session module is jest-mocked so the AI lane's drives are
 * observable; the recorder keeps the REAL session objects, so the host
 * still drives a REAL session (never a fabricated double).
 */
const mockCapturedAdaptationSessions: NeatRacingAdaptationSession[] = [];

/**
 * Every simulation tick the AI lane drove the adaptation session at (in
 * drive order).
 *
 * WHY: the tick ADVANCE gate's invariant is "at most one drive per
 * simulation tick", so the recorded ticks ARE the contract.
 */
const mockCapturedAdaptationDriveTicks: number[] = [];

/**
 * The car-0 POSITION term of EVERY recorded adaptation drive (in drive
 * order; null when the tick input carried no reference car).
 *
 * WHY: the rival progress meters are host-private state that only the
 * panel populates and only the drive consumes — the drive's own position
 * term (one plus the rivals counted ahead) is the lane-honest surface
 * that reveals them, so the round-boundary pins record what the drive
 * FED instead of reaching into the host's private state.
 */
const mockCapturedAdaptationTickCar0Positions: Array<number | null> = [];

/* ------------------------------------------------------------------------- *
 * Mode switches through the REAL selector (DOM change events).
 * ------------------------------------------------------------------------- */

/**
 * Switches the boot-mounted REAL race mode selector through its DOM
 * control: the select lives inside the reserved HUD output container,
 * and its change listener reports the switch to the host's carrier.
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

/* ------------------------------------------------------------------------- *
 * Frame driving.
 * ------------------------------------------------------------------------- */

/** RAF callbacks captured from the boot loop; tests drive them manually. */
let scheduledRafCallbacks: FrameRequestCallback[] = [];

/**
 * Drives `count` frames by invoking the boot loop's captured RAF
 * callbacks with advancing timestamps (no timers, no real rAF).
 */
function driveFrames(
  count: number,
  startTimestampMs = FIRST_FRAME_TIMESTAMP_MS,
): void {
  for (let frame = 0; frame < count; frame += 1) {
    const callback = scheduledRafCallbacks.at(-1);
    if (callback === undefined) {
      throw new Error('the loop never scheduled a frame callback');
    }
    callback(startTimestampMs + frame * FRAME_STEP_MS);
  }
}

/**
 * The macrotask turns one promise flush drains (the AI lane's adaptation
 * tick is async: the session publishes its live node counts when its
 * tick resolves).
 */
const PROMISE_FLUSH_TURN_COUNT = 3;

/**
 * Flushes the pending promise chains.
 *
 * WHY: the AI lane's adaptation tick resolves asynchronously, so the
 * session publishes its live node counts a few microtask turns after
 * the frame that started the tick. A macrotask turn drains every pending
 * microtask chain, so a short flush lets the session publish before the
 * next frame reads it.
 */
async function flushPendingPromises(): Promise<void> {
  for (let turn = 0; turn < PROMISE_FLUSH_TURN_COUNT; turn += 1) {
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
  }
}

/** Presses the given keyboard keys for one live-driving window. */
function pressKeys(keys: readonly ('ArrowUp' | 'ArrowRight')[]): void {
  for (const key of keys) {
    window.dispatchEvent(new KeyboardEvent('keydown', { key, code: key }));
  }
}

/** Releases the given keyboard keys. */
function releaseKeys(keys: readonly ('ArrowUp' | 'ArrowRight')[]): void {
  for (const key of keys) {
    window.dispatchEvent(new KeyboardEvent('keyup', { key, code: key }));
  }
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
 * The inert WebGL2 boot double shared by every live-branch test (the
 * composer path only needs the stable gl identity, never real draws).
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

/* ------------------------------------------------------------------------- *
 * Synthetic promoted outcomes (built by the REAL goal machine).
 * ------------------------------------------------------------------------- */

/** A drivability sample that passes the real gate: 25 >= 0.75 * 30, 0.2 <= 0.35, 0.05 <= 0.1. */
const PASSING_DRIVABILITY: NeatRacingDrivabilityMeasurement = {
  meanForwardSpeed: 25,
  scriptedBaselineLapPace: 30,
  oscillationMetric: 0.2,
  offTrackFraction: 0.05,
};

/** The level-1 node-count promotion threshold (the shared band source). */
const LVL1_NODE_THRESHOLD = resolveOscillationPromotionNodeThreshold(
  NEATRACING_AI_MODE_START_LEVEL,
);

/**
 * The carrier's LIVE network size at rung 1: the level-1 observation
 * width (82 input nodes) plus the controller's output count (2 output
 * nodes). The goal machine must be fed AT LEAST the carrier's whole
 * network — never a fabricated zero.
 */
const CARRIER_BASE_NODE_COUNT =
  NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT +
  NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT;

/**
 * Builds one honest forward crossing fixture at the level-1 band.
 */
function buildForwardCrossing(
  overrides: Partial<NeatRacingAiModeGoalCrossing> = {},
): NeatRacingAiModeGoalCrossing {
  return {
    forwardCrossing: true,
    lapsSinceLastCrossing: NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS,
    nodeCount: LVL1_NODE_THRESHOLD + 1,
    drivability: PASSING_DRIVABILITY,
    ...overrides,
  };
}

/**
 * Records a chain of crossings through the REAL machine and returns the
 * LAST outcome: after `NEATRACING_AI_MIN_SESSION_LAPS` counted crossings
 * the 20th call is the crossing that flips eligibility and promotes.
 */
function recordCrossingChain(
  session: NeatRacingAiModeGoalSession,
  count: number,
  crossing: NeatRacingAiModeGoalCrossing = buildForwardCrossing(),
): NeatRacingAiModeGoalCrossingOutcome {
  let current = session;
  let outcome: NeatRacingAiModeGoalCrossingOutcome | undefined;
  for (let index = 0; index < count; index += 1) {
    outcome = recordNeatRacingAiModeGoalCrossing(current, crossing);
    current = outcome.session;
  }
  if (outcome === undefined) {
    throw new Error('the crossing chain must not be empty');
  }
  return outcome;
}

/* ------------------------------------------------------------------------- *
 * Synthetic along-track progress fixtures (a 4-chord closed square).
 * ------------------------------------------------------------------------- */

/** Synthetic chord width (meters) for the progress fixtures. */
const SYNTHETIC_TRACK_WIDTH = 8;

/** Synthetic chord length (meters): four chords close a 40 m loop. */
const SYNTHETIC_CHORD_LENGTH_M = 10;

/**
 * Builds a closed counter-clockwise square of four 10 m chords.
 */
function buildSyntheticSegments(): TrackSegment[] {
  return [
    {
      startX: 0,
      startY: 0,
      endX: SYNTHETIC_CHORD_LENGTH_M,
      endY: 0,
      width: SYNTHETIC_TRACK_WIDTH,
    },
    {
      startX: SYNTHETIC_CHORD_LENGTH_M,
      startY: 0,
      endX: SYNTHETIC_CHORD_LENGTH_M,
      endY: SYNTHETIC_CHORD_LENGTH_M,
      width: SYNTHETIC_TRACK_WIDTH,
    },
    {
      startX: SYNTHETIC_CHORD_LENGTH_M,
      startY: SYNTHETIC_CHORD_LENGTH_M,
      endX: 0,
      endY: SYNTHETIC_CHORD_LENGTH_M,
      width: SYNTHETIC_TRACK_WIDTH,
    },
    {
      startX: 0,
      startY: SYNTHETIC_CHORD_LENGTH_M,
      endX: 0,
      endY: 0,
      width: SYNTHETIC_TRACK_WIDTH,
    },
  ];
}

/* ------------------------------------------------------------------------- *
 * The harness beforeEach (the browser-entry suite's reset discipline).
 * ------------------------------------------------------------------------- */

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
  mockedCreateHud.mockImplementation(createCapturingHud);
  mockedProbeEpisode.mockReturnValue(PROBE_RESULT);
  mockedCreateSimWorkerSession.mockImplementation(() => {
    const boot = createCapturingSimWorkerSession(createRealSimWorkerSession);
    capturedSimWorkerBoots.push(boot);
    return boot.session;
  });

  hudUpdates.length = 0;
  capturedSimWorkerBoots.length = 0;
  mockCapturedAdaptationSessions.length = 0;
  mockCapturedAdaptationDriveTicks.length = 0;
  mockCapturedAdaptationTickCar0Positions.length = 0;
  framesReplyDoctor = null;
  scheduledRafCallbacks = [];
  window.requestAnimationFrame = (callback: FrameRequestCallback): number => {
    scheduledRafCallbacks.push(callback);
    return scheduledRafCallbacks.length;
  };
  liveStopHandle = null;
  // The panel canvas' 2d context is jsdom-null by default (the adapter's
  // honest undrawable path); stub the prototype so no jsdom
  // "Not implemented" noise leaks into the run.
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
 * (a) The boot init contract.
 * ------------------------------------------------------------------------- */

describe('the sim-worker boot init contract (AC-033)', () => {
  it('boots exactly one capturing worker session and sends one init request first', () => {
    bootLiveLoop();

    expect(capturedBootAt(0).capturedRequests).toHaveLength(1);
  });

  it('carries the generated demo track seed in the tuple', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    expect(payload.seed).toBe(payload.trackSpec.seed);
  });

  it('boots the level-1 lone-car fleet (agentCount 1)', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    expect(payload.agentCount).toBe(resolveNeatRacingLevelCarCount(1));
  });

  it('carries the packed schema version', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    expect(payload.packSchemaVersion).toBe(NEATRACING_PACK_SCHEMA_VERSION);
  });

  it('carries the honest level-1 opponent snapshot', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    const snapshot: NeatRacingPromotionRoundSnapshot = {
      level: 1,
      carCount: resolveNeatRacingLevelCarCount(1),
    };
    expect(payload.opponentSnapshot).toEqual(snapshot);
  });

  it('carries the demo track identity', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    expect(payload.trackId).toBe('neatracing-demo-loop');
  });

  it('boots the worker in manual race mode always (the human keeps car 0)', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    expect(payload.raceMode).toBe('manual');
  });

  it('carries the default feature flags', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    expect(payload.featureFlags).toEqual(NEATRACING_DEFAULT_FEATURE_FLAGS);
  });

  it('hands the generated demo track spec to the worker', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    expect(payload.trackSpec.segments.length).toBeGreaterThan(0);
  });
});

/* ------------------------------------------------------------------------- *
 * (b) The frames rebinding.
 * ------------------------------------------------------------------------- */

describe('the frames rebinding through the sim-worker protocol', () => {
  it('publishes the protocol rules clone to the HUD read model', () => {
    bootLiveLoop();
    mockedAdvanceFrameSteps.mockReturnValue(1);
    pressKeys(['ArrowUp', 'ArrowRight']);
    driveFrames(DRIVEN_FRAME_COUNT);
    releaseKeys(['ArrowUp', 'ArrowRight']);

    expect(lastHudUpdate().raceState.level).toBe(1);
  });

  it('drives the HUD speed from the protocol car-0 speed', () => {
    bootLiveLoop();
    mockedAdvanceFrameSteps.mockReturnValue(1);
    pressKeys(['ArrowUp']);
    driveFrames(DRIVEN_FRAME_COUNT);
    releaseKeys(['ArrowUp']);

    expect(lastHudUpdate().speedUnitsPerS).toBeGreaterThan(0);
  });

  it('binds the drift seam to the protocol clone fields (one speed authority)', () => {
    bootLiveLoop();
    mockedAdvanceFrameSteps.mockReturnValue(1);
    pressKeys(['ArrowUp']);
    driveFrames(DRIVEN_FRAME_COUNT);
    releaseKeys(['ArrowUp']);

    const seam = (
      window as unknown as { neatRacingDriftState?: { speedMps: number } }
    ).neatRacingDriftState;
    expect(seam?.speedMps).toBe(lastHudUpdate().speedUnitsPerS);
  });

  it('keeps the level-1 fleet a lone car 0 with no rivals', () => {
    bootLiveLoop();
    mockedAdvanceFrameSteps.mockReturnValue(1);
    pressKeys(['ArrowUp']);
    driveFrames(DRIVEN_FRAME_COUNT);
    releaseKeys(['ArrowUp']);

    expect(lastCapturedBoot().realSession.snapshot.cars.length).toBe(1);
  });
});

/* ------------------------------------------------------------------------- *
 * (c) The lap-facts pipeline.
 * ------------------------------------------------------------------------- */

describe('the lap-facts pipeline over the worker lap authority', () => {
  it('reports no crossing when laps do not advance', () => {
    expect(resolveNeatRacingLapFactAdvance(0, 0).crossed).toBe(false);
  });

  it('reports one crossing with the forward-lap diff on the first lap', () => {
    expect(resolveNeatRacingLapFactAdvance(0, 1)).toEqual({
      lapBaseline: 1,
      lapsSinceLastCrossing: 1,
      crossed: true,
    });
  });

  it('stays silent at an unchanged lap count after adopting the baseline', () => {
    expect(resolveNeatRacingLapFactAdvance(1, 1).crossed).toBe(false);
  });

  it('aggregates a multi-lap diff into one crossing', () => {
    expect(resolveNeatRacingLapFactAdvance(1, 3)).toEqual({
      lapBaseline: 3,
      lapsSinceLastCrossing: 2,
      crossed: true,
    });
  });

  it('consumes driven frames without inventing a completed lap', () => {
    bootLiveLoop();
    mockedAdvanceFrameSteps.mockReturnValue(1);
    pressKeys(['ArrowUp', 'ArrowRight']);
    driveFrames(DRIVEN_FRAME_COUNT);
    releaseKeys(['ArrowUp', 'ArrowRight']);

    expect(lastHudUpdate().raceState.lapsCompleted).toBe(0);
  });
});

/* ------------------------------------------------------------------------- *
 * (d) The AI-mode goal lifecycle.
 * ------------------------------------------------------------------------- */

describe('the AI-mode goal lifecycle (LD-10: the machine keeps racing)', () => {
  it('boots the scripted-baseline probe once on a switch into ai', () => {
    bootLiveLoop();

    switchRaceMode('ai');

    expect(mockedProbeEpisode).toHaveBeenCalledTimes(1);
  });

  it('reuses the one probe across AI-lane restarts', () => {
    bootLiveLoop();
    switchRaceMode('ai');

    switchRaceMode('manual');
    switchRaceMode('ai');

    expect(mockedProbeEpisode).toHaveBeenCalledTimes(1);
  });

  it('resolves the ai goal session on the start level below the lap floor', () => {
    const session = resolveNeatRacingAiModeGoalSessionForMode('ai');
    if (session === null) {
      throw new Error('an ai switch must boot a goal session');
    }

    expect(describeNeatRacingAiModeGoalPhase(session)).toEqual({
      kind: 'lapFloor',
      remainingLaps: NEATRACING_AI_MIN_SESSION_LAPS,
    });
  });

  it('parks the lifecycle for a manual switch (no goal session)', () => {
    expect(resolveNeatRacingAiModeGoalSessionForMode('manual')).toBeNull();
  });

  it('returns no rebase for a non-forward crossing', () => {
    const outcome = recordNeatRacingAiModeGoalCrossing(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      buildForwardCrossing({ forwardCrossing: false }),
    );

    expect(resolveNeatRacingAiModeRebaseRung(outcome).rebaseRung).toBeNull();
  });

  it('returns no rebase for a crossing without a lap', () => {
    const outcome = recordNeatRacingAiModeGoalCrossing(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      buildForwardCrossing({ lapsSinceLastCrossing: 0.5 }),
    );

    expect(resolveNeatRacingAiModeRebaseRung(outcome).rebaseRung).toBeNull();
  });

  it('returns no rebase for a non-promoting counted lap', () => {
    const outcome = recordNeatRacingAiModeGoalCrossing(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      buildForwardCrossing({ nodeCount: 0, lapsSinceLastCrossing: 1 }),
    );

    expect(resolveNeatRacingAiModeRebaseRung(outcome).rebaseRung).toBeNull();
  });

  it('resolves a promoted crossing onto the promoted rung', () => {
    const promoted = recordCrossingChain(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      NEATRACING_AI_MIN_SESSION_LAPS,
    );

    expect(resolveNeatRacingAiModeRebaseRung(promoted).rebaseRung).toBe(
      promoted.session.level,
    );
  });

  it('adopts a fresh goal session on the promoted rung', () => {
    const promoted = recordCrossingChain(
      createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL),
      NEATRACING_AI_MIN_SESSION_LAPS,
    );

    expect(promoted.session).toEqual({
      level: NEATRACING_AI_MODE_START_LEVEL + 1,
      sessionLaps: 0,
    });
  });
});

/* ------------------------------------------------------------------------- *
 * (e) The restart carrier in BOTH switch directions.
 * ------------------------------------------------------------------------- */

describe('the restart carrier in BOTH switch directions (AC-033)', () => {
  it('stops and discards the session when the selector switches to ai', () => {
    bootLiveLoop();

    switchRaceMode('ai');

    expect(isStoppedBoot(capturedBootAt(0))).toBe(true);
  });

  it('boots a fresh initialized round on the carried level after a switch', () => {
    bootLiveLoop();

    switchRaceMode('ai');

    const freshBoot = capturedBootAt(1);
    expect(firstInitPayload(freshBoot).agentCount).toBe(
      resolveNeatRacingLevelCarCount(1),
    );
  });

  it('stops the previous session when the selector switches back to manual', () => {
    bootLiveLoop();
    switchRaceMode('ai');

    switchRaceMode('manual');

    expect(isStoppedBoot(capturedBootAt(1))).toBe(true);
  });

  it('restarts the HUD read model from the fresh round', () => {
    bootLiveLoop();
    mockedAdvanceFrameSteps.mockReturnValue(1);
    pressKeys(['ArrowUp']);
    driveFrames(DRIVEN_FRAME_COUNT);
    releaseKeys(['ArrowUp']);
    switchRaceMode('ai');
    switchRaceMode('manual');
    mockedAdvanceFrameSteps.mockReturnValue(0);
    driveFrames(1, DRIVEN_FRAME_COUNT * FRAME_STEP_MS + FRAME_STEP_MS);

    expect(lastHudUpdate().speedUnitsPerS).toBe(0);
  });

  it('leaves the fresh round idle until the next frame arrives', () => {
    bootLiveLoop();

    switchRaceMode('ai');

    expect(lastCapturedBoot().capturedRequests).toHaveLength(1);
  });
});

/* ------------------------------------------------------------------------- *
 * (f) The most-forward network panel.
 * ------------------------------------------------------------------------- */

describe('the most-forward network panel (mode-gated, rivals only)', () => {
  it('mounts the panel canvas inside the reserved HUD output container', () => {
    bootLiveLoop();

    switchRaceMode('ai');

    const panel = document.querySelector<HTMLCanvasElement>(
      `#${HUD_OUTPUT_ID} #${NETWORK_PANEL_CANVAS_ID}`,
    );
    expect(panel).toBeInstanceOf(HTMLCanvasElement);
  });

  it('sizes the mounted panel through the adapter resize authority', () => {
    bootLiveLoop();

    switchRaceMode('ai');

    expect(document.getElementById(NETWORK_PANEL_CANVAS_ID)).toBeInstanceOf(
      HTMLCanvasElement,
    );
    expect(
      (document.getElementById(NETWORK_PANEL_CANVAS_ID) as HTMLCanvasElement)
        .width,
    ).toBeGreaterThan(0);
  });

  it('drives level-1 frames on the idle panel path without throwing', () => {
    bootLiveLoop();
    switchRaceMode('ai');

    expect(() => driveFrames(4)).not.toThrow();
  });

  it('unmounts the panel canvas when the selector switches back to manual', () => {
    bootLiveLoop();
    switchRaceMode('ai');

    switchRaceMode('manual');

    expect(document.getElementById(NETWORK_PANEL_CANVAS_ID)).toBeNull();
  });

  it('wins the panel with the finite candidate over a non-finite runner', () => {
    const candidates: NeatRacingNetworkPanelCandidate[] = [
      { agentId: 1, progress: Number.NaN, network: undefined },
      { agentId: 2, progress: 30, network: undefined },
    ];

    expect(resolveMostForwardNeatRacingAgent(candidates)?.agentId).toBe(2);
  });

  it('wins exact progress ties on the lowest agent id', () => {
    const candidates: NeatRacingNetworkPanelCandidate[] = [
      { agentId: 2, progress: 10, network: undefined },
      { agentId: 1, progress: 10, network: undefined },
    ];

    expect(resolveMostForwardNeatRacingAgent(candidates)?.agentId).toBe(1);
  });

  it('elected no winner when every candidate progress is non-finite', () => {
    const candidates: NeatRacingNetworkPanelCandidate[] = [
      { agentId: 1, progress: Number.NaN, network: undefined },
    ];

    expect(
      resolveMostForwardNeatRacingAgent(candidates, undefined),
    ).toBeUndefined();
  });

  it('accumulates a monotone along-track total in meters', () => {
    const index = buildNeatRacingTrackChordIndex(buildSyntheticSegments());
    const start = resolveNeatRacingCarAlongProgressM(index, 1, 0, null);
    const next = resolveNeatRacingCarAlongProgressM(index, 3, 0, start);

    expect(next.totalM).toBe(3);
  });

  it('backward half-loop wrap DECREMENTS the total', () => {
    // The 1 -> 38 jump RAISES the along position past halfLoopM (delta
    // +37 > 20), a BACKWARD wrap: the total decrements by the wrapped
    // displacement (1 + 37 - 40 = -2; L = 40 on this square loop).
    const index = buildNeatRacingTrackChordIndex(buildSyntheticSegments());
    const start = resolveNeatRacingCarAlongProgressM(index, 1, 0, null);
    const wrapped = resolveNeatRacingCarAlongProgressM(index, 0, 2, start);

    expect(wrapped.totalM).toBe(-2);
  });

  it('forward half-loop wrap grows the total by the loop tail + head', () => {
    // The 31 -> 1 jump DROPS the along position past -halfLoopM (delta
    // -30 < -20), a FORWARD wrap: the total grows by the loop tail + head
    // (31 - 30 + 40 = 41).
    const index = buildNeatRacingTrackChordIndex(buildSyntheticSegments());
    const beforeLine = resolveNeatRacingCarAlongProgressM(index, 0, 9, null);
    const across = resolveNeatRacingCarAlongProgressM(index, 1, 0, beforeLine);

    expect(across.totalM).toBe(41);
  });

  it('decrements the total on a plain backward move without a wrap', () => {
    const index = buildNeatRacingTrackChordIndex(buildSyntheticSegments());
    const start = resolveNeatRacingCarAlongProgressM(index, 3, 0, null);
    const back = resolveNeatRacingCarAlongProgressM(index, 1, 0, start);

    expect(back.totalM).toBe(1);
  });
});

/* ------------------------------------------------------------------------- *
 * (g) The stop handle teardown.
 * ------------------------------------------------------------------------- */

describe('the stop handle teardown (AC-033)', () => {
  it('sends the terminal stop request to the live session', () => {
    bootLiveLoop();

    liveStopHandle?.();

    expect(isStoppedBoot(lastCapturedBoot())).toBe(true);
  });

  it('leaves the stopped session terminal with no further requests', () => {
    bootLiveLoop();

    liveStopHandle?.();

    expect(lastCapturedBoot().capturedRequests.at(-1)?.type).toBe('stop');
  });

  it('removes the panel canvas with its resize authority on stop', () => {
    bootLiveLoop();
    switchRaceMode('ai');

    liveStopHandle?.();

    expect(document.getElementById(NETWORK_PANEL_CANVAS_ID)).toBeNull();
  });

  it('stops a degraded fallback boot without throwing', () => {
    mockedInitRenderingContext.mockReturnValue({
      ok: false,
      gl: null,
      message: NEATRACING_FALLBACK_MESSAGE,
      fallbackElement: document.createElement('div'),
      isContextLost: (): boolean => false,
    });
    const stop = bootLiveLoop();

    expect(() => stop()).not.toThrow();
  });
});

/* ------------------------------------------------------------------------- *
 * The panel seam's jest-mocked 2d context (the adapter suite's idiom).
 * ------------------------------------------------------------------------- */

/**
 * Builds the jest-mocked 2d context the boundary pin mounts the network
 * panel with.
 *
 * WHY: the rival progress meters are host-private state only the panel
 * populates (frame step 7) and only the drive consumes (frame step 5), so
 * the stale-rival witness needs the panel's honest populate path to RUN —
 * the jsdom prototype stub returns null (the harness's default honest
 * UNDRAWABLE path) and the panel would stay dead. This stub is the SAME
 * established idiom the visualizer adapter suite pins its draw path with
 * (a complete jest-mocked CanvasRenderingContext2D over a jsdom canvas),
 * so the panel resolves and draws against the REAL adapter code while
 * nothing touches a live GPU surface.
 *
 * @returns The mocked 2d context (with its jsdom canvas attached).
 */
function buildNetworkPanelStubContext(): CanvasRenderingContext2D {
  const canvas = document.createElement('canvas');
  canvas.width = 900;
  canvas.height = 240;
  return {
    canvas,
    clearRect: jest.fn(),
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    bezierCurveTo: jest.fn(),
    quadraticCurveTo: jest.fn(),
    ellipse: jest.fn(),
    stroke: jest.fn(),
    rect: jest.fn(),
    fill: jest.fn(),
    fillRect: jest.fn(),
    arc: jest.fn(),
    fillText: jest.fn(),
    strokeText: jest.fn(),
    measureText: jest.fn().mockReturnValue({ width: 0 }),
    drawImage: jest.fn(),
    save: jest.fn(),
    restore: jest.fn(),
    translate: jest.fn(),
    scale: jest.fn(),
    rotate: jest.fn(),
    closePath: jest.fn(),
    clip: jest.fn(),
    setLineDash: jest.fn(),
    getLineDash: jest.fn().mockReturnValue([]),
    setTransform: jest.fn(),
    strokeRect: jest.fn(),
    strokeStyle: '',
    lineWidth: 1,
    fillStyle: '',
    font: '',
    textAlign: 'center',
    textBaseline: 'middle',
  } as unknown as CanvasRenderingContext2D;
}

/* ------------------------------------------------------------------------- *
 * (h) The live AI-lane adaptation session (the LIVE carrier node count).
 * ------------------------------------------------------------------------- */

describe('the live AI-lane adaptation session', () => {
  it('boots the manual round with a seam and the AI round with it', () => {
    bootLiveLoop();

    // The MANUAL round carries the seam TOO: the amended spec pins
    // "a manual round ALWAYS carries the seam" (the AC-029 manual-lane
    // replacement per LD-12), so the human races against the SAME
    // evolving agents in BOTH race modes.
    expect(
      typeof firstInitPayload(lastCapturedBoot()).controller?.computeControl,
    ).toBe('function');

    // LD-12: the worker race mode stays 'manual', so car 0 remains the
    // human car.
    expect(firstInitPayload(lastCapturedBoot()).raceMode).toBe('manual');

    // The switch into 'ai' REBOOTS the round, so the seam rides the
    // FRESH round's init payload (the LAST captured boot).
    switchRaceMode('ai');

    expect(
      typeof firstInitPayload(lastCapturedBoot()).controller?.computeControl,
    ).toBe('function');
  });

  it('feeds the goal machine the carrier LIVE node count', async () => {
    const promotionSpy = jest.spyOn(
      promotionRules,
      'evaluateNeatRacingPromotion',
    );
    try {
      bootLiveLoop();
      switchRaceMode('ai');

      // Frame one: the AI lane's adaptation tick runs (the harness
      // releases no simulation ticks, so the cadence gate passes on the
      // tick counter zero) and the session publishes its live node
      // counts when the tick resolves.
      driveFrames(1);
      await flushPendingPromises();

      // The doctored lap counter publishes the forward-lap crossing the
      // real lap authority never reached (the doctor returns a FRESH
      // payload, so the worker's own state is never mutated).
      framesReplyDoctor = (payload) => ({
        ...payload,
        rules: { ...payload.rules, lapsCompleted: 1 },
      });
      driveFrames(1);

      expect(promotionSpy).toHaveBeenCalled();
      const observedNodeCount =
        promotionSpy.mock.calls.at(-1)?.[0]?.nodeCount ?? Number.NaN;
      expect(Number.isFinite(observedNodeCount)).toBe(true);
      // The feed is LIVE: the carrier's whole network (82 inputs + 2
      // outputs at rung 1), never a fabricated zero.
      expect(observedNodeCount).toBeGreaterThanOrEqual(CARRIER_BASE_NODE_COUNT);
    } finally {
      // The spy is restored here, so the next test's
      // `jest.resetAllMocks()` never inherits a broken predicate spy.
      promotionSpy.mockRestore();
    }
  });

  it('drives the adaptation session at most once per simulation tick', async () => {
    bootLiveLoop();
    switchRaceMode('ai');

    // The harness releases NO simulation tick per drawn frame (the
    // frame-step double returns zero), so every driven frame observes
    // the SAME simulation tick (zero). The promise flush BETWEEN the
    // frames releases the in-flight flag, so the tick ADVANCE gate is
    // the ONLY thing that keeps the repeated drive out.
    for (let frame = 0; frame < ADAPTATION_REPEAT_FRAME_COUNT; frame += 1) {
      driveFrames(1);
      await flushPendingPromises();
    }

    // ONE created session per MODE BOOT: the initial MANUAL boot plus
    // the AI switch = TWO created sessions (the manual round now boots
    // its own adaptation session).
    expect(mockCapturedAdaptationSessions).toHaveLength(2);
    // The drive is NOT mode-gated (the seam rides BOTH modes — AC-029),
    // but this frame plan never drives a frame before the switch, so the
    // manual round's session stays un-driven and only the AI session
    // records a tick.
    expect(mockCapturedAdaptationDriveTicks).toEqual([0]);
  });

  it('resets the rival meters at the round boundary and drives the MANUAL session (R1 + R2, AC-031)', async () => {
    // The drawable panel is REQUIRED for the stale-rival witness: the
    // rival meters are host-private state the panel populates (frame step
    // 7) and the drive consumes (frame step 5 — which runs BEFORE the
    // panel in the same frame). The jest-mocked 2d stub is the adapter
    // seam's OWN established test idiom; until this test overrides it,
    // the harness keeps the honest undrawable prototype stub.
    jest
      .spyOn(HTMLCanvasElement.prototype, 'getContext')
      .mockImplementation(() => buildNetworkPanelStubContext());
    // Every driven frame releases ONE cadence slot (4 simulation steps),
    // so each frame is a deterministic adaptation-tick candidate.
    mockedAdvanceFrameSteps.mockReturnValue(4);
    bootLiveLoop();
    switchRaceMode('ai');

    // The 2-car doctor: the FRESH frames reply carries the real car 0
    // plus ONE rival cloned from it and displaced ahead along the demo
    // track — relative to car 0 on EVERY frame, so the rival's meter
    // trails car 0's along by the fixed gap (the worker's own fleet never
    // grows and its state is never mutated).
    const chordIndex = buildNeatRacingTrackChordIndex(
      firstInitPayload(capturedBootAt(0)).trackSpec.segments,
    );
    const rivalGapM =
      (RIVAL_AHEAD_CHORD_COUNT * chordIndex.loopLengthM) /
      chordIndex.chords.length;
    framesReplyDoctor = (payload) => {
      const car = payload.cars[0] ?? null;
      if (car === null) {
        return payload;
      }
      const carAlongM = resolveNeatRacingCarAlongProgressM(
        chordIndex,
        car.x,
        car.y,
        null,
      ).totalM;
      const rivalAlongM = (carAlongM + rivalGapM) % chordIndex.loopLengthM;
      const rivalChord = chordIndex.chords.findLast(
        (chord) => chord.startOffsetM <= rivalAlongM,
      );
      if (rivalChord === undefined) {
        return payload;
      }
      const rival = {
        ...car,
        x: (rivalChord.startX + rivalChord.endX) / 2,
        y: (rivalChord.startY + rivalChord.endY) / 2,
      };
      return { ...payload, cars: [car, rival] };
    };

    // AI frame one: the drive reads NULL meters (the round just booted)
    // and the panel populates the rival meter afterwards.
    driveFrames(1);
    await flushPendingPromises();
    // AI frame two: the drive consumes the POPULATED rival meter — the
    // stale-meter witness arms (the position term becomes 2).
    driveFrames(1);
    await flushPendingPromises();

    // THE ROUND BOUNDARY: the manual switch stops the outgoing round and
    // reboots the worker round — the R1 rival-meter reset rides the round
    // restart, so the fresh round reads NO stale rival ahead.
    switchRaceMode('manual');

    // The boundary-side frame: the MANUAL round's session drives (R2 —
    // the seam rides both race modes) and the drive's position term
    // returns to its fresh-round value.
    driveFrames(1);
    await flushPendingPromises();

    // THREE drives (two AI + the boundary-side MANUAL one at a FRESH tick
    // lineage) pin R2, and the 1 -> 2 -> 1 position chain pins R1: a
    // missing boundary reset would keep the rival's pre-boundary meter
    // alive and the boundary-side drive would read 2 instead of 1.
    expect(mockCapturedAdaptationDriveTicks).toEqual([4, 8, 4]);
    expect(mockCapturedAdaptationTickCar0Positions).toEqual([1, 2, 1]);
  });
});

/* ------------------------------------------------------------------------- *
 * (i) The AI lane's carrier-backed human drive (the AI drive of car 0).
 * ------------------------------------------------------------------------- */

describe('the AI lane\u2019s carrier-backed human drive (the AI drive of car 0)', () => {
  function buildLiveBootSessionTuple(): NeatRacingReproducibilityTuple {
    const payload = firstInitPayload(capturedBootAt(0));
    return {
      seed: payload.seed,
      agentCount: payload.agentCount,
      packSchemaVersion: payload.packSchemaVersion,
      opponentSnapshot: payload.opponentSnapshot,
      trackId: payload.trackId,
      featureFlags: payload.featureFlags,
    };
  }

  it('boots the goal lifecycle at the carried boot level', () => {
    const session = resolveNeatRacingAiModeGoalSessionForMode('ai', 2);
    if (session === null) {
      throw new Error('an ai switch must boot a goal session');
    }

    expect(session).toEqual({
      level: 2,
      sessionLaps: 0,
    });
  });

  it('rejects a carried boot level below the observation ladder', () => {
    // The runtime guard lives in the goal-session factory (the TS level
    // type keeps the literal out statically, so the pin casts): the
    // RESOLVER must surface the RANGE ERROR, not swallow it.
    const carried = 0 as NeatRacingObservationLevel;
    expect(() =>
      resolveNeatRacingAiModeGoalSessionForMode('ai', carried),
    ).toThrow(RangeError);
  });

  it('drives car 0 through the LIVE carrier-backed seam consult', () => {
    bootLiveLoop();
    switchRaceMode('ai');

    const session = mockCapturedAdaptationSessions.at(-1);
    if (session === undefined) {
      throw new Error('an ai switch must boot an adaptation session');
    }
    const control = session.controller.computeControl(
      [createInitialPhysicsCarState(0, { x: 2, y: 0, heading: 0 })],
      0,
    );
    if (control === null) {
      throw new Error('the human slot must consult the carrier-backed seam');
    }

    expect(control.throttle).toBeGreaterThanOrEqual(0);
    expect(control.throttle).toBeLessThanOrEqual(1);
    // The mapping law: the brake axis is fixed at zero and no turn-rate
    // channel is ever emitted.
    expect(control.brake).toBe(NEATRACING_FLEET_CONTROL_BRAKE);
  });

  it('keeps the human slot null with the carrier-drive option absent', () => {
    bootLiveLoop();

    const session = createNeatRacingAdaptationSession({
      tuple: buildLiveBootSessionTuple(),
      trackSpec: firstInitPayload(capturedBootAt(0)).trackSpec,
      carCount: 2,
    });
    const car: PhysicsCarState = createInitialPhysicsCarState(0, {
      x: 2,
      y: 0,
      heading: 0,
    });

    expect(session.controller.computeControl([car], 0)).toBeNull();
  });

  it('backs the human slot with the opted carrier slot when present', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    const session = createNeatRacingAdaptationSession({
      tuple: buildLiveBootSessionTuple(),
      trackSpec: payload.trackSpec,
      carCount: 2,
      humanCarrierDriveSlot: 1,
    });
    const car: PhysicsCarState = createInitialPhysicsCarState(0, {
      x: 2,
      y: 0,
      heading: 0,
    });
    const control = session.controller.computeControl([car], 0);

    expect(control).not.toBeNull();
  });

  it('rejects a carrier-drive slot at the human slot', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    expect(() =>
      createNeatRacingAdaptationSession({
        tuple: buildLiveBootSessionTuple(),
        trackSpec: payload.trackSpec,
        carCount: 2,
        humanCarrierDriveSlot: 0,
      }),
    ).toThrow(RangeError);
  });

  it('rejects a carrier-drive slot at the fleet edge', () => {
    bootLiveLoop();

    const payload = firstInitPayload(capturedBootAt(0));
    expect(() =>
      createNeatRacingAdaptationSession({
        tuple: buildLiveBootSessionTuple(),
        trackSpec: payload.trackSpec,
        carCount: 2,
        humanCarrierDriveSlot: 2,
      }),
    ).toThrow(RangeError);
  });
});
