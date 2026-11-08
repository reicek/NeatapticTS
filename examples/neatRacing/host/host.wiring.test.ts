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
  runNeatRacingScriptedBaselineProbeEpisode: jest.fn(),
}));

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
import {
  runNeatRacingScriptedBaselineProbeEpisode,
  resolveNeatRacingLevelCarCount,
  NEATRACING_AI_MIN_SESSION_LAPS,
  NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS,
  type NeatRacingScriptedBaselineProbeResult,
  type NeatRacingDrivabilityMeasurement,
  type NeatRacingPromotionRoundSnapshot,
} from '../ai/promotion.rules';
import { resolveOscillationPromotionNodeThreshold } from '../ai/oscillation-penalty.utils';
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
import type {
  NeatRacingSimWorkerInitPayload,
  NeatRacingSimWorkerRequest,
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

const {
  createSimWorkerSession: createRealSimWorkerSession,
} = jest.requireActual<
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
      return realSession.handleRequest(request);
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
function driveFrames(count: number, startTimestampMs = FIRST_FRAME_TIMESTAMP_MS): void {
  for (let frame = 0; frame < count; frame += 1) {
    const callback = scheduledRafCallbacks.at(-1);
    if (callback === undefined) {
      throw new Error('the loop never scheduled a frame callback');
    }
    callback(startTimestampMs + frame * FRAME_STEP_MS);
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
const mockedProbeEpisode = jest.mocked(runNeatRacingScriptedBaselineProbeEpisode);
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

    const seam = (window as unknown as { neatRacingDriftState?: { speedMps: number } })
      .neatRacingDriftState;
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

    expect(document.getElementById(NETWORK_PANEL_CANVAS_ID))
      .toBeInstanceOf(HTMLCanvasElement);
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
    const across = resolveNeatRacingCarAlongProgressM(
      index,
      1,
      0,
      beforeLine,
    );

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
