/**
 * @jest-environment jsdom
 */

/**
 * RED contract tests for the Flappy-style 3-2-1-GO countdown (plan slice
 * P5-S27-1, RED lane RED-S27-3; AC-106; GAP-5-03).
 *
 * THE GAP (GAP-5-03): there is NO countdown anywhere in `examples/neatRacing`
 * today. The HUD (`host/hud.ts`) renders the speed / lap / time / traction
 * readouts ONLY, and the host hands the sampled keyboard input straight to
 * the simulation worker, so the throttle is NEVER held before the start.
 *
 * THE GREEN TARGET (slice P5-S27-4, AC-106): a NEW pure countdown resolver
 * `host/hud.countdown.ts`, an EDITED `host/hud.ts` (the DOM overlay that
 * renders 3, then 2, then 1, then GO) and an EDITED
 * `browser-entry/browser-entry.ts` (the HOLD-until-GO gate that ZEROES the
 * sampled throttle before GO).
 *
 * THE RUNTIME-ONLY RULE: this lane NEVER imports `host/hud.countdown.ts` -
 * the module does not exist yet, and importing it would fail by an IMPORT
 * ERROR instead of by an ASSERTION. Every contract below probes the EXISTING
 * modules ONLY:
 *
 * - the HUD surface (`host/hud.ts`) - the DOM overlay that renders the labels;
 * - the race-rules read model (`physics/race-rules.ts`) - the clock source
 *   (`raceState.elapsedS`);
 * - the host per-frame sample site (`browser-entry/browser-entry.ts`,
 *   `driveSimProtocolFrame`) - the throttle CONSUMER that hands ONE sample to
 *   EVERY catch-up tick of that frame.
 *
 * Every red below is therefore a NAMED RUNTIME failure.
 */

/**
 * The boot doubles of the PROVEN jsdom harness
 * (`browser-entry/browser-entry.test.ts`), re-based on this folder.
 *
 * Only the three seams the drive loop actually reaches are doubled:
 * the rendering context (the boot path acquires it), the renderer size
 * (the letterbox seam) and the simulation-worker session (the throttle
 * CONSUMER under test). The composer passes stay REAL: the composition
 * attempt is gated on `gl.createShader`, so the inert GL fixture simply
 * degrades to the inert composer and never touches them.
 */
jest.mock('../engine/context.service', () => ({
  ...jest.requireActual('../engine/context.service'),
  initRenderingContext: jest.fn(),
  runShaderWarmup: jest.fn(),
}));

jest.mock('../browser-entry/canvas-dimensions.utils', () => ({
  ...jest.requireActual('../browser-entry/canvas-dimensions.utils'),
  updateFullWindowRendererSize: jest.fn(),
}));

jest.mock('../workers/simulation-worker/simulation-worker', () => ({
  ...jest.requireActual('../workers/simulation-worker/simulation-worker'),
  createSimWorkerSession: jest.fn(),
}));

import { afterEach, beforeEach, describe, expect, it } from '@jest/globals';

import type { RenderingContextInitResult } from '../engine/context.service';
import {
  initRenderingContext,
  runShaderWarmup,
} from '../engine/context.service';
import type { CanvasDimensions } from '../browser-entry/canvas-dimensions.utils';
import { updateFullWindowRendererSize } from '../browser-entry/canvas-dimensions.utils';
import { createSimWorkerSession } from '../workers/simulation-worker/simulation-worker';
import type {
  NeatRacingSimWorkerRequest,
  NeatRacingSimWorkerResponse,
  NeatRacingSimWorkerSession,
} from '../workers/simulation-worker/simulation-worker';
import { createHud } from './hud';
import type { NeatRacingHud } from './hud';
import { createRaceRulesState } from '../physics/race-rules';
import { createInitialPhysicsCarState } from '../physics/physics.step.service';
import type {
  PhysicsCarState,
  PhysicsStepInput,
} from '../physics/physics.step.service';
// Side-effect import: evaluating the host entry registers
// `window.neatRacingStart` on the jsdom host.
import '../browser-entry/browser-entry';
import type {
  NeatRacingStart,
  NeatRacingStop,
} from '../browser-entry/browser-entry';

/** The reserved host-shell container the demo canvas mounts into. */
const HOST_OUTPUT_ID = 'neatracing-output';

/**
 * The reserved host-shell container the HUD + mode selector mount into
 * (spec section 10.2; the boot glue resolves-or-throws against it).
 */
const RESERVED_HUD_OUTPUT_ID = 'neatracing-hud-output';

/**
 * The countdown tokens the HUD surface publishes, in the published order
 * (AC-106: 3, then 2, then 1, then GO).
 */
const EXPECTED_COUNTDOWN_TOKENS: readonly string[] = ['3', '2', '1', 'GO'];

/** The countdown sweep step (s) - a fine grid over the 3-2-1-GO window. */
const COUNTDOWN_SWEEP_STEP_S = 0.25;

/** The countdown sweep horizon (s) - past every countdown phase. */
const COUNTDOWN_SWEEP_HORIZON_S = 20;

/** The frozen wall-clock reading (ms) - a Date.now()-driven clock cannot move. */
const FROZEN_NOW_MS = 1_700_000_000_000;

/** The FIRST probe's race clock (s) - inside the '3' phase. */
const FIRST_PROBE_ELAPSED_S = 0.1;

/** The SECOND probe's race clock (s) - inside the '2' phase. */
const SECOND_PROBE_ELAPSED_S = 1.1;

/** The PRE-GO race clock (s) the fake worker session publishes. */
const PRE_GO_ELAPSED_S = 0;

/** The POST-GO race clock (s) the fake worker session publishes. */
const POST_GO_ELAPSED_S = 5;

/** The frame timestamp step (ms) between the driven frames (60 Hz). */
const DRIVEN_FRAME_STEP_MS = 1000 / 60;

/** The first driven frame's timestamp (ms). */
const FIRST_FRAME_TIMESTAMP_MS = 0;

/** The rAF stub installed by the drive-loop harness (restored after each test). */
const ORIGINAL_REQUEST_ANIMATION_FRAME = window.requestAnimationFrame;

/** The cancel-rAF stub installed by the drive-loop harness (restored after each test). */
const ORIGINAL_CANCEL_ANIMATION_FRAME = window.cancelAnimationFrame;

/**
 * Normalises one element's text into a countdown token.
 *
 * A trailing '!' is stripped ('3!' -> '3') so the neon recipe's bang
 * spelling still reads as the token, and the comparison is
 * case-insensitive ('go!' -> 'GO'). A text that is NOT one of the four
 * countdown tokens yields null.
 *
 * @param text - The element text to normalise.
 * @returns The countdown token, or null when the text is not one.
 */
function toCountdownToken(text: string): string | null {
  const trimmed = text.trim();
  const withoutBang = trimmed.endsWith('!')
    ? trimmed.slice(0, trimmed.length - 1).trim()
    : trimmed;
  const upper = withoutBang.toUpperCase();
  return EXPECTED_COUNTDOWN_TOKENS.includes(upper) ? upper : null;
}

/**
 * Collects the DISTINCT countdown tokens the HUD surface publishes.
 *
 * The surface is walked in DOCUMENT ORDER (the container first, then its
 * descendants), so a NESTED wrapper that repeats the token
 * ('<div>3<span>3</span></div>') contributes the token ONCE - the
 * de-duplication keeps the FIRST-SEEN order.
 *
 * @param container - The HUD surface to walk.
 * @returns The distinct countdown tokens, in first-seen order.
 */
function collectCountdownTokens(container: HTMLElement): string[] {
  const tokens: string[] = [];
  const candidates: Element[] = [
    container,
    ...Array.from(container.querySelectorAll('*')),
  ];
  for (const candidate of candidates) {
    const token = toCountdownToken(candidate.textContent ?? '');
    if (token !== null && !tokens.includes(token)) {
      tokens.push(token);
    }
  }
  return tokens;
}

/**
 * Sweeps the race clock over the countdown window and returns the DISTINCT
 * countdown tokens the HUD published, in FIRST-SEEN order.
 *
 * The clock is swept on a FINE grid, because the HUD REPAINTS its labels in
 * place: a token is observable in the DOM only while the clock sits inside
 * that token's phase, so the sweep must visit every phase.
 *
 * @param hud - The HUD surface under test.
 * @returns The distinct countdown tokens, in first-seen order.
 */
function sweepCountdownTokens(hud: NeatRacingHud): string[] {
  const observed: string[] = [];
  for (
    let elapsedS = 0;
    elapsedS <= COUNTDOWN_SWEEP_HORIZON_S;
    elapsedS += COUNTDOWN_SWEEP_STEP_S
  ) {
    hud.update({
      speedUnitsPerS: 0,
      raceState: { ...createRaceRulesState(), elapsedS },
    });
    for (const token of collectCountdownTokens(hud.container)) {
      if (!observed.includes(token)) {
        observed.push(token);
      }
    }
  }
  return observed;
}

/**
 * Creates the inert WebGL2 context double the boot path acquires.
 *
 * Inert on purpose: the double only has to survive being handed to the
 * (mocked) warmup, and the composer composition attempt is gated on
 * `gl.createShader`, which this fixture deliberately does NOT provide.
 */
function createMockWebGl2(): WebGL2RenderingContext {
  const gl = {
    clear: (): void => undefined,
    clearColor: (): void => undefined,
    drawArrays: (): void => undefined,
    viewport: (): void => undefined,
  };
  return gl as unknown as WebGL2RenderingContext;
}

/** The live-branch init double: `ok: true` with a stable inert gl. */
function createLiveInitResultFixture(): RenderingContextInitResult {
  return {
    ok: true,
    gl: createMockWebGl2(),
    isContextLost: (): boolean => false,
  };
}

/** Standard 1280x720 backing-store size-double result. */
function createMockRendererSizeUpdate(): CanvasDimensions {
  return { width: 1280, height: 720 };
}

/**
 * The car-0 state the fake worker session publishes in every reply.
 *
 * The tick is deliberately NOT a multiple of the host's adaptation cadence
 * (4), so the fake reply never drives the adaptation engine.
 */
function createFakeCarState(): PhysicsCarState {
  return {
    ...createInitialPhysicsCarState(),
    tick: 1,
  };
}

/**
 * Returns the boot function installed by the host entry.
 *
 * @throws When the host entry has not registered `window.neatRacingStart`.
 */
function getBootStart(): NeatRacingStart {
  const start = window.neatRacingStart;
  if (typeof start !== 'function') {
    throw new Error('neatRacingStart was not registered on window');
  }
  return start;
}

describe('the Neat Racing 3-2-1-GO countdown (RED-S27-3, AC-106)', () => {
  describe('the HUD countdown surface (host/hud.ts)', () => {
    /** The REAL HUD surface under test (mounted per test). */
    let hud: NeatRacingHud;
    /** Restores the frozen wall clock after the clock-source probe. */
    let restoreDateNow: (() => void) | null;

    beforeEach(() => {
      restoreDateNow = null;
      // The HUD resolves-or-throws against its container, so the reserved
      // host container is mounted first.
      document.body.innerHTML = `<div id="${HOST_OUTPUT_ID}"></div>`;
      hud = createHud(HOST_OUTPUT_ID);
    });

    afterEach(() => {
      restoreDateNow?.();
      restoreDateNow = null;
    });

    it('publishes the countdown phase on the HUD surface', () => {
      // Act: sweep the race clock over the countdown window and collect the
      // DISTINCT countdown tokens the HUD published.
      const observed = sweepCountdownTokens(hud);

      // Assert
      expect(observed).toEqual(EXPECTED_COUNTDOWN_TOKENS);
    });

    it('reads the countdown clock from raceState.elapsedS and never from Date.now()', () => {
      // Arrange: the WALL CLOCK is frozen, so a Date.now()-driven countdown
      // could never advance between the two probes.
      const dateNowSpy = jest.spyOn(Date, 'now').mockReturnValue(FROZEN_NOW_MS);
      restoreDateNow = (): void => {
        dateNowSpy.mockRestore();
      };

      // Act: the SAME HUD, TWO DIFFERENT race-clock values.
      hud.update({
        speedUnitsPerS: 0,
        raceState: {
          ...createRaceRulesState(),
          elapsedS: FIRST_PROBE_ELAPSED_S,
        },
      });
      const firstTokens = collectCountdownTokens(hud.container);
      hud.update({
        speedUnitsPerS: 0,
        raceState: {
          ...createRaceRulesState(),
          elapsedS: SECOND_PROBE_ELAPSED_S,
        },
      });
      const secondTokens = collectCountdownTokens(hud.container);

      // Assert: the published tokens must follow the RACE clock, so the two
      // race-clock values must publish two DIFFERENT countdown phases.
      expect(firstTokens).not.toEqual(secondTokens);
    });
  });

  describe('the HOLD-until-GO throttle gate (browser-entry/browser-entry.ts)', () => {
    /** Every frame sample the host handed to the worker, in drive order. */
    let capturedFrameSamples: Array<PhysicsStepInput | undefined>;
    /** The race clock the fake worker session publishes in its next reply. */
    let publishedElapsedS: number;
    /** The rAF callbacks the host loop scheduled (driven MANUALLY). */
    let scheduledRafCallbacks: FrameRequestCallback[];
    /** The stop handle of the booted demo (torn down after the test). */
    let stopDemo: NeatRacingStop | null;

    beforeEach(() => {
      jest.resetAllMocks();

      // The live-branch doubles: the boot path sees a WebGL2 context and a
      // fixed 1280x720 letterbox result regardless of the jsdom host.
      jest
        .mocked(initRenderingContext)
        .mockReturnValue(createLiveInitResultFixture());
      jest.mocked(runShaderWarmup).mockReturnValue({
        warmedProgramCount: 0,
        offscreenDrawCount: 0,
      });
      jest
        .mocked(updateFullWindowRendererSize)
        .mockReturnValue(createMockRendererSizeUpdate());

      capturedFrameSamples = [];
      publishedElapsedS = PRE_GO_ELAPSED_S;

      /**
       * The fake simulation-worker session: it answers the boot's `init`
       * request with silence and every `frame` request with ONE `frames`
       * reply that carries car 0 and the test-controlled race clock. The
       * frame sample is CAPTURED, so the throttle the host published is
       * observable.
       */
      jest
        .mocked(createSimWorkerSession)
        .mockImplementation((): NeatRacingSimWorkerSession => ({
          handleRequest: (
            request: NeatRacingSimWorkerRequest,
          ): NeatRacingSimWorkerResponse[] => {
            if (request.type !== 'frame') {
              return [];
            }
            capturedFrameSamples.push(request.payload.sample);
            return [
              {
                type: 'frames',
                payload: {
                  fromTick: 0,
                  toTick: 0,
                  cars: [createFakeCarState()],
                  rules: {
                    ...createRaceRulesState(),
                    elapsedS: publishedElapsedS,
                  },
                },
              },
            ];
          },
          snapshot: {
            phase: 'running',
            tick: 0,
            cars: [],
            rules: createRaceRulesState(),
            lapLinePrevSide: null,
          },
        }));

      // Replace the jsdom RAF implementation with a capturing stub so the
      // host loop's frames are driven manually instead of by timers.
      scheduledRafCallbacks = [];
      window.requestAnimationFrame = (
        callback: FrameRequestCallback,
      ): number => {
        scheduledRafCallbacks.push(callback);
        return scheduledRafCallbacks.length;
      };

      stopDemo = null;

      // The host shell: the canvas container plus the reserved HUD container
      // the boot glue mounts the HUD + mode selector into.
      document.body.innerHTML = `<div id="${HOST_OUTPUT_ID}"><div id="${RESERVED_HUD_OUTPUT_ID}"></div></div>`;
    });

    afterEach(() => {
      stopDemo?.();
      stopDemo = null;
      window.requestAnimationFrame = ORIGINAL_REQUEST_ANIMATION_FRAME;
      window.cancelAnimationFrame = ORIGINAL_CANCEL_ANIMATION_FRAME;
    });

    /**
     * Drives `frameCount` host frames by invoking the captured rAF callbacks
     * in order (the host re-schedules itself after every frame).
     *
     * @param frameCount - The number of frames to drive.
     */
    function driveFrames(frameCount: number): void {
      for (let frame = 0; frame < frameCount; frame += 1) {
        const callback = scheduledRafCallbacks.at(frame);
        if (callback === undefined) {
          throw new Error(
            `the host scheduled no rAF callback for frame ${frame}`,
          );
        }
        callback(FIRST_FRAME_TIMESTAMP_MS + frame * DRIVEN_FRAME_STEP_MS);
      }
    }

    it('holds the sampled throttle at zero until the countdown reaches GO', () => {
      // Arrange: the demo boots against the fake worker session, and the UP
      // key is held down for the whole probe.
      stopDemo = getBootStart()(HOST_OUTPUT_ID);
      window.dispatchEvent(
        new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true }),
      );

      // Act 1: ONE frame while the worker publishes a PRE-GO race clock.
      publishedElapsedS = PRE_GO_ELAPSED_S;
      driveFrames(1);

      // Act 2: the worker publishes a POST-GO race clock. The host rebinds
      // the rules from the reply, so the first of the two frames carries the
      // POST-GO clock into the host's OWN rules copy and the second one
      // releases the throttle on it.
      publishedElapsedS = POST_GO_ELAPSED_S;
      driveFrames(2);

      const preGoThrottle = capturedFrameSamples.at(0)?.throttle;
      const postGoThrottle = capturedFrameSamples.at(-1)?.throttle;

      // Assert: the NON-VACUITY guard comes FIRST (the hold must NOT last
      // forever - a gate that never releases would satisfy the hold clause
      // alone), then the discriminating HOLD clause.
      expect(postGoThrottle).toBe(1);
      expect(preGoThrottle).toBe(0);
    });
  });
});
