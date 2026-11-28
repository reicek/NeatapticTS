/**
 * @jest-environment jsdom
 */

/**
 * RED contract tests for the TRACK-DEPARTURE ADVISORY hint (plan slice
 * `P5-S28-red`, the RED lane of the "track-departure fix" step; AC-107;
 * GAP-5-04).
 *
 * THE GAP (GAP-5-04): there is NO departure advisory anywhere in
 * `examples/neatRacing` today. The HUD (`host/hud.ts`) renders the speed /
 * lap / time / traction readouts and the countdown overlay ONLY, so a car
 * that has LEFT the drivable surface gets NO hint at all - the driver
 * only sees the car rolling away over the open grass, with no signal
 * that the track was left behind.
 *
 * THE GREEN TARGET (slice `P5-S28-law`): a HUD "return to the track" hint
 * derived from the ONE published FINE classification
 * (`physics/collision.utils.ts` -> `TrackSurfaceClassification.isDrivable`
 * - the road core, the curb band and the grass apron are DRIVABLE, the
 * open grass beyond the apron is NOT), painted INSIDE the HUD wrapper as
 * an overlay (the AC-106 overlay discipline: `pointer-events: none`,
 * repainted every frame, NOT a new member on the public `NeatRacingHud`
 * interface).
 *
 * THE RUNTIME-ONLY RULE: this lane NEVER imports the NEW
 * `host/hud.departure.ts` module - it does not exist yet, and importing
 * it would fail by an IMPORT ERROR instead of by an ASSERTION. Every
 * contract below probes the EXISTING modules ONLY:
 *
 * - the HUD surface (`host/hud.ts`) - the DOM overlay under test;
 * - the ONE published surface classifier (`physics/collision.utils.ts`) -
 *   the FINE band law that says whether a pose is still drivable;
 * - the demo-track catalog (`track/track.catalog.ts`) - the SERVED
 *   geometry the demo actually boots (read from the demo's OWN track
 *   selector, never hard-coded);
 * - the host boot seam (`browser-entry/browser-entry.ts`) - the END-TO-END
 *   path the probe drives (the demo boots, the fake worker replies, the
 *   HUD repaints from the reply).
 *
 * Every red below is therefore a NAMED RUNTIME failure.
 *
 * THE WORDING-AGNOSTIC PROBE: the plan does NOT specify the hint's
 * wording, so the probe NEVER hard-codes any text. It scopes itself to
 * the HUD subtree (the PARENT of the speed readout - the HUD mounts
 * first into the reserved HUD container, so the mode selector and the
 * track selector are its SIBLINGS and never pollute the probe), collects
 * the LEAF texts and treats every text that matches NONE of the KNOWN
 * readouts as a departure hint. The wording is the implementer's choice;
 * the EXISTENCE of the hint is the contract.
 *
 * THE TWO FIXTURE POSES are both derived from the SERVED geometry (never
 * from a hard-coded coordinate): the ON-SURFACE pose is the MIDPOINT of
 * the first chord of the served ribbon (heading along that chord, and
 * verified through the published classifier to be road-core), and the
 * DEPARTED pose is the FIRST PERPENDICULAR offset from that same
 * midpoint whose FINE classification is NOT drivable - the car that has
 * LEFT the drivable surface.
 */

/**
 * The boot doubles of the PROVEN jsdom harness
 * (`browser-entry/browser-entry.test.ts`), re-based on this folder.
 *
 * Only the three seams the drive loop actually reaches are doubled: the
 * rendering context (the boot path acquires it), the renderer size (the
 * letterbox seam) and the simulation-worker session (the frame reply
 * under test). The composer passes stay REAL: the composition attempt is
 * gated on `gl.createShader`, so the inert GL fixture simply degrades to
 * the inert composer and never touches them.
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
import { formatHudSpeed } from './hud';
import { createRaceRulesState } from '../physics/race-rules';
import { createInitialPhysicsCarState } from '../physics/physics.step.service';
import type { PhysicsCarState } from '../physics/physics.step.service';
import { createTrackSurfaceClassifier } from '../physics/collision.utils';
import type { TrackSurfaceClassification } from '../physics/collision.utils';
import {
  NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
  resolveTrackVariantSpec,
} from '../track/track.catalog';
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
 * The reserved host-shell container the HUD + mode selector + track
 * selector mount into (spec section 10.2; the boot glue resolves-or-
 * throws against it).
 */
const RESERVED_HUD_OUTPUT_ID = 'neatracing-hud-output';

/** DOM id of the booted demo's track selector `<select>`. */
const TRACK_SELECTOR_DOM_ID = 'neatracing-track-selector';

/** The step (m) of the perpendicular departure walk. */
const DEPARTURE_WALK_STEP_M = 0.25;

/** The horizon (m) of the perpendicular departure walk. */
const DEPARTURE_WALK_HORIZON_M = 40;

/** The number of steps the perpendicular departure walk takes. */
const DEPARTURE_WALK_STEPS = DEPARTURE_WALK_HORIZON_M / DEPARTURE_WALK_STEP_M;

/** The perpendicular walk signs (both sides of the chord). */
const DEPARTURE_WALK_SIGNS: readonly number[] = [1, -1];

/** The forward speed (m/s) the fake worker session publishes for car 0. */
const PUBLISHED_SPEED_UNITS_PER_S = 20;

/** The race clock (s) the fake worker session publishes (past the countdown). */
const PUBLISHED_ELAPSED_S = 5;

/** The number of host frames the probe drives. */
const DRIVEN_FRAME_COUNT = 30;

/** The frame timestamp step (ms) between the driven frames (60 Hz). */
const DRIVEN_FRAME_STEP_MS = 1000 / 60;

/** The first driven frame's timestamp (ms). */
const FIRST_FRAME_TIMESTAMP_MS = 0;

/** The rAF stub installed by the drive-loop harness (restored after each test). */
const ORIGINAL_REQUEST_ANIMATION_FRAME = window.requestAnimationFrame;

/** The cancel-rAF stub installed by the drive-loop harness (restored after each test). */
const ORIGINAL_CANCEL_ANIMATION_FRAME = window.cancelAnimationFrame;

/**
 * The speed readout pattern: the ONE published HUD text that identifies
 * the speed label, and therefore the HUD wrapper (its parent).
 */
const SPEED_READOUT_PATTERN = /^\d+ km\/h$/;

/** The KNOWN HUD readout patterns (NONE of them is a departure hint). */
const KNOWN_READOUT_PATTERNS: readonly RegExp[] = [
  SPEED_READOUT_PATTERN,
  /^Lap \d+\/\d+$/,
  /^\d+:\d{2}\.\d$/,
  /^Traction \d+%$/,
];

/**
 * The KNOWN HUD texts (NONE of them is a departure hint): the static
 * controls affordance and the countdown tokens (the empty token clears
 * the overlay).
 */
const KNOWN_READOUT_TEXTS: readonly string[] = [
  'WASD / ARROWS to drive',
  '3',
  '2',
  '1',
  'GO',
  '',
];

/** One car pose in world metres (the fixture's pose vocabulary). */
interface CarPose {
  /** World x position in metres. */
  readonly x: number;
  /** World y position in metres. */
  readonly y: number;
  /** Heading in radians. */
  readonly heading: number;
}

/** The served track spec (the catalog resolver's own return type). */
type ServedTrackSpec = ReturnType<typeof resolveTrackVariantSpec>;

/** The FINE surface classifier probe over the served ribbon. */
type ServedSurfaceClassifier = (car: PhysicsCarState) => TrackSurfaceClassification;

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

/**
 * Builds the car state the fake worker session publishes for car 0.
 *
 * The pose comes from the fixture (the served geometry), the speed is a
 * NON-ZERO forward speed (so the speed readout proves the HUD repainted
 * from THIS reply) and the tick is 1 (never a multiple of the host's
 * adaptation cadence, so the fake reply never drives the adaptation
 * engine).
 *
 * @param pose - The world pose the fake session publishes.
 * @returns The published car-0 state.
 */
function createPublishedCarState(pose: CarPose): PhysicsCarState {
  return {
    ...createInitialPhysicsCarState(0, {
      x: pose.x,
      y: pose.y,
      heading: pose.heading,
    }),
    speed: PUBLISHED_SPEED_UNITS_PER_S,
    tick: 1,
  };
}

/**
 * Reads the SERVED track id from the booted demo's own track selector.
 *
 * The demo boots on the catalog default, but the probe NEVER assumes it:
 * the id is read from the mounted `<select>` (the control the host owns)
 * and only falls back to the catalog default when the control carries no
 * selection at all.
 *
 * @returns The served catalog variant id.
 * @throws When the track selector is not mounted.
 */
function readServedTrackId(): string {
  const select = document.getElementById(TRACK_SELECTOR_DOM_ID);
  if (!(select instanceof HTMLSelectElement)) {
    throw new Error(
      `the track selector #${TRACK_SELECTOR_DOM_ID} is not mounted`,
    );
  }
  const servedTrackId = select.value.trim();
  return servedTrackId.length > 0
    ? servedTrackId
    : NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID;
}

/**
 * Builds the car state the classifier probe reads (the pose is all the
 * classifier looks at).
 *
 * @param pose - The world pose to classify.
 * @returns The car state carrying the pose.
 */
function createPoseCarState(pose: CarPose): PhysicsCarState {
  return createInitialPhysicsCarState(0, {
    x: pose.x,
    y: pose.y,
    heading: pose.heading,
  });
}

/**
 * Resolves the ON-SURFACE pose: the MIDPOINT of the first chord of the
 * served ribbon, heading along that chord.
 *
 * The pose is VERIFIED through the ONE published classifier: the
 * midpoint must classify as road-core (`isRoadOnly === true`), so the
 * fixture can never drift onto the apron by accident.
 *
 * @param spec - The served track spec.
 * @param classify - The FINE classifier over the served ribbon.
 * @returns The on-surface pose.
 * @throws When the served ribbon is degenerate or the midpoint is NOT
 *   road-core.
 */
function resolveOnSurfacePose(
  spec: ServedTrackSpec,
  classify: ServedSurfaceClassifier,
): CarPose {
  const segment = spec.segments[0];
  if (segment === undefined) {
    throw new Error('the served ribbon carries no chord');
  }
  const chordX = segment.endX - segment.startX;
  const chordY = segment.endY - segment.startY;
  const chordLengthM = Math.hypot(chordX, chordY);
  if (chordLengthM <= 0) {
    throw new Error('the first chord of the served ribbon is degenerate');
  }
  const pose: CarPose = {
    x: (segment.startX + segment.endX) / 2,
    y: (segment.startY + segment.endY) / 2,
    heading: Math.atan2(chordY, chordX),
  };
  if (!classify(createPoseCarState(pose)).isRoadOnly) {
    throw new Error(
      `the first-chord midpoint (${String(pose.x)}, ${String(pose.y)}) is NOT road-core`,
    );
  }
  return pose;
}

/**
 * Resolves the DEPARTED pose: the FIRST PERPENDICULAR offset from the
 * on-surface midpoint whose FINE classification is NOT drivable.
 *
 * The walk steps 0.25 m at a time along BOTH perpendicular signs (the
 * chord normal) up to 40 m, and returns the first pose the classifier
 * reports as `isDrivable === false` - the car that has LEFT the
 * drivable surface (beyond the grass apron).
 *
 * @param spec - The served track spec.
 * @param onSurfacePose - The on-surface pose the walk starts from.
 * @param classify - The FINE classifier over the served ribbon.
 * @returns The departed pose.
 * @throws When the served ribbon is degenerate or no non-drivable pose
 *   was found inside the walk horizon.
 */
function resolveDepartedPose(
  spec: ServedTrackSpec,
  onSurfacePose: CarPose,
  classify: ServedSurfaceClassifier,
): CarPose {
  const segment = spec.segments[0];
  if (segment === undefined) {
    throw new Error('the served ribbon carries no chord');
  }
  const chordX = segment.endX - segment.startX;
  const chordY = segment.endY - segment.startY;
  const chordLengthM = Math.hypot(chordX, chordY);
  if (chordLengthM <= 0) {
    throw new Error('the first chord of the served ribbon is degenerate');
  }
  const normalX = -chordY / chordLengthM;
  const normalY = chordX / chordLengthM;
  for (let step = 1; step <= DEPARTURE_WALK_STEPS; step += 1) {
    const offsetM = step * DEPARTURE_WALK_STEP_M;
    for (const sign of DEPARTURE_WALK_SIGNS) {
      const pose: CarPose = {
        x: onSurfacePose.x + normalX * offsetM * sign,
        y: onSurfacePose.y + normalY * offsetM * sign,
        heading: onSurfacePose.heading,
      };
      if (!classify(createPoseCarState(pose)).isDrivable) {
        return pose;
      }
    }
  }
  throw new Error(
    `no non-drivable pose was found within ${String(DEPARTURE_WALK_HORIZON_M)} m of the first-chord midpoint`,
  );
}

/**
 * Finds the HUD wrapper: the PARENT of the element whose trimmed text is
 * the speed readout.
 *
 * The HUD mounts FIRST into the reserved HUD container, so the mode
 * selector and the track selector are its SIBLINGS - scoping the probe
 * to the wrapper keeps the probe on the HUD subtree ONLY.
 *
 * @returns The HUD wrapper element.
 * @throws When the speed readout (or its parent) is missing.
 */
function findHudWrapper(): HTMLElement {
  const speedElement = Array.from(document.querySelectorAll('*')).find(
    (element): boolean =>
      SPEED_READOUT_PATTERN.test((element.textContent ?? '').trim()),
  );
  if (speedElement === undefined) {
    throw new Error('the HUD speed readout was not found in the document');
  }
  const wrapper = speedElement.parentElement;
  if (wrapper === null) {
    throw new Error('the HUD speed readout carries no parent element');
  }
  return wrapper;
}

/**
 * Collects the LEAF texts inside the HUD wrapper (the elements with no
 * element children), de-duplicated in first-seen order.
 *
 * @param wrapper - The HUD wrapper to walk.
 * @returns The distinct non-empty leaf texts.
 */
function collectHudLeafTexts(wrapper: HTMLElement): string[] {
  const texts: string[] = [];
  for (const element of Array.from(wrapper.querySelectorAll('*'))) {
    if (element.children.length > 0) {
      continue;
    }
    const text = (element.textContent ?? '').trim();
    if (text.length > 0 && !texts.includes(text)) {
      texts.push(text);
    }
  }
  return texts;
}

/**
 * Reports whether one HUD text is a KNOWN readout (and therefore NOT a
 * departure hint).
 *
 * @param text - The HUD leaf text to classify.
 * @returns TRUE when the text is one of the published readouts.
 */
function isKnownReadoutText(text: string): boolean {
  if (KNOWN_READOUT_TEXTS.includes(text)) {
    return true;
  }
  return KNOWN_READOUT_PATTERNS.some((pattern): boolean => pattern.test(text));
}

/**
 * Collects the DEPARTURE HINT texts the HUD publishes: every leaf text
 * inside the HUD wrapper that matches NONE of the KNOWN readouts.
 *
 * The hint's WORDING is not specified by the plan, so the probe NEVER
 * hard-codes any text - the EXISTENCE of an unknown leaf text inside the
 * HUD subtree is the contract.
 *
 * @returns The departure hint texts, in first-seen order.
 */
function collectDepartureHintTexts(): string[] {
  const wrapper = findHudWrapper();
  return collectHudLeafTexts(wrapper).filter(
    (text): boolean => !isKnownReadoutText(text),
  );
}

/**
 * Asserts (as a FIXTURE guard, never as a contract) that the HUD
 * republished the fake worker's reply.
 *
 * The guard proves the frame really ran and the HUD really mounted: the
 * speed readout carries the PUBLISHED speed (never the boot paint's
 * zero), so a probe that never drove a frame fails here instead of
 * masquerading as a missing departure hint.
 *
 * @throws When the HUD did not repaint from the published reply.
 */
function assertHudPaintedPublishedCar(): void {
  const wrapper = findHudWrapper();
  const speedText = (wrapper.textContent ?? '').includes(
    formatHudSpeed(PUBLISHED_SPEED_UNITS_PER_S),
  )
    ? formatHudSpeed(PUBLISHED_SPEED_UNITS_PER_S)
    : '';
  if (speedText.length === 0) {
    throw new Error(
      'the HUD did not repaint from the published reply (the speed readout carries no published speed)',
    );
  }
}

describe('the Neat Racing track-departure ADVISORY hint (RED-S28-1, AC-107)', () => {
  describe('the HUD departure-hint surface (host/hud.ts)', () => {
    /** The pose the fake worker session publishes for car 0. */
    let publishedPose: CarPose;
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

      // The fake simulation-worker session: it answers the boot's `init`
      // request with silence and every `frame` request with ONE `frames`
      // reply that carries car 0 at the PUBLISHED pose and the
      // test-controlled race clock (past the countdown).
      publishedPose = { x: 0, y: 0, heading: 0 };
      jest
        .mocked(createSimWorkerSession)
        .mockImplementation((): NeatRacingSimWorkerSession => ({
          handleRequest: (
            request: NeatRacingSimWorkerRequest,
          ): NeatRacingSimWorkerResponse[] => {
            if (request.type !== 'frame') {
              return [];
            }
            return [
              {
                type: 'frames',
                payload: {
                  fromTick: 0,
                  toTick: 0,
                  cars: [createPublishedCarState(publishedPose)],
                  rules: {
                    ...createRaceRulesState(),
                    elapsedS: PUBLISHED_ELAPSED_S,
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

      // The host shell: the canvas container plus the reserved HUD
      // container the boot glue mounts the HUD + selectors into.
      document.body.innerHTML = `<div id="${HOST_OUTPUT_ID}"><div id="${RESERVED_HUD_OUTPUT_ID}"></div></div>`;
    });

    afterEach(() => {
      stopDemo?.();
      stopDemo = null;
      window.requestAnimationFrame = ORIGINAL_REQUEST_ANIMATION_FRAME;
      window.cancelAnimationFrame = ORIGINAL_CANCEL_ANIMATION_FRAME;
    });

    /**
     * Drives `frameCount` host frames by invoking the captured rAF
     * callbacks in order (the host re-schedules itself after every frame).
     *
     * @param frameCount - The number of frames to drive.
     * @throws When the host scheduled no callback for a driven frame.
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

    it('publishes a departure hint for a car driven OFF the drivable surface', () => {
      // Arrange: the demo boots against the fake worker session, and the
      // SERVED geometry comes from the demo's own track selector.
      stopDemo = getBootStart()(HOST_OUTPUT_ID);
      const spec = resolveTrackVariantSpec(readServedTrackId());
      const classify = createTrackSurfaceClassifier(spec.segments);
      const onSurfacePose = resolveOnSurfacePose(spec, classify);
      const departedPose = resolveDepartedPose(spec, onSurfacePose, classify);

      // Act: the fake worker publishes car 0 at the DEPARTED pose and the
      // host repaints the HUD from that reply.
      publishedPose = departedPose;
      driveFrames(DRIVEN_FRAME_COUNT);
      assertHudPaintedPublishedCar();

      const departureHintTexts = collectDepartureHintTexts();

      // Assert: the departure hint IS published (RED today - the HUD
      // publishes NO departure hint at all).
      expect(departureHintTexts.length).toBeGreaterThan(0);
    });

    it('publishes NO departure hint while the car is ON the drivable surface', () => {
      // Arrange: the demo boots against the fake worker session, and the
      // SERVED geometry comes from the demo's own track selector.
      stopDemo = getBootStart()(HOST_OUTPUT_ID);
      const spec = resolveTrackVariantSpec(readServedTrackId());
      const classify = createTrackSurfaceClassifier(spec.segments);
      const onSurfacePose = resolveOnSurfacePose(spec, classify);

      // Act: the fake worker publishes car 0 at the ON-SURFACE pose (the
      // road core) and the host repaints the HUD from that reply.
      publishedPose = onSurfacePose;
      driveFrames(DRIVEN_FRAME_COUNT);
      assertHudPaintedPublishedCar();

      const departureHintTexts = collectDepartureHintTexts();

      // Assert: the HUD publishes NO departure hint while the car is ON the
      // drivable surface (GREEN today, and it MUST stay GREEN).
      expect(departureHintTexts).toHaveLength(0);
    });
  });
});
