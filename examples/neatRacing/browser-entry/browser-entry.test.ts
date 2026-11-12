/**
 * @jest-environment jsdom
 */

/**
 * RED -> IMPLEMENT -> GREEN contract tests for the Neat Racing boot stub
 * (spec slice P1-S1-A).
 *
 * The suite pins two contracts from the step packet:
 *
 * - AC-001 (partial): `window.neatRacingStart` is a function that mounts the
 *   game canvas into the page shell, returns a stop handle, and never leaves
 *   page-level scroll behind (the shell CSS owns the no-scroll contract).
 * - AC-037 (partial): the host shell loads the host entry per the flagship
 *   demo contract (bundle from docs/assets, boot through the global start).
 *
 * @example
 * ```ts
 * window.neatRacingStart('neatracing-output');
 * ```
 */

/**
 * Step 13 (P1-boot-glue) wiring-contract doubles.
 *
 * The slice red_test note: "browser-entry.test.ts wiring contract fails while
 * the boot stub only mounts the canvas; spy doubles over context.service +
 * render-loop + canvas-dimensions assert each wiring call and its order."
 * The factories below keep the real module exports (so the fallback message
 * and other constants can never drift) and replace only the three boot seams
 * with `jest.fn()` spies.
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

jest.mock('./canvas-dimensions.utils', () => ({
  ...jest.requireActual('./canvas-dimensions.utils'),
  updateFullWindowRendererSize: jest.fn(),
}));

/**
 * Step 14 (P1-boot-composition) reserved-container mount doubles.
 *
 * The slice red_test note: "browser-entry.test.ts fails while
 * NEATRACING_BOOT_PROGRAM_REGISTRY is still the empty readonly never[] and
 * index.html lacks the neatracing-hud-output container; asserts the registry
 * keys equal NEATRACING_WARMED_PROGRAM_KEYS, the count parity vs
 * NEATRACING_WARMED_PROGRAM_COUNT, and the createHud/createRaceModeSelector
 * mount calls with the reserved id". The factories below keep the real module
 * exports (so the resolve-or-throw host contract constants never drift) and
 * spy only the two reserved-container mount seams: the asserted calls are the
 * boot composition seam, not the DOM state the host shell owns.
 */
jest.mock('../host/hud', () => ({
  ...jest.requireActual('../host/hud'),
  createHud: jest.fn(),
}));

jest.mock('../host/mode-selector', () => ({
  ...jest.requireActual('../host/mode-selector'),
  createRaceModeSelector: jest.fn(),
}));

/**
 * Phase-3 Step-01 (P3-S1-wiring) wiring doubles.
 *
 * The slice red_test note: "browser-entry.test.ts wiring contract fails
 * while composerFrame.enabled is a hardcoded true and no controller tick
 * exists - wiring doubles (P1-boot-glue invocationCallOrder precedent) mock
 * the controller + composer modules and pin exactly one tickAdaptiveQuality
 * per rendered frame (zero-step frames included) ahead of the composer
 * draw, the stage-record-derived off-switch value handed to the frame
 * (stage 0 true, stage 1 bloom-off false), and the clamped-negative-stage
 * non-undefined knob."
 *
 * The factories below keep every warmup helper REAL through
 * `requireActual` (the nine-program warmup registry contract must never
 * drift) and replace only the eight pass-factory seams with `jest.fn()`
 * spies; the composer seam is fully mocked so the composition attempt
 * succeeds on the fake GL fixture. The adaptive-QUALITY controller is NOT
 * mocked at module scope: its spies are installed per test with
 * `jest.spyOn` and restored by `afterEach`, so every other describe keeps
 * the REAL controller.
 */
jest.mock('../engine/passes/bloom.pass', () => ({
  ...jest.requireActual('../engine/passes/bloom.pass'),
  createBloomPass: jest.fn(),
}));

jest.mock('../engine/passes/car.pass', () => ({
  ...jest.requireActual('../engine/passes/car.pass'),
  createCarPass: jest.fn(),
}));

jest.mock('../engine/passes/glass.pass', () => ({
  ...jest.requireActual('../engine/passes/glass.pass'),
  createGlassPass: jest.fn(),
}));

jest.mock('../engine/passes/track.pass', () => ({
  ...jest.requireActual('../engine/passes/track.pass'),
  createTrackPass: jest.fn(),
}));

jest.mock('../track/grid.pass', () => ({
  ...jest.requireActual('../track/grid.pass'),
  createTrackGridPass: jest.fn(),
}));

jest.mock('../track/right-world.pass', () => ({
  ...jest.requireActual('../track/right-world.pass'),
  createRightWorldPass: jest.fn(),
}));

jest.mock('../engine/passes/pulse.pass', () => ({
  ...jest.requireActual('../engine/passes/pulse.pass'),
  createPulsePass: jest.fn(),
}));

jest.mock('../engine/passes/starfield.pass', () => ({
  ...jest.requireActual('../engine/passes/starfield.pass'),
  createStarfieldPass: jest.fn(),
}));

jest.mock('../host/frame-composer', () => ({
  createFrameComposer: jest.fn(),
}));

import { afterEach, beforeEach, describe, expect, it } from '@jest/globals';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import {
  NEATRACING_FALLBACK_MESSAGE,
  NEATRACING_WARMED_PROGRAM_COUNT,
  NEATRACING_WARMED_PROGRAM_KEYS,
  initRenderingContext,
  runShaderWarmup,
} from '../engine/context.service';
import type { RenderingContextInitResult } from '../engine/context.service';
import type { NeatRacingHud } from '../host/hud';
import { createHud } from '../host/hud';
import type { NeatRacingRaceModeSelector } from '../host/mode-selector';
import { createRaceModeSelector } from '../host/mode-selector';
import { advanceFrameSteps } from '../host/render-loop.utils';
// The shared traction law, used here as the ORACLE for the seam contract:
// the physics state publishes `computeTractionSeverity(slip, speed)`, so
// the seam must carry that EXACT number for the slip and the speed it
// publishes - the browser never re-derives the severity.
import { computeTractionSeverity } from '../physics/drift.model';
// The fixed physics timestep, used here as the ORACLE for the EIGHTH seam
// field: the pose spring integrates the pose with the published rate
// (`pose = prev.pose + rate * dt`), so the published pose rate is the
// crab's own rate and not a re-derived number.
import { NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS } from '../physics/physics.step.service';
// The chase-camera law, used here as the ORACLE for the drift-camera
// READER contract (Step 14X, RED-5): the browser entry publishes the
// follow offset the law computed, so the published offset MUST equal the
// law replayed from the seam's OWN published drift inputs. The separation
// is NOT a separate module instance (Jest gives ONE module registry per
// test file): the reader's path is the browser entry's OWN call into the
// law, and the oracle's path is THIS TEST's own replay from the seam's
// published drift inputs - two independent code paths over one module.
import {
  createChaseCameraFollowState,
  stepChaseCameraFollow,
} from './chase-camera';
// Side-effect import: evaluating the boot stub registers
// `window.neatRacingStart` on the jsdom host.
import './browser-entry';
import {
  NEATRACING_CAR_SCENE_YAW_OFFSET_RAD,
  carSceneWorldZ,
  carSceneYawRad,
  lookAtInto,
  multiplyViewProjectionInto,
  perspectiveInto,
} from './browser-entry';
import type { NeatRacingStart } from './browser-entry';
import type { CanvasDimensions } from './canvas-dimensions.utils';
import { updateFullWindowRendererSize } from './canvas-dimensions.utils';
// The adaptive-quality controller ladder (the AC-035 ORACLE): the resolved
// stage record's `bloomEnabled` value is what the composed frame must
// carry. The controller functions are SPIED (never module-mocked) for the
// wiring contract - the namespace import is the per-test spy target.
import * as adaptiveQuality from '../engine/adaptive-quality';
import { NEATRACING_ADAPTIVE_QUALITY_STAGES } from '../engine/adaptive-quality';
// The composed-frame type + the composer seam double targets
// (P3-S1-wiring): the composition attempt succeeds on the fake GL fixture
// through these doubles.
import { createFrameComposer } from '../host/frame-composer';
import type { NeatRacingFrameComposerFrame } from '../host/frame-composer';
// The pass-factory seam doubles (P3-S1-wiring): the factories themselves
// are stubbed with inert pass doubles so the composition attempt never
// touches the fake GL context beyond the capability gate.
import { createBloomPass } from '../engine/passes/bloom.pass';
import { createCarPass } from '../engine/passes/car.pass';
import { createGlassPass } from '../engine/passes/glass.pass';
import { createTrackPass } from '../engine/passes/track.pass';
import { createTrackGridPass } from '../track/grid.pass';
import { createRightWorldPass } from '../track/right-world.pass';
import { createPulsePass } from '../engine/passes/pulse.pass';
import { createStarfieldPass } from '../engine/passes/starfield.pass';

const HOST_SHELL_PATH = resolve(
  process.cwd(),
  'examples',
  'neatRacing',
  'index.html',
);

/** Reads the source host shell (not a generated docs copy). */
function readHostShell(): string {
  return readFileSync(HOST_SHELL_PATH, 'utf-8');
}

/** The frame-composer source path the AC-035 source pin reads. */
const FRAME_COMPOSER_PATH = resolve(
  process.cwd(),
  'examples',
  'neatRacing',
  'host',
  'frame-composer.ts',
);

/** Reads the frame-composer source (not a generated docs copy). */
function readFrameComposerSource(): string {
  return readFileSync(FRAME_COMPOSER_PATH, 'utf-8');
}

/** Resets the jsdom document to a minimal host shell between tests. */
function resetHostShell(): void {
  document.body.innerHTML = '<div id="neatracing-output"></div>';
}

/**
 * Returns the boot function installed by the boot stub.
 *
 * Throws a descriptive error when the boot stub has not registered
 * `window.neatRacingStart`, which is the RED failure mode of this slice.
 */
function getBootStart(): NeatRacingStart {
  const start = window.neatRacingStart;
  if (typeof start !== 'function') {
    throw new Error('neatRacingStart was not registered on window');
  }
  return start;
}

/** Boot-seam doubles for the Step 02 boot modules (module-scope spies). */
const mockedInitRenderingContext = jest.mocked(initRenderingContext);
const mockedRunShaderWarmup = jest.mocked(runShaderWarmup);
const mockedAdvanceFrameSteps = jest.mocked(advanceFrameSteps);
const mockedUpdateFullWindowRendererSize = jest.mocked(
  updateFullWindowRendererSize,
);

/** Mount-seam doubles for the Step 14 boot composition (module-scope spies). */
const mockedCreateHud = jest.mocked(createHud);
const mockedCreateRaceModeSelector = jest.mocked(createRaceModeSelector);

/** Composer + pass seam doubles for the P3-S1-wiring composition gate. */
const mockedCreateFrameComposer = jest.mocked(createFrameComposer);
const mockedCreateBloomPass = jest.mocked(createBloomPass);
const mockedCreateCarPass = jest.mocked(createCarPass);
const mockedCreateGlassPass = jest.mocked(createGlassPass);
const mockedCreateTrackPass = jest.mocked(createTrackPass);
const mockedCreateTrackGridPass = jest.mocked(createTrackGridPass);
const mockedCreateRightWorldPass = jest.mocked(createRightWorldPass);
const mockedCreatePulsePass = jest.mocked(createPulsePass);
const mockedCreateStarfieldPass = jest.mocked(createStarfieldPass);

/** Timestamp step (ms) between the driven frames of the wiring probe. */
const WIRING_FRAME_STEP_MS = 1000 / 60;

/** Timestamp handed to the captured RAF callback when tests drive frame one. */
const DRIVEN_FRAME_TIMESTAMP_MS = 0;

/**
 * Number of driven frames the sliding seam contract releases.
 *
 * The car needs a real acceleration phase before the drift law publishes
 * a non-zero slip, so the contract drives a short race (one simulation
 * step per frame), not a single frame.
 */
const SLIDING_SEAM_FRAME_COUNT = 120;

/** Timestamp step (ms) between the driven frames of the sliding contract. */
const SLIDING_SEAM_FRAME_STEP_MS = 1000 / 60;

/**
 * Reserved host DOM id the batch 94 handoff gives to the HUD container
 * (spec section 10.2; `neatracing-hud-output` in the host shell and both
 * mount factories). RED note: index.html does not reserve it yet.
 */
const RESERVED_HUD_OUTPUT_ID = 'neatracing-hud-output';

/** Live-branch fixture: the init result double plus its stable gl reference. */
interface LiveInitResultFixture {
  readonly result: RenderingContextInitResult;
  readonly gl: WebGL2RenderingContext;
}

/**
 * Creates a mock WebGL2 context with the methods the boot path exercises.
 *
 * Inert on purpose: this double only has to survive being handed to
 * `runShaderWarmup` and compared by identity in wiring assertions.
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

/**
 * Live-branch init double: `ok: true` with a stable inert gl, matching the
 * successful WebGL2 acquisition the wiring path must warm up and drive.
 */
function createLiveInitResultFixture(): LiveInitResultFixture {
  const gl = createMockWebGl2();
  return {
    gl,
    result: {
      ok: true,
      gl,
      isContextLost: (): boolean => false,
    },
  };
}

/**
 * Fallback-branch init double: stamps the service fallback message onto the
 * notice exactly like the production service would, so the boot path must
 * append the service-built element rather than rewrite its content.
 */
function createFallbackInitResultFixture(
  fallbackNotice: HTMLElement,
): RenderingContextInitResult {
  fallbackNotice.textContent = NEATRACING_FALLBACK_MESSAGE;
  return {
    ok: false,
    gl: null,
    message: NEATRACING_FALLBACK_MESSAGE,
    fallbackElement: fallbackNotice,
    isContextLost: (): boolean => false,
  };
}

/** Standard 1280x720 backing-store size-double result. */
function createMockRendererSizeUpdate(): CanvasDimensions {
  return { width: 1280, height: 720 };
}

/**
 * Inert HUD controller double for the Step 14 boot-composition mounts.
 *
 * The red contract asserts only the reserved-id mount call, so the double
 * stays inert; per-frame HUD updates stay with the drive-loop slice.
 */
function createInertHud(): NeatRacingHud {
  return {
    container: document.createElement('div'),
    speedLabel: document.createElement('div'),
    lapLabel: document.createElement('div'),
    timeLabel: document.createElement('div'),
    update: (): void => undefined,
  };
}

/**
 * Inert race mode selector double for the Step 14 boot-composition mounts.
 *
 * The red contract asserts only the reserved-id mount call; the mode and
 * switch API behavior stays pinned by the host mode-selector suite.
 */
function createInertRaceModeSelector(): NeatRacingRaceModeSelector {
  return {
    select: document.createElement('select'),
    mode: 'manual',
    setMode: (): void => undefined,
    onSwitch: (): void => undefined,
  };
}

describe('camera matrix helpers (P1-composition-green regression)', () => {
  it('maps a representative world point to positive, finite clip-space w', () => {
    const projection = new Float32Array(16);
    perspectiveInto(projection, Math.PI / 4, 16 / 9, 0.1, 1000);
    const view = new Float32Array(16);
    lookAtInto(view, 10, 5, 10, 2, 0.5, 3);
    const viewProjection = new Float32Array(16);
    multiplyViewProjectionInto(viewProjection, view, projection);

    // Representative world point near the track origin.
    const world = new Float32Array([2, 0.5, 3, 1]);
    const clip = new Float32Array(4);
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let col = 0; col < 4; col += 1) {
        sum += viewProjection[col * 4 + row] * world[col];
      }
      clip[row] = sum;
    }

    expect(clip[3]).toBeGreaterThan(0);
    expect(Number.isFinite(clip[0] / clip[3])).toBe(true);
    expect(Number.isFinite(clip[1] / clip[3])).toBe(true);
    expect(Number.isFinite(clip[2] / clip[3])).toBe(true);
  });
});

describe('Neat Racing browser entry boot stub', () => {
  beforeEach(() => {
    resetHostShell();
  });

  describe('AC-001 (partial): boot stub mounts the game canvas', () => {
    it('registers window.neatRacingStart as a function', () => {
      expect(typeof window.neatRacingStart).toBe('function');
    });

    it('mounts the game canvas into the page shell', () => {
      getBootStart()('neatracing-output');
      const output = document.getElementById('neatracing-output');
      expect(output?.querySelector('canvas')).toBeInstanceOf(HTMLCanvasElement);
    });

    it('returns a stop handle that removes the mounted canvas', () => {
      const stop = getBootStart()('neatracing-output');
      stop();
      const output = document.getElementById('neatracing-output');
      expect(output?.querySelector('canvas')).toBeNull();
    });

    it('throws a descriptive error when the output container is missing', () => {
      expect(() => getBootStart()('missing-output')).toThrow(/missing-output/);
    });
  });

  describe('AC-037 (partial): host HTML shell per the flagship demo contract', () => {
    it('exists with the game page markup', () => {
      expect(readHostShell().length).toBeGreaterThan(0);
    });

    it('prevents page-level scroll via overflow hidden', () => {
      expect(readHostShell()).toContain('overflow: hidden');
    });

    it('provides the neatracing-output host container', () => {
      expect(readHostShell()).toContain('id="neatracing-output"');
    });

    it('loads the host entry and boots through window.neatRacingStart', () => {
      const html = readHostShell();
      expect(html).toContain('neatracing.bundle.js');
      expect(html).toContain('neatRacingStart');
    });
  });
});

/**
 * Step 13 (P1-boot-glue) wiring contract.
 *
 * RED while the boot stub only mounts the canvas: none of the three boot
 * seams (context.service, render-loop.utils, canvas-dimensions.utils) are
 * imported yet, so every wiring assertion below fails on the legible
 * "expected to be called, received 0 calls" mismatch. These tests answer
 * AC-002 completion only; they do not pin any frame-simulation behavior.
 */
describe('scene handedness mapping (Step 14B round-3: mirrored ground + nose yaw)', () => {
  it('maps the top-down math y into the mirrored scene z with world z = -y', () => {
    expect(carSceneWorldZ(12)).toBe(-12);
    expect(carSceneWorldZ(-3)).toBe(3);
    expect(carSceneWorldZ(0)).toBe(0);
  });

  it('offsets the top-down heading by the pinned nose yaw so the model +z nose tracks scene forward', () => {
    // rotationYMat4 maps the model +z nose to (sin yaw, 0, cos yaw); the
    // mirrored scene forward is (cos heading, 0, -sin heading) because world
    // z = -y, which yaw = PI/2 + heading produces exactly: at heading 0 the
    // nose points along scene +x, and positive heading (A/left steer) tips
    // the nose toward scene -z (screen left).
    expect(NEATRACING_CAR_SCENE_YAW_OFFSET_RAD).toBeCloseTo(Math.PI / 2, 12);
    expect(carSceneYawRad(0)).toBeCloseTo(Math.PI / 2, 12);
    expect(carSceneYawRad(1)).toBeCloseTo(Math.PI / 2 + 1, 12);
    expect(carSceneYawRad(-Math.PI / 2)).toBeCloseTo(0, 12);
  });
});

/**
 * The EXACT field count the live drift debug seam must carry (Step 14P,
 * slice `P1-drift-rewire-red`; the RED-8 acceptance criterion, grown by
 * Step 14Q2, slice `P1-drift-traction-seam`, and by Step 14R, slice
 * `P1-drift-presentation-car`; AC-088).
 *
 * The count is an EXACT number, never a range: the seam grows by exactly
 * ONE field per growth wave - the camera follow offset in the re-wiring
 * wave, the PUBLISHED traction severity in the traction wave, and the
 * PUBLISHED pose-spring rate in this one - so seven fields are a defect
 * and nine fields are a defect.
 */
const DRIFT_SEAM_FIELD_COUNT = 8;

/**
 * The pinned eight-field seam key set (sorted).
 *
 * The five live fields (poseRad, slipAngleRad, speedMps, tickHz, tick) plus
 * the camera follow offset, the PUBLISHED traction severity and the
 * PUBLISHED pose-spring rate.
 */
const DRIFT_SEAM_KEYS = [
  'followOffsetRad',
  'poseRad',
  'poseRateRadPerS',
  'slipAngleRad',
  'speedMps',
  'tick',
  'tickHz',
  'tractionSeverity',
];

/**
 * The drift debug seam, read through the PUBLIC window handle.
 *
 * The seam type (`NeatRacingDriftState`, browser-entry.ts) is MODULE-LOCAL
 * and UNEXPORTED - the slice forbids the export, because a new public
 * export would be a new public API and would break the
 * one-assignment-per-frame seam rule. The handle is therefore widened to a
 * loose record here: the assertions read the LIVE seam object, never a
 * re-derived local expression.
 */
type DriftSeamView = Record<string, unknown>;

/**
 * Reads the live drift debug seam off the window handle.
 *
 * @returns The live seam, or `undefined` while no frame has been driven.
 */
function readDriftSeam(): DriftSeamView | undefined {
  return (window as unknown as { neatRacingDriftState?: DriftSeamView })
    .neatRacingDriftState;
}

/**
 * Reads a numeric field off the live seam.
 *
 * @param seam - The live seam view.
 * @param key - The field name to read.
 * @returns The field value, or `NaN` when the field is absent or is not a
 *   number, so a law comparison fails LEGIBLY instead of passing on an
 *   `undefined` field.
 */
function readSeamNumber(seam: DriftSeamView | undefined, key: string): number {
  const value = seam?.[key];
  return typeof value === 'number' ? value : Number.NaN;
}

/**
 * Counts the drift seam writes by installing a window property accessor.
 *
 * The production seam write is a plain `window.neatRacingDriftState =
 * {...}` assignment inside the zero-arg frame builder, so an accessor
 * observes every write: the one-assignment-per-frame seam rule is then
 * observable from the LIVE seam instead of from a re-derived expression.
 *
 * @returns The write counter and the accessor restorer.
 */
function countDriftSeamAssignments(): {
  count: () => number;
  restore: () => void;
} {
  const original = Object.getOwnPropertyDescriptor(
    window,
    'neatRacingDriftState',
  );
  let writes = 0;
  let current: DriftSeamView | undefined;
  Object.defineProperty(window, 'neatRacingDriftState', {
    configurable: true,
    get: (): DriftSeamView | undefined => current,
    set: (value: DriftSeamView): void => {
      writes += 1;
      current = value;
    },
  });
  return {
    count: (): number => writes,
    restore: (): void => {
      if (original !== undefined) {
        Object.defineProperty(window, 'neatRacingDriftState', original);
        return;
      }
      delete (window as unknown as { neatRacingDriftState?: unknown })
        .neatRacingDriftState;
    },
  };
}

describe('Neat Racing browser entry boot wiring (Step 13, P1-boot-glue)', () => {
  /** Live-branch fixture captured before each test re-stamps the doubles. */
  let liveFixture: LiveInitResultFixture;
  /** RAF callbacks captured from the boot loop; tests drive them manually. */
  let scheduledRafCallbacks: FrameRequestCallback[];

  beforeEach(() => {
    jest.resetAllMocks();

    // Live-branch doubles: the boot path sees a WebGL2 context and a fixed
    // 1280x720 letterbox result regardless of the jsdom host capabilities.
    liveFixture = createLiveInitResultFixture();
    mockedInitRenderingContext.mockReturnValue(liveFixture.result);
    mockedRunShaderWarmup.mockReturnValue({
      warmedProgramCount: 0,
      offscreenDrawCount: 0,
    });
    mockedUpdateFullWindowRendererSize.mockReturnValue(
      createMockRendererSizeUpdate(),
    );
    mockedAdvanceFrameSteps.mockReturnValue(0);

    // Replace the jsdom RAF implementation with a capturing stub so the
    // boot loop's frames are driven manually instead of by timers.
    scheduledRafCallbacks = [];
    window.requestAnimationFrame = (callback: FrameRequestCallback): number => {
      scheduledRafCallbacks.push(callback);
      return scheduledRafCallbacks.length;
    };

    resetHostShell();
  });

  describe('boot path wiring order and seams (AC-002 completion)', () => {
    it('initializes the WebGL2 rendering context exactly once with the mounted canvas', () => {
      getBootStart()('neatracing-output');
      const mountedCanvas = document
        .getElementById('neatracing-output')
        ?.querySelector('canvas');

      expect(mockedInitRenderingContext).toHaveBeenCalledTimes(1);
      expect(mockedInitRenderingContext.mock.calls[0][0].canvas).toBe(
        mountedCanvas,
      );
      expect(mockedInitRenderingContext.mock.calls[0][0].documentRef).toBe(
        document,
      );
    });

    it('runs the shader warmup on the acquired context before the first frame', () => {
      getBootStart()('neatracing-output');
      scheduledRafCallbacks[0]?.(DRIVEN_FRAME_TIMESTAMP_MS);

      expect(mockedRunShaderWarmup).toHaveBeenCalledTimes(1);
      expect(mockedRunShaderWarmup.mock.calls[0][0]).toBe(liveFixture.gl);
      expect(mockedRunShaderWarmup.mock.invocationCallOrder[0]).toBeLessThan(
        mockedAdvanceFrameSteps.mock.invocationCallOrder[0],
      );
    });

    it('applies the full-window stretch renderer size at boot from the live viewport', () => {
      getBootStart()('neatracing-output');
      const mountedCanvas = document
        .getElementById('neatracing-output')
        ?.querySelector('canvas');

      expect(mockedUpdateFullWindowRendererSize).toHaveBeenCalledTimes(1);
      const sizeUpdateArg = mockedUpdateFullWindowRendererSize.mock.calls[0][0];
      expect(sizeUpdateArg.canvas).toBe(mountedCanvas);
      expect(sizeUpdateArg).toMatchObject({
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
      });
    });

    it('re-applies the full-window stretch renderer size when the viewport resizes', () => {
      getBootStart()('neatracing-output');
      const bootSizeUpdates =
        mockedUpdateFullWindowRendererSize.mock.calls.length;

      window.dispatchEvent(new Event('resize'));

      expect(
        mockedUpdateFullWindowRendererSize.mock.calls.length,
      ).toBeGreaterThan(bootSizeUpdates);
    });

    it('asserts the AC-002 boot path order across the wiring doubles', () => {
      getBootStart()('neatracing-output');
      scheduledRafCallbacks[0]?.(DRIVEN_FRAME_TIMESTAMP_MS);

      expect(
        mockedInitRenderingContext.mock.invocationCallOrder[0],
      ).toBeLessThan(mockedRunShaderWarmup.mock.invocationCallOrder[0]);
      expect(mockedRunShaderWarmup.mock.invocationCallOrder[0]).toBeLessThan(
        mockedUpdateFullWindowRendererSize.mock.invocationCallOrder[0],
      );
      expect(
        mockedUpdateFullWindowRendererSize.mock.invocationCallOrder[0],
      ).toBeLessThan(mockedAdvanceFrameSteps.mock.invocationCallOrder[0]);
    });

    it('keeps the RAF loop alive by requesting the next frame after the first', () => {
      getBootStart()('neatracing-output');

      expect(scheduledRafCallbacks.length).toBe(1);

      scheduledRafCallbacks[0]?.(DRIVEN_FRAME_TIMESTAMP_MS);
      expect(scheduledRafCallbacks.length).toBe(2);
    });
  });

  describe('graceful fallback without WebGL2 (AC-002 completion)', () => {
    it('appends the fallback notice with the service message and stays bootable', () => {
      const fallbackNotice = document.createElement('div');
      mockedInitRenderingContext.mockReturnValue(
        createFallbackInitResultFixture(fallbackNotice),
      );
      const stop = getBootStart()('neatracing-output');
      const output = document.getElementById('neatracing-output');

      expect(output?.contains(fallbackNotice)).toBe(true);
      expect(fallbackNotice.textContent).toBe(NEATRACING_FALLBACK_MESSAGE);
      expect(typeof stop).toBe('function');
    });

    it('does not run the shader warmup when the context init degrades to the fallback', () => {
      mockedInitRenderingContext.mockReturnValue(
        createFallbackInitResultFixture(document.createElement('div')),
      );

      getBootStart()('neatracing-output');

      expect(mockedInitRenderingContext).toHaveBeenCalledTimes(1);
      expect(mockedRunShaderWarmup).not.toHaveBeenCalled();
    });
  });

  /**
   * The DRIFT DEBUG SEAM contract (Step 14P, slice `P1-drift-rewire-red`;
   * the RED-8 acceptance criterion and the plan's `red_contract_api`
   * section 6).
   *
   * THE SEAM OWNER: the drift debug seam is assigned INSIDE the zero-arg
   * `buildComposerFrame` closure (browser-entry.ts:931, the write at
   * :980-986), so it is observable ONLY through the module's own public
   * window handle (`window.neatRacingDriftState`) after a frame is driven.
   * The module-local `NeatRacingDriftState` type is NEVER exported (the
   * export is forbidden by this slice: a new public export would be a new
   * public API and would break the one-assignment-per-frame seam rule),
   * so the contracts read the LIVE handle through a widened local view and
   * never name the type.
   *
   * RED-8: the seam grows by EXACTLY ONE field - the camera follow offset
   * - so the field COUNT is asserted as the EXACT number six (never a
   * range) and the field SET is asserted against the pinned key list. The
   * seam MECHANICS are retained: ONE assignment per frame, which the
   * assignment-count contract pins with a window property accessor.
   *
   * Step 14Q2 (slice `P1-drift-traction-seam`; AC-088): the seam grows by
   * EXACTLY ONE more field - the PUBLISHED traction severity - so the
   * field COUNT is now the EXACT number seven (never a range) and the
   * field SET is asserted against the pinned seven-key list. The value is
   * READ from the physics state and NEVER re-derived in the browser, which
   * the traction-severity contract pins against the SHARED law value of
   * the slip and the speed the seam itself publishes.
   *
   * Step 14R (slice `P1-drift-presentation-car`; AC-088): the seam grows by
   * EXACTLY ONE more field - the PUBLISHED pose-spring rate - so the field
   * COUNT is now the EXACT number eight (never a range) and the field SET
   * is asserted against the pinned eight-key list. The EIGHTH field is a
   * NAMED channel and it is NOT the traction severity (that is the SEVENTH
   * field): the pose rate is the crab's own rate, READ from the physics
   * state, so the browser half can measure the ring-down and the far-side
   * pop in simulation time instead of differencing the pose across
   * render-rate dependent frames.
   */
  describe('drift debug seam (Step 14P RED-8 + Step 14Q2 + Step 14R, P1-drift-presentation-car)', () => {
    /** Restores the seam accessor after the assignment-count contract. */
    let restoreSeamAccessor: (() => void) | null = null;

    afterEach(() => {
      restoreSeamAccessor?.();
      restoreSeamAccessor = null;
    });

    /**
     * Boots the page and drives ONE frame, so the zero-arg frame builder
     * runs and the seam is written.
     */
    function driveSeamFrame(): void {
      getBootStart()('neatracing-output');
      scheduledRafCallbacks[0]?.(DRIVEN_FRAME_TIMESTAMP_MS);
    }

    // [RED-8-SEAM-FIELD-COUNT]
    it('carries exactly eight fields on the live drift seam', () => {
      // Arrange: the seam is written by the frame builder, so a frame runs.
      driveSeamFrame();

      // Act
      const seam = readDriftSeam();

      // Assert
      expect(seam).toBeDefined();
      expect(Object.keys(seam ?? {})).toHaveLength(DRIFT_SEAM_FIELD_COUNT);
    });

    // [RED-8-SEAM-KEY-SET]
    it('carries the pinned eight-field key set on the live drift seam', () => {
      // Arrange
      driveSeamFrame();

      // Act
      const seam = readDriftSeam();

      // Assert
      expect(Object.keys(seam ?? {}).sort()).toEqual(DRIFT_SEAM_KEYS);
    });

    // [RED-8-SEAM-FOLLOW-OFFSET]
    it('publishes the camera follow offset as a finite number on the live drift seam', () => {
      // Arrange
      driveSeamFrame();

      // Act
      const seam = readDriftSeam();

      // Assert
      expect(typeof seam?.followOffsetRad).toBe('number');
      expect(Number.isFinite(seam?.followOffsetRad)).toBe(true);
    });

    // [14Q2-SEAM-TRACTION-SEVERITY]
    it('publishes the traction severity as a finite number on the live drift seam', () => {
      // Arrange
      driveSeamFrame();

      // Act
      const seam = readDriftSeam();

      // Assert
      expect(typeof seam?.tractionSeverity).toBe('number');
      expect(Number.isFinite(seam?.tractionSeverity)).toBe(true);
    });

    // [14Q2-SEAM-TRACTION-SEVERITY-LAW]
    it('publishes the traction severity as the SHARED law value of the published slip and speed', () => {
      // Arrange
      driveSeamFrame();

      // Act
      const seam = readDriftSeam();

      // Assert: the physics state publishes
      // `computeTractionSeverity(slip, speed)`, so the seam carries that
      // EXACT number for the slip and the speed it publishes: the
      // browser READS the published severity and never re-derives it.
      expect(seam?.tractionSeverity).toBe(
        computeTractionSeverity(
          readSeamNumber(seam, 'slipAngleRad'),
          readSeamNumber(seam, 'speedMps'),
        ),
      );
    });

    // [14Q2-SEAM-TRACTION-SEVERITY-SLIDING]
    it('publishes a NON-ZERO traction severity while the car slides', () => {
      // Arrange: release ONE simulation step per frame and hold the
      // throttle plus a steering key, so the car accelerates into a slide
      // and the physics state publishes a non-zero severity.
      mockedAdvanceFrameSteps.mockReturnValue(1);
      getBootStart()('neatracing-output');
      window.dispatchEvent(
        new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true }),
      );
      window.dispatchEvent(
        new KeyboardEvent('keydown', { code: 'ArrowRight', cancelable: true }),
      );
      for (let frame = 0; frame < SLIDING_SEAM_FRAME_COUNT; frame += 1) {
        scheduledRafCallbacks[frame]?.(
          DRIVEN_FRAME_TIMESTAMP_MS + frame * SLIDING_SEAM_FRAME_STEP_MS,
        );
      }

      // Act
      const seam = readDriftSeam();

      // Assert: on a slide the published severity is positive, and it is
      // STILL the shared law value of the published slip and speed.
      expect(seam?.tractionSeverity).toBeGreaterThan(0);
      expect(seam?.tractionSeverity).toBe(
        computeTractionSeverity(
          readSeamNumber(seam, 'slipAngleRad'),
          readSeamNumber(seam, 'speedMps'),
        ),
      );
    });

    // [14R-SEAM-POSE-RATE]
    it('publishes the pose-spring rate as a finite number on the live drift seam', () => {
      // Arrange
      driveSeamFrame();

      // Act
      const seam = readDriftSeam();

      // Assert: the EIGHTH field is the PUBLISHED pose rate, READ from the
      // physics state (`car.drift.pose.poseRate`) and never re-derived in
      // the browser. It is a NAMED channel of its own, NOT the traction
      // severity (that is the SEVENTH field).
      expect(typeof seam?.poseRateRadPerS).toBe('number');
      expect(Number.isFinite(seam?.poseRateRadPerS)).toBe(true);
    });

    // [14R-SEAM-POSE-RATE-IS-THE-POSE-RATE]
    it('publishes the RATE of the pose it publishes beside it', () => {
      // Arrange: drive a real slide (one simulation step per frame) and
      // record the seam after every driven frame.
      mockedAdvanceFrameSteps.mockReturnValue(1);
      getBootStart()('neatracing-output');
      window.dispatchEvent(
        new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true }),
      );
      window.dispatchEvent(
        new KeyboardEvent('keydown', { code: 'ArrowRight', cancelable: true }),
      );
      const samples: Array<{
        readonly tick: number;
        readonly poseRad: number;
        readonly poseRateRadPerS: number;
      }> = [];
      for (let frame = 0; frame < SLIDING_SEAM_FRAME_COUNT; frame += 1) {
        scheduledRafCallbacks[frame]?.(
          DRIVEN_FRAME_TIMESTAMP_MS + frame * SLIDING_SEAM_FRAME_STEP_MS,
        );
        const seam = readDriftSeam();
        samples.push({
          tick: readSeamNumber(seam, 'tick'),
          poseRad: readSeamNumber(seam, 'poseRad'),
          poseRateRadPerS: readSeamNumber(seam, 'poseRateRadPerS'),
        });
      }

      // Act: take the LAST pair of samples that are exactly ONE simulation
      // tick apart (the frame accumulator may release a tick in a later
      // frame, so only a one-tick step can be differenced).
      let poseStep: { readonly delta: number; readonly rate: number } | null =
        null;
      for (let index = 1; index < samples.length; index += 1) {
        const previous = samples[index - 1];
        const current = samples[index];
        if (current.tick - previous.tick === 1) {
          poseStep = {
            delta: current.poseRad - previous.poseRad,
            rate: current.poseRateRadPerS,
          };
        }
      }

      // Assert: the pose spring integrates the pose with the PUBLISHED
      // rate (`pose = prev.pose + rate * dt`), so the eighth field is the
      // crab's own rate - the quantity the browser half would otherwise
      // have to ESTIMATE by differencing the pose across rendered frames.
      expect(poseStep).not.toBeNull();
      expect(poseStep?.delta).toBeCloseTo(
        (poseStep?.rate ?? 0) * NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
        10,
      );
    });

    // [RED-8-SEAM-ONE-ASSIGNMENT-PER-FRAME]
    it('assigns the drift seam exactly once per driven frame', () => {
      // Arrange: the accessor counts the seam writes.
      const assignments = countDriftSeamAssignments();
      restoreSeamAccessor = assignments.restore;
      getBootStart()('neatracing-output');

      // Act + Assert: one write for the first driven frame ...
      scheduledRafCallbacks[0]?.(DRIVEN_FRAME_TIMESTAMP_MS);
      expect(assignments.count()).toBe(1);

      // ... and exactly one more for the second driven frame.
      scheduledRafCallbacks[1]?.(DRIVEN_FRAME_TIMESTAMP_MS);
      expect(assignments.count()).toBe(2);
    });
  });

  /**
   * Step 14X (P1-driftcam-red) drift-camera READER contract.
   *
   * RED while the camera law has NO aim-offset cap: the browser entry
   * publishes the chase-camera follow offset on the live drift seam
   * (`followOffsetRad`), so the reader publishes the UNCAPPED offset.
   *
   * WHY TWO CLAUSES:
   *
   * 1. [RED-5-PUBLISHED-CAP] (RED) - the published |followOffsetRad| NEVER
   *    exceeds the RECORDED cap. The cap is enforced AT THE SOURCE (inside
   *    the camera law), so EVERY reader - the HUD reader and this browser
   *    reader - can only ever see the CAPPED offset. The reader can never
   *    re-derive a bigger offset of its own.
   * 2. [RED-5-PUBLISHED-AGREES-WITH-THE-LAW] (GREEN rail) - the published
   *    offset EQUALS the offset the camera law computes from the seam's OWN
   *    published drift inputs. The physics publishes
   *    `travelHeadingRad = wrap(heading + slipAngle)`, so the law's
   *    divergence IS the published slip.
   *
   * WHY THE RAIL IS NOT VACUOUS: the published offset is READ from the live
   * seam (the browser's own reader) while the law's value is REPLAYED by
   * the test from the seam's published drift inputs - two independent code
   * paths. A replay that ignored the published drift would answer 0, and
   * the guards below prove the probe really drifted.
   */
  describe('drift camera reader (Step 14X, P1-driftcam-red; RED-5)', () => {
    /**
     * The probe length (frames) of the drift-camera reader probe.
     *
     * MEASURED (the post-14X re-pin by `P1-driftcam-repin`): the probe
     * drives ONE simulation tick per frame with the throttle and a
     * steering key held. The published |followOffsetRad| peaks at
     * 0.0992038678428377 rad after 120 frames (ledger row RP-1) and at
     * 0.14962536105320104 rad after 240 frames (ledger row RP-2); the
     * slide saturates there (the same peak is measured after 360, 480
     * and 600 frames - ledger row RP-3).
     * BOTH peaks are BELOW the RECORDED cap: the desensitised law follows
     * the slide SLOWER, and the cap itself is enforced AT THE SOURCE
     * inside the camera law, so EVERY reader can only ever publish the
     * CAPPED offset. The cap clause still discriminates: the SAME
     * 240-frame probe published a peak ABOVE the RECORDED cap under the
     * pre-14X law (the pre-wave measurement of the Step 14X re-baseline
     * ledger, row R-2), so dropping the source cap breaks the clause.
     */
    const DRIFT_CAMERA_PROBE_FRAME_COUNT = 240;
    /** The RECORDED aim-offset cap (rad) of the drift-camera wave. */
    const RECORDED_MAX_AIM_OFFSET_RAD = 0.17448929499999988;
    /** The tolerance of the reader comparison (rad). */
    const CAMERA_RAD_TOLERANCE = 1e-9;

    /** One live seam sample of the drift-camera reader. */
    interface DriftSeamCameraSample {
      /** The published physics tick. */
      readonly tick: number;
      /** The published slip angle (rad). */
      readonly slipAngleRad: number;
      /** The published speed (m/s). */
      readonly speedMps: number;
      /** The published chase-camera follow offset (rad). */
      readonly followOffsetRad: number;
    }

    /**
     * Drives a REAL slide (one simulation step per frame, the throttle and
     * a steering key held) and records the live seam after EVERY frame.
     *
     * @returns The per-frame seam samples, in frame order.
     */
    function driveSlidingSeamTrace(
      frameCount: number = DRIFT_CAMERA_PROBE_FRAME_COUNT,
    ): DriftSeamCameraSample[] {
      mockedAdvanceFrameSteps.mockReturnValue(1);
      getBootStart()('neatracing-output');
      window.dispatchEvent(
        new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true }),
      );
      window.dispatchEvent(
        new KeyboardEvent('keydown', { code: 'ArrowRight', cancelable: true }),
      );
      const samples: DriftSeamCameraSample[] = [];
      for (let frame = 0; frame < frameCount; frame += 1) {
        scheduledRafCallbacks[frame]?.(
          DRIVEN_FRAME_TIMESTAMP_MS + frame * SLIDING_SEAM_FRAME_STEP_MS,
        );
        const seam = readDriftSeam();
        samples.push({
          tick: readSeamNumber(seam, 'tick'),
          slipAngleRad: readSeamNumber(seam, 'slipAngleRad'),
          speedMps: readSeamNumber(seam, 'speedMps'),
          followOffsetRad: readSeamNumber(seam, 'followOffsetRad'),
        });
      }
      return samples;
    }

    /**
     * Replays the camera law from the seam's OWN published drift inputs.
     *
     * The law's divergence is `signedShortestDeltaRad(heading, travel)` and
     * the physics publishes `travelHeadingRad = wrap(heading + slipAngle)`,
     * so the divergence IS the published slip; the heading cancels out of
     * the shortest-delta, so the replay runs with a zero heading.
     *
     * @param samples - The per-frame seam samples.
     * @returns The replayed offset after every sample, in frame order.
     */
    function replayLawFromSeamTrace(
      samples: readonly DriftSeamCameraSample[],
    ): number[] {
      let follow = createChaseCameraFollowState();
      const offsetsRad: number[] = [];
      for (const sample of samples) {
        follow = stepChaseCameraFollow(follow, {
          ticks: Math.max(0, sample.tick - follow.lastTick),
          headingRad: 0,
          travelHeadingRad: sample.slipAngleRad,
          speedMps: sample.speedMps,
          tick: sample.tick,
        });
        offsetsRad.push(follow.offsetRad);
      }
      return offsetsRad;
    }

    // [RED-5-PUBLISHED-CAP]
    it('never publishes a follow offset above the recorded aim-offset cap', () => {
      // Arrange: drive a REAL slide and record the live seam per frame.
      const trace = driveSlidingSeamTrace();

      // Act: the worst-case published offset.
      let publishedPeakRad = 0;
      for (const sample of trace) {
        publishedPeakRad = Math.max(
          publishedPeakRad,
          Math.abs(sample.followOffsetRad),
        );
      }

      // GUARDS first: the probe MUST REALLY have slid (a probe that never
      // drifted would satisfy the cap clause vacuously) ...
      expect(publishedPeakRad).toBeGreaterThan(0);
      // ... and the reader MUST REALLY be live: the published offset MUST
      // CHANGE across the frames (a constant reader could never be caught
      // by the agreement clause).
      expect(trace[trace.length - 1].followOffsetRad).not.toBe(
        trace[0].followOffsetRad,
      );

      // THEN the relational clause: the published |offset| NEVER exceeds
      // the RECORDED cap. MEASURED today: the 240-frame probe publishes a
      // peak of 0.14962536105320104 rad (ledger row RP-2), which is BELOW
      // the cap - the cap is enforced AT THE SOURCE inside the camera
      // law, so EVERY reader can only ever publish the CAPPED offset.
      expect(publishedPeakRad).toBeLessThanOrEqual(
        RECORDED_MAX_AIM_OFFSET_RAD + CAMERA_RAD_TOLERANCE,
      );
    });

    // [RED-5-PUBLISHED-AGREES-WITH-THE-LAW]
    it('publishes exactly the follow offset the camera law computes from the seam drift inputs', () => {
      // Arrange
      const trace = driveSlidingSeamTrace();

      // Act: replay the law from the seam's OWN published drift inputs.
      const replayedOffsetsRad = replayLawFromSeamTrace(trace);
      let worstDifferenceRad = 0;
      for (let index = 0; index < trace.length; index += 1) {
        worstDifferenceRad = Math.max(
          worstDifferenceRad,
          Math.abs(trace[index].followOffsetRad - replayedOffsetsRad[index]),
        );
      }

      // GUARDS first: the seam MUST REALLY have published a slide (the
      // replay is driven by the published slip and speed, so a probe that
      // never drifted would replay 0) ...
      let publishedSlipPeakRad = 0;
      let publishedOffsetPeakRad = 0;
      for (const sample of trace) {
        publishedSlipPeakRad = Math.max(
          publishedSlipPeakRad,
          Math.abs(sample.slipAngleRad),
        );
        publishedOffsetPeakRad = Math.max(
          publishedOffsetPeakRad,
          Math.abs(sample.followOffsetRad),
        );
      }
      expect(publishedSlipPeakRad).toBeGreaterThan(0);
      expect(publishedOffsetPeakRad).toBeGreaterThan(0);

      // THEN the relational clause: EVERY published frame agrees with the
      // law's replay of the seam's own drift inputs. MEASURED today: the
      // worst difference over the probe is 4.163336342344337e-17 rad
      // (ledger row RP-5) - the reader is a READER, not a second law.
      expect(worstDifferenceRad).toBeLessThan(CAMERA_RAD_TOLERANCE);
    });
  });
});

/**
 * Step 14 (P1-boot-composition) boot composition contract.
 *
 * RED while `NEATRACING_BOOT_PROGRAM_REGISTRY` is still the empty
 * `readonly never[]` inside `browser-entry.ts` and the boot stub never
 * touches the HUD or the race mode selector: the warmup registry arg
 * carries none of the pinned program keys (empty array vs the nine pinned
 * keys), and both mount spies show the legible "received 0 calls" mismatch.
 *
 * These tests answer the AC-040 nine-program parity (batch 102 gap iii) and
 * the AC-024 (partial) reserved-container mounts (batch 94 handoff). They
 * pin the boot seam only; the live drive loop stays with P1-drive-loop-live.
 */
describe('Neat Racing browser entry boot composition (Step 14, P1-boot-composition)', () => {
  /** Live-branch fixture captured before each test re-stamps the doubles. */
  let liveFixture: LiveInitResultFixture;
  /** RAF callbacks captured from the boot loop; frames are not driven here. */
  let scheduledRafCallbacks: FrameRequestCallback[];

  beforeEach(() => {
    jest.resetAllMocks();

    // Live-branch doubles identical to the Step 13 wiring block: the boot
    // path sees a WebGL2 context so the warmup and mount seams stay on the
    // live branch.
    liveFixture = createLiveInitResultFixture();
    mockedInitRenderingContext.mockReturnValue(liveFixture.result);
    mockedRunShaderWarmup.mockReturnValue({
      warmedProgramCount: 0,
      offscreenDrawCount: 0,
    });
    mockedUpdateFullWindowRendererSize.mockReturnValue(
      createMockRendererSizeUpdate(),
    );
    mockedAdvanceFrameSteps.mockReturnValue(0);
    mockedCreateHud.mockReturnValue(createInertHud());
    mockedCreateRaceModeSelector.mockReturnValue(createInertRaceModeSelector());

    // Capture RAF callbacks so the boot loop never runs on jsdom timers.
    scheduledRafCallbacks = [];
    window.requestAnimationFrame = (callback: FrameRequestCallback): number => {
      scheduledRafCallbacks.push(callback);
      return scheduledRafCallbacks.length;
    };

    resetHostShell();
  });

  describe('AC-040 (completion): nine-program warmup registry parity (batch 148 OBS-1 / OBS-5)', () => {
    it('warms the registered programs in the pinned NEATRACING_WARMED_PROGRAM_KEYS order', () => {
      getBootStart()('neatracing-output');

      const registryArg = mockedRunShaderWarmup.mock.calls[0]?.[1] ?? [];

      expect(registryArg.map((program) => program.key)).toEqual(
        NEATRACING_WARMED_PROGRAM_KEYS,
      );
    });

    it('populates a registry whose length matches NEATRACING_WARMED_PROGRAM_COUNT', () => {
      getBootStart()('neatracing-output');

      const registryArg = mockedRunShaderWarmup.mock.calls[0]?.[1] ?? [];

      expect(registryArg.length).toBe(NEATRACING_WARMED_PROGRAM_COUNT);
    });
  });

  describe('AC-024 (partial): reserved neatracing-hud-output container mounts (batch 94 handoff)', () => {
    it('mounts the HUD into the reserved neatracing-hud-output container at boot', () => {
      getBootStart()('neatracing-output');

      expect(mockedCreateHud).toHaveBeenCalledWith(RESERVED_HUD_OUTPUT_ID);
    });

    it('mounts the race mode selector into the reserved neatracing-hud-output container at boot', () => {
      getBootStart()('neatracing-output');

      expect(mockedCreateRaceModeSelector).toHaveBeenCalledWith(
        RESERVED_HUD_OUTPUT_ID,
      );
    });
  });
});

/**
 * Phase-3 Step-01 (P3-S1-wiring) adaptive-quality controller wiring
 * contract.
 *
 * RED while `composerFrame.enabled` is a hardcoded `true` in the browser
 * entry's composer-frame construction and no controller tick exists, and
 * while frame-composer.ts holds the independent
 * `INITIAL_BAG_BLOOM_ENABLED = true` literal. The wiring doubles open the
 * composition gate (a fake GL that presents `createShader`) and stub the
 * eight pass factories + the composer seam with inert doubles (every
 * warmup helper stays real), then pin:
 *
 * - exactly ONE `tickAdaptiveQuality` per RENDERED frame - the probe runs
 *   on ZERO-step frames (the mocked worker advance releases no simulation
 *   steps) so "zero-step frames included" is measured, not assumed -
 *   AHEAD of the composer draw (invocationCallOrder);
 * - the stage-record-derived off-switch HANDED TO THE COMPOSED FRAME:
 *   stage 0 full-quality `true` and stage 1 bloom-off `false` once the
 *   controller drops;
 * - the defensively clamped negative-stage non-undefined knob;
 * - and (the readHostShell readFileSync precedent) the source pin on
 *   frame-composer.ts: the initial bloom-bag neutral must derive from the
 *   controller stage ladder, never from an independent `true` literal.
 */
describe('Neat Racing adaptive-quality controller wiring (Phase-3 Step-01, P3-S1-wiring)', () => {
  /**
   * Inert pass double consumed only by the composer teardown: the
   * composition stubs hand one to every pass factory. The composer seam
   * never draws through a pass handle in these tests, so `draw` and
   * `dispose` stay inert.
   */
  const inertPassDouble = {
    draw: (): void => undefined,
    dispose: (): void => undefined,
  };

  /** The composed frame capture seam installed by the composer double. */
  let seamDrawFrame: jest.Mock;
  /** RAF callbacks captured from the boot loop; tests drive them manually. */
  let scheduledRafCallbacks: FrameRequestCallback[];

  beforeEach(() => {
    jest.resetAllMocks();

    // Live-branch doubles PLUS the createShader capability so the
    // composition attempt is ENTERED (the gate stays closed for every
    // other describe's headless GL fixture, so the pass-factory doubles
    // below never leak into other suites).
    const compositionGl = {
      clear: (): void => undefined,
      clearColor: (): void => undefined,
      drawArrays: (): void => undefined,
      viewport: (): void => undefined,
      createShader: (): unknown => ({}),
    } as unknown as WebGL2RenderingContext;
    mockedInitRenderingContext.mockReturnValue({
      ok: true,
      gl: compositionGl,
      isContextLost: (): boolean => false,
    });

    mockedRunShaderWarmup.mockReturnValue({
      warmedProgramCount: 0,
      offscreenDrawCount: 0,
    });
    mockedUpdateFullWindowRendererSize.mockReturnValue(
      createMockRendererSizeUpdate(),
    );
    // ZERO-step frames: the (mocked) worker advance releases NO simulation
    // steps, yet every driven frame is still a RENDERED frame - the exact
    // "zero-step frames included" surface the packet pins.
    mockedAdvanceFrameSteps.mockReturnValue(0);
    mockedCreateHud.mockReturnValue(createInertHud());
    mockedCreateRaceModeSelector.mockReturnValue(createInertRaceModeSelector());

    // The composer seam double: createFrameComposer succeeds and hands
    // back a capture-only composer, so every composed frame is
    // observable.
    seamDrawFrame = jest.fn();
    mockedCreateFrameComposer.mockImplementation(() => ({
      drawFrame: seamDrawFrame,
      dispose: (): void => undefined,
    }));
    // The eight pass-factory seams return inert doubles (compose-time per
    // pass construction stays on the doubles, never on the fake GL).
    mockedCreateBloomPass.mockReturnValue(
      inertPassDouble as unknown as ReturnType<typeof createBloomPass>,
    );
    mockedCreateCarPass.mockReturnValue(
      inertPassDouble as unknown as ReturnType<typeof createCarPass>,
    );
    mockedCreateGlassPass.mockReturnValue(
      inertPassDouble as unknown as ReturnType<typeof createGlassPass>,
    );
    mockedCreateTrackPass.mockReturnValue(
      inertPassDouble as unknown as ReturnType<typeof createTrackPass>,
    );
    mockedCreateTrackGridPass.mockReturnValue(
      inertPassDouble as unknown as ReturnType<typeof createTrackGridPass>,
    );
    mockedCreateRightWorldPass.mockReturnValue(
      inertPassDouble as unknown as ReturnType<typeof createRightWorldPass>,
    );
    mockedCreatePulsePass.mockReturnValue(
      inertPassDouble as unknown as ReturnType<typeof createPulsePass>,
    );
    mockedCreateStarfieldPass.mockReturnValue(
      inertPassDouble as unknown as ReturnType<typeof createStarfieldPass>,
    );

    // Replace the jsdom RAF implementation with a capturing stub so the
    // boot loop's frames are driven manually instead of by timers.
    scheduledRafCallbacks = [];
    window.requestAnimationFrame = (callback: FrameRequestCallback): number => {
      scheduledRafCallbacks.push(callback);
      return scheduledRafCallbacks.length;
    };

    resetHostShell();
  });

  afterEach(() => {
    // The controller spies are per-test `jest.spyOn` handles: restore them
    // so every OTHER describe keeps the REAL controller (the module-scope
    // jest.fn doubles above are re-stubbed by the next beforeEach).
    jest.restoreAllMocks();
  });

  /**
   * Drives `frameCount` rendered frames through the captured RAF chain,
   * each at the pinned 60 Hz timestamp step from the base timestamp.
   */
  function driveWiringFrames(frameCount: number): void {
    for (let frame = 0; frame < frameCount; frame += 1) {
      scheduledRafCallbacks[frame]?.(
        DRIVEN_FRAME_TIMESTAMP_MS + frame * WIRING_FRAME_STEP_MS,
      );
    }
  }

  /**
   * The composed frame handed to the composer draw seam at `frameIndex`.
   *
   * Throws a legible error when the frame is missing, so a wiring failure
   * never "passes" on an undefined knob.
   */
  function readDrawnFrameAt(frameIndex: number): NeatRacingFrameComposerFrame {
    const frame = (seamDrawFrame.mock.calls.at(frameIndex)?.[0] ??
      undefined) as NeatRacingFrameComposerFrame | undefined;
    if (frame === undefined) {
      throw new Error(`no composed frame was drawn at index ${frameIndex}`);
    }
    return frame;
  }

  it('ticks the adaptive-quality controller exactly once per rendered frame, ahead of the composer draw', () => {
    const tickSpy = jest.spyOn(adaptiveQuality, 'tickAdaptiveQuality');
    getBootStart()('neatracing-output');
    driveWiringFrames(3);

    expect(tickSpy).toHaveBeenCalledTimes(3);
    expect(seamDrawFrame).toHaveBeenCalledTimes(3);
    for (let frame = 0; frame < 3; frame += 1) {
      expect(tickSpy.mock.invocationCallOrder[frame]).toBeLessThan(
        seamDrawFrame.mock.invocationCallOrder[frame],
      );
    }
  });

  it('still ticks exactly once per rendered frame when the timestamp does not advance between frames', () => {
    const tickSpy = jest.spyOn(adaptiveQuality, 'tickAdaptiveQuality');
    getBootStart()('neatracing-output');
    scheduledRafCallbacks[0]?.(DRIVEN_FRAME_TIMESTAMP_MS);
    scheduledRafCallbacks[1]?.(DRIVEN_FRAME_TIMESTAMP_MS);

    expect(tickSpy).toHaveBeenCalledTimes(2);
    expect(seamDrawFrame).toHaveBeenCalledTimes(2);
  });

  it('hands the full-quality stage record knob to the composed frame while the controller holds stage 0', () => {
    const tickSpy = jest.spyOn(adaptiveQuality, 'tickAdaptiveQuality');
    tickSpy.mockReturnValue(0);
    getBootStart()('neatracing-output');
    driveWiringFrames(1);

    expect(tickSpy).toHaveBeenCalledTimes(1);
    expect(readDrawnFrameAt(0).enabled).toBe(
      NEATRACING_ADAPTIVE_QUALITY_STAGES[0].bloomEnabled,
    );
  });

  it('hands the bloom-off stage record knob to the composed frame once the controller drops to stage 1', () => {
    const tickSpy = jest.spyOn(adaptiveQuality, 'tickAdaptiveQuality');
    tickSpy.mockReturnValueOnce(0).mockReturnValueOnce(1);
    getBootStart()('neatracing-output');
    // The composer-frame bag is IDENTITY-STABLE (the zero per-frame
    // allocation rule): the off-switch is stamped IN PLACE per rendered
    // frame, so each frame's knob is read by value BEFORE the next frame
    // refreshes the same bag.
    driveWiringFrames(1);
    expect(readDrawnFrameAt(0).enabled).toBe(
      NEATRACING_ADAPTIVE_QUALITY_STAGES[0].bloomEnabled,
    );

    scheduledRafCallbacks[1]?.(
      DRIVEN_FRAME_TIMESTAMP_MS + WIRING_FRAME_STEP_MS,
    );
    expect(readDrawnFrameAt(1).enabled).toBe(
      NEATRACING_ADAPTIVE_QUALITY_STAGES[1].bloomEnabled,
    );
  });

  it('keeps the composed frame knob defined when the controller stage resolves from a negative stage', () => {
    const tickSpy = jest.spyOn(adaptiveQuality, 'tickAdaptiveQuality');
    tickSpy.mockReturnValue(-1);
    getBootStart()('neatracing-output');
    driveWiringFrames(1);

    expect(tickSpy).toHaveBeenCalledTimes(1);
    expect(readDrawnFrameAt(0).enabled).toBeDefined();
    expect(readDrawnFrameAt(0).enabled).toBe(
      NEATRACING_ADAPTIVE_QUALITY_STAGES[0].bloomEnabled,
    );
  });

  it('derives the frame-composer initial bloom-bag neutral from the adaptive-quality stage ladder (source pin)', () => {
    const composerSource = readFrameComposerSource();

    expect(composerSource).not.toMatch(
      /INITIAL_BAG_BLOOM_ENABLED\s*(:\s*boolean\s*)?=\s*true/,
    );
    expect(composerSource).toMatch(/NEATRACING_ADAPTIVE_QUALITY_STAGES/);
  });
});
