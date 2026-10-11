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

import { beforeEach, describe, expect, it } from '@jest/globals';
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

/** Timestamp handed to the captured RAF callback when tests drive frame one. */
const DRIVEN_FRAME_TIMESTAMP_MS = 0;

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

  describe('AC-040 (completion): nine-program warmup registry parity (batch 102 gap iii)', () => {
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
