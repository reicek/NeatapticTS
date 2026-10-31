/// <reference lib="dom" />

/**
 * Browser entrypoint (boot glue) for the Neat Racing voxel racer demo.
 *
 * Step 13 (P1-boot-glue) wires the live render path: the boot function mounts
 * the game canvas, acquires the WebGL2 context through the engine context
 * service, warms the registered shader programs, and applies the fixed-canvas
 * letterbox strategy. Step 14 (P1-boot-composition) completed the boot shell:
 * the warmup registry is assembled per boot in the pinned
 * `NEATRACING_WARMED_PROGRAM_KEYS` order and the HUD overlay plus the race
 * mode selector are mounted into the shell's reserved `neatracing-hud-output`
 * container. Step 14 (P1-drive-loop-live) now drives the live drive loop:
 * keyboard input is tracked on the window, the composer scene is built
 * synchronously from the static car/wheel voxel-model JSON assets (degrading
 * to an inert composer when the context lacks program support or composition
 * throws), and every RAF frame is handed to the session-owned fixed-timestep
 * drive (`physics → rules → HUD → composer`; one request at boot plus one
 * re-request per frame). When WebGL2 is unavailable, boot degrades gracefully by
 * appending the service-built fallback notice into the host
 * container instead of throwing (editor-entry precedent), so the demo page
 * stays readable and the stop handle stays functional.
 *
 * The host shell (`index.html`) loads the host bundle per the flagship demo
 * contract and calls `window.neatRacingStart(outputId)`; keeping the global
 * install here (rather than in the shell script) means the page never needs
 * to know how the canvas is created or torn down.
 *
 * @module
 */

import {
  initRenderingContext,
  runShaderWarmup,
} from '../engine/context.service';
import type {
  RenderingContextInitResult,
  ShaderWarmupProgram,
} from '../engine/context.service';
import {
  toBloomWarmupPrograms,
  createBloomPass,
} from '../engine/passes/bloom.pass';
import { toCarWarmupProgram, createCarPass } from '../engine/passes/car.pass';
import {
  toGlassWarmupProgram,
  createGlassPass,
} from '../engine/passes/glass.pass';
import {
  toTrackWarmupProgram,
  createTrackPass,
} from '../engine/passes/track.pass';
import {
  createTrackGridPass,
  toTrackGridWarmupProgram,
} from '../track/grid.pass';
import { createRightWorldPass } from '../track/right-world.pass';
import type { RightWorldPassDrawState } from '../track/right-world.pass';
import {
  collectActivePulses,
  createPulsePass,
  NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
  toPulseWarmupProgram,
} from '../engine/passes/pulse.pass';
import type { AmbientPulse } from '../engine/passes/pulse.pass';
import {
  toStarfieldWarmupProgram,
  createStarfieldPass,
  type StarfieldPass,
} from '../engine/passes/starfield.pass';
import { createFrameComposer } from '../host/frame-composer';
import { computeVisualWheelSpinDeltaRad } from '../voxel/wheel.mount.utils';
import { fillCarRenderPose } from '../car/generate-car';
import {
  createChaseCameraFollowState,
  resolveChaseCameraAimRad,
  resolveChaseCameraBasis,
  stepChaseCameraFollow,
} from './chase-camera';
import type {
  ChaseCameraBasis,
  ChaseCameraFollowState,
  ChaseCameraFollowStep,
} from './chase-camera';
import type { NeatRacingFrameComposerFrame } from '../host/frame-composer';
import type { NeatRacingHudState } from '../host/hud';
import { createHud } from '../host/hud';
import { sampleKeyboardInput } from '../host/input';
import type { NeatRacingRaceMode } from '../host/mode-selector';
import { createRaceModeSelector } from '../host/mode-selector';
import { createRaceSession } from '../host/race-session';
import type { NeatRacingRaceSession } from '../host/race-session';
import { NEATRACING_FIXED_TIMESTEP_SECONDS } from '../host/render-loop.utils';
import { createTrackSurfaceResolver } from '../physics/collision.utils';
import { stepRaceRules } from '../physics/race-rules';
import { stepAutopilotInput } from '../physics/ai.autopilot';
import {
  createInitialPhysicsCarState,
  stepPhysics,
} from '../physics/physics.step.service';
import type { PhysicsCarSpawnPose } from '../physics/physics.step.service';
import type { VoxelBinModel } from '../voxel/bin.codec';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../voxel/lod.distance.utils';
import { meshVoxelModel } from '../voxel/greedy-mesher';
import { decodeVoxelModel } from '../voxel/model.decode.utils';
import { validateVoxelModelDocument } from '../voxel/model.schema';
import { NEATRACING_TEAM_ID_RED } from '../voxel/team-remap';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type { TrackSpec } from '../../racing_curriculum/track/track.generator.types';
import carModelDocument from '../assets/models/car-f1.model.json';
import wheelModelDocument from '../assets/models/wheel.model.json';
import {
  applyFullWindowStretchStyle,
  logRendererSizeAnnotation,
  resolveCanvasBackingDimensions,
  updateFullWindowRendererSize,
} from './canvas-dimensions.utils';

/**
 * Teardown handle returned by the boot stub.
 *
 * Calling it stops the render loop, removes the resize listener and any
 * appended fallback notice, and removes the mounted game canvas so an
 * embedding host can unmount the demo cleanly (used by tests and future
 * hot-reload paths).
 */
export type NeatRacingStop = () => void;

/**
 * Boot function installed on `window` by the host bundle.
 *
 * @param outputId - Host container element id that will own the game canvas.
 * @returns A stop handle that removes the mounted canvas.
 */
export type NeatRacingStart = (outputId: string) => NeatRacingStop;

/**
 * The 14N drift debug seam: the read-only, EXACTLY eight-field view of the
 * live drift channel that the browser half measures. It is a debug
 * surface, not a gameplay API.
 */
interface NeatRacingDriftState {
  /**
   * The chase-camera follow offset (rad): how far the camera aim is
   * rotated off the nose heading, toward the PUBLISHED travel direction.
   *
   * Step 14P: the SIXTH seam field, added together with the camera
   * re-wiring. It is the camera's own share of the shared slip scalar, so
   * the browser half can measure how far the 3D view actually swung and
   * compare it with the published slip.
   */
  readonly followOffsetRad: number;
  /** The visible body yaw (rad) - the pose the renderer draws. */
  readonly poseRad: number;
  /**
   * The PUBLISHED pose-spring rate (rad/s): how fast the visible body
   * yaw is moving at this tick.
   *
   * Step 14R (AC-088): the EIGHTH seam field, added together with the
   * drift-presentation car pass. The value is READ from the physics state
   * (`car.drift.pose.poseRate`) and is NEVER re-derived in the browser.
   *
   * It is a NAMED channel and it is NOT the traction severity (that is the
   * SEVENTH field): the pose rate is the crab's own rate, the exact
   * per-tick published quantity the browser half would otherwise have to
   * ESTIMATE by differencing the pose across rendered frames - and the
   * browser half is render-rate dependent, so a frame difference measures
   * the camera, not the car. With the published rate the fixture can
   * measure the crab's ring-down and its far-side pop DIRECTLY, in
   * simulation time, next to the pose it belongs to.
   */
  readonly poseRateRadPerS: number;
  /** The published physics slip angle (rad). */
  readonly slipAngleRad: number;
  /** The longitudinal speed (m/s). */
  readonly speedMps: number;
  /** The fixed simulation tick rate (Hz). */
  readonly tickHz: number;
  /**
   * The deterministic simulation tick counter (`car.tick`).
   *
   * 14N W3: the FIFTH seam field, added together with the fixture reader
   * and the fixture-gate document. The browser half is render-rate
   * dependent (the host releases 0-4 simulation ticks per rendered
   * frame), so a FRAME count can never address a simulation instant: a
   * 600-tick phase spans about 1622 frames at 163 fps and about 330
   * frames at 30 fps. `car.tick` is the deterministic 60 Hz clock the
   * production already owns, so the browser half can budget its phases
   * in TICKS and compute a frame-rate-independent rate (deg per tick
   * delta) instead of a per-frame step.
   */
  readonly tick: number;
  /**
   * The PUBLISHED traction severity: how much traction the car is
   * spending, in the closed unit interval `[0, 1]`.
   *
   * Step 14Q2 (AC-088): the SEVENTH seam field, added together with the
   * published state field (the slice `P1-drift-traction-scalar`). The
   * value is READ from the physics state (`car.tractionSeverity`) and is
   * NEVER re-derived in the browser, so the browser half reads the SAME
   * number the tires, the HUD and the car pass read.
   */
  readonly tractionSeverity: number;
}

/**
 * The mutable drift-seam BAG: the SAME field set as the published view
 * with the `readonly` modifiers stripped, so the frame builder can
 * refresh the seam IN PLACE.
 *
 * Step 14Q2 (AC-088): the zero per-frame allocation rule holds - the seam
 * object is a STABLE BAG, not a fresh object per frame. ONE bag is born
 * per boot and the window handle keeps pointing at the SAME object, so
 * this seam growth (and the next one) adds a field WITHOUT adding a
 * per-frame allocation.
 */
type NeatRacingDriftStateBag = {
  -readonly [Key in keyof NeatRacingDriftState]: NeatRacingDriftState[Key];
};

/** Element id given to the game canvas mounted by the boot stub. */
const NEATRACING_CANVAS_ID = 'neatracing-canvas';

/**
 * Reserved host shell id that the HUD overlay and the race mode selector
 * mount into at boot (AC-024 partial; batch 94 handoff). The shell reserves
 * the element in `index.html`; the mount factories resolve-or-throw against
 * it, so the boot path owns only the ordering.
 */
const NEATRACING_HUD_OUTPUT_ID = 'neatracing-hud-output';

/**
 * Boot-session teardown bookkeeping (frame loop, resize listener, fallback
 * notice, stopped flag) shared by the wired boot surfaces.
 */
interface BootSession {
  /** True once the stop handle has torn the session down. */
  stopped: boolean;
  /** Handle of the most recently scheduled animation frame, or null. */
  frameHandle: number | null;
  /** Resize listener installed at boot, or null when never installed. */
  resizeListener: EventListener | null;
  /** Fallback notice appended at boot, or null on the live path. */
  fallbackNotice: HTMLElement | null;
  /** Keyboard-down listener installed by the drive loop, or null. */
  keydownListener: ((event: KeyboardEvent) => void) | null;
  /** Keyboard-up listener installed by the drive loop, or null. */
  keyupListener: ((event: KeyboardEvent) => void) | null;
  /** Composer + pass teardown installed by the drive loop, or null. */
  driveTeardown: (() => void) | null;
  /** Starfield pass updated on resize by the boot listener, or null. */
  starfieldPass: StarfieldPass | null;
}

/**
 * Mount the game canvas into the host shell, wire the live render path, and
 * return a stop handle.
 *
 * The mount → overlays → init → warmup → letterbox → RAF loop order is the
 * AC-002 boot contract: the canvas is mounted first, the HUD overlay and the
 * race mode selector are mounted into the reserved shell container (AC-024
 * partial), the context is acquired exactly once for it, the shader warmup
 * runs against the per-boot program registry before the first frame, the
 * letterbox strategy is applied from the live viewport, and the
 * fixed-timestep accumulator is advanced from `requestAnimationFrame` (one
 * request at boot, one re-request per tick). Without WebGL2 the boot degrades
 * gracefully by appending the service-built fallback notice (editor-entry
 * precedent) while returning a functional stop handle.
 *
 * @param outputId - Host container element id that will own the game canvas.
 * @returns A stop handle that removes the mounted canvas.
 * @throws When no element with `outputId` exists in the document.
 *
 * @example
 * ```ts
 * const stop = window.neatRacingStart('neatracing-output');
 * stop(); // unmounts the demo canvas
 * ```
 */
function neatRacingStart(outputId: string): NeatRacingStop {
  const output = getOutputContainer(outputId);
  const canvas = mountGameCanvas(output);

  // Step 1: Mount the boot shell overlays — HUD first, then the race mode
  // selector — into the reserved neatracing-hud-output container (AC-024
  // partial; batch 94 handoff). The factories resolve-or-throw against the
  // shell and own their own DOM wiring; the returned controllers feed the
  // live drive loop (per-drawn-frame hud.update + the restart switch carrier).
  const overlays = createBootShellOverlays();

  // Step 2: Acquire the WebGL2 context for the freshly mounted canvas exactly
  // once (AC-002). The optional-chain tolerance keeps an absent init result
  // (stubbed service doubles) on the fallback branch instead of throwing.
  const init: RenderingContextInitResult | undefined = initRenderingContext({
    canvas,
    documentRef: document,
  });
  const gl = init?.ok ? init.gl : null;
  const session = createBootSession();

  // Step 3: Degrade gracefully without WebGL2 — append the service-built
  // fallback notice (when the page has a DOM shell) and keep the stop handle.
  if (gl === null) {
    return finishFallbackBoot(
      output,
      canvas,
      session,
      init?.fallbackElement ?? null,
    );
  }

  // Step 4: Warm the pinned nine-program registry once before frame 1
  // (AC-040, P8) so the first visible frame carries no shader-compilation
  // hitch. The registry is assembled per boot against this context in the
  // pinned NEATRACING_WARMED_PROGRAM_KEYS order.
  runShaderWarmup(gl, createBootProgramRegistry(gl));

  // Step 5: Apply the fixed-canvas letterbox strategy from the live viewport
  // (spec §6.2) before the first frame is scheduled.
  applyRendererSize(canvas);

  // Step 6: Build the live drive loop — decode the static voxel assets,
  // assemble the composer passes once (degrading to an inert composer when
  // the context lacks program support or composition throws), and start the
  // RAF loop. The session owns the fixed-timestep accumulator: each frame
  // advances it (physics → rules → HUD → composer). One request at boot,
  // one re-request per frame (AC-003 cadence).
  const driveSession = startLiveDriveLoop(gl, session, overlays, canvas);
  startRenderLoop(session, driveSession);

  // Step 7: Re-apply the letterbox strategy directly (non-debounced) on every
  // viewport resize (spec §6.2), and keep the starfield NDC mapping in sync
  // with the computed backing width (OBS-F resize awareness).
  attachResizeListener(canvas, session, session.starfieldPass);

  return createStopHandle(canvas, session);
}

/**
 * Resolves the host output container, throwing a descriptive error when the
 * shell does not provide it (mounted-canvas contract, AC-001).
 *
 * @param outputId - Host container element id to resolve.
 * @returns The resolved host container.
 * @throws When no element with `outputId` exists in the document.
 */
function getOutputContainer(outputId: string): HTMLElement {
  const output = document.getElementById(outputId);
  if (!(output instanceof HTMLElement)) {
    throw new Error(`Output container #${outputId} not found`);
  }
  return output;
}

/**
 * Creates the game canvas and mounts it into the host container.
 *
 * The canvas is mounted before context acquisition so the service memoizes
 * the WebGL2 context against the actual mounted element.
 *
 * @param output - Host container that will own the canvas.
 * @returns The freshly mounted game canvas.
 */
function mountGameCanvas(output: HTMLElement): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.id = NEATRACING_CANVAS_ID;
  output.appendChild(canvas);
  return canvas;
}

/**
 * Mounts the HUD overlay and the race mode selector into the reserved host
 * container at boot (AC-024 partial; batch 94 handoff) and returns the
 * controllers.
 *
 * The HUD mounts first, then the race mode selector, mirroring the AC-024
 * ordering (read model before control surface). The factories resolve the
 * reserved {@link NEATRACING_HUD_OUTPUT_ID} container themselves and throw
 * when the shell omits it (index.html owns the reservation), so the boot path
 * owns only the ordering. Stubbed service doubles may mount nothing; the
 * drive loop stays wired on inert fallback controllers in that case
 * (P1-drive-loop-live: AC-024 contract requires the wiring chain to survive
 * undefined controllers).
 *
 * @returns The mounted HUD and race-mode-selector controllers (inert
 *   fallback controllers when a mount factory yields nothing).
 */
function createBootShellOverlays(): BootShellOverlays {
  return {
    hud: createHud(NEATRACING_HUD_OUTPUT_ID) ?? createHudFallback(),
    selector:
      createRaceModeSelector(NEATRACING_HUD_OUTPUT_ID) ??
      createSelectorFallback(),
  };
}

/**
 * Boot-mounted overlay controllers handed to the live drive loop.
 */
interface BootShellOverlays {
  /** HUD controller (AC-024 read model) updated once per drawn frame. */
  hud: { update(state: NeatRacingHudState): void };
  /** Race mode selector (AC-024 control surface) driving restarts. */
  selector: {
    /** Currently active race mode. */
    readonly mode: NeatRacingRaceMode;
    /** Registers a switch listener (no unsubscribe in phase 1). */
    onSwitch(listener: (mode: NeatRacingRaceMode) => void): void;
  };
}

/** Race mode carried by the inert selector fallback (manual play). */
const NEATRACING_DEFAULT_RACE_MODE: NeatRacingRaceMode = 'manual';

/**
 * Inert HUD fallback for stubbed mount surfaces (no-op updates).
 *
 * @returns An HUD-compatible controller whose updates do nothing.
 */
function createHudFallback(): BootShellOverlays['hud'] {
  return { update: (): void => undefined };
}

/**
 * Inert selector fallback for stubbed mount surfaces (manual mode).
 *
 * @returns A selector-compatible controller with a no-op switch carrier.
 */
function createSelectorFallback(): BootShellOverlays['selector'] {
  return {
    mode: NEATRACING_DEFAULT_RACE_MODE,
    onSwitch: (): void => undefined,
  };
}

/**
 * Creates the boot-session teardown bookkeeping with every surface unset.
 *
 * @returns A fresh boot session for the current boot call.
 */
function createBootSession(): BootSession {
  return {
    stopped: false,
    frameHandle: null,
    resizeListener: null,
    fallbackNotice: null,
    keydownListener: null,
    keyupListener: null,
    driveTeardown: null,
    starfieldPass: null,
  };
}

/**
 * Completes the degraded boot when WebGL2 is unavailable.
 *
 * Appends the service-built fallback notice into the boot container (the
 * editor-entry precedent for a reduced view) and returns the same stop-handle
 * contract so an embedding host can always unmount the boot cleanly. Rendering
 * is never started on this path: there is no live context to draw into.
 *
 * @param output - Host container owning the boot content.
 * @param canvas - Mounted game canvas removed by the stop handle.
 * @param session - Boot session tracking the notice for symmetric teardown.
 * @param fallbackNotice - Service-built notice element, or null when the
 *   init result carried no DOM notice (headless hosts).
 * @returns A stop handle that removes the mounted boot content.
 */
function finishFallbackBoot(
  output: HTMLElement,
  canvas: HTMLCanvasElement,
  session: BootSession,
  fallbackNotice: HTMLElement | null,
): NeatRacingStop {
  if (fallbackNotice !== null) {
    output.appendChild(fallbackNotice);
    session.fallbackNotice = fallbackNotice;
  }
  return createStopHandle(canvas, session);
}

/**
 * Builds the per-boot warmup registry in the pinned AC-040 key order:
 * starfield, grid, pulse, track, car, glass, bloom-threshold, bloom-blur,
 * bloom-additive (`NEATRACING_WARMED_PROGRAM_KEYS`).
 *
 * The legacy 1 m² engine grid pass has been replaced by the 2 m² track-grid
 * pass (AC-045). The `grid` registry slot now warms the track-grid shader so
 * the canonical key count stays unchanged and no dual-path draw code remains.
 *
 * @param gl - Live WebGL2 context the warmup programs compile against.
 * @returns The warmup programs in the pinned registry-key order, keyed so
 *   the `grid` slot compiles the track-grid program.
 */
function createBootProgramRegistry(
  gl: WebGL2RenderingContext,
): readonly ShaderWarmupProgram[] {
  const [bloomThreshold, bloomBlur, bloomAdditive] = toBloomWarmupPrograms(gl);
  const trackGridWarmup = toTrackGridWarmupProgram(gl);
  return [
    toStarfieldWarmupProgram(gl),
    {
      key: 'grid',
      compileAndLink: trackGridWarmup.compileAndLink,
      drawOffscreen: trackGridWarmup.drawOffscreen,
    },
    toPulseWarmupProgram(gl),
    toTrackWarmupProgram(gl),
    toCarWarmupProgram(gl),
    toGlassWarmupProgram(gl),
    bloomThreshold,
    bloomBlur,
    bloomAdditive,
  ];
}

/**
 * Resolves the computed backing dimensions for the starfield pass.
 *
 * Uses the letterbox-strategy resolver from canvas-dimensions.utils so the
 * pass maps stars across the intended fixed-render-width backing store rather
 * than whatever the raw canvas.width happens to be at construction time
 * (OBS-F). This is also the source of truth for the resize listener.
 *
 * @param canvas - Mounted game canvas (or structural stub).
 * @returns Computed backing-store dimensions in pixels.
 */
function resolveStarfieldBackingDimensions(
  canvas: HTMLCanvasElement,
): { width: number; height: number } {
  return resolveCanvasBackingDimensions({
    clientWidth: canvas.clientWidth,
    clientHeight: canvas.clientHeight,
    viewportWidth: window.innerWidth,
    viewportHeight: window.innerHeight,
  });
}

/**
 * Applies the fixed-canvas letterbox strategy from the live viewport once.
 *
 * Resizes the canvas backing store, positions the CSS display box with the
 * matching letterbox layout, and records a window-size annotation so the
 * next browser smoke test can cross-check the full-page render box.
 *
 * @param canvas - Canvas whose backing store and CSS box are resized.
 */
function applyRendererSize(canvas: HTMLCanvasElement): void {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  applyFullWindowStretchStyle(canvas);
  const dimensions = updateFullWindowRendererSize({
    canvas,
    viewportWidth,
    viewportHeight,
  });
  logRendererSizeAnnotation(
    canvas,
    dimensions,
    {
      width: dimensions.width,
      height: dimensions.height,
      offsetX: 0,
      offsetY: 0,
    },
    viewportWidth,
    viewportHeight,
  );
}

/**
 * Static demo-track fixture (pinned seed/layout so the generated loop is
 * deterministic per boot). Regenerating layouts per boot is out of scope for
 * phase 1; the recycled generator output feeds both the physics surface
 * resolver and the lap-line detector.
 */
const NEATRACING_DEMO_TRACK_GENERATION = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
} as const;

/**
 * Chase-camera pitch (radians) fed to the starfield camera pose.
 *
 * The chase GEOMETRY (the eye distance, the height and the look-ahead) is
 * owned by the shared chase-camera module (`./chase-camera`), so the camera
 * placement, the look target and the camera forward can never be derived
 * from two different sets of constants.
 */
const NEATRACING_CHASE_CAMERA_PITCH_RADIANS = 0.12;
/** Chase-camera vertical field of view (radians). */
const NEATRACING_CHASE_CAMERA_FOV_RADIANS = Math.PI / 3;
/**
 * NDC y coordinate of the visible horizon for the starfield pass.
 *
 * Derived from the fixed chase-camera pitch and vertical FOV so the starfield
 * tile ends exactly at the geometric horizon and stars never render below the
 * ground/grid region (OBS-E).
 */
const NEATRACING_STARFIELD_HORIZON_NDC_Y =
  Math.tan(NEATRACING_CHASE_CAMERA_PITCH_RADIANS) /
  Math.tan(NEATRACING_CHASE_CAMERA_FOV_RADIANS / 2);
/** Chase-camera near clip plane (world units). */
const NEATRACING_CHASE_CAMERA_NEAR = 1;
/** Chase-camera far clip plane (world units). */
const NEATRACING_CHASE_CAMERA_FAR = 3000;
/** Wheel spin fallback radius (world units) when the wheel asset is absent. */
const NEATRACING_WHEEL_FALLBACK_RADIUS = 8;

/**
 * Scene handedness (Step 14B round-3): the physics integrator works in the
 * top-down math plane where positive heading rotates counter-clockwise, and
 * the WebGL2 camera reads left-handed screen axes from `lookAtInto` (with the
 * camera looking along scene +x at yaw 0, screen right is scene +z). Mirroring
 * the top-down y into scene z (`worldZ = -y`) aligns the two: positive heading
 * tips the car nose toward scene -z, which the chase camera reads as screen
 * LEFT, matching the A key. The yaw offset then rotates the car model's +z
 * nose (per the wheel mount car-frame contract) so the nose points along the
 * mirrored scene forward `(cos heading, 0, -sin heading)` instead of the raw
 * mirrored top-down forward.
 */
export const NEATRACING_CAR_SCENE_YAW_OFFSET_RAD = Math.PI / 2;

/**
 * Maps a top-down math-plane y coordinate into the mirrored scene z axis
 * (`worldZ = -y`), the single ground-plane handedness seam for the frame
 * composer's car pose, camera eye/target, and baked track slabs.
 *
 * @param topDownY - Physics-plane y coordinate in meters.
 * @returns Scene z coordinate in meters.
 */
export function carSceneWorldZ(topDownY: number): number {
  // Normalize the signed zero so mapping the world origin yields +0.
  return topDownY === 0 ? 0 : -topDownY;
}

/**
 * Maps a top-down math heading into the mirrored scene's car yaw so the
 * model +z nose points along the mirrored scene forward.
 *
 * @param topDownHeadingRadians - Physics-plane heading (0 = +x, CCW positive).
 * @returns Scene yaw fed to the car body matrix and camera pose.
 */
export function carSceneYawRad(topDownHeadingRadians: number): number {
  return NEATRACING_CAR_SCENE_YAW_OFFSET_RAD + topDownHeadingRadians;
}

/**
 * Writes a right-handed perspective projection (column-major, WebGL clip
 * conventions) into `out` in place.
 *
 * @param out - 16-float matrix destination (overwritten).
 * @param fovYRadians - Vertical field of view in radians.
 * @param aspect - Width/height aspect ratio.
 * @param near - Near clip distance.
 * @param far - Far clip distance.
 */
export function perspectiveInto(
  out: Float32Array,
  fovYRadians: number,
  aspect: number,
  near: number,
  far: number,
): void {
  const focal = 1 / Math.tan(fovYRadians / 2);
  const range = near - far;
  out.fill(0);
  out[0] = focal / aspect;
  out[5] = focal;
  out[10] = (far + near) / range;
  out[11] = -1;
  out[14] = (2 * far * near) / range;
}

/**
 * Writes a right-handed look-at view matrix (column-major) into `out` with a
 * world-up vector of `(0, 1, 0)`.
 *
 * @param out - 16-float matrix destination (overwritten).
 * @param eyeX - Camera eye X.
 * @param eyeY - Camera eye Y.
 * @param eyeZ - Camera eye Z.
 * @param targetX - Look target X.
 * @param targetY - Look target Y.
 * @param targetZ - Look target Z.
 */
export function lookAtInto(
  out: Float32Array,
  eyeX: number,
  eyeY: number,
  eyeZ: number,
  targetX: number,
  targetY: number,
  targetZ: number,
): void {
  // zAxis = normalize(eye - target) (points from target toward the eye).
  const zx = eyeX - targetX;
  const zy = eyeY - targetY;
  const zz = eyeZ - targetZ;
  const zLength = Math.hypot(zx, zy, zz) || 1;
  const z0 = zx / zLength;
  const z1 = zy / zLength;
  const z2 = zz / zLength;
  // xAxis = normalize(cross(worldUp, zAxis)) with worldUp = (0, 1, 0).
  const xAxisLength = Math.hypot(z2, z0) || 1;
  const x0 = z2 / xAxisLength;
  const x2 = -z0 / xAxisLength;
  // yAxis = cross(zAxis, xAxis).
  const y0 = z1 * x2;
  const y1 = z2 * x0 - z0 * x2;
  const y2 = -z1 * x0;
  // Build the view matrix: rows are the camera axes in world space, so the
  // rotation part is the transpose of the [right, up, back] basis matrix.
  out[0] = x0;
  out[1] = y0;
  out[2] = z0;
  out[3] = 0;
  out[4] = 0;
  out[5] = y1;
  out[6] = z1;
  out[7] = 0;
  out[8] = x2;
  out[9] = y2;
  out[10] = z2;
  out[11] = 0;
  out[12] = -(x0 * eyeX + x2 * eyeZ);
  out[13] = -(y0 * eyeX + y1 * eyeY + y2 * eyeZ);
  out[14] = -(z0 * eyeX + z1 * eyeY + z2 * eyeZ);
  out[15] = 1;
}

/**
 * Writes `projection * view` (column-major) into `out`. All three matrices
 * are 16-float column-major; `out` must be distinct from the inputs.
 *
 * The GLSL world passes consume `uViewProjection * worldPosition`, so the
 * combined matrix must apply the projection after the view transform. This
 * helper is intentionally named with the operands `(out, view, projection)`
 * to match the call-site wording while computing `projection * view`.
 *
 * @param out - 16-float matrix destination.
 * @param view - View matrix.
 * @param projection - Projection matrix.
 */
export function multiplyViewProjectionInto(
  out: Float32Array,
  view: Float32Array,
  projection: Float32Array,
): void {
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) {
        sum += projection[k * 4 + row] * view[column * 4 + k];
      }
      out[column * 4 + row] = sum;
    }
  }
}

/**
 * Builds the live drive session and starts the RAF-driven drive: keyboard
 * presses are tracked on the window, the composer scene is assembled once
 * (decode + mesh the static voxel assets, per-pass program construction), and
 * every frame runs through the session-owned fixed-timestep drive
 * (`physics → rules → HUD → composer`).
 *
 * Composition is best-effort within a live render path: when the context
 * cannot compile programs (fake/headless GL surfaces) or the scene assembly
 * throws, the session keeps running against an inert composer that swallows
 * draw frames, so the drive loop (input, physics, rules, HUD) stays live
 * without render-path noise. The composer and its passes are disposed by the
 * boot stop handle through the session teardown hook.
 *
 * The lap-line detector reads signed sides of segment 0's chord in the ground
 * plane (generator x maps to world x; generator y maps to world z) and
 * reports any strict sign flip as a crossing; direction filtering and
 * checkpoint ordering are carried gaps for later slices. The detector
 * baseline re-binds on every restart via a second boot-owned switch
 * subscription (the session itself still subscribes exactly one restart
 * listener at creation, preserving its contract).
 *
 * @param gl - Live WebGL2 context for the mounted canvas.
 * @param session - Boot session receiving teardown bookkeeping.
 * @param overlays - Boot-mounted HUD + race mode selector controllers.
 * @param canvas - Mounted game canvas (aspect source for the camera).
 * @returns The drive session handed to the RAF loop.
 */
function startLiveDriveLoop(
  gl: WebGL2RenderingContext,
  session: BootSession,
  overlays: BootShellOverlays,
  canvas: HTMLCanvasElement,
) {
  // Track geometry: the recycled generator with the pinned demo fixture; the
  // resolver feeds the physics surface band, segments[0] feeds the lap line.
  const trackSpec: TrackSpec = generateTrack(NEATRACING_DEMO_TRACK_GENERATION);
  const surfaceResolver = createTrackSurfaceResolver(trackSpec.segments);

  // Place the car on the first segment centerline so the race starts on
  // the track and the track ribbon is visible from frame 0.
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

  // Detector state: latched along-track side (meters past/before the
  // finish plane); null until the first qualifying sample, cleared on
  // restart and whenever the car leaves the gate window around the
  // start point.
  let lapLinePrevSide: number | null = null;
  // Session holder bound right after creation; frame/physics closures read
  // it only once the RAF loop starts (strictly after the binding below).
  const driveHolder: { session: NeatRacingRaceSession | null } = {
    session: null,
  };

  // Keyboard: track a stable presses record on the window; the session
  // samples it once per frame (the phase-1 drive contract).
  const presses: {
    up: boolean;
    down: boolean;
    left: boolean;
    right: boolean;
  } = {
    up: false,
    down: false,
    left: false,
    right: false,
  };
  const keydownListener = (event: KeyboardEvent): void => {
    switch (event.code) {
      case 'KeyW':
      case 'ArrowUp':
        presses.up = true;
        event.preventDefault();
        break;
      case 'KeyS':
      case 'ArrowDown':
        presses.down = true;
        event.preventDefault();
        break;
      case 'KeyA':
      case 'ArrowLeft':
        presses.left = true;
        event.preventDefault();
        break;
      case 'KeyD':
      case 'ArrowRight':
        presses.right = true;
        event.preventDefault();
        break;
      default:
        break;
    }
  };
  const keyupListener = (event: KeyboardEvent): void => {
    switch (event.code) {
      case 'KeyW':
      case 'ArrowUp':
        presses.up = false;
        break;
      case 'KeyS':
      case 'ArrowDown':
        presses.down = false;
        break;
      case 'KeyA':
      case 'ArrowLeft':
        presses.left = false;
        break;
      case 'KeyD':
      case 'ArrowRight':
        presses.right = false;
        break;
      default:
        break;
    }
  };
  session.keydownListener = keydownListener;
  window.addEventListener('keydown', keydownListener);
  session.keyupListener = keyupListener;
  window.addEventListener('keyup', keyupListener);

  // Composer seam: inert until the composition attempt below succeeds; the
  // session always receives a drawing seam (zero-step/failed-composition
  // frames simply draw nothing).
  let drawComposerFrame: (
    frame: NeatRacingFrameComposerFrame,
  ) => void = (): void => undefined;
  let disposeComposition: (() => void) | null = null;

  // Lap-line detector: the finish line is the plane through segment 0's
  // start point perpendicular to the track. A lap counts when the car's
  // along-track position flips sign while it stays inside the lateral
  // gate around the start point (crossing that plane anywhere else on
  // the loop, or lateral drift along the start straight, must not
  // count). Any-flip semantics for phase 1 (direction filtering is a
  // carried gap).
  const NEATRACING_LAP_LINE_LATERAL_GATE_M = 5;
  const NEATRACING_LAP_LINE_ALONG_EPS_M = 0.5;
  const resolveForwardLapLineCrossing = (): boolean => {
    const car = driveHolder.session?.state.car ?? null;
    if (car === null) {
      return false;
    }
    const firstSegment = trackSpec.segments[0];
    if (firstSegment === undefined) {
      return false;
    }
    const chordX = firstSegment.endX - firstSegment.startX;
    const chordZ = firstSegment.endY - firstSegment.startY;
    const chordLength = Math.hypot(chordX, chordZ);
    if (chordLength === 0) {
      return false;
    }
    const relX = car.x - firstSegment.startX;
    const relY = car.y - firstSegment.startY;
    const alongM = (relX * chordX + relY * chordZ) / chordLength;
    const lateralM = (chordX * relY - chordZ * relX) / chordLength;
    if (Math.abs(lateralM) > NEATRACING_LAP_LINE_LATERAL_GATE_M) {
      lapLinePrevSide = null;
      return false;
    }
    if (Math.abs(alongM) < NEATRACING_LAP_LINE_ALONG_EPS_M) {
      return false;
    }
    const baseline = lapLinePrevSide;
    lapLinePrevSide = alongM;
    if (baseline === null) {
      return false;
    }
    return baseline * alongM < 0;
  };

  // Reusable frame building bags: one draw frame object refreshed in place
  // per frame (the composer consumes it immediately, so reuse is safe).
  const viewProjection = new Float32Array(16);
  const view = new Float32Array(16);
  const projection = new Float32Array(16);
  // The chase-camera basis (Step 14P): the camera eye, the look target and
  // the camera forward in ONE stable bag, refreshed IN PLACE every frame
  // (the zero per-frame allocation rule). The composer frame's
  // `cameraWorld` IS the basis camera eye - one array, no copy - so the
  // camera can never be placed by a SECOND set of chase constants.
  const chaseBasis: ChaseCameraBasis = {
    cameraWorld: [0, 0, 0],
    lookTarget: [0, 0, 0],
    forwardVector: [0, 0, 0],
  };
  const cameraWorld = chaseBasis.cameraWorld;
  const cameraPose = { yawRadians: 0, pitchRadians: 0 };
  const carPoseBag = {
    worldX: 0,
    worldZ: 0,
    yawRad: 0,
    teamColorId: NEATRACING_TEAM_ID_RED,
    wheelSpinRad: 0,
    steerAngleRad: 0,
    driftYawOffsetRad: 0,
    tractionSeverity: 0,
  };
  const carsBag = [carPoseBag];
  // The drift debug seam BAG (Step 14Q2): ONE stable object, refreshed IN
  // PLACE every frame, so the seam can carry the PUBLISHED traction
  // severity WITHOUT a per-frame allocation (the zero per-frame
  // allocation rule). The window handle is assigned this SAME bag once
  // per frame, so the seam never hands out a fresh object per frame. The
  // seeds below are never published: the bag is refreshed BEFORE every
  // assignment.
  const driftSeamBag: NeatRacingDriftStateBag = {
    followOffsetRad: 0,
    poseRad: 0,
    poseRateRadPerS: 0,
    slipAngleRad: 0,
    speedMps: 0,
    tick: 0,
    tickHz: 0,
    tractionSeverity: 0,
  };
  // In-place ambient pulse bag (the frame property is a stable readonly
  // reference by composer contract; contents are refreshed per frame the
  // same way `cameraWorld` and `carPoseBag` mutate their stable buffers).
  const pulseListBag: AmbientPulse[] = [];
  let wheelRadius = NEATRACING_WHEEL_FALLBACK_RADIUS;
  // The chase-camera follow state (Step 14P): a STABLE in-place bag - the
  // frame never holds a fresh follow object (the zero per-frame allocation
  // rule). The follow advances in TICKS, so the aim is frame-rate
  // independent: the same number of ticks always yields the same offset.
  const chaseFollow: ChaseCameraFollowState = createChaseCameraFollowState();
  const chaseFollowStep: ChaseCameraFollowStep = {
    ticks: 0,
    headingRad: 0,
    travelHeadingRad: 0,
    speedMps: 0,
    tick: 0,
  };
  // Display wheel spin accumulates the clamped visual omega across frames
  // (AC-015 strobing cap); it resets when a new session binds or when a
  // restart winds the tick counter backwards.
  let wheelSpinAccumulatorRad = 0;
  let wheelSpinLastTick = 0;
  let wheelSpinBoundSession: object | null = null;
  const rightWorldCarPosition: [number, number, number] = [0, 0, 0];
  const rightWorldDrawState: RightWorldPassDrawState = {
    viewProjection,
    cameraWorld,
    cameraRight: [1, 0, 0],
    cameraUp: [0, 1, 0],
    timeSeconds: 0,
    carPosition: rightWorldCarPosition,
  };
  const composerFrame: NeatRacingFrameComposerFrame = {
    viewProjection,
    cameraWorld,
    cameraPose,
    pulses: pulseListBag,
    cars: carsBag,
    rightWorld: rightWorldDrawState,
    enabled: true,
  };

  // Frame builder: chase camera hovering behind and above the car, looking
  // slightly ahead along the heading. Scene handedness mirrors the top-down
  // math plane (worldZ = -y), so the scene forward is (cos, -sin) and the car
  // yaw carries the nose offset (see NEATRACING_CAR_SCENE_YAW_OFFSET_RAD).
  // The wheel spin accumulates the clamped visual omega per stepped tick
  // (AC-015 strobing cap) instead of replaying the uncapped physical integral.
  const buildComposerFrame = (): NeatRacingFrameComposerFrame => {
    const boundSession = driveHolder.session;
    if (boundSession !== null) {
      const car = boundSession.state.car;
      const carSceneZ = carSceneWorldZ(car.y);
      // Step 14P: the chase camera follows the TRAVEL direction (the
      // PUBLISHED slip), not the nose. The follow advances in TICKS - the
      // deterministic tick counter is the clock - so the aim is identical
      // whether the frame released one tick or four.
      chaseFollowStep.ticks = Math.max(0, car.tick - chaseFollow.lastTick);
      chaseFollowStep.headingRad = car.heading;
      chaseFollowStep.travelHeadingRad = car.travelHeadingRad;
      chaseFollowStep.speedMps = car.speed;
      chaseFollowStep.tick = car.tick;
      stepChaseCameraFollow(chaseFollow, chaseFollowStep, chaseFollow);
      // The ONE aim law: the nose heading plus the follow offset.
      const chaseAimRad = resolveChaseCameraAimRad(
        car.heading,
        chaseFollow.offsetRad,
      );
      // The ONE basis: the camera eye, the look target and the camera
      // forward are all derived from the aim ALONE (the shared
      // chase-camera module owns the geometry).
      resolveChaseCameraBasis(car.x, car.y, chaseAimRad, chaseBasis);
      if (
        wheelSpinBoundSession !== boundSession ||
        car.tick < wheelSpinLastTick
      ) {
        wheelSpinBoundSession = boundSession;
        wheelSpinAccumulatorRad = 0;
        wheelSpinLastTick = car.tick;
      }
      const wheelSpinDeltaTicks = car.tick - wheelSpinLastTick;
      wheelSpinLastTick = car.tick;
      if (wheelSpinDeltaTicks > 0) {
        wheelSpinAccumulatorRad += computeVisualWheelSpinDeltaRad(
          wheelSpinDeltaTicks,
          NEATRACING_FIXED_TIMESTEP_SECONDS,
          car.speed,
          wheelRadius,
        );
      }
      fillCarRenderPose(
        carPoseBag,
        car.x,
        carSceneZ,
        carSceneYawRad(car.heading),
        wheelSpinAccumulatorRad,
        NEATRACING_TEAM_ID_RED,
        boundSession.state.input?.steer,
      );
      // AC-047C / AC-087: feed the pre-computed visual drift pose from the
      // physics state directly into the car pass. The pose now lives in the
      // NESTED drift channel (`car.drift.pose.pose`); it is recomputed once
      // per 60 Hz physics tick, never per frame.
      carPoseBag.driftYawOffsetRad = car.drift.pose.pose;
      // AC-088: the car pass CONSUMES the published traction severity, so
      // the ONE shared scalar the physics state publishes reaches the
      // tires (the per-tire grip) and the front-wheel steer yaw. The value
      // is READ here and NEVER re-derived in the browser.
      carPoseBag.tractionSeverity = car.tractionSeverity;
      // 14N drift debug seam: ONE assignment per frame, EXACTLY eight
      // fields, and no new public gameplay API (it is a `window.` debug
      // surface, not an exported module symbol). The browser half
      // MEASURES the pose instead of inferring it. The fifth field is
      // the deterministic simulation tick counter, so the browser half
      // can budget its phases in TICKS and measure a rate per tick
      // instead of per rendered frame. The sixth field (Step 14P) is the
      // chase-camera follow offset: the camera's own share of the
      // shared slip scalar, so the browser half can measure how far the
      // 3D view actually swung. The seventh field (Step 14Q2) is the
      // PUBLISHED traction severity: the ONE shared scalar the physics
      // state publishes, READ here and NEVER re-derived in the browser.
      // The eighth field (Step 14R) is the PUBLISHED pose-spring rate
      // (rad/s): the crab's own rate, read from the physics state, so
      // the browser half can measure the ring-down and the far-side pop
      // in simulation time instead of differencing the pose across
      // render-rate dependent frames. It is NOT the severity (that is the
      // seventh field).
      //
      // The seam is a STABLE BAG (the zero per-frame allocation rule):
      // the fields are refreshed IN PLACE and the window handle is
      // assigned the SAME bag, so the seam never allocates a fresh
      // object per frame.
      driftSeamBag.poseRad = car.drift.pose.pose;
      driftSeamBag.poseRateRadPerS = car.drift.pose.poseRate;
      driftSeamBag.slipAngleRad = car.slipAngle;
      driftSeamBag.speedMps = car.speed;
      driftSeamBag.tickHz = 1 / NEATRACING_FIXED_TIMESTEP_SECONDS;
      driftSeamBag.tick = car.tick;
      driftSeamBag.followOffsetRad = chaseFollow.offsetRad;
      driftSeamBag.tractionSeverity = car.tractionSeverity;
      window.neatRacingDriftState = driftSeamBag;
      // Ambient pulse window (TP-20): the pure time-window collector turns
      // the deterministic physics tick count into the alive-set, newest
      // first, so the expanding ground rings replay identically per seed.
      // The bag refreshes in place to keep the frame's readonly reference
      // identity-stable for the composer.
      const nextPulses = collectActivePulses(
        car.tick * NEATRACING_FIXED_TIMESTEP_SECONDS * 1000,
        NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
      );
      pulseListBag.length = 0;
      for (const pulse of nextPulses) {
        pulseListBag.push(pulse);
      }
      // The starfield yaw follows the TRAVEL direction (Step 14P), so the
      // sky swings with the slide instead of with the nose.
      cameraPose.yawRadians = carSceneYawRad(chaseAimRad);
      cameraPose.pitchRadians = NEATRACING_CHASE_CAMERA_PITCH_RADIANS;
      const aspect = canvas.height > 0 ? canvas.width / canvas.height : 16 / 9;
      perspectiveInto(
        projection,
        NEATRACING_CHASE_CAMERA_FOV_RADIANS,
        aspect,
        NEATRACING_CHASE_CAMERA_NEAR,
        NEATRACING_CHASE_CAMERA_FAR,
      );
      const targetX = chaseBasis.lookTarget[0];
      const targetY = chaseBasis.lookTarget[1];
      const targetZ = chaseBasis.lookTarget[2];
      lookAtInto(
        view,
        cameraWorld[0],
        cameraWorld[1],
        cameraWorld[2],
        targetX,
        targetY,
        targetZ,
      );
      multiplyViewProjectionInto(viewProjection, view, projection);
      // Right-world billboarding basis: derive normalized camera-right/up from
      // the chase look direction so the public/agent sprites always face the
      // active camera. The basis is refreshed in-place on the shared draw-state
      // bag (zero per-frame allocation). The forward vector comes from the
      // shared chase basis (Step 14P), so the sprites and the camera can never
      // disagree about where the camera looks.
      const fX = chaseBasis.forwardVector[0];
      const fY = chaseBasis.forwardVector[1];
      const fZ = chaseBasis.forwardVector[2];
      const rX = fY * 0 - fZ * 1;
      const rY = fZ * 0 - fX * 0;
      const rZ = fX * 1 - fY * 0;
      const rightLength = Math.hypot(rX, rY, rZ) || 1;
      rightWorldDrawState.cameraRight[0] = rX / rightLength;
      rightWorldDrawState.cameraRight[1] = rY / rightLength;
      rightWorldDrawState.cameraRight[2] = rZ / rightLength;
      rightWorldDrawState.cameraUp[0] =
        rightWorldDrawState.cameraRight[1] * fZ -
        rightWorldDrawState.cameraRight[2] * fY;
      rightWorldDrawState.cameraUp[1] =
        rightWorldDrawState.cameraRight[2] * fX -
        rightWorldDrawState.cameraRight[0] * fZ;
      rightWorldDrawState.cameraUp[2] =
        rightWorldDrawState.cameraRight[0] * fY -
        rightWorldDrawState.cameraRight[1] * fX;
      // Feed the live clock and car position into the right-world sprite pass
      // so billboards animate and look at the car (OBS-2 + OBS-3). The bag is
      // updated in-place to keep the frame object reference-stable.
      rightWorldDrawState.timeSeconds =
        car.tick * NEATRACING_FIXED_TIMESTEP_SECONDS;
      rightWorldCarPosition[0] = car.x;
      rightWorldCarPosition[1] = 0;
      rightWorldCarPosition[2] = carSceneZ;
    }
    return composerFrame;
  };

  // Composition attempt: gated on program-compile capability so fake/headless
  // GL fixtures (no createShader) degrade without console noise, and guarded
  // against assembly failures with the same inert degradation.
  if (typeof gl.createShader === 'function') {
    try {
      const bodyModel = decodeVoxelModel(
        validateVoxelModelDocument(carModelDocument),
      );
      const wheelModel = decodeVoxelModel(
        validateVoxelModelDocument(wheelModelDocument),
      );
      const bodyBins: VoxelBinModel = {
        dims: bodyModel.dims,
        palette: bodyModel.palette,
        quads: meshVoxelModel(bodyModel).quads,
      };
      const wheelBins: VoxelBinModel = {
        dims: wheelModel.dims,
        palette: wheelModel.palette,
        quads: meshVoxelModel(wheelModel).quads,
      };
      const starfieldBackdrop = resolveStarfieldBackingDimensions(canvas);
      const starfieldPass = createStarfieldPass(gl, {
        backdropWidthPx: starfieldBackdrop.width,
        backdropHeightPx: starfieldBackdrop.height,
        horizonNdcY: NEATRACING_STARFIELD_HORIZON_NDC_Y,
      });
      session.starfieldPass = starfieldPass;
      const pulsePass = createPulsePass(gl);
      const trackPass = createTrackPass(gl);
      const trackGridPass = createTrackGridPass(gl);
      const rightWorldPass = createRightWorldPass(gl);
      const carPass = createCarPass(gl, { body: bodyBins, wheel: wheelBins });
      const glassPass = createGlassPass(gl, {
        body: bodyModel,
        wheel: wheelModel,
      });
      const bloomPass = createBloomPass(gl);
      const composerSeam = createFrameComposer({
        gl,
        passes: {
          starfield: starfieldPass,
          pulse: pulsePass,
          track: trackPass,
          trackGrid: trackGridPass,
          rightWorld: rightWorldPass,
          car: carPass,
          glass: glassPass,
          bloom: bloomPass,
        },
      });
      drawComposerFrame = (frame): void => composerSeam.drawFrame(frame);
      wheelRadius = Math.max(
        (wheelBins.dims[2] / 2) * NEATRACING_VOXEL_WORLD_SIZE,
        0.5,
      );
      disposeComposition = (): void => {
        composerSeam.dispose();
        // Starfield and track-grid passes expose no dispose seam (construct-once
        // program handles that stay live for the page); the disposable
        // passes ride with the composer teardown below.
        pulsePass.dispose();
        rightWorldPass.dispose();
        carPass.dispose();
        glassPass.dispose();
        bloomPass.dispose();
      };
    } catch (error) {
      // Degrade to the inert composer: the drive loop stays live and the
      // failure is surfaced to the console for operator visibility.
      console.warn('neatRacing: composer composition degraded', error);
    }
  }

  // The session owns the drive contract: one sampled driving input per
  // frame (the autopilot seam when the selector is on `ai` — the page
  // opens as a running race — else the keyboard sample), physics then
  // rules per released tick, one hud.update per drawn frame, and exactly
  // one composer drawFrame per frame (zero-step frames included).
  const driveSession = createRaceSession({
    composer: { drawFrame: (frame) => drawComposerFrame(frame) },
    hud: overlays.hud,
    modeSelector: overlays.selector,
    presses,
    sampleKeyboardInput,
    sampleAutoInput: (car) => stepAutopilotInput(car, trackSpec.splineSamples),
    stepPhysics: (car, input) =>
      stepPhysics(car, input, surfaceResolver, trackSpec.segments),
    stepRaceRules,
    resolveForwardLapLineCrossing,
    createInitialCarState: () =>
      createInitialPhysicsCarState(trackSpec.seed, spawnPose),
    buildComposerFrame,
  });
  driveHolder.session = driveSession;

  // Boot-owned restart carrier: the session subscribed its own restart
  // listener exactly once at creation (contract); this second boot-owned
  // subscription only re-binds the detector baseline so a fresh car never
  // registers a spurious crossing against the previous car's side sample.
  overlays.selector.onSwitch((): void => {
    lapLinePrevSide = null;
  });

  // Composer + pass teardown rides on the boot stop handle.
  session.driveTeardown = disposeComposition;
  return driveSession;
}

/**
 * Runs the session-driven render loop.
 *
 * Exactly one `requestAnimationFrame` is requested at boot; each frame hands
 * the timestamp to the drive session, which advances its own fixed-timestep
 * accumulator (physics → rules → HUD → composer) and re-requests the next
 * frame so the loop stays alive. The boot no longer carries an accumulator:
 * the drive loop slice replaced the boot-owned `advanceFrameSteps` wiring
 * with the session-owned one (No-Deferred-Cleanup: the old boot-side wiring
 * is removed with this change). A stopped session never schedules again, so
 * the stop handle terminates the loop deterministically.
 *
 * @param session - Boot session storing the current frame handle.
 * @param driveSession - Session whose `handleFrame` drives one frame.
 */
function startRenderLoop(
  session: BootSession,
  driveSession: { handleFrame(timestampMs: number): void },
): void {
  const driveFrame = (timestampMs: number): void => {
    if (session.stopped) {
      return;
    }
    driveSession.handleFrame(timestampMs);
    session.frameHandle = window.requestAnimationFrame(driveFrame);
  };
  session.frameHandle = window.requestAnimationFrame(driveFrame);
}

/**
 * Installs the non-debounced viewport-resize listener that re-applies the
 * letterbox strategy from the live viewport (spec §6.2).
 *
 * Also re-uploads the starfield pass backdrop size so the star map stays
 * aligned with the computed backing width after a resize (OBS-F).
 *
 * @param canvas - Canvas whose backing store tracks the viewport.
 * @param session - Boot session storing the listener for teardown.
 * @param starfieldPass - Starfield pass whose NDC mapping is updated.
 */
function attachResizeListener(
  canvas: HTMLCanvasElement,
  session: BootSession,
  starfieldPass: StarfieldPass | null,
): void {
  session.resizeListener = (): void => {
    applyRendererSize(canvas);
    if (starfieldPass !== null) {
      const backdrop = resolveStarfieldBackingDimensions(canvas);
      starfieldPass.setBackdropSize(backdrop.width, backdrop.height);
    }
  };
  window.addEventListener('resize', session.resizeListener);
}

/**
 * Builds the boot stop handle that tears down every wired surface
 * symmetrically: the RAF loop, the keyboard listeners, the resize listener,
 * the composer + passes, any appended fallback notice, and the mounted game
 * canvas (AC-001).
 *
 * @param canvas - Mounted game canvas removed by the handle.
 * @param session - Boot session holding the surfaces to tear down.
 * @returns The stop handle.
 */
function createStopHandle(
  canvas: HTMLCanvasElement,
  session: BootSession,
): NeatRacingStop {
  return (): void => {
    session.stopped = true;
    if (session.frameHandle !== null) {
      window.cancelAnimationFrame(session.frameHandle);
      session.frameHandle = null;
    }
    if (session.keydownListener !== null) {
      window.removeEventListener('keydown', session.keydownListener);
      session.keydownListener = null;
    }
    if (session.keyupListener !== null) {
      window.removeEventListener('keyup', session.keyupListener);
      session.keyupListener = null;
    }
    if (session.resizeListener !== null) {
      window.removeEventListener('resize', session.resizeListener);
      session.resizeListener = null;
    }
    session.driveTeardown?.();
    session.driveTeardown = null;
    session.starfieldPass = null;
    session.fallbackNotice?.remove();
    session.fallbackNotice = null;
    canvas.remove();
  };
}

declare global {
  interface Window {
    /** Neat Racing demo boot function installed by the host bundle. */
    neatRacingStart?: NeatRacingStart;
    /**
     * The 14N drift debug seam: the read-only eight-field view of the live
     * drift channel. Assigned once per frame by the browser entry.
     */
    neatRacingDriftState?: NeatRacingDriftState;
  }
}

window.neatRacingStart = neatRacingStart;
