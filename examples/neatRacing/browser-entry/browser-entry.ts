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
  toGridWarmupProgram,
  createGridPass,
} from '../engine/passes/grid.pass';
import {
  toTrackWarmupProgram,
  createTrackPass,
} from '../engine/passes/track.pass';
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
} from '../engine/passes/starfield.pass';
import { createFrameComposer } from '../host/frame-composer';
import { computeVisualWheelSpinDeltaRad } from '../voxel/wheel.mount.utils';
import type { NeatRacingFrameComposerFrame } from '../host/frame-composer';
import type { NeatRacingHudState } from '../host/hud';
import { createHud } from '../host/hud';
import { sampleKeyboardInput } from '../host/input';
import type { NeatRacingRaceMode } from '../host/mode-selector';
import { createRaceModeSelector } from '../host/mode-selector';
import { createRaceSession } from '../host/race-session';
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
  startRenderLoop(session, startLiveDriveLoop(gl, session, overlays, canvas));

  // Step 7: Re-apply the letterbox strategy directly (non-debounced) on every
  // viewport resize (spec §6.2).
  attachResizeListener(canvas, session);

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
 * Every real pass exposes a lazy warmup seam (`to*WarmupProgram`), so the
 * entries compose without GL work until `runShaderWarmup` invokes them.
 * Rebuilt per boot against the live context, the registry handed to
 * `runShaderWarmup` carries exactly the nine pinned keys before frame 1.
 *
 * @param gl - Live WebGL2 context the warmup programs compile against.
 * @returns The nine warmup programs in the pinned registry-key order.
 */
function createBootProgramRegistry(
  gl: WebGL2RenderingContext,
): readonly ShaderWarmupProgram[] {
  const [bloomThreshold, bloomBlur, bloomAdditive] = toBloomWarmupPrograms(gl);
  return [
    toStarfieldWarmupProgram(gl),
    toGridWarmupProgram(gl),
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

/** Chase-camera distance behind the car (world units). */
const NEATRACING_CHASE_CAMERA_BACK = 5.5;
/** Chase-camera height above the ground plane (world units). */
const NEATRACING_CHASE_CAMERA_HEIGHT = 2.4;
/** Chase-camera look-ahead distance along the car heading (world units). */
const NEATRACING_CHASE_CAMERA_LOOKAHEAD = 8;
/** Chase-camera pitch (radians) fed to the starfield camera pose. */
const NEATRACING_CHASE_CAMERA_PITCH_RADIANS = 0.12;
/** Chase-camera vertical field of view (radians). */
const NEATRACING_CHASE_CAMERA_FOV_RADIANS = Math.PI / 3;
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
  const driveHolder: {
    session: {
      state: {
        car: {
          x: number;
          y: number;
          heading: number;
          speed: number;
          tick: number;
        };
      };
    } | null;
  } = { session: null };

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
  const cameraWorld: [number, number, number] = [0, 0, 0];
  const cameraPose = { yawRadians: 0, pitchRadians: 0 };
  const carPoseBag = {
    worldX: 0,
    worldZ: 0,
    yawRad: 0,
    teamColorId: NEATRACING_TEAM_ID_RED,
    wheelSpinRad: 0,
  };
  const carsBag = [carPoseBag];
  // In-place ambient pulse bag (the frame property is a stable readonly
  // reference by composer contract; contents are refreshed per frame the
  // same way `cameraWorld` and `carPoseBag` mutate their stable buffers).
  const pulseListBag: AmbientPulse[] = [];
  let wheelRadius = NEATRACING_WHEEL_FALLBACK_RADIUS;
  // Display wheel spin accumulates the clamped visual omega across frames
  // (AC-015 strobing cap); it resets when a new session binds or when a
  // restart winds the tick counter backwards.
  let wheelSpinAccumulatorRad = 0;
  let wheelSpinLastTick = 0;
  let wheelSpinBoundSession: object | null = null;
  const composerFrame: NeatRacingFrameComposerFrame = {
    viewProjection,
    cameraWorld,
    cameraPose,
    pulses: pulseListBag,
    cars: carsBag,
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
      const forwardX = Math.cos(car.heading);
      const sceneForwardZ = -Math.sin(car.heading);
      const carSceneZ = carSceneWorldZ(car.y);
      cameraWorld[0] = car.x - forwardX * NEATRACING_CHASE_CAMERA_BACK;
      cameraWorld[1] = NEATRACING_CHASE_CAMERA_HEIGHT;
      cameraWorld[2] = carSceneZ - sceneForwardZ * NEATRACING_CHASE_CAMERA_BACK;
      carPoseBag.worldX = car.x;
      carPoseBag.worldZ = carSceneZ;
      carPoseBag.yawRad = carSceneYawRad(car.heading);
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
      carPoseBag.wheelSpinRad = wheelSpinAccumulatorRad;
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
      cameraPose.yawRadians = carPoseBag.yawRad;
      cameraPose.pitchRadians = NEATRACING_CHASE_CAMERA_PITCH_RADIANS;
      const aspect = canvas.height > 0 ? canvas.width / canvas.height : 16 / 9;
      perspectiveInto(
        projection,
        NEATRACING_CHASE_CAMERA_FOV_RADIANS,
        aspect,
        NEATRACING_CHASE_CAMERA_NEAR,
        NEATRACING_CHASE_CAMERA_FAR,
      );
      lookAtInto(
        view,
        cameraWorld[0],
        cameraWorld[1],
        cameraWorld[2],
        car.x + forwardX * NEATRACING_CHASE_CAMERA_LOOKAHEAD,
        NEATRACING_CHASE_CAMERA_HEIGHT / 2,
        carSceneZ + sceneForwardZ * NEATRACING_CHASE_CAMERA_LOOKAHEAD,
      );
      multiplyViewProjectionInto(viewProjection, view, projection);
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
      const starfieldPass = createStarfieldPass(gl);
      const gridPass = createGridPass(gl);
      const pulsePass = createPulsePass(gl);
      const trackPass = createTrackPass(gl);
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
          grid: gridPass,
          pulse: pulsePass,
          track: trackPass,
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
        // Starfield and grid passes expose no dispose seam (construct-once
        // program handles that stay live for the page); the disposable
        // passes ride with the composer teardown below.
        pulsePass.dispose();
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
    stepPhysics: (car, input) => stepPhysics(car, input, surfaceResolver),
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
 * @param canvas - Canvas whose backing store tracks the viewport.
 * @param session - Boot session storing the listener for teardown.
 */
function attachResizeListener(
  canvas: HTMLCanvasElement,
  session: BootSession,
): void {
  session.resizeListener = (): void => {
    applyRendererSize(canvas);
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
    session.fallbackNotice?.remove();
    session.fallbackNotice = null;
    canvas.remove();
  };
}

declare global {
  interface Window {
    /** Neat Racing demo boot function installed by the host bundle. */
    neatRacingStart?: NeatRacingStart;
  }
}

window.neatRacingStart = neatRacingStart;
