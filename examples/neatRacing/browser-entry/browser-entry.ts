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
 * Step 05 (P2-S4-C) re-plumbs the live drive loop onto the Phase-2
 * simulation-worker protocol (AC-033): the host now owns a REAL sim-worker
 * session (`createSimWorkerSession`) instead of the phase-1 host session, so
 * the HUMAN drives car 0 while the worker's fleet carries the evolving
 * rivals. The worker owns the spawning, the physics, the lap-line detector
 * and the race rules; the host owns the input sampling, the composer
 * bindings, the AI-mode goal lifecycle, the most-forward network panel and
 * the teardown.
 *
 * The worker module is a PURE realm-free state machine apart from its
 * transport tail (`self.onmessage = ...`), which is harmless in the main
 * thread: the page never posts window messages, so the assignment only
 * parks an unused handler.
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
// The adaptive-quality controller (AC-035): the drive loop ticks it EXACTLY
// once per rendered frame and the composed frame carries the RESOLVED
// (defensively clamped) stage record's bloom off-switch - never a hardcoded
// literal.
import {
  createInitialAdaptiveQualityState,
  tickAdaptiveQuality,
  NEATRACING_ADAPTIVE_QUALITY_MAX_STAGE_INDEX,
  NEATRACING_ADAPTIVE_QUALITY_STAGES,
  type AdaptiveQualityStage,
  type AdaptiveQualityState,
} from '../engine/adaptive-quality';
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
import type { NeatRacingKeyboardSample } from '../host/input';
import type { NeatRacingRaceMode } from '../host/mode-selector';
import { createRaceModeSelector } from '../host/mode-selector';
import { NEATRACING_FIXED_TIMESTEP_SECONDS } from '../host/render-loop.utils';
import { createTrackSurfaceResolver } from '../physics/collision.utils';
import { createRaceRulesState } from '../physics/race-rules';
import type { RaceRulesState } from '../physics/race-rules';
import type {
  PhysicsCarState,
  PhysicsSurfaceType,
} from '../physics/physics.step.service';
import {
  createNeatRacingAiModeGoalSession,
  recordNeatRacingAiModeGoalCrossing,
  NEATRACING_AI_MODE_START_LEVEL,
} from '../ai/ai-mode.goals';
import type {
  NeatRacingAiModeGoalCrossingOutcome,
  NeatRacingAiModeGoalSession,
} from '../ai/ai-mode.goals';
import type { NeatRacingObservationLevel } from '../ai/observation.assembler';
import {
  resolveNeatRacingLevelCarCount,
  runNeatRacingScriptedBaselineProbeEpisode,
} from '../ai/promotion.rules';
import { detectSteeringOscillation } from '../ai/runtime.adaptation';
import {
  drawNeatRacingNetworkPanelFromFrame,
  installNeatRacingNetworkResize,
  resolveMostForwardNeatRacingAgent,
  resolveNeatRacingNetworkPanelVisualizationFrame,
} from '../visualizer/network-panel.adapter';
import type {
  NetworkVisualizationResolvedFrame,
  NeatRacingNetworkPanelCandidate,
} from '../visualizer/network-panel.adapter';
import { createSimWorkerSession } from '../workers/simulation-worker/simulation-worker';
import type {
  NeatRacingSimWorkerFramesPayload,
  NeatRacingSimWorkerRaceMode,
  NeatRacingSimWorkerSession,
} from '../workers/simulation-worker/simulation-worker';
import {
  NEATRACING_DEFAULT_FEATURE_FLAGS,
  NEATRACING_PACK_SCHEMA_VERSION,
} from '../workers/simulation-worker/frame.pack.service';
import type { VoxelBinModel } from '../voxel/bin.codec';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../voxel/lod.distance.utils';
import { meshVoxelModel } from '../voxel/greedy-mesher';
import { decodeVoxelModel } from '../voxel/model.decode.utils';
import { validateVoxelModelDocument } from '../voxel/model.schema';
import {
  NEATRACING_TEAM_ID_BLUE,
  NEATRACING_TEAM_ID_CYAN,
  NEATRACING_TEAM_ID_GREEN,
  NEATRACING_TEAM_ID_RED,
} from '../voxel/team-remap';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type {
  TrackSegment,
  TrackSpec,
} from '../../racing_curriculum/track/track.generator.types';
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
 * The demo track identity the host sends to the simulation worker.
 *
 * The worker protocol carries a `trackId` so a round can be attributed to
 * the track it ran on. The demo loop has no canonical track id anywhere in
 * the repository (the worker's own suites use their own fixture ids), so the
 * host owns this honest demo identity instead of borrowing a test fixture.
 */
const NEATRACING_DEMO_TRACK_ID = 'neatracing-demo-loop';

/**
 * Element id of the most-forward-agent network panel canvas.
 *
 * The panel canvas is a SIBLING of the HUD subtree (it is mounted into the
 * reserved {@link NEATRACING_HUD_OUTPUT_ID} container as its own element) so
 * the HUD's own DOM contract is never disturbed.
 */
const NEATRACING_NETWORK_PANEL_CANVAS_ID = 'neatracing-network-panel';

/**
 * The rival team colours, cycled by rival index.
 *
 * Car 0 (the human) always keeps {@link NEATRACING_TEAM_ID_RED}; the rivals
 * cycle this palette so two cars never share a colour on the same frame.
 */
const NEATRACING_RIVAL_TEAM_COLOR_IDS: readonly number[] = [
  NEATRACING_TEAM_ID_BLUE,
  NEATRACING_TEAM_ID_GREEN,
  NEATRACING_TEAM_ID_CYAN,
];

/**
 * The along-track progress scan window (in chords).
 *
 * The host-owned progress meter scans the chords around the incumbent chord
 * index first and falls back to a FULL scan when the window clipped a
 * boundary, so a car that jumped far along the track still resolves its
 * position exactly.
 */
const NEATRACING_PROGRESS_CHORD_SCAN_WINDOW = 8;

/**
 * The fraction of the loop length that separates a forward wrap from a
 * backward wrap in the host-owned progress meter.
 */
const NEATRACING_PROGRESS_HALF_LOOP_FRACTION = 0.5;

/**
 * The physics surface that counts as "on the road" for the host-owned
 * off-track fraction.
 *
 * The physics surface band is a three-value union (`road`, `curb`,
 * `grass`); only `road` counts as on-track for the drivability measurement.
 */
const NEATRACING_PHYSICS_SURFACE_ROAD: PhysicsSurfaceType = 'road';

/**
 * The evolved-network node count the demo lane can honestly report.
 *
 * The eval worker's growth engine is NOT integrated yet (the workers are
 * transport-only in this slice), so the live demo has NO evolved carriers to
 * count. The number is reported as ZERO - never fabricated - because the
 * per-tick rival control and the network growth live inside the workers
 * (E11(a)). The demo crossings therefore ride the non-promoting `lapCounted`
 * path and the machine keeps racing (LD-10).
 */
const NEATRACING_DEMO_EVOLVED_NODE_COUNT = 0;

/**
 * The race mode the host boots the simulation worker with.
 *
 * The host ALWAYS boots the worker in `manual` mode: the human drives car 0
 * in BOTH UI modes (LD-12), and the worker's `ai` race mode would hand car
 * 0's controls to the machine.
 */
const NEATRACING_DEMO_WORKER_RACE_MODE: NeatRacingSimWorkerRaceMode =
  'manual';

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
  /** Live simulation-worker session, or null once it is stopped. */
  simSession: NeatRacingSimWorkerSession | null;
  /** Network-panel teardown (resize uninstall + canvas removal), or null. */
  networkPanelTeardown: (() => void) | null;
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
  // the context lacks program support or composition throws), boot the
  // simulation-worker session, and start the RAF loop. The worker owns the
  // fixed-timestep accumulator: each frame feeds the protocol one pace
  // request and rebinds the reply (physics → rules snapshots). One request
  // at boot, one re-request per frame (AC-003 cadence).
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
    simSession: null,
    networkPanelTeardown: null,
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
 * Resolves the defensively clamped adaptive-quality stage record for a
 * controller stage index (AC-035): negative, out-of-range or non-finite
 * controller outputs clamp into the stage ladder so the off-switch knob is
 * ALWAYS a defined stage record (never `undefined`).
 *
 * @param stageIndex - The raw stage index handed back by the ticked
 *   controller.
 * @returns The resolved (defensively clamped) stage record.
 */
function resolveAdaptiveQualityStageClamped(
  stageIndex: number,
): AdaptiveQualityStage {
  const clampedStageIndex = Number.isFinite(stageIndex)
    ? Math.min(
        Math.max(Math.trunc(stageIndex), 0),
        NEATRACING_ADAPTIVE_QUALITY_MAX_STAGE_INDEX,
      )
    : 0;
  return NEATRACING_ADAPTIVE_QUALITY_STAGES[clampedStageIndex];
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
 * One track chord: a straight piece of the generated loop plus its
 * along-track arc-length offset.
 *
 * WHY: the host-owned along-track progress meter measures a car's position
 * by the NEAREST-CHORD rule over the generated segments, so the meter needs
 * the chord geometry (start, end, length) and the arc-length offset of each
 * chord in loop order.
 */
export interface NeatRacingTrackChord {
  /** The chord start point (generator x / y plane). */
  readonly startX: number;
  /** The chord start point (generator x / y plane). */
  readonly startY: number;
  /** The chord end point (generator x / y plane). */
  readonly endX: number;
  /** The chord end point (generator x / y plane). */
  readonly endY: number;
  /** The chord length (world units). */
  readonly lengthM: number;
  /** The chord's arc-length offset from the loop start (world units). */
  readonly startOffsetM: number;
}

/**
 * The boot-time chord index the host-owned progress meter rides.
 */
export interface NeatRacingTrackChordIndex {
  /** The chords in loop order. */
  readonly chords: readonly NeatRacingTrackChord[];
  /** The total loop length (world units). */
  readonly loopLengthM: number;
}

/**
 * The host-owned along-track progress state of ONE car.
 *
 * UNIT: the progress meter measures METERS along the track's centerline
 * (the sum of the generated segments' chord lengths).
 */
export interface NeatRacingCarProgressState {
  /** The monotone along-track total (meters). */
  readonly totalM: number;
  /** The along-track offset inside the loop (meters, `0..loopLengthM`). */
  readonly alongM: number;
  /** The incumbent chord index (the nearest chord). */
  readonly chordIndex: number;
}

/**
 * The host decision surface for one AI-mode goal crossing.
 */
export interface NeatRacingAiModeRebaseDecision {
  /**
   * The level the host round must rebase onto, or null when the crossing
   * did not promote the goal session.
   */
  readonly rebaseRung: NeatRacingObservationLevel | null;
}

/**
 * The host-owned lap-fact bookkeeping for one rendered frame.
 */
export interface NeatRacingLapFactAdvance {
  /** The baseline the host adopted (the rules' laps completed). */
  readonly lapBaseline: number;
  /** The laps completed since the last crossing (the protocol counter). */
  readonly lapsSinceLastCrossing: number;
  /** True when the frame carried a forward crossing. */
  readonly crossed: boolean;
}

/**
 * Builds the chord index the host-owned progress meter rides.
 *
 * The chords are the generated track's segments in loop order; the arc-length
 * offsets are ONE prefix-sum over the chord lengths, so the index is built
 * once per boot and never rebuilt per frame.
 *
 * @param segments - The generated track's segments in loop order.
 * @returns The chord index and the total loop length.
 *
 * @example
 * ```ts
 * const index = buildNeatRacingTrackChordIndex(trackSpec.segments);
 * ```
 */
export function buildNeatRacingTrackChordIndex(
  segments: readonly TrackSegment[],
): NeatRacingTrackChordIndex {
  const chords: NeatRacingTrackChord[] = [];
  let startOffsetM = 0;
  for (const segment of segments) {
    const lengthM = Math.hypot(
      segment.endX - segment.startX,
      segment.endY - segment.startY,
    );
    chords.push({
      startX: segment.startX,
      startY: segment.startY,
      endX: segment.endX,
      endY: segment.endY,
      lengthM,
      startOffsetM,
    });
    startOffsetM += lengthM;
  }
  return { chords, loopLengthM: startOffsetM };
}

/**
 * Resolves the squared distance from a point to the nearest point of a
 * chord (the clamped projection).
 *
 * @param chord - The chord to measure against.
 * @param pointX - The point's generator x coordinate.
 * @param pointY - The point's generator y coordinate.
 * @returns The squared distance.
 */
function resolveChordDistanceSq(
  chord: NeatRacingTrackChord,
  pointX: number,
  pointY: number,
): number {
  const chordX = chord.endX - chord.startX;
  const chordY = chord.endY - chord.startY;
  const lengthSq = chord.lengthM * chord.lengthM;
  const relX = pointX - chord.startX;
  const relY = pointY - chord.startY;
  if (lengthSq === 0) {
    return relX * relX + relY * relY;
  }
  const t = Math.min(
    1,
    Math.max(0, (relX * chordX + relY * chordY) / lengthSq),
  );
  const offsetX = relX - t * chordX;
  const offsetY = relY - t * chordY;
  return offsetX * offsetX + offsetY * offsetY;
}

/**
 * Resolves the along-track offset of a point on a chord (meters).
 *
 * @param chord - The chord to measure against.
 * @param pointX - The point's generator x coordinate.
 * @param pointY - The point's generator y coordinate.
 * @returns The along-track offset from the loop start (meters).
 */
function resolveChordAlongM(
  chord: NeatRacingTrackChord,
  pointX: number,
  pointY: number,
): number {
  const chordX = chord.endX - chord.startX;
  const chordY = chord.endY - chord.startY;
  const lengthSq = chord.lengthM * chord.lengthM;
  const relX = pointX - chord.startX;
  const relY = pointY - chord.startY;
  if (lengthSq === 0) {
    return chord.startOffsetM;
  }
  const t = Math.min(
    1,
    Math.max(0, (relX * chordX + relY * chordY) / lengthSq),
  );
  return chord.startOffsetM + t * chord.lengthM;
}

/**
 * Resolves the nearest chord of a point (the nearest-chord rule).
 *
 * The scan is WINDOWED around the incumbent chord index (a car moves a few
 * chords per frame) and falls back to a FULL scan whenever the windowed
 * winner sits on the window boundary, so the nearest chord is always the
 * true nearest one.
 *
 * @param index - The boot-time chord index.
 * @param pointX - The point's generator x coordinate.
 * @param pointY - The point's generator y coordinate.
 * @param incumbentIndex - The incumbent chord index of the previous frame.
 * @returns The index of the nearest chord.
 */
function resolveNearestChordIndex(
  index: NeatRacingTrackChordIndex,
  pointX: number,
  pointY: number,
  incumbentIndex: number,
): number {
  const chords = index.chords;
  if (chords.length === 0) {
    return 0;
  }
  const windowStart = Math.max(
    0,
    incumbentIndex - NEATRACING_PROGRESS_CHORD_SCAN_WINDOW,
  );
  const windowEnd = Math.min(
    chords.length - 1,
    incumbentIndex + NEATRACING_PROGRESS_CHORD_SCAN_WINDOW,
  );
  let bestIndex = windowStart;
  let bestDistanceSq = Number.POSITIVE_INFINITY;
  for (let chordIndex = windowStart; chordIndex <= windowEnd; chordIndex += 1) {
    const chord = chords[chordIndex];
    if (chord === undefined) {
      continue;
    }
    const distanceSq = resolveChordDistanceSq(chord, pointX, pointY);
    if (distanceSq < bestDistanceSq) {
      bestDistanceSq = distanceSq;
      bestIndex = chordIndex;
    }
  }
  const onWindowBoundary = bestIndex === windowStart || bestIndex === windowEnd;
  if (!onWindowBoundary) {
    return bestIndex;
  }
  // FULL-SCAN FALLBACK: the windowed winner sits on the boundary, so the
  // true nearest chord may live outside the window.
  let fullBestIndex = 0;
  let fullBestDistanceSq = Number.POSITIVE_INFINITY;
  for (let chordIndex = 0; chordIndex < chords.length; chordIndex += 1) {
    const chord = chords[chordIndex];
    if (chord === undefined) {
      continue;
    }
    const distanceSq = resolveChordDistanceSq(chord, pointX, pointY);
    if (distanceSq < fullBestDistanceSq) {
      fullBestDistanceSq = distanceSq;
      fullBestIndex = chordIndex;
    }
  }
  return fullBestIndex;
}

/**
 * Resolves the host-owned along-track progress of one car.
 *
 * UNIT: METERS along the track's centerline (the sum of the generated
 * segments' chord lengths). The position inside the loop comes from the
 * NEAREST-CHORD rule (the clamped projection onto the nearest chord); the
 * total accumulates the forward distance and detects a half-loop wrap.
 *
 * WRAP LAW: the wrap direction is read from the SIGN of `deltaM`; the
 * loop-length correction applies OPPOSITE that sign. A FORWARD crossing
 * of the loop start DROPS the global along-track position (`deltaM` very
 * negative) and grows the total by the loop tail + head
 * (`deltaM + loopLengthM`, the +L correction opposite the negative
 * sign). A BACKWARD crossing RAISES the position (`deltaM` very
 * positive) and DECREMENTS the total by the wrapped shorter displacement
 * (`deltaM - loopLengthM`): the rivals never own the rules' laps (the
 * worker's lap detector is car-0-only), so the host accepts the
 * decrement instead of inventing a lap counter of its own.
 *
 * @param index - The boot-time chord index.
 * @param carX - The car's generator x coordinate.
 * @param carY - The car's generator y coordinate.
 * @param previous - The previous progress state, or null on the first frame.
 * @returns The car's progress state.
 *
 * @example
 * ```ts
 * const progress = resolveNeatRacingCarAlongProgressM(
 *   index,
 *   car.x,
 *   car.y,
 *   previous,
 * );
 * ```
 */
export function resolveNeatRacingCarAlongProgressM(
  index: NeatRacingTrackChordIndex,
  carX: number,
  carY: number,
  previous: NeatRacingCarProgressState | null,
): NeatRacingCarProgressState {
  const chords = index.chords;
  if (chords.length === 0 || index.loopLengthM <= 0) {
    return { totalM: 0, alongM: 0, chordIndex: 0 };
  }
  const incumbentIndex = previous?.chordIndex ?? 0;
  const chordIndex = resolveNearestChordIndex(
    index,
    carX,
    carY,
    incumbentIndex,
  );
  const chord = chords[chordIndex];
  if (chord === undefined) {
    return { totalM: 0, alongM: 0, chordIndex: 0 };
  }
  const alongM = resolveChordAlongM(chord, carX, carY);
  if (previous === null) {
    return { totalM: alongM, alongM, chordIndex };
  }
  const deltaM = alongM - previous.alongM;
  const halfLoopM = index.loopLengthM * NEATRACING_PROGRESS_HALF_LOOP_FRACTION;
  if (deltaM > halfLoopM) {
    const totalM = previous.totalM + deltaM - index.loopLengthM;
    return { totalM, alongM, chordIndex };
  }
  if (deltaM < -halfLoopM) {
    const totalM = previous.totalM + deltaM + index.loopLengthM;
    return { totalM, alongM, chordIndex };
  }
  return { totalM: previous.totalM + deltaM, alongM, chordIndex };
}

/**
 * Resolves the host's rebase decision for one AI-mode goal crossing.
 *
 * The decision surface is PURE: the host applies the rebase (re-boot the
 * worker round on the promoted rung) and the suite can exercise the same
 * decision without a live promoted round.
 *
 * @param outcome - The goal machine's outcome for the crossing.
 * @returns The rebase decision (null when the round does not rebase).
 *
 * @example
 * ```ts
 * const decision = resolveNeatRacingAiModeRebaseRung(outcome);
 * if (decision.rebaseRung !== null) {
 *   rebaseHostRound(decision.rebaseRung);
 * }
 * ```
 */
export function resolveNeatRacingAiModeRebaseRung(
  outcome: NeatRacingAiModeGoalCrossingOutcome,
): NeatRacingAiModeRebaseDecision {
  if (outcome.kind !== 'promoted') {
    return { rebaseRung: null };
  }
  // The promoted rung is the PROMOTED SESSION's level (the machine already
  // clamped the evaluation's `toLevel` onto the shared observation ladder).
  return { rebaseRung: outcome.session.level };
}

/**
 * Resolves the AI-mode goal session a mode switch must boot.
 *
 * A switch INTO `ai` boots a FRESH goal session on the start level; a
 * switch to `manual` parks the AI-mode lifecycle (the adopted session stays
 * adopted and NO further goal facts are fed), so the resolver returns null.
 *
 * @param mode - The UI race mode the host switched to.
 * @returns The fresh goal session, or null when the lifecycle parks.
 *
 * @example
 * ```ts
 * const booted = resolveNeatRacingAiModeGoalSessionForMode('ai');
 * ```
 */
export function resolveNeatRacingAiModeGoalSessionForMode(
  mode: NeatRacingRaceMode,
): NeatRacingAiModeGoalSession | null {
  if (mode !== 'ai') {
    return null;
  }
  return createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL);
}

/**
 * Resolves the host-owned lap-fact bookkeeping for one rendered frame.
 *
 * The worker owns the lap detector (car 0 only); the host keeps the
 * baseline it adopted at the round boot and derives the laps completed since
 * the last crossing from the protocol's forward-lap counter.
 *
 * @param lapBaseline - The laps completed the host adopted as its baseline.
 * @param lapsCompleted - The laps completed the rules clone now reports.
 * @returns The lap-fact advance (no crossing when nothing changed).
 *
 * @example
 * ```ts
 * const advance = resolveNeatRacingLapFactAdvance(0, 1);
 * // { lapBaseline: 1, lapsSinceLastCrossing: 1, crossed: true }
 * ```
 */
export function resolveNeatRacingLapFactAdvance(
  lapBaseline: number,
  lapsCompleted: number,
): NeatRacingLapFactAdvance {
  if (lapsCompleted <= lapBaseline) {
    return { lapBaseline, lapsSinceLastCrossing: 0, crossed: false };
  }
  return {
    lapBaseline: lapsCompleted,
    lapsSinceLastCrossing: lapsCompleted - lapBaseline,
    crossed: true,
  };
}

/**
 * The host-owned display wheel-spin state of ONE car.
 */
interface NeatRacingWheelSpinState {
  /** The accumulated display wheel spin (radians). */
  accumulatorRad: number;
  /** The last published physics tick the accumulator saw. */
  lastTick: number;
}

/**
 * The host-owned race state: what the drive loop reads from the worker
 * protocol on every frame.
 */
interface NeatRacingHostRaceState {
  /**
   * Car 0 - the human's car and the AUTHORITY for every host binding
   * (the chase camera, the HUD, the drift seam, the right-world pass).
   */
  car: PhysicsCarState | null;
  /** The rival fleet (the worker's cars 1..n) in fleet order. */
  rivals: PhysicsCarState[];
  /** The worker's race-rules clone (the HUD's read model). */
  rules: RaceRulesState;
  /** The current frame's keyboard sample (the front-wheel steer source). */
  input: NeatRacingKeyboardSample | null;
}

/**
 * The host-owned AI-mode goal lifecycle state.
 */
interface NeatRacingAiModeGoalState {
  /** The adopted goal session (parked while the UI mode is `manual`). */
  session: NeatRacingAiModeGoalSession;
  /** The sum of car 0's speeds over the CURRENT lap window. */
  speedSum: number;
  /** The number of frames in the CURRENT lap window. */
  frameCount: number;
  /** The number of off-road frames in the CURRENT lap window. */
  offTrackFrameCount: number;
  /** The host's own per-frame steer samples over the CURRENT lap window. */
  steerSeries: number[];
}

/**
 * The host-owned most-forward network panel state.
 */
interface NeatRacingNetworkPanelState {
  /** The mounted panel canvas, or null when the panel is unmounted. */
  canvas: HTMLCanvasElement | null;
  /** The incumbent (sticky) most-forward agent id, or null. */
  incumbentAgentId: number | null;
}

/**
 * Builds the live drive session and starts the RAF-driven drive: keyboard
 * presses are tracked on the window, the composer scene is assembled once
 * (decode + mesh the static voxel assets, per-pass program construction), and
 * every frame runs through the simulation-worker protocol
 * (`sample → worker frame → rebind → lap facts → HUD → panel → composer`).
 *
 * Composition is best-effort within a live render path: when the context
 * cannot compile programs (fake/headless GL surfaces) or the scene assembly
 * throws, the session keeps running against an inert composer that swallows
 * draw frames, so the drive loop (input, the worker protocol, the HUD)
 * stays live
 * without render-path noise. The composer and its passes are disposed by the
 * boot stop handle through the session teardown hook.
 *
 * Step 05 (P2-S4-C) moved the SIMULATION out of the host: the worker owns
 * the spawning, the physics, the race rules and the lap-line detector, and
 * the host owns the input sampling, the composer bindings, the AI-mode goal
 * lifecycle, the most-forward network panel and the teardown. Car 0 is the
 * AUTHORITY for the chase camera, the HUD, the drift seam and the
 * right-world pass; the rivals (cars 1..n) are the worker's fleet.
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
  // Track geometry: the recycled generator with the pinned demo fixture. The
  // resolver feeds the host's off-track measurement and the chord index
  // feeds the host-owned along-track progress meter.
  const trackSpec: TrackSpec = generateTrack(NEATRACING_DEMO_TRACK_GENERATION);
  const surfaceResolver = createTrackSurfaceResolver(trackSpec.segments);
  const chordIndex = buildNeatRacingTrackChordIndex(trackSpec.segments);

  // The host-owned race state: car 0 is the AUTHORITY for the chase camera,
  // the HUD, the drift seam and the right-world pass; the rivals are the
  // worker's fleet (cars 1..n).
  const raceState: NeatRacingHostRaceState = {
    car: null,
    rivals: [],
    rules: createRaceRulesState(),
    input: null,
  };
  // The lap baseline the host adopted at the round boot. The worker's
  // lap-line detector is the SINGLE authority for car 0's forward
  // crossings, so the host only tracks the baseline it compares against.
  let lapBaseline = 0;

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
  // The rival pose bags are allocated ONLY when the fleet GROWS (the array
  // identity and the existing bags stay stable, so the composer's readonly
  // `cars` reference never changes). The parallel spin and progress states
  // keep the SAME length as the bag array.
  const rivalSpinStates: NeatRacingWheelSpinState[] = [];
  const rivalProgressStates: (NeatRacingCarProgressState | null)[] = [];
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
  let wheelSpinBoundSession: NeatRacingSimWorkerSession | null = null;
  const rightWorldCarPosition: [number, number, number] = [0, 0, 0];
  const rightWorldDrawState: RightWorldPassDrawState = {
    viewProjection,
    cameraWorld,
    cameraRight: [1, 0, 0],
    cameraUp: [0, 1, 0],
    timeSeconds: 0,
    carPosition: rightWorldCarPosition,
  };
  // The adaptive-quality controller (AC-035): ONE state instance per drive
  // session, ticked EXACTLY once per rendered frame by the drive loop; the
  // composed frame's bloom off-switch is the RESOLVED (defensively
  // clamped) stage record's value. The driven rAF timestamp is the
  // controller's frame clock; the tracker below publishes a zero delta for
  // the first rendered frame (the controller ignores non-positive deltas).
  const adaptiveQualityState: AdaptiveQualityState =
    createInitialAdaptiveQualityState();
  let adaptiveLastFrameTimestampMs: number | null = null;
  const composerFrame: NeatRacingFrameComposerFrame = {
    viewProjection,
    cameraWorld,
    cameraPose,
    pulses: pulseListBag,
    cars: carsBag,
    rightWorld: rightWorldDrawState,
    // AC-035: the construction-time neutral IS the controller ladder's
    // full-quality stage record (defensively clamped lookup), refreshed
    // per rendered frame from the ticked controller in the drive loop.
    enabled:
      resolveAdaptiveQualityStageClamped(adaptiveQualityState.stage)
        .bloomEnabled,
  };

  // Frame builder: chase camera hovering behind and above the car, looking
  // slightly ahead along the heading. Scene handedness mirrors the top-down
  // math plane (worldZ = -y), so the scene forward is (cos, -sin) and the car
  // yaw carries the nose offset (see NEATRACING_CAR_SCENE_YAW_OFFSET_RAD).
  // The wheel spin accumulates the clamped visual omega per stepped tick
  // (AC-015 strobing cap) instead of replaying the uncapped physical integral.
  const buildComposerFrame = (): NeatRacingFrameComposerFrame => {
    const boundSession = session.simSession;
    const car = raceState.car;
    if (car !== null) {
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
        raceState.input?.steer,
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
      // The rivals (the worker's fleet, cars 1..n) are refreshed in their
      // own stable pose bags; car 0 keeps its own bag above.
      syncRivalPoseBags();
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

  // =====================================================================
  // The simulation-worker protocol wiring (P2-S4-C).
  //
  // The worker owns spawning, the physics fleet, the race rules and the
  // car-0 lap-line detector. The host boots ONE worker session per racing
  // round, feeds it the once-per-frame keyboard sample, and rebinds its
  // composer/HUD state from the per-frame `frames` reply:
  //
  // `sample → worker frame → LAST `frames` reply → rebind → lap facts →
  // HUD → panel → composer`
  //
  // LD-12 (the human takes car 0 in BOTH UI modes): every round is booted
  // with the worker raceMode 'manual' ALWAYS. The worker's own 'ai' race
  // mode would drive car 0 with the worker autopilot seam and steal the
  // human's controls, so it is NOT booted here; the AI-mode goal machine
  // lives in the HOST (ai-mode.goals) and is fed by the host's honest lap
  // windows. Car 0 stays the human's car and the AUTHORITY for the chase
  // camera, the HUD, the drift seam and the right-world pass in both
  // modes; the rivals (cars 1..n) are the worker's fleet.
  //
  // Rivals join at the promotion rungs (NEATRACING_TEAM_LAYOUTS: level 1
  // is a lone car 0 with no rivals, level 2 carries 2 cars, level 3
  // carries 5 cars), so the level-1 boot carries the lone-car fleet and
  // the most-forward network panel rides its idle path until a promoted
  // rebase grows the fleet.
  // =====================================================================

  /**
   * The AI-mode goal lifecycle: the adopted goal-machine session plus the
   * HONEST per-lap-window measurement accumulators. Null until the first
   * switch into 'ai'; parked (kept adopted, never fed again) by a switch
   * back to 'manual' because the per-frame feed gates on the selector
   * mode.
   */
  let goalState: NeatRacingAiModeGoalState | null = null;
  /**
   * The scripted-baseline lap pace (m/s) of the generated demo track:
   * computed ONCE per AI-lane lifecycle at the first AI boot by the pure
   * deterministic probe (up to its pinned 12k-tick budget — boot cost,
   * accepted) and reused across restarts of the same track (never re-run
   * per crossing).
   */
  let scriptedBaselineLapPaceM: number | null = null;
  /** The most-forward network panel state (mounted canvas + incumbent). */
  const panelState: NeatRacingNetworkPanelState = {
    canvas: null,
    incumbentAgentId: null,
  };
  /** The panel's cached idle frame (resolved once per mount). */
  let panelIdleFrame: NetworkVisualizationResolvedFrame | null = null;
  /**
   * The panel's cached 2D context, resolved ONCE per mount; null parks the
   * panel draw (a headless or 2D-less canvas never throws per frame).
   */
  let panelContext: CanvasRenderingContext2D | null = null;
  /** The session identity the panel progress states are bound to. */
  let panelProgressBoundSession: NeatRacingSimWorkerSession | null = null;
  /** The session identity the rival wheel-spin accumulators are bound to. */
  let rivalSpinBoundSession: NeatRacingSimWorkerSession | null = null;

  /**
   * Boots one fresh worker session on the given level.
   *
   * The init request carries the GENERATED track spec (the worker owns
   * spawning from it), the demo rung's fleet size, the pack schema version,
   * the honest opponent snapshot for the rung, the pinned demo track id and
   * the default feature flags. The raceMode is the worker 'manual' ALWAYS
   * (LD-12: see the section banner above).
   *
   * @param bootLevel - The rules level the round races on: at loop boot the
   *   rules' starting level, on restarts the carrier's next-race level.
   * @returns The freshly booted (ready) worker session.
   */
  const bootSimWorkerRound = (
    bootLevel: number,
  ): NeatRacingSimWorkerSession => {
    const simSession = createSimWorkerSession();
    const agentCount = resolveNeatRacingLevelCarCount(bootLevel);
    simSession.handleRequest({
      type: 'init',
      payload: {
        seed: trackSpec.seed,
        agentCount,
        packSchemaVersion: NEATRACING_PACK_SCHEMA_VERSION,
        opponentSnapshot: { level: bootLevel, carCount: agentCount },
        trackId: NEATRACING_DEMO_TRACK_ID,
        featureFlags: NEATRACING_DEFAULT_FEATURE_FLAGS,
        trackSpec,
        raceMode: NEATRACING_DEMO_WORKER_RACE_MODE,
      },
    });
    return simSession;
  };

  /**
   * Restarts the worker round on the given level.
   *
   * The old session receives the terminal `{type:'stop'}` request and is
   * DISCARDED (`'stop'` is terminal — a stopped session is never reused);
   * a fresh session is booted and init'd on the new level. Everything that
   * is bound to the OLD round's identity is reset: the lap baseline, the
   * AI-mode lap window, the panel progress/incumbent rebinding key, and
   * the chase + wheel-spin rebindings (the composer's next frame resets
   * their keyed states against the new session).
   *
   * @param bootLevel - The rules level the fresh round races on.
   */
  const restartSimWorkerRound = (bootLevel: number): void => {
    session.simSession?.handleRequest({ type: 'stop' });
    session.simSession = bootSimWorkerRound(bootLevel);
    // The fresh round's rules start the protocol laps at zero again.
    lapBaseline = 0;
    // The chase + wheel-spin rebinding keys on the session identity: the
    // next frame re-bases both against the NEW session (its tick counter
    // restarts at zero).
    wheelSpinBoundSession = null;
    rivalSpinBoundSession = null;
    chaseFollow.offsetRad = 0;
    chaseFollow.lastTick = 0;
    // The panel progress is a host-owned along-track meter: a fresh round
    // restarts it from the fresh spawn, so the rebinding key and the
    // incumbent reset (keyed states clear on the next panel update).
    panelProgressBoundSession = null;
    panelState.incumbentAgentId = null;
    // The AI-mode lap window (if booted) starts over on the fresh round.
    resetAiModeGoalWindow();
  };

  /**
   * Resets the AI-mode lap-window accumulators (the current crossing's
   * measurement window). The adopted goal session is never touched: only
   * the window measurements reset.
   */
  const resetAiModeGoalWindow = (): void => {
    if (goalState === null) {
      return;
    }
    goalState.speedSum = 0;
    goalState.frameCount = 0;
    goalState.offTrackFrameCount = 0;
    goalState.steerSeries.length = 0;
  };

  /**
   * Boots (or re-boots) the AI-mode goal lifecycle on a switch into 'ai'.
   *
   * Every switch into 'ai' boots a FRESH goal session on the AI-mode start
   * level through the exported mode resolver (its session boot stays on
   * the live wiring path, not test-only); the scripted-baseline probe
   * resolves once per AI-lane lifecycle (the same generated track always
   * scores the same pace, so restarts of the same track reuse the one
   * probe result). The measurement window starts empty.
   *
   * @returns The freshly booted goal state.
   */
  const bootAiModeGoalLifecycle = (): NeatRacingAiModeGoalState => {
    if (scriptedBaselineLapPaceM === null) {
      scriptedBaselineLapPaceM =
        runNeatRacingScriptedBaselineProbeEpisode({ trackSpec }).meanLapSpeed;
    }
    const session = resolveNeatRacingAiModeGoalSessionForMode(
      overlays.selector.mode,
    );
    if (session === null) {
      // The 'ai' switch branch guards the mode, so the resolver's parked
      // 'manual' null cannot fire on the live path; the guard mirrors the
      // suite's contract guard.
      throw new Error('an ai switch must boot a goal session');
    }
    goalState = {
      session,
      speedSum: 0,
      frameCount: 0,
      offTrackFrameCount: 0,
      steerSeries: [],
    };
    return goalState;
  };

  /**
   * Feeds one forward lap crossing into the AI-mode goal machine and
   * applies its outcome (a promoted rebase reboots the worker round on the
   * promoted rung).
   *
   * The measurements are HONEST lap-window facts of the host's OWN car-0
   * traffic: the mean published speed over the window, the fraction of
   * window frames off the road surface, the TP-09 steering-oscillation
   * metric of the host's OWN steer samples (the single metric source law),
   * and the ONE-time scripted baseline. The carrier node count is
   * honestly 0 (E11(a): per-tick rival control and growth live inside the
   * workers; this slice's crossings ride the machine's non-promoting
   * `lapCounted` path and the demos keep racing per LD-10).
   */
  const feedAiModeGoalCrossing = (lapsSinceLastCrossing: number): void => {
    const currentGoal = goalState;
    if (
      currentGoal === null ||
      scriptedBaselineLapPaceM === null ||
      raceState.car === null ||
      currentGoal.frameCount === 0
    ) {
      return;
    }
    const outcome = recordNeatRacingAiModeGoalCrossing(currentGoal.session, {
      forwardCrossing: true,
      lapsSinceLastCrossing,
      nodeCount: NEATRACING_DEMO_EVOLVED_NODE_COUNT,
      drivability: {
        meanForwardSpeed: currentGoal.speedSum / currentGoal.frameCount,
        scriptedBaselineLapPace: scriptedBaselineLapPaceM,
        oscillationMetric: detectSteeringOscillation(currentGoal.steerSeries),
        offTrackFraction:
          currentGoal.offTrackFrameCount / currentGoal.frameCount,
      },
    });
    // The adopted session follows the outcome in EVERY kind.
    currentGoal.session = outcome.session;
    // The measurement window resets after every fed crossing.
    resetAiModeGoalWindow();
    const decision = resolveNeatRacingAiModeRebaseRung(outcome);
    if (decision.rebaseRung !== null) {
      // The promoted rebase: the worker round reboots on the promoted rung
      // and the panel progress restarts from the fresh spawn (restart
      // helper).
      restartSimWorkerRound(decision.rebaseRung);
    }
  };

  /** Feeds the per-frame AI-mode measurement window (mode-gated). */
  const accumulateAiModeGoalWindow = (
    sample: NeatRacingKeyboardSample,
    crossed: boolean,
    lapsSinceLastCrossing: number,
  ): void => {
    const currentGoal = goalState;
    const car = raceState.car;
    // The lifecycle parks by GATE, not by teardown: a 'manual' selector
    // leaves the adopted session adopted and starves the window, so a
    // later 'ai' return boots a FRESH session and an empty window anyway.
    if (
      currentGoal === null ||
      car === null ||
      overlays.selector.mode !== 'ai'
    ) {
      return;
    }
    currentGoal.speedSum += car.speed;
    currentGoal.frameCount += 1;
    if (
      surfaceResolver(car) !== NEATRACING_PHYSICS_SURFACE_ROAD
    ) {
      currentGoal.offTrackFrameCount += 1;
    }
    currentGoal.steerSeries.push(sample.steer);
    if (crossed) {
      feedAiModeGoalCrossing(lapsSinceLastCrossing);
    }
  };

  /**
   * Mounts the most-forward network panel canvas inside the reserved HUD
   * output container (a SIBLING of the HUD's own subtree, under its own
   * id, never a child of the HUD the container hosts) and installs the
   * adapter's resize authority as the ONE sizing owner.
   */
  const mountNetworkPanel = (): void => {
    if (panelState.canvas !== null) {
      return;
    }
    const panelHost = document.getElementById(NEATRACING_HUD_OUTPUT_ID);
    if (panelHost === null) {
      // The reserved container is gone (host shell removed it): degrade
      // with the file's console.warn pattern and keep the loop live.
      console.warn('neatRacing: network panel host container missing');
      return;
    }
    const panelCanvas = document.createElement('canvas');
    panelCanvas.id = NEATRACING_NETWORK_PANEL_CANVAS_ID;
    panelHost.appendChild(panelCanvas);
    const resizeAuthority = installNeatRacingNetworkResize(
      panelCanvas,
      panelHost,
      (): void => {
        // The per-frame draw repaints immediately; the cache drops so the
        // idle frame re-resolves at the resized backing store.
        panelIdleFrame = null;
      },
    );
    session.networkPanelTeardown = (): void => {
      resizeAuthority.uninstall();
      panelCanvas.remove();
    };
    panelState.canvas = panelCanvas;
    panelState.incumbentAgentId = null;
    panelIdleFrame = null;
    panelContext = panelCanvas.getContext('2d');
  };

  /** Removes the network panel and its resize authority (idempotent). */
  const unmountNetworkPanel = (): void => {
    session.networkPanelTeardown?.();
    session.networkPanelTeardown = null;
    panelState.canvas = null;
    panelState.incumbentAgentId = null;
    panelIdleFrame = null;
    panelContext = null;
  };

  /**
   * Draws the most-forward network panel from the current rivals.
   *
   * The candidates are the RIVALS ONLY (car 0 is the human's panel-exempt
   * car): each carries its host-owned along-track progress in meters and
   * an ALWAYS-undefined network (E11(a): the transport-only worker exposes
   * no evolved controller networks this slice — a networkless winner is
   * never replaced by a networked runner-up, review F3 / AC-034; the
   * winner with no network renders the adapter's idle state). Level 1 has
   * no rivals, so the panel rides its idle path entirely.
   */
  const updateNetworkPanel = (): void => {
    if (panelState.canvas === null || panelContext === null) {
      return;
    }
    const boundSession = session.simSession;
    if (panelProgressBoundSession !== boundSession) {
      // The progress meter re-binds to the fresh round: the along-track
      // totals restart from the fresh spawn and the incumbent clears.
      panelProgressBoundSession = boundSession;
      rivalProgressStates.fill(null);
      panelState.incumbentAgentId = null;
    }
    const candidates: NeatRacingNetworkPanelCandidate[] = [];
    for (
      let rivalIndex = 0;
      rivalIndex < raceState.rivals.length;
      rivalIndex += 1
    ) {
      const rival = raceState.rivals[rivalIndex];
      if (rival === undefined) {
        continue;
      }
      const progress = resolveNeatRacingCarAlongProgressM(
        chordIndex,
        rival.x,
        rival.y,
        rivalProgressStates[rivalIndex] ?? null,
      );
      rivalProgressStates[rivalIndex] = progress;
      candidates.push({
        agentId: rivalIndex + 1,
        network: undefined,
        progress: progress.totalM,
      });
    }
    const winner = resolveMostForwardNeatRacingAgent(
      candidates,
      panelState.incumbentAgentId ?? undefined,
    );
    panelState.incumbentAgentId = winner?.agentId ?? null;
    if (winner !== undefined && winner.network !== undefined) {
      // Future-transport path (a real network snapshot resolves a fresh
      // frame each draw); no transport networks exist this slice.
      const resolvedFrame = resolveNeatRacingNetworkPanelVisualizationFrame(
        panelContext,
        winner.network,
      );
      drawNeatRacingNetworkPanelFromFrame(panelContext, resolvedFrame);
      return;
    }
    if (panelIdleFrame === null) {
      panelIdleFrame = resolveNeatRacingNetworkPanelVisualizationFrame(
        panelContext,
        undefined,
      );
    }
    drawNeatRacingNetworkPanelFromFrame(panelContext, panelIdleFrame);
  };

  /**
   * Refreshes the rival pose bags (the worker's fleet, cars 1..n) in place.
   *
   * A bag is allocated ONLY when the fleet GROWS (a promoted rebase); the
   * existing bag identities stay stable, so the composer's readonly
   * `cars` reference never changes. A fresh round with a shorter fleet
   * shrinks the parallel state arrays. Rival team colors cycle BLUE →
   * GREEN → CYAN by rival index; car 0 keeps NEATRACING_TEAM_ID_RED.
   *
   * BOUNDARY: the rivals' steering samples are the worker's internal
   * autopilot state and never travel with the `frames` reply, so the bags
   * render neutral front-wheel steer instead of inventing a sample.
   */
  const syncRivalPoseBags = (): void => {
    const rivals = raceState.rivals;
    while (carsBag.length - 1 < rivals.length) {
      carsBag.push({
        worldX: 0,
        worldZ: 0,
        yawRad: 0,
        teamColorId: NEATRACING_TEAM_ID_BLUE,
        wheelSpinRad: 0,
        steerAngleRad: 0,
        driftYawOffsetRad: 0,
        tractionSeverity: 0,
      });
      rivalSpinStates.push({ accumulatorRad: 0, lastTick: 0 });
      rivalProgressStates.push(null);
    }
    while (carsBag.length - 1 > rivals.length) {
      carsBag.pop();
      rivalSpinStates.pop();
      rivalProgressStates.pop();
    }
    if (rivalSpinBoundSession !== session.simSession) {
      // The rival wheel-spin accumulators re-base onto the fresh round
      // (its tick counter restarts at zero, like car 0's above).
      rivalSpinBoundSession = session.simSession;
      for (const rivalSpin of rivalSpinStates) {
        rivalSpin.accumulatorRad = 0;
        rivalSpin.lastTick = 0;
      }
    }
    for (
      let rivalIndex = 0;
      rivalIndex < rivals.length;
      rivalIndex += 1
    ) {
      const rival = rivals[rivalIndex];
      const rivalPose = carsBag[rivalIndex + 1];
      const rivalSpin = rivalSpinStates[rivalIndex];
      if (rival === undefined || rivalPose === undefined) {
        continue;
      }
      if (rivalSpin !== undefined && rival.tick < rivalSpin.lastTick) {
        rivalSpin.accumulatorRad = 0;
      }
      const spinDeltaTicks =
        rival.tick - (rivalSpin?.lastTick ?? rival.tick);
      if (rivalSpin !== undefined) {
        rivalSpin.lastTick = rival.tick;
        if (spinDeltaTicks > 0) {
          rivalSpin.accumulatorRad += computeVisualWheelSpinDeltaRad(
            spinDeltaTicks,
            NEATRACING_FIXED_TIMESTEP_SECONDS,
            rival.speed,
            wheelRadius,
          );
        }
      }
      fillCarRenderPose(
        rivalPose,
        rival.x,
        carSceneWorldZ(rival.y),
        carSceneYawRad(rival.heading),
        rivalSpin?.accumulatorRad ?? 0,
        NEATRACING_RIVAL_TEAM_COLOR_IDS[
          rivalIndex % NEATRACING_RIVAL_TEAM_COLOR_IDS.length
        ] ?? NEATRACING_TEAM_ID_BLUE,
      );
      // AC-047C / AC-088 for the rivals, too: the pre-computed visual
      // drift pose and the PUBLISHED traction severity ride the worker's
      // fleet snapshot; the host READS both and never re-derives either.
      rivalPose.driftYawOffsetRad = rival.drift.pose.pose;
      rivalPose.tractionSeverity = rival.tractionSeverity;
    }
  };

  // Boot the round the loop opens with: the P1 rules seeding fixes the
  // starting level, and the fresh rules state carries it (`nextRaceLevel`
  // is null until a race-completion boundary sets it).
  session.simSession = bootSimWorkerRound(raceState.rules.level);

  /**
   * Drives ONE frame through the worker protocol.
   *
   * The input samples ONCE per frame (the phase-1 cadence), the worker
   * replies are taken for the LAST `frames` message, and the host rebinds:
   * car 0 into the authority bag, the rivals into their pose bags, the
   * rules into the HUD's read model, the lap facts from the protocol's
   * forward-lap counter, the AI-mode window (mode-gated), the most-forward
   * panel (mode-gated), the adaptive-quality controller tick (exactly once
   * per rendered frame, AC-035) and exactly one composer draw — zero-step
   * frames included.
   *
   * The controller tick consumes the WALL-CLOCK rAF delta (P3-S1-A
   * OBS-9, carried): the RENDERED quality toggles derive from live frame
   * pacing and are therefore not replay-stable, while physics and the
   * chase camera stay deterministic — the tick never feeds the
   * simulation.
   *
   * @param timestampMs - The rAF-style monotonic frame timestamp.
   */
  const driveSimProtocolFrame = (timestampMs: number): void => {
    const simSession = session.simSession;
    if (simSession === null) {
      return;
    }
    const sample = sampleKeyboardInput(presses);
    raceState.input = sample;
    // Step 1: one protocol frame request per drawn frame.
    const responses = simSession.handleRequest({
      type: 'frame',
      payload: { timestampMs, sample },
    });
    // Step 2: the LAST `frames` reply rebinds the host state.
    let framesPayload: NeatRacingSimWorkerFramesPayload | null = null;
    for (const simResponse of responses) {
      if (simResponse.type === 'frames') {
        framesPayload = simResponse.payload;
      }
    }
    if (framesPayload === null) {
      // Degrade (the file's console.warn pattern): a frame without a
      // `frames` reply leaves the previous bindings drawn and the HUD
      // untouched.
      console.warn('neatRacing: simulation worker frames reply missing');
      return;
    }
    raceState.rivals = framesPayload.cars.slice(1);
    raceState.rules = framesPayload.rules;
    const car = framesPayload.cars[0] ?? null;
    raceState.car = car;
    // Step 3: the worker's own car-0 lap authority drives the host's
    // forward-lap facts (single authority: the protocol counter).
    const lapAdvance = resolveNeatRacingLapFactAdvance(
      lapBaseline,
      raceState.rules.lapsCompleted,
    );
    lapBaseline = lapAdvance.lapBaseline;
    // Step 4: the HUD read model updates EVERY drawn frame (the protocol
    // car 0 speed, the rules clone and the PUBLISHED severity).
    if (car !== null) {
      overlays.hud.update({
        speedUnitsPerS: car.speed,
        raceState: raceState.rules,
        tractionSeverity: car.tractionSeverity,
      });
    }
    // Step 5: the AI-mode measurement window (mode-gated; the lap crossing
    // feeds the goal machine and may rebase the round, which swaps the
    // session identity the rebinding keys read on the NEXT frame).
    accumulateAiModeGoalWindow(
      sample,
      lapAdvance.crossed,
      lapAdvance.lapsSinceLastCrossing,
    );
    // Step 6: the most-forward network panel draw (mode-gated).
    if (overlays.selector.mode === 'ai') {
      updateNetworkPanel();
    }
    // Step 7: exactly one composer drawFrame per frame (zero-step frames
    // included), with the adaptive-quality controller ticked EXACTLY once
    // per rendered frame AHEAD of the composer draw (AC-035). The composed
    // frame's bloom off-switch is the RESOLVED stage record's value from
    // that tick; the per-frame path allocates nothing (the stable
    // composer-frame bag is refreshed in place, zero per-frame rule).
    const adaptiveFrameMs =
      adaptiveLastFrameTimestampMs === null
        ? 0
        : timestampMs - adaptiveLastFrameTimestampMs;
    adaptiveLastFrameTimestampMs = timestampMs;
    composerFrame.enabled = resolveAdaptiveQualityStageClamped(
      tickAdaptiveQuality(adaptiveQualityState, adaptiveFrameMs),
    ).bloomEnabled;
    drawComposerFrame(buildComposerFrame());
  };

  // The selector's restart carrier (Phase-2 JSDoc: the host shell owns the
  // restart, in BOTH switch directions). The worker's rules clone right
  // before the switch decides the fresh round's level: the carried
  // next-race level when a race-completion boundary set it, the session
  // level otherwise. A switch into 'ai' boots a fresh goal lifecycle and
  // mounts the panel; a switch into 'manual' parks the lifecycle (the
  // adopted session stays adopted, no further goal facts are fed) and
  // unmounts the panel.
  overlays.selector.onSwitch((): void => {
    restartSimWorkerRound(
      raceState.rules.nextRaceLevel ?? raceState.rules.level,
    );
    if (overlays.selector.mode === 'ai') {
      bootAiModeGoalLifecycle();
      mountNetworkPanel();
      return;
    }
    unmountNetworkPanel();
  });

  session.driveTeardown = disposeComposition;
  return { handleFrame: driveSimProtocolFrame };
}

/**
 * Runs the session-driven render loop.
 *
 * Exactly one `requestAnimationFrame` is requested at boot; each frame hands
 * the timestamp to the drive session, which forwards it through the
 * simulation-worker protocol (sample → worker frame → rebind → HUD → panel →
 * composer) and re-requests the next frame so the loop stays alive. The boot
 * itself carries no accumulator: the worker owns the fixed-timestep
 * accumulator (P2-S4-C), so the host's per-frame entry point is a pure
 * protocol hop. A stopped session never schedules again, so the stop handle
 * terminates the loop deterministically.
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
    // The network panel rides its resize authority + canvas (teardown
    // installed at the panel mount); the worker session receives the
    // terminal `{type:'stop'}` request and every live surface is dropped.
    session.networkPanelTeardown?.();
    session.networkPanelTeardown = null;
    session.simSession?.handleRequest({ type: 'stop' });
    session.simSession = null;
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
