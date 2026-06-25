/**
 * Racing curriculum browser shell.
 *
 * This folder is the browser-side presentation layer for the racing curriculum.
 * It builds a two-region DOM host — track canvas on the left and a focused
 * controller network view on the right — and runs the rendering and control
 * loop. Runtime controls live below the track, inside the canvas region. The
 * host owns only DOM regions and drawing; simulation stepping and controller
 * inference are driven by local services that mirror the worker-authoritative
 * protocol defined in
 * {@link ../workers/simulation-worker/simulation-worker.evolution.types.ts}.
 *
 * The layout intentionally mirrors the Flappy Bird parity shape: a left canvas
 * region, a right sidebar network-visualizer region, and no separate bottom
 * visualizer strip. The right sidebar routes through the shared Flappy network
 * visualizer via the racing adapter in `network-view/`; the host owns hover
 * state, resolved-frame caching, and `installRacingNetworkResize` viewport sync.
 *
 * Read this boundary as the browser-side answer to one practical question:
 * how do you inspect a learned racing controller in the browser without letting
 * DOM concerns leak into physics or evolution? The answer is a thin host that
 * owns stable regions and rendering, while the controller and simulation state
 * are passed in as plain data.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Start["start()"]:::accent --> Host["host/\nDOM and canvas shell"]:::base
 *   Start --> Controller["deterministic controller\n + local physics"]:::base
 *   Host --> Canvas["track canvas\n(left region)"]:::base
 *   Host --> Network["network-view/\n(right region)"]:::base
 *   Network --> Resize["host.resize.service\nviewport sync"]:::base
 * Network --> Tooltip["host.ts\nhover + redraw controller"]:::base
 * ```
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Browser["Main thread browser host"]:::accent --> Regions["two-region layout"]:::base
 *   Regions --> Controls["runtime controls\n(below track)"]:::base
 *   Browser --> Loop["requestAnimationFrame\nfixed-timestep loop"]:::base
 *   Loop --> Worker["worker-ready seam\n(local fallback)"]:::base
 * ```
 *
 * For the browser execution model, see
 * [Web Workers (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)
 * and
 * [Transferable objects (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferring_objects).
 * For the fixed-timestep game-loop pattern, see
 * [Fix Your Timestep! (Gaffer On Games)](https://gafferongames.com/post/fix_your_timestep/).
 *
 * @example
 * ```ts
 * import { start } from './browser-entry/browser-entry';
 *
 * const handle = await start('racing-curriculum-output');
 * // Later: handle.stop();
 * ```
 */

import { createRacingHost } from './host/host';
import type { RacingNetworkHudNodes } from './host/host.types';
import { Network, methods } from '../../../src/browser-entry.ts';
import { generateTrack } from '../track/track.generator';
import type {
  TrackGenerationViewport,
  TrackSpec,
} from '../track/track.generator.types';
import { stepEnvironment } from '../environment/environment.step.service';
import {
  computeWorldTransform,
  createRacingRenderState,
  renderRacingFrame,
} from '../renderer/racing.renderer';
import { resolveSplineSampleFrame } from '../track/track.spline.utils';
import {
  createNgeController,
  type NgeControllerTickEvidence,
  resolveGuidanceAlphaForTier,
} from '../controller/nge.controller';
import {
  createRuntimeAdaptationEngine,
  type RuntimeAdaptationEngine,
  type RuntimeAdaptationTelemetry,
} from '../controller/runtime.adaptation';
import type {
  CarControlOutput,
  EnvironmentState,
  RacingCarState,
  TireStateTuple,
} from '../environment/environment.types';
import { FLAPPY_NEON_PALETTE } from '../../flappy_bird/constants/constants.palette';
import { FLAPPY_MONOSPACE_FONT_FAMILY } from '../../flappy_bird/constants/constants.frame';
import {
  FLAPPY_SCREEN_PADDING_PX,
  FLAPPY_UI_CANVAS_INSET_SHADOW,
  FLAPPY_UI_DOUBLE_PANEL_BORDER,
  FLAPPY_UI_NETWORK_HOST_BACKGROUND,
  FLAPPY_UI_OUTER_FRAME_BACKGROUND,
  FLAPPY_UI_OUTER_FRAME_MIN_SIDE_PADDING_PX,
  FLAPPY_UI_OUTER_FRAME_SIDE_PADDING_OFFSET_PX,
  FLAPPY_UI_UNIFIED_INSET_SHADOW,
} from '../../flappy_bird/constants/constants.layout';
import {
  FLAPPY_HOST_PANEL_PADDING,
  FLAPPY_HOST_STATS_SPLIT_GAP,
  FLAPPY_HOST_TABLE_FONT_SIZE,
  FLAPPY_HOST_TABLE_HOST_PADDING,
} from '../../flappy_bird/browser-entry/host/host.constants';

// ── Animation constants ───────────────────────────────────────────────────────

/** Fixed simulation timestep in milliseconds (60 Hz). */
const FIXED_TIMESTEP_MS = 1000 / 60;

/** Maximum physics catchup steps per rAF tick (prevents spiral-of-death). */
const MAX_CATCHUP_STEPS_PER_FRAME = 4;

/** Max device-pixel ratio used for the baseline canvas backbuffer. */
const MAX_CANVAS_DEVICE_PIXEL_RATIO = 2;
/** Refresh cadence for the focused network canvas in milliseconds. */
const FOCUSED_NETWORK_REFRESH_INTERVAL_MS = 5000;

/** Track determinism key: seed 42, layout v1. */
const DEMO_TRACK_SEED = 42;
const DEMO_TRACK_LAYOUT_VERSION = 1;
/** Active browser harness tier for the NGE controller integration. */
const ACTIVE_CURRICULUM_TIER = 1;
/** Population size sent to the simulation worker during host initialisation. */
const RACING_CURRICULUM_POPULATION_SIZE = 10;
/** Small hidden layer used to pin a deterministic solo driving policy. */
const CONTROLLER_HIDDEN_LAYER_SIZES = [4] as const;
/** Observation channel index for the lateral-offset feature. */
const OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET = 16;
/** Observation channel index for the heading-error feature. */
const OBSERVATION_INDEX_OPTIMAL_LINE_HEADING_ERROR = 17;
/** Observation channel index for the lateral-speed feature. */
const OBSERVATION_INDEX_LATERAL_SPEED = 5;
/** Observation channel index for sin(carHeading) — used as the subtracted baseline for look-ahead differential signals. */
const OBSERVATION_INDEX_SIN_CAR_HEADING = 2;
/**
 * Observation channel index for sin(tangentHeading) at look-ahead offset 1.
 *
 * Look-ahead offset 1 = 18 spline samples ahead ≈ one full track segment.
 * The look-ahead block starts at channel 20; each segment occupies 8 channels;
 * sin(tangent) is at +4 within each group. So: 20 + 1×8 + 4 = 32.
 *
 * Wired against OBSERVATION_INDEX_SIN_CAR_HEADING to form a differential
 * signal: sin(tangentAhead) − sin(carHeading) ≈ sin(headingErrorAhead),
 * giving the controller anticipatory steering before curves arrive.
 * @internal
 */
const OBSERVATION_INDEX_LOOK_AHEAD_NEAR_SIN_TANGENT = 32;

/** Full-health tire tuple used by the solo Tier 1 harness stability guard. */
const FULL_HEALTH_TIRE_STATE = [1, 1, 1, 1] as const;
/** Tier at which live tire wear becomes visible in the browser shell. */
const TIRE_WEAR_START_TIER = 4;
/** Tier 4 packs use a four-car 2v2 grid. */
const TIER_FOUR_TEAM_LAYOUT = [0, 0, 1, 1] as const;
/** Tier 5+ packs use a six-car 3v3 grid. */
const TIER_FIVE_TEAM_LAYOUT = [0, 0, 0, 1, 1, 1] as const;
/** Tier 1 packs use a two-car 1v1 grid so both team guiding lines are visible. */
const TIER_ONE_TEAM_LAYOUT = [0, 1] as const;
/** Default Tier 1 adaptation cadence in fixed-timestep ticks (very frequent). */
const DEFAULT_RUNTIME_ADAPTATION_CADENCE_INTERVAL_TICKS = 1;
/** Minimum mutation intensity while adaptation remains active in higher tiers. */
const MIN_ADAPTATION_MUTATION_STEPS = 1;
/** Maximum mutation intensity for one within-tier adaptation pass. */
const MAX_ADAPTATION_MUTATION_STEPS = 3;
/** Default cadence mode for runtime adaptation controls. */
const DEFAULT_ADAPTATION_CADENCE_MODE = 'ticks' as const;
/** Lightweight trend epsilon used to classify adaptation momentum. */
const IMPROVEMENT_TREND_EPSILON = 0.01;
/** Bounded rolling score window used by runtime adaptation decisions. */
const RUNTIME_ADAPTATION_SCORE_HISTORY_CAP = 48;
/** Default hard cap for mutable runtime controller node count. */
const RUNTIME_ADAPTATION_MAX_NODES = 192;
/** Default hard cap for mutable runtime controller connection count. */
const RUNTIME_ADAPTATION_MAX_CONNECTIONS = 640;
/** Tier 1 keeps adaptation very frequent while bounding mutation pressure. */
const RUNTIME_ADAPTATION_TIER_ONE_MUTATION_COOLDOWN_TICKS = 4;
/** Tier 1 keeps rollback retries short so adaptation remains active while driving. */
const RUNTIME_ADAPTATION_TIER_ONE_ROLLBACK_COOLDOWN_TICKS = 2;
/** Rollback cooldown keeps repeated unsafe retries out of the hot loop. */
const RUNTIME_ADAPTATION_ROLLBACK_COOLDOWN_TICKS = 12;
/** Scale UI threshold controls to the engine's score-delta domain. */
const RUNTIME_ADAPTATION_IMPROVEMENT_THRESHOLD_SCALE = 0.001;

// ── POC worker seam (physics-only delegation) ────────────────────────────────
//
// This seam sends a full EnvironmentState to the worker each tick and receives
// a stepped EnvironmentState back.  Controller inference, evolution, and
// curriculum progress all run on the host (main) thread.
//
// The worker-authoritative protocol (defined in
// simulation-worker.evolution.types.ts) inverts this relationship:
//   Host → Worker: init | request-generation | start-race | request-race-step | stop
//   Worker → Host: generation-ready | race-step | runtime-status | error
//
// Under that protocol the worker owns Team A/B population containers, rolling
// opponent snapshots, controller inference, and race-step snapshot production.
// The host receives compact typed-array race-step frames and renders them at
// display cadence — it never drives simulation ticks directly.
//
// The local types below belong to the physics-only seam.  They will be
// replaced by the worker-authoritative message types once that protocol is
// wired end-to-end.

type RacingWorkerStepRequest = {
  type: 'step';
  requestId: number;
  envState: EnvironmentState;
  control: readonly CarControlOutput[] | CarControlOutput;
};

type RacingWorkerStepResponse = {
  type: 'step-result';
  requestId: number;
  envState: EnvironmentState;
};

type RacingSimulationWorkerScope = {
  onmessage: ((event: MessageEvent<RacingWorkerStepRequest>) => void) | null;
  postMessage(message: RacingWorkerStepResponse): void;
};

type PendingWorkerStep = {
  resolve: (envState: EnvironmentState) => void;
  reject: (reason: Error) => void;
};

// ── Panel text labels ─────────────────────────────────────────────────────────

const CANVAS_TOOLTIP_HEADING = 'Track Playback';

let racingTooltipIdCounter = 0;

const FLAPPY_SELECTOR_BUTTON_PADDING = '6px 8px';
const FLAPPY_SELECTOR_BUTTON_RADIUS_PX = 10;
const FLAPPY_SELECTOR_BUTTON_FONT_SIZE = '11px';
const FLAPPY_SELECTOR_TOOLTIP_OFFSET_PX = 10;
const FLAPPY_SELECTOR_TOOLTIP_MAX_WIDTH_PX = 260;
const FLAPPY_SELECTOR_TOOLTIP_PADDING = '10px 12px';
const FLAPPY_SELECTOR_TOOLTIP_RADIUS_PX = 12;
const FLAPPY_SELECTOR_TOOLTIP_HEADING_FONT_SIZE = '11px';
const FLAPPY_SELECTOR_TOOLTIP_BODY_FONT_SIZE = '10px';
const FLAPPY_SELECTOR_TRANSITION =
  'border-color 120ms ease-out, box-shadow 120ms ease-out, color 120ms ease-out, background 120ms ease-out, transform 120ms ease-out';
const FLAPPY_SELECTOR_TOOLTIP_TRANSITION =
  'opacity 120ms ease-out, transform 120ms ease-out';

/**
 * Public run handle for the racing curriculum browser shell.
 *
 * Returned by `start(...)`. Use `stop()` to cancel the animation loop; await
 * `done` to observe clean teardown.
 */
export interface RacingCurriculumRunHandle {
  /** Resolves when the animation loop has fully stopped. */
  done: Promise<void>;
  /** Returns `true` while the animation loop is still running. */
  isRunning: () => boolean;
  /** Requests a graceful stop of the animation loop. */
  stop: () => void;
}

// ── DOM panel element sets ────────────────────────────────────────────────────

/** Live-updating redraw hook and text node references for the network panel. */
/** Live-updating text node references for the telemetry panel. */
interface TelemetryPanelNodes {
  tickValue: Text;
  lapValue: Text;
  adaptationEnabledValue: Text;
  cadenceModeValue: Text;
  cadenceIntervalValue: Text;
  mutationIntensityValue: Text;
  growthPruneBiasValue: Text;
  commitThresholdValue: Text;
  rollbackSensitivityValue: Text;
  commitCountValue: Text;
  rollbackCountValue: Text;
  recentTrendValue: Text;
  networkSizeValue: Text;
  networkDeltaValue: Text;
  lastChangeReasonValue: Text;
  controlsElement: HTMLDivElement;
  syncRuntimeControls: () => void;
}

type AdaptationCadenceMode = 'laps' | 'ticks';

interface RuntimeTuningConfig {
  adaptationEnabled: boolean;
  cadenceMode: AdaptationCadenceMode;
  cadenceInterval: number;
  mutationIntensity: number;
  growthPruneBias: number;
  commitThreshold: number;
  rollbackSensitivity: number;
}

interface RuntimeTelemetryState {
  commitCount: number;
  rollbackCount: number;
  recentImprovementTrend: string;
  lastChangeReason: string;
  previousLapTick: number | null;
  previousLapDuration: number | null;
  lastLapImprovementRatio: number;
  networkDeltaNodes: number;
  networkDeltaConnections: number;
  adaptationScoreHistory: number[];
}

interface RuntimeAdaptationState {
  engine: RuntimeAdaptationEngine;
  engineConfigSignature: string;
  tuning: RuntimeTuningConfig;
  telemetry: RuntimeTelemetryState;
}

/** Curriculum tier contract from the racing plan ladder. */
type CurriculumTier = 1 | 2 | 3 | 4 | 5 | 6;

/** Observation tier supported by the owner-local controller seam. */
type SupportedObservationTier = 1 | 2 | 3 | 4 | 5;

/** Team index for the browser-local race pack grid. */
type CurriculumTeamIndex = 0 | 1;

/** Track bucket used by the tier-aware browser track generator. */
type TrackSizeBucket = 'small' | 'medium' | 'large';

type LapProgressState = {
  lastClosestSplineSampleIndex: number;
  completedLaps: number;
};

type CurriculumProgressState = {
  tier: CurriculumTier;
  lapProgress: LapProgressState;
};

type CurriculumEpisodeState = {
  trackSpec: TrackSpec;
  envState: EnvironmentState;
};

type TierSignalEvidenceAccumulator = {
  tickCount: number;
  cumulativeLateralErrorNormalized: number;
  cumulativeHeadingAlignment01: number;
  cumulativeCenterGuideNeed01: number;
};

type TierSignalEvidenceSummary = {
  meanLateralErrorNormalized: number;
  meanHeadingAlignment01: number;
  meanCenterGuideNeed01: number;
};

/**
 * Fallback promotion floor while richer co-evolution promotion logic is unavailable.
 *
 * Keep this at 3 laps minimum so tiers cannot end too quickly even if future
 * threshold checks become more permissive.
 * @internal
 */
const LAP_COMPLETIONS_REQUIRED_FOR_TIER_ADVANCE = 3;
/** Highest curriculum tier in the racing plan ladder. */
const MAX_CURRICULUM_TIER: CurriculumTier = 6;
/** Highest tier allowed by the fallback auto-promotion policy. */
const MAX_FALLBACK_AUTOPROMOTION_TIER: CurriculumTier = 4;
/** Highest observation tier implemented in the browser controller seam. */
const MAX_SUPPORTED_OBSERVATION_TIER: SupportedObservationTier = 5;
/** Wrap threshold used to detect one completed lap from nearest spline sample indices. */
const LAP_WRAP_HIGH_WATERMARK_RATIO = 0.75;
/** Wrap threshold used to detect one completed lap from nearest spline sample indices. */
const LAP_WRAP_LOW_WATERMARK_RATIO = 0.25;
/** Edge padding ratio reserved when fitting generated track geometry to the viewport. */
const DEFAULT_TRACK_VIEWPORT_EDGE_PADDING_RATIO = 0.08;

/**
 * Starts the Tier 0 racing curriculum browser demo.
 *
 * Sets up the two-region layout created by {@link createRacingHost} — a track
 * canvas on the left and a focused-controller network view on the right —
 * generates a deterministic track, and launches a `requestAnimationFrame`
 * animation loop with fixed-timestep physics driven by a deterministic NGE
 * controller network.
 *
 * Runtime controls live below the track, inside the canvas region. The network
 * sidebar stays in sync with the viewport through
 * {@link installRacingNetworkResize} and is redrawn from the shared
 * {@link drawRacingNetworkVisualization} racing adapter.
 *
 * @param container - Host element or element id.
 * @returns Lightweight run handle.
 *
 * @example
 * ```ts
 * const handle = await start('racing-curriculum-output');
 * // Later: handle.stop();
 * ```
 * @example
 * ```ts
 * const handle = await start(document.getElementById('racing-output')!);
 * console.log(handle.isRunning);
 * handle.stop();
 * await handle.done;
 * ```
 */
export async function start(
  container: HTMLElement | string = 'racing-curriculum-output',
): Promise<RacingCurriculumRunHandle> {
  // Step 1: Resolve container and build host DOM shell.
  const containerElement = resolveContainerElement(container);
  injectRacingStyles();
  const hostHandle = createRacingHost(containerElement);
  setupCanvasStage(
    hostHandle.canvasRegionElement,
    hostHandle.canvasElement,
    ACTIVE_CURRICULUM_TIER,
  );
  hostHandle.applyViewportLayout(window.innerWidth);
  syncCanvasToDisplaySize(hostHandle.canvasElement);

  // Step 2: Generate the tier-aware track and matching starting grid.
  let episodeState = createCurriculumEpisodeState(
    ACTIVE_CURRICULUM_TIER,
    resolveTrackGenerationViewport(hostHandle.canvasElement),
  );
  let trackSpec = episodeState.trackSpec;

  // Step 3: Initialise the packed race state on the sampled spline lane center.
  let envState = episodeState.envState;

  // Step 4: Initialise the deterministic NGE controller and render state.
  let curriculumProgress = createInitialCurriculumProgress(trackSpec, envState);
  let activeObservationTier = resolveObservationTierForCurriculumTier(
    curriculumProgress.tier,
  );
  let controllerNetwork = createDeterministicRacingControllerNetwork(
    activeObservationTier,
  );
  let controller = createNgeController(controllerNetwork, {
    tier: activeObservationTier,
  });
  let tierSignalEvidenceAccumulator =
    createEmptyTierSignalEvidenceAccumulator();
  let guidanceAlpha = resolveGuidanceAlphaForCurriculumTier(
    curriculumProgress.tier,
  );
  const runtimeAdaptationState = createRuntimeAdaptationState(
    ACTIVE_CURRICULUM_TIER,
  );
  const previousNetworkSize = resolveNetworkSize(controllerNetwork);
  const renderState = createRacingRenderState();
  const simulationWorker = createRacingSimulationWorker();
  simulationWorker?.postMessage({
    type: 'init',
    populationSize: RACING_CURRICULUM_POPULATION_SIZE,
    rngSeed: DEMO_TRACK_SEED,
    tier: ACTIVE_CURRICULUM_TIER,
  });
  const pendingWorkerSteps = new Map<number, PendingWorkerStep>();
  let nextWorkerRequestId = 0;
  const handleWorkerMessage = (
    event: MessageEvent<RacingWorkerStepResponse>,
  ): void => {
    handleRacingWorkerMessage(pendingWorkerSteps, event);
  };
  simulationWorker?.addEventListener('message', handleWorkerMessage);

  // Step 5: Build info panels inside the host regions.
  hostHandle.renderNetworkArchitecture(controllerNetwork);
  const stageCardElement =
    hostHandle.canvasRegionElement.querySelector('.racing-stage-card');
  const runtimeControls = setupRuntimeControls(
    (stageCardElement as HTMLElement | null) ?? hostHandle.canvasRegionElement,
    runtimeAdaptationState.tuning,
  );
  const handleRuntimeTuningKeydown = (keyboardEvent: KeyboardEvent): void => {
    const eventTarget = keyboardEvent.target;
    const isFormElement =
      eventTarget instanceof HTMLInputElement ||
      eventTarget instanceof HTMLSelectElement ||
      eventTarget instanceof HTMLTextAreaElement;
    if (isFormElement) {
      return;
    }

    const tuning = runtimeAdaptationState.tuning;
    let handled = true;
    switch (keyboardEvent.code) {
      case 'KeyA':
        tuning.adaptationEnabled = !tuning.adaptationEnabled;
        break;
      case 'KeyC':
        tuning.cadenceMode = tuning.cadenceMode === 'laps' ? 'ticks' : 'laps';
        break;
      case 'BracketLeft':
        tuning.cadenceInterval = clampInteger(
          tuning.cadenceInterval - 1,
          1,
          360,
        );
        break;
      case 'BracketRight':
        tuning.cadenceInterval = clampInteger(
          tuning.cadenceInterval + 1,
          1,
          360,
        );
        break;
      case 'KeyM':
        tuning.mutationIntensity = clampNumber(
          tuning.mutationIntensity + 0.1,
          0.2,
          3,
        );
        break;
      case 'KeyN':
        tuning.mutationIntensity = clampNumber(
          tuning.mutationIntensity - 0.1,
          0.2,
          3,
        );
        break;
      case 'KeyG':
        tuning.growthPruneBias = clampNumber(
          tuning.growthPruneBias + 0.1,
          -1,
          1,
        );
        break;
      case 'KeyH':
        tuning.growthPruneBias = clampNumber(
          tuning.growthPruneBias - 0.1,
          -1,
          1,
        );
        break;
      case 'KeyK':
        tuning.commitThreshold = clampNumber(
          tuning.commitThreshold + 0.01,
          0,
          0.5,
        );
        break;
      case 'KeyJ':
        tuning.commitThreshold = clampNumber(
          tuning.commitThreshold - 0.01,
          0,
          0.5,
        );
        break;
      case 'KeyR':
        tuning.rollbackSensitivity = clampNumber(
          tuning.rollbackSensitivity + 0.01,
          0,
          0.5,
        );
        break;
      case 'KeyF':
        tuning.rollbackSensitivity = clampNumber(
          tuning.rollbackSensitivity - 0.01,
          0,
          0.5,
        );
        break;
      default:
        handled = false;
    }

    if (!handled) {
      return;
    }

    runtimeAdaptationState.telemetry.lastChangeReason = `manual tuning ${keyboardEvent.code}`;
    runtimeControls.syncRuntimeControls();
    keyboardEvent.preventDefault();
  };
  window.addEventListener('keydown', handleRuntimeTuningKeydown);

  // Step 6: Keep layout mode and canvas backbuffer in sync with the viewport.
  const handleViewportResize = (): void => {
    hostHandle.applyViewportLayout(window.innerWidth);
    syncCanvasToDisplaySize(hostHandle.canvasElement);
  };
  handleViewportResize();
  window.addEventListener('resize', handleViewportResize);

  // Step 7: Launch animation loop.
  let running = true;
  let animationFrameId = 0;
  let lastFrameTimestampMs: number | null = null;
  let accumulatedMs = 0;
  let initialControlTick = controller.computeControlWithEvidence(
    envState,
    trackSpec,
  );
  let lastControlOutput = initialControlTick.control;

  const animationStep = async (nowMs: number): Promise<void> => {
    if (!running) return;

    // Compute elapsed time since last frame, capped at 4 steps to prevent spiral.
    if (lastFrameTimestampMs !== null) {
      accumulatedMs += nowMs - lastFrameTimestampMs;
    }
    lastFrameTimestampMs = nowMs;

    let stepsThisFrame = 0;
    while (
      accumulatedMs >= FIXED_TIMESTEP_MS &&
      stepsThisFrame < MAX_CATCHUP_STEPS_PER_FRAME
    ) {
      const previousCurriculumTier = curriculumProgress.tier;
      const previousCompletedLaps =
        curriculumProgress.lapProgress.completedLaps;
      const controlTickResult = controller.computeControlWithEvidence(
        envState,
        trackSpec,
      );
      lastControlOutput = controlTickResult.control;
      tierSignalEvidenceAccumulator = collectTierSignalEvidence(
        tierSignalEvidenceAccumulator,
        controlTickResult.evidence,
      );
      const steppedEnvironmentState = simulationWorker
        ? await requestRacingWorkerStep(
            simulationWorker,
            pendingWorkerSteps,
            ++nextWorkerRequestId,
            envState,
            resolveControlFanOut(lastControlOutput, envState.cars?.length ?? 1),
          )
        : stepEnvironment(
            envState,
            resolveControlFanOut(lastControlOutput, envState.cars?.length ?? 1),
          );
      envState = stabilizeCurriculumTierTireGrip(
        steppedEnvironmentState,
        curriculumProgress.tier,
      );
      curriculumProgress = resolveNextCurriculumProgressState(
        curriculumProgress,
        trackSpec,
        envState,
      );
      updateRuntimeImprovementTrend(
        runtimeAdaptationState.telemetry,
        previousCompletedLaps,
        curriculumProgress.lapProgress.completedLaps,
        envState.tick,
      );
      const didRefreshRuntimeAdaptationEngine = refreshRuntimeAdaptationEngine(
        runtimeAdaptationState,
        curriculumProgress.tier,
      );
      if (didRefreshRuntimeAdaptationEngine) {
        runtimeAdaptationState.telemetry.lastChangeReason =
          'runtime adaptation engine refreshed';
      }

      if (curriculumProgress.tier !== previousCurriculumTier) {
        // Step 1: Rebuild the controller with the promoted tier observation width.
        activeObservationTier = resolveObservationTierForCurriculumTier(
          curriculumProgress.tier,
        );
        controllerNetwork = remapControllerNetworkForObservationTier(
          controllerNetwork,
          activeObservationTier,
        );
        controller = createNgeController(controllerNetwork, {
          tier: activeObservationTier,
        });
        hostHandle.renderNetworkArchitecture(controllerNetwork);
        // Step 2: Rebuild the track and race-local state for the promoted tier.
        episodeState = createCurriculumEpisodeState(
          curriculumProgress.tier,
          resolveTrackGenerationViewport(hostHandle.canvasElement),
        );
        trackSpec = episodeState.trackSpec;
        envState = stabilizeCurriculumTierTireGrip(
          episodeState.envState,
          curriculumProgress.tier,
        );
        // Step 3: Refresh control output at the promoted tier/start-state seam.
        initialControlTick = controller.computeControlWithEvidence(
          envState,
          trackSpec,
        );
        lastControlOutput = initialControlTick.control;
        tierSignalEvidenceAccumulator =
          createEmptyTierSignalEvidenceAccumulator();
        curriculumProgress = {
          ...curriculumProgress,
          lapProgress: createInitialLapProgress(trackSpec, envState),
        };
        const promotedNetworkSize = resolveNetworkSize(controllerNetwork);
        runtimeAdaptationState.telemetry.networkDeltaNodes =
          promotedNetworkSize.nodes - previousNetworkSize.nodes;
        runtimeAdaptationState.telemetry.networkDeltaConnections =
          promotedNetworkSize.connections - previousNetworkSize.connections;
        previousNetworkSize.nodes = promotedNetworkSize.nodes;
        previousNetworkSize.connections = promotedNetworkSize.connections;
        runtimeAdaptationState.telemetry.lastChangeReason =
          'tier promotion remap';
        refreshRuntimeAdaptationEngine(
          runtimeAdaptationState,
          curriculumProgress.tier,
        );
        runtimeAdaptationState.engine.reset();
        runtimeAdaptationState.telemetry.adaptationScoreHistory.length = 0;
        guidanceAlpha = resolveGuidanceAlphaForCurriculumTier(
          curriculumProgress.tier,
        );
        setupCanvasStage(
          hostHandle.canvasRegionElement,
          hostHandle.canvasElement,
          curriculumProgress.tier,
        );
      } else {
        const tierSignalEvidenceSummary = summarizeTierSignalEvidence(
          tierSignalEvidenceAccumulator,
        );
        const didCommitRuntimeAdaptation = applyWithinTierAdaptation(
          controllerNetwork,
          curriculumProgress.tier,
          runtimeAdaptationState,
          tierSignalEvidenceSummary,
          previousNetworkSize,
          envState.tick,
          curriculumProgress.lapProgress.completedLaps,
        );
        tierSignalEvidenceAccumulator =
          createEmptyTierSignalEvidenceAccumulator();
        if (didCommitRuntimeAdaptation) {
          hostHandle.renderNetworkArchitecture(controllerNetwork);
        }
      }

      accumulatedMs -= FIXED_TIMESTEP_MS;
      stepsThisFrame++;
    }

    // Render the latest physics state.
    syncCanvasToDisplaySize(hostHandle.canvasElement);
    const worldTransform = computeWorldTransform(
      hostHandle.canvasElement,
      trackSpec,
    );
    renderRacingFrame(
      hostHandle.canvasElement,
      trackSpec,
      envState,
      renderState,
      worldTransform,
      { guidanceAlpha },
    );

    // Update telemetry readouts (text-only, no innerHTML churn).
    updateTelemetryPanelNodes(
      runtimeControls,
      runtimeAdaptationState,
      controllerNetwork,
      envState,
      curriculumProgress.lapProgress.completedLaps,
      hostHandle.networkHud,
    );

    animationFrameId = requestAnimationFrame(animationStep);
  };

  hostHandle.rootElement.dataset.running = 'true';
  animationFrameId = requestAnimationFrame(animationStep);
  const focusedNetworkRefreshIntervalId = window.setInterval(() => {
    if (!running) {
      return;
    }

    hostHandle.renderNetworkArchitecture(controllerNetwork);
  }, FOCUSED_NETWORK_REFRESH_INTERVAL_MS);

  const handle: RacingCurriculumRunHandle = {
    done: Promise.resolve(),
    isRunning: () => running,
    stop: () => {
      if (!running) return;
      running = false;
      window.removeEventListener('resize', handleViewportResize);
      window.removeEventListener('keydown', handleRuntimeTuningKeydown);
      hostHandle.resizeRedrawController.uninstall();
      window.clearInterval(focusedNetworkRefreshIntervalId);
      cancelAnimationFrame(animationFrameId);
      simulationWorker?.removeEventListener('message', handleWorkerMessage);
      simulationWorker?.terminate();
      pendingWorkerSteps.clear();
      hostHandle.rootElement.dataset.running = 'false';
    },
  };

  return handle;
}

// ── Private helpers ───────────────────────────────────────────────────────────

/**
 * Resolves a container target into a concrete host element.
 *
 * @param container - Host element or element id.
 * @returns Resolved host element.
 * @internal
 */
function resolveContainerElement(container: HTMLElement | string): HTMLElement {
  if (typeof container !== 'string') {
    return container;
  }

  const resolvedContainer = document.getElementById(container);

  if (resolvedContainer === null) {
    throw new RangeError(
      `Racing curriculum container "${container}" was not found.`,
    );
  }

  return resolvedContainer;
}

/**
 * Injects a `<style>` block into `document.head` with all racing-host
 * layout and panel CSS.
 *
 * Idempotent: will not inject if already present (checked by id attribute).
 * @internal
 */
function injectRacingStyles(): void {
  if (document.getElementById('racing-curriculum-styles')) return;

  const outerFrameSidePaddingPx = Math.max(
    FLAPPY_UI_OUTER_FRAME_MIN_SIDE_PADDING_PX,
    FLAPPY_SCREEN_PADDING_PX - FLAPPY_UI_OUTER_FRAME_SIDE_PADDING_OFFSET_PX,
  );

  const styleElement = document.createElement('style');
  styleElement.id = 'racing-curriculum-styles';
  styleElement.textContent = `
    :root {
      --racing-host-background: ${FLAPPY_UI_OUTER_FRAME_BACKGROUND};
      --racing-panel-background: ${FLAPPY_NEON_PALETTE.hudPanelBackground};
      --racing-network-background: ${FLAPPY_UI_NETWORK_HOST_BACKGROUND};
      --racing-stage-background: ${FLAPPY_NEON_PALETTE.background};
      --racing-panel-border-double: ${FLAPPY_UI_DOUBLE_PANEL_BORDER};
      --racing-panel-shadow: ${FLAPPY_UI_UNIFIED_INSET_SHADOW};
      --racing-canvas-shadow: ${FLAPPY_UI_CANVAS_INSET_SHADOW};
      --racing-text: ${FLAPPY_NEON_PALETTE.hudText};
      --racing-text-muted: rgba(159, 220, 255, 0.78);
      --racing-accent: ${FLAPPY_NEON_PALETTE.hudAccent};
      --racing-highlight: ${FLAPPY_NEON_PALETTE.statusText};
      --racing-border-color: ${FLAPPY_NEON_PALETTE.hudPanelBorder};
      --racing-mono: ${FLAPPY_MONOSPACE_FONT_FAMILY};
      --racing-panel-padding: ${FLAPPY_HOST_PANEL_PADDING};
      --racing-card-gap: ${FLAPPY_HOST_STATS_SPLIT_GAP};
      --racing-card-title-size: ${FLAPPY_HOST_TABLE_FONT_SIZE};
      --racing-table-padding: ${FLAPPY_HOST_TABLE_HOST_PADDING};
      --racing-wide-layout-columns: minmax(0, 1.7fr) minmax(300px, 0.92fr);
      --racing-control-padding: ${FLAPPY_SELECTOR_BUTTON_PADDING};
      --racing-control-radius: ${FLAPPY_SELECTOR_BUTTON_RADIUS_PX}px;
      --racing-control-font-size: ${FLAPPY_SELECTOR_BUTTON_FONT_SIZE};
      --racing-control-background: rgba(6, 11, 20, 0.55);
      --racing-control-idle-shadow: inset 0 0 6px rgba(15, 181, 255, 0.08);
      --racing-control-hover-shadow: 0 0 8px rgba(15, 181, 255, 0.72), inset 0 0 10px rgba(0, 229, 255, 0.16);
    }
    .racing-host {
      display: grid;
      grid-template-columns: var(--racing-wide-layout-columns);
      grid-template-rows: minmax(0, 1fr);
      gap: var(--racing-card-gap);
      width: 100%;
      height: 100%;
      min-height: 100%;
      box-sizing: border-box;
      align-items: stretch;
      padding: 0 ${outerFrameSidePaddingPx}px ${outerFrameSidePaddingPx}px ${outerFrameSidePaddingPx}px;
      border: var(--racing-panel-border-double);
      background: var(--racing-host-background);
      box-shadow: var(--racing-panel-shadow);
      overflow: hidden;
    }
    .racing-host--narrow {
      grid-template-columns: 1fr;
      grid-template-rows: auto auto;
    }
    .racing-host__region {
      min-width: 0;
      min-height: 0;
    }
    .racing-host__region--canvas {
      grid-column: 1;
      grid-row: 1;
      display: flex;
      min-height: clamp(420px, 72dvh, 860px);
    }
    .racing-stage-card,
    .racing-card {
      display: grid;
      gap: 12px;
      padding: var(--racing-panel-padding);
      border: var(--racing-panel-border-double);
      background: var(--racing-panel-background);
      box-shadow: var(--racing-panel-shadow);
      box-sizing: border-box;
      min-height: 0;
    }
    .racing-stage-card {
      grid-template-rows: auto auto auto minmax(0, 1fr) auto;
      width: 100%;
      overflow: hidden;
    }
    .racing-stage-card__subtitle {
      color: var(--racing-text-muted);
      font-family: var(--racing-mono);
      font-size: 10px;
      letter-spacing: 0.06em;
      line-height: 1.45;
      text-transform: uppercase;
    }
    .racing-stage-card__meta {
      display: flex;
      flex-wrap: wrap;
      gap: var(--racing-card-gap);
    }
    .racing-stage-card__footer {
      color: var(--racing-text-muted);
      font-family: var(--racing-mono);
      font-size: 10px;
      line-height: 1.45;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .racing-stage {
      position: relative;
      min-height: 0;
      border: var(--racing-panel-border-double);
      overflow: hidden;
      background: linear-gradient(180deg, var(--racing-network-background), var(--racing-stage-background));
      box-shadow: var(--racing-canvas-shadow);
    }
    .racing-host__region--canvas canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .racing-host__region--network {
      grid-column: 2;
      grid-row: 1;
      display: flex;
      min-height: clamp(420px, 48dvh, 860px);
    }
    .racing-host--narrow .racing-host__region--network {
      grid-column: 1;
      grid-row: 2;
    }
    .racing-network-canvas-host {
      position: relative;
      width: 100%;
      height: 100%;
      min-height: 0;
      border: var(--racing-panel-border-double);
      background: var(--racing-network-background);
      box-shadow: var(--racing-canvas-shadow);
      overflow: hidden;
    }
    .racing-network-canvas {
      width: 100%;
      height: 100%;
      display: block;
      cursor: crosshair;
    }
    .racing-network-canvas-wrapper {
      position: relative;
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
      overflow: hidden;
      border-top: 1px solid rgba(15, 181, 255, 0.34);
    }
    .racing-network-hud {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 10px var(--racing-panel-padding);
      background: linear-gradient(180deg, rgba(0, 21, 34, 0.96), rgba(4, 11, 19, 0.99));
      border-bottom: 1px solid rgba(15, 181, 255, 0.34);
      box-shadow: inset 0 0 12px rgba(15, 181, 255, 0.08), 0 0 10px rgba(15, 181, 255, 0.12);
    }
    .racing-network-hud__cell {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
    }
    .racing-network-hud__label {
      color: var(--racing-text-muted);
      font-family: var(--racing-mono);
      font-size: 9px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .racing-network-hud__value {
      color: var(--racing-text);
      font-family: var(--racing-mono);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .racing-network-hud__run {
      color: var(--racing-accent);
      text-shadow: 0 0 8px rgba(255, 154, 46, 0.45);
    }
    .racing-network-hud__status {
      color: var(--racing-highlight);
      text-shadow: 0 0 8px rgba(255, 92, 255, 0.45);
    }
    .racing-network-help-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 8px var(--racing-panel-padding);
      background: rgba(0, 21, 34, 0.78);
      border-top: 1px solid rgba(15, 181, 255, 0.18);
    }
    .racing-network-help-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border: 1px solid rgba(15, 181, 255, 0.34);
      border-radius: var(--racing-control-radius);
      background: rgba(6, 11, 20, 0.72);
      color: var(--racing-text-muted);
      font-family: var(--racing-mono);
      font-size: 9px;
      letter-spacing: 0.04em;
      line-height: 1.4;
      box-shadow: var(--racing-control-idle-shadow);
    }
    .racing-network-help-chip__emoji {
      font-size: 10px;
      line-height: 1;
    }
    .racing-network-help-chip__text {
      text-transform: uppercase;
    }
    .racing-side-stack,
    .racing-lower-panels {
      display: grid;
      gap: var(--racing-card-gap);
      width: 100%;
    }
    .racing-side-stack {
      grid-template-rows: auto auto;
      align-content: start;
    }
    .racing-lower-panels {
      grid-template-columns: var(--racing-wide-layout-columns);
    }
    .racing-host--narrow .racing-lower-panels {
      grid-template-columns: 1fr;
    }
    .racing-card--stretch {
      min-height: 100%;
      align-content: start;
    }
    .racing-card__body {
      display: grid;
      gap: 10px;
    }
    .racing-panel__header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding-bottom: 6px;
      border-bottom: 1px solid rgba(15, 181, 255, 0.34);
    }
    .racing-panel__title {
      color: var(--racing-accent);
      font-family: var(--racing-mono);
      font-size: var(--racing-card-title-size);
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .racing-panel__row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 4px 0;
      border-bottom: 1px solid rgba(15, 181, 255, 0.12);
    }
    .racing-panel__label {
      color: var(--racing-text-muted);
      font-family: var(--racing-mono);
      font-size: 11px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .racing-panel__value {
      color: var(--racing-text);
      font-family: var(--racing-mono);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-align: right;
      text-transform: uppercase;
    }
    .racing-panel__note {
      color: var(--racing-text-muted);
      font-family: var(--racing-mono);
      font-size: 10px;
      line-height: 1.45;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .racing-telemetry {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 16px;
    }
    .racing-controls {
      display: grid;
      gap: 8px;
    }
    .racing-control-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(120px, 1fr) auto;
      align-items: center;
      gap: 10px;
      padding: 4px 0;
      border-bottom: 1px solid rgba(15, 181, 255, 0.12);
    }
    .racing-control-row .racing-panel__label {
      font-size: 10px;
    }
    .racing-control-input {
      width: 100%;
      min-height: 24px;
      padding: 2px 6px;
      border: 1px solid var(--racing-border-color);
      border-radius: 4px;
      background: rgba(4, 12, 24, 0.9);
      color: var(--racing-text);
      font-family: var(--racing-mono);
      font-size: 10px;
      letter-spacing: 0.05em;
    }
    .racing-control-value {
      color: var(--racing-accent);
      font-family: var(--racing-mono);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-align: right;
      text-transform: uppercase;
      min-width: 56px;
    }
    .racing-status-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: var(--racing-control-padding);
      border: 1px solid var(--racing-border-color);
      border-radius: var(--racing-control-radius);
      background: var(--racing-control-background);
      color: var(--racing-text);
      font-family: var(--racing-mono);
      font-size: var(--racing-control-font-size);
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      white-space: nowrap;
      box-shadow: var(--racing-control-idle-shadow);
    }
    .racing-status-chip__label {
      color: var(--racing-accent);
    }
    .racing-callout {
      padding: ${FLAPPY_HOST_TABLE_HOST_PADDING};
      border: 1px solid rgba(15, 181, 255, 0.34);
      background: rgba(0, 21, 34, 0.76);
      color: var(--racing-text-muted);
      font-family: var(--racing-mono);
      font-size: 10px;
      line-height: 1.45;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      box-shadow: var(--racing-control-idle-shadow);
    }
    .racing-help {
      position: relative;
      display: inline-flex;
      align-items: center;
    }
    .racing-help__button {
      min-width: 30px;
      padding: var(--racing-control-padding);
      border: 1px solid var(--racing-border-color);
      border-radius: var(--racing-control-radius);
      background: var(--racing-control-background);
      color: var(--racing-text);
      font-family: var(--racing-mono);
      font-size: var(--racing-control-font-size);
      font-weight: 700;
      letter-spacing: 0.08em;
      cursor: help;
      box-shadow: var(--racing-control-idle-shadow);
      transition: ${FLAPPY_SELECTOR_TRANSITION};
      text-transform: uppercase;
    }
    .racing-help:hover .racing-help__button,
    .racing-help:focus-within .racing-help__button {
      transform: translateY(-1px);
      box-shadow: var(--racing-control-hover-shadow);
    }
    .racing-tooltip {
      position: absolute;
      left: 50%;
      bottom: calc(100% + ${FLAPPY_SELECTOR_TOOLTIP_OFFSET_PX}px);
      width: max-content;
      max-width: min(${FLAPPY_SELECTOR_TOOLTIP_MAX_WIDTH_PX}px, calc(100vw - 48px));
      padding: ${FLAPPY_SELECTOR_TOOLTIP_PADDING};
      border: 1px solid var(--racing-border-color);
      border-radius: ${FLAPPY_SELECTOR_TOOLTIP_RADIUS_PX}px;
      background: linear-gradient(180deg, rgba(0, 21, 34, 0.96), rgba(4, 11, 19, 0.99));
      box-shadow: 0 0 14px rgba(15, 181, 255, 0.34), inset 0 0 12px rgba(15, 181, 255, 0.08);
      backdrop-filter: blur(6px);
      color: var(--racing-text);
      opacity: 0;
      pointer-events: none;
      transform: translate(-50%, 6px);
      transition: ${FLAPPY_SELECTOR_TOOLTIP_TRANSITION};
      z-index: 20;
    }
    .racing-help:hover .racing-tooltip,
    .racing-help:focus-within .racing-tooltip {
      opacity: 1;
      transform: translate(-50%, 0);
    }
    .racing-tooltip__arrow {
      position: absolute;
      left: 50%;
      bottom: -6px;
      width: 12px;
      height: 12px;
      transform: translateX(-50%) rotate(45deg);
      background: rgba(0, 21, 34, 0.98);
      border-right: 1px solid var(--racing-border-color);
      border-bottom: 1px solid var(--racing-border-color);
      box-shadow: 0 0 10px rgba(15, 181, 255, 0.26);
    }
    .racing-tooltip__heading {
      margin-bottom: 6px;
      color: var(--racing-accent);
      font-family: var(--racing-mono);
      font-size: ${FLAPPY_SELECTOR_TOOLTIP_HEADING_FONT_SIZE};
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      text-shadow: 0 0 8px rgba(255, 154, 46, 0.35);
    }
    .racing-tooltip__line {
      color: var(--racing-text);
      font-family: var(--racing-mono);
      font-size: ${FLAPPY_SELECTOR_TOOLTIP_BODY_FONT_SIZE};
      line-height: 1.45;
      white-space: normal;
    }
    .racing-network-tooltip {
      position: fixed;
      top: 0;
      left: 0;
      z-index: 100;
      padding: ${FLAPPY_SELECTOR_TOOLTIP_PADDING};
      border: 1px solid var(--racing-border-color);
      border-radius: ${FLAPPY_SELECTOR_TOOLTIP_RADIUS_PX}px;
      background: linear-gradient(180deg, rgba(0, 21, 34, 0.96), rgba(4, 11, 19, 0.99));
      box-shadow: 0 0 14px rgba(15, 181, 255, 0.34), inset 0 0 12px rgba(15, 181, 255, 0.08);
      color: var(--racing-text);
      font-family: var(--racing-mono);
      font-size: ${FLAPPY_SELECTOR_TOOLTIP_BODY_FONT_SIZE};
      line-height: 1.45;
      pointer-events: none;
      display: none;
      white-space: nowrap;
    }
    @media (max-width: 960px) {
      .racing-stage-card,
      .racing-card {
        padding: ${FLAPPY_HOST_PANEL_PADDING};
      }
      .racing-host__region--canvas {
        min-height: clamp(360px, 54dvh, 640px);
      }
    }
  `;
  document.head.append(styleElement);
}

/**
 * Rebuilds the canvas region into a proper stage card with explanatory HUD chips.
 *
 * @param region - Canvas host region.
 * @param canvasElement - Playback canvas element to place inside the stage.
 * @internal
 */
function setupCanvasStage(
  region: HTMLElement,
  canvasElement: HTMLCanvasElement,
  tier: CurriculumTier,
): void {
  region.replaceChildren();
  const stageNarrative = resolveStageNarrativeForTier(tier);

  const stageCardElement = document.createElement('div');
  stageCardElement.className = 'racing-stage-card';

  const headerElement = createPanelHeader(
    'Track Playback',
    CANVAS_TOOLTIP_HEADING,
    stageNarrative.tooltipLines,
  );
  const subtitleElement = document.createElement('div');
  subtitleElement.className = 'racing-stage-card__subtitle';
  subtitleElement.textContent = stageNarrative.subtitle;

  const metaElement = document.createElement('div');
  metaElement.className = 'racing-stage-card__meta';
  metaElement.append(
    createStatusChip('Controller', 'Live NGE controller'),
    createStatusChip('Track', 'Spline-smoothed visual'),
    createStatusChip('Seed', '42 • v1 • medium'),
  );

  const stageElement = document.createElement('div');
  stageElement.className = 'racing-stage';
  stageElement.append(canvasElement);

  const footerElement = document.createElement('div');
  footerElement.className = 'racing-stage-card__footer';
  footerElement.textContent = stageNarrative.footer;

  stageCardElement.append(
    headerElement,
    subtitleElement,
    metaElement,
    stageElement,
    footerElement,
  );
  region.append(stageCardElement);
}

/**
 * Builds the live runtime-tuning controls and telemetry readouts.
 * Builds the live runtime-tuning controls and telemetry readouts.
 *
 * The controls live inside the canvas region below the stage, matching the
 * Flappy Bird parity layout where runtime widgets share the left column.
 *
 * @param region - The stage-card body element inside the canvas region.
 * @param tuningConfig - Mutable runtime tuning configuration.
 * @returns References to live-updating telemetry text nodes plus a sync hook.
 * @internal
 */
function setupRuntimeControls(
  region: HTMLElement,
  tuningConfig: RuntimeTuningConfig,
): TelemetryPanelNodes {
  const tickValue = document.createTextNode('0');
  const lapValue = document.createTextNode('0');
  const adaptationEnabledValue = document.createTextNode('on');
  const cadenceModeValue = document.createTextNode('laps');
  const cadenceIntervalValue = document.createTextNode(
    String(DEFAULT_RUNTIME_ADAPTATION_CADENCE_INTERVAL_TICKS),
  );
  const mutationIntensityValue = document.createTextNode('1.0');
  const growthPruneBiasValue = document.createTextNode('0.0');
  const commitThresholdValue = document.createTextNode('0.06');
  const rollbackSensitivityValue = document.createTextNode('0.45');
  const commitCountValue = document.createTextNode('0');
  const rollbackCountValue = document.createTextNode('0');
  const recentTrendValue = document.createTextNode('flat');
  const networkSizeValue = document.createTextNode('N0 / C0');
  const networkDeltaValue = document.createTextNode('ΔN0 / ΔC0');
  const lastChangeReasonValue = document.createTextNode('startup baseline');

  const runtimeCard = createPanelCard('Runtime Tuning', 'Runtime Controls', [
    'Tune adaptation behavior live while the simulation loop keeps running.',
    'The right column mirrors active values so keyboard or pointer changes stay visible.',
  ]);
  const controlsElement = document.createElement('div');
  controlsElement.className = 'racing-controls';
  const adaptationEnabledInput = document.createElement('input');
  adaptationEnabledInput.type = 'checkbox';
  adaptationEnabledInput.className = 'racing-control-input';
  const cadenceModeInput = document.createElement('select');
  cadenceModeInput.className = 'racing-control-input';
  cadenceModeInput.append(
    new Option('Laps', 'laps'),
    new Option('Ticks', 'ticks'),
  );
  const cadenceIntervalInput = document.createElement('input');
  cadenceIntervalInput.type = 'range';
  cadenceIntervalInput.min = '1';
  cadenceIntervalInput.max = '360';
  cadenceIntervalInput.step = '1';
  cadenceIntervalInput.className = 'racing-control-input';
  const mutationIntensityInput = document.createElement('input');
  mutationIntensityInput.type = 'range';
  mutationIntensityInput.min = '0.2';
  mutationIntensityInput.max = '3';
  mutationIntensityInput.step = '0.1';
  mutationIntensityInput.className = 'racing-control-input';
  const growthPruneBiasInput = document.createElement('input');
  growthPruneBiasInput.type = 'range';
  growthPruneBiasInput.min = '-1';
  growthPruneBiasInput.max = '1';
  growthPruneBiasInput.step = '0.1';
  growthPruneBiasInput.className = 'racing-control-input';
  const commitThresholdInput = document.createElement('input');
  commitThresholdInput.type = 'range';
  commitThresholdInput.min = '0';
  commitThresholdInput.max = '0.5';
  commitThresholdInput.step = '0.01';
  commitThresholdInput.className = 'racing-control-input';
  const rollbackSensitivityInput = document.createElement('input');
  rollbackSensitivityInput.type = 'range';
  rollbackSensitivityInput.min = '0';
  rollbackSensitivityInput.max = '0.5';
  rollbackSensitivityInput.step = '0.01';
  rollbackSensitivityInput.className = 'racing-control-input';

  const syncControlInputsFromTuning = (): void => {
    adaptationEnabledInput.checked = tuningConfig.adaptationEnabled;
    cadenceModeInput.value = tuningConfig.cadenceMode;
    cadenceIntervalInput.value = String(tuningConfig.cadenceInterval);
    mutationIntensityInput.value = tuningConfig.mutationIntensity.toFixed(1);
    growthPruneBiasInput.value = tuningConfig.growthPruneBias.toFixed(1);
    commitThresholdInput.value = tuningConfig.commitThreshold.toFixed(2);
    rollbackSensitivityInput.value =
      tuningConfig.rollbackSensitivity.toFixed(2);
  };

  const syncTuningReadout = (): void => {
    adaptationEnabledValue.textContent = tuningConfig.adaptationEnabled
      ? 'on'
      : 'off';
    cadenceModeValue.textContent = tuningConfig.cadenceMode;
    cadenceIntervalValue.textContent = String(tuningConfig.cadenceInterval);
    mutationIntensityValue.textContent =
      tuningConfig.mutationIntensity.toFixed(1);
    growthPruneBiasValue.textContent = tuningConfig.growthPruneBias.toFixed(1);
    commitThresholdValue.textContent = tuningConfig.commitThreshold.toFixed(2);
    rollbackSensitivityValue.textContent =
      tuningConfig.rollbackSensitivity.toFixed(2);
  };

  const syncRuntimeControls = (): void => {
    syncControlInputsFromTuning();
    syncTuningReadout();
  };

  syncControlInputsFromTuning();

  adaptationEnabledInput.addEventListener('change', () => {
    tuningConfig.adaptationEnabled = adaptationEnabledInput.checked;
    syncTuningReadout();
  });
  cadenceModeInput.addEventListener('change', () => {
    tuningConfig.cadenceMode =
      cadenceModeInput.value as RuntimeTuningConfig['cadenceMode'];
    syncTuningReadout();
  });
  cadenceIntervalInput.addEventListener('input', () => {
    tuningConfig.cadenceInterval = clampInteger(
      Number(cadenceIntervalInput.value),
      1,
      360,
    );
    syncTuningReadout();
  });
  mutationIntensityInput.addEventListener('input', () => {
    tuningConfig.mutationIntensity = clampNumber(
      Number(mutationIntensityInput.value),
      0.2,
      3,
    );
    syncTuningReadout();
  });
  growthPruneBiasInput.addEventListener('input', () => {
    tuningConfig.growthPruneBias = clampNumber(
      Number(growthPruneBiasInput.value),
      -1,
      1,
    );
    syncTuningReadout();
  });
  commitThresholdInput.addEventListener('input', () => {
    tuningConfig.commitThreshold = clampNumber(
      Number(commitThresholdInput.value),
      0,
      0.5,
    );
    syncTuningReadout();
  });
  rollbackSensitivityInput.addEventListener('input', () => {
    tuningConfig.rollbackSensitivity = clampNumber(
      Number(rollbackSensitivityInput.value),
      0,
      0.5,
    );
    syncTuningReadout();
  });

  controlsElement.append(
    buildControlRowWithInput(
      'Adaptation Enabled',
      adaptationEnabledInput,
      adaptationEnabledValue,
    ),
    buildControlRowWithInput(
      'Cadence Mode',
      cadenceModeInput,
      cadenceModeValue,
    ),
    buildControlRowWithInput(
      'Cadence Interval',
      cadenceIntervalInput,
      cadenceIntervalValue,
    ),
    buildControlRowWithInput(
      'Mutation Intensity',
      mutationIntensityInput,
      mutationIntensityValue,
    ),
    buildControlRowWithInput(
      'Growth vs Prune',
      growthPruneBiasInput,
      growthPruneBiasValue,
    ),
    buildControlRowWithInput(
      'Commit Threshold',
      commitThresholdInput,
      commitThresholdValue,
    ),
    buildControlRowWithInput(
      'Rollback Sensitivity',
      rollbackSensitivityInput,
      rollbackSensitivityValue,
    ),
  );

  const adaptationTelemetryGrid = document.createElement('div');
  adaptationTelemetryGrid.className = 'racing-telemetry';
  adaptationTelemetryGrid.append(
    buildPanelRowWithLiveNode('Laps', lapValue),
    buildPanelRowWithLiveNode('Tick', tickValue),
    buildPanelRowWithLiveNode('Commit Count', commitCountValue),
    buildPanelRowWithLiveNode('Rollback Count', rollbackCountValue),
    buildPanelRowWithLiveNode('Trend', recentTrendValue),
    buildPanelRowWithLiveNode('Network Size', networkSizeValue),
    buildPanelRowWithLiveNode('Network Δ', networkDeltaValue),
    buildPanelRowWithLiveNode('Last Change', lastChangeReasonValue),
  );
  const runtimeTuningKeyboardHint = buildCallout(
    'Keyboard: A toggle adaptation, C cycle cadence mode, [/] cadence interval, N/M mutation, H/G growth-prune bias, J/K commit threshold, F/R rollback sensitivity.',
  );
  syncRuntimeControls();
  runtimeCard.bodyElement.append(
    controlsElement,
    adaptationTelemetryGrid,
    runtimeTuningKeyboardHint,
  );

  region.append(runtimeCard.cardElement);

  return {
    tickValue,
    lapValue,
    adaptationEnabledValue,
    cadenceModeValue,
    cadenceIntervalValue,
    mutationIntensityValue,
    growthPruneBiasValue,
    commitThresholdValue,
    rollbackSensitivityValue,
    commitCountValue,
    rollbackCountValue,
    recentTrendValue,
    networkSizeValue,
    networkDeltaValue,
    lastChangeReasonValue,
    controlsElement,
    syncRuntimeControls,
  };
}

interface PanelCardElements {
  cardElement: HTMLDivElement;
  bodyElement: HTMLDivElement;
}

/**
 * Creates a reusable panel card with a title and hover/focus tooltip.
 *
 * @param title - Visible card title.
 * @param tooltipHeading - Tooltip heading text.
 * @param tooltipBodyLines - Tooltip body copy.
 * @returns Card shell and body element for further population.
 * @internal
 */
function createPanelCard(
  title: string,
  tooltipHeading: string,
  tooltipBodyLines: readonly string[],
): PanelCardElements {
  const cardElement = document.createElement('div');
  cardElement.className = 'racing-card';

  const headerElement = createPanelHeader(
    title,
    tooltipHeading,
    tooltipBodyLines,
  );
  const bodyElement = document.createElement('div');
  bodyElement.className = 'racing-card__body';

  cardElement.append(headerElement, bodyElement);

  return { cardElement, bodyElement };
}

/**
 * Creates a title row with a Flappy-style explanatory tooltip chip.
 *
 * @param title - Visible title text.
 * @param tooltipHeading - Tooltip heading text.
 * @param tooltipBodyLines - Tooltip body copy.
 * @returns Header element.
 * @internal
 */
function createPanelHeader(
  title: string,
  tooltipHeading: string,
  tooltipBodyLines: readonly string[],
): HTMLDivElement {
  const headerElement = document.createElement('div');
  headerElement.className = 'racing-panel__header';

  const titleElement = document.createElement('div');
  titleElement.className = 'racing-panel__title';
  titleElement.textContent = title;

  headerElement.append(
    titleElement,
    createTooltipHelp(tooltipHeading, tooltipBodyLines),
  );
  return headerElement;
}

/**
 * Creates a compact status chip for the stage metadata row.
 *
 * @param label - Chip label.
 * @param value - Chip value.
 * @returns Status chip element.
 * @internal
 */
function createStatusChip(label: string, value: string): HTMLDivElement {
  const chipElement = document.createElement('div');
  chipElement.className = 'racing-status-chip';

  const labelElement = document.createElement('span');
  labelElement.className = 'racing-status-chip__label';
  labelElement.textContent = label;

  const valueElement = document.createElement('span');
  valueElement.textContent = value;

  chipElement.append(labelElement, valueElement);
  return chipElement;
}

/**
 * Creates a reusable explanatory callout.
 *
 * @param text - Callout body text.
 * @returns Callout element.
 * @internal
 */
function buildCallout(text: string): HTMLDivElement {
  const calloutElement = document.createElement('div');
  calloutElement.className = 'racing-callout';
  calloutElement.textContent = text;
  return calloutElement;
}

/**
 * Creates a hover/focus tooltip chip styled to match the rest of the HUD.
 *
 * @param heading - Tooltip heading.
 * @param bodyLines - Tooltip body lines.
 * @returns Tooltip help wrapper.
 * @internal
 */
function createTooltipHelp(
  heading: string,
  bodyLines: readonly string[],
): HTMLDivElement {
  const wrapperElement = document.createElement('div');
  wrapperElement.className = 'racing-help';

  const tooltipId = `racing-tooltip-${racingTooltipIdCounter++}`;
  const buttonElement = document.createElement('button');
  buttonElement.className = 'racing-help__button';
  buttonElement.type = 'button';
  buttonElement.textContent = '?';
  buttonElement.setAttribute('aria-describedby', tooltipId);
  buttonElement.setAttribute('aria-label', `More info: ${heading}`);

  const tooltipElement = document.createElement('div');
  tooltipElement.className = 'racing-tooltip';
  tooltipElement.id = tooltipId;
  tooltipElement.setAttribute('role', 'tooltip');

  const tooltipArrowElement = document.createElement('div');
  tooltipArrowElement.className = 'racing-tooltip__arrow';

  const tooltipHeadingElement = document.createElement('div');
  tooltipHeadingElement.className = 'racing-tooltip__heading';
  tooltipHeadingElement.textContent = heading;
  tooltipElement.append(tooltipHeadingElement);

  for (const bodyLine of bodyLines) {
    const tooltipLineElement = document.createElement('div');
    tooltipLineElement.className = 'racing-tooltip__line';
    tooltipLineElement.textContent = bodyLine;
    tooltipElement.append(tooltipLineElement);
  }

  tooltipElement.append(tooltipArrowElement);

  wrapperElement.append(buttonElement, tooltipElement);
  return wrapperElement;
}

/**
 * Keeps the canvas backbuffer aligned with its CSS display size.
 *
 * @param canvasElement - Playback canvas.
 * @internal
 */
function syncCanvasToDisplaySize(canvasElement: HTMLCanvasElement): void {
  const devicePixelRatio = Math.min(
    window.devicePixelRatio || 1,
    MAX_CANVAS_DEVICE_PIXEL_RATIO,
  );
  const canvasBounds = canvasElement.getBoundingClientRect();
  const parentElement = canvasElement.parentElement;
  const cssDisplayWidth = resolvePositiveViewportDimension(
    canvasElement.clientWidth,
    canvasBounds.width,
    parentElement?.clientWidth ?? 0,
    canvasElement.width / devicePixelRatio,
  );
  const cssDisplayHeight = resolvePositiveViewportDimension(
    canvasElement.clientHeight,
    canvasBounds.height,
    parentElement?.clientHeight ?? 0,
    canvasElement.height / devicePixelRatio,
  );
  const displayWidth = Math.max(
    1,
    Math.round(cssDisplayWidth * devicePixelRatio),
  );
  const displayHeight = Math.max(
    1,
    Math.round(cssDisplayHeight * devicePixelRatio),
  );

  if (
    canvasElement.width === displayWidth &&
    canvasElement.height === displayHeight
  ) {
    return;
  }

  canvasElement.width = displayWidth;
  canvasElement.height = displayHeight;
}

/**
 * Builds a label/value panel row element whose value is backed by a live text node.
 *
 * @param label - Left label text.
 * @param liveTextNode - Text node whose content will be mutated each frame.
 * @returns Completed row element.
 * @internal
 */
function buildPanelRowWithLiveNode(
  label: string,
  liveTextNode: Text,
): HTMLElement {
  const row = document.createElement('div');
  row.className = 'racing-panel__row';

  const labelSpan = document.createElement('span');
  labelSpan.className = 'racing-panel__label';
  labelSpan.textContent = label;

  const valueSpan = document.createElement('span');
  valueSpan.className = 'racing-panel__value';
  valueSpan.append(liveTextNode);

  row.append(labelSpan, valueSpan);
  return row;
}

/**
 * Builds a runtime control row with an input and a visible live value mirror.
 *
 * @param label - Control label shown in the left column.
 * @param inputElement - Interactive input element.
 * @param liveValueNode - Text node reflecting the current active value.
 * @returns Completed control row.
 * @internal
 */
function buildControlRowWithInput(
  label: string,
  inputElement: HTMLInputElement | HTMLSelectElement,
  liveValueNode: Text,
): HTMLElement {
  const rowElement = document.createElement('label');
  rowElement.className = 'racing-control-row';

  const labelSpan = document.createElement('span');
  labelSpan.className = 'racing-panel__label';
  labelSpan.textContent = label;

  const liveValueSpan = document.createElement('span');
  liveValueSpan.className = 'racing-control-value';
  liveValueSpan.append(liveValueNode);

  rowElement.append(labelSpan, inputElement, liveValueSpan);
  return rowElement;
}

/**
 * Clamps one number inside an inclusive numeric range.
 *
 * @param value - Candidate value.
 * @param minimum - Inclusive minimum.
 * @param maximum - Inclusive maximum.
 * @returns Clamped numeric value.
 * @internal
 */
function clampNumber(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

/**
 * Clamps one number to an integer inside an inclusive range.
 *
 * @param value - Candidate value.
 * @param minimum - Inclusive minimum.
 * @param maximum - Inclusive maximum.
 * @returns Clamped integer value.
 * @internal
 */
function clampInteger(value: number, minimum: number, maximum: number): number {
  return Math.round(clampNumber(value, minimum, maximum));
}

/**
 * Creates a small deterministic public network that drives the solo browser harness.
 *
 * @returns Deterministically parameterized controller network.
 */
export function createDeterministicRacingControllerNetwork(
  observationTier: SupportedObservationTier = 1,
): Network {
  const resolvedInputCount =
    resolveControllerInputCountForObservationTier(observationTier);
  const controllerNetwork = Network.createMLP(
    resolvedInputCount,
    [...CONTROLLER_HIDDEN_LAYER_SIZES],
    2,
  );
  configureDeterministicControllerParameters(controllerNetwork);
  return controllerNetwork;
}

/**
 * Pins one small parameter layout so the browser harness uses a stable steering policy.
 *
 * @param controllerNetwork - Newly created public network facade.
 * @internal
 */
function configureDeterministicControllerParameters(
  controllerNetwork: Network,
): void {
  const inputNodes = controllerNetwork.nodes
    .filter((node) => node.type === 'input')
    .toSorted(
      (leftNode, rightNode) =>
        resolveNodeIndex(leftNode) - resolveNodeIndex(rightNode),
    );
  const hiddenNodes = controllerNetwork.nodes
    .filter((node) => node.type === 'hidden')
    .toSorted(
      (leftNode, rightNode) =>
        resolveNodeIndex(leftNode) - resolveNodeIndex(rightNode),
    );
  const outputNodes = controllerNetwork.nodes
    .filter((node) => node.type === 'output')
    .toSorted(
      (leftNode, rightNode) =>
        resolveNodeIndex(leftNode) - resolveNodeIndex(rightNode),
    );

  if (
    inputNodes.length !== controllerNetwork.input ||
    hiddenNodes.length !== CONTROLLER_HIDDEN_LAYER_SIZES[0] ||
    outputNodes.length !== 2
  ) {
    throw new Error('Racing curriculum controller expected a N -> 4 -> 2 MLP.');
  }

  hiddenNodes[0].squash = methods.Activation.relu;
  hiddenNodes[1].squash = methods.Activation.relu;
  hiddenNodes[2].squash = methods.Activation.relu;
  hiddenNodes[3].squash = methods.Activation.relu;
  outputNodes[0].squash = methods.Activation.tanh;
  outputNodes[1].squash = methods.Activation.tanh;

  hiddenNodes.forEach((hiddenNode) => {
    hiddenNode.bias = 0;
  });
  outputNodes[0].bias = 1.1;
  outputNodes[1].bias = 0;

  const weightByEdgeKey = new Map<string, number>([
    [
      createEdgeKey(
        resolveNodeIndex(
          inputNodes[OBSERVATION_INDEX_OPTIMAL_LINE_HEADING_ERROR],
        ),
        resolveNodeIndex(hiddenNodes[0]),
      ),
      2.5,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(
          inputNodes[OBSERVATION_INDEX_OPTIMAL_LINE_HEADING_ERROR],
        ),
        resolveNodeIndex(hiddenNodes[1]),
      ),
      -2.5,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(
          inputNodes[OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET],
        ),
        resolveNodeIndex(hiddenNodes[2]),
      ),
      1.5,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(
          inputNodes[OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET],
        ),
        resolveNodeIndex(hiddenNodes[3]),
      ),
      -1.5,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(inputNodes[OBSERVATION_INDEX_LATERAL_SPEED]),
        resolveNodeIndex(hiddenNodes[2]),
      ),
      0.4,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(inputNodes[OBSERVATION_INDEX_LATERAL_SPEED]),
        resolveNodeIndex(hiddenNodes[3]),
      ),
      -0.4,
    ],
    // Look-ahead: sin(tangentAhead) − sin(carHeading) ≈ sin(headingErrorAhead)
    // Wires the near look-ahead tangent against the car's own heading sin so
    // h0/h1 see an anticipatory heading error before the curve arrives.
    [
      createEdgeKey(
        resolveNodeIndex(
          inputNodes[OBSERVATION_INDEX_LOOK_AHEAD_NEAR_SIN_TANGENT],
        ),
        resolveNodeIndex(hiddenNodes[0]),
      ),
      0.25,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(
          inputNodes[OBSERVATION_INDEX_LOOK_AHEAD_NEAR_SIN_TANGENT],
        ),
        resolveNodeIndex(hiddenNodes[1]),
      ),
      -0.25,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(inputNodes[OBSERVATION_INDEX_SIN_CAR_HEADING]),
        resolveNodeIndex(hiddenNodes[0]),
      ),
      -0.25,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(inputNodes[OBSERVATION_INDEX_SIN_CAR_HEADING]),
        resolveNodeIndex(hiddenNodes[1]),
      ),
      0.25,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(hiddenNodes[0]),
        resolveNodeIndex(outputNodes[0]),
      ),
      -0.95,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(hiddenNodes[1]),
        resolveNodeIndex(outputNodes[0]),
      ),
      -0.95,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(hiddenNodes[2]),
        resolveNodeIndex(outputNodes[0]),
      ),
      -0.55,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(hiddenNodes[3]),
        resolveNodeIndex(outputNodes[0]),
      ),
      -0.55,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(hiddenNodes[0]),
        resolveNodeIndex(outputNodes[1]),
      ),
      1.85,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(hiddenNodes[1]),
        resolveNodeIndex(outputNodes[1]),
      ),
      -1.85,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(hiddenNodes[2]),
        resolveNodeIndex(outputNodes[1]),
      ),
      -1.35,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(hiddenNodes[3]),
        resolveNodeIndex(outputNodes[1]),
      ),
      1.35,
    ],
  ]);

  controllerNetwork.connections.forEach((connection) => {
    const edgeKey = createEdgeKey(
      resolveNodeIndex(connection.from),
      resolveNodeIndex(connection.to),
    );
    connection.weight = weightByEdgeKey.get(edgeKey) ?? 0;
  });
}

/**
 * Resolves a stable node index for deterministic parameter pinning.
 *
 * @param nodeWithIndex - Node-like value exposing an optional numeric index.
 * @returns Stable node index.
 * @internal
 */
function resolveNodeIndex(nodeWithIndex: { index?: number }): number {
  if (typeof nodeWithIndex.index !== 'number') {
    throw new Error(
      'Racing curriculum controller expected every node to expose an index.',
    );
  }

  return nodeWithIndex.index;
}

/**
 * Builds one stable directed edge key for deterministic connection pinning.
 *
 * @param sourceNodeIndex - Source node index.
 * @param targetNodeIndex - Target node index.
 * @returns Stable edge key.
 * @internal
 */
function createEdgeKey(
  sourceNodeIndex: number,
  targetNodeIndex: number,
): string {
  return `${sourceNodeIndex}->${targetNodeIndex}`;
}

/**
 * Builds one stable role edge key for phenotype remapping.
 *
 * @param sourceRole - Source node role (`type:position`).
 * @param targetRole - Target node role (`type:position`).
 * @returns Stable role-edge key.
 * @internal
 */
function createRoleEdgeKey(sourceRole: string, targetRole: string): string {
  return `${sourceRole}->${targetRole}`;
}

/**
 * Updates telemetry panel text nodes from the current environment state.
 *
 * @param nodes - Live text node references.
 * @param envState - Current physics state.
 * @internal
 */
function updateTelemetryPanelNodes(
  nodes: TelemetryPanelNodes,
  runtimeAdaptationState: RuntimeAdaptationState,
  controllerNetwork: Network,
  envState: EnvironmentState,
  completedLaps: number,
  networkHud?: RacingNetworkHudNodes,
): void {
  const networkSize = resolveNetworkSize(controllerNetwork);
  const runtimeTelemetry = runtimeAdaptationState.telemetry;
  const runtimeTuning = runtimeAdaptationState.tuning;

  nodes.lapValue.textContent = String(completedLaps);
  nodes.tickValue.textContent = String(envState.tick);
  nodes.adaptationEnabledValue.textContent = runtimeTuning.adaptationEnabled
    ? 'on'
    : 'off';
  nodes.cadenceModeValue.textContent = runtimeTuning.cadenceMode;
  nodes.cadenceIntervalValue.textContent = String(
    runtimeTuning.cadenceInterval,
  );
  nodes.mutationIntensityValue.textContent =
    runtimeTuning.mutationIntensity.toFixed(1);
  nodes.growthPruneBiasValue.textContent =
    runtimeTuning.growthPruneBias.toFixed(1);
  nodes.commitThresholdValue.textContent =
    runtimeTuning.commitThreshold.toFixed(2);
  nodes.rollbackSensitivityValue.textContent =
    runtimeTuning.rollbackSensitivity.toFixed(2);
  nodes.commitCountValue.textContent = String(runtimeTelemetry.commitCount);
  nodes.rollbackCountValue.textContent = String(runtimeTelemetry.rollbackCount);
  nodes.recentTrendValue.textContent = runtimeTelemetry.recentImprovementTrend;
  nodes.networkSizeValue.textContent = `N${networkSize.nodes} / C${networkSize.connections}`;
  nodes.networkDeltaValue.textContent = `ΔN${runtimeTelemetry.networkDeltaNodes} / ΔC${runtimeTelemetry.networkDeltaConnections}`;
  nodes.lastChangeReasonValue.textContent = runtimeTelemetry.lastChangeReason;

  if (!networkHud) {
    return;
  }

  networkHud.sizeValue.textContent = `N${networkSize.nodes} / C${networkSize.connections}`;
  networkHud.lastChangeValue.textContent = runtimeTelemetry.lastChangeReason;
  networkHud.statusValue.textContent = resolveNetworkHudStatus(
    runtimeTuning.adaptationEnabled,
    runtimeTelemetry.recentImprovementTrend,
  );
}

/**
 * Resolves the short status label shown in the network panel HUD strip.
 *
 * @param adaptationEnabled - Whether runtime adaptation is currently active.
 * @param recentTrend - Latest improvement trend telemetry value.
 * @returns Uppercase status label.
 */
export function resolveNetworkHudStatus(
  adaptationEnabled: boolean,
  recentTrend: string,
): string {
  if (!adaptationEnabled) {
    return 'HOLD';
  }

  if (recentTrend === 'improving') {
    return 'ADAPTING';
  }

  if (recentTrend === 'regressing') {
    return 'CAUTION';
  }

  return 'STABLE';
}

/**
 * Updates the focused-network panel live-value text nodes from the current
 * control output and curriculum state.
 *
 * @param nodes - Live text node references for the network panel.
 * @param control - Latest controller output (throttle / steer).
 * @param tick - Current simulation tick.
 * @param tier - Current curriculum tier.
 * @internal
 */
/**
 * Runs a deterministic controller probe without touching browser DOM or
 * rendering.  This is a headless smoke-test entry point used by regression tests
 * and benchmark harnesses.
 *
 * @param tickCount - Number of fixed-timestep ticks to simulate.
 * @param tier - Curriculum tier used for controller observation width.
 * @returns Heading and steering summary used by regression tests.
 */
export function runDeterministicControllerProbe(
  tickCount = 240,
  tier: CurriculumTier = 1,
): {
  meanAbsoluteSteer: number;
  nonTrivialSteerSamples: number;
  headingDeltaRadians: number;
} {
  const episodeState = createCurriculumEpisodeState(tier);
  const trackSpec = episodeState.trackSpec;
  const observationTier = resolveObservationTierForCurriculumTier(tier);
  const controller = createNgeController(
    createDeterministicRacingControllerNetwork(observationTier),
    { tier: observationTier },
  );
  let envState = episodeState.envState;
  const initialHeading = envState.carHeading;
  const steerSamples: number[] = [];

  for (let tickIndex = 0; tickIndex < tickCount; tickIndex++) {
    const controlOutput = controller.computeControl(envState, trackSpec);
    steerSamples.push(controlOutput.steer);
    const steppedEnvironmentState = stepEnvironment(envState, controlOutput);
    envState = stabilizeCurriculumTierTireGrip(steppedEnvironmentState, tier);
  }

  const meanAbsoluteSteer =
    steerSamples.reduce(
      (accumulatedSteer, steerSample) =>
        accumulatedSteer + Math.abs(steerSample),
      0,
    ) / steerSamples.length;
  const nonTrivialSteerSamples = steerSamples.filter(
    (steerSample) => Math.abs(steerSample) > 0.05,
  ).length;

  return {
    meanAbsoluteSteer,
    nonTrivialSteerSamples,
    headingDeltaRadians: Math.abs(envState.carHeading - initialHeading),
  };
}

/**
 * Creates the tier-aware starting environment state for the browser shell.
 *
 * @param trackSpec - Frozen track specification for the current tier.
 * @param curriculumTier - Active curriculum tier.
 * @returns Environment state seeded with the tier-appropriate packed roster.
 */
export function createCurriculumEnvironmentState(
  trackSpec: TrackSpec,
  curriculumTier: CurriculumTier,
): EnvironmentState {
  const firstSegment = trackSpec.segments[0];
  const firstSplineSample = trackSpec.splineSamples[0];
  const initialHeading = firstSplineSample
    ? resolveSplineSampleFrame(
        trackSpec.splineSamples,
        firstSplineSample.globalIndex,
      ).tangentHeadingRadians
    : Math.atan2(
        firstSegment.endY - firstSegment.startY,
        firstSegment.endX - firstSegment.startX,
      );
  const startingCars = resolveCurriculumRacePackCars(
    trackSpec,
    curriculumTier,
    initialHeading,
  );
  const primaryCar = startingCars[0];

  return {
    tick: 0,
    carX: primaryCar?.carX ?? firstSplineSample?.x ?? firstSegment.startX,
    carY: primaryCar?.carY ?? firstSplineSample?.y ?? firstSegment.startY,
    carHeading: primaryCar?.carHeading ?? initialHeading,
    teamIndex: primaryCar?.teamIndex ?? 0,
    tireState:
      primaryCar?.tireState ?? ([...FULL_HEALTH_TIRE_STATE] as TireStateTuple),
    cars: startingCars,
    trackSpec,
  };
}

/**
 * Creates the tier-aware track and environment state used by the browser shell.
 *
 * @param curriculumTier - Active curriculum tier.
 * @param trackViewport - Optional canvas-aware viewport used for track shaping.
 * @returns Frozen track spec plus the matching environment roster for that tier.
 */
export function createCurriculumEpisodeState(
  curriculumTier: CurriculumTier,
  trackViewport?: TrackGenerationViewport,
): CurriculumEpisodeState {
  const trackSpec = generateTrack({
    seed: DEMO_TRACK_SEED,
    layoutVersion: DEMO_TRACK_LAYOUT_VERSION,
    sizeBucket: resolveTrackSizeBucketForCurriculumTier(curriculumTier),
    viewport: trackViewport,
  });

  return {
    trackSpec,
    envState: createCurriculumEnvironmentState(trackSpec, curriculumTier),
  };
}

/**
 * Resolves canvas-aware viewport metadata for deterministic track generation.
 *
 * @param canvasElement - Active simulation canvas.
 * @returns Viewport dimensions and edge padding for initial track shaping.
 * @internal
 */
function resolveTrackGenerationViewport(
  canvasElement: HTMLCanvasElement,
): TrackGenerationViewport {
  const canvasBounds = canvasElement.getBoundingClientRect();
  const parentElement = canvasElement.parentElement;
  const parentBounds = parentElement?.getBoundingClientRect();
  const viewportWidth = resolvePositiveViewportDimension(
    canvasElement.clientWidth,
    canvasBounds.width,
    parentElement?.clientWidth ?? 0,
    parentBounds?.width ?? 0,
    canvasElement.width,
  );
  const viewportHeight = resolvePositiveViewportDimension(
    canvasElement.clientHeight,
    canvasBounds.height,
    parentElement?.clientHeight ?? 0,
    parentBounds?.height ?? 0,
    canvasElement.height,
  );

  return {
    width: viewportWidth,
    height: viewportHeight,
    edgePaddingRatio: DEFAULT_TRACK_VIEWPORT_EDGE_PADDING_RATIO,
  };
}

/**
 * Chooses the first positive finite viewport dimension from preferred fallbacks.
 *
 * @param dimensionCandidates - Ordered viewport dimension candidates.
 * @returns Positive finite viewport dimension in pixels.
 * @internal
 */
function resolvePositiveViewportDimension(
  ...dimensionCandidates: readonly number[]
): number {
  for (const dimensionCandidate of dimensionCandidates) {
    if (Number.isFinite(dimensionCandidate) && dimensionCandidate > 0) {
      return dimensionCandidate;
    }
  }

  return 1;
}

/**
 * Resolves the course width bucket for a curriculum tier.
 *
 * Tier 4 and above use the large course so the multi-agent pack has enough
 * lateral room to read clearly in the browser demo.
 *
 * @param curriculumTier - Active curriculum tier.
 * @returns Track size bucket for the tier.
 */
export function resolveTrackSizeBucketForCurriculumTier(
  curriculumTier: CurriculumTier,
): TrackSizeBucket {
  if (curriculumTier >= 4) {
    return 'large';
  }

  return 'medium';
}

/**
 * Resolves the canonical packed roster layout for the active curriculum tier.
 *
 * @param curriculumTier - Active curriculum tier.
 * @returns Ordered team indices for the tier-specific race pack.
 * @internal
 */
function resolveCurriculumRacePackLayout(
  curriculumTier: CurriculumTier,
): readonly CurriculumTeamIndex[] {
  if (curriculumTier >= 5) {
    return TIER_FIVE_TEAM_LAYOUT;
  }

  if (curriculumTier >= 4) {
    return TIER_FOUR_TEAM_LAYOUT;
  }

  if (curriculumTier === 1) {
    return TIER_ONE_TEAM_LAYOUT;
  }

  return [0];
}

/**
 * Resolves the tier-specific race-pack cars on the inner-lane centerline.
 *
 * Cars are seeded on the inner-lane centerline of the first spline sample so the
 * demo starts on the optimal racing line. Team 0 (the primary car) sits on the
 * inward side of the loop; additional team slots step outward.
 *
 * @param trackSpec - Frozen track specification for the current tier.
 * @param curriculumTier - Active curriculum tier.
 * @param initialHeading - Heading resolved from the sampled track frame.
 * @returns Ordered roster seeded from the starting grid.
 * @internal
 */
function resolveCurriculumRacePackCars(
  trackSpec: TrackSpec,
  curriculumTier: CurriculumTier,
  initialHeading: number,
): readonly RacingCarState[] {
  const firstSplineSample = trackSpec.splineSamples[0];
  const sampledFrame = firstSplineSample
    ? resolveSplineSampleFrame(
        trackSpec.splineSamples,
        firstSplineSample.globalIndex,
      )
    : {
        tangentHeadingRadians: initialHeading,
        normalX: 0,
        normalY: 1,
      };
  const teamLayout = resolveCurriculumRacePackLayout(curriculumTier);
  const trackWidth =
    firstSplineSample?.width ?? trackSpec.segments[0]?.width ?? 24;
  const laneCount = firstSplineSample?.laneCount ?? 2;
  const laneWidthWorld =
    firstSplineSample?.laneWidthWorld ?? trackWidth / laneCount;
  const innerOffsetWorld =
    firstSplineSample?.innerOffsetWorld ?? trackWidth / 2 - laneWidthWorld / 2;
  const gridSpacingWorldUnits = Math.max(6, trackWidth * 0.5);
  const teamSlotCounts: Record<CurriculumTeamIndex, number> = { 0: 0, 1: 0 };

  return teamLayout.map((teamIndex) => {
    const teamSlotIndex = teamSlotCounts[teamIndex];
    teamSlotCounts[teamIndex] = teamSlotIndex + 1;
    const longitudinalOffsetWorldUnits = -teamSlotIndex * gridSpacingWorldUnits;
    const lateralOffsetWorldUnits =
      teamIndex === 0 ? innerOffsetWorld : -innerOffsetWorld;
    const startX = firstSplineSample?.x ?? trackSpec.segments[0]?.startX ?? 0;
    const startY = firstSplineSample?.y ?? trackSpec.segments[0]?.startY ?? 0;

    return {
      carX:
        startX +
        Math.cos(sampledFrame.tangentHeadingRadians) *
          longitudinalOffsetWorldUnits +
        sampledFrame.normalX * lateralOffsetWorldUnits,
      carY:
        startY +
        Math.sin(sampledFrame.tangentHeadingRadians) *
          longitudinalOffsetWorldUnits +
        sampledFrame.normalY * lateralOffsetWorldUnits,
      carHeading: sampledFrame.tangentHeadingRadians,
      teamIndex,
      tireState: [...FULL_HEALTH_TIRE_STATE],
    } satisfies RacingCarState;
  });
}

/**
 * Returns the stepped environment state without forcing tire-health overrides.
 *
 * The browser shell now keeps live tire wear untouched so renderer corner colors
 * can reflect the active simulation for every curriculum tier.
 *
 * @param envState - Newly stepped environment state.
 * @param curriculumTier - Active curriculum tier.
 * @returns Unmodified stepped environment state.
 */
export function stabilizeCurriculumTierTireGrip(
  envState: EnvironmentState,
  curriculumTier: CurriculumTier,
): EnvironmentState {
  if (curriculumTier >= TIRE_WEAR_START_TIER) {
    return envState;
  }

  const fullHealthTireState = [...FULL_HEALTH_TIRE_STATE] as TireStateTuple;

  return {
    ...envState,
    tireState: fullHealthTireState,
    cars: envState.cars?.map((carState) => ({
      ...carState,
      tireState: [...FULL_HEALTH_TIRE_STATE] as TireStateTuple,
    })),
  };
}

function createInitialCurriculumProgress(
  trackSpec: TrackSpec,
  envState: EnvironmentState,
): CurriculumProgressState {
  return {
    tier: ACTIVE_CURRICULUM_TIER,
    lapProgress: createInitialLapProgress(trackSpec, envState),
  };
}

function createInitialLapProgress(
  trackSpec: TrackSpec,
  envState: EnvironmentState,
): LapProgressState {
  return {
    lastClosestSplineSampleIndex: resolveClosestSplineSampleIndex(
      trackSpec,
      envState,
    ),
    completedLaps: 0,
  };
}

function resolveNextCurriculumProgressState(
  currentProgress: CurriculumProgressState,
  trackSpec: TrackSpec,
  envState: EnvironmentState,
): CurriculumProgressState {
  const closestSplineSampleIndex = resolveClosestSplineSampleIndex(
    trackSpec,
    envState,
  );
  const nextCompletedLapCount = resolveNextCompletedLapCount(
    currentProgress.lapProgress,
    closestSplineSampleIndex,
    trackSpec.splineSamples.length,
  );

  const promotionResult = resolveTierPromotionFromLapCount(
    currentProgress.tier,
    nextCompletedLapCount,
  );

  if (promotionResult.didAdvance) {
    return {
      tier: promotionResult.nextTier,
      lapProgress: {
        lastClosestSplineSampleIndex: closestSplineSampleIndex,
        completedLaps: promotionResult.remainingLaps,
      },
    };
  }

  return {
    ...currentProgress,
    lapProgress: {
      lastClosestSplineSampleIndex: closestSplineSampleIndex,
      completedLaps: nextCompletedLapCount,
    },
  };
}

function resolveNextCompletedLapCount(
  lapProgress: LapProgressState,
  closestSplineSampleIndex: number,
  sampleCount: number,
): number {
  if (sampleCount <= 0) {
    return lapProgress.completedLaps;
  }

  const wrappedFromEndToStart =
    lapProgress.lastClosestSplineSampleIndex >
      sampleCount * LAP_WRAP_HIGH_WATERMARK_RATIO &&
    closestSplineSampleIndex < sampleCount * LAP_WRAP_LOW_WATERMARK_RATIO;

  return wrappedFromEndToStart
    ? lapProgress.completedLaps + 1
    : lapProgress.completedLaps;
}

/**
 * Applies the racing-curriculum fallback promotion rule:
 * advance one tier whenever the winner completes at least three laps.
 *
 * @param currentTier - Active curriculum tier.
 * @param completedLaps - Completed laps within the current tier race window.
 * @returns Promotion decision with next tier and remaining lap carry.
 */
export function resolveTierPromotionFromLapCount(
  currentTier: CurriculumTier,
  completedLaps: number,
): {
  nextTier: CurriculumTier;
  didAdvance: boolean;
  remainingLaps: number;
} {
  if (currentTier >= MAX_FALLBACK_AUTOPROMOTION_TIER) {
    return {
      nextTier: currentTier,
      didAdvance: false,
      remainingLaps: completedLaps,
    };
  }

  if (
    completedLaps >= LAP_COMPLETIONS_REQUIRED_FOR_TIER_ADVANCE &&
    currentTier < MAX_CURRICULUM_TIER
  ) {
    return {
      nextTier: (currentTier + 1) as CurriculumTier,
      didAdvance: true,
      remainingLaps: completedLaps - LAP_COMPLETIONS_REQUIRED_FOR_TIER_ADVANCE,
    };
  }

  return {
    nextTier: currentTier,
    didAdvance: false,
    remainingLaps: completedLaps,
  };
}

function resolveClosestSplineSampleIndex(
  trackSpec: TrackSpec,
  envState: EnvironmentState,
): number {
  let closestSplineSampleIndex = 0;
  let closestDistanceWorld = Number.POSITIVE_INFINITY;

  for (const splineSample of trackSpec.splineSamples) {
    const distanceToSampleWorld = Math.hypot(
      splineSample.x - envState.carX,
      splineSample.y - envState.carY,
    );

    if (distanceToSampleWorld < closestDistanceWorld) {
      closestDistanceWorld = distanceToSampleWorld;
      closestSplineSampleIndex = splineSample.globalIndex;
    }
  }

  return closestSplineSampleIndex;
}

function resolveObservationTierForCurriculumTier(
  tier: CurriculumTier,
): SupportedObservationTier {
  return Math.min(
    tier,
    MAX_SUPPORTED_OBSERVATION_TIER,
  ) as SupportedObservationTier;
}

function resolveGuidanceAlphaForCurriculumTier(tier: CurriculumTier): number {
  if (tier === 1) {
    return resolveGuidanceAlphaForTier(1);
  }

  if (tier >= 2) {
    return resolveGuidanceAlphaForTier(2);
  }

  return resolveGuidanceAlphaForTier(0);
}

/**
 * Clamps one floating-point value into the closed `[0, 1]` interval.
 *
 * @param value - Incoming floating-point value.
 * @returns Clamped unit-interval value.
 * @internal
 */
function clampUnitInterval(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function resolveControllerInputCountForObservationTier(
  observationTier: SupportedObservationTier,
): number {
  if (observationTier === 1) {
    return 70;
  }

  if (observationTier === 2) {
    return 77;
  }

  if (observationTier === 3) {
    return 91;
  }

  return 95;
}

/**
 * Resolves stage narrative copy for the active curriculum tier.
 *
 * @param tier - Active curriculum tier.
 * @returns Tier-scoped subtitle, footer, and tooltip copy.
 * @internal
 */
function resolveStageNarrativeForTier(tier: CurriculumTier): {
  subtitle: string;
  footer: string;
  tooltipLines: readonly string[];
} {
  const tierLabel = `Tier ${tier}`;

  if (tier <= 2) {
    return {
      subtitle: `${tierLabel} solo NGE harness with a spline-smoothed visual circuit and ${tier === 1 ? 'faded' : 'removed'} optimal-line guidance.`,
      footer:
        'The car and controller are both live in this solo harness. The help chips explain active UI slots and how later tiers broaden the same runtime contract.',
      tooltipLines: [
        `This left stage now runs the ${tierLabel} solo harness: one car, one deterministic seed, and live NGE inference.`,
        'The visual track stays spline-smoothed, but the control loop now flows through the owner-local observation assembler and the public Network.activate(...) seam.',
        'Later tiers keep this shell and widen the same controller path with self-radio and race-pack authority instead of replacing the browser contract again.',
      ],
    };
  }

  return {
    subtitle: `${tierLabel} keeps the same live shell while widening observation authority after guidance removal.`,
    footer:
      'Controller authority stays live through the same browser seam while higher tiers prepare race-pack context instead of reshaping the page contract.',
    tooltipLines: [
      `This left stage now runs the ${tierLabel} progression step with live NGE inference and no optimal-line guidance.`,
      'The same owner-local observation assembler and public Network.activate(...) seam remain in control while tier authority expands.',
      'Later tiers widen self-radio and race-pack context in-place so this browser contract stays stable as the curriculum advances.',
    ],
  };
}

/**
 * Expands a single controller output across every active car in the roster.
 *
 * @param controlOutput - Controller output for the current step.
 * @param carCount - Number of cars that should receive motion input.
 * @returns Per-car control array aligned to roster order.
 * @internal
 */
function resolveControlFanOut(
  controlOutput: CarControlOutput,
  carCount: number,
): readonly CarControlOutput[] {
  return Array.from({ length: Math.max(1, carCount) }, () => ({
    throttle: controlOutput.throttle,
    steer: controlOutput.steer,
  }));
}

/**
 * Creates a fresh tier-signal accumulator for per-tick center-guide evidence.
 *
 * @returns Zeroed evidence accumulator.
 * @internal
 */
function createEmptyTierSignalEvidenceAccumulator(): TierSignalEvidenceAccumulator {
  return {
    tickCount: 0,
    cumulativeLateralErrorNormalized: 0,
    cumulativeHeadingAlignment01: 0,
    cumulativeCenterGuideNeed01: 0,
  };
}

/**
 * Adds one tick of center-guide evidence to the current accumulator.
 *
 * @param currentAccumulator - Current per-lap evidence accumulator.
 * @param evidence - Tick evidence from the active observation seam.
 * @returns Updated accumulator.
 * @internal
 */
function collectTierSignalEvidence(
  currentAccumulator: TierSignalEvidenceAccumulator,
  evidence: NgeControllerTickEvidence,
): TierSignalEvidenceAccumulator {
  return {
    tickCount: currentAccumulator.tickCount + 1,
    cumulativeLateralErrorNormalized:
      currentAccumulator.cumulativeLateralErrorNormalized +
      evidence.lateralErrorNormalized,
    cumulativeHeadingAlignment01:
      currentAccumulator.cumulativeHeadingAlignment01 +
      evidence.headingAlignment01,
    cumulativeCenterGuideNeed01:
      currentAccumulator.cumulativeCenterGuideNeed01 +
      evidence.centerGuideNeed01,
  };
}

/**
 * Computes mean center-guide evidence for the last adaptation window.
 *
 * @param accumulator - Per-lap evidence accumulator.
 * @returns Mean evidence summary.
 * @internal
 */
function summarizeTierSignalEvidence(
  accumulator: TierSignalEvidenceAccumulator,
): TierSignalEvidenceSummary {
  const safeTickCount = Math.max(1, accumulator.tickCount);

  return {
    meanLateralErrorNormalized:
      accumulator.cumulativeLateralErrorNormalized / safeTickCount,
    meanHeadingAlignment01:
      accumulator.cumulativeHeadingAlignment01 / safeTickCount,
    meanCenterGuideNeed01:
      accumulator.cumulativeCenterGuideNeed01 / safeTickCount,
  };
}

function createRuntimeAdaptationState(
  curriculumTier: CurriculumTier,
): RuntimeAdaptationState {
  const tuning: RuntimeTuningConfig = {
    adaptationEnabled: true,
    cadenceMode: DEFAULT_ADAPTATION_CADENCE_MODE,
    cadenceInterval: DEFAULT_RUNTIME_ADAPTATION_CADENCE_INTERVAL_TICKS,
    mutationIntensity: 1,
    growthPruneBias: 0,
    commitThreshold: 0.06,
    rollbackSensitivity: 0.45,
  };

  return {
    engine: createRuntimeAdaptationEngineForTier(tuning, curriculumTier),
    engineConfigSignature: resolveRuntimeAdaptationEngineConfigSignature(
      tuning,
      curriculumTier,
    ),
    tuning,
    telemetry: {
      commitCount: 0,
      rollbackCount: 0,
      recentImprovementTrend: 'flat',
      lastChangeReason: 'startup baseline',
      previousLapTick: null,
      previousLapDuration: null,
      lastLapImprovementRatio: 0,
      networkDeltaNodes: 0,
      networkDeltaConnections: 0,
      adaptationScoreHistory: [],
    },
  };
}

function updateRuntimeImprovementTrend(
  runtimeTelemetry: RuntimeTelemetryState,
  previousCompletedLaps: number,
  currentCompletedLaps: number,
  currentTick: number,
): void {
  if (currentCompletedLaps <= previousCompletedLaps) {
    return;
  }

  if (runtimeTelemetry.previousLapTick === null) {
    runtimeTelemetry.previousLapTick = currentTick;
    runtimeTelemetry.recentImprovementTrend = 'baseline';
    runtimeTelemetry.lastLapImprovementRatio = 0;
    return;
  }

  const currentLapDuration = currentTick - runtimeTelemetry.previousLapTick;
  runtimeTelemetry.previousLapTick = currentTick;

  if (runtimeTelemetry.previousLapDuration === null) {
    runtimeTelemetry.previousLapDuration = currentLapDuration;
    runtimeTelemetry.recentImprovementTrend = 'warming';
    runtimeTelemetry.lastLapImprovementRatio = 0;
    return;
  }

  const improvementRatio =
    (runtimeTelemetry.previousLapDuration - currentLapDuration) /
    Math.max(1, runtimeTelemetry.previousLapDuration);
  runtimeTelemetry.previousLapDuration = currentLapDuration;
  runtimeTelemetry.lastLapImprovementRatio = improvementRatio;
  runtimeTelemetry.recentImprovementTrend =
    improvementRatio > IMPROVEMENT_TREND_EPSILON
      ? `up +${(improvementRatio * 100).toFixed(1)}%`
      : improvementRatio < -IMPROVEMENT_TREND_EPSILON
        ? `down ${(improvementRatio * 100).toFixed(1)}%`
        : 'flat';
}

function resolveNetworkSize(network: Network): {
  nodes: number;
  connections: number;
} {
  return {
    nodes: network.nodes.length,
    connections: network.connections.length,
  };
}

/**
 * Applies a single slow morph cycle to the focused controller network.
 *
 * @param controllerNetwork - Live controller network owned by the browser harness.
 * @internal
 */
function applyWithinTierAdaptation(
  controllerNetwork: ReturnType<
    typeof createDeterministicRacingControllerNetwork
  >,
  curriculumTier: CurriculumTier,
  runtimeAdaptationState: RuntimeAdaptationState,
  tierSignalEvidenceSummary: TierSignalEvidenceSummary,
  previousNetworkSize: { nodes: number; connections: number },
  currentTick: number,
  completedLaps: number,
): boolean {
  const runtimeTelemetry = runtimeAdaptationState.telemetry;
  const runtimeTuning = runtimeAdaptationState.tuning;

  if (!runtimeTuning.adaptationEnabled) {
    runtimeTelemetry.networkDeltaNodes = 0;
    runtimeTelemetry.networkDeltaConnections = 0;
    runtimeTelemetry.lastChangeReason = 'adaptation_disabled (+0.000)';
    return false;
  }

  const adaptationScoreSample = resolveRuntimeAdaptationScoreSample(
    tierSignalEvidenceSummary,
    curriculumTier,
  );
  pushRuntimeAdaptationScoreSample(
    runtimeTelemetry.adaptationScoreHistory,
    adaptationScoreSample,
  );
  const adaptationTelemetry = runtimeAdaptationState.engine.adaptOnTick({
    tick: currentTick,
    network: controllerNetwork,
    scoreHistory: runtimeTelemetry.adaptationScoreHistory,
    completedLaps,
  });

  if (adaptationTelemetry.committed) {
    runtimeTelemetry.commitCount += 1;
  } else if (adaptationTelemetry.operations.length > 0) {
    runtimeTelemetry.rollbackCount += 1;
  }

  runtimeTelemetry.networkDeltaNodes =
    adaptationTelemetry.networkSizeAfter.nodes -
    adaptationTelemetry.networkSizeBefore.nodes;
  runtimeTelemetry.networkDeltaConnections =
    adaptationTelemetry.networkSizeAfter.connections -
    adaptationTelemetry.networkSizeBefore.connections;
  runtimeTelemetry.lastChangeReason =
    resolveRuntimeAdaptationReason(adaptationTelemetry);

  previousNetworkSize.nodes = adaptationTelemetry.networkSizeAfter.nodes;
  previousNetworkSize.connections =
    adaptationTelemetry.networkSizeAfter.connections;

  return adaptationTelemetry.committed;
}

function createRuntimeAdaptationEngineForTier(
  tuningConfig: RuntimeTuningConfig,
  curriculumTier: CurriculumTier,
): RuntimeAdaptationEngine {
  const cadenceInterval = Math.max(1, tuningConfig.cadenceInterval);
  const cadence =
    tuningConfig.cadenceMode === 'laps'
      ? {
          mode: 'lap_boundary' as const,
          boundaryInterval: cadenceInterval,
        }
      : {
          mode: 'every_n_ticks' as const,
          everyNTicks: cadenceInterval,
        };
  const normalizedMutationIntensity = clampInteger(
    Math.round(tuningConfig.mutationIntensity),
    MIN_ADAPTATION_MUTATION_STEPS,
    MAX_ADAPTATION_MUTATION_STEPS,
  );
  const effectiveImprovementThreshold = clampNumber(
    (tuningConfig.commitThreshold + tuningConfig.rollbackSensitivity * 0.5) *
      RUNTIME_ADAPTATION_IMPROVEMENT_THRESHOLD_SCALE,
    0,
    0.01,
  );
  return createRuntimeAdaptationEngine({
    cadence,
    improvementThreshold: effectiveImprovementThreshold,
    minimumEvidenceWindow: 4,
    random: createBiasedStructuralRandomSource(tuningConfig.growthPruneBias),
    limits: {
      maxStructuralEditsPerStep:
        curriculumTier <= 1
          ? normalizedMutationIntensity
          : Math.max(
              MIN_ADAPTATION_MUTATION_STEPS,
              normalizedMutationIntensity - 1,
            ),
      maxNodes: RUNTIME_ADAPTATION_MAX_NODES,
      maxConnections: RUNTIME_ADAPTATION_MAX_CONNECTIONS,
      mutationCooldownTicks:
        curriculumTier <= 1
          ? RUNTIME_ADAPTATION_TIER_ONE_MUTATION_COOLDOWN_TICKS
          : Math.max(
              RUNTIME_ADAPTATION_TIER_ONE_MUTATION_COOLDOWN_TICKS,
              tuningConfig.cadenceInterval,
            ),
      rollbackCooldownTicks:
        curriculumTier <= 1
          ? RUNTIME_ADAPTATION_TIER_ONE_ROLLBACK_COOLDOWN_TICKS
          : RUNTIME_ADAPTATION_ROLLBACK_COOLDOWN_TICKS,
    },
  });
}

function refreshRuntimeAdaptationEngine(
  runtimeAdaptationState: RuntimeAdaptationState,
  curriculumTier: CurriculumTier,
): boolean {
  const nextConfigSignature = resolveRuntimeAdaptationEngineConfigSignature(
    runtimeAdaptationState.tuning,
    curriculumTier,
  );
  if (runtimeAdaptationState.engineConfigSignature === nextConfigSignature) {
    return false;
  }

  runtimeAdaptationState.engine = createRuntimeAdaptationEngineForTier(
    runtimeAdaptationState.tuning,
    curriculumTier,
  );
  runtimeAdaptationState.engineConfigSignature = nextConfigSignature;
  return true;
}

function resolveRuntimeAdaptationEngineConfigSignature(
  tuningConfig: RuntimeTuningConfig,
  curriculumTier: CurriculumTier,
): string {
  return [
    curriculumTier,
    tuningConfig.cadenceMode,
    tuningConfig.mutationIntensity.toFixed(1),
    tuningConfig.growthPruneBias.toFixed(1),
    tuningConfig.commitThreshold.toFixed(2),
    tuningConfig.rollbackSensitivity.toFixed(2),
    tuningConfig.cadenceInterval,
  ].join('|');
}

function createBiasedStructuralRandomSource(
  growthPruneBias: number,
): () => number {
  const clampedBias = clampNumber(growthPruneBias, -1, 1);
  if (clampedBias === 0) {
    return Math.random;
  }

  const biasExponent = 1 + Math.abs(clampedBias) * 4;
  if (clampedBias > 0) {
    return () => 1 - Math.pow(1 - Math.random(), biasExponent);
  }

  return () => Math.pow(Math.random(), biasExponent);
}

function pushRuntimeAdaptationScoreSample(
  scoreHistory: number[],
  scoreSample: number,
): void {
  scoreHistory.push(scoreSample);
  if (scoreHistory.length > RUNTIME_ADAPTATION_SCORE_HISTORY_CAP) {
    scoreHistory.splice(
      0,
      scoreHistory.length - RUNTIME_ADAPTATION_SCORE_HISTORY_CAP,
    );
  }
}

function resolveRuntimeAdaptationScoreSample(
  tierSignalEvidenceSummary: TierSignalEvidenceSummary,
  curriculumTier: CurriculumTier,
): number {
  const guidanceEvidenceWeight =
    resolveGuidanceEvidenceWeightForTier(curriculumTier);
  const headingMisalignment01 =
    1 - tierSignalEvidenceSummary.meanHeadingAlignment01;
  const weightedAdaptationNeed01 = clampUnitInterval(
    tierSignalEvidenceSummary.meanCenterGuideNeed01 * guidanceEvidenceWeight +
      headingMisalignment01 * (1 - guidanceEvidenceWeight),
  );

  return 1 - weightedAdaptationNeed01;
}

function resolveRuntimeAdaptationReason(
  adaptationTelemetry: RuntimeAdaptationTelemetry,
): string {
  const scoreDelta =
    adaptationTelemetry.scoreAfter - adaptationTelemetry.scoreBefore;
  const operationSummary =
    adaptationTelemetry.operations.length === 0
      ? 'none'
      : adaptationTelemetry.operations.join('+');
  return `${adaptationTelemetry.reason}:${operationSummary} (${scoreDelta >= 0 ? '+' : ''}${scoreDelta.toFixed(3)})`;
}

/**
 * Resolves guidance evidence weight for adaptation scoring by curriculum tier.
 *
 * Tier 1 prioritizes center-guide error strongly. Tier 2+ keeps adaptation
 * active while reducing direct dependence on the center-guide seam.
 *
 * @param curriculumTier - Active curriculum tier.
 * @returns Guidance evidence weight in [0, 1].
 * @internal
 */
function resolveGuidanceEvidenceWeightForTier(
  curriculumTier: CurriculumTier,
): number {
  if (curriculumTier <= 1) {
    return 0.9;
  }

  if (curriculumTier === 2) {
    return 0.5;
  }

  if (curriculumTier === 3) {
    return 0.35;
  }

  return 0.2;
}

/**
 * Carries the evolved controller phenotype into the promoted observation tier.
 *
 * New observation tiers may widen the input surface. Existing evolved hidden and
 * output behavior is preserved by role-based remapping from the previous network
 * into the newly shaped network.
 *
 * @param sourceNetwork - Evolved network from the previous tier.
 * @param nextObservationTier - Observation tier for the promoted curriculum tier.
 * @returns Remapped network with carried phenotype and widened input seam.
 * @internal
 */
function remapControllerNetworkForObservationTier(
  sourceNetwork: Network,
  nextObservationTier: SupportedObservationTier,
): Network {
  const targetNetwork =
    createDeterministicRacingControllerNetwork(nextObservationTier);
  const sourceRoleByNodeIndex = createNodeRoleMap(sourceNetwork);
  const targetRoleByNodeIndex = createNodeRoleMap(targetNetwork);
  const sourceConnectionWeightsByRole = new Map<string, number>();

  sourceNetwork.connections.forEach((connection) => {
    const sourceFromRole = sourceRoleByNodeIndex.get(
      resolveNodeIndex(connection.from),
    );
    const sourceToRole = sourceRoleByNodeIndex.get(
      resolveNodeIndex(connection.to),
    );

    if (sourceFromRole === undefined || sourceToRole === undefined) {
      return;
    }

    sourceConnectionWeightsByRole.set(
      createRoleEdgeKey(sourceFromRole, sourceToRole),
      connection.weight,
    );
  });

  targetNetwork.connections.forEach((connection) => {
    const targetFromRole = targetRoleByNodeIndex.get(
      resolveNodeIndex(connection.from),
    );
    const targetToRole = targetRoleByNodeIndex.get(
      resolveNodeIndex(connection.to),
    );

    if (targetFromRole === undefined || targetToRole === undefined) {
      return;
    }

    const mappedWeight = sourceConnectionWeightsByRole.get(
      createRoleEdgeKey(targetFromRole, targetToRole),
    );

    if (mappedWeight !== undefined) {
      connection.weight = mappedWeight;
    }
  });

  remapNodeBiasesAndActivations(sourceNetwork, targetNetwork);
  return targetNetwork;
}

/**
 * Copies hidden and output bias/activation values from one network to another.
 *
 * @param sourceNetwork - Previous tier network.
 * @param targetNetwork - Next tier network.
 * @internal
 */
function remapNodeBiasesAndActivations(
  sourceNetwork: Network,
  targetNetwork: Network,
): void {
  const sourceHiddenNodes = resolveSortedNodesByType(sourceNetwork, 'hidden');
  const targetHiddenNodes = resolveSortedNodesByType(targetNetwork, 'hidden');
  const sourceOutputNodes = resolveSortedNodesByType(sourceNetwork, 'output');
  const targetOutputNodes = resolveSortedNodesByType(targetNetwork, 'output');
  const sharedHiddenCount = Math.min(
    sourceHiddenNodes.length,
    targetHiddenNodes.length,
  );
  const sharedOutputCount = Math.min(
    sourceOutputNodes.length,
    targetOutputNodes.length,
  );

  for (
    let hiddenNodeIndex = 0;
    hiddenNodeIndex < sharedHiddenCount;
    hiddenNodeIndex++
  ) {
    targetHiddenNodes[hiddenNodeIndex].bias =
      sourceHiddenNodes[hiddenNodeIndex].bias;
    targetHiddenNodes[hiddenNodeIndex].squash =
      sourceHiddenNodes[hiddenNodeIndex].squash;
  }

  for (
    let outputNodeIndex = 0;
    outputNodeIndex < sharedOutputCount;
    outputNodeIndex++
  ) {
    targetOutputNodes[outputNodeIndex].bias =
      sourceOutputNodes[outputNodeIndex].bias;
    targetOutputNodes[outputNodeIndex].squash =
      sourceOutputNodes[outputNodeIndex].squash;
  }
}

/**
 * Builds a stable role map (`input:N`, `hidden:N`, `output:N`) for a network.
 *
 * @param network - Network whose nodes should be role-mapped.
 * @returns Role mapping keyed by node index.
 * @internal
 */
function createNodeRoleMap(network: Network): Map<number, string> {
  const roleByNodeIndex = new Map<number, string>();
  const nodeTypes = ['input', 'hidden', 'output'] as const;

  nodeTypes.forEach((nodeType) => {
    const nodesOfType = resolveSortedNodesByType(network, nodeType);

    nodesOfType.forEach((node, nodeTypeIndex) => {
      roleByNodeIndex.set(
        resolveNodeIndex(node),
        `${nodeType}:${nodeTypeIndex}`,
      );
    });
  });

  return roleByNodeIndex;
}

/**
 * Resolves nodes of one type in stable index order.
 *
 * @param network - Source network.
 * @param nodeType - Target node type.
 * @returns Stable node list for the type.
 * @internal
 */
function resolveSortedNodesByType(
  network: Network,
  nodeType: 'input' | 'hidden' | 'output',
) {
  return network.nodes
    .filter((node) => node.type === nodeType)
    .toSorted(
      (leftNode, rightNode) =>
        resolveNodeIndex(leftNode) - resolveNodeIndex(rightNode),
    );
}

/**
 * Creates a worker instance that can step the racing simulation off-thread.
 *
 * The worker reuses the current bundle URL so the same file can serve both the
 * browser host and the worker runtime.
 *
 * @returns Worker instance when the current bundle URL is available; otherwise null.
 * @internal
 */
function createRacingSimulationWorker(): Worker | null {
  const workerUrl = resolveRacingSimulationWorkerUrl();

  if (workerUrl === null) {
    return null;
  }

  return new Worker(workerUrl);
}

/**
 * Resolves the current racing bundle URL for worker bootstrap.
 *
 * @returns Script URL when the host is running from a DOM script element.
 * @internal
 */
function resolveRacingSimulationWorkerUrl(): string | null {
  if (typeof document === 'undefined') {
    return null;
  }

  const currentScriptElement = document.currentScript;

  if (!(currentScriptElement instanceof HTMLScriptElement)) {
    return null;
  }

  if (currentScriptElement.src.length === 0) {
    return null;
  }

  return currentScriptElement.src;
}

/**
 * Sends one stepping request to the racing simulation worker.
 *
 * @param worker - Live worker instance.
 * @param pendingWorkerSteps - Resolver map for in-flight requests.
 * @param requestId - Monotonic request identifier.
 * @param envState - Current environment snapshot.
 * @param control - Per-car control input for the next fixed timestep.
 * @returns Stepped environment snapshot returned by the worker.
 * @internal
 */
function requestRacingWorkerStep(
  worker: Worker,
  pendingWorkerSteps: Map<number, PendingWorkerStep>,
  requestId: number,
  envState: EnvironmentState,
  control: readonly CarControlOutput[] | CarControlOutput,
): Promise<EnvironmentState> {
  return new Promise<EnvironmentState>((resolve, reject) => {
    pendingWorkerSteps.set(requestId, { resolve, reject });
    worker.postMessage({
      type: 'step',
      requestId,
      envState,
      control,
    } satisfies RacingWorkerStepRequest);
  });
}

/**
 * Resolves worker responses back into the pending request map.
 *
 * @param pendingWorkerSteps - In-flight request resolvers.
 * @param event - Worker message event.
 * @internal
 */
function handleRacingWorkerMessage(
  pendingWorkerSteps: Map<number, PendingWorkerStep>,
  event: MessageEvent<RacingWorkerStepResponse>,
): void {
  const workerResponse = event.data;

  if (workerResponse.type !== 'step-result') {
    return;
  }

  const pendingStep = pendingWorkerSteps.get(workerResponse.requestId);

  if (pendingStep === undefined) {
    return;
  }

  pendingWorkerSteps.delete(workerResponse.requestId);
  pendingStep.resolve(workerResponse.envState);
}

/**
 * Boots the worker-side stepping bridge when this bundle runs in a worker context.
 * @internal
 */
function maybeBootstrapRacingSimulationWorker(): void {
  if (!isDedicatedRacingSimulationWorkerContext()) {
    return;
  }

  const workerGlobalScope = self as RacingSimulationWorkerScope;

  workerGlobalScope.onmessage = (
    event: MessageEvent<RacingWorkerStepRequest>,
  ): void => {
    const workerRequest = event.data;

    if (workerRequest.type !== 'step') {
      return;
    }

    workerGlobalScope.postMessage({
      type: 'step-result',
      requestId: workerRequest.requestId,
      envState: stepEnvironment(workerRequest.envState, workerRequest.control),
    } satisfies RacingWorkerStepResponse);
  };
}

/**
 * Detects whether the current runtime is the worker-side racing bundle.
 *
 * @returns True when the bundle is executing in a dedicated worker context.
 * @internal
 */
function isDedicatedRacingSimulationWorkerContext(): boolean {
  return (
    typeof window === 'undefined' &&
    typeof document === 'undefined' &&
    typeof self !== 'undefined' &&
    'postMessage' in self
  );
}

maybeBootstrapRacingSimulationWorker();
