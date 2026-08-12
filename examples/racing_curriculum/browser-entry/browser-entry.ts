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

import { resolveAccelerationConfig } from '../../../src/acceleration/acceleration.config';
import { autoEnableAcceleration } from '../../../src/acceleration/acceleration.orchestrator';
import type {
  AccelerationMode,
  AccelerationStatus,
} from '../../../src/acceleration/acceleration.types';
import { createRacingHost } from './host/host';
import type { RacingNetworkHudNodes } from './host/host.types';
import { Network, Node, methods } from '../../../src/browser-entry.ts';
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
  type NgeController,
  type NgeControllerTickEvidence,
  resolveGuidanceAlphaForTier,
} from '../controller/nge.controller';
import {
  createPerCarAdaptationEngines,
  evaluateRacingTrendScore,
  type RacingQualitySignal,
  type RuntimeAdaptationEngine,
  type RuntimeAdaptationTelemetry,
} from '../controller/runtime.adaptation';
import {
  derivePerCarObservationState,
  TOTAL_TIER4_INPUT_SIZE,
} from '../controller/observation.assembler';
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
/** Maximum backing-store width in device pixels; CSS stretches beyond this. */
const MAX_CANVAS_WIDTH_PX = 1600;
/** Maximum backing-store height in device pixels; CSS stretches beyond this. */
const MAX_CANVAS_HEIGHT_PX = 900;
/** Refresh cadence for the focused network canvas in milliseconds. */
const FOCUSED_NETWORK_REFRESH_INTERVAL_MS = 5000;
/** Rolling score window size supplied to the runtime adaptation evaluator. */
const RACING_RUNTIME_SCORE_HISTORY_WINDOW = 60;

/**
 * Resolved acceleration configuration forwarded into the per-car runtime
 * adaptation engines. The demo explicitly requests a large parallel variant
 * count so the grow-stabilize cycle can exercise the NGE variant evaluator.
 */
const DEMO_ACCELERATION_CONFIG = resolveAccelerationConfig({
  parallelVariantCount: 256,
});

/** Track determinism key: seed 42, layout v1. */
const DEMO_TRACK_SEED = 42;
const DEMO_TRACK_LAYOUT_VERSION = 1;
/** Default browser harness tier for the NGE controller integration. */
const DEFAULT_CURRICULUM_TIER: CurriculumTier = 1;
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
/** Tier 2 packs use a two-car 1v1 grid with one car per team and the self-radio seam live. */
const TIER_TWO_TEAM_LAYOUT = [0, 1] as const;
/** Tier 3 fallback packs use a four-car 2v2 grid so promotion does not collapse to one car. */
const TIER_THREE_TEAM_LAYOUT = [0, 0, 1, 1] as const;
/**
 * Team slot index reserved for the blue (inner-lane) team.
 * Exported so renderer and observation contracts can agree on the red/blue
 * baseline without magic numbers.
 */
export const TEAM_BLUE_INDEX = 0 as const;
/**
 * Team slot index reserved for the red (outer-lane) team.
 * Exported alongside {@link TEAM_BLUE_INDEX} to keep Tier 1/Tier 2 color and
 * lane assignments explicit and deterministic.
 */
export const TEAM_RED_INDEX = 1 as const;

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
  requestId: string;
  envState: EnvironmentState;
  control: readonly CarControlOutput[] | CarControlOutput;
};

type RacingWorkerStepResponse = {
  type: 'step-result';
  requestId: string;
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

/**
 * Options accepted by {@link start} to override the default curriculum tier.
 *
 * The browser harness defaults to Tier 1 (two-car 1v1 pack). Callers that need
 * a higher-tier pack — for example Tier 3 (four-car 2v2) — pass `{ tier: 3 }`
 * so the host builds the correct roster, controller map, and observation tier
 * before the animation loop begins.
 */
export interface RacingCurriculumStartOptions {
  /** Curriculum tier to launch; defaults to {@link DEFAULT_CURRICULUM_TIER}. */
  readonly tier?: number;
}

// ── DOM panel element sets ────────────────────────────────────────────────────

/** Live-updating text node references for the telemetry panel. */
interface TelemetryPanelNodes {
  tickValue: Text;
  lapValue: Text;
  networkSizeValue: Text;
  networkDeltaValue: Text;
  lastChangeReasonValue: Text;
  lapTimeValue: Text;
  accelerationValue: Text;
  controlsElement: HTMLDivElement;
  syncRuntimeControls: () => void;
}

/** Curriculum tier contract from the racing plan ladder. */
type CurriculumTier = 1 | 2 | 3 | 4 | 5 | 6;

/** Observation tier supported by the owner-local controller seam. */
type SupportedObservationTier = 1 | 2 | 3 | 4 | 5 | 6;

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

/** Candidate for tier promotion with all metrics needed for agent selection. */
type PromotionCandidate = {
  /** Car index in the roster. */
  carIndex: number;
  /** Best lap time in ms (team-level metric shared across candidates). */
  bestLapTimeMs: number;
  /** Median hidden-node count for this candidate's network. */
  medianHiddenNodeCount: number;
  /** Driving quality score from the trend evaluator. */
  drivingQuality: number;
  /** Reference to the candidate's network. */
  network: Network;
  /** Forward-pass output sample for behavioral diversity computation. */
  outputSample: number[];
};

/** Configurable selection criteria for tier promotion. */
type PromotionSelectionConfig = {
  /** Criterion for ranking promotion candidates. */
  selectionCriteria: 'bestLapTime' | 'mostGrowth' | 'bestDrivingQuality';
  /** Fraction of candidates to promote (0..1). */
  selectionRatio: number;
  /** Minimum number of agents to promote. */
  minSelected: number;
};

/** Highest curriculum tier in the racing plan ladder. */
const MAX_CURRICULUM_TIER: CurriculumTier = 6;
/**
 * N_floor per curriculum tier: median hidden-node count required for promotion.
 *
 * Values from the racing plan ladder. A team's median hidden-node count
 * must meet or exceed the tier's N_floor before promotion is granted.
 */
const TIER_N_FLOOR: Readonly<Record<number, number>> = {
  1: 1_000,
  2: 2_000,
  3: 8_000,
  4: 20_000,
  5: 40_000,
  6: 75_000,
};
/** Default agent selection ratio for promotion: top 50% of qualifying candidates. */
const DEFAULT_PROMOTION_SELECTION_RATIO = 0.5;
/** Minimum number of agents to promote when candidates qualify. */
const MIN_PROMOTED_AGENTS = 1;
/** Highest observation tier implemented in the browser controller seam. */
const MAX_SUPPORTED_OBSERVATION_TIER: SupportedObservationTier = 6;
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
 * @param options - Optional launch configuration; `tier` overrides the default
 *   curriculum tier so callers can start directly at Tier 3 (four-car 2v2) or
 *   higher without waiting for auto-promotion.
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
 * @example
 * ```ts
 * // Launch directly at Tier 3 (four-car 2v2 pack).
 * const handle = await start(host, { tier: 3 });
 * handle.stop();
 * ```
 */
export async function start(
  container: HTMLElement | string = 'racing-curriculum-output',
  options?: RacingCurriculumStartOptions,
): Promise<RacingCurriculumRunHandle> {
  // Step 0: Resolve the curriculum tier from caller options.
  const resolvedTier = resolveCurriculumTierFromOptions(options);

  // Step 1: Resolve container and build host DOM shell.
  const containerElement = resolveContainerElement(container);
  injectRacingStyles();
  const hostHandle = createRacingHost(containerElement);
  const stageCardNodes = setupCanvasStage(
    hostHandle.canvasRegionElement,
    hostHandle.canvasElement,
    resolvedTier,
  );
  hostHandle.applyViewportLayout(window.innerWidth);
  syncCanvasToDisplaySize(hostHandle.canvasElement);

  // Step 2: Generate the tier-aware track and matching starting grid.
  let episodeState = createCurriculumEpisodeState(
    resolvedTier,
    resolveTrackGenerationViewport(hostHandle.canvasElement),
  );
  let trackSpec = episodeState.trackSpec;

  // Step 3: Initialise the packed race state on the sampled spline lane center.
  let envState = episodeState.envState;

  // Step 4: Initialise one independent NGE controller per car and render state.
  let curriculumProgress = createInitialCurriculumProgress(
    trackSpec,
    envState,
    resolvedTier,
  );
  let activeObservationTier = resolveObservationTierForCurriculumTier(
    curriculumProgress.tier,
  );
  const carCount = envState.cars?.length ?? 1;
  const controllerNetworkByCarIndex = new Map<number, Network>();
  const controllerByCarIndex = new Map<number, NgeController>();
  const buildPerCarControllers = (
    observationTier: SupportedObservationTier,
    rosterSize: number,
  ): void => {
    controllerNetworkByCarIndex.clear();
    controllerByCarIndex.clear();
    for (let carIndex = 0; carIndex < rosterSize; carIndex++) {
      const perCarNetwork =
        createDeterministicRacingControllerNetwork(observationTier);
      controllerNetworkByCarIndex.set(carIndex, perCarNetwork);
      controllerByCarIndex.set(
        carIndex,
        createNgeController(perCarNetwork, { tier: observationTier }),
      );
    }
  };
  buildPerCarControllers(activeObservationTier, carCount);

  // Step 4a: Wire per-car runtime adaptation engines on top of the seeded
  // deterministic controllers.  The engines are worker-authoritative and mutate
  // the live focused network so the demo exercises real NGE growth.
  const adaptationEngineByCarIndex = new Map<number, RuntimeAdaptationEngine>();
  const scoreHistoryByCarIndex = new Map<
    number,
    (number | RacingQualitySignal)[]
  >();
  const latestAdaptationTelemetryByCarIndex = new Map<
    number,
    RuntimeAdaptationTelemetry
  >();
  const previousCarPositionByCarIndex = new Map<
    number,
    { carX: number; carY: number }
  >();
  const buildPerCarAdaptationEngines = (rosterSize: number): void => {
    adaptationEngineByCarIndex.clear();
    scoreHistoryByCarIndex.clear();
    latestAdaptationTelemetryByCarIndex.clear();
    previousCarPositionByCarIndex.clear();
    const engines = createPerCarAdaptationEngines(rosterSize, {
      evaluateScore: evaluateRacingTrendScore,
      improvementThreshold: 0.01,
      cadence: { mode: 'every_n_ticks', everyNTicks: 4 },
      limits: { mutationCooldownTicks: 40, rollbackCooldownTicks: 5 },
      accelerationConfig: DEMO_ACCELERATION_CONFIG,
    });
    for (let carIndex = 0; carIndex < rosterSize; carIndex++) {
      adaptationEngineByCarIndex.set(carIndex, engines.get(carIndex)!);
      scoreHistoryByCarIndex.set(carIndex, []);
    }
  };
  buildPerCarAdaptationEngines(carCount);

  let focusedCarIndex = 0;
  let focusedControllerNetwork =
    controllerNetworkByCarIndex.get(focusedCarIndex)!;

  // Resolve the actual acceleration backend the library will use so the HUD can
  // display the chosen backend, any fallback reason, and the variant count.
  const accelerationStatus = await autoEnableAcceleration({
    nodeCount: focusedControllerNetwork.nodes.length,
    batchParallelCount: DEMO_ACCELERATION_CONFIG.parallelVariantCount!,
    config: DEMO_ACCELERATION_CONFIG,
  });
  const accelerationDisplayLabel = formatAccelerationStatus(
    accelerationStatus,
    DEMO_ACCELERATION_CONFIG.parallelVariantCount!,
  );
  hostHandle.networkHud.titleValue.textContent = accelerationDisplayLabel;
  if (stageCardNodes.accelerationStatusElement) {
    stageCardNodes.accelerationStatusElement.textContent =
      accelerationDisplayLabel;
  }
  if (stageCardNodes.accelerationChipElement) {
    stageCardNodes.accelerationChipElement.className = `racing-status-chip ${resolveAccelerationChipClass(accelerationStatus.mode)}`;
  }

  let tierSignalEvidenceAccumulator =
    createEmptyTierSignalEvidenceAccumulator();
  let guidanceAlpha = resolveGuidanceAlphaForCurriculumTier(
    curriculumProgress.tier,
  );

  // Tier promotion gate state: lap-time improvement tracking (Gate A) and
  // agent selection config for the co-evolution promotion policy.
  let tierBestLapTimeMs: number | null = null;
  let lapStartTick = 0;
  let promotedCarIndices: number[] = [];

  /**
   * Rebuilds the curriculum episode, race state, and per-car controllers for a
   * target curriculum tier while carrying any supplied source networks.
   *
   * This is the single rebuild seam used both by automatic tier promotion and
   * by the runtime tier selector. Callers decide which networks (if any) are
   * carried into the new tier shape; missing slots receive a fresh deterministic
   * substrate.
   *
   * @param targetTier - Curriculum tier to switch to.
   * @param sourceNetworkByCarIndex - Optional map of networks to remap into the
   *   new tier shape, keyed by car index.
   * @internal
   */
  function rebuildCurriculumStateForTier(
    targetTier: CurriculumTier,
    sourceNetworkByCarIndex?: ReadonlyMap<number, Network>,
  ): void {
    activeObservationTier = resolveObservationTierForCurriculumTier(targetTier);
    curriculumProgress = {
      tier: targetTier,
      lapProgress: createInitialLapProgress(trackSpec, envState),
    };
    episodeState = createCurriculumEpisodeState(
      targetTier,
      resolveTrackGenerationViewport(hostHandle.canvasElement),
    );
    trackSpec = episodeState.trackSpec;
    envState = stabilizeCurriculumTierTireGrip(
      episodeState.envState,
      targetTier,
    );

    const targetCarCount = envState.cars?.length ?? 1;
    controllerNetworkByCarIndex.clear();
    controllerByCarIndex.clear();
    for (let carIndex = 0; carIndex < targetCarCount; carIndex++) {
      const sourceNetwork = sourceNetworkByCarIndex?.get(carIndex);
      const baseNetwork =
        sourceNetwork ??
        createDeterministicRacingControllerNetwork(activeObservationTier);
      const remappedNetwork = remapControllerNetworkForObservationTier(
        baseNetwork,
        activeObservationTier,
      );
      controllerNetworkByCarIndex.set(carIndex, remappedNetwork);
      controllerByCarIndex.set(
        carIndex,
        createNgeController(remappedNetwork, { tier: activeObservationTier }),
      );
    }

    buildPerCarAdaptationEngines(targetCarCount);
    focusedCarIndex = 0;
    focusedControllerNetwork =
      controllerNetworkByCarIndex.get(focusedCarIndex)!;
    simulationTick = 0;
    lapStartTick = 0;
    tierBestLapTimeMs = null;
    tierSignalEvidenceAccumulator = createEmptyTierSignalEvidenceAccumulator();
    guidanceAlpha = resolveGuidanceAlphaForCurriculumTier(targetTier);
    hostHandle.renderNetworkArchitecture(focusedControllerNetwork);
    setupCanvasStage(
      hostHandle.canvasRegionElement,
      hostHandle.canvasElement,
      targetTier,
      accelerationStatus,
    );
  }

  const promotionSelectionConfig: PromotionSelectionConfig = {
    selectionCriteria: 'bestLapTime',
    selectionRatio: DEFAULT_PROMOTION_SELECTION_RATIO,
    minSelected: MIN_PROMOTED_AGENTS,
  };

  const renderState = createRacingRenderState();
  const simulationWorker = createRacingSimulationWorker();
  simulationWorker?.postMessage({
    type: 'init',
    populationSize: RACING_CURRICULUM_POPULATION_SIZE,
    rngSeed: DEMO_TRACK_SEED,
    tier: resolvedTier,
  });
  const pendingWorkerSteps = new Map<string, PendingWorkerStep>();
  let nextWorkerRequestId = 0;
  const handleWorkerMessage = (
    event: MessageEvent<RacingWorkerStepResponse>,
  ): void => {
    handleRacingWorkerMessage(pendingWorkerSteps, event);
  };
  simulationWorker?.addEventListener('message', handleWorkerMessage);

  // Step 5: Build info panels inside the host regions.
  hostHandle.renderNetworkArchitecture(focusedControllerNetwork);
  const stageCardElement =
    hostHandle.canvasRegionElement.querySelector('.racing-stage-card');
  const runtimeControls = setupRuntimeControls(
    (stageCardElement as HTMLElement | null) ?? hostHandle.canvasRegionElement,
    {
      initialTier: curriculumProgress.tier,
      onSelectTier: (selectedTier) => {
        rebuildCurriculumStateForTier(
          selectedTier,
          new Map(controllerNetworkByCarIndex),
        );
      },
    },
  );

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
  let simulationTick = 0;

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
      const perCarControls = resolvePerCarControls(
        controllerByCarIndex,
        envState,
        trackSpec,
      );
      const focusedTickResult = controllerByCarIndex
        .get(0)!
        .computeControlWithEvidence(
          derivePerCarObservationState(envState, 0),
          trackSpec,
        );

      // Drive every car's runtime adaptation engine with a rolling
      // composite driving-quality signal so the demo exercises live network
      // growth across the whole roster, not just the focused car.
      const carCountThisStep = envState.cars?.length ?? 1;
      for (let carIndex = 0; carIndex < carCountThisStep; carIndex++) {
        const perCarScoreHistory = scoreHistoryByCarIndex.get(carIndex)!;
        const carState = envState.cars?.[carIndex];
        const trackProgress = resolvePerCarTrackProgress(trackSpec, carState);
        const forwardSpeed = resolvePerCarForwardSpeed(
          carState,
          previousCarPositionByCarIndex.get(carIndex),
        );
        const headingAlignment = resolvePerCarHeadingAlignment(
          trackSpec,
          carState,
        );
        const offTrackPenalty = resolvePerCarOffTrackPenalty(
          trackSpec,
          carState,
        );
        const physicsReward = carState?.reward ?? 0;
        if (carState) {
          previousCarPositionByCarIndex.set(carIndex, {
            carX: carState.carX,
            carY: carState.carY,
          });
        }
        const racingQualitySignal: RacingQualitySignal = {
          trackProgress,
          forwardSpeed,
          headingAlignment,
          offTrackPenalty,
          physicsReward,
        };
        perCarScoreHistory.push(racingQualitySignal);
        if (perCarScoreHistory.length > RACING_RUNTIME_SCORE_HISTORY_WINDOW) {
          perCarScoreHistory.shift();
        }
        const adaptationTelemetry = await adaptationEngineByCarIndex
          .get(carIndex)!
          .adaptOnTick({
            tick: simulationTick,
            network: controllerNetworkByCarIndex.get(carIndex)!,
            scoreHistory: perCarScoreHistory,
          });
        latestAdaptationTelemetryByCarIndex.set(carIndex, adaptationTelemetry);
      }
      simulationTick += 1;

      tierSignalEvidenceAccumulator = collectTierSignalEvidence(
        tierSignalEvidenceAccumulator,
        focusedTickResult.evidence,
      );
      const steppedEnvironmentState = simulationWorker
        ? await requestRacingWorkerStep(
            simulationWorker,
            pendingWorkerSteps,
            String(++nextWorkerRequestId),
            envState,
            perCarControls,
          )
        : stepEnvironment(envState, perCarControls);
      envState = stabilizeCurriculumTierTireGrip(
        steppedEnvironmentState,
        curriculumProgress.tier,
      );
      // Step: Detect lap completion and evaluate tier promotion gates.
      const closestSplineSampleIndex = resolveClosestSplineSampleIndex(
        trackSpec,
        envState,
      );
      const nextCompletedLapCount = resolveNextCompletedLapCount(
        curriculumProgress.lapProgress,
        closestSplineSampleIndex,
        trackSpec.splineSamples.length,
      );
      const lapCompleted =
        nextCompletedLapCount > curriculumProgress.lapProgress.completedLaps;

      let nextTier: CurriculumTier = curriculumProgress.tier;
      let didPromote = false;

      if (lapCompleted) {
        // Gate A: Lap-time improvement — the first lap establishes the
        // baseline; subsequent laps must improve (be faster) to pass.
        const lapTimeMs = (simulationTick - lapStartTick) * FIXED_TIMESTEP_MS;
        lapStartTick = simulationTick;
        const lapTimeImproved =
          tierBestLapTimeMs === null || lapTimeMs < tierBestLapTimeMs;
        if (lapTimeImproved) {
          tierBestLapTimeMs =
            tierBestLapTimeMs === null
              ? lapTimeMs
              : Math.min(tierBestLapTimeMs, lapTimeMs);
        }

        // Gate B + agent selection: resolve tier promotion with per-car data.
        const promotionResult = resolveTierPromotion({
          currentTier: curriculumProgress.tier,
          completedLaps: nextCompletedLapCount,
          lapTimeImproved,
          bestLapTimeMs: tierBestLapTimeMs,
          controllerNetworkByCarIndex,
          scoreHistoryByCarIndex,
          selectionConfig: promotionSelectionConfig,
        });
        nextTier = promotionResult.nextTier;
        didPromote = promotionResult.didAdvance;
        promotedCarIndices = promotionResult.promotedCarIndices;
      }

      if (didPromote) {
        curriculumProgress = {
          tier: nextTier,
          lapProgress: {
            lastClosestSplineSampleIndex: closestSplineSampleIndex,
            completedLaps: 0,
          },
        };
      } else {
        curriculumProgress = {
          ...curriculumProgress,
          lapProgress: {
            lastClosestSplineSampleIndex: closestSplineSampleIndex,
            completedLaps: nextCompletedLapCount,
          },
        };
      }

      if (curriculumProgress.tier !== previousCurriculumTier) {
        // Carry only promoted agents' evolved networks into the next tier.
        // Non-promoted and newly added car slots receive a fresh deterministic
        // substrate remapped to the promoted tier shape.
        const promotedSet = new Set(promotedCarIndices);
        const previousNetworksByCarIndex = new Map(controllerNetworkByCarIndex);
        const carriedNetworks = new Map<number, Network>();
        for (const [carIndex, network] of previousNetworksByCarIndex) {
          if (promotedSet.has(carIndex)) {
            carriedNetworks.set(carIndex, network);
          }
        }
        rebuildCurriculumStateForTier(curriculumProgress.tier, carriedNetworks);
      } else {
        tierSignalEvidenceAccumulator =
          createEmptyTierSignalEvidenceAccumulator();
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
    const renderOverlayFrame = buildRacingRenderOverlayFrame(
      envState,
      curriculumProgress.tier,
    );
    renderRacingFrame(
      hostHandle.canvasElement,
      trackSpec,
      envState,
      renderState,
      worldTransform,
      { guidanceAlpha, frame: renderOverlayFrame },
    );

    // Update telemetry readouts (text-only, no innerHTML churn).
    updateTelemetryPanelNodes(
      runtimeControls,
      focusedControllerNetwork,
      envState,
      curriculumProgress.lapProgress.completedLaps,
      tierBestLapTimeMs,
      latestAdaptationTelemetryByCarIndex,
      0,
      accelerationDisplayLabel,
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

    // Round-robin through all cars so the visualizer shows every agent's
    // network architecture, not just car 0.
    const carCount = controllerNetworkByCarIndex.size;
    if (carCount > 0) {
      focusedCarIndex = (focusedCarIndex + 1) % carCount;
      focusedControllerNetwork =
        controllerNetworkByCarIndex.get(focusedCarIndex)!;
    }

    hostHandle.renderNetworkArchitecture(focusedControllerNetwork);
  }, FOCUSED_NETWORK_REFRESH_INTERVAL_MS);

  const handle: RacingCurriculumRunHandle = {
    done: Promise.resolve(),
    isRunning: () => running,
    stop: () => {
      if (!running) return;
      running = false;
      window.removeEventListener('resize', handleViewportResize);
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
    .racing-tier-selector {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      padding: 6px;
      border: 1px solid rgba(15, 181, 255, 0.22);
      border-radius: var(--racing-control-radius);
      background: rgba(6, 11, 20, 0.45);
    }
    .racing-tier-selector__label {
      color: var(--racing-text-muted);
      font-family: var(--racing-mono);
      font-size: 9px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding-right: 6px;
    }
    .racing-tier-button {
      min-width: 28px;
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
      cursor: pointer;
      box-shadow: var(--racing-control-idle-shadow);
      transition: ${FLAPPY_SELECTOR_TRANSITION};
    }
    .racing-tier-button:hover,
    .racing-tier-button:focus-visible {
      transform: translateY(-1px);
      box-shadow: var(--racing-control-hover-shadow);
    }
    .racing-tier-button--active {
      color: var(--racing-accent);
      border-color: var(--racing-accent);
      background: rgba(255, 154, 46, 0.12);
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
    .racing-status-chip--gpu {
      color: #4ade80;
      border-color: rgba(74, 222, 128, 0.4);
    }
    .racing-status-chip--worker {
      color: #fbbf24;
      border-color: rgba(251, 191, 36, 0.4);
    }
    .racing-status-chip--cpu {
      color: #f87171;
      border-color: rgba(248, 113, 113, 0.4);
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
 * Resolves the CSS class that color-codes the acceleration metadata chip by
 * the selected backend mode.
 *
 * @param mode - Resolved acceleration backend mode.
 * @returns Class name to apply to the acceleration status chip.
 * @internal
 */
function resolveAccelerationChipClass(mode: AccelerationMode): string {
  return `racing-status-chip--${mode}`;
}

/**
 * Rebuilds the canvas region into a proper stage card with explanatory HUD chips.
 *
 * The returned acceleration chip reference lets the caller update the chip once
 * the async acceleration backend resolution finishes.
 *
 * @param region - Canvas host region.
 * @param canvasElement - Playback canvas element to place inside the stage.
 * @param tier - Curriculum tier driving the stage narrative.
 * @param accelerationStatus - Optional resolved acceleration status; when
 *   supplied the chip is created with the resolved label and color-coded
 *   immediately. When omitted the chip starts as "detecting…" and the caller
 *   should update it via the returned reference.
 * @returns Reference to the acceleration chip element and its value text node.
 * @internal
 */
function setupCanvasStage(
  region: HTMLElement,
  canvasElement: HTMLCanvasElement,
  tier: CurriculumTier,
  accelerationStatus?: AccelerationStatus,
): {
  accelerationChipElement: HTMLDivElement | undefined;
  accelerationStatusElement: HTMLSpanElement | undefined;
} {
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
  const initialAccelerationLabel = accelerationStatus
    ? formatAccelerationStatus(
        accelerationStatus,
        DEMO_ACCELERATION_CONFIG.parallelVariantCount!,
      )
    : 'detecting…';
  const accelerationChipElement = createStatusChip(
    'Acceleration',
    initialAccelerationLabel,
  );
  if (accelerationStatus) {
    accelerationChipElement.className = `racing-status-chip ${resolveAccelerationChipClass(accelerationStatus.mode)}`;
  }
  metaElement.append(
    createStatusChip(
      'Controller',
      `Live NGE controller • ${DEMO_ACCELERATION_CONFIG.parallelVariantCount}x`,
    ),
    createStatusChip('Track', 'Spline-smoothed visual'),
    createStatusChip('Seed', '42 • v1 • medium'),
    accelerationChipElement,
  );
  const accelerationStatusElement =
    accelerationChipElement.querySelector<HTMLSpanElement>(
      '.racing-status-chip__value',
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

  return {
    accelerationChipElement,
    accelerationStatusElement: accelerationStatusElement ?? undefined,
  };
}

/**
 * Builds the live telemetry readouts panel for the racing canvas region.
 *
 * The panel lives inside the canvas region below the stage, matching the
 * Flappy Bird parity layout where runtime widgets share the left column.
 * Adaptation now runs worker-side; the host only displays tick, lap, and
 * network size telemetry. A small tier selector lets the user manually switch
 * between the six observation tiers without reloading the page.
 *
 * @param region - The stage-card body element inside the canvas region.
 * @param options - Optional initial tier and tier-selection callback.
 * @returns References to live-updating telemetry text nodes plus a sync hook.
 * @internal
 */
function setupRuntimeControls(
  region: HTMLElement,
  options?: {
    readonly initialTier?: CurriculumTier;
    readonly onSelectTier?: (tier: CurriculumTier) => void;
  },
): TelemetryPanelNodes {
  const tickValue = document.createTextNode('0');
  const lapValue = document.createTextNode('0');
  const networkSizeValue = document.createTextNode('N0 / C0');
  const networkDeltaValue = document.createTextNode(formatNetworkDelta(0, 0));
  const lastChangeReasonValue = document.createTextNode(
    `${humanizeAdaptationReason('pending')} (local adaptation)`,
  );
  const lapTimeValue = document.createTextNode('—');
  const accelerationValue = document.createTextNode('detecting…');

  const runtimeCard = createPanelCard('Runtime Telemetry', 'Runtime Controls', [
    'Adaptation runs locally in the browser. This panel shows live telemetry.',
    'Use the observation-tier selector to switch between Tier 1 (70 inputs) and Tier 6 (124 inputs).',
  ]);
  const controlsElement = document.createElement('div');
  controlsElement.className = 'racing-controls';

  const tierSelectorElement = document.createElement('div');
  tierSelectorElement.className = 'racing-tier-selector';
  const tierSelectorLabel = document.createElement('div');
  tierSelectorLabel.className = 'racing-tier-selector__label';
  tierSelectorLabel.textContent = 'Observation Tier';
  tierSelectorElement.append(tierSelectorLabel);

  const TIER_SELECTOR_BUTTONS: readonly {
    tier: CurriculumTier;
    label: string;
  }[] = [
    { tier: 1, label: 'T1' },
    { tier: 2, label: 'T2' },
    { tier: 3, label: 'T3' },
    { tier: 4, label: 'T4' },
    { tier: 5, label: 'T5' },
    { tier: 6, label: 'T6' },
  ];
  const activeTier = options?.initialTier ?? 1;
  const tierButtons: HTMLButtonElement[] = [];
  for (const tierButtonSpec of TIER_SELECTOR_BUTTONS) {
    const tierButton = document.createElement('button');
    tierButton.type = 'button';
    tierButton.className = 'racing-tier-button';
    if (tierButtonSpec.tier === activeTier) {
      tierButton.classList.add('racing-tier-button--active');
    }
    tierButton.textContent = tierButtonSpec.label;
    tierButton.addEventListener('click', () => {
      options?.onSelectTier?.(tierButtonSpec.tier);
      for (const button of tierButtons) {
        button.classList.toggle(
          'racing-tier-button--active',
          button === tierButton,
        );
      }
    });
    tierButtons.push(tierButton);
    tierSelectorElement.append(tierButton);
  }
  controlsElement.append(tierSelectorElement);

  const telemetryGrid = document.createElement('div');
  telemetryGrid.className = 'racing-telemetry';
  telemetryGrid.append(
    buildPanelRowWithLiveNode('Laps', lapValue),
    buildPanelRowWithLiveNode('Tick', tickValue),
    buildPanelRowWithLiveNode('Best Lap', lapTimeValue),
    buildPanelRowWithLiveNode('Network Size', networkSizeValue),
    buildPanelRowWithLiveNode('Network Δ', networkDeltaValue),
    buildPanelRowWithLiveNode('Last Change', lastChangeReasonValue),
    buildPanelRowWithLiveNode('Acceleration', accelerationValue),
  );

  const syncRuntimeControls = (): void => {
    // No host-side tuning inputs to sync — telemetry is updated per-frame.
  };

  runtimeCard.bodyElement.append(controlsElement, telemetryGrid);
  region.append(runtimeCard.cardElement);

  return {
    tickValue,
    lapValue,
    networkSizeValue,
    networkDeltaValue,
    lastChangeReasonValue,
    lapTimeValue,
    accelerationValue,
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
  valueElement.className = 'racing-status-chip__value';
  valueElement.textContent = value;

  chipElement.append(labelElement, valueElement);
  return chipElement;
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
  const displayWidth = Math.min(
    MAX_CANVAS_WIDTH_PX,
    Math.max(1, Math.round(cssDisplayWidth * devicePixelRatio)),
  );
  const displayHeight = Math.min(
    MAX_CANVAS_HEIGHT_PX,
    Math.max(1, Math.round(cssDisplayHeight * devicePixelRatio)),
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
 * Creates a small deterministic public network that drives the solo browser harness.
 *
 * The network shape depends on the observation tier so the input dimension matches
 * the assembled per-car observation vector and the output dimension matches the
 * tier's control contract.
 *
 * @param observationTier - Supported observation tier (1–6) that selects the
 *   controller input/output shape. Defaults to tier 1.
 * @returns Deterministically parameterized controller network.
 * @throws {Error} When `observationTier` is outside the supported 1–6 range.
 *
 * @example
 * ```ts
 * const controller = createDeterministicRacingControllerNetwork(4);
 * console.log(controller.input, controller.output); // 103, 2
 * ```
 */
export function createDeterministicRacingControllerNetwork(
  observationTier: SupportedObservationTier = 1,
): Network {
  if (
    !Number.isInteger(observationTier) ||
    observationTier < 1 ||
    observationTier > 6
  ) {
    throw new Error(
      `Unsupported observation tier ${observationTier}. Supported tiers: 1–6.`,
    );
  }

  const resolvedInputCount =
    resolveControllerInputCountForObservationTier(observationTier);
  const resolvedOutputCount = observationTier === 2 ? 9 : 2;
  const controllerNetwork = Network.createMLP(
    resolvedInputCount,
    [...CONTROLLER_HIDDEN_LAYER_SIZES],
    resolvedOutputCount,
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
    (outputNodes.length !== 2 && outputNodes.length !== 9)
  ) {
    throw new Error(
      'Racing curriculum controller expected a N -> 4 -> 2 or N -> 4 -> 9 MLP.',
    );
  }

  hiddenNodes[0].squash = methods.Activation.relu;
  hiddenNodes[1].squash = methods.Activation.relu;
  hiddenNodes[2].squash = methods.Activation.relu;
  hiddenNodes[3].squash = methods.Activation.relu;

  outputNodes.forEach((outputNode, outputIndex) => {
    outputNode.squash = methods.Activation.tanh;
    // Keep the historical throttle-bias shortcut on the first output; all
    // other outputs default to zero bias.
    outputNode.bias = outputIndex === 0 ? 1.1 : 0;
  });

  hiddenNodes.forEach((hiddenNode) => {
    hiddenNode.bias = 0;
  });

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
 * Updates telemetry panel text nodes from the current environment state.
 *
 * @param nodes - Live text node references.
 * @param controllerNetwork - Current focused controller network.
 * @param envState - Current physics state.
 * @param completedLaps - Current completed lap count.
 * @param bestLapTimeMs - Team-level best lap time in milliseconds, or `null`
 *   when no lap has been completed yet.
 * @param latestTelemetryByCarIndex - Latest per-car runtime adaptation
 *   telemetry, keyed by car index.
 * @param focusCarIndex - Car index whose delta and reason should be shown.
 * @param networkHud - Optional network HUD nodes to update.
 * @internal
 */
function updateTelemetryPanelNodes(
  nodes: TelemetryPanelNodes,
  controllerNetwork: Network,
  envState: EnvironmentState,
  completedLaps: number,
  bestLapTimeMs: number | null,
  latestTelemetryByCarIndex: ReadonlyMap<number, RuntimeAdaptationTelemetry>,
  focusCarIndex: number,
  accelerationLabel: string,
  networkHud?: RacingNetworkHudNodes,
): void {
  const networkSize = resolveNetworkSize(controllerNetwork);
  const adaptationSummary = resolveFocusedAdaptationSummary(
    latestTelemetryByCarIndex,
    focusCarIndex,
  );

  nodes.lapValue.textContent = String(completedLaps);
  nodes.tickValue.textContent = String(envState.tick);
  nodes.lapTimeValue.textContent =
    bestLapTimeMs === null ? '—' : `${bestLapTimeMs.toFixed(0)} ms`;
  nodes.networkSizeValue.textContent = `N${networkSize.nodes} / C${networkSize.connections}`;
  nodes.networkDeltaValue.textContent = formatNetworkDelta(
    adaptationSummary.nodeDelta,
    adaptationSummary.connectionDelta,
  );
  nodes.lastChangeReasonValue.textContent = `${humanizeAdaptationReason(
    adaptationSummary.reason,
  )} (local adaptation)`;
  nodes.accelerationValue.textContent = accelerationLabel;

  if (!networkHud) {
    return;
  }

  networkHud.sizeValue.textContent = `N${networkSize.nodes} / C${networkSize.connections}`;
  networkHud.lastChangeValue.textContent = `${humanizeAdaptationReason(
    adaptationSummary.reason,
  )} (local adaptation)`;
  networkHud.statusValue.textContent = resolveNetworkHudStatus(
    true,
    adaptationSummary.trend,
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
 * Builds a compact overlay frame from the live environment state so the
 * Tier 4+ renderer can draw tire-corner colors and pit-stop occupancy overlays.
 *
 * @param envState - Current physics state.
 * @param curriculumTier - Active curriculum tier (determines feature flags).
 * @returns Packed overlay frame for {@link renderRacingFrame}, or `undefined`
 *   when the environment carries no per-car roster.
 * @internal
 */
function buildRacingRenderOverlayFrame(
  envState: EnvironmentState,
  curriculumTier: CurriculumTier,
):
  | {
      carTeam: Uint8Array;
      tireState: Float32Array;
      featureFlags: number;
      pitStatus?: Uint8Array;
    }
  | undefined {
  const cars = envState.cars;
  const carCount = cars?.length ?? 1;

  // The renderer only enables overlays when a full per-car roster is present.
  if (carCount < 2 || cars === undefined) {
    return undefined;
  }

  const carTeam = new Uint8Array(carCount);
  const tireState = new Float32Array(carCount * 4);
  for (const [carIndex, car] of cars.entries()) {
    carTeam[carIndex] = car.teamIndex;
    const offset = carIndex * 4;
    tireState[offset] = car.tireState[0];
    tireState[offset + 1] = car.tireState[1];
    tireState[offset + 2] = car.tireState[2];
    tireState[offset + 3] = car.tireState[3];
  }

  return {
    featureFlags: resolveFeatureFlagsForCurriculumTier(curriculumTier),
    carTeam,
    tireState,
    pitStatus: buildPackedPitStatus(envState, carCount),
  };
}

/** Number of pit shelf slots (three per team). */
const PIT_SLOT_COUNT = 6;
/** Number of pit shelf slots per team. */
const PIT_SLOTS_PER_TEAM = 3;
/** Sentinel car index used for an unoccupied pit shelf slot. */
const NO_CAR_INDEX = 255;

/**
 * Packs the six-record pit shelf into the renderer's per-team tuple.
 *
 * The renderer expects either a four-car stride-2 layout
 * `[teamA_car, teamA_ticks, teamB_car, teamB_ticks]` or a six-car stride-3
 * layout `[teamA_car, teamA_ticks, teamA_waiting, teamB_car, teamB_ticks,
 * teamB_waiting]`. The browser collapses each team's three shelf slots to the
 * first occupied record and zeroes the remaining slots.
 *
 * @param envState - Current physics state.
 * @param carCount - Number of cars in the race roster.
 * @returns Packed pit status tuple, or `undefined` if no occupancy data exists.
 * @internal
 */
function buildPackedPitStatus(
  envState: EnvironmentState,
  carCount: number,
): Uint8Array | undefined {
  const occupancy = envState.pitOccupancy ?? envState.pitStatus;
  if (occupancy === undefined || occupancy.length < PIT_SLOT_COUNT) {
    return undefined;
  }

  const stride = carCount >= 6 ? 3 : 2;
  const packed = new Uint8Array(stride * 2);

  for (const teamIndex of [0, 1] as const) {
    const teamBase = teamIndex * PIT_SLOTS_PER_TEAM;
    const firstOccupied = occupancy
      .slice(teamBase, teamBase + PIT_SLOTS_PER_TEAM)
      .find((record) => record.occupyingCarIndex !== NO_CAR_INDEX);

    const packedBase = teamIndex * stride;
    if (firstOccupied !== undefined) {
      packed[packedBase] = firstOccupied.occupyingCarIndex;
      packed[packedBase + 1] = firstOccupied.remainingStopTicks;
    } else {
      packed[packedBase] = NO_CAR_INDEX;
      packed[packedBase + 1] = 0;
    }
  }

  return packed;
}

/** Bit mask enabling tire-corner overlay rendering. */
const FEATURE_FLAG_TIRES_ENABLED = 0b001;
/** Bit mask enabling radio-overlay rendering. */
const FEATURE_FLAG_RADIO_ENABLED = 0b010;
/** Bit mask enabling pit-stop occupancy overlay rendering. */
const FEATURE_FLAG_PITS_ENABLED = 0b100;

/**
 * Resolves renderer feature flags from the active curriculum tier.
 *
 * @param curriculumTier - Active curriculum tier.
 * @returns Bit mask enabling tires, radio, and pit overlays for Tier 4+.
 * @internal
 */
function resolveFeatureFlagsForCurriculumTier(
  curriculumTier: CurriculumTier,
): number {
  if (curriculumTier < TIRE_WEAR_START_TIER) {
    return 0;
  }

  return (
    FEATURE_FLAG_TIRES_ENABLED |
    FEATURE_FLAG_RADIO_ENABLED |
    FEATURE_FLAG_PITS_ENABLED
  );
}

/**
 * Computes the focused car's most recent node / connection delta, adaptation
 * reason, and inferred trend from captured telemetry.
 *
 * @param latestTelemetryByCarIndex - Latest per-car runtime adaptation
 *   telemetry, keyed by car index.
 * @param focusCarIndex - Car index whose summary should be returned.
 * @returns Network delta, human-readable reason, and inferred trend.
 * @internal
 */
function resolveFocusedAdaptationSummary(
  latestTelemetryByCarIndex: ReadonlyMap<number, RuntimeAdaptationTelemetry>,
  focusCarIndex: number,
): {
  nodeDelta: number;
  connectionDelta: number;
  reason: string;
  trend: 'improving' | 'regressing' | 'flat';
} {
  const telemetry = latestTelemetryByCarIndex.get(focusCarIndex);
  if (telemetry === undefined) {
    return {
      nodeDelta: 0,
      connectionDelta: 0,
      reason: 'pending',
      trend: 'flat',
    };
  }

  const nodeDelta =
    telemetry.networkSizeAfter.nodes - telemetry.networkSizeBefore.nodes;
  const connectionDelta =
    telemetry.networkSizeAfter.connections -
    telemetry.networkSizeBefore.connections;
  const scoreDelta = telemetry.scoreAfter - telemetry.scoreBefore;

  return {
    nodeDelta,
    connectionDelta,
    reason: telemetry.reason,
    trend:
      scoreDelta > 0 ? 'improving' : scoreDelta < 0 ? 'regressing' : 'flat',
  };
}

/**
 * Maps an internal adaptation reason to a human-readable telemetry label.
 *
 * @param reason - Raw reason returned by {@link RuntimeAdaptationEngine}.
 * @returns Display-friendly label.
 * @internal
 */
function humanizeAdaptationReason(reason: string): string {
  switch (reason) {
    case 'committed':
      return 'committed';
    case 'cadence_not_reached':
      return 'cadence not reached';
    case 'insufficient_evidence':
      return 'insufficient evidence';
    case 'mutation_cooldown_active':
      return 'mutation cooldown';
    case 'rollback_cooldown_active':
      return 'rollback cooldown';
    case 'growth_throttled':
      return 'growth throttled';
    case 'no_candidate_operations':
      return 'no candidates';
    case 'safety_checks_failed':
      return 'safety checks failed';
    case 'improvement_below_threshold':
      return 'below threshold';
    default:
      return reason;
  }
}

/**
 * Formats a node/connection delta pair for the telemetry panel.
 *
 * @param nodeDelta - Change in node count.
 * @param connectionDelta - Change in connection count.
 * @returns Compact delta label such as "ΔN0 / ΔC0" or "ΔN+2 / ΔC-1".
 * @internal
 */
function formatNetworkDelta(
  nodeDelta: number,
  connectionDelta: number,
): string {
  const nodeSign = nodeDelta > 0 ? '+' : '';
  const connectionSign = connectionDelta > 0 ? '+' : '';
  return `ΔN${nodeSign}${nodeDelta} / ΔC${connectionSign}${connectionDelta}`;
}

/**
 * Formats the resolved acceleration backend status for HUD readouts.
 *
 * @param status - Resolved acceleration status from the library.
 * @param parallelVariantCount - Variant count actually requested by the demo.
 * @returns A short label such as "GPU (1024 variants)" or
 *   "CPU · GPU blocked: no WebGPU (1024 variants)".
 * @internal
 */
function formatAccelerationStatus(
  status: AccelerationStatus,
  parallelVariantCount: number,
): string {
  const modeLabel = status.mode.toUpperCase();
  const fallbackReason = status.gapReasons?.[0];
  const suffix = fallbackReason
    ? ` · ${status.mode} blocked: ${fallbackReason}`
    : '';
  return `${modeLabel}${suffix} • ${parallelVariantCount}x`;
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

  if (curriculumTier === 3) {
    return TIER_THREE_TEAM_LAYOUT;
  }

  if (curriculumTier === 1) {
    return TIER_ONE_TEAM_LAYOUT;
  }

  if (curriculumTier === 2) {
    return TIER_TWO_TEAM_LAYOUT;
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
    const carTeamIndex = teamIndex as 0 | 1;
    const longitudinalOffsetWorldUnits = -teamSlotIndex * gridSpacingWorldUnits;
    const lateralOffsetWorldUnits =
      carTeamIndex === TEAM_BLUE_INDEX ? innerOffsetWorld : -innerOffsetWorld;
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
  curriculumTier: CurriculumTier,
): CurriculumProgressState {
  return {
    tier: curriculumTier,
    lapProgress: createInitialLapProgress(trackSpec, envState),
  };
}

/**
 * Resolve the curriculum tier from caller-provided start options.
 *
 * Falls back to {@link DEFAULT_CURRICULUM_TIER} when the caller omits `tier`
 * or passes a value outside the valid `CurriculumTier` range.
 *
 * @param options - Caller options passed to {@link start}.
 * @returns The validated curriculum tier to launch.
 */
function resolveCurriculumTierFromOptions(
  options?: RacingCurriculumStartOptions,
): CurriculumTier {
  if (options?.tier === undefined) {
    return DEFAULT_CURRICULUM_TIER;
  }
  const tier = options.tier;
  if (
    typeof tier === 'number' &&
    Number.isInteger(tier) &&
    tier >= 1 &&
    tier <= 6
  ) {
    return tier as CurriculumTier;
  }
  return DEFAULT_CURRICULUM_TIER;
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
 * Resolves the median hidden-node count across all controller networks.
 *
 * Used by Gate B of the tier promotion check: the team's median hidden-node
 * count must meet or exceed the tier's N_floor before promotion is granted.
 *
 * @param controllerNetworkByCarIndex - Per-car controller networks.
 * @returns Median hidden-node count, or 0 if no networks exist.
 * @internal
 */
function resolveMedianHiddenNodeCount(
  controllerNetworkByCarIndex: Map<number, Network>,
): number {
  const hiddenCounts = Array.from(controllerNetworkByCarIndex.values())
    .map(
      (network) =>
        network.nodes.filter((node) => node.type === 'hidden').length,
    )
    .toSorted((a, b) => a - b);
  if (hiddenCounts.length === 0) {
    return 0;
  }
  return hiddenCounts[Math.floor(hiddenCounts.length / 2)]!;
}

/**
 * Resolves the mean driving quality from a car's score history.
 *
 * Mirrors the `toDrivingQuality` weighting from `runtime.adaptation.ts` since
 * that helper is not exported.
 *
 * @param scoreHistory - Rolling score window of numeric scores or composite signals.
 * @returns Mean driving quality, or 0 if history is empty.
 * @internal
 */
function resolveMeanDrivingQuality(
  scoreHistory: readonly (number | RacingQualitySignal)[],
): number {
  if (scoreHistory.length === 0) {
    return 0;
  }
  return (
    scoreHistory.reduce<number>(
      (sum: number, entry: number | RacingQualitySignal) => {
        if (typeof entry === 'number') {
          return sum + entry;
        }
        const signal = entry as RacingQualitySignal;
        return (
          sum +
          signal.trackProgress * 0.35 +
          signal.forwardSpeed * 0.25 +
          signal.headingAlignment * 0.3 -
          signal.offTrackPenalty * 0.1 +
          (signal.physicsReward ?? 0) * 0.1
        );
      },
      0,
    ) / scoreHistory.length
  );
}

/**
 * Builds an observation vector of the given input size from a single score
 * history entry.
 *
 * Composite signals tile their five fields to match the network input
 * dimension, mirroring `resolveObservationVector` from
 * `runtime.adaptation.ts` which is not exported.
 *
 * @param scoreHistory - Rolling score window.
 * @param inputSize - Number of input nodes in the network.
 * @returns Input vector suitable for `network.activate`.
 * @internal
 */
function resolveObservationVectorForNetwork(
  scoreHistory: readonly (number | RacingQualitySignal)[],
  inputSize: number,
): number[] {
  if (scoreHistory.length === 0 || inputSize <= 0) {
    return Array(inputSize).fill(0);
  }
  const entry = scoreHistory.at(-1)!;
  if (typeof entry === 'number') {
    return Array(inputSize).fill(entry);
  }
  const signalValues = [
    entry.trackProgress,
    entry.forwardSpeed,
    entry.headingAlignment,
    entry.offTrackPenalty,
    entry.physicsReward ?? 0,
  ];
  const observation: number[] = [];
  for (let i = 0; i < inputSize; i++) {
    observation.push(signalValues[i % signalValues.length] ?? 0);
  }
  return observation;
}

/**
 * Builds promotion candidates from per-car networks and score histories.
 *
 * Each candidate includes a forward-pass output sample for behavioral diversity
 * computation. The observation vector is derived from the car's last score
 * history entry and tiled to the network's input size.
 *
 * @param controllerNetworkByCarIndex - Per-car controller networks.
 * @param scoreHistoryByCarIndex - Per-car score histories.
 * @param bestLapTimeMs - Team-level best lap time in ms.
 * @returns Array of promotion candidates sorted by car index.
 * @internal
 */
function buildPromotionCandidates(
  controllerNetworkByCarIndex: Map<number, Network>,
  scoreHistoryByCarIndex: Map<number, (number | RacingQualitySignal)[]>,
  bestLapTimeMs: number,
): PromotionCandidate[] {
  const candidates: PromotionCandidate[] = [];
  for (const [carIndex, network] of controllerNetworkByCarIndex) {
    const hiddenNodeCount = network.nodes.filter(
      (node) => node.type === 'hidden',
    ).length;
    const scoreHistory = scoreHistoryByCarIndex.get(carIndex) ?? [];
    const drivingQuality = resolveMeanDrivingQuality(scoreHistory);
    const observation = resolveObservationVectorForNetwork(
      scoreHistory,
      network.input,
    );
    const outputSample = [...network.activate(observation)];
    candidates.push({
      carIndex,
      bestLapTimeMs,
      medianHiddenNodeCount: hiddenNodeCount,
      drivingQuality,
      network,
      outputSample,
    });
  }
  return candidates.toSorted((a, b) => a.carIndex - b.carIndex);
}

/**
 * Selects agents for tier promotion based on configurable criteria.
 *
 * Sorts candidates by the selected criterion and picks the top fraction
 * defined by `selectionRatio`, ensuring at least `minSelected` agents.
 *
 * @param candidates - Promotion candidates to select from.
 * @param config - Selection configuration.
 * @returns Indices of selected car indices for promotion.
 * @internal
 */
function selectForPromotion(
  candidates: PromotionCandidate[],
  config: PromotionSelectionConfig,
): number[] {
  if (candidates.length === 0) {
    return [];
  }
  const sorted = [...candidates].toSorted((a, b) => {
    switch (config.selectionCriteria) {
      case 'bestLapTime':
        return a.bestLapTimeMs - b.bestLapTimeMs;
      case 'mostGrowth':
        return b.medianHiddenNodeCount - a.medianHiddenNodeCount;
      case 'bestDrivingQuality':
        return b.drivingQuality - a.drivingQuality;
      default:
        return b.drivingQuality - a.drivingQuality;
    }
  });
  const count = Math.max(
    config.minSelected,
    Math.ceil(sorted.length * config.selectionRatio),
  );
  return sorted.slice(0, count).map((c) => c.carIndex);
}

/**
 * Computes behavioral diversity across promotion candidates using output
 * variance.
 *
 * For each candidate, computes the mean squared deviation of its output sample
 * from the population mean across all output dimensions. Candidates with higher
 * diversity scores produce outputs that differ more from the average.
 *
 * @param candidates - Promotion candidates with output samples.
 * @returns Map from car index to diversity score.
 * @internal
 */
function computeBehavioralDiversity(
  candidates: PromotionCandidate[],
): Map<number, number> {
  if (candidates.length <= 1) {
    return new Map(candidates.map((c) => [c.carIndex, 0]));
  }
  const outputDim = candidates[0]!.outputSample.length;
  const populationMean = new Array(outputDim).fill(0);
  for (const candidate of candidates) {
    for (let dim = 0; dim < outputDim; dim++) {
      populationMean[dim] += candidate.outputSample[dim] ?? 0;
    }
  }
  for (let dim = 0; dim < outputDim; dim++) {
    populationMean[dim] /= candidates.length;
  }

  const diversityByCarIndex = new Map<number, number>();
  for (const candidate of candidates) {
    let squaredDeviation = 0;
    for (let dim = 0; dim < outputDim; dim++) {
      const diff = (candidate.outputSample[dim] ?? 0) - populationMean[dim]!;
      squaredDeviation += diff * diff;
    }
    diversityByCarIndex.set(candidate.carIndex, squaredDeviation / outputDim);
  }
  return diversityByCarIndex;
}

/**
 * Ensures at least one behaviorally diverse agent is retained in the promoted
 * set.
 *
 * If the most diverse candidate is not already in the selected set, replaces
 * the last selected agent with it to preserve behavioral diversity.
 *
 * @param selectedIndices - Currently selected car indices for promotion.
 * @param candidates - All promotion candidates.
 * @returns Updated selected indices with diversity preservation.
 * @internal
 */
function retainDiverseAgent(
  selectedIndices: number[],
  candidates: PromotionCandidate[],
): number[] {
  if (candidates.length <= 1 || selectedIndices.length <= 1) {
    return selectedIndices;
  }
  const diversityByCarIndex = computeBehavioralDiversity(candidates);
  const mostDiverseCandidate = candidates.toSorted(
    (a, b) =>
      (diversityByCarIndex.get(b.carIndex) ?? 0) -
      (diversityByCarIndex.get(a.carIndex) ?? 0),
  )[0]!;
  const selectedSet = new Set(selectedIndices);
  if (selectedSet.has(mostDiverseCandidate.carIndex)) {
    return selectedIndices;
  }
  // Replace the last selected agent with the most diverse one.
  const updated = [...selectedIndices];
  updated[updated.length - 1] = mostDiverseCandidate.carIndex;
  return updated;
}

/**
 * Resolves tier promotion by evaluating two gates and selecting agents.
 *
 * Gate A (lap-time improvement): the team must show lap-time improvement. The
 * first lap is accepted as a baseline; subsequent laps must be faster.
 * Gate B (N_floor): the team's median hidden-node count must meet or exceed
 * the tier's N_floor.
 *
 * When both gates pass, agents are selected for promotion using the configured
 * selection criteria, and behavioral diversity is preserved.
 *
 * @param params - Promotion parameters including tier, lap data, networks, and config.
 * @returns Promotion decision with next tier, advance flag, and promoted car indices.
 * @internal
 */
function resolveTierPromotion(params: {
  currentTier: CurriculumTier;
  completedLaps: number;
  lapTimeImproved: boolean;
  bestLapTimeMs: number | null;
  controllerNetworkByCarIndex: Map<number, Network>;
  scoreHistoryByCarIndex: Map<number, (number | RacingQualitySignal)[]>;
  selectionConfig: PromotionSelectionConfig;
}): {
  nextTier: CurriculumTier;
  didAdvance: boolean;
  remainingLaps: number;
  promotedCarIndices: number[];
} {
  const {
    currentTier,
    completedLaps,
    lapTimeImproved,
    bestLapTimeMs,
    controllerNetworkByCarIndex,
    scoreHistoryByCarIndex,
    selectionConfig,
  } = params;

  if (currentTier >= MAX_CURRICULUM_TIER) {
    return {
      nextTier: currentTier,
      didAdvance: false,
      remainingLaps: completedLaps,
      promotedCarIndices: [],
    };
  }

  // Gate A: Lap-time improvement.
  if (!lapTimeImproved) {
    return {
      nextTier: currentTier,
      didAdvance: false,
      remainingLaps: completedLaps,
      promotedCarIndices: [],
    };
  }

  // Gate B: N_floor — median hidden-node count must meet or exceed tier floor.
  const medianHiddenCount = resolveMedianHiddenNodeCount(
    controllerNetworkByCarIndex,
  );
  const nFloor = TIER_N_FLOOR[currentTier] ?? 0;
  if (medianHiddenCount < nFloor) {
    return {
      nextTier: currentTier,
      didAdvance: false,
      remainingLaps: completedLaps,
      promotedCarIndices: [],
    };
  }

  // Both gates passed: select agents for promotion.
  const effectiveBestLapTime = bestLapTimeMs ?? 0;
  const candidates = buildPromotionCandidates(
    controllerNetworkByCarIndex,
    scoreHistoryByCarIndex,
    effectiveBestLapTime,
  );
  const selectedIndices = selectForPromotion(candidates, selectionConfig);
  const promotedIndices = retainDiverseAgent(selectedIndices, candidates);

  return {
    nextTier: (currentTier + 1) as CurriculumTier,
    didAdvance: true,
    remainingLaps: 0,
    promotedCarIndices: promotedIndices,
  };
}

/** Lap threshold required before a tier-1-through-4 car is eligible for promotion. */
const LAP_COUNT_PROMOTION_THRESHOLD = 3;

/**
 * Simplified lap-count-based tier promotion check used by the all-cars
 * methodology to decide whether every car on the grid has completed enough
 * laps to advance as a group.
 *
 * Tiers 1–4 advance after `LAP_COUNT_PROMOTION_THRESHOLD` completed laps.
 * Tier 5 holds for cross-team fairness confirmation and never auto-advances.
 * Tier 6 is the ceiling and cannot advance further.
 *
 * @param currentTier - Current curriculum tier (1–6).
 * @param completedLaps - Number of laps completed by the car at this tier.
 * @returns Promotion decision with next tier, advance flag, and remaining laps.
 *
 * @example
 * ```ts
 * const result = resolveTierPromotionFromLapCount(1, 3);
 * console.log(result); // { nextTier: 2, didAdvance: true, remainingLaps: 0 }
 * ```
 */
export function resolveTierPromotionFromLapCount(
  currentTier: number,
  completedLaps: number,
): { nextTier: number; didAdvance: boolean; remainingLaps: number } {
  if (currentTier >= MAX_CURRICULUM_TIER) {
    return {
      nextTier: MAX_CURRICULUM_TIER,
      didAdvance: false,
      remainingLaps: completedLaps,
    };
  }

  // Tier 5 holds for cross-team fairness confirmation.
  if (currentTier >= 5) {
    return {
      nextTier: currentTier,
      didAdvance: false,
      remainingLaps: completedLaps,
    };
  }

  if (completedLaps >= LAP_COUNT_PROMOTION_THRESHOLD) {
    return {
      nextTier: currentTier + 1,
      didAdvance: true,
      remainingLaps: 0,
    };
  }

  return {
    nextTier: currentTier,
    didAdvance: false,
    remainingLaps: LAP_COUNT_PROMOTION_THRESHOLD - completedLaps,
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

/**
 * Maximum forward speed in world units per second, used to normalize
 * per-car velocity into a `[0, 1]` driving-quality signal.
 *
 * Mirrors `MAX_FORWARD_SPEED_UNITS_PER_SECOND` from
 * `environment.step.service.ts` so the browser entry does not depend on
 * an unexported module-local constant.
 */
const SIGNAL_MAX_FORWARD_SPEED_UNITS_PER_SECOND = 108;

/**
 * Fixed physics timestep in seconds (60 Hz), used to convert per-tick
 * position deltas into a normalized speed signal.
 */
const SIGNAL_FIXED_TIMESTEP_SECONDS = 1 / 60;

/**
 * Resolve the closest spline sample index for a specific car position.
 *
 * @param trackSpec - Track specification with ordered spline samples.
 * @param carX - Car X position in world units.
 * @param carY - Car Y position in world units.
 * @returns Global index of the closest spline sample, or 0 if the track
 * has no samples.
 * @internal
 */
function resolveClosestSplineSampleIndexForCar(
  trackSpec: TrackSpec,
  carX: number,
  carY: number,
): number {
  if (trackSpec.splineSamples.length === 0) {
    return 0;
  }

  let closestIndex = 0;
  let closestDistance = Number.POSITIVE_INFINITY;

  for (const sample of trackSpec.splineSamples) {
    const distance = Math.hypot(sample.x - carX, sample.y - carY);
    if (distance < closestDistance) {
      closestDistance = distance;
      closestIndex = sample.globalIndex;
    }
  }

  return closestIndex;
}

/**
 * Compute per-car track progress as a normalized `[0, 1]` value from the
 * car's physical position on the track, not from shared curriculum state.
 *
 * @param trackSpec - Track specification with ordered spline samples.
 * @param carState - Per-car racing state, or undefined if the car slot is
 * empty.
 * @returns Normalized track progress fraction.
 * @internal
 */
function resolvePerCarTrackProgress(
  trackSpec: TrackSpec,
  carState: RacingCarState | undefined,
): number {
  if (!carState || trackSpec.splineSamples.length === 0) {
    return 0;
  }

  const closestIndex = resolveClosestSplineSampleIndexForCar(
    trackSpec,
    carState.carX,
    carState.carY,
  );

  return closestIndex / trackSpec.splineSamples.length;
}

/**
 * Compute per-car forward speed as a normalized `[0, 1]` value from the
 * physical position delta between consecutive ticks, not from control
 * output.
 *
 * @param carState - Per-car racing state, or undefined if the car slot is
 * empty.
 * @param previousPosition - Previous car position `{ carX, carY }`, or
 * undefined on the first tick.
 * @returns Normalized forward speed fraction.
 * @internal
 */
function resolvePerCarForwardSpeed(
  carState: RacingCarState | undefined,
  previousPosition: { carX: number; carY: number } | undefined,
): number {
  if (!carState || !previousPosition) {
    return 0;
  }

  const deltaX = carState.carX - previousPosition.carX;
  const deltaY = carState.carY - previousPosition.carY;
  const distanceWorld = Math.hypot(deltaX, deltaY);
  const speedUnitsPerSecond = distanceWorld / SIGNAL_FIXED_TIMESTEP_SECONDS;
  const normalizedSpeed =
    speedUnitsPerSecond / SIGNAL_MAX_FORWARD_SPEED_UNITS_PER_SECOND;

  return Math.max(0, Math.min(1, normalizedSpeed));
}

/**
 * Compute per-car heading alignment as a `[0, 1]` value from the dot
 * product of the car's heading vector and the track tangent at the closest
 * spline sample.
 *
 * @param trackSpec - Track specification with ordered spline samples.
 * @param carState - Per-car racing state, or undefined if the car slot is
 * empty.
 * @returns Normalized heading alignment fraction.
 * @internal
 */
function resolvePerCarHeadingAlignment(
  trackSpec: TrackSpec,
  carState: RacingCarState | undefined,
): number {
  if (!carState || trackSpec.splineSamples.length === 0) {
    return 0;
  }

  const closestIndex = resolveClosestSplineSampleIndexForCar(
    trackSpec,
    carState.carX,
    carState.carY,
  );

  const frame = resolveSplineSampleFrame(trackSpec.splineSamples, closestIndex);

  const carHeadingX = Math.cos(carState.carHeading);
  const carHeadingY = Math.sin(carState.carHeading);
  const dotProduct =
    carHeadingX * Math.cos(frame.tangentHeadingRadians) +
    carHeadingY * Math.sin(frame.tangentHeadingRadians);

  return Math.max(0, (dotProduct + 1) / 2);
}

/**
 * Compute per-car off-track penalty as a `[0, 1]` value from the lateral
 * distance to the closest spline sample, normalized by half the track
 * width at that sample.
 *
 * @param trackSpec - Track specification with ordered spline samples.
 * @param carState - Per-car racing state, or undefined if the car slot is
 * empty.
 * @returns Normalized off-track penalty fraction.
 * @internal
 */
function resolvePerCarOffTrackPenalty(
  trackSpec: TrackSpec,
  carState: RacingCarState | undefined,
): number {
  if (!carState || trackSpec.splineSamples.length === 0) {
    return 0;
  }

  const closestIndex = resolveClosestSplineSampleIndexForCar(
    trackSpec,
    carState.carX,
    carState.carY,
  );

  const closestSample = trackSpec.splineSamples[closestIndex];
  if (!closestSample) {
    return 0;
  }

  const lateralDistance = Math.hypot(
    carState.carX - closestSample.x,
    carState.carY - closestSample.y,
  );
  const halfWidth = Math.max(closestSample.width / 2, 1);

  return Math.max(0, Math.min(1, lateralDistance / halfWidth));
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
 * Resolves the controller input dimension for a supported observation tier.
 *
 * The browser harness remaps every tier to a stable channel count that stays
 * byte-for-byte compatible with the coevolution service's controller input
 * dimension. Tier 6 is explicitly assigned the literal value `124` so the
 * source remains greppable and stable under refactor.
 *
 * @param observationTier - Active observation tier (1–6).
 * @returns Number of input channels required by the tier controller.
 * @internal
 */
function resolveControllerInputCountForObservationTier(
  observationTier: SupportedObservationTier,
): number {
  const INPUT_COUNTS_BY_TIER: Record<SupportedObservationTier, number> = {
    1: 70,
    2: 77,
    3: 91,
    4: TOTAL_TIER4_INPUT_SIZE,
    5: TOTAL_TIER4_INPUT_SIZE,
    6: 124,
  };

  return INPUT_COUNTS_BY_TIER[observationTier];
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
 * Computes one control output per car from the active per-car controller map.
 *
 * Each controller receives its own per-car observation state via
 * `derivePerCarObservationState` so that every agent acts on its own pose,
 * team, and tire state. The resulting array is aligned to roster order.
 *
 * @param controllerByCarIndex - Per-car controller map keyed by car index.
 * @param envState - Current multi-car environment snapshot.
 * @param trackSpec - Frozen track geometry.
 * @returns Per-car control array aligned to roster order.
 * @internal
 */
function resolvePerCarControls(
  controllerByCarIndex: Map<number, NgeController>,
  envState: EnvironmentState,
  trackSpec: TrackSpec,
): readonly CarControlOutput[] {
  const rosterSize = Math.max(1, envState.cars?.length ?? 1);

  return Array.from({ length: rosterSize }, (_, carIndex) => {
    const perCarController = controllerByCarIndex.get(carIndex);

    if (perCarController === undefined) {
      return { throttle: 0, steer: 0 };
    }

    return perCarController.computeControl(
      derivePerCarObservationState(envState, carIndex),
      trackSpec,
    );
  });
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
 * Carries the evolved controller phenotype into the promoted observation tier.
 *
 * Instead of creating a fresh MLP and copying only role-matched weights, this
 * function extends the existing evolved network in-place. New input nodes
 * needed for the wider observation surface are added and connected to existing
 * hidden nodes with zero-weight connections. All existing hidden nodes, output
 * nodes, connections, learned weights, and NGE-adaptation-grown structure are
 * preserved.
 *
 * @param sourceNetwork - Evolved network from the previous tier.
 * @param nextObservationTier - Observation tier for the promoted curriculum tier.
 * @returns The same network instance, extended with new input nodes if needed.
 * @internal
 */
function remapControllerNetworkForObservationTier(
  sourceNetwork: Network,
  nextObservationTier: SupportedObservationTier,
): Network {
  const nextInputCount =
    resolveControllerInputCountForObservationTier(nextObservationTier);
  const currentInputNodes = resolveSortedNodesByType(sourceNetwork, 'input');
  const currentInputCount = currentInputNodes.length;

  if (nextInputCount === currentInputCount) {
    return sourceNetwork;
  }

  // Step 1: Remove excess input nodes when the next tier is narrower than the
  // current network. Existing hidden and output structure is preserved because
  // excess input nodes only own outgoing zero-weight connections.
  if (nextInputCount < currentInputCount) {
    const nodesToRemove = currentInputNodes.slice(nextInputCount);
    for (const inputNode of nodesToRemove) {
      sourceNetwork.remove(inputNode);
    }
    sourceNetwork.input = nextInputCount;
    return sourceNetwork;
  }

  // Step 2: Determine how many new input nodes are needed.
  const newInputCount = nextInputCount - currentInputCount;

  // Step 3: Add new input nodes to the existing network.
  const hiddenNodes = resolveSortedNodesByType(sourceNetwork, 'hidden');

  for (let inputIndex = 0; inputIndex < newInputCount; inputIndex++) {
    const newInputNode = new Node('input');
    newInputNode.bias = 0;
    sourceNetwork.nodes.push(newInputNode);

    // Step 4: Connect each new input node to every existing hidden node with
    // zero-weight connections so the new inputs are inert until learning
    // shapes them.
    for (const hiddenNode of hiddenNodes) {
      sourceNetwork.connect(newInputNode, hiddenNode, 0);
    }
  }

  // Step 5: Update the network's input count so activation vectors match the
  // new observation width.
  sourceNetwork.input = nextInputCount;

  return sourceNetwork;
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
  pendingWorkerSteps: Map<string, PendingWorkerStep>,
  requestId: string,
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
  pendingWorkerSteps: Map<string, PendingWorkerStep>,
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
