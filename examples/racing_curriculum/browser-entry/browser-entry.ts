import { createRacingHost } from './host/host';
import { Network, methods } from '../../../src/browser-entry.ts';
import { generateTrack } from '../track/track.generator';
import { stepEnvironment } from '../environment/environment.step.service';
import {
  computeWorldTransform,
  createRacingRenderState,
  renderRacingFrame,
} from '../renderer/racing.renderer';
import { resolveSplineSampleFrame } from '../track/track.spline.utils';
import {
  createNgeController,
  resolveGuidanceAlphaForTier,
} from '../controller/nge.controller';
import type { EnvironmentState } from '../environment/environment.types';
import {
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_NEON_PALETTE,
  FLAPPY_SCREEN_PADDING_PX,
  FLAPPY_UI_CANVAS_INSET_SHADOW,
  FLAPPY_UI_DOUBLE_PANEL_BORDER,
  FLAPPY_UI_NETWORK_HOST_BACKGROUND,
  FLAPPY_UI_OUTER_FRAME_BACKGROUND,
  FLAPPY_UI_OUTER_FRAME_MIN_SIDE_PADDING_PX,
  FLAPPY_UI_OUTER_FRAME_SIDE_PADDING_OFFSET_PX,
  FLAPPY_UI_UNIFIED_INSET_SHADOW,
} from '../../flappy_bird/constants/constants';
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

/** Track determinism key: seed 42, layout v1, medium size. */
const DEMO_TRACK_SEED = 42;
const DEMO_TRACK_LAYOUT_VERSION = 1;
const DEMO_TRACK_SIZE_BUCKET = 'medium';
/** Active browser harness tier for the Step 04 NGE controller integration. */
const ACTIVE_CURRICULUM_TIER = 1;
/** Tier 1 observation width before the Tier 2 radio tail is appended. */
const CONTROLLER_INPUT_COUNT = 70;
/** Small hidden layer used to pin a deterministic solo driving policy. */
const CONTROLLER_HIDDEN_LAYER_SIZES = [4] as const;
/** Observation channel index for the lateral-offset feature. */
const OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET = 16;
/** Observation channel index for the heading-error feature. */
const OBSERVATION_INDEX_OPTIMAL_LINE_HEADING_ERROR = 17;
/** Observation channel index for the lateral-speed feature. */
const OBSERVATION_INDEX_LATERAL_SPEED = 5;

// ── Panel text labels ─────────────────────────────────────────────────────────

const CANVAS_TOOLTIP_HEADING = 'Track Playback';
const CANVAS_TOOLTIP_LINES = [
  'This left stage now runs the Tier 1 solo harness: one car, one deterministic seed, and live NGE inference.',
  'The visual track stays spline-smoothed, but the control loop now flows through the owner-local observation assembler and the public Network.activate(...) seam.',
  'Later tiers keep this shell and widen the same controller path with self-radio and race-pack authority instead of replacing the browser contract again.',
];
const CONTROLLER_TOOLTIP_HEADING = 'Controller Status';
const CONTROLLER_TOOLTIP_LINES = [
  'This panel now reports the live Tier 1 NGE controller instead of the old scripted baseline seam.',
  'The browser harness keeps the same fixed-timestep telemetry loop, but the control path now runs through the public network inference surface.',
  'Tier 2 reuses this seam and only widens the observation vector plus the self-radio tail.',
];
const NETWORK_SLOT_TOOLTIP_HEADING = 'Future Network View';
const NETWORK_SLOT_TOOLTIP_LINES = [
  'The right column is still reserved for the richer live controller/network explanation surface.',
  'Step 04 keeps the slot informative rather than empty while the controller seam is now genuinely live.',
];
const TELEMETRY_TOOLTIP_HEADING = 'Telemetry';
const TELEMETRY_TOOLTIP_LINES = [
  'Telemetry reports the authoritative fixed-timestep playback state for the visible car.',
  'These numbers are the baseline proof surface while the richer race-pack and team dashboards are still deferred.',
];
const RACE_PACK_TOOLTIP_HEADING = 'Race Pack Slot';
const RACE_PACK_TOOLTIP_LINES = [
  'Lap counters, sector timing, and multi-car comparisons belong here once race-pack authority exists.',
  'Tier 0 keeps the slot visible so later tiers can populate it without changing the browser-shell contract.',
];

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

/** Live-updating text node references for the network panel. */
interface NetworkPanelNodes {
  throttleValue: Text;
  steerValue: Text;
  tickValue: Text;
}

/** Live-updating text node references for the telemetry panel. */
interface TelemetryPanelNodes {
  posXValue: Text;
  posYValue: Text;
  headingValue: Text;
  tickValue: Text;
}

/**
 * Starts the Tier 0 racing curriculum browser demo.
 *
 * Sets up the three-panel layout (canvas, network info, telemetry), generates
 * a deterministic track, and launches a `requestAnimationFrame` animation loop
 * with fixed-timestep physics driven by a deterministic NGE controller network.
 *
 * @param container - Host element or element id.
 * @returns Lightweight run handle.
 *
 * @example
 * ```ts
 * const handle = await start('racing-curriculum-output');
 * // Later: handle.stop();
 * ```
 */
export async function start(
  container: HTMLElement | string = 'racing-curriculum-output',
): Promise<RacingCurriculumRunHandle> {
  // Step 1: Resolve container and build host DOM shell.
  const containerElement = resolveContainerElement(container);
  injectRacingStyles();
  const hostHandle = createRacingHost(containerElement);
  setupCanvasStage(hostHandle.canvasRegionElement, hostHandle.canvasElement);

  // Step 2: Generate deterministic track.
  const trackSpec = generateTrack({
    seed: DEMO_TRACK_SEED,
    layoutVersion: DEMO_TRACK_LAYOUT_VERSION,
    sizeBucket: DEMO_TRACK_SIZE_BUCKET,
  });

  // Step 3: Initialise car on the sampled spline lane center and tangent.
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
  let envState: EnvironmentState = {
    tick: 0,
    carX: firstSplineSample?.x ?? firstSegment.startX,
    carY: firstSplineSample?.y ?? firstSegment.startY,
    carHeading: initialHeading,
  };

  // Step 4: Initialise the deterministic NGE controller and render state.
  const controller = createNgeController(
    createDeterministicRacingControllerNetwork(),
    {
      tier: ACTIVE_CURRICULUM_TIER,
    },
  );
  const guidanceAlpha = resolveGuidanceAlphaForTier(ACTIVE_CURRICULUM_TIER);
  const renderState = createRacingRenderState();

  // Step 5: Build info panels inside the host regions.
  const networkPanelNodes = setupNetworkPanel(hostHandle.networkRegionElement);
  const telemetryPanelNodes = setupTelemetryPanel(
    hostHandle.visualizerRegionElement,
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
  let lastControlOutput = controller.computeControl(envState, trackSpec);

  const animationStep = (nowMs: number): void => {
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
      lastControlOutput = controller.computeControl(envState, trackSpec);
      envState = stepEnvironment(envState, lastControlOutput);
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

    // Update info panels (text-only, no innerHTML churn).
    updateNetworkPanelNodes(
      networkPanelNodes,
      lastControlOutput.throttle,
      lastControlOutput.steer,
      envState.tick,
    );
    updateTelemetryPanelNodes(telemetryPanelNodes, envState);

    animationFrameId = requestAnimationFrame(animationStep);
  };

  hostHandle.rootElement.dataset.running = 'true';
  animationFrameId = requestAnimationFrame(animationStep);

  const handle: RacingCurriculumRunHandle = {
    done: Promise.resolve(),
    isRunning: () => running,
    stop: () => {
      if (!running) return;
      running = false;
      window.removeEventListener('resize', handleViewportResize);
      cancelAnimationFrame(animationFrameId);
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
      grid-template-rows: minmax(0, 1fr) auto;
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
      grid-template-rows: auto auto auto;
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
    .racing-host__region--visualizer {
      grid-column: 1 / -1;
      grid-row: 2;
    }
    .racing-host--narrow .racing-host__region--visualizer {
      grid-row: 3;
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
 */
function setupCanvasStage(
  region: HTMLElement,
  canvasElement: HTMLCanvasElement,
): void {
  region.replaceChildren();

  const stageCardElement = document.createElement('div');
  stageCardElement.className = 'racing-stage-card';

  const headerElement = createPanelHeader(
    'Track Playback',
    CANVAS_TOOLTIP_HEADING,
    CANVAS_TOOLTIP_LINES,
  );
  const subtitleElement = document.createElement('div');
  subtitleElement.className = 'racing-stage-card__subtitle';
  subtitleElement.textContent =
    'Tier 1 solo NGE harness with a spline-smoothed visual circuit and faded optimal-line guidance.';

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
  footerElement.textContent =
    'The car and controller are both live in this solo harness. The help chips call out which surrounding UI slots are active today and which still stay future-facing.';

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
 * Populates the network-info region with controller status rows.
 *
 * Creates DOM structure once; returns text node references for live updates.
 *
 * @param region - The network panel host element.
 * @returns References to the live-updating text nodes.
 */
function setupNetworkPanel(region: HTMLElement): NetworkPanelNodes {
  region.replaceChildren();

  const sideStackElement = document.createElement('div');
  sideStackElement.className = 'racing-side-stack';

  const controllerCard = createPanelCard(
    'Controller Status',
    CONTROLLER_TOOLTIP_HEADING,
    CONTROLLER_TOOLTIP_LINES,
  );

  const throttleValue = document.createTextNode('0.00');
  const steerValue = document.createTextNode('0.00');
  const tickValue = document.createTextNode('0');

  controllerCard.bodyElement.append(
    buildPanelRow('Controller', 'Live NGE controller'),
    buildPanelRow('Live NGE AI', 'Active'),
    buildPanelRowWithLiveNode('Throttle', throttleValue),
    buildPanelRowWithLiveNode('Steer', steerValue),
    buildPanelRowWithLiveNode('Tick', tickValue),
  );

  const noteElement = document.createElement('div');
  noteElement.className = 'racing-panel__note';
  noteElement.textContent =
    'This controller now runs through the owner-local observation seam and the public Network.activate(...) surface. Tier 2 can widen the same seam with self-radio channels without reshaping the browser shell.';
  controllerCard.bodyElement.append(noteElement);

  const networkSlotCard = createPanelCard(
    'Network View Slot',
    NETWORK_SLOT_TOOLTIP_HEADING,
    NETWORK_SLOT_TOOLTIP_LINES,
  );
  networkSlotCard.bodyElement.append(
    buildPanelRow('Status', 'Future-facing'),
    buildPanelRow('Current role', 'Explain the live seam'),
    buildCallout(
      'This right-side column stays visible on purpose: the controller seam is already live, but the richer network visualization still lands later without forcing another layout rewrite.',
    ),
  );

  sideStackElement.append(
    controllerCard.cardElement,
    networkSlotCard.cardElement,
  );
  region.append(sideStackElement);

  return { throttleValue, steerValue, tickValue };
}

/**
 * Populates the telemetry region with car state readouts.
 *
 * Creates DOM structure once; returns text node references for live updates.
 *
 * @param region - The visualizer panel host element.
 * @returns References to the live-updating text nodes.
 */
function setupTelemetryPanel(region: HTMLElement): TelemetryPanelNodes {
  region.replaceChildren();

  const lowerPanelsElement = document.createElement('div');
  lowerPanelsElement.className = 'racing-lower-panels';

  const telemetryCard = createPanelCard(
    'Telemetry',
    TELEMETRY_TOOLTIP_HEADING,
    TELEMETRY_TOOLTIP_LINES,
  );

  const telemetryGrid = document.createElement('div');
  telemetryGrid.className = 'racing-telemetry';

  const posXValue = document.createTextNode('0.0');
  const posYValue = document.createTextNode('0.0');
  const headingValue = document.createTextNode('0°');
  const tickValue = document.createTextNode('0');

  telemetryGrid.append(
    buildPanelRowWithLiveNode('Pos X', posXValue),
    buildPanelRowWithLiveNode('Pos Y', posYValue),
    buildPanelRowWithLiveNode('Heading', headingValue),
    buildPanelRowWithLiveNode('Tick', tickValue),
  );

  telemetryCard.bodyElement.append(telemetryGrid);

  const racePackCard = createPanelCard(
    'Race Pack Slot',
    RACE_PACK_TOOLTIP_HEADING,
    RACE_PACK_TOOLTIP_LINES,
  );
  racePackCard.bodyElement.append(
    buildPanelRow('Lap counter', 'Deferred'),
    buildPanelRow('Sector timing', 'Deferred'),
    buildCallout(
      'This panel is intentionally informative instead of blank: later tiers will populate it with lap, sector, and opponent context once the race-pack authority exists.',
    ),
  );

  lowerPanelsElement.append(
    telemetryCard.cardElement,
    racePackCard.cardElement,
  );
  region.append(lowerPanelsElement);

  return { posXValue, posYValue, headingValue, tickValue };
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
 */
function syncCanvasToDisplaySize(canvasElement: HTMLCanvasElement): void {
  const devicePixelRatio = Math.min(
    window.devicePixelRatio || 1,
    MAX_CANVAS_DEVICE_PIXEL_RATIO,
  );
  const displayWidth = Math.max(
    1,
    Math.round(canvasElement.clientWidth * devicePixelRatio),
  );
  const displayHeight = Math.max(
    1,
    Math.round(canvasElement.clientHeight * devicePixelRatio),
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
 * Builds a static label/value panel row element.
 *
 * @param label - Left label text.
 * @param value - Right value text (static).
 * @returns Completed row element.
 */
function buildPanelRow(label: string, value: string): HTMLElement {
  const row = document.createElement('div');
  row.className = 'racing-panel__row';

  const labelSpan = document.createElement('span');
  labelSpan.className = 'racing-panel__label';
  labelSpan.textContent = label;

  const valueSpan = document.createElement('span');
  valueSpan.className = 'racing-panel__value';
  valueSpan.textContent = value;

  row.append(labelSpan, valueSpan);
  return row;
}

/**
 * Builds a label/value panel row element whose value is backed by a live text node.
 *
 * @param label - Left label text.
 * @param liveTextNode - Text node whose content will be mutated each frame.
 * @returns Completed row element.
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
 * Updates network panel text nodes for the current tick.
 *
 * @param nodes - Live text node references.
 * @param throttle - Current throttle output.
 * @param steer - Current steer output.
 * @param tick - Current simulation tick.
 */
function updateNetworkPanelNodes(
  nodes: NetworkPanelNodes,
  throttle: number,
  steer: number,
  tick: number,
): void {
  nodes.throttleValue.textContent = throttle.toFixed(2);
  nodes.steerValue.textContent = steer.toFixed(3);
  nodes.tickValue.textContent = String(tick);
}

/**
 * Creates a small deterministic public network that drives the solo browser harness.
 *
 * @returns Deterministically parameterized controller network.
 */
function createDeterministicRacingControllerNetwork(): Network {
  const controllerNetwork = Network.createMLP(
    CONTROLLER_INPUT_COUNT,
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
    inputNodes.length !== CONTROLLER_INPUT_COUNT ||
    hiddenNodes.length !== CONTROLLER_HIDDEN_LAYER_SIZES[0] ||
    outputNodes.length !== 2
  ) {
    throw new Error(
      'Racing curriculum controller expected a 70 -> 4 -> 2 MLP.',
    );
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
      1,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(
          inputNodes[OBSERVATION_INDEX_OPTIMAL_LINE_HEADING_ERROR],
        ),
        resolveNodeIndex(hiddenNodes[1]),
      ),
      -1,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(
          inputNodes[OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET],
        ),
        resolveNodeIndex(hiddenNodes[2]),
      ),
      1,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(
          inputNodes[OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET],
        ),
        resolveNodeIndex(hiddenNodes[3]),
      ),
      -1,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(inputNodes[OBSERVATION_INDEX_LATERAL_SPEED]),
        resolveNodeIndex(hiddenNodes[2]),
      ),
      0.25,
    ],
    [
      createEdgeKey(
        resolveNodeIndex(inputNodes[OBSERVATION_INDEX_LATERAL_SPEED]),
        resolveNodeIndex(hiddenNodes[3]),
      ),
      -0.25,
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
 * @param envState - Current physics state.
 */
function updateTelemetryPanelNodes(
  nodes: TelemetryPanelNodes,
  envState: EnvironmentState,
): void {
  nodes.posXValue.textContent = envState.carX.toFixed(1);
  nodes.posYValue.textContent = envState.carY.toFixed(1);
  nodes.headingValue.textContent = `${((envState.carHeading * 180) / Math.PI).toFixed(1)}°`;
  nodes.tickValue.textContent = String(envState.tick);
}
