/// <reference lib="dom" />

/**
 * Browser entrypoint for the Neatenstein neon raycasting demo.
 *
 * This small facade is loaded by the host HTML shell as an IIFE bundle. It
 * exposes `window.neatensteinStart` which wires the visible canvas to the
 * display worker via {@link ./host/renderer-bridge} and begins feeding render
 * state snapshots. Game state is initialized through
 * {@link ./host/game/state} so the renderer has a deterministic simulation
 * clock to display.
 *
 * @module
 */

import {
  NEATENSTEIN_FALLBACK_CANVAS_WIDTH,
  NEATENSTEIN_FALLBACK_STATUS_TEXT_RGB,
  NEATENSTEIN_WORKER_BUNDLE_FILENAME,
} from './constants';
import { forwardWorkerInput } from './host/game/controls';
import { NEATENSTEIN_ENEMY_MAX_CONCURRENT } from './host/game/constants';
import { createGameState } from './host/game/state';
import type { GameState } from './host/game/types';
import {
  createNeonStatusBar,
  type NeonStatusBarHud,
  createHumanModeSelector,
  createDeathFeedbackIndicator,
  type DeathFeedbackIndicator,
  createWaveAnnouncement,
} from './host/hud';
import { createInputRouter, type InputRouter } from './host/input';
import { createNeatensteinRendererBridge } from './host/renderer-bridge';
import type { NeatensteinRendererBridge } from './host/renderer-bridge';
import { NEATENSTEIN_BACKGROUND_RGB } from './renderer/framebuffer';

/**
 * Exported shape expected by the host shell on `window`.
 *
 * @param outputId - Host container element id that owns the HIVE DENSITY HUD
 *   overlay.
 * @param canvasId - Visible canvas element id to bind the renderer to.
 * @returns A teardown function that cancels the render loop, detaches input, and
 *   terminates the worker.
 */
export type NeatensteinStart = (
  outputId: string,
  canvasId: string,
) => NeatensteinStop;

/** Teardown function returned by {@link NeatensteinStart}. */
export type NeatensteinStop = () => void;

/**
 * Host script that loaded this IIFE bundle.
 *
 * Captured at bundle evaluation time so the worker URL can be resolved relative
 * to the host script's location, not the page's base URI. This keeps the demo
 * working whether it is served from `docs/examples/neatenstein/` or
 * `examples/neatenstein/`.
 */
const hostScript = document.currentScript;

/**
 * Resolve the published worker URL relative to the host bundle.
 *
 * The worker asset lives next to the host bundle under `docs/assets/`. Resolving
 * against `document.currentScript.src` works correctly under any server root.
 */
function resolveWorkerUrl(): string {
  if (!hostScript || !(hostScript instanceof HTMLScriptElement)) {
    throw new Error(
      'Neatenstein bundle must be loaded through a <script> tag so the worker URL can be resolved relative to the host bundle.',
    );
  }
  return new URL(NEATENSTEIN_WORKER_BUNDLE_FILENAME, hostScript.src).href;
}

/**
 * Runtime capability check for the OffscreenCanvas worker transfer path.
 *
 * Returns `true` only when both the visible canvas can produce an
 * {@link OffscreenCanvas} and the worker-side type is present.
 */
function supportsWorkerOffscreenCanvas(): boolean {
  return (
    typeof HTMLCanvasElement !== 'undefined' &&
    typeof OffscreenCanvas !== 'undefined' &&
    typeof HTMLCanvasElement.prototype.transferControlToOffscreen === 'function'
  );
}

/**
 * Format an RGB triple as a CSS `rgb(...)` string.
 *
 * Small local helper so the fallback status path can share the same canonical
 * background color as the worker/CPU/GPU renderers.
 */
function formatRgb(color: { r: number; g: number; b: number }): string {
  return `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;
}

/**
 * Draw a centered status message on the visible canvas.
 *
 * Used by the CPU fallback path when OffscreenCanvas is unavailable so the
 * page shows an explanation instead of a blank screen.
 */
function drawCanvasStatus(canvas: HTMLCanvasElement, message: string): void {
  const context = canvas.getContext('2d');
  if (!context) {
    return;
  }

  context.fillStyle = formatRgb(NEATENSTEIN_BACKGROUND_RGB);
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = formatRgb(NEATENSTEIN_FALLBACK_STATUS_TEXT_RGB);
  context.font = '0.875rem system-ui, Segoe UI, Arial, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(message, canvas.width / 2, canvas.height / 2);
}

/**
 * Start the Neatenstein demo on the host page.
 *
 * Resolves the canvas, sets the canvas backing store to the CSS-derived render
 * size (480px height with width proportional to the canvas's CSS box aspect
 * ratio), spawns the display worker bridge, initializes a deterministic game
 * state, and starts the render loop that ships simulation snapshots to the
 * worker.
 *
 * Uses a CPU fallback when OffscreenCanvas is not available so the entry point
 * never crashes on load.
 *
 * @param outputId - Host container element id that owns the HIVE DENSITY HUD
 *   overlay.
 * @param canvasId - Visible canvas element id to bind the renderer to.
 * @returns A stop function that cancels rendering, detaches input, and destroys
 * the renderer bridge.
 */
function neatensteinStart(outputId: string, canvasId: string): NeatensteinStop {
  const canvas = document.getElementById(canvasId);
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error(`Canvas element #${canvasId} not found`);
  }

  // Keep the backing store at a fixed 480px height with a width proportional
  // to the canvas's CSS display aspect ratio. The CSS shell stretches the
  // canvas to fill the available container space, and the browser upscales
  // this fixed-height backing store to the stretched display size. Reading
  // client dimensions gives the actual rendered box, which differs from the
  // viewport when the page includes status bars, gaps, or flex layout.
  const htmlCanvas = canvas;

  /**
   * Fixed render height in pixels used for the canvas backing store and for
   * the worker-tier render dimensions forwarded in simState.
   */
  const NEATENSTEIN_FIXED_RENDER_HEIGHT = 480;

  /**
   * Compute the CSS-derived render dimensions for the visible canvas.
   *
   * The width is proportional to the canvas CSS box aspect ratio so the
   * rendered scene is never stretched, while the height stays fixed at 480px
   * to keep the raycaster projection math stable.
   *
   * @param element - Visible host canvas.
   * @returns Render width and height in pixels.
   */
  function resolveCanvasRenderDimensions(element: HTMLCanvasElement): {
    width: number;
    height: number;
  } {
    const clientWidth = element.clientWidth;
    const clientHeight = element.clientHeight;

    if (clientWidth && clientHeight) {
      return {
        width: Math.round(
          NEATENSTEIN_FIXED_RENDER_HEIGHT * (clientWidth / clientHeight),
        ),
        height: NEATENSTEIN_FIXED_RENDER_HEIGHT,
      };
    }

    // Fall back to viewport dimensions when the canvas has not been laid out yet.
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const fallbackWidth =
      viewportWidth && viewportHeight
        ? Math.round(
            NEATENSTEIN_FIXED_RENDER_HEIGHT * (viewportWidth / viewportHeight),
          )
        : NEATENSTEIN_FALLBACK_CANVAS_WIDTH;

    return {
      width: fallbackWidth,
      height: NEATENSTEIN_FIXED_RENDER_HEIGHT,
    };
  }

  /**
   * Apply computed render dimensions to the visible canvas backing store.
   *
   * Safe to call only when the canvas is still owned by the host (i.e., the
   * CPU fallback tier). The worker tier transfers the canvas to the worker,
   * after which direct width/height assignment throws.
   *
   * @param element - Visible host canvas.
   * @param dimensions - Render width and height in pixels.
   */
  function applyCanvasBackingStore(
    element: HTMLCanvasElement,
    dimensions: { width: number; height: number },
  ): void {
    element.width = dimensions.width;
    element.height = dimensions.height;
  }

  let currentRenderDimensions = resolveCanvasRenderDimensions(htmlCanvas);
  applyCanvasBackingStore(htmlCanvas, currentRenderDimensions);

  let bridge: NeatensteinRendererBridge | null = null;

  const useWorkerTier = supportsWorkerOffscreenCanvas();
  const tier = useWorkerTier ? 'worker' : 'cpu';

  /**
   * React to a change in the visible canvas CSS box.
   *
   * For the worker tier, the host canvas is transferred to the worker, so the
   * host cannot mutate its backing store. Instead the new dimensions are routed
   * to the bridge, which posts them to the worker, and the render loop uses
   * the CSS-derived dimensions directly. For the CPU fallback tier, the host
   * still owns the canvas and updates the backing store as before.
   */
  function updateRendererSize(): void {
    currentRenderDimensions = resolveCanvasRenderDimensions(htmlCanvas);

    if (useWorkerTier && bridge !== null) {
      bridge.resize(
        currentRenderDimensions.width,
        currentRenderDimensions.height,
      );
      return;
    }

    applyCanvasBackingStore(htmlCanvas, currentRenderDimensions);
  }

  let resizeObserver: ResizeObserver | null = null;
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(updateRendererSize);
    resizeObserver.observe(htmlCanvas);
  }

  const handleResize = () => {
    updateRendererSize();
  };
  window.addEventListener('resize', handleResize);

  if (!useWorkerTier) {
    drawCanvasStatus(
      canvas,
      'OffscreenCanvas not available; using CPU fallback renderer.',
    );
  }

  // Seed a deterministic game state so the render loop has a simulation clock.
  const initialState = createGameState({ seed: 1 });

  // Wire the neon status bar HUD overlay into the reserved host container.
  const statusBar = createNeonStatusBar(outputId);

  // Wire the death feedback indicator into the same host container. The
  // indicator displays the adaptation direction (stronger/weaker/shifted)
  // derived from arms-race generation results so the player gets a visual
  // read on co-evolution pressure.
  const deathFeedback = createDeathFeedbackIndicator(outputId);

  // Wire the human-mode selector into the same host container. The selector
  // exposes a mode flag ('auto' or 'human') that callers can pass into the
  // arms-race configuration when invoking runArmsRaceGeneration.
  const humanModeSelector = createHumanModeSelector(outputId);

  // Wire the wave announcement overlay — shows "Wave N" centered on screen
  // when a new wave of enemies spawns, using the same neon typography and
  // cyan glow as the flappy_bird generation display.
  const waveAnnouncement = createWaveAnnouncement(outputId);

  bridge = createNeatensteinRendererBridge({
    canvas,
    workerUrl: resolveWorkerUrl(),
    tier,
    mapSeed: initialState.seed,
  });

  // Wire the status bar to render frames posted by the worker so the health
  // segments, ammo segments, and kill/death labels update on each frame.
  let lastFrameHealth = 0;
  let lastFrameMaxHealth = 100;
  let lastFrameAmmo = 0;
  let lastFrameMaxAmmo = 50;
  let lastFrameKills = 0;
  let lastFrameDeaths = 0;
  let lastWaveNumber = 0;

  bridge.setFrameConsumer((frame) => {
    lastFrameHealth = frame.playerHealth ?? 0;
    lastFrameMaxHealth = frame.playerMaxHealth ?? 100;
    lastFrameAmmo = frame.playerAmmo ?? 0;
    lastFrameMaxAmmo = frame.playerMaxAmmo ?? 50;
    lastFrameKills = frame.playerKills ?? 0;
    lastFrameDeaths = frame.playerDeaths ?? 0;
    statusBar.update({
      playerHealth: lastFrameHealth,
      playerMaxHealth: lastFrameMaxHealth,
      playerAmmo: lastFrameAmmo,
      playerMaxAmmo: lastFrameMaxAmmo,
      playerKills: lastFrameKills,
      playerDeaths: lastFrameDeaths,
    });

    // Detect wave transitions and show the "Wave N" announcement overlay.
    // Wave 1 = spawnCount 0-8 (first 8 enemies), Wave 2 = spawnCount 9-16,
    // etc. Using (spawnCount - 1) ensures the wave number only advances when
    // the first enemy of the new wave actually spawns, not when the 8th
    // enemy of the current wave spawns.
    const spawnCount = frame.spawnCount ?? 0;
    const waveNumber =
      Math.floor(
        Math.max(0, spawnCount - 1) / NEATENSTEIN_ENEMY_MAX_CONCURRENT,
      ) + 1;
    if (waveNumber > lastWaveNumber) {
      lastWaveNumber = waveNumber;
      waveAnnouncement.show(waveNumber);
    }
  });

  const inputRouter = createInputRouter();
  inputRouter.attach(canvas);

  const cancelRenderLoop = startRenderLoop(
    canvas,
    bridge,
    initialState,
    inputRouter,
    () => currentRenderDimensions,
    statusBar,
    deathFeedback,
    () => humanModeSelector.mode,
    () => ({
      playerHealth: lastFrameHealth,
      playerMaxHealth: lastFrameMaxHealth,
      playerAmmo: lastFrameAmmo,
      playerMaxAmmo: lastFrameMaxAmmo,
      playerKills: lastFrameKills,
      playerDeaths: lastFrameDeaths,
    }),
  );

  const stop = () => {
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    window.removeEventListener('resize', handleResize);
    cancelRenderLoop();
    inputRouter.detach();
    bridge.destroy();
  };

  window.neatensteinStop = stop;

  return stop;
}

/**
 * Start the worker-paced host render loop.
 *
 * The loop posts a {@link NeatensteinRenderState} snapshot to the display
 * worker on **every** animation frame. Instead of scheduling `requestAnimationFrame`
 * continuously at display refresh rate, the next rAF is triggered by the
 * bridge's `onFrameReady` callback, which fires when the worker finishes
 * rendering a frame and has no pending state. This makes the loop purely
 * worker-paced, eliminating wasted rAF ticks when the worker renders slower
 * than the display refresh rate.
 *
 * The bridge applies worker-busy backpressure so only one snapshot is in
 * flight at a time, naturally throttling to the worker's actual render
 * capacity. Each snapshot carries a `deltaMs` field derived from consecutive
 * `requestAnimationFrame` timestamps so the worker can drive simulation
 * stepping with FPS-scaled timing instead of a fixed timestep.
 *
 * Input forwarding remains unthrottled so mouse/keyboard/touch look stays
 * responsive.
 *
 * @param canvas - The visible canvas bound to the worker renderer.
 * @param bridge - Host/worker bridge that forwards simulation snapshots.
 * @param initialState - Deterministic game state providing camera and seed.
 * @param inputRouter - Host input router used to build movement snapshots.
 * @param getRenderDimensions - Returns the current CSS-derived render
 *   dimensions. Mutable because resize updates it.
 * @param statusBar - Neon status bar HUD overlay updated on each animation
 *   frame with hive density and the latest frame vitals.
 * @param deathFeedback - Death feedback indicator updated on each animation
 *   frame with the adaptation signal derived from hive-density changes.
 * @param getHumanMode - Returns the current human-mode selector value ('auto'
 *   or 'human') so it can be forwarded in the render state to downstream
 *   consumers including the arms-race configuration.
 * @param getLatestFrameState - Returns the latest player vitals (health, ammo,
 *   kills, deaths) received from the worker frame so the status bar can merge
 *   them with the per-frame hive density.
 * @returns A function that cancels the queued animation frame.
 */
function startRenderLoop(
  canvas: HTMLCanvasElement,
  bridge: NeatensteinRendererBridge,
  initialState: GameState,
  inputRouter: InputRouter,
  getRenderDimensions: () => { width: number; height: number },
  statusBar: NeonStatusBarHud,
  deathFeedback: DeathFeedbackIndicator,
  getHumanMode: () => 'auto' | 'human',
  getLatestFrameState: () => {
    playerHealth: number;
    playerMaxHealth: number;
    playerAmmo: number;
    playerMaxAmmo: number;
    playerKills: number;
    playerDeaths: number;
  },
): () => void {
  let simTick = initialState.seed;
  let cameraYaw = initialState.player.angleRad;
  let animationFrameId: number | null = null;
  let lastTimestamp: number | null = null;

  /**
   * Previous-frame hive density used to derive the death feedback adaptation
   * signal. The delta between consecutive frames drives the indicator's
   * direction label (stronger/weaker/shifted).
   */
  let prevHiveDensity = 0;

  /**
   * Reference timestep for FPS-scaled simulation stepping.
   *
   * The host no longer uses a fixed timestep constant — delta-time drives the
   * clock. This local value is only used to scale `simTick` increments so a
   * 60 Hz display yields ~1 tick/frame, 30 Hz yields ~2, etc.
   */
  const REFERENCE_TIMESTEP_MS = 16;

  /**
   * Upper bound for the rAF delta-time in milliseconds.
   *
   * When a tab is suspended/resumed or the main thread stalls for a few
   * hundred milliseconds, the raw delta can be several seconds long. Clamping
   * it prevents the next physics tick from tunnelling through walls (point-
   * sample collision can step past thin wall segments when the single-step
   * displacement exceeds the grid cell size).
   */
  const MAX_DELTA_MS = 4 * REFERENCE_TIMESTEP_MS; // 64 ms ≈ four reference frames

  /**
   * Render loop: ship an updated render state snapshot to the worker on each
   * animation frame with a delta-time field derived from consecutive rAF
   * timestamps.
   *
   * The next rAF is NOT scheduled at the end of tick — it is triggered by the
   * `onFrameReady` callback registered on the bridge, which fires when the
   * worker finishes rendering and is idle. This makes the loop purely
   * worker-paced.
   *
   * Input forwarding stays unthrottled so mouse/keyboard/touch look remains
   * responsive.
   *
   * @param timestamp - High-resolution animation-frame timestamp in ms.
   */
  function tick(timestamp: number): void {
    // Compute delta-time from consecutive rAF timestamps. On the first frame
    // there is no previous timestamp, so deltaMs is 0. Clamp the delta to
    // MAX_DELTA_MS so a suspended tab or long frame cannot produce a single
    // physics step large enough to tunnel through walls.
    const deltaMs = Math.min(
      lastTimestamp === null ? 0 : timestamp - lastTimestamp,
      MAX_DELTA_MS,
    );
    lastTimestamp = timestamp;

    // FPS-scaled simulation stepping: increment simTick proportionally to the
    // frame delta so simulation progress is consistent across varying refresh
    // rates.
    simTick += Math.max(1, Math.round(deltaMs / REFERENCE_TIMESTEP_MS));

    const snapshot = inputRouter.getSnapshot();

    // Forward look deltas first so the worker can apply them to the incoming
    // simulation state for this frame.
    forwardWorkerInput(bridge.worker, snapshot);

    const renderDimensions = getRenderDimensions();

    // HIVE DENSITY: enemy population density relative to the concurrency cap,
    // clamped to [0, 1]. The host render state carries the initial enemy count;
    // the worker maintains the live population. Forwarding this field keeps the
    // HUD overlay synchronized with the render state snapshot posted to the
    // worker on each animation frame.
    const hiveDensity = Math.min(
      1,
      Math.max(
        0,
        initialState.enemies.length / NEATENSTEIN_ENEMY_MAX_CONCURRENT,
      ),
    );

    // Post simState on every animation frame. The bridge applies worker-busy
    // backpressure so only one snapshot is in flight at a time; additional
    // frames are deferred until the worker acknowledges. The deltaMs field
    // lets the worker use FPS-scaled timing for its simulation stepping.
    const renderState = {
      canvasWidth: renderDimensions.width,
      canvasHeight: renderDimensions.height,
      simTick,
      cameraX: initialState.player.position.x,
      cameraY: initialState.player.position.y,
      cameraYaw,
      mapSeed: initialState.seed,
      movement: snapshot.movement,
      enemies: [],
      deltaMs,
      hiveDensity,
      humanMode: getHumanMode(),
    };

    statusBar.update({
      hiveDensity,
      ...getLatestFrameState(),
    });

    // DEATH FEEDBACK: derive a simple adaptation signal from the hive-density
    // delta between consecutive frames. The signal mirrors the
    // AdaptationSignal shape produced by computeAdaptationSignal in the
    // harness death-feedback module so the HUD indicator stays consistent with
    // the arms-race result wiring.
    const densityDelta = hiveDensity - prevHiveDensity;
    prevHiveDensity = hiveDensity;
    const direction =
      densityDelta > 0.01
        ? 'stronger'
        : densityDelta < -0.01
          ? 'weaker'
          : 'shifted';
    deathFeedback.update({
      direction,
      aggressionDelta: densityDelta,
      movementDelta: 0,
      positioningDelta: 0,
    });

    bridge.postSimState(renderState);

    // The authoritative camera yaw is accumulated on the host so it persists
    // across frames while the worker uses the per-frame delta for responsive
    // visual feedback.
    cameraYaw += snapshot.look.yawDelta;

    // The next rAF is scheduled by the onFrameReady callback (registered
    // below) when the worker finishes rendering and is idle. This makes the
    // loop purely worker-paced instead of running continuously at display
    // refresh rate, eliminating wasted rAF ticks when the worker renders
    // slower than the display.
  }

  // Register the worker-paced rAF trigger: when the worker finishes a frame
  // and has no pending state, schedule the next rAF. The flushed-state path
  // does NOT call onFrameReady — the flushed state's own ack will trigger it.
  bridge.setOnFrameReady(() => {
    animationFrameId = requestAnimationFrame(tick);
  });

  animationFrameId = requestAnimationFrame(tick);

  return () => {
    bridge.setOnFrameReady(null);
    // Defensive guard for teardown being called before the first animation
    // frame is scheduled. This branch is not reachable through the public
    // start flow, so it is excluded from branch coverage.
    /* istanbul ignore next */
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
  };
}

declare global {
  interface Window {
    /** Neatenstein demo start function installed by the host bundle. */
    neatensteinStart?: NeatensteinStart;
    /** Neatenstein demo teardown function installed by the host bundle. */
    neatensteinStop?: NeatensteinStop;
  }
}

window.neatensteinStart = neatensteinStart;
