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
import { selectMugshotDirection } from './host/hud-mugshot';
import { createInputRouter, type InputRouter } from './host/input';
import { createNeatensteinRendererBridge } from './host/renderer-bridge';
import type { NeatensteinRendererBridge } from './host/renderer-bridge';
import {
  computeDeltaMs,
  advanceSimTick,
  computeHiveDensityRatio,
  buildRenderState,
  deriveDeathFeedbackDirection,
  resolveWaveNumber,
} from './render-loop.utils';
import {
  resolveWorkerUrl,
  supportsWorkerOffscreenCanvas,
  drawCanvasStatus,
} from './bootstrap.utils';
import {
  resolveCanvasRenderDimensions,
  applyCanvasBackingStore,
  updateRendererSize,
} from './canvas-dimensions.utils';
import {
  REFERENCE_TIMESTEP_MS,
  DEFAULT_MAX_HEALTH,
  DEFAULT_MAX_AMMO,
  DELTA_MULTIPLIER,
  RENDER_TIER_WORKER,
  RENDER_TIER_CPU,
} from './constants';
import { DOM_EVENT_RESIZE } from './host/dom-events.constants';
import type { NeatensteinStart, NeatensteinStop } from './browser-entry.types';

/**
 * Exported shape expected by the host shell on `window`.
 *
 * @deprecated Import from `./browser-entry.types` instead. This re-export
 *   preserves the public API for existing consumers.
 */
export type { NeatensteinStart, NeatensteinStop } from './browser-entry.types';

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

  let currentRenderDimensions = resolveCanvasRenderDimensions(htmlCanvas);
  applyCanvasBackingStore(htmlCanvas, currentRenderDimensions);

  let bridge: NeatensteinRendererBridge | null = null;

  const useWorkerTier = supportsWorkerOffscreenCanvas();
  const tier = useWorkerTier ? RENDER_TIER_WORKER : RENDER_TIER_CPU;

  /** React to a change in the visible canvas CSS box via imported executor. */
  const handleResize = () => {
    currentRenderDimensions = updateRendererSize({
      canvas: htmlCanvas,
      useWorkerTier,
      bridge,
    });
  };

  let resizeObserver: ResizeObserver | null = null;
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(htmlCanvas);
  }

  window.addEventListener(DOM_EVENT_RESIZE, handleResize);

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
    workerUrl: resolveWorkerUrl(hostScript),
    tier,
    mapSeed: initialState.seed,
  });

  // Wire the status bar to render frames posted by the worker so the health
  // segments, ammo segments, and kill/death labels update on each frame.
  let lastFrameHealth = 0;
  let lastFrameMaxHealth = DEFAULT_MAX_HEALTH;
  let lastFrameAmmo = 0;
  let lastFrameMaxAmmo = DEFAULT_MAX_AMMO;
  let lastFrameKills = 0;
  let lastFrameDeaths = 0;
  let lastFrameGeneration = 0;
  let lastWaveNumber = 0;

  /** Health ratio for the mugshot overlay, defaulting to full-health teal. */
  let lastMugshotHealthRatio = 1.0;

  bridge.setFrameConsumer((frame) => {
    lastFrameHealth = frame.playerHealth ?? 0;
    lastFrameMaxHealth = frame.playerMaxHealth ?? DEFAULT_MAX_HEALTH;
    lastFrameAmmo = frame.playerAmmo ?? 0;
    lastFrameMaxAmmo = frame.playerMaxAmmo ?? DEFAULT_MAX_AMMO;
    lastFrameKills = frame.playerKills ?? 0;
    lastFrameDeaths = frame.playerDeaths ?? 0;
    lastFrameGeneration = frame.generation ?? 0;

    // Compute the mugshot health ratio from the raw frame fields, defaulting
    // to full-health (1.0) when either field is absent or maxHealth is zero.
    const rawHealth = frame.playerHealth;
    const rawMaxHealth = frame.playerMaxHealth;
    if (rawHealth != null && rawMaxHealth != null && rawMaxHealth > 0) {
      lastMugshotHealthRatio = Math.max(
        0,
        Math.min(1, rawHealth / rawMaxHealth),
      );
    } else {
      lastMugshotHealthRatio = 1.0;
    }

    statusBar.update({
      playerHealth: lastFrameHealth,
      playerMaxHealth: lastFrameMaxHealth,
      playerAmmo: lastFrameAmmo,
      playerMaxAmmo: lastFrameMaxAmmo,
      playerKills: lastFrameKills,
      playerDeaths: lastFrameDeaths,
      generation: lastFrameGeneration,
    });

    // Detect wave transitions and show the "Wave N" announcement overlay.
    // Wave 1 = spawnCount 0-8 (first 8 enemies), Wave 2 = spawnCount 9-16,
    // etc. Using (spawnCount - 1) ensures the wave number only advances when
    // the first enemy of the new wave actually spawns, not when the 8th
    // enemy of the current wave spawns.
    const spawnCount = frame.spawnCount ?? 0;
    const waveNumber = resolveWaveNumber(
      spawnCount,
      NEATENSTEIN_ENEMY_MAX_CONCURRENT,
    );
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
      generation: lastFrameGeneration,
    }),
    () => lastMugshotHealthRatio,
  );

  const stop = () => {
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    window.removeEventListener(DOM_EVENT_RESIZE, handleResize);
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
 * @param getMugshotHealthRatio - Returns the latest health ratio for the
 *   mugshot overlay, defaulting to 1.0 (full-health teal) when the frame
 *   fields are absent.
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
  getMugshotHealthRatio: () => number,
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

  const MAX_DELTA_MS = DELTA_MULTIPLIER * REFERENCE_TIMESTEP_MS; // 64 ms ≈ four reference frames

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
    // --- Compute delta-time ---
    const deltaMs = computeDeltaMs(lastTimestamp, timestamp, MAX_DELTA_MS);
    lastTimestamp = timestamp;

    // --- Advance simulation tick ---
    simTick = advanceSimTick(simTick, deltaMs, REFERENCE_TIMESTEP_MS);

    // --- Forward input ---
    const snapshot = inputRouter.getSnapshot();
    forwardWorkerInput(bridge.worker, snapshot);

    // --- Compute hive density ---
    const renderDimensions = getRenderDimensions();
    const hiveDensity = computeHiveDensityRatio(
      initialState.enemies.length,
      NEATENSTEIN_ENEMY_MAX_CONCURRENT,
    );

    // --- Build render state ---
    const renderState = buildRenderState({
      canvasWidth: renderDimensions.width,
      canvasHeight: renderDimensions.height,
      simTick,
      cameraX: initialState.player.position.x,
      cameraY: initialState.player.position.y,
      cameraYaw,
      mapSeed: initialState.seed,
      movement: snapshot.movement,
      deltaMs,
      hiveDensity,
      humanMode: getHumanMode(),
    });

    // --- Update status bar ---
    statusBar.update({
      hiveDensity,
      ...getLatestFrameState(),
    });

    // --- Update mugshot overlay ---
    statusBar.mugshot.update(
      selectMugshotDirection({ yawDelta: snapshot.look.yawDelta }),
      getMugshotHealthRatio(),
    );

    // --- Update death feedback ---
    const densityDelta = hiveDensity - prevHiveDensity;
    prevHiveDensity = hiveDensity;
    deathFeedback.update({
      direction: deriveDeathFeedbackDirection(densityDelta),
      aggressionDelta: densityDelta,
      movementDelta: 0,
      positioningDelta: 0,
    });

    // --- Post sim state to worker ---
    bridge.postSimState(renderState);

    // --- Accumulate camera yaw on host ---
    cameraYaw += snapshot.look.yawDelta;
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
