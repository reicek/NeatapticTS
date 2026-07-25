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
  NEATENSTEIN_FALLBACK_CANVAS_HEIGHT,
  NEATENSTEIN_FALLBACK_CANVAS_WIDTH,
  NEATENSTEIN_FALLBACK_STATUS_TEXT_RGB,
  NEATENSTEIN_WORKER_BUNDLE_FILENAME,
  NEATENSTEIN_GPU_COLUMN_COUNT,
  NEATENSTEIN_WORKER_COLUMN_COUNT,
} from './constants';
import { forwardWorkerInput } from './host/game/controls';
import { createGameState } from './host/game/state';
import type { GameState } from './host/game/types';
import { createInputRouter, type InputRouter } from './host/input';
import { createNeatensteinRendererBridge } from './host/renderer-bridge';
import type { NeatensteinRendererBridge } from './host/renderer-bridge';
import { NEATENSTEIN_BACKGROUND_RGB } from './renderer/framebuffer';

/**
 * Exported shape expected by the host shell on `window`.
 *
 * @param outputId - Host container element id (currently unused; reserved for future HUD).
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

const NEATENSTEIN_MAX_CANVAS_WIDTH = NEATENSTEIN_GPU_COLUMN_COUNT * 2;
const NEATENSTEIN_MAX_CANVAS_HEIGHT = NEATENSTEIN_WORKER_COLUMN_COUNT * 2;

/**
 * Scale a canvas backing-store size so it uses the largest render resolution
 * supported by the Neatenstein raycasting output bounds while preserving the
 * source aspect ratio.
 *
 * The maximum width is derived from the GPU raycasting column count, and the
 * maximum height is derived from the worker-tier column count. They are aliased
 * locally as canvas bounds because this file is concerned with the host canvas
 * backing-store size rather than renderer internals.
 *
 * The returned size will:
 *
 * - preserve the source canvas aspect ratio
 * - use as much of the maximum output bounds as possible
 * - never exceed `NEATENSTEIN_MAX_CANVAS_WIDTH`
 * - never exceed `NEATENSTEIN_MAX_CANVAS_HEIGHT`
 *
 * This function may upscale or downscale the source dimensions. That is
 * intentional: the CSS canvas size provides the aspect ratio, while the
 * raycasting bounds define the desired maximum render resolution.
 *
 * @param sourceWidth - Source canvas width in CSS pixels.
 * @param sourceHeight - Source canvas height in CSS pixels.
 * @returns The largest backing-store size that fits within the raycasting output bounds.
 */
function fitCanvasBackingStoreToMax(
  sourceWidth: number,
  sourceHeight: number,
): { width: number; height: number } {
  // Guard against invalid source sizes. The caller normally falls back before
  // this point, but this keeps the helper safe if reused elsewhere.
  if (sourceWidth <= 0 || sourceHeight <= 0) {
    return {
      width: NEATENSTEIN_MAX_CANVAS_WIDTH,
      height: NEATENSTEIN_MAX_CANVAS_HEIGHT,
    };
  }

  // Choose the limiting axis by taking the smaller scale factor.
  //
  // Do not include `1` here. A `1` cap would make this a downscale-only helper,
  // preventing smaller CSS canvases from using the full available raycasting
  // output resolution.
  const scale = Math.min(
    NEATENSTEIN_MAX_CANVAS_WIDTH / sourceWidth,
    NEATENSTEIN_MAX_CANVAS_HEIGHT / sourceHeight,
  );

  return {
    // Canvas backing-store dimensions must be integer device pixels.
    // Clamp to at least 1px so unusual aspect ratios never floor to zero.
    width: Math.max(1, Math.floor(sourceWidth * scale)),
    height: Math.max(1, Math.floor(sourceHeight * scale)),
  };
}

/**
 * Start the Neatenstein demo on the host page.
 *
 * Resolves the canvas, matches the canvas backing store to its CSS pixel size,
 * caps the backing store to the supported maximum output size while preserving
 * aspect ratio, spawns the display worker bridge, initializes a deterministic
 * game state, and starts the render loop that ships simulation snapshots to the
 * worker.
 *
 * Uses a CPU fallback when OffscreenCanvas is not available so the entry point
 * never crashes on load.
 *
 * @param _outputId - Host container element id, reserved for future HUD output.
 * @param canvasId - Visible canvas element id to bind the renderer to.
 * @returns A stop function that cancels rendering, detaches input, and destroys
 * the renderer bridge.
 */
function neatensteinStart(
  _outputId: string,
  canvasId: string,
): NeatensteinStop {
  const canvas = document.getElementById(canvasId);
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error(`Canvas element #${canvasId} not found`);
  }

  // Match the canvas backing store to its CSS pixel size for crisp rendering.
  // The backing store controls the actual render resolution, while CSS controls
  // the displayed layout size.
  //
  // When jsdom, tests, hidden containers, or a zero-layout viewport report zero
  // client dimensions, fall back to the computed CSS size so the renderer still
  // receives a usable drawing buffer.
  let backingWidth = canvas.clientWidth;
  let backingHeight = canvas.clientHeight;

  if (backingWidth === 0 || backingHeight === 0) {
    const style = getComputedStyle(canvas);

    // Prefer the authored CSS size when layout dimensions are unavailable.
    // If parsing fails, fall back to the existing Neatenstein defaults.
    backingWidth =
      Number.parseInt(style.width, 10) || NEATENSTEIN_FALLBACK_CANVAS_WIDTH;
    backingHeight =
      Number.parseInt(style.height, 10) || NEATENSTEIN_FALLBACK_CANVAS_HEIGHT;
  }

  // Resize the canvas backing store to the largest render size supported by the
  // raycasting output bounds while preserving the canvas's CSS aspect ratio.
  //
  // The CSS dimensions are used only to determine aspect ratio. The backing
  // store may be upscaled or downscaled so the worker receives the maximum
  // usable render resolution without stretching the scene.
  const fittedBackingStore = fitCanvasBackingStoreToMax(
    backingWidth,
    backingHeight,
  );

  canvas.width = fittedBackingStore.width;
  canvas.height = fittedBackingStore.height;

  const useWorkerTier = supportsWorkerOffscreenCanvas();
  const tier = useWorkerTier ? 'worker' : 'cpu';

  if (!useWorkerTier) {
    drawCanvasStatus(
      canvas,
      'OffscreenCanvas not available; using CPU fallback renderer.',
    );
  }

  // Seed a deterministic game state so the render loop has a simulation clock.
  const initialState = createGameState({ seed: 1 });

  const bridge = createNeatensteinRendererBridge({
    canvas,
    workerUrl: resolveWorkerUrl(),
    tier,
    mapSeed: initialState.seed,
  });

  const inputRouter = createInputRouter();
  inputRouter.attach(canvas);

  const cancelRenderLoop = startRenderLoop(
    canvas,
    bridge,
    initialState,
    inputRouter,
  );

  const stop = () => {
    cancelRenderLoop();
    inputRouter.detach();
    bridge.destroy();
  };

  window.neatensteinStop = stop;

  return stop;
}

/**
 * Run the host render loop for a worker-bound canvas.
 *
 * Keeps a local simulation tick and forwards a render state snapshot to the
 * worker every animation frame.
 *
 * @param canvas - The visible canvas bound to the worker renderer.
 * @param bridge - Host/worker bridge that forwards simulation snapshots.
 * @param initialState - Deterministic game state providing camera and seed.
 * @returns A function that cancels the queued animation frame.
 */
function startRenderLoop(
  canvas: HTMLCanvasElement,
  bridge: NeatensteinRendererBridge,
  initialState: GameState,
  inputRouter: InputRouter,
): () => void {
  let simTick = initialState.seed;
  let cameraYaw = initialState.player.angleRad;
  let animationFrameId: number | null = null;

  /**
   * Render loop: ship an updated render state snapshot to the worker every
   * animation frame.
   */
  function tick(): void {
    simTick += 1;
    const snapshot = inputRouter.getSnapshot();

    // Forward look deltas first so the worker can apply them to the incoming
    // simulation state for this frame.
    forwardWorkerInput(bridge.worker, snapshot);

    bridge.postSimState({
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
      simTick,
      cameraX: initialState.player.position.x,
      cameraY: initialState.player.position.y,
      cameraYaw,
      mapSeed: initialState.seed,
      movement: snapshot.movement,
    });

    // The authoritative camera yaw is accumulated on the host so it persists
    // across frames while the worker uses the per-frame delta for responsive
    // visual feedback.
    cameraYaw += snapshot.look.yawDelta;

    animationFrameId = requestAnimationFrame(tick);
  }

  animationFrameId = requestAnimationFrame(tick);

  return () => {
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
