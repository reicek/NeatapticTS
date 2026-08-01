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

  bridge = createNeatensteinRendererBridge({
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
    () => currentRenderDimensions,
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
 * Run the host render loop for a worker-bound canvas.
 *
 * Keeps a local simulation tick and forwards a render state snapshot to the
 * worker every animation frame.
 *
 * @param canvas - The visible canvas bound to the worker renderer.
 * @param bridge - Host/worker bridge that forwards simulation snapshots.
 * @param initialState - Deterministic game state providing camera and seed.
 * @param inputRouter - Host input router used to build movement snapshots.
 * @param getRenderDimensions - Returns the current CSS-derived render
 *   dimensions. Mutable because resize updates it.
 * @returns A function that cancels the queued animation frame.
 */
function startRenderLoop(
  canvas: HTMLCanvasElement,
  bridge: NeatensteinRendererBridge,
  initialState: GameState,
  inputRouter: InputRouter,
  getRenderDimensions: () => { width: number; height: number },
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

    const renderDimensions = getRenderDimensions();

    bridge.postSimState({
      canvasWidth: renderDimensions.width,
      canvasHeight: renderDimensions.height,
      simTick,
      cameraX: initialState.player.position.x,
      cameraY: initialState.player.position.y,
      cameraYaw,
      mapSeed: initialState.seed,
      movement: snapshot.movement,
      enemies: [],
    });

    // The authoritative camera yaw is accumulated on the host so it persists
    // across frames while the worker uses the per-frame delta for responsive
    // visual feedback.
    cameraYaw += snapshot.look.yawDelta;

    animationFrameId = requestAnimationFrame(tick);
  }

  animationFrameId = requestAnimationFrame(tick);

  return () => {
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
