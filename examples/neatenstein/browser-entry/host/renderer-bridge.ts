/**
 * Host-side renderer bridge for the Neatenstein neon raycasting demo.
 *
 * This module owns the boundary between the main browser thread and the
 * display worker. It is responsible for:
 *
 * - spawning the module worker,
 * - constraining the render size before messages cross the worker boundary,
 * - transferring an {@link OffscreenCanvas} on the `worker` tier,
 * - forwarding simulation state to the worker,
 * - forwarding input snapshots to the worker,
 * - exposing the latest rendered frame request id for CPU/GPU tiers,
 * - and terminating the worker on teardown.
 *
 * The bridge intentionally remains thin, but it enforces the canvas/backing
 * store size invariant because this is the last host-side boundary before the
 * worker owns direct rendering.
 *
 * @module
 */

import {
  NEATENSTEIN_GPU_COLUMN_COUNT,
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
  NEATENSTEIN_WORKER_BUNDLE_FILENAME,
  NEATENSTEIN_WORKER_COLUMN_COUNT,
  type NeatensteinTier,
} from '../constants';
import type { NeatensteinRenderState } from '../renderer/frame';
import type { InputSnapshot } from './input';

/**
 * Default worker bundle URL used when the caller does not supply one.
 */
const DEFAULT_WORKER_URL = `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`;

/**
 * Maximum bridge-managed canvas width.
 *
 * The source constant is named for the GPU raycasting column count, but at the
 * bridge boundary it represents the maximum horizontal backing-store size that
 * should be sent to the display worker.
 */
const NEATENSTEIN_BRIDGE_MAX_CANVAS_WIDTH = NEATENSTEIN_GPU_COLUMN_COUNT * 2;

/**
 * Maximum bridge-managed canvas height.
 *
 * The source constant is named for the worker column count in renderer terms,
 * but at the bridge boundary it represents the maximum vertical backing-store
 * size that should be sent to the display worker.
 */
const NEATENSTEIN_BRIDGE_MAX_CANVAS_HEIGHT =
  NEATENSTEIN_WORKER_COLUMN_COUNT * 2;

/**
 * Configuration needed to create a renderer bridge.
 */
export interface NeatensteinRendererBridgeOptions {
  /** The visible canvas element on the host page. */
  canvas: HTMLCanvasElement;
  /** Absolute or relative URL to the ESM worker bundle. */
  workerUrl?: string;
  /** Selected renderer tier. */
  tier: NeatensteinTier;
  /** Deterministic seed used to build the wall grid on the worker. */
  mapSeed: number;
}

/**
 * Public surface of the host renderer bridge.
 */
export interface NeatensteinRendererBridge {
  /** The spawned display worker. */
  worker: Worker;
  /** Latest frame request id received from the worker. */
  requestId: number;
  /** Forward a simulation state snapshot to the worker. */
  postSimState(state: NeatensteinRenderState): void;
  /** Forward an input snapshot to the worker so it can advance the sim tick. */
  forwardWorkerInput(snapshot: InputSnapshot): void;
  /** Terminate the worker and release the bridge. */
  destroy(): void;
}

/**
 * Integer render size after applying Neatenstein output constraints.
 */
interface ConstrainedBridgeRenderSize {
  /** Constrained backing-store width in pixels. */
  width: number;
  /** Constrained backing-store height in pixels. */
  height: number;
}

/**
 * Return whether a value is usable as a positive render dimension.
 *
 * @param value - Candidate dimension.
 * @returns Whether the value is finite and positive.
 */
function isPositiveFiniteDimension(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/**
 * Resolve a constrained canvas backing-store size.
 *
 * The source dimensions provide the aspect ratio. The returned dimensions use
 * as much of the Neatenstein maximum render bounds as possible while preserving
 * that aspect ratio.
 *
 * This function may upscale or downscale relative to the source size. That is
 * intentional: the source size defines shape, while the Neatenstein bounds
 * define maximum render resolution.
 *
 * @param sourceWidth - Source width in pixels.
 * @param sourceHeight - Source height in pixels.
 * @returns Constrained integer size, or `null` when the source is invalid.
 */
function resolveConstrainedBridgeRenderSize(
  sourceWidth: number,
  sourceHeight: number,
): ConstrainedBridgeRenderSize | null {
  if (
    !isPositiveFiniteDimension(sourceWidth) ||
    !isPositiveFiniteDimension(sourceHeight)
  ) {
    return null;
  }

  // Choose the limiting scale so neither axis exceeds the max bounds.
  // Do not cap this at 1: smaller canvases should still be able to use the
  // maximum backing-store resolution while preserving their aspect ratio.
  const scale = Math.min(
    NEATENSTEIN_BRIDGE_MAX_CANVAS_WIDTH / sourceWidth,
    NEATENSTEIN_BRIDGE_MAX_CANVAS_HEIGHT / sourceHeight,
  );

  return {
    width: Math.max(1, Math.floor(sourceWidth * scale)),
    height: Math.max(1, Math.floor(sourceHeight * scale)),
  };
}

/**
 * Resolve the best initial backing-store size for a visible canvas.
 *
 * The bridge prefers the canvas backing-store dimensions because earlier host
 * setup may already have constrained them. If they are unavailable, it falls
 * back to layout dimensions.
 *
 * @param canvas - Visible host canvas.
 * @returns Constrained render size, or `null` if no usable dimensions exist.
 */
function resolveInitialCanvasRenderSize(
  canvas: HTMLCanvasElement,
): ConstrainedBridgeRenderSize | null {
  const backingWidth = canvas.width;
  const backingHeight = canvas.height;

  if (
    isPositiveFiniteDimension(backingWidth) &&
    isPositiveFiniteDimension(backingHeight)
  ) {
    return resolveConstrainedBridgeRenderSize(backingWidth, backingHeight);
  }

  return resolveConstrainedBridgeRenderSize(
    canvas.clientWidth,
    canvas.clientHeight,
  );
}

/**
 * Apply a constrained backing-store size to the visible canvas.
 *
 * This must happen before `transferControlToOffscreen()` for the worker tier so
 * the transferred canvas starts with the correct dimensions.
 *
 * @param canvas - Visible host canvas.
 * @param size - Constrained backing-store size.
 */
function applyCanvasBackingStoreSize(
  canvas: HTMLCanvasElement,
  size: ConstrainedBridgeRenderSize,
): void {
  if (canvas.width !== size.width) {
    canvas.width = size.width;
  }

  if (canvas.height !== size.height) {
    canvas.height = size.height;
  }
}

/**
 * Return a simulation state with constrained render dimensions.
 *
 * This makes the bridge a hard boundary: even if an upstream caller sends
 * unconstrained dimensions, the worker receives the maximum allowed size for
 * the same aspect ratio.
 *
 * @param state - Raw simulation/render state from the host loop.
 * @returns State with constrained canvas dimensions.
 */
function constrainRenderState(
  state: NeatensteinRenderState,
): NeatensteinRenderState {
  const constrainedSize = resolveConstrainedBridgeRenderSize(
    state.canvasWidth,
    state.canvasHeight,
  );

  if (constrainedSize === null) {
    return state;
  }

  return {
    ...state,
    canvasWidth: constrainedSize.width,
    canvasHeight: constrainedSize.height,
  };
}

/**
 * Return whether a value looks like a worker protocol message object.
 *
 * @param value - Raw message payload.
 * @returns Whether the value can be inspected as a message record.
 */
function isMessageRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object';
}

/**
 * Create a host-side bridge to the Neatenstein display worker.
 *
 * For the `worker` tier, the canvas backing store is constrained first and then
 * transferred to the worker through
 * {@link HTMLCanvasElement.transferControlToOffscreen}. For `cpu` and `gpu`
 * tiers, the canvas stays on the host and the worker posts packed frames back.
 *
 * The bridge queues the latest simulation state and latest input snapshot until
 * the worker acknowledges initialization. This avoids losing early host-loop
 * messages during worker startup.
 *
 * @param options - Canvas, worker URL, and tier selection.
 * @returns A bridge object with lifecycle and messaging methods.
 *
 * @example
 * ```ts
 * const bridge = createNeatensteinRendererBridge({
 *   canvas: document.getElementById('game') as HTMLCanvasElement,
 *   workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
 *   tier: 'worker',
 *   mapSeed: 1,
 * });
 *
 * bridge.postSimState({
 *   canvasWidth: 640,
 *   canvasHeight: 480,
 *   simTick: 1,
 *   cameraX: 12.5,
 *   cameraY: 12.5,
 *   cameraYaw: 0,
 *   mapSeed: 1,
 * });
 * ```
 */
export function createNeatensteinRendererBridge(
  options: NeatensteinRendererBridgeOptions,
): NeatensteinRendererBridge {
  const { canvas, workerUrl = DEFAULT_WORKER_URL, tier, mapSeed } = options;

  const initialRenderSize = resolveInitialCanvasRenderSize(canvas);
  if (initialRenderSize !== null) {
    // Enforce the render-size invariant before creating/transferring the worker
    // canvas. This is especially important for OffscreenCanvas ownership.
    applyCanvasBackingStoreSize(canvas, initialRenderSize);
  }

  const worker = new Worker(workerUrl, { type: 'module' });

  let destroyed = false;
  let initialized = false;
  let latestRequestId = 0;
  let pendingState: NeatensteinRenderState | null = null;
  let pendingInput: InputSnapshot | null = null;

  let offscreen: OffscreenCanvas | undefined;
  const transferList: Transferable[] = [];

  if (tier === 'worker') {
    if (typeof canvas.transferControlToOffscreen !== 'function') {
      worker.terminate();

      throw new Error(
        'Neatenstein worker tier requires HTMLCanvasElement.transferControlToOffscreen().',
      );
    }

    // Transfer after the backing store has been constrained.
    offscreen = canvas.transferControlToOffscreen();
    transferList.push(offscreen);
  }

  /**
   * Send a simulation state immediately if the bridge is alive.
   *
   * @param state - Constrained state to send.
   */
  function postSimStateNow(state: NeatensteinRenderState): void {
    if (destroyed) {
      return;
    }

    worker.postMessage({ type: 'simState', state });
  }

  /**
   * Send an input snapshot immediately if the bridge is alive.
   *
   * @param snapshot - Input snapshot to send.
   */
  function forwardWorkerInputNow(snapshot: InputSnapshot): void {
    if (destroyed) {
      return;
    }

    worker.postMessage({
      type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
      input: snapshot,
    });
  }

  /**
   * Flush startup messages that arrived before the worker acknowledged init.
   *
   * Only the latest state and latest input are kept because render/input loops
   * can produce many messages before the worker is ready.
   */
  function flushPendingMessages(): void {
    if (destroyed || !initialized) {
      return;
    }

    if (pendingInput !== null) {
      forwardWorkerInputNow(pendingInput);
      pendingInput = null;
    }

    if (pendingState !== null) {
      postSimStateNow(pendingState);
      pendingState = null;
    }
  }

  const bridge: NeatensteinRendererBridge = {
    worker,

    get requestId(): number {
      return latestRequestId;
    },

    postSimState(state: NeatensteinRenderState): void {
      if (destroyed) {
        return;
      }

      const constrainedState = constrainRenderState(state);

      if (!initialized) {
        // Keep only the newest render state so worker startup cannot create a
        // backlog of obsolete frames.
        pendingState = constrainedState;
        return;
      }

      postSimStateNow(constrainedState);
    },

    forwardWorkerInput(snapshot: InputSnapshot): void {
      if (destroyed) {
        return;
      }

      if (!initialized) {
        // Keep the latest input snapshot until the worker is ready.
        pendingInput = snapshot;
        return;
      }

      forwardWorkerInputNow(snapshot);
    },

    destroy(): void {
      if (destroyed) {
        return;
      }

      destroyed = true;
      pendingState = null;
      pendingInput = null;
      worker.terminate();
    },
  };

  worker.onmessage = (event: MessageEvent) => {
    if (destroyed || !isMessageRecord(event.data)) {
      return;
    }

    const data = event.data;

    if (data.type === 'initialized') {
      initialized = true;
      flushPendingMessages();
      return;
    }

    if (
      data.type === 'frame' &&
      isMessageRecord(data.frame) &&
      typeof data.frame.requestId === 'number'
    ) {
      latestRequestId = data.frame.requestId;
    }
  };

  worker.postMessage(
    {
      type: 'init',
      tier,
      version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
      mapSeed,
      ...(initialRenderSize
        ? {
            canvasWidth: initialRenderSize.width,
            canvasHeight: initialRenderSize.height,
          }
        : {}),
      ...(offscreen ? { canvas: offscreen } : {}),
    },
    transferList,
  );

  return bridge;
}
