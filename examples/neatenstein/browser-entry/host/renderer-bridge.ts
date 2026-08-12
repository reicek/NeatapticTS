/**
 * Host-side renderer bridge for the Neatenstein neon raycasting demo.
 *
 * This module owns the boundary between the main browser thread and the
 * display worker. It is responsible for:
 *
 * - spawning the module worker,
 * - forwarding the host-derived render size to the worker without constraints,
 * - transferring an {@link OffscreenCanvas} on the `worker` tier,
 * - forwarding simulation state to the worker,
 * - forwarding input snapshots to the worker,
 * - exposing the latest rendered frame request id for CPU/GPU tiers,
 * - and terminating the worker on teardown.
 *
 * The bridge intentionally remains thin; it forwards the host-provided canvas
 * backing-store dimensions because this is the last host-side boundary before
 * the worker owns direct rendering.
 *
 * @module
 */

import {
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
  NEATENSTEIN_WORKER_BUNDLE_FILENAME,
  type NeatensteinTier,
} from '../constants';
import type {
  NeatensteinRenderFrame,
  NeatensteinRenderState,
} from '../renderer/frame';
import type { InputSnapshot } from './input';

/**
 * Default worker bundle URL used when the caller does not supply one.
 */
const DEFAULT_WORKER_URL = `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`;

/**
 * Configuration needed to create a host side renderer bridge that spawns the
 * display worker and transfers the OffscreenCanvas to the selected tier.
 */
export interface NeatensteinRendererBridgeOptions {
  /** The visible canvas element on the host page. */
  canvas: HTMLCanvasElement;
  /** Absolute or relative URL to the worker bundle. */
  workerUrl?: string;
  /** Selected renderer tier. */
  tier: NeatensteinTier;
  /** Deterministic seed used to build the wall grid on the worker. */
  mapSeed: number;
}

/**
 * Public surface of the host renderer bridge that callers use to forward
 * state, send input, and consume rendered frames from the display worker.
 */
export interface NeatensteinRendererBridge {
  /** The spawned display worker. */
  worker: Worker;
  /** Latest frame request id received from the worker. */
  requestId: number;
  /** Forward a simulation state snapshot to the worker. The bridge applies
   * worker-busy backpressure so only one snapshot is in flight at a time;
   * additional calls defer the latest state until the worker acknowledges. */
  postSimState(state: NeatensteinRenderState): void;
  /** Forward a host-resized CSS-box dimension to the worker. */
  resize(width: number, height: number): void;
  /** Forward an input snapshot to the worker so it can advance the sim tick. */
  forwardWorkerInput(snapshot: InputSnapshot): void;
  /** Terminate the worker and release the bridge. */
  destroy(): void;
  /**
   * Register a callback that receives every rendered frame produced by the
   * worker on the `cpu` and `gpu` tiers.
   *
   * The consumer is called synchronously when a `frame` message arrives, before
   * the bridge updates its latest request id. This lets the host overlay or
   * capture pipeline consume the same frame payload without an extra copy.
   *
   * @param consumer - Function invoked with each incoming render frame.
   */
  setFrameConsumer(consumer: (frame: NeatensteinRenderFrame) => void): void;
  /**
   * Register a callback invoked when the worker finishes rendering a frame
   * and has no pending state to flush. The host uses this to schedule the
   * next `requestAnimationFrame`, making the render loop purely worker-paced
   * instead of running continuously at display refresh rate.
   *
   * The callback is NOT called when a deferred `pendingState` is flushed on
   * the frame ack — the flushed state's own ack will trigger it once the
   * worker finishes that render.
   *
   * Pass `null` to clear the callback.
   *
   * @param callback - Function invoked when the worker is idle and ready for
   *   the next frame, or `null` to clear.
   */
  setOnFrameReady(callback: (() => void) | null): void;
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
 * Resolve the initial canvas dimensions the bridge forwards to the worker.
 *
 * The bridge prefers the canvas backing-store dimensions because the host
 * browser entry already set them to the fixed 480px-height, viewport-aspect-ratio
 * size. If those are unavailable, it falls back to layout dimensions.
 *
 * @param canvas - Visible host canvas.
 * @returns A dimension pair, or `null` if no usable dimensions exist.
 */
function resolveInitialCanvasRenderSize(
  canvas: HTMLCanvasElement,
): { width: number; height: number } | null {
  const backingWidth = canvas.width;
  const backingHeight = canvas.height;

  if (
    isPositiveFiniteDimension(backingWidth) &&
    isPositiveFiniteDimension(backingHeight)
  ) {
    return { width: backingWidth, height: backingHeight };
  }

  if (
    isPositiveFiniteDimension(canvas.clientWidth) &&
    isPositiveFiniteDimension(canvas.clientHeight)
  ) {
    return { width: canvas.clientWidth, height: canvas.clientHeight };
  }

  return null;
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
 * Create and initialize a host-side bridge to the Neatenstein display worker.
 *
 * For the `worker` tier, the canvas backing store is already sized by the host
 * browser entry and is then transferred to the worker through
 * {@link HTMLCanvasElement.transferControlToOffscreen}. For `cpu` and `gpu`
 * tiers, the canvas stays on the host and the worker posts packed frames back.
 *
 * The bridge forwards the host-provided canvas dimensions in the `init`
 * message and passes subsequent {@link NeatensteinRenderState} values through
 * unchanged, so the worker renders at the fixed 480px-height, viewport-aspect-ratio
 * size.
 *
 * The bridge queues the latest simulation state and latest input snapshot until
 * the worker acknowledges initialization. This avoids losing early host-loop
 * messages during worker startup.
 *
 * After initialization, the bridge applies worker-busy backpressure: only one
 * simState message is in flight at a time. When the worker acknowledges a
 * rendered frame, any deferred state (the latest snapshot received while the
 * worker was busy) is flushed immediately. This naturally throttles posting to
 * the worker's actual render capacity regardless of the display refresh rate.
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

  // Read the canvas dimensions set by the host browser entry. The host already
  // sized the backing store to the fixed 480px-height, viewport-aspect-ratio
  // render size, so the bridge only forwards those dimensions to the worker
  // without re-constraining.
  const initialRenderSize = resolveInitialCanvasRenderSize(canvas);

  const worker = new Worker(workerUrl);

  let destroyed = false;
  let initialized = false;
  let latestRequestId = 0;
  let pendingState: NeatensteinRenderState | null = null;
  let pendingInput: InputSnapshot | null = null;
  let frameConsumer: ((frame: NeatensteinRenderFrame) => void) | null = null;
  let onFrameReadyCallback: (() => void) | null = null;
  let workerBusy = false;

  let offscreen: OffscreenCanvas | undefined;
  const transferList: Transferable[] = [];

  if (tier === 'worker') {
    if (typeof canvas.transferControlToOffscreen !== 'function') {
      worker.terminate();

      throw new Error(
        'Neatenstein worker tier requires HTMLCanvasElement.transferControlToOffscreen().',
      );
    }

    // Transfer after the host already sized the backing store.
    offscreen = canvas.transferControlToOffscreen();
    transferList.push(offscreen);
  }

  /**
   * Send a simulation state immediately if the bridge is alive.
   *
   * @param state - Render state to send.
   */
  function postSimStateNow(state: NeatensteinRenderState): void {
    worker.postMessage({ type: 'simState', state });
  }

  /**
   * Send a host-resized CSS-box dimension to the worker immediately.
   *
   * @param width - Host-derived render width in pixels.
   * @param height - Host-derived render height in pixels.
   */
  function postResizeMessage(width: number, height: number): void {
    worker.postMessage({ type: 'resize', width, height });
  }

  /**
   * Send an input snapshot immediately if the bridge is alive.
   *
   * @param snapshot - Input snapshot to send.
   */
  function forwardWorkerInputNow(snapshot: InputSnapshot): void {
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
    if (pendingInput !== null) {
      forwardWorkerInputNow(pendingInput);
      pendingInput = null;
    }

    if (pendingState !== null) {
      postSimStateNow(pendingState);
      pendingState = null;
      workerBusy = true;
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

      if (!initialized) {
        // Keep only the newest render state so worker startup cannot create a
        // backlog of obsolete frames.
        pendingState = state;
        return;
      }

      if (workerBusy) {
        // Worker is still rendering the previous frame. Defer this state until
        // the next frame acknowledgment to prevent message-queue flooding on
        // high refresh-rate displays. Only the latest state is kept.
        pendingState = state;
        return;
      }

      workerBusy = true;
      postSimStateNow(state);
    },

    resize(width: number, height: number): void {
      if (destroyed) {
        return;
      }

      postResizeMessage(width, height);
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
      frameConsumer = null;
      onFrameReadyCallback = null;
      workerBusy = false;
      worker.terminate();
    },

    setFrameConsumer(consumer: (frame: NeatensteinRenderFrame) => void): void {
      frameConsumer = consumer;
    },

    setOnFrameReady(callback: (() => void) | null): void {
      onFrameReadyCallback = callback;
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

      if (frameConsumer !== null) {
        frameConsumer(data.frame as unknown as NeatensteinRenderFrame);
      }

      // Backpressure: the worker has finished rendering this frame. Clear the
      // busy flag so the next postSimState can post immediately. If a deferred
      // state is pending (arrived while the worker was busy), flush it now so
      // the worker starts the next render without waiting for another rAF
      // tick. When no state is pending, notify the host via onFrameReady so it
      // can schedule the next requestAnimationFrame — making the loop purely
      // worker-paced instead of running continuously at display refresh rate.
      workerBusy = false;
      if (pendingState !== null) {
        const state = pendingState;
        pendingState = null;
        workerBusy = true;
        postSimStateNow(state);
      } else if (onFrameReadyCallback !== null) {
        onFrameReadyCallback();
      }
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
