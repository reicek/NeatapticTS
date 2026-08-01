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
  /** Absolute or relative URL to the ESM worker bundle. */
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
  /** Forward a simulation state snapshot to the worker. */
  postSimState(state: NeatensteinRenderState): void;
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

  const worker = new Worker(workerUrl, { type: 'module' });

  let destroyed = false;
  let initialized = false;
  let latestRequestId = 0;
  let pendingState: NeatensteinRenderState | null = null;
  let pendingInput: InputSnapshot | null = null;
  let frameConsumer: ((frame: NeatensteinRenderFrame) => void) | null = null;

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

      postSimStateNow(state);
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
      worker.terminate();
    },

    setFrameConsumer(consumer: (frame: NeatensteinRenderFrame) => void): void {
      frameConsumer = consumer;
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
