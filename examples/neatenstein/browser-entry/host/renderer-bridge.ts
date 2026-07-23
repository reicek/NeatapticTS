/**
 * Host-side renderer bridge for the Neatenstein neon raycasting demo.
 *
 * This module owns the boundary between the main browser thread and the
 * display worker. It is responsible for:
 *
 * - spawning the module worker,
 * - transferring an {@link OffscreenCanvas} on the `worker` tier,
 * - forwarding simulation state to the worker,
 * - exposing the latest rendered frame request id for CPU/GPU tiers,
 * - and terminating the worker on teardown.
 *
 * The bridge is intentionally thin: all rendering decisions live in the worker
 * or in the renderer modules, not here.
 *
 * @module
 */

import {
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
  NEATENSTEIN_WORKER_BUNDLE_FILENAME,
  type NeatensteinTier,
} from '../constants';
import type { InputSnapshot } from './input';
import type { NeatensteinRenderState } from '../renderer/frame';

/**
 * Default worker bundle URL used when the caller does not supply one.
 */
const DEFAULT_WORKER_URL = `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`;

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
  /** Latest frame request id received from the worker (CPU/GPU tiers only). */
  requestId: number;
  /** Forward a simulation state snapshot to the worker. */
  postSimState(state: NeatensteinRenderState): void;
  /** Forward an input snapshot to the worker so it can advance the sim tick. */
  forwardWorkerInput(snapshot: InputSnapshot): void;
  /** Terminate the worker and release the bridge. */
  destroy(): void;
}

/**
 * Create a host-side bridge to the Neatenstein display worker.
 *
 * For the `worker` tier the canvas control is transferred to the worker via
 * {@link HTMLCanvasElement.transferControlToOffscreen}. For `cpu` and `gpu`
 * tiers the canvas stays on the host and the worker posts packed frames back.
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
 * bridge.postSimState({
 *   canvasWidth: 640,
 *   canvasHeight: 360,
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
  const worker = new Worker(workerUrl, { type: 'module' });

  let offscreen: OffscreenCanvas | undefined;
  const transferList: Transferable[] = [];

  if (tier === 'worker') {
    offscreen = canvas.transferControlToOffscreen();
    transferList.push(offscreen);
  }

  worker.postMessage(
    {
      type: 'init',
      tier,
      version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
      mapSeed,
      ...(offscreen ? { canvas: offscreen } : {}),
    },
    transferList,
  );

  const bridge: NeatensteinRendererBridge = {
    worker,
    requestId: 0,
    postSimState(state: NeatensteinRenderState) {
      worker.postMessage({ type: 'simState', state });
    },
    forwardWorkerInput(snapshot: InputSnapshot) {
      worker.postMessage({
        type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
        input: snapshot,
      });
    },
    destroy() {
      worker.terminate();
    },
  };

  if (tier !== 'worker') {
    worker.onmessage = (event: MessageEvent) => {
      const data = event.data;
      if (
        data &&
        typeof data === 'object' &&
        data.type === 'frame' &&
        data.frame &&
        typeof data.frame.requestId === 'number'
      ) {
        bridge.requestId = data.frame.requestId;
      }
    };
  }

  return bridge;
}
