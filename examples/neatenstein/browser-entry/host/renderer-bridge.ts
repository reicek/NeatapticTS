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

import { NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION } from '../constants';
import type { NeatensteinRenderState } from '../renderer/frame';

/**
 * Supported renderer tiers.
 *
 * - `worker`  â†’ computation and rasterization happen on a dedicated worker
 *   using an {@link OffscreenCanvas}.
 * - `cpu`     â†’ computation happens on a worker; the host blits the packed
 *   frame to a main-thread canvas.
 * - `gpu`     â†’ same split as `cpu`, reserved for future GPU-backed
 *   computation.
 */
export type NeatensteinTier = 'worker' | 'cpu' | 'gpu';

/**
 * Configuration needed to create a renderer bridge.
 */
export interface NeatensteinRendererBridgeOptions {
  /** The visible canvas element on the host page. */
  canvas: HTMLCanvasElement;
  /** Absolute or relative URL to the ESM worker bundle. */
  workerUrl: string;
  /** Selected renderer tier. */
  tier: NeatensteinTier;
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
 *   workerUrl: '/assets/neatenstein.worker.esm.js',
 *   tier: 'worker',
 * });
 * bridge.postSimState({ canvasWidth: 640, canvasHeight: 360, simTick: 1 });
 * ```
 */
export function createNeatensteinRendererBridge(
  options: NeatensteinRendererBridgeOptions,
): NeatensteinRendererBridge {
  const { canvas, workerUrl, tier } = options;
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
