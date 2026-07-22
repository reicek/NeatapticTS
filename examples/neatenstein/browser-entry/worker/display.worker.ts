/**
 * Display worker entrypoint for the Neatenstein neon raycasting demo.
 *
 * This worker receives an `init` message that selects a renderer tier, then
 * either:
 *
 * - renders directly to an {@link OffscreenCanvas} for the `worker` tier, or
 * - builds a packed {@link NeatensteinRenderFrame} and posts it back to the
 *   host for the `cpu` and `gpu` tiers.
 *
 * The worker tier is driven by `simState` messages from the host; the CPU/GPU
 * tiers schedule their own `requestAnimationFrame` ticks after initialization.
 *
 * @module
 */

/// <reference lib="webworker" />

import {
  NEATENSTEIN_CPU_COLUMN_COUNT,
  NEATENSTEIN_GPU_COLUMN_COUNT,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
  NEATENSTEIN_WORKER_COLUMN_COUNT,
} from '../constants';
import {
  buildNeatensteinRenderFrame,
  type NeatensteinRenderState,
} from '../renderer/frame';

type DisplayTier = 'worker' | 'cpu' | 'gpu';

let currentTier: DisplayTier | null = null;
let workerCanvas: OffscreenCanvas | null = null;
let latestState: NeatensteinRenderState | null = null;
let rafScheduled = false;

/**
 * Map a tier name to the column count used for packed frames.
 *
 * @param tier - The active renderer tier.
 * @returns The column count for that tier.
 */
function resolveColumnCount(tier: DisplayTier): number {
  switch (tier) {
    case 'gpu':
      return NEATENSTEIN_GPU_COLUMN_COUNT;
    case 'worker':
      return NEATENSTEIN_WORKER_COLUMN_COUNT;
    case 'cpu':
    default:
      return NEATENSTEIN_CPU_COLUMN_COUNT;
  }
}

/**
 * Build a packed frame from the latest state and post it back to the host.
 *
 * For the `worker` tier this also clears the transferred OffscreenCanvas so
 * the frame boundary is visible in a real browser context.
 */
function buildAndPostFrame(): void {
  if (!latestState || !currentTier) {
    return;
  }

  const columnCount = resolveColumnCount(currentTier);
  const frame = buildNeatensteinRenderFrame(latestState, columnCount);

  if (currentTier === 'worker' && workerCanvas) {
    const ctx = workerCanvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, latestState.canvasWidth, latestState.canvasHeight);
    }
  }

  self.postMessage({ type: 'frame', frame });
}

/**
 * Animation-frame loop for CPU/GPU tiers.
 *
 * The loop keeps re-scheduling itself while the worker is running on a tier
 * that ships packed frames to the host.
 */
function rafTick(): void {
  if (currentTier === 'cpu' || currentTier === 'gpu') {
    buildAndPostFrame();
    self.requestAnimationFrame(rafTick);
  }
}

/**
 * Start the CPU/GPU animation loop if it is not already running.
 */
function startCpuGpuLoop(): void {
  if (!rafScheduled) {
    rafScheduled = true;
    self.requestAnimationFrame(rafTick);
  }
}

self.onmessage = (event: MessageEvent) => {
  const data = event.data;
  if (!data || typeof data !== 'object') {
    return;
  }

  if (data.type === 'init') {
    const tier = data.tier as DisplayTier;
    currentTier = tier;
    if (data.canvas) {
      workerCanvas = data.canvas as OffscreenCanvas;
    }

    self.postMessage({
      type: 'initialized',
      tier,
      version: data.version ?? NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    });

    if (tier === 'cpu' || tier === 'gpu') {
      startCpuGpuLoop();
    }
  } else if (data.type === 'simState') {
    latestState = data.state as NeatensteinRenderState;
    buildAndPostFrame();
  }
};
