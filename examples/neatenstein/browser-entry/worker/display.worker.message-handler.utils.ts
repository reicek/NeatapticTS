/**
 * Message-handler executor utilities for the Neatenstein display worker.
 *
 * Extracted from `display.worker.ts` as part of the B2 architecture-debt
 * refactoring.  These utilities provide individual message-case handlers
 * that are orchestrated by {@link ./display.worker.message-handler}.
 *
 * @module
 */

/// <reference lib="webworker" />

import {
  NEATENSTEIN_DEFAULT_SEED,
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
  WORKER_MSG_INIT,
  WORKER_MSG_RESIZE,
  WORKER_MSG_SIM_STATE,
  WORKER_MSG_INITIALIZED,
  RENDER_TIER_WORKER,
  RENDER_TIER_CPU,
  RENDER_TIER_GPU,
} from '../constants';
import { buildNeatensteinMap, createCollisionMap } from '../renderer/map';
import { createGameState } from '../host/game/tick';
import { createEnemyControllerState } from '../shared/enemy-controller';
import { createMlpEnemyPopulation } from '../harness/enemy-mlp';
import {
  mergePendingTickInput,
  inputMessageToTickInput,
} from './display.worker.input.utils';
import { syncWorkerCanvasSize } from './display.worker.canvas.utils';
import {
  createDisplayWorkerState,
  runSimStepParallel,
} from './display.worker.sim.utils';
import { delegateEvaluation } from './display.worker.eval-delegation.utils';
import {
  getWorkerState,
  setWorkerState,
  getSimTickCounter,
  setSimTickCounter,
} from './display.worker';
import type { DisplayTier } from './display.worker.types';
import type { NeatensteinRenderState } from '../renderer/frame';

/**
 * Check whether a message data object has a recognised worker message type.
 *
 * @param data - Raw message data.
 * @returns `true` when the data is a non-null object with a `type` property.
 */
export function isWorkerMessage(data: unknown): data is { type: string } & Record<string, unknown> {
  return (
    data !== null &&
    typeof data === 'object' &&
    'type' in data
  );
}

/**
 * Resolve the display tier from an init message, or `null` when invalid.
 *
 * @param rawTier - Untyped tier value from the message.
 * @returns The validated {@link DisplayTier} or `null`.
 */
export function resolveDisplayTier(rawTier: unknown): DisplayTier | null {
  if (
    rawTier === RENDER_TIER_WORKER ||
    rawTier === RENDER_TIER_CPU ||
    rawTier === RENDER_TIER_GPU
  ) {
    return rawTier as DisplayTier;
  }
  return null;
}

/**
 * Extract the map seed from an init message, falling back to the default.
 *
 * @param mapSeed - Untyped seed value from the message.
 * @returns A finite numeric seed.
 */
export function resolveMapSeed(mapSeed: unknown): number {
  return typeof mapSeed === 'number' && Number.isFinite(mapSeed)
    ? mapSeed
    : NEATENSTEIN_DEFAULT_SEED;
}

/**
 * Process the `init` message: reset state, build the map, and post back
 * the `initialized` confirmation.
 *
 * @param data - The init message data (must include `tier`).
 * @param applyResize - Callback to apply a pending canvas resize.
 */
export function handleInitMessage(
  data: { tier: unknown; canvas?: OffscreenCanvas; mapSeed?: number; version?: number },
): void {
  const tier = resolveDisplayTier(data.tier);
  if (tier === null) return;

  const preservedResize = getWorkerState().pendingResizeDimensions;
  setWorkerState(createDisplayWorkerState());
  const s = getWorkerState();
  s.pendingResizeDimensions = preservedResize;
  s.currentTier = tier;
  setSimTickCounter(0);

  if (data.canvas) {
    s.workerCanvas = data.canvas as OffscreenCanvas;
  }

  if (s.workerCanvas !== null && s.pendingResizeDimensions !== null) {
    syncWorkerCanvasSize(
      s.workerCanvas,
      s.pendingResizeDimensions.width,
      s.pendingResizeDimensions.height,
    );
    s.pendingResizeDimensions = null;
  }

  const seed = resolveMapSeed(data.mapSeed);

  s.wallMap = buildNeatensteinMap(seed);
  s.collisionMap = createCollisionMap(s.wallMap, NEATENSTEIN_MAP_SIZE);
  s.gameState = createGameState({ seed });
  s.enemyControllerState = createEnemyControllerState(s.gameState);
  s.enemyPopulation = createMlpEnemyPopulation({ seed: s.gameState.seed });

  const version = data.version ?? NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION;

  self.postMessage({ type: WORKER_MSG_INITIALIZED, tier, version });
}

/**
 * Process the `simState` message: store the latest render state, run the
 * deterministic simulation step, and build/post the next frame.
 *
 * @param renderState - The incoming render state from the host.
 * @param buildAndPostFrame - Callback that builds and posts the frame.
 */
export function handleSimStateMessage(
  renderState: NeatensteinRenderState,
  buildAndPostFrame: () => void,
): void {
  const s = getWorkerState();
  s.latestState = renderState;

  const updated = runSimStepParallel(
    s,
    delegateEvaluation,
    getSimTickCounter(),
  );
  if (updated) setWorkerState(updated);
  setSimTickCounter(getSimTickCounter() + 1);

  buildAndPostFrame();
}

/**
 * Process the input message: merge the incoming tick input into the
 * pending queue.
 *
 * @param input - Raw input value from the message.
 */
export function handleInputMessage(input: unknown): void {
  const s = getWorkerState();
  const nextInput = inputMessageToTickInput(input);
  s.pendingTickInput = mergePendingTickInput(
    s.pendingTickInput,
    nextInput,
  );
}

/** Re-export of WORKER_MSG_INIT for the orchestrator. */
export { WORKER_MSG_INIT as INIT_TYPE };
/** Re-export of WORKER_MSG_RESIZE for the orchestrator. */
export { WORKER_MSG_RESIZE as RESIZE_TYPE };
/** Re-export of WORKER_MSG_SIM_STATE for the orchestrator. */
export { WORKER_MSG_SIM_STATE as SIM_STATE_TYPE };
/** Re-export of NEATENSTEIN_INPUT_MESSAGE_TYPE for the orchestrator. */
export { NEATENSTEIN_INPUT_MESSAGE_TYPE as INPUT_TYPE };