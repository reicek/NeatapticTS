/**
 * Eval-worker delegation executors for the Neatenstein display worker.
 *
 * Extracted from `display.worker.ts` as part of the B2 architecture-debt
 * refactoring.  These executors own the lifecycle of the dedicated eval
 * worker that performs arms-race generation evaluation off the render loop.
 *
 * @module
 */

/// <reference lib="webworker" />

import { EVAL_MSG_EVALUATE, EVAL_MSG_EVAL_COMPLETE } from '../constants';
import { NEATENSTEIN_MAIN_NEAT_INPUTS } from '../harness/neat-io-config';
import type { MlpSnapshot } from '../harness/types';
import type {
  EvalRequestPayload,
  EvalCompletePayload,
} from './display.worker.types';
import { getWorkerState, setWorkerState } from './display.worker';

/**
 * Resolve the eval worker URL from the display worker's location.
 *
 * @returns Absolute URL to the eval worker bundle, or `null` when the
 *   location cannot be resolved (e.g. in test environments).
 */
export function resolveEvalWorkerUrl(): string | null {
  try {
    const href = self.location?.href;
    if (typeof href !== 'string' || !href) return null;
    return href.replace('neatenstein.worker.js', 'neatenstein.eval-worker.js');
  } catch {
    return null;
  }
}

/**
 * Handle the `evalComplete` message from the eval worker.
 *
 * Deserializes the champion network via `Network.fromJSON()`, applies the
 * advanced generation to `gameState`, stores the champion network, and
 * clears the launch guard.
 *
 * @param event - Message event from the eval worker.
 */
export async function handleEvalComplete(event: MessageEvent): Promise<void> {
  const data = event.data as EvalCompletePayload | null;
  if (
    !data ||
    typeof data !== 'object' ||
    data.type !== EVAL_MSG_EVAL_COMPLETE
  ) {
    return;
  }

  // Lazy-load Network for deserialization (avoids pulling neataptic into
  // the static import chain, which triggers GPUDevice type errors in the
  // test environment).
  const { Network } = await import('neataptic');

  const championNetwork = Network.fromJSON(data.championNetworkJSON);

  const s = getWorkerState();
  // Apply the result (generation always matches pendingGeneration + 1).
  s.gameState = { ...s.gameState!, generation: data.generation };
  // Store the champion main-agent network for Phase 4's player controller.
  s.championMainNetwork = championNetwork;
  s.lastChampionInputCount = NEATENSTEIN_MAIN_NEAT_INPUTS;

  // Clear the launch guard.
  s.pendingGeneration = null;
  setWorkerState(s);
}

/**
 * Get or create the dedicated eval worker.
 *
 * The worker is created lazily on the first call. In the browser, the URL
 * is derived from the display worker's location. In tests, a mock worker
 * is injected via the test-hooks module.
 *
 * @returns The eval worker instance, or `null` when no worker can be
 *   created (e.g. the `Worker` constructor is unavailable).
 */
export function getOrCreateEvalWorker(): Worker | null {
  const s = getWorkerState();
  if (s.evalWorker) return s.evalWorker;

  const url = resolveEvalWorkerUrl();
  if (!url) return null;

  try {
    const workerCtor = (globalThis as { Worker?: typeof Worker }).Worker;
    if (!workerCtor) return null;
    s.evalWorker = new workerCtor(url);
    s.evalWorker.onmessage = handleEvalComplete;
  } catch {
    s.evalWorker = null;
  }

  setWorkerState(s);
  return s.evalWorker;
}

/**
 * Delegate the arms-race generation evaluation to the eval worker.
 *
 * Posts an evaluate request to the eval worker via `postMessage`. The
 * evaluation runs entirely in the eval worker, off the display worker's
 * render loop — no blocking `await` on the main thread. When the eval
 * worker completes, it posts back an `evalComplete` message which is
 * handled by {@link handleEvalComplete}.
 *
 * @param seed - Game seed.
 * @param generation - Current generation (post-advanceWave).
 * @param enemySnapshot - Frozen enemy snapshot from advanceWave.
 * @param humanModeBool - Whether the game is in auto mode.
 */
export function delegateEvaluation(
  seed: number,
  generation: number,
  enemySnapshot: MlpSnapshot,
  humanModeBool: boolean,
): void {
  const worker = getOrCreateEvalWorker();
  if (!worker) return;

  const payload: EvalRequestPayload = {
    type: EVAL_MSG_EVALUATE,
    seed,
    generation,
    enemySnapshot,
    humanMode: humanModeBool,
  };
  worker.postMessage(payload);
}
