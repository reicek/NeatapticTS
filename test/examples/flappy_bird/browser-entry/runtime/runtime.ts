import { createCanvasHost, updateStatsTableValues } from '../host/host';
import {
  createRuntimeTelemetryState,
  disconnectRuntimeTelemetry,
} from './runtime.telemetry.service';
import { createEvolutionWorker } from '../worker-channel/worker-channel';
import {
  FLAPPY_BROWSER_ELITISM_COUNT,
  FLAPPY_BROWSER_POPULATION_SIZE,
  DEFAULT_CONTAINER_ID,
  FLAPPY_HUD_INITIALIZING_TEXT,
  FLAPPY_HUD_ZERO_TEXT,
} from '../../constants/constants';
import {
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from '../../constants/constants';
import {
  resolveRequiredRuntimeHostElement,
  resolveRuntimeHudErrorStatus,
} from './runtime.errors';
import { runRuntimeEvolutionLoop } from './runtime.evolution-loop.service';
import { installRuntimeBrowserGlobals } from './runtime.browser-globals.service';
import type { RuntimeContainerTarget, RuntimeRunHandle } from './runtime.types';

export type { RuntimeRunHandle as FlappyBirdRunHandle } from './runtime.types';

/**
 * Starts the Flappy Bird NeatapticTS browser demo and returns lifecycle controls.
 *
 * This function is intentionally orchestration-focused:
 * 1) resolve runtime dependencies (DOM host, worker, host UI),
 * 2) initialize worker and telemetry plumbing,
 * 3) run the evolve -> playback -> HUD fold loop until stopped,
 * 4) expose a small stop/isRunning/done handle for callers.
 *
 * @example
 * ```ts
 * const runHandle = await start('flappy-bird-output');
 * // later
 * runHandle.stop();
 * await runHandle.done;
 * ```
 *
 * @param container - Element id or HTMLElement to host the demo.
 * @returns Run handle for stop/state control.
 */
export const start = async (
  container: RuntimeContainerTarget = DEFAULT_CONTAINER_ID,
): Promise<RuntimeRunHandle> => {
  // Step 1: Resolve and validate the host element where the demo will render.
  const hostElement = resolveRequiredRuntimeHostElement(container);

  /** Network input width used when rendering architecture labels. */
  const inputSize = FLAPPY_NETWORK_INPUT_SIZE;
  /** Network output width used when rendering architecture labels. */
  const outputSize = FLAPPY_NETWORK_OUTPUT_SIZE;
  /** Worker initialization population size for browser evolution/playback. */
  const populationSize = FLAPPY_BROWSER_POPULATION_SIZE;
  /** Worker initialization elitism count for browser evolution/playback. */
  const elitismCount = FLAPPY_BROWSER_ELITISM_COUNT;

  // Step 2: Create the browser host (canvas + HUD + network panel render hook).
  const { canvas, context, statsValueByKey, renderNetworkArchitecture } =
    createCanvasHost(hostElement);

  // Step 3: Attach optional runtime instrumentation observer when enabled.
  const runtimeTelemetryState = createRuntimeTelemetryState();

  // Step 4: Paint initial status before worker bootstrapping starts.
  updateStatsTableValues(statsValueByKey, {
    status: FLAPPY_HUD_INITIALIZING_TEXT,
    birds: `${FLAPPY_HUD_ZERO_TEXT}/${populationSize}`,
  });

  // Step 5: Create the evolution worker channel used by the orchestrator loop.
  const evolutionWorker = createEvolutionWorker();

  /** Mutable stop flag consulted by all loop and callback paths. */
  let stopped = false;
  /** Resolver captured for the externally exposed completion promise. */
  let resolveDone: (() => void) | undefined;
  /** Promise that resolves exactly once after stop/teardown is complete. */
  const done = new Promise<void>((resolve) => {
    resolveDone = resolve;
  });

  /**
   * Stops the running demo and performs idempotent resource teardown.
   *
   * Teardown includes:
   * - disconnecting instrumentation observers,
   * - notifying and terminating the worker,
   * - resolving the `done` promise exactly once.
   *
   * @returns Nothing.
   */
  const stop = () => {
    // Step 1: Keep stop idempotent for repeated calls from multiple paths.
    if (stopped) {
      return;
    }

    // Step 2: Flip loop guard first so no new async work starts.
    stopped = true;

    // Step 3: Release runtime resources in deterministic order.
    disconnectRuntimeTelemetry(runtimeTelemetryState);
    evolutionWorker.postMessage({ type: 'stop' });
    evolutionWorker.terminate();

    // Step 4: Resolve external completion handle.
    resolveDone?.();
  };

  /**
   * Reports whether the demo loop is currently active.
   *
   * @returns `true` when not stopped.
   */
  const isStopped = () => stopped;

  /**
   * Reports whether the demo loop is currently active.
   *
   * @returns `true` when not stopped.
   */
  const isRunning = () => !stopped;

  // Step 6: Start orchestration loop and route unexpected errors to HUD + teardown.
  void runRuntimeEvolutionLoop({
    evolutionWorker,
    canvas,
    context,
    statsValueByKey,
    renderNetworkArchitecture,
    populationSize,
    elitismCount,
    inputSize,
    outputSize,
    runtimeTelemetryState,
    isStopped,
  }).catch((error: unknown) => {
    updateStatsTableValues(statsValueByKey, {
      status: resolveRuntimeHudErrorStatus(error),
    });
    stop();
  });

  return { stop, isRunning, done };
};

installRuntimeBrowserGlobals(start);
