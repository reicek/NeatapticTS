import Network from '../../../../src/architecture/network';
import {
  createCanvasHost,
  updateStatsTableValues,
} from './browser-entry.host.utils';
import { resolveNetworkArchitectureLabel } from './network-view/network-view';
import { animatePopulationEpisode } from './playback/playback';
import {
  createMinorGcObserver,
  resolveEventsPerMinute,
  resolveHudUpdatesPerSecond,
  trimSamplesToWindow,
} from './browser-entry.telemetry.utils';
import {
  createEvolutionWorker,
  requestWorkerGeneration,
} from './browser-entry.worker-channel.utils';
import {
  FLAPPY_BROWSER_ELITISM_COUNT,
  FLAPPY_BROWSER_POPULATION_SIZE,
  DEFAULT_CONTAINER_ID,
  FLAPPY_HUD_INITIALIZING_TEXT,
  FLAPPY_HUD_OFF_TEXT,
  FLAPPY_HUD_UPDATE_INTERVAL_FRAMES,
  FLAPPY_HUD_UPDATES_WINDOW_MS,
  FLAPPY_HUD_ZERO_DECIMAL_TEXT,
  FLAPPY_HUD_ZERO_TEXT,
  FLAPPY_MINOR_GC_WINDOW_MS,
  FLAPPY_STATUS_EVOLVING_TEXT,
  FLAPPY_STATUS_PLAYING_TEXT,
} from '../constants/constants';
import type { FlappyBirdRunHandle, RuntimeWindow } from './browser-entry.types';
import {
  FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from '../constants/constants';

export type { FlappyBirdRunHandle } from './browser-entry.types';

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
  container: string | HTMLElement = DEFAULT_CONTAINER_ID,
): Promise<FlappyBirdRunHandle> => {
  // Step 1: Resolve and validate the host element where the demo will render.
  const hostElement =
    typeof container === 'string'
      ? document.getElementById(container)
      : container;

  if (!hostElement) {
    throw new Error(`Flappy demo container not found: ${String(container)}`);
  }

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

  /** Sliding timestamp window used to compute HUD updates/sec telemetry. */
  const hudUpdateTimestampsMs: number[] = [];
  /** Sliding timestamp window used to compute minor GC/min telemetry. */
  const minorGcTimestampsMs: number[] = [];

  // Step 3: Attach optional runtime instrumentation observer when enabled.
  const gcObserver = FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
    ? createMinorGcObserver(minorGcTimestampsMs)
    : undefined;

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
    gcObserver?.disconnect();
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
  const isRunning = () => !stopped;

  // Step 6: Start orchestration loop and route unexpected errors to HUD + teardown.
  void runEvolutionLoop().catch((error: unknown) => {
    updateStatsTableValues(statsValueByKey, {
      status: `error: ${String((error as Error)?.message ?? error)}`,
    });
    stop();
  });

  return { stop, isRunning, done };

  /**
   * Runs generation orchestration and playback until stopped.
   *
   * @returns Nothing.
   */
  async function runEvolutionLoop(): Promise<void> {
    // Step 1: Track best-so-far metrics across all completed generations.
    let bestRunFrames = 0;
    let bestRunPipes = 0;

    // Step 2: Initialize worker runtime with deterministic seed + config.
    evolutionWorker.postMessage({
      type: 'init',
      payload: {
        populationSize,
        elitismCount,
        rngSeed: 0x1234abcd,
      },
    });

    // Step 3: Keep iterating generations until a stop signal is observed.
    while (!stopped) {
      // Step 3.1: Request and await the next evolved generation payload.
      const generationPayload = await requestWorkerGeneration(evolutionWorker);
      if (stopped) {
        break;
      }

      // Step 3.2: Resolve generation-level values and best-network visualization model.
      const bestFitness = generationPayload.bestFitness;
      const bestNetwork = generationPayload.bestNetworkJson
        ? Network.fromJSON(generationPayload.bestNetworkJson)
        : undefined;
      const bestArchitectureLabel = resolveNetworkArchitectureLabel(
        bestNetwork,
        inputSize,
        outputSize,
      );

      // Step 3.3: Hydrate current-generation HUD values before playback begins.
      updateStatsTableValues(statsValueByKey, {
        currentHeader: `Current run · Gen ${generationPayload.generation}`,
        currentFrames: FLAPPY_HUD_ZERO_TEXT,
        currentPipes: FLAPPY_HUD_ZERO_TEXT,
        currentMaxFrames: FLAPPY_HUD_ZERO_TEXT,
        currentMaxPipes: FLAPPY_HUD_ZERO_TEXT,
        currentArchitecture: bestArchitectureLabel,
        telemetryHeader: 'Instrumentation',
        telemetryActivationsPerFrame: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
          ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
          : FLAPPY_HUD_OFF_TEXT,
        telemetrySimulationStepsPerRaf: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
          ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
          : FLAPPY_HUD_OFF_TEXT,
        telemetryHudUpdatesPerSecond: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
          ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
          : FLAPPY_HUD_OFF_TEXT,
        telemetryMinorGcPerMinute: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
          ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
          : FLAPPY_HUD_OFF_TEXT,
        bestHeader: 'Best run',
        bestFrames: String(bestRunFrames),
        bestPipes: String(bestRunPipes),
        bestMaxFrames: String(bestRunFrames),
        bestMaxPipes: String(bestRunPipes),
        bestArchitecture: bestArchitectureLabel,
        status: FLAPPY_STATUS_PLAYING_TEXT,
        birds: `${FLAPPY_HUD_ZERO_TEXT}/${populationSize}`,
      });

      // Step 3.4: Render active network architecture in the side panel.
      renderNetworkArchitecture(bestNetwork, inputSize, outputSize);

      // Step 3.5: Run playback and stream per-frame HUD updates.
      const playbackSummary = await animatePopulationEpisode(
        canvas,
        context,
        evolutionWorker,
        (frameStats) => {
          // Step 3.5.1: Throttle HUD writes to reduce layout/repaint churn.
          if (frameStats.frameIndex % FLAPPY_HUD_UPDATE_INTERVAL_FRAMES !== 0) {
            return;
          }

          // Step 3.5.2: Maintain sliding telemetry windows when instrumentation is enabled.
          if (FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION) {
            const nowMs = performance.now();
            hudUpdateTimestampsMs.push(nowMs);
            trimSamplesToWindow(
              hudUpdateTimestampsMs,
              FLAPPY_HUD_UPDATES_WINDOW_MS,
              nowMs,
            );
            trimSamplesToWindow(
              minorGcTimestampsMs,
              FLAPPY_MINOR_GC_WINDOW_MS,
              nowMs,
            );
          }

          // Step 3.5.3: Publish current frame counters + instrumentation values.
          updateStatsTableValues(statsValueByKey, {
            birds: `${frameStats.activeBirdCount}/${populationSize}`,
            currentFrames: String(frameStats.frameIndex),
            currentPipes: String(frameStats.leaderPipesPassed),
            currentMaxFrames: String(frameStats.leaderFramesSurvived),
            currentMaxPipes: String(frameStats.leaderPipesPassed),
            telemetryActivationsPerFrame: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
              ? frameStats.activationCallsPerFrame.toFixed(2)
              : FLAPPY_HUD_OFF_TEXT,
            telemetrySimulationStepsPerRaf:
              FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
                ? frameStats.simulationStepsPerRaf.toFixed(2)
                : FLAPPY_HUD_OFF_TEXT,
            telemetryHudUpdatesPerSecond: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
              ? resolveHudUpdatesPerSecond(hudUpdateTimestampsMs).toFixed(2)
              : FLAPPY_HUD_OFF_TEXT,
            telemetryMinorGcPerMinute: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
              ? resolveEventsPerMinute(minorGcTimestampsMs).toFixed(2)
              : FLAPPY_HUD_OFF_TEXT,
          });
        },
      );

      // Step 3.6: Fold generation winner into cross-generation maxima.
      bestRunFrames = Math.max(
        bestRunFrames,
        playbackSummary.winnerFramesSurvived,
      );
      bestRunPipes = Math.max(bestRunPipes, playbackSummary.winnerPipesPassed);

      // Step 3.7: Finalize generation HUD summary and switch status back to evolving.
      updateStatsTableValues(statsValueByKey, {
        bestFrames: String(playbackSummary.winnerFramesSurvived),
        bestPipes: String(playbackSummary.winnerPipesPassed),
        bestMaxFrames: String(bestRunFrames),
        bestMaxPipes: String(bestRunPipes),
        status: FLAPPY_STATUS_EVOLVING_TEXT,
        birds: `${FLAPPY_HUD_ZERO_TEXT}/${populationSize}`,
      });

      // Step 3.8: Emit compact generation summary to console (best-effort only).
      try {
        console.log(
          `[flappy_bird] gen=${generationPayload.generation} fitness=${bestFitness.toFixed(0)} winnerPipes=${playbackSummary.winnerPipesPassed} winnerFrames=${playbackSummary.winnerFramesSurvived} avgPipes=${playbackSummary.averagePipesPassed.toFixed(2)} p90Frames=${playbackSummary.p90FramesSurvived}`,
        );
      } catch {
        // ignore console write issues
      }
    }
  }
};

/**
 * Publishes browser globals for demo auto-start and host-driven control.
 *
 * This keeps parity with the asciiMaze entry style:
 * - `window.flappyBird.start(...)` for explicit invocation,
 * - `window.flappyBirdStart(...)` for compatibility,
 * - one guarded auto-start for standalone HTML usage.
 */
try {
  // Step 1: Resolve runtime window shape used by the browser demo host.
  const runtimeWindow = window as unknown as RuntimeWindow;

  // Step 2: Publish canonical entry points.
  runtimeWindow.flappyBird = runtimeWindow.flappyBird ?? {};
  runtimeWindow.flappyBird.start = start;
  runtimeWindow.flappyBirdStart = (containerElement?: unknown) =>
    start(containerElement as never);

  // Step 3: Auto-start exactly once when loaded in standalone pages.
  if (!runtimeWindow.flappyBird._autoStarted) {
    runtimeWindow.flappyBird._autoStarted = true;
    setTimeout(() => {
      try {
        start().catch(() => undefined);
      } catch {
        // ignore
      }
    }, 20);
  }
} catch {
  // ignore global wiring failures (non-browser environments).
}
