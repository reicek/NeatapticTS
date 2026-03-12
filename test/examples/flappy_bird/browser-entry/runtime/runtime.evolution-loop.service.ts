import Network from '../../../../../src/architecture/network';
import { updateStatsTableValues } from '../host/host';
import { resolveNetworkArchitectureLabel } from '../network-view/network-view';
import { animatePopulationEpisode } from '../playback/playback';
import { requestWorkerGeneration } from '../worker-channel/worker-channel';
import {
  FLAPPY_HUD_UPDATE_INTERVAL_FRAMES,
  FLAPPY_HUD_ZERO_TEXT,
  FLAPPY_DEFAULT_RNG_SEED,
  FLAPPY_STATUS_EVOLVING_TEXT,
  FLAPPY_STATUS_PLAYING_TEXT,
} from '../../constants/constants';
import {
  resolveInitialRuntimeTelemetryHudValues,
  resolveRuntimeTelemetryHudValues,
} from './runtime.telemetry.service';
import type {
  FlappyStatsTableCells,
  NetworkVisualizationHandle,
} from '../browser-entry.types';
import type { RuntimeTelemetryState } from './runtime.telemetry.service';

/**
 * Long-running evolution/playback orchestration for the browser runtime.
 *
 * This loop is the heart of the interactive demo. It repeatedly asks the worker
 * for the next evolved generation, updates the HUD and network view, plays back
 * that generation on the canvas, then folds the outcome into best-so-far
 * browser state.
 */

/**
 * Dependencies required to run the browser runtime evolution loop.
 *
 * Grouping these dependencies into one object keeps the public loop entry more
 * declarative and avoids a long positional parameter list.
 */
export interface RuntimeEvolutionLoopOptions {
  evolutionWorker: Worker;
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  statsValueByKey: FlappyStatsTableCells;
  renderNetworkArchitecture: NetworkVisualizationHandle['renderNetworkArchitecture'];
  populationSize: number;
  elitismCount: number;
  inputSize: number;
  outputSize: number;
  runtimeTelemetryState: RuntimeTelemetryState;
  isStopped: () => boolean;
}

/**
 * Runs generation orchestration and playback until a stop signal is observed.
 *
 * The loop alternates between two phases:
 * 1. Evolve off-thread until the worker emits the next best-generation summary.
 * 2. Play that generation back on the main thread while streaming HUD updates.
 *
 * This rhythm makes the demo feel like a live training dashboard instead of a
 * one-shot batch job.
 *
 * @param options - Runtime evolution dependencies and mutable state accessors.
 * @returns Nothing.
 */
export async function runRuntimeEvolutionLoop(
  options: RuntimeEvolutionLoopOptions,
): Promise<void> {
  const {
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
  } = options;

  // Step 1: Track best-so-far metrics across all completed generations.
  let bestRunFrames = 0;
  let bestRunPipes = 0;

  // Step 2: Initialize worker runtime with deterministic seed + config.
  evolutionWorker.postMessage({
    type: 'init',
    payload: {
      populationSize,
      elitismCount,
      rngSeed: FLAPPY_DEFAULT_RNG_SEED,
    },
  });

  // Step 3: Keep iterating generations until a stop signal is observed.
  while (!isStopped()) {
    // Step 3.1: Request and await the next evolved generation payload.
    const generationPayload = await requestWorkerGeneration(evolutionWorker);
    if (isStopped()) {
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
      ...resolveInitialRuntimeTelemetryHudValues(),
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

        // Step 3.5.2: Publish current frame counters + telemetry values.
        updateStatsTableValues(statsValueByKey, {
          birds: `${frameStats.activeBirdCount}/${populationSize}`,
          currentFrames: String(frameStats.frameIndex),
          currentPipes: String(frameStats.leaderPipesPassed),
          currentMaxFrames: String(frameStats.leaderFramesSurvived),
          currentMaxPipes: String(frameStats.leaderPipesPassed),
          ...resolveRuntimeTelemetryHudValues(
            frameStats,
            runtimeTelemetryState,
          ),
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
