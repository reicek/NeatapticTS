import Network from '../../../../src/architecture/network';
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
import {
  presentRuntimeGenerationPreview,
  startRuntimeStartupPreview,
} from './runtime.startup-preview.service';
import type {
  FlappyStatsTableCells,
  NetworkVisualizationHandle,
} from '../browser-entry.types';
import type { WorkerChannelGenerationPayload } from '../worker-channel/worker-channel.types';
import type { RuntimeStartupPreviewHandle } from './runtime.types';
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
  let shouldShowStartupPreview = true;

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
    const generationPayload = await requestGenerationWithOptionalStartupPreview(
      {
        evolutionWorker,
        canvas,
        context,
        isStopped,
        showStartupPreview: shouldShowStartupPreview,
      },
    );
    if (isStopped()) {
      break;
    }
    shouldShowStartupPreview = false;

    // Step 3.2: Resolve generation-level values and best-network visualization model.
    const bestFitness = generationPayload.bestFitness;
    const bestNetwork = generationPayload.bestNetworkJson
      ? Network.fromJSON(generationPayload.bestNetworkJson)
      : undefined;
    const generationPopulationNetworks = resolveGenerationPopulationNetworks(
      generationPayload,
      bestNetwork,
    );
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
      birds: `${FLAPPY_HUD_ZERO_TEXT}/${populationSize}`,
    });

    // Step 3.4: Render active network architecture in the side panel.
    renderNetworkArchitecture(bestNetwork, inputSize, outputSize);

    // Step 3.5: Present the ready generation on the main canvas before playback begins.
    await presentRuntimeGenerationPreview({
      canvas,
      context,
      isStopped,
      legendText: resolveGenerationPresentationLegendText(
        generationPayload.generation,
      ),
    });
    if (isStopped()) {
      break;
    }

    // Step 3.6: Switch the HUD into playing state once the title card has cleared.
    updateStatsTableValues(statsValueByKey, {
      status: FLAPPY_STATUS_PLAYING_TEXT,
    });

    // Step 3.7: Run playback and stream per-frame HUD updates.
    const playbackSummary = await animatePopulationEpisode(
      canvas,
      context,
      evolutionWorker,
      (frameStats) => {
        // Step 3.7.1: Throttle HUD writes to reduce layout/repaint churn.
        if (frameStats.frameIndex % FLAPPY_HUD_UPDATE_INTERVAL_FRAMES !== 0) {
          return;
        }

        // Step 3.7.2: Publish current frame counters + telemetry values.
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
      ({ championBirdIndex }) => {
        // Step 3.7.3: Redraw the side panel from the current champion network.
        const championNetwork =
          generationPopulationNetworks[championBirdIndex] ?? bestNetwork;
        if (!championNetwork) {
          return;
        }

        updateStatsTableValues(statsValueByKey, {
          currentArchitecture: resolveNetworkArchitectureLabel(
            championNetwork,
            inputSize,
            outputSize,
          ),
        });
        renderNetworkArchitecture(championNetwork, inputSize, outputSize);
      },
    );

    // Step 3.8: Fold generation winner into cross-generation maxima.
    bestRunFrames = Math.max(
      bestRunFrames,
      playbackSummary.winnerFramesSurvived,
    );
    bestRunPipes = Math.max(bestRunPipes, playbackSummary.winnerPipesPassed);

    // Step 3.9: Finalize generation HUD summary and switch status back to evolving.
    updateStatsTableValues(statsValueByKey, {
      bestFrames: String(playbackSummary.winnerFramesSurvived),
      bestPipes: String(playbackSummary.winnerPipesPassed),
      bestMaxFrames: String(bestRunFrames),
      bestMaxPipes: String(bestRunPipes),
      status: FLAPPY_STATUS_EVOLVING_TEXT,
      birds: `${FLAPPY_HUD_ZERO_TEXT}/${populationSize}`,
    });

    // Step 3.10: Emit compact generation summary to console (best-effort only).
    try {
      console.log(
        `[flappy_bird] gen=${generationPayload.generation} fitness=${bestFitness.toFixed(0)} winnerPipes=${playbackSummary.winnerPipesPassed} winnerFrames=${playbackSummary.winnerFramesSurvived} avgPipes=${playbackSummary.averagePipesPassed.toFixed(2)} p90Frames=${playbackSummary.p90FramesSurvived}`,
      );
    } catch {
      // ignore console write issues
    }
  }
}

/**
 * Requests the next generation and optionally shows the first-load startup preview.
 *
 * The browser should only show the animated loading preview while the very
 * first generation is still booting. Later generations can reuse the same
 * canvas style through the separate generation-presentation path.
 *
 * @param options - Generation request inputs and preview-gating state.
 * @returns The next generation payload from the worker.
 */
async function requestGenerationWithOptionalStartupPreview(options: {
  evolutionWorker: Worker;
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  isStopped: () => boolean;
  showStartupPreview: boolean;
}): Promise<WorkerChannelGenerationPayload> {
  // Step 1: Start the animated loading preview only for the first generation.
  const startupPreviewHandle = options.showStartupPreview
    ? startRuntimeStartupPreview({
        canvas: options.canvas,
        context: options.context,
        isStopped: options.isStopped,
      })
    : undefined;

  try {
    // Step 2: Await the next worker generation while the preview animates.
    const generationPayload = await requestWorkerGeneration(
      options.evolutionWorker,
    );
    if (options.isStopped()) {
      startupPreviewHandle?.stop();
      return generationPayload;
    }

    // Step 3: Fade the startup preview out before birds are allowed to appear.
    await finalizeStartupPreview(startupPreviewHandle);
    return generationPayload;
  } catch (error) {
    // Step 4: Stop the preview immediately if generation boot fails.
    startupPreviewHandle?.stop();
    throw error;
  }
}

/**
 * Completes and tears down the startup preview when one is active.
 *
 * @param startupPreviewHandle - Optional preview handle created for first-load boot.
 * @returns Nothing.
 */
async function finalizeStartupPreview(
  startupPreviewHandle: RuntimeStartupPreviewHandle | undefined,
): Promise<void> {
  // Step 1: Skip teardown when no startup preview is active.
  if (!startupPreviewHandle) {
    return;
  }

  // Step 2: Wait for the fade-out to finish, then stop the preview loop.
  try {
    await startupPreviewHandle.complete();
  } finally {
    startupPreviewHandle.stop();
  }
}

/**
 * Resolves the browser-side network cache for the current playback generation.
 *
 * Playback birds are created from the generation population in stable array
 * order, so the browser can reuse this ordered cache to redraw the network
 * panel when the red-bird champion changes.
 *
 * @param generationPayload - Worker generation-ready payload.
 * @param bestNetwork - Current generation best-network fallback.
 * @returns Ordered population networks for the upcoming playback session.
 */
function resolveGenerationPopulationNetworks(
  generationPayload: {
    populationNetworksJson?: Array<Record<string, unknown>>;
  },
  bestNetwork: Network | undefined,
): Network[] {
  // Step 1: Prefer the full serialized population when the worker provides it.
  if (
    Array.isArray(generationPayload.populationNetworksJson) &&
    generationPayload.populationNetworksJson.length > 0
  ) {
    return generationPayload.populationNetworksJson.map((populationNetwork) =>
      Network.fromJSON(populationNetwork),
    );
  }

  // Step 2: Fall back to the best network when no population payload exists.
  return bestNetwork ? [bestNetwork] : [];
}

/**
 * Resolves the centered legend text used to present one ready generation.
 *
 * @param generation - Ready generation number from the worker payload.
 * @returns Generation presentation legend text.
 */
function resolveGenerationPresentationLegendText(generation: number): string {
  return `GENERATION ${generation}`;
}
