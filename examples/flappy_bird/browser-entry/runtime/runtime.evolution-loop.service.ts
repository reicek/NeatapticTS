import Network from '../../../../src/architecture/network';
import { updateStatsTableValues } from '../host/host';
import {
  animatePopulationEpisode,
  type PlaybackEpisodeSummary,
} from '../playback/playback';
import { requestWorkerGeneration } from '../worker-channel/worker-channel';
import {
  FLAPPY_HUD_UPDATE_INTERVAL_FRAMES,
  FLAPPY_HUD_PLACEHOLDER_TEXT,
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
  startRuntimeEvolvingPreview,
  startRuntimeStartupPreview,
} from './runtime.startup-preview.service';
import type { HostArchitectureSelectorController } from '../host/host.types';
import type {
  FlappyStatsTableCells,
  NetworkVisualizationHandle,
} from '../browser-entry.types';
import type { WorkerChannelGenerationPayload } from '../worker-channel/worker-channel.types';
import type { RuntimeStartupPreviewHandle } from './runtime.types';
import type {
  ExampleArchitectureProfile,
  ExampleArchitectureProfileId,
} from '../../../architectureProfiles';
import {
  persistRuntimeArchitectureChampions,
  persistRuntimeArchitectureHistory,
  resolveRuntimeArchitectureSelectorItems,
  updateRuntimeArchitectureHistory,
  type RuntimeArchitectureChampionByProfileId,
  type RuntimeArchitectureHistoryByProfileId,
} from './runtime.architecture-profile.service';
import type { RuntimeTelemetryState } from './runtime.telemetry.service';
import type { SerializedNetwork } from '../browser-entry.worker.types';
import type { WorkerInitMessage } from '../../flappy-evolution-worker/flappy-evolution-worker.types';

const FLAPPY_BROWSER_CHAMPION_LOG_PREFIX = '[flappy-browser]';
const RUNTIME_CHAMPION_FINGERPRINT_OFFSET_BASIS = 0x811c9dc5;
const RUNTIME_CHAMPION_FINGERPRINT_PRIME = 0x01000193;

type RuntimeChampionSummary = {
  connectionCount: number;
  fingerprint: string;
  hiddenNodeCount: number;
  inputNodeCount: number;
  nodeCount: number;
  outputNodeCount: number;
};

/**
 * Long-running evolution/playback orchestration for the browser runtime.
 *
 * This loop is the heart of the interactive demo. It repeatedly asks the worker
 * for the next playable generation payload, updates the HUD and network view,
 * plays back that population on the canvas, then folds the outcome into the
 * generation summary section and the cross-generation history used by the
 * architecture selector.
 */

/**
 * Dependencies required to run the browser runtime evolution loop.
 *
 * Grouping these dependencies into one object keeps the public loop entry more
 * declarative and avoids a long positional parameter list.
 */
export interface RuntimeEvolutionLoopOptions {
  architectureSelectorController: HostArchitectureSelectorController;
  evolutionWorker: Worker;
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  statsValueByKey: FlappyStatsTableCells;
  renderNetworkArchitecture: NetworkVisualizationHandle['renderNetworkArchitecture'];
  availableArchitectureProfiles: ExampleArchitectureProfile[];
  initialArchitectureChampionByProfileId: RuntimeArchitectureChampionByProfileId;
  initialArchitectureHistoryByProfileId: RuntimeArchitectureHistoryByProfileId;
  populationSize: number;
  elitismCount: number;
  inputSize: number;
  outputSize: number;
  selectedArchitectureProfileId: ExampleArchitectureProfileId;
  selectedArchitectureProfileLabel: string;
  runtimeTelemetryState: RuntimeTelemetryState;
  isStopped: () => boolean;
}

/**
 * Runs generation orchestration and playback until a stop signal is observed.
 *
 * The loop alternates between two phases:
 * 1. Wait off-thread until the worker emits the next playable population summary.
 * 2. Play that population back on the main thread while streaming HUD updates.
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
    architectureSelectorController,
    evolutionWorker,
    canvas,
    context,
    statsValueByKey,
    renderNetworkArchitecture,
    availableArchitectureProfiles,
    initialArchitectureChampionByProfileId,
    initialArchitectureHistoryByProfileId,
    populationSize,
    elitismCount,
    inputSize,
    outputSize,
    selectedArchitectureProfileId,
    selectedArchitectureProfileLabel,
    runtimeTelemetryState,
    isStopped,
  } = options;

  // Step 1: Track cross-generation maxima used by architecture-history badges.
  let bestRunFrames = 0;
  let bestRunPipes = 0;
  let architectureChampionByProfileId = initialArchitectureChampionByProfileId;
  let architectureHistoryByProfileId = initialArchitectureHistoryByProfileId;
  let shouldShowStartupPreview = true;
  let expectedGeneration = 1;

  // Step 2: Initialize worker runtime with deterministic seed + config.

  evolutionWorker.postMessage({
    type: 'init',
    payload: resolveWorkerInitPayload({
      architectureProfileId: selectedArchitectureProfileId,
      championByProfileId: architectureChampionByProfileId,
      elitismCount,
      populationSize,
      rngSeed: FLAPPY_DEFAULT_RNG_SEED,
    }),
  });

  // Step 3: Keep iterating generations until a stop signal is observed.
  while (!isStopped()) {
    // Step 3.1: Request and await the next playable generation payload.
    const generationPayload = await requestGenerationWithOptionalStartupPreview(
      {
        evolutionWorker,
        canvas,
        context,
        isStopped,
        showStartupPreview: shouldShowStartupPreview,
        waitLegendText: resolveEvolutionWaitLegendText(expectedGeneration),
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
    const generationPopulationSize = resolveGenerationPopulationSize(
      generationPopulationNetworks,
      populationSize,
    );
    const bestArchitectureLabel = selectedArchitectureProfileLabel;

    // Step 3.3: Hydrate current-generation HUD values before playback begins.
    updateStatsTableValues(statsValueByKey, {
      currentHeader: resolveCurrentRunHeaderText(generationPayload.generation),
      currentFrames: FLAPPY_HUD_ZERO_TEXT,
      currentPipes: FLAPPY_HUD_ZERO_TEXT,
      currentArchitecture: bestArchitectureLabel,
      ...resolveInitialRuntimeTelemetryHudValues(),
      ...resolveGenerationSummaryHudValues({
        architectureLabel: bestArchitectureLabel,
        bestFitness,
      }),
      birds: `${FLAPPY_HUD_ZERO_TEXT}/${generationPopulationSize}`,
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
          birds: `${frameStats.activeBirdCount}/${generationPopulationSize}`,
          currentFrames: String(frameStats.frameIndex),
          currentPipes: String(frameStats.leaderPipesPassed),
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

        renderNetworkArchitecture(championNetwork, inputSize, outputSize);
      },
    );

    // Step 3.8: Fold generation winner into cross-generation maxima.
    bestRunFrames = Math.max(
      bestRunFrames,
      playbackSummary.winnerFramesSurvived,
    );
    bestRunPipes = Math.max(bestRunPipes, playbackSummary.winnerPipesPassed);

    const nextArchitectureProgressUpdate =
      resolveRuntimeArchitectureProgressUpdate({
        candidateBestScore: {
          pipesPassed: bestRunPipes,
          framesSurvived: bestRunFrames,
        },
        candidateChampionNetworkJson: generationPayload.bestNetworkJson,
        championByProfileId: architectureChampionByProfileId,
        historyByProfileId: architectureHistoryByProfileId,
        profileId: selectedArchitectureProfileId,
      });
    if (nextArchitectureProgressUpdate.didImprove) {
      architectureHistoryByProfileId =
        nextArchitectureProgressUpdate.historyByProfileId;
      architectureChampionByProfileId =
        nextArchitectureProgressUpdate.championByProfileId;
      persistRuntimeArchitectureHistory(architectureHistoryByProfileId);
      persistRuntimeArchitectureChampions(architectureChampionByProfileId);
      architectureSelectorController.updateItems(
        resolveRuntimeArchitectureSelectorItems({
          availableProfiles: availableArchitectureProfiles,
          selectedProfileId: selectedArchitectureProfileId,
          historyByProfileId: architectureHistoryByProfileId,
        }),
      );
    }

    const nextExpectedGeneration = generationPayload.generation + 1;

    // Step 3.9: Finalize generation HUD summary and switch status back to evolving.
    updateStatsTableValues(statsValueByKey, {
      currentHeader: resolveCurrentRunHeaderText(nextExpectedGeneration),
      currentFrames: FLAPPY_HUD_ZERO_TEXT,
      currentPipes: FLAPPY_HUD_ZERO_TEXT,
      ...resolveGenerationSummaryHudValues({
        architectureLabel: bestArchitectureLabel,
        bestFitness,
        playbackSummary,
      }),
      status: FLAPPY_STATUS_EVOLVING_TEXT,
      birds: `${FLAPPY_HUD_ZERO_TEXT}/${generationPopulationSize}`,
    });
    expectedGeneration = nextExpectedGeneration;

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
  waitLegendText: string;
}): Promise<WorkerChannelGenerationPayload> {
  // Step 1: Show either the first-load preview or the between-generation evolving overlay.
  const startupPreviewHandle = options.showStartupPreview
    ? startRuntimeStartupPreview({
        canvas: options.canvas,
        context: options.context,
        isStopped: options.isStopped,
      })
    : startRuntimeEvolvingPreview({
        canvas: options.canvas,
        context: options.context,
        isStopped: options.isStopped,
        legendText: options.waitLegendText,
      });

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
 * Resolves which population size the browser HUD should display for the current generation.
 *
 * Browser sessions may start with one population budget and later downshift to a
 * smaller one after a successful run. The worker generation payload already
 * carries the actual serialized population, so the HUD should prefer that real
 * size over the startup budget whenever it is available.
 *
 * @param generationPopulationNetworks - Browser-side cache of the current generation population.
 * @param fallbackPopulationSize - Startup budget used before a generation payload is available.
 * @returns Population size that should be displayed in the HUD for this generation.
 */
export function resolveGenerationPopulationSize(
  generationPopulationNetworks: Network[],
  fallbackPopulationSize: number,
): number {
  return generationPopulationNetworks.length > 0
    ? generationPopulationNetworks.length
    : fallbackPopulationSize;
}

/**
 * Resolves the worker init payload for the selected architecture profile.
 *
 * @param options - Worker startup values plus the browser-local champion table.
 * @returns Worker init payload with an optional champion seed override.
 */
export function resolveWorkerInitPayload(options: {
  architectureProfileId: ExampleArchitectureProfileId;
  championByProfileId: RuntimeArchitectureChampionByProfileId;
  elitismCount: number;
  populationSize: number;
  rngSeed: number;
}): WorkerInitMessage['payload'] {
  const championNetworkJson =
    options.championByProfileId[options.architectureProfileId];

  if (championNetworkJson) {
    logRuntimeChampionLoaded({
      championNetworkJson,
      profileId: options.architectureProfileId,
    });
  }

  return {
    architectureProfileId: options.architectureProfileId,
    ...(championNetworkJson ? { championNetworkJson } : {}),
    elitismCount: options.elitismCount,
    populationSize: options.populationSize,
    rngSeed: options.rngSeed,
  };
}

/**
 * Resolves the next browser-local architecture record state after one playback run.
 *
 * @param options - Current history/champion tables plus the candidate run result.
 * @returns Updated history and champion tables plus an improvement flag.
 */
export function resolveRuntimeArchitectureProgressUpdate(options: {
  candidateBestScore: {
    pipesPassed: number;
    framesSurvived: number;
  };
  candidateChampionNetworkJson?: SerializedNetwork;
  championByProfileId: RuntimeArchitectureChampionByProfileId;
  historyByProfileId: RuntimeArchitectureHistoryByProfileId;
  profileId: ExampleArchitectureProfileId;
}): {
  championByProfileId: RuntimeArchitectureChampionByProfileId;
  didImprove: boolean;
  historyByProfileId: RuntimeArchitectureHistoryByProfileId;
} {
  const previousBestScore = options.historyByProfileId[options.profileId];
  const nextHistoryByProfileId = updateRuntimeArchitectureHistory(
    options.historyByProfileId,
    options.profileId,
    options.candidateBestScore,
  );
  const didImprove = nextHistoryByProfileId !== options.historyByProfileId;

  if (!didImprove || !options.candidateChampionNetworkJson) {
    return {
      championByProfileId: options.championByProfileId,
      didImprove,
      historyByProfileId: nextHistoryByProfileId,
    };
  }

  logRuntimeChampionSaved({
    championNetworkJson: options.candidateChampionNetworkJson,
    currentBestScore: options.candidateBestScore,
    previousBestScore,
    profileId: options.profileId,
  });

  return {
    championByProfileId: {
      ...options.championByProfileId,
      [options.profileId]: options.candidateChampionNetworkJson,
    },
    didImprove: true,
    historyByProfileId: nextHistoryByProfileId,
  };
}

function logRuntimeChampionLoaded(options: {
  championNetworkJson: SerializedNetwork;
  profileId: ExampleArchitectureProfileId;
}): void {
  if (!shouldLogRuntimeChampionPersistence()) {
    return;
  }

  const championSummary = resolveRuntimeChampionSummary(
    options.championNetworkJson,
  );

  console.info(
    `${FLAPPY_BROWSER_CHAMPION_LOG_PREFIX} loading saved champion profile=${options.profileId} fingerprint=${championSummary.fingerprint} nodes=${championSummary.nodeCount} connections=${championSummary.connectionCount} inputs=${championSummary.inputNodeCount} hidden=${championSummary.hiddenNodeCount} outputs=${championSummary.outputNodeCount}`,
  );
}

function logRuntimeChampionSaved(options: {
  championNetworkJson: SerializedNetwork;
  currentBestScore: {
    pipesPassed: number;
    framesSurvived: number;
  };
  previousBestScore:
    | {
        pipesPassed: number;
        framesSurvived: number;
      }
    | undefined;
  profileId: ExampleArchitectureProfileId;
}): void {
  if (!shouldLogRuntimeChampionPersistence()) {
    return;
  }

  const championSummary = resolveRuntimeChampionSummary(
    options.championNetworkJson,
  );
  const previousPipesPassed = options.previousBestScore?.pipesPassed ?? 0;
  const previousFramesSurvived = options.previousBestScore?.framesSurvived ?? 0;

  console.info(
    `${FLAPPY_BROWSER_CHAMPION_LOG_PREFIX} saving champion profile=${options.profileId} pipes=${options.currentBestScore.pipesPassed} frames=${options.currentBestScore.framesSurvived} previousPipes=${previousPipesPassed} previousFrames=${previousFramesSurvived} fingerprint=${championSummary.fingerprint} nodes=${championSummary.nodeCount} connections=${championSummary.connectionCount} inputs=${championSummary.inputNodeCount} hidden=${championSummary.hiddenNodeCount} outputs=${championSummary.outputNodeCount}`,
  );
}

function resolveRuntimeChampionSummary(
  championNetworkJson: SerializedNetwork,
): RuntimeChampionSummary {
  const serializedChampion =
    resolveSerializedChampionString(championNetworkJson);
  const serializedNodes = resolveSerializedChampionNodes(championNetworkJson);
  const connectionCount =
    resolveSerializedChampionConnections(championNetworkJson).length;
  const inputNodeCount = serializedNodes.filter(
    (serializedNode) =>
      resolveSerializedChampionNodeType(serializedNode) === 'input',
  ).length;
  const outputNodeCount = serializedNodes.filter(
    (serializedNode) =>
      resolveSerializedChampionNodeType(serializedNode) === 'output',
  ).length;
  const nodeCount = serializedNodes.length;

  return {
    connectionCount,
    fingerprint: resolveSerializedChampionFingerprint(serializedChampion),
    hiddenNodeCount: Math.max(0, nodeCount - inputNodeCount - outputNodeCount),
    inputNodeCount,
    nodeCount,
    outputNodeCount,
  };
}

function resolveSerializedChampionString(
  championNetworkJson: SerializedNetwork,
): string {
  try {
    return JSON.stringify(championNetworkJson) ?? '';
  } catch {
    return '';
  }
}

function resolveSerializedChampionFingerprint(
  serializedChampion: string,
): string {
  let nextHash = RUNTIME_CHAMPION_FINGERPRINT_OFFSET_BASIS;

  for (
    let characterIndex = 0;
    characterIndex < serializedChampion.length;
    characterIndex += 1
  ) {
    nextHash ^= serializedChampion.charCodeAt(characterIndex);
    nextHash = Math.imul(nextHash, RUNTIME_CHAMPION_FINGERPRINT_PRIME);
  }

  return `0x${(nextHash >>> 0).toString(16).padStart(8, '0')}`;
}

function resolveSerializedChampionNodes(
  championNetworkJson: SerializedNetwork,
): Array<Record<string, unknown>> {
  const serializedNodes = championNetworkJson.nodes;

  return Array.isArray(serializedNodes)
    ? serializedNodes.filter(
        (serializedNode): serializedNode is Record<string, unknown> =>
          typeof serializedNode === 'object' && serializedNode !== null,
      )
    : [];
}

function resolveSerializedChampionConnections(
  championNetworkJson: SerializedNetwork,
): Array<Record<string, unknown>> {
  const serializedConnections = championNetworkJson.connections;

  return Array.isArray(serializedConnections)
    ? serializedConnections.filter(
        (
          serializedConnection,
        ): serializedConnection is Record<string, unknown> =>
          typeof serializedConnection === 'object' &&
          serializedConnection !== null,
      )
    : [];
}

function resolveSerializedChampionNodeType(
  serializedNode: Record<string, unknown>,
): string | undefined {
  return typeof serializedNode.type === 'string'
    ? serializedNode.type
    : undefined;
}

function shouldLogRuntimeChampionPersistence(): boolean {
  return resolveNodeEnvForRuntimeLogs() !== 'test';
}

function resolveNodeEnvForRuntimeLogs(): string | undefined {
  return (globalThis as { process?: { env?: { NODE_ENV?: string } } }).process
    ?.env?.NODE_ENV;
}

/**
 * Resolves the centered legend text shown while the worker evolves the next generation.
 *
 * @param generation - Next generation number expected from the worker.
 * @returns Evolving overlay text.
 */
export function resolveEvolutionWaitLegendText(generation: number): string {
  return `Evolving Gen ${generation}...`;
}

/**
 * Resolves the generation-summary HUD values shown beside the live run counters.
 *
 * The summary intentionally shows metrics the worker already computes for the
 * whole population so the browser can present higher-signal data without doing
 * extra aggregation on the main thread.
 *
 * @param input - Summary source values for the active generation.
 * @returns HUD-ready summary values with placeholders until playback completes.
 */
export function resolveGenerationSummaryHudValues(input: {
  architectureLabel: string;
  bestFitness: number;
  playbackSummary?: PlaybackEpisodeSummary;
}): {
  summaryHeader: string;
  summaryFitness: string;
  summaryWinnerFrames: string;
  summaryWinnerPipes: string;
  summaryAveragePipes: string;
  summaryP90Frames: string;
  summaryArchitecture: string;
} {
  const summary = input.playbackSummary;

  return {
    summaryHeader: 'Generation summary',
    summaryFitness: input.bestFitness.toFixed(0),
    summaryWinnerFrames: summary
      ? String(summary.winnerFramesSurvived)
      : FLAPPY_HUD_PLACEHOLDER_TEXT,
    summaryWinnerPipes: summary
      ? String(summary.winnerPipesPassed)
      : FLAPPY_HUD_PLACEHOLDER_TEXT,
    summaryAveragePipes: summary
      ? summary.averagePipesPassed.toFixed(2)
      : FLAPPY_HUD_PLACEHOLDER_TEXT,
    summaryP90Frames: summary
      ? String(summary.p90FramesSurvived)
      : FLAPPY_HUD_PLACEHOLDER_TEXT,
    summaryArchitecture: input.architectureLabel,
  };
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

function resolveCurrentRunHeaderText(generation: number): string {
  return `Current run · Gen ${generation}`;
}
