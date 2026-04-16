import { createCanvasHost, updateStatsTableValues } from '../host/host';
import { createEvolutionWorker } from '../worker-channel/worker-channel';
import {
  FLAPPY_BROWSER_ELITISM_COUNT,
  FLAPPY_BROWSER_POPULATION_SIZE,
  FLAPPY_HUD_INITIALIZING_TEXT,
  FLAPPY_HUD_ZERO_TEXT,
} from '../../constants/constants';
import {
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from '../../constants/constants';
import { resolveRequiredRuntimeHostElement } from './runtime.errors';
import {
  resolveAvailableRuntimeArchitectureProfiles,
  resolveRuntimeArchitectureHistory,
  resolveRuntimeArchitectureSelectorItems,
  resolveSelectedRuntimeArchitectureProfile,
} from './runtime.architecture-profile.service';
import { createRuntimeTelemetryState } from './runtime.telemetry.service';
import type {
  RuntimeContainerTarget,
  RuntimeStartConfig,
  RuntimeStartContext,
  RuntimeStartOptions,
} from './runtime.types';
import type { ExampleArchitectureProfileId } from '../../../architectureProfiles';

const FLAPPY_BROWSER_SPARSE_POPULATION_SIZE = 30;
const FLAPPY_BROWSER_SPARSE_ELITISM_COUNT = 6;
const FLAPPY_BROWSER_NARX_POPULATION_SIZE = 40;
const FLAPPY_BROWSER_NARX_ELITISM_COUNT = 8;
const FLAPPY_BROWSER_GRU_POPULATION_SIZE = 18;
const FLAPPY_BROWSER_GRU_ELITISM_COUNT = 4;
const FLAPPY_BROWSER_LSTM_POPULATION_SIZE = 6;
const FLAPPY_BROWSER_LSTM_ELITISM_COUNT = 1;

/**
 * Runtime startup helpers for the Flappy Bird browser demo.
 *
 * These functions cover the pre-loop phase: resolve the host container, build a
 * typed browser view, derive static config, create telemetry state, spawn the
 * worker, and paint the initial HUD before evolution begins.
 */

/**
 * Creates the shared runtime startup dependencies used by the entry orchestration.
 *
 * This is the main bootstrap fold for browser startup. After it returns, the
 * runtime has a host view, a worker channel, telemetry state, and a resolved
 * configuration object.
 *
 * @param container - Element id or HTMLElement to host the demo.
 * @returns Shared runtime start context for setup and loop launch.
 */
export function createRuntimeStartContext(
  container: RuntimeContainerTarget,
  runtimeStartOptions: RuntimeStartOptions,
): RuntimeStartContext {
  // Step 1: Resolve and validate the runtime host element.
  const hostElement = resolveRequiredRuntimeHostElement(container);

  // Step 2: Construct the static runtime configuration values.
  const config = createRuntimeStartConfig(runtimeStartOptions);

  // Step 3: Create the browser host, telemetry state, and worker channel.
  return {
    config,
    hostElement,
    viewContext: createCanvasHost(hostElement, {
      architectureSelectorItems: resolveRuntimeArchitectureSelectorItems({
        availableProfiles: config.availableArchitectureProfiles,
        selectedProfileId: config.selectedArchitectureProfile.id,
        historyByProfileId: config.architectureHistoryByProfileId,
      }),
      onSelectArchitectureProfile: runtimeStartOptions.onSelectArchitectureProfile,
    }),
    runtimeTelemetryState: createRuntimeTelemetryState(),
    evolutionWorker: createEvolutionWorker(),
  };
}

/**
 * Paints the initial runtime HUD values before the evolution loop starts.
 *
 * The HUD is seeded immediately so the page communicates that startup is in
 * progress rather than appearing blank while the worker and loop are booting.
 *
 * @param runtimeStartContext - Shared runtime start context.
 * @returns Nothing.
 */
export function initializeRuntimeHud(
  runtimeStartContext: RuntimeStartContext,
): void {
  // Step 1: Publish the initializing state and empty bird counters.
  updateStatsTableValues(runtimeStartContext.viewContext.statsValueByKey, {
    currentArchitecture:
      runtimeStartContext.config.selectedArchitectureProfile.label,
    summaryArchitecture:
      runtimeStartContext.config.selectedArchitectureProfile.label,
    status: FLAPPY_HUD_INITIALIZING_TEXT,
    birds: `${FLAPPY_HUD_ZERO_TEXT}/${runtimeStartContext.config.populationSize}`,
  });
}

/**
 * Resolves the static runtime configuration used during browser startup.
 *
 * Centralizing the configuration fold here makes the runtime entry read as
 * orchestration instead of constant plumbing.
 *
 * @returns Runtime configuration derived from shared constants.
 */
function createRuntimeStartConfig(
  runtimeStartOptions: RuntimeStartOptions,
): RuntimeStartConfig {
  const selectedArchitectureProfile = resolveSelectedRuntimeArchitectureProfile(
    runtimeStartOptions.architectureProfileId,
  );
  const runtimeBudget = resolveRuntimePopulationBudget(
    selectedArchitectureProfile.id,
  );

  // Step 1: Fold shared runtime constants into one descriptive config object.
  return {
    architectureHistoryByProfileId: resolveRuntimeArchitectureHistory(),
    availableArchitectureProfiles: resolveAvailableRuntimeArchitectureProfiles(),
    inputSize: FLAPPY_NETWORK_INPUT_SIZE,
    outputSize: FLAPPY_NETWORK_OUTPUT_SIZE,
    populationSize: runtimeBudget.populationSize,
    elitismCount: runtimeBudget.elitismCount,
    selectedArchitectureProfile,
  };
}

/**
 * Resolves the browser evolution budget for one architecture profile.
 *
 * Sparse and NARX keep wider browser budgets than the dense MLP baseline so
 * the interactive demo still has room to discover pipe-clearing behavior in a
 * small number of generations. GRU and LSTM now stay materially smaller than
 * NARX because their gated recurrent blocks still cause visible main-thread
 * stutter at broader browser flock sizes.
 *
 * @param architectureProfileId - Selected shared Flappy profile id.
 * @returns Browser-local population and elitism settings.
 */
function resolveRuntimePopulationBudget(
  architectureProfileId: ExampleArchitectureProfileId,
): Pick<RuntimeStartConfig, 'populationSize' | 'elitismCount'> {
  // Step 1: Widen lighter Sparse runs a bit because they stay comparatively cheap.
  if (architectureProfileId === 'random-sparse') {
    return {
      populationSize: FLAPPY_BROWSER_SPARSE_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_SPARSE_ELITISM_COUNT,
    };
  }

  // Step 2: Keep NARX broader than the baseline while trimming its browser cost a bit.
  if (architectureProfileId === 'narx') {
    return {
      populationSize: FLAPPY_BROWSER_NARX_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_NARX_ELITISM_COUNT,
    };
  }

  // Step 3: Keep GRU meaningfully above the MLP baseline without reintroducing visible stutter.
  if (architectureProfileId === 'gru') {
    return {
      populationSize: FLAPPY_BROWSER_GRU_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_GRU_ELITISM_COUNT,
    };
  }

  // Step 4: Cap LSTM near the baseline because the heavier recurrent shelf still stutters first.
  if (architectureProfileId === 'lstm') {
    return {
      populationSize: FLAPPY_BROWSER_LSTM_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_LSTM_ELITISM_COUNT,
    };
  }

  // Step 5: Keep the shared lightweight browser baseline for MLP.
  return {
    populationSize: FLAPPY_BROWSER_POPULATION_SIZE,
    elitismCount: FLAPPY_BROWSER_ELITISM_COUNT,
  };
}
