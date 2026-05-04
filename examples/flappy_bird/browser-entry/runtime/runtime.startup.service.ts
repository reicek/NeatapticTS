import { createCanvasHost, updateStatsTableValues } from '../host/host';
import { createEvolutionWorker } from '../worker-channel/worker-channel';
import {
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
import { resolveRuntimePopulationBudget } from './runtime.population-budget';
import { createRuntimeTelemetryState } from './runtime.telemetry.service';
import type {
  RuntimeContainerTarget,
  RuntimeStartConfig,
  RuntimeStartContext,
  RuntimeStartOptions,
} from './runtime.types';

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
      onSelectArchitectureProfile:
        runtimeStartOptions.onSelectArchitectureProfile,
      onResetScores: runtimeStartOptions.onResetScores,
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
    availableArchitectureProfiles:
      resolveAvailableRuntimeArchitectureProfiles(),
    inputSize: FLAPPY_NETWORK_INPUT_SIZE,
    outputSize: FLAPPY_NETWORK_OUTPUT_SIZE,
    populationSize: runtimeBudget.populationSize,
    elitismCount: runtimeBudget.elitismCount,
    selectedArchitectureProfile,
  };
}
