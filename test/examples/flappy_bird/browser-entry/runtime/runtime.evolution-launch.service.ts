import { updateStatsTableValues } from '../host/host';
import { resolveRuntimeHudErrorStatus } from './runtime.errors';
import { runRuntimeEvolutionLoop } from './runtime.evolution-loop.service';
import type {
  RuntimeMutableLifecycleState,
  RuntimeStartContext,
} from './runtime.types';

/**
 * Starts the runtime evolution loop and routes unexpected failures to the HUD.
 *
 * @param runtimeStartContext - Shared runtime start context.
 * @param runtimeLifecycleState - Mutable lifecycle state used for stop checks.
 * @param stop - Idempotent stop function bound to the current runtime handle.
 * @returns Nothing.
 */
export function launchRuntimeEvolution(
  runtimeStartContext: RuntimeStartContext,
  runtimeLifecycleState: RuntimeMutableLifecycleState,
  stop: () => void,
): void {
  const { config, evolutionWorker, runtimeTelemetryState, viewContext } =
    runtimeStartContext;

  // Step 1: Start the long-running evolution orchestration loop.
  void runRuntimeEvolutionLoop({
    evolutionWorker,
    canvas: viewContext.canvas,
    context: viewContext.context,
    statsValueByKey: viewContext.statsValueByKey,
    renderNetworkArchitecture: viewContext.renderNetworkArchitecture,
    populationSize: config.populationSize,
    elitismCount: config.elitismCount,
    inputSize: config.inputSize,
    outputSize: config.outputSize,
    runtimeTelemetryState,
    isStopped: () => runtimeLifecycleState.stopped,
  }).catch((error: unknown) => {
    // Step 2: Surface unexpected failures in the HUD and tear down the runtime.
    updateStatsTableValues(viewContext.statsValueByKey, {
      status: resolveRuntimeHudErrorStatus(error),
    });
    stop();
  });
}
