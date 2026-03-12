import type {
  ResolvePlaybackStepRequestInput,
  ResolvePlaybackStepRequestResult,
} from './playback.worker-channel.types';

/**
 * Resolves step count and request payload for the next worker playback batch.
 *
 * @param input - Current frame budget and viewport dimensions.
 * @returns Request payload plus carried-over fractional frame budget.
 */
export function resolvePlaybackStepRequest(
  input: ResolvePlaybackStepRequestInput,
): ResolvePlaybackStepRequestResult {
  const updatedSimulationFrameBudget =
    input.simulationFrameBudget + input.emulationSpeedMultiplier;
  const simulationStepsThisRender = Math.max(
    1,
    Math.floor(updatedSimulationFrameBudget),
  );
  const simulationFrameBudgetRemainder =
    updatedSimulationFrameBudget - simulationStepsThisRender;

  return {
    simulationStepsThisRender,
    simulationFrameBudgetRemainder,
    playbackStepRequest: {
      simulationSteps: simulationStepsThisRender,
      visibleWorldWidthPx: input.visibleWorldWidthPx,
      visibleWorldHeightPx: input.visibleWorldHeightPx,
    },
  };
}