import type {
  ResolvePlaybackStepRequestInput,
  ResolvePlaybackStepRequestResult,
} from './playback.worker-channel.types';

/**
 * Playback batch-request helpers for the browser worker channel.
 *
 * The playback loop accumulates simulation budget in fractional units, then
 * converts that budget into integer worker step requests on each render tick.
 */

/**
 * Resolves step count and request payload for the next worker playback batch.
 *
 * This is the pacing bridge between browser rendering and worker simulation.
 * Rather than sending a fixed step count every frame, the loop carries forward
 * fractional remainder so long-term playback speed stays closer to the intended
 * emulation rate.
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
