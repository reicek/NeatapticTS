import type { FlappyTrainerRuntimeState } from './trainer.types';

/**
 * Registers graceful stop signal handlers.
 *
 * Educational note:
 * Long-running evolutionary runs should stop cleanly when the user presses
 * `Ctrl+C`. This service flips runtime intent instead of abruptly tearing down
 * the process mid-generation.
 *
 * @param trainerRuntimeState - Mutable trainer runtime state.
 * @returns Nothing.
 */
export function registerTrainerStopSignals(
  trainerRuntimeState: FlappyTrainerRuntimeState,
): void {
  process.on('SIGINT', function onSigInt(): void {
    handleTrainerStopSignal(trainerRuntimeState);
  });

  process.on('SIGTERM', function onSigTerm(): void {
    handleTrainerStopSignal(trainerRuntimeState);
  });
}

/**
 * Handles one stop signal update.
 *
 * @returns Nothing.
 *
 * @param trainerRuntimeState - Mutable trainer runtime state.
 */
function handleTrainerStopSignal(
  trainerRuntimeState: FlappyTrainerRuntimeState,
): void {
  trainerRuntimeState.shouldStop = true;
}
