import type { FlappyTrainerRuntimeState } from './trainer.types';

/**
 * Registers graceful stop signal handlers.
 *
 * @param trainerRuntimeState - Mutable trainer runtime state.
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
 * @param trainerRuntimeState - Mutable trainer runtime state.
 */
function handleTrainerStopSignal(
  trainerRuntimeState: FlappyTrainerRuntimeState,
): void {
  trainerRuntimeState.shouldStop = true;
}
