/**
 * Process-signal bridge for cooperative trainer shutdown.
 *
 * The trainer should stop between generations, not by tearing the process down
 * in the middle of evaluation. This file converts OS-level stop signals into one
 * shared runtime intent flag that the main loop can observe safely.
 */
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
 * The handler does the minimum possible work because signal paths should stay
 * predictable and side-effect light.
 *
 * @param trainerRuntimeState - Mutable trainer runtime state.
 * @returns Nothing.
 */
function handleTrainerStopSignal(
  trainerRuntimeState: FlappyTrainerRuntimeState,
): void {
  trainerRuntimeState.shouldStop = true;
}
