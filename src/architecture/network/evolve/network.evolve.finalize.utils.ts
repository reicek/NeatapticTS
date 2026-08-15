import Network from '../../network/network';
import type { EvolveOptions, NeatRuntime } from '../network.types';
import type { EvolutionSummary } from './network.evolve.utils.types';

/**
 * Finalization helper that adopts the best evolved genome into the caller's
 * network when one exists.
 *
 * If no best genome is available, the optional NEAT warning hook is invoked so
 * callers can surface diagnostic context without throwing from finalize flow.
 *
 * @param network - Network instance to update with the best genome.
 * @param neatInstance - NEAT runtime with optional warning hook.
 * @param bestGenome - Best evolved genome or undefined when none exists.
 * @param clearState - Whether to clear network state after adoption.
 */
export function adoptBestGenomeOrWarn(
  network: Network,
  neatInstance: NeatRuntime,
  bestGenome: Network | undefined,
  clearState: boolean,
): void {
  if (bestGenome) {
    network.nodes = bestGenome.nodes;
    network.connections = bestGenome.connections;
    network.selfconns = bestGenome.selfconns;
    network.gates = bestGenome.gates;
    network.refreshExplicitIORoles();
    if (clearState) network.clear();
    return;
  }

  if (!neatInstance._warnIfNoBestGenome) return;
  try {
    neatInstance._warnIfNoBestGenome();
  } catch {
    // Ignore warning errors
  }
}

/**
 * Best-effort shutdown for worker terminators attached to evolve options.
 *
 * This keeps finalize paths resilient when background evaluators were used and
 * avoids leaking worker resources if teardown throws.
 *
 * @param evolveOptions - Evolve options object.
 * @returns Nothing.
 */
export function terminateWorkersSafely(evolveOptions: EvolveOptions): void {
  try {
    evolveOptions._workerTerminators?.();
  } catch {
    // Ignore termination errors
  }
}

/**
 * Build the final evolution summary payload returned by the evolve loop.
 *
 * @param error - Final loop error.
 * @param iterations - Final generation count.
 * @param loopStartTime - Loop start timestamp.
 * @returns Evolution summary object.
 */
export function buildEvolutionSummary(
  error: number,
  iterations: number,
  loopStartTime: number,
): EvolutionSummary {
  return { error, iterations, time: Date.now() - loopStartTime };
}
