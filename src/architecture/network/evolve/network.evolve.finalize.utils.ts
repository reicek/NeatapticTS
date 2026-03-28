import Network from '../../network/network';
import type { EvolveOptions, NeatRuntime } from '../network.types';
import type { EvolutionSummary } from './network.evolve.utils.types';

/**
 * Adopt best genome structure or emit warning when unavailable.
 *
 * @param network - Network instance being evolved.
 * @param neatInstance - Active NEAT instance.
 * @param bestGenome - Best genome snapshot.
 * @param clearState - Whether to clear network after adoption.
 * @returns Nothing.
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
 * Terminate worker resources registered in options.
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
 * Build final evolve return payload.
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
