/**
 * GPU-aware controller integration for the racing-curriculum simulation worker.
 *
 * This module provides the worker-side decision helper that chooses when to
 * pass the `useGPU` hint to {@link Network.activate}. It mirrors the Phase 2
 * crossover threshold findings: in the racing-browser demo the GPU dispatch
 * overhead only pays off once the batch is large enough.
 *
 * The module does not import the WebGPU API directly; it delegates the
 * threshold decision to the generic acceleration layer via
 * {@link shouldAutoEnableGpu}. This keeps the worker boundary thin and
 * avoids duplicating device-management or eligibility logic.
 */

import type { Network } from '../../../../src/browser-entry.ts';
import { shouldAutoEnableGpu } from '../../../../src/acceleration/index.ts';

/**
 * Decide whether a racing generation batch should opt into the GPU path.
 *
 * The decision is delegated to the generic acceleration layer's
 * {@link shouldAutoEnableGpu} helper, which compares the network size and
 * batch parallelism against the library-wide GPU node and batch thresholds.
 * Any structural incompatibilities are handled by {@link Network.activate}'s
 * internal fallback at activation time.
 *
 * @param agentCount - Number of cars / networks in the batch.
 * @param network - Representative network from the batch.
 * @returns True when the generic acceleration policy recommends GPU for this
 *   network and batch size.
 */
export function shouldUseGPUForBatch(
  agentCount: number,
  network: Network,
): boolean {
  return shouldAutoEnableGpu(network.nodes.length, agentCount);
}

/**
 * Synchronous controller handle returned by
 * {@link createGPUAwareRaceController}.
 */
export type GPUAwareRaceController = {
  /** Runs inference, passing the GPU hint when the batch is eligible. */
  activate(inputs: number[]): number[];
};

/**
 * Create a race controller that passes the GPU hint to {@link Network.activate}
 * when the generation is large enough to justify GPU dispatch.
 *
 * The controller is synchronous because the racing tick loop is synchronous.
 * When the GPU hint is passed but the network is not actually GPU-ready
 * (missing device, lost device, ineligible structure), {@link Network.activate}
 * falls back to the CPU path and returns a plain number array.
 *
 * @param network - Network that will drive one car.
 * @param agentCount - Total number of cars / networks in the batch. Used to
 *   apply the Phase 2 crossover threshold.
 * @returns A controller handle that internally decides whether to pass
 *   `useGPU: true`.
 */
export function createGPUAwareRaceController(
  network: Network,
  agentCount: number,
): GPUAwareRaceController {
  return {
    activate(inputs: number[]): number[] {
      const useGPU = shouldUseGPUForBatch(agentCount, network);

      const output = useGPU
        ? network.activate(inputs, { useGPU: true })
        : network.activate(inputs, { useGPU: false });
      if (output instanceof Promise) {
        throw new Error(
          'Unexpected async GPU activation: racing worker tick loop is synchronous. ' +
            'Ensure the network has a bound, eligible WebGPU device before enabling GPU.',
        );
      }
      return output;
    },
  };
}
