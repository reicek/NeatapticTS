/**
 * GPU-aware controller integration for the racing-curriculum simulation worker.
 *
 * This module provides the worker-side decision helper that chooses when to
 * pass the `useGPU` hint to {@link Network.activate}. It mirrors the Phase 2
 * crossover threshold findings: in the racing-browser demo the GPU dispatch
 * overhead only pays off once the batch is large enough.
 *
 * The module does not import the WebGPU API directly; it relies on the
 * existing GPU seam in `src/architecture/network/gpu/` and on the
 * `Network.activate` opt-in flag. This keeps the worker boundary thin and
 * avoids duplicating device-management logic.
 */

import type { Network } from '../../../../src/browser-entry.ts';
import { SUPPORTED_ACTIVATION_INDICES } from '../../../../src/architecture/network/gpu/network.gpu.kernel.ts';

/**
 * Phase 2 crossover threshold for the racing-browser worker.
 *
 * Below this agent count the per-car CPU path is cheaper because the fixed
 * WebGPU dispatch and readback overhead dominates. At or above the threshold
 * the parallel GPU path begins to amortize that overhead.
 *
 * @see plans/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md — Resolved questions
 */
export const RACING_BROWSER_GPU_THRESHOLD = 130;

/** Activation indices the current GPU kernel supports. */
const SUPPORTED_ACTIVATIONS = new Set<number>(
  SUPPORTED_ACTIVATION_INDICES as unknown as number[],
);

/**
 * Decide whether a racing generation batch should opt into the GPU path.
 *
 * The decision uses the Phase 2 racing-browser crossover threshold and a
 * lightweight structural eligibility check on the representative network.
 * Device readiness is intentionally checked at activation time by
 * {@link Network.activate} so this predicate can be used in worker planning
 * without requiring a live WebGPU device.
 *
 * @param agentCount - Number of cars / networks in the batch.
 * @param network - Representative network from the batch.
 * @returns True when the batch is large enough and the network is
 *   structurally GPU-compatible.
 */
export function shouldUseGPUForBatch(
  agentCount: number,
  network: Network,
): boolean {
  if (agentCount < RACING_BROWSER_GPU_THRESHOLD) {
    return false;
  }

  return isNetworkStructurallyGPUEligible(network);
}

/**
 * Check structural GPU eligibility without requiring a live WebGPU device.
 *
 * Mirrors the device-independent portion of the eligibility checks used by
 * the batched GPU seam so worker-side batch planning can decide before a
 * device is bound.
 *
 * @param network - Network to inspect.
 * @returns True when the network has no gating, self-connections, or
 *   unsupported activations.
 */
function isNetworkStructurallyGPUEligible(network: Network): boolean {
  if (network.gates.length > 0) {
    return false;
  }

  if (network.selfconns.length > 0) {
    return false;
  }

  const hasUnsupportedActivation = network.nodes.some((node) => {
    const index = (node.squash as { index?: number }).index;
    return typeof index === 'number' && !SUPPORTED_ACTIVATIONS.has(index);
  });
  if (hasUnsupportedActivation) {
    return false;
  }

  return true;
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
