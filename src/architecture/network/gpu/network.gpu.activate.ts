import type Network from '../network';
import { canUseGPU } from './network.gpu.capability';
import { SUPPORTED_ACTIVATION_INDICES } from './network.gpu.kernel';

/**
 * WebGPU single-network activation seam (mock-aware placeholder).
 *
 * Performs a minimal eligibility check and then returns a zeroed output buffer
 * with the same shape as the CPU activation path. The zeroed placeholder is
 * intentionally wrong so the red parity tests fail at tolerance assertions
 * rather than at "not implemented".
 *
 * @param device - WebGPU device used to run the forward kernel.
 * @param network - Network whose fast-slab topology has been uploaded.
 * @param inputs - Input vector of length `network.input`.
 * @returns A promise resolving to a zeroed Float32Array of CPU output length.
 * @throws Error when the network is ineligible for GPU inference.
 */
export async function activateGPU(
  device: GPUDevice,
  network: Network,
  inputs: Float32Array | number[],
): Promise<Float32Array> {
  void inputs;

  const supportedActivations = new Set<number>(
    SUPPORTED_ACTIVATION_INDICES as unknown as number[],
  );

  if (!canUseGPU(network, device, supportedActivations)) {
    throw new Error(
      'activateGPU: network is not eligible for GPU inference (gated topology, self-connection, or missing device)',
    );
  }

  const outputLength = network.output;
  return new Float32Array(outputLength);
}
