import type Network from '../network';
import { canUseGPU } from './network.gpu.capability';
import { activateGPU } from './network.gpu.activate';
import { SUPPORTED_ACTIVATION_INDICES } from './network.gpu.kernel';

/**
 * Activation indices that the current placeholder GPU kernel understands.
 *
 * Mirrors the WGSL switch in {@link createActivationKernel} so the eligibility
 * predicate and the GPU seam agree on what is dispatchable.
 */
const SUPPORTED_ACTIVATIONS = new Set<number>(
  SUPPORTED_ACTIVATION_INDICES as unknown as number[],
);

/**
 * Check whether a supplied GPU device is present and has not been lost.
 *
 * Real WebGPU devices do not expose a synchronous lost flag; the mock device
 * used in owner-local tests carries a private `__lost` marker so the fallback
 * seam can react synchronously in test fixtures. In production the device
 * wrapper that calls this seam is expected to track loss before invoking the
 * library path.
 *
 * @param device - Device to inspect, or null/undefined when WebGPU is absent.
 * @returns True when the device is present and not marked lost.
 */
function isDeviceUsable(
  device: GPUDevice | null | undefined,
): device is GPUDevice {
  if (device === null || device === undefined) {
    return false;
  }

  const maybeLost = device as unknown as { __lost?: boolean };
  return maybeLost.__lost !== true;
}

/**
 * Transparent single-network activation seam.
 *
 * Dispatches to the WebGPU fast path when the network and device are eligible,
 * otherwise falls back to the CPU `network.activate()` implementation. Both
 * paths return a `Float32Array` of the same output length so callers do not
 * need to know which path was taken.
 *
 * @param network - Network to activate.
 * @param inputs - Input vector of length `network.input`.
 * @param device - Optional WebGPU device. When null, missing, or lost, the CPU
 *   path is used.
 * @returns Promise resolving to the network output.
 */
export async function dispatchActivation(
  network: Network,
  inputs: Float32Array | number[],
  device?: GPUDevice | null,
): Promise<Float32Array> {
  if (
    isDeviceUsable(device) &&
    canUseGPU(network, device, SUPPORTED_ACTIVATIONS)
  ) {
    return activateGPU(device, network, inputs);
  }

  const cpuOutput = network.activate(inputs);
  return new Float32Array(cpuOutput);
}
