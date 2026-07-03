/**
 * Transparent CPU fallback and GPU eligibility predicate.
 *
 * This module decides whether the public `Network.activate(..., { useGPU: true })`
 * overload can safely take the WebGPU fast path, and provides a standalone
 * dispatch helper that falls back to the CPU path automatically when the GPU
 * path is unavailable. Keeping the eligibility decision in one place ensures
 * the public CPU seam and any direct GPU dispatch helpers agree on when the
 * GPU path is safe to use.
 *
 * @see [WebGPU](https://en.wikipedia.org/wiki/WebGPU) on Wikipedia for
 * background on the browser GPU compute API.
 */

import type Network from '../network';
import { canUseGPU } from './network.gpu.capability';
import { activateGPU } from './network.gpu.activate';
import { isDeviceReady } from './network.gpu.device';
import { SUPPORTED_ACTIVATION_INDICES } from './network.gpu.kernel';

/**
 * Activation indices that the current placeholder GPU kernel understands.
 *
 * Mirrors the WGSL switch in `createActivationKernel` so the eligibility
 * predicate and the GPU seam agree on what is dispatchable.
 */
const SUPPORTED_ACTIVATIONS = new Set<number>(
  SUPPORTED_ACTIVATION_INDICES as unknown as number[],
);

/**
 * Shared GPU eligibility predicate used by the single-network fallback seam and
 * by `Network.activate`. A network is eligible only when:
 *
 * - a usable WebGPU device is present and has not been lost,
 * - the network stores slab weights in float32 (the GPU kernel is f32-only),
 * - `canUseGPU` reports the network is structurally eligible.
 *
 * This keeps the fallback decision in one place so the public CPU seam and the
 * standalone dispatch seam agree on when the GPU path is safe to use.
 *
 * @param network - Network to evaluate for GPU inference.
 * @param device - WebGPU device, or null/undefined when WebGPU is unavailable.
 * @returns Type guard that narrows device to GPUDevice when true.
 *
 * @example
 * ```ts
 * const adapter = await navigator.gpu.requestAdapter({
 *   powerPreference: 'high-performance',
 * });
 * const device = await adapter?.requestDevice();
 * if (isGPUEligible(network, device)) {
 *   // device is narrowed to GPUDevice here
 *   const output = await activateGPU(device, network, inputs);
 * }
 * ```
 */
export function isGPUEligible(
  network: Network,
  device: GPUDevice | null | undefined,
): device is GPUDevice {
  if (!device || !network._useFloat32Weights) {
    return false;
  }
  return (
    isDeviceReady(device) && canUseGPU(network, device, SUPPORTED_ACTIVATIONS)
  );
}

/**
 * Transparent single-network activation seam.
 *
 * Dispatches to the WebGPU fast path when the network and device are eligible,
 * otherwise falls back to the CPU `network.activate()` implementation. Both
 * paths return a `Float32Array` of the same output length so callers do not
 * need to know which path was taken.
 *
 * Eligibility is evaluated by `isGPUEligible`: it rejects missing or
 * lost devices, networks with float32 weights disabled, and structurally
 * unsupported networks.
 *
 * @param network - Network to activate.
 * @param inputs - Input vector of length `network.input`.
 * @param device - Optional WebGPU device. When null, missing, lost, or the
 *   network is ineligible, the CPU path is used.
 * @returns Promise resolving to the network output.
 *
 * @example
 * ```ts
 * const adapter = await navigator.gpu.requestAdapter({
 *   powerPreference: 'high-performance',
 * });
 * const device = await adapter?.requestDevice();
 * const output = await dispatchActivation(network, [0.5, -0.2], device);
 * // output is Float32Array from GPU if eligible, otherwise from CPU
 * ```
 */
export async function dispatchActivation(
  network: Network,
  inputs: Float32Array | number[],
  device?: GPUDevice | null,
): Promise<Float32Array> {
  if (isGPUEligible(network, device)) {
    return activateGPU(device, network, inputs);
  }

  const cpuOutput = network.activate(inputs);
  return new Float32Array(cpuOutput);
}
