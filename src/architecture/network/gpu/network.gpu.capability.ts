import type Network from '../network';
import { isDeviceReady } from './network.gpu.device';

/**
 * Minimum GPU eligibility predicate.
 *
 * Decides whether a network is structurally eligible for the GPU inference
 * path without requiring a real WebGPU backend. It rejects missing devices,
 * networks with gating connections, networks that contain self-connections,
 * and nodes whose activation index is both present and unsupported. Buffer
 * sizing is only coarsely estimated so the predicate can run against mock
 * devices as well as real hardware.
 *
 * This predicate is one input to `isGPUEligible`, which also checks
 * device readiness and the float32 slab flag. Most callers should use
 * `isGPUEligible` rather than calling `canUseGPU` directly.
 *
 * @param network - Network to evaluate for GPU inference.
 * @param device - WebGPU device, or null when WebGPU is unavailable.
 * @param supportedActivations - Worker-registry activation indices the GPU
 *   kernel supports. Nodes without an explicit index are skipped so networks
 *   built from high-level constructors can still be evaluated; nodes without a
 *   squash function are also skipped so `activateGPU` can report the
 *   missing-squash error with its own message.
 * @returns True when the network is structurally eligible for the GPU path.
 *
 * @example
 * ```ts
 * const supported = new Set<number>([0, 1, 2, 3]);
 * const eligible = canUseGPU(network, device, supported);
 * ```
 */
export function canUseGPU(
  network: Network,
  device: GPUDevice | null,
  supportedActivations: ReadonlySet<number>,
): boolean {
  if (!isDeviceReady(device)) {
    return false;
  }

  if (network.gates.length > 0) {
    return false;
  }

  if (network.selfconns.length > 0) {
    return false;
  }

  const hasSelfConnection = network.connections.some(
    (connection) => connection.from === connection.to,
  );
  if (hasSelfConnection) {
    return false;
  }

  const hasUnsupportedActivation = network.nodes.some((node) => {
    if (!node.squash) {
      return false;
    }
    const index = (node.squash as { index?: number }).index;
    return typeof index === 'number' && !supportedActivations.has(index);
  });
  if (hasUnsupportedActivation) {
    return false;
  }

  const estimatedBytes = Math.max(
    network.nodes.length * 4,
    network.connections.length * 4,
  );
  const limit = device!.limits.maxStorageBufferBindingSize!;
  if (estimatedBytes > limit) {
    return false;
  }

  return true;
}
