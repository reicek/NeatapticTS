import type Network from '../network';

/**
 * Minimum GPU eligibility predicate for the red-test seam.
 *
 * Decides whether a network is structurally eligible for the GPU inference
 * path without requiring a real WebGPU backend. It rejects missing devices,
 * networks with gating connections, networks that contain self-connections,
 * and nodes whose activation index is both present and unsupported. Buffer
 * sizing is only coarsely estimated so the predicate can run against the mock
 * devices used in owner-local tests.
 *
 * @param network - Network to evaluate for GPU inference.
 * @param device - WebGPU device, or null when WebGPU is unavailable.
 * @param supportedActivations - Worker-registry activation indices the GPU
 *   kernel supports. Nodes without an explicit index are skipped because the
 *   current createMLP fixtures do not yet carry worker-registry indices.
 * @returns True when the network is structurally eligible for the GPU path.
 */
export function canUseGPU(
  network: Network,
  device: GPUDevice | null,
  supportedActivations: ReadonlySet<number>,
): boolean {
  if (device === null) {
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
  const limit = device.limits.maxStorageBufferBindingSize!;
  if (estimatedBytes > limit) {
    return false;
  }

  return true;
}
