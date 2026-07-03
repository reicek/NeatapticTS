import type Network from '../network';
import { canUseGPU } from './network.gpu.capability';

/**
 * WebGPU buffer usage flag: the buffer may be the destination of a copy command.
 *
 * Mirror of the `GPUBufferUsage.COPY_DST` bit so the module compiles before a
 * full WebGPU type package is added.
 */
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;

/**
 * WebGPU buffer usage flag: the buffer may be bound as a storage buffer in a
 * compute shader.
 *
 * Mirror of the `GPUBufferUsage.STORAGE` bit.
 */
const GPU_BUFFER_USAGE_STORAGE = 0x0080;

/**
 * WebGPU buffer usage flag: the buffer may be the source of a copy command.
 *
 * Mirror of the `GPUBufferUsage.COPY_SRC` bit.
 */
const GPU_BUFFER_USAGE_COPY_SRC = 0x0004;

/**
 * Combined usage for every slab buffer uploaded to the GPU: storage for the
 * compute kernel plus copy-destination for `queue.writeBuffer` uploads.
 */
const GPU_BUFFER_USAGE_DEFAULT =
  GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST;

/**
 * Create a WebGPU storage buffer that can receive `queue.writeBuffer` uploads.
 *
 * Every buffer produced by the upload path must be usable as a read-only
 * (or read-write) storage binding and as a copy destination. This helper
 * validates the request against the device's binding and buffer size limits
 * before delegating to `device.createBuffer`.
 *
 * @param device - WebGPU device used to allocate the buffer.
 * @param byteLength - Desired buffer size in bytes. Must be finite and
 *   non-negative.
 * @param label - Debug label attached to the buffer.
 * @param usage - Additional usage flags merged with the mandatory
 *   `STORAGE | COPY_DST` bits. Defaults to no extra flags.
 * @returns A freshly created `GPUBuffer` with the mandatory usage bits set.
 * @throws Error when `byteLength` is invalid or exceeds device limits.
 */
export function createGPUBuffer(
  device: GPUDevice,
  byteLength: number,
  label: string,
  usage: number = 0,
): GPUBuffer {
  if (!Number.isFinite(byteLength) || byteLength < 0) {
    throw new Error(
      `Invalid GPU buffer size for "${label}": ${String(byteLength)}`,
    );
  }

  const maxStorageBufferBindingSize =
    device.limits.maxStorageBufferBindingSize!;
  const maxBufferSize = device.limits.maxBufferSize!;

  if (byteLength > maxStorageBufferBindingSize) {
    throw new Error(
      `Buffer "${label}" size ${byteLength} exceeds maxStorageBufferBindingSize ${String(maxStorageBufferBindingSize)}`,
    );
  }

  if (byteLength > maxBufferSize) {
    throw new Error(
      `Buffer "${label}" size ${byteLength} exceeds maxBufferSize ${String(maxBufferSize)}`,
    );
  }

  const combinedUsage = usage | GPU_BUFFER_USAGE_DEFAULT;

  return device.createBuffer({
    label,
    size: byteLength,
    usage: combinedUsage,
  });
}

/**
 * Raw connection slab used to build GPU-friendly adjacency arrays.
 *
 * The cast is intentional: GPU upload is a consumer of the same private layout
 * that slab activation uses.
 */
interface ConnectionSlab {
  from: Uint32Array;
  to: Uint32Array;
  weights: Float32Array | Float64Array;
  flags: Uint8Array;
}

/**
 * GPU-side buffer handles and metadata produced by uploading a network slab.
 *
 * The implementation creates one WebGPU buffer per slab/activation array
 * and records `nodeCount`/`connectionCount` so the compute pipeline can size its
 * dispatches without re-reading CPU structures.
 */
export interface GPUBufferSet {
  weights: GPUBuffer;
  from: GPUBuffer;
  to: GPUBuffer;
  flags: GPUBuffer;
  inStart: GPUBuffer;
  inOrder: GPUBuffer;
  outputs: GPUBuffer;
  bias: GPUBuffer;
  topoLevels: GPUBuffer;
  params: GPUBuffer;
  nodeCount: number;
  connectionCount: number;
  topoLevelsArray: Uint32Array;
}

/**
 * Build the incoming-CSR adjacency arrays needed by the gather kernel.
 *
 * `inStart[node]` and `inStart[node + 1]` bound the slice of `inOrder` that
 * lists connection indices feeding into `node`. The ordering is deterministic
 * because it follows the connection index order returned by the slab.
 *
 * @param slab - Connection slab with `from`/`to` source/target arrays.
 * @param nodeCount - Number of nodes in the network.
 * @param connectionCount - Number of connections in the network.
 * @returns Incoming CSR offsets and connection order arrays.
 */
function buildIncomingCSR(
  slab: ConnectionSlab,
  nodeCount: number,
  connectionCount: number,
): { inStart: Uint32Array; inOrder: Uint32Array } {
  const inStart = new Uint32Array(nodeCount + 1);
  const inOrder = new Uint32Array(connectionCount);

  if (connectionCount === 0) {
    return { inStart, inOrder };
  }

  const counts = new Uint32Array(nodeCount);
  for (let connection = 0; connection < connectionCount; connection += 1) {
    counts[slab.to[connection]] += 1;
  }

  for (let node = 0; node < nodeCount; node += 1) {
    inStart[node + 1] = inStart[node] + counts[node];
  }

  const next = new Uint32Array(inStart.subarray(0, nodeCount));
  for (let connection = 0; connection < connectionCount; connection += 1) {
    const destination = slab.to[connection];
    const position = next[destination];
    next[destination] = position + 1;
    inOrder[position] = connection;
  }

  return { inStart, inOrder };
}

/**
 * Build the outgoing-CSR adjacency arrays used for topological level sorting.
 *
 * @param slab - Connection slab with `from`/`to` source/target arrays.
 * @param nodeCount - Number of nodes in the network.
 * @param connectionCount - Number of connections in the network.
 * @returns Outgoing CSR offsets and connection order arrays.
 */
function buildOutgoingCSR(
  slab: ConnectionSlab,
  nodeCount: number,
  connectionCount: number,
): { outStart: Uint32Array; outOrder: Uint32Array } {
  const outStart = new Uint32Array(nodeCount + 1);
  const outOrder = new Uint32Array(connectionCount);

  const counts = new Uint32Array(nodeCount);
  for (let connection = 0; connection < connectionCount; connection += 1) {
    counts[slab.from[connection]] += 1;
  }

  for (let node = 0; node < nodeCount; node += 1) {
    outStart[node + 1] = outStart[node] + counts[node];
  }

  const next = new Uint32Array(outStart.subarray(0, nodeCount));
  for (let connection = 0; connection < connectionCount; connection += 1) {
    const source = slab.from[connection];
    const position = next[source];
    next[source] = position + 1;
    outOrder[position] = connection;
  }

  return { outStart, outOrder };
}

/**
 * Compute a topological level for every node in a feed-forward network.
 *
 * Input nodes have level `0`; every other node's level is one greater than the
 * maximum level among its incoming sources. The Kahn-style traversal is
 * deterministic and produces the same levels for the same topology, which the
 * GPU kernel uses to schedule per-level dispatches without cross-thread races.
 *
 * @param slab - Connection slab with `from`/`to` source/target arrays.
 * @param nodeCount - Number of nodes in the network.
 * @param connectionCount - Number of connections in the network.
 * @returns A `nodeCount`-length array of unsigned topological levels.
 */
function buildTopoLevels(
  slab: ConnectionSlab,
  nodeCount: number,
  connectionCount: number,
): Uint32Array {
  const levels = new Uint32Array(nodeCount);

  if (connectionCount === 0) {
    return levels;
  }

  const { outStart, outOrder } = buildOutgoingCSR(
    slab,
    nodeCount,
    connectionCount,
  );
  const inDegree = new Uint32Array(nodeCount);
  for (let connection = 0; connection < connectionCount; connection += 1) {
    inDegree[slab.to[connection]] += 1;
  }

  const queue: number[] = [];
  for (let node = 0; node < nodeCount; node += 1) {
    if (inDegree[node] === 0) {
      queue.push(node);
    }
  }

  let head = 0;
  while (head < queue.length) {
    const source = queue[head];
    head += 1;

    const start = outStart[source];
    const end = outStart[source + 1];
    for (let index = start; index < end; index += 1) {
      const connection = outOrder[index];
      const target = slab.to[connection];
      const candidate = levels[source] + 1;
      if (levels[target] < candidate) {
        levels[target] = candidate;
      }
      inDegree[target] -= 1;
      if (inDegree[target] === 0) {
        queue.push(target);
      }
    }
  }

  return levels;
}

/**
 * Build a dense per-node bias array for the GPU gather kernel.
 *
 * @param network - Network whose node biases will be uploaded.
 * @returns Float32 bias values ordered by node index.
 */
function buildBiasArray(network: Network): Float32Array {
  const nodeCount = network.nodes.length;
  const bias = new Float32Array(nodeCount);

  for (let node = 0; node < nodeCount; node += 1) {
    bias[node] = network.nodes[node].bias;
  }

  return bias;
}

/**
 * Upload a network's fast-slab structures to WebGPU buffers.
 *
 * The upload path reuses the existing CPU slab arrays without
 * re-serialization: it creates one `GPUBuffer` per slab/CSR array via
 * `createGPUBuffer`, writes each slab exactly once with
 * `queue.writeBuffer`, and returns the buffer handles plus node/connection
 * counts. The buffer order matches `GPU_BUFFER_BINDING` so the compute
 * kernel can bind them with stable indices.
 *
 * @param device - Mock or real WebGPU device used to allocate buffers.
 * @param network - Network whose fast-slab layout will be uploaded.
 * @returns Handles for the uploaded slab buffers and network metadata.
 * @throws Error when the network is not eligible for the GPU path.
 * @throws Error when a requested buffer size exceeds device limits.
 */
export function uploadNetworkToGPU(
  device: GPUDevice,
  network: Network,
): GPUBufferSet {
  // Step 1: Reject ineligible networks using the capability predicate.
  if (!canUseGPU(network, device, new Set<number>())) {
    throw new Error('Network is not eligible for GPU upload');
  }

  // Step 2: Read the packed connection slab and build GPU-friendly arrays.
  const slab = network.getConnectionSlab() as unknown as ConnectionSlab;
  const nodeCount = network.nodes.length;
  const connectionCount = network.connections.length;
  const { inStart, inOrder } = buildIncomingCSR(
    slab,
    nodeCount,
    connectionCount,
  );
  const topoLevels = buildTopoLevels(slab, nodeCount, connectionCount);
  const bias = buildBiasArray(network);

  // Step 3: Create one GPU buffer per slab/CSR array.
  const weightsBuffer = createGPUBuffer(
    device,
    slab.weights.byteLength,
    'network_weights',
  );
  const fromBuffer = createGPUBuffer(
    device,
    slab.from.byteLength,
    'network_from',
  );
  const toBuffer = createGPUBuffer(device, slab.to.byteLength, 'network_to');
  const flagsBuffer = createGPUBuffer(
    device,
    slab.flags.byteLength,
    'network_flags',
  );
  const inStartBuffer = createGPUBuffer(
    device,
    inStart.byteLength,
    'network_inStart',
  );
  const inOrderBuffer = createGPUBuffer(
    device,
    inOrder.byteLength,
    'network_inOrder',
  );
  const outputsBuffer = createGPUBuffer(
    device,
    nodeCount * Float32Array.BYTES_PER_ELEMENT,
    'network_outputs',
    GPU_BUFFER_USAGE_COPY_SRC,
  );
  const biasBuffer = createGPUBuffer(device, bias.byteLength, 'network_bias');
  const topoLevelsBuffer = createGPUBuffer(
    device,
    topoLevels.byteLength,
    'network_topoLevels',
  );
  const paramsBuffer = createGPUBuffer(
    device,
    2 * Uint32Array.BYTES_PER_ELEMENT,
    'network_params',
  );

  // Step 4: Upload each slab/CSR array to its GPU buffer.
  device.queue.writeBuffer(weightsBuffer, 0, slab.weights);
  device.queue.writeBuffer(fromBuffer, 0, slab.from);
  device.queue.writeBuffer(toBuffer, 0, slab.to);
  device.queue.writeBuffer(flagsBuffer, 0, slab.flags);
  device.queue.writeBuffer(inStartBuffer, 0, inStart);
  device.queue.writeBuffer(inOrderBuffer, 0, inOrder);
  device.queue.writeBuffer(biasBuffer, 0, bias);
  device.queue.writeBuffer(topoLevelsBuffer, 0, topoLevels);
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([0, nodeCount]));
  device.queue.writeBuffer(outputsBuffer, 0, new Float32Array(nodeCount));

  // Step 5: Return the buffer set and metadata required by the compute path.
  return {
    weights: weightsBuffer,
    from: fromBuffer,
    to: toBuffer,
    flags: flagsBuffer,
    inStart: inStartBuffer,
    inOrder: inOrderBuffer,
    outputs: outputsBuffer,
    bias: biasBuffer,
    topoLevels: topoLevelsBuffer,
    params: paramsBuffer,
    nodeCount,
    connectionCount,
    topoLevelsArray: topoLevels,
  };
}

/**
 * Re-upload the weights and bias arrays for a network whose topology has not
 * changed.
 *
 * The GPU kernel reads weights and bias every dispatch, so these buffers must
 * be kept in sync with the CPU network state across activations. Topology
 * buffers are not re-uploaded here; callers recreate the full `GPUBufferSet`
 * when the topology changes.
 *
 * @param device - WebGPU device that owns the buffers.
 * @param bufferSet - Topology buffers created by `uploadNetworkToGPU`.
 * @param network - Network whose current weights and bias will be uploaded.
 */
export function uploadDynamicNetworkBuffers(
  device: GPUDevice,
  bufferSet: GPUBufferSet,
  network: Network,
): void {
  const slab = network.getConnectionSlab() as unknown as ConnectionSlab;
  device.queue.writeBuffer(bufferSet.weights, 0, slab.weights);
  device.queue.writeBuffer(bufferSet.bias, 0, buildBiasArray(network));
}

/**
 * Destroy every GPU buffer in a previously uploaded buffer set.
 *
 * @param device - WebGPU device that owns the buffers (unused by this helper,
 *   kept in the signature for API symmetry).
 * @param bufferSet - Buffer set returned by `uploadNetworkToGPU`.
 */
export function destroyGPUBufferSet(
  device: GPUDevice,
  bufferSet: GPUBufferSet,
): void {
  void device;
  for (const buffer of [
    bufferSet.weights,
    bufferSet.from,
    bufferSet.to,
    bufferSet.flags,
    bufferSet.inStart,
    bufferSet.inOrder,
    bufferSet.outputs,
    bufferSet.bias,
    bufferSet.topoLevels,
    bufferSet.params,
  ]) {
    buffer.destroy();
  }
}
