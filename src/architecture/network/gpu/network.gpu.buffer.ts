import type Network from '../network';
import { canUseGPU } from './network.gpu.capability';
import type { GPUBufferSet } from './network.gpu.types';

export type { GPUBufferSet };

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
 * WebGPU buffer usage flag: the buffer may be bound as a uniform buffer in a
 * compute shader.
 *
 * Mirror of the `GPUBufferUsage.UNIFORM` bit.
 */
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;

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
 * Create a WebGPU uniform buffer that can receive `queue.writeBuffer` uploads.
 *
 * The network parameter buffer is bound as a uniform because it is tiny
 * (a few scalar uniforms) and read once per workgroup. Uniform buffers are
 * limited by `maxUniformBufferBindingSize`, which is much smaller than the
 * storage-buffer limit, so this helper validates against the correct limit.
 *
 * @param device - WebGPU device used to allocate the buffer.
 * @param byteLength - Desired buffer size in bytes. Must be finite and
 *   non-negative.
 * @param label - Debug label attached to the buffer.
 * @returns A freshly created `GPUBuffer` with `UNIFORM | COPY_DST` usage.
 * @throws Error when `byteLength` is invalid or exceeds device limits.
 */
export function createGPUUniformBuffer(
  device: GPUDevice,
  byteLength: number,
  label: string,
): GPUBuffer {
  if (!Number.isFinite(byteLength) || byteLength < 0) {
    throw new Error(
      `Invalid GPU buffer size for "${label}": ${String(byteLength)}`,
    );
  }

  const maxUniformBufferBindingSize =
    device.limits.maxUniformBufferBindingSize!;
  const maxBufferSize = device.limits.maxBufferSize!;

  if (byteLength > maxUniformBufferBindingSize) {
    throw new Error(
      `Buffer "${label}" size ${byteLength} exceeds maxUniformBufferBindingSize ${String(maxUniformBufferBindingSize)}`,
    );
  }

  if (byteLength > maxBufferSize) {
    throw new Error(
      `Buffer "${label}" size ${byteLength} exceeds maxBufferSize ${String(maxBufferSize)}`,
    );
  }

  return device.createBuffer({
    label,
    size: byteLength,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
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
export function buildIncomingCSR(
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
export function buildOutgoingCSR(
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
export function buildTopoLevels(
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
 * Byte stride of one connection struct on the GPU.
 *
 * The WGSL `Connection` struct is `{ from_node: u32, to_node: u32, weight: f32,
 * flags: u32 }`, which is 16 bytes after alignment. Packing all four fields
 * into one struct means a single storage-buffer read brings the whole
 * connection into cache.
 */
const GPU_CONNECTION_STRUCT_BYTES = 16;

/**
 * Byte stride of one node struct on the GPU.
 *
 * The WGSL `Node` struct is `{ activation_state: f32, derivative_state: f32,
 * error: f32, flags: u32 }`, which is 16 bytes after alignment. Reading a node
 * fetches its state, bias (packed into the derivative slot for the forward
 * pass), error, and flags in one contiguous read.
 */
const GPU_NODE_STRUCT_BYTES = 16;

/**
 * Byte size of the per-dispatch params uniform.
 *
 * The WGSL `Params` struct is `{ level: u32, node_count: u32,
 * connection_count: u32, output_start: u32 }`, padded to 16 bytes so it is a
 * valid uniform binding.
 */
const GPU_PARAMS_BYTES = 16;

/**
 * Pack the connection slab into one contiguous struct array.
 *
 * Each connection is laid out as `{ from_node: u32, to_node: u32, weight: f32,
 * flags: u32 }`. Connections are ordered by the incoming-CSR order produced by
 * `buildIncomingCSR`, which groups them by target node and preserves the
 * connection-index order used by the CPU fast-slab path. The kernel can then
 * iterate the struct buffer linearly for each node, keeping f32 summation
 * order consistent with the CPU and avoiding cross-platform drift.
 *
 * @param slab - Connection slab with `from`, `to`, `weights`, and `flags`.
 * @param nodeCount - Number of nodes in the network, including inputs and
 *   outputs. Drives the incoming-CSR offsets used to order the buffer.
 * @param connectionCount - Number of active connections to pack. The slab may
 *   over-allocate, so only this many entries are uploaded.
 * @returns An `ArrayBuffer` ready for `queue.writeBuffer`.
 */
function buildConnectionsArray(
  slab: ConnectionSlab,
  nodeCount: number,
  connectionCount: number,
): ArrayBuffer {
  const { inOrder } = buildIncomingCSR(slab, nodeCount, connectionCount);
  const buffer = new ArrayBuffer(connectionCount * GPU_CONNECTION_STRUCT_BYTES);
  const view = new DataView(buffer);

  for (let index = 0; index < connectionCount; index += 1) {
    const connectionIndex = inOrder[index];
    const offset = index * GPU_CONNECTION_STRUCT_BYTES;
    view.setUint32(offset, slab.from[connectionIndex], true);
    view.setUint32(offset + 4, slab.to[connectionIndex], true);
    view.setFloat32(offset + 8, Number(slab.weights[connectionIndex]), true);
    view.setUint32(offset + 12, slab.flags[connectionIndex], true);
  }

  return buffer;
}

/**
 * Pack node state into one contiguous struct array.
 *
 * Each node is laid out as `{ activation_state: f32, derivative_state: f32,
 * error: f32, flags: u32 }`. The forward-pass kernel reads the bias from the
 * `derivative_state` slot because the plan's node struct keeps `bias` there
 * (the slot is unused by the forward pass otherwise). Callers should treat the
 * `derivative_state` field as the per-node bias while the kernel is running.
 *
 * @param network - Network whose node state will be packed.
 * @returns An `ArrayBuffer` ready for `queue.writeBuffer`.
 */
function buildNodesArray(network: Network): ArrayBuffer {
  const nodeCount = network.nodes.length;
  const buffer = new ArrayBuffer(nodeCount * GPU_NODE_STRUCT_BYTES);
  const floats = new Float32Array(buffer);
  const uints = new Uint32Array(buffer);

  for (let node = 0; node < nodeCount; node += 1) {
    const nodeRef = network.nodes[node];
    const offset = node * 4;
    floats[offset] = nodeRef.state ?? 0;
    floats[offset + 1] = nodeRef.bias;
    floats[offset + 2] = nodeRef.error.responsibility ?? 0;
    uints[offset + 3] = 0;
  }

  return buffer;
}

/**
 * Count how many distinct topological levels are present in a level array.
 *
 * Levels start at `0` for input nodes, so the number of passes needed by the
 * dispatch loop is `max(levels) + 1`.
 *
 * @param levels - Per-node topological level array.
 * @returns Number of distinct levels.
 */
function computeTopoLevelCount(levels: Uint32Array): number {
  let maxLevel = 0;
  for (let index = 0; index < levels.length; index += 1) {
    if (levels[index] > maxLevel) {
      maxLevel = levels[index];
    }
  }
  return maxLevel + 1;
}

/**
 * Upload a network's fast-slab structures to WebGPU buffers.
 *
 * The upload path packs connections and nodes into two struct arrays and then
 * creates only four GPU buffers: connections, nodes, outputs, and params.
 * Keeping the binding count at four sits below the WebGPU default limit for
 * storage buffers per shader stage and removes the need to request a custom
 * `maxStorageBuffersPerShaderStage` limit.
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
  const connectionsArray = buildConnectionsArray(
    slab,
    nodeCount,
    connectionCount,
  );
  const nodesArray = buildNodesArray(network);
  const topoLevels = buildTopoLevels(slab, nodeCount, connectionCount);

  // Step 3: Create the four buffers required by the struct-packed contract.
  const connectionsBuffer = createGPUBuffer(
    device,
    connectionsArray.byteLength,
    'network_connections',
  );
  const nodesBuffer = createGPUBuffer(
    device,
    nodesArray.byteLength,
    'network_nodes',
  );
  const outputsBuffer = createGPUBuffer(
    device,
    nodeCount * Float32Array.BYTES_PER_ELEMENT,
    'network_outputs',
    GPU_BUFFER_USAGE_COPY_SRC,
  );
  const paramsBuffer = createGPUUniformBuffer(
    device,
    GPU_PARAMS_BYTES,
    'network_params',
  );

  // Step 4: Upload the packed arrays and initialize mutable buffers.
  device.queue.writeBuffer(connectionsBuffer, 0, connectionsArray);
  device.queue.writeBuffer(nodesBuffer, 0, nodesArray);
  device.queue.writeBuffer(outputsBuffer, 0, new Float32Array(nodeCount));
  device.queue.writeBuffer(
    paramsBuffer,
    0,
    new Uint32Array([
      0,
      nodeCount,
      connectionCount,
      nodeCount - network.output,
    ]),
  );

  // Step 5: Return the buffer set and metadata required by the compute path.
  return {
    connections: connectionsBuffer,
    nodes: nodesBuffer,
    outputs: outputsBuffer,
    params: paramsBuffer,
    nodeCount,
    connectionCount,
    topoLevelsArray: topoLevels,
    topoLevelCount: computeTopoLevelCount(topoLevels),
  };
}

/**
 * Re-upload the weights and bias arrays for a network whose topology has not
 * changed.
 *
 * The GPU kernel reads weights and node biases every dispatch, so these fields
 * must be kept in sync with the CPU network state across activations. Because
 * the values live inside struct arrays, the whole connections buffer and the
 * whole nodes buffer are rewritten. Topology metadata does not change here;
 * callers recreate the full `GPUBufferSet` when the topology changes.
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
  device.queue.writeBuffer(
    bufferSet.connections,
    0,
    buildConnectionsArray(slab, bufferSet.nodeCount, bufferSet.connectionCount),
  );
  device.queue.writeBuffer(bufferSet.nodes, 0, buildNodesArray(network));
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
    bufferSet.connections,
    bufferSet.nodes,
    bufferSet.outputs,
    bufferSet.params,
  ]) {
    buffer.destroy();
  }
}
