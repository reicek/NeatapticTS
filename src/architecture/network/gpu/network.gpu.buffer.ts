/**
 * CPU-side preparation and upload for the WebGPU activation path.
 *
 * Before the GPU can run a forward pass, the network's connection slab,
 * node state, topological levels, and incoming-edge CSR offsets must be packed
 * into GPU-friendly arrays and copied to the device. This module owns that
 * preparation. The work is CPU-bound: on an RTX 4070 the CPU preparation slice
 * grows from 13.2% of wall time for a 64-node network to 42.3–44.9% for 4k–8k
 * node networks, so this path is also the place to look when optimizing large-
 * network latency.
 *
 * The upload path splits data into static and dynamic buffers. Static buffers
 * (the connection array sorted by source rank, topological levels, and CSR
 * start offsets) change only when the topology changes. Dynamic buffers (the
 * full connection struct array with current weights and the full node struct
 * array with current biases) are rewritten on every activation through
 * `uploadDynamicNetworkBuffers()`. Keeping the split narrow avoids paying the
 * topological-sort cost on every forward pass.
 *
 * The packed layout uses a compressed sparse row (CSR) representation for
 * incoming edges and a Kahn-style topological sort to level nodes. Both are
 * standard graph algorithms that keep the GPU kernel simple: one thread per
 * node can gather its inputs by walking a contiguous slice of the connection
 * array.
 *
 * WebGPU buffer upload strategy follows the hierarchy Brandon Jones documents in
 * the `toji.dev` WebGPU best-practices guide. Data written once and rarely
 * changed should be created with `mappedAtCreation: true`, filled directly from
 * the CPU, and then unmapped. Data updated every frame should use
 * `queue.writeBuffer()`, which queues an asynchronous GPU-side copy and avoids
 * stalling the CPU. Mappable buffers and `mapAsync()` should be reserved for
 * readback, because mapping waits until the GPU is finished with the buffer.
 * Destroying and recreating buffers on the hot path is expensive and is avoided
 * here by caching buffer sets and staging buffers. The static upload path uses
 * `createBuffer` followed by `queue.writeBuffer()`; a one-shot static upload could
 * instead use `mappedAtCreation: true` for the slab. The dynamic weight/bias
 * updates already follow the `writeBuffer` rule.
 *
 * TensorFlow.js codifies this with its `BufferManager`: a pool of
 * `GPUBuffer` handles keyed by `${size}_${usage}` with `acquireBuffer` and
 * `releaseBuffer` lifecycles, so inference never pays allocation or destruction
 * overhead. burn's `burn-wgpu` compute server uses the same idea with a
 * size-keyed handle pool. NeatapticTS keeps topology buffer sets alive across
 * activations and reuses one staging buffer per output size, which moves in the
 * same direction but still re-uploads and reads back every pass.
 *
 * @see [Compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
 * @see [Topological sorting](https://en.wikipedia.org/wiki/Topological_sorting)
 * @see [toji.dev — WebGPU buffer uploads](https://toji.dev/webgpu-best-practices/buffer-uploads)
 *   for the `mappedAtCreation` / `writeBuffer` / `mapAsync` trade-off.
 * @see [toji.dev — writeBuffer when in doubt](https://toji.dev/webgpu-best-practices/buffer-uploads#when-in-doubt-writebuffer)
 * @see [toji.dev — buffers written to frequently](https://toji.dev/webgpu-best-practices/buffer-uploads#buffers-that-are-written-to-frequently)
 * @see [TensorFlow.js BufferManager](https://github.com/tensorflow/tfjs/blob/7f5309fef0a47545e34049903dbdae0f97285f7e/tfjs-backend-webgpu/src/buffer_manager.ts)
 * @see [burn-wgpu compute server](https://github.com/tracel-ai/burn/blob/v0.12.1/burn-wgpu/src/compute/server.rs)
 * @see [WebGPU Performance Guide](https://github.com/reicek/NeatapticTS/blob/main/docs/webgpu-performance-guide.md)
 *
 * @module
 */
import type Network from '../network';
import type Node from '../../node/node';
import { canUseGPU } from './network.gpu.capability';
import { SUPPORTED_ACTIVATION_INDICES } from './network.gpu.kernel';
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
 * because it follows the connection index order returned by the slab. Using a
 * compressed sparse row layout lets the GPU kernel gather a node's inputs with
 * one contiguous storage-buffer read per incoming edge instead of chasing
 * pointers.
 *
 * @param slab - Connection slab with `from`/`to` source/target arrays.
 * @param nodeCount - Number of nodes in the network.
 * @param connectionCount - Number of connections in the network.
 * @returns Incoming CSR offsets and connection order arrays.
 * @see [Compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
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
 * The outgoing CSR mirrors the incoming CSR but lets the topological walk start
 * from source nodes and follow forward edges. It is computed once per topology
 * change, so its cost is amortized across many activations.
 *
 * @param slab - Connection slab with `from`/`to` source/target arrays.
 * @param nodeCount - Number of nodes in the network.
 * @param connectionCount - Number of connections in the network.
 * @returns Outgoing CSR offsets and connection order arrays.
 * @see [Compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
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
 * This step is part of the static upload set and runs only when the topology
 * changes.
 *
 * @param slab - Connection slab with `from`/`to` source/target arrays.
 * @param nodeCount - Number of nodes in the network.
 * @param connectionCount - Number of connections in the network.
 * @returns A `nodeCount`-length array of unsigned topological levels.
 * @see [Topological sorting](https://en.wikipedia.org/wiki/Topological_sorting)
 * @see [Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm)
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
 * Resolve the deterministic tie-break scalar used by the CPU topological sort.
 *
 * The CPU fast-slab path emits nodes in Kahn order and sorts each zero-in-degree
 * wave with this same rule, so matching it exactly lets the GPU pack incoming
 * edges in the same source-node order.
 *
 * @param node - Node whose stable gene id or index will be read.
 * @returns Deterministic scalar for ordering.
 */
function resolveStableNodeTieBreak(node: Node): number {
  if (typeof node.geneId === 'number' && Number.isFinite(node.geneId)) {
    return node.geneId;
  }

  if (typeof node.index === 'number' && Number.isFinite(node.index)) {
    return node.index;
  }

  return Number.MAX_SAFE_INTEGER;
}

/**
 * Compute the source-node topological rank used to order GPU incoming edges.
 *
 * The CPU fast-slab path accumulates outgoing activations by walking nodes in
 * topological order (all level-0 nodes in stable tie-break order, then level-1,
 * and so on). By sorting each target node's incoming slice by the source's rank
 * in that same order, the GPU gather kernel sums the exact same f32 terms in the
 * exact same order, eliminating cross-path rounding drift.
 *
 * @param network - Network whose nodes supply the stable tie-break values.
 * @param slab - Connection slab with `from`/`to` source/target arrays.
 * @param nodeCount - Number of nodes in the network.
 * @param connectionCount - Number of connections in the network.
 * @returns Per-node rank in the CPU-equivalent topological walk.
 */
function buildSourceTopoRanks(
  network: Network,
  slab: ConnectionSlab,
  nodeCount: number,
  connectionCount: number,
): Uint32Array {
  const levels = buildTopoLevels(slab, nodeCount, connectionCount);
  const order = Array.from({ length: nodeCount }, (_, index) => index).toSorted(
    (leftIndex, rightIndex) => {
      const levelDiff = levels[leftIndex] - levels[rightIndex];
      if (levelDiff !== 0) {
        return levelDiff;
      }
      return (
        resolveStableNodeTieBreak(network.nodes[leftIndex]) -
        resolveStableNodeTieBreak(network.nodes[rightIndex])
      );
    },
  );

  const ranks = new Uint32Array(nodeCount);
  for (let rank = 0; rank < nodeCount; rank += 1) {
    ranks[order[rank]] = rank;
  }

  return ranks;
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
export const GPU_NODE_STRUCT_BYTES = 16;

/**
 * Write input activations into the `activation_state` slot of the first
 * `inputs.length` node structs.
 *
 * The WGSL `Node` struct stores `activation_state` at byte offset zero of each
 * 16-byte struct, so input node `i` must be written at `i * GPU_NODE_STRUCT_BYTES`
 * rather than at `i * Float32Array.BYTES_PER_ELEMENT`. Centralising this logic in
 * one helper prevents contiguous-write bugs when multiple upload paths need to
 * seed the node buffer with input values.
 *
 * @param device - WebGPU device whose queue will perform the write.
 * @param nodesBuffer - GPU node buffer created by `uploadNetworkToGPU`.
 * @param inputs - Input vector to scatter into the node struct array.
 */
export function writeInputValuesToNodeStruct(
  device: GPUDevice,
  nodesBuffer: GPUBuffer,
  inputs: Float32Array,
): void {
  for (let index = 0; index < inputs.length; index += 1) {
    device.queue.writeBuffer(
      nodesBuffer,
      index * GPU_NODE_STRUCT_BYTES,
      inputs,
      index,
      1,
    );
  }
}

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
 * flags: u32 }`. Connections are sorted by `(target_node, source_topological_rank)`
 * so that each target node's incoming slice `[inStart[node], inStart[node+1])`
 * is iterated in the same source-node order the CPU fast-slab path uses. Because
 * f32 summation is order-dependent, matching the accumulation order gives the
 * GPU gather kernel the same rounded result as the CPU push path instead of
 * relying on looser tolerances.
 *
 * This step is part of the dynamic upload set, so it runs on every activation.
 * Its cost is linear in the connection count and becomes a measurable fraction
 * of wall time for large networks (up to ~44.9% CPU preparation for 4k nodes on
 * an RTX 4070). Avoiding it requires keeping the topology unchanged and using
 * a weight-only update path, which this module does not provide.
 *
 * @param network - Network whose nodes and connection slab will be packed.
 * @param connectionCount - Number of active connections to pack. The slab may
 *   over-allocate, so only this many entries are uploaded.
 * @returns An `ArrayBuffer` ready for `queue.writeBuffer`.
 * @see [Compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
 */
export function buildConnectionsArray(
  network: Network,
  connectionCount: number,
): ArrayBuffer {
  const slab = network.getConnectionSlab() as unknown as ConnectionSlab;
  const nodeCount = network.nodes.length;
  const sourceRanks = buildSourceTopoRanks(
    network,
    slab,
    nodeCount,
    connectionCount,
  );
  const { inStart, inOrder } = buildIncomingCSR(
    slab,
    nodeCount,
    connectionCount,
  );

  if (connectionCount > 0) {
    for (let node = 0; node < nodeCount; node += 1) {
      const start = inStart[node];
      const end = inStart[node + 1];
      if (end - start > 1) {
        const slice = inOrder.subarray(start, end);
        const sorted = Array.from(slice).toSorted(
          (leftConnection, rightConnection) =>
            sourceRanks[slab.from[leftConnection]] -
            sourceRanks[slab.from[rightConnection]],
        );
        slice.set(sorted);
      }
    }
  }

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
 * `derivative_state` slot because the node struct stores `bias` there (the slot
 * is unused by the forward pass otherwise). Callers should treat the
 * `derivative_state` field as the per-node bias while the kernel is running.
 *
 * Like `buildConnectionsArray()`, this step is part of the dynamic upload
 * set and is rewritten on every activation. Its cost is linear in the node
 * count and is included in the CPU-preparation share reported in the
 * performance guide.
 *
 * @param network - Network whose node state will be packed.
 * @returns An `ArrayBuffer` ready for `queue.writeBuffer`.
 */
export function buildNodesArray(network: Network): ArrayBuffer {
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
export function computeTopoLevelCount(levels: Uint32Array): number {
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
 * creates six GPU buffers: connections, nodes, outputs, params, topological
 * levels, and incoming-CSR start offsets. The six-buffer layout still sits
 * below the WebGPU default limit for storage buffers per shader stage and
 * removes the need to request a custom `maxStorageBuffersPerShaderStage`
 * limit.
 *
 * This function performs the static upload: the connection array, topological
 * levels, and CSR start offsets change only when the topology changes and are
 * cached through `ensureNetworkGPUState()`. Callers must still call
 * `uploadDynamicNetworkBuffers()` before each activation to refresh
 * weights and biases.
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
  // Networks that carry a valid worker-registry activation index are still
  // eligible for upload; the kernel compiler checks whether that index is
  // supported when the pipeline is built.
  const supportedActivations = new Set<number>(
    SUPPORTED_ACTIVATION_INDICES as unknown as number[],
  );
  if (!canUseGPU(network, device, supportedActivations)) {
    throw new Error('Network is not eligible for GPU upload');
  }

  // Step 2: Read the packed connection slab and build GPU-friendly arrays.
  const nodeCount = network.nodes.length;
  const connectionCount = network.connections.length;
  const slab = network.getConnectionSlab() as unknown as ConnectionSlab;
  const connectionsArray = buildConnectionsArray(network, connectionCount);
  const nodesArray = buildNodesArray(network);
  const { inStart } = buildIncomingCSR(slab, nodeCount, connectionCount);
  const topoLevels = buildTopoLevels(slab, nodeCount, connectionCount);

  // Step 3: Create the six buffers required by the struct-packed contract.
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
  const topoLevelsBuffer = createGPUBuffer(
    device,
    topoLevels.byteLength,
    'network_topo_levels',
  );
  const inStartBuffer = createGPUBuffer(
    device,
    inStart.byteLength,
    'network_in_start',
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
  device.queue.writeBuffer(topoLevelsBuffer, 0, topoLevels);
  device.queue.writeBuffer(inStartBuffer, 0, inStart);

  // Step 5: Return the buffer set and metadata required by the compute path.
  return {
    connections: connectionsBuffer,
    nodes: nodesBuffer,
    outputs: outputsBuffer,
    params: paramsBuffer,
    topoLevels: topoLevelsBuffer,
    inStart: inStartBuffer,
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
 * This whole-buffer rewrite is the dynamic-upload cost shown in the performance
 * guide. On an RTX 4070 it accounts for a growing share of wall time as the
 * network grows, reaching roughly 24–33% of the single-network GPU path for
 * 1k–8k nodes. Callers that evaluate the same static cohort many times can use
 * the batched activation path's `skipUpload` flag to avoid paying this cost on
 * every call.
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
  device.queue.writeBuffer(
    bufferSet.connections,
    0,
    buildConnectionsArray(network, bufferSet.connectionCount),
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
    bufferSet.topoLevels,
    bufferSet.inStart,
  ]) {
    buffer.destroy();
  }
}

/**
 * Allocate a fresh, independent GPU buffer set for a single concurrent request.
 *
 * Every call creates a new set of WebGPU buffers. This keeps concurrent or
 * interleaved activations of the same network instance from reading or writing
 * each other's node/output state, which is the critical requirement for
 * parallel multi-agent evaluation.
 *
 * @param device - WebGPU device that owns the newly created buffers.
 * @param network - Network whose slab will be uploaded.
 * @returns A freshly allocated `GPUBufferSet` isolated from any other request.
 */
export function createConcurrentBufferSet(
  device: GPUDevice,
  network: Network,
): GPUBufferSet {
  return uploadNetworkToGPU(device, network);
}
