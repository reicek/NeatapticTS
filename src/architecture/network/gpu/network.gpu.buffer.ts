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
 * Internal network state used to read the CSR adjacency arrays produced by
 * the fast-slab path. The cast is intentional: GPU upload is a consumer of the
 * same private layout that slab activation uses.
 */
interface NetworkSlabInternals {
  _outStart?: Uint32Array | null;
  _outOrder?: Uint32Array | null;
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
  outStart: GPUBuffer;
  outOrder: GPUBuffer;
  outputs: GPUBuffer;
  nodeCount: number;
  connectionCount: number;
}

/**
 * Upload a network's fast-slab structures to WebGPU buffers.
 *
 * The upload path reuses the existing CPU slab arrays without
 * re-serialization: it creates one `GPUBuffer` per slab/CSR array via
 * {@link createGPUBuffer}, writes each slab exactly once with
 * `queue.writeBuffer`, and returns the buffer handles plus node/connection
 * counts. The buffer order matches {@link GPU_BUFFER_BINDING} so the compute
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

  // Step 2: Read the packed connection slab and CSR adjacency arrays.
  const slab = network.getConnectionSlab();
  const internals = network as unknown as NetworkSlabInternals;
  const outStart = internals._outStart ?? new Uint32Array(0);
  const outOrder = internals._outOrder ?? new Uint32Array(0);

  // Step 3: Create one GPU buffer per slab array.
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
  const outStartBuffer = createGPUBuffer(
    device,
    outStart.byteLength,
    'network_outStart',
  );
  const outOrderBuffer = createGPUBuffer(
    device,
    outOrder.byteLength,
    'network_outOrder',
  );
  const outputsBuffer = createGPUBuffer(
    device,
    network.nodes.length * Float32Array.BYTES_PER_ELEMENT,
    'network_outputs',
  );

  // Step 4: Upload each slab to its GPU buffer.
  device.queue.writeBuffer(weightsBuffer, 0, slab.weights);
  device.queue.writeBuffer(fromBuffer, 0, slab.from);
  device.queue.writeBuffer(toBuffer, 0, slab.to);
  device.queue.writeBuffer(flagsBuffer, 0, slab.flags);
  device.queue.writeBuffer(outStartBuffer, 0, outStart);
  device.queue.writeBuffer(outOrderBuffer, 0, outOrder);
  device.queue.writeBuffer(
    outputsBuffer,
    0,
    new Float32Array(network.nodes.length),
  );

  // Step 5: Return the buffer set and metadata required by the compute path.
  return {
    weights: weightsBuffer,
    from: fromBuffer,
    to: toBuffer,
    flags: flagsBuffer,
    outStart: outStartBuffer,
    outOrder: outOrderBuffer,
    outputs: outputsBuffer,
    nodeCount: network.nodes.length,
    connectionCount: network.connections.length,
  };
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
    bufferSet.outStart,
    bufferSet.outOrder,
    bufferSet.outputs,
  ]) {
    buffer.destroy();
  }
}
