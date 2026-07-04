/**
 * WebGPU single-network activation seam.
 *
 * This folder implements the optional GPU fast path for `Network.activate`.
 * Neural networks in NeatapticTS normally run on the CPU, which is the
 * deterministic source of truth for evolution, replay, and cross-machine
 * benchmarks. When the runtime has a working WebGPU device and the network is
 * eligible, the same forward pass can be dispatched to the GPU for higher
 * throughput. The seam is deliberately thin: this module wires together the
 * kernel compiler, buffer upload, and device probe so each piece stays focused.
 *
 * The high-level entry point is `activateGPU`. Most callers should use the
 * public `Network.activate(..., { useGPU: true })` overload instead, because
 * it automatically falls back to the CPU path when the network or device is
 * ineligible.
 *
 * GPU output is expected to agree with the CPU path within an absolute tolerance
 * of `1e-3` and a mean absolute error of `≤ 1e-4`. Use the CPU path for
 * deterministic replay and cross-machine regression tests.
 *
 * ```mermaid
 * flowchart TD
 *     A["Network.activate(input, { useGPU: true })"] --> B{isGPUEligible?}
 *     B -->|yes| C[Upload slab to GPU]
 *     C --> D[Compile / cache kernel]
 *     D --> E[Dispatch compute]
 *     E --> F[Read back outputs]
 *     F --> G([Float32Array])
 *     B -->|no| H[CPU activate]
 *     H --> G
 * ```
 *
 * @see [WebGPU](https://en.wikipedia.org/wiki/WebGPU) on Wikipedia for
 * background on the browser GPU compute API.
 *
 * @module
 */

import type Network from '../network';
import { ACTIVATION_FUNCTIONS } from '../../../multithreading/multi.utils';
import { canUseGPU } from './network.gpu.capability';
import {
  buildGPUPipeline,
  createActivationKernel,
  createBindGroupLayout,
  SUPPORTED_ACTIVATION_INDICES,
} from './network.gpu.kernel';
import {
  destroyGPUBufferSet,
  uploadDynamicNetworkBuffers,
  uploadNetworkToGPU,
  writeInputValuesToNodeStruct,
  type GPUBufferSet,
} from './network.gpu.buffer';
import { GPU_BUFFER_BINDING } from './network.gpu.types';

/**
 * Per-network cached GPU state.
 *
 * The buffer set and bind group are reused across activations while the
 * network topology (node count, connection set, and adjacency structure) stays
 * unchanged. The compiled pipeline lives in a separate per-device cache keyed
 * by the generated WGSL shader so that networks with different topologies but
 * the same activation function share one compiled pipeline.
 */
interface NetworkGPUState {
  device: GPUDevice;
  topologyHash: string;
  bufferSet: GPUBufferSet;
  bindGroup: GPUBindGroup;
}

/** Per-network cache of uploaded GPU state, keyed by the live network object. */
const networkGPUStateCache = new WeakMap<Network, NetworkGPUState>();

/**
 * Per-device cache for the standard activation bind-group layout.
 *
 * All activation pipelines in this module use the same four-entry
 * struct-packed layout, so a single layout per device is sufficient and avoids
 * redundant layout creation.
 */
const activationBindGroupLayoutCache = new WeakMap<
  GPUDevice,
  GPUBindGroupLayout
>();

/**
 * Per-device cache for compiled activation pipelines.
 *
 * The cache key is the generated WGSL source string, which now depends on both
 * the activation function switch and the embedded topology constants.
 * Identical WGSL implies an identical `GPUComputePipeline` and
 * `GPUPipelineLayout`, so the same compiled pipeline can drive any network that
 * shares the same topology and activation.
 */
const activationPipelineCache = new WeakMap<
  GPUDevice,
  Map<string, GPUComputePipeline>
>();

/**
 * Compute a deterministic topology hash for a network.
 *
 * The hash includes node count and the ordered from/to indices of every
 * connection. Networks that differ only in weights therefore share a hash,
 * which lets the GPU buffer cache reuse the uploaded static structure across
 * weight-only mutations such as backprop updates.
 */
function computeTopologyHash(network: Network): string {
  const nodeCount = network.nodes.length;
  const connections = network.connections;

  let hash = nodeCount;
  for (let i = 0; i < connections.length; i++) {
    const connection = connections[i];
    hash = ((hash << 5) - hash + connection.from.index!) | 0;
    hash = ((hash << 5) - hash + connection.to.index!) | 0;
  }

  return `${connections.length}:${hash}`;
}

/**
 * Return the shared bind-group layout for activation kernels on this device,
 * creating and caching it on first use.
 */
function getActivationBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
  let layout = activationBindGroupLayoutCache.get(device);
  if (!layout) {
    layout = createBindGroupLayout(device);
    activationBindGroupLayoutCache.set(device, layout);
  }
  return layout;
}

/**
 * Return the per-device pipeline cache map, creating it on first use.
 */
function getActivationPipelineCache(
  device: GPUDevice,
): Map<string, GPUComputePipeline> {
  let cache = activationPipelineCache.get(device);
  if (!cache) {
    cache = new Map();
    activationPipelineCache.set(device, cache);
  }
  return cache;
}

/**
 * Compile (or reuse) the activation compute pipeline for a network.
 *
 * The pipeline is keyed by the generated WGSL source, so networks that share an
 * activation function share one compiled pipeline even when their topologies
 * differ. The temporary activation-index annotation on the first node is
 * restored before returning, keeping the mutation scoped to this seam.
 */
function getOrCreateActivationPipeline(
  device: GPUDevice,
  network: Network,
): GPUComputePipeline {
  const activationContext = prepareActivationContext(network);

  try {
    const source = createActivationKernel(network);
    const cache = getActivationPipelineCache(device);
    const cached = cache.get(source);
    if (cached) {
      return cached;
    }

    const shaderModule = device.createShaderModule({ code: source });
    const bindGroupLayout = getActivationBindGroupLayout(device);
    const pipeline = buildGPUPipeline(device, shaderModule, bindGroupLayout);
    cache.set(source, pipeline);
    return pipeline;
  } finally {
    activationContext.restore();
  }
}

/**
 * Ensure the cached buffer set and bind group for a network match the current
 * topology, and return the compatible compiled pipeline.
 *
 * Creates or reuses GPU state as needed. When only the activation function
 * changes, the buffer set (which is independent of activation) is kept and only
 * the pipeline is replaced.
 */
function ensureNetworkGPUState(
  device: GPUDevice,
  network: Network,
): { state: NetworkGPUState; pipeline: GPUComputePipeline } {
  const topologyHash = computeTopologyHash(network);
  const cached = networkGPUStateCache.get(network);

  if (
    cached &&
    cached.device === device &&
    cached.topologyHash === topologyHash
  ) {
    const pipeline = getOrCreateActivationPipeline(device, network);
    return { state: cached, pipeline };
  }

  if (cached) {
    destroyGPUBufferSet(device, cached.bufferSet);
  }

  const bufferSet = uploadNetworkToGPU(device, network);
  const pipeline = getOrCreateActivationPipeline(device, network);
  const bindGroupLayout = getActivationBindGroupLayout(device);
  const bindGroup = createActivationBindGroup(
    device,
    bindGroupLayout,
    bufferSet,
  );

  const state: NetworkGPUState = {
    device,
    topologyHash,
    bufferSet,
    bindGroup,
  };

  networkGPUStateCache.set(network, state);
  return { state, pipeline };
}

/**
 * Run a single-network forward pass on the supplied WebGPU device.
 *
 * @param device - WebGPU device used to run the forward kernel.
 * @param network - Network whose fast-slab topology will be uploaded.
 * @param inputs - Input vector of length `network.input`.
 * @returns A promise resolving to a Float32Array of output-node values.
 * @throws Error when the network is ineligible for GPU inference.
 * @throws Error when the network has no nodes or the first node's activation
 *   cannot be mapped to a supported worker-registry index.
 *
 * @example
 * ```ts
 * const adapter = await navigator.gpu.requestAdapter({
 *   powerPreference: 'high-performance',
 * });
 * const device = await adapter?.requestDevice();
 * if (device) {
 *   const output = await activateGPU(device, network, [0.5, -0.2]);
 * }
 * ```
 */
export async function activateGPU(
  device: GPUDevice,
  network: Network,
  inputs: Float32Array | number[],
): Promise<Float32Array> {
  const supportedActivations = new Set<number>(
    SUPPORTED_ACTIVATION_INDICES as unknown as number[],
  );

  if (!canUseGPU(network, device, supportedActivations)) {
    throw new Error(
      'activateGPU: network is not eligible for GPU inference (gated topology, self-connection, unsupported activation, or missing device)',
    );
  }

  if (network.nodes.length === 0) {
    throw new Error('activateGPU: network has no nodes');
  }

  const typedInputs =
    inputs instanceof Float32Array ? inputs : new Float32Array(inputs);

  if (typedInputs.length !== network.input) {
    throw new Error(
      `activateGPU: expected ${network.input} inputs, received ${typedInputs.length}`,
    );
  }

  const { state, pipeline } = ensureNetworkGPUState(device, network);
  const { bufferSet, bindGroup } = state;

  uploadDynamicNetworkBuffers(device, bufferSet, network);
  writeInputValuesToNodeStruct(device, bufferSet.nodes, typedInputs);
  await dispatchActivationKernel(
    device,
    bufferSet,
    pipeline,
    bindGroup,
    network.output,
  );

  return readOutputValues(device, network, bufferSet);
}

/**
 * Local extension of the ambient GPU command encoder so we can copy a storage
 * buffer to a mappable staging buffer. The WebGPU ambient types in this repo are
 * intentionally minimal; the cast is justified because `copyBufferToBuffer` is
 * part of the actual WebGPU API surface.
 */
interface GPUCommandEncoderCopy extends GPUCommandEncoder {
  copyBufferToBuffer(
    sourceBuffer: GPUBuffer,
    sourceOffset: number,
    destinationBuffer: GPUBuffer,
    destinationOffset: number,
    size: number,
  ): void;
}

/** Number of threads per compute workgroup for the activation kernel. */
const ACTIVATION_WORKGROUP_SIZE = 64;

/** WebGPU buffer usage flag for mappable readback buffers. */
const GPU_BUFFER_USAGE_MAP_READ = 0x0001;

/** WebGPU buffer usage flag for copy destinations. */
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;

/** WebGPU map mode for reading mapped buffers. */
const GPU_MAP_MODE_READ = 0x0001;

/**
 * Stable cross-module activation key used by the runtime registry.
 *
 * `Symbol.for` keeps the key identical even when `methods/activation` and the
 * worker registry are loaded from different bundles or mocked contexts, which is
 * exactly the boundary that breaks strict function identity.
 */
const ACTIVATION_KEY_SYMBOL = Symbol.for('neataptic.activation.key');

/**
 * Map from activation base name to worker-registry index.
 *
 * The worker registry function names end in "Activation" (e.g.
 * `logisticActivation`); the runtime registry and the WGSL naming table use
 * the shorter base form (e.g. `logistic`). Stripping the suffix gives a single
 * stable lookup key that works for both.
 */
const WORKER_ACTIVATION_INDEX_BY_NAME = new Map<string, number>(
  ACTIVATION_FUNCTIONS.map((activation, index) => {
    const baseName = activation.name.replace(/Activation$/, '');
    return [baseName, index];
  }),
);

/**
 * Names that the runtime registry uses but the worker registry names
 * differently.
 *
 * The runtime registry exposes `sigmoid` as an alias for the logistic function,
 * while the worker registry only stores the canonical `logisticActivation`.
 */
const ACTIVATION_NAME_ALIASES = new Map<string, string>([
  ['sigmoid', 'logistic'],
]);

/**
 * Sample inputs used to compare an unknown squash against the built-in
 * activation registry. The spread covers negative, zero, and positive values
 * so that most activation families produce distinct fingerprints.
 */
const ACTIVATION_SAMPLE_INPUTS = [-2, -1, 0, 1, 2];

/**
 * Tolerance for floating-point equality when matching an unknown squash to a
 * built-in activation. The threshold is tight because behaviour-matching is
 * only intended for thin wrappers around the exact same math.
 */
const ACTIVATION_SAMPLE_TOLERANCE = 1e-9;

/**
 * Test whether a candidate squash produces the same values as a reference
 * built-in activation across a small deterministic input grid.
 *
 * This lets the GPU path support thin wrappers (for example the benchmark
 * harness wrapping `Neataptic.methods.Activation.logistic` with a custom
 * symbol key) without requiring the wrapper to carry the exact same function
 * object as the worker registry.
 */
function matchesBuiltInActivation(
  candidate: (value: number, derivate?: boolean) => number,
  reference: (value: number, derivate?: boolean) => number,
): boolean {
  for (const sample of ACTIVATION_SAMPLE_INPUTS) {
    try {
      const candidateValue = candidate(sample, false);
      const referenceValue = reference(sample, false);
      if (
        !Number.isFinite(candidateValue) ||
        !Number.isFinite(referenceValue) ||
        Math.abs(candidateValue - referenceValue) > ACTIVATION_SAMPLE_TOLERANCE
      ) {
        return false;
      }
    } catch {
      return false;
    }
  }

  return true;
}

/**
 * Look up the worker-registry activation index for a built-in squash function.
 *
 * The lookup is intentionally robust across module-loading boundaries and mock
 * environments where the same activation may be imported from different source
 * files and therefore fails a strict `===` comparison. It first tries the
 * runtime-registry symbol key, then falls back to the function name, and
 * finally falls back to a deterministic behaviour match against the built-in
 * activation functions so that thin wrappers around supported activations are
 * still dispatchable.
 *
 * @param squash - Activation function attached to a node.
 * @returns The corresponding worker index, or `undefined` when the function is
 *   not part of the canonical registry.
 */
export function resolveActivationIndex(
  squash: (value: number, derivate?: boolean) => number,
): number | undefined {
  // Fast path: strict identity still works when the same function object is
  // reused by both the node and the worker registry.
  const identityIndex = ACTIVATION_FUNCTIONS.indexOf(squash);
  if (identityIndex >= 0) {
    return identityIndex;
  }

  // Symbol-based lookup: works across module boundaries because the runtime
  // registry annotates built-in activations with a shared Symbol.for key.
  const keyedSquash = squash as typeof squash & {
    [ACTIVATION_KEY_SYMBOL]?: string;
  };
  const symbolKey = keyedSquash[ACTIVATION_KEY_SYMBOL];
  if (typeof symbolKey === 'string') {
    const aliasedName = ACTIVATION_NAME_ALIASES.get(symbolKey) ?? symbolKey;
    const index = WORKER_ACTIVATION_INDEX_BY_NAME.get(aliasedName);
    if (index !== undefined) {
      return index;
    }
  }

  // Name-based fallback: handles functions that were not annotated by the
  // runtime registry but still carry a recognisable name.
  const name = squash.name;
  if (name) {
    const baseName = name.replace(/Activation$/, '');
    const aliasedName = ACTIVATION_NAME_ALIASES.get(baseName) ?? baseName;
    const index = WORKER_ACTIVATION_INDEX_BY_NAME.get(aliasedName);
    if (index !== undefined) {
      return index;
    }
  }

  // Behaviour-based fallback: supports wrappers around built-in activations
  // that carry a custom symbol key or name (common in bundled browser pages)
  // but are mathematically equivalent to a supported function.
  for (let index = 0; index < ACTIVATION_FUNCTIONS.length; index += 1) {
    if (matchesBuiltInActivation(squash, ACTIVATION_FUNCTIONS[index])) {
      return index;
    }
  }

  return undefined;
}

/**
 * Temporarily annotate the first node's squash with its worker-registry index
 * so `compileActivationKernel` can generate the correct WGSL switch, then restore
 * the original value.
 *
 * We only set the index during compilation because `uploadNetworkToGPU` uses an
 * empty supported-activation set in its eligibility check and would otherwise
 * reject networks whose nodes carry any index. Restoring the original value
 * keeps the mutation scoped to this seam.
 *
 * @param network - Network whose first node squash will be temporarily annotated.
 * @returns A context object with the resolved index and a `restore()` callback.
 * @throws Error when the first node has no squash or it is not a built-in worker
 *   activation.
 */
function prepareActivationContext(network: Network): {
  index: number;
  restore: () => void;
} {
  const firstNode = network.nodes[0];
  const squash = firstNode.squash as
    ((value: number, derivate?: boolean) => number) | undefined;

  if (!squash) {
    throw new Error('activateGPU: first node has no squash function');
  }

  const squashWithIndex = squash as {
    index?: number;
  } & ((value: number, derivate?: boolean) => number);
  const savedIndex = squashWithIndex.index;
  const index = resolveActivationIndex(squash);

  if (index === undefined) {
    throw new Error(
      'activateGPU: first node uses an activation that is not in the worker registry',
    );
  }

  squashWithIndex.index = index;

  return {
    index,
    restore: () => {
      squashWithIndex.index = savedIndex;
    },
  };
}

/**
 * Build the bind group that wires the six struct-packed kernel buffers into
 * the pipeline layout. The bind group can be reused across activations as long
 * as the underlying buffers are the same.
 */
function createActivationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  bufferSet: GPUBufferSet,
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      {
        binding: GPU_BUFFER_BINDING.connections,
        resource: { buffer: bufferSet.connections },
      },
      {
        binding: GPU_BUFFER_BINDING.nodes,
        resource: { buffer: bufferSet.nodes },
      },
      {
        binding: GPU_BUFFER_BINDING.outputs,
        resource: { buffer: bufferSet.outputs },
      },
      {
        binding: GPU_BUFFER_BINDING.params,
        resource: { buffer: bufferSet.params },
      },
      {
        binding: GPU_BUFFER_BINDING.topoLevels,
        resource: { buffer: bufferSet.topoLevels },
      },
      {
        binding: GPU_BUFFER_BINDING.inStart,
        resource: { buffer: bufferSet.inStart },
      },
    ],
  });
}

/**
 * Dispatch the activation kernel once per topological level.
 *
 * The params uniform carries the current level, total node count, connection
 * count, and output-node start index. Threads for nodes that do not belong to
 * the current level early-exit, so the same global dispatch size can be reused
 * for every level while still guaranteeing that all source values are available
 * from previous levels.
 */
async function dispatchActivationKernel(
  device: GPUDevice,
  bufferSet: GPUBufferSet,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  outputNodeCount: number,
): Promise<void> {
  const levelCount = bufferSet.topoLevelCount;
  const workgroupCount = Math.ceil(
    bufferSet.nodeCount / ACTIVATION_WORKGROUP_SIZE,
  );
  const params = new Uint32Array(4);
  params[1] = bufferSet.nodeCount;
  params[2] = bufferSet.connectionCount;
  params[3] = bufferSet.nodeCount - outputNodeCount;

  for (let level = 1; level < levelCount; level++) {
    params[0] = level;
    device.queue.writeBuffer(bufferSet.params, 0, params);

    const commandEncoder = device.createCommandEncoder({
      label: `network_activation_level_${level}`,
    });
    const pass = commandEncoder.beginComputePass({
      label: `network_activation_pass_level_${level}`,
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();

    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
}

/**
 * Copy the output-node slice of the GPU output buffer to a mappable staging
 * buffer, await the mapping, and return a detached Float32Array copy.
 */
async function readOutputValues(
  device: GPUDevice,
  network: Network,
  bufferSet: GPUBufferSet,
): Promise<Float32Array> {
  const outputNodeCount = network.output;
  const outputByteLength = outputNodeCount * Float32Array.BYTES_PER_ELEMENT;
  const outputStartOffset =
    (bufferSet.nodeCount - outputNodeCount) * Float32Array.BYTES_PER_ELEMENT;

  const stagingBuffer = device.createBuffer({
    label: 'network_outputs_staging',
    size: outputByteLength,
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  const copyEncoder = device.createCommandEncoder({
    label: 'network_outputs_copy',
  }) as unknown as GPUCommandEncoderCopy;
  copyEncoder.copyBufferToBuffer(
    bufferSet.outputs,
    outputStartOffset,
    stagingBuffer,
    0,
    outputByteLength,
  );
  device.queue.submit([copyEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await stagingBuffer.mapAsync(GPU_MAP_MODE_READ);
  const mappedRange = stagingBuffer.getMappedRange();
  const output = new Float32Array(mappedRange.slice(0));
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return output;
}
