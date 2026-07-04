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
  createConcurrentBufferSet,
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
  /** Per-level params uniforms so every dispatch level can use a fixed value. */
  levelParamsBuffers: (GPUBuffer | undefined)[];
  /** Per-level bind groups selecting the matching params buffer. */
  levelBindGroups: (GPUBindGroup | undefined)[];
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
 * The cache key is the generated WGSL source string. Topology-specific data
 * such as `topoLevels` and `inStart` are supplied through read-only storage
 * buffers at runtime rather than embedded in the shader, so the generated WGSL
 * depends only on the activation function switch and the fixed storage-buffer
 * layout. Identical WGSL implies an identical `GPUComputePipeline` and
 * `GPUPipelineLayout`, so the same compiled pipeline can drive any network that
 * shares the same activation function.
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
    destroyLevelParamsBuffers(cached.levelParamsBuffers);
  }

  const bufferSet = uploadNetworkToGPU(device, network);
  const pipeline = getOrCreateActivationPipeline(device, network);
  const levelParamsBuffers = createLevelParamsBuffers(
    device,
    bufferSet,
    network.output,
  );
  const levelBindGroups = createLevelBindGroups(
    device,
    pipeline,
    bufferSet,
    levelParamsBuffers,
  );

  const state: NetworkGPUState = {
    device,
    topologyHash,
    bufferSet,
    levelParamsBuffers,
    levelBindGroups,
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
  const { bufferSet, levelBindGroups } = state;

  uploadDynamicNetworkBuffers(device, bufferSet, network);
  writeInputValuesToNodeStruct(device, bufferSet.nodes, typedInputs);

  const commandEncoder = device.createCommandEncoder({
    label: 'network_activation',
  });
  encodeActivationKernel(commandEncoder, bufferSet, pipeline, levelBindGroups);

  return readOutputValues(commandEncoder, device, network, bufferSet);
}

/**
 * Run a single-network forward pass on the GPU without caching the buffer set.
 *
 * This is the concurrent-safe counterpart to {@link activateGPU}. Every call
 * uploads a fresh slab and creates a new bind group, so multiple requests that
 * target the same `Network` instance cannot overwrite each other's node or
 * output buffers. Pipelines are still shared through the per-device pipeline
 * cache, so identical topologies reuse a single compiled kernel.
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
 * const output = await activateGPUWithFreshState(device, network, [0.5, -0.2]);
 * ```
 */
export async function activateGPUWithFreshState(
  device: GPUDevice,
  network: Network,
  inputs: Float32Array | number[],
): Promise<Float32Array> {
  const supportedActivations = new Set<number>(
    SUPPORTED_ACTIVATION_INDICES as unknown as number[],
  );

  if (!canUseGPU(network, device, supportedActivations)) {
    throw new Error(
      'activateGPUWithFreshState: network is not eligible for GPU inference (gated topology, self-connection, unsupported activation, or missing device)',
    );
  }

  if (network.nodes.length === 0) {
    throw new Error('activateGPUWithFreshState: network has no nodes');
  }

  const typedInputs =
    inputs instanceof Float32Array ? inputs : new Float32Array(inputs);

  if (typedInputs.length !== network.input) {
    throw new Error(
      `activateGPUWithFreshState: expected ${network.input} inputs, received ${typedInputs.length}`,
    );
  }

  const bufferSet = createConcurrentBufferSet(device, network);
  const pipeline = getOrCreateActivationPipeline(device, network);
  const levelParamsBuffers = createLevelParamsBuffers(
    device,
    bufferSet,
    network.output,
  );
  const levelBindGroups = createLevelBindGroups(
    device,
    pipeline,
    bufferSet,
    levelParamsBuffers,
  );

  uploadDynamicNetworkBuffers(device, bufferSet, network);
  writeInputValuesToNodeStruct(device, bufferSet.nodes, typedInputs);

  const commandEncoder = device.createCommandEncoder({
    label: 'network_activation_fresh',
  });
  encodeActivationKernel(commandEncoder, bufferSet, pipeline, levelBindGroups);

  const outputs = await readOutputValues(
    commandEncoder,
    device,
    network,
    bufferSet,
  );
  destroyLevelParamsBuffers(levelParamsBuffers);
  destroyGPUBufferSet(device, bufferSet);
  return outputs;
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
 *
 * @param paramsBuffer - Optional params uniform buffer. When omitted, the
 *   buffer set's default params buffer is used.
 */
function createActivationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  bufferSet: GPUBufferSet,
  paramsBuffer?: GPUBuffer,
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
        resource: { buffer: paramsBuffer ?? bufferSet.params },
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
 * WebGPU buffer usage flag for uniform buffers.
 */
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;

/** Byte size of the per-dispatch params uniform buffer. */
const GPU_PARAMS_BYTES = 16;

/**
 * Create one params uniform buffer per topological level that needs a GPU
 * dispatch.
 *
 * Level 0 is skipped because input nodes are seeded directly by the caller.
 * Each buffer stores a fixed level index plus the dimension constants from the
 * uploaded buffer set so the kernel can early-exit threads that do not belong
 * to the current level. Keeping the params buffer immutable per level lets the
 * single-network path record every level into one command encoder without
 * serializing queue writes between dispatches.
 *
 * @param device - WebGPU device used to allocate buffers.
 * @param bufferSet - Uploaded network slab buffers.
 * @param outputNodeCount - Number of output nodes in the network.
 * @returns Array of params buffers indexed by level. Index 0 is `undefined`
 *   because level 0 is not dispatched.
 */
function createLevelParamsBuffers(
  device: GPUDevice,
  bufferSet: GPUBufferSet,
  outputNodeCount: number,
): (GPUBuffer | undefined)[] {
  const levelCount = bufferSet.topoLevelCount;
  const levelParamsBuffers: (GPUBuffer | undefined)[] = new Array(levelCount);
  levelParamsBuffers[0] = undefined;

  for (let level = 1; level < levelCount; level++) {
    const paramsBuffer = device.createBuffer({
      label: `network_activation_params_level_${level}`,
      size: GPU_PARAMS_BYTES,
      usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
    });
    const params = new Uint32Array([
      level,
      bufferSet.nodeCount,
      bufferSet.connectionCount,
      bufferSet.nodeCount - outputNodeCount,
    ]);
    device.queue.writeBuffer(paramsBuffer, 0, params);
    levelParamsBuffers[level] = paramsBuffer;
  }

  return levelParamsBuffers;
}

/**
 * Destroy params buffers created for per-level dispatch.
 *
 * @param levelParamsBuffers - Array of per-level params buffers.
 */
function destroyLevelParamsBuffers(
  levelParamsBuffers: (GPUBuffer | undefined)[],
): void {
  for (const buffer of levelParamsBuffers) {
    if (buffer !== undefined) {
      buffer.destroy();
    }
  }
}

/**
 * Create per-level bind groups that wire the kernel buffers and the matching
 * level params uniform together.
 *
 * @param device - WebGPU device used to create bind groups.
 * @param pipeline - Compiled activation pipeline.
 * @param bufferSet - Uploaded network slab buffers.
 * @param levelParamsBuffers - Per-level params buffers from
 *   `createLevelParamsBuffers`.
 * @returns Array of bind groups indexed by level. Index 0 is `undefined`.
 */
function createLevelBindGroups(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bufferSet: GPUBufferSet,
  levelParamsBuffers: (GPUBuffer | undefined)[],
): (GPUBindGroup | undefined)[] {
  const bindGroupLayout = pipeline.getBindGroupLayout(0);
  const levelBindGroups: (GPUBindGroup | undefined)[] = new Array(
    levelParamsBuffers.length,
  );

  for (let level = 1; level < levelParamsBuffers.length; level++) {
    const paramsBuffer = levelParamsBuffers[level];
    if (paramsBuffer !== undefined) {
      levelBindGroups[level] = createActivationBindGroup(
        device,
        bindGroupLayout,
        bufferSet,
        paramsBuffer,
      );
    }
  }

  return levelBindGroups;
}

/**
 * Record the activation kernel dispatches for every topological level into the
 * supplied command encoder.
 *
 * The params uniform is supplied by a per-level bind group, so no queue writes or
 * intermediate submissions are needed between levels. The caller must submit
 * the encoder and wait on `device.queue.onSubmittedWorkDone()` before reading
 * any output buffer.
 */
function encodeActivationKernel(
  commandEncoder: GPUCommandEncoder,
  bufferSet: GPUBufferSet,
  pipeline: GPUComputePipeline,
  levelBindGroups: (GPUBindGroup | undefined)[],
): void {
  const levelCount = bufferSet.topoLevelCount;
  const workgroupCount = Math.ceil(
    bufferSet.nodeCount / ACTIVATION_WORKGROUP_SIZE,
  );

  for (let level = 1; level < levelCount; level++) {
    const bindGroup = levelBindGroups[level];
    if (!bindGroup) {
      throw new Error(`activateGPU: missing bind group for level ${level}`);
    }

    const pass = commandEncoder.beginComputePass({
      label: `network_activation_pass_level_${level}`,
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
  }
}

/**
 * Copy the output-node slice of the GPU output buffer to a mappable staging
 * buffer using the supplied command encoder, submit the encoder, await GPU
 * completion, and return a detached Float32Array copy.
 */
async function readOutputValues(
  commandEncoder: GPUCommandEncoder,
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

  const copyEncoder = commandEncoder as unknown as GPUCommandEncoderCopy;
  copyEncoder.copyBufferToBuffer(
    bufferSet.outputs,
    outputStartOffset,
    stagingBuffer,
    0,
    outputByteLength,
  );
  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await stagingBuffer.mapAsync(GPU_MAP_MODE_READ);
  const mappedRange = stagingBuffer.getMappedRange();
  const output = new Float32Array(mappedRange.slice(0));
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return output;
}
