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
 * Real profiling on an RTX 4070 shows that GPU inference is not universally
 * faster. For a single small network, the dominant cost is synchronization:
 * a 64-node network spends 75.5% of its wall time waiting for the GPU and only
 * 13.2% preparing CPU-side data. The measured `mapAsync()` round trip after
 * `queue.submit()` is 6–25 ms depending on network size, while the actual GPU
 * compute work is usually well under 1 ms. The bottleneck is the readback
 * barrier, not ALU throughput. At 4k nodes the picture flips: GPU compute is
 * the majority of useful work, CPU preparation is 44.9% of wall time, and the
 * GPU path is clearly faster than the CPU path. Batching is the best way to
 * amortize the fixed synchronization cost locally.
 *
 * WebGPU exposes three timelines — content (CPU), device (GPU internal), and
 * queue (submitted work) — and a buffer-mapping state machine:
 * `unmapped → pending map → mapped`. `mapAsync()` resolves on the content
 * timeline only after all previously submitted operations on that buffer's
 * queue timeline have completed. That guarantee makes an explicit
 * `device.queue.onSubmittedWorkDone()` barrier redundant for readback and is
 * intentionally omitted here.
 *
 * The wider WebGPU/ML ecosystem uses a different shape to avoid this trap.
 * TensorFlow.js pools GPU buffers in a `Map<string, GPUBuffer[]>` keyed by
 * `${size}_${usage}` through `acquireBuffer` / `releaseBuffer` and keeps tensors
 * GPU-resident during inference; readback happens only when the caller
 * explicitly calls `tensor.data()`. burn's `burn-wgpu` compute server does the
 * same with a size-keyed handle pool and batches dispatches into a single
 * `queue.submit()`. NeatapticTS still reads back every activation, which is the
 * anomaly relative to that pattern. The recommended migration is:
 *
 * 1. Keep per-agent node/output buffers GPU-resident across activations.
 * 2. Pool staging readback buffers by size, as this module already does.
 * 3. Use a ring of 2–3 staging buffers so the CPU can map/read frame N while
 *    frame N+1 is being copied into another buffer, eliminating the blocking
 *    wait between the GPU copy and the CPU read.
 * 4. Only call `mapAsync()` when an external consumer actually needs the
 *    output values.
 *
 * This path caches the topology buffers, the compiled pipeline, and one output
 * staging buffer per output size, which removes buffer churn and recompilation
 * overhead. The dominant remaining cost is per-call readback, which the batched
 * activation path amortizes through `skipUpload` and repeated `iterations`,
 * but does not eliminate. For a deeper walkthrough of the measured bottleneck
 * and the optimization strategies, see the
 * [WebGPU Performance Guide](../../../docs/webgpu-performance-guide.md).
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
 * @see [WebGPU buffer mapping](https://www.w3.org/TR/webgpu/#buffer-mapping)
 * in the W3C WebGPU specification.
 * @see [WebGPU buffer map states](https://www.w3.org/TR/webgpu/#enumdef-gpubuffermapstate)
 *   for the normative `unmapped`/`pending`/`mapped` states.
 * @see [WebGPU programming-model timelines](https://www.w3.org/TR/webgpu/#programming-model-timelines)
 *   for the content, device, and queue timelines.
 * @see [TensorFlow.js BufferManager](https://github.com/tensorflow/tfjs/blob/7f5309fef0a47545e34049903dbdae0f97285f7e/tfjs-backend-webgpu/src/buffer_manager.ts)
 *   for the size/usage-keyed buffer pool.
 * @see [TensorFlow.js WebGPU backend](https://github.com/tensorflow/tfjs/blob/7f5309fef0a47545e34049903dbdae0f97285f7e/tfjs-backend-webgpu/src/backend_webgpu.ts)
 *   for GPU-resident tensors and deferred readback.
 * @see [toji.dev — WebGPU buffer uploads](https://toji.dev/webgpu-best-practices/buffer-uploads)
 *   for upload strategy guidance.
 * @see [webgpufundamentals — efficiently using mappable buffers](https://webgpufundamentals.org/webgpu/lessons/webgpu-copying-data.html#efficiently-using-mappable-buffers)
 *   for staging-ring best practices.
 * @see [burn-wgpu compute server](https://github.com/tracel-ai/burn/blob/v0.12.1/burn-wgpu/src/compute/server.rs)
 *   for a cross-library size-keyed handle pool.
 * @see [WebGPU Performance Guide](../../../docs/webgpu-performance-guide.md)
 *   for measured overhead, optimization strategies, and CPU/GPU crossover analysis.
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
  /** Per-level workgroup dispatch count sized to the actual nodes in each level. */
  levelWorkgroupCounts: number[];
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
 * Per-device cache for reusable output staging buffers.
 *
 * `readOutputValues` needs a small mappable buffer for every activation. Creating
 * and destroying that buffer for every call dominates CPU time when many agents
 * run on the same device, so this cache keeps one staging buffer per unique output
 * byte size and reuses it across activations.
 */
const activationOutputStagingBufferCache = new WeakMap<
  GPUDevice,
  Map<number, GPUBuffer>
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
 * the pipeline is replaced. When the network moves to a different `GPUDevice`,
 * the old per-device output staging buffer is destroyed so no cross-device
 * resource leaks outlive the buffers recorded for that device.
 *
 * @param device - WebGPU device used to run the forward kernel.
 * @param network - Network whose topology must be reflected in GPU buffers.
 * @returns Object containing the cached state and the activation pipeline.
 *
 * @example
 * ```ts
 * const { state, pipeline } = ensureNetworkGPUState(device, network);
 * // state.bufferSet holds the uploaded slab; pipeline can be reused across
 * // activations as long as the activation function does not change.
 * ```
 */
export function ensureNetworkGPUState(
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
    if (cached.device !== device) {
      const outputByteLength = network.output * Float32Array.BYTES_PER_ELEMENT;
      destroyOutputStagingBuffer(cached.device, outputByteLength);
    }
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
  const levelWorkgroupCounts = computeLevelWorkgroupCounts(bufferSet);

  const state: NetworkGPUState = {
    device,
    topologyHash,
    bufferSet,
    levelParamsBuffers,
    levelBindGroups,
    levelWorkgroupCounts,
  };

  networkGPUStateCache.set(network, state);
  return { state, pipeline };
}

/**
 * Compute the workgroup dispatch count for each topological level.
 *
 * The activation kernel launches one thread per node and each thread checks
 * its topological level against the dispatch level. Using a per-level dispatch
 * size avoids launching empty workgroups for levels with far fewer nodes than
 * the full network (for example a small output layer on a large hidden layer),
 * which removes the dominant source of dispatch overhead for multi-layer
 * perceptrons.
 *
 * @param bufferSet - Uploaded buffer metadata including `topoLevelsArray`.
 * @returns Array where index `level` is the number of workgroups to dispatch
 *   for that level. Index 0 is unused because input nodes are never dispatched.
 */
function computeLevelWorkgroupCounts(bufferSet: GPUBufferSet): number[] {
  const counts = new Array<number>(bufferSet.topoLevelCount).fill(0);
  for (let nodeIndex = 0; nodeIndex < bufferSet.nodeCount; nodeIndex += 1) {
    const level = bufferSet.topoLevelsArray[nodeIndex];
    counts[level] += 1;
  }
  for (let level = 0; level < counts.length; level += 1) {
    counts[level] = Math.max(
      1,
      Math.ceil(counts[level] / ACTIVATION_WORKGROUP_SIZE),
    );
  }
  return counts;
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
  const { bufferSet, levelBindGroups, levelWorkgroupCounts } = state;

  uploadDynamicNetworkBuffers(device, bufferSet, network);
  writeInputValuesToNodeStruct(device, bufferSet.nodes, typedInputs);

  const commandEncoder = device.createCommandEncoder({
    label: 'network_activation',
  });
  encodeActivationKernel(
    commandEncoder,
    bufferSet,
    pipeline,
    levelBindGroups,
    levelWorkgroupCounts,
  );

  return readOutputValues(commandEncoder, device, network, bufferSet);
}

/**
 * Run a single-network forward pass on the GPU without caching the buffer set.
 *
 * This is the concurrent-safe counterpart to `activateGPU()`. Every call
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

  const outputByteLength = network.output * Float32Array.BYTES_PER_ELEMENT;
  const freshOutputStagingBuffer = device.createBuffer({
    label: 'network_outputs_fresh_staging',
    size: outputByteLength,
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

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
  const levelWorkgroupCounts = computeLevelWorkgroupCounts(bufferSet);

  uploadDynamicNetworkBuffers(device, bufferSet, network);
  writeInputValuesToNodeStruct(device, bufferSet.nodes, typedInputs);

  const commandEncoder = device.createCommandEncoder({
    label: 'network_activation_fresh',
  });
  encodeActivationKernel(
    commandEncoder,
    bufferSet,
    pipeline,
    levelBindGroups,
    levelWorkgroupCounts,
  );

  const outputs = await readOutputValues(
    commandEncoder,
    device,
    network,
    bufferSet,
    freshOutputStagingBuffer,
  );
  freshOutputStagingBuffer.destroy();
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
 * @param paramsBuffer - Per-level params uniform buffer.
 */
function createActivationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  bufferSet: GPUBufferSet,
  paramsBuffer: GPUBuffer,
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
        resource: { buffer: paramsBuffer },
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
 * All levels are recorded into a single compute pass because dispatches within
 * the same pass execute in submission order and the WebGPU memory model makes
 * each level's node writes visible to the next level without requiring separate
 * compute-pass boundaries. This removes per-level pass overhead while keeping
 * the correct dependency ordering. The params uniform is supplied by a per-level
 * bind group, so no queue writes or intermediate submissions are needed between
 * levels. The caller submits the encoder; awaiting `mapAsync` on the output
 * staging buffer is sufficient synchronization for readback.
 *
 * @param commandEncoder - Encoder that will hold the compute pass.
 * @param bufferSet - Uploaded network slab buffers.
 * @param pipeline - Compiled activation pipeline.
 * @param levelBindGroups - Per-level bind groups from `createLevelBindGroups`.
 * @param levelWorkgroupCounts - Optional per-level workgroup dispatch counts.
 *   When provided, each level is dispatched with exactly the workgroups needed
 *   for its node count instead of the whole-network ceiling.
 */
export function encodeActivationKernel(
  commandEncoder: GPUCommandEncoder,
  bufferSet: GPUBufferSet,
  pipeline: GPUComputePipeline,
  levelBindGroups: (GPUBindGroup | undefined)[],
  levelWorkgroupCounts?: number[],
): void {
  const levelCount = bufferSet.topoLevelCount;
  const globalWorkgroupCount = Math.ceil(
    bufferSet.nodeCount / ACTIVATION_WORKGROUP_SIZE,
  );

  // Validate bind groups before starting the compute pass so that missing
  // groups surface with the intended error instead of a lower-level WebGPU
  // failure.
  for (let level = 1; level < levelCount; level += 1) {
    if (!levelBindGroups[level]) {
      throw new Error(`activateGPU: missing bind group for level ${level}`);
    }
  }

  const pass = commandEncoder.beginComputePass({
    label: 'network_activation_pass',
  });

  pass.setPipeline(pipeline);

  for (let level = 1; level < levelCount; level += 1) {
    const bindGroup = levelBindGroups[level]!;
    const workgroupCount =
      levelWorkgroupCounts?.[level] ?? globalWorkgroupCount;
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount);
  }

  pass.end();
}

/**
 * Return the per-device staging-buffer size map, creating it on first use.
 */
function getOutputStagingBufferCache(
  device: GPUDevice,
): Map<number, GPUBuffer> {
  let cache = activationOutputStagingBufferCache.get(device);
  if (!cache) {
    cache = new Map();
    activationOutputStagingBufferCache.set(device, cache);
  }
  return cache;
}

/**
 * Destroy a cached output staging buffer for the given device and byte size.
 *
 * Called when a network's cached GPU state is evicted to a different
 * `GPUDevice`. The staging buffer is tied to the device that created it, so
 * destroying it prevents the old device's resources from outliving the network
 * buffers that were recorded for that device.
 *
 * @param device - WebGPU device that owns the cached staging buffer.
 * @param byteLength - Exact byte size of the staging buffer to remove.
 */
function destroyOutputStagingBuffer(
  device: GPUDevice,
  byteLength: number,
): void {
  const cache = activationOutputStagingBufferCache.get(device);
  if (!cache) {
    return;
  }
  const buffer = cache.get(byteLength);
  if (buffer) {
    buffer.destroy();
    cache.delete(byteLength);
  }
}

/**
 * Fetch or create a reusable mappable staging buffer of the requested size.
 *
 * Buffers are keyed by byte size per device. A cached buffer is recreated only
 * when it has been destroyed, so activations that produce the same output size
 * (the common case for repeated evaluation of a cohort) reuse a single buffer
 * instead of allocating, mapping, and destroying one per call.
 *
 * @param device - WebGPU device that owns the staging buffer.
 * @param byteLength - Required staging buffer size in bytes.
 * @returns A GPU buffer with `MAP_READ | COPY_DST` usage.
 */
function getOrCreateOutputStagingBuffer(
  device: GPUDevice,
  byteLength: number,
): GPUBuffer {
  const cache = getOutputStagingBufferCache(device);
  let buffer = cache.get(byteLength);
  const isDestroyed =
    buffer !== undefined &&
    typeof (buffer as unknown as { destroyed?: boolean }).destroyed ===
      'boolean' &&
    (buffer as unknown as { destroyed: boolean }).destroyed;

  if (buffer === undefined || isDestroyed) {
    buffer = device.createBuffer({
      label: 'network_outputs_staging',
      size: byteLength,
      usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
    });
    cache.set(byteLength, buffer);
  }

  return buffer;
}

/**
 * Copy the output-node slice of the GPU output buffer to a mappable staging
 * buffer using the supplied command encoder, submit the encoder, and return a
 * detached Float32Array copy.
 *
 * The staging buffer is reused from the per-device output-staging cache rather
 * than allocated per call. After `queue.submit`, `mapAsync` transitions the
 * staging buffer from `unmapped` to `pending map`; the WebGPU implementation
 * completes the transition to `mapped` only after the queue operations that
 * target the buffer have finished. That makes `mapAsync` a sufficient
 * synchronization point for readback, so an explicit
 * `device.queue.onSubmittedWorkDone()` wait is unnecessary and is omitted to
 * reduce CPU-GPU round trips.
 *
 * This is the function where the readback bottleneck shows up in practice.
 * Project measurements on an RTX 4070 put the `queue.submit()` → `mapAsync()`
 * round trip at 6–25 ms, while the GPU compute itself is typically under 1 ms
 * for the network sizes this library evaluates. For a 64-node network the wait
 * accounts for about 75% of wall time. The mitigation recommended in WebGPU
 * best-practice guides is a ring of 2–3 staging buffers: the CPU maps and reads
 * frame N while the GPU copies the next frame into a different buffer, so the
 * CPU never blocks on the GPU's copy completion.
 *
 * WebGPU's three timelines explain why `mapAsync()` alone is enough. The copy
 * command executes on the queue timeline; `mapAsync()` resolves on the content
 * timeline only after all previously submitted work on that queue timeline has
 * finished. Adding `onSubmittedWorkDone()` would wait for the same signal a
 * second time from the CPU side without changing when the buffer becomes
 * mappable.
 *
 * The mapping state machine for the staging buffer is:
 *
 * ```mermaid
 * stateDiagram-v2
 *   [*] --> unmapped : createBuffer
 *   unmapped --> pendingMap : mapAsync(READ)
 *   pendingMap --> mapped : queue work finishes
 *   mapped --> unmapped : unmap()
 *   unmapped --> [*] : destroy()
 * ```
 *
 * @param commandEncoder - Encoder with the recorded activation passes.
 * @param device - WebGPU device that owns the staging buffer.
 * @param network - Network being evaluated; determines output node count.
 * @param bufferSet - Uploaded network slab buffers.
 * @param providedStagingBuffer - Optional dedicated staging buffer. When
 *   supplied, readback uses it directly instead of the shared per-device cache.
 *   Callers are responsible for destroying the supplied buffer. This avoids
 *   data races when multiple activations are in flight concurrently.
 * @returns Promise resolving to a detached copy of the output values.
 * @see [WebGPU buffer mapping](https://www.w3.org/TR/webgpu/#buffer-mapping)
 * @see [WebGPU buffer map states](https://www.w3.org/TR/webgpu/#enumdef-gpubuffermapstate)
 * @see [WebGPU programming-model timelines](https://www.w3.org/TR/webgpu/#programming-model-timelines)
 * @see [webgpufundamentals — efficiently using mappable buffers](https://webgpufundamentals.org/webgpu/lessons/webgpu-copying-data.html#efficiently-using-mappable-buffers)
 */
async function readOutputValues(
  commandEncoder: GPUCommandEncoder,
  device: GPUDevice,
  network: Network,
  bufferSet: GPUBufferSet,
  providedStagingBuffer?: GPUBuffer,
): Promise<Float32Array> {
  const outputNodeCount = network.output;
  const outputByteLength = outputNodeCount * Float32Array.BYTES_PER_ELEMENT;
  const outputStartOffset =
    (bufferSet.nodeCount - outputNodeCount) * Float32Array.BYTES_PER_ELEMENT;

  const stagingBuffer =
    providedStagingBuffer ??
    getOrCreateOutputStagingBuffer(device, outputByteLength);

  const copyEncoder = commandEncoder as unknown as GPUCommandEncoderCopy;
  copyEncoder.copyBufferToBuffer(
    bufferSet.outputs,
    outputStartOffset,
    stagingBuffer,
    0,
    outputByteLength,
  );
  device.queue.submit([commandEncoder.finish()]);

  await stagingBuffer.mapAsync(GPU_MAP_MODE_READ);
  const mappedRange = stagingBuffer.getMappedRange();
  const output = new Float32Array(mappedRange.slice(0));
  stagingBuffer.unmap();

  return output;
}
