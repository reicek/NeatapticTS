/**
 * Public type surface for the WebGPU inference fast path.
 *
 * The canonical WebGPU interfaces are declared ambiently in
 * `gpu.types.d.ts` so they are available to both source files and automated
 * tests without a runtime dependency. This module re-exports the subset of
 * those names used by the device probe and capability predicate, giving
 * callers a single local import surface if they prefer explicit module
 * references over global declarations.
 */

/** Alias for the WebGPU adapter interface used by the GPU inference fast path. */
export type GPUAdapterType = GPUAdapter;

/**
 * Alias for the WebGPU device interface used by the GPU inference fast path.
 *
 * The local alias keeps imports explicit and avoids collision with any global
 * GPUDevice declaration variants across test and build environments.
 */
export type GPUDeviceType = GPUDevice;

/**
 * Alias for the WebGPU supported limits type used by the GPU inference fast path.
 */
export type GPUSupportedLimitsType = GPUSupportedLimits;

/**
 * Alias for the WebGPU adapter request options used by the GPU inference fast path.
 */
export type GPURequestAdapterOptionsType = GPURequestAdapterOptions;

/**
 * Stable WebGPU binding indices for the struct-packed network upload contract.
 *
 * The activation kernel binds six buffers: a struct array of connections, a
 * struct array of nodes, the per-node output buffer, a small per-dispatch
 * params uniform, the per-node topological level array, and the incoming-CSR
 * start-offset array. Packing fields into structs improves cache locality:
 * reading one connection fetches `from_node`, `to_node`, `weight`, and
 * `flags` from one contiguous 16-byte region, and reading one node fetches
 * `activation_state`, `derivative_state`, `error`, and `flags` from one
 * contiguous 16-byte region. The six-buffer layout still sits below the WebGPU
 * default `maxStorageBuffersPerShaderStage` limit, so the code does not request
 * a custom limit for that resource.
 *
 * @example
 * ```ts
 * const binding = GPU_BUFFER_BINDING.connections; // 0
 * ```
 */
export const GPU_BUFFER_BINDING = {
  /** Connections struct array `{ from_node, to_node, weight, flags }`. */
  connections: 0,
  /** Nodes struct array `{ activation_state, derivative_state, error, flags }`. */
  nodes: 1,
  /** Per-node activation/output buffer used for readback. */
  outputs: 2,
  /** Per-dispatch level and dimension uniform. */
  params: 3,
  /** Per-node topological level array. */
  topoLevels: 4,
  /** Incoming-CSR start-offset array (length `nodeCount + 1`). */
  inStart: 5,
} as const;

/**
 * Names of the buffers that participate in the GPU upload contract.
 *
 * Each name maps to a stable binding index in `GPU_BUFFER_BINDING`.
 */
export type GPUBufferName = keyof typeof GPU_BUFFER_BINDING;

/**
 * Number of bindings used by the GPU forward-pass kernel.
 *
 * This count matches the length of `GPU_BUFFER_BINDING` and the number
 * of entries in the kernel bind-group layout.
 */
export const GPU_BUFFER_BINDING_COUNT = 6;

/**
 * GPU-side buffer handles and metadata produced by uploading a network slab.
 *
 * The set contains exactly six WebGPU buffers bound to the compute kernel in
 * the order described by `GPU_BUFFER_BINDING`: the connection struct
 * array, the node struct array, the per-node output buffer, the per-dispatch
 * params uniform, the topological level array, and the incoming-CSR start-offset
 * array. The `inStart` buffer stores `nodeCount + 1` offsets into the connection
 * array so that the gather kernel can read each node's incoming edges as one
 * contiguous slice.
 *
 * The buffers are split by update frequency. Static buffers — `connections`
 * (when topology is stable), `topoLevels`, and `inStart` — are uploaded once
 * and reused. Dynamic buffers — `connections` weights and `nodes` biases — are
 * rewritten every activation through `queue.writeBuffer()`. The `outputs` buffer
 * is written by the GPU and then copied to a mappable staging buffer for
 * readback, following the rule that `mapAsync()` should be reserved for
 * readback while `writeBuffer()` handles CPU-to-GPU updates.
 *
 * The implementation also records `nodeCount` and `connectionCount` so the
 * compute pipeline can size its dispatches without re-reading CPU structures.
 * The `topoLevelsArray` is kept here because the CPU dispatch loop still needs
 * to know how many levels to launch.
 * @see [toji.dev — WebGPU buffer uploads](https://toji.dev/webgpu-best-practices/buffer-uploads)
 */
export interface GPUBufferSet {
  /** Struct array buffer `{ from_node, to_node, weight, flags }`. */
  connections: GPUBuffer;
  /** Struct array buffer `{ activation_state, derivative_state, error, flags }`. */
  nodes: GPUBuffer;
  /** Per-node output buffer for readback. */
  outputs: GPUBuffer;
  /** Per-dispatch params uniform. */
  params: GPUBuffer;
  /** Per-node topological level array. */
  topoLevels: GPUBuffer;
  /** Incoming-CSR start-offset array (length `nodeCount + 1`). */
  inStart: GPUBuffer;
  /** Number of nodes in the uploaded network. */
  nodeCount: number;
  /** Number of connections in the uploaded network. */
  connectionCount: number;
  /** Topological level assigned to every node. */
  topoLevelsArray: Uint32Array;
  /** Number of distinct topological levels (max level + 1). */
  topoLevelCount: number;
}

/**
 * Pipeline cache scoped to one WebGPU device.
 *
 * Maps a deterministic topology key to the compiled compute pipeline so that
 * identical network topologies share one pipeline even when their weights differ.
 */
export type GPUActivationPipelineCache = Map<string, GPUComputePipeline>;

/**
 * Deterministic string key that identifies a network topology for the pipeline
 * cache.
 */
export type GPUKernelTopologyKey = string;
