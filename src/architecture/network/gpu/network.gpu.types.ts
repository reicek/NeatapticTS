/**
 * Public type surface for the WebGPU inference fast path.
 *
 * The canonical WebGPU interfaces are declared ambiently in
 * `gpu.types.d.ts` so they are available to both source files and owner-local
 * tests without a runtime dependency. This module re-exports the subset of
 * those names used by the device probe and capability predicate, giving
 * callers a single local import surface if they prefer explicit module
 * references over global declarations.
 */

export type GPUAdapterType = GPUAdapter;
export type GPUDeviceType = GPUDevice;
export type GPUSupportedLimitsType = GPUSupportedLimits;
export type GPURequestAdapterOptionsType = GPURequestAdapterOptions;

/**
 * Stable WebGPU binding indices for the slab-to-GPU upload contract.
 *
 * The compute kernel's bind group layout and WGSL declarations must use these
 * exact indices so the shader reads the uploaded arrays in the order produced
 * by {@link uploadNetworkToGPU}. Keeping the mapping in one exported table
 * prevents drift between the upload path and the kernel.
 *
 * @example
 * ```ts
 * const binding = GPU_BUFFER_BINDING.weights; // 0
 * ```
 */
export const GPU_BUFFER_BINDING = {
  weights: 0,
  from: 1,
  to: 2,
  flags: 3,
  outStart: 4,
  outOrder: 5,
  outputs: 6,
} as const;

/**
 * Names of the slab buffers that participate in the GPU upload contract.
 *
 * Each name maps to a stable binding index in {@link GPU_BUFFER_BINDING}.
 */
export type GPUBufferName = keyof typeof GPU_BUFFER_BINDING;

/**
 * Number of storage-buffer bindings used by the GPU forward-pass kernel.
 *
 * This count matches the length of {@link GPU_BUFFER_BINDING} and the number
 * of entries in the kernel bind-group layout.
 */
export const GPU_BUFFER_BINDING_COUNT = 7;

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
