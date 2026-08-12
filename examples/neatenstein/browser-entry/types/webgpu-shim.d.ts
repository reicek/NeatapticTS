/**
 * Minimal WebGPU ambient type declarations for the neatenstein example.
 *
 * The full `@webgpu/types` package is not a dependency of this project.
 * These stub types satisfy the TypeScript compiler when transitively
 * importing the `neataptic` entry point (which pulls in GPU acceleration
 * modules that reference WebGPU types).  The stubs are intentionally
 * permissive (`any`) — the neatenstein example never instantiates or calls
 * WebGPU APIs; the types exist only so that type-checking of the import
 * chain succeeds.
 */

/* eslint-disable @typescript-eslint/no-explicit-any -- WebGPU shim stubs */

declare global {
  // Use permissive `any` stubs — the neatenstein example never calls WebGPU
  // APIs directly; these types exist solely to satisfy the TypeScript compiler
  // when transitively importing the `neataptic` entry point.
  type GPUAdapter = any;
  type GPUBindGroup = any;
  type GPUBindGroupLayout = any;
  type GPUBuffer = any;
  type GPUCommandEncoder = any;
  type GPUComputePipeline = any;
  type GPUDevice = any;
  type GPURequestAdapterOptions = any;
  type GPUShaderModule = any;
  type GPUSupportedLimits = any;
  interface Navigator {
    gpu: any;
  }
}

export {};
