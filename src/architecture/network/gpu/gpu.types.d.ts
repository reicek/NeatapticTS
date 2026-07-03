/**
 * Minimal ambient WebGPU types for the GPU inference fast path.
 *
 * These declarations allow the GPU capability and device probes to compile
 * before the project pulls in a full WebGPU type package. They intentionally
 * expose only the surface exercised by the inference seam and its unit
 * doubles.
 */

declare global {
  interface GPUSupportedLimits {
    maxStorageBufferBindingSize?: number;
    maxBufferSize?: number;
    [key: string]: number | undefined;
  }

  interface GPUDeviceLostInfo {
    reason: 'destroyed' | 'unknown';
    message: string;
  }

  type GPUMapModeFlags = number;

  interface GPUBufferDescriptor {
    label?: string;
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }

  interface GPUBuffer {
    label?: string;
    size: number;
    usage: number;
    mapAsync(
      mode?: GPUMapModeFlags,
      offset?: number,
      size?: number,
    ): Promise<void>;
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    unmap(): void;
    destroy(): void;
  }

  // Intentionally empty marker interfaces used by the ambient WebGPU stub.
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface GPUBindGroupLayout {}
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface GPUBindGroup {}
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface GPUPipelineLayout {}

  interface GPUShaderModuleDescriptor {
    code: string;
    label?: string;
  }

  interface GPUShaderModule {
    getCompilationInfo(): Promise<{ messages: unknown[] }>;
  }

  interface GPUComputePipelineDescriptor {
    layout: GPUPipelineLayout | 'auto';
    compute: {
      module: GPUShaderModule;
      entryPoint: string;
    };
  }

  interface GPUComputePipeline {
    getBindGroupLayout(index: number): GPUBindGroupLayout;
  }

  interface GPUComputePassEncoder {
    setPipeline(pipeline: GPUComputePipeline): void;
    setBindGroup(index: number, bindGroup: GPUBindGroup): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    end(): void;
  }

  interface GPUCommandEncoder {
    beginComputePass(descriptor?: { label?: string }): GPUComputePassEncoder;
    finish(): GPUCommandBuffer;
  }

  // Empty marker interface used by the ambient WebGPU stub.
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface GPUCommandBuffer {}

  interface GPUQueue {
    writeBuffer(
      buffer: GPUBuffer,
      bufferOffset: number,
      data: ArrayBufferView | ArrayBuffer,
      dataOffset?: number,
      size?: number,
    ): void;
    submit(commandBuffers: GPUCommandBuffer[]): void;
    onSubmittedWorkDone(): Promise<void>;
  }

  interface GPUDevice {
    limits: GPUSupportedLimits;
    lost: Promise<GPUDeviceLostInfo>;
    queue: GPUQueue;
    destroy(): void;
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    createBindGroupLayout(descriptor?: unknown): GPUBindGroupLayout;
    createBindGroup(descriptor?: unknown): GPUBindGroup;
    createPipelineLayout(descriptor?: unknown): GPUPipelineLayout;
    createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
    createComputePipeline(
      descriptor: GPUComputePipelineDescriptor,
    ): GPUComputePipeline;
    createCommandEncoder(descriptor?: { label?: string }): GPUCommandEncoder;
  }

  interface GPUAdapter {
    limits: GPUSupportedLimits;
    requestDevice(descriptor?: unknown): Promise<GPUDevice>;
  }

  interface GPURequestAdapterOptions {
    powerPreference?: 'low-power' | 'high-performance';
    forceFallbackAdapter?: boolean;
  }

  type GPUTextureFormat = string;

  type WGSLLanguageFeatures = Iterable<string>;

  interface GPU {
    requestAdapter(
      options?: GPURequestAdapterOptions,
    ): Promise<GPUAdapter | null>;
    getPreferredCanvasFormat(): GPUTextureFormat;
    wgslLanguageFeatures: WGSLLanguageFeatures;
  }

  interface Navigator {
    gpu?: GPU;
  }

  interface WorkerNavigator {
    gpu?: GPU;
  }
}

export {};
