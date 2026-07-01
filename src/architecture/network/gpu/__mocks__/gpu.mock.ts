/**
 * Reusable WebGPU mock helpers for owner-local GPU tests.
 *
 * The returned devices are intentionally shallow: they record the calls the
 * implementation is expected to make without pulling in a real WebGPU backend.
 */

export interface MockGPURecordings {
  buffers: Array<{ label?: string; size: number; usage: number }>;
  shaderModules: string[];
  bindGroupLayouts: unknown[];
  pipelineLayouts: unknown[];
  pipelines: GPUComputePipelineDescriptor[];
  submissions: Array<{
    workgroupCountX: number;
    workgroupCountY?: number;
    workgroupCountZ?: number;
  }>;
  writeBuffers: Array<{
    buffer: GPUBuffer;
    offset: number;
    byteLength: number;
  }>;
  mapAsyncCalls: Array<{ bufferLabel?: string }>;
}

export interface MockGPUDevice extends GPUDevice {
  recorded: MockGPURecordings;
  fakeLose(): void;
  __lost: boolean;
}

/**
 * Build a fake WebGPU device that records every call made by the inference path.
 */
export function createMockGPUDevice(
  limits?: Partial<GPUSupportedLimits>,
): MockGPUDevice {
  const recorded: MockGPURecordings = {
    buffers: [],
    shaderModules: [],
    bindGroupLayouts: [],
    pipelineLayouts: [],
    pipelines: [],
    submissions: [],
    writeBuffers: [],
    mapAsyncCalls: [],
  };

  const defaultLimits = {
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
  } as GPUSupportedLimits;

  const mergedLimits = { ...defaultLimits, ...limits } as GPUSupportedLimits;

  let resolveLost!: (info: GPUDeviceLostInfo) => void;
  const lostPromise = new Promise<GPUDeviceLostInfo>((resolve) => {
    resolveLost = resolve;
  });

  const device = {
    recorded,
    __lost: false,
    limits: mergedLimits,
    lost: lostPromise,
    queue: {
      writeBuffer: jest.fn(
        (
          buffer: GPUBuffer,
          offset: number,
          data: ArrayBufferView | ArrayBuffer,
        ) => {
          const byteLength =
            (data as ArrayBufferView).byteLength ??
            (data as ArrayBuffer).byteLength ??
            0;
          recorded.writeBuffers.push({ buffer, offset, byteLength });
        },
      ),
      submit: jest.fn(),
      onSubmittedWorkDone: jest.fn(async () => undefined),
    },
    destroy: jest.fn(),
    createBuffer: jest.fn((descriptor: GPUBufferDescriptor) => {
      const buffer = {
        label: descriptor.label,
        size: descriptor.size,
        usage: descriptor.usage,
        mapAsync: jest.fn(async () => {
          recorded.mapAsyncCalls.push({ bufferLabel: descriptor.label });
        }),
        getMappedRange: jest.fn(() => new ArrayBuffer(descriptor.size)),
        unmap: jest.fn(),
        destroy: jest.fn(),
      } as unknown as GPUBuffer;
      recorded.buffers.push(buffer);
      return buffer;
    }),
    createBindGroupLayout: jest.fn((descriptor?: unknown) => {
      recorded.bindGroupLayouts.push(descriptor);
      return {} as unknown as GPUBindGroupLayout;
    }),
    createBindGroup: jest.fn(() => ({}) as unknown as GPUBindGroup),
    createPipelineLayout: jest.fn((descriptor?: unknown) => {
      recorded.pipelineLayouts.push(descriptor);
      return {} as unknown as GPUPipelineLayout;
    }),
    createShaderModule: jest.fn((descriptor: GPUShaderModuleDescriptor) => {
      recorded.shaderModules.push(descriptor.code as string);
      return {
        getCompilationInfo: jest.fn(async () => ({ messages: [] })),
      } as unknown as GPUShaderModule;
    }),
    createComputePipeline: jest.fn(
      (descriptor: GPUComputePipelineDescriptor) => {
        recorded.pipelines.push(descriptor);
        return {
          getBindGroupLayout: jest.fn(
            () => ({}) as unknown as GPUBindGroupLayout,
          ),
        } as unknown as GPUComputePipeline;
      },
    ),
    createCommandEncoder: jest.fn(() => {
      const pass = {
        setPipeline: jest.fn(),
        setBindGroup: jest.fn(),
        dispatchWorkgroups: jest.fn((x: number, y?: number, z?: number) => {
          recorded.submissions.push({
            workgroupCountX: x,
            workgroupCountY: y,
            workgroupCountZ: z,
          });
        }),
        end: jest.fn(),
      };
      return {
        beginComputePass: jest.fn(() => pass),
        finish: jest.fn(() => ({}) as unknown as GPUCommandBuffer),
      } as unknown as GPUCommandEncoder;
    }),
    fakeLose: function (this: MockGPUDevice) {
      this.__lost = true;
      resolveLost({
        reason: 'destroyed',
        message: 'mock lost',
      } as GPUDeviceLostInfo);
    },
  } as unknown as MockGPUDevice;

  return device;
}

/**
 * Build a fake `navigator` object that exposes a mock `gpu` property.
 */
export function createMockNavigatorGPU(options?: {
  adapter?: GPUAdapter | null;
  device?: GPUDevice | null;
}): Navigator & { gpu: GPU } {
  const adapter = options?.adapter ?? null;

  const gpu = {
    requestAdapter: jest.fn(
      async () => adapter,
    ) as unknown as GPU['requestAdapter'],
    getPreferredCanvasFormat: jest.fn(() => 'bgra8unorm' as GPUTextureFormat),
    wgslLanguageFeatures: new Set<string>() as unknown as WGSLLanguageFeatures,
  } as unknown as GPU;

  return { gpu } as unknown as Navigator & { gpu: GPU };
}
