/**
 * Reusable WebGPU mock helpers for owner-local GPU tests.
 *
 * The returned devices are intentionally shallow: they record the calls the
 * implementation is expected to make without pulling in a real WebGPU backend.
 * Optionally, a mock device can emulate a CPU forward pass so that parity
 * tests can compare GPU read-back values against the CPU source of truth.
 */

import type Network from '../../network';
import { GPU_BUFFER_BINDING } from '../network.gpu.types';

/**
 * Context supplied to the optional per-dispatch output generator.
 */
export interface MockGenerateOutputContext {
  /** Zero-based index of this dispatch within the command encoder. */
  dispatchIndex: number;
  /** Total number of f32 elements that fit in the bound output buffer. */
  outputCount: number;
  /** Input slice uploaded by the caller (length equals network.input when emulating). */
  inputs: Float32Array;
}

/**
 * Optional callback that can write deterministic output values into the mock.
 */
export type MockGenerateOutput = (
  context: MockGenerateOutputContext,
) => Float32Array | number[];

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
  copyBuffers: Array<{
    source: GPUBuffer;
    sourceOffset: number;
    destination: GPUBuffer;
    destinationOffset: number;
    size: number;
  }>;
  mapAsyncCalls: Array<{ bufferLabel?: string }>;
}

export interface MockGPUDevice extends GPUDevice {
  recorded: MockGPURecordings;
  fakeLose(): void;
  __lost: boolean;
  /** When set, the compute pass emulates `network.activate()` for CPU-vs-GPU parity. */
  emulateNetwork?: Network;
  /** Optional deterministic generator that overrides the default zero output. */
  generateOutput?: MockGenerateOutput;
}

export interface MockGPUDeviceOptions {
  /** WebGPU limits to merge over the default mock limits. */
  limits?: Partial<GPUSupportedLimits>;
  /** Network whose `activate()` should be replayed on each dispatch. */
  emulateNetwork?: Network;
  /** Custom output generator used when emulateNetwork is not provided. */
  generateOutput?: MockGenerateOutput;
}

/**
 * Narrow a generic GPUBuffer to the mock's internal backing-store shape.
 */
interface MockBuffer {
  __data: ArrayBuffer;
  size: number;
}

function hasMockData(buffer: GPUBuffer): buffer is GPUBuffer & MockBuffer {
  return (
    '__data' in buffer &&
    (buffer as unknown as MockBuffer).__data instanceof ArrayBuffer
  );
}

/**
 * Read a bound buffer as a typed array, returning an empty array if the buffer
 * is not a mock-backed GPUBuffer.
 */
function readBoundFloatArray(
  entries: Array<{ binding: number; resource: { buffer: GPUBuffer } }>,
  binding: number,
): Float32Array {
  const entry = entries.find((e) => e.binding === binding);
  if (!entry || !hasMockData(entry.resource.buffer)) {
    return new Float32Array(0);
  }
  return new Float32Array(entry.resource.buffer.__data);
}

/**
 * Connection struct mirrored from the WGSL `Connection` layout.
 */
interface MockConnection {
  fromNode: number;
  toNode: number;
  weight: number;
  flags: number;
}

/**
 * Read the bound connection buffer as an array of connection structs.
 */
function readBoundConnections(
  entries: Array<{ binding: number; resource: { buffer: GPUBuffer } }>,
): MockConnection[] {
  const entry = entries.find(
    (e) => e.binding === GPU_BUFFER_BINDING.connections,
  );
  if (!entry || !hasMockData(entry.resource.buffer)) {
    return [];
  }

  const view = new DataView(entry.resource.buffer.__data);
  const connectionCount = entry.resource.buffer.size / 16;
  const connections: MockConnection[] = [];

  for (let i = 0; i < connectionCount; i++) {
    const offset = i * 16;
    connections.push({
      fromNode: view.getUint32(offset, true),
      toNode: view.getUint32(offset + 4, true),
      weight: view.getFloat32(offset + 8, true),
      flags: view.getUint32(offset + 12, true),
    });
  }

  return connections;
}

/**
 * Per-dispatch params mirrored from the WGSL `Params` layout.
 */
interface MockParams {
  level: number;
  nodeCount: number;
  connectionCount: number;
  outputStart: number;
}

/**
 * Read the bound params uniform as a typed struct.
 */
function readBoundParams(
  entries: Array<{ binding: number; resource: { buffer: GPUBuffer } }>,
): MockParams | undefined {
  const entry = entries.find((e) => e.binding === GPU_BUFFER_BINDING.params);
  if (!entry || !hasMockData(entry.resource.buffer)) {
    return undefined;
  }

  const view = new DataView(entry.resource.buffer.__data);
  return {
    level: view.getUint32(0, true),
    nodeCount: view.getUint32(4, true),
    connectionCount: view.getUint32(8, true),
    outputStart: view.getUint32(12, true),
  };
}

/**
 * Compute a simple topological level for every node from the connection list.
 *
 * Level 0 contains nodes with no incoming connections; every other node gets
 * one level above its highest predecessor. This is sufficient for the mock
 * because GPU-eligible networks are feed-forward and the real kernel already
 * skips disabled connections via its flags check.
 */
function computeMockLevels(
  connections: MockConnection[],
  nodeCount: number,
): Uint32Array {
  const levels = new Uint32Array(nodeCount);
  let changed = true;

  while (changed) {
    changed = false;
    for (const connection of connections) {
      if (
        connection.fromNode === connection.toNode ||
        connection.fromNode < 0 ||
        connection.fromNode >= nodeCount ||
        connection.toNode < 0 ||
        connection.toNode >= nodeCount
      ) {
        continue;
      }
      const candidate = levels[connection.fromNode] + 1;
      if (candidate > levels[connection.toNode]) {
        levels[connection.toNode] = candidate;
        changed = true;
      }
    }
  }

  return levels;
}

/**
 * Compute the pre-activation values for every non-input node.
 *
 * This mirrors the struct-packed gather-and-activate kernel: each node's
 * pre-activation value is the sum of incoming weights times the current source
 * activations (read from the bound node buffer) plus the node's bias. Input
 * nodes are never processed by a dispatch, so they retain the values uploaded
 * by the caller.
 */
function computeMockForwardPass(
  entries: Array<{ binding: number; resource: { buffer: GPUBuffer } }>,
  nodeCount: number,
): Float32Array {
  const nodes = readBoundFloatArray(entries, GPU_BUFFER_BINDING.nodes);
  const outputs = readBoundFloatArray(entries, GPU_BUFFER_BINDING.outputs);
  const connections = readBoundConnections(entries);

  const result = new Float32Array(nodeCount);
  const hasNodeBuffer = nodes.length > 0;

  for (let node = 0; node < nodeCount; node++) {
    let sum = nodes[node * 4 + 1] ?? 0;

    for (const connection of connections) {
      if (connection.toNode !== node) {
        continue;
      }
      if ((connection.flags & 1) === 0) {
        continue;
      }
      const sourceActivation = hasNodeBuffer
        ? (nodes[connection.fromNode * 4] ?? 0)
        : (outputs[connection.fromNode] ?? 0);
      sum += connection.weight * sourceActivation;
    }

    result[node] = sum;
  }

  return result;
}

/**
 * Build a fake WebGPU device that records every call made by the inference path.
 *
 * @param options - Optional limits, emulator network, or output generator.
 */
export function createMockGPUDevice(
  options: MockGPUDeviceOptions | Partial<GPUSupportedLimits> = {},
): MockGPUDevice {
  const normalized =
    'limits' in options ||
    'emulateNetwork' in options ||
    'generateOutput' in options
      ? (options as MockGPUDeviceOptions)
      : { limits: options as Partial<GPUSupportedLimits> };
  const { limits, emulateNetwork, generateOutput } = normalized;
  const recorded: MockGPURecordings = {
    buffers: [],
    shaderModules: [],
    bindGroupLayouts: [],
    pipelineLayouts: [],
    pipelines: [],
    submissions: [],
    writeBuffers: [],
    copyBuffers: [],
    mapAsyncCalls: [],
  };

  const defaultLimits = {
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxUniformBufferBindingSize: 64 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
  } as GPUSupportedLimits;

  const mergedLimits = { ...defaultLimits, ...limits } as GPUSupportedLimits;

  let resolveLost!: (info: GPUDeviceLostInfo) => void;
  const lostPromise = new Promise<GPUDeviceLostInfo>((resolve) => {
    resolveLost = resolve;
  });

  let dispatchIndex = 0;

  const device = {
    recorded,
    __lost: false,
    emulateNetwork,
    generateOutput,
    limits: mergedLimits,
    lost: lostPromise,
    queue: {
      writeBuffer: jest.fn(
        (
          buffer: GPUBuffer,
          offset: number,
          data: ArrayBufferView | ArrayBuffer,
          dataOffset?: number,
          size?: number,
        ) => {
          let byteLength: number;

          const bytesPerElement = ArrayBuffer.isView(data)
            ? ((data.constructor as unknown as { BYTES_PER_ELEMENT?: number })
                .BYTES_PER_ELEMENT ?? 1)
            : 1;

          if (ArrayBuffer.isView(data)) {
            const view = data as ArrayBufferView;
            const elementCount =
              size ??
              (data as unknown as { length?: number }).length ??
              view.byteLength / bytesPerElement;
            byteLength = elementCount * bytesPerElement;
          } else {
            byteLength = size ?? (data as ArrayBuffer).byteLength ?? 0;
          }

          recorded.writeBuffers.push({ buffer, offset, byteLength });

          if (hasMockData(buffer)) {
            let sourceOffsetBytes: number;
            let sourceBuffer: ArrayBuffer;

            if (ArrayBuffer.isView(data)) {
              const view = data as ArrayBufferView;
              sourceOffsetBytes =
                view.byteOffset + (dataOffset ?? 0) * bytesPerElement;
              sourceBuffer = view.buffer as ArrayBuffer;
            } else {
              sourceOffsetBytes = dataOffset ?? 0;
              sourceBuffer = data as ArrayBuffer;
            }

            const dst = new Uint8Array(buffer.__data, offset, byteLength);
            const src = new Uint8Array(
              sourceBuffer,
              sourceOffsetBytes,
              byteLength,
            );
            dst.set(src);
          }
        },
      ),
      submit: jest.fn(),
      onSubmittedWorkDone: jest.fn(async () => undefined),
    },
    destroy: jest.fn(),
    createBuffer: jest.fn((descriptor: GPUBufferDescriptor) => {
      const data = new ArrayBuffer(descriptor.size);
      const buffer = {
        label: descriptor.label,
        size: descriptor.size,
        usage: descriptor.usage,
        __data: data,
        mapAsync: jest.fn(async () => {
          recorded.mapAsyncCalls.push({ bufferLabel: descriptor.label });
        }),
        getMappedRange: jest.fn(() => data),
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
    createBindGroup: jest.fn(
      (descriptor?: {
        entries?: Array<{ binding: number; resource: { buffer: GPUBuffer } }>;
      }) => {
        return {
          entries: descriptor?.entries ?? [],
        } as unknown as GPUBindGroup;
      },
    ),
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
      let boundEntries: Array<{
        binding: number;
        resource: { buffer: GPUBuffer };
      }> = [];
      const pass = {
        setPipeline: jest.fn(),
        setBindGroup: jest.fn(
          (_index: number, group: { entries: typeof boundEntries }) => {
            boundEntries = group.entries ? [...group.entries] : [];
          },
        ),
        dispatchWorkgroups: jest.fn((x: number, y?: number, z?: number) => {
          recorded.submissions.push({
            workgroupCountX: x,
            workgroupCountY: y,
            workgroupCountZ: z,
          });

          const outputsEntry = boundEntries.find(
            (entry) => entry.binding === GPU_BUFFER_BINDING.outputs,
          );
          if (!outputsEntry || !hasMockData(outputsEntry.resource.buffer)) {
            dispatchIndex += 1;
            return;
          }

          const outBuffer = outputsEntry.resource
            .buffer as unknown as MockBuffer;
          const nodeCount = outBuffer.size / Float32Array.BYTES_PER_ELEMENT;

          if (emulateNetwork) {
            const inputCount = emulateNetwork.input;
            const outputCount = emulateNetwork.output;
            const nodesEntry = boundEntries.find(
              (entry) => entry.binding === GPU_BUFFER_BINDING.nodes,
            );
            const nodeBuffer =
              nodesEntry && hasMockData(nodesEntry.resource.buffer)
                ? (nodesEntry.resource.buffer as unknown as MockBuffer).__data
                : outBuffer.__data;
            const inputArray = new Float32Array(nodeBuffer, 0, inputCount);
            const cpuOutput = emulateNetwork.activate(Array.from(inputArray));
            const outputArray = new Float32Array(outBuffer.__data);
            for (let i = 0; i < outputCount; i++) {
              outputArray[nodeCount - outputCount + i] = cpuOutput[i];
            }
          } else if (generateOutput) {
            const preActivation = computeMockForwardPass(
              boundEntries,
              nodeCount,
            );
            const generated = generateOutput({
              dispatchIndex,
              outputCount: nodeCount,
              inputs: preActivation,
            });
            const generatedFloat =
              generated instanceof Float32Array
                ? generated
                : new Float32Array(generated);

            const params = readBoundParams(boundEntries);
            const connections = readBoundConnections(boundEntries);
            const levels =
              params && connections.length > 0
                ? computeMockLevels(connections, nodeCount)
                : new Uint32Array(nodeCount);
            const currentLevel = params?.level ?? 0;

            const nodesEntry = boundEntries.find(
              (entry) => entry.binding === GPU_BUFFER_BINDING.nodes,
            );
            const nodeArray =
              nodesEntry && hasMockData(nodesEntry.resource.buffer)
                ? new Float32Array(nodesEntry.resource.buffer.__data)
                : undefined;
            const outputArray = new Float32Array(outBuffer.__data);

            for (let node = 0; node < nodeCount; node++) {
              if (levels[node] !== currentLevel) {
                continue;
              }
              const value = generatedFloat[node] ?? 0;
              outputArray[node] = value;
              if (nodeArray) {
                nodeArray[node * 4] = value;
              }
            }
          }

          dispatchIndex += 1;
        }),
        end: jest.fn(),
      };
      return {
        beginComputePass: jest.fn(() => pass),
        copyBufferToBuffer: jest.fn(
          (
            source: GPUBuffer,
            sourceOffset: number,
            destination: GPUBuffer,
            destinationOffset: number,
            copySize: number,
          ) => {
            recorded.copyBuffers.push({
              source,
              sourceOffset,
              destination,
              destinationOffset,
              size: copySize,
            });

            if (hasMockData(source) && hasMockData(destination)) {
              new Uint8Array(
                destination.__data,
                destinationOffset,
                copySize,
              ).set(new Uint8Array(source.__data, sourceOffset, copySize));
            }
          },
        ),
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
