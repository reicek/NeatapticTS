import type Network from '../network';
import { createMockGPUDevice, type MockGPUDevice } from './__mocks__/gpu.mock';
import {
  buildGPUPipeline,
  compileActivationKernel,
  createActivationKernel,
  createBindGroupLayout,
  SUPPORTED_ACTIVATION_INDICES as ExportedSupportedIndices,
} from './network.gpu.kernel';
import {
  GPU_BUFFER_BINDING,
  GPU_BUFFER_BINDING_COUNT,
} from './network.gpu.types';

// GPUShaderStage is not available at runtime in the jsdom test environment,
// so we mirror the COMPUTE stage bit here for explicit visibility assertions.
const GPU_SHADER_STAGE_COMPUTE = 0x0004;

// First-kernel activation subset, indexed into the worker registry in
// src/multithreading/multi.utils.ts ACTIVATION_FUNCTIONS.
const SUPPORTED_ACTIVATION_INDICES = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13];

function createFakeNetwork(activationIndex: number): Network {
  return {
    nodes: [{ squash: Object.assign(() => 0, { index: activationIndex }) }],
    connections: [],
    getConnectionSlab: () => ({
      from: new Uint32Array(0),
      to: new Uint32Array(0),
      weights: new Float32Array(0),
      flags: new Uint8Array(0),
    }),
  } as unknown as Network;
}

function createFakeShaderModule(): GPUShaderModule {
  return {
    getCompilationInfo: jest.fn(async () => ({ messages: [] })),
  } as unknown as GPUShaderModule;
}

describe('network.gpu.kernel', () => {
  let device: MockGPUDevice;

  beforeEach(() => {
    device = createMockGPUDevice();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('createActivationKernel', () => {
    it('is exported and callable', () => {
      expect(typeof createActivationKernel).toBe('function');
    });

    it('returns a non-empty WGSL source string', () => {
      const network = createFakeNetwork(4);
      const source = createActivationKernel(network);

      expect(source.length).toBeGreaterThan(0);
    });

    it('includes a @compute forward entry point with workgroup_size(64)', () => {
      const network = createFakeNetwork(4);
      const source = createActivationKernel(network);

      expect(source).toMatch(
        /@compute[\s\S]*?workgroup_size\s*\(\s*64\b[\s\S]*?fn\s+forward\s*\(\s*@builtin\(global_invocation_id\)\s+global_invocation_id:\s*vec3<u32>\s*\)/,
      );
    });

    it('includes a switch statement for activation dispatch', () => {
      const network = createFakeNetwork(4);
      const source = createActivationKernel(network);

      expect(source).toMatch(/switch\s*\(/);
    });

    it('covers every supported activation index from worker registry', () => {
      const network = createFakeNetwork(4);
      const source = createActivationKernel(network);
      const foundIndices: number[] = [];
      const casePattern = /case\s+(\d+)\s*:/g;
      let match = casePattern.exec(source);

      while (match !== null) {
        foundIndices.push(parseInt(match[1], 10));
        match = casePattern.exec(source);
      }

      expect([...new Set(foundIndices)].sort((a, b) => a - b)).toEqual(
        SUPPORTED_ACTIVATION_INDICES,
      );
    });

    it('throws for unsupported activation index', () => {
      const network = createFakeNetwork(99);

      expect(() => createActivationKernel(network)).toThrow(
        /activation index 99 is not supported/i,
      );
    });

    it('throws when the network has no nodes', () => {
      const network = { nodes: [], connections: [] } as unknown as Network;

      expect(() => createActivationKernel(network)).toThrow(
        /network topology is not eligible for gpu activation kernel/i,
      );
    });

    it('throws when the first node has no indexed squash function', () => {
      const network = {
        nodes: [{ squash: undefined }],
        connections: [],
      } as unknown as Network;

      expect(() => createActivationKernel(network)).toThrow(
        /network topology is not eligible for gpu activation kernel/i,
      );
    });

    it('throws when the first node squash has no activation index', () => {
      const network = {
        nodes: [{ squash: () => 0 }],
        connections: [],
      } as unknown as Network;

      expect(() => createActivationKernel(network)).toThrow(
        /network topology is not eligible for gpu activation kernel/i,
      );
    });
  });

  describe('compileActivationKernel', () => {
    it('returns the compute pipeline created by the device', () => {
      const network = createFakeNetwork(4);

      const pipeline = compileActivationKernel(device, network);

      expect(pipeline).toBe(
        (device.createComputePipeline as jest.Mock).mock.results[0].value,
      );
    });

    it('creates a shader module from generated WGSL', () => {
      const network = createFakeNetwork(4);

      compileActivationKernel(device, network);

      expect(device.recorded.shaderModules.length).toBe(1);
    });

    it('creates a bind group layout for the kernel buffers', () => {
      const network = createFakeNetwork(4);

      compileActivationKernel(device, network);

      expect(device.createBindGroupLayout).toHaveBeenCalledTimes(1);
    });

    it('reuses the cached pipeline for an identical topology', () => {
      const network = createFakeNetwork(4);

      compileActivationKernel(device, network);
      compileActivationKernel(device, network);

      expect({
        pipelines: device.recorded.pipelines.length,
        shaders: device.recorded.shaderModules.length,
        layouts: device.recorded.bindGroupLayouts.length,
      }).toEqual({
        pipelines: 1,
        shaders: 1,
        layouts: 1,
      });
    });

    it('creates a new pipeline when the topology differs', () => {
      const smallNetwork = createFakeNetwork(4);
      const largeNetwork = {
        nodes: [
          { squash: Object.assign(() => 0, { index: 4 }) },
          { squash: Object.assign(() => 0, { index: 4 }) },
        ],
        connections: [],
        getConnectionSlab: () => ({
          from: new Uint32Array(0),
          to: new Uint32Array(0),
          weights: new Float32Array(0),
          flags: new Uint8Array(0),
        }),
      } as unknown as Network;

      compileActivationKernel(device, smallNetwork);
      compileActivationKernel(device, largeNetwork);

      expect(device.recorded.pipelines.length).toBe(2);
    });

    it('includes CSR from/to arrays in the topology key', () => {
      const networkA = {
        nodes: [{ squash: Object.assign(() => 0, { index: 4 }) }],
        connections: [{ id: 'a' }],
        _connFrom: new Uint32Array([1]),
        _connTo: new Uint32Array([2]),
        getConnectionSlab: () => ({
          from: new Uint32Array([1]),
          to: new Uint32Array([2]),
          weights: new Float32Array([0]),
          flags: new Uint8Array([0]),
        }),
      } as unknown as Network;
      const networkB = {
        nodes: [{ squash: Object.assign(() => 0, { index: 4 }) }],
        connections: [{ id: 'b' }],
        _connFrom: new Uint32Array([2]),
        _connTo: new Uint32Array([1]),
        getConnectionSlab: () => ({
          from: new Uint32Array([2]),
          to: new Uint32Array([1]),
          weights: new Float32Array([0]),
          flags: new Uint8Array([0]),
        }),
      } as unknown as Network;

      compileActivationKernel(device, networkA);
      compileActivationKernel(device, networkB);

      expect(device.recorded.pipelines.length).toBe(2);
    });

    it('throws for unsupported activation index', () => {
      const network = createFakeNetwork(99);

      expect(() => compileActivationKernel(device, network)).toThrow(
        /activation index 99 is not supported/i,
      );
    });

    it('throws when the network has no nodes', () => {
      const network = { nodes: [], connections: [] } as unknown as Network;

      expect(() => compileActivationKernel(device, network)).toThrow(
        /network topology is not eligible for gpu activation kernel/i,
      );
    });
  });

  describe('GPU buffer contract constants', () => {
    it('exports the expected binding indices and count', () => {
      expect({
        binding: GPU_BUFFER_BINDING,
        count: GPU_BUFFER_BINDING_COUNT,
      }).toEqual({
        binding: {
          connections: 0,
          nodes: 1,
          outputs: 2,
          params: 3,
        },
        count: 4,
      });
    });
  });

  describe('SUPPORTED_ACTIVATION_INDICES re-export', () => {
    it('exports the worker activation subset from the kernel module', () => {
      expect(ExportedSupportedIndices).toEqual([
        0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13,
      ]);
    });
  });

  describe('createBindGroupLayout', () => {
    it('is exported and callable', () => {
      expect(typeof createBindGroupLayout).toBe('function');
    });

    it('calls device.createBindGroupLayout exactly once', () => {
      createBindGroupLayout(device);

      expect(device.createBindGroupLayout).toHaveBeenCalledTimes(1);
    });

    it('creates four struct-buffer and uniform entries', () => {
      createBindGroupLayout(device);
      const descriptor = (device.createBindGroupLayout as jest.Mock).mock
        .calls[0][0];

      expect(descriptor.entries.length).toBe(4);
    });

    it('sets compute visibility on every entry', () => {
      createBindGroupLayout(device);
      const descriptor = (device.createBindGroupLayout as jest.Mock).mock
        .calls[0][0];

      expect(
        descriptor.entries.every(
          (entry: { visibility: number }) =>
            entry.visibility === GPU_SHADER_STAGE_COMPUTE,
        ),
      ).toBe(true);
    });

    it('orders entries as connections, nodes, outputs, params', () => {
      createBindGroupLayout(device);
      const descriptor = (device.createBindGroupLayout as jest.Mock).mock
        .calls[0][0];
      const observed = descriptor.entries.map(
        (entry: { binding: number; buffer: { type: string } }) => ({
          binding: entry.binding,
          type: entry.buffer.type,
        }),
      );

      expect(observed).toEqual([
        { binding: 0, type: 'read-only-storage' },
        { binding: 1, type: 'storage' },
        { binding: 2, type: 'storage' },
        { binding: 3, type: 'uniform' },
      ]);
    });
  });

  describe('buildGPUPipeline', () => {
    let shaderModule: GPUShaderModule;
    let bindGroupLayout: GPUBindGroupLayout;

    beforeEach(() => {
      shaderModule = createFakeShaderModule();
      bindGroupLayout = {} as unknown as GPUBindGroupLayout;
    });

    it('is exported and callable', () => {
      expect(typeof buildGPUPipeline).toBe('function');
    });

    it('creates a compute pipeline with the supplied shader module and bind group layout', () => {
      buildGPUPipeline(device, shaderModule, bindGroupLayout);

      expect(device.createComputePipeline).toHaveBeenCalledTimes(1);
    });

    it('uses entryPoint "forward" and a pipeline layout built from the bind group layout', () => {
      buildGPUPipeline(device, shaderModule, bindGroupLayout);
      const descriptor = (device.createComputePipeline as jest.Mock).mock
        .calls[0][0];
      const pipelineLayout = (device.createPipelineLayout as jest.Mock).mock
        .results[0].value;

      expect(descriptor).toEqual(
        expect.objectContaining({
          layout: pipelineLayout,
          compute: expect.objectContaining({
            module: shaderModule,
            entryPoint: 'forward',
          }),
        }),
      );
    });
  });
});
