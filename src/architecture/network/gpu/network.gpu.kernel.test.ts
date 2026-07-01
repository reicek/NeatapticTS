import type Network from '../network';
import { createMockGPUDevice, type MockGPUDevice } from './__mocks__/gpu.mock';
import {
  buildGPUPipeline,
  createActivationKernel,
  createBindGroupLayout,
} from './network.gpu.kernel';

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
        /@compute[\s\S]*?workgroup_size\s*\(\s*64\b[\s\S]*?fn\s+forward\s*\(\s*\)/,
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

  describe('createBindGroupLayout', () => {
    it('is exported and callable', () => {
      expect(typeof createBindGroupLayout).toBe('function');
    });

    it('calls device.createBindGroupLayout exactly once', () => {
      createBindGroupLayout(device);

      expect(device.createBindGroupLayout).toHaveBeenCalledTimes(1);
    });

    it('creates seven storage-buffer entries', () => {
      createBindGroupLayout(device);
      const descriptor = (device.createBindGroupLayout as jest.Mock).mock
        .calls[0][0];

      expect(descriptor.entries.length).toBe(7);
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

    it('orders entries as weights, from, to, flags, outStart, outOrder, outputs', () => {
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
        { binding: 1, type: 'read-only-storage' },
        { binding: 2, type: 'read-only-storage' },
        { binding: 3, type: 'read-only-storage' },
        { binding: 4, type: 'read-only-storage' },
        { binding: 5, type: 'read-only-storage' },
        { binding: 6, type: 'storage' },
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
