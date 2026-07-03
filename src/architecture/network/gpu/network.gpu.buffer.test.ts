import {
  createGPUBuffer,
  createGPUUniformBuffer,
  destroyGPUBufferSet,
  uploadNetworkToGPU,
  uploadDynamicNetworkBuffers,
  type GPUBufferSet,
} from './network.gpu.buffer';
import {
  GPU_BUFFER_BINDING,
  GPU_BUFFER_BINDING_COUNT,
} from './network.gpu.types';
import Network from '../network';
import * as capability from './network.gpu.capability';
import { createMockGPUDevice } from './__mocks__/gpu.mock';

// WebGPU buffer usage constants are not available at runtime in the test
// environment, so we mirror the spec values here for explicit assertions.
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_COPY_SRC = 0x0004;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;

function createEligibleMLP(): Network {
  const network = Network.createMLP(2, [3], 1);
  network.activate([0.1, 0.2]);
  return network;
}

function createFakeBufferSet(): GPUBufferSet {
  function makeBuffer(): GPUBuffer {
    return {
      destroy: jest.fn(),
      label: 'fake',
      size: 0,
      usage: 0,
      mapAsync: jest.fn(),
      getMappedRange: jest.fn(),
      unmap: jest.fn(),
    } as unknown as GPUBuffer;
  }

  return {
    connections: makeBuffer(),
    nodes: makeBuffer(),
    outputs: makeBuffer(),
    params: makeBuffer(),
    topoLevelsArray: new Uint32Array(6),
    topoLevelCount: 1,
    nodeCount: 6,
    connectionCount: 9,
  };
}

describe('network.gpu.buffer', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('createGPUBuffer', () => {
    it.each([[NaN], [Infinity], [-1]])(
      'rejects non-finite/negative byteLength (%s)',
      (byteLength) => {
        const device = createMockGPUDevice();

        expect(() => createGPUBuffer(device, byteLength, 'bad')).toThrow(
          /Invalid GPU buffer size for "bad"/,
        );
      },
    );

    it('rejects sizes above maxStorageBufferBindingSize', () => {
      const device = createMockGPUDevice({ maxStorageBufferBindingSize: 100 });

      expect(() => createGPUBuffer(device, 101, 'big-storage')).toThrow(
        /exceeds maxStorageBufferBindingSize 100/,
      );
    });

    it('rejects sizes above maxBufferSize', () => {
      const device = createMockGPUDevice({
        maxStorageBufferBindingSize: 200,
        maxBufferSize: 100,
      });

      expect(() => createGPUBuffer(device, 101, 'big-buffer')).toThrow(
        /exceeds maxBufferSize 100/,
      );
    });

    it('defaults usage to STORAGE | COPY_DST', () => {
      const device = createMockGPUDevice();
      const expectedUsage =
        GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST;

      createGPUBuffer(device, 16, 'default-usage');

      expect(device.recorded.buffers[0]?.usage).toBe(expectedUsage);
    });
  });

  describe('createGPUUniformBuffer', () => {
    it.each([[NaN], [Infinity], [-1]])(
      'rejects non-finite/negative byteLength (%s)',
      (byteLength) => {
        const device = createMockGPUDevice();

        expect(() => createGPUUniformBuffer(device, byteLength, 'bad')).toThrow(
          /Invalid GPU buffer size for "bad"/,
        );
      },
    );

    it('rejects sizes above maxUniformBufferBindingSize', () => {
      const device = createMockGPUDevice({ maxUniformBufferBindingSize: 100 });

      expect(() => createGPUUniformBuffer(device, 101, 'big-uniform')).toThrow(
        /exceeds maxUniformBufferBindingSize 100/,
      );
    });

    it('rejects sizes above maxBufferSize', () => {
      const device = createMockGPUDevice({
        maxUniformBufferBindingSize: 200,
        maxBufferSize: 100,
      });

      expect(() => createGPUUniformBuffer(device, 101, 'big-buffer')).toThrow(
        /exceeds maxBufferSize 100/,
      );
    });

    it('creates a buffer with UNIFORM | COPY_DST usage', () => {
      const device = createMockGPUDevice();
      const expectedUsage =
        GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST;

      createGPUUniformBuffer(device, 16, 'uniform-usage');

      expect(device.recorded.buffers[0]?.usage).toBe(expectedUsage);
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

  describe('uploadNetworkToGPU', () => {
    it('throws when the network is not GPU eligible', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(false);

      expect(() => uploadNetworkToGPU(device, network)).toThrow(
        /not eligible for GPU|GPU eligibility/i,
      );
    });

    it('creates one GPUBuffer for every struct buffer', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);

      expect(device.recorded.buffers.length).toBe(4);
    });

    it('sizes each buffer to the byteLength of its struct array', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);

      const nodeCount = network.nodes.length;
      const connectionCount = network.connections.length;
      const expectedSizes = {
        connections: connectionCount * 16,
        nodes: nodeCount * 16,
        outputs: nodeCount * Float32Array.BYTES_PER_ELEMENT,
        params: 4 * Uint32Array.BYTES_PER_ELEMENT,
      };
      const recordedSizes = {
        connections: device.recorded.buffers[0]?.size ?? 0,
        nodes: device.recorded.buffers[1]?.size ?? 0,
        outputs: device.recorded.buffers[2]?.size ?? 0,
        params: device.recorded.buffers[3]?.size ?? 0,
      };

      expect(recordedSizes).toEqual(expectedSizes);
    });

    it('uses STORAGE and COPY_DST usage on data buffers and UNIFORM on params', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);

      const [connections, nodes, outputs, params] = device.recorded.buffers;
      const storageUsage = GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST;
      const uniformUsage = GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST;

      expect(connections?.usage).toBe(storageUsage);
      expect(nodes?.usage).toBe(storageUsage);
      expect(outputs?.usage).toBe(storageUsage | GPU_BUFFER_USAGE_COPY_SRC);
      expect(params?.usage).toBe(uniformUsage);
    });

    it('returns a GPUBufferSet with the created buffers and metadata', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      const bufferSet = uploadNetworkToGPU(device, network);

      expect(bufferSet).toEqual(
        expect.objectContaining({
          connections: device.recorded.buffers[0],
          nodes: device.recorded.buffers[1],
          outputs: device.recorded.buffers[2],
          params: device.recorded.buffers[3],
          nodeCount: network.nodes.length,
          connectionCount: network.connections.length,
        }),
      );
    });

    it('records a queue.writeBuffer call for every struct buffer', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);

      expect(device.recorded.writeBuffers.length).toBe(4);
    });

    it('produces deterministic buffer bookkeeping across repeated uploads', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);
      uploadNetworkToGPU(device, network);

      expect(device.recorded.buffers.length).toBe(8);
    });
    it('uploads a network with no connections', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      network.activate([0.1, 0.2]);
      network.connections = [];
      network._slabDirty = true;
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      const bufferSet = uploadNetworkToGPU(device, network);

      expect(bufferSet.connectionCount).toBe(0);
    });

    it('falls back to zero when node state or error responsibility is undefined', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      const node = network.nodes[0] as unknown as {
        state: number | undefined;
        error: { responsibility: number | undefined };
      };
      node.state = undefined;
      node.error.responsibility = undefined;

      const bufferSet = uploadNetworkToGPU(device, network);
      const floats = new Float32Array(
        (bufferSet.nodes as unknown as { __data: ArrayBuffer }).__data,
      );

      expect({
        state: floats[0],
        responsibility: floats[2],
      }).toEqual({
        state: 0,
        responsibility: 0,
      });
    });
  });

  describe('destroyGPUBufferSet', () => {
    it('calls destroy() on every buffer in the set', () => {
      const device = createMockGPUDevice();
      const bufferSet = createFakeBufferSet();

      destroyGPUBufferSet(device, bufferSet);

      expect(
        [
          bufferSet.connections,
          bufferSet.nodes,
          bufferSet.outputs,
          bufferSet.params,
        ].every(
          (buffer) =>
            ((buffer as unknown as { destroy: jest.Mock }).destroy.mock?.calls
              .length ?? 0) === 1,
        ),
      ).toBe(true);
    });
  });

  describe('resolveStableNodeTieBreak fallback branches', () => {
    it('falls back to node.index when geneId is missing', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      for (const node of network.nodes.slice(0, network.input)) {
        (node as unknown as { geneId?: number }).geneId = undefined;
      }

      uploadNetworkToGPU(device, network);

      expect(device.recorded.buffers.length).toBe(4);
    });

    it('falls back to MAX_SAFE_INTEGER when both geneId and index are missing', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      for (const node of network.nodes.slice(0, network.input)) {
        const mutableNode = node as unknown as {
          geneId?: number;
          index?: number;
        };
        mutableNode.geneId = undefined;
        mutableNode.index = undefined;
      }

      uploadNetworkToGPU(device, network);

      expect(device.recorded.buffers.length).toBe(4);
    });
  });

  describe('uploadDynamicNetworkBuffers', () => {
    it('writes updated weights and node data into the existing buffers', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      const bufferSet = uploadNetworkToGPU(device, network);
      const previousWriteCount = device.recorded.writeBuffers.length;

      uploadDynamicNetworkBuffers(device, bufferSet, network);

      expect(device.recorded.writeBuffers.length).toBe(previousWriteCount + 2);
    });
  });
});
