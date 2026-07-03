import {
  createGPUBuffer,
  destroyGPUBufferSet,
  uploadNetworkToGPU,
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
const GPU_BUFFER_USAGE_STORAGE = 0x0080;

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
    weights: makeBuffer(),
    from: makeBuffer(),
    to: makeBuffer(),
    flags: makeBuffer(),
    inStart: makeBuffer(),
    inOrder: makeBuffer(),
    outputs: makeBuffer(),
    bias: makeBuffer(),
    topoLevels: makeBuffer(),
    params: makeBuffer(),
    topoLevelsArray: new Uint32Array(6),
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

  describe('GPU buffer contract constants', () => {
    it('exports the expected binding indices and count', () => {
      expect({
        binding: GPU_BUFFER_BINDING,
        count: GPU_BUFFER_BINDING_COUNT,
      }).toEqual({
        binding: {
          weights: 0,
          from: 1,
          to: 2,
          flags: 3,
          inStart: 4,
          inOrder: 5,
          outputs: 6,
          bias: 7,
          topoLevels: 8,
          params: 9,
        },
        count: 10,
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

    it('creates one GPUBuffer for every slab', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);

      expect(device.recorded.buffers.length).toBe(10);
    });

    it('sizes each buffer to the byteLength of its source slab', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);

      const nodeCount = network.nodes.length;
      const connectionCount = network.connections.length;
      const slab = network.getConnectionSlab();
      const expectedSizes = {
        weights: slab.weights.byteLength,
        from: slab.from.byteLength,
        to: slab.to.byteLength,
        flags: slab.flags.byteLength,
        inStart: (nodeCount + 1) * Uint32Array.BYTES_PER_ELEMENT,
        inOrder: connectionCount * Uint32Array.BYTES_PER_ELEMENT,
        outputs: nodeCount * Float32Array.BYTES_PER_ELEMENT,
        bias: nodeCount * Float32Array.BYTES_PER_ELEMENT,
        topoLevels: nodeCount * Uint32Array.BYTES_PER_ELEMENT,
        params: 2 * Uint32Array.BYTES_PER_ELEMENT,
      };
      const recordedSizes = {
        weights: device.recorded.buffers[0]?.size ?? 0,
        from: device.recorded.buffers[1]?.size ?? 0,
        to: device.recorded.buffers[2]?.size ?? 0,
        flags: device.recorded.buffers[3]?.size ?? 0,
        inStart: device.recorded.buffers[4]?.size ?? 0,
        inOrder: device.recorded.buffers[5]?.size ?? 0,
        outputs: device.recorded.buffers[6]?.size ?? 0,
        bias: device.recorded.buffers[7]?.size ?? 0,
        topoLevels: device.recorded.buffers[8]?.size ?? 0,
        params: device.recorded.buffers[9]?.size ?? 0,
      };

      expect(recordedSizes).toEqual(expectedSizes);
    });

    it('uses STORAGE and COPY_DST usage on every created buffer', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);

      const expectedUsage =
        GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST;
      expect(
        device.recorded.buffers.every(
          (buffer) => (buffer.usage & expectedUsage) === expectedUsage,
        ),
      ).toBe(true);
    });

    it('returns a GPUBufferSet with the created buffers and metadata', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      const bufferSet = uploadNetworkToGPU(device, network);

      expect(bufferSet).toEqual(
        expect.objectContaining({
          weights: device.recorded.buffers[0],
          from: device.recorded.buffers[1],
          to: device.recorded.buffers[2],
          flags: device.recorded.buffers[3],
          inStart: device.recorded.buffers[4],
          inOrder: device.recorded.buffers[5],
          outputs: device.recorded.buffers[6],
          bias: device.recorded.buffers[7],
          topoLevels: device.recorded.buffers[8],
          params: device.recorded.buffers[9],
          nodeCount: network.nodes.length,
          connectionCount: network.connections.length,
        }),
      );
    });

    it('records a queue.writeBuffer call for every slab', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);

      expect(device.recorded.writeBuffers.length).toBe(10);
    });

    it('produces deterministic buffer bookkeeping across repeated uploads', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);
      uploadNetworkToGPU(device, network);

      expect(device.recorded.buffers.length).toBe(20);
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

  });

  describe('destroyGPUBufferSet', () => {
    it('calls destroy() on every buffer in the set', () => {
      const device = createMockGPUDevice();
      const bufferSet = createFakeBufferSet();

      destroyGPUBufferSet(device, bufferSet);

      expect(
        [
          bufferSet.weights,
          bufferSet.from,
          bufferSet.to,
          bufferSet.flags,
          bufferSet.inStart,
          bufferSet.inOrder,
          bufferSet.outputs,
          bufferSet.bias,
          bufferSet.topoLevels,
          bufferSet.params,
        ].every(
          (buffer) =>
            ((buffer as unknown as { destroy: jest.Mock }).destroy.mock?.calls
              .length ?? 0) === 1,
        ),
      ).toBe(true);
    });
  });
});
