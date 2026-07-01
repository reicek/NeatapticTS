import Network from '../network';
import * as capability from './network.gpu.capability';
import { createMockGPUDevice } from './__mocks__/gpu.mock';
import {
  destroyGPUBufferSet,
  uploadNetworkToGPU,
  type GPUBufferSet,
} from './network.gpu.buffer';

// WebGPU buffer usage constants are not available at runtime in the test
// environment, so we mirror the spec values here for explicit assertions.
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;

interface NetworkInternals {
  _outStart?: Uint32Array | null;
  _outOrder?: Uint32Array | null;
}

function createEligibleMLP(): Network {
  const network = Network.createMLP(2, [3], 1);
  network.activate([0.1, 0.2]);
  return network;
}

function getNetworkInternals(network: Network): NetworkInternals {
  return network as unknown as NetworkInternals;
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
    outStart: makeBuffer(),
    outOrder: makeBuffer(),
    outputs: makeBuffer(),
    nodeCount: 6,
    connectionCount: 9,
  };
}

describe('network.gpu.buffer', () => {
  afterEach(() => {
    jest.restoreAllMocks();
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

      expect(device.recorded.buffers.length).toBe(7);
    });

    it('sizes each buffer to the byteLength of its source slab', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);

      const slab = network.getConnectionSlab();
      const internals = getNetworkInternals(network);
      const expectedSizes = {
        weights: slab.weights.byteLength,
        from: slab.from.byteLength,
        to: slab.to.byteLength,
        flags: slab.flags.byteLength,
        outStart: (internals._outStart?.byteLength ?? 0),
        outOrder: (internals._outOrder?.byteLength ?? 0),
        outputs: network.nodes.length * Float32Array.BYTES_PER_ELEMENT,
      };
      const recordedSizes = {
        weights: device.recorded.buffers[0]?.size ?? 0,
        from: device.recorded.buffers[1]?.size ?? 0,
        to: device.recorded.buffers[2]?.size ?? 0,
        flags: device.recorded.buffers[3]?.size ?? 0,
        outStart: device.recorded.buffers[4]?.size ?? 0,
        outOrder: device.recorded.buffers[5]?.size ?? 0,
        outputs: device.recorded.buffers[6]?.size ?? 0,
      };

      expect(recordedSizes).toEqual(expectedSizes);
    });

    it('falls back to empty adjacency arrays when the CSR context is missing', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);
      const internals = getNetworkInternals(network);
      internals._outStart = null;
      internals._outOrder = null;

      uploadNetworkToGPU(device, network);

      expect({
        outStart: device.recorded.buffers[4]?.size ?? 0,
        outOrder: device.recorded.buffers[5]?.size ?? 0,
      }).toEqual({ outStart: 0, outOrder: 0 });
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
          outStart: device.recorded.buffers[4],
          outOrder: device.recorded.buffers[5],
          outputs: device.recorded.buffers[6],
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

      expect(device.recorded.writeBuffers.length).toBe(7);
    });

    it('produces deterministic buffer bookkeeping across repeated uploads', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      uploadNetworkToGPU(device, network);
      uploadNetworkToGPU(device, network);

      expect(device.recorded.buffers.length).toBe(14);
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
          bufferSet.outStart,
          bufferSet.outOrder,
          bufferSet.outputs,
        ].every(
          (buffer) =>
            ((buffer as unknown as { destroy: jest.Mock }).destroy.mock?.calls
              .length ?? 0) === 1,
        ),
      ).toBe(true);
    });
  });
});
