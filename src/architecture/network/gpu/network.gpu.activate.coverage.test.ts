import Network from '../network';
import { ACTIVATION_FUNCTIONS } from '../../../multithreading/multi.utils';
import {
  activateGPU,
  activateGPUWithFreshState,
  ensureNetworkGPUState,
  encodeActivationKernel,
} from './network.gpu.activate';
import * as capability from './network.gpu.capability';
import { createMockGPUDevice } from './__mocks__/gpu.mock';
import type { GPUBufferSet } from './network.gpu.buffer';

type ActivationFunction = (value: number, derivate?: boolean) => number;

describe('network.gpu.activate coverage', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('activateGPU input validation', () => {
    it('throws when the network has no nodes', async () => {
      const fakeNetwork = {
        nodes: [],
        gates: [],
        selfconns: [],
        connections: [],
        input: 2,
        output: 1,
      } as unknown as Network;
      const device = createMockGPUDevice();

      await expect(activateGPU(device, fakeNetwork, [0, 0])).rejects.toThrow(
        'activateGPU: network has no nodes',
      );
    });

    it('throws when the input length does not match network.input', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();

      await expect(activateGPU(device, network, [0])).rejects.toThrow(
        'activateGPU: expected 2 inputs, received 1',
      );
    });

    it('accepts a Float32Array input', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();

      const result = await activateGPU(
        device,
        network,
        new Float32Array([0.1, 0.2]),
      );

      expect(result.length).toBe(network.output);
    });
  });

  describe('activateGPU activation lookup', () => {
    it('resolves the first node squash by strict identity when it is a registry function', async () => {
      const network = Network.createMLP(2, [3], 1);
      network.nodes[0].squash = ACTIVATION_FUNCTIONS[0] as ActivationFunction;
      const device = createMockGPUDevice();

      const result = await activateGPU(device, network, [0.1, 0.2]);

      expect(result.length).toBe(network.output);
    });

    it('resolves the first node squash by name when identity and symbol key are absent', async () => {
      const network = Network.createMLP(2, [3], 1);
      function sigmoidActivation(value: number) {
        return value;
      }
      network.nodes[0].squash =
        sigmoidActivation as unknown as ActivationFunction;
      const device = createMockGPUDevice();

      const result = await activateGPU(device, network, [0.1, 0.2]);

      expect(result.length).toBe(network.output);
    });

    it('resolves the first node squash by symbol key when present', async () => {
      const network = Network.createMLP(2, [3], 1);
      const ACTIVATION_KEY_SYMBOL = Symbol.for('neataptic.activation.key');
      const fakeSigmoid = Object.assign(
        function sigmoidActivation(value: number) {
          return value;
        },
        { [ACTIVATION_KEY_SYMBOL]: 'sigmoid' },
      );
      network.nodes[0].squash = fakeSigmoid as unknown as ActivationFunction;
      const device = createMockGPUDevice();

      const result = await activateGPU(device, network, [0.1, 0.2]);

      expect(result.length).toBe(network.output);
    });

    it('falls back to name lookup when the symbol key is unknown', async () => {
      const network = Network.createMLP(2, [3], 1);
      const ACTIVATION_KEY_SYMBOL = Symbol.for('neataptic.activation.key');
      const namedSigmoid = Object.assign(
        function sigmoidActivation(value: number) {
          return value;
        },
        { [ACTIVATION_KEY_SYMBOL]: 'unknownKey' },
      );
      network.nodes[0].squash = namedSigmoid as unknown as ActivationFunction;
      const device = createMockGPUDevice();

      const result = await activateGPU(device, network, [0.1, 0.2]);

      expect(result.length).toBe(network.output);
    });

    it('throws when the first node squash has no name or symbol key and does not match a built-in', async () => {
      const network = Network.createMLP(2, [3], 1);
      const noNameActivation = (() =>
        function (value: number) {
          return value + 1;
        })();
      network.nodes[0].squash =
        noNameActivation as unknown as ActivationFunction;
      const device = createMockGPUDevice();

      await expect(activateGPU(device, network, [0.1, 0.2])).rejects.toThrow(
        'activateGPU: first node uses an activation that is not in the worker registry',
      );
    });

    it('throws when the first node has no squash function', async () => {
      const network = Network.createMLP(2, [3], 1);
      (network.nodes[0] as { squash?: ActivationFunction }).squash = undefined;
      const device = createMockGPUDevice();

      await expect(activateGPU(device, network, [0.1, 0.2])).rejects.toThrow(
        'activateGPU: first node has no squash function',
      );
    });

    it('throws when the first node squash is not in the worker registry and does not match a built-in', async () => {
      const network = Network.createMLP(2, [3], 1);
      function customActivation(value: number) {
        return value + 1;
      }
      network.nodes[0].squash =
        customActivation as unknown as ActivationFunction;
      const device = createMockGPUDevice();

      await expect(activateGPU(device, network, [0.1, 0.2])).rejects.toThrow(
        'activateGPU: first node uses an activation that is not in the worker registry',
      );
    });

    it('throws when a candidate activation throws during behavior matching', async () => {
      const network = Network.createMLP(2, [3], 1);
      function throwingActivation(value: number) {
        if (value <= 0) {
          throw new Error('negative input');
        }
        return value;
      }
      network.nodes[0].squash =
        throwingActivation as unknown as ActivationFunction;
      const device = createMockGPUDevice();

      await expect(activateGPU(device, network, [0.1, 0.2])).rejects.toThrow(
        'activateGPU: first node uses an activation that is not in the worker registry',
      );
    });

    it('resolves a thin wrapper around a built-in activation by behaviour match', async () => {
      const network = Network.createMLP(2, [3], 1);
      const ACTIVATION_KEY_SYMBOL = Symbol.for('neataptic.activation.key');
      const wrappedLogistic = Object.assign(
        function logisticActivation2(value: number) {
          return ACTIVATION_FUNCTIONS[0](value);
        },
        { [ACTIVATION_KEY_SYMBOL]: 'logisticActivation2' },
      );
      for (const node of network.nodes) {
        if (typeof node.squash === 'function') {
          node.squash = wrappedLogistic as unknown as ActivationFunction;
        }
      }
      const device = createMockGPUDevice({ emulateNetwork: network });
      const cpuOutput = network.activate([0.1, 0.2]);

      const gpuOutput = await activateGPU(device, network, [0.1, 0.2]);

      const maxAbsoluteDifference = Math.max(
        ...cpuOutput.map((value, index) =>
          Math.abs(value - (gpuOutput[index] ?? Number.POSITIVE_INFINITY)),
        ),
      );
      expect(maxAbsoluteDifference).toBeLessThanOrEqual(1e-6);
    });
  });

  describe('Network.activate GPU branch', () => {
    it('returns emulated CPU outputs when useGPU is true and a device is attached', async () => {
      const network = Network.createMLP(2, [3], 1);
      const inputs = [0.5, -0.5];
      const expected = new Float32Array(network.clone().activate(inputs));
      network.gpuDevice = createMockGPUDevice({ emulateNetwork: network });

      const result = await network.activate(inputs, { useGPU: true });

      expect(result).toEqual(expected);
    });

    it('uses the CPU path when the options bag omits useGPU', () => {
      const network = Network.createMLP(2, [3], 1);
      const inputs = [0.5, -0.5];
      const expected = network.activate(inputs);

      const result = network.activate(inputs, { training: false });

      expect(result).toEqual(expected);
    });
  });
  describe('activateGPU state cache', () => {
    it('destroys the previous buffer set when the device changes between activations', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device1 = createMockGPUDevice();
      const device2 = createMockGPUDevice();

      await activateGPU(device1, network, [0.1, 0.2]);
      const firstBuffers = [...device1.recorded.buffers];

      await activateGPU(device2, network, [0.1, 0.2]);

      expect(
        firstBuffers.every(
          (buffer) =>
            ((buffer as unknown as { destroy: jest.Mock }).destroy.mock?.calls
              .length ?? 0) === 1,
        ),
      ).toBe(true);
    });

    it('reuses the pipeline, shader module, bind group layout, and persistent buffers on the second activation', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      const isPersistentBuffer = (buffer: { label?: string }) =>
        buffer.label !== 'network_outputs_staging';

      network.getConnectionSlab();
      await activateGPU(device, network, [0.1, 0.2]);
      const persistentBufferCountAfterFirstCall =
        device.recorded.buffers.filter(isPersistentBuffer).length;

      await activateGPU(device, network, [0.1, 0.2]);

      expect(
        device.recorded.pipelines.length === 1 &&
          device.recorded.shaderModules.length === 1 &&
          device.recorded.bindGroupLayouts.length === 1 &&
          device.recorded.buffers.filter(isPersistentBuffer).length ===
            persistentBufferCountAfterFirstCall,
      ).toBe(true);
    });
  });

  describe('activateGPU eligibility', () => {
    it('throws when the network is not eligible for GPU inference', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(false);

      await expect(activateGPU(device, network, [0.1, 0.2])).rejects.toThrow(
        'activateGPU: network is not eligible for GPU inference',
      );
    });
  });

  describe('activateGPUWithFreshState input validation', () => {
    it('throws when the network is not eligible for GPU inference', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(false);

      await expect(
        activateGPUWithFreshState(device, network, [0.1, 0.2]),
      ).rejects.toThrow(
        'activateGPUWithFreshState: network is not eligible for GPU inference',
      );
    });

    it('throws when the network has no nodes', async () => {
      const fakeNetwork = {
        nodes: [],
        gates: [],
        selfconns: [],
        connections: [],
        input: 2,
        output: 1,
      } as unknown as Network;
      const device = createMockGPUDevice();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);

      await expect(
        activateGPUWithFreshState(device, fakeNetwork, [0, 0]),
      ).rejects.toThrow('activateGPUWithFreshState: network has no nodes');
    });

    it('throws when the input length does not match network.input', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();

      await expect(
        activateGPUWithFreshState(device, network, [0]),
      ).rejects.toThrow(
        'activateGPUWithFreshState: expected 2 inputs, received 1',
      );
    });

    it('accepts a Float32Array input', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();

      const result = await activateGPUWithFreshState(
        device,
        network,
        new Float32Array([0.1, 0.2]),
      );

      expect(result.length).toBe(network.output);
    });
  });

  describe('encodeActivationKernel', () => {
    it('throws when a level bind group is missing', () => {
      const bufferSet = {
        nodeCount: 4,
        topoLevelCount: 3,
      } as unknown as GPUBufferSet;
      const pipeline = {} as unknown as GPUComputePipeline;
      const commandEncoder = {} as unknown as GPUCommandEncoder;
      const levelBindGroups: (GPUBindGroup | undefined)[] = [
        undefined,
        undefined,
        undefined,
      ];

      expect(() =>
        encodeActivationKernel(
          commandEncoder,
          bufferSet,
          pipeline,
          levelBindGroups,
        ),
      ).toThrow('activateGPU: missing bind group for level 1');
    });

    it('uses per-level workgroup counts when provided', () => {
      const bufferSet = {
        nodeCount: 4,
        topoLevelCount: 3,
      } as unknown as GPUBufferSet;
      const pipeline = {} as unknown as GPUComputePipeline;
      const dispatchCalls: number[] = [];
      const pass = {
        setPipeline: jest.fn(),
        setBindGroup: jest.fn(),
        dispatchWorkgroups: jest.fn((x: number) => {
          dispatchCalls.push(x);
        }),
        end: jest.fn(),
      };
      const commandEncoder = {
        beginComputePass: jest.fn(() => pass),
      } as unknown as GPUCommandEncoder;
      const levelBindGroups: (GPUBindGroup | undefined)[] = [
        undefined,
        {} as unknown as GPUBindGroup,
        {} as unknown as GPUBindGroup,
      ];
      const levelWorkgroupCounts = [0, 7, 9];

      encodeActivationKernel(
        commandEncoder,
        bufferSet,
        pipeline,
        levelBindGroups,
        levelWorkgroupCounts,
      );

      expect(dispatchCalls).toEqual([7, 9]);
    });

    it('falls back to the global workgroup count when per-level counts are omitted', () => {
      const bufferSet = {
        nodeCount: 4,
        topoLevelCount: 3,
      } as unknown as GPUBufferSet;
      const pipeline = {} as unknown as GPUComputePipeline;
      const dispatchCalls: number[] = [];
      const pass = {
        setPipeline: jest.fn(),
        setBindGroup: jest.fn(),
        dispatchWorkgroups: jest.fn((x: number) => {
          dispatchCalls.push(x);
        }),
        end: jest.fn(),
      };
      const commandEncoder = {
        beginComputePass: jest.fn(() => pass),
      } as unknown as GPUCommandEncoder;
      const levelBindGroups: (GPUBindGroup | undefined)[] = [
        undefined,
        {} as unknown as GPUBindGroup,
        {} as unknown as GPUBindGroup,
      ];

      encodeActivationKernel(
        commandEncoder,
        bufferSet,
        pipeline,
        levelBindGroups,
      );

      expect(dispatchCalls).toEqual([1, 1]);
    });
  });

  describe('bind group layout cache', () => {
    it('shares one layout across networks with different activation functions', async () => {
      const device = createMockGPUDevice();
      const networkA = Network.createMLP(2, [3], 1);
      const networkB = Network.createMLP(2, [3], 1);
      networkB.nodes[0].squash = ACTIVATION_FUNCTIONS[1] as ActivationFunction;

      await activateGPU(device, networkA, [0.1, 0.2]);
      await activateGPU(device, networkB, [0.1, 0.2]);

      expect(device.recorded.bindGroupLayouts.length).toBe(1);
    });
  });

  describe('output staging buffer lifecycle', () => {
    it('recreates the output staging buffer when the cached buffer was destroyed', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();

      await activateGPU(device, network, [0.1, 0.2]);
      const stagingBuffer = device.recorded.buffers.find(
        (buffer) => buffer.label === 'network_outputs_staging',
      );
      (stagingBuffer as unknown as { destroyed: boolean }).destroyed = true;

      await activateGPU(device, network, [0.1, 0.2]);

      expect(
        device.recorded.buffers.filter(
          (buffer) => buffer.label === 'network_outputs_staging',
        ).length,
      ).toBe(2);
    });

    it('returns early when there is no per-device staging buffer cache', async () => {
      const network = Network.createMLP(2, [3], 1);
      const deviceA = createMockGPUDevice();
      const deviceB = createMockGPUDevice();
      const { state } = ensureNetworkGPUState(deviceA, network);
      state.device = createMockGPUDevice();

      const result = ensureNetworkGPUState(deviceB, network);

      expect(result).toEqual(
        expect.objectContaining({
          state: expect.any(Object),
          pipeline: expect.any(Object),
        }),
      );
    });

    it('tolerates a missing cache entry when the device changes', async () => {
      const network = Network.createMLP(2, [3], 1);
      const deviceA = createMockGPUDevice();
      const deviceB = createMockGPUDevice();
      ensureNetworkGPUState(deviceA, network);
      (network as unknown as { output: number }).output = 100;

      const result = ensureNetworkGPUState(deviceB, network);

      expect(result).toEqual(
        expect.objectContaining({
          state: expect.any(Object),
          pipeline: expect.any(Object),
        }),
      );
    });

    it('destroys the cached output staging buffer when the device changes', async () => {
      const network = Network.createMLP(2, [3], 1);
      const deviceA = createMockGPUDevice();
      const deviceB = createMockGPUDevice();

      await activateGPU(deviceA, network, [0.1, 0.2]);
      const stagingBuffer = deviceA.recorded.buffers.find(
        (buffer) => buffer.label === 'network_outputs_staging',
      ) as unknown as GPUBuffer | undefined;
      expect(stagingBuffer).toBeDefined();
      const destroySpy = jest.spyOn(stagingBuffer!, 'destroy');

      ensureNetworkGPUState(deviceB, network);

      expect(destroySpy).toHaveBeenCalledTimes(1);
    });

    it('destroys a cached output staging buffer with the exact matching byte size', async () => {
      const network = Network.createMLP(2, [3], 1);
      const deviceA = createMockGPUDevice();
      const deviceB = createMockGPUDevice();

      await activateGPU(deviceA, network, [0.1, 0.2]);
      const destroySpies = deviceA.recorded.buffers
        .filter((buffer) => buffer.label === 'network_outputs_staging')
        .map((buffer) => jest.spyOn(buffer as unknown as GPUBuffer, 'destroy'));

      ensureNetworkGPUState(deviceB, network);

      expect(
        destroySpies.reduce((sum, spy) => sum + spy.mock.calls.length, 0),
      ).toBe(1);
    });

    it('takes no action when the cached staging byte size does not match', async () => {
      const network = Network.createMLP(2, [3], 1);
      const deviceA = createMockGPUDevice();
      const deviceB = createMockGPUDevice();

      await activateGPU(deviceA, network, [0.1, 0.2]);
      const destroySpies = deviceA.recorded.buffers
        .filter((buffer) => buffer.label === 'network_outputs_staging')
        .map((buffer) => jest.spyOn(buffer as unknown as GPUBuffer, 'destroy'));

      (network as unknown as { output: number }).output = 2;
      ensureNetworkGPUState(deviceB, network);

      expect(
        destroySpies.reduce((sum, spy) => sum + spy.mock.calls.length, 0),
      ).toBe(0);
    });
  });

  describe('createLevelBindGroups guard coverage', () => {
    it('throws when a level params buffer is missing during fresh-state activation', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      jest.spyOn(device, 'createBuffer').mockImplementation((descriptor) => {
        const { label, size, usage } = descriptor as GPUBufferDescriptor;
        if (label?.startsWith('network_activation_params_level_')) {
          return undefined as unknown as GPUBuffer;
        }
        return {
          label,
          size,
          usage,
          destroy: jest.fn(),
          mapAsync: jest.fn(async () => undefined),
          getMappedRange: jest.fn(() => new ArrayBuffer(size)),
          unmap: jest.fn(),
        } as unknown as GPUBuffer;
      });
      jest.spyOn(device.queue, 'writeBuffer').mockImplementation(() => {
        /* no-op so undefined params buffers do not crash the mock */
      });

      await expect(
        activateGPUWithFreshState(device, network, [0.1, 0.2]),
      ).rejects.toThrow('activateGPU: missing bind group for level 1');
    });
  });
});
