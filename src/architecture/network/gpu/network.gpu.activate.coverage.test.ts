import Network from '../network';
import { ACTIVATION_FUNCTIONS } from '../../../multithreading/multi.utils';
import { activateGPU, activateGPUWithFreshState } from './network.gpu.activate';
import * as capability from './network.gpu.capability';
import { createMockGPUDevice } from './__mocks__/gpu.mock';

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
});
