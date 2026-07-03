import Network from '../network';
import { dispatchActivation } from './network.gpu.fallback';
import { createMockGPUDevice } from './__mocks__/gpu.mock';

describe('network.gpu.fallback', () => {
  describe('dispatchActivation', () => {
    it('falls back to CPU activate when device is null', async () => {
      const network = Network.createMLP(2, [3], 1);
      const inputs = [0.5, -0.5];
      const expected = network.activate(inputs);

      const result = await dispatchActivation(network, inputs, null);

      expect(result).toEqual(new Float32Array(expected));
    });

    it('falls back to CPU activate when device is lost', async () => {
      const network = Network.createMLP(2, [3], 1);
      const inputs = [0.5, -0.5];
      const device = createMockGPUDevice();
      device.fakeLose();
      const expected = network.activate(inputs);

      const result = await dispatchActivation(network, inputs, device);

      expect(result).toEqual(new Float32Array(expected));
    });

    it('falls back to CPU activate for an ineligible network', async () => {
      const network = Network.createMLP(2, [3], 1);
      const gaterNode = network.nodes[4];
      const connection = network.connections[0];
      network.gate(gaterNode, connection);
      const inputs = [0.5, -0.5];
      const device = createMockGPUDevice();
      // Evaluate on an independent clone so the CPU fallback path starts from
      // the same cleared state as the expected reference.
      const expected = network.clone().activate(inputs);

      const result = await dispatchActivation(network, inputs, device);

      expect(result).toEqual(new Float32Array(expected));
    });

    it('clears gpuDevice when the bound device reports lost', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      network.gpuDevice = device;
      device.fakeLose();
      await device.lost;

      expect(network.gpuDevice).toBeUndefined();
    });

    it('short-circuits reassigning the same gpuDevice', () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      const thenSpy = jest.spyOn(device.lost, 'then');
      network.gpuDevice = device;
      network.gpuDevice = device;

      expect(thenSpy).toHaveBeenCalledTimes(1);
      thenSpy.mockRestore();
    });

    it('clears gpuDevice when assigned undefined', () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      network.gpuDevice = device;
      network.gpuDevice = undefined;

      expect(network.gpuDevice).toBeUndefined();
    });

    it('does not clear a replacement device when an older device reports lost', async () => {
      const network = Network.createMLP(2, [3], 1);
      const deviceA = createMockGPUDevice();
      const deviceB = createMockGPUDevice();
      network.gpuDevice = deviceA;
      network.gpuDevice = deviceB;
      deviceA.fakeLose();
      await deviceA.lost;

      expect(network.gpuDevice).toBe(deviceB);
    });

    it('GPU path returns a zero placeholder that fails parity with CPU output', async () => {
      const network = Network.createMLP(2, [3], 1);
      const inputs = [0.5, -0.5];
      const device = createMockGPUDevice();

      const result = await dispatchActivation(network, inputs, device);

      expect(result).toEqual(new Float32Array(network.output).fill(0));
    });
  });
});
