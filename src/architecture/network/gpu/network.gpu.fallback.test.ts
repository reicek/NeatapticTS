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

    it('GPU path returns a zero placeholder that fails parity with CPU output', async () => {
      const network = Network.createMLP(2, [3], 1);
      const inputs = [0.5, -0.5];
      const device = createMockGPUDevice();
      // Compute the CPU reference on an independent clone so recurrent/gated
      // state does not bleed into the GPU seam comparison.
      const expected = network.clone().activate(inputs);

      const result = await dispatchActivation(network, inputs, device);

      expect(result).toEqual(new Float32Array(expected));
    });
  });
});
