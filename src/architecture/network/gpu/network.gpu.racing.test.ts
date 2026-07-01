import Network from '../network';
import { evaluateRacingGeneration } from './network.gpu.racing';
import { createMockGPUDevice } from './__mocks__/gpu.mock';

const DEFAULT_THRESHOLD = 4;

describe('network.gpu.racing', () => {
  describe('evaluateRacingGeneration', () => {
    it('GPU batch path returns a zero placeholder that fails parity with CPU output', async () => {
      const network = Network.createMLP(2, [3], 1);
      const batchSize = DEFAULT_THRESHOLD + 1;
      const networks = Array.from({ length: batchSize }, () => network);
      const inputMatrix = new Float32Array(batchSize * network.input).fill(0.5);
      const device = createMockGPUDevice();

      const result = await evaluateRacingGeneration(
        networks,
        inputMatrix,
        device,
        { gpuBatchThreshold: DEFAULT_THRESHOLD },
      );

      // Compute the CPU reference on independent clones so repeated activations
      // do not drift due to recurrent/gated state.
      const expectedRows = networks.map((_, index) => {
        const clone = network.clone();
        const start = index * network.input;
        return clone.activate(
          inputMatrix.subarray(start, start + network.input),
        );
      });
      const expected = new Float32Array(expectedRows.flat());

      expect(result).toEqual(expected);
    });

    it('returns an empty output matrix for an empty generation', async () => {
      const result = await evaluateRacingGeneration(
        [],
        new Float32Array(0),
        createMockGPUDevice(),
        { gpuBatchThreshold: DEFAULT_THRESHOLD },
      );

      expect(result).toEqual(new Float32Array(0));
    });

    it('uses the default GPU batch threshold when options are omitted', async () => {
      const network = Network.createMLP(2, [3], 1);
      const activateSpy = jest.spyOn(network, 'activate');
      const batchSize = 1;
      const networks = Array.from({ length: batchSize }, () => network);
      const inputMatrix = new Float32Array(batchSize * network.input).fill(0.5);
      const device = createMockGPUDevice();

      await evaluateRacingGeneration(networks, inputMatrix, device);

      expect(activateSpy).toHaveBeenCalledTimes(batchSize);
    });

    it('falls back to CPU per-network activation when batch size is below threshold', async () => {
      const network = Network.createMLP(2, [3], 1);
      const activateSpy = jest.spyOn(network, 'activate');
      const batchSize = DEFAULT_THRESHOLD - 1;
      const networks = Array.from({ length: batchSize }, () => network);
      const inputMatrix = new Float32Array(batchSize * network.input).fill(0.5);
      const device = createMockGPUDevice();

      await evaluateRacingGeneration(networks, inputMatrix, device, {
        gpuBatchThreshold: DEFAULT_THRESHOLD,
      });

      expect(activateSpy).toHaveBeenCalledTimes(batchSize);
    });

    it('falls back to CPU per-network activation when any network is ineligible', async () => {
      const network = Network.createMLP(2, [3], 1);
      const activateSpy = jest.spyOn(Network.prototype, 'activate');
      const ineligibleNetwork = Network.createMLP(2, [3], 1);
      const gaterNode = ineligibleNetwork.nodes[4];
      const connection = ineligibleNetwork.connections[0];
      ineligibleNetwork.gate(gaterNode, connection);
      const networks = [network, ineligibleNetwork, network];
      const inputMatrix = new Float32Array(
        networks.length * network.input,
      ).fill(0.5);
      const device = createMockGPUDevice();

      await evaluateRacingGeneration(networks, inputMatrix, device, {
        gpuBatchThreshold: DEFAULT_THRESHOLD,
      });

      expect(activateSpy).toHaveBeenCalledTimes(networks.length);
      activateSpy.mockRestore();
    });

    it('falls back to CPU per-network activation when device is null', async () => {
      const network = Network.createMLP(2, [3], 1);
      const activateSpy = jest.spyOn(network, 'activate');
      const batchSize = DEFAULT_THRESHOLD + 1;
      const networks = Array.from({ length: batchSize }, () => network);
      const inputMatrix = new Float32Array(batchSize * network.input).fill(0.5);

      await evaluateRacingGeneration(networks, inputMatrix, null, {
        gpuBatchThreshold: DEFAULT_THRESHOLD,
      });

      expect(activateSpy).toHaveBeenCalledTimes(batchSize);
    });

    it('falls back to CPU per-network activation when device is lost', async () => {
      const network = Network.createMLP(2, [3], 1);
      const activateSpy = jest.spyOn(network, 'activate');
      const batchSize = DEFAULT_THRESHOLD + 1;
      const networks = Array.from({ length: batchSize }, () => network);
      const inputMatrix = new Float32Array(batchSize * network.input).fill(0.5);
      const device = createMockGPUDevice();
      device.fakeLose();

      await evaluateRacingGeneration(networks, inputMatrix, device, {
        gpuBatchThreshold: DEFAULT_THRESHOLD,
      });

      expect(activateSpy).toHaveBeenCalledTimes(batchSize);
    });

    it('returns a result matrix with one row per network regardless of path', async () => {
      const network = Network.createMLP(2, [3], 1);
      const batchSize = DEFAULT_THRESHOLD - 1;
      const networks = Array.from({ length: batchSize }, () => network);
      const inputMatrix = new Float32Array(batchSize * network.input).fill(0.5);
      const device = createMockGPUDevice();

      const result = await evaluateRacingGeneration(
        networks,
        inputMatrix,
        device,
        { gpuBatchThreshold: DEFAULT_THRESHOLD },
      );

      expect(result.length).toBe(batchSize * network.output);
    });
  });
});
