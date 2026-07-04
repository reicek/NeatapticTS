import Network from '../network';
import {
  evaluateConcurrentAgents,
  evaluateBatchGeneration,
} from './network.gpu.batch-evaluation';
import { createMockGPUDevice } from './__mocks__/gpu.mock';

const DEFAULT_THRESHOLD = 4;

describe('network.gpu.batch-evaluation', () => {
  describe('evaluateBatchGeneration', () => {
    it('GPU batch path returns a zero placeholder that fails parity with CPU output', async () => {
      const network = Network.createMLP(2, [3], 1);
      const batchSize = DEFAULT_THRESHOLD + 1;
      const networks = Array.from({ length: batchSize }, () => network);
      const inputMatrix = new Float32Array(batchSize * network.input).fill(0.5);
      const device = createMockGPUDevice();

      const result = await evaluateBatchGeneration(
        networks,
        inputMatrix,
        device,
        { gpuBatchThreshold: DEFAULT_THRESHOLD },
      );

      expect(result).toEqual(
        new Float32Array(batchSize * network.output).fill(0),
      );
    });

    it('returns an empty output matrix for an empty generation', async () => {
      const result = await evaluateBatchGeneration(
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

      await evaluateBatchGeneration(networks, inputMatrix, device);

      expect(activateSpy).toHaveBeenCalledTimes(batchSize);
    });

    it('falls back to CPU per-network activation when batch size is below threshold', async () => {
      const network = Network.createMLP(2, [3], 1);
      const activateSpy = jest.spyOn(network, 'activate');
      const batchSize = DEFAULT_THRESHOLD - 1;
      const networks = Array.from({ length: batchSize }, () => network);
      const inputMatrix = new Float32Array(batchSize * network.input).fill(0.5);
      const device = createMockGPUDevice();

      await evaluateBatchGeneration(networks, inputMatrix, device, {
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

      await evaluateBatchGeneration(networks, inputMatrix, device, {
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

      await evaluateBatchGeneration(networks, inputMatrix, null, {
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

      await evaluateBatchGeneration(networks, inputMatrix, device, {
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

      const result = await evaluateBatchGeneration(
        networks,
        inputMatrix,
        device,
        { gpuBatchThreshold: DEFAULT_THRESHOLD },
      );

      expect(result.length).toBe(batchSize * network.output);
    });
  });

  describe('evaluateConcurrentAgents', () => {
    it('returns an empty array for an empty request list', async () => {
      const result = await evaluateConcurrentAgents(createMockGPUDevice(), []);

      expect(result).toEqual([]);
    });

    it('does not collide when multiple requests target the same network instance', async () => {
      const network = Network.createMLP(2, [3], 1);
      const firstInputs = [0.1, 0.2];
      const secondInputs = [0.3, 0.4];
      const requests = [
        { network, inputs: firstInputs },
        { network, inputs: secondInputs },
      ];
      const device = createMockGPUDevice({ emulateNetwork: network });

      const outputs = await evaluateConcurrentAgents(device, requests);

      expect(outputs).toEqual([
        new Float32Array(network.activate(firstInputs)),
        new Float32Array(network.activate(secondInputs)),
      ]);
    });

    it('returns one correct output per request across different network instances', async () => {
      const network = Network.createMLP(2, [3], 1);
      const firstInputs = [0.1, 0.2];
      const secondInputs = [0.3, 0.4];
      const requests = [
        { network: network.clone(), inputs: firstInputs },
        { network: network.clone(), inputs: secondInputs },
      ];
      const device = createMockGPUDevice({ emulateNetwork: network });

      const outputs = await evaluateConcurrentAgents(device, requests);

      expect({
        length: outputs.length,
        first: outputs[0],
        second: outputs[1],
      }).toEqual({
        length: requests.length,
        first: new Float32Array(network.activate(firstInputs)),
        second: new Float32Array(network.activate(secondInputs)),
      });
    });

    it('creates only one pipeline per unique topology for interleaved requests', async () => {
      const networkA = Network.createMLP(2, [3], 1);
      const networkB = Network.createMLP(2, [4], 1);
      const requests = [
        { network: networkA.clone(), inputs: [0.1, 0.2] },
        { network: networkB.clone(), inputs: [0.3, 0.4] },
        { network: networkA.clone(), inputs: [0.5, 0.6] },
      ];
      const device = createMockGPUDevice();

      await evaluateConcurrentAgents(device, requests);

      // Topology constants now live in storage buffers, so the generated WGSL is
      // identical for both topologies and only one compiled pipeline is needed.
      expect(device.recorded.pipelines.length).toBe(1);
    });
  });
});
