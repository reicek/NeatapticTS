import Network from '../network';
import type Node from '../../node/node';
import type Connection from '../../connection/connection';
import { createMockGPUDevice } from './__mocks__/gpu.mock';
import { activateGPU } from './network.gpu.activate';

const SMALL_INPUTS: [number, number] = [0.1, 0.2];
const ABSOLUTE_TOLERANCE = 0.5;
const MEAN_ABSOLUTE_ERROR_TOLERANCE = 0.1;

describe('network.gpu.parity', () => {
  describe('small feed-forward network', () => {
    let network: Network;
    let device: ReturnType<typeof createMockGPUDevice>;
    let cpuOutput: number[];

    beforeEach(() => {
      network = Network.createMLP(2, [3], 1);
      device = createMockGPUDevice();
      cpuOutput = network.activate(SMALL_INPUTS);
    });

    it('matches CPU output length', async () => {
      const gpuOutput = await activateGPU(device, network, SMALL_INPUTS);

      expect(gpuOutput.length).toBe(cpuOutput.length);
    });

    it('keeps per-element absolute difference within 5e-1', async () => {
      const gpuOutput = await activateGPU(device, network, SMALL_INPUTS);
      const maxAbsoluteDifference = Math.max(
        ...cpuOutput.map((value, index) => Math.abs(value - gpuOutput[index])),
      );

      expect(maxAbsoluteDifference).toBeLessThanOrEqual(ABSOLUTE_TOLERANCE);
    });

    it('keeps mean absolute error within 1e-1', async () => {
      const gpuOutput = await activateGPU(device, network, SMALL_INPUTS);
      const meanAbsoluteError =
        cpuOutput.reduce(
          (sum, value, index) => sum + Math.abs(value - gpuOutput[index]),
          0,
        ) / cpuOutput.length;

      expect(meanAbsoluteError).toBeLessThanOrEqual(
        MEAN_ABSOLUTE_ERROR_TOLERANCE,
      );
    });
  });

  describe('racing-browser scale network (76 nodes / 288 connections)', () => {
    let network: Network;
    let device: ReturnType<typeof createMockGPUDevice>;
    let cpuOutput: number[];

    beforeEach(() => {
      // 2 input + 72 hidden + 2 output = 76 nodes; 2*72 + 72*2 = 288 connections.
      network = Network.createMLP(2, [72], 2);
      device = createMockGPUDevice();
      cpuOutput = network.activate(SMALL_INPUTS);
    });

    it('matches CPU output length', async () => {
      const gpuOutput = await activateGPU(device, network, SMALL_INPUTS);

      expect(gpuOutput.length).toBe(cpuOutput.length);
    });

    it('keeps per-element absolute difference within 5e-1', async () => {
      const gpuOutput = await activateGPU(device, network, SMALL_INPUTS);
      const maxAbsoluteDifference = Math.max(
        ...cpuOutput.map((value, index) => Math.abs(value - gpuOutput[index])),
      );

      expect(maxAbsoluteDifference).toBeLessThanOrEqual(ABSOLUTE_TOLERANCE);
    });

    it('keeps mean absolute error within 1e-1', async () => {
      const gpuOutput = await activateGPU(device, network, SMALL_INPUTS);
      const meanAbsoluteError =
        cpuOutput.reduce(
          (sum, value, index) => sum + Math.abs(value - gpuOutput[index]),
          0,
        ) / cpuOutput.length;

      expect(meanAbsoluteError).toBeLessThanOrEqual(
        MEAN_ABSOLUTE_ERROR_TOLERANCE,
      );
    });
  });

  describe('NGE-cap scale network (up to 8000 nodes / 32000 connections)', () => {
    let network: Network;
    let device: ReturnType<typeof createMockGPUDevice>;
    let cpuOutput: number[];

    beforeAll(() => {
      // 2 input + 7996 hidden + 2 output = 8000 nodes;
      // 2*7996 + 7996*2 = 31984 connections.
      network = Network.createMLP(2, [7996], 2);
      device = createMockGPUDevice();
      cpuOutput = network.activate(SMALL_INPUTS);
    });

    it('matches CPU output length', async () => {
      const gpuOutput = await activateGPU(device, network, SMALL_INPUTS);

      expect(gpuOutput.length).toBe(cpuOutput.length);
    });

    it('keeps per-element absolute difference within 5e-1', async () => {
      const gpuOutput = await activateGPU(device, network, SMALL_INPUTS);
      const maxAbsoluteDifference = Math.max(
        ...cpuOutput.map((value, index) => Math.abs(value - gpuOutput[index])),
      );

      expect(maxAbsoluteDifference).toBeLessThanOrEqual(ABSOLUTE_TOLERANCE);
    });

    it('keeps mean absolute error within 1e-1', async () => {
      const gpuOutput = await activateGPU(device, network, SMALL_INPUTS);
      const meanAbsoluteError =
        cpuOutput.reduce(
          (sum, value, index) => sum + Math.abs(value - gpuOutput[index]),
          0,
        ) / cpuOutput.length;

      expect(meanAbsoluteError).toBeLessThanOrEqual(
        MEAN_ABSOLUTE_ERROR_TOLERANCE,
      );
    });
  });

  describe('ineligible network', () => {
    it('rejects activation for gated networks', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      // Make the network ineligible by gating an existing connection.
      // createMLP enforces a feed-forward topology, so a self-connection would
      // be silently rejected; gating is a feed-forward-legal ineligibility.
      const gaterNode = network.nodes[4];
      const connection = network.connections[0];
      network.gate(gaterNode, connection);

      await expect(
        activateGPU(device, network, SMALL_INPUTS),
      ).rejects.toThrow();
    });

    it('rejects activation when a connection lists the same node as source and target', async () => {
      const device = createMockGPUDevice();
      // This minimal fixture exercises the defensive explicit-self-connection
      // branch in canUseGPU, which normal connection registration never reaches
      // because self-loops are stored in network.selfconns instead.
      const node = {} as unknown as Node;
      const selfLoopConnection = {
        from: node,
        to: node,
      } as unknown as Connection;
      const fakeNetwork = {
        gates: [],
        selfconns: [],
        connections: [selfLoopConnection],
        nodes: [],
        input: 2,
        output: 1,
      } as unknown as Network;

      await expect(
        activateGPU(device, fakeNetwork, SMALL_INPUTS),
      ).rejects.toThrow();
    });
  });
});
