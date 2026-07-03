/**
 * GPU-aware racing worker integration tests.
 *
 * Verifies the Phase 2 crossover threshold helper and the controller wrapper
 * that passes the `useGPU` hint to {@link Network.activate}.
 */

import { Network } from '../../../../src/browser-entry.ts';
import {
  createGPUAwareRaceController,
  RACING_BROWSER_GPU_THRESHOLD,
  shouldUseGPUForBatch,
  type GPUAwareRaceController,
} from './simulation-worker.gpu';
import { createMockGPUDevice } from '../../../../src/architecture/network/gpu/__mocks__/gpu.mock.ts';


/**
 * Return a node index inside the single hidden layer of a `createMLP(2, [3], 1)`
 * network. The layout is input[0,1] → hidden[2,3,4] → output[5], so any
 * hidden node is eligible.
 */
function hiddenNodeIndex(network: Network): number {
  return network.nodes.findIndex((node) => node.type === 'hidden');
}

/**
 * Patch a node's squash function so it carries an unsupported activation index.
 * The patched function still returns its input, which is enough for structural
 * eligibility tests because we never activate through it.
 */
function setUnsupportedActivation(network: Network, nodeIndex: number): void {
  const node = network.nodes[nodeIndex];
  const unsupportedActivation = (x: number): number => x;
  (unsupportedActivation as { index?: number }).index = -1;
  node.squash = unsupportedActivation as unknown as typeof node.squash;
}

describe('simulation-worker.gpu', () => {
  describe('RACING_BROWSER_GPU_THRESHOLD', () => {
    it('matches the Phase 2 racing-browser crossover threshold', () => {
      expect(RACING_BROWSER_GPU_THRESHOLD).toBe(130);
    });
  });

  describe('shouldUseGPUForBatch', () => {
    it('returns false when the agent count is below the threshold', () => {
      const network = Network.createMLP(2, [3], 1);

      expect(
        shouldUseGPUForBatch(RACING_BROWSER_GPU_THRESHOLD - 1, network),
      ).toBe(false);
    });

    it('returns true at the threshold for a structurally eligible network', () => {
      const network = Network.createMLP(2, [3], 1);

      expect(shouldUseGPUForBatch(RACING_BROWSER_GPU_THRESHOLD, network)).toBe(
        true,
      );
    });

    it('returns true above the threshold for a structurally eligible network', () => {
      const network = Network.createMLP(2, [3], 1);

      expect(
        shouldUseGPUForBatch(RACING_BROWSER_GPU_THRESHOLD + 1, network),
      ).toBe(true);
    });

    it('returns false for a network with gated connections', () => {
      const network = Network.createMLP(2, [3], 1);
      const gaterNode = network.nodes[hiddenNodeIndex(network)];
      const connection = network.connections[0];
      network.gate(gaterNode, connection);

      expect(shouldUseGPUForBatch(RACING_BROWSER_GPU_THRESHOLD, network)).toBe(
        false,
      );
    });

    it('returns false for a network with self-connections', () => {
      const network = Network.createMLP(2, [3], 1);
      network.setEnforceAcyclic(false);
      const hiddenNode = network.nodes[hiddenNodeIndex(network)];
      network.connect(hiddenNode, hiddenNode);

      expect(shouldUseGPUForBatch(RACING_BROWSER_GPU_THRESHOLD, network)).toBe(
        false,
      );
    });

    it('returns false for a network with unsupported activations', () => {
      const network = Network.createMLP(2, [3], 1);
      const hiddenIndex = hiddenNodeIndex(network);
      setUnsupportedActivation(network, hiddenIndex);

      expect(shouldUseGPUForBatch(RACING_BROWSER_GPU_THRESHOLD, network)).toBe(
        false,
      );
    });
  });

  describe('createGPUAwareRaceController', () => {
    it('passes useGPU: true to network.activate when the batch is eligible', () => {
      const network = Network.createMLP(2, [3], 1);
      const activateSpy = jest.spyOn(network, 'activate');
      const controller: GPUAwareRaceController = createGPUAwareRaceController(
        network,
        RACING_BROWSER_GPU_THRESHOLD,
      );

      const inputs = [0.1, 0.2];
      controller.activate(inputs);

      expect(activateSpy).toHaveBeenCalledWith(inputs, { useGPU: true });
      activateSpy.mockRestore();
    });

    it('passes useGPU: false when the batch is below the threshold', () => {
      const network = Network.createMLP(2, [3], 1);
      const activateSpy = jest.spyOn(network, 'activate');
      const controller: GPUAwareRaceController = createGPUAwareRaceController(
        network,
        RACING_BROWSER_GPU_THRESHOLD - 1,
      );

      const inputs = [0.1, 0.2];
      controller.activate(inputs);

      expect(activateSpy).toHaveBeenCalledWith(inputs, { useGPU: false });
      activateSpy.mockRestore();
    });

    it('returns the CPU fallback output when the GPU path is not available', () => {
      const network = Network.createMLP(2, [3], 1);
      const controller: GPUAwareRaceController = createGPUAwareRaceController(
        network,
        RACING_BROWSER_GPU_THRESHOLD,
      );

      const inputs = [0.1, 0.2];
      const output = controller.activate(inputs);

      expect(Array.isArray(output)).toBe(true);
    });

    it('throws when the GPU path unexpectedly returns a Promise', () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice({ emulateNetwork: network });
      const controller: GPUAwareRaceController = createGPUAwareRaceController(
        network,
        RACING_BROWSER_GPU_THRESHOLD,
      );

      expect(() => controller.activate([0.1, 0.2])).toThrow(
        'Unexpected async GPU activation',
      );
    });
  });
});
