/**
 * Red tests for GPU eligibility re-verification after network mutations.
 *
 * When a network is structurally modified (add/remove node, connect, mutate),
 * the acceleration layer must invalidate cached eligibility and re-verify the
 * network against the GPU capability predicate before the next GPU dispatch.
 */

import Network from '../network';
import { createMockGPUDevice } from './__mocks__/gpu.mock';
import { isGPUEligible } from './network.gpu.fallback';

describe('network.gpu.eligibility.mutation', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('structural re-verification', () => {
    it('remains eligible after a simple weight mutation', () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();

      network.mutate({ mutation: 'MOD_WEIGHTS' });

      expect(isGPUEligible(network, device)).toBe(true);
    });

    it('becomes ineligible after adding a self-connection', () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      const fromNode = network.nodes[0];
      const toNode = network.nodes[0];

      network.connect(fromNode, toNode);

      expect(isGPUEligible(network, device)).toBe(false);
    });

    it('becomes ineligible after adding a gate', () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      const gater = network.nodes[4];
      const connection = network.connections[0];

      network.gate(gater, connection);

      expect(isGPUEligible(network, device)).toBe(false);
    });

    it('remains eligible after adding a feed-forward connection', () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      const inputNode = network.nodes[0];
      const outputNode = network.nodes.at(-1)!;

      network.connect(inputNode, outputNode);

      expect(isGPUEligible(network, device)).toBe(true);
    });
  });

  describe('network lastActivationBackend invalidation', () => {
    it('resets lastActivationBackend to undefined after adding a connection', () => {
      const network = Network.createMLP(2, [3], 1);
      network.activate([0.5, -0.5], { backend: 'cpu' });
      const inputNode = network.nodes[0];
      const outputNode = network.nodes.at(-1)!;

      network.connect(inputNode, outputNode);

      expect(network.lastActivationBackend).toBeUndefined();
    });

    it('resets lastActivationBackend to undefined after adding a node between layers', () => {
      const network = Network.createMLP(2, [3], 1);
      network.activate([0.5, -0.5], { backend: 'cpu' });

      network.addNodeBetween();

      expect(network.lastActivationBackend).toBeUndefined();
    });

    it('resets lastActivationBackend to undefined after removing a node', () => {
      const network = Network.createMLP(2, [3], 1);
      network.activate([0.5, -0.5], { backend: 'cpu' });
      const hiddenNode = network.nodes[2];

      network.remove(hiddenNode);

      expect(network.lastActivationBackend).toBeUndefined();
    });

    it('resets lastActivationBackend to undefined after disconnecting nodes', () => {
      const network = Network.createMLP(2, [3], 1);
      network.activate([0.5, -0.5], { backend: 'cpu' });
      const fromNode = network.nodes[0];
      const toNode = network.nodes[2];

      network.disconnect(fromNode, toNode);

      expect(network.lastActivationBackend).toBeUndefined();
    });

    it('resets lastActivationBackend to undefined after clearing network state', () => {
      const network = Network.createMLP(2, [3], 1);
      network.activate([0.5, -0.5], { backend: 'cpu' });

      network.clear();

      expect(network.lastActivationBackend).toBeUndefined();
    });

    it('resets lastActivationBackend to undefined after a batched connection', () => {
      const network = Network.createMLP(2, [3], 1);
      network.activate([0.5, -0.5], { backend: 'cpu' });
      const inputNode = network.nodes[0];
      const outputNode = network.nodes.at(-1)!;

      network.connectBatch([{ from: inputNode, to: outputNode }]);

      expect(network.lastActivationBackend).toBeUndefined();
    });
  });

  describe('getGPUEligibility after mutation', () => {
    it('reports the updated eligibility after a mutation', () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();

      network.gate(network.nodes[4], network.connections[0]);

      const eligibility = network.getGPUEligibility();
      expect(eligibility.eligible).toBe(false);
    });

    it('includes a reason that mentions gating when ineligible due to a gate', () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();

      network.gate(network.nodes[4], network.connections[0]);

      const eligibility = network.getGPUEligibility();
      expect(eligibility.reason).toEqual(expect.stringMatching(/gate/i));
    });

    it('reports not eligible after a self-connection is added', () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();

      const before = network.getGPUEligibility();
      expect(before.eligible).toBe(true);

      const node = network.nodes[2];
      network.connect(node, node);

      const after = network.getGPUEligibility();
      expect(after.eligible).toBe(false);
    });
  });

  describe('eligibility transitions after mutation', () => {
    it('becomes eligible again after removing a gate that blocked the gpu path', () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      network.gpuDevice = device;

      const gaterNode = network.nodes[4];
      const targetConnection = network.connections[0];
      network.gate(gaterNode, targetConnection);
      expect(network.getGPUEligibility().eligible).toBe(false);

      network.ungate(targetConnection);

      expect(network.getGPUEligibility().eligible).toBe(true);
    });
  });

  describe('observer notification after mutation', () => {
    it('notifies observer of a backend change when mutation forces gpu to cpu fallback', async () => {
      const changes: unknown[] = [];
      const observer = {
        onBackendChange: (event: unknown) => changes.push(event),
      };
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();

      await network.activate([0.5, -0.5], { backend: 'gpu', observer });

      const gaterNode = network.nodes[4];
      const targetConnection = network.connections[0];
      network.gate(gaterNode, targetConnection);

      await network.activate([0.5, -0.5], { backend: 'gpu', observer });

      expect(changes.length).toBe(2);
    });
  });
});
