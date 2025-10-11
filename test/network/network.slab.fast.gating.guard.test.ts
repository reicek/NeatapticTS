/**
 * network.slab.fast.gating.guard.test.ts
 * Validates that when gating is present, fastSlabActivate falls back to legacy path (guard).
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';
import Node from '../../src/architecture/node';

const runFastSlabActivate = (network: Network, values: number[]): number[] => {
  const fastSlabActivate = Reflect.get(network, 'fastSlabActivate') as (
    input: number[]
  ) => number[];
  return fastSlabActivate.call(network, values);
};

const getGateCollection = (network: Network): Array<unknown> =>
  Reflect.get(network, 'gates') as Array<unknown>;

const markTopologyDirty = (network: Network): void => {
  Reflect.set(network, '_topoDirty', true);
  Reflect.set(network, '_nodeIndexDirty', true);
};

// Minimal gating scenario: create a gater connection (simulate by assigning to net.gates array directly if API limited)

describe('network.slab.fast.gating.guard', () => {
  it('falls back to legacy path when gating detected', () => {
    // Arrange
    config.enableNodePooling = false;
    const networkUnderTest = new Network(2, 1, { enforceAcyclic: true });
    // Create a hidden node and a dummy gated connection by pushing to gates array (simplified)
    const hiddenNode = new Node('hidden');
    networkUnderTest.nodes.push(hiddenNode);
    const gateCollection = getGateCollection(networkUnderTest);
    gateCollection.push({ dummy: true });
    markTopologyDirty(networkUnderTest);
    const inputVector = [0.3, -0.1];
    const legacyOutput = networkUnderTest.activate([...inputVector], false);
    const fastOutput = runFastSlabActivate(networkUnderTest, [...inputVector]);
    // Assert: outputs equal (fallback used) and gating array present
    expect(
      JSON.stringify(fastOutput) === JSON.stringify(legacyOutput) &&
        getGateCollection(networkUnderTest).length > 0
    ).toBe(true);
  });
});
