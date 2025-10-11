/**
 * network.slab.fast.parity.test.ts
 * Ensures fastSlabActivate matches legacy activate output for eligible networks (acyclic, no gates/dropout/noise).
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

const randomInputs = (count: number): number[] =>
  Array.from({ length: count }, () => Math.random() * 2 - 1);

const markSlabDirty = (network: Network): void => {
  Reflect.set(network, '_slabDirty', true);
};

const runFastSlabActivate = (network: Network, values: number[]): number[] => {
  const fastSlabActivate = Reflect.get(network, 'fastSlabActivate') as (
    input: number[]
  ) => number[];
  return fastSlabActivate.call(network, values);
};

describe('network.slab.fast.parity', () => {
  it('produces identical outputs to legacy activate on random seeds when eligible', () => {
    // Arrange
    config.enableNodePooling = false;
    const net = new Network(5, 3, { enforceAcyclic: true });
    // Grow a simple acyclic structure with addNodeBetween mutations
    for (let mutationIndex = 0; mutationIndex < 8; mutationIndex++) {
      if (net.connections.length) net.addNodeBetween();
    }
    // Force slab dirty then build both paths
    markSlabDirty(net);
    const input = randomInputs(5);
    const legacy = net.activate([...input], false);
    const fast = runFastSlabActivate(net, [...input]);
    // Assert single expectation
    expect(JSON.stringify(fast)).toBe(JSON.stringify(legacy));
  });
});
