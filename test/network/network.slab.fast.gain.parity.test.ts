/**
 * network.slab.fast.gain.parity.test.ts
 * Ensures fast slab path matches legacy path when non-neutral gains trigger lazy gain slab allocation.
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

const markSlabDirty = (network: Network): void => {
  Reflect.set(network, '_slabDirty', true);
};

const runFastSlabActivate = (network: Network, values: number[]): number[] => {
  const fastSlabActivate = Reflect.get(network, 'fastSlabActivate') as (
    input: number[]
  ) => number[];
  return fastSlabActivate.call(network, values);
};

describe('network.slab.fast.gain.parity', () => {
  it('matches legacy activate with non-neutral gains (lazy slab allocation)', () => {
    config.enableNodePooling = false;
    const networkUnderTest = new Network(4, 2, { enforceAcyclic: true });
    // Grow a few hidden nodes to ensure multiple connections
    for (let mutationIndex = 0; mutationIndex < 6; mutationIndex++) {
      if (networkUnderTest.connections.length) {
        networkUnderTest.addNodeBetween();
      }
    }
    // Assign a non-neutral gain to several connections to force gain slab allocation
    for (
      let connectionIndex = 0;
      connectionIndex < networkUnderTest.connections.length;
      connectionIndex += 2
    ) {
      networkUnderTest.connections[connectionIndex].gain = 1.2;
    }
    markSlabDirty(networkUnderTest);
    const inputVector = [0.1, -0.2, 0.05, 0.9]; // first 4 inputs
    const legacyOutput = networkUnderTest.activate([...inputVector], false);
    const fastOutput = runFastSlabActivate(networkUnderTest, [...inputVector]);
    expect(JSON.stringify(fastOutput)).toBe(JSON.stringify(legacyOutput));
  });
});
