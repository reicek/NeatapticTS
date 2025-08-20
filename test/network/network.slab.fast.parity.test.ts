/**
 * network.slab.fast.parity.test.ts
 * Ensures fastSlabActivate matches legacy activate output for eligible networks (acyclic, no gates/dropout/noise).
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

function randomInputs(n: number) {
  const arr: number[] = [];
  for (let i = 0; i < n; i++) arr.push(Math.random() * 2 - 1);
  return arr;
}

describe('network.slab.fast.parity', () => {
  it('produces identical outputs to legacy activate on random seeds when eligible', () => {
    // Arrange
    config.enableNodePooling = false;
    const net = new Network(5, 3, { enforceAcyclic: true });
    // Grow a simple acyclic structure with addNodeBetween mutations
    for (let i = 0; i < 8; i++) {
      if (net.connections.length) net.addNodeBetween();
    }
    // Force slab dirty then build both paths
    (net as any)._slabDirty = true;
    const input = randomInputs(5);
    const legacy = (net as any).activate(input.slice(), false);
    const fast = (net as any).fastSlabActivate(input.slice());
    // Assert single expectation
    expect(JSON.stringify(fast)).toBe(JSON.stringify(legacy));
  });
});
