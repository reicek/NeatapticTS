/**
 * network.slab.fast.gain.parity.test.ts
 * Ensures fast slab path matches legacy path when non-neutral gains trigger lazy gain slab allocation.
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

describe('network.slab.fast.gain.parity', () => {
  it('matches legacy activate with non-neutral gains (lazy slab allocation)', () => {
    config.enableNodePooling = false;
    const net = new Network(4, 2, { enforceAcyclic: true });
    // Grow a few hidden nodes to ensure multiple connections
    for (let i = 0; i < 6; i++)
      if (net.connections.length) net.addNodeBetween();
    // Assign a non-neutral gain to several connections to force gain slab allocation
    for (let i = 0; i < net.connections.length; i += 2)
      net.connections[i].gain = 1.2;
    (net as any)._slabDirty = true;
    const input = [0.1, -0.2, 0.05, 0.9]; // first 4 inputs
    const legacy = (net as any).activate(input.slice(), false);
    const fast = (net as any).fastSlabActivate(input.slice());
    expect(JSON.stringify(fast)).toBe(JSON.stringify(legacy));
  });
});
