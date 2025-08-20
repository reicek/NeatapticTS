/**
 * network.slab.gain.omission.test.ts
 * Ensures gain slab omitted when all gains neutral (1) and allocated lazily when a non-neutral gain appears.
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

function hasGainSlab(net: any) {
  return !!net._connGain;
}

describe('network.slab.gain.omission', () => {
  it('omits gain slab until non-neutral gain set and releases if reverted', () => {
    config.enableNodePooling = false;
    const net = new Network(3, 2, { enforceAcyclic: true });
    (net as any)._slabDirty = true;
    (net as any).getConnectionSlab();
    const initialHas = hasGainSlab(net);
    // If no connections, trivially pass
    if (net.connections.length === 0) {
      expect(true).toBe(true);
      return;
    }
    // Set a non-neutral gain on first connection
    net.connections[0].gain = 1.5;
    (net as any)._slabDirty = true;
    (net as any).getConnectionSlab();
    const afterSet = hasGainSlab(net);
    // Revert to neutral
    net.connections[0].gain = 1;
    (net as any)._slabDirty = true;
    (net as any).getConnectionSlab();
    const afterRevert = hasGainSlab(net);
    expect(
      initialHas === false && afterSet === true && afterRevert === false
    ).toBe(true);
  });
});
