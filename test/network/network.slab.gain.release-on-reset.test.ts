/**
 * network.slab.gain.release-on-reset.test.ts
 * Ensures that once all connection gains revert to 1, optional gain slab is released on next rebuild.
 */
import Network from '../../src/architecture/network';

function anyGainArray(net: any) {
  return !!net._connGain;
}

describe('network.slab.gain.release-on-reset', () => {
  it('releases optional gain slab after all gains reset to neutral', () => {
    const net: any = new Network(4, 2, { enforceAcyclic: true });
    // Introduce a gating gain effect by setting gains manually
    net.connections.forEach((c: any, idx: number) => {
      if (idx === 0) c.gain = 0.5;
    });
    net._slabDirty = true;
    net.getConnectionSlab();
    expect(anyGainArray(net)).toBe(true); // optional gain slab allocated

    // Reset all gains to neutral
    net.connections.forEach((c: any) => {
      c.gain = 1;
    });
    net._slabDirty = true;
    net.getConnectionSlab();
    expect(anyGainArray(net)).toBe(false); // optional gain slab should be released
  });
});
