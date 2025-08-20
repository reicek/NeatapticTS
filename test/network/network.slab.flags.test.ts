/**
 * network.slab.flags.test.ts
 * Ensures bit-packed flags array mirrors Connection _flags transitions (enable/disable, dropConnect mask simulation).
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

describe('network.slab.flags', () => {
  it('reflects enable/disable transitions in slab flag byte without full rebuild of capacity', () => {
    config.enableNodePooling = false;
    const net = new Network(3, 2, { enforceAcyclic: true });
    const slab1 = (net as any).getConnectionSlab();
    const cap1 = slab1.capacity;
    // Disable first connection
    if (net.connections.length === 0) return expect(true).toBe(true);
    net.connections[0].enabled = false;
    (net as any)._slabDirty = true;
    const slab2 = (net as any).getConnectionSlab();
    const flagByte = slab2.flags[0];
    // Re-enable and rebuild
    net.connections[0].enabled = true;
    (net as any)._slabDirty = true;
    const slab3 = (net as any).getConnectionSlab();
    const flagByteRe = slab3.flags[0];
    expect(
      cap1 === slab2.capacity && // capacity reused
        (flagByte & 0b1) === 0 && // disabled bit cleared
        (flagByteRe & 0b1) === 1 && // re-enabled set
        slab3.version > slab1.version // version increments
    ).toBe(true);
  });
});
