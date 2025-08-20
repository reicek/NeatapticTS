/**
 * network.slab.plasticity.test.ts
 * Verifies plasticity flag bit (bit3) and optional plasticity rate slab allocation is pay-for-use.
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

function countPlastic(flags: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < flags.length; i++) if (flags[i] & 0b1000) n++;
  return n;
}

describe('network.slab.plasticity', () => {
  it('allocates plastic slab only when at least one connection has plasticityRate > 0 and releases when cleared', () => {
    config.enableNodePooling = false;
    config.enableSlabArrayPooling = true;
    const net = new Network(4, 2, { enforceAcyclic: true });
    (net as any)._slabDirty = true;
    let slab = (net as any).getConnectionSlab();
    const baseVersion = slab.version;
    const hadPlasticInitially = !!slab.plastic;
    // Assign plasticityRate to first connection
    if (net.connections.length === 0) {
      expect(true).toBe(true);
      return;
    }
    (net.connections[0] as any).plasticityRate = 0.05;
    (net as any)._slabDirty = true;
    slab = (net as any).getConnectionSlab();
    const plasticAfterSet = slab.plastic;
    const plasticCount = countPlastic(slab.flags.subarray(0, slab.used));
    // Clear plasticityRate
    (net.connections[0] as any).plasticityRate = 0;
    (net as any)._slabDirty = true;
    const slabAfterClear = (net as any).getConnectionSlab();
    const plasticCleared = slabAfterClear.plastic === null;
    expect(
      hadPlasticInitially === false &&
        plasticAfterSet !== null &&
        plasticCount >= 1 &&
        slab.version > baseVersion &&
        plasticCleared
    ).toBe(true);
  });
});
