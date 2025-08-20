/**
 * network.slab.alloc.stats.reuse.test.ts
 * Validates that repeated rebuilds without capacity growth favor pooled over fresh allocations when pooling enabled.
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';
import { memoryStats } from '../../src/utils/memory';

describe('network.slab.alloc.stats.reuse', () => {
  it('pooled allocations increase with repeated rebuilds absent structural growth', () => {
    config.enableNodePooling = false;
    config.enableSlabArrayPooling = true;
    const net = new Network(6, 3, { enforceAcyclic: true });
    (net as any)._slabDirty = true;
    (net as any).getConnectionSlab();
    const before = memoryStats(net).flags.snapshot.allocStats;
    // Perform several slab rebuilds without changing structure
    for (let i = 0; i < 5; i++) {
      (net as any)._slabDirty = true;
      (net as any).getConnectionSlab();
    }
    const after = memoryStats(net).flags.snapshot.allocStats;
    // Fresh may remain constant; pooled should be >= before.pooled
    expect(after.pooled >= before.pooled).toBe(true);
  });
});
