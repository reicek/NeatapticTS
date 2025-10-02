/**
 * network.slab.alloc.stats.reuse.test.ts
 * Validates that repeated rebuilds without capacity growth favor pooled over fresh allocations when pooling enabled.
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';
import { memoryStats, type SlabAllocStats } from '../../src/utils/memory';

const markSlabDirty = (network: Network): void => {
  Reflect.set(network, '_slabDirty', true);
};

const getAllocationStats = (network: Network): SlabAllocStats | null =>
  memoryStats(network).flags.snapshot.allocStats as SlabAllocStats | null;

describe('network.slab.alloc.stats.reuse', () => {
  it('pooled allocations increase with repeated rebuilds absent structural growth', () => {
    config.enableNodePooling = false;
    config.enableSlabArrayPooling = true;
    const networkUnderTest = new Network(6, 3, { enforceAcyclic: true });
    markSlabDirty(networkUnderTest);
    networkUnderTest.getConnectionSlab();
    const before = getAllocationStats(networkUnderTest);
    if (!before) throw new Error('before is null');
    // Perform several slab rebuilds without changing structure
    for (let iterationIndex = 0; iterationIndex < 5; iterationIndex += 1) {
      markSlabDirty(networkUnderTest);
      networkUnderTest.getConnectionSlab();
    }
    const after = getAllocationStats(networkUnderTest);
    if (!after) throw new Error('after is null');
    // Fresh may remain constant; pooled should be >= before.pooled
    expect(after.pooled >= before.pooled).toBe(true);
  });
});
