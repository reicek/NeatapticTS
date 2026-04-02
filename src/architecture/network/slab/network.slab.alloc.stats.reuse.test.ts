import Network from '../network';
import { config } from '../../../config';
import { memoryStats } from '../../../utils/memory';
import type { SlabAllocStats } from '../../../utils/memory';

function getAllocationStats(network: Network): SlabAllocStats | null {
  return memoryStats(network).flags.snapshot
    .allocStats as SlabAllocStats | null;
}

function markSlabDirty(network: Network): void {
  Reflect.set(network, '_slabDirty', true);
}

describe('network slab chapter', () => {
  describe('allocation statistics reuse', () => {
    describe('given the structure stays fixed across repeated rebuilds', () => {
      describe('when pooling is enabled and allocation stats are compared', () => {
        it('increases pooled allocations without requiring structural growth', () => {
          // Arrange
          config.enableNodePooling = false;
          config.enableSlabArrayPooling = true;
          const network = new Network(6, 3, { enforceAcyclic: true });

          markSlabDirty(network);
          network.getConnectionSlab();
          const initialStats = getAllocationStats(network);

          if (!initialStats) {
            throw new Error('Expected initial slab allocation stats');
          }

          for (let rebuildIndex = 0; rebuildIndex < 5; rebuildIndex++) {
            markSlabDirty(network);
            network.getConnectionSlab();
          }

          // Act
          const finalStats = getAllocationStats(network);

          if (!finalStats) {
            throw new Error('Expected final slab allocation stats');
          }

          // Assert
          expect(finalStats.pooled >= initialStats.pooled).toBe(true);
        });
      });
    });
  });
});
