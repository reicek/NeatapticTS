/**
 * Phase 3 – Initial slab packing tests.
 * Focus: version counter increments on rebuild + presence of new flags/gain arrays.
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

describe('phase3.slab.initial', () => {
  describe('slab rebuild version & arrays', () => {
    it('slab version increments and exposes flags/gain arrays', () => {
      // Arrange
      config.enableNodePooling = false; // pooling not required here
      const net = new Network(3, 2, { enforceAcyclic: true });
      // Force a connection add/remove mutation to dirty slabs (simulate structural change)
      const initialVersion = (net as any)._slabVersion || 0;
      // Act
      const slab1 = (net as any).getConnectionSlab();
      const v1 = slab1.version;
      // Add a node between to change connections and mark slab dirty
      if (net.connections.length) net.addNodeBetween();
      (net as any)._slabDirty = true;
      const slab2 = (net as any).getConnectionSlab();
      const v2 = slab2.version;
      // Assert (single expectation bundling core invariants)
      expect(
        v2 > v1 &&
          ArrayBuffer.isView(slab2.flags) &&
          ArrayBuffer.isView(slab2.gain) &&
          slab2.flags.length === slab2.weights.length &&
          slab2.gain.length === slab2.weights.length
      ).toBe(true);
    });
  });
});
