/**
 * network.slab.versioning.test.ts
 * Verifies initial slab packing adds flags/gain arrays and increments a version counter on structural change.
 * Focus: educational coverage for SoA rebuild mechanics (single expectation style).
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

describe('network.slab.versioning', () => {
  describe('rebuild increments version & exposes parallel arrays', () => {
    it('version increases after structural mutation and flags/gain align with weights', () => {
      // Arrange
      config.enableNodePooling = false; // pooling not relevant for slab packing test
      const net = new Network(3, 2, { enforceAcyclic: true });
      const slabA = (net as any).getConnectionSlab();
      const vA = slabA.version;
      // Act: mutate structure (split a random connection) marking slab dirty
      if (net.connections.length) net.addNodeBetween();
      (net as any)._slabDirty = true; // force rebuild
      const slabB = (net as any).getConnectionSlab();
      const vB = slabB.version;
      // Assert (single expectation bundling invariants)
      expect(
        vB > vA &&
          ArrayBuffer.isView(slabB.flags) &&
          ArrayBuffer.isView(slabB.gain) &&
          slabB.flags.length === slabB.weights.length &&
          slabB.gain.length === slabB.weights.length
      ).toBe(true);
    });
  });
});
