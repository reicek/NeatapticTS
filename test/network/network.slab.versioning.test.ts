/**
 * network.slab.versioning.test.ts
 * Verifies initial slab packing adds flags/gain arrays and increments a version counter on structural change.
 * Focus: educational coverage for SoA rebuild mechanics (single expectation style).
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

type ConnectionSlab = ReturnType<Network['getConnectionSlab']>;
interface NetworkInternals {
  _slabDirty: boolean;
}

const getConnectionSlab = (net: Network): ConnectionSlab =>
  (
    net as unknown as { getConnectionSlab: () => ConnectionSlab }
  ).getConnectionSlab();

const setNetworkInternal = <Key extends keyof NetworkInternals>(
  net: Network,
  key: Key,
  value: NetworkInternals[Key],
) => {
  Reflect.set(net, key, value);
};

describe('network.slab.versioning', () => {
  describe('rebuild increments version & exposes parallel arrays', () => {
    it('version increases after structural mutation and flags/gain align with weights', () => {
      // Arrange
      config.enableNodePooling = false; // pooling not relevant for slab packing test
      const net = new Network(3, 2, { enforceAcyclic: true });
      const slabA = getConnectionSlab(net);
      const vA = slabA.version;
      // Act: mutate structure (split a random connection) marking slab dirty
      if (net.connections.length) net.addNodeBetween();
      setNetworkInternal(net, '_slabDirty', true); // force rebuild
      const slabB = getConnectionSlab(net);
      const vB = slabB.version;
      // Assert (single expectation bundling invariants)
      expect(
        vB > vA &&
          ArrayBuffer.isView(slabB.flags) &&
          ArrayBuffer.isView(slabB.gain) &&
          slabB.flags.length === slabB.weights.length &&
          slabB.gain.length === slabB.weights.length,
      ).toBe(true);
    });
  });
});
