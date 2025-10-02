/**
 * Phase 3 – Initial slab packing tests.
 * Focus: version counter increments on rebuild + presence of new flags/gain arrays.
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

describe('phase3.slab.initial', () => {
  describe('slab rebuild version & arrays', () => {
    it('slab version increments and exposes flags/gain arrays', () => {
      // Arrange
      config.enableNodePooling = false; // pooling not required here
      const net = new Network(3, 2, { enforceAcyclic: true });
      // Act
      const slab1 = getConnectionSlab(net);
      const v1 = slab1.version;
      // Add a node between to change connections and mark slab dirty
      if (net.connections.length) net.addNodeBetween();
      setNetworkInternal(net, '_slabDirty', true);
      const slab2 = getConnectionSlab(net);
      const v2 = slab2.version;
      // Assert (single expectation bundling core invariants)
      expect(
        v2 > v1 &&
          ArrayBuffer.isView(slab2.flags) &&
          ArrayBuffer.isView(slab2.gain) &&
          slab2.flags.length === slab2.weights.length &&
          slab2.gain.length === slab2.weights.length,
      ).toBe(true);
    });
  });
});
