/**
 * network.slab.gain.release-on-reset.test.ts
 * Ensures that once all connection gains revert to 1, optional gain slab is released on next rebuild.
 */
import Network from '../../src/architecture/network';

type ConnectionSlab = ReturnType<Network['getConnectionSlab']>;
interface NetworkInternals {
  _slabDirty: boolean;
  _connGain?: Float32Array | Float64Array | null;
}

const hasGainArray = (net: Network): boolean =>
  Boolean(Reflect.get(net, '_connGain'));

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

describe('network.slab.gain.release-on-reset', () => {
  it('releases optional gain slab after all gains reset to neutral', () => {
    const net = new Network(4, 2, { enforceAcyclic: true });
    // Introduce a gating gain effect by setting gains manually
    net.connections.forEach((connection, index) => {
      if (index === 0) connection.gain = 0.5;
    });
    setNetworkInternal(net, '_slabDirty', true);
    getConnectionSlab(net);
    expect(hasGainArray(net)).toBe(true); // optional gain slab allocated

    // Reset all gains to neutral
    net.connections.forEach((connection) => {
      connection.gain = 1;
    });
    setNetworkInternal(net, '_slabDirty', true);
    getConnectionSlab(net);
    expect(hasGainArray(net)).toBe(false); // optional gain slab should be released
  });
});
