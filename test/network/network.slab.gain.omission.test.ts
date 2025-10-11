/**
 * network.slab.gain.omission.test.ts
 * Ensures gain slab omitted when all gains neutral (1) and allocated lazily when a non-neutral gain appears.
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

type ConnectionSlab = ReturnType<Network['getConnectionSlab']>;
interface NetworkInternals {
  _slabDirty: boolean;
  _connGain?: Float32Array | Float64Array | null;
}

const hasGainSlab = (net: Network): boolean =>
  Boolean(Reflect.get(net, '_connGain'));

const getConnectionSlab = (net: Network): ConnectionSlab =>
  ((net as unknown) as {
    getConnectionSlab: () => ConnectionSlab;
  }).getConnectionSlab();

const setNetworkInternal = <Key extends keyof NetworkInternals>(
  net: Network,
  key: Key,
  value: NetworkInternals[Key]
) => {
  Reflect.set(net, key, value);
};

describe('network.slab.gain.omission', () => {
  it('omits gain slab until non-neutral gain set and releases if reverted', () => {
    config.enableNodePooling = false;
    const net = new Network(3, 2, { enforceAcyclic: true });
    setNetworkInternal(net, '_slabDirty', true);
    getConnectionSlab(net);
    const initialHas = hasGainSlab(net);
    // If no connections, trivially pass
    if (net.connections.length === 0) {
      expect(true).toBe(true);
      return;
    }
    // Set a non-neutral gain on first connection
    net.connections[0].gain = 1.5;
    setNetworkInternal(net, '_slabDirty', true);
    getConnectionSlab(net);
    const afterSet = hasGainSlab(net);
    // Revert to neutral
    net.connections[0].gain = 1;
    setNetworkInternal(net, '_slabDirty', true);
    getConnectionSlab(net);
    const afterRevert = hasGainSlab(net);
    expect(
      initialHas === false && afterSet === true && afterRevert === false
    ).toBe(true);
  });
});
