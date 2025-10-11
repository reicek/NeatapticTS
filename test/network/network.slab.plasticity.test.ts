/**
 * network.slab.plasticity.test.ts
 * Verifies plasticity flag bit (bit3) and optional plasticity rate slab allocation is pay-for-use.
 */
import Network from '../../src/architecture/network';
import { config } from '../../src/config';

const countPlastic = (flags: Uint8Array): number => {
  let flagged = 0;
  for (let index = 0; index < flags.length; index++) {
    if (flags[index] & 0b1000) flagged++;
  }
  return flagged;
};

type ConnectionSlab = ReturnType<Network['getConnectionSlab']>;
type NetworkConnection = Network['connections'][number];
interface NetworkInternals {
  _slabDirty: boolean;
}

const setNetworkInternal = <Key extends keyof NetworkInternals>(
  net: Network,
  key: Key,
  value: NetworkInternals[Key]
) => {
  Reflect.set(net, key, value);
};

const getConnectionSlab = (net: Network): ConnectionSlab =>
  ((net as unknown) as {
    getConnectionSlab: () => ConnectionSlab;
  }).getConnectionSlab();

describe('network.slab.plasticity', () => {
  it('allocates plastic slab only when at least one connection has plasticityRate > 0 and releases when cleared', () => {
    config.enableNodePooling = false;
    config.enableSlabArrayPooling = true;
    const net = new Network(4, 2, { enforceAcyclic: true });
    setNetworkInternal(net, '_slabDirty', true);
    let slab = getConnectionSlab(net);
    const baseVersion = slab.version;
    const hadPlasticInitially = !!slab.plastic;
    // Assign plasticityRate to first connection
    if (net.connections.length === 0) {
      expect(true).toBe(true);
      return;
    }
    const firstConnection = net.connections[0] as NetworkConnection & {
      plasticityRate?: number;
    };
    firstConnection.plasticityRate = 0.05;
    setNetworkInternal(net, '_slabDirty', true);
    slab = getConnectionSlab(net);
    const plasticAfterSet = slab.plastic;
    const plasticCount = countPlastic(slab.flags.subarray(0, slab.used));
    // Clear plasticityRate
    firstConnection.plasticityRate = 0;
    setNetworkInternal(net, '_slabDirty', true);
    const slabAfterClear = getConnectionSlab(net);
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
