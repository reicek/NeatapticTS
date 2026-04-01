import Network from '../network';
import { config } from '../../../config';

type ConnectionSlab = ReturnType<Network['getConnectionSlab']>;
type NetworkConnection = Network['connections'][number];

interface NetworkInternals {
  _slabDirty: boolean;
}

function countPlasticFlags(flags: Uint8Array): number {
  let plasticFlagCount = 0;

  for (let flagIndex = 0; flagIndex < flags.length; flagIndex++) {
    if (flags[flagIndex] & 0b1000) {
      plasticFlagCount++;
    }
  }

  return plasticFlagCount;
}

function getConnectionSlab(network: Network): ConnectionSlab {
  return (
    network as unknown as {
      getConnectionSlab: () => ConnectionSlab;
    }
  ).getConnectionSlab();
}

function setNetworkInternal<Key extends keyof NetworkInternals>(
  network: Network,
  key: Key,
  value: NetworkInternals[Key],
): void {
  Reflect.set(network, key, value);
}

describe('network slab chapter', () => {
  describe('plasticity slab allocation', () => {
    describe('given one connection gains and then loses a plasticity rate', () => {
      describe('when the rebuilt slab view is inspected across both rebuilds', () => {
        it('allocates the plastic slab on demand and releases it after reset', () => {
          // Arrange
          config.enableNodePooling = false;
          config.enableSlabArrayPooling = true;
          const network = new Network(4, 2, { enforceAcyclic: true });

          if (network.connections.length === 0) {
            throw new Error(
              'Expected at least one connection for plasticity coverage',
            );
          }

          setNetworkInternal(network, '_slabDirty', true);
          const initialSlab = getConnectionSlab(network);
          const hadPlasticInitially = initialSlab.plastic !== null;
          const initialVersion = initialSlab.version;
          const firstConnection = network
            .connections[0] as NetworkConnection & {
            plasticityRate?: number;
          };

          firstConnection.plasticityRate = 0.05;
          setNetworkInternal(network, '_slabDirty', true);
          const plasticSlab = getConnectionSlab(network);
          const plasticFlagCount = countPlasticFlags(
            plasticSlab.flags.subarray(0, plasticSlab.used),
          );

          firstConnection.plasticityRate = 0;

          // Act
          setNetworkInternal(network, '_slabDirty', true);
          const resetSlab = getConnectionSlab(network);

          // Assert
          expect(
            hadPlasticInitially === false &&
              plasticSlab.plastic !== null &&
              plasticFlagCount >= 1 &&
              plasticSlab.version > initialVersion &&
              resetSlab.plastic === null,
          ).toBe(true);
        });
      });
    });
  });
});
