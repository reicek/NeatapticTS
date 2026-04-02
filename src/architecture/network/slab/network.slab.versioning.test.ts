import Network from '../network';
import { config } from '../../../config';

type ConnectionSlab = ReturnType<Network['getConnectionSlab']>;

interface NetworkInternals {
  _slabDirty: boolean;
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
  describe('rebuild bookkeeping', () => {
    describe('given slab arrays are rebuilt after a structural change', () => {
      describe('when the rebuilt slab view is inspected', () => {
        it('increments the version and keeps flags and gain aligned with weights', () => {
          // Arrange
          config.enableNodePooling = false;
          const network = new Network(3, 2, { enforceAcyclic: true });
          const initialSlab = getConnectionSlab(network);
          const initialVersion = initialSlab.version;

          if (network.connections.length) {
            network.addNodeBetween();
          }

          // Act
          setNetworkInternal(network, '_slabDirty', true);
          const rebuiltSlab = getConnectionSlab(network);

          // Assert
          expect(
            rebuiltSlab.version > initialVersion &&
              ArrayBuffer.isView(rebuiltSlab.flags) &&
              ArrayBuffer.isView(rebuiltSlab.gain) &&
              rebuiltSlab.flags.length === rebuiltSlab.weights.length &&
              rebuiltSlab.gain.length === rebuiltSlab.weights.length,
          ).toBe(true);
        });
      });
    });
  });
});
