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
  describe('flag packing', () => {
    describe('given one connection is disabled and then re-enabled', () => {
      describe('when the rebuilt slab flags are inspected', () => {
        it('updates the enabled bit without forcing capacity growth', () => {
          // Arrange
          config.enableNodePooling = false;
          const network = new Network(3, 2, { enforceAcyclic: true });
          const initialSlab = getConnectionSlab(network);

          if (network.connections.length === 0) {
            throw new Error(
              'Expected at least one connection for slab flag coverage',
            );
          }

          const initialCapacity = initialSlab.capacity;

          network.connections[0].enabled = false;
          setNetworkInternal(network, '_slabDirty', true);
          const disabledSlab = getConnectionSlab(network);
          const disabledFlagByte = disabledSlab.flags[0];

          network.connections[0].enabled = true;

          // Act
          setNetworkInternal(network, '_slabDirty', true);
          const reenabledSlab = getConnectionSlab(network);
          const reenabledFlagByte = reenabledSlab.flags[0];

          // Assert
          expect(
            initialCapacity === disabledSlab.capacity &&
              (disabledFlagByte & 0b1) === 0 &&
              (reenabledFlagByte & 0b1) === 1 &&
              reenabledSlab.version > initialSlab.version,
          ).toBe(true);
        });
      });
    });
  });
});
