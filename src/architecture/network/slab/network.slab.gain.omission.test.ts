import Network from '../network';
import { config } from '../../../config';

type ConnectionSlab = ReturnType<Network['getConnectionSlab']>;

interface GainSlabInternals {
  _slabDirty: boolean;
  _connGain?: Float32Array | Float64Array | null;
}

function getConnectionSlab(network: Network): ConnectionSlab {
  return (
    network as unknown as {
      getConnectionSlab: () => ConnectionSlab;
    }
  ).getConnectionSlab();
}

function setGainSlabInternal<Key extends keyof GainSlabInternals>(
  network: Network,
  key: Key,
  value: GainSlabInternals[Key],
): void {
  Reflect.set(network, key, value);
}

function hasGainSlab(network: Network): boolean {
  return Boolean(Reflect.get(network, '_connGain'));
}

describe('network slab chapter', () => {
  describe('gain slab omission', () => {
    describe('given all gains start neutral', () => {
      describe('when one gain becomes non-neutral and later returns to one', () => {
        it('allocates the optional gain slab only while it is needed', () => {
          // Arrange
          config.enableNodePooling = false;
          const network = new Network(3, 2, { enforceAcyclic: true });

          setGainSlabInternal(network, '_slabDirty', true);
          getConnectionSlab(network);
          const hadGainInitially = hasGainSlab(network);

          if (network.connections.length === 0) {
            throw new Error(
              'Expected at least one connection for gain slab coverage',
            );
          }

          network.connections[0].gain = 1.5;
          setGainSlabInternal(network, '_slabDirty', true);
          getConnectionSlab(network);
          const hasGainAfterNonNeutralWrite = hasGainSlab(network);

          network.connections[0].gain = 1;

          // Act
          setGainSlabInternal(network, '_slabDirty', true);
          getConnectionSlab(network);
          const hasGainAfterReset = hasGainSlab(network);

          // Assert
          expect(
            hadGainInitially === false &&
              hasGainAfterNonNeutralWrite === true &&
              hasGainAfterReset === false,
          ).toBe(true);
        });
      });
    });

    describe('given several gains are non-neutral before a rebuild', () => {
      describe('when all gains are reset to one before the next rebuild', () => {
        it('releases the retained gain slab', () => {
          // Arrange
          const network = new Network(4, 2, { enforceAcyclic: true });

          if (network.connections.length === 0) {
            throw new Error(
              'Expected at least one connection for gain slab release coverage',
            );
          }

          network.connections.forEach((connection, connectionIndex) => {
            if (connectionIndex === 0) {
              connection.gain = 0.5;
            }
          });

          setGainSlabInternal(network, '_slabDirty', true);
          getConnectionSlab(network);
          const hadGainAfterActivation = hasGainSlab(network);

          network.connections.forEach((connection) => {
            connection.gain = 1;
          });

          // Act
          setGainSlabInternal(network, '_slabDirty', true);
          getConnectionSlab(network);
          const hasGainAfterReset = hasGainSlab(network);

          // Assert
          expect(
            hadGainAfterActivation === true && hasGainAfterReset === false,
          ).toBe(true);
        });
      });
    });
  });
});
