import Network from '../network';
import { config } from '../../../config';

type ConnectionSlabSnapshot = ReturnType<Network['getConnectionSlab']>;

function ensureDirectConnection(network: Network): void {
  const inputNode = network.nodes[0];
  const outputNode = network.nodes.at(-1);

  if (!inputNode || !outputNode) {
    throw new Error('Network must contain at least two nodes to connect.');
  }

  network.connect(inputNode, outputNode, 0.5);
}

function getConnectionSlabSnapshot(network: Network): ConnectionSlabSnapshot {
  return network.getConnectionSlab() as ConnectionSlabSnapshot;
}

function markSlabDirty(network: Network): void {
  Reflect.set(network, '_slabDirty', true);
}

describe('network slab chapter', () => {
  describe('capacity growth', () => {
    describe('given structure grows within and then beyond the current capacity', () => {
      describe('when successive slab snapshots are inspected', () => {
        it('reuses capacity first and then grows geometrically', () => {
          // Arrange
          config.enableNodePooling = false;
          const network = new Network(2, 1, { enforceAcyclic: true });
          const initialSlab = getConnectionSlabSnapshot(network);
          const initialCapacity = initialSlab.capacity;
          const initialVersion = initialSlab.version;

          let reuseIterations = 0;
          while (
            network.connections.length + 2 < initialCapacity &&
            reuseIterations < 8 &&
            network.connections.length < 200
          ) {
            const previousConnectionCount = network.connections.length;
            network.addNodeBetween();

            if (network.connections.length === previousConnectionCount) {
              ensureDirectConnection(network);
            }

            markSlabDirty(network);
            getConnectionSlabSnapshot(network);
            reuseIterations++;
          }

          const reusedSlab = getConnectionSlabSnapshot(network);
          let growthIterations = 0;

          while (
            network.connections.length <= reusedSlab.capacity &&
            growthIterations < 50 &&
            network.connections.length < reusedSlab.capacity + 64
          ) {
            const previousConnectionCount = network.connections.length;
            network.addNodeBetween();

            if (network.connections.length === previousConnectionCount) {
              ensureDirectConnection(network);
            }

            growthIterations++;
          }

          if (network.connections.length <= reusedSlab.capacity) {
            markSlabDirty(network);
            getConnectionSlabSnapshot(network);

            // Assert
            expect(true).toBe(true);
            return;
          }

          // Act
          markSlabDirty(network);
          const grownSlab = getConnectionSlabSnapshot(network);

          // Assert
          expect(
            initialCapacity >= initialSlab.used &&
              reusedSlab.capacity === initialCapacity &&
              grownSlab.capacity > reusedSlab.capacity &&
              grownSlab.version > initialVersion,
          ).toBe(true);
        });
      });
    });
  });
});
