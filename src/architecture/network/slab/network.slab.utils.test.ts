import Network from '../network';
import {
  canUseFastSlab,
  fastSlabActivate,
  getConnectionSlab,
  getSlabAllocationStats,
  getSlabVersion,
  rebuildConnectionSlabAsync,
} from './network.slab.utils';

function markAdjacencyDirty(network: Network): void {
  Reflect.set(network, '_adjDirty', true);
}

function markSlabDirty(network: Network): void {
  Reflect.set(network, '_slabDirty', true);
}

async function withTemporaryWindow<T>(callback: () => Promise<T>): Promise<T> {
  const hadWindow = Reflect.has(globalThis, 'window');
  const originalWindow = Reflect.get(globalThis, 'window');

  Reflect.set(globalThis, 'window', {});

  try {
    return await callback();
  } finally {
    if (hadWindow) {
      Reflect.set(globalThis, 'window', originalWindow);
    } else {
      Reflect.deleteProperty(globalThis, 'window');
    }
  }
}

async function withoutWindow<T>(callback: () => Promise<T>): Promise<T> {
  const hadWindow = Reflect.has(globalThis, 'window');
  const originalWindow = Reflect.get(globalThis, 'window');

  if (hadWindow) {
    Reflect.deleteProperty(globalThis, 'window');
  }

  try {
    return await callback();
  } finally {
    if (hadWindow) {
      Reflect.set(globalThis, 'window', originalWindow);
    }
  }
}

describe('network slab chapter', () => {
  describe('utility wrappers', () => {
    describe('given async slab rebuild runs without a browser window', () => {
      describe('when the direct async utility is called', () => {
        it('falls back to the synchronous rebuild path', async () => {
          // Arrange
          const network = new Network(3, 2, { enforceAcyclic: true });

          // Act
          const slabSnapshot = await withoutWindow(async () => {
            await rebuildConnectionSlabAsync.call(network, 4);

            return {
              used: getConnectionSlab.call(network).used,
              version: getSlabVersion.call(network),
            };
          });

          // Assert
          expect(slabSnapshot).toEqual({
            used: network.connections.length,
            version: 1,
          });
        });
      });
    });

    describe('given a clean network is rebuilt in a browser-like environment', () => {
      describe('when the direct async utility is called', () => {
        it('returns early without incrementing the slab version', async () => {
          // Arrange
          const network = new Network(3, 1, { enforceAcyclic: true });

          getConnectionSlab.call(network);
          const initialVersion = getSlabVersion.call(network);

          // Act
          const finalVersion = await withTemporaryWindow(async () => {
            await rebuildConnectionSlabAsync.call(network, 8);
            return getSlabVersion.call(network);
          });

          // Assert
          expect(finalVersion).toBe(initialVersion);
        });
      });
    });

    describe('given a dirty network is rebuilt in a browser-like environment', () => {
      describe('when the direct async utility is called', () => {
        it('publishes the rebuilt slab view with optional arrays', async () => {
          // Arrange
          const network = new Network(4, 2, { enforceAcyclic: true });

          network.connections[0].gain = 1.25;
          network.connections[0].plastic = true;
          network.connections[0].plasticityRate = 0.35;
          markSlabDirty(network);

          // Act
          const slabSnapshot = await withTemporaryWindow(async () => {
            await rebuildConnectionSlabAsync.call(network);
            return getConnectionSlab.call(network);
          });

          // Assert
          expect({
            hasGain: slabSnapshot.gain !== null,
            hasPlastic: slabSnapshot.plastic !== null,
            used: slabSnapshot.used,
            versionPositive: slabSnapshot.version > 0,
          }).toEqual({
            hasGain: true,
            hasPlastic: true,
            used: network.connections.length,
            versionPositive: true,
          });
        });
      });
    });

    describe('given a rebuilt slab network is queried through direct helper exports', () => {
      describe('when eligibility and version are read', () => {
        it('returns a boolean eligibility flag and a positive version', () => {
          // Arrange
          const network = new Network(2, 1, { enforceAcyclic: true });

          getConnectionSlab.call(network);

          // Act
          const slabState = {
            eligibilityType: typeof canUseFastSlab.call(network, false),
            versionPositive: getSlabVersion.call(network) > 0,
          };

          // Assert
          expect(slabState).toEqual({
            eligibilityType: 'boolean',
            versionPositive: true,
          });
        });
      });

      describe('when allocation statistics are read', () => {
        it('returns a serializable numeric snapshot', () => {
          // Arrange
          getSlabAllocationStats();

          // Act
          const allocationStats = getSlabAllocationStats();

          // Assert
          expect({
            freshType: typeof allocationStats.fresh,
            hasPool: typeof allocationStats.pool === 'object',
            pooledType: typeof allocationStats.pooled,
          }).toEqual({
            freshType: 'number',
            hasPool: true,
            pooledType: 'number',
          });
        });
      });

      describe('when fast slab activation is called directly', () => {
        it('matches the legacy activation output with and without adjacency rebuild', () => {
          // Arrange
          const network = new Network(4, 2, { enforceAcyclic: true });
          const inputVector = [0.25, -0.1, 0.4, 0.8];
          const legacyOutput = network.activate([...inputVector], false);

          markSlabDirty(network);
          markAdjacencyDirty(network);

          // Act
          const rebuiltAdjacencyOutput = fastSlabActivate.call(network, [
            ...inputVector,
          ]);
          const reusedAdjacencyOutput = fastSlabActivate.call(network, [
            ...inputVector,
          ]);

          // Assert
          expect({
            rebuiltAdjacencyOutput,
            reusedAdjacencyOutput,
          }).toStrictEqual({
            rebuiltAdjacencyOutput: legacyOutput,
            reusedAdjacencyOutput: legacyOutput,
          });
        });
      });
    });
  });
});
