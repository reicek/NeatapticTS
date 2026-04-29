import {
  memoryStats,
  registerTrackedNetwork,
  resetMemoryTracking,
  unregisterTrackedNetwork,
  type NetworkView,
} from './memory';

function buildTrackedNetwork(
  connectionCount: number,
  nodeCount: number,
): NetworkView {
  return {
    connections: Array.from({ length: connectionCount }, () => ({})),
    nodes: Array.from({ length: nodeCount }, () => ({})),
  };
}

describe('utils chapter', () => {
  describe('memory registry helpers', () => {
    beforeEach(() => {
      // Arrange
      resetMemoryTracking();
    });

    afterEach(() => {
      jest.restoreAllMocks();
      jest.resetModules();
      resetMemoryTracking();
    });

    describe('given the same network is registered twice', () => {
      describe('when capturing memoryStats without explicit targets', () => {
        it('keeps only one tracked copy of the network', () => {
          // Arrange
          const trackedNetwork = buildTrackedNetwork(2, 1);
          registerTrackedNetwork(trackedNetwork);
          registerTrackedNetwork(trackedNetwork);

          // Act
          const snapshot = memoryStats();

          // Assert
          expect({
            connections: snapshot.connections,
            nodes: snapshot.nodes,
          }).toStrictEqual({
            connections: 2,
            nodes: 1,
          });
        });
      });
    });

    describe('given a null network registration', () => {
      describe('when capturing memoryStats without explicit targets', () => {
        it('ignores the null registration entry', () => {
          // Arrange
          registerTrackedNetwork(null);

          // Act
          const snapshot = memoryStats();

          // Assert
          expect({
            connections: snapshot.connections,
            nodes: snapshot.nodes,
          }).toStrictEqual({
            connections: 0,
            nodes: 0,
          });
        });
      });
    });

    describe('given a tracked network already exists', () => {
      describe('when resetMemoryTracking runs', () => {
        it('clears the tracked registry used by memoryStats', () => {
          // Arrange
          registerTrackedNetwork(buildTrackedNetwork(3, 2));

          // Act
          resetMemoryTracking();
          const snapshot = memoryStats();

          // Assert
          expect({
            connections: snapshot.connections,
            nodes: snapshot.nodes,
          }).toStrictEqual({
            connections: 0,
            nodes: 0,
          });
        });
      });
    });

    describe('given a tracked network is explicitly unregistered', () => {
      describe('when capturing memoryStats without explicit targets', () => {
        it('removes that network from the registry', () => {
          // Arrange
          const trackedNetwork = buildTrackedNetwork(4, 2);
          registerTrackedNetwork(trackedNetwork);

          // Act
          unregisterTrackedNetwork(trackedNetwork);
          const snapshot = memoryStats();

          // Assert
          expect({
            connections: snapshot.connections,
            nodes: snapshot.nodes,
          }).toStrictEqual({
            connections: 0,
            nodes: 0,
          });
        });
      });
    });

    describe('given an unrelated network is unregistered', () => {
      describe('when capturing memoryStats without explicit targets', () => {
        it('leaves the existing tracked network intact', () => {
          // Arrange
          const trackedNetwork = buildTrackedNetwork(5, 3);
          const unrelatedNetwork = buildTrackedNetwork(1, 1);
          registerTrackedNetwork(trackedNetwork);

          // Act
          unregisterTrackedNetwork(unrelatedNetwork);
          const snapshot = memoryStats();

          // Assert
          expect({
            connections: snapshot.connections,
            nodes: snapshot.nodes,
          }).toStrictEqual({
            connections: 5,
            nodes: 3,
          });
        });
      });
    });

    describe('given the node-pool export is not a function', () => {
      describe('when capturing memoryStats from an isolated module instance', () => {
        it('falls back to a null nodePool snapshot', () => {
          // Arrange
          let nodePoolSnapshot: unknown;

          // Act
          jest.doMock('../architecture/nodePool', () => ({
            nodePoolStats: null,
          }));

          jest.isolateModules(() => {
            const isolatedMemoryModule =
              require('./memory') as typeof import('./memory');
            nodePoolSnapshot = isolatedMemoryModule.memoryStats(
              buildTrackedNetwork(0, 0),
            ).pools.nodePool;
          });

          jest.dontMock('../architecture/nodePool');

          // Assert
          expect(nodePoolSnapshot).toBeNull();
        });
      });
    });
  });
});
