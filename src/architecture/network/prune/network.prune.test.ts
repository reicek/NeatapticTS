import Network from '../network';
import {
  getCurrentSparsity,
  maybePrune,
  pruneToSparsity,
} from '../network.utils';

type PruningMethod = 'magnitude' | 'snip';

type PruningConfigSnapshot = {
  start: number;
  end: number;
  frequency: number;
  targetSparsity: number;
  method: PruningMethod;
  regrowFraction?: number;
  lastPruneIter?: number;
};

const xorDataset = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

function setPruningConfig(
  network: Network,
  pruningConfig: PruningConfigSnapshot,
): void {
  Reflect.set(network, '_pruningConfig', pruningConfig);
}

function getPruningConfig(network: Network): PruningConfigSnapshot | undefined {
  return Reflect.get(network, '_pruningConfig') as
    PruningConfigSnapshot | undefined;
}

function captureInitialConnectionBaseline(network: Network): void {
  Reflect.set(network, '_initialConnectionCount', network.connections.length);
}

describe('network prune chapter', () => {
  describe('configurePruning()', () => {
    describe('given magnitude pruning is scheduled during training', () => {
      describe('when the schedule reaches its target window', () => {
        it('keeps sparsity positive while retaining at least forty percent of baseline connections', () => {
          // Arrange
          const network = Network.createMLP(2, [4, 4], 1);
          const initialConnectionCount = network.connections.length;
          network.configurePruning({
            start: 1,
            end: 5,
            targetSparsity: 0.5,
            regrowFraction: 0,
            frequency: 1,
          });

          // Act
          for (
            let trainingIteration = 0;
            trainingIteration < 5;
            trainingIteration++
          ) {
            network.train(xorDataset, {
              iterations: 1,
              rate: 0.1,
              error: 0.00001,
              log: 0,
            });
          }

          const scheduleProducedExpectedSparsity =
            network.getCurrentSparsity() > 0 &&
            network.connections.length >=
              Math.floor(initialConnectionCount * 0.4);

          // Assert
          expect(scheduleProducedExpectedSparsity).toBe(true);
        });
      });
    });

    describe('given pruning regrowth is enabled during training', () => {
      describe('when the schedule prunes twice', () => {
        it('retains more than thirty percent of the original connections', () => {
          // Arrange
          const network = Network.createMLP(2, [3], 1);
          const initialConnectionCount = network.connections.length;
          network.configurePruning({
            start: 1,
            end: 2,
            targetSparsity: 0.4,
            regrowFraction: 0.5,
          });

          // Act
          network.train(xorDataset, {
            iterations: 1,
            rate: 0.1,
            error: 0.00001,
            log: 0,
          });
          network.train(xorDataset, {
            iterations: 1,
            rate: 0.1,
            error: 0.00001,
            log: 0,
          });
          const retainedEnoughConnections =
            network.connections.length > initialConnectionCount * 0.3;

          // Assert
          expect(retainedEnoughConnections).toBe(true);
        });
      });
    });
  });

  describe('maybePrune()', () => {
    describe('given no pruning config has been set on the network', () => {
      describe('when maybePrune() is called', () => {
        it('keeps the connection count unchanged', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 471,
            enforceAcyclic: true,
          });
          const connectionCountBeforePrune = network.connections.length;

          // Act
          maybePrune.call(network, 5);
          const connectionCountAfterPrune = network.connections.length;

          // Assert
          expect(connectionCountAfterPrune).toBe(connectionCountBeforePrune);
        });
      });
    });

    describe('given pruning config is set but no connection baseline was captured', () => {
      describe('when maybePrune() is called within the schedule window', () => {
        it('keeps the connection count unchanged', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 472,
            enforceAcyclic: true,
          });
          setPruningConfig(network, {
            start: 0,
            end: 10,
            frequency: 1,
            targetSparsity: 0.5,
            method: 'magnitude',
          });
          // No captureInitialConnectionBaseline call
          const connectionCountBeforePrune = network.connections.length;

          // Act
          maybePrune.call(network, 5);
          const connectionCountAfterPrune = network.connections.length;

          // Assert
          expect(connectionCountAfterPrune).toBe(connectionCountBeforePrune);
        });
      });
    });

    describe('given the current iteration is before the pruning window', () => {
      describe('when maybePrune() is called', () => {
        it('keeps the connection count unchanged', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 460,
            enforceAcyclic: true,
          });
          setPruningConfig(network, {
            start: 5,
            end: 10,
            frequency: 1,
            targetSparsity: 0.5,
            method: 'magnitude',
          });
          captureInitialConnectionBaseline(network);
          const connectionCountBeforePrune = network.connections.length;

          // Act
          maybePrune.call(network, 0);
          const connectionCountAfterPrune = network.connections.length;

          // Assert
          expect(connectionCountAfterPrune).toBe(connectionCountBeforePrune);
        });
      });
    });

    describe('given the active pruning window ramps to zero current sparsity', () => {
      describe('when maybePrune() is called at the window start', () => {
        it('stamps lastPruneIter without changing the connection count', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 461,
            enforceAcyclic: true,
          });
          const initialConnectionCount = network.connections.length;
          captureInitialConnectionBaseline(network);
          setPruningConfig(network, {
            start: 0,
            end: 10,
            frequency: 1,
            targetSparsity: 0.5,
            method: 'magnitude',
            regrowFraction: 0,
          });

          // Act
          maybePrune.call(network, 0);
          const pruningConfig = getPruningConfig(network);
          const prunedWithoutStructuralChange =
            pruningConfig?.lastPruneIter === 0 &&
            network.connections.length === initialConnectionCount;

          // Assert
          expect(prunedWithoutStructuralChange).toBe(true);
        });
      });
    });

    describe('given the pruning window reaches its target sparsity', () => {
      describe('when maybePrune() is called at the window end', () => {
        it('reduces the connection count', () => {
          // Arrange
          const network = new Network(3, 2, {
            seed: 462,
            enforceAcyclic: true,
          });
          setPruningConfig(network, {
            start: 0,
            end: 2,
            frequency: 1,
            targetSparsity: 0.5,
            method: 'magnitude',
          });
          captureInitialConnectionBaseline(network);
          const connectionCountBeforePrune = network.connections.length;

          // Act
          maybePrune.call(network, 2);
          const connectionCountAfterPrune = network.connections.length;

          // Assert
          expect(connectionCountAfterPrune).toBeLessThan(
            connectionCountBeforePrune,
          );
        });
      });
    });

    describe('given SNIP ranking is requested', () => {
      describe('when maybePrune() runs at full progress', () => {
        it('reduces the connection count', () => {
          // Arrange
          const network = new Network(3, 2, {
            seed: 463,
            enforceAcyclic: true,
          });
          setPruningConfig(network, {
            start: 0,
            end: 3,
            frequency: 1,
            targetSparsity: 0.4,
            method: 'snip',
          });
          captureInitialConnectionBaseline(network);
          const connectionCountBeforePrune = network.connections.length;

          // Act
          maybePrune.call(network, 3);
          const connectionCountAfterPrune = network.connections.length;

          // Assert
          expect(connectionCountAfterPrune).toBeLessThan(
            connectionCountBeforePrune,
          );
        });
      });
    });

    describe('given regrowth is enabled during scheduled pruning', () => {
      describe('when pruning runs on identical baseline networks', () => {
        it('keeps at least as many connections as pruning without regrowth', () => {
          // Arrange
          const pruneOnlyNetwork = new Network(3, 2, {
            seed: 464,
            enforceAcyclic: true,
          });
          const pruneAndRegrowNetwork = new Network(3, 2, {
            seed: 464,
            enforceAcyclic: true,
          });

          setPruningConfig(pruneOnlyNetwork, {
            start: 0,
            end: 0,
            frequency: 1,
            targetSparsity: 0.5,
            method: 'magnitude',
          });
          setPruningConfig(pruneAndRegrowNetwork, {
            start: 0,
            end: 0,
            frequency: 1,
            targetSparsity: 0.5,
            method: 'magnitude',
            regrowFraction: 1,
          });
          captureInitialConnectionBaseline(pruneOnlyNetwork);
          captureInitialConnectionBaseline(pruneAndRegrowNetwork);

          // Act
          maybePrune.call(pruneOnlyNetwork, 0);
          maybePrune.call(pruneAndRegrowNetwork, 0);
          const regrowthPreservedAtLeastAsManyConnections =
            pruneAndRegrowNetwork.connections.length >=
            pruneOnlyNetwork.connections.length;

          // Assert
          expect(regrowthPreservedAtLeastAsManyConnections).toBe(true);
        });
      });
    });
  });

  describe('getCurrentSparsity()', () => {
    describe('given no pruning baseline was captured', () => {
      describe('when getCurrentSparsity() is called', () => {
        it('returns zero', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 465,
            enforceAcyclic: true,
          });

          // Act
          const currentSparsity = getCurrentSparsity.call(network);

          // Assert
          expect(currentSparsity).toBe(0);
        });
      });
    });

    describe('given a baseline was captured before one connection was removed', () => {
      describe('when getCurrentSparsity() is called', () => {
        it('reports positive sparsity', () => {
          // Arrange
          const network = new Network(2, 2, {
            seed: 466,
            enforceAcyclic: true,
          });
          captureInitialConnectionBaseline(network);
          const removableConnection = network.connections[0];
          if (!removableConnection) {
            throw new Error('Expected at least one connection to remove');
          }

          network.disconnect(removableConnection.from, removableConnection.to);

          // Act
          const currentSparsity = getCurrentSparsity.call(network);

          // Assert
          expect(currentSparsity > 0).toBe(true);
        });
      });
    });
  });

  describe('pruneToSparsity()', () => {
    describe('given magnitude ranking is requested', () => {
      describe('when pruneToSparsity() is called', () => {
        it('reduces the connection count', () => {
          // Arrange
          const network = new Network(4, 2, {
            seed: 467,
            enforceAcyclic: true,
          });
          const connectionCountBeforePrune = network.connections.length;

          // Act
          pruneToSparsity.call(network, 0.3, 'magnitude');
          const connectionCountAfterPrune = network.connections.length;

          // Assert
          expect(connectionCountAfterPrune).toBeLessThan(
            connectionCountBeforePrune,
          );
        });
      });
    });

    describe('given SNIP ranking has no gradient statistics to use', () => {
      describe('when pruneToSparsity() is called', () => {
        it('still reduces the connection count through magnitude fallback', () => {
          // Arrange
          const network = new Network(4, 2, {
            seed: 468,
            enforceAcyclic: true,
          });
          const connectionCountBeforePrune = network.connections.length;

          // Act
          pruneToSparsity.call(network, 0.2, 'snip');
          const connectionCountAfterPrune = network.connections.length;

          // Assert
          expect(connectionCountAfterPrune).toBeLessThan(
            connectionCountBeforePrune,
          );
        });
      });
    });

    describe('given the requested target sparsity is zero', () => {
      describe('when pruneToSparsity() is called', () => {
        it('keeps the connection count unchanged', () => {
          // Arrange
          const network = new Network(3, 1, {
            seed: 469,
            enforceAcyclic: true,
          });
          const connectionCountBeforePrune = network.connections.length;

          // Act
          pruneToSparsity.call(network, 0);
          const connectionCountAfterPrune = network.connections.length;

          // Assert
          expect(connectionCountAfterPrune).toBe(connectionCountBeforePrune);
        });
      });
    });

    describe('given the network is already at or below the target sparsity', () => {
      describe('when pruneToSparsity() is called a second time at the same target', () => {
        it('keeps the connection count unchanged on the second call', () => {
          // Arrange – first call captures baseline and prunes to 30% sparsity
          const network = new Network(4, 2, {
            seed: 473,
            enforceAcyclic: true,
          });
          pruneToSparsity.call(network, 0.3);
          const connectionCountAfterFirstPrune = network.connections.length;

          // Act – second call finds no excess connections
          pruneToSparsity.call(network, 0.3);
          const connectionCountAfterSecondPrune = network.connections.length;

          // Assert
          expect(connectionCountAfterSecondPrune).toBe(
            connectionCountAfterFirstPrune,
          );
        });
      });
    });

    describe('given the requested target sparsity exceeds one', () => {
      describe('when pruneToSparsity() is called', () => {
        it('still leaves at least one connection in the network', () => {
          // Arrange
          const network = new Network(3, 1, {
            seed: 470,
            enforceAcyclic: true,
          });

          // Act
          pruneToSparsity.call(network, 1);
          const remainingConnectionCount = network.connections.length;

          // Assert
          expect(remainingConnectionCount > 0).toBe(true);
        });
      });
    });
  });
});
