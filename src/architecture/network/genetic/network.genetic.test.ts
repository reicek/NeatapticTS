import { Network, methods } from '../../../neataptic';
import Node from '../../node';

function createCrossOverCallback(
  firstParent: Network | null | undefined,
  secondParent: Network | null | undefined,
  equalFlag?: boolean,
): () => Network {
  if (typeof equalFlag === 'boolean') {
    return () =>
      Network.crossOver(
        firstParent as Network,
        secondParent as Network,
        equalFlag,
      );
  }

  return () =>
    Network.crossOver(firstParent as Network, secondParent as Network);
}

function suppressConsoleWarn(callback: () => void): void {
  const originalWarn = console.warn;
  console.warn = jest.fn();

  try {
    callback();
  } finally {
    console.warn = originalWarn;
  }
}

function buildParentNetworkWithAddedNodes(addedNodeCount: number): Network {
  const network = new Network(2, 1);

  for (
    let nodeMutationIndex = 0;
    nodeMutationIndex < addedNodeCount;
    nodeMutationIndex++
  ) {
    network.mutate(methods.mutation.ADD_NODE);
  }

  return network;
}

function buildAcyclicParentNetwork(
  seed: number,
  addedNodeCount: number,
  addedConnectionCount: number,
): Network {
  const network = new Network(2, 2, { seed, enforceAcyclic: true });

  for (
    let nodeMutationIndex = 0;
    nodeMutationIndex < addedNodeCount;
    nodeMutationIndex++
  ) {
    network.mutate(methods.mutation.ADD_NODE);
  }

  for (
    let connectionMutationIndex = 0;
    connectionMutationIndex < addedConnectionCount;
    connectionMutationIndex++
  ) {
    network.mutate(methods.mutation.ADD_CONN);
  }

  return network;
}

function areAllConnectionsFeedForward(network: Network): boolean {
  return network.connections.every((candidateConnection) => {
    const sourceNode = candidateConnection.from as Node | undefined;
    const targetNode = candidateConnection.to as Node | undefined;

    if (!sourceNode || !targetNode) {
      return false;
    }

    const sourceNodeIndex = network.nodes.indexOf(sourceNode);
    const targetNodeIndex = network.nodes.indexOf(targetNode);

    return (
      sourceNodeIndex !== -1 &&
      targetNodeIndex !== -1 &&
      sourceNodeIndex < targetNodeIndex
    );
  });
}

describe('network genetic chapter', () => {
  describe('Network.crossOver()', () => {
    describe('given both parent networks enforce acyclic topology', () => {
      describe('when crossover materializes the offspring graph', () => {
        it('keeps every offspring connection feed-forward', () => {
          // Arrange
          const parentNetwork1 = buildAcyclicParentNetwork(410, 20, 60);
          const parentNetwork2 = buildAcyclicParentNetwork(411, 40, 20);

          // Act
          const offspringNetwork = Network.crossOver(
            parentNetwork1,
            parentNetwork2,
          );
          const offspringIsFeedForward =
            areAllConnectionsFeedForward(offspringNetwork);

          // Assert
          expect(offspringIsFeedForward).toBe(true);
        });
      });
    });

    describe('given equal-fitness parents with different hidden-node counts', () => {
      let firstParent: Network;
      let secondParent: Network;
      let offspringNetwork: Network;

      beforeEach(() => {
        // Arrange
        firstParent = buildParentNetworkWithAddedNodes(1);
        secondParent = buildParentNetworkWithAddedNodes(2);
        firstParent.score = 1;
        secondParent.score = 1;

        // Act
        offspringNetwork = Network.crossOver(firstParent, secondParent, true);
      });

      describe('when the offspring size is chosen symmetrically', () => {
        it('keeps the node count at or above the smaller parent', () => {
          // Assert
          expect(offspringNetwork.nodes.length).toBeGreaterThanOrEqual(
            Math.min(firstParent.nodes.length, secondParent.nodes.length),
          );
        });

        it('keeps the node count at or below the larger parent', () => {
          // Assert
          expect(offspringNetwork.nodes.length).toBeLessThanOrEqual(
            Math.max(firstParent.nodes.length, secondParent.nodes.length),
          );
        });
      });
    });

    describe('given the fitter parent has more hidden nodes', () => {
      let secondParent: Network;
      let offspringNetwork: Network;

      beforeEach(() => {
        // Arrange
        const firstParent = buildParentNetworkWithAddedNodes(1);
        secondParent = buildParentNetworkWithAddedNodes(2);
        firstParent.score = 1;
        secondParent.score = 2;

        // Act
        offspringNetwork = Network.crossOver(firstParent, secondParent, false);
      });

      describe('when fitter-parent inheritance decides offspring size', () => {
        it('matches the fitter parent node count', () => {
          // Assert
          expect(offspringNetwork.nodes.length).toBe(secondParent.nodes.length);
        });
      });
    });

    describe('given the parent networks expose incompatible interfaces', () => {
      describe('when the input width differs', () => {
        it('throws', () => {
          // Arrange
          const firstParent = new Network(2, 1);
          const secondParent = new Network(3, 1);
          const crossOverCallback = () =>
            Network.crossOver(firstParent, secondParent);

          // Assert
          expect(crossOverCallback).toThrow();
        });
      });

      describe('when the output width differs', () => {
        it('throws', () => {
          // Arrange
          const firstParent = new Network(2, 1);
          const secondParent = new Network(2, 2);
          const crossOverCallback = () =>
            Network.crossOver(firstParent, secondParent);

          // Assert
          expect(crossOverCallback).toThrow();
        });
      });
    });

    describe('given parent references are missing', () => {
      describe('when both parents are undefined', () => {
        it('throws', () => {
          // Arrange
          const crossOverCallback = createCrossOverCallback(
            undefined,
            undefined,
          );

          // Act / Assert
          suppressConsoleWarn(() => {
            expect(crossOverCallback).toThrow();
          });
        });
      });

      describe('when both parents are null', () => {
        it('throws', () => {
          // Arrange
          const crossOverCallback = createCrossOverCallback(null, null);

          // Act / Assert
          suppressConsoleWarn(() => {
            expect(crossOverCallback).toThrow();
          });
        });
      });

      describe('when the first parent is undefined', () => {
        it('throws', () => {
          // Arrange
          const crossOverCallback = createCrossOverCallback(
            undefined,
            new Network(2, 1),
          );

          // Act / Assert
          suppressConsoleWarn(() => {
            expect(crossOverCallback).toThrow();
          });
        });
      });

      describe('when the second parent is undefined', () => {
        it('throws', () => {
          // Arrange
          const crossOverCallback = createCrossOverCallback(
            new Network(2, 1),
            undefined,
          );

          // Act / Assert
          suppressConsoleWarn(() => {
            expect(crossOverCallback).toThrow();
          });
        });
      });
    });

    describe('given the parent graphs are structurally simple', () => {
      describe('when both parents expose no registered connections', () => {
        let firstParent: Network;
        let offspringNetwork: Network;

        beforeEach(() => {
          // Arrange
          firstParent = new Network(2, 1);
          firstParent.connections = [];
          const secondParent = new Network(2, 1);
          secondParent.connections = [];

          // Act
          offspringNetwork = Network.crossOver(firstParent, secondParent, true);
        });

        describe('when the offspring is materialized', () => {
          it('preserves the parent node count', () => {
            // Assert
            expect(offspringNetwork.nodes.length).toBe(
              firstParent.nodes.length,
            );
          });
        });
      });

      describe('when both parents have the same score and size', () => {
        let firstParent: Network;
        let offspringNetwork: Network;

        beforeEach(() => {
          // Arrange
          firstParent = new Network(2, 1);
          const secondParent = new Network(2, 1);
          firstParent.score = 1;
          secondParent.score = 1;

          // Act
          offspringNetwork = Network.crossOver(firstParent, secondParent, true);
        });

        describe('when crossover chooses from equivalent parents', () => {
          it('keeps the shared node count unchanged', () => {
            // Assert
            expect(offspringNetwork.nodes.length).toBe(
              firstParent.nodes.length,
            );
          });
        });
      });
    });
  });
});
