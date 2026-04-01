import Network from '../network';
import Node from '../../node';
import { Architect } from '../../../neataptic';
import * as methods from '../../../methods/methods';
import { computeTopoOrder, hasPath } from '../network.utils';

function setTopoOrder(network: Network, topologyOrder: Node[] | null): void {
  Reflect.set(network, '_topoOrder', topologyOrder);
}

function getTopoOrder(network: Network): Node[] | null {
  return Reflect.get(network, '_topoOrder') as Node[] | null;
}

function setTopoDirty(network: Network, isDirty: boolean): void {
  Reflect.set(network, '_topoDirty', isDirty);
}

function getTopoDirty(network: Network): boolean {
  return Boolean(Reflect.get(network, '_topoDirty'));
}

function getNetworkRandomGenerator(network: Network): () => number {
  return (Reflect.get(network, '_rand') as () => number) ?? Math.random;
}

describe('network topology chapter', () => {
  describe('getTopologyIntent()', () => {
    describe('given a perceptron builder creates the network', () => {
      describe('when the public topology intent is read', () => {
        it('marks the network as feed-forward', () => {
          // Arrange
          const network = Architect.perceptron(3, 4, 1);
          const expectedIntent = 'feed-forward';

          // Act
          const topologyIntent = network.getTopologyIntent();

          // Assert
          expect(topologyIntent).toBe(expectedIntent);
        });
      });

      describe('when the internal acyclic flag is inspected', () => {
        it('enables acyclic enforcement', () => {
          // Arrange
          const network = Architect.perceptron(3, 4, 1);

          // Act
          const enforceAcyclic = Reflect.get(network, '_enforceAcyclic');

          // Assert
          expect(enforceAcyclic).toBe(true);
        });
      });
    });

    describe('given the public constructor receives a feed-forward topology intent', () => {
      describe('when the public topology intent is read', () => {
        it('preserves the feed-forward intent', () => {
          // Arrange
          const network = new Network(2, 1, {
            topologyIntent: 'feed-forward',
          });
          const expectedIntent = 'feed-forward';

          // Act
          const topologyIntent = network.getTopologyIntent();

          // Assert
          expect(topologyIntent).toBe(expectedIntent);
        });
      });
    });

    describe('given a feed-forward perceptron is serialized', () => {
      describe('when the JSON payload is inspected', () => {
        it('stores the topology intent', () => {
          // Arrange
          const network = Architect.perceptron(2, 3, 1);
          const expectedIntent = 'feed-forward';

          // Act
          const jsonPayload = network.toJSON() as { topologyIntent?: string };

          // Assert
          expect(jsonPayload.topologyIntent).toBe(expectedIntent);
        });
      });
    });

    describe('given a serialized feed-forward perceptron is rebuilt', () => {
      describe('when the rebuilt intent is read', () => {
        it('restores the feed-forward intent', () => {
          // Arrange
          const network = Architect.perceptron(2, 3, 1);
          const expectedIntent = 'feed-forward';

          // Act
          const rebuiltNetwork = Network.fromJSON(
            network.toJSON() as Record<string, unknown>,
          );
          const topologyIntent = rebuiltNetwork.getTopologyIntent();

          // Assert
          expect(topologyIntent).toBe(expectedIntent);
        });
      });
    });
  });

  describe('setEnforceAcyclic()', () => {
    describe('given a legacy caller toggles the acyclic flag', () => {
      describe('when the public topology intent is read afterward', () => {
        it('keeps the semantic intent aligned', () => {
          // Arrange
          const network = new Network(2, 1);
          const expectedIntent = 'feed-forward';

          // Act
          network.setEnforceAcyclic(true);
          const topologyIntent = network.getTopologyIntent();

          // Assert
          expect(topologyIntent).toBe(expectedIntent);
        });
      });
    });

    describe('given recurrent mutations are attempted repeatedly', () => {
      describe('when acyclic enforcement is enabled first', () => {
        it('prevents backward and self connections from being added', () => {
          // Arrange
          const network = new Network(2, 1);
          network.setEnforceAcyclic(true);
          const connectionCountBeforeMutation = network.connections.length;
          const selfConnectionCountBeforeMutation = network.selfconns.length;

          // Act
          for (
            let mutationAttemptIndex = 0;
            mutationAttemptIndex < 10;
            mutationAttemptIndex++
          ) {
            network.mutate(methods.mutation.ADD_BACK_CONN);
            network.mutate(methods.mutation.ADD_SELF_CONN);
          }

          const mutationAttemptsPreservedAcyclicShape =
            network.connections.length === connectionCountBeforeMutation &&
            network.selfconns.length === selfConnectionCountBeforeMutation;

          // Assert
          expect(mutationAttemptsPreservedAcyclicShape).toBe(true);
        });
      });
    });
  });

  describe('computeTopoOrder()', () => {
    describe('given acyclic topology is enforced', () => {
      describe('when structural mutations dirty the cached order before activation', () => {
        it('recomputes a topological order covering all nodes', () => {
          // Arrange
          const network = new Network(3, 2, { enforceAcyclic: true });

          network.mutate(methods.mutation.ADD_NODE);
          network.mutate(methods.mutation.ADD_CONN);

          // Act
          network.activate([0, 0, 0]);
          const topologyOrder = Reflect.get(network, '_topoOrder') as unknown;
          const hasValidOrder =
            Array.isArray(topologyOrder) &&
            topologyOrder.length === network.nodes.length;

          // Assert
          expect(hasValidOrder).toBe(true);
        });
      });
    });

    describe('given a cycle is introduced into an otherwise feed-forward graph', () => {
      describe('when computeTopoOrder() is called directly', () => {
        it('falls back to raw node order length', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 72, enforceAcyclic: true });
          const hiddenNode = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );
          network.nodes.push(hiddenNode);
          network.connect(network.nodes[0], hiddenNode);
          network.connect(hiddenNode, network.nodes[1]);
          network.connect(network.nodes[1], hiddenNode);

          // Act
          computeTopoOrder.call(network);
          const topologyOrder = getTopoOrder(network);

          // Assert
          expect(topologyOrder?.length).toBe(network.nodes.length);
        });
      });
    });

    describe('given acyclic enforcement is disabled', () => {
      describe('when computeTopoOrder() is called directly', () => {
        it('clears the cached order and resets the dirty flag', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 73,
            enforceAcyclic: false,
          });
          setTopoOrder(network, [network.nodes[0]]);
          setTopoDirty(network, true);

          // Act
          computeTopoOrder.call(network);
          const topologyStateWasCleared =
            getTopoOrder(network) === null && getTopoDirty(network) === false;

          // Assert
          expect(topologyStateWasCleared).toBe(true);
        });
      });
    });
  });

  describe('hasPath()', () => {
    describe('given the target node is reachable', () => {
      describe('when hasPath() is called', () => {
        it('returns true', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 74, enforceAcyclic: true });
          const inputNode = network.nodes[0];
          const outputNode = network.nodes.at(-1);
          if (!outputNode) {
            throw new Error('Expected an output node');
          }

          // Act
          const targetIsReachable = hasPath.call(
            network,
            inputNode,
            outputNode,
          );

          // Assert
          expect(targetIsReachable).toBe(true);
        });
      });
    });

    describe('given the target node is unreachable', () => {
      describe('when hasPath() is called', () => {
        it('returns false', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 75, enforceAcyclic: true });
          network.connections.slice().forEach((connection) => {
            network.disconnect(connection.from, connection.to);
          });
          const inputNode = network.nodes[0];
          const outputNode = network.nodes.at(-1);
          if (!outputNode) {
            throw new Error('Expected an output node');
          }

          // Act
          const targetIsReachable = hasPath.call(
            network,
            inputNode,
            outputNode,
          );

          // Assert
          expect(targetIsReachable).toBe(false);
        });
      });
    });
  });
});
