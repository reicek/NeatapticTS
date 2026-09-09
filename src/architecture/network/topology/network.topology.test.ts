import Network from '../network';
import Node from '../../node';
import { Architect } from '../../../neataptic';
import * as methods from '../../../methods/methods';
import { computeTopoOrder, hasPath } from '../network.utils';
import * as topologyUtilsFacade from './network.topology.utils';

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

interface ActivationScheduleSnapshot {
  stateSemantics?: 'carry';
  mode: 'acyclic' | 'recurrent';
  steps: ActivationScheduleStepSnapshot[];
  outputNodeIds: number[];
}

interface ActivationScheduleStepSnapshot {
  kind: 'wave' | 'recurrent-component';
  nodeIds: number[];
  iterations?: number;
}

function getActivationSchedule(
  network: Network,
): ActivationScheduleSnapshot | null {
  return Reflect.get(
    network,
    '_activationSchedule',
  ) as ActivationScheduleSnapshot | null;
}

function setActivationSchedule(
  network: Network,
  activationSchedule: ActivationScheduleSnapshot | null,
): void {
  Reflect.set(network, '_activationSchedule', activationSchedule);
}

function getNetworkRandomGenerator(network: Network): () => number {
  return (Reflect.get(network, '_rand') as () => number) ?? (() => 0.5);
}

describe('network topology chapter', () => {
  describe('network.topology.utils facade re-exports', () => {
    it('exports all topology facade functions as callable values', () => {
      expect({
        computeTopoOrder: typeof topologyUtilsFacade.computeTopoOrder,
        hasPath: typeof topologyUtilsFacade.hasPath,
        getTopologyIntent: typeof topologyUtilsFacade.getTopologyIntent,
        hasFeedForwardTopologyContract:
          typeof topologyUtilsFacade.hasFeedForwardTopologyContract,
        setEnforceAcyclic: typeof topologyUtilsFacade.setEnforceAcyclic,
        setTopologyIntent: typeof topologyUtilsFacade.setTopologyIntent,
        createMLP: typeof topologyUtilsFacade.createMLP,
        rebuildConnections: typeof topologyUtilsFacade.rebuildConnections,
      }).toEqual({
        computeTopoOrder: 'function',
        hasPath: 'function',
        getTopologyIntent: 'function',
        hasFeedForwardTopologyContract: 'function',
        setEnforceAcyclic: 'function',
        setTopologyIntent: 'function',
        createMLP: 'function',
        rebuildConnections: 'function',
      });
    });
  });

  describe('explicit IO role storage', () => {
    describe('given the public constructor creates a starter graph', () => {
      describe('when the explicit role metadata is read', () => {
        it('stores ordered input and output node gene ids', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 901 });
          const expectedRoles = {
            inputNodeIds: network.nodes
              .filter((node) => node.type === 'input')
              .map((node) => node.geneId),
            outputNodeIds: network.nodes
              .filter((node) => node.type === 'output')
              .map((node) => node.geneId),
          };

          // Act
          const explicitRoles = {
            inputNodeIds: network.inputNodeIds,
            outputNodeIds: network.outputNodeIds,
          };

          // Assert
          expect(explicitRoles).toEqual(expectedRoles);
        });
      });
    });

    describe('given a perceptron builder replaces the network node list', () => {
      describe('when the explicit role metadata is read', () => {
        it('refreshes ordered input and output node gene ids', () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);
          const expectedRoles = {
            inputNodeIds: network.nodes
              .filter((node) => node.type === 'input')
              .map((node) => node.geneId),
            outputNodeIds: network.nodes
              .filter((node) => node.type === 'output')
              .map((node) => node.geneId),
          };

          // Act
          const explicitRoles = {
            inputNodeIds: network.inputNodeIds,
            outputNodeIds: network.outputNodeIds,
          };

          // Assert
          expect(explicitRoles).toEqual(expectedRoles);
        });
      });
    });
  });

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

      describe('when an acyclic graph has a parallel hidden layer', () => {
        it('builds a deterministic activation schedule grouped by wave', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 76,
            enforceAcyclic: true,
          });
          const inputNodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'input',
          );
          const outputNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'output',
          );

          if (!outputNode) {
            throw new Error('Expected one output node for schedule test');
          }

          const hiddenNodeLeft = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );
          const hiddenNodeRight = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );

          network.nodes = [
            inputNodes[1],
            inputNodes[0],
            hiddenNodeRight,
            hiddenNodeLeft,
            outputNode,
          ];

          network.connections.slice().forEach((connection) => {
            network.disconnect(connection.from, connection.to);
          });

          network.connect(inputNodes[0], hiddenNodeLeft);
          network.connect(inputNodes[1], hiddenNodeLeft);
          network.connect(inputNodes[0], hiddenNodeRight);
          network.connect(inputNodes[1], hiddenNodeRight);
          network.connect(hiddenNodeLeft, outputNode);
          network.connect(hiddenNodeRight, outputNode);

          const expectedSchedule = {
            mode: 'acyclic' as const,
            steps: [
              {
                kind: 'wave' as const,
                nodeIds: inputNodes
                  .map((nodeEntry) => nodeEntry.geneId)
                  .toSorted((left, right) => left - right),
              },
              {
                kind: 'wave' as const,
                nodeIds: [
                  hiddenNodeLeft.geneId,
                  hiddenNodeRight.geneId,
                ].toSorted((left, right) => left - right),
              },
              {
                kind: 'wave' as const,
                nodeIds: [outputNode.geneId],
              },
            ],
            outputNodeIds: [outputNode.geneId],
          };

          // Act
          computeTopoOrder.call(network);
          const activationSchedule = getActivationSchedule(network);

          // Assert
          expect(activationSchedule).toEqual(expectedSchedule);
        });
      });

      describe('when the same acyclic graph is stored in a different node-array order', () => {
        it('keeps the cached activation schedule stable', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 77,
            enforceAcyclic: true,
          });
          const inputNodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'input',
          );
          const outputNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'output',
          );

          if (!outputNode) {
            throw new Error(
              'Expected one output node for schedule stability test',
            );
          }

          const hiddenNode = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );

          network.nodes = [
            inputNodes[1],
            inputNodes[0],
            hiddenNode,
            outputNode,
          ];
          network.connections.slice().forEach((connection) => {
            network.disconnect(connection.from, connection.to);
          });
          network.connect(inputNodes[0], hiddenNode);
          network.connect(inputNodes[1], hiddenNode);
          network.connect(hiddenNode, outputNode);

          computeTopoOrder.call(network);
          const initialSchedule = structuredClone(
            getActivationSchedule(network),
          );

          network.nodes = network.nodes.toReversed();
          setTopoDirty(network, true);

          // Act
          computeTopoOrder.call(network);
          const reorderedSchedule = getActivationSchedule(network);

          // Assert
          expect(reorderedSchedule).toEqual(initialSchedule);
        });
      });
    });

    describe('given recurrent topology is allowed', () => {
      describe('when a cyclic hidden component sits between inputs and outputs', () => {
        it('builds a deterministic recurrent schedule and clears the acyclic order cache', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 78,
            enforceAcyclic: false,
          });
          const inputNodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'input',
          );
          const outputNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'output',
          );

          if (!outputNode) {
            throw new Error(
              'Expected one output node for recurrent schedule test',
            );
          }

          const hiddenNodeLeft = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );
          const hiddenNodeRight = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );

          network.nodes = [
            hiddenNodeRight,
            inputNodes[1],
            outputNode,
            hiddenNodeLeft,
            inputNodes[0],
          ];

          network.connections.slice().forEach((connection) => {
            network.disconnect(connection.from, connection.to);
          });

          network.connect(inputNodes[0], hiddenNodeLeft);
          network.connect(inputNodes[1], hiddenNodeRight);
          network.connect(hiddenNodeLeft, hiddenNodeRight);
          network.connect(hiddenNodeRight, hiddenNodeLeft);
          network.connect(hiddenNodeLeft, outputNode);
          network.connect(hiddenNodeRight, outputNode);

          const expectedTopologyState = {
            topologyOrder: null,
            activationSchedule: {
              mode: 'recurrent' as const,
              steps: [
                {
                  kind: 'wave' as const,
                  nodeIds: inputNodes
                    .map((nodeEntry) => nodeEntry.geneId)
                    .toSorted((left, right) => left - right),
                },
                {
                  kind: 'recurrent-component' as const,
                  nodeIds: [
                    hiddenNodeLeft.geneId,
                    hiddenNodeRight.geneId,
                  ].toSorted((left, right) => left - right),
                  iterations: 1,
                },
                {
                  kind: 'wave' as const,
                  nodeIds: [outputNode.geneId],
                },
              ],
              outputNodeIds: [outputNode.geneId],
              stateSemantics: 'carry' as const,
            },
            topoDirty: false,
          };

          // Act
          computeTopoOrder.call(network);
          const topologyState = {
            topologyOrder: getTopoOrder(network),
            activationSchedule: getActivationSchedule(network),
            topoDirty: getTopoDirty(network),
          };

          // Assert
          expect(topologyState).toEqual(expectedTopologyState);
        });
      });

      describe('when the same recurrent graph is stored in a different node-array order', () => {
        it('keeps the cached recurrent schedule stable', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 79,
            enforceAcyclic: false,
          });
          const inputNodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'input',
          );
          const outputNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'output',
          );

          if (!outputNode) {
            throw new Error(
              'Expected one output node for recurrent stability test',
            );
          }

          const hiddenNodeLeft = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );
          const hiddenNodeRight = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );

          network.nodes = [
            inputNodes[0],
            hiddenNodeRight,
            outputNode,
            hiddenNodeLeft,
            inputNodes[1],
          ];
          network.connections.slice().forEach((connection) => {
            network.disconnect(connection.from, connection.to);
          });

          network.connect(inputNodes[0], hiddenNodeLeft);
          network.connect(inputNodes[1], hiddenNodeRight);
          network.connect(hiddenNodeLeft, hiddenNodeRight);
          network.connect(hiddenNodeRight, hiddenNodeLeft);
          network.connect(hiddenNodeLeft, outputNode);
          network.connect(hiddenNodeRight, outputNode);

          computeTopoOrder.call(network);
          const initialSchedule = structuredClone(
            getActivationSchedule(network),
          );

          network.nodes = network.nodes.toReversed();
          setTopoDirty(network, true);

          // Act
          computeTopoOrder.call(network);
          const reorderedSchedule = getActivationSchedule(network);

          // Assert
          expect(reorderedSchedule).toEqual(initialSchedule);
        });
      });

      describe('when a single hidden node only carries a self-loop recurrence', () => {
        it('marks the singleton step as a recurrent component', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 80,
            enforceAcyclic: false,
          });
          const inputNode = network.nodes[0];
          const outputNode = network.nodes[1];
          const hiddenNode = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );

          network.nodes = [outputNode, hiddenNode, inputNode];
          network.connections.slice().forEach((connection) => {
            network.disconnect(connection.from, connection.to);
          });

          network.connect(inputNode, hiddenNode);
          network.connect(hiddenNode, hiddenNode);
          network.connect(hiddenNode, outputNode);

          const expectedSchedule = {
            mode: 'recurrent' as const,
            steps: [
              {
                kind: 'wave' as const,
                nodeIds: [inputNode.geneId],
              },
              {
                kind: 'recurrent-component' as const,
                nodeIds: [hiddenNode.geneId],
                iterations: 1,
              },
              {
                kind: 'wave' as const,
                nodeIds: [outputNode.geneId],
              },
            ],
            outputNodeIds: [outputNode.geneId],
            stateSemantics: 'carry' as const,
          };

          // Act
          computeTopoOrder.call(network);
          const activationSchedule = getActivationSchedule(network);

          // Assert
          expect(activationSchedule).toEqual(expectedSchedule);
        });
      });
    });

    describe('given a cycle is introduced into an otherwise feed-forward graph', () => {
      describe('when computeTopoOrder() is called directly', () => {
        it('falls back to raw node order and clears the acyclic schedule', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 72, enforceAcyclic: true });
          const inputNode = network.nodes[0];
          const outputNode = network.nodes[1];
          const hiddenNode = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );
          network.nodes.push(hiddenNode);

          inputNode.connect(hiddenNode);
          hiddenNode.connect(outputNode);
          outputNode.connect(hiddenNode);
          Network.rebuildConnections(network);

          // Act
          computeTopoOrder.call(network);
          const topologyOrder = getTopoOrder(network);
          const topologyState = {
            hasFallbackOrder:
              (topologyOrder?.length ?? 0) === network.nodes.length,
            activationSchedule: getActivationSchedule(network),
          };

          // Assert
          expect(topologyState).toEqual({
            hasFallbackOrder: true,
            activationSchedule: null,
          });
        });
      });
    });

    describe('given acyclic enforcement is disabled', () => {
      describe('when computeTopoOrder() is called directly', () => {
        it('replaces stale acyclic cache state with a recurrent schedule and clears the dirty flag', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 73,
            enforceAcyclic: false,
          });
          const inputNodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'input',
          );
          const outputNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'output',
          );

          if (!outputNode) {
            throw new Error(
              'Expected one output node for recurrent replacement test',
            );
          }

          const hiddenNode = new Node(
            'hidden',
            undefined,
            getNetworkRandomGenerator(network),
          );

          network.nodes = [
            inputNodes[1],
            outputNode,
            hiddenNode,
            inputNodes[0],
          ];
          network.connections.slice().forEach((connection) => {
            network.disconnect(connection.from, connection.to);
          });
          network.connect(inputNodes[0], hiddenNode);
          network.connect(inputNodes[1], hiddenNode);
          network.connect(hiddenNode, hiddenNode);
          network.connect(hiddenNode, outputNode);

          setTopoOrder(network, [network.nodes[0]]);
          setActivationSchedule(network, {
            mode: 'acyclic',
            steps: [
              {
                kind: 'wave',
                nodeIds: [network.nodes[0].geneId],
              },
            ],
            outputNodeIds: network.outputNodeIds,
          });
          setTopoDirty(network, true);

          // Act
          computeTopoOrder.call(network);
          const topologyStateWasCleared = {
            topologyOrder: getTopoOrder(network),
            activationSchedule: getActivationSchedule(network),
            topoDirty: getTopoDirty(network),
          };

          // Assert
          expect(topologyStateWasCleared).toEqual({
            topologyOrder: null,
            activationSchedule: {
              mode: 'recurrent',
              steps: [
                {
                  kind: 'wave',
                  nodeIds: inputNodes
                    .map((nodeEntry) => nodeEntry.geneId)
                    .toSorted((left, right) => left - right),
                },
                {
                  kind: 'recurrent-component',
                  nodeIds: [hiddenNode.geneId],
                  iterations: 1,
                },
                {
                  kind: 'wave',
                  nodeIds: [outputNode.geneId],
                },
              ],
              outputNodeIds: [outputNode.geneId],
              stateSemantics: 'carry',
            },
            topoDirty: false,
          });
        });
      });
    });
  });

  describe('hasPath()', () => {
    describe('given the from and to node are the same node', () => {
      describe('when hasPath() is called', () => {
        it('returns true', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 76, enforceAcyclic: true });
          const sameNode = network.nodes[0];

          // Act
          const selfReachable = hasPath.call(network, sameNode, sameNode);

          // Assert
          expect(selfReachable).toBe(true);
        });
      });
    });

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
