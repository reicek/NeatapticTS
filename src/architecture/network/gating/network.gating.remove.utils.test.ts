import mutation from '../../../methods/mutation/mutation';
import type Connection from '../../connection';
import type Node from '../../node';
import type Network from '../../network/network';
import {
  NetworkGatingRemovalNodeNotFoundError,
  NetworkGatingStructuralAnchorRemovalError,
} from './network.gating.errors';
import {
  assertNodeRemovableAndGetIndex,
  createBridgingConnections,
  disconnectInboundConnections,
  disconnectNodeSelfLoop,
  disconnectOutboundConnections,
  reassignPreservedGaters,
  removeNodeAtIndex,
  resolveSubNodeMutationConfig,
  ungateConnectionsGatedByNode,
} from './network.gating.remove.utils';

const originalSubNodeMutation = mutation.SUB_NODE;

describe('network gating removal utility chapter', () => {
  afterEach(() => {
    mutation.SUB_NODE = originalSubNodeMutation;
    jest.restoreAllMocks();
  });

  describe('assertNodeRemovableAndGetIndex', () => {
    describe('given a hidden node belongs to the network', () => {
      it('returns the node index in the network list', () => {
        // Arrange
        const hiddenNode = createNode({ type: 'hidden' });
        const network = createNetwork({
          nodes: [createNode({ type: 'input' }), hiddenNode],
        });

        // Act
        const nodeIndex = assertNodeRemovableAndGetIndex(network, hiddenNode);

        // Assert
        expect(nodeIndex).toBe(1);
      });
    });

    describe('given the node is an input anchor', () => {
      it('throws the structural-anchor removal error', () => {
        // Arrange
        const inputNode = createNode({ type: 'input' });
        const network = createNetwork({ nodes: [inputNode] });

        // Act
        const removeInputNode = () =>
          assertNodeRemovableAndGetIndex(network, inputNode);

        // Assert
        expect(removeInputNode).toThrow(
          NetworkGatingStructuralAnchorRemovalError,
        );
      });
    });

    describe('given the node is an output anchor', () => {
      it('throws the structural-anchor removal error', () => {
        // Arrange
        const outputNode = createNode({ type: 'output' });
        const network = createNetwork({ nodes: [outputNode] });

        // Act
        const removeOutputNode = () =>
          assertNodeRemovableAndGetIndex(network, outputNode);

        // Assert
        expect(removeOutputNode).toThrow(
          NetworkGatingStructuralAnchorRemovalError,
        );
      });
    });

    describe('given the node is absent from the network', () => {
      it('throws the node-not-found removal error', () => {
        // Arrange
        const hiddenNode = createNode({ type: 'hidden' });
        const network = createNetwork({ nodes: [] });

        // Act
        const removeMissingNode = () =>
          assertNodeRemovableAndGetIndex(network, hiddenNode);

        // Assert
        expect(removeMissingNode).toThrow(
          NetworkGatingRemovalNodeNotFoundError,
        );
      });
    });
  });

  describe('resolveSubNodeMutationConfig', () => {
    describe('given SUB_NODE is configured as one mutation config object', () => {
      it('returns that config object unchanged', () => {
        // Arrange
        const expectedConfig = { keep_gates: true, name: 'sub-node-object' };
        mutation.SUB_NODE = expectedConfig as typeof mutation.SUB_NODE;

        // Act
        const mutationConfig = resolveSubNodeMutationConfig();

        // Assert
        expect(mutationConfig).toBe(expectedConfig);
      });
    });

    describe('given SUB_NODE is configured as an array of mutation configs', () => {
      it('returns the first config entry', () => {
        // Arrange
        const expectedConfig = { keep_gates: false, name: 'sub-node-array-0' };
        mutation.SUB_NODE = [
          expectedConfig,
          { keep_gates: true, name: 'sub-node-array-1' },
        ] as unknown as typeof mutation.SUB_NODE;

        // Act
        const mutationConfig = resolveSubNodeMutationConfig();

        // Assert
        expect(mutationConfig).toBe(expectedConfig);
      });
    });
  });

  describe('disconnectNodeSelfLoop', () => {
    describe('given a node is being removed', () => {
      it('disconnects the node from itself before rewiring', () => {
        // Arrange
        const node = createNode({ type: 'hidden' });
        const disconnect = jest.fn();
        const network = createNetwork({ disconnect });

        // Act
        disconnectNodeSelfLoop(network, node);

        // Assert
        expect(disconnect).toHaveBeenCalledWith(node, node);
      });
    });
  });

  describe('disconnectInboundConnections', () => {
    describe('given gate preservation is disabled and the inbound list lacks toReversed', () => {
      it('falls back to reversed iteration, collects predecessors, and does not preserve gaters', () => {
        // Arrange
        const removedNode = createNode({ type: 'hidden' });
        const predecessorNodeA = createNode({ type: 'hidden' });
        const predecessorNodeB = createNode({ type: 'hidden' });
        const inboundConnectionA = createConnection({
          from: predecessorNodeA,
          to: removedNode,
        });
        const inboundConnectionB = createConnection({
          from: predecessorNodeB,
          to: removedNode,
        });
        removedNode.connections.in = createConnectionList(
          [inboundConnectionA, inboundConnectionB],
          true,
        );
        const disconnect = jest.fn();
        const network = createNetwork({ disconnect });
        const preservedGaters: Node[] = [];

        // Act
        const predecessorNodes = disconnectInboundConnections(
          network,
          removedNode,
          preservedGaters,
          undefined,
        );

        // Assert
        expect({
          predecessorNodes,
          preservedGaters,
          disconnectCalls: disconnect.mock.calls,
        }).toEqual({
          predecessorNodes: [predecessorNodeB, predecessorNodeA],
          preservedGaters: [],
          disconnectCalls: [
            [predecessorNodeB, removedNode],
            [predecessorNodeA, removedNode],
          ],
        });
      });
    });

    describe('given gate preservation is enabled and the gater is not the removed node', () => {
      it('preserves the gater for later reassignment', () => {
        // Arrange
        const removedNode = createNode({ type: 'hidden' });
        const predecessorNode = createNode({ type: 'hidden' });
        const externalGater = createNode({ type: 'hidden' });
        const inboundConnection = createConnection({
          from: predecessorNode,
          to: removedNode,
          gater: externalGater,
        });
        removedNode.connections.in = createConnectionList([inboundConnection]);
        const network = createNetwork({ disconnect: jest.fn() });
        const preservedGaters: Node[] = [];

        // Act
        const predecessorNodes = disconnectInboundConnections(
          network,
          removedNode,
          preservedGaters,
          { keep_gates: true },
        );

        // Assert
        expect({ predecessorNodes, preservedGaters }).toEqual({
          predecessorNodes: [predecessorNode],
          preservedGaters: [externalGater],
        });
      });
    });
  });

  describe('disconnectOutboundConnections', () => {
    describe('given the removed node is also the gater and the outbound list uses toReversed', () => {
      it('disconnects successors without preserving the removed node as a gater', () => {
        // Arrange
        const removedNode = createNode({ type: 'hidden' });
        const successorNode = createNode({ type: 'hidden' });
        const outboundConnection = createConnection({
          from: removedNode,
          to: successorNode,
          gater: removedNode,
        });
        removedNode.connections.out = createConnectionList([
          outboundConnection,
        ]);
        const disconnect = jest.fn();
        const network = createNetwork({ disconnect });
        const preservedGaters: Node[] = [];

        // Act
        const successorNodes = disconnectOutboundConnections(
          network,
          removedNode,
          preservedGaters,
          { keep_gates: true },
        );

        // Assert
        expect({
          successorNodes,
          preservedGaters,
          disconnectCalls: disconnect.mock.calls,
        }).toEqual({
          successorNodes: [successorNode],
          preservedGaters: [],
          disconnectCalls: [[removedNode, successorNode]],
        });
      });
    });

    describe('given the outbound list lacks toReversed', () => {
      it('falls back to reversed iteration while disconnecting successors', () => {
        // Arrange
        const removedNode = createNode({ type: 'hidden' });
        const successorNodeA = createNode({ type: 'hidden' });
        const successorNodeB = createNode({ type: 'hidden' });
        const outboundConnectionA = createConnection({
          from: removedNode,
          to: successorNodeA,
        });
        const outboundConnectionB = createConnection({
          from: removedNode,
          to: successorNodeB,
        });
        removedNode.connections.out = createConnectionList(
          [outboundConnectionA, outboundConnectionB],
          true,
        );
        const disconnect = jest.fn();
        const network = createNetwork({ disconnect });

        // Act
        const successorNodes = disconnectOutboundConnections(
          network,
          removedNode,
          [],
          { keep_gates: true },
        );

        // Assert
        expect({
          successorNodes,
          disconnectCalls: disconnect.mock.calls,
        }).toEqual({
          successorNodes: [successorNodeB, successorNodeA],
          disconnectCalls: [
            [removedNode, successorNodeB],
            [removedNode, successorNodeA],
          ],
        });
      });
    });
  });

  describe('createBridgingConnections', () => {
    describe('given the predecessor and successor are the same node', () => {
      it('skips the self-bridge pair', () => {
        // Arrange
        const sharedNode = createNode({ type: 'hidden' });
        const connect = jest.fn();
        const network = createNetwork({ connect });

        // Act
        const bridgingConnections = createBridgingConnections(
          network,
          [sharedNode],
          [sharedNode],
        );

        // Assert
        expect({
          bridgingConnections,
          connectCalls: connect.mock.calls,
        }).toEqual({
          bridgingConnections: [],
          connectCalls: [],
        });
      });
    });

    describe('given one valid pair creates no connection and a later pair does create one', () => {
      it('keeps only the successfully created bridge connections', () => {
        // Arrange
        const predecessorNodeA = createNode({ type: 'hidden' });
        const predecessorNodeB = createNode({ type: 'hidden' });
        const successorNode = createNode({ type: 'hidden' });
        const createdConnection = createConnection({
          from: predecessorNodeB,
          to: successorNode,
        });
        const connect = jest
          .fn()
          .mockReturnValueOnce([])
          .mockReturnValueOnce([createdConnection]);
        const network = createNetwork({ connect });

        // Act
        const bridgingConnections = createBridgingConnections(
          network,
          [predecessorNodeA, predecessorNodeB],
          [successorNode],
        );

        // Assert
        expect({
          bridgingConnections,
          connectCalls: connect.mock.calls,
        }).toEqual({
          bridgingConnections: [createdConnection],
          connectCalls: [
            [predecessorNodeA, successorNode],
            [predecessorNodeB, successorNode],
          ],
        });
      });
    });
  });

  describe('reassignPreservedGaters', () => {
    describe('given no bridge connections remain for reassignment', () => {
      it('returns without attempting any gate operation', () => {
        // Arrange
        const network = createNetwork({ gate: jest.fn() });
        const preservedGaters = [createNode({ type: 'hidden' })];

        // Act
        reassignPreservedGaters(network, preservedGaters, []);

        // Assert
        expect((network.gate as jest.Mock).mock.calls).toEqual([]);
      });
    });

    describe('given preserved gaters and bridge connections are available', () => {
      it('gates one randomly selected bridge connection and removes it from the pool', () => {
        // Arrange
        jest.spyOn(Math, 'random').mockReturnValue(0.75);
        const gaterNode = createNode({ type: 'hidden' });
        const bridgeConnectionA = createConnection({
          from: createNode({ type: 'hidden' }),
          to: createNode({ type: 'hidden' }),
        });
        const bridgeConnectionB = createConnection({
          from: createNode({ type: 'hidden' }),
          to: createNode({ type: 'hidden' }),
        });
        const bridgingConnections = [bridgeConnectionA, bridgeConnectionB];
        const gate = jest.fn();
        const network = createNetwork({ gate });

        // Act
        reassignPreservedGaters(network, [gaterNode], bridgingConnections);

        // Assert
        expect({ gateCalls: gate.mock.calls, bridgingConnections }).toEqual({
          gateCalls: [[gaterNode, bridgeConnectionB]],
          bridgingConnections: [bridgeConnectionA],
        });
      });
    });
  });

  describe('ungateConnectionsGatedByNode', () => {
    describe('given the gated list uses toReversed', () => {
      it('ungates the stored connections in reverse order', () => {
        // Arrange
        const node = createNode({ type: 'hidden' });
        const gatedConnectionA = createConnection();
        const gatedConnectionB = createConnection();
        node.connections.gated = createConnectionList([
          gatedConnectionA,
          gatedConnectionB,
        ]);
        const ungate = jest.fn();
        const network = createNetwork({ ungate });

        // Act
        ungateConnectionsGatedByNode(network, node);

        // Assert
        expect(ungate.mock.calls).toEqual([
          [gatedConnectionB],
          [gatedConnectionA],
        ]);
      });
    });

    describe('given the gated list lacks toReversed', () => {
      it('falls back to reversed iteration while ungating the stored connections', () => {
        // Arrange
        const node = createNode({ type: 'hidden' });
        const gatedConnectionA = createConnection();
        const gatedConnectionB = createConnection();
        node.connections.gated = createConnectionList(
          [gatedConnectionA, gatedConnectionB],
          true,
        );
        const ungate = jest.fn();
        const network = createNetwork({ ungate });

        // Act
        ungateConnectionsGatedByNode(network, node);

        // Assert
        expect(ungate.mock.calls).toEqual([
          [gatedConnectionB],
          [gatedConnectionA],
        ]);
      });
    });
  });

  describe('removeNodeAtIndex', () => {
    describe('given one node is removed from the network list', () => {
      it('splices the node list and marks node indexing as dirty', () => {
        // Arrange
        const retainedNode = createNode({ type: 'hidden' });
        const removedNode = createNode({ type: 'hidden' });
        const network = createNetwork({
          nodes: [retainedNode, removedNode],
        }) as Network & { _nodeIndexDirty?: boolean };

        // Act
        removeNodeAtIndex(network, 1);

        // Assert
        expect({
          nodes: network.nodes,
          isDirty: network._nodeIndexDirty,
        }).toEqual({
          nodes: [retainedNode],
          isDirty: true,
        });
      });
    });
  });
});

function createNetwork(
  overrides: Partial<
    Pick<Network, 'connect' | 'disconnect' | 'gate' | 'nodes' | 'ungate'>
  > = {},
): Network {
  return {
    nodes: [],
    disconnect: jest.fn(),
    connect: jest.fn(),
    gate: jest.fn(),
    ungate: jest.fn(),
    ...overrides,
  } as unknown as Network;
}

function createNode(overrides: {
  gatedConnections?: Connection[];
  inboundConnections?: Connection[];
  isProjectingTo?: (targetNode: Node) => boolean;
  outboundConnections?: Connection[];
  type: string;
}): Node {
  return {
    type: overrides.type,
    connections: {
      in: overrides.inboundConnections ?? [],
      out: overrides.outboundConnections ?? [],
      gated: overrides.gatedConnections ?? [],
    },
    isProjectingTo: overrides.isProjectingTo ?? (() => false),
  } as unknown as Node;
}

function createConnection(
  overrides: Partial<Pick<Connection, 'from' | 'gater' | 'to'>> = {},
): Connection {
  return {
    from: overrides.from ?? createNode({ type: 'hidden' }),
    to: overrides.to ?? createNode({ type: 'hidden' }),
    gater: overrides.gater,
  } as unknown as Connection;
}

function createConnectionList(
  connections: Connection[],
  disableToReversed = false,
): Connection[] {
  const connectionList = [...connections] as Connection[];

  if (disableToReversed) {
    Object.defineProperty(connectionList, 'toReversed', {
      configurable: true,
      value: undefined,
      writable: true,
    });
  }

  return connectionList;
}
