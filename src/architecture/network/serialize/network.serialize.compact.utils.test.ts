import Connection from '../../connection';
import Node from '../../node';
import { methods } from '../../../neataptic';
import {
  collectNodeGeneIds,
  collectSerializedConnections,
  rebuildConnectionsFromCompactPayload,
  rebuildNodesFromCompactPayload,
  refreshNodeIndices,
} from './network.serialize.compact.utils';
import type {
  NetworkInternals,
  SerializedConnection,
} from './network.serialize.utils.types';

type CompactRuntime = NetworkInternals & {
  connections: Connection[];
  gates: Connection[];
  nodes: Node[];
  selfconns: Connection[];
};

function createNodeWithOptionalGeneId(
  nodeType: Node['type'],
  geneId?: number,
): Node {
  const node = new Node(nodeType);

  if (typeof geneId === 'number') {
    node.geneId = geneId;
    return node;
  }

  Reflect.set(node, 'geneId', undefined);
  return node;
}

function createExportRuntime(
  nodes: Node[],
  connections: Connection[],
): NetworkInternals {
  return {
    connections,
    nodes,
    selfconns: [],
  } as unknown as NetworkInternals;
}

function createConnectionRebuildRuntime(nodes: Node[]): CompactRuntime {
  const runtime = {
    connections: [] as Connection[],
    gates: [] as Connection[],
    nodes,
    selfconns: [] as Connection[],
    connect(sourceNode: Node, targetNode: Node, weight: number) {
      const createdConnection = new Connection(sourceNode, targetNode, weight);
      this.connections.push(createdConnection);
      return [createdConnection];
    },
    gate(gaterNode: Node, createdConnection: Connection) {
      createdConnection.gater = gaterNode;
      this.gates.push(createdConnection);
    },
  };

  return runtime as unknown as CompactRuntime;
}

function createSerializedConnection(input: {
  enabled?: boolean;
  from: number;
  fromGeneId?: number;
  gater?: number | null;
  gaterGeneId?: number | null;
  innovation?: number;
  to: number;
  toGeneId?: number;
  weight?: number;
}): SerializedConnection {
  return {
    enabled: input.enabled ?? true,
    from: input.from,
    fromGeneId: input.fromGeneId,
    gater: input.gater ?? null,
    gaterGeneId: input.gaterGeneId ?? null,
    innovation: input.innovation ?? 501,
    to: input.to,
    toGeneId: input.toGeneId,
    weight: input.weight ?? 0.75,
  };
}

describe('network serialize compact utilities chapter', () => {
  describe('collectNodeGeneIds', () => {
    describe('given one gated compact export includes a hidden node without a gene id', () => {
      describe('when compact node and connection metadata are collected', () => {
        it('serializes the missing hidden-node gene id as null while keeping gated connection indices', () => {
          // Arrange
          const inputNode = createNodeWithOptionalGeneId('input', 11);
          const hiddenNode = createNodeWithOptionalGeneId('hidden');
          const outputNode = createNodeWithOptionalGeneId('output', 33);
          const gatedConnection = new Connection(inputNode, outputNode, 0.5);
          gatedConnection.gater = hiddenNode;
          const nodes = [inputNode, hiddenNode, outputNode];
          refreshNodeIndices(nodes);
          const runtime = createExportRuntime(nodes, [gatedConnection]);

          // Act
          const compactSignature = {
            nodeGeneIds: collectNodeGeneIds(nodes),
            serializedConnections: collectSerializedConnections(runtime),
          };

          // Assert
          expect(compactSignature).toEqual({
            nodeGeneIds: [11, null, 33],
            serializedConnections: [
              {
                enabled: true,
                from: 0,
                fromGeneId: 11,
                gater: 1,
                gaterGeneId: undefined,
                innovation: gatedConnection.innovation,
                to: 2,
                toGeneId: 33,
                weight: 0.5,
              },
            ],
          });
        });
      });
    });
  });

  describe('rebuildNodesFromCompactPayload', () => {
    describe('given the compact payload omits the hidden-node squash key', () => {
      describe('when runtime nodes are rebuilt from the compact arrays', () => {
        it('rebuilds the middle node as hidden and falls back to the identity squash function', () => {
          // Arrange
          const runtime = {
            nodes: [] as Node[],
          } as unknown as NetworkInternals;

          // Act
          rebuildNodesFromCompactPayload(runtime, {
            activations: [0.1, 0.2, 0.3],
            input: 1,
            nodeGeneIds: [11, null, 33],
            output: 1,
            squashes: ['identity', undefined, 'identity'] as unknown as string[],
            states: [0, 0, 0],
          });

          // Assert
          expect({
            nodeTypes: runtime.nodes.map((node) => node.type),
            squash: runtime.nodes[1].squash,
          }).toEqual({
            nodeTypes: ['input', 'hidden', 'output'],
            squash: methods.Activation.identity,
          });
        });
      });
    });
  });

  describe('rebuildConnectionsFromCompactPayload', () => {
    describe('given one serialized connection includes a valid compact gater gene id', () => {
      describe('when the connection is rebuilt', () => {
        it('restores the gate and all persisted endpoint gene ids', () => {
          // Arrange
          const inputNode = createNodeWithOptionalGeneId('input');
          const hiddenNode = createNodeWithOptionalGeneId('hidden');
          const outputNode = createNodeWithOptionalGeneId('output');
          const nodes = [inputNode, hiddenNode, outputNode];
          refreshNodeIndices(nodes);
          const runtime = createConnectionRebuildRuntime(nodes);

          // Act
          rebuildConnectionsFromCompactPayload({
            networkInternals: runtime,
            serializedConnections: [
              createSerializedConnection({
                from: 0,
                fromGeneId: 101,
                gater: 1,
                gaterGeneId: 202,
                innovation: 701,
                to: 2,
                toGeneId: 303,
              }),
            ],
          });

          // Assert
          expect({
            gateCount: runtime.gates.length,
            gaterIndex: runtime.connections[0].gater?.index ?? null,
            nodeGeneIds: nodes.map((node) => node.geneId ?? null),
          }).toEqual({
            gateCount: 1,
            gaterIndex: 1,
            nodeGeneIds: [101, 202, 303],
          });
        });
      });
    });

    describe('given one serialized connection includes an out-of-bounds gater index and gene id', () => {
      describe('when the connection is rebuilt', () => {
        it('skips both gate attachment and gater gene-id restoration', () => {
          // Arrange
          const inputNode = createNodeWithOptionalGeneId('input');
          const hiddenNode = createNodeWithOptionalGeneId('hidden');
          const outputNode = createNodeWithOptionalGeneId('output');
          const nodes = [inputNode, hiddenNode, outputNode];
          refreshNodeIndices(nodes);
          const runtime = createConnectionRebuildRuntime(nodes);
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => undefined);

          try {
            // Act
            rebuildConnectionsFromCompactPayload({
              networkInternals: runtime,
              serializedConnections: [
                createSerializedConnection({
                  from: 0,
                  fromGeneId: 101,
                  gater: 999,
                  gaterGeneId: 202,
                  innovation: 702,
                  to: 2,
                  toGeneId: 303,
                }),
              ],
            });

            // Assert
            expect({
              gateCount: runtime.gates.length,
              hiddenGeneId: hiddenNode.geneId ?? null,
            }).toEqual({
              gateCount: 0,
              hiddenGeneId: null,
            });
          } finally {
            warnSpy.mockRestore();
          }
        });
      });
    });
  });
});