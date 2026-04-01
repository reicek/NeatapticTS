import type {
  ConnectionWithMetadata,
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';
import {
  applySplitWithExistingRecord,
  applySplitWithNewRecord,
  buildSplitDescriptor,
  disconnectOriginalConnection,
} from './mutation.add-node';

describe('neat mutation add-node chapter', () => {
  describe('applySplitWithNewRecord', () => {
    describe('given a novel connection split', () => {
      it('stores a reusable innovation record for the split descriptor', () => {
        // Arrange
        const { genome, connectionToSplit } = createSplitScenario();
        const mutationController = createMutationController();
        const splitDescriptor = buildSplitDescriptor(connectionToSplit);
        disconnectOriginalConnection(genome, connectionToSplit);

        // Act
        applySplitWithNewRecord(
          genome,
          connectionToSplit,
          splitDescriptor,
          DeterministicNode,
          mutationController,
        );
        const recordedSplit = mutationController._nodeSplitInnovations.get(
          splitDescriptor.splitKey,
        );

        // Assert
        expect(recordedSplit).toEqual({
          newNodeGeneId: 900,
          inInnov: 11,
          outInnov: 12,
        });
      });
    });
  });

  describe('applySplitWithExistingRecord', () => {
    describe('given an already known split record', () => {
      it('reuses the stored node gene id and replacement innovations', () => {
        // Arrange
        const { genome, connectionToSplit } = createSplitScenario();
        const splitDescriptor = buildSplitDescriptor(connectionToSplit);
        const splitRecord = {
          newNodeGeneId: 41,
          inInnov: 7,
          outInnov: 8,
        };
        disconnectOriginalConnection(genome, connectionToSplit);

        // Act
        applySplitWithExistingRecord(
          genome,
          connectionToSplit,
          splitDescriptor,
          splitRecord,
          DeterministicNode,
        );
        const insertedNode = genome.nodes.find(
          (node) => node.type === 'hidden',
        );
        const incomingConnection = genome.connections.find(
          (connection) => connection.to === insertedNode,
        );
        const outgoingConnection = genome.connections.find(
          (connection) => connection.from === insertedNode,
        );

        // Assert
        expect({
          nodeGeneId: insertedNode?.geneId,
          incomingInnovation: incomingConnection?.innovation,
          outgoingInnovation: outgoingConnection?.innovation,
        }).toEqual({
          nodeGeneId: 41,
          incomingInnovation: 7,
          outgoingInnovation: 8,
        });
      });
    });
  });
});

class DeterministicNode implements NodeWithMetadata {
  type: NodeWithMetadata['type'];
  geneId?: number;
  connections: NodeWithMetadata['connections'];

  constructor(type: NodeWithMetadata['type']) {
    this.type = type;
    this.geneId = 900;
    this.connections = {
      in: [],
      out: [],
    };
  }

  isProjectingTo(targetNode: NodeWithMetadata): boolean {
    return this.connections.out.some(
      (connection) => connection.to === targetNode,
    );
  }
}

function createSplitScenario(): {
  genome: GenomeWithMetadata;
  connectionToSplit: ConnectionWithMetadata;
} {
  const inputNode = createNode('input', 1);
  const outputNode = createNode('output', 2);
  const genome = createGenome([inputNode, outputNode]);
  const [connectionToSplit] = genome.connect!(inputNode, outputNode, 0.75);

  return {
    genome,
    connectionToSplit,
  };
}

function createMutationController(): NeatControllerForMutation {
  return {
    population: [],
    options: {},
    _getRNG: () => () => 0,
    selectMutationMethod: async () => null,
    _mutateAddNodeReuse: async () => undefined,
    _mutateAddConnReuse: () => undefined,
    _invalidateGenomeCaches: () => undefined,
    _operatorStats: new Map(),
    _nodeSplitInnovations: new Map(),
    _connInnovations: new Map(),
    _nextGlobalInnovation: 11,
  };
}

function createGenome(nodes: NodeWithMetadata[]): GenomeWithMetadata {
  const genome: GenomeWithMetadata = {
    nodes,
    connections: [],
    gates: [],
    input: 1,
    output: 1,
    connect: (from, to, weight = 1) => {
      const connection: ConnectionWithMetadata = {
        from,
        to,
        weight,
      };
      from.connections.out.push(connection);
      to.connections.in.push(connection);
      genome.connections.push(connection);
      return [connection];
    },
    disconnect: (from, to) => {
      const matchingConnections = genome.connections.filter(
        (connection) => connection.from === from && connection.to === to,
      );

      genome.connections = genome.connections.filter(
        (connection) => !(connection.from === from && connection.to === to),
      );
      from.connections.out = from.connections.out.filter(
        (connection) => !matchingConnections.includes(connection),
      );
      to.connections.in = to.connections.in.filter(
        (connection) => !matchingConnections.includes(connection),
      );
    },
  };

  return genome;
}

function createNode(
  type: NodeWithMetadata['type'],
  geneId: number,
): NodeWithMetadata {
  const connections: NodeWithMetadata['connections'] = {
    in: [],
    out: [],
  };

  return {
    type,
    geneId,
    connections,
    isProjectingTo: (targetNode) =>
      connections.out.some((connection) => connection.to === targetNode),
  };
}
