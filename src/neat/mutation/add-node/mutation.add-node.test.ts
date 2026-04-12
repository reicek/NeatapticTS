import type {
  ConnectionWithMetadata,
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';
import { createInnovationTracker } from '../../innovation-tracker/innovation-tracker';
import {
  applySplitWithExistingRecord,
  applySplitWithNewRecord,
  buildSplitDescriptor,
  chooseConnectionForSplit,
  disconnectOriginalConnection,
  ensureBootstrapConnection,
} from './mutation.add-node';

describe('neat mutation add-node chapter', () => {
  describe('ensureBootstrapConnection', () => {
    describe('given the genome has no connections and exposes one input-output pair', () => {
      it('seeds exactly one bootstrap connection', () => {
        // Arrange
        const genome = createGenome([
          createNode('input', 1),
          createNode('output', 2),
        ]);

        // Act
        ensureBootstrapConnection(genome, createMutationController());

        // Assert
        expect({
          connectionCount: genome.connections.length,
          endpointTypes: genome.connections.map((connection) => [
            connection.from.type,
            connection.to.type,
          ]),
        }).toEqual({
          connectionCount: 1,
          endpointTypes: [['input', 'output']],
        });
      });
    });

    describe('given the genome already has one connection', () => {
      it('keeps the existing connection shelf unchanged', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const outputNode = createNode('output', 2);
        const genome = createGenome([inputNode, outputNode]);
        genome.connect?.(inputNode, outputNode, 0.75);

        // Act
        ensureBootstrapConnection(genome, createMutationController());

        // Assert
        expect(genome.connections.length).toBe(1);
      });
    });

    describe('given the genome lacks one side of the public interface', () => {
      it('returns without creating a bootstrap edge', () => {
        // Arrange
        const genome = createGenome([createNode('input', 1)]);

        // Act
        ensureBootstrapConnection(genome, createMutationController());

        // Assert
        expect(genome.connections.length).toBe(0);
      });
    });
  });

  describe('buildSplitDescriptor', () => {
    describe('given the split connection already has a historical innovation', () => {
      it('keys the split descriptor by that structural event instead of only endpoints', () => {
        // Arrange
        const { connectionToSplit } = createSplitScenario(17);

        // Act
        const splitDescriptor = buildSplitDescriptor(connectionToSplit);

        // Assert
        expect(splitDescriptor).toEqual({
          splitKey: 'splitConnectionInnovation:17',
          originalWeight: 0.75,
        });
      });
    });

    describe('given two connections share endpoints but not historical identity', () => {
      it('keeps their split descriptors distinct', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const outputNode = createNode('output', 2);
        const firstConnection = createConnection(inputNode, outputNode, 0.75, 17);
        const secondConnection = createConnection(inputNode, outputNode, 0.75, 18);

        // Act
        const splitKeys = [
          buildSplitDescriptor(firstConnection).splitKey,
          buildSplitDescriptor(secondConnection).splitKey,
        ];

        // Assert
        expect(splitKeys).toEqual([
          'splitConnectionInnovation:17',
          'splitConnectionInnovation:18',
        ]);
      });
    });

    describe('given the split connection lacks historical innovation metadata', () => {
      it('falls back to the legacy endpoint key', () => {
        // Arrange
        const { connectionToSplit } = createSplitScenario(null);

        // Act
        const splitDescriptor = buildSplitDescriptor(connectionToSplit);

        // Assert
        expect(splitDescriptor.splitKey).toBe('legacyEndpoints:1->2');
      });
    });
  });

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
        const recordedSplit =
          mutationController._innovationTracker.nodeSplitRecords.get(
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
          () => 0.5,
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

  describe('chooseConnectionForSplit', () => {
    describe('given no enabled connections are available', () => {
      it('returns null', () => {
        // Arrange
        const mutationController = createMutationController();

        // Act
        const chosenConnection = chooseConnectionForSplit([], mutationController);

        // Assert
        expect(chosenConnection).toBeNull();
      });
    });

    describe('given several enabled connections are available', () => {
      it('selects the connection at the sampled ordinal', () => {
        // Arrange
        const firstConnection = createConnection(
          createNode('input', 1),
          createNode('hidden', 2),
          0.25,
          11,
        );
        const secondConnection = createConnection(
          createNode('hidden', 3),
          createNode('output', 4),
          0.5,
          12,
        );
        const mutationController = createMutationController(0.75);

        // Act
        const chosenConnection = chooseConnectionForSplit(
          [firstConnection, secondConnection],
          mutationController,
        );

        // Assert
        expect(chosenConnection?.innovation).toBe(12);
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

function createSplitScenario(connectionInnovation: number | null = 5): {
  genome: GenomeWithMetadata;
  connectionToSplit: ConnectionWithMetadata;
} {
  const inputNode = createNode('input', 1);
  const outputNode = createNode('output', 2);
  const genome = createGenome([inputNode, outputNode]);
  const [connectionToSplit] = genome.connect!(inputNode, outputNode, 0.75);
  if (typeof connectionInnovation === 'number') {
    connectionToSplit.innovation = connectionInnovation;
  }

  return {
    genome,
    connectionToSplit,
  };
}

function createConnection(
  from: NodeWithMetadata,
  to: NodeWithMetadata,
  weight: number,
  innovation?: number,
): ConnectionWithMetadata {
  return {
    from,
    to,
    weight,
    innovation,
  };
}

function createMutationController(randomSample = 0): NeatControllerForMutation {
  return {
    population: [],
    options: {},
    _getRNG: () => () => randomSample,
    selectMutationMethod: async () => null,
    _mutateAddNodeReuse: async () => undefined,
    _mutateAddConnReuse: () => undefined,
    _invalidateGenomeCaches: () => undefined,
    _operatorStats: new Map(),
    _innovationTracker: {
      ...createInnovationTracker(),
      nextInnovationId: 11,
    },
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
