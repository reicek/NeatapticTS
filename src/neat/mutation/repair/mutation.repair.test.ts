import { createInnovationTracker } from '../../innovation-tracker/innovation-tracker';
import {
  connectIfCandidatesExistForDeadEnds,
  ensureHiddenConnectivityForDeadEnds,
} from './mutation.dead-ends';
import type {
  ConnectionWithMetadata,
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';

describe('neat mutation repair chapter', () => {
  describe('connectIfCandidatesExistForDeadEnds', () => {
    describe('given feed-forward repair only has a backward hidden candidate available', () => {
      it('skips the illegal recurrent repair edge', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const earlierHiddenNode = createNode('hidden', 2);
        const laterHiddenNode = createNode('hidden', 3);
        const genome = createGenome([
          inputNode,
          earlierHiddenNode,
          laterHiddenNode,
        ]);
        genome.getTopologyIntent = () => 'feed-forward';
        genome._enforceAcyclic = true;
        const mutationController = createMutationController({
          allowRecurrent: true,
        });

        // Act
        connectIfCandidatesExistForDeadEnds(
          genome,
          laterHiddenNode,
          [earlierHiddenNode],
          false,
          mutationController,
        );

        // Assert
        expect(genome.connections.length).toBe(0);
      });
    });
  });

  describe('ensureHiddenConnectivityForDeadEnds', () => {
    describe('given one hidden node belongs to a validated recurrent module descriptor', () => {
      it('skips outbound repair for that module-owned hidden node', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const protectedHiddenNode = createNode('hidden', 2);
        const outputNode = createNode('output', 3);
        const genome = createGenome([inputNode, protectedHiddenNode, outputNode]);
        const mutationController = createMutationController({
          allowRecurrent: true,
        });
        const genomeWithTemporalDescriptor = genome as GenomeWithMetadata & {
          _serializedExtensions?: {
            version: number;
            values: {
              recurrentModules: Array<{
                moduleId: string;
                kind: 'gru';
                nodeGeneIdsByRole: Record<string, number[]>;
                connectionInnovations: number[];
              }>;
            };
          };
        };
        const moduleConnection = genome.connect?.(inputNode, protectedHiddenNode)[0];

        if (!moduleConnection) {
          throw new Error('Expected recurrent module seed connection');
        }

        moduleConnection.innovation = 999;
        genomeWithTemporalDescriptor._serializedExtensions = {
          version: 1,
          values: {
            recurrentModules: [
              {
                moduleId: 'module:gru:2',
                kind: 'gru',
                nodeGeneIdsByRole: {
                  output: [protectedHiddenNode.geneId ?? 0],
                },
                connectionInnovations: [999],
              },
            ],
          },
        };

        // Act
        ensureHiddenConnectivityForDeadEnds(
          genomeWithTemporalDescriptor,
          {
            inputNodes: [inputNode],
            outputNodes: [outputNode],
            hiddenNodes: [protectedHiddenNode],
          },
          mutationController,
        );

        // Assert
        expect(genome.connections).toHaveLength(1);
      });
    });
  });
});

function createMutationController(input: {
  allowRecurrent?: boolean;
}): NeatControllerForMutation {
  return {
    population: [],
    options: {
      allowRecurrent: input.allowRecurrent,
    },
    _getRNG: () => () => 0,
    selectMutationMethod: async () => null,
    _mutateAddNodeReuse: async () => undefined,
    _mutateAddConnReuse: () => undefined,
    _invalidateGenomeCaches: () => undefined,
    _operatorStats: new Map(),
    _innovationTracker: createInnovationTracker(),
  };
}

function createGenome(nodes: NodeWithMetadata[]): GenomeWithMetadata {
  const genome: GenomeWithMetadata = {
    nodes,
    connections: [],
    gates: [],
    input: 1,
    output: 0,
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
  const connections = {
    in: [],
    out: [],
    self: [],
  } as NodeWithMetadata['connections'] & {
    self: ConnectionWithMetadata[];
  };

  return {
    type,
    geneId,
    connections,
    isProjectingTo: (targetNode) =>
      connections.out.some((connection) => connection.to === targetNode),
  };
}