import { createInnovationTracker } from '../../innovation-tracker/innovation-tracker';
import * as mutationAddConn from '../add-conn/mutation.add-conn';
import {
  chooseRandomNodeForDeadEnds,
  collectNodeGroupsForDeadEnds,
  connectIfCandidatesExistForDeadEnds,
  ensureHiddenConnectivityForDeadEnds,
  ensureInputConnectivityForDeadEnds,
  ensureOutputConnectivityForDeadEnds,
} from './mutation.dead-ends';
import type {
  ConnectionWithMetadata,
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';

describe('neat mutation repair chapter', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('collectNodeGroupsForDeadEnds', () => {
    describe('given the genome contains input, hidden, and output nodes', () => {
      it('returns the nodes grouped by repair role', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const hiddenNode = createNode('hidden', 2);
        const outputNode = createNode('output', 3);
        const genome = createGenome([inputNode, hiddenNode, outputNode]);

        // Act
        const nodeGroups = collectNodeGroupsForDeadEnds(genome);

        // Assert
        expect(nodeGroups).toEqual({
          inputNodes: [inputNode],
          outputNodes: [outputNode],
          hiddenNodes: [hiddenNode],
        });
      });
    });
  });

  describe('chooseRandomNodeForDeadEnds', () => {
    describe('given the candidate list is empty', () => {
      it('returns null instead of sampling the controller RNG', () => {
        // Arrange
        const mutationController = createMutationController({});

        // Act
        const chosenNode = chooseRandomNodeForDeadEnds([], mutationController);

        // Assert
        expect(chosenNode).toBeNull();
      });
    });
  });

  describe('connectIfCandidatesExistForDeadEnds', () => {
    describe('given the repair pool is empty', () => {
      it('leaves the genome unchanged', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const outputNode = createNode('output', 2);
        const genome = createGenome([inputNode, outputNode]);
        const mutationController = createMutationController({});

        // Act
        connectIfCandidatesExistForDeadEnds(
          genome,
          inputNode,
          [],
          false,
          mutationController,
        );

        // Assert
        expect(genome.connections).toHaveLength(0);
      });
    });

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

    describe('given the RNG lands beyond the legal candidate range', () => {
      it('returns without creating a repair connection', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const outputNode = createNode('output', 2);
        const genome = createGenome([inputNode, outputNode]);
        const mutationController = createMutationController({
          randomValue: 1,
        });
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockReturnValue(true);

        // Act
        connectIfCandidatesExistForDeadEnds(
          genome,
          inputNode,
          [outputNode],
          false,
          mutationController,
        );

        // Assert
        expect(genome.connections).toHaveLength(0);
      });
    });
  });

  describe('ensureInputConnectivityForDeadEnds', () => {
    describe('given an input node already has an outgoing connection', () => {
      it('keeps the existing projection unchanged', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const hiddenNode = createNode('hidden', 2);
        const genome = createGenome([inputNode, hiddenNode]);
        genome.connect?.(inputNode, hiddenNode);
        const mutationController = createMutationController({});

        // Act
        ensureInputConnectivityForDeadEnds(
          genome,
          {
            inputNodes: [inputNode],
            outputNodes: [],
            hiddenNodes: [hiddenNode],
          },
          mutationController,
        );

        // Assert
        expect(readConnectionGeneIdPairs(genome)).toEqual([[1, 2]]);
      });
    });

    describe('given the network has no hidden nodes to target', () => {
      it('connects the stranded input directly to an output node', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const outputNode = createNode('output', 2);
        const genome = createGenome([inputNode, outputNode]);
        const mutationController = createMutationController({});
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockReturnValue(true);
        jest
          .spyOn(mutationAddConn, 'connectChosenPairWithInnovationReuse')
          .mockImplementation((networkToEdit, chosenPair) => {
            return networkToEdit.connect?.(chosenPair[0], chosenPair[1])[0];
          });

        // Act
        ensureInputConnectivityForDeadEnds(
          genome,
          {
            inputNodes: [inputNode],
            outputNodes: [outputNode],
            hiddenNodes: [],
          },
          mutationController,
        );

        // Assert
        expect(readConnectionGeneIdPairs(genome)).toEqual([[1, 2]]);
      });
    });

    describe('given a hidden repair candidate is legal', () => {
      it('repairs the stranded input without falling back to an output node', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const hiddenNode = createNode('hidden', 2);
        const outputNode = createNode('output', 3);
        const genome = createGenome([inputNode, hiddenNode, outputNode]);
        const mutationController = createMutationController({});
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockReturnValue(true);
        jest
          .spyOn(mutationAddConn, 'connectChosenPairWithInnovationReuse')
          .mockImplementation((networkToEdit, chosenPair) => {
            return networkToEdit.connect?.(chosenPair[0], chosenPair[1])[0];
          });

        // Act
        ensureInputConnectivityForDeadEnds(
          genome,
          {
            inputNodes: [inputNode],
            outputNodes: [outputNode],
            hiddenNodes: [hiddenNode],
          },
          mutationController,
        );

        // Assert
        expect(readConnectionGeneIdPairs(genome)).toEqual([[1, 2]]);
      });
    });

    describe('given hidden repair candidates exist but only output fallback is legal', () => {
      it('falls back to connecting the stranded input directly to an output node', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const hiddenNode = createNode('hidden', 2);
        const outputNode = createNode('output', 3);
        const genome = createGenome([inputNode, hiddenNode, outputNode]);
        const mutationController = createMutationController({});
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockImplementation((_genome, chosenPair) => chosenPair[1] === outputNode);
        jest
          .spyOn(mutationAddConn, 'connectChosenPairWithInnovationReuse')
          .mockImplementation((networkToEdit, chosenPair) => {
            return networkToEdit.connect?.(chosenPair[0], chosenPair[1])[0];
          });

        // Act
        ensureInputConnectivityForDeadEnds(
          genome,
          {
            inputNodes: [inputNode],
            outputNodes: [outputNode],
            hiddenNodes: [hiddenNode],
          },
          mutationController,
        );

        // Assert
        expect(readConnectionGeneIdPairs(genome)).toEqual([[1, 3]]);
      });
    });
  });

  describe('ensureOutputConnectivityForDeadEnds', () => {
    describe('given an output node already has an incoming connection', () => {
      it('keeps the existing inbound projection unchanged', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const outputNode = createNode('output', 2);
        const genome = createGenome([inputNode, outputNode]);
        genome.connect?.(inputNode, outputNode);
        const mutationController = createMutationController({});

        // Act
        ensureOutputConnectivityForDeadEnds(
          genome,
          {
            inputNodes: [inputNode],
            outputNodes: [outputNode],
            hiddenNodes: [],
          },
          mutationController,
        );

        // Assert
        expect(readConnectionGeneIdPairs(genome)).toEqual([[1, 2]]);
      });
    });

    describe('given the network has no hidden nodes upstream', () => {
      it('connects the stranded output directly from an input node', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const outputNode = createNode('output', 2);
        const genome = createGenome([inputNode, outputNode]);
        const mutationController = createMutationController({});
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockReturnValue(true);
        jest
          .spyOn(mutationAddConn, 'connectChosenPairWithInnovationReuse')
          .mockImplementation((networkToEdit, chosenPair) => {
            return networkToEdit.connect?.(chosenPair[0], chosenPair[1])[0];
          });

        // Act
        ensureOutputConnectivityForDeadEnds(
          genome,
          {
            inputNodes: [inputNode],
            outputNodes: [outputNode],
            hiddenNodes: [],
          },
          mutationController,
        );

        // Assert
        expect(readConnectionGeneIdPairs(genome)).toEqual([[1, 2]]);
      });
    });

    describe('given a hidden repair candidate is legal', () => {
      it('repairs the stranded output without falling back to an input node', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const hiddenNode = createNode('hidden', 2);
        const outputNode = createNode('output', 3);
        const genome = createGenome([inputNode, hiddenNode, outputNode]);
        const mutationController = createMutationController({});
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockReturnValue(true);
        jest
          .spyOn(mutationAddConn, 'connectChosenPairWithInnovationReuse')
          .mockImplementation((networkToEdit, chosenPair) => {
            return networkToEdit.connect?.(chosenPair[0], chosenPair[1])[0];
          });

        // Act
        ensureOutputConnectivityForDeadEnds(
          genome,
          {
            inputNodes: [inputNode],
            outputNodes: [outputNode],
            hiddenNodes: [hiddenNode],
          },
          mutationController,
        );

        // Assert
        expect(readConnectionGeneIdPairs(genome)).toEqual([[2, 3]]);
      });
    });

    describe('given hidden repair candidates exist but only input fallback is legal', () => {
      it('falls back to connecting a stranded output directly from an input node', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const hiddenNode = createNode('hidden', 2);
        const outputNode = createNode('output', 3);
        const genome = createGenome([inputNode, hiddenNode, outputNode]);
        const mutationController = createMutationController({});
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockImplementation((_genome, chosenPair) => chosenPair[0] === inputNode);
        jest
          .spyOn(mutationAddConn, 'connectChosenPairWithInnovationReuse')
          .mockImplementation((networkToEdit, chosenPair) => {
            return networkToEdit.connect?.(chosenPair[0], chosenPair[1])[0];
          });

        // Act
        ensureOutputConnectivityForDeadEnds(
          genome,
          {
            inputNodes: [inputNode],
            outputNodes: [outputNode],
            hiddenNodes: [hiddenNode],
          },
          mutationController,
        );

        // Assert
        expect(readConnectionGeneIdPairs(genome)).toEqual([[1, 3]]);
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

    describe('given an unprotected hidden node has neither incoming nor outgoing edges', () => {
      it('repairs both sides using the legal candidate pools', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const hiddenNodeToRepair = createNode('hidden', 2);
        const siblingHiddenNode = createNode('hidden', 3);
        const outputNode = createNode('output', 4);
        const genome = createGenome([
          inputNode,
          hiddenNodeToRepair,
          siblingHiddenNode,
          outputNode,
        ]);
        const mutationController = createMutationController({});
        jest
          .spyOn(mutationAddConn, 'canApplyChosenPairForConn')
          .mockReturnValue(true);
        jest
          .spyOn(mutationAddConn, 'connectChosenPairWithInnovationReuse')
          .mockImplementation((networkToEdit, chosenPair) => {
            return networkToEdit.connect?.(chosenPair[0], chosenPair[1])[0];
          });

        // Act
        ensureHiddenConnectivityForDeadEnds(
          genome,
          {
            inputNodes: [inputNode],
            outputNodes: [outputNode],
            hiddenNodes: [hiddenNodeToRepair, siblingHiddenNode],
          },
          mutationController,
        );

        // Assert
        expect(readConnectionGeneIdPairs(genome)).toEqual([
          [1, 2],
          [2, 4],
          [1, 3],
          [3, 4],
        ]);
      });
    });

    describe('given an unprotected hidden node already has both sides connected', () => {
      it('keeps the existing hidden path unchanged', () => {
        // Arrange
        const inputNode = createNode('input', 1);
        const hiddenNode = createNode('hidden', 2);
        const outputNode = createNode('output', 3);
        const genome = createGenome([inputNode, hiddenNode, outputNode]);
        genome.connect?.(inputNode, hiddenNode);
        genome.connect?.(hiddenNode, outputNode);
        const mutationController = createMutationController({});

        // Act
        ensureHiddenConnectivityForDeadEnds(
          genome,
          {
            inputNodes: [inputNode],
            outputNodes: [outputNode],
            hiddenNodes: [hiddenNode],
          },
          mutationController,
        );

        // Assert
        expect(readConnectionGeneIdPairs(genome)).toEqual([
          [1, 2],
          [2, 3],
        ]);
      });
    });
  });
});

function createMutationController(input: {
  allowRecurrent?: boolean;
  randomValue?: number;
}): NeatControllerForMutation {
  return {
    population: [],
    options: {
      allowRecurrent: input.allowRecurrent,
    },
    _getRNG: () => () => input.randomValue ?? 0,
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

function readConnectionGeneIdPairs(
  genome: GenomeWithMetadata,
): Array<[number | undefined, number | undefined]> {
  return genome.connections.map((connection) => [
    connection.from.geneId,
    connection.to.geneId,
  ]);
}