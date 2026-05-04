import type {
  ConnectionWithMetadata,
  NeatControllerForMutation,
  GenomeWithMetadata,
  NodeWithMetadata,
} from '../shared/mutation.types';
import { createInnovationTracker } from '../../innovation-tracker/innovation-tracker';
import {
  assignInnovationForConnection,
  buildDirectionalKeyForConn,
  canApplyChosenPairForConn,
  collectCandidatePairsForConn,
  connectChosenPairWithInnovationReuse,
  choosePairForConn,
  createsCycle,
  filterPairsWithInnovations,
  selectPairPool,
  shouldAbortForCycle,
} from './mutation.add-conn';

describe('neat mutation add-connection chapter', () => {
  describe('collectCandidatePairsForConn', () => {
    describe('given feed-forward connection growth with default second arg', () => {
      it('returns only forward candidates when called without the recurrent flag', () => {
        // Arrange – calls without second arg → covers default-param branch (line 80)
        const genome = createCandidatePairGenome();

        // Act
        const candidatePairs = summarizePairs(
          collectCandidatePairsForConn(genome),
        );

        // Assert
        expect(candidatePairs).toEqual(['1->2', '1->3', '2->3']);
      });
    });

    describe('given a fully-connected genome', () => {
      it('returns an empty candidate list when all forward edges already exist', () => {
        // Arrange – add all 3 forward edges so isAbsentDirectedEdge returns false each time (line 114 FALSE arm)
        const genome = createCandidatePairGenome();
        const [inputNode, hiddenNode, outputNode] = genome.nodes;
        connectNodes(genome, inputNode, hiddenNode);
        connectNodes(genome, inputNode, outputNode);
        connectNodes(genome, hiddenNode, outputNode);

        // Act
        const candidatePairs = collectCandidatePairsForConn(genome, false);

        // Assert
        expect(candidatePairs).toHaveLength(0);
      });
    });

    describe('given feed-forward connection growth', () => {
      it('returns only forward candidates', () => {
        // Arrange
        const genome = createCandidatePairGenome();

        // Act
        const candidatePairs = summarizePairs(
          collectCandidatePairsForConn(genome, false),
        );

        // Assert
        expect(candidatePairs).toEqual(['1->2', '1->3', '2->3']);
      });
    });

    describe('given recurrent connection growth is allowed', () => {
      it('includes forward, self, and backward candidates', () => {
        // Arrange
        const genome = createCandidatePairGenome();

        // Act
        const candidatePairs = summarizePairs(
          collectCandidatePairsForConn(genome, true),
        );

        // Assert
        expect(candidatePairs).toEqual([
          '1->2',
          '1->3',
          '2->2',
          '2->3',
          '3->2',
          '3->3',
        ]);
      });

      it('excludes a disabled directed edge because revival belongs to re-enable logic', () => {
        // Arrange
        const { genome } = createDormantEdgeGenome();

        // Act
        const candidatePairs = summarizePairs(
          collectCandidatePairsForConn(genome, true),
        );

        // Assert
        expect(candidatePairs).toEqual([
          '1->2',
          '1->3',
          '2->2',
          '2->3',
          '3->3',
        ]);
      });
    });
  });

  describe('buildDirectionalKeyForConn', () => {
    it('preserves the source-to-target orientation for a forward edge', () => {
      // Arrange
      const sourceNode = createNode('hidden', 2);
      const targetNode = createNode('hidden', 3);

      // Act
      const connectionKey = buildDirectionalKeyForConn(sourceNode, targetNode);

      // Assert
      expect(connectionKey).toBe('2->3');
    });

    it('keeps the reverse direction distinct for a backward edge', () => {
      // Arrange
      const sourceNode = createNode('hidden', 3);
      const targetNode = createNode('hidden', 2);

      // Act
      const connectionKey = buildDirectionalKeyForConn(sourceNode, targetNode);

      // Assert
      expect(connectionKey).toBe('3->2');
    });

    it('represents a self edge with its own exact directional key', () => {
      // Arrange
      const sourceNode = createNode('hidden', 2);

      // Act
      const connectionKey = buildDirectionalKeyForConn(sourceNode, sourceNode);

      // Assert
      expect(connectionKey).toBe('2->2');
    });

    describe('given node gene ids are missing', () => {
      it('falls back to zero gene ids in the directional key', () => {
        // Arrange
        const sourceNode = createNode('hidden', 2) as NodeWithMetadata & {
          geneId?: number;
        };
        const targetNode = createNode('hidden', 3) as NodeWithMetadata & {
          geneId?: number;
        };
        Reflect.deleteProperty(sourceNode, 'geneId');
        Reflect.deleteProperty(targetNode, 'geneId');

        // Act
        const connectionKey = buildDirectionalKeyForConn(
          sourceNode,
          targetNode,
        );

        // Assert
        expect(connectionKey).toBe('0->0');
      });
    });
  });

  describe('assignInnovationForConnection', () => {
    describe('given a brand-new node pair', () => {
      describe('when the connection innovation is allocated for the first time', () => {
        it('stores one reusable innovation id under the directional tracker key', () => {
          // Arrange
          const sourceNode = createNode('hidden', 2);
          const targetNode = createNode('hidden', 3);
          const pairNodes = {
            connectionKey: buildDirectionalKeyForConn(sourceNode, targetNode),
          };
          const connection: ConnectionWithMetadata = {
            from: sourceNode,
            to: targetNode,
            weight: 1,
          };
          const mutationController = createMutationController();

          // Act
          assignInnovationForConnection(
            connection,
            pairNodes,
            mutationController,
          );

          // Assert
          expect({
            connectionInnovation: connection.innovation,
            directionalInnovation:
              mutationController._innovationTracker.connectionInnovations.get(
                pairNodes.connectionKey,
              ),
            trackedInnovationCount:
              mutationController._innovationTracker.connectionInnovations.size,
          }).toEqual({
            connectionInnovation: 11,
            directionalInnovation: 11,
            trackedInnovationCount: 1,
          });
        });
      });

      describe('when opposite recurrent-capable directions are discovered separately', () => {
        it('assigns distinct innovations to each exact direction', () => {
          // Arrange
          const forwardSourceNode = createNode('hidden', 2);
          const forwardTargetNode = createNode('hidden', 3);
          const backwardSourceNode = createNode('hidden', 3);
          const backwardTargetNode = createNode('hidden', 2);
          const forwardPairNodes = {
            connectionKey: buildDirectionalKeyForConn(
              forwardSourceNode,
              forwardTargetNode,
            ),
          };
          const backwardPairNodes = {
            connectionKey: buildDirectionalKeyForConn(
              backwardSourceNode,
              backwardTargetNode,
            ),
          };
          const forwardConnection: ConnectionWithMetadata = {
            from: forwardSourceNode,
            to: forwardTargetNode,
            weight: 1,
          };
          const backwardConnection: ConnectionWithMetadata = {
            from: backwardSourceNode,
            to: backwardTargetNode,
            weight: 1,
          };
          const mutationController = createMutationController();

          // Act
          assignInnovationForConnection(
            forwardConnection,
            forwardPairNodes,
            mutationController,
          );
          assignInnovationForConnection(
            backwardConnection,
            backwardPairNodes,
            mutationController,
          );

          // Assert
          expect({
            forwardInnovation: forwardConnection.innovation,
            backwardInnovation: backwardConnection.innovation,
            trackedInnovations: Array.from(
              mutationController._innovationTracker.connectionInnovations.entries(),
            ),
          }).toEqual({
            forwardInnovation: 11,
            backwardInnovation: 12,
            trackedInnovations: [
              ['2->3', 11],
              ['3->2', 12],
            ],
          });
        });
      });
    });

    describe('given a historically known recurrent-capable direction', () => {
      describe('when that exact backward edge is absent and recreated later', () => {
        it('reuses the existing directional innovation id without advancing the tracker cursor', () => {
          // Arrange
          const sourceNode = createNode('hidden', 3);
          const targetNode = createNode('hidden', 2);
          const pairNodes = {
            connectionKey: buildDirectionalKeyForConn(sourceNode, targetNode),
          };
          const connection: ConnectionWithMetadata = {
            from: sourceNode,
            to: targetNode,
            weight: 1,
          };
          const mutationController = createMutationController();
          mutationController._innovationTracker.connectionInnovations.set(
            pairNodes.connectionKey,
            41,
          );

          // Act
          assignInnovationForConnection(
            connection,
            pairNodes,
            mutationController,
          );

          // Assert
          expect({
            connectionInnovation: connection.innovation,
            nextInnovationId:
              mutationController._innovationTracker.nextInnovationId,
          }).toEqual({
            connectionInnovation: 41,
            nextInnovationId: 11,
          });
        });
      });
    });
  });

  describe('createsCycle', () => {
    describe('given the proposed target can already reach the proposed source', () => {
      describe('when the cycle detector walks forward from that target node', () => {
        it('reports that the new edge would close a loop', () => {
          // Arrange
          const { firstHiddenNode, secondHiddenNode } =
            createCycleGuardScenario();

          // Act
          const wouldCreateCycle = createsCycle(
            firstHiddenNode,
            secondHiddenNode,
          );

          // Assert
          expect(wouldCreateCycle).toBe(true);
        });
      });
    });

    describe('given the proposed target has no path back to the proposed source', () => {
      describe('when the cycle detector walks forward from that target node', () => {
        it('reports that the new edge remains acyclic', () => {
          // Arrange
          const { secondHiddenNode, outputNode } = createCycleGuardScenario();

          // Act
          const wouldCreateCycle = createsCycle(secondHiddenNode, outputNode);

          // Assert
          expect(wouldCreateCycle).toBe(false);
        });
      });
    });

    describe('given the target node graph revisits a previously seen node', () => {
      describe('when the cycle detector walks forward from that target node', () => {
        it('skips the already-visited node and still reports an acyclic result', () => {
          // Arrange
          const sourceNode = createNode('hidden', 2);
          const targetNode = createNode('hidden', 3);
          const loopNode = createNode('hidden', 4);
          const cyclicalConnection = {
            from: targetNode,
            to: loopNode,
          } as ConnectionWithMetadata;
          const returnConnection = {
            from: loopNode,
            to: targetNode,
          } as ConnectionWithMetadata;
          targetNode.connections.out.push(cyclicalConnection);
          loopNode.connections.in.push(cyclicalConnection);
          loopNode.connections.out.push(returnConnection);
          targetNode.connections.in.push(returnConnection);

          // Act
          const wouldCreateCycle = createsCycle(sourceNode, targetNode);

          // Assert
          expect(wouldCreateCycle).toBe(false);
        });
      });
    });
  });

  describe('shouldAbortForCycle', () => {
    describe('given acyclic enforcement is disabled', () => {
      describe('when a cycle-forming pair is inspected', () => {
        it('allows the add-connection path to continue', () => {
          // Arrange
          const { genome, firstHiddenNode, secondHiddenNode } =
            createCycleGuardScenario();

          // Act
          const shouldAbortMutation = shouldAbortForCycle(genome, {
            sourceNode: firstHiddenNode,
            targetNode: secondHiddenNode,
          });

          // Assert
          expect(shouldAbortMutation).toBe(false);
        });
      });
    });

    describe('given acyclic enforcement is enabled', () => {
      describe('when a cycle-forming pair is inspected', () => {
        it('aborts the add-connection path before the edge is created', () => {
          // Arrange
          const { genome, firstHiddenNode, secondHiddenNode } =
            createCycleGuardScenario();
          genome._enforceAcyclic = true;

          // Act
          const shouldAbortMutation = shouldAbortForCycle(genome, {
            sourceNode: firstHiddenNode,
            targetNode: secondHiddenNode,
          });

          // Assert
          expect(shouldAbortMutation).toBe(true);
        });
      });
    });
  });

  describe('choosePairForConn', () => {
    describe('given no candidate pairs exist', () => {
      it('returns null when the pairs array is empty', () => {
        // Arrange & Act (covers line 229 TRUE arm)
        const chosenPair = choosePairForConn([], createMutationController());

        // Assert
        expect(chosenPair).toBeNull();
      });
    });

    describe('given exactly one candidate pair exists', () => {
      it('returns that pair without sampling the RNG', () => {
        // Arrange (covers line 232 TRUE arm)
        const singlePair = [
          createNode('hidden', 2),
          createNode('hidden', 3),
        ] as [NodeWithMetadata, NodeWithMetadata];

        // Act
        const chosenPair = choosePairForConn(
          [singlePair],
          createMutationController(),
        );

        // Assert
        expect(chosenPair).toBe(singlePair);
      });
    });

    describe('given multiple candidate pairs exist', () => {
      it('selects the pair chosen by the controller RNG', () => {
        // Arrange
        const firstPair = [
          createNode('hidden', 2),
          createNode('hidden', 3),
        ] as [NodeWithMetadata, NodeWithMetadata];
        const secondPair = [
          createNode('hidden', 3),
          createNode('hidden', 4),
        ] as [NodeWithMetadata, NodeWithMetadata];

        // Act
        const chosenPair = choosePairForConn([firstPair, secondPair], {
          ...createMutationController(),
          _getRNG: () => () => 0.75,
        });

        // Assert
        expect(chosenPair).toBe(secondPair);
      });
    });
  });

  describe('canApplyChosenPairForConn', () => {
    describe('given a valid feed-forward pair with default third arg', () => {
      it('accepts the pair under the default feed-forward policy', () => {
        // Arrange – no third arg → covers default-param branch (line 255)
        const { genome, firstHiddenNode, inputNode } =
          createConnectionReuseGenome();

        // Act
        const canApplyPair = canApplyChosenPairForConn(genome, [
          inputNode,
          firstHiddenNode,
        ]);

        // Assert
        expect(canApplyPair).toBe(true);
      });
    });

    describe('given a valid feed-forward pair', () => {
      it('accepts the pair under the feed-forward policy', () => {
        // Arrange
        const { genome, firstHiddenNode, inputNode } =
          createConnectionReuseGenome();

        // Act
        const canApplyPair = canApplyChosenPairForConn(
          genome,
          [inputNode, firstHiddenNode],
          false,
        );

        // Assert
        expect(canApplyPair).toBe(true);
      });
    });

    describe('given a valid recurrent pair', () => {
      it('accepts the pair under the recurrent policy', () => {
        // Arrange
        const { genome, firstHiddenNode, secondHiddenNode } =
          createConnectionReuseGenome();

        // Act
        const canApplyPair = canApplyChosenPairForConn(
          genome,
          [secondHiddenNode, firstHiddenNode],
          true,
        );

        // Assert
        expect(canApplyPair).toBe(true);
      });
    });
  });

  describe('selectPairPool', () => {
    describe('given reuse candidates exist', () => {
      it('returns the reuse pool', () => {
        // Arrange
        const allPairs = [
          [createNode('hidden', 2), createNode('hidden', 3)] as [
            NodeWithMetadata,
            NodeWithMetadata,
          ],
          [createNode('hidden', 3), createNode('hidden', 4)] as [
            NodeWithMetadata,
            NodeWithMetadata,
          ],
        ];
        const reusePairs = [allPairs[1]];

        // Act
        const selectedPool = selectPairPool(allPairs, reusePairs);

        // Assert
        expect(selectedPool).toBe(reusePairs);
      });
    });

    describe('given no reuse candidates exist but hidden-hidden candidates do', () => {
      it('returns the hidden pair pool', () => {
        // Arrange
        const allPairs = [
          [createNode('input', 1), createNode('hidden', 2)] as [
            NodeWithMetadata,
            NodeWithMetadata,
          ],
          [createNode('hidden', 3), createNode('hidden', 4)] as [
            NodeWithMetadata,
            NodeWithMetadata,
          ],
          [createNode('hidden', 4), createNode('output', 5)] as [
            NodeWithMetadata,
            NodeWithMetadata,
          ],
        ];

        // Act
        const selectedPool = selectPairPool(allPairs, []);

        // Assert
        expect(selectedPool).toEqual([allPairs[1]]);
      });
    });

    describe('given neither reuse nor hidden-hidden candidates exist', () => {
      it('returns the full candidate pool', () => {
        // Arrange
        const allPairs = [
          [createNode('input', 1), createNode('output', 2)] as [
            NodeWithMetadata,
            NodeWithMetadata,
          ],
          [createNode('input', 3), createNode('output', 4)] as [
            NodeWithMetadata,
            NodeWithMetadata,
          ],
        ];

        // Act
        const selectedPool = selectPairPool(allPairs, []);

        // Assert
        expect(selectedPool).toBe(allPairs);
      });
    });
  });

  describe('filterPairsWithInnovations', () => {
    describe('given only one candidate pair already has a recorded innovation', () => {
      it('returns only the reusable pair', () => {
        // Arrange
        const sourceNode = createNode('hidden', 2);
        const targetNode = createNode('hidden', 3);
        const unrelatedSourceNode = createNode('hidden', 4);
        const unrelatedTargetNode = createNode('hidden', 5);
        const reusablePair = [sourceNode, targetNode] as [
          NodeWithMetadata,
          NodeWithMetadata,
        ];
        const unreusedPair = [unrelatedSourceNode, unrelatedTargetNode] as [
          NodeWithMetadata,
          NodeWithMetadata,
        ];
        const mutationController = createMutationController();
        mutationController._innovationTracker.connectionInnovations.set(
          buildDirectionalKeyForConn(sourceNode, targetNode),
          42,
        );

        // Act
        const reusablePairs = filterPairsWithInnovations(
          [reusablePair, unreusedPair],
          mutationController,
        );

        // Assert
        expect(reusablePairs).toEqual([reusablePair]);
      });
    });
  });

  describe('connectChosenPairWithInnovationReuse', () => {
    describe('given a valid pair with default fourth arg', () => {
      it('uses feed-forward policy when allowRecurrentConnections is omitted', () => {
        // Arrange – no fourth arg → covers default-param branch (line 292)
        const sourceNode = createNode('input', 1);
        const targetNode = createNode('hidden', 2);
        const createdConnection: ConnectionWithMetadata = {
          from: sourceNode,
          to: targetNode,
          weight: 1,
        };
        const genome: GenomeWithMetadata = {
          nodes: [sourceNode, targetNode],
          connections: [],
          gates: [],
          input: 1,
          output: 1,
          connect: () => [createdConnection],
        } as GenomeWithMetadata;

        // Act
        const returnedConnection = connectChosenPairWithInnovationReuse(
          genome,
          [sourceNode, targetNode],
          createMutationController(),
        );

        // Assert
        expect(returnedConnection).toBe(createdConnection);
      });
    });

    describe('given the chosen pair is not legal for the active topology policy', () => {
      it('returns undefined without materializing a connection', () => {
        // Arrange
        const { genome, secondHiddenNode } = createConnectionReuseGenome();

        // Act
        const createdConnection = connectChosenPairWithInnovationReuse(
          genome,
          [createNode('hidden', 99), secondHiddenNode],
          createMutationController(),
          false,
        );

        // Assert
        expect(createdConnection).toBeUndefined();
      });
    });

    describe('given the source node lives in the output shelf during feed-forward growth', () => {
      it('returns undefined without materializing a connection', () => {
        // Arrange
        const { genome, firstHiddenNode, secondHiddenNode } =
          createConnectionReuseGenome();

        // Act
        const createdConnection = connectChosenPairWithInnovationReuse(
          genome,
          [secondHiddenNode, firstHiddenNode],
          createMutationController(),
          false,
        );

        // Assert
        expect(createdConnection).toBeUndefined();
      });
    });

    describe('given the genome connect hook creates one new connection', () => {
      it('returns the created connection after assigning innovation metadata', () => {
        // Arrange
        const sourceNode = createNode('input', 1);
        const targetNode = createNode('hidden', 2);
        const createdConnection: ConnectionWithMetadata = {
          from: sourceNode,
          to: targetNode,
          weight: 1,
        };
        const genome: GenomeWithMetadata = {
          nodes: [sourceNode, targetNode],
          connections: [],
          gates: [],
          input: 1,
          output: 1,
          connect: () => [createdConnection],
        } as GenomeWithMetadata;

        // Act
        const returnedConnection = connectChosenPairWithInnovationReuse(
          genome,
          [sourceNode, targetNode],
          createMutationController(),
          false,
        );

        // Assert
        expect(returnedConnection).toBe(createdConnection);
      });
    });

    describe('given a legal feed-forward pair but the genome connect hook returns nothing', () => {
      it('returns undefined after the connection hook fails', () => {
        // Arrange
        const { genome, firstHiddenNode, inputNode } =
          createConnectionReuseGenome();

        // Act
        const createdConnection = connectChosenPairWithInnovationReuse(
          genome,
          [inputNode, firstHiddenNode],
          createMutationController(),
          false,
        );

        // Assert
        expect(createdConnection).toBeUndefined();
      });
    });

    describe('given a legal recurrent pair but the genome connect hook returns nothing', () => {
      it('returns undefined after the connection hook fails', () => {
        // Arrange
        const { genome, firstHiddenNode, secondHiddenNode } =
          createConnectionReuseGenome();

        // Act
        const createdConnection = connectChosenPairWithInnovationReuse(
          genome,
          [secondHiddenNode, firstHiddenNode],
          createMutationController(),
          true,
        );

        // Assert
        expect(createdConnection).toBeUndefined();
      });
    });
  });
});

function createCycleGuardScenario(): {
  genome: GenomeWithMetadata;
  firstHiddenNode: NodeWithMetadata;
  secondHiddenNode: NodeWithMetadata;
  outputNode: NodeWithMetadata;
} {
  const inputNode = createNode('input', 1);
  const firstHiddenNode = createNode('hidden', 2);
  const secondHiddenNode = createNode('hidden', 3);
  const outputNode = createNode('output', 4);

  const genome: GenomeWithMetadata = {
    nodes: [inputNode, firstHiddenNode, secondHiddenNode, outputNode],
    connections: [],
    gates: [],
    input: 1,
    output: 1,
  };

  connectNodes(genome, secondHiddenNode, inputNode);
  connectNodes(genome, inputNode, firstHiddenNode);

  return {
    genome,
    firstHiddenNode,
    secondHiddenNode,
    outputNode,
  };
}

function createConnectionReuseGenome(): {
  genome: GenomeWithMetadata;
  inputNode: NodeWithMetadata;
  firstHiddenNode: NodeWithMetadata;
  secondHiddenNode: NodeWithMetadata;
} {
  const inputNode = createNode('input', 1);
  const firstHiddenNode = createNode('hidden', 2);
  const secondHiddenNode = createNode('hidden', 3);

  return {
    genome: {
      nodes: [inputNode, firstHiddenNode, secondHiddenNode],
      connections: [],
      gates: [],
      input: 1,
      output: 1,
      connect: () => [] as ConnectionWithMetadata[],
    } as GenomeWithMetadata,
    inputNode,
    firstHiddenNode,
    secondHiddenNode,
  };
}

function createCandidatePairGenome(): GenomeWithMetadata {
  const inputNode = createNode('input', 1);
  const hiddenNode = createNode('hidden', 2);
  const outputNode = createNode('output', 3);

  return {
    nodes: [inputNode, hiddenNode, outputNode],
    connections: [],
    gates: [],
    input: 1,
    output: 1,
  };
}

function createDormantEdgeGenome(): { genome: GenomeWithMetadata } {
  const inputNode = createNode('input', 1);
  const hiddenNode = createNode('hidden', 2);
  const outputNode = createNode('output', 3);
  const genome: GenomeWithMetadata = {
    nodes: [inputNode, hiddenNode, outputNode],
    connections: [],
    gates: [],
    input: 1,
    output: 1,
  };

  connectNodes(genome, outputNode, hiddenNode, false);

  return { genome };
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
    _innovationTracker: {
      ...createInnovationTracker(),
      nextInnovationId: 11,
    },
  };
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

function connectNodes(
  genome: GenomeWithMetadata,
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
  enabled: boolean = true,
): void {
  const connection: ConnectionWithMetadata = {
    from: sourceNode,
    to: targetNode,
    weight: 1,
    enabled,
  };

  sourceNode.connections.out.push(connection);
  targetNode.connections.in.push(connection);
  genome.connections.push(connection);
}

function summarizePairs(
  pairs: Array<[NodeWithMetadata, NodeWithMetadata]>,
): string[] {
  return pairs.map(
    ([sourceNode, targetNode]) => `${sourceNode.geneId}->${targetNode.geneId}`,
  );
}
