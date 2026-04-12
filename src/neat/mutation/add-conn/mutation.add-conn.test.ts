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
  collectCandidatePairsForConn,
  createsCycle,
  shouldAbortForCycle,
} from './mutation.add-conn';

describe('neat mutation add-connection chapter', () => {
  describe('collectCandidatePairsForConn', () => {
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
