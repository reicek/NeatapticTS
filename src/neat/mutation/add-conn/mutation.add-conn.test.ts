import type {
  ConnectionWithMetadata,
  NeatControllerForMutation,
  GenomeWithMetadata,
  NodeWithMetadata,
} from '../shared/mutation.types';
import {
  assignInnovationForConnection,
  buildLegacyKeyForConn,
  buildSymmetricKeyForConn,
  createsCycle,
  shouldAbortForCycle,
} from './mutation.add-conn';

describe('neat mutation add-connection chapter', () => {
  describe('assignInnovationForConnection', () => {
    describe('given a brand-new node pair', () => {
      describe('when the connection innovation is allocated for the first time', () => {
        it('stores the same innovation id under symmetric and legacy keys', () => {
          // Arrange
          const sourceNode = createNode('hidden', 2);
          const targetNode = createNode('hidden', 3);
          const pairNodes = {
            symmetricKey: buildSymmetricKeyForConn(sourceNode, targetNode),
            legacyForwardKey: buildLegacyKeyForConn(sourceNode, targetNode),
            legacyReverseKey: buildLegacyKeyForConn(targetNode, sourceNode),
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
            symmetricInnovation: mutationController._connInnovations.get(
              pairNodes.symmetricKey,
            ),
            forwardInnovation: mutationController._connInnovations.get(
              pairNodes.legacyForwardKey,
            ),
            reverseInnovation: mutationController._connInnovations.get(
              pairNodes.legacyReverseKey,
            ),
          }).toEqual({
            connectionInnovation: 11,
            symmetricInnovation: 11,
            forwardInnovation: 11,
            reverseInnovation: 11,
          });
        });
      });
    });

    describe('given a historically known node pair', () => {
      describe('when a replacement connection is created later', () => {
        it('reuses the existing innovation id without advancing the counter', () => {
          // Arrange
          const sourceNode = createNode('hidden', 2);
          const targetNode = createNode('hidden', 3);
          const pairNodes = {
            symmetricKey: buildSymmetricKeyForConn(sourceNode, targetNode),
            legacyForwardKey: buildLegacyKeyForConn(sourceNode, targetNode),
            legacyReverseKey: buildLegacyKeyForConn(targetNode, sourceNode),
          };
          const connection: ConnectionWithMetadata = {
            from: sourceNode,
            to: targetNode,
            weight: 1,
          };
          const mutationController = createMutationController();
          mutationController._connInnovations.set(pairNodes.symmetricKey, 41);

          // Act
          assignInnovationForConnection(
            connection,
            pairNodes,
            mutationController,
          );

          // Assert
          expect({
            connectionInnovation: connection.innovation,
            nextGlobalInnovation: mutationController._nextGlobalInnovation,
          }).toEqual({
            connectionInnovation: 41,
            nextGlobalInnovation: 11,
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
): void {
  const connection: ConnectionWithMetadata = {
    from: sourceNode,
    to: targetNode,
    weight: 1,
  };

  sourceNode.connections.out.push(connection);
  targetNode.connections.in.push(connection);
  genome.connections.push(connection);
}
