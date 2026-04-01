import { updateOperatorStatsIfNeeded } from './mutation.flow';
import type {
  GenomeWithMetadata,
  MutationMethod,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';

function createNode(type: NodeWithMetadata['type']): NodeWithMetadata {
  return {
    type,
    connections: { in: [], out: [] },
  };
}

function createGenome(input: {
  nodeCount: number;
  connectionCount: number;
}): GenomeWithMetadata {
  const nodes = [
    createNode('input'),
    createNode('input'),
    ...Array.from({ length: Math.max(0, input.nodeCount - 3) }, () =>
      createNode('hidden'),
    ),
    createNode('output'),
  ].slice(0, input.nodeCount);
  const sourceNode = nodes[0]!;
  const targetNode = nodes.at(-1)!;

  return {
    nodes,
    connections: Array.from({ length: input.connectionCount }, () => ({
      from: sourceNode,
      to: targetNode,
      weight: 1,
    })),
    gates: [],
    input: 2,
    output: 1,
  };
}

function createMutationController(): NeatControllerForMutation {
  return {
    population: [],
    options: {
      mutation: [],
      operatorAdaptation: { enabled: true },
    },
    _getRNG: () => () => 0,
    selectMutationMethod: async () => null,
    _mutateAddNodeReuse: async () => undefined,
    _mutateAddConnReuse: () => undefined,
    _invalidateGenomeCaches: () => undefined,
    _operatorStats: new Map(),
    _nodeSplitInnovations: new Map(),
    _connInnovations: new Map(),
    _nextGlobalInnovation: 1,
  };
}

describe('neat mutation flow chapter', () => {
  describe('updateOperatorStatsIfNeeded', () => {
    describe('given operator adaptation is enabled and the mutation grows structure', () => {
      it('records one successful attempt for the operator', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 4 });
        const mutationController = createMutationController();
        const mutationMethod: MutationMethod = { name: 'ADD_NODE' };

        // Act
        updateOperatorStatsIfNeeded(
          genome,
          mutationMethod,
          { beforeNodes: 2, beforeConns: 4 },
          mutationController,
        );

        // Assert
        expect(Array.from(mutationController._operatorStats.entries())).toEqual(
          [['ADD_NODE', { success: 1, attempts: 1 }]],
        );
      });
    });
  });
});
