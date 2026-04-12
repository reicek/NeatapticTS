import { applyMutationOperator, updateOperatorStatsIfNeeded } from './mutation.flow';
import { createInnovationTracker } from '../../innovation-tracker/innovation-tracker';
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
    _innovationTracker: createInnovationTracker(),
  };
}

describe('neat mutation flow chapter', () => {
  describe('applyMutationOperator', () => {
    describe('given a structural operator descriptor matches by name only', () => {
      it('routes ADD_NODE through the reuse-aware controller helper', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const addNodeCalls: GenomeWithMetadata[] = [];
        genome.mutate = jest.fn();
        const mutationController = createMutationController();
        mutationController._mutateAddNodeReuse = async (candidateGenome) => {
          addNodeCalls.push(candidateGenome);
        };
        const methods = {
          mutation: {
            ADD_NODE: { name: 'ADD_NODE' },
            ADD_CONN: { name: 'ADD_CONN' },
            MOD_WEIGHT: { name: 'MOD_WEIGHT', min: -0.1, max: 0.1 },
          },
        };
        const selectedMethod: MutationMethod = { name: 'ADD_NODE' };

        // Act
        await applyMutationOperator(
          genome,
          selectedMethod,
          mutationController,
          methods,
        );

        // Assert
        expect({
          addNodeCalls,
          mutateCalls: (genome.mutate as jest.Mock).mock.calls.length,
        }).toEqual({
          addNodeCalls: [genome],
          mutateCalls: 0,
        });
      });
    });
  });

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
