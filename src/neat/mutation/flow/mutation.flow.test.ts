import {
  applyAddConnMutation,
  applyMutationOperator,
  captureStructuralSizes,
  initializeAdaptiveMutation,
  maybeAddExtraConnection,
  mutateGenome,
  resolveEffectiveAmount,
  resolveEffectiveRate,
  selectConcreteMutationMethod,
  updateOperatorStatsIfNeeded,
} from './mutation.flow';
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

type MutationControllerOverrides = {
  optionOverrides?: Partial<NeatControllerForMutation['options']>;
  randomValues?: number[];
  selectMutationMethod?: NeatControllerForMutation['selectMutationMethod'];
  mutateAddNodeReuse?: NeatControllerForMutation['_mutateAddNodeReuse'];
  mutateAddConnReuse?: NeatControllerForMutation['_mutateAddConnReuse'];
  invalidateGenomeCaches?: NeatControllerForMutation['_invalidateGenomeCaches'];
  operatorStats?: NeatControllerForMutation['_operatorStats'];
};

function createDeterministicRng(randomValues: number[]): () => number {
  let randomValueIndex = 0;

  return () => {
    const sampledValue =
      randomValues[randomValueIndex] ?? randomValues.at(-1) ?? 0;
    randomValueIndex++;
    return sampledValue;
  };
}

function createMutationController(
  overrides: MutationControllerOverrides = {},
): NeatControllerForMutation {
  const randomNumberGenerator = createDeterministicRng(
    overrides.randomValues ?? [0],
  );

  return {
    population: [],
    options: {
      mutation: [],
      operatorAdaptation: { enabled: true },
      ...overrides.optionOverrides,
    },
    _getRNG: () => randomNumberGenerator,
    selectMutationMethod: overrides.selectMutationMethod ?? (async () => null),
    _mutateAddNodeReuse:
      overrides.mutateAddNodeReuse ?? (async () => undefined),
    _mutateAddConnReuse: overrides.mutateAddConnReuse ?? (() => undefined),
    _invalidateGenomeCaches:
      overrides.invalidateGenomeCaches ?? (() => undefined),
    _operatorStats: overrides.operatorStats ?? new Map(),
    _innovationTracker: createInnovationTracker(),
  };
}

function createMutationMethods(overrides: Record<string, unknown> = {}): {
  mutation: unknown;
} {
  return {
    mutation: {
      ADD_NODE: { name: 'ADD_NODE' },
      ADD_CONN: { name: 'ADD_CONN' },
      MOD_WEIGHT: { name: 'MOD_WEIGHT', min: -0.1, max: 0.1 },
      ADD_GATE: { name: 'ADD_GATE' },
      SUB_NODE: { name: 'SUB_NODE' },
      SUB_CONN: { name: 'SUB_CONN' },
      ADD_SELF_CONN: { name: 'ADD_SELF_CONN' },
      ADD_BACK_CONN: { name: 'ADD_BACK_CONN' },
      ...overrides,
    },
  };
}

describe('neat mutation flow chapter', () => {
  describe('mutateGenome', () => {
    describe('given the rng gate rejects mutation', () => {
      it('initializes adaptive state and skips operator selection', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        let selectionCalls = 0;
        const mutationController = createMutationController({
          randomValues: [0.9],
          optionOverrides: {
            adaptiveMutation: { enabled: true, initialRate: 0.2 },
            mutationRate: 0.1,
            mutationAmount: 2,
          },
          selectMutationMethod: async () => {
            selectionCalls++;
            return { name: 'ADD_CONN' };
          },
        });

        // Act
        await mutateGenome(genome, mutationController, createMutationMethods());

        // Assert
        expect({
          mutationRate: genome._mutRate,
          selectionCalls,
          connectionCount: genome.connections.length,
        }).toEqual({
          mutationRate: 0.1,
          selectionCalls: 0,
          connectionCount: 1,
        });
      });
    });

    describe('given one selected descriptor is empty before a valid operator is chosen', () => {
      it('skips the empty selection and applies the later operator attempt', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const invalidatedGenomes: GenomeWithMetadata[] = [];
        let selectionIndex = 0;
        const mutationController = createMutationController({
          randomValues: [0, 0, 0, 0],
          optionOverrides: {
            mutationRate: 1,
            mutationAmount: 2,
            operatorAdaptation: { enabled: true },
          },
          selectMutationMethod: async () => {
            const selectedMethods: MutationMethod[] = [
              { name: '' },
              { name: 'ADD_CONN' },
            ];
            const selectedMethod = selectedMethods[selectionIndex] ?? null;
            selectionIndex++;
            return selectedMethod;
          },
          mutateAddConnReuse: (candidateGenome) => {
            candidateGenome.connections.push({
              from: candidateGenome.nodes[0]!,
              to: candidateGenome.nodes.at(-1)!,
              weight: 1,
            });
          },
          invalidateGenomeCaches: (candidateGenome) => {
            invalidatedGenomes.push(candidateGenome);
          },
        });

        // Act
        await mutateGenome(genome, mutationController, createMutationMethods());

        // Assert
        expect({
          selectionIndex,
          invalidatedCount: invalidatedGenomes.length,
          connectionCount: genome.connections.length,
          operatorStats: Array.from(
            mutationController._operatorStats.entries(),
          ),
        }).toEqual({
          selectionIndex: 2,
          invalidatedCount: 1,
          connectionCount: 3,
          operatorStats: [['ADD_CONN', { success: 1, attempts: 1 }]],
        });
      });
    });
  });

  describe('initializeAdaptiveMutation', () => {
    describe('given adaptive mutation is disabled', () => {
      it('leaves the genome mutation state unchanged', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: false },
          },
        });

        // Act
        initializeAdaptiveMutation(genome, mutationController);

        // Assert
        expect({
          mutationRate: genome._mutRate,
          mutationAmount: genome._mutAmount,
        }).toEqual({
          mutationRate: undefined,
          mutationAmount: undefined,
        });
      });
    });

    describe('given the genome already carries adaptive mutation state', () => {
      it('keeps the existing per-genome rate and amount', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        genome._mutRate = 0.25;
        genome._mutAmount = 4;
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: true, adaptAmount: true },
            mutationRate: 0.9,
            mutationAmount: 7,
          },
        });

        // Act
        initializeAdaptiveMutation(genome, mutationController);

        // Assert
        expect({
          mutationRate: genome._mutRate,
          mutationAmount: genome._mutAmount,
        }).toEqual({
          mutationRate: 0.25,
          mutationAmount: 4,
        });
      });
    });

    describe('given explicit controller mutation settings are available', () => {
      it('bootstraps the genome from those controller values', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: true, adaptAmount: true },
            mutationRate: 0.8,
            mutationAmount: 3,
          },
        });

        // Act
        initializeAdaptiveMutation(genome, mutationController);

        // Assert
        expect({
          mutationRate: genome._mutRate,
          mutationAmount: genome._mutAmount,
        }).toEqual({
          mutationRate: 0.8,
          mutationAmount: 3,
        });
      });
    });

    describe('given only adaptive defaults are available', () => {
      it('uses the initial rate and default amount fallback', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: {
              enabled: true,
              initialRate: 0.4,
              adaptAmount: true,
            },
          },
        });

        // Act
        initializeAdaptiveMutation(genome, mutationController);

        // Assert
        expect({
          mutationRate: genome._mutRate,
          mutationAmount: genome._mutAmount,
        }).toEqual({
          mutationRate: 0.4,
          mutationAmount: 1,
        });
      });
    });

    describe('given no adaptive defaults are configured', () => {
      it('falls back to the mutation-rate constant without assigning an amount', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: true, adaptAmount: false },
          },
        });

        // Act
        initializeAdaptiveMutation(genome, mutationController);

        // Assert
        expect({
          mutationRate: genome._mutRate,
          mutationAmount: genome._mutAmount,
        }).toEqual({
          mutationRate: 0.7,
          mutationAmount: undefined,
        });
      });
    });
  });

  describe('resolveEffectiveRate', () => {
    describe('given the controller specifies a global mutation rate', () => {
      it('returns the controller override', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        genome._mutRate = 0.25;
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: true },
            mutationRate: 0.55,
          },
        });

        // Act
        const effectiveRate = resolveEffectiveRate(genome, mutationController);

        // Assert
        expect(effectiveRate).toBe(0.55);
      });
    });

    describe('given adaptive mutation already assigned a genome rate', () => {
      it('uses the genome-specific rate', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        genome._mutRate = 0.33;
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: true },
          },
        });

        // Act
        const effectiveRate = resolveEffectiveRate(genome, mutationController);

        // Assert
        expect(effectiveRate).toBe(0.33);
      });
    });

    describe('given adaptive mutation has no genome-specific rate yet', () => {
      it('falls back to the default mutation-rate constant', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: true },
          },
        });

        // Act
        const effectiveRate = resolveEffectiveRate(genome, mutationController);

        // Assert
        expect(effectiveRate).toBe(0.7);
      });
    });

    describe('given adaptive mutation is disabled and no rate is configured', () => {
      it('returns the default mutation-rate constant', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: false },
          },
        });

        // Act
        const effectiveRate = resolveEffectiveRate(genome, mutationController);

        // Assert
        expect(effectiveRate).toBe(0.7);
      });
    });
  });

  describe('resolveEffectiveAmount', () => {
    describe('given adaptive mutation already assigned a genome amount', () => {
      it('uses the genome-specific attempt budget', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        genome._mutAmount = 5;
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: true, adaptAmount: true },
            mutationAmount: 2,
          },
        });

        // Act
        const effectiveAmount = resolveEffectiveAmount(
          genome,
          mutationController,
        );

        // Assert
        expect(effectiveAmount).toBe(5);
      });
    });

    describe('given adaptive mutation amount is enabled without a genome amount', () => {
      it('falls back to the default attempt count', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: true, adaptAmount: true },
          },
        });

        // Act
        const effectiveAmount = resolveEffectiveAmount(
          genome,
          mutationController,
        );

        // Assert
        expect(effectiveAmount).toBe(1);
      });
    });

    describe('given adaptive amount is disabled', () => {
      it('uses the configured controller amount', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: true, adaptAmount: false },
            mutationAmount: 4,
          },
        });

        // Act
        const effectiveAmount = resolveEffectiveAmount(
          genome,
          mutationController,
        );

        // Assert
        expect(effectiveAmount).toBe(4);
      });
    });

    describe('given no mutation amount is configured anywhere', () => {
      it('returns the default attempt count', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          optionOverrides: {
            adaptiveMutation: { enabled: false },
          },
        });

        // Act
        const effectiveAmount = resolveEffectiveAmount(
          genome,
          mutationController,
        );

        // Assert
        expect(effectiveAmount).toBe(1);
      });
    });
  });

  describe('selectConcreteMutationMethod', () => {
    describe('given selection returns one concrete operator descriptor', () => {
      it('returns that operator directly', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationMethod = { name: 'ADD_CONN' };
        const mutationController = createMutationController({
          selectMutationMethod: async () => mutationMethod,
        });

        // Act
        const resolvedMethod = await selectConcreteMutationMethod(
          genome,
          mutationController,
        );

        // Assert
        expect(resolvedMethod).toEqual(mutationMethod);
      });
    });

    describe('given selection returns no operator', () => {
      it('normalizes the result to null', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          selectMutationMethod: async () => null,
        });

        // Act
        const resolvedMethod = await selectConcreteMutationMethod(
          genome,
          mutationController,
        );

        // Assert
        expect(resolvedMethod).toBeNull();
      });
    });

    describe('given selection returns a legacy operator pool', () => {
      it('samples one operator using the controller rng', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          randomValues: [0.75],
          selectMutationMethod: async () => [
            { name: 'ADD_NODE' },
            { name: 'ADD_CONN' },
          ],
        });

        // Act
        const resolvedMethod = await selectConcreteMutationMethod(
          genome,
          mutationController,
        );

        // Assert
        expect(resolvedMethod).toEqual({ name: 'ADD_CONN' });
      });
    });

    describe('given selection returns an empty legacy operator pool', () => {
      it('normalizes the missing sampled operator to null', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const mutationController = createMutationController({
          selectMutationMethod: async () => [],
        });

        // Act
        const resolvedMethod = await selectConcreteMutationMethod(
          genome,
          mutationController,
        );

        // Assert
        expect(resolvedMethod).toBeNull();
      });
    });
  });

  describe('captureStructuralSizes', () => {
    describe('given a genome already has nodes and connections', () => {
      it('returns the current node and connection counts', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 5, connectionCount: 3 });

        // Act
        const structuralSizes = captureStructuralSizes(genome);

        // Assert
        expect(structuralSizes).toEqual({ beforeNodes: 5, beforeConns: 3 });
      });
    });
  });

  describe('applyMutationOperator', () => {
    describe('given a structural operator descriptor matches by name only', () => {
      it('routes ADD_NODE through the reuse-aware controller helper', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const addNodeCalls: GenomeWithMetadata[] = [];
        genome.mutate = jest.fn();
        const mutationController = createMutationController({
          mutateAddNodeReuse: async (candidateGenome) => {
            addNodeCalls.push(candidateGenome);
          },
        });
        const methods = createMutationMethods();
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

    describe('given add-node reuse leaves no connection available for weight nudging', () => {
      it('still invalidates caches after the structural helper runs', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 0 });
        const addNodeCalls: GenomeWithMetadata[] = [];
        const invalidatedGenomes: GenomeWithMetadata[] = [];
        const mutationController = createMutationController({
          mutateAddNodeReuse: async (candidateGenome) => {
            addNodeCalls.push(candidateGenome);
          },
          invalidateGenomeCaches: (candidateGenome) => {
            invalidatedGenomes.push(candidateGenome);
          },
        });

        // Act
        await applyMutationOperator(
          genome,
          { name: 'ADD_NODE' },
          mutationController,
          createMutationMethods(),
        );

        // Assert
        expect({
          addNodeCalls,
          invalidatedGenomes,
          connectionCount: genome.connections.length,
        }).toEqual({
          addNodeCalls: [genome],
          invalidatedGenomes: [genome],
          connectionCount: 0,
        });
      });
    });

    describe('given a structural operator descriptor matches ADD_CONN by name only', () => {
      it('routes ADD_CONN through the reuse-aware controller helper', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const addConnCalls: GenomeWithMetadata[] = [];
        const mutationController = createMutationController({
          mutateAddConnReuse: (candidateGenome) => {
            addConnCalls.push(candidateGenome);
          },
        });

        // Act
        await applyMutationOperator(
          genome,
          { name: 'ADD_CONN' },
          mutationController,
          createMutationMethods(),
        );

        // Assert
        expect(addConnCalls).toEqual([genome]);
      });
    });

    describe('given a local operator is one of the cache invalidators', () => {
      it('delegates to genome mutation and clears caches', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const invalidatedGenomes: GenomeWithMetadata[] = [];
        const mutationController = createMutationController({
          invalidateGenomeCaches: (candidateGenome) => {
            invalidatedGenomes.push(candidateGenome);
          },
        });
        const methods = createMutationMethods();
        const selectedMethod = (
          methods.mutation as Record<string, MutationMethod>
        ).ADD_GATE;
        genome.mutate = jest.fn();

        // Act
        await applyMutationOperator(
          genome,
          selectedMethod,
          mutationController,
          methods,
        );

        // Assert
        expect({
          mutateCalls: (genome.mutate as jest.Mock).mock.calls,
          invalidatedGenomes,
        }).toEqual({
          mutateCalls: [[selectedMethod]],
          invalidatedGenomes: [genome],
        });
      });
    });

    describe('given a local operator does not affect cached topology', () => {
      it('delegates to genome mutation without clearing caches', async () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const invalidatedGenomes: GenomeWithMetadata[] = [];
        const mutationController = createMutationController({
          invalidateGenomeCaches: (candidateGenome) => {
            invalidatedGenomes.push(candidateGenome);
          },
        });
        const selectedMethod: MutationMethod = { name: 'MOD_BIAS' };
        genome.mutate = jest.fn();

        // Act
        await applyMutationOperator(
          genome,
          selectedMethod,
          mutationController,
          createMutationMethods(),
        );

        // Assert
        expect({
          mutateCalls: (genome.mutate as jest.Mock).mock.calls,
          invalidatedGenomes,
        }).toEqual({
          mutateCalls: [[selectedMethod]],
          invalidatedGenomes: [],
        });
      });
    });
  });

  describe('applyAddConnMutation', () => {
    describe('given MOD_WEIGHT is not an object descriptor', () => {
      it('uses the default weight bounds for the deterministic nudge', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const addConnCalls: GenomeWithMetadata[] = [];
        const invalidatedGenomes: GenomeWithMetadata[] = [];
        const mutationController = createMutationController({
          randomValues: [0, 0],
          mutateAddConnReuse: (candidateGenome) => {
            addConnCalls.push(candidateGenome);
          },
          invalidateGenomeCaches: (candidateGenome) => {
            invalidatedGenomes.push(candidateGenome);
          },
        });

        // Act
        applyAddConnMutation(genome, mutationController, {
          mutation: {
            MOD_WEIGHT: 'MOD_WEIGHT',
          },
        });

        // Assert
        expect({
          addConnCalls,
          invalidatedGenomes,
          weights: genome.connections.map((connection) => connection.weight),
        }).toEqual({
          addConnCalls: [genome],
          invalidatedGenomes: [genome],
          weights: [0],
        });
      });
    });
  });

  describe('maybeAddExtraConnection', () => {
    describe('given the exploration roll falls below the extra-connection threshold', () => {
      it('adds one extra connection through the reuse helper', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const addConnCalls: GenomeWithMetadata[] = [];
        const mutationController = createMutationController({
          randomValues: [0],
          mutateAddConnReuse: (candidateGenome) => {
            addConnCalls.push(candidateGenome);
          },
        });

        // Act
        maybeAddExtraConnection(genome, mutationController);

        // Assert
        expect(addConnCalls).toEqual([genome]);
      });
    });

    describe('given the exploration roll stays above the extra-connection threshold', () => {
      it('skips the extra connection hook', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 1 });
        const addConnCalls: GenomeWithMetadata[] = [];
        const mutationController = createMutationController({
          randomValues: [1],
          mutateAddConnReuse: (candidateGenome) => {
            addConnCalls.push(candidateGenome);
          },
        });

        // Act
        maybeAddExtraConnection(genome, mutationController);

        // Assert
        expect(addConnCalls).toEqual([]);
      });
    });
  });

  describe('updateOperatorStatsIfNeeded', () => {
    describe('given operator adaptation is disabled', () => {
      it('leaves the operator statistics map unchanged', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 4 });
        const mutationController = createMutationController({
          optionOverrides: {
            operatorAdaptation: { enabled: false },
          },
        });
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
          [],
        );
      });
    });

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

    describe('given only the connection count grows', () => {
      it('records the attempt as a structural success', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 4 });
        const mutationController = createMutationController();
        const mutationMethod: MutationMethod = { name: 'ADD_CONN' };

        // Act
        updateOperatorStatsIfNeeded(
          genome,
          mutationMethod,
          { beforeNodes: 3, beforeConns: 3 },
          mutationController,
        );

        // Assert
        expect(Array.from(mutationController._operatorStats.entries())).toEqual(
          [['ADD_CONN', { success: 1, attempts: 1 }]],
        );
      });
    });

    describe('given the mutation does not grow structure', () => {
      it('records an attempt without increasing the success counter', () => {
        // Arrange
        const genome = createGenome({ nodeCount: 3, connectionCount: 4 });
        const mutationController = createMutationController();
        const mutationMethod: MutationMethod = { name: 'SUB_CONN' };

        // Act
        updateOperatorStatsIfNeeded(
          genome,
          mutationMethod,
          { beforeNodes: 3, beforeConns: 4 },
          mutationController,
        );

        // Assert
        expect(Array.from(mutationController._operatorStats.entries())).toEqual(
          [['SUB_CONN', { success: 0, attempts: 1 }]],
        );
      });
    });
  });
});
