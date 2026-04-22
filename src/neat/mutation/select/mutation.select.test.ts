import * as methods from '../../../methods/methods';
import { createInnovationTracker } from '../../innovation-tracker/innovation-tracker';
import {
  applyOperatorAdaptationForSelect,
  applyOperatorBanditForSelect,
  applyPhasedComplexityForSelect,
  isBlockedByRecurrentPolicyForSelect,
  isBlockedByStructuralLimitsForSelect,
  isLegacyFFWPoolForSelect,
  isOperatorNamePrefixedForSelect,
  normalizeMutationPoolForSelect,
  resolveFFWPolicyForSelect,
  sampleFromPoolForSelect,
} from './mutation.select';
import type {
  GenomeWithMetadata,
  MutationMethod,
  NeatControllerForMutation,
} from '../shared/mutation.types';

function createMutationMethod(name: string): MutationMethod {
  return { name };
}

function createLegacyFfwPool(): MutationMethod[] {
  return methods.mutation.FFW as unknown as MutationMethod[];
}

function createSelectionController(input: {
  operatorStats?: Map<string, { success: number; attempts: number }>;
  operatorAdaptation?: NeatControllerForMutation['options']['operatorAdaptation'];
  operatorBandit?: NeatControllerForMutation['options']['operatorBandit'];
  phasedComplexity?: NeatControllerForMutation['options']['phasedComplexity'];
  mutationPool?: Array<MutationMethod | undefined>;
  maxNodes?: number;
  maxConns?: number;
  maxGates?: number;
  allowRecurrent?: boolean;
  phase?: NeatControllerForMutation['_phase'];
  randomValue?: number;
}): NeatControllerForMutation {
  return {
    population: [],
    options: {
      mutation: input.mutationPool ?? [],
      phasedComplexity: input.phasedComplexity,
      operatorAdaptation: input.operatorAdaptation,
      operatorBandit: input.operatorBandit,
      maxNodes: input.maxNodes,
      maxConns: input.maxConns,
      maxGates: input.maxGates,
      allowRecurrent: input.allowRecurrent,
    },
    _getRNG: () => () => input.randomValue ?? 0,
    selectMutationMethod: async () => null,
    _mutateAddNodeReuse: async () => undefined,
    _mutateAddConnReuse: () => undefined,
    _invalidateGenomeCaches: () => undefined,
    _operatorStats: input.operatorStats ?? new Map(),
    _innovationTracker: createInnovationTracker(),
    _phase: input.phase,
  };
}

function createStructuralGenome(input: {
  nodeCount?: number;
  connectionCount?: number;
  gateCount?: number;
  topologyIntent?: 'feed-forward' | 'unconstrained';
  enforceAcyclic?: boolean;
}): GenomeWithMetadata {
  return {
    nodes: Array.from({ length: input.nodeCount ?? 0 }, () => ({})),
    connections: Array.from({ length: input.connectionCount ?? 0 }, () => ({})),
    gates: Array.from({ length: input.gateCount ?? 0 }, () => ({})),
    _enforceAcyclic: input.enforceAcyclic,
    getTopologyIntent: input.topologyIntent
      ? () => input.topologyIntent as 'feed-forward' | 'unconstrained'
      : undefined,
  } as unknown as GenomeWithMetadata;
}

describe('neat mutation selection chapter', () => {
  describe('resolveFFWPolicyForSelect', () => {
    describe('given the controller keeps the legacy direct FFW policy and the caller wants the raw pool', () => {
      it('returns the preserved FFW operator array', () => {
        // Arrange
        const selectionController = createSelectionController({});
        selectionController.options.mutation =
          createLegacyFfwPool() as unknown as NeatControllerForMutation['options']['mutation'];

        // Act
        const resolvedPolicy = resolveFFWPolicyForSelect(
          selectionController,
          methods,
          true,
        );

        // Assert
        expect(resolvedPolicy).toBe(methods.mutation.FFW);
      });
    });

    describe('given the controller keeps one nested legacy FFW policy and the caller wants a concrete method', () => {
      it('samples one mutation method from the nested FFW array', () => {
        // Arrange
        const selectionController = createSelectionController({
          mutationPool: [methods.mutation.FFW as unknown as MutationMethod],
          randomValue: 0,
        });

        // Act
        const resolvedPolicy = resolveFFWPolicyForSelect(
          selectionController,
          methods,
          false,
        );

        // Assert
        expect(resolvedPolicy).toBe(createLegacyFfwPool()[0]);
      });
    });

    describe('given the controller does not use a legacy FFW policy', () => {
      it('returns null', () => {
        // Arrange
        const selectionController = createSelectionController({
          mutationPool: [createMutationMethod('ADD_NODE')],
        });

        // Act
        const resolvedPolicy = resolveFFWPolicyForSelect(
          selectionController,
          methods,
          false,
        );

        // Assert
        expect(resolvedPolicy).toBeNull();
      });
    });

    describe('given the controller keeps one nested legacy FFW policy and the caller wants a concrete method', () => {
      it('samples one mutation method from the nested FFW array', () => {
        // Arrange
        const selectionController = createSelectionController({
          mutationPool: [methods.mutation.FFW as unknown as MutationMethod],
          randomValue: 0,
        });

        // Act
        const resolvedPolicy = resolveFFWPolicyForSelect(
          selectionController,
          methods,
          false,
        );

        // Assert
        expect(resolvedPolicy).toBe(createLegacyFfwPool()[0]);
      });
    });

    describe('given the controller does not use a legacy FFW policy', () => {
      it('returns null', () => {
        // Arrange
        const selectionController = createSelectionController({
          mutationPool: [createMutationMethod('ADD_NODE')],
        });

        // Act
        const resolvedPolicy = resolveFFWPolicyForSelect(
          selectionController,
          methods,
          false,
        );

        // Assert
        expect(resolvedPolicy).toBeNull();
      });
    });
  });

  describe('normalizeMutationPoolForSelect', () => {
    describe('given the configured pool matches the legacy FFW ordering and tests request the raw policy', () => {
      it('returns the canonical FFW operator array', () => {
        // Arrange
        const selectionController = createSelectionController({
          mutationPool: createLegacyFfwPool(),
        });

        // Act
        const normalizedPool = normalizeMutationPoolForSelect(
          selectionController,
          methods,
          true,
        );

        // Assert
        expect(normalizedPool).toBe(methods.mutation.FFW);
      });
    });

    describe('given the configured pool keeps one nested operator list', () => {
      it('unwraps that nested list into the concrete candidate pool', () => {
        // Arrange
        const addNodeMethod = createMutationMethod('ADD_NODE');
        const subtractConnectionMethod = createMutationMethod('SUB_CONN');
        const selectionController = createSelectionController({
          mutationPool: [
            [addNodeMethod, subtractConnectionMethod] as unknown as MutationMethod,
          ],
        });

        // Act
        const normalizedPool = normalizeMutationPoolForSelect(
          selectionController,
          methods,
          false,
        );

        // Assert
        expect(normalizedPool).toEqual([
          addNodeMethod,
          subtractConnectionMethod,
        ]);
      });
    });
  });

  describe('isLegacyFFWPoolForSelect', () => {
    describe('given the configured mutation policy is not an array shape', () => {
      it('rejects the policy as a legacy FFW pool', () => {
        // Arrange
        const configuredPool = methods.mutation.ADD_NODE as unknown as MutationMethod[];

        // Act
        const isLegacyFfwPool = isLegacyFFWPoolForSelect(
          configuredPool,
          methods,
        );

        // Assert
        expect(isLegacyFfwPool).toBe(false);
      });
    });

    describe('given the configured pool length differs from the canonical FFW ordering', () => {
      it('rejects the policy as a legacy FFW pool', () => {
        // Arrange
        const configuredPool = [createMutationMethod('ADD_NODE')];

        // Act
        const isLegacyFfwPool = isLegacyFFWPoolForSelect(
          configuredPool,
          methods,
        );

        // Assert
        expect(isLegacyFfwPool).toBe(false);
      });
    });

    describe('given the configured pool matches the canonical FFW ordering', () => {
      it('recognizes the pool as the legacy FFW preset', () => {
        // Arrange
        const configuredPool = [...createLegacyFfwPool()];

        // Act
        const isLegacyFfwPool = isLegacyFFWPoolForSelect(
          configuredPool,
          methods,
        );

        // Assert
        expect(isLegacyFfwPool).toBe(true);
      });
    });
  });

  describe('applyOperatorAdaptationForSelect', () => {
    describe('given operator adaptation is disabled', () => {
      it('returns the original pool unchanged', () => {
        // Arrange
        const pool = [createMutationMethod('ADD_NODE')];
        const selectionController = createSelectionController({
          operatorAdaptation: { enabled: false },
        });

        // Act
        const adaptedPool = applyOperatorAdaptationForSelect(
          pool,
          selectionController,
        );

        // Assert
        expect(adaptedPool).toBe(pool);
      });
    });

    describe('given one operator with enough successful history to earn a boost', () => {
      it('duplicates that operator in the candidate pool according to the configured boost factor', () => {
        // Arrange
        const addNodeMethod = createMutationMethod('ADD_NODE');
        const subtractConnectionMethod = createMutationMethod('SUB_CONN');
        const selectionController = createSelectionController({
          operatorAdaptation: { enabled: true, boost: 3 },
          operatorStats: new Map([
            ['ADD_NODE', { success: 8, attempts: 10 }],
            ['SUB_CONN', { success: 1, attempts: 10 }],
          ]),
        });

        // Act
        const adaptedPool = applyOperatorAdaptationForSelect(
          [addNodeMethod, subtractConnectionMethod],
          selectionController,
        );

        // Assert
        expect(adaptedPool.map((method) => method.name)).toEqual([
          'ADD_NODE',
          'ADD_NODE',
          'ADD_NODE',
          'SUB_CONN',
        ]);
      });
    });

    describe('given operator adaptation is disabled', () => {
      it('returns the original pool unchanged', () => {
        // Arrange
        const pool = [createMutationMethod('ADD_NODE')];
        const selectionController = createSelectionController({
          operatorAdaptation: { enabled: false },
        });

        // Act
        const adaptedPool = applyOperatorAdaptationForSelect(
          pool,
          selectionController,
        );

        // Assert
        expect(adaptedPool).toBe(pool);
      });
    });

    describe('given operator adaptation uses the default boost and another operator has no stats record yet', () => {
      it('boosts only the tracked successful operator and leaves the untracked one unboosted', () => {
        // Arrange
        const addNodeMethod = createMutationMethod('ADD_NODE');
        const subtractConnectionMethod = createMutationMethod('SUB_CONN');
        const selectionController = createSelectionController({
          operatorAdaptation: { enabled: true },
          operatorStats: new Map([
            ['ADD_NODE', { success: 9, attempts: 10 }],
          ]),
        });

        // Act
        const adaptedPool = applyOperatorAdaptationForSelect(
          [addNodeMethod, subtractConnectionMethod],
          selectionController,
        );

        // Assert
        expect(adaptedPool.map((method) => method.name)).toEqual([
          'ADD_NODE',
          'ADD_NODE',
          'SUB_CONN',
        ]);
      });
    });
  });

  describe('applyOperatorBanditForSelect', () => {
    describe('given the operator bandit is disabled', () => {
      it('returns the fallback method unchanged', () => {
        // Arrange
        const fallbackMethod = createMutationMethod('ADD_NODE');
        const selectionController = createSelectionController({
          operatorBandit: { enabled: false },
        });

        // Act
        const selectedMethod = applyOperatorBanditForSelect(
          [fallbackMethod],
          fallbackMethod,
          selectionController,
        );

        // Assert
        expect(selectedMethod).toBe(fallbackMethod);
      });
    });

    describe('given a pool whose operators have no prior stats entries', () => {
      it('creates operator stats records for every candidate before scoring', () => {
        // Arrange
        const addNodeMethod = createMutationMethod('ADD_NODE');
        const subtractConnectionMethod = createMutationMethod('SUB_CONN');
        const selectionController = createSelectionController({
          operatorBandit: { enabled: true, c: 1.4, minAttempts: 1 },
        });

        // Act
        applyOperatorBanditForSelect(
          [addNodeMethod, subtractConnectionMethod],
          addNodeMethod,
          selectionController,
        );

        // Assert
        expect(Array.from(selectionController._operatorStats.keys())).toEqual([
          'ADD_NODE',
          'SUB_CONN',
        ]);
      });
    });

    describe('given the operator bandit is disabled', () => {
      it('returns the fallback method unchanged', () => {
        // Arrange
        const fallbackMethod = createMutationMethod('ADD_NODE');
        const selectionController = createSelectionController({
          operatorBandit: { enabled: false },
        });

        // Act
        const selectedMethod = applyOperatorBanditForSelect(
          [fallbackMethod],
          fallbackMethod,
          selectionController,
        );

        // Assert
        expect(selectedMethod).toBe(fallbackMethod);
      });
    });

    describe('given one operator below the minimum-attempt threshold', () => {
      it('chooses the under-sampled operator because exploration outranks the fallback score', () => {
        // Arrange
        const addNodeMethod = createMutationMethod('ADD_NODE');
        const subtractConnectionMethod = createMutationMethod('SUB_CONN');
        const selectionController = createSelectionController({
          operatorBandit: { enabled: true, c: 1.4, minAttempts: 3 },
          operatorStats: new Map([
            ['ADD_NODE', { success: 6, attempts: 6 }],
            ['SUB_CONN', { success: 0, attempts: 1 }],
          ]),
        });

        // Act
        const selectedMethod = applyOperatorBanditForSelect(
          [addNodeMethod, subtractConnectionMethod],
          addNodeMethod,
          selectionController,
        );

        // Assert
        expect(selectedMethod.name).toBe('SUB_CONN');
      });
    });

    describe('given the bandit uses its default exploration settings', () => {
      it('prefers the higher-scoring operator once both candidates exceed the default minimum attempts', () => {
        // Arrange
        const addNodeMethod = createMutationMethod('ADD_NODE');
        const subtractConnectionMethod = createMutationMethod('SUB_CONN');
        const selectionController = createSelectionController({
          operatorBandit: { enabled: true },
          operatorStats: new Map([
            ['ADD_NODE', { success: 6, attempts: 6 }],
            ['SUB_CONN', { success: 0, attempts: 6 }],
          ]),
        });

        // Act
        const selectedMethod = applyOperatorBanditForSelect(
          [addNodeMethod, subtractConnectionMethod],
          subtractConnectionMethod,
          selectionController,
        );

        // Assert
        expect(selectedMethod.name).toBe('ADD_NODE');
      });
    });
  });

  describe('applyPhasedComplexityForSelect', () => {
    describe('given phased complexity has no active phase yet', () => {
      it('returns the original pool unchanged', () => {
        // Arrange
        const pool = [createMutationMethod('ADD_NODE')];
        const selectionController = createSelectionController({
          phase: undefined,
          phasedComplexity: { enabled: true },
        });

        // Act
        const phasedPool = applyPhasedComplexityForSelect(
          pool,
          selectionController,
        );

        // Assert
        expect(phasedPool).toBe(pool);
      });
    });

    describe('given a simplify-phase pool that still contains an undefined entry after normalization', () => {
      it('drops the undefined entry and duplicates subtractive operators only', () => {
        // Arrange
        const addNodeMethod = createMutationMethod('ADD_NODE');
        const subtractConnectionMethod = createMutationMethod('SUB_CONN');
        const selectionController = createSelectionController({
          mutationPool: [addNodeMethod, undefined, subtractConnectionMethod],
          phase: 'simplify',
          phasedComplexity: { enabled: true },
          operatorBandit: { enabled: false },
        });
        const normalizedPool = normalizeMutationPoolForSelect(
          selectionController,
          { mutation: {} },
          false,
        );

        // Act
        const phasedPool = applyPhasedComplexityForSelect(
          normalizedPool,
          selectionController,
        );

        // Assert
        expect(phasedPool.map((method) => method.name)).toEqual([
          'ADD_NODE',
          'SUB_CONN',
          'SUB_CONN',
        ]);
      });
    });

    describe('given phased complexity has no active phase yet', () => {
      it('returns the original pool unchanged', () => {
        // Arrange
        const pool = [createMutationMethod('ADD_NODE')];
        const selectionController = createSelectionController({
          phase: undefined,
          phasedComplexity: { enabled: true },
        });

        // Act
        const phasedPool = applyPhasedComplexityForSelect(
          pool,
          selectionController,
        );

        // Assert
        expect(phasedPool).toBe(pool);
      });
    });

    describe('given a complexify-phase pool that still contains an undefined entry after normalization', () => {
      it('drops the undefined entry and duplicates additive operators only', () => {
        // Arrange
        const addNodeMethod = createMutationMethod('ADD_NODE');
        const subtractConnectionMethod = createMutationMethod('SUB_CONN');
        const selectionController = createSelectionController({
          mutationPool: [addNodeMethod, undefined, subtractConnectionMethod],
          phase: 'complexify',
          phasedComplexity: { enabled: true },
          operatorBandit: { enabled: false },
        });
        const normalizedPool = normalizeMutationPoolForSelect(
          selectionController,
          { mutation: {} },
          false,
        );

        // Act
        const phasedPool = applyPhasedComplexityForSelect(
          normalizedPool,
          selectionController,
        );

        // Assert
        expect(phasedPool.map((method) => method.name)).toEqual([
          'ADD_NODE',
          'SUB_CONN',
          'ADD_NODE',
        ]);
      });
    });

    describe('given a simplify-phase pool without subtractive operators', () => {
      it('returns the filtered pool without adding extra operators', () => {
        // Arrange
        const selectionController = createSelectionController({
          mutationPool: [createMutationMethod('ADD_NODE'), undefined],
          phase: 'simplify',
          phasedComplexity: { enabled: true },
        });
        const normalizedPool = normalizeMutationPoolForSelect(
          selectionController,
          methods,
          false,
        );

        // Act
        const phasedPool = applyPhasedComplexityForSelect(
          normalizedPool,
          selectionController,
        );

        // Assert
        expect(phasedPool.map((method) => method.name)).toEqual([
          'ADD_NODE',
        ]);
      });
    });

    describe('given a complexify-phase pool without additive operators', () => {
      it('returns the filtered pool without adding extra operators', () => {
        // Arrange
        const selectionController = createSelectionController({
          mutationPool: [createMutationMethod('SUB_CONN'), undefined],
          phase: 'complexify',
          phasedComplexity: { enabled: true },
        });
        const normalizedPool = normalizeMutationPoolForSelect(
          selectionController,
          methods,
          false,
        );

        // Act
        const phasedPool = applyPhasedComplexityForSelect(
          normalizedPool,
          selectionController,
        );

        // Assert
        expect(phasedPool.map((method) => method.name)).toEqual([
          'SUB_CONN',
        ]);
      });
    });

    describe('given phased complexity uses one non-mutating phase label', () => {
      it('returns the filtered pool unchanged', () => {
        // Arrange
        const addNodeMethod = createMutationMethod('ADD_NODE');
        const selectionController = createSelectionController({
          mutationPool: [addNodeMethod, undefined],
          phase: 'stabilize' as unknown as NeatControllerForMutation['_phase'],
          phasedComplexity: { enabled: true },
        });
        const normalizedPool = normalizeMutationPoolForSelect(
          selectionController,
          methods,
          false,
        );

        // Act
        const phasedPool = applyPhasedComplexityForSelect(
          normalizedPool,
          selectionController,
        );

        // Assert
        expect(phasedPool).toEqual([addNodeMethod]);
      });
    });
  });

  describe('isBlockedByStructuralLimitsForSelect', () => {
    describe('given ADD_NODE reaches the configured node ceiling', () => {
      it('blocks the operator before selection returns it', () => {
        // Arrange
        const selectionController = createSelectionController({ maxNodes: 1 });
        const genome = createStructuralGenome({ nodeCount: 1 });

        // Act
        const isBlocked = isBlockedByStructuralLimitsForSelect(
          methods.mutation.ADD_NODE as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(true);
      });
    });

    describe('given ADD_CONN reaches the configured connection ceiling', () => {
      it('blocks the operator before selection returns it', () => {
        // Arrange
        const selectionController = createSelectionController({ maxConns: 1 });
        const genome = createStructuralGenome({ connectionCount: 1 });

        // Act
        const isBlocked = isBlockedByStructuralLimitsForSelect(
          methods.mutation.ADD_CONN as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(true);
      });
    });

    describe('given ADD_GATE reaches the configured gate ceiling', () => {
      it('blocks the operator before selection returns it', () => {
        // Arrange
        const selectionController = createSelectionController({ maxGates: 1 });
        const genome = createStructuralGenome({ gateCount: 1 });

        // Act
        const isBlocked = isBlockedByStructuralLimitsForSelect(
          methods.mutation.ADD_GATE as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(true);
      });
    });

    describe('given ADD_GATE stays below the configured gate ceiling', () => {
      it('allows the operator to remain eligible', () => {
        // Arrange
        const selectionController = createSelectionController({ maxGates: 2 });
        const genome = createStructuralGenome({ gateCount: 1 });

        // Act
        const isBlocked = isBlockedByStructuralLimitsForSelect(
          methods.mutation.ADD_GATE as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(false);
      });
    });

    describe('given ADD_NODE has no configured ceiling', () => {
      it('falls back to the unbounded default and allows the operator', () => {
        // Arrange
        const selectionController = createSelectionController({});
        const genome = createStructuralGenome({ nodeCount: 999 });

        // Act
        const isBlocked = isBlockedByStructuralLimitsForSelect(
          methods.mutation.ADD_NODE as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(false);
      });
    });

    describe('given ADD_GATE has no configured ceiling', () => {
      it('falls back to the unbounded default and allows the operator', () => {
        // Arrange
        const selectionController = createSelectionController({});
        const genome = createStructuralGenome({ gateCount: 999 });

        // Act
        const isBlocked = isBlockedByStructuralLimitsForSelect(
          methods.mutation.ADD_GATE as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(false);
      });
    });

    describe('given ADD_NODE stays below the configured node ceiling', () => {
      it('allows the operator to remain eligible', () => {
        // Arrange
        const selectionController = createSelectionController({ maxNodes: 2 });
        const genome = createStructuralGenome({ nodeCount: 1 });

        // Act
        const isBlocked = isBlockedByStructuralLimitsForSelect(
          methods.mutation.ADD_NODE as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(false);
      });
    });

    describe('given ADD_NODE has no configured ceiling', () => {
      it('falls back to the unbounded default and allows the operator', () => {
        // Arrange
        const selectionController = createSelectionController({});
        const genome = createStructuralGenome({ nodeCount: 999 });

        // Act
        const isBlocked = isBlockedByStructuralLimitsForSelect(
          methods.mutation.ADD_NODE as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(false);
      });
    });

    describe('given ADD_CONN stays below the configured connection ceiling', () => {
      it('allows the operator to remain eligible', () => {
        // Arrange
        const selectionController = createSelectionController({ maxConns: 2 });
        const genome = createStructuralGenome({ connectionCount: 1 });

        // Act
        const isBlocked = isBlockedByStructuralLimitsForSelect(
          methods.mutation.ADD_CONN as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(false);
      });
    });

    describe('given ADD_CONN has no configured ceiling', () => {
      it('falls back to the unbounded default and allows the operator', () => {
        // Arrange
        const selectionController = createSelectionController({});
        const genome = createStructuralGenome({ connectionCount: 999 });

        // Act
        const isBlocked = isBlockedByStructuralLimitsForSelect(
          methods.mutation.ADD_CONN as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(false);
      });
    });
  });

  describe('isBlockedByRecurrentPolicyForSelect', () => {
    describe('given recurrent mutation is enabled globally but the genome stays feed-forward', () => {
      it('blocks ADD_BACK_CONN so the genome contract remains acyclic', () => {
        // Arrange
        const selectionController = createSelectionController({
          allowRecurrent: true,
        });
        const genome = createStructuralGenome({
          topologyIntent: 'feed-forward',
          enforceAcyclic: true,
        });

        // Act
        const isBlocked = isBlockedByRecurrentPolicyForSelect(
          methods.mutation.ADD_BACK_CONN as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(true);
      });
    });

    describe('given recurrent mutation is enabled and the genome is unconstrained', () => {
      it('allows ADD_SELF_CONN to proceed', () => {
        // Arrange
        const selectionController = createSelectionController({
          allowRecurrent: true,
        });
        const genome = createStructuralGenome({
          topologyIntent: 'unconstrained',
          enforceAcyclic: false,
        });

        // Act
        const isBlocked = isBlockedByRecurrentPolicyForSelect(
          methods.mutation.ADD_SELF_CONN as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(false);
      });
    });

    describe('given only the legacy acyclic flag is present on the genome', () => {
      it('still blocks recurrent operators conservatively', () => {
        // Arrange
        const selectionController = createSelectionController({
          allowRecurrent: true,
        });
        const genome = createStructuralGenome({
          enforceAcyclic: true,
        });

        // Act
        const isBlocked = isBlockedByRecurrentPolicyForSelect(
          methods.mutation.ADD_BACK_CONN as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(true);
      });
    });

    describe('given the mutation method is not a recurrent operator', () => {
      it('returns false immediately', () => {
        // Arrange
        const selectionController = createSelectionController({
          allowRecurrent: false,
        });
        const genome = createStructuralGenome({
          topologyIntent: 'feed-forward',
          enforceAcyclic: true,
        });

        // Act
        const isBlocked = isBlockedByRecurrentPolicyForSelect(
          methods.mutation.ADD_CONN as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(false);
      });
    });

    describe('given the mutation method is not a recurrent operator', () => {
      it('returns false immediately', () => {
        // Arrange
        const selectionController = createSelectionController({
          allowRecurrent: false,
        });
        const genome = createStructuralGenome({
          topologyIntent: 'feed-forward',
          enforceAcyclic: true,
        });

        // Act
        const isBlocked = isBlockedByRecurrentPolicyForSelect(
          methods.mutation.ADD_CONN as unknown as MutationMethod,
          genome,
          selectionController,
          methods,
        );

        // Assert
        expect(isBlocked).toBe(false);
      });
    });
  });

  describe('sampleFromPoolForSelect', () => {
    describe('given the candidate pool is empty', () => {
      it('returns null', () => {
        // Arrange
        const selectionController = createSelectionController({ randomValue: 0 });

        // Act
        const sampledMethod = sampleFromPoolForSelect([], selectionController);

        // Assert
        expect(sampledMethod).toBeNull();
      });
    });

    describe('given the candidate pool contains one deterministic method', () => {
      it('returns that method unchanged', () => {
        // Arrange
        const mutationMethod = methods.mutation
          .MOD_WEIGHT as unknown as MutationMethod;
        const selectionController = createSelectionController({
          mutationPool: [mutationMethod],
          randomValue: 0,
        });

        // Act
        const sampledMethod = sampleFromPoolForSelect(
          [mutationMethod],
          selectionController,
        );

        // Assert
        expect(sampledMethod).toBe(mutationMethod);
      });
    });

    describe('given the sampled index falls outside the pool bounds', () => {
      it('returns null', () => {
        // Arrange
        const mutationMethod = createMutationMethod('ADD_NODE');
        const selectionController = createSelectionController({
          randomValue: 1,
        });

        // Act
        const sampledMethod = sampleFromPoolForSelect(
          [mutationMethod],
          selectionController,
        );

        // Assert
        expect(sampledMethod).toBeNull();
      });
    });
  });

  describe('isOperatorNamePrefixedForSelect', () => {
    describe('given the mutation method has no public name', () => {
      it('returns false', () => {
        // Arrange
        const namelessMethod = {} as MutationMethod;

        // Act
        const hasPrefix = isOperatorNamePrefixedForSelect(
          namelessMethod,
          'ADD_',
        );

        // Assert
        expect(hasPrefix).toBe(false);
      });
    });
  });
});
