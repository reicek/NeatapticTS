import * as methods from '../../../methods/methods';
import {
  applyOperatorAdaptationForSelect,
  applyOperatorBanditForSelect,
  applyPhasedComplexityForSelect,
  isBlockedByStructuralLimitsForSelect,
  normalizeMutationPoolForSelect,
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

function createSelectionController(input: {
  operatorStats?: Map<string, { success: number; attempts: number }>;
  operatorAdaptation?: NeatControllerForMutation['options']['operatorAdaptation'];
  operatorBandit?: NeatControllerForMutation['options']['operatorBandit'];
  phasedComplexity?: NeatControllerForMutation['options']['phasedComplexity'];
  mutationPool?: Array<MutationMethod | undefined>;
  maxNodes?: number;
  maxConns?: number;
  maxGates?: number;
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
    },
    _getRNG: () => () => input.randomValue ?? 0,
    selectMutationMethod: async () => null,
    _mutateAddNodeReuse: async () => undefined,
    _mutateAddConnReuse: () => undefined,
    _invalidateGenomeCaches: () => undefined,
    _operatorStats: input.operatorStats ?? new Map(),
    _nodeSplitInnovations: new Map(),
    _connInnovations: new Map(),
    _nextGlobalInnovation: 1,
    _phase: input.phase,
  };
}

function createStructuralGenome(input: {
  nodeCount?: number;
  connectionCount?: number;
  gateCount?: number;
}): GenomeWithMetadata {
  return {
    nodes: Array.from({ length: input.nodeCount ?? 0 }, () => ({})),
    connections: Array.from({ length: input.connectionCount ?? 0 }, () => ({})),
    gates: Array.from({ length: input.gateCount ?? 0 }, () => ({})),
  } as unknown as GenomeWithMetadata;
}

describe('neat mutation selection chapter', () => {
  describe('applyOperatorAdaptationForSelect', () => {
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
  });

  describe('applyOperatorBanditForSelect', () => {
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
  });

  describe('applyPhasedComplexityForSelect', () => {
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
  });

  describe('sampleFromPoolForSelect', () => {
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
  });
});
