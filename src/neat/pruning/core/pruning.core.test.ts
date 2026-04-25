import type { NeatLikeForPruning } from './pruning.types';
import {
  applyAdaptivePruneLevelToPopulation,
  applyPruningToPopulation,
  computeRampFraction,
  initializeAdaptivePruningState,
  resolveActiveAdaptivePruningOptions,
  resolveActiveEvolutionPruningOptions,
} from './pruning.core';

type RecordedPruneCall = { method: string | undefined; sparsity: number };
type RecordedGenome = NeatLikeForPruning['population'][number] & {
  calls: RecordedPruneCall[];
};
type PruningCoreTestHost = Omit<NeatLikeForPruning, 'population'> & {
  population: RecordedGenome[];
};

function createRecordedGenome(input: {
  connectionCount?: number;
  nodeCount?: number;
  supportsPruning?: boolean;
}): RecordedGenome {
  const calls: RecordedPruneCall[] = [];
  const genome = {
    calls,
    connections: Array.from({ length: input.connectionCount ?? 0 }, () => ({})),
    nodes: Array.from({ length: input.nodeCount ?? 0 }, () => ({})),
  } as RecordedGenome;

  if (input.supportsPruning !== false) {
    genome.pruneToSparsity = (sparsity: number, method?: string) => {
      calls.push({ method, sparsity });
    };
  }

  return genome;
}

function createPruningCoreHost(input: {
  adaptivePruneLevel?: number;
  generation?: number;
  evolutionPruning?: NeatLikeForPruning['options']['evolutionPruning'];
  genomes?: RecordedGenome[];
} = {}): PruningCoreTestHost {
  return {
    _adaptivePruneLevel: input.adaptivePruneLevel,
    generation: input.generation ?? 0,
    options: {
      adaptivePruning: undefined,
      evolutionPruning: input.evolutionPruning,
    },
    population:
      input.genomes ??
      [
        createRecordedGenome({ connectionCount: 4, nodeCount: 2 }),
        createRecordedGenome({ connectionCount: 6, nodeCount: 3 }),
      ],
  };
}

describe('neat pruning core chapter', () => {
  describe('resolveActiveEvolutionPruningOptions', () => {
    describe('given scheduled pruning is not configured', () => {
      it('returns null', () => {
        // Arrange
        const pruningHost = createPruningCoreHost({
          evolutionPruning: undefined,
          generation: 4,
        });

        // Act
        const activeOptions = resolveActiveEvolutionPruningOptions(pruningHost);

        // Assert
        expect(activeOptions).toBeNull();
      });
    });

    describe('given the current generation falls outside the configured pruning interval', () => {
      it('returns null', () => {
        // Arrange
        const pruningHost = createPruningCoreHost({
          evolutionPruning: {
            interval: 3,
            startGeneration: 1,
            targetSparsity: 0.4,
          },
          generation: 2,
        });

        // Act
        const activeOptions = resolveActiveEvolutionPruningOptions(pruningHost);

        // Assert
        expect(activeOptions).toBeNull();
      });
    });
  });

  describe('resolveActiveAdaptivePruningOptions', () => {
    describe('given adaptive pruning is disabled', () => {
      it('returns null', () => {
        // Arrange
        const pruningHost = createPruningCoreHost();
        pruningHost.options.adaptivePruning = { enabled: false };

        // Act
        const activeOptions = resolveActiveAdaptivePruningOptions(pruningHost);

        // Assert
        expect(activeOptions).toBeNull();
      });
    });
  });

  describe('computeRampFraction', () => {
    describe('given scheduled pruning disables ramping', () => {
      it('returns the fully applied ramp fraction', () => {
        // Arrange
        const pruningHost = createPruningCoreHost({ generation: 4 });

        // Act
        const rampFraction = computeRampFraction(pruningHost, {
          rampGenerations: 0,
          startGeneration: 2,
        });

        // Assert
        expect(rampFraction).toBe(1);
      });
    });
  });

  describe('applyPruningToPopulation', () => {
    describe('given one genome does not implement pruning support', () => {
      it('skips the unsupported genome and applies the configured method to the supported one', () => {
        // Arrange
        const unsupportedGenome = createRecordedGenome({
          connectionCount: 4,
          nodeCount: 2,
          supportsPruning: false,
        });
        const supportedGenome = createRecordedGenome({
          connectionCount: 6,
          nodeCount: 3,
        });
        const pruningHost = createPruningCoreHost({
          genomes: [unsupportedGenome, supportedGenome],
        });

        // Act
        applyPruningToPopulation(
          pruningHost,
          { method: 'snip' },
          0.4,
        );

        // Assert
        expect(pruningHost.population.map((genome) => genome.calls)).toEqual([
          [],
          [{ method: 'snip', sparsity: 0.4 }],
        ]);
      });
    });
  });

  describe('initializeAdaptivePruningState', () => {
    describe('given the shared adaptive prune level already exists', () => {
      it('keeps the existing level unchanged', () => {
        // Arrange
        const pruningHost = createPruningCoreHost({ adaptivePruneLevel: 0.3 });

        // Act
        initializeAdaptivePruningState(pruningHost);

        // Assert
        expect(pruningHost._adaptivePruneLevel).toBe(0.3);
      });
    });
  });

  describe('applyAdaptivePruneLevelToPopulation', () => {
    describe('given one genome does not implement adaptive pruning support', () => {
      it('skips the unsupported genome and applies the shared magnitude prune level to the supported one', () => {
        // Arrange
        const unsupportedGenome = createRecordedGenome({
          connectionCount: 4,
          nodeCount: 2,
          supportsPruning: false,
        });
        const supportedGenome = createRecordedGenome({
          connectionCount: 6,
          nodeCount: 3,
        });
        const pruningHost = createPruningCoreHost({
          genomes: [unsupportedGenome, supportedGenome],
        });

        // Act
        applyAdaptivePruneLevelToPopulation(pruningHost, 0.25);

        // Assert
        expect(pruningHost.population.map((genome) => genome.calls)).toEqual([
          [],
          [{ method: 'magnitude', sparsity: 0.25 }],
        ]);
      });
    });
  });
});