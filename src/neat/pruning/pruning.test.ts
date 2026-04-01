import { applyAdaptivePruning, applyEvolutionPruning } from './pruning';
import type { NeatLikeForPruning } from './core/pruning.types';

type RecordedPruneCall = { sparsity: number; method: string | undefined };
type RecordedGenome = NeatLikeForPruning['population'][number] & {
  calls: RecordedPruneCall[];
};
type PruningTestHost = Omit<NeatLikeForPruning, 'population'> & {
  population: RecordedGenome[];
};

function createPruneRecorder() {
  const calls: RecordedPruneCall[] = [];

  return {
    calls,
    pruneToSparsity(sparsity: number, method?: string) {
      calls.push({ sparsity, method });
    },
  };
}

function createPruningHost(input: {
  generation?: number;
  evolutionPruning?: NeatLikeForPruning['options']['evolutionPruning'];
  adaptivePruning?: NeatLikeForPruning['options']['adaptivePruning'];
  nodeCounts?: number[];
  connectionCounts?: number[];
}): PruningTestHost {
  const nodeCounts = input.nodeCounts ?? [4, 4];
  const connectionCounts = input.connectionCounts ?? [10, 10];

  return {
    options: {
      evolutionPruning: input.evolutionPruning,
      adaptivePruning: input.adaptivePruning,
    },
    generation: input.generation ?? 0,
    population: nodeCounts.map((nodeCount, genomeIndex) => {
      const pruneRecorder = createPruneRecorder();

      return {
        nodes: Array.from({ length: nodeCount }, () => ({})),
        connections: Array.from(
          { length: connectionCounts[genomeIndex] ?? 0 },
          () => ({}),
        ),
        pruneToSparsity: pruneRecorder.pruneToSparsity,
        calls: pruneRecorder.calls,
      };
    }),
  };
}

describe('neat pruning chapter', () => {
  describe('applyEvolutionPruning', () => {
    describe('given a generation before scheduled pruning starts', () => {
      it('leaves compatible genomes unpruned', () => {
        // Arrange
        const pruningHost = createPruningHost({
          generation: 0,
          evolutionPruning: {
            startGeneration: 1,
            interval: 1,
            targetSparsity: 0.5,
          },
        });

        // Act
        applyEvolutionPruning.call(pruningHost);

        // Assert
        expect(pruningHost.population.map((genome) => genome.calls)).toEqual([
          [],
          [],
        ]);
      });
    });

    describe('given a scheduled pruning generation inside the ramp window', () => {
      it('fans the ramp-scaled sparsity target out to every compatible genome', () => {
        // Arrange
        const pruningHost = createPruningHost({
          generation: 2,
          evolutionPruning: {
            startGeneration: 0,
            interval: 1,
            targetSparsity: 0.5,
            rampGenerations: 4,
          },
        });

        // Act
        applyEvolutionPruning.call(pruningHost);

        // Assert
        expect(pruningHost.population.map((genome) => genome.calls)).toEqual([
          [{ sparsity: 0.25, method: 'magnitude' }],
          [{ sparsity: 0.25, method: 'magnitude' }],
        ]);
      });
    });
  });

  describe('applyAdaptivePruning', () => {
    describe('given a population whose observed metric exceeds the target remaining complexity', () => {
      it('raises the shared prune level and applies it across the compatible genomes', () => {
        // Arrange
        const pruningHost = createPruningHost({
          adaptivePruning: {
            enabled: true,
            metric: 'connections',
            targetSparsity: 0.4,
            adjustRate: 0.2,
            tolerance: 0,
          },
          connectionCounts: [10, 10],
        });

        // Act
        applyAdaptivePruning.call(pruningHost);

        // Assert
        expect({
          level: pruningHost._adaptivePruneLevel,
          appliedCalls: pruningHost.population.map((genome) => genome.calls),
        }).toEqual({
          level: 0.2,
          appliedCalls: [
            [{ sparsity: 0.2, method: 'magnitude' }],
            [{ sparsity: 0.2, method: 'magnitude' }],
          ],
        });
      });
    });
  });
});
