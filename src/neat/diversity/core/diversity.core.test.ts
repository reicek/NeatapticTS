import { calculateDiversityStats } from './diversity.core';
import type { CompatComputer, GenomeWithMetrics } from './diversity.types';

function createSingleGenomeWithoutLineageDepth(): GenomeWithMetrics {
  return {
    nodes: [{ connections: { out: [{}] } }, { connections: { out: [] } }],
    connections: [{}],
  };
}

function createUnusedCompatibilityComputer(): CompatComputer {
  return {
    _compatibilityDistance() {
      throw new Error(
        'Expected no compatibility comparisons for a single sampled genome.',
      );
    },
  };
}

describe('diversity core chapter', () => {
  describe('calculateDiversityStats', () => {
    describe('given a single genome without lineage depth metadata', () => {
      it('returns zero-valued lineage and compatibility aggregates', () => {
        // Arrange
        const population = [createSingleGenomeWithoutLineageDepth()];
        const compatibilityComputer = createUnusedCompatibilityComputer();

        // Act
        const diversityStats = calculateDiversityStats(
          population,
          compatibilityComputer,
        );

        // Assert
        expect(diversityStats).toEqual({
          lineageMeanDepth: 0,
          lineageMeanPairDist: 0,
          meanNodes: 2,
          meanConns: 1,
          nodeVar: 0,
          connVar: 0,
          meanCompat: 0,
          graphletEntropy: 0,
          population: 1,
        });
      });
    });
  });
});
