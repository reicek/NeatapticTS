import { computeDiversityStats } from './diversity';
import type { CompatComputer, GenomeWithMetrics } from './core/diversity.types';

function createGenomeWithMetrics(
  nodeCount: number,
  connectionCount: number,
  depth?: number,
): GenomeWithMetrics {
  return {
    nodes: Array.from({ length: nodeCount }, () => ({
      connections: { out: [] },
    })),
    connections: Array.from({ length: connectionCount }, () => ({})),
    _depth: depth,
  };
}

function createCompatibilityComputer(): CompatComputer {
  return {
    _compatibilityDistance(firstGenome, secondGenome) {
      return (
        Math.abs(firstGenome.nodes.length - secondGenome.nodes.length) +
        Math.abs(
          firstGenome.connections.length - secondGenome.connections.length,
        )
      );
    },
  };
}

describe('neat diversity chapter', () => {
  describe('computeDiversityStats', () => {
    describe('given an empty population', () => {
      it('returns undefined instead of a synthetic diversity report', () => {
        // Arrange
        const compatibilityComputer = createCompatibilityComputer();

        // Act
        const diversityStats = computeDiversityStats([], compatibilityComputer);

        // Assert
        expect(diversityStats).toBeUndefined();
      });
    });

    describe('given a small population with structural and lineage spread', () => {
      it('returns the expected bounded diversity summary fields and aggregates', () => {
        // Arrange
        const compatibilityComputer = createCompatibilityComputer();
        const population = [
          createGenomeWithMetrics(3, 2, 1),
          createGenomeWithMetrics(5, 4, 2),
          createGenomeWithMetrics(4, 3, 4),
        ];

        // Act
        const diversityStats = computeDiversityStats(
          population,
          compatibilityComputer,
        );

        // Assert
        expect(diversityStats).toEqual({
          lineageMeanDepth: 7 / 3,
          lineageMeanPairDist: 2,
          meanNodes: 4,
          meanConns: 3,
          nodeVar: 2 / 3,
          connVar: 2 / 3,
          meanCompat: 8 / 3,
          graphletEntropy: 0,
          population: 3,
        });
      });
    });
  });
});
