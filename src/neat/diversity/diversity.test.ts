import Network from '../../architecture/network/network';
import {
  buildEmptyDiversityStats,
  computeDiversityStats,
  MAX_COMPATIBILITY_SAMPLE,
  MAX_LINEAGE_PAIR_SAMPLE,
  structuralEntropy,
} from './diversity';
import {
  calculateStructuralEntropy,
  MAX_COMPATIBILITY_SAMPLE as CORE_MAX_COMPATIBILITY_SAMPLE,
  MAX_LINEAGE_PAIR_SAMPLE as CORE_MAX_LINEAGE_PAIR_SAMPLE,
} from './core/diversity.core';
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
  describe('structuralEntropy', () => {
    describe('given a network graph', () => {
      it('delegates to the core entropy helper for structural fingerprinting', () => {
        // Arrange
        const graph = new Network(2, 1);

        // Act
        const chapterEntropy = structuralEntropy(graph);
        const coreEntropy = calculateStructuralEntropy(graph);

        // Assert
        expect(chapterEntropy).toBe(coreEntropy);
      });
    });
  });

  describe('buildEmptyDiversityStats', () => {
    describe('given a population size fallback', () => {
      it('returns a zeroed snapshot with the provided population echoed', () => {
        // Arrange
        const populationSize = 7;

        // Act
        const emptyStats = buildEmptyDiversityStats(populationSize);

        // Assert
        expect(emptyStats).toEqual({
          lineageMeanDepth: 0,
          lineageMeanPairDist: 0,
          meanNodes: 0,
          meanConns: 0,
          nodeVar: 0,
          connVar: 0,
          meanCompat: 0,
          graphletEntropy: 0,
          population: 7,
        });
      });
    });
  });

  describe('public sample caps', () => {
    describe('given root diversity re-exports', () => {
      it('exposes the compatibility sample cap from core without drift', () => {
        // Arrange
        const compatibilityCapFromChapter = MAX_COMPATIBILITY_SAMPLE;

        // Act
        const compatibilityCapFromCore = CORE_MAX_COMPATIBILITY_SAMPLE;

        // Assert
        expect(compatibilityCapFromChapter).toBe(compatibilityCapFromCore);
      });

      it('exposes the lineage pair sample cap from core without drift', () => {
        // Arrange
        const lineageCapFromChapter = MAX_LINEAGE_PAIR_SAMPLE;

        // Act
        const lineageCapFromCore = CORE_MAX_LINEAGE_PAIR_SAMPLE;

        // Assert
        expect(lineageCapFromChapter).toBe(lineageCapFromCore);
      });
    });
  });

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
