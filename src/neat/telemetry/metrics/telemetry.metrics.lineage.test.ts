import {
  buildLineageEntry,
  computeLineageStats,
} from './telemetry.metrics.lineage';
import type { GenomeDetailed } from '../../shared/neat.shared.types';
import type { TelemetryGenome } from '../types/telemetry.types';

function createPairRngFactory(
  firstValue: number,
  secondValue: number,
): () => () => number {
  return () => {
    const randomValues = [firstValue, secondValue];
    let randomValueIndex = 0;

    return () => {
      const nextValue =
        randomValues[randomValueIndex] ?? randomValues.at(-1) ?? 0;
      randomValueIndex += 1;
      return nextValue;
    };
  };
}

describe('neat telemetry lineage metrics chapter', () => {
  describe('buildLineageEntry', () => {
    describe('given a best-genome snapshot and previous inbreeding count', () => {
      it('packages the lineage block with the rounded mean depth and inbreeding value', () => {
        // Arrange
        const bestGenomeSnapshot = {
          _parents: [7, 8],
          _depth: 3,
        } as GenomeDetailed;

        // Act
        const lineageEntry = buildLineageEntry(
          { _prevInbreedingCount: 4 },
          bestGenomeSnapshot,
          1.25,
          0.5,
        );

        // Assert
        expect(lineageEntry).toEqual({
          parents: [7, 8],
          depthBest: 3,
          meanDepth: 1.25,
          inbreeding: 4,
          ancestorUniq: 0.5,
        });
      });
    });
  });

  describe('computeLineageStats', () => {
    describe('given lineage-enabled telemetry with genomes at different ancestry depths', () => {
      it('reports the mean depth and sampled pairwise depth distance', () => {
        // Arrange
        const genomes = [
          { _depth: 0 },
          { _depth: 1 },
          { _depth: 2 },
        ] as TelemetryGenome[];

        // Act
        const lineageStats = computeLineageStats(
          true,
          genomes,
          genomes.length,
          1,
          createPairRngFactory(0, 0.99),
        );

        // Assert
        expect(lineageStats).toEqual({
          lineageMeanDepth: 1,
          lineageMeanPairDist: 2,
        });
      });
    });
  });
});
