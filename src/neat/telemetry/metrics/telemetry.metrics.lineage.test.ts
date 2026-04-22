import {
  applyLineageStatsMonoObjective,
  applyLineageStatsMultiObjective,
  buildLineageContext,
  buildLineageEntry,
  collectDepths,
  computeAncestorUniquenessSampled,
  computeLineageStats,
  computeMeanDepth,
  computePairJaccardDistance,
  countAncestorIntersection,
  isLineageEligible,
  pickDistinctPairIndices,
} from './telemetry.metrics.lineage';
import { buildAnc, computeAncestorUniqueness } from '../../lineage/lineage';
import type { GenomeDetailed } from '../../shared/neat.shared.types';
import type { TelemetryEntryRecord } from '../types/telemetry.types';
import type { TelemetryGenome } from '../types/telemetry.types';

jest.mock('../../lineage/lineage', () => {
  const actualModule = jest.requireActual('../../lineage/lineage');

  return {
    ...actualModule,
    buildAnc: jest.fn(),
    computeAncestorUniqueness: jest.fn(),
  };
});

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

const mockedBuildAnc = jest.mocked(buildAnc);
const mockedComputeAncestorUniqueness = jest.mocked(computeAncestorUniqueness);
type LineageTelemetryContext = {
  _lineageEnabled?: boolean;
  _getRNG?: () => () => number;
  _lastMeanDepth?: number;
  _prevInbreedingCount?: number;
};

describe('neat telemetry lineage metrics chapter', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

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

      it('falls back to empty parents, zero depth, and zero inbreeding when lineage fields are absent', () => {
        // Arrange
        const bestGenomeSnapshot = {} as GenomeDetailed;

        // Act
        const lineageEntry = buildLineageEntry(
          {},
          bestGenomeSnapshot,
          2.345,
          0.125,
        );

        // Assert
        expect(lineageEntry).toEqual({
          parents: [],
          depthBest: 0,
          meanDepth: 2.35,
          inbreeding: 0,
          ancestorUniq: 0.125,
        });
      });
    });
  });

  describe('computeLineageStats', () => {
    describe('given lineage telemetry is disabled', () => {
      it('returns zeroed lineage stats without sampling pairs', () => {
        // Arrange
        const genomes = [{ _depth: 2 }] as TelemetryGenome[];

        // Act
        const lineageStats = computeLineageStats(
          false,
          genomes,
          genomes.length,
          1,
          createPairRngFactory(0, 0),
        );

        // Assert
        expect(lineageStats).toEqual({
          lineageMeanDepth: 0,
          lineageMeanPairDist: 0,
        });
      });
    });

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

      it('defaults missing depths to zero and keeps pair distance at zero when no pairs are sampled', () => {
        // Arrange
        const genomes = [{}, { _depth: 1 }] as TelemetryGenome[];

        // Act
        const lineageStats = computeLineageStats(
          true,
          genomes,
          genomes.length,
          0,
          createPairRngFactory(0, 0),
        );

        // Assert
        expect(lineageStats).toEqual({
          lineageMeanDepth: 0.5,
          lineageMeanPairDist: 0,
        });
      });

      it('adjusts the second sampled index when the rng picks the same genome twice', () => {
        // Arrange
        const genomes = [{ _depth: 0 }, { _depth: 2 }] as TelemetryGenome[];

        // Act
        const lineageStats = computeLineageStats(
          true,
          genomes,
          genomes.length,
          1,
          createPairRngFactory(0, 0),
        );

        // Assert
        expect(lineageStats).toEqual({
          lineageMeanDepth: 1,
          lineageMeanPairDist: 2,
        });
      });
    });
  });

  describe('applyLineageStatsMultiObjective', () => {
    describe('given lineage telemetry is disabled', () => {
      it('leaves the entry unchanged', () => {
        // Arrange
        const entry = {} as TelemetryEntryRecord;

        // Act
        applyLineageStatsMultiObjective(
          { _lineageEnabled: false },
          [{ _depth: 1 } as GenomeDetailed],
          entry,
        );

        // Assert
        expect(entry).toEqual({});
      });
    });

    describe('given lineage telemetry is enabled with explicit lineage values', () => {
      it('records the best-genome lineage snapshot and preserves provided rng and inbreeding data', () => {
        // Arrange
        let receivedRngFactory: (() => () => number) | undefined;
        mockedComputeAncestorUniqueness.mockImplementation(function (this: {
          _getRNG?: () => () => number;
        }) {
          receivedRngFactory = this._getRNG;
          return 0.456;
        });
        const entry = {} as TelemetryEntryRecord;
        const rngFactory = createPairRngFactory(0.25, 0.75);
        const telemetryContext: LineageTelemetryContext = {
          _lineageEnabled: true,
          _getRNG: rngFactory,
          _prevInbreedingCount: 3,
        };

        // Act
        applyLineageStatsMultiObjective(
          telemetryContext,
          [
            { _parents: [10, 11], _depth: 2 } as GenomeDetailed,
            { _depth: 4 } as GenomeDetailed,
          ],
          entry,
        );

        // Assert
        expect({
          lineage: entry.lineage,
          meanDepth: telemetryContext._lastMeanDepth,
          helperCalls: mockedComputeAncestorUniqueness.mock.calls.length,
          sameRngFactory: receivedRngFactory === rngFactory,
        }).toEqual({
          lineage: {
            parents: [10, 11],
            depthBest: 2,
            meanDepth: 3,
            inbreeding: 3,
            ancestorUniq: 0.456,
          },
          meanDepth: 3,
          helperCalls: 1,
          sameRngFactory: true,
        });
      });

      it('falls back to empty parent and depth defaults when the best genome omits lineage fields', () => {
        // Arrange
        mockedComputeAncestorUniqueness.mockImplementation(function (this: {
          _getRNG?: () => () => number;
        }) {
          this._getRNG?.()();
          return 0.111;
        });
        const entry = {} as TelemetryEntryRecord;
        const telemetryContext: LineageTelemetryContext = {
          _lineageEnabled: true,
        };

        // Act
        applyLineageStatsMultiObjective(
          telemetryContext,
          [{}, { _depth: 2 }] as GenomeDetailed[],
          entry,
        );

        // Assert
        expect({
          lineage: entry.lineage,
          meanDepth: telemetryContext._lastMeanDepth,
        }).toEqual({
          lineage: {
            parents: [],
            depthBest: 0,
            meanDepth: 1,
            inbreeding: 0,
            ancestorUniq: 0.111,
          },
          meanDepth: 1,
        });
      });
    });
  });

  describe('applyLineageStatsMonoObjective', () => {
    describe('given lineage telemetry is enabled for a sampled population snapshot', () => {
      it('builds a lineage entry from sampled ancestor uniqueness', () => {
        // Arrange
        mockedBuildAnc
          .mockReturnValueOnce(new Set([1, 2]))
          .mockReturnValueOnce(new Set([2, 3]));
        const entry = {} as TelemetryEntryRecord;
        const telemetryContext: LineageTelemetryContext = {
          _lineageEnabled: true,
          _getRNG: createPairRngFactory(0, 0.99),
          _prevInbreedingCount: 2,
        };

        // Act
        applyLineageStatsMonoObjective(
          telemetryContext,
          [
            { _parents: [4, 5], _depth: 1 } as GenomeDetailed,
            { _parents: [5, 6], _depth: 3 } as GenomeDetailed,
          ],
          entry,
        );

        // Assert
        expect({
          lineage: entry.lineage,
          meanDepth: telemetryContext._lastMeanDepth,
        }).toEqual({
          lineage: {
            parents: [4, 5],
            depthBest: 1,
            meanDepth: 2,
            inbreeding: 2,
            ancestorUniq: 0.667,
          },
          meanDepth: 2,
        });
      });

      it('leaves the entry unchanged when lineage sampling is not eligible', () => {
        // Arrange
        const entry = {} as TelemetryEntryRecord;

        // Act
        applyLineageStatsMonoObjective(
          { _lineageEnabled: false },
          [{ _depth: 1 } as GenomeDetailed],
          entry,
        );

        // Assert
        expect(entry).toEqual({});
      });
    });
  });

  describe('isLineageEligible', () => {
    describe('given lineage is disabled for a non-empty population', () => {
      it('reports that lineage metrics should not run', () => {
        // Arrange
        const populationSnapshot = [{ _depth: 1 }] as GenomeDetailed[];

        // Act
        const isEligible = isLineageEligible(
          { _lineageEnabled: false },
          populationSnapshot,
        );

        // Assert
        expect(isEligible).toBe(false);
      });
    });
  });

  describe('collectDepths', () => {
    describe('given a population snapshot with missing depth values', () => {
      it('normalizes missing depths to zero', () => {
        // Arrange
        const populationSnapshot = [{ _depth: 2 }, {}] as GenomeDetailed[];

        // Act
        const depthValues = collectDepths(populationSnapshot);

        // Assert
        expect(depthValues).toEqual([2, 0]);
      });
    });
  });

  describe('computeMeanDepth', () => {
    describe('given an empty depth list', () => {
      it('falls back to zero instead of dividing by zero', () => {
        // Arrange
        const depthValues: number[] = [];

        // Act
        const meanDepth = computeMeanDepth(depthValues);

        // Assert
        expect(meanDepth).toBe(0);
      });
    });
  });

  describe('computeAncestorUniquenessSampled', () => {
    describe('given a single-genome population snapshot', () => {
      it('returns zero without sampling any ancestor pairs', () => {
        // Arrange
        const populationSnapshot = [{ _depth: 1 }] as GenomeDetailed[];

        // Act
        const ancestorUniqueness = computeAncestorUniquenessSampled(
          { _getRNG: createPairRngFactory(0, 0) },
          populationSnapshot,
        );

        // Assert
        expect(ancestorUniqueness).toBe(0);
      });
    });

    describe('given sampled ancestor sets are empty', () => {
      it('skips undefined Jaccard distances and returns zero', () => {
        // Arrange
        mockedBuildAnc.mockReturnValue(new Set<number>());
        const populationSnapshot = [{}, {}] as GenomeDetailed[];

        // Act
        const ancestorUniqueness = computeAncestorUniquenessSampled(
          { _getRNG: createPairRngFactory(0, 0.99) },
          populationSnapshot,
        );

        // Assert
        expect(ancestorUniqueness).toBe(0);
      });
    });
  });

  describe('pickDistinctPairIndices', () => {
    describe('given no rng factory is provided and Math.random repeats the same value', () => {
      it('falls back to Math.random and advances the second index', () => {
        // Arrange
        const mathRandomSpy = jest
          .spyOn(Math, 'random')
          .mockReturnValueOnce(0.1)
          .mockReturnValueOnce(0.1);

        // Act
        const pairIndices = pickDistinctPairIndices({}, 2);

        // Assert
        expect(pairIndices).toEqual({ firstIndex: 0, secondIndex: 1 });

        // Cleanup
        mathRandomSpy.mockRestore();
      });
    });
  });

  describe('computePairJaccardDistance', () => {
    describe('given both genomes have no discovered ancestors', () => {
      it('returns undefined instead of a distance', () => {
        // Arrange
        mockedBuildAnc.mockReturnValue(new Set<number>());
        const populationSnapshot = [{}, {}] as GenomeDetailed[];

        // Act
        const jaccardDistance = computePairJaccardDistance(
          {},
          populationSnapshot,
          0,
          1,
        );

        // Assert
        expect(jaccardDistance).toBeUndefined();
      });
    });

    describe('given ancestor sets overlap partially', () => {
      it('returns the Jaccard distance for the sampled pair', () => {
        // Arrange
        mockedBuildAnc
          .mockReturnValueOnce(new Set([1, 2]))
          .mockReturnValueOnce(new Set([2, 3]));
        const populationSnapshot = [{}, {}] as GenomeDetailed[];

        // Act
        const jaccardDistance = computePairJaccardDistance(
          { _getRNG: createPairRngFactory(0.2, 0.8) },
          populationSnapshot,
          0,
          1,
        );

        // Assert
        expect(jaccardDistance).toBeCloseTo(2 / 3, 10);
      });
    });
  });

  describe('buildLineageContext', () => {
    describe('given the telemetry context already exposes an rng factory', () => {
      it('preserves the existing rng factory on the lineage helper context', () => {
        // Arrange
        const populationSnapshot = [{ _depth: 1 }] as GenomeDetailed[];
        const rngFactory = createPairRngFactory(0.3, 0.7);

        // Act
        const lineageContext = buildLineageContext(
          { _getRNG: rngFactory },
          populationSnapshot,
        );

        // Assert
        expect({
          population: lineageContext.population,
          sameRngFactory: lineageContext._getRNG === rngFactory,
        }).toEqual({
          population: populationSnapshot,
          sameRngFactory: true,
        });
      });
    });

    describe('given no rng factory is present on the telemetry context', () => {
      it('provides the population snapshot and a Math.random-backed rng factory', () => {
        // Arrange
        const populationSnapshot = [{ _depth: 1 }] as GenomeDetailed[];
        const mathRandomSpy = jest.spyOn(Math, 'random').mockReturnValue(0.4);

        // Act
        const lineageContext = buildLineageContext({}, populationSnapshot);
        const sampledValue = lineageContext._getRNG?.()();

        // Assert
        expect({
          population: lineageContext.population,
          sampledValue,
        }).toEqual({
          population: populationSnapshot,
          sampledValue: 0.4,
        });

        // Cleanup
        mathRandomSpy.mockRestore();
      });
    });
  });

  describe('countAncestorIntersection', () => {
    describe('given one ancestor id is shared between two sets', () => {
      it('counts only the overlapping ancestor ids', () => {
        // Arrange
        const ancestorsA = new Set([1, 2]);
        const ancestorsB = new Set([2, 3]);

        // Act
        const intersectionCount = countAncestorIntersection(ancestorsA, ancestorsB);

        // Assert
        expect(intersectionCount).toBe(1);
      });
    });
  });
});
