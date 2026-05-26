import * as telemetryMetrics from './telemetry.metrics';

describe('telemetry metrics compatibility barrel', () => {
  describe('when recorder-facing helpers are read through the barrel', () => {
    it('exposes every helper as a callable function', () => {
      const recorderFacingHelperNames = [
        'getCachedEntropy',
        'computeDegreeCounts',
        'buildDegreeHistogram',
        'computeEntropyFromHistogram',
        'setCachedEntropy',
        'getTelemetryCoreSnapshot',
        'stripUnselectedTelemetryKeys',
        'mergeTelemetryCoreFields',
        'safelyApplyTelemetrySelect',
        'applyFastModeDefaults',
        'computeCompatibilityStats',
        'computeEntropyStats',
        'computeGraphletEntropy',
        'pickDistinctIndices',
        'countEnabledEdges',
        'computeOperatorStatsSnapshot',
        'readOperatorStats',
        'computeHyperVolumeProxy',
        'computeParetoFrontSizes',
        'applyObjectiveImportance',
        'applyObjectiveAges',
        'applyObjectiveEvents',
        'applySpeciesAllocation',
        'applyObjectivesSnapshot',
        'applyHypervolumeTelemetry',
        'applyRngState',
        'computeLineageStats',
        'applyLineageStatsMultiObjective',
        'applyLineageStatsMonoObjective',
        'isLineageEligible',
        'collectDepths',
        'computeMeanDepth',
        'computeAncestorUniquenessSampled',
        'pickDistinctPairIndices',
        'computePairJaccardDistance',
        'buildLineageContext',
        'countAncestorIntersection',
        'buildLineageEntry',
        'collectPopulationCounts',
        'computeMeanCounts',
        'computeMaxCounts',
        'computeEnabledRatios',
        'computeMeanEnabledRatio',
        'computeAndStoreGrowthValues',
        'buildComplexityEntry',
        'applyComplexityStatsMultiObjective',
        'applyComplexityStatsMonoObjective',
        'applyPerformanceStats',
      ] as const;
      const resolvedHelpers = recorderFacingHelperNames.map(
        (helperName) => telemetryMetrics[helperName],
      );

      expect(
        resolvedHelpers.every(
          (candidateHelper) => typeof candidateHelper === 'function',
        ),
      ).toBe(true);
    });
  });
});
