import {
  getCachedDiversityStats,
  getPerformanceStatsSnapshot,
} from '../../accessors/telemetry.accessors';
import type { DiversityStats } from '../../../diversity/diversity';

/**
 * Narrow telemetry-facade host surface required by the runtime-metrics chapter.
 *
 * This chapter groups the two read paths that summarize recent runtime state:
 * coarse timing snapshots and the cached diversity view used by diagnostics and
 * telemetry dashboards.
 */
export interface TelemetryFacadeRuntimeHost {
  population: { length: number };
  _diversityStats?: DiversityStats;
  _lastEvalDuration?: number;
  _lastEvolveDuration?: number;
  _computeDiversityStats(): DiversityStats;
}

/**
 * Return coarse timing metrics for the last evaluation and evolution passes.
 *
 * Keeping this beside the diversity snapshot helper makes the runtime chapter a
 * compact place to inspect the latest controller-health signals without mixing
 * them with lineage, species, or archive reads.
 *
 * @param host - `Neat` instance tracking performance timings.
 * @returns Snapshot of the last evaluation and evolution durations.
 */
export function getPerformanceStats(host: TelemetryFacadeRuntimeHost) {
  return getPerformanceStatsSnapshot(host);
}

/**
 * Return cached diversity metrics, computing a fallback snapshot when needed.
 *
 * This keeps the public facade resilient: callers can always ask for diversity
 * stats even before a full metrics pass has run.
 *
 * @param host - `Neat` instance exposing cached diversity state.
 * @returns Diversity metrics for the current population.
 */
export function getDiversityStats(
  host: TelemetryFacadeRuntimeHost,
): DiversityStats {
  if (!host._diversityStats) {
    return host._computeDiversityStats();
  }

  return (
    getCachedDiversityStats(host) ??
    buildEmptyDiversityStats(host.population.length)
  );
}

function buildEmptyDiversityStats(populationSize: number): DiversityStats {
  return {
    lineageMeanDepth: 0,
    lineageMeanPairDist: 0,
    meanNodes: 0,
    meanConns: 0,
    nodeVar: 0,
    connVar: 0,
    meanCompat: 0,
    graphletEntropy: 0,
    population: populationSize,
  };
}
