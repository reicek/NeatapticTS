import type { OperatorStatsRecord } from '../../../shared/neat.shared.types';
import { readOperatorStats } from '../../metrics/telemetry.metrics.operator';

/**
 * Narrow telemetry-facade host surface required by the operator-stats chapter.
 *
 * This chapter isolates the mutation-operator summary read path so adaptive
 * scheduling diagnostics stay separate from novelty maintenance and archive
 * inspection in the root telemetry facade.
 */
export interface TelemetryFacadeOperatorStatsHost {
  _operatorStats: Map<string, OperatorStatsRecord>;
}

/**
 * Return aggregated mutation/operator statistics.
 *
 * Keeping this accessor in its own chapter makes the telemetry facade easier
 * to scan: archive helpers answer multi-objective questions, novelty helpers
 * manage behavior descriptors, and this chapter explains operator activity.
 *
 * @param host - `Neat` instance recording operator attempts and successes.
 * @returns Operator summaries suitable for dashboards and debugging.
 */
export function getOperatorStats(host: TelemetryFacadeOperatorStatsHost): {
  name: string;
  success: number;
  attempts: number;
}[] {
  return readOperatorStats(host._operatorStats);
}
