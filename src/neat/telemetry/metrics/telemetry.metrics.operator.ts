import type { OperatorStatsMap } from '../types/telemetry.types';

/**
 * Snapshot operator statistics into a telemetry-friendly array.
 *
 * @param operatorStats - Operator stats map (opName -> success/attempts).
 * @returns Operator stats snapshot array.
 */
export function computeOperatorStatsSnapshot(
  operatorStats: OperatorStatsMap | undefined,
): Array<{ op: string; succ: number; att: number }> {
  // Step 1: Convert operator stats map into a list.
  return Array.from((operatorStats ?? new Map()).entries()).map(
    ([operationName, stats]) => ({
      op: operationName,
      succ: stats.success,
      att: stats.attempts,
    }),
  );
}

/**
 * Convert operator stats map into the public accessor shape.
 *
 * @param operatorStats - Operator stats map stored on the host.
 * @returns Public operator summaries for dashboards and tests.
 */
export function readOperatorStats(
  operatorStats: OperatorStatsMap | undefined,
): { name: string; success: number; attempts: number }[] {
  return Array.from((operatorStats ?? new Map()).entries()).map(
    ([operationName, stats]) => ({
      name: operationName,
      success: stats.success,
      attempts: stats.attempts,
    }),
  );
}
