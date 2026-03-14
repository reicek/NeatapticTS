import { OPERATOR_DECAY_DEFAULT } from '../core/adaptive.core.constants';
import type { OperatorAdaptationConfig } from '../core/adaptive.core.types';

/**
 * Resolve the decay factor for operator statistics.
 *
 * @param config - Operator adaptation configuration.
 * @returns Decay factor for exponential smoothing.
 */
export function resolveOperatorDecay(config: OperatorAdaptationConfig): number {
  return config.decay ?? OPERATOR_DECAY_DEFAULT;
}

/**
 * Collect operator statistic entries for processing.
 *
 * @param stats - Operator statistics map.
 * @returns Array of operator stat entries.
 */
export function collectOperatorStatsEntries(
  stats: Map<string, { success: number; attempts: number }>,
): Array<[string, { success: number; attempts: number }]> {
  return Array.from(stats.entries());
}

/**
 * Apply exponential decay to each operator statistic entry.
 *
 * @param stats - Operator statistics map.
 * @param entries - Operator stat entries to update.
 * @param decay - Decay factor.
 * @returns {void}
 */
export function applyOperatorDecay(
  stats: Map<string, { success: number; attempts: number }>,
  entries: Array<[string, { success: number; attempts: number }]>,
  decay: number,
): void {
  for (const [operatorId, operatorStat] of entries) {
    const nextStat = decayOperatorStat(operatorStat, decay);
    stats.set(operatorId, nextStat);
  }
}

/**
 * Apply decay to a single operator statistic record.
 *
 * @param operatorStat - Operator statistic record.
 * @param decay - Decay factor.
 * @returns Decayed operator statistic record.
 */
export function decayOperatorStat(
  operatorStat: { success: number; attempts: number },
  decay: number,
): { success: number; attempts: number } {
  return {
    success: operatorStat.success * decay,
    attempts: operatorStat.attempts * decay,
  };
}