import { OPERATOR_DECAY_DEFAULT } from '../core/adaptive.core.constants';
import type { OperatorAdaptationConfig } from '../core/adaptive.core.types';

/**
 * Operator-stat decay helpers for adaptive mutation.
 *
 * These helpers maintain the slower-moving memory of which mutation operators
 * have succeeded recently. They stay separate from per-genome mutation tuning so
 * the generated chapter can distinguish genome-local pressure from controller-
 * level evidence decay.
 */

/* Module introduction boundary for generated README output. */

/**
 * Resolve the decay factor for operator statistics.
 *
 * Centralizing the default decay factor keeps the caller focused on the update
 * cycle instead of repeatedly restating configuration fallback rules.
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
 * Snapshotting the entries before update keeps the decay pass simple and makes
 * the generated docs show that the helper operates over a stable view of the
 * current operator table.
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
 * Operator decay is the controller-level counterpart to per-genome mutation
 * tuning. Rather than rewriting genomes directly, it softens stale wins and
 * attempts so later operator selection can weight recent performance more
 * heavily.
 *
 * @param stats - Operator statistics map.
 * @param entries - Operator stat entries to update.
 * @param decay - Decay factor.
 * @returns Nothing.
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
 * This helper keeps the decay rule explicit and uniform for both success and
 * attempt counts so the operator table preserves ratios while shrinking older
 * evidence.
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
