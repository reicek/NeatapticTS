/**
 * Combat-pressure → reproduction-mode hysteresis policy.
 *
 * This module implements an external overlay that inspects the last three
 * generations of combat pressure and selects a reproduction mode by majority
 * vote. The selected mode is written back to the canonical
 * {@link NgeReproductionPolicy.mode} field only when the policy declares the
 * mode itself evolvable (`modeIsEvolvable: true`).
 *
 * The policy is deliberately distinct from the juvenile `NgeHysteresisState`
 * grow gate: it governs *which* reproduction operator the lineage uses, not
 * whether a juvenile module may grow.
 *
 * Mode mapping:
 * - dominating majority → `'parthenogenesis'`
 * - struggling majority → `'polyandric'`
 * - stalemate majority → `'sexual'`
 *
 * Ties are resolved deterministically by the priority order above.
 */
import type {
  NgeReproductionPolicy,
  NgeReproductionPolicyMode,
} from '../nge-dna/neat.nge-dna.types';

/**
 * One generation's combat-pressure signal.
 *
 * Exactly one of the three booleans should be true for a given generation,
 * but the counter tolerates mixed signals by counting each flag independently
 * and applying the majority vote.
 */
export interface ReproductionModePressureSignal {
  /** Generation number for telemetry and ordering. */
  generation: number;
  /** True when the lineage is dominating the enemy swarm this generation. */
  isDominating: boolean;
  /** True when the lineage is struggling against the enemy swarm this generation. */
  isStruggling: boolean;
  /** True when the engagement is a stalemate this generation. */
  isStalemate: boolean;
}

/**
 * Input consumed by {@link reproductionModeHysteresis}.
 */
export interface ReproductionModeHysteresisInput {
  /** Reproduction policy whose `mode` may be updated. */
  policy: NgeReproductionPolicy;
  /** Last-generation combat-pressure signals, oldest-to-newest. Only the final three are considered. */
  generationPressure: readonly ReproductionModePressureSignal[];
}

/**
 * Result returned by {@link reproductionModeHysteresis}.
 */
export interface ReproductionModeHysteresisResult {
  /** Selected reproduction mode from the 3-generation majority vote. */
  mode: NgeReproductionPolicyMode;
  /** Policy with `mode` updated only when `modeIsEvolvable` is true. */
  policy: NgeReproductionPolicy;
}

/**
 * Compute a 3-generation majority-vote reproduction mode from combat pressure.
 *
 * When `policy.modeIsEvolvable` is true, the selected mode is written back into
 * the returned policy; otherwise the policy is returned unchanged so a fixed
 * mode cannot be overwritten by the hysteresis overlay.
 *
 * If no generation pressure is supplied, the function falls back to
 * `'parthenogenesis'` as the conservative default.
 *
 * @param input - combat-pressure window and target policy
 * @returns selected mode and the possibly updated policy
 *
 * @example
 * ```ts
 * const result = reproductionModeHysteresis({
 *   policy: { mode: 'parthenogenesis', modeIsEvolvable: true, ... },
 *   generationPressure: [
 *     { generation: 1, isDominating: false, isStruggling: true, isStalemate: false },
 *     { generation: 2, isDominating: false, isStruggling: true, isStalemate: false },
 *     { generation: 3, isDominating: false, isStruggling: true, isStalemate: false },
 *   ],
 * });
 * console.log(result.mode); // 'polyandric'
 * ```
 */
export function reproductionModeHysteresis(
  input: ReproductionModeHysteresisInput,
): ReproductionModeHysteresisResult {
  const { policy, generationPressure } = input;
  const selectedMode = selectModeFromWindow(generationPressure);

  return policy.modeIsEvolvable
    ? { mode: selectedMode, policy: { ...policy, mode: selectedMode } }
    : { mode: selectedMode, policy };
}

/**
 * Select the reproduction mode implied by the last three pressure signals.
 *
 * Counts each flag independently across the trailing window and returns the
 * mode associated with the highest count. Ties resolve deterministically in
 * the order parthenogenesis → polyandric → sexual.
 *
 * @param generationPressure - pressure signals, oldest-to-newest
 * @returns selected reproduction mode
 */
function selectModeFromWindow(
  generationPressure: readonly ReproductionModePressureSignal[],
): NgeReproductionPolicyMode {
  if (generationPressure.length === 0) {
    return 'parthenogenesis';
  }

  const window = generationPressure.slice(-3);
  let dominatingCount = 0;
  let strugglingCount = 0;
  let stalemateCount = 0;

  for (const signal of window) {
    if (signal.isDominating) dominatingCount++;
    if (signal.isStruggling) strugglingCount++;
    if (signal.isStalemate) stalemateCount++;
  }

  const modeCounts = [
    { mode: 'parthenogenesis' as const, count: dominatingCount },
    { mode: 'polyandric' as const, count: strugglingCount },
    { mode: 'sexual' as const, count: stalemateCount },
  ];

  return modeCounts.reduce((best, current) =>
    current.count > best.count ? current : best,
  ).mode;
}
