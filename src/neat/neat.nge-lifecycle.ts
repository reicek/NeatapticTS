import type { EquilibriumCandidate } from './nge-adult/neat.nge-adult.types';
import { assimilateEquilibriumCandidate } from './nge-assimilation/neat.nge-assimilation';
import type {
  NgeAssimilationCandidate,
  NgeAssimilationPolicy,
  NgeAssimilationResult,
} from './nge-assimilation/neat.nge-assimilation.types';
import {
  computeFocusScores,
  planGrowthMorphs,
  resolveFocusConfig,
} from './nge-juvenile/neat.nge-juvenile';
import type {
  NgeFocusScore,
  NgeGrowthBudget,
  NgeHysteresisState,
  NgeJuvenilePhaseConfig,
  NgeModuleMetricsSnapshot,
  NgeMorphDelta,
} from './nge-juvenile/neat.nge-juvenile.types';

/**
 * Inputs for the juvenile stage of the NGE lifecycle runner.
 */
export interface NgeJuvenileLifecycleInput {
  /** Lifecycle stage selector. */
  stage: 'juvenile';
  /** Module whose juvenile focus and growth will be evaluated. */
  moduleId: string;
  /** Latest module metrics snapshot for the active evaluation window. */
  metrics: NgeModuleMetricsSnapshot;
  /** DNA-configured growth caps and current live counts for the module. */
  budget: NgeGrowthBudget;
  /** Juvenile phase configuration, partial or fully resolved. */
  config: Partial<NgeJuvenilePhaseConfig>;
  /** Persistent hysteresis state carried into the current window. */
  hysteresis: NgeHysteresisState;
  /** Optional pre-computed focus score supplied by the caller. */
  focusScore?: NgeFocusScore;
  /** Optional pre-computed dry-run morph deltas supplied by the caller. */
  deltas?: NgeMorphDelta[];
}

/**
 * Inputs for the adult stage of the NGE lifecycle runner, including an equilibrium
 * candidate and the assimilation envelope needed to write back structural priors.
 */
export interface NgeAdultLifecycleInput {
  /** Lifecycle stage selector. */
  stage: 'adult';
  /** Equilibrium event emitted by the adult phase that triggers assimilation. */
  equilibriumCandidate: EquilibriumCandidate;
  /** Structured assimilation candidate assembled from the adult boundary. */
  candidate: NgeAssimilationCandidate;
  /** Resolved policy controlling validation, budget handling, and encoding behavior. */
  policy: NgeAssimilationPolicy;
}

/**
 * Result emitted by one call to the NGE lifecycle staging runner.
 */
export interface NgeLifecycleResult {
  /** Lifecycle stage reached after this runner call. */
  stage: 'juvenile' | 'assimilation' | 'adult';
  /** Juvenile focus score and planned dry-run morph deltas, when the juvenile stage ran. */
  juvenileResult?: {
    /** Focus score used to drive juvenile growth planning. */
    focusScore: NgeFocusScore;
    /** Planned dry-run morph deltas in edge-first priority order. */
    deltas: NgeMorphDelta[];
  };
  /** Assimilation result produced when an adult equilibrium candidate is processed. */
  assimilationResult?: NgeAssimilationResult;
}

/**
 * Run one NGE lifecycle staging step, sequencing juvenile growth, equilibrium
 * assimilation, and adult cooling as the provided stage requires.
 *
 * - `juvenile` recomputes the focus vector and growth morph plan for the supplied
 *   module, then transitions to the adult stage.
 * - `adult` runs the assimilation pass for the supplied equilibrium candidate and
 *   policy, producing a structural-prior delta while remaining in the adult stage.
 *
 * @param input - Runtime inputs for the selected lifecycle stage.
 * @returns The lifecycle result naming the reached stage and any stage-specific outputs.
 *
 * @example
 * ```ts
 * const result = runNgeLifecycle({
 *   stage: 'juvenile',
 *   moduleId: 'module:alpha',
 *   metrics,
 *   budget,
 *   config,
 *   hysteresis,
 * });
 * console.log(result.stage); // 'adult'
 * ```
 */
export function runNgeLifecycle(
  input: NgeJuvenileLifecycleInput | NgeAdultLifecycleInput,
): NgeLifecycleResult {
  if (input.stage === 'juvenile') {
    // Step 1: Resolve the juvenile config and score module focus for the window.
    const resolvedConfig = resolveFocusConfig(input.config);
    const focusVector = computeFocusScores([input.metrics], resolvedConfig);
    const focusScore = focusVector.scores[0];

    // Step 2: Plan the dry-run growth morphs that the juvenile stage may apply.
    const deltas = planGrowthMorphs(
      input.moduleId,
      focusScore,
      input.metrics,
      input.budget,
      resolvedConfig,
      input.hysteresis,
    );

    // Step 3: Transition to the adult stage after one juvenile planning cycle.
    return {
      stage: 'adult',
      juvenileResult: { focusScore, deltas },
    };
  }

  // Step 4: In the adult stage, assimilate the equilibrium candidate into structural priors.
  const assimilationResult = assimilateEquilibriumCandidate(
    input.candidate,
    input.policy,
  );

  return {
    stage: 'adult',
    assimilationResult,
  };
}
