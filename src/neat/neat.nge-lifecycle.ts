/**
 * NGE lifecycle staging runner.
 *
 * This module sequences one window of the Neuro-evolutionary Genesis Engine
 * (NGE) lifecycle. The `juvenile` stage scores module focus, plans dry-run
 * growth morphs, and optionally applies them to a live network. The `adult`
 * stage takes an equilibrium candidate and writes it back as structural priors
 * through the assimilation boundary.
 *
 * The runner is deliberately narrow: it expects the caller to supply metrics,
 * budgets, hysteresis, and an optional network. No `examples/` or demo
 * scaffolding is required; a headless test can drive the same growth engine
 * that a benchmark curriculum or collective application would use at runtime.
 *
 * ```mermaid
 * stateDiagram-v2
 *   [*] --> Juvenile : runNgeLifecycle({ stage: 'juvenile' })
 *   Juvenile --> Adult : focus scored, morphs planned or applied
 *   Adult --> Assimilation : equilibriumCandidate supplied
 *   Assimilation --> [*] : structural priors written
 * ```
 *
 * ## Determinism note
 *
 * When a `seed` is supplied with a live network, the runner seeds the network
 * RNG and pins the global connection innovation counter to the network's
 * current maximum innovation before any morph is applied. That makes the same
 * DNA + seed + experience stream reproducible at the level of topology and
 * innovation IDs. Omitting the seed leaves the engine non-deterministic but
 * does not affect classic NEAT when NGE is disabled.
 */

import Connection from '../architecture/connection/connection';
import type Network from '../architecture/network';
import type { EquilibriumCandidate } from './nge-adult/neat.nge-adult.types';
import { assimilateEquilibriumCandidate } from './nge-assimilation/neat.nge-assimilation';
import type {
  NgeAssimilationCandidate,
  NgeAssimilationPolicy,
  NgeAssimilationResult,
} from './nge-assimilation/neat.nge-assimilation.types';
import {
  applyMorphDeltas,
  commitGrowth,
  computeFocusScores,
  planGrowthMorphs,
  resolveFocusConfig,
} from './nge-juvenile/neat.nge-juvenile';
import type {
  MorphApplyBudget,
  MorphApplyOutcome,
} from './nge-juvenile/neat.nge-juvenile.apply';
import type { NgeGrowthMorphKind } from './nge-juvenile/neat.nge-juvenile.grow';
import type {
  NgeFocusScore,
  NgeGrowthBudget,
  NgeHysteresisState,
  NgeJuvenilePhaseConfig,
  NgeModuleMetricsSnapshot,
  NgeMorphDelta,
  NgePruneBudget,
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
  /** Optional live network to mutate when the apply phase should run. */
  network?: Network;
  /** DNA-configured prune floors required when the apply phase runs. */
  pruneBudget?: NgePruneBudget;
  /**
   * Optional deterministic seed that drives the network's random choices for this
   * lifecycle window. When supplied, the seed is applied to the live network before
   * morph application so repeated runs with the same seed and inputs reproduce the
   * same structural choices.
   */
  seed?: number;
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
  /** Apply outcomes from `applyMorphDeltas`, present when the apply phase ran. */
  applyOutcomes?: MorphApplyOutcome[];
  /** Updated hysteresis state after growth commit, present when the apply phase ran. */
  hysteresis?: NgeHysteresisState;
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
 *   network,
 *   pruneBudget,
 * });
 * console.log(result.stage); // 'adult'
 * console.log(result.applyOutcomes?.length); // number of applied morphs
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

    // Step 3: When a live network and prune budget are supplied, apply the
    //        planned morphs and commit growth hysteresis.
    if (input.network !== undefined && input.pruneBudget !== undefined) {
      // Re-seed the network RNG when the caller asked for deterministic morph
      // choices for this window. Then pin the global connection innovation
      // counter to the network's current max innovation so repeated runs with
      // the same seed and inputs assign identical innovation IDs to newly
      // created connections.
      if (input.seed !== undefined) {
        input.network.setSeed(input.seed);
      }
      syncInnovationCounterToNetwork(input.network);

      const applyBudget: MorphApplyBudget = {
        growth: input.budget,
        prune: input.pruneBudget,
      };
      const applyOutcomes = applyMorphDeltas(
        input.network,
        deltas,
        applyBudget,
      );

      const committedKind = extractCommittedGrowthKind(applyOutcomes);
      const updatedHysteresis =
        committedKind !== undefined
          ? commitGrowth(input.hysteresis, committedKind, resolvedConfig)
          : input.hysteresis;

      return {
        stage: 'adult',
        juvenileResult: { focusScore, deltas },
        applyOutcomes,
        hysteresis: updatedHysteresis,
      };
    }

    // Step 4: Dry-run path — return planned deltas without mutating any inputs.
    return {
      stage: 'adult',
      juvenileResult: { focusScore, deltas },
    };
  }

  // Step 5: In the adult stage, assimilate the equilibrium candidate into structural priors.
  const assimilationResult = assimilateEquilibriumCandidate(
    input.candidate,
    input.policy,
  );

  return {
    stage: 'adult',
    assimilationResult,
  };
}

/**
 * Pin the global connection innovation counter to a deterministic value for the
 * current network state.
 *
 * The NEAT connection allocator assigns monotonic innovation IDs from a shared
 * static counter. That counter is not reset per `Network` construction, so two
 * identical seeded networks built in the same process receive different absolute
 * innovation IDs. Before applying NGE morphs, we reset the counter to
 * `max(network connection innovation) + 1`, which is deterministic for a fixed
 * network state, so the same seed + experience stream yields bitwise-identical
 * innovation assignments for newly grown edges.
 *
 * @param network - Live network whose current connection innovations define the
 *   deterministic starting point.
 * @returns Nothing.
 */
function syncInnovationCounterToNetwork(network: Network): void {
  const innovations = network.connections.map((connection) =>
    Number(connection.innovation),
  );
  const maxInnovation = innovations.reduce(
    (maxValue, innovation) => Math.max(maxValue, innovation),
    0,
  );
  Connection.resetInnovationCounter(maxInnovation + 1);
}

/** Growth morph kinds that `commitGrowth` accepts. */
const GROWTH_MORPH_KINDS: ReadonlySet<NgeMorphDelta['kind']> = new Set([
  'edgeDensify',
  'slotExpand',
  'nodeAdd',
]);

/**
 * Find the first applied growth morph kind from apply outcomes.
 *
 * Used to determine whether `commitGrowth` should be called and which morph
 * kind to report. Prune-only or all-skipped outcome sets return `undefined`.
 *
 * @param outcomes - Apply outcomes produced by `applyMorphDeltas`.
 * @returns The first applied growth morph kind, or `undefined` when none applied.
 */
function extractCommittedGrowthKind(
  outcomes: readonly MorphApplyOutcome[],
): NgeGrowthMorphKind | undefined {
  for (const outcome of outcomes) {
    if (outcome.status === 'applied' && GROWTH_MORPH_KINDS.has(outcome.kind)) {
      return outcome.kind as NgeGrowthMorphKind;
    }
  }
  return undefined;
}
