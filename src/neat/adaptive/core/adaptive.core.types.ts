/**
 * Contract map for the adaptive helper boundary.
 *
 * The adaptive subtree works because each policy chapter can stay focused on a
 * single feedback loop while still sharing one precise agreement about what it
 * may read, what it may rewrite, and which option family owns each tuning
 * decision. This file is that agreement.
 *
 * Read the contracts in three passes:
 *
 * - start with `NeatLikeWithAdaptive` to see the runtime host surface and the
 *   scratch fields adaptive controllers are allowed to maintain,
 * - continue with the exported `*Config` aliases to see how complexity,
 *   acceptance, mutation, operator adaptation, and lineage feedback each slice
 *   the broader options object,
 * - finish with `Genome`, `MutationSettings`, `MutationPartitions`, and
 *   `MutationOutcome` when you want the normalized working shapes used inside
 *   adaptive mutation helpers.
 *
 * The matching defaults and mode labels live in `adaptive.core.constants.ts`.
 * This file stays focused on contracts so the generated chapter reads as a
 * bounded vocabulary map rather than a second controller implementation.
 *
 * ```mermaid
 * flowchart TD
 *   Host[NeatLikeWithAdaptive host] --> Options[Adaptive option families]
 *   Host --> Population[Population runtime state]
 *   Host --> Scratch[Adaptive scratch fields and telemetry]
 *   Options --> Complexity[Complexity and phased schedules]
 *   Options --> Acceptance[Acceptance and minimal criterion]
 *   Options --> Mutation[Mutation and operator adaptation]
 *   Options --> Lineage[Ancestor uniqueness and lineage pressure]
 * ```
 */
export interface NeatLikeWithAdaptive {
  /** Adaptive and scheduling configuration options for NEAT behaviors. */
  options: {
    /** Complexity-budget scheduling configuration. */
    complexityBudget?: {
      /** Enables complexity budget scheduling. */
      enabled?: boolean;
      /** Scheduling mode (for example: 'adaptive' or 'linear'). */
      mode?: string;
      /** Window size used to measure improvement trends. */
      improvementWindow?: number;
      /** Multiplicative increase factor applied on improvement. */
      increaseFactor?: number;
      /** Multiplicative decay factor applied on stagnation. */
      stagnationFactor?: number;
      /** Starting node cap for linear/initial budgeting. */
      maxNodesStart?: number;
      /** Ending node cap for linear/maximum budgeting. */
      maxNodesEnd?: number;
      /** Minimum node cap allowed during budgeting. */
      minNodes?: number;
      /** Starting connection cap for linear/initial budgeting. */
      maxConnsStart?: number;
      /** Ending connection cap for linear/maximum budgeting. */
      maxConnsEnd?: number;
      /** Horizon (in generations) for linear schedules. */
      horizon?: number;
    };
    /** Phased complexity configuration (complexify/simplify cycles). */
    phasedComplexity?: {
      /** Enables phased complexity scheduling. */
      enabled?: boolean;
      /** Explicit phase definitions with generation and caps. */
      phases?: Array<{
        /** Generation at which the phase begins. */
        generation: number;
        /** Node cap to apply during the phase. */
        maxNodes?: number;
        /** Connection cap to apply during the phase. */
        maxConns?: number;
      }>;
      /** Default duration (in generations) for a phase. */
      phaseLength?: number;
      /** Initial phase name (for example: 'complexify'). */
      initialPhase?: string;
    };
    /** Minimal-criterion configuration for acceptance thresholds. */
    minimalCriterion?: {
      /** Enables minimal-criterion gating. */
      enabled?: boolean;
      /** Strategy mode for minimal criterion. */
      mode?: string;
      /** Target complexity threshold for acceptance. */
      targetComplexity?: number;
      /** Learning rate for threshold adjustments. */
      learningRate?: number;
    };
    /** Adaptive minimal-criterion configuration. */
    minimalCriterionAdaptive?: {
      /** Enables adaptive minimal-criterion thresholding. */
      enabled?: boolean;
      /** Initial threshold for acceptance. */
      initialThreshold?: number;
      /** Target acceptance proportion in the population. */
      targetAcceptance?: number;
      /** Adjustment rate for threshold updates. */
      adjustRate?: number;
    };
    /** Ancestor-uniqueness configuration (static). */
    ancestorUniqueness?: {
      /** Enables ancestor-uniqueness tracking. */
      enabled?: boolean;
      /** Target ancestor-uniqueness rate. */
      targetRate?: number;
      /** Learning rate for uniqueness adjustments. */
      learningRate?: number;
    };
    /** Adaptive ancestor-uniqueness configuration. */
    ancestorUniqAdaptive?: {
      /** Enables adaptive ancestor-uniqueness adjustments. */
      enabled?: boolean;
      /** Cooldown (in generations) between adjustments. */
      cooldown?: number;
      /** Lower bound triggering an increase in diversity pressure. */
      lowThreshold?: number;
      /** Upper bound triggering a decrease in diversity pressure. */
      highThreshold?: number;
      /** Adjustment magnitude applied when thresholds are crossed. */
      adjust?: number;
      /** Strategy mode (for example: 'epsilon' or 'lineagePressure'). */
      mode?: string;
    };
    /** Self-adaptive mutation configuration. */
    adaptiveMutation?: {
      /** Enables per-genome mutation adaptation. */
      enabled?: boolean;
      /** Learning rate for mutation adaptation. */
      learningRate?: number;
      /** Minimum mutation rate allowed. */
      min?: number;
      /** Maximum mutation rate allowed. */
      max?: number;
      /** Adapt mutation parameters every N generations. */
      adaptEvery?: number;
      /** Standard deviation for mutation perturbations. */
      sigma?: number;
      /** Minimum per-genome mutation rate (clamp). */
      minRate?: number;
      /** Maximum per-genome mutation rate (clamp). */
      maxRate?: number;
      /** Adaptation strategy identifier. */
      strategy?: string;
      /** Enables mutation amount adaptation. */
      adaptAmount?: boolean;
      /** Minimum mutation amount allowed. */
      minAmount?: number;
      /** Maximum mutation amount allowed. */
      maxAmount?: number;
      /** Initial mutation rate used as a baseline. */
      initialRate?: number;
      /** Standard deviation for mutation-amount perturbations. */
      amountSigma?: number;
    };
    /** Adaptive operator-selection configuration. */
    operatorAdaptation?: {
      /** Enables operator adaptation. */
      enabled?: boolean;
      /** Learning rate for operator statistics updates. */
      learningRate?: number;
      /** Alpha parameter for operator adaptation. */
      alpha?: number;
      /** Decay factor for operator success/attempt statistics. */
      decay?: number;
    };
    /** Multi-objective configuration options. */
    multiObjective?: {
      /** Adaptive epsilon configuration for multi-objective selection. */
      adaptiveEpsilon?: {
        /** Enables adaptive epsilon. */
        enabled?: boolean;
      };
      /** Dominance epsilon for Pareto comparisons. */
      dominanceEpsilon?: number;
    };
    /** Lineage pressure configuration. */
    lineagePressure?: {
      /** Enables lineage pressure. */
      enabled?: boolean;
      /** Lineage pressure mode. */
      mode?: string;
      /** Lineage pressure strength value. */
      strength?: number;
    };
    /** Global maximum node cap. */
    maxNodes?: number;
    /** Global maximum connection cap. */
    maxConns?: number;
    /** Base mutation rate. */
    mutationRate?: number;
    /** Base mutation amount. */
    mutationAmount?: number;
  };
  /** Current population of genomes. */
  population: Array<{
    /** Fitness score for the genome. */
    score?: number;
    /** Per-genome mutation rate override. */
    _mutRate?: number | null;
    /** Per-genome mutation amount override. */
    _mutAmount?: number | null;
    /** Additional dynamic genome fields. */
    [key: string]: unknown;
  }>;
  /** Number of input nodes in the network. */
  input: number;
  /** Number of output nodes in the network. */
  output: number;
  /** Current generation index. */
  generation: number;
  /** Rolling history of best scores for complexity budgeting. */
  _cbHistory?: number[];
  /** Current complexity budget for nodes. */
  _cbMaxNodes?: number;
  /** Current complexity budget for connections. */
  _cbMaxConns?: number;
  /** Novelty archive used by some adaptive strategies. */
  _noveltyArchive?: unknown[];
  /** Minimal criterion baseline value. */
  _mcBaseline?: number;
  /** Minimal criterion current level. */
  _mcLevel?: number;
  /** Last observed inbreeding count. */
  _lastInbreedingCount?: number;
  /** Inbreeding penalty applied to selection. */
  _inbreedingPenalty?: number;
  /** Adaptive mutation rate computed by strategy. */
  _mutationRateAdaptive?: number;
  /** Operator success/attempt statistics by operator id. */
  _operatorStats?: Map<string, { success: number; attempts: number }>;
  /** Current phase name for phased complexity. */
  _phase?: string;
  /** Generation when the current phase started. */
  _phaseStartGeneration?: number;
  /** Minimal criterion acceptance threshold. */
  _mcThreshold?: number;
  /** Generation of the last ancestor-uniqueness adjustment. */
  _lastAncestorUniqAdjustGen?: number;
  /** Telemetry history (including lineage metrics). */
  _telemetry?: Array<{ lineage?: { ancestorUniq?: number } }>;
  /** RNG provider for deterministic randomness. */
  _getRNG?: () => () => number;
}

/**
 * Shared config view for complexity-budget helpers.
 *
 * Use this alias when a helper only cares about node and connection caps,
 * schedule shape, and improvement-window tuning for the adaptive budget loop.
 * It is the smallest view needed for the controller that grows or shrinks the
 * allowed topology budget over time.
 */
export type ComplexityBudgetConfig = NonNullable<
  NeatLikeWithAdaptive['options']['complexityBudget']
>;

/**
 * Shared config view for phased-complexity helpers.
 *
 * This isolates the alternating complexify/simplify schedule from the broader
 * adaptive options object so phase-oriented helpers can stay narrow and think
 * in terms of mode transitions instead of the entire adaptive policy surface.
 */
export type PhasedComplexityConfig = NonNullable<
  NeatLikeWithAdaptive['options']['phasedComplexity']
>;

/**
 * Shared config view for adaptive minimal-criterion helpers.
 *
 * Helpers use this slice when they are only adjusting the acceptance
 * threshold, not inspecting the rest of the controller policy surface. This is
 * the acceptance-side tuning vocabulary, not a whole-population runtime view.
 */
export type MinimalCriterionAdaptiveConfig = NonNullable<
  NeatLikeWithAdaptive['options']['minimalCriterionAdaptive']
>;

/**
 * Shared config view for ancestor-uniqueness feedback helpers.
 *
 * This captures the thresholds, cooldowns, and mode switches used when the
 * controller nudges diversity pressure in response to lineage concentration.
 * Read it as the lineage-feedback slice of the broader adaptive options object.
 */
export type AncestorUniqAdaptiveConfig = NonNullable<
  NeatLikeWithAdaptive['options']['ancestorUniqAdaptive']
>;

/**
 * Shared config view for per-genome adaptive mutation helpers.
 *
 * The mutation adaptation loop reads this slice to clamp rates, normalize
 * perturbation scales, and decide how often genome-local parameters are
 * refreshed. It is the mutation-side policy vocabulary before defaults are
 * resolved into `MutationSettings`.
 */
export type AdaptiveMutationConfig = NonNullable<
  NeatLikeWithAdaptive['options']['adaptiveMutation']
>;

/**
 * Shared config view for operator-stat adaptation helpers.
 *
 * This is the policy surface for helpers that bias mutation-operator choice
 * using historical success and attempt statistics. It stays separate from the
 * broader adaptive mutation config because operator-choice decay is a different
 * feedback loop from per-genome rate tuning.
 */
export type OperatorAdaptationConfig = NonNullable<
  NeatLikeWithAdaptive['options']['operatorAdaptation']
>;

/**
 * Normalized adaptive-mutation settings after config fallback resolution.
 *
 * This is the working form used after helpers merge user options with shared
 * defaults, so downstream logic does not need to repeatedly re-interpret
 * optional config fields. Read it as the mutation chapter's resolved call
 * frame: one object with every clamp, sigma, strategy, and baseline already in
 * concrete form.
 */
export type MutationSettings = {
  strategy: string;
  sigmaBase: number;
  minRate: number;
  maxRate: number;
  initialRate: number;
  adaptAmount: boolean;
  amountSigma: number;
  minAmount: number;
  maxAmount: number;
  mutationAmountDefault: number;
  generation: number;
  populationSize: number;
};

/**
 * Outcome flags used to detect whether mutation pressure stayed balanced.
 *
 * Mutation adaptation uses this tiny result object to summarize whether recent
 * adjustments produced both upward and downward movement rather than collapsing
 * into one-sided pressure.
 */
export type MutationOutcome = { hasIncrease: boolean; hasDecrease: boolean };

/**
 * Score-ranked population halves used by two-tier and explore-low strategies.
 *
 * This keeps the mutation strategies' working partitions explicit so helpers
 * can talk about "top half" and "bottom half" without recomputing or loosely
 * describing that split.
 */
export type MutationPartitions = { topHalf: Genome[]; bottomHalf: Genome[] };

/**
 * Shared genome view used by the adaptive helpers.
 *
 * This is intentionally opaque beyond the adaptive scratch fields already
 * exposed through `NeatLikeWithAdaptive['population']`. The adaptive core cares
 * about per-genome scores and adaptive overrides, not full network structure.
 */
export type Genome = NeatLikeWithAdaptive['population'][number];
