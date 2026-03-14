/**
 * Minimal NEAT controller shape required by the adaptive helper boundary.
 *
 * The adaptive folders share this vocabulary so each teaching-oriented README
 * can focus on its local heuristics without redefining the controller surface.
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

export type ComplexityBudgetConfig = NonNullable<
  NeatLikeWithAdaptive['options']['complexityBudget']
>;
export type PhasedComplexityConfig = NonNullable<
  NeatLikeWithAdaptive['options']['phasedComplexity']
>;
export type MinimalCriterionAdaptiveConfig = NonNullable<
  NeatLikeWithAdaptive['options']['minimalCriterionAdaptive']
>;
export type AncestorUniqAdaptiveConfig = NonNullable<
  NeatLikeWithAdaptive['options']['ancestorUniqAdaptive']
>;
export type AdaptiveMutationConfig = NonNullable<
  NeatLikeWithAdaptive['options']['adaptiveMutation']
>;
export type OperatorAdaptationConfig = NonNullable<
  NeatLikeWithAdaptive['options']['operatorAdaptation']
>;

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

export type MutationOutcome = { hasIncrease: boolean; hasDecrease: boolean };
export type MutationPartitions = { topHalf: Genome[]; bottomHalf: Genome[] };
export type Genome = NeatLikeWithAdaptive['population'][number];