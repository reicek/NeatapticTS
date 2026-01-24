/**
 * Genome with score, novelty, and clearing capabilities.
 *
 * This interface describes the minimal genome shape required by evaluation
 * helpers. It intentionally stays permissive for compatibility with legacy
 * genome variants while documenting the expected properties.
 */
export interface GenomeForEvaluation {
  /** Optional fitness score assigned during evaluation. */
  score?: number;
  /** Optional method to clear internal state between evaluations. */
  clear?: () => void;
  /** Connection list used for structural variance tuning. */
  connections: unknown[];
  /** Optional node list used by some novelty descriptors. */
  nodes?: unknown[];
  /** Optional novelty value computed during evaluation. */
  _novelty?: number;
  /**
   * Index signature for legacy/extended genome fields.
   *
   * This keeps the evaluation helpers forward-compatible with custom genome
   * extensions in downstream projects.
   */
  [key: string]: unknown;
}

/**
 * Novelty archive entry with descriptor and novelty score.
 *
 * Entries store a descriptor vector alongside the computed novelty so the
 * archive can seed future novelty calculations.
 */
export interface NoveltyArchiveEntry {
  /** Descriptor vector representing a genome's behavior. */
  desc: number[];
  /** Novelty score associated with the descriptor. */
  novelty: number;
}

/**
 * Diversity statistics tracked during evaluation.
 *
 * The values are optional because different evaluations may only compute a
 * subset of metrics.
 */
export interface DiversityStats {
  /** Variance of entropy across the population. */
  varEntropy?: number;
  /** Mean entropy across the population. */
  meanEntropy?: number;
  /** Additional metrics added by adaptive tuning logic. */
  [key: string]: unknown;
}

/**
 * Objective definition for multi-objective optimization.
 *
 * Objectives are registered dynamically to guide evaluation and selection.
 */
export interface ObjectiveDef {
  /** Unique key used to reference the objective. */
  key: string;
  /** Direction indicating whether higher or lower values are preferred. */
  direction?: string;
  /** Optional scoring function for the objective. */
  fn?: (genome: GenomeForEvaluation) => number;
}

/**
 * NEAT controller interface for evaluation.
 *
 * This interface models the subset of a NEAT controller used by the evaluation
 * helpers. It includes options, population data, and optional adaptive tuning
 * hooks.
 */
export interface NeatControllerForEval {
  /** Runtime options that influence evaluation and tuning behavior. */
  options: {
    /** Run fitness once for the whole population when true. */
    fitnessPopulation?: boolean;
    /** Clear genome internal state before scoring when true. */
    clear?: boolean;
    /** Configuration for novelty search and blending. */
    novelty?: {
      /** Enable novelty computation when true. */
      enabled?: boolean;
      /** Descriptor function mapping a genome to a behavior vector. */
      descriptor?: (genome: GenomeForEvaluation) => number[];
      /** Number of neighbors used for novelty scoring. */
      k?: number;
      /** Blend factor between fitness and novelty. */
      blendFactor?: number;
      /** Threshold required to add a genome to the novelty archive. */
      archiveAddThreshold?: number;
    };
    /** Configuration for entropy-sharing sigma tuning. */
    entropySharingTuning?: {
      /** Enable entropy-sharing tuning when true. */
      enabled?: boolean;
      /** Target variance for entropy distribution. */
      targetEntropyVar?: number;
      /** Rate applied to increase/decrease sigma. */
      adjustRate?: number;
      /** Minimum sigma value allowed. */
      minSigma?: number;
      /** Maximum sigma value allowed. */
      maxSigma?: number;
    };
    /** Configuration for compatibility threshold tuning. */
    entropyCompatTuning?: {
      /** Enable compatibility threshold tuning when true. */
      enabled?: boolean;
      /** Target mean entropy. */
      targetEntropy?: number;
      /** Deadband around the target entropy. */
      deadband?: number;
      /** Rate applied to adjust the threshold. */
      adjustRate?: number;
      /** Minimum compatibility threshold allowed. */
      minThreshold?: number;
      /** Maximum compatibility threshold allowed. */
      maxThreshold?: number;
    };
    /** Configuration for auto distance coefficient tuning. */
    autoDistanceCoeffTuning?: {
      /** Enable coefficient tuning when true. */
      enabled?: boolean;
      /** Rate applied to adjust coefficients. */
      adjustRate?: number;
      /** Minimum coefficient allowed. */
      minCoeff?: number;
      /** Maximum coefficient allowed. */
      maxCoeff?: number;
    };
    /** Multi-objective optimization settings. */
    multiObjective?: {
      /** Enable multi-objective optimization when true. */
      enabled?: boolean;
      /** Automatically register entropy objective when true. */
      autoEntropy?: boolean;
      /** Dynamic objective configuration. */
      dynamic?: {
        /** Enable dynamic objective handling when true. */
        enabled?: boolean;
      };
    };
    /** Enable speciation logic when true. */
    speciation?: boolean;
    /** Target number of species when speciation tuning is active. */
    targetSpecies?: number;
    /** Enable compatibility threshold adjustment when true. */
    compatAdjust?: boolean;
    /** Species allocation settings. */
    speciesAllocation?: {
      /** Enable extended history tracking when true. */
      extendedHistory?: boolean;
    };
    /** Current sharing sigma value used for fitness sharing. */
    sharingSigma?: number;
    /** Current compatibility threshold value. */
    compatibilityThreshold?: number;
    /** Excess coefficient used in distance calculations. */
    excessCoeff?: number;
    /** Disjoint coefficient used in distance calculations. */
    disjointCoeff?: number;
    /** Additional, controller-specific options. */
    [key: string]: unknown;
  };
  /** Current population of genomes to evaluate. */
  population: GenomeForEvaluation[];
  /** Fitness delegate invoked per genome or per population. */
  fitness: (
    genomeOrPop: GenomeForEvaluation | GenomeForEvaluation[],
  ) => Promise<number | void>;
  /** Optional novelty archive storing descriptors and novelty scores. */
  _noveltyArchive?: NoveltyArchiveEntry[];
  /** Optional diversity statistics storage used by tuning steps. */
  _diversityStats?: DiversityStats;
  /** Optional last observed connection variance for tuning. */
  _lastConnVar?: number | null;
  /** Optional speciation method. */
  _speciate?: () => void;
  /** Optional objective getter. */
  _getObjectives?: () => ObjectiveDef[];
  /** Optional entropy calculation helper. */
  _structuralEntropy?: (genome: GenomeForEvaluation) => number;
  /** Optional method for registering objectives. */
  registerObjective?: (
    key: string,
    direction: string,
    fn: (g: GenomeForEvaluation) => number,
  ) => void;
  /** Optional pending objective additions queue. */
  _pendingObjectiveAdds?: string[];
  /** Optional cached objective list (invalidated on changes). */
  _objectivesList?: unknown;
}

/** Default neighbor count for novelty calculation. */
export const NOVELTY_DEFAULT_NEIGHBORS = 3;
/** Default blend factor for novelty vs. fitness. */
export const NOVELTY_DEFAULT_BLEND = 0.3;
/** Maximum number of entries stored in the novelty archive. */
export const NOVELTY_ARCHIVE_CAP = 200;
/** Default target variance for entropy sharing. */
export const ENTROPY_VAR_TARGET_DEFAULT = 0.2;
/** Default adjustment rate for entropy sharing. */
export const ENTROPY_VAR_ADJUST_DEFAULT = 0.1;
/** Default minimum sigma for entropy sharing. */
export const ENTROPY_VAR_MIN_SIGMA_DEFAULT = 0.1;
/** Default maximum sigma for entropy sharing. */
export const ENTROPY_VAR_MAX_SIGMA_DEFAULT = 10;
/** Lower band multiplier for entropy variance tuning. */
export const ENTROPY_VAR_LOW_BAND = 0.9;
/** Upper band multiplier for entropy variance tuning. */
export const ENTROPY_VAR_HIGH_BAND = 1.1;
/** Default target entropy for compatibility tuning. */
export const ENTROPY_TARGET_DEFAULT = 0.5;
/** Default deadband for compatibility tuning. */
export const ENTROPY_DEADBAND_DEFAULT = 0.05;
/** Default adjustment rate for compatibility tuning. */
export const ENTROPY_ADJUST_DEFAULT = 0.05;
/** Default compatibility threshold when not provided. */
export const COMPAT_THRESHOLD_DEFAULT = 3;
/** Default minimum compatibility threshold. */
export const COMPAT_MIN_THRESHOLD_DEFAULT = 0.5;
/** Default maximum compatibility threshold. */
export const COMPAT_MAX_THRESHOLD_DEFAULT = 10;
/** Default adjustment rate for auto distance coefficient tuning. */
export const AUTO_COEFF_ADJUST_DEFAULT = 0.05;
/** Default minimum coefficient for auto distance coefficient tuning. */
export const AUTO_COEFF_MIN_DEFAULT = 0.05;
/** Default maximum coefficient for auto distance coefficient tuning. */
export const AUTO_COEFF_MAX_DEFAULT = 8;
/** Default coefficient value when not provided. */
export const DISTANCE_COEFF_DEFAULT = 1;
/** Variance decrease threshold multiplier. */
export const VARIANCE_DECREASE_THRESHOLD = 0.95;
/** Variance increase threshold multiplier. */
export const VARIANCE_INCREASE_THRESHOLD = 1.05;

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns Promise<void> after fitness evaluation completes.
 */
export async function runFitnessEvaluation(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): Promise<void> {
  // Step 1: Select population-level or per-genome fitness evaluation.
  if (evaluationOptions.fitnessPopulation) {
    // Step 2: Clear genome state when requested.
    clearGenomeStateIfRequested(controller, evaluationOptions, (genome) =>
      genome.clear?.(),
    );
    // Step 3: Execute population-level fitness.
    await controller.fitness(controller.population);
    return;
  }

  // Step 2: Evaluate each genome individually.
  for (const genome of controller.population) {
    if (evaluationOptions.clear && genome.clear) genome.clear();
    const fitnessValue = await controller.fitness(genome);
    genome.score = fitnessValue as number;
  }
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runNoveltyBlendAndArchive(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Guard and evaluate novelty if enabled.
  try {
    const noveltyOptions = evaluationOptions.novelty;
    if (
      !noveltyOptions?.enabled ||
      typeof noveltyOptions.descriptor !== 'function'
    ) {
      return;
    }

    const kNeighbors = getNoveltyNeighborCount(noveltyOptions);
    const blendFactor = getNoveltyBlendFactor(noveltyOptions);
    const descriptors = buildNoveltyDescriptors(controller, noveltyOptions);
    const distanceMatrix = buildDistanceMatrix(descriptors);
    applyNoveltyToPopulation(
      controller,
      descriptors,
      distanceMatrix,
      kNeighbors,
      blendFactor,
      noveltyOptions,
    );
  } catch {
    // Intentionally ignore novelty computation errors to allow evaluation to continue
  }
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @returns void.
 */
export function ensureDiversityStatsContainer(
  controller: NeatControllerForEval,
): void {
  // Step 1: Ensure a diversity stats container exists for tuning.
  if (!controller._diversityStats) controller._diversityStats = {};
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runEntropySharingTuning(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Adjust sharing sigma when entropy sharing tuning is enabled.
  try {
    const entropySharingOptions = evaluationOptions.entropySharingTuning;
    if (!entropySharingOptions?.enabled) return;
    const currentVarEntropy = controller._diversityStats?.varEntropy;
    if (typeof currentVarEntropy !== 'number') return;
    const nextSigma = computeNextSharingSigma(
      entropySharingOptions,
      currentVarEntropy,
      controller.options.sharingSigma ?? 0,
    );
    controller.options.sharingSigma = nextSigma;
  } catch {
    // Intentionally ignore entropy sharing tuning errors
  }
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runEntropyCompatibilityTuning(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Adjust compatibility threshold when entropy tuning is enabled.
  try {
    const entropyCompatOptions = evaluationOptions.entropyCompatTuning;
    if (!entropyCompatOptions?.enabled) return;
    const meanEntropy = controller._diversityStats?.meanEntropy;
    if (typeof meanEntropy !== 'number') return;
    const nextThreshold = computeNextCompatibilityThreshold(
      entropyCompatOptions,
      meanEntropy,
      controller.options.compatibilityThreshold ?? COMPAT_THRESHOLD_DEFAULT,
    );
    controller.options.compatibilityThreshold = nextThreshold;
  } catch {
    // Intentionally ignore entropy-compatibility tuning errors
  }
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runLightweightSpeciation(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Run speciation when controller features are enabled.
  try {
    if (!shouldRunSpeciation(evaluationOptions)) return;
    controller._speciate?.();
  } catch {
    // Intentionally ignore speciation errors during evaluation
  }
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Apply variance-driven coefficient tuning.
  try {
    const autoDistanceCoeffOptions = evaluationOptions.autoDistanceCoeffTuning;
    if (!autoDistanceCoeffOptions?.enabled || !evaluationOptions.speciation) {
      return;
    }
    const connectionSizes = controller.population.map(
      (genome) => genome.connections.length,
    );
    const meanConnectionSize = computeMean(connectionSizes);
    const connectionVariance = computeVariance(
      connectionSizes,
      meanConnectionSize,
    );
    applyAutoDistanceCoefficientTuning(
      controller,
      autoDistanceCoeffOptions,
      connectionVariance,
    );
  } catch {
    // Intentionally ignore auto-distance coefficient tuning errors
  }
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runAutoEntropyObjectiveInjection(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Inject entropy objective when requested.
  try {
    if (!shouldAutoInjectEntropy(evaluationOptions)) return;
    const objectiveKeys =
      controller._getObjectives?.()?.map((o) => o.key) ?? [];
    if (objectiveKeys.includes('entropy')) return;
    registerEntropyObjective(controller);
  } catch {
    // Intentionally ignore auto-entropy objective injection errors
  }
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @param clearAction - Action that clears a genome's internal state.
 * @returns void.
 */
function clearGenomeStateIfRequested(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
  clearAction: (genome: GenomeForEvaluation) => void,
): void {
  // Step 1: Clear genome state when the option is enabled.
  if (!evaluationOptions.clear) return;
  controller.population.forEach((genome) => clearAction(genome));
}

/**
 * @param noveltyOptions - Novelty configuration.
 * @returns Number of neighbors to consider.
 */
function getNoveltyNeighborCount(
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): number {
  // Step 1: Enforce at least one neighbor.
  return Math.max(1, noveltyOptions.k || NOVELTY_DEFAULT_NEIGHBORS);
}

/**
 * @param noveltyOptions - Novelty configuration.
 * @returns Blend factor for novelty vs. fitness.
 */
function getNoveltyBlendFactor(
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): number {
  // Step 1: Default to a moderate blend if not provided.
  return noveltyOptions.blendFactor ?? NOVELTY_DEFAULT_BLEND;
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param noveltyOptions - Novelty configuration.
 * @returns Descriptor vectors for each genome.
 */
function buildNoveltyDescriptors(
  controller: NeatControllerForEval,
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): number[][] {
  // Step 1: Map genomes through the descriptor function.
  return controller.population.map((genome) => {
    try {
      return noveltyOptions.descriptor?.(genome) ?? [];
    } catch {
      return [];
    }
  });
}

/**
 * @param descriptors - Descriptor vectors.
 * @returns Distance matrix.
 */
function buildDistanceMatrix(descriptors: number[][]): number[][] {
  // Step 1: Compute distances between descriptor vectors.
  return descriptors.map((rowDescriptor, rowIndex) =>
    descriptors.map((columnDescriptor, columnIndex) =>
      computeDescriptorDistance(
        rowDescriptor,
        columnDescriptor,
        rowIndex === columnIndex,
      ),
    ),
  );
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param descriptors - Descriptor vectors for each genome.
 * @param distanceMatrix - Distance matrix.
 * @param kNeighbors - Neighbor count.
 * @param blendFactor - Blend factor.
 * @param noveltyOptions - Novelty configuration.
 * @returns void.
 */
function applyNoveltyToPopulation(
  controller: NeatControllerForEval,
  descriptors: number[][],
  distanceMatrix: number[][],
  kNeighbors: number,
  blendFactor: number,
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): void {
  // Step 1: Compute novelty and blend it into scores.
  controller.population.forEach((genome, genomeIndex) => {
    const novelty = computeNoveltyScore(
      distanceMatrix[genomeIndex],
      kNeighbors,
    );
    genome._novelty = novelty;
    blendNoveltyIntoScore(genome, novelty, blendFactor);
    addGenomeToNoveltyArchive(
      controller,
      descriptors[genomeIndex],
      novelty,
      noveltyOptions,
    );
  });
}

/**
 * @param distanceRow - Distance values for a single genome.
 * @param kNeighbors - Neighbor count.
 * @returns Novelty score.
 */
function computeNoveltyScore(
  distanceRow: number[],
  kNeighbors: number,
): number {
  // Step 1: Average the k nearest neighbors.
  const sortedDistances = distanceRow.toSorted((left, right) => left - right);
  const neighbors = sortedDistances.slice(1, kNeighbors + 1);
  if (neighbors.length === 0) return 0;
  return (
    neighbors.reduce((accumulated, value) => accumulated + value, 0) /
    neighbors.length
  );
}

/**
 * @param genome - Genome to update.
 * @param novelty - Novelty value.
 * @param blendFactor - Blend factor.
 * @returns void.
 */
function blendNoveltyIntoScore(
  genome: GenomeForEvaluation,
  novelty: number,
  blendFactor: number,
): void {
  // Step 1: Blend novelty into score when a numeric score is present.
  if (typeof genome.score !== 'number') return;
  genome.score =
    (1 - blendFactor) * (genome.score ?? 0) + blendFactor * novelty;
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param descriptor - Genome descriptor.
 * @param novelty - Novelty score.
 * @param noveltyOptions - Novelty configuration.
 * @returns void.
 */
function addGenomeToNoveltyArchive(
  controller: NeatControllerForEval,
  descriptor: number[],
  novelty: number,
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): void {
  // Step 1: Ensure archive exists.
  if (!controller._noveltyArchive) controller._noveltyArchive = [];
  // Step 2: Evaluate archive eligibility.
  const archiveAddThreshold = noveltyOptions.archiveAddThreshold ?? Infinity;
  const shouldAdd =
    noveltyOptions.archiveAddThreshold === 0 || novelty > archiveAddThreshold;
  if (!shouldAdd) return;
  // Step 3: Append while under the cap.
  if (controller._noveltyArchive.length < NOVELTY_ARCHIVE_CAP) {
    controller._noveltyArchive.push({ desc: descriptor, novelty });
  }
}

/**
 * @param left - Left descriptor.
 * @param right - Right descriptor.
 * @param isSame - Whether the descriptors are the same index.
 * @returns Euclidean distance.
 */
function computeDescriptorDistance(
  left: number[],
  right: number[],
  isSame: boolean,
): number {
  // Step 1: Return zero for identical indices.
  if (isSame) return 0;
  // Step 2: Compute Euclidean distance on the common prefix.
  const commonLength = Math.min(left.length, right.length);
  const squaredSum = left
    .slice(0, commonLength)
    .reduce((accumulated, leftValue, index) => {
      const delta = leftValue - (right[index] ?? 0);
      return accumulated + delta * delta;
    }, 0);
  return Math.sqrt(squaredSum);
}

/**
 * @param entropySharingOptions - Tuning options.
 * @param currentVarEntropy - Current variance of entropy.
 * @param currentSigma - Current sigma value.
 * @returns Next sigma value.
 */
function computeNextSharingSigma(
  entropySharingOptions: NonNullable<
    NeatControllerForEval['options']['entropySharingTuning']
  >,
  currentVarEntropy: number,
  currentSigma: number,
): number {
  // Step 1: Prepare tuning constants.
  const targetVar =
    entropySharingOptions.targetEntropyVar ?? ENTROPY_VAR_TARGET_DEFAULT;
  const adjustRate =
    entropySharingOptions.adjustRate ?? ENTROPY_VAR_ADJUST_DEFAULT;
  const minSigma =
    entropySharingOptions.minSigma ?? ENTROPY_VAR_MIN_SIGMA_DEFAULT;
  const maxSigma =
    entropySharingOptions.maxSigma ?? ENTROPY_VAR_MAX_SIGMA_DEFAULT;
  // Step 2: Adjust sigma based on variance band.
  if (currentVarEntropy < targetVar * ENTROPY_VAR_LOW_BAND) {
    return Math.max(minSigma, currentSigma * (1 - adjustRate));
  }
  if (currentVarEntropy > targetVar * ENTROPY_VAR_HIGH_BAND) {
    return Math.min(maxSigma, currentSigma * (1 + adjustRate));
  }
  return currentSigma;
}

/**
 * @param entropyCompatOptions - Tuning options.
 * @param meanEntropy - Current mean entropy.
 * @param currentThreshold - Current compatibility threshold.
 * @returns Next compatibility threshold.
 */
function computeNextCompatibilityThreshold(
  entropyCompatOptions: NonNullable<
    NeatControllerForEval['options']['entropyCompatTuning']
  >,
  meanEntropy: number,
  currentThreshold: number,
): number {
  // Step 1: Prepare tuning constants.
  const targetEntropy =
    entropyCompatOptions.targetEntropy ?? ENTROPY_TARGET_DEFAULT;
  const deadband = entropyCompatOptions.deadband ?? ENTROPY_DEADBAND_DEFAULT;
  const adjustRate = entropyCompatOptions.adjustRate ?? ENTROPY_ADJUST_DEFAULT;
  const minThreshold =
    entropyCompatOptions.minThreshold ?? COMPAT_MIN_THRESHOLD_DEFAULT;
  const maxThreshold =
    entropyCompatOptions.maxThreshold ?? COMPAT_MAX_THRESHOLD_DEFAULT;
  // Step 2: Adjust within the deadband.
  if (meanEntropy < targetEntropy - deadband) {
    return Math.max(minThreshold, currentThreshold * (1 - adjustRate));
  }
  if (meanEntropy > targetEntropy + deadband) {
    return Math.min(maxThreshold, currentThreshold * (1 + adjustRate));
  }
  return currentThreshold;
}

/**
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns Whether speciation should be run.
 */
function shouldRunSpeciation(
  evaluationOptions: NeatControllerForEval['options'],
): boolean {
  // Step 1: Require speciation and at least one related feature enabled.
  if (!evaluationOptions.speciation) return false;
  return Boolean(
    evaluationOptions.targetSpecies ||
    evaluationOptions.compatAdjust ||
    evaluationOptions.speciesAllocation?.extendedHistory,
  );
}

/**
 * @param values - Input values.
 * @returns Mean of the values.
 */
function computeMean(values: number[]): number {
  // Step 1: Guard empty arrays.
  if (values.length === 0) return 0;
  return (
    values.reduce((accumulated, value) => accumulated + value, 0) /
    values.length
  );
}

/**
 * @param values - Input values.
 * @param meanValue - Precomputed mean.
 * @returns Variance of the values.
 */
function computeVariance(values: number[], meanValue: number): number {
  // Step 1: Guard empty arrays.
  if (values.length === 0) return 0;
  const squaredSum = values.reduce(
    (accumulated, value) => accumulated + (value - meanValue) ** 2,
    0,
  );
  return squaredSum / values.length;
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param autoDistanceCoeffOptions - Tuning options.
 * @param connectionVariance - Variance of connection counts.
 * @returns void.
 */
function applyAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  autoDistanceCoeffOptions: NonNullable<
    NeatControllerForEval['options']['autoDistanceCoeffTuning']
  >,
  connectionVariance: number,
): void {
  // Step 1: Initialize and bootstrap if needed.
  const tuningBounds = getDistanceCoefficientBounds(autoDistanceCoeffOptions);
  const adjustRate =
    autoDistanceCoeffOptions.adjustRate ?? AUTO_COEFF_ADJUST_DEFAULT;
  if (
    controller._lastConnVar === undefined ||
    controller._lastConnVar === null
  ) {
    initializeConnectionVarianceBootstrap(
      controller,
      connectionVariance,
      tuningBounds,
      adjustRate,
    );
  }

  // Step 2: Apply tuning based on variance delta.
  if (
    connectionVariance <
    (controller._lastConnVar ?? 0) * VARIANCE_DECREASE_THRESHOLD
  ) {
    applyDistanceCoefficientIncrease(controller, tuningBounds, adjustRate);
  } else if (
    connectionVariance >
    (controller._lastConnVar ?? 0) * VARIANCE_INCREASE_THRESHOLD
  ) {
    applyDistanceCoefficientDecrease(controller, tuningBounds, adjustRate);
  }
  controller._lastConnVar = connectionVariance;
}

/**
 * @param autoDistanceCoeffOptions - Tuning options.
 * @returns Bounds for coefficients.
 */
function getDistanceCoefficientBounds(
  autoDistanceCoeffOptions: NonNullable<
    NeatControllerForEval['options']['autoDistanceCoeffTuning']
  >,
): { minCoeff: number; maxCoeff: number } {
  // Step 1: Resolve bounds.
  return {
    minCoeff: autoDistanceCoeffOptions.minCoeff ?? AUTO_COEFF_MIN_DEFAULT,
    maxCoeff: autoDistanceCoeffOptions.maxCoeff ?? AUTO_COEFF_MAX_DEFAULT,
  };
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param connectionVariance - Current connection variance.
 * @param bounds - Min/max coefficients.
 * @param adjustRate - Adjustment rate.
 * @returns void.
 */
function initializeConnectionVarianceBootstrap(
  controller: NeatControllerForEval,
  connectionVariance: number,
  bounds: { minCoeff: number; maxCoeff: number },
  adjustRate: number,
): void {
  // Step 1: Record baseline variance.
  controller._lastConnVar = connectionVariance;
  // Step 2: Apply a deterministic nudge so tuning has a visible effect.
  try {
    controller.options.excessCoeff = Math.min(
      bounds.maxCoeff,
      (controller.options.excessCoeff ?? DISTANCE_COEFF_DEFAULT) *
        (1 + adjustRate),
    );
    controller.options.disjointCoeff = Math.min(
      bounds.maxCoeff,
      (controller.options.disjointCoeff ?? DISTANCE_COEFF_DEFAULT) *
        (1 + adjustRate),
    );
  } catch {
    // Intentionally ignore coefficient adjustment errors during bootstrap
  }
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param bounds - Min/max coefficients.
 * @param adjustRate - Adjustment rate.
 * @returns void.
 */
function applyDistanceCoefficientIncrease(
  controller: NeatControllerForEval,
  bounds: { minCoeff: number; maxCoeff: number },
  adjustRate: number,
): void {
  // Step 1: Increase coefficients within bounds.
  controller.options.excessCoeff = Math.min(
    bounds.maxCoeff,
    (controller.options.excessCoeff ?? DISTANCE_COEFF_DEFAULT) *
      (1 + adjustRate),
  );
  controller.options.disjointCoeff = Math.min(
    bounds.maxCoeff,
    (controller.options.disjointCoeff ?? DISTANCE_COEFF_DEFAULT) *
      (1 + adjustRate),
  );
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param bounds - Min/max coefficients.
 * @param adjustRate - Adjustment rate.
 * @returns void.
 */
function applyDistanceCoefficientDecrease(
  controller: NeatControllerForEval,
  bounds: { minCoeff: number; maxCoeff: number },
  adjustRate: number,
): void {
  // Step 1: Decrease coefficients within bounds.
  controller.options.excessCoeff = Math.max(
    bounds.minCoeff,
    (controller.options.excessCoeff ?? DISTANCE_COEFF_DEFAULT) *
      (1 - adjustRate),
  );
  controller.options.disjointCoeff = Math.max(
    bounds.minCoeff,
    (controller.options.disjointCoeff ?? DISTANCE_COEFF_DEFAULT) *
      (1 - adjustRate),
  );
}

/**
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns Whether entropy objective should be injected.
 */
function shouldAutoInjectEntropy(
  evaluationOptions: NeatControllerForEval['options'],
): boolean {
  // Step 1: Require multi-objective and auto-entropy without dynamic override.
  if (!evaluationOptions.multiObjective?.enabled) return false;
  if (!evaluationOptions.multiObjective.autoEntropy) return false;
  return !evaluationOptions.multiObjective.dynamic?.enabled;
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @returns void.
 */
function registerEntropyObjective(controller: NeatControllerForEval): void {
  // Step 1: Register entropy objective and invalidate cache.
  controller.registerObjective?.(
    'entropy',
    'max',
    (genome) => controller._structuralEntropy?.(genome) ?? 0,
  );
  controller._pendingObjectiveAdds?.push('entropy');
  controller._objectivesList = undefined;
}
