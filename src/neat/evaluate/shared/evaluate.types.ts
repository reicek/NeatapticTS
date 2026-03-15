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
   * Index signature for legacy or extended genome fields.
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

      /** Rate applied to increase or decrease sigma. */
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
    fn: (genome: GenomeForEvaluation) => number,
  ) => void;

  /** Optional pending objective additions queue. */
  _pendingObjectiveAdds?: string[];

  /** Optional cached objective list invalidated on changes. */
  _objectivesList?: unknown;
}
