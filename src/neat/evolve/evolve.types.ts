import type Network from '../../architecture/network';

/**
 * Runtime interface for a genome carrying evolution metadata.
 *
 * This mirrors the dynamic properties attached at runtime during evolution,
 * without pulling in the full Genome class to avoid circular dependencies.
 *
 * @remarks
 * These fields are intentionally permissive because evolution attaches
 * metadata (ids, ancestry, multi-objective ranks) dynamically.
 */
export interface GenomeWithMetadata {
  /** Node list for the genome. */
  nodes: unknown[];
  /** Connection list for the genome. */
  connections: unknown[];
  /** Fitness score for the genome. */
  score?: number;
  /** Unique runtime id. */
  _id?: number;
  /** Shared fitness value used in speciation. */
  _sharedFitness?: number;
  /** Crowding distance for multi-objective selection. */
  _crowdingDistance?: number;
  /** Pareto front rank for multi-objective selection. */
  _frontRank?: number;
  /** Cached structural entropy. */
  _structuralEntropy?: number;
  /** Multi-objective rank. */
  _moRank?: number;
  /** Multi-objective crowding metric. */
  _moCrowd?: number;
  /** Compatibility cache (for speciation distance). */
  _compatCache?: Record<string, number>;
  /** Lineage parent ids. */
  _parents?: number[];
  /** Lineage depth. */
  _depth?: number;
  /** Re-enable probability for disabled connections. */
  _reenableProb?: number;
  /** Optional cleanup hook. */
  clear?: () => void;
  /** Optional mutate hook. */
  mutate?: (method: MutationMethod) => void;
  /** Serialize genome into JSON. */
  toJSON?: () => Record<string, unknown>;
  /** Clone the genome instance. */
  clone?: () => GenomeWithMetadata;
}

/**
 * Runtime interface for species metadata used in allocation and stats.
 */
export interface SpeciesWithMetadata {
  /** Member genomes in this species. */
  members: GenomeWithMetadata[];
  /** Species id. */
  id: number;
  /** Generation when the species was created. */
  generation: number;
  /** Shared fitness across the species. */
  sharedFitness?: number;
  /** Average shared fitness across members. */
  avgSharedFitness?: number;
  /** Offspring count allocated this generation. */
  offspring?: number;
  /** Best score seen in this species. */
  bestScore?: number;
  /** Generation of last improvement. */
  lastImproved: number;
}

/**
 * Species history snapshot record used for telemetry/exports.
 */
export interface SpeciesHistoryRecord {
  /** Generation for this snapshot. */
  generation: number;
  /** Per-species stats at the snapshot. */
  stats: Array<{
    /** Species id. */
    id: number;
    /** Number of members. */
    size: number;
    /** Average shared fitness. */
    avgSharedFitness?: number;
    /** Best score recorded. */
    bestScore?: number;
    /** Generation when last improved. */
    lastImproved?: number;
  }>;
}

/**
 * Mutation method descriptor used by runtime mutation hooks.
 */
export interface MutationMethod {
  /** Mutation operator name. */
  name: string;
}

/**
 * Objective descriptor for multi-objective evaluation.
 */
export interface ObjectiveDescriptor {
  /** Objective key identifier. */
  key: string;
  /** Accessor that extracts a numeric objective value. */
  accessor: (genome: GenomeWithMetadata) => number;
}

/**
 * Multi-objective configuration block.
 */
export interface MultiObjectiveOptions {
  /** Enable multi-objective scoring and ranking. */
  enabled?: boolean;
  /** Adaptive epsilon configuration for Pareto dominance. */
  adaptiveEpsilon?: {
    enabled?: boolean;
    targetFront?: number;
    adjust?: number;
    min?: number;
    max?: number;
    cooldown?: number;
  };
  /** Epsilon used for dominance checks. */
  dominanceEpsilon?: number;
  /** Pruning policy for inactive objectives. */
  pruneInactive?: {
    enabled?: boolean;
    window?: number;
    rangeEps?: number;
    protect?: string[];
  };
  /** Objective list (keys + metadata). */
  objectives?: Array<{ key: string; [key: string]: unknown }>;
  /** Dynamic objective scheduling rules. */
  dynamic?: {
    enabled?: boolean;
    addComplexityAt?: number;
    addEntropyAt?: number;
    dropEntropyOnStagnation?: number;
    readdEntropyAfter?: number;
  };
  /** Auto-entropy fallback flag. */
  autoEntropy?: boolean;
}

/**
 * NEAT controller subset used by evolve orchestrations.
 *
 * @remarks
 * This is a minimal, runtime-focused surface used by evolve utilities
 * to avoid circular dependencies on the full controller class.
 */
export interface NeatControllerForEvolution {
  /** Input size for new networks. */
  input: number;
  /** Output size for new networks. */
  output: number;
  /** Population for the current generation. */
  population: GenomeWithMetadata[];
  /** Current generation index. */
  generation: number;
  /** Evolution options. */
  options: {
    popsize?: number;
    elitism?: number;
    provenance?: number;
    selection?: unknown;
    crossover?: unknown;
    mutation?: unknown;
    multiObjective?: MultiObjectiveOptions;
    speciation?: {
      enabled?: boolean;
    };
    speciesAllocation?: {
      extendedHistory?: boolean;
      minOffspring?: number;
    };
    speciesAgeBonus?: {
      youngThreshold?: number;
      youngMultiplier?: number;
      oldThreshold?: number;
      oldMultiplier?: number;
    };
    pruning?: {
      enabled?: boolean;
    };
    telemetry?: {
      enabled?: boolean;
    };
    stagnationInjection?: {
      enabled?: boolean;
      threshold?: number;
      rate?: number;
    };
    globalStagnationGenerations?: number;
    network?: Network;
    minHidden?: number;
    autoCompatTuning?: {
      enabled?: boolean;
      target?: number;
      adjustRate?: number;
      minCoeff?: number;
      maxCoeff?: number;
    };
    targetSpecies?: number;
    excessCoeff?: number;
    disjointCoeff?: number;
    crossSpeciesMatingProb?: number;
    survivalThreshold?: number;
    equal?: boolean;
    reenableProb?: number;
  };
  /** Cached objective list. */
  _objectivesList?: ObjectiveDescriptor[];
  /** Best score in previous generation. */
  _bestScoreLastGen?: number;
  /** Last generation where the global best improved. */
  _lastGlobalImproveGeneration?: number;
  /** Best global score ever seen. */
  _bestGlobalScore: number;
  /** Diversity stats hook. */
  _computeDiversityStats?: () => void;
  /** Objectives getter hook. */
  _getObjectives?: () => ObjectiveDescriptor[];
  /** Current species list. */
  _species?: SpeciesWithMetadata[];
  /** Species history snapshots. */
  _speciesHistory?: SpeciesHistoryRecord[];
  /** RNG factory. */
  _getRNG: () => () => number;
  /** Speciation hook. */
  _speciate?: () => void;
  /** Fitness sharing hook. */
  _applyFitnessSharing?: () => void;
  /** Structural entropy accessor. */
  _structuralEntropy?: (genome: GenomeWithMetadata) => number;
  /** Fitness suppression state (tests). */
  _fitnessSuppressedOnce?: boolean;
  /** Objective suppression flag (tests). */
  _suppressFitnessObjective?: boolean;
  /** Last objective importance stats. */
  _lastObjImportance?: Record<string, { range: number; var: number }>;
  /** Suppress tournament errors when needed. */
  _suppressTournamentError?: boolean;
  /** Invalidate caches hook. */
  _invalidateGenomeCaches?: (genome: GenomeWithMetadata) => void;
  /** Population pruning hook. */
  _prunePopulation?: () => void;
  /** Telemetry record hook. */
  _recordTelemetry?: () => void;
  /** Next genome id counter. */
  _nextGenomeId: number;
  /** Lineage tracking enabled flag. */
  _lineageEnabled?: boolean;
  /** Pareto archive history. */
  _paretoArchive: Array<{
    gen: number;
    size: number;
    genomes: Array<{
      id: number;
      score: number;
      nodes: number;
      connections: number;
    }>;
  }>;
  /** Pareto objective vectors history. */
  _paretoObjectivesArchive: Array<{
    gen: number;
    vectors: Array<{ id: number; values: number[] }>;
  }>;
  /** Last epsilon adjustment generation. */
  _lastEpsilonAdjustGen: number;
  /** Stale objective counts. */
  _objectiveStale: Map<string, number>;
  /** Pending objective additions. */
  _pendingObjectiveAdds: string[];
  /** Pending objective removals. */
  _pendingObjectiveRemoves: string[];
  /** Entropy objective removal generation. */
  _entropyDropped?: number;
  /** Objective age tracking. */
  _objectiveAges: Map<string, number>;
  /** Last offspring allocation snapshot. */
  _lastOffspringAlloc: Array<{ id: number; alloc: number }>;
  /** Previous inbreeding count. */
  _prevInbreedingCount: number;
  /** Last inbreeding count. */
  _lastInbreedingCount: number;
  /** Species member sorter hook. */
  _sortSpeciesMembers: (species: SpeciesWithMetadata) => void;
  /** Update species stagnation hook. */
  _updateSpeciesStagnation: () => void;
  /** Last evolve duration in milliseconds. */
  _lastEvolveDuration: number;
  /** Evaluate population hook. */
  evaluate: () => Promise<void>;
  /** Sort population hook. */
  sort: () => void;
  /** Mutate population hook. */
  mutate: () => Promise<void>;
  /** Offspring generation hook. */
  getOffspring: () => Promise<GenomeWithMetadata>;
  /** Parent selection hook. */
  selectParent: () => GenomeWithMetadata;
  /** Objective registration hook. */
  registerObjective: (
    key: string,
    direction: 'min' | 'max',
    accessor: (genome: GenomeWithMetadata) => number,
  ) => void;
  /** Ensure a minimum number of hidden nodes in a genome. */
  ensureMinHiddenNodes: (genome: GenomeWithMetadata) => Promise<void>;
  /** Ensure no dead-end connections in a genome. */
  ensureNoDeadEnds: (genome: GenomeWithMetadata) => void;
}
