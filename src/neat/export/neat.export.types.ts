import type {
  InnovationTracker,
  InnovationTrackerJSON,
} from '../innovation-tracker/innovation-tracker.types';
import type {
  SpeciesHistoryEntry,
  SpeciesLastStats,
} from '../shared/neat.shared.types';

/** Format version used when a checkpoint predates explicit version tags. */
export const LEGACY_CHECKPOINT_FORMAT_VERSION = 0;

/** Format version for controller-only meta checkpoints. */
export const CURRENT_META_FORMAT_VERSION = 1;

/** Format version for strict full-state checkpoint bundles. */
export const CURRENT_STATE_FORMAT_VERSION = 1;

/** Checkpoint mode marker for strict full-resume bundles. */
export const FULL_CHECKPOINT_MODE = 'full' as const;

/**
 * JSON representation of an individual genome (network). The concrete shape is
 * produced by `Network#toJSON()` and re-hydrated via `Network.fromJSON()`. The
 * export chapter may also attach one `controllerMeta` object alongside that
 * network payload so full checkpoints can preserve stable genome ids, lineage,
 * and evaluation-side annotations without widening the network serializer.
 *
 * Treat this as a persistence boundary rather than a strict schema promise. The
 * export helpers preserve whatever `Network#toJSON()` emits, which lets the
 * broader architecture evolve without forcing this chapter to hard-code every
 * possible serialized field, while the reserved `controllerMeta` pocket keeps
 * controller-owned resume data explicit and versionable.
 */
export interface GenomeJSON {
  /** Optional controller-owned metadata captured alongside the network JSON. */
  controllerMeta?: GenomeControllerMetaJSON;
  [key: string]: unknown;
}

/**
 * Controller-owned genome metadata preserved in exported population snapshots.
 *
 * The runtime `Network` JSON only describes the structural graph. A resumed
 * NEAT run also needs controller annotations such as stable genome ids,
 * lineage, rank-side metrics, and any cached score evidence that should still
 * exist immediately after restore.
 */
export interface GenomeControllerMetaJSON {
  /** Fitness score assigned before export, when present. */
  score?: number;
  /** Stable genome id used by species membership and lineage reads. */
  genomeId?: number;
  /** Optional network-local RNG state for deterministic genome-level mutation replay. */
  networkRngState?: number;
  /** Shared-fitness value cached during speciation. */
  sharedFitness?: number;
  /** Crowding distance used by multi-objective ranking. */
  crowdingDistance?: number;
  /** Pareto front rank captured during multi-objective selection. */
  frontRank?: number;
  /** Cached structural entropy, when the run computed it. */
  structuralEntropy?: number;
  /** Multi-objective rank shortcut used by some helpers. */
  multiObjectiveRank?: number;
  /** Multi-objective crowding metric. */
  multiObjectiveCrowding?: number;
  /** Optional lineage parent ids. */
  parents?: number[];
  /** Optional lineage depth. */
  depth?: number;
  /** Connection re-enable probability attached to the genome. */
  reenableProb?: number;
  /** Disabled-connection re-enable successes recorded this generation. */
  reenableSuccess?: number;
  /** Disabled-connection re-enable attempts recorded this generation. */
  reenableAttempts?: number;
  /** Compatibility/history fallback mode for deliberately partial genomes. */
  compatInnovationMode?: 'require-explicit' | 'allow-fallback';
  /** Optional novelty score from the last evaluation pass. */
  novelty?: number;
}

/**
 * Controller runtime state that does not depend on the live population object
 * graph and can therefore travel through the meta-only checkpoint path.
 */
export interface NeatRuntimeMetaJSON {
  /**
   * Next genome id reserved for newly created children or imports.
   *
   * Stable genome ids are used by full-checkpoint speciation restore. They are
   * also the glue that lets history and telemetry surfaces rebind the species
   * registry onto freshly rehydrated network instances.
   */
  nextGenomeId?: number;
  /**
   * Next connection innovation counter used by architecture-level edge construction.
   *
   * Deterministic replay requires that the monotonic innovation cursor resumes
   * from the exact same value after restore. Otherwise the very next structural
   * mutation may allocate different innovation numbers and diverge the run.
   */
  nextConnectionInnovation?: number;
  /**
   * Next stable node gene id used by architecture-level node construction.
   *
   * Node gene ids are part of the proper-NEAT identity contract. The counter is
   * stored here so restored runs do not accidentally reuse a previously
   * assigned id.
   */
  nextNodeGeneId?: number;
  /**
   * Next runtime node index used by endpoint bookkeeping.
   *
   * Runtime node indexes are used by several legacy-adjacent surfaces (for
   * example connection endpoint indices) even when the canonical identity key
   * is the node gene id. Persisting the index allocator keeps restore
   * deterministic and prevents accidental collisions.
   */
  nextNodeIndex?: number;
  /** Whether lineage metadata is currently active for the controller. */
  lineageEnabled?: boolean;
  /**
   * Opaque controller RNG state used to resume the same future random stream.
   *
   * This is the controller-owned random stream that should continue advancing
   * deterministically after import. The full replay promise assumes this state
   * is restored before selection, mutation, and crossover consume randomness.
   */
  rngState?: number;
  /** Last observed inbreeding count retained for telemetry. */
  lastInbreedingCount?: number;
  /** Generation index of the last global improvement checkpoint. */
  lastGlobalImproveGeneration?: number;
  /** Rolling species-history archive used by read-side telemetry. */
  speciesHistory?: SpeciesHistoryEntry[];
}

/**
 * Species registry row captured inside a full checkpoint.
 *
 * Species state points back to population genomes by stable genome id so the
 * import path can rebind the live species registry onto the freshly restored
 * `Network` instances.
 */
export interface SpeciesCheckpointJSON {
  /** Stable species id. */
  id: number;
  /** Member genome ids currently assigned to the species. */
  memberGenomeIds: number[];
  /** Representative genome id used for the next assignment pass. */
  representativeGenomeId?: number;
  /** Serialized representative anchor when it no longer lives in the current population. */
  representativeGenome?: GenomeJSON;
  /** Best score observed for the species so far. */
  bestScore?: number;
  /** Generation index when the species last improved. */
  lastImproved?: number;
  /** Shared-fitness aggregate kept for allocation/telemetry. */
  sharedFitness?: number;
  /** Average shared fitness across the species' current members. */
  avgSharedFitness?: number;
  /** Offspring allocation snapshot when available. */
  offspring?: number;
  /** Optional creation-generation style metadata from richer species records. */
  generation?: number;
}

/**
 * Speciation-specific checkpoint state required by the full resume path.
 */
export interface SpeciationCheckpointJSON {
  /** Next species id reserved for future new-species allocation. */
  nextSpeciesId?: number;
  /** Live species registry, stored by stable genome id references. */
  species?: SpeciesCheckpointJSON[];
  /** Species creation generations used by age protection. */
  speciesCreated?: Array<[number, number]>;
  /** Previous-generation member ids used by continuity-aware reads. */
  prevSpeciesMembers?: Array<[number, number[]]>;
  /** Rolling per-species aggregate stats. */
  speciesLastStats?: Array<[number, SpeciesLastStats]>;
  /** Compatibility-threshold integral accumulator. */
  compatIntegral?: number;
  /** Optional EMA of observed species counts. */
  compatSpeciesEMA?: number;
}

/**
 * Serialized meta information describing a NEAT run, excluding the concrete
 * population genomes. This allows you to persist and resume experiment context
 * without committing to a particular population snapshot, while still carrying
 * controller counters and history that do not depend on live genome instances.
 */
export interface NeatMetaJSON {
  /** Meta checkpoint format version. Missing means legacy pre-versioned export. */
  formatVersion?: number;
  /** Number of input nodes expected by evolved networks. */
  input: number;
  /** Number of output nodes produced by evolved networks. */
  output: number;
  /** Current evolutionary generation index (0-based). */
  generation: number;
  /** Full options object (hyper-parameters) used to configure NEAT. */
  options: Record<string, unknown>;
  /** Innovation tracker payload for deterministic structural mutation resume. */
  innovationTracker: InnovationTrackerJSON;
  /** Controller runtime state that can travel without the population array. */
  runtime?: NeatRuntimeMetaJSON;
}

/**
 * Genome with `toJSON()` serialization method.
 *
 * This is the smallest runtime contract needed by the export helpers when they
 * only care about turning one genome into a JSON payload.
 */
export interface GenomeWithSerialization {
  toJSON: () => GenomeJSON;
}

/**
 * Internal genome view combining network serialization with controller-owned
 * metadata used by export and restore helpers.
 */
export type GenomeControllerCarrier = GenomeWithSerialization & {
  getRNGState?: () => number | undefined;
  setRNGState?: (state: number) => void;
  score?: number;
  _id?: number;
  _sharedFitness?: number;
  _crowdingDistance?: number;
  _frontRank?: number;
  _structuralEntropy?: number;
  _moRank?: number;
  _moCrowd?: number;
  _parents?: number[];
  _depth?: number;
  _reenableProb?: number;
  _reenableSuccess?: number;
  _reenableAttempts?: number;
  _compatInnovationMode?: 'require-explicit' | 'allow-fallback';
  _novelty?: number;
};

/**
 * Internal species row shape used while serializing and restoring checkpoints.
 */
export type SpeciesControllerCarrier = {
  id: number;
  members: GenomeControllerCarrier[];
  representative?: GenomeControllerCarrier;
  bestScore?: number;
  lastImproved?: number;
  sharedFitness?: number;
  avgSharedFitness?: number;
  offspring?: number;
  generation?: number;
};

/**
 * NEAT controller interface for export operations.
 *
 * The persistence helpers intentionally depend on this narrow host shape instead
 * of the concrete `Neat` class. That keeps export and restore logic reusable in
 * tests and static-style helper flows without coupling the file to the full
 * controller implementation.
 */
export interface NeatControllerForExport {
  input: number;
  output: number;
  generation: number;
  options: Record<string, unknown> & {
    popsize?: number;
    genomeExtensions?: {
      connectionGain?: boolean;
      nodeResponse?: boolean;
      disabledConnectionReenableProbability?: boolean;
      [k: string]: unknown;
    };
    rng?: () => number;
    seed?: unknown;
  };
  population: GenomeControllerCarrier[];
  _rng?: () => number;
  _rngState?: number;
  _innovationTracker: InnovationTracker;
  _nextGenomeId?: number;
  _lineageEnabled?: boolean;
  _lastInbreedingCount?: number;
  _lastGlobalImproveGeneration?: number;
  _speciesHistory?: SpeciesHistoryEntry[];
  _species?: SpeciesControllerCarrier[];
  _nextSpeciesId?: number;
  _speciesCreated?: Map<number, number>;
  _prevSpeciesMembers?: Map<number, Set<number>>;
  _speciesLastStats?: Map<number, SpeciesLastStats>;
  _compatIntegral?: number;
  _compatSpeciesEMA?: number;
}

/**
 * Network class with static `fromJSON()` method.
 *
 * Import helpers use this contract when rebuilding genomes from serialized JSON
 * without needing to know the concrete network implementation details.
 */
export interface NetworkClass {
  fromJSON: (json: Record<string, unknown>) => GenomeControllerCarrier;
}

/**
 * NEAT class constructor interface.
 *
 * Static-style restore helpers depend on this constructor shape so they can
 * rebuild a controller instance from persisted meta data and then optionally
 * rehydrate the population.
 */
export interface NeatConstructor {
  new (
    input: number,
    output: number,
    fitness: (network: GenomeWithSerialization) => number | Promise<number>,
    options?: Record<string, unknown>,
  ): NeatControllerForExport;
  fromJSON?: (
    meta: NeatMetaJSON,
    fitness: (network: GenomeWithSerialization) => number | Promise<number>,
  ) => NeatControllerForExport;
}

/**
 * Top-level bundle containing both NEAT meta information and the full array of
 * serialized genomes (population). This is what you get from `exportState()` and
 * feed into `importStateImpl()` to resume exactly where you left off.
 *
 * If `NeatMetaJSON` is the controller checkpoint and `GenomeJSON[]` is the pool
 * of candidate solutions, `NeatStateJSON` is the combined pause-and-resume
 * artifact that preserves both layers together plus the species-side runtime
 * state needed by the strict full-checkpoint restore path.
 */
export interface NeatStateJSON {
  /** Full checkpoint format version. Missing means legacy pre-versioned export. */
  formatVersion?: number;
  /** Checkpoint mode marker for the strict full-resume path. */
  checkpointMode?: typeof FULL_CHECKPOINT_MODE;
  /** Serialized NEAT meta (innovation history, generation, options, etc.). */
  neat: NeatMetaJSON;
  /** Array of serialized genomes representing the current population. */
  population: GenomeJSON[];
  /** Species and threshold state needed by the full checkpoint path. */
  speciation?: SpeciationCheckpointJSON;
}
