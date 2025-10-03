/**
 * Speciation options for NEAT speciation controller.
 * Extends NeatOptions with additional fields for compatibility threshold control and species allocation.
 */
export type SpeciationOptions = NeatOptions & {
  compatibilityThreshold?: number;
  minThreshold?: number;
  maxThreshold?: number;
  targetSpecies?: number;
  speciesAllocation?: { extendedHistory?: boolean };
  speciesAgeProtection?: { grace?: number; oldPenalty?: number };
};

/**
 * Rolling statistics tracked for each species between generations.
 * These values inform stagnation heuristics and adaptive controllers.
 */
export interface SpeciesLastStats {
  meanNodes: number;
  meanConns: number;
  best: number;
}

/**
 * Minimal runtime surface required by speciation helpers.
 * Tests and harnesses can narrow the options type via the generic parameter.
 *
 * @template TOptions - Specialised speciation options passed to the helper.
 */
export interface SpeciationHarnessContext<
  TOptions extends SpeciationOptions = SpeciationOptions,
> extends NeatLike {
  population: GenomeDetailed[];
  _species: SpeciesLike[];
  _nextSpeciesId: number;
  generation: number;
  options: TOptions;
  _speciesCreated: Map<number, number>;
  _prevSpeciesMembers: Map<number, Set<number>>;
  _speciesLastStats: Map<number, SpeciesLastStats>;
  _speciesHistory: Array<Record<string, unknown>>;
  _compatIntegral?: number;
  _compatSpeciesEMA?: number;
  _getRNG?: () => () => number;
  _compatibilityDistance: (
    genomeA: GenomeDetailed,
    genomeB: GenomeDetailed,
  ) => number;
  _fallbackInnov: (connection: ConnectionLike) => number;
  _structuralEntropy: (genome: GenomeDetailed) => number;
}
/**
 * Shared lightweight structural types for modular NEAT components.
 *
 * These are deliberately kept small & structural (duck-typed) so that helper
 * modules can interoperate without importing the concrete (heavier) `Neat`
 * class, avoiding circular references while the codebase is being
 * progressively extracted / refactored.
 *
 * Guidelines:
 * - Prefer adding narrowly scoped interfaces instead of widening existing ones.
 * - Avoid leaking implementation details; keep contracts minimal.
 * - Feature‑detect optional telemetry fields – they may be omitted to save cost.
 */

/**
 * Generic map type used as a stop‑gap where the precise shape is still in flux.
 * Prefer a specific interface once the surface stabilises.
 */
export type AnyObj = Record<string, unknown>;

/**
 * Minimal surface every helper currently expects from a NEAT instance while
 * extraction continues. Kept intentionally loose; prefer concrete fields
 * when helpers are stabilised. Represented as a simple record to avoid an
 * empty interface that duplicates its supertype.
 */
export type NeatLike = Record<string, unknown>;

// Objective system ---------------------------------------------------------
/**
 * Descriptor for a single optimisation objective (single or multi‑objective runs).
 *
 * @example Add a maximisation objective for accuracy
 * ```ts
 * const accuracyObj: ObjectiveDescriptor = {
 *   key: 'accuracy',
 *   direction: 'max',
 *   accessor: g => g.score ?? 0
 * };
 * ```
 *
 * @example Add a minimisation objective for network complexity
 * ```ts
 * const complexityObj: ObjectiveDescriptor = {
 *   key: 'complexity',
 *   direction: 'min',
 *   accessor: g => (g.nodes.length + g.connections.length)
 * };
* ```
*/
export interface ObjectiveDescriptor {
  key: string;
  direction: 'max' | 'min';
  accessor: (g: GenomeLike) => number;
}

/**
 * Minimal genome structural surface used by several helpers (incrementally expanded).
 *
 * NOTE: `nodes` & `connections` intentionally remain `any[]` until a stable
 * `NodeLike` / `ConnectionLike` abstraction is finalised.
 */
export interface GenomeLike {
  /** Collection of node objects (structure intentionally opaque for now). */
  /** Collection of node objects (structure intentionally opaque for now). */

  /** Collection of connection objects (structure intentionally opaque for now). */
  connections: unknown[];
  /** Primary fitness / score (convention: higher is better unless objective flips). */
  score?: number;
  /** Number of network input nodes (cached for convenience in some helpers). */
  input?: number;
  /** Number of network output nodes. */
  output?: number;
}

/**
 * Lightweight node representation used by telemetry and structural helpers.
 */
export interface NodeLike {
  geneId?: number;
  [k: string]: unknown;
}

/**
 * Lightweight connection representation used by telemetry and structural helpers.
 */
export interface ConnectionLike {
  from: NodeLike | Record<string, unknown>;
  to: NodeLike | Record<string, unknown>;
  enabled?: boolean;
  /** Optional innovation identifier for tracking historical origin of a connection */
  innovation?: number;
  [k: string]: unknown;
}

/**
 * More concrete genome surface used by telemetry and lineage helpers.
 * Extends the minimal `GenomeLike` with node/connection shapes and a few
 * internal bookkeeping fields used by telemetry.
 */
export interface GenomeDetailed extends GenomeLike {
  nodes: NodeLike[];
  connections: ConnectionLike[];
  _id: number;
  _depth?: number;
  _moRank?: number;
  _parents?: number[];
  [k: string]: unknown;
}

/**
 * Internal species representation used by helpers. Kept minimal and structural.
 */
export interface SpeciesLike {
  id: number;
  members: GenomeDetailed[] | GenomeLike[];
  bestScore?: number;
  lastImproved?: number;
  [k: string]: unknown;
}

/**
 * Options subset used by telemetry helpers. Kept narrow to avoid leaking
 * full runtime options into the helper type surface.
 */
export interface NeatOptions {
  multiObjective?: {
    enabled?: boolean;
    complexityMetric?: 'nodes' | 'connections';
  };
  rngState?: boolean;
  telemetry?: {
    hypervolume?: boolean;
    complexity?: boolean;
    performance?: boolean;
  };
  maxNodes?: number;
  maxConns?: number;
  diversityMetrics?: {
    enabled?: boolean;
    pairSample?: number;
    graphletSample?: number;
  };
  fastMode?: boolean;
  novelty?: { enabled?: boolean; k?: number };
  speciesAllocation?: {
    extendedHistory?: boolean;
    // Add other properties as needed
    [k: string]: unknown;
  };
  /** Soft age protection settings for species (grace generations and penalty) */
  speciesAgeProtection?: {
    grace?: number;
    oldPenalty?: number;
  };
  /** PID-like compatibility adjustment configuration */
  compatAdjust?: {
    smoothingWindow?: number;
    decay?: number;
    kp?: number;
    ki?: number;
    minThreshold?: number;
    maxThreshold?: number;
  };
  /** Automatic coefficient tuning options for compatibility coefficients */
  autoCompatTuning?: {
    enabled?: boolean;
    target?: number;
    adjustRate?: number;
    minCoeff?: number;
    maxCoeff?: number;
  };
  /** Working coefficients used by compatibility distance (may be tuned) */
  excessCoeff?: number;
  disjointCoeff?: number;
  /** Optional target species count used by controllers */
  targetSpecies?: number;
  [k: string]: unknown;
}

/**
 * Diversity statistics captured each generation. Individual fields may be
 * omitted in telemetry output if diversity tracking is partially disabled to
 * reduce runtime cost.
 */
export interface DiversityStats {
  /** Mean pairwise compatibility distance among sampled genomes. */
  meanCompat: number;
  /** Variance of compatibility distance (spread of structural diversity). */
  varCompat: number;
  /** Mean symbolic / activation entropy across networks (higher = more varied). */
  meanEntropy: number;
  /** Variance of entropy values. */
  varEntropy: number;
  /** Entropy derived from graphlet distribution (local motif diversity). */
  graphletEntropy: number;
  /** Mean lineage depth (generational distance from originating ancestor). */
  lineageMeanDepth: number;
  /** Mean pairwise lineage distance (ancestral divergence indicator). */
  lineageMeanPairDist: number;
}

/**
 * Per-generation statistic for a genetic operator.
 *
 * Success is operator‑specific (e.g. produced a structurally valid mutation).
 * A high attempt count with low success can indicate constraints becoming tight
 * (e.g. structural budgets reached) – useful for adaptive operator scheduling.
 *
 * @property op Operator identifier (stable string token).
 * @property succ Successful applications that produced a change.
 * @property att Total attempts (succ <= att). Success rate = succ / att (guard att>0).
 */
export interface OperatorStat {
  op: string;
  succ: number; // success count
  att: number; // attempt count
}
/** Aggregated success / attempt counters over a window or entire run. */
export interface OperatorStatsRecord {
  success: number;
  attempts: number;
}

/**
 * Contribution / dispersion metrics for an objective over a recent window.
 * Used to gauge whether an objective meaningfully influences selection.
 *
 * @property range Difference between max & min observed objective values.
 * @property var Statistical variance across the sampled objective values.
 */
export interface ObjImportanceEntry {
  range: number;
  var: number;
}
/** Map of objective key to its importance metrics (range / variance). */
export interface ObjImportance {
  [key: string]: ObjImportanceEntry;
}
/** Map of objective key to age in generations since introduction. */
export interface ObjAges {
  [key: string]: number;
}
/** Dynamic objective lifecycle event (addition or removal). */
export interface ObjEvent {
  type: 'add' | 'remove';
  key: string;
}
/** Offspring allocation for a species during reproduction. */
export interface SpeciesAlloc {
  id: number;
  alloc: number;
}

/**
 * Snapshot of lineage & ancestry statistics for the current generation.
 *
 * @property parents Parent genome identifiers (e.g. indices / ids) for a focal elite or sample.
 * @property depthBest Depth (generations) of the best genome's lineage path.
 * @property meanDepth Average depth across all genomes (evolutionary age proxy).
 * @property inbreeding Count / score of recent inbreeding detections (heuristic).
 * @property ancestorUniq Jaccard‑style uniqueness proxy of ancestral sets (higher = more unique ancestry).
 */
export interface LineageSnapshot {
  parents: number[];
  depthBest: number;
  meanDepth: number; // average depth across population
  inbreeding: number; // prior generation inbreeding count
  ancestorUniq: number; // Jaccard-based uniqueness proxy
}

/**
 * Aggregate structural complexity metrics capturing size & growth pressure.
 *
 * @property meanNodes Mean number of nodes across population.
 * @property meanConns Mean number of connections across population.
 * @property maxNodes Maximum node count encountered this generation.
 * @property maxConns Maximum connection count encountered this generation.
 * @property meanEnabledRatio Mean proportion of enabled vs total connections.
 * @property growthNodes Net node growth (current mean - previous mean).
 * @property growthConns Net connection growth (current mean - previous mean).
 * @property budgetMaxNodes Node budget ceiling (constraint parameter) at eval time.
 * @property budgetMaxConns Connection budget ceiling at eval time.
 */
export interface ComplexityMetrics {
  meanNodes: number;
  meanConns: number;
  maxNodes: number;
  maxConns: number;
  meanEnabledRatio: number;
  growthNodes: number;
  growthConns: number;
  budgetMaxNodes: number;
  budgetMaxConns: number;
}

/**
 * Timing metrics for coarse evolutionary phases (milliseconds).
 *
 * @property evalMs Time spent evaluating population fitness.
 * @property evolveMs Time spent performing evolutionary operators / reproduction.
 */
export interface PerformanceMetrics {
  evalMs?: number;
  evolveMs?: number;
}

/**
 * Telemetry summary for one generation.
 *
 * Optional properties are feature‑dependent; consumers MUST test for presence.
 *
 * @example
 * ```ts
 * function logSummary(t: TelemetryEntry) {
 *   console.log(`Gen ${t.gen} best=${t.best.toFixed(4)} species=${t.species}`);
 *   if (t.diversity) console.log('Mean compat', t.diversity.meanCompat);
 * }
 * ```
 *
 * @property gen Generation index starting at 0.
 * @property best Best scalar fitness / objective value (primary fitness).
 * @property species Number of extant species.
 * @property hyper Hypervolume proxy (when multi‑objective) or placeholder metric.
 * @property fronts Sizes of first few Pareto fronts (multi‑objective only).
 * @property diversity Diversity statistics (if tracking enabled).
 * @property ops Operator success/attempt counts for this generation.
 * @property objImportance Objective dispersion metrics (may be empty object but never undefined).
 * @property objAges Objective ages in generations.
 * @property objEvents Objective lifecycle events that occurred this generation.
 * @property speciesAlloc Offspring allocation suggestions / results per species.
 * @property objectives Ordered list of objective keys currently active.
 * @property rng Serializable RNG state / seed snapshot.
 * @property lineage Lineage / ancestry snapshot (if enabled).
 * @property hv Rounded hypervolume metric (alternate to `hyper` if both present).
 * @property complexity Structural complexity metrics.
 * @property perf Performance timing metrics.
 */
export interface TelemetryEntry {
  gen: number;
  best: number;
  species: number;
  hyper: number; // hypervolume-like proxy
  // allow additional optional telemetry fields to be attached dynamically
  [k: string]: unknown;
  fronts?: number[]; // first few pareto front sizes when MO enabled
  diversity?: DiversityStats;
  ops: OperatorStat[];
  objImportance: ObjImportance; // always present (may be empty object)
  objAges?: ObjAges;
  objEvents?: ObjEvent[];
  speciesAlloc?: SpeciesAlloc[];
  objectives?: string[];
  rng?: number; // rng state when exported
  lineage?: LineageSnapshot; // only present when lineage tracking enabled
  hv?: number; // optional rounded hypervolume value
  complexity?: ComplexityMetrics;
  perf?: PerformanceMetrics;
}

/**
 * Species statistics at a single historical snapshot (generation boundary).
 *
 * @property id Species identifier.
 * @property size Number of genomes presently in the species.
 * @property bestScore Best fitness achieved by any member so far.
 * @property lastImproved Generations since last improvement (0 = improved this gen).
 */
export interface SpeciesHistoryStat {
  id: number;
  size: number;
  bestScore: number;
  lastImproved: number;
}
/** Species statistics captured for a particular generation. */
export interface SpeciesHistoryEntry {
  generation: number;
  stats: SpeciesHistoryStat[];
}

/**
 * Extended per-species historical snapshot with optional backfilled metrics
 * that may be computed lazily (innovationRange, enabledRatio).
 */
export interface SpeciesHistoryStatExtended extends SpeciesHistoryStat {
  innovationRange?: number;
  enabledRatio?: number;
}
