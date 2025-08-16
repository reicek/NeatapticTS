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
export type AnyObj = { [k: string]: any };

/**
 * Minimal surface every helper currently expects from a NEAT instance while
 * extraction continues. At present it carries no guaranteed properties besides
 * an index signature. As helpers converge, promote concrete, documented fields.
 */
export interface NeatLike extends AnyObj {}

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
 *
 * @property key Stable identifier (used as a key in telemetry & objective maps).
 * @property direction Optimisation direction – maximise ("max") or minimise ("min").
 * @property accessor Pure extractor returning a scalar value for the objective.
 */
export interface ObjectiveDescriptor {
  key: string;
  direction: 'max' | 'min';
  accessor: (g: GenomeLike) => number; // narrowed from any
}

/**
 * Minimal genome structural surface used by several helpers (incrementally expanded).
 *
 * NOTE: `nodes` & `connections` intentionally remain `any[]` until a stable
 * `NodeLike` / `ConnectionLike` abstraction is finalised.
 */
export interface GenomeLike {
  /** Collection of node objects (structure intentionally opaque for now). */
  nodes: any[]; // will refine with NodeLike
  /** Collection of connection objects (structure intentionally opaque for now). */
  connections: any[];
  /** Primary fitness / score (convention: higher is better unless objective flips). */
  score?: number;
  /** Number of network input nodes (cached for convenience in some helpers). */
  input?: number;
  /** Number of network output nodes. */
  output?: number;
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
