/**
 * Shared structural contracts used across the NEAT controller chapters.
 *
 * This shared chapter keeps the light-weight, controller-facing type surface in
 * one direct-path location so speciation, telemetry, objectives, species, and
 * tests can import the same contracts without depending on the heavier root
 * `Neat` implementation.
 *
 * The practical value of this boundary is not that it lists many interfaces.
 * Its value is that it gives the rest of the controller a common language.
 * Without that common layer, every extracted chapter would gradually invent its
 * own slightly different idea of what a genome looks like, which option slice
 * matters, or which telemetry record is considered complete enough to share.
 * That kind of drift is subtle but expensive: it makes later chapters harder to
 * read, harder to compose, and harder to trust.
 *
 * Read this file as the controller's common language layer. Most NEAT chapters
 * answer behavioral questions such as how speciation works, what telemetry
 * records, or how objective lists are resolved. This shared surface answers the
 * structural question underneath them: what is the smallest contract each
 * chapter needs in order to cooperate without pulling in the full runtime
 * facade?
 *
 * The contracts cluster into five practical families:
 *
 * 1. host and genome shapes such as `NeatLike`, `GenomeLike`, and
 *    `GenomeDetailed`,
 * 2. objective contracts used by `objectives/` and `multiobjective/`,
 * 3. speciation and species-history contracts used by `speciation/` and
 *    `species/`,
 * 4. telemetry and diagnostics contracts used by `telemetry/`, `diversity/`,
 *    and `lineage/`,
 * 5. small test-facing seams reused by harnesses and extracted helpers.
 *
 * ```mermaid
 * flowchart TD
 *   Shared[shared contract chapter] --> Host[Host and genome vocabulary]
 *   Shared --> Policy[Policy slices and option families]
 *   Shared --> Species[Speciation and species state]
 *   Shared --> Evidence[Telemetry diversity and lineage evidence]
 *   Shared --> TestSeams[Test and helper seams]
 *   Host --> NeatLike[NeatLike GenomeLike GenomeDetailed]
 *   Policy --> Options[NeatOptions SpeciationOptions]
 *   Species --> Registry[SpeciesLike SpeciesLastStats SpeciationHarnessContext]
 *   Evidence --> Telemetry[TelemetryEntry DiversityStats LineageSnapshot]
 * ```
 *
 * Treat the chapter as a reading map rather than a flat glossary. The strongest
 * way to use it is to start with the smallest shared vocabulary, then move into
 * the option families that shape controller policy, and only then descend into
 * the richer evidence contracts used by telemetry and diagnostics.
 *
 * Practical reading order:
 *
 * 1. Start with `NeatLike`, `GenomeLike`, and `GenomeDetailed` to understand
 *    the base structural vocabulary.
 * 2. Continue to `NeatOptions` and `SpeciationOptions` when a helper reads or
 *    rewrites controller policy.
 * 3. Read `TelemetryEntry`, `DiversityStats`, and `LineageSnapshot` when you
 *    want the evidence layer shared across reporting chapters.
 * 4. Finish with species-history and archive contracts when you need the
 *    longer-lived reporting or checkpoint surfaces.
 */

/**
 * Speciation options for the NEAT speciation controller.
 *
 * Extends {@link NeatOptions} with speciation-specific configuration used by:
 * - Compatibility-threshold based species assignment
 * - Adaptive threshold controllers (PID-like)
 * - Species allocation telemetry (history snapshots)
 *
 * This is the shared policy slice that lets speciation helpers stay narrowly
 * typed without importing the full controller configuration object. Read it as
 * the option family that shapes how species are formed, protected, and recorded
 * over time.
 */
export type SpeciationOptions = NeatOptions & {
  /**
   * Compatibility threshold used to decide whether a genome belongs to an
   * existing species.
   *
   * Smaller thresholds create more species (stricter matching). Larger
   * thresholds create fewer species.
   */
  compatibilityThreshold?: number;

  /**
   * Lower bound for compatibility threshold when clamping is enabled.
   */
  minThreshold?: number;

  /**
   * Upper bound for compatibility threshold when clamping is enabled.
   */
  maxThreshold?: number;

  /**
   * Desired (target) number of species used by adaptive controllers.
   */
  targetSpecies?: number;

  /**
   * Speciation allocation and history settings.
   */
  speciesAllocation?: { extendedHistory?: boolean };

  /**
   * Species age protection settings.
   *
   * `grace` controls how long young species are protected.
   * `oldPenalty` is a multiplicative factor applied to older species.
   */
  speciesAgeProtection?: { grace?: number; oldPenalty?: number };
};

/**
 * Rolling statistics tracked for each species between generations.
 * These values inform stagnation heuristics and adaptive controllers.
 *
 * These are controller-maintenance values rather than polished reporting rows.
 * They exist so live speciation logic can remember recent behavior before the
 * `species/` chapter later projects that state into user-facing summaries.
 */
export interface SpeciesLastStats {
  /**
   * Mean number of nodes across the species' members.
   */
  meanNodes: number;

  /**
   * Mean number of connections across the species' members.
   */
  meanConns: number;

  /**
   * Best fitness observed among the species' members.
   */
  best: number;
}

/**
 * Minimal runtime surface required by speciation helpers.
 * Tests and harnesses can narrow the options type via the generic parameter.
 *
 * This host contract is intentionally more specific than `NeatLike` because
 * speciation is one of the most stateful controller phases. The helpers need
 * the live population, the species registry, threshold-tuning accumulators,
 * and the compatibility read itself, but they still do not need the entire
 * public `Neat` facade.
 *
 * @template TOptions - Specialised speciation options passed to the helper.
 */
export interface SpeciationHarnessContext<
  TOptions extends SpeciationOptions = SpeciationOptions,
> extends NeatLike {
  /**
   * Current population for the generation.
   */
  population: GenomeDetailed[];

  /**
   * Current list of extant species.
   */
  _species: SpeciesLike[];

  /**
   * Next species id to assign when creating a new species.
   */
  _nextSpeciesId: number;

  /**
   * Current generation index (starting at 0).
   */
  generation: number;

  /**
   * Current speciation-related options.
   */
  options: TOptions;

  /**
   * Map of species id -> generation when that species was created.
   */
  _speciesCreated: Map<number, number>;

  /**
   * Snapshot of previous generation memberships.
   *
   * Map of species id -> set of genome ids present in the previous generation.
   */
  _prevSpeciesMembers: Map<number, Set<number>>;

  /**
   * Rolling per-species aggregate stats used for telemetry and heuristics.
   */
  _speciesLastStats: Map<number, SpeciesLastStats>;

  /**
   * Species history records (structure depends on configured history format).
   */
  _speciesHistory: Array<Record<string, unknown>>;

  /**
   * Integral accumulator for PID-like compatibility threshold adjustment.
   */
  _compatIntegral?: number;

  /**
   * Exponential moving average of observed species counts (optional telemetry).
   */
  _compatSpeciesEMA?: number;

  /**
   * Optional RNG factory.
   *
   * @returns A function that returns a uniform random number in [0, 1).
   */
  _getRNG?: () => () => number;

  /**
   * Compute the compatibility distance between two genomes.
   *
   * @param genomeA - First genome.
   * @param genomeB - Second genome.
   * @returns Non-negative distance value; smaller means more similar.
   */
  _compatibilityDistance: (
    genomeA: GenomeDetailed,
    genomeB: GenomeDetailed,
  ) => number;

  /**
   * Resolve a fallback innovation id for a connection when `connection.innovation`
   * is missing.
   *
   * @param connection - Connection to extract/derive an innovation identifier from.
   * @returns Numeric innovation identifier.
   */
  _fallbackInnov: (connection: ConnectionLike) => number;

  /**
   * Compute a structural entropy value for a genome.
   *
   * Used by telemetry and diversity/structure summaries.
   *
   * @param genome - Genome to evaluate.
   * @returns Entropy-like scalar; higher typically indicates more varied structure.
   */
  _structuralEntropy: (genome: GenomeDetailed) => number;
}
// Shared contract symbols continue below.

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
 *
 * Treat this as the lowest common denominator host type. Serious helper
 * chapters should usually narrow it quickly into a richer local contract, but
 * keeping a tiny shared base makes extracted utilities easier to compose
 * without inventing a fake monolithic controller interface.
 *
 * Pedagogically, this is the chapter's "start here" host contract. It tells a
 * reader that the shared layer values portability over completeness, and that a
 * richer local host contract should only appear when a chapter can justify why.
 */
export type NeatLike = Record<string, unknown>;

// Objective system ---------------------------------------------------------
/**
 * Descriptor for a single optimisation objective (single or multi‑objective runs).
 *
 * This is the shared currency between the `objectives/` chapter that manages
 * objective lists and the `multiobjective/` chapter that later ranks genomes by
 * those objectives. The contract stays intentionally small: a stable key, a
 * direction, and one accessor that can read the relevant numeric evidence from
 * a genome.
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
  /**
   * Stable identifier for the objective.
   */
  key: string;

  /**
   * Whether the objective should be maximized or minimized.
   */
  direction: 'max' | 'min';

  /**
   * Extract a numeric objective value from a genome.
   *
   * @param g - Genome to evaluate.
   * @returns Scalar objective value.
   */
  accessor: (g: GenomeLike) => number;
}

/**
 * Minimal genome structural surface used by several helpers (incrementally expanded).
 *
 * NOTE: `nodes` and `connections` remain intentionally structural/opaque
 * until a stable public abstraction is finalised.
 *
 * Read this as the portable genome contract. It is enough for helpers that
 * need counts, loose structural access, or a primary score, but not enough to
 * encode every network behavior. More specialized chapters should extend this
 * shape only when they truly need richer node, connection, or lineage detail.
 *
 * That restraint matters because this contract sits near the base of the shared
 * vocabulary. If it grows too eager, every downstream chapter inherits more of
 * the runtime than it actually needs.
 */
export interface GenomeLike {
  /**
   * Collection of node objects (structure intentionally opaque for now).
   */
  nodes: unknown[];

  /**
   * Collection of connection objects (structure intentionally opaque for now).
   */
  connections: unknown[];

  /**
   * Primary fitness/score.
   *
   * Convention: higher is better unless a multi-objective direction flips it.
   */
  score?: number;

  /**
   * Number of network input nodes (cached for convenience in some helpers).
   */
  input?: number;

  /**
   * Number of network output nodes.
   */
  output?: number;
}

/**
 * Lightweight node representation used by telemetry and structural helpers.
 */
export interface NodeLike {
  /**
   * Optional gene identifier.
   *
   * When present, may be used for telemetry or debugging.
   */
  geneId?: number;

  /**
   * Additional implementation-specific properties.
   */
  [k: string]: unknown;
}

/**
 * Lightweight connection representation used by telemetry and structural helpers.
 */
export interface ConnectionLike {
  /**
   * Source node of the connection.
   */
  from: NodeLike | Record<string, unknown>;

  /**
   * Target node of the connection.
   */
  to: NodeLike | Record<string, unknown>;

  /**
   * Whether the connection is enabled.
   */
  enabled?: boolean;

  /**
   * Optional innovation identifier for tracking the historical origin of a
   * connection.
   */
  innovation?: number;

  /**
   * Additional implementation-specific properties.
   */
  [k: string]: unknown;
}

/**
 * More concrete genome surface used by telemetry and lineage helpers.
 * Extends the minimal `GenomeLike` with node/connection shapes and a few
 * internal bookkeeping fields used by telemetry.
 *
 * Use this contract when a helper needs more than anonymous structure. The
 * extra fields here are the ones that repeatedly matter to the read-side NEAT
 * chapters: stable genome identity, parent tracking, lineage depth, and the
 * richer node and connection shapes needed for telemetry, history, and
 * compatibility-adjacent reporting.
 *
 * In chapter terms, `GenomeDetailed` is the point where the shared vocabulary
 * stops being merely structural and becomes historically meaningful. Once a
 * helper needs `_id`, `_parents`, or `_depth`, it is no longer talking about a
 * generic network shape; it is talking about a genome with an evolutionary
 * past that later reporting chapters may want to inspect.
 */
export interface GenomeDetailed extends GenomeLike {
  /**
   * Node list with a stable lightweight shape.
   */
  nodes: NodeLike[];

  /**
   * Connection list with a stable lightweight shape.
   */
  connections: ConnectionLike[];

  /**
   * Unique genome identifier used by speciation/telemetry.
   */
  _id: number;

  /**
   * Optional lineage depth (generations from origin) used by lineage telemetry.
   */
  _depth?: number;

  /**
   * Optional Pareto front rank used by multi-objective selection.
   */
  _moRank?: number;

  /**
   * Optional parent identifiers (genome ids) used by lineage tracing.
   */
  _parents?: number[];

  /**
   * Additional implementation-specific properties.
   */
  [k: string]: unknown;
}

/**
 * Internal species representation used by helpers. Kept minimal and structural.
 *
 * This is the live registry shape behind the stronger `speciation/` and
 * `species/` chapters. It is intentionally not a polished reporting format;
 * instead it holds the small amount of state those chapters repeatedly need to
 * maintain or summarize: membership, best score, improvement timing, and room
 * for implementation-specific bookkeeping.
 */
export interface SpeciesLike {
  /**
   * Unique species identifier.
   */
  id: number;

  /**
   * Member genomes currently assigned to the species.
   */
  members: GenomeDetailed[] | GenomeLike[];

  /**
   * Best fitness achieved by any member so far.
   */
  bestScore?: number;

  /**
   * Generation index when `bestScore` last improved.
   */
  lastImproved?: number;

  /**
   * Additional implementation-specific properties.
   */
  [k: string]: unknown;
}

/**
 * Options subset used by telemetry helpers. Kept narrow to avoid leaking
 * full runtime options into the helper type surface.
 *
 * The purpose of this contract is not to model every `Neat` option. It is to
 * capture the policy knobs that recur across extracted helper chapters:
 * telemetry flags, diversity sampling, compatibility tuning, species history,
 * novelty, and a few structural budget fields. If a helper needs more than
 * this, that is usually a sign it wants a chapter-local host contract instead
 * of a wider shared type.
 *
 * Read this as the shared policy slice, not the complete public options model.
 * It exists so extracted chapters can agree on recurring controller settings
 * without smuggling the full `Neat` facade through every helper signature.
 */
export interface NeatOptions {
  /**
   * Multi-objective configuration.
   */
  multiObjective?: {
    /** Whether multi-objective optimization is enabled. */
    enabled?: boolean;
    /** Complexity metric used by some built-in objectives/telemetry. */
    complexityMetric?: 'nodes' | 'connections';
  };

  /** Whether to store/export RNG state for deterministic replay. */
  rngState?: boolean;

  /** Telemetry feature flags. */
  telemetry?: {
    /** Track/emit hypervolume-like metrics (multi-objective runs). */
    hypervolume?: boolean;
    /** Track/emit complexity metrics. */
    complexity?: boolean;
    /** Track/emit performance timing metrics. */
    performance?: boolean;
  };

  /** Optional hard ceiling for number of nodes. */
  maxNodes?: number;

  /** Optional hard ceiling for number of connections. */
  maxConns?: number;

  /** Diversity metric configuration. */
  diversityMetrics?: {
    /** Whether diversity tracking is enabled. */
    enabled?: boolean;
    /** Sample size for pairwise compatibility distance estimates. */
    pairSample?: number;
    /** Sample size for graphlet-based estimates (local motif diversity). */
    graphletSample?: number;
  };

  /** Enable fast-mode shortcuts (trade accuracy for speed). */
  fastMode?: boolean;

  /** Novelty search configuration. */
  novelty?: { enabled?: boolean; k?: number };

  /** Speciation allocation/history settings. */
  speciesAllocation?: {
    /** When true, record extended per-species history entries. */
    extendedHistory?: boolean;
    // Add other properties as needed
    [k: string]: unknown;
  };

  /**
   * Soft age protection settings for species.
   *
   * `grace` sets the young-species grace period.
   * `oldPenalty` is applied to older species' member scores.
   */
  speciesAgeProtection?: {
    grace?: number;
    oldPenalty?: number;
  };

  /**
   * PID-like compatibility adjustment configuration.
   *
   * These gains/limits may be used to adapt `compatibilityThreshold` over time.
   */
  compatAdjust?: {
    /** Window size for smoothing observed species counts (when used). */
    smoothingWindow?: number;
    /** Optional decay factor for integral/EMA terms (when used). */
    decay?: number;
    /** Proportional gain. */
    kp?: number;
    /** Integral gain. */
    ki?: number;
    /** Lower clamp bound for the threshold. */
    minThreshold?: number;
    /** Upper clamp bound for the threshold. */
    maxThreshold?: number;
  };

  /** Automatic coefficient tuning options for compatibility coefficients. */
  autoCompatTuning?: {
    /** Whether coefficient tuning is enabled. */
    enabled?: boolean;
    /** Target compatibility distance value (or proxy) to aim for. */
    target?: number;
    /** Step size / learning rate for coefficient adjustments. */
    adjustRate?: number;
    /** Lower clamp bound for tuned coefficients. */
    minCoeff?: number;
    /** Upper clamp bound for tuned coefficients. */
    maxCoeff?: number;
  };

  /** Working coefficient used by compatibility distance (may be tuned). */
  excessCoeff?: number;

  /** Working coefficient used by compatibility distance (may be tuned). */
  disjointCoeff?: number;

  /** Optional target species count used by controllers. */
  targetSpecies?: number;

  /**
   * Additional implementation-specific options.
   */
  [k: string]: unknown;
}

/**
 * Diversity statistics captured each generation. Individual fields may be
 * omitted in telemetry output if diversity tracking is partially disabled to
 * reduce runtime cost.
 *
 * This contract is the compact summary that lets telemetry, diagnostics, and
 * dashboards talk about population spread without rerunning the heavier
 * compatibility, entropy, and lineage calculations. The fields are aggregated
 * on purpose: they are meant for trend reading, not for reconstructing every
 * pairwise comparison after the fact.
 *
 * This is one half of the chapter's evidence language. Diversity tells the
 * controller how structurally spread out the population currently looks, which
 * is useful for dashboards, adaptive policies, and regression checks even when
 * the full pairwise evidence would be too expensive to keep around.
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
  /**
   * Operator identifier (stable string token).
   */
  op: string;

  /**
   * Successful applications that produced a change.
   */
  succ: number;

  /**
   * Total attempts (must satisfy `succ <= att`).
   */
  att: number;
}
/** Aggregated success / attempt counters over a window or entire run. */
export interface OperatorStatsRecord {
  /** Total successful operations. */
  success: number;
  /** Total operation attempts. */
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
  /** Difference between max and min observed objective values. */
  range: number;
  /** Statistical variance across the sampled objective values. */
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
/**
 * Dynamic objective lifecycle event (addition or removal).
 *
 * @deprecated Use `ObjectiveEvent` instead.
 */
export type ObjEvent = ObjectiveEvent;
/** Offspring allocation for a species during reproduction. */
export interface SpeciesAlloc {
  /** Species identifier. */
  id: number;
  /** Allocated offspring count/weight for the species. */
  alloc: number;
}

/**
 * Snapshot of lineage & ancestry statistics for the current generation.
 *
 * This is the ancestry companion to `DiversityStats`. Diversity answers how far
 * apart genomes currently look; lineage answers how much recent family overlap
 * still exists beneath those structures. Telemetry and adaptive controllers use
 * both because structural spread and family spread are related but not
 * interchangeable signals.
 *
 * Read `LineageSnapshot` and `DiversityStats` together. One describes present
 * structural variety, the other describes how independent that variety really
 * is. NEAT benefits from both views because a population can look diverse while
 * still clustering around a narrow recent ancestry.
 *
 * @property parents Parent genome identifiers (e.g. indices / ids) for a focal elite or sample.
 * @property depthBest Depth (generations) of the best genome's lineage path.
 * @property meanDepth Average depth across all genomes (evolutionary age proxy).
 * @property inbreeding Count / score of recent inbreeding detections (heuristic).
 * @property ancestorUniq Jaccard‑style uniqueness proxy of ancestral sets (higher = more unique ancestry).
 */
export interface LineageSnapshot {
  /** Parent genome identifiers for a focal elite or sample. */
  parents: number[];
  /** Depth (generations) of the best genome's lineage path. */
  depthBest: number;
  /** Average lineage depth across the population (evolutionary age proxy). */
  meanDepth: number;
  /** Count/score of recent inbreeding detections (heuristic). */
  inbreeding: number;
  /** Jaccard-style uniqueness proxy of ancestral sets (higher = more unique). */
  ancestorUniq: number;
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
  /** Mean number of nodes across the population. */
  meanNodes: number;
  /** Mean number of connections across the population. */
  meanConns: number;
  /** Maximum node count encountered this generation. */
  maxNodes: number;
  /** Maximum connection count encountered this generation. */
  maxConns: number;
  /** Mean proportion of enabled vs total connections. */
  meanEnabledRatio: number;
  /** Net node growth (current mean - previous mean). */
  growthNodes: number;
  /** Net connection growth (current mean - previous mean). */
  growthConns: number;
  /** Node budget ceiling (constraint parameter) at evaluation time. */
  budgetMaxNodes: number;
  /** Connection budget ceiling (constraint parameter) at evaluation time. */
  budgetMaxConns: number;
}

/**
 * Timing metrics for coarse evolutionary phases (milliseconds).
 *
 * These timings are intentionally coarse. They exist to reveal where a run is
 * spending time at the chapter level, not to replace a profiler. That makes
 * them useful inside telemetry exports and regression-oriented dashboards where
 * simple evaluation-versus-evolution trends are more valuable than exhaustive
 * trace detail.
 *
 * @property evalMs Time spent evaluating population fitness.
 * @property evolveMs Time spent performing evolutionary operators / reproduction.
 */
export interface PerformanceMetrics {
  /** Time spent evaluating population fitness (ms). */
  evalMs?: number;
  /** Time spent performing evolutionary operators/reproduction (ms). */
  evolveMs?: number;
}

/**
 * Telemetry summary for one generation.
 *
 * Optional properties are feature‑dependent; consumers MUST test for presence.
 *
 * This is the shared "one generation of evidence" contract behind the
 * telemetry subtree. Instead of forcing callers to stitch together diversity,
 * lineage, operator stats, objectives, complexity, and performance from many
 * different buffers, the recorder and export helpers fold those signals into
 * one generation-stamped summary row.
 *
 * In the shared chapter, this is the fullest evidence contract. Earlier types
 * define the vocabulary; `TelemetryEntry` shows how that vocabulary is folded
 * into one readable generation snapshot that exporters, dashboards, tests, and
 * diagnostics can all share.
 *
 * Read the fields in families:
 * - run position and headline outcome: `gen`, `best`, `species`, `hyper`
 * - diversity and lineage evidence: `diversity`, `lineage`
 * - objective and Pareto context: `objectives`, `objImportance`, `objAges`, `objEvents`, `fronts`
 * - operator and runtime diagnostics: `ops`, `complexity`, `perf`, `rng`
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
  /** Generation index starting at 0. */
  gen: number;
  /** Best scalar fitness/objective value for the generation. */
  best: number;
  /** Number of extant species. */
  species: number;

  /** Hypervolume-like proxy metric (multi-objective runs) or placeholder metric. */
  hyper: number;

  /** Allow additional optional telemetry fields to be attached dynamically. */
  [k: string]: unknown;

  /** Sizes of the first few Pareto fronts (multi-objective only). */
  fronts?: number[];

  /** Diversity statistics (if tracking enabled). */
  diversity?: DiversityStats;

  /** Operator success/attempt counts for this generation. */
  ops: OperatorStat[];

  /** Objective dispersion metrics (always present; may be empty object). */
  objImportance: ObjImportance;

  /** Objective ages in generations. */
  objAges?: ObjAges;

  /** Objective lifecycle events that occurred this generation. */
  objEvents?: ObjectiveEvent[];

  /** Offspring allocation suggestions/results per species. */
  speciesAlloc?: SpeciesAlloc[];

  /** Ordered list of objective keys currently active. */
  objectives?: string[];

  /** Serializable RNG state/seed snapshot (when exported). */
  rng?: number;

  /** Lineage/ancestry snapshot (if enabled). */
  lineage?: LineageSnapshot;

  /** Optional rounded hypervolume value (alternate to `hyper` if both present). */
  hv?: number;

  /** Structural complexity metrics. */
  complexity?: ComplexityMetrics;

  /** Performance timing metrics. */
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
  /** Species identifier. */
  id: number;
  /** Number of genomes presently in the species. */
  size: number;
  /** Best fitness achieved by any member so far. */
  bestScore: number;
  /** Generations since last improvement (0 = improved this generation). */
  lastImproved: number;
}
/**
 * Species statistics captured for a particular generation.
 *
 * This is the history-buffer unit used by the `species/` chapter. One entry
 * answers "what did the species table look like at this generation boundary?"
 * without requiring callers to inspect the live registry directly.
 */
export interface SpeciesHistoryEntry {
  /** Generation index for this snapshot. */
  generation: number;
  /** Per-species stats captured at the generation boundary. */
  stats: SpeciesHistoryStat[];
}

/**
 * Extended per-species historical snapshot with optional backfilled metrics
 * that may be computed lazily (innovationRange, enabledRatio).
 */
export interface SpeciesHistoryStatExtended extends SpeciesHistoryStat {
  /** Range of innovation ids observed among members (max - min), if computed. */
  innovationRange?: number;
  /** Ratio of enabled to total connections among members, if computed. */
  enabledRatio?: number;
}

/**
 * Pareto archive entry capturing a genome plus its objective values.
 *
 * This contract is intentionally sparse because the archive is primarily a
 * preservation boundary. The stored genome captures the candidate itself, while
 * `objectives` preserves the exact multi-objective evidence that justified its
 * place on the frontier at the time it was archived.
 */
export interface ParetoArchiveEntry {
  /** Representative genome for the archive entry. */
  genome: GenomeLike;
  /** Objective values for the stored genome, in objective-list order. */
  objectives: number[];

  /** Additional implementation-specific metadata. */
  [key: string]: unknown;
}

/**
 * Objective add/remove lifecycle event for telemetry and auditing.
 */
export interface ObjectiveEvent {
  /** Generation index where the event occurred. */
  gen: number;
  /** Whether the objective was added or removed. */
  type: 'add' | 'remove';
  /** Objective key affected by the event. */
  key: string;
}
