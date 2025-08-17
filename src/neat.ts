import Network from './architecture/network';
import type {
  TelemetryEntry,
  ObjectiveDescriptor,
  SpeciesHistoryEntry,
  OperatorStatsRecord,
} from './neat/neat.types';
import * as methods from './methods/methods';
import { selection as selectionMethods } from './methods/selection';
import NodeType from './architecture/node'; // Import the Node type with a different name to avoid conflicts
// Static imports (post-migration from runtime require delegates)
import {
  ensureMinHiddenNodes,
  selectMutationMethod,
  ensureNoDeadEnds,
  mutate,
  mutateAddNodeReuse,
  mutateAddConnReuse,
} from './neat/neat.mutation';
import { evolve } from './neat/neat.evolve';
import { evaluate } from './neat/neat.evaluate';
import { createPool, spawnFromParent, addGenome } from './neat/neat.helpers';
import {
  _getObjectives,
  registerObjective,
  clearObjectives,
} from './neat/neat.objectives';
import {
  computeDiversityStats,
  structuralEntropy,
} from './neat/neat.diversity';
import { fastNonDominated } from './neat/neat.multiobjective';
import { _fallbackInnov, _compatibilityDistance } from './neat/neat.compat';
import {
  _speciate,
  _applyFitnessSharing,
  _sortSpeciesMembers,
  _updateSpeciesStagnation,
} from './neat/neat.speciation';
import { getSpeciesStats, getSpeciesHistory } from './neat/neat.species';
import {
  exportTelemetryJSONL,
  exportTelemetryCSV,
  exportSpeciesHistoryCSV,
} from './neat/neat.telemetry.exports';
import { sort, getParent, getFittest, getAverage } from './neat/neat.selection';
import {
  exportPopulation,
  importPopulation,
  exportState,
  importStateImpl,
  toJSONImpl,
  fromJSONImpl,
} from './neat/neat.export';

/**
 * Configuration options for Neat evolutionary runs.
 *
 * Each property is optional and the class applies sensible defaults when a
 * field is not provided. Options control population size, mutation rates,
 * compatibility coefficients, selection strategy and other behavioral knobs.
 *
 * Example:
 * const opts: NeatOptions = { popsize: 100, mutationRate: 0.5 };
 * const neat = new Neat(3, 1, fitnessFn, opts);
 *
 * Note: this type is intentionally permissive to support staged migration and
 * legacy callers; prefer providing a typed options object where possible.
 */
type Options = { [k: string]: any };
// Public re-export for library consumers
export type NeatOptions = Options;
export default class Neat {
  input: number;
  output: number;
  fitness: (network: Network) => number;
  options: Options;
  population: Network[] = [];
  generation: number = 0;
  // Deterministic RNG state (lazy init)
  /**
   * Internal numeric state for the deterministic xorshift RNG when no user RNG
   * is provided. Stored as a 32-bit unsigned integer.
   */
  private _rngState?: number;
  /**
   * Cached RNG function; created lazily and seeded from `_rngState` when used.
   */
  private _rng?: () => number;
  // Internal bookkeeping and caches (kept permissive during staggered migration)
  /** Array of current species (internal representation). */
  private _species: any[] = [];
  /** Operator statistics used by adaptive operator selection. */
  private _operatorStats: Map<string, OperatorStatsRecord> = new Map();
  /** Map of node-split innovations used to reuse innovation ids for node splits. */
  private _nodeSplitInnovations: Map<string, any> = new Map();
  /** Map of connection innovations keyed by a string identifier. */
  private _connInnovations: Map<string, number> = new Map();
  /** Counter for issuing global innovation numbers when explicit numbers are used. */
  private _nextGlobalInnovation: number = 1;
  /** Counter for assigning unique genome ids. */
  private _nextGenomeId: number = 1;
  /** Whether lineage metadata should be recorded on genomes. */
  private _lineageEnabled: boolean = false;
  /** Last observed count of inbreeding (used for detecting excessive cloning). */
  private _lastInbreedingCount: number = 0;
  /** Previous inbreeding count snapshot. */
  private _prevInbreedingCount: number = 0;
  /** Optional phase marker for multi-stage experiments. */
  private _phase?: string;
  /** Telemetry buffer storing diagnostic snapshots per generation. */
  private _telemetry: any[] = [];
  /** Map of species id -> set of member genome ids from previous generation. */
  private _prevSpeciesMembers: Map<number, Set<number>> = new Map();
  /** Last recorded stats per species id. */
  private _speciesLastStats: Map<number, any> = new Map();
  /** Time-series history of species stats (for exports/telemetry). */
  private _speciesHistory: any[] = [];
  /** Archive of Pareto front metadata for multi-objective tracking. */
  private _paretoArchive: any[] = [];
  /** Archive storing Pareto objectives snapshots. */
  private _paretoObjectivesArchive: any[] = [];
  /** Novelty archive used by novelty search (behavior representatives). */
  private _noveltyArchive: any[] = [];
  /** Map tracking stale counts for objectives by key. */
  private _objectiveStale: Map<string, number> = new Map();
  /** Map tracking ages for objectives by key. */
  private _objectiveAges: Map<string, number> = new Map();
  /** Queue of recent objective activation/deactivation events for telemetry. */
  private _objectiveEvents: any[] = [];
  /** Pending objective keys to add during safe phases. */
  private _pendingObjectiveAdds: string[] = [];
  /** Pending objective keys to remove during safe phases. */
  private _pendingObjectiveRemoves: string[] = [];
  /** Last allocated offspring set (used by adaptive allocators). */
  private _lastOffspringAlloc?: any[];
  /** Adaptive prune level for complexity control (optional). */
  private _adaptivePruneLevel?: number;
  /** Duration of the last evaluation run (ms). */
  private _lastEvalDuration?: number;
  /** Duration of the last evolve run (ms). */
  private _lastEvolveDuration?: number;
  /** Cached diversity metrics (computed lazily). */
  private _diversityStats?: any;
  /** Cached list of registered objectives. */
  private _objectivesList?: any[];
  /** Generation index where the last global improvement occurred. */
  private _lastGlobalImproveGeneration: number = 0;
  /** Best score observed in the last generation (used for improvement detection). */
  private _bestScoreLastGen?: number;
  // Speciation controller state
  /** Map of speciesId -> creation generation for bookkeeping. */
  private _speciesCreated: Map<number, number> = new Map();
  /** Exponential moving average for compatibility threshold (adaptive speciation). */
  private _compatSpeciesEMA?: number;
  /** Integral accumulator used by adaptive compatibility controllers. */
  private _compatIntegral: number = 0;
  /** Generation when epsilon compatibility was last adjusted. */
  private _lastEpsilonAdjustGen: number = -Infinity;
  /** Generation when ancestor uniqueness adjustment was last applied. */
  private _lastAncestorUniqAdjustGen: number = -Infinity;
  // Adaptive minimal criterion & complexity
  /** Adaptive minimal criterion threshold (optional). */
  private _mcThreshold?: number;

  // Lightweight RNG accessor used throughout migrated modules
  private _getRNG(): () => number {
    if (!this._rng) {
      // Allow user-provided RNG in options for deterministic tests
      const optRng = (this.options as any)?.rng;
      if (typeof optRng === 'function') this._rng = optRng;
      else {
        // Deterministic xorshift32 seeded by _rngState; if absent initialize lazily
        if (this._rngState === undefined) {
          // initialize with a non-zero seed derived from time & population length for variability
          let seed =
            (Date.now() ^ ((this.population.length + 1) * 0x9e3779b1)) >>> 0;
          if (seed === 0) seed = 0x1a2b3c4d;
          this._rngState = seed >>> 0;
        }
        this._rng = () => {
          // xorshift32
          let x = this._rngState! >>> 0;
          x ^= x << 13;
          x >>>= 0;
          x ^= x >> 17;
          x >>>= 0;
          x ^= x << 5;
          x >>>= 0;
          this._rngState = x >>> 0;
          return (x >>> 0) / 0xffffffff;
        };
      }
    }
    return this._rng!;
  }
  // Delegate ensureMinHiddenNodes to migrated mutation helper for smaller class surface
  /**
   * Ensure a network has the minimum number of hidden nodes according to
   * configured policy. Delegates to migrated helper implementation.
   *
   * @param network Network instance to adjust.
   * @param multiplierOverride Optional multiplier to override configured policy.
   */
  ensureMinHiddenNodes(network: Network, multiplierOverride?: number) {
    return ensureMinHiddenNodes.call(this as any, network, multiplierOverride);
  }
  /**
   * Construct a new Neat instance.
   * Kept permissive during staged migration; accepts the same signature tests expect.
   *
   * @example
   * // Create a neat instance for 3 inputs and 1 output with default options
   * const neat = new Neat(3, 1, (net) => evaluateFitness(net));
   */
  constructor(
    input?: number,
    output?: number,
    fitness?: any,
    options: any = {}
  ) {
    // Assign basic fields; other internals are initialized above as class fields
    this.input = input ?? 0;
    this.output = output ?? 0;
    this.fitness = fitness ?? ((n: Network) => 0);
    this.options = options || {};
    // --- Default option hydration (only assign when undefined to respect caller overrides) ---
    const opts: any = this.options;
    // Core sizes / rates
    if (opts.popsize === undefined) opts.popsize = 50;
    if (opts.elitism === undefined) opts.elitism = 0;
    if (opts.provenance === undefined) opts.provenance = 0;
    if (opts.mutationRate === undefined) opts.mutationRate = 0.7; // tests expect 0.7
    if (opts.mutationAmount === undefined) opts.mutationAmount = 1;
    if (opts.fitnessPopulation === undefined) opts.fitnessPopulation = false;
    if (opts.clear === undefined) opts.clear = false;
    if (opts.equal === undefined) opts.equal = false;
    if (opts.compatibilityThreshold === undefined)
      opts.compatibilityThreshold = 3;
    // Structural caps
    if (opts.maxNodes === undefined) opts.maxNodes = Infinity;
    if (opts.maxConns === undefined) opts.maxConns = Infinity;
    if (opts.maxGates === undefined) opts.maxGates = Infinity;
    // Compatibility distance coefficients
    if (opts.excessCoeff === undefined) opts.excessCoeff = 1;
    if (opts.disjointCoeff === undefined) opts.disjointCoeff = 1;
    if (opts.weightDiffCoeff === undefined) opts.weightDiffCoeff = 0.5;
    // Mutation list default (shallow copy so tests can check identity scenarios)
    if (opts.mutation === undefined)
      opts.mutation = methods.mutation.ALL
        ? methods.mutation.ALL.slice()
        : methods.mutation.FFW
        ? [methods.mutation.FFW]
        : [];
    // Selection method defaults
    if (opts.selection === undefined) {
      // prefer dedicated selection module; fallback to methods.selection if legacy export
      opts.selection =
        (selectionMethods && selectionMethods.TOURNAMENT) ||
        (methods as any).selection?.TOURNAMENT ||
        selectionMethods.FITNESS_PROPORTIONATE;
    }
    if (opts.crossover === undefined)
      opts.crossover = methods.crossover
        ? methods.crossover.SINGLE_POINT
        : undefined;
    // Novelty archive defaults
    if (opts.novelty === undefined) opts.novelty = { enabled: false };
    // Diversity metrics container
    if (opts.diversityMetrics === undefined)
      opts.diversityMetrics = { enabled: true };
    // fastMode auto defaults
    if (opts.fastMode && opts.diversityMetrics) {
      if (opts.diversityMetrics.pairSample == null)
        opts.diversityMetrics.pairSample = 20;
      if (opts.diversityMetrics.graphletSample == null)
        opts.diversityMetrics.graphletSample = 30;
      if (opts.novelty?.enabled && opts.novelty.k == null) opts.novelty.k = 5;
    }
    // Initialize novelty archive backing array for size accessor
    (this as any)._noveltyArchive = [];
    // Speciation defaults
    if (opts.speciation === undefined) opts.speciation = false;
    // Objective system container
    if (
      opts.multiObjective &&
      opts.multiObjective.enabled &&
      !Array.isArray(opts.multiObjective.objectives)
    )
      opts.multiObjective.objectives = [];
    // Ensure population initialization consistent with original behavior
    this.population = this.population || [];
    // If a network or population seed provided, create initial pool
    try {
      if ((this.options as any).network !== undefined)
        this.createPool((this.options as any).network);
      else if ((this.options as any).popsize) this.createPool(null);
    } catch {}
    // Enable lineage tracking if requested via options
    if (
      (this.options as any).lineage?.enabled ||
      (this.options as any).provenance > 0
    )
      this._lineageEnabled = true;
    // Backwards compat: some tests use `lineageTracking` boolean option
    if ((this.options as any).lineageTracking === true)
      this._lineageEnabled = true;
    if (options.lineagePressure?.enabled && this._lineageEnabled !== true) {
      // lineagePressure requires lineage metadata
      this._lineageEnabled = true;
    }
  }
  /**
   * Evolves the population by selecting, mutating, and breeding genomes.
   * This method is delegated to `src/neat/neat.evolve.ts` during the migration.
   *
   * @example
   * // Run a single evolution step (async)
   * await neat.evolve();
   */
  async evolve(): Promise<Network> {
    return evolve.call(this as any);
  }

  async evaluate(): Promise<any> {
    return evaluate.call(this as any);
  }

  /**
   * Create initial population pool. Delegates to helpers if present.
   */
  createPool(network: Network | null): void {
    try {
      if (createPool && typeof createPool === 'function')
        return createPool.call(this as any, network);
    } catch {}
    // Fallback basic implementation
    this.population = [];
    /**
     * Size of the initial pool to create when seeding the population. Taken
     * from options.popsize with a sensible default for backward compatibility.
     */
    const poolSize = this.options.popsize || 50;
    for (let idx = 0; idx < poolSize; idx++) {
      // Clone or create a fresh genome for the pool
      const genomeCopy = network
        ? Network.fromJSON((network as any).toJSON())
        : new Network(this.input, this.output, {
            minHidden: this.options.minHidden,
          });
      // Clear any serialized score so newly-created genomes start unevaluated
      genomeCopy.score = undefined;
      try {
        this.ensureNoDeadEnds(genomeCopy);
      } catch {}
      (genomeCopy as any)._reenableProb = this.options.reenableProb;
      (genomeCopy as any)._id = this._nextGenomeId++;
      if (this._lineageEnabled) {
        (genomeCopy as any)._parents = [];
        (genomeCopy as any)._depth = 0;
      }
      this.population.push(genomeCopy);
    }
  }

  // RNG snapshot / restore helpers used by tests
  /**
   * Return the current opaque RNG numeric state used by the instance.
   * Useful for deterministic test replay and debugging.
   */
  snapshotRNGState() {
    return this._rngState;
  }
  /**
   * Restore a previously-snapshotted RNG state. This restores the internal
   * seed but does not re-create the RNG function until next use.
   *
   * @param state Opaque numeric RNG state produced by `snapshotRNGState()`.
   */
  restoreRNGState(state: any) {
    // Restore numeric RNG state (opaque to callers)
    this._rngState = state;
    // invalidate RNG so next call re-reads seed
    this._rng = undefined;
  }
  /**
   * Import an RNG state (alias for restore; kept for compatibility).
   * @param state Numeric RNG state.
   */
  importRNGState(state: any) {
    this._rngState = state;
    this._rng = undefined;
  }
  /**
   * Export the current RNG state for external persistence or tests.
   */
  exportRNGState() {
    return this._rngState;
  }
  /**
   * Generates an offspring by crossing over two parent networks.
   * Uses the crossover method described in the Instinct algorithm.
   * @returns A new network created from two parents.
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6 Instinct: neuro-evolution on steroids by Thomas Wagenaar}
   */
  getOffspring(): Network {
    let parent1: Network;
    let parent2: Network;
    try {
      parent1 = this.getParent();
    } catch {
      parent1 = this.population[0];
    }
    try {
      parent2 = this.getParent();
    } catch {
      parent2 =
        this.population[
          Math.floor(this._getRNG()() * this.population.length)
        ] || this.population[0];
    }
    const offspring = Network.crossOver(
      parent1,
      parent2,
      this.options.equal || false
    );
    (offspring as any)._reenableProb = this.options.reenableProb;
    (offspring as any)._id = this._nextGenomeId++;
    if (this._lineageEnabled) {
      (offspring as any)._parents = [
        (parent1 as any)._id,
        (parent2 as any)._id,
      ];
      const depth1 = (parent1 as any)._depth ?? 0;
      const depth2 = (parent2 as any)._depth ?? 0;
      (offspring as any)._depth = 1 + Math.max(depth1, depth2);
      if ((parent1 as any)._id === (parent2 as any)._id)
        this._lastInbreedingCount++;
    }
    // Ensure the offspring has the minimum required hidden nodes
    this.ensureMinHiddenNodes(offspring);
    this.ensureNoDeadEnds(offspring); // Ensure no dead ends or blind I/O
    return offspring;
  }

  /** Emit a standardized warning when evolution loop finds no valid best genome (test hook). */
  _warnIfNoBestGenome() {
    try {
      console.warn(
        'Evolution completed without finding a valid best genome (no fitness improvements recorded).'
      );
    } catch {}
  }

  /**
   * Spawn a new genome derived from a single parent while preserving Neat bookkeeping.
   *
   * This helper performs a canonical "clone + slight mutation" workflow while
   * keeping `Neat`'s internal invariants intact. It is intended for callers that
   * want a child genome derived from a single parent but do not want to perform the
   * bookkeeping and registration steps manually. The function deliberately does NOT
   * add the returned child to `this.population` so callers are free to inspect or
   * further modify the child and then register it via `addGenome()` (or push it
   * directly if they understand the consequences).
   *
   * Behavior summary:
   * - Clone the provided `parent` (`parent.clone()` when available, else JSON round-trip).
   * - Clear fitness/score on the child and assign a fresh unique `_id`.
   * - If lineage tracking is enabled, set `(child as any)._parents = [parent._id]`
   *   and `(child as any)._depth = (parent._depth ?? 0) + 1`.
   * - Enforce structural invariants by calling `ensureMinHiddenNodes(child)` and
   *   `ensureNoDeadEnds(child)` so the child is valid for subsequent mutation/evaluation.
   * - Apply `mutateCount` mutations selected via `selectMutationMethod` and driven by
   *   the instance RNG (`_getRNG()`); mutation exceptions are caught and ignored to
   *   preserve best-effort behavior during population seeding/expansion.
   * - Invalidate per-genome caches with `_invalidateGenomeCaches(child)` before return.
   *
   * Important: the returned child is not registered in `Neat.population` — call
   * `addGenome(child, [parentId])` to insert it and keep telemetry/lineage consistent.
   *
   * @param parent - Source genome to derive from. Must be a `Network` instance.
   * @param mutateCount - Number of mutation operations to apply to the spawned child (default: 1).
   * @returns A new `Network` instance derived from `parent`. The child is unregistered.
   */
  spawnFromParent(parent: Network, mutateCount: number = 1): Network {
    return spawnFromParent.call(this as any, parent, mutateCount);
  }

  /**
   * Register an externally-created genome into the `Neat` population.
   *
   * Use this method when code constructs or mutates a `Network` outside of the
   * usual reproduction pipeline and needs to insert it into `neat.population`
   * while preserving lineage, id assignment, and structural invariants. The
   * method performs best-effort safety actions and falls back to pushing the
   * genome even if invariant enforcement throws, which mirrors the forgiving
   * behavior used in dynamic population expansion.
   *
   * Behavior summary:
   * - Clears the genome's `score` and assigns `_id` using Neat's counter.
   * - When lineage is enabled, attaches the provided `parents` array (copied)
   *   and estimates `_depth` as `max(parent._depth) + 1` when parent ids are
   *   resolvable from the current population.
   * - Enforces structural invariants (`ensureMinHiddenNodes` and
   *   `ensureNoDeadEnds`) and invalidates caches via
   *   `_invalidateGenomeCaches(genome)`.
   * - Pushes the genome into `this.population`.
   *
   * Note: Because depth estimation requires parent objects to be discoverable
   * in `this.population`, callers that generate intermediate parent genomes
   * should register them via `addGenome` before relying on automatic depth
   * estimation for their children.
   *
   * @param genome - The external `Network` to add.
   * @param parents - Optional array of parent ids to record on the genome.
   */
  addGenome(genome: Network, parents?: number[]): void {
    return addGenome.call(this as any, genome as any, parents as any);
  }

  /**
   * Selects a mutation method for a given genome based on constraints.
   * Ensures that the mutation respects the maximum nodes, connections, and gates.
   * @param genome - The genome to mutate.
   * @returns The selected mutation method or null if no valid method is available.
   */
  selectMutationMethod(genome: Network, rawReturnForTest: boolean = true): any {
    try {
      return selectMutationMethod.call(this as any, genome, rawReturnForTest);
    } catch {
      return null;
    }
  }

  /** Delegate ensureNoDeadEnds to mutation module (added for backward compat). */
  ensureNoDeadEnds(network: Network) {
    try {
      return ensureNoDeadEnds.call(this as any, network);
    } catch {
      return; // silent fail (used defensively in seeding paths)
    }
  }

  /** Minimum hidden size considering explicit minHidden or multiplier policy. */
  getMinimumHiddenSize(multiplierOverride?: number): number {
    const o: any = this.options;
    if (typeof o.minHidden === 'number') return o.minHidden;
    const mult = multiplierOverride ?? o.minHiddenMultiplier;
    if (typeof mult === 'number' && isFinite(mult)) {
      return Math.max(0, Math.round(mult * (this.input + this.output)));
    }
    return 0;
  }

  /** Produce `count` deterministic random samples using instance RNG. */
  sampleRandom(count: number): number[] {
    const rng = this._getRNG();
    const arr: number[] = [];
    for (let i = 0; i < count; i++) arr.push(rng());
    return arr;
  }

  /** Internal: return cached objective descriptors, building if stale. */
  private _getObjectives(): ObjectiveDescriptor[] {
    return _getObjectives.call(this as any) as ObjectiveDescriptor[];
  }

  /** Public helper returning just the objective keys (tests rely on). */
  getObjectiveKeys(): string[] {
    // Map objective descriptors to their key strings
    return (this._getObjectives() as ObjectiveDescriptor[]).map(
      (obj) => obj.key
    );
  }

  /** Invalidate per-genome caches (compatibility distance, forward pass, etc.). */
  private _invalidateGenomeCaches(genome: any) {
    if (!genome || typeof genome !== 'object') return;
    delete genome._compatCache;
    // Network forward cache fields (best-effort, ignore if absent)
    delete genome._outputCache;
    delete genome._traceCache;
  }

  /** Compute and cache diversity statistics used by telemetry & tests. */
  private _computeDiversityStats() {
    this._diversityStats = computeDiversityStats(this.population, this);
  }

  // Removed thin wrappers _structuralEntropy and _fastNonDominated; modules used directly where needed.
  /** Compatibility wrapper retained for tests that reference (neat as any)._structuralEntropy */
  private _structuralEntropy(genome: Network): number {
    return structuralEntropy(genome);
  }

  /**
   * Applies mutations to the population based on the mutation rate and amount.
   * Each genome is mutated using the selected mutation methods.
   * Slightly increases the chance of ADD_CONN mutation for more connectivity.
   */
  mutate(): void {
    return mutate.call(this as any);
  }
  // Perform ADD_NODE honoring global innovation reuse mapping
  private _mutateAddNodeReuse(genome: Network) {
    return mutateAddNodeReuse.call(this as any, genome);
  }
  private _mutateAddConnReuse(genome: Network) {
    return mutateAddConnReuse.call(this as any, genome);
  }

  // --- Speciation helpers (properly scoped) ---
  private _fallbackInnov(conn: any): number {
    return _fallbackInnov.call(this as any, conn);
  }
  _compatibilityDistance(netA: Network, netB: Network): number {
    return _compatibilityDistance.call(this as any, netA, netB);
  }
  /**
   * Assign genomes into species based on compatibility distance and maintain species structures.
   * This function creates new species for unassigned genomes and prunes empty species.
   * It also records species-level history used for telemetry and adaptive controllers.
   */
  private _speciate() {
    return _speciate.call(this as any);
  }
  /**
   * Apply fitness sharing within species. When `sharingSigma` > 0 this uses a kernel-based
   * sharing; otherwise it falls back to classic per-species averaging. Sharing reduces
   * effective fitness for similar genomes to promote diversity.
   */
  private _applyFitnessSharing() {
    return _applyFitnessSharing.call(this as any);
  }
  /**
   * Sort members of a species in-place by descending score.
   * @param sp - Species object with `members` array.
   */
  private _sortSpeciesMembers(sp: { members: Network[] }) {
    return _sortSpeciesMembers.call(this as any, sp);
  }
  /**
   * Update species stagnation tracking and remove species that exceeded the allowed stagnation.
   */
  private _updateSpeciesStagnation() {
    return _updateSpeciesStagnation.call(this as any);
  }
  /**
   * Return a concise summary for each current species.
   *
   * Educational context: In NEAT, populations are partitioned into species based
   * on genetic compatibility. Each species groups genomes that are similar so
   * selection and reproduction can preserve diversity between groups. This
   * accessor provides a lightweight view suitable for telemetry, visualization
   * and teaching examples without exposing full genome objects.
   *
   * The returned array contains objects with these fields:
   * - id: numeric species identifier
   * - size: number of members currently assigned to the species
   * - bestScore: the best observed fitness score for the species
   * - lastImproved: generation index when the species last improved its best score
   *
   * Notes for learners:
   * - Species sizes and lastImproved are typical signals used to detect
   *   stagnation and apply protective or penalizing measures.
   * - This function intentionally avoids returning full member lists to
   *   prevent accidental mutation of internal state; use `getSpeciesHistory`
   *   for richer historical data.
   *
   * @returns An array of species summary objects.
   */
  getSpeciesStats(): {
    id: number;
    size: number;
    bestScore: number;
    lastImproved: number;
  }[] {
    return getSpeciesStats.call(this as any);
  }
  /**
   * Returns the historical species statistics recorded each generation.
   *
   * Educational context: Species history captures per-generation snapshots
   * of species-level metrics (size, best score, last improvement) and is
   * useful for plotting trends, teaching about speciation dynamics, and
   * driving adaptive controllers.
   *
   * The returned array contains entries with a `generation` index and a
   * `stats` array containing per-species summaries recorded at that
   * generation.
   *
   * @returns An array of generation-stamped species stat snapshots.
   */
  getSpeciesHistory(): SpeciesHistoryEntry[] {
    return getSpeciesHistory.call(this as any) as SpeciesHistoryEntry[];
  }
  /**
   * Returns the number of entries currently stored in the novelty archive.
   *
   * Educational context: The novelty archive stores representative behaviors
   * used by behavior-based novelty search. Monitoring its size helps teach
   * how behavioral diversity accumulates over time and can be used to
   * throttle archive growth.
   *
   * @returns Number of archived behaviors.
   */
  getNoveltyArchiveSize(): number {
    return this._noveltyArchive ? this._noveltyArchive.length : 0;
  }
  /**
   * Returns compact multi-objective metrics for each genome in the current
   * population. The metrics include Pareto rank and crowding distance (if
   * computed), along with simple size and score measures useful in
   * instructional contexts.
   *
   * @returns Array of per-genome MO metric objects.
   */
  getMultiObjectiveMetrics(): {
    rank: number;
    crowding: number;
    score: number;
    nodes: number;
    connections: number;
  }[] {
    return this.population.map((genome) => ({
      rank: (genome as any)._moRank ?? 0,
      crowding: (genome as any)._moCrowd ?? 0,
      score: genome.score || 0,
      nodes: genome.nodes.length,
      connections: genome.connections.length,
    }));
  }
  /**
   * Returns a summary of mutation/operator statistics used by operator
   * adaptation and bandit selection.
   *
   * Educational context: Operator statistics track how often mutation
   * operators are attempted and how often they succeed. These counters are
   * used by adaptation mechanisms to bias operator selection towards
   * successful operators.
   *
   * @returns Array of { name, success, attempts } objects.
   */
  getOperatorStats(): { name: string; success: number; attempts: number }[] {
    return Array.from(this._operatorStats.entries()).map(
      ([operatorName, stats]) => ({
        name: operatorName,
        success: stats.success,
        attempts: stats.attempts,
      })
    );
  }
  /**
   * Manually apply evolution-time pruning once using the current generation
   * index and configuration in `options.evolutionPruning`.
   *
   * Educational usage: While pruning normally occurs automatically inside
   * the evolve loop, exposing this method lets learners trigger the pruning
   * logic in isolation to observe its effect on network sparsity.
   *
   * Implementation detail: Delegates to the migrated helper in
   * `neat.pruning.ts` so the core class surface remains thin.
   */
  applyEvolutionPruning(): void {
    try {
      require('./neat/neat.pruning').applyEvolutionPruning.call(this as any);
    } catch {}
  }
  /**
   * Run the adaptive pruning controller once. This adjusts the internal
   * `_adaptivePruneLevel` based on the configured metric (nodes or
   * connections) and invokes per-genome pruning when an adjustment is
   * warranted.
   *
   * Educational usage: Allows step-wise observation of how the adaptive
   * controller converges population complexity toward a target sparsity.
   */
  applyAdaptivePruning(): void {
    try {
      require('./neat/neat.pruning').applyAdaptivePruning.call(this as any);
    } catch {}
  }
  /**
   * Return the internal telemetry buffer.
   *
   * Telemetry entries are produced per-generation when telemetry is enabled
   * and include diagnostic metrics (diversity, performance, lineage, etc.).
   * This accessor returns the raw buffer for external inspection or export.
   *
   * @returns Array of telemetry snapshot objects.
   */
  getTelemetry(): any[] {
    return this._telemetry;
  }
  /**
   * Export telemetry as JSON Lines (one JSON object per line).
   *
   * Useful for piping telemetry to external loggers or analysis tools.
   *
   * @returns A newline-separated string of JSON objects.
   */
  exportTelemetryJSONL(): string {
    return exportTelemetryJSONL.call(this as any);
  }
  /**
   * Export recent telemetry entries as CSV.
   *
   * The exporter attempts to flatten commonly-used nested fields (complexity,
   * perf, lineage) into columns. This is a best-effort exporter intended for
   * human inspection and simple ingestion.
   *
   * @param maxEntries Maximum number of recent telemetry entries to include.
   * @returns CSV string (may be empty when no telemetry present).
   */
  exportTelemetryCSV(maxEntries = 500): string {
    return exportTelemetryCSV.call(this as any, maxEntries);
  }
  /**
   * Export telemetry as CSV with flattened columns for common nested fields.
   */
  clearTelemetry() {
    this._telemetry = [];
  }
  /** Clear all collected telemetry entries. */
  getObjectives(): { key: string; direction: 'max' | 'min' }[] {
    return (this._getObjectives() as ObjectiveDescriptor[]).map((o) => ({
      key: o.key,
      direction: o.direction,
    }));
  }
  getObjectiveEvents(): { gen: number; type: 'add' | 'remove'; key: string }[] {
    return this._objectiveEvents.slice();
  }
  /** Get recent objective add/remove events. */
  getLineageSnapshot(limit = 20): { id: number; parents: number[] }[] {
    return this.population.slice(0, limit).map((genome) => ({
      id: (genome as any)._id ?? -1,
      parents: Array.isArray((genome as any)._parents)
        ? (genome as any)._parents.slice()
        : [],
    }));
  }
  /**
   * Return an array of {id, parents} for the first `limit` genomes in population.
   */
  exportSpeciesHistoryCSV(maxEntries = 200): string {
    return exportSpeciesHistoryCSV.call(this as any, maxEntries);
  }
  /**
   * Export species history as CSV.
   *
   * Produces rows for each recorded per-species stat entry within the
   * specified window. Useful for quick inspection or spreadsheet analysis.
   *
   * @param maxEntries Maximum history entries to include (default: 200).
   * @returns CSV string (may be empty).
   */
  getParetoFronts(maxFronts = 3): Network[][] {
    if (!this.options.multiObjective?.enabled) return [[...this.population]];
    // reconstruct fronts from stored ranks (avoids re-sorting again)
    const fronts: Network[][] = [];
    for (let frontIdx = 0; frontIdx < maxFronts; frontIdx++) {
      const front = this.population.filter(
        (genome) => ((genome as any)._moRank ?? 0) === frontIdx
      );
      if (!front.length) break;
      fronts.push(front);
    }
    return fronts;
  }
  /**
   * Return the latest cached diversity statistics.
   *
   * Educational context: diversity metrics summarize how genetically and
   * behaviorally spread the population is. They can include lineage depth,
   * pairwise genetic distances, and other aggregated measures used by
   * adaptive controllers, novelty search, and telemetry. This accessor returns
   * whatever precomputed diversity object the Neat instance holds (may be
   * undefined if not computed for the current generation).
   *
   * @returns Arbitrary diversity summary object or undefined.
   */
  getDiversityStats() {
    return this._diversityStats;
  }
  registerObjective(
    key: string,
    direction: 'min' | 'max',
    // Widen accessor parameter type to match underlying registerObjective expectation (GenomeLike)
    accessor: (g: any) => number
  ) {
    return registerObjective.call(this as any, key, direction, accessor);
  }
  /**
   * Register a custom objective for multi-objective optimization.
   *
   * Educational context: multi-objective optimization lets you optimize for
   * multiple, potentially conflicting goals (e.g., maximize fitness while
   * minimizing complexity). Each objective is identified by a unique key and
   * an accessor function mapping a genome to a numeric score. Registering an
   * objective makes it visible to the internal MO pipeline and clears any
   * cached objective list so changes take effect immediately.
   *
   * @param key Unique objective key.
   * @param direction 'min' or 'max' indicating optimization direction.
   * @param accessor Function mapping a genome to a numeric objective value.
   */
  /**
   * Clear all registered multi-objective objectives.
   *
   * Removes any objectives configured for multi-objective optimization and
   * clears internal caches. Useful for tests or when reconfiguring the MO
   * setup at runtime.
   */
  clearObjectives() {
    return clearObjectives.call(this as any);
  }
  // Advanced archives & performance accessors
  /**
   * Get recent Pareto archive entries (meta information about archived fronts).
   *
   * Educational context: when performing multi-objective search we may store
   * representative Pareto-front snapshots over time. This accessor returns the
   * most recent archive entries up to the provided limit.
   *
   * @param maxEntries Maximum number of entries to return (default: 50).
   * @returns Array of archived Pareto metadata entries.
   */
  getParetoArchive(maxEntries = 50) {
    return this._paretoArchive.slice(-maxEntries);
  }
  /**
   * Export Pareto front archive as JSON Lines for external analysis.
   *
   * Each line is a JSON object representing one archived Pareto snapshot.
   *
   * @param maxEntries Maximum number of entries to include (default: 100).
   * @returns Newline-separated JSON objects.
   */
  exportParetoFrontJSONL(maxEntries = 100): string {
    const slice = this._paretoObjectivesArchive.slice(-maxEntries);
    return slice.map((e) => JSON.stringify(e)).join('\n');
  }
  /**
   * Return recent performance statistics (durations in milliseconds) for the
   * most recent evaluation and evolve operations.
   *
   * Provides wall-clock timing useful for profiling and teaching how runtime
   * varies with network complexity or population settings.
   *
   * @returns Object with { lastEvalMs, lastEvolveMs }.
   */
  getPerformanceStats() {
    return {
      lastEvalMs: this._lastEvalDuration,
      lastEvolveMs: this._lastEvolveDuration,
    };
  }
  // Utility exports / maintenance
  /**
   * Export species history as JSON Lines for storage and analysis.
   *
   * Each line is a JSON object containing a generation index and per-species
   * stats recorded at that generation. Useful for long-term tracking.
   *
   * @param maxEntries Maximum history entries to include (default: 200).
   * @returns Newline-separated JSON objects.
   */
  exportSpeciesHistoryJSONL(maxEntries = 200): string {
    const slice = this._speciesHistory.slice(-maxEntries);
    return slice.map((e) => JSON.stringify(e)).join('\n');
  }
  /**
   * Reset the novelty archive (clear entries).
   *
   * The novelty archive is used to keep representative behaviors for novelty
   * search. Clearing it removes stored behaviors.
   */
  resetNoveltyArchive() {
    this._noveltyArchive = [];
  }
  /**
   * Clear the Pareto archive.
   *
   * Removes any stored Pareto-front snapshots retained by the algorithm.
   */
  clearParetoArchive() {
    this._paretoArchive = [];
  }

  /**
   * Sorts the population in descending order of fitness scores.
   * Ensures that the fittest genomes are at the start of the population array.
   */
  sort(): void {
    return sort.call(this as any);
  }

  /**
   * Selects a parent genome for breeding based on the selection method.
   * Supports multiple selection strategies, including POWER, FITNESS_PROPORTIONATE, and TOURNAMENT.
   * @returns The selected parent genome.
   * @throws Error if tournament size exceeds population size.
   */
  getParent(): Network {
    return getParent.call(this as any);
  }

  /**
   * Retrieves the fittest genome from the population.
   * Ensures that the population is evaluated and sorted before returning the result.
   * @returns The fittest genome in the population.
   */
  getFittest(): Network {
    return getFittest.call(this as any);
  }

  /**
   * Calculates the average fitness score of the population.
   * Ensures that the population is evaluated before calculating the average.
   * @returns The average fitness score of the population.
   */
  getAverage(): number {
    return getAverage.call(this as any);
  }

  /**
   * Exports the current population as an array of JSON objects.
   * Useful for saving the state of the population for later use.
   * @returns An array of JSON representations of the population.
   */
  export(): any[] {
    return exportPopulation.call(this as any);
  }

  /**
   * Imports a population from an array of JSON objects.
   * Replaces the current population with the imported one.
   * @param json - An array of JSON objects representing the population.
   */
  import(json: any[]): void {
    return importPopulation.call(this as any, json as any);
  }

  /**
   * Convenience: export full evolutionary state (meta + population genomes).
   * Combines innovation registries and serialized genomes for easy persistence.
   */
  exportState(): any {
    return exportState.call(this as any);
  }

  /**
   * Convenience: restore full evolutionary state previously produced by exportState().
   * @param bundle Object with shape { neat, population }
   * @param fitness Fitness function to attach
   */
  static importState(bundle: any, fitness: (n: Network) => number): Neat {
    return importStateImpl.call(Neat as any, bundle, fitness) as Neat;
  }
  /**
   * Import a previously exported state bundle and rehydrate a Neat instance.
   */
  // Serialize NEAT meta (without population) for persistence of innovation history
  toJSON(): any {
    return toJSONImpl.call(this as any);
  }

  static fromJSON(json: any, fitness: (n: Network) => number): Neat {
    return fromJSONImpl.call(Neat as any, json, fitness) as Neat;
  }
}
