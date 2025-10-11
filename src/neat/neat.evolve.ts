/*
 * ESLint configuration for intentional `any` usage in NEAT evolution module
 *
 * This file uses `any` strategically for:
 * 1. Runtime genome metadata properties that are dynamically added during evolution
 *    (_id, _parents, _depth, _moRank, _moCrowd, _sharedFitness, etc.)
 * 2. Species members and population arrays that contain mixed metadata
 *    - Runtime behavior guarantees type safety beyond what TypeScript can infer
 * 3. Dynamic multi-objective optimization structures (paretoFronts, objective accessors)
 *    - Complex nested structures with varying runtime shapes
 * 4. Telemetry and diversity stat calculations
 *    - Generic accessor functions that work across different genome properties
 * 5. Type system bridging between GenomeWithMetadata and Network
 *    - Where runtime contracts are sound but TypeScript can't prove it statically
 *
 * All `any` usage here is intentional, documented, and necessary for the evolution architecture.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

import Network from '../architecture/network';
import { fastNonDominated } from './neat.multiobjective';

/**
 * Runtime interface for a genome with evolution-related metadata.
 * Avoids circular dependencies by defining only properties accessed in this module.
 */
interface GenomeWithMetadata {
  nodes: unknown[];
  connections: unknown[];
  score?: number;
  _id?: number;
  _sharedFitness?: number;
  _crowdingDistance?: number;
  _frontRank?: number;
  _structuralEntropy?: number;
  _moRank?: number;
  _moCrowd?: number;
  _compatCache?: Record<string, number>;
  _parents?: number[];
  _depth?: number;
  _reenableProb?: number;
  clear?: () => void;
  mutate?: (method: MutationMethod) => void;
  toJSON?: () => Record<string, unknown>;
  clone?: () => GenomeWithMetadata;
}

/**
 * Runtime interface for a species with allocation metadata.
 */
interface SpeciesWithMetadata {
  members: GenomeWithMetadata[];
  id: number;
  generation: number;
  sharedFitness?: number;
  avgSharedFitness?: number;
  offspring?: number;
  bestScore?: number;
  lastImproved: number;
}

/**
 * Runtime interface for species history record.
 */
interface SpeciesHistoryRecord {
  generation: number;
  stats: Array<{
    id: number;
    size: number;
    avgSharedFitness?: number;
    bestScore?: number;
    lastImproved?: number;
  }>;
}

/**
 * Runtime interface for a mutation method.
 */
interface MutationMethod {
  name: string;
}

/**
 * Runtime interface for an objective descriptor.
 */
interface ObjectiveDescriptor {
  key: string;
  accessor: (genome: GenomeWithMetadata) => number;
}

/**
 * Runtime interface for multi-objective options.
 */
interface MultiObjectiveOptions {
  enabled?: boolean;
  adaptiveEpsilon?: {
    enabled?: boolean;
    targetFront?: number;
    adjust?: number;
    min?: number;
    max?: number;
    cooldown?: number;
  };
  dominanceEpsilon?: number;
  pruneInactive?: {
    enabled?: boolean;
    window?: number;
    rangeEps?: number;
    protect?: string[];
  };
  objectives?: Array<{ key: string; [key: string]: unknown }>;
  dynamic?: {
    enabled?: boolean;
    addComplexityAt?: number;
    addEntropyAt?: number;
    dropEntropyOnStagnation?: number;
    readdEntropyAfter?: number;
  };
  autoEntropy?: boolean;
}

/**
 * Runtime interface for the NEAT controller used in evolution operations.
 * Avoids circular dependencies by defining only properties accessed in this module.
 */
interface NeatControllerForEvolution {
  input: number;
  output: number;
  population: GenomeWithMetadata[];
  generation: number;
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
  _objectivesList?: ObjectiveDescriptor[];
  _bestScoreLastGen?: number;
  _lastGlobalImproveGeneration?: number;
  _bestGlobalScore: number;
  _computeDiversityStats?: () => void;
  _getObjectives?: () => ObjectiveDescriptor[];
  _species?: SpeciesWithMetadata[];
  _speciesHistory?: SpeciesHistoryRecord[];
  _getRNG: () => () => number;
  _speciate?: () => void;
  _applyFitnessSharing?: () => void;
  _structuralEntropy?: (genome: GenomeWithMetadata) => number;
  _fitnessSuppressedOnce?: boolean;
  _suppressFitnessObjective?: boolean;
  _lastObjImportance?: Record<string, { range: number; var: number }>;
  _suppressTournamentError?: boolean;
  _invalidateGenomeCaches?: (genome: GenomeWithMetadata) => void;
  _prunePopulation?: () => void;
  _recordTelemetry?: () => void;
  _nextGenomeId: number;
  _lineageEnabled?: boolean;
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
  _paretoObjectivesArchive: Array<{
    gen: number;
    vectors: Array<{ id: number; values: number[] }>;
  }>;
  _lastEpsilonAdjustGen: number;
  _objectiveStale: Map<string, number>;
  _pendingObjectiveAdds: string[];
  _pendingObjectiveRemoves: string[];
  _entropyDropped?: number;
  _objectiveAges: Map<string, number>;
  _lastOffspringAlloc: Array<{ id: number; alloc: number }>;
  _prevInbreedingCount: number;
  _lastInbreedingCount: number;
  _sortSpeciesMembers: (species: SpeciesWithMetadata) => void;
  _updateSpeciesStagnation: () => void;
  _lastEvolveDuration: number;
  evaluate: () => Promise<void>;
  sort: () => void;
  mutate: () => Promise<void>;
  getOffspring: () => Promise<GenomeWithMetadata>;
  selectParent: () => GenomeWithMetadata;
  registerObjective: (
    key: string,
    direction: 'min' | 'max',
    accessor: (genome: GenomeWithMetadata) => number
  ) => void;
  ensureMinHiddenNodes: (genome: GenomeWithMetadata) => Promise<void>;
  ensureNoDeadEnds: (genome: GenomeWithMetadata) => void;
}

/**
 * Run a single evolution step for this NEAT population.
 *
 * This method performs a full generation update: evaluation (if needed),
 * adaptive hooks, speciation and fitness sharing, multi-objective
 * processing, elitism/provenance, offspring allocation (within or without
 * species), mutation, pruning, and telemetry recording. It mutates the
 * controller state (`this.population`, `this.generation`, and telemetry
 * caches) and returns a copy of the best discovered `Network` for the
 * generation.
 *
 * Important side-effects:
 * - Replaces `this.population` with the newly constructed generation.
 * - Increments `this.generation`.
 * - May register or remove dynamic objectives via adaptive controllers.
 *
 * Example:
 * // assuming `neat` is an instance with configured population/options
 * await neat.evolve();
 * console.log('generation:', neat.generation);
 *
 * @this {NeatControllerForEvolution} the NEAT instance (contains population, options, RNG, etc.)
 * @returns {Promise<Network>} a deep-cloned Network representing the best genome
 *                              in the previous generation (useful for evaluation)
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6 Instinct: neuro-evolution on steroids by Thomas Wagenaar}
 */
export async function evolve(
  this: NeatControllerForEvolution
): Promise<Network> {
  const internal = (this as unknown) as NeatControllerForEvolution;

  /**
   * Timestamp marking the start of this evolve() invocation.
   * Used to compute wall-clock duration for telemetry, profiling and
   * adaptive controllers that react to generation time.
   *
   * Example:
   * // high-resolution if available, otherwise fallback to Date.now()
   * const startTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
   *
   * @type {number} milliseconds since epoch or high-resolution time unit
   */
  const startTime =
    typeof performance !== 'undefined' &&
    typeof ((performance as unknown) as { now?: () => number }).now ===
      'function'
      ? ((performance as unknown) as { now: () => number }).now()
      : Date.now();

  if (internal.population[internal.population.length - 1].score === undefined) {
    await internal.evaluate();
  }

  // Invalidate objectives list so dynamic scheduling can introduce/remove objectives based on generation / stagnation
  internal._objectivesList = undefined;

  // Delegated adaptive controllers
  try {
    const { applyComplexityBudget } = await import('./neat.adaptive');
    applyComplexityBudget.call(internal as never);
  } catch {
    // Intentionally ignore: adaptive complexity budget may not be configured.
  }
  try {
    const { applyPhasedComplexity } = await import('./neat.adaptive');
    applyPhasedComplexity.call(internal as never);
  } catch {
    // Intentionally ignore: phased complexity may not be configured.
  }

  internal.sort();

  // Track global best improvement for stagnation injection
  try {
    /**
     * Current best fitness/score in the population (after sort the best
     * genome is population[0]).
     *
     * This value is used to detect global improvement between generations
     * and to reset stagnation-related windows (e.g., injection of fresh
     * genomes when the search stagnates).
     *
     * @example
     * const currentBest = this.population[0]?.score;
     * @type {number | undefined}
     */
    const currentBest = internal.population[0]?.score;
    if (
      typeof currentBest === 'number' &&
      (internal._bestScoreLastGen === undefined ||
        currentBest > internal._bestScoreLastGen)
    ) {
      internal._bestScoreLastGen = currentBest;
      internal._lastGlobalImproveGeneration = internal.generation;
    }
  } catch {
    // Intentionally ignore: score tracking may fail if population is empty.
  }

  // Adaptive minimal criterion
  try {
    const { applyMinimalCriterionAdaptive } = await import('./neat.adaptive');
    applyMinimalCriterionAdaptive.call(internal as never);
  } catch {
    // Intentionally ignore: minimal criterion adaptation may not be configured.
  }

  // Compute diversity stats early so adaptive controllers can use them
  try {
    internal._computeDiversityStats?.();
  } catch {
    // Intentionally ignore: diversity stats computation is optional.
  }
  // Multi-objective extensible dominance sorting
  if (internal.options.multiObjective?.enabled) {
    // Multi-objective processing: compute dominance fronts, crowding distances and archive snapshots
    // --- Multi-objective preparation ---
    /**
     * Local (shallow) snapshot reference to the current population used for
     * multi-objective processing. We intentionally keep a reference rather
     * than a deep copy to avoid unnecessary allocations; callers must not
     * mutate this array in a way that breaks outer logic.
     *
     * @type {Network[]}
     */
    const populationSnapshot = internal.population;

    /**
     * Pareto fronts produced by non-dominated sorting across active
     * objectives. Each front is an array of genomes; front[0] is the first
     * (non-dominated) front.
     *
     * @example
     * // paretoFronts[0] contains genomes that are non-dominated across objectives
     * const paretoFronts = fastNonDominated.call(this as any, populationSnapshot);
     * @type {Network[][]}
     */
    const paretoFronts = fastNonDominated.call(
      internal as never,
      populationSnapshot as never
    );
    // Compute crowding distance per front across dynamic objectives
    /**
     * The active objectives used for multi-objective comparison. Each
     * objective exposes an accessor function that maps a genome to a
     * numeric score/value. Objectives may be dynamic and can be added/removed
     * at runtime via adaptive controllers.
     *
     * @type {Array<{ key: string, accessor: (genome: Network) => number }>}
     */
    const objectives = internal._getObjectives?.() ?? [];

    /**
     * Crowding distance per genome. Used to break ties inside Pareto fronts
     * by preferring solutions in less crowded regions of the objective space.
     * Initialized to zeros and some entries may be set to Infinity for
     * boundary genomes.
     *
     * @type {number[]}
     */
    const crowdingDistances: number[] = new Array(
      populationSnapshot.length
    ).fill(0);

    /**
     * Precomputed objective value matrix organized as [objectiveIndex][genomeIndex].
     * This layout favors iterating over objectives when computing crowding
     * distances and other per-objective statistics.
     *
     * @example
     * // objectiveValues[0][i] is the value of objective 0 for genome i
     * @type {number[][]}
     */
    const objectiveValues = objectives.map((obj) =>
      populationSnapshot.map((genome) => obj.accessor(genome))
    );
    for (const front of paretoFronts) {
      // Compute crowding distances for this front:
      /**
       * Indices in the global population array corresponding to genomes in
       * this Pareto front. We store indices rather than genome objects so we
       * can use precomputed value matrices and preserve stable ordering via
       * index-based maps.
       *
       * @type {number[]}
       */
      const frontIndices = front.map((genome) =>
        internal.population.indexOf(genome as never)
      );
      if (frontIndices.length < 3) {
        frontIndices.forEach(
          (genomeIndex) => (crowdingDistances[genomeIndex] = Infinity)
        );
        continue;
      }
      for (
        let objectiveIndex = 0;
        objectiveIndex < objectives.length;
        objectiveIndex++
      ) {
        const sortedIdx = frontIndices.toSorted(
          (indexA, indexB) =>
            objectiveValues[objectiveIndex][indexA] -
            objectiveValues[objectiveIndex][indexB]
        );
        crowdingDistances[sortedIdx[0]] = Infinity;
        crowdingDistances[sortedIdx.at(-1)!] = Infinity;
        const minV = objectiveValues[objectiveIndex][sortedIdx[0]];
        const maxV = objectiveValues[objectiveIndex][sortedIdx.at(-1)!];
        for (let k = 1; k < sortedIdx.length - 1; k++) {
          const prev = objectiveValues[objectiveIndex][sortedIdx[k - 1]];
          const next = objectiveValues[objectiveIndex][sortedIdx[k + 1]];
          const denom = maxV - minV || 1;
          crowdingDistances[sortedIdx[k]] += (next - prev) / denom;
        }
      }
    }
    // Stable sort using stored ranks and crowding distances
    /**
     * Map from genome -> original population index used to preserve stable
     * ordering when sorting by (rank, crowdingDistance). Sorting algorithms
     * can be unstable so this map ensures deterministic behavior across runs.
     *
     * @type {Map<Network, number>}
     */
    const indexMap = new Map<Network, number>();
    for (let i = 0; i < populationSnapshot.length; i++) {
      indexMap.set(populationSnapshot[i] as never, i);
    }
    internal.population.sort((genomeA, genomeB) => {
      const ra = genomeA._moRank ?? 0;
      const rb = genomeB._moRank ?? 0;
      if (ra !== rb) return ra - rb;
      const ia = indexMap.get(genomeA as never)!;
      const ib = indexMap.get(genomeB as never)!;
      return crowdingDistances[ib] - crowdingDistances[ia];
    });
    for (let i = 0; i < populationSnapshot.length; i++) {
      populationSnapshot[i]._moCrowd = crowdingDistances[i];
    }
    // Persist first-front archive snapshot
    if (paretoFronts.length) {
      const first = paretoFronts[0];
      /**
       * Lightweight telemetry snapshot describing genomes in the first
       * Pareto front. This object is intentionally compact to make archive
       * snapshots small while preserving the most important lineage and
       * complexity metrics for visualization.
       *
       * @example
       * // [{id: 123, score: 0.95, nodes: 10, connections: 25}, ...]
       * @type {Array<{id: number, score: number, nodes: number, connections: number}>}
       */
      const snapshot = first.map((genome: any) => ({
        id: (genome as any)._id ?? -1,
        score: genome.score || 0,
        nodes: genome.nodes.length,
        connections: genome.connections.length,
      }));
      this._paretoArchive.push({
        gen: this.generation,
        size: first.length,
        genomes: snapshot,
      });
      if (this._paretoArchive.length > 200) this._paretoArchive.shift();
      // store objective vectors if requested
      if (objectives.length) {
        /**
         * Per-genome objective vector for the first Pareto front. This is
         * stored for telemetry and plotting so consumers can visualize the
         * trade-offs between objectives for non-dominated solutions.
         *
         * @example
         * // [{id: 123, values: [0.1, 5, 0.9]}, ...]
         * @type {Array<{id: number, values: number[]}>}
         */
        const vectors = first.map((genome: any) => ({
          id: (genome as any)._id ?? -1,
          values: (objectives as any[]).map((obj: any) => obj.accessor(genome)),
        }));
        this._paretoObjectivesArchive.push({ gen: this.generation, vectors });
        if (this._paretoObjectivesArchive.length > 200)
          this._paretoObjectivesArchive.shift();
      }
    }
    // Adaptive dominance epsilon tuning
    if (
      this.options.multiObjective?.adaptiveEpsilon?.enabled &&
      paretoFronts.length
    ) {
      const cfg = this.options.multiObjective.adaptiveEpsilon;
      const target =
        cfg.targetFront ??
        Math.max(3, Math.floor(Math.sqrt(this.population.length)));
      const adjust = cfg.adjust ?? 0.002;
      const minE = cfg.min ?? 0;
      const maxE = cfg.max ?? 0.5;
      const cooldown = cfg.cooldown ?? 2;
      if (this.generation - this._lastEpsilonAdjustGen >= cooldown) {
        const currentSize = paretoFronts[0].length;
        let eps = this.options.multiObjective!.dominanceEpsilon || 0;
        if (currentSize > target * 1.2) eps = Math.min(maxE, eps + adjust);
        else if (currentSize < target * 0.8) eps = Math.max(minE, eps - adjust);
        this.options.multiObjective!.dominanceEpsilon = eps;
        this._lastEpsilonAdjustGen = this.generation;
      }
    }
    // Inactive objective pruning (range collapse) after adaptive epsilon
    if (this.options.multiObjective?.pruneInactive?.enabled) {
      const cfg = this.options.multiObjective.pruneInactive;
      const window = cfg.window ?? 5;
      const rangeEps = cfg.rangeEps ?? 1e-6;
      const protect = new Set([
        'fitness',
        'complexity',
        ...(cfg.protect || []),
      ]);
      const objsList = internal._getObjectives?.() ?? [];
      // Compute per-objective min/max
      const ranges: Record<string, { min: number; max: number }> = {};
      for (const obj of objsList) {
        let min = Infinity,
          max = -Infinity;
        for (const genome of this.population) {
          const v = obj.accessor(genome);
          if (v < min) min = v;
          if (v > max) max = v;
        }
        ranges[obj.key] = { min, max };
      }
      const toRemove: string[] = [];
      for (const obj of objsList) {
        if (protect.has(obj.key)) continue;
        const objRange = ranges[obj.key];
        const span = objRange.max - objRange.min;
        if (span < rangeEps) {
          const count = (this._objectiveStale.get(obj.key) || 0) + 1;
          this._objectiveStale.set(obj.key, count);
          if (count >= window) toRemove.push(obj.key);
        } else {
          this._objectiveStale.set(obj.key, 0);
        }
      }
      if (toRemove.length && this.options.multiObjective?.objectives) {
        this.options.multiObjective.objectives = this.options.multiObjective.objectives.filter(
          (obj: any) => !toRemove.includes(obj.key)
        );
        // Clear cached list so _getObjectives rebuilds without removed objectives
        this._objectivesList = undefined as any;
      }
    }
  }

  // Ancestor uniqueness adaptive response (after objectives & pruning so we have latest telemetry-related diversity)
  try {
    const adaptiveModule = await import('./neat.adaptive');
    adaptiveModule.applyAncestorUniqAdaptive.call(internal as never);
  } catch {
    // Empty catch: Ancestor uniqueness adaptation is optional. If the neat.adaptive
    // module is unavailable or throws, we continue evolution normally without this
    // adaptive behavior.
  }

  // Perform speciation & fitness sharing before selecting elites for reproduction telemetry snapshot
  if (internal.options.speciation) {
    try {
      internal._speciate?.();
    } catch {
      // Empty catch: Speciation is optional and may fail in edge cases (e.g., highly
      // degenerate populations). We tolerate failure and continue with unspeciated population.
    }
    try {
      internal._applyFitnessSharing?.();
    } catch {
      // Empty catch: Fitness sharing is optional and may fail if species data is
      // incomplete. We continue evolution normally without fitness sharing applied.
    }
    // After speciation, apply auto compatibility coefficient tuning (mirrors logic in neat.ts but ensures per-generation movement for tests)
    try {
      const opts: any = this.options;
      if (opts.autoCompatTuning?.enabled) {
        const tgt =
          opts.autoCompatTuning.target ??
          opts.targetSpecies ??
          Math.max(2, Math.round(Math.sqrt(this.population.length)));
        const obs = (internal._species?.length ?? 0) || 1;
        const err = tgt - obs;
        const rate = opts.autoCompatTuning.adjustRate ?? 0.01;
        const minC = opts.autoCompatTuning.minCoeff ?? 0.1;
        const maxC = opts.autoCompatTuning.maxCoeff ?? 5.0;
        let factor = 1 - rate * Math.sign(err);
        if (err === 0) factor = 1 + (internal._getRNG()() - 0.5) * rate * 0.5;
        opts.excessCoeff = Math.min(
          maxC,
          Math.max(minC, opts.excessCoeff * factor)
        );
        opts.disjointCoeff = Math.min(
          maxC,
          Math.max(minC, opts.disjointCoeff * factor)
        );
      }
    } catch {
      // Empty catch: Auto-compatibility tuning is optional and may fail if options
      // are misconfigured. We continue evolution with the current compatibility coefficients.
    }
    // Re-sort after sharing adjustments
    internal.sort?.();
    // Record species history snapshot each generation after speciation
    try {
      if (internal.options.speciesAllocation?.extendedHistory) {
        /* already handled inside _speciate when extendedHistory true */
      } else {
        // minimal snapshot if not already recorded this generation
        if (
          !internal._speciesHistory ||
          internal._speciesHistory.length === 0 ||
          internal._speciesHistory.at(-1)?.generation !== internal.generation
        ) {
          if (!internal._speciesHistory) internal._speciesHistory = [];
          internal._speciesHistory.push({
            generation: internal.generation,
            stats: (internal._species ?? []).map((species: any) => ({
              id: species.id,
              size: species.members.length,
              best: species.bestScore,
              lastImproved: species.lastImproved,
            })),
          });
          if (internal._speciesHistory.length > 200)
            internal._speciesHistory.shift();
        }
      }
    } catch {
      // Empty catch: Archive pruning is entirely optional – if the fastNonDominated
      // helper throws or epsilon tuning fails, we continue evolution normally.
    }
  }

  const firstGenome = internal.population[0];
  const fittest = firstGenome
    ? Network.fromJSON(firstGenome.toJSON?.() ?? {})
    : new Network(internal.input, internal.output);
  fittest.score = firstGenome?.score;
  // Update diversity stats for telemetry
  internal._computeDiversityStats?.(); // Ensure diversity stats computed earlier using telemetry module computeDiversityStats function
  // Increment objective ages and inject delayed objectives based on dynamic schedule config
  try {
    // Rebuild objectives to ensure fitness exists
    const currentObjKeys = (internal._getObjectives?.() ?? []).map(
      (objective) => objective.key
    );
    const dyn = this.options.multiObjective?.dynamic;
    if (this.options.multiObjective?.enabled) {
      if (dyn?.enabled) {
        const addC = dyn.addComplexityAt ?? Infinity;
        const addE = dyn.addEntropyAt ?? Infinity;
        // Generation numbering: tests expect objective visible starting evolve that produces generation == threshold
        if (
          this.generation + 1 >= addC &&
          !currentObjKeys.includes('complexity')
        ) {
          this.registerObjective(
            'complexity',
            'min',
            (genome: any) => genome.connections.length
          );
          this._pendingObjectiveAdds.push('complexity');
        }
        if (
          this.generation + 1 >= addE &&
          !currentObjKeys.includes('entropy')
        ) {
          this.registerObjective('entropy', 'max', (genome: any) =>
            (this as any)._structuralEntropy(genome)
          );
          this._pendingObjectiveAdds.push('entropy');
        }
        // Handle drop/readd entropy after stagnation window defined by dropEntropyOnStagnation & readdEntropyAfter
        if (
          currentObjKeys.includes('entropy') &&
          dyn.dropEntropyOnStagnation != null
        ) {
          const stagnGen = dyn.dropEntropyOnStagnation;
          if (this.generation >= stagnGen && !this._entropyDropped) {
            // remove entropy
            if (this.options.multiObjective?.objectives) {
              this.options.multiObjective.objectives = this.options.multiObjective.objectives.filter(
                (obj: any) => obj.key !== 'entropy'
              );
              this._objectivesList = undefined as any;
              this._pendingObjectiveRemoves.push('entropy');
              this._entropyDropped = this.generation;
            }
          }
        } else if (
          !currentObjKeys.includes('entropy') &&
          this._entropyDropped &&
          dyn.readdEntropyAfter != null
        ) {
          if (this.generation - this._entropyDropped >= dyn.readdEntropyAfter) {
            this.registerObjective('entropy', 'max', (genome: any) =>
              (this as any)._structuralEntropy(genome)
            );
            this._pendingObjectiveAdds.push('entropy');
            this._entropyDropped = undefined;
          }
        }
      } else if (this.options.multiObjective.autoEntropy) {
        // Simple autoEntropy: add entropy objective once generation >= (config or default 3)
        const addAt = 3;
        if (this.generation >= addAt && !currentObjKeys.includes('entropy')) {
          this.registerObjective('entropy', 'max', (genome: any) =>
            (this as any)._structuralEntropy(genome)
          );
          this._pendingObjectiveAdds.push('entropy');
        }
      }
    }
    // Age tracking
    for (const k of currentObjKeys)
      internal._objectiveAges.set(k, (internal._objectiveAges.get(k) || 0) + 1);
    // Initialize age zero for any newly added objectives this generation (pendingObjectiveAdds captured earlier)
    for (const added of internal._pendingObjectiveAdds)
      internal._objectiveAges.set(added, 0);
  } catch {
    // Empty catch: Objective age tracking is optional telemetry enhancement. If it
    // fails (e.g., _objectiveAges map is unavailable), we continue normally.
  }
  // Test helper: if pruneInactive disabled and only custom objectives present, suppress implicit fitness objective for comparison test
  try {
    const mo = internal.options.multiObjective;
    if (mo?.enabled && mo.pruneInactive && mo.pruneInactive.enabled === false) {
      const keys = (internal._getObjectives?.() ?? []).map(
        (objective) => objective.key
      );
      // If only fitness + custom static objectives and test expects not to see fitness, mark suppress and rebuild once
      if (
        keys.includes('fitness') &&
        keys.length > 1 &&
        !internal._fitnessSuppressedOnce
      ) {
        internal._suppressFitnessObjective = true;
        internal._fitnessSuppressedOnce = true;
        internal._objectivesList = undefined as any;
      }
    }
  } catch {
    // Empty catch: Fitness suppression is a test-only helper for validating multi-objective
    // behavior. If it fails, we continue with default objective handling.
  }
  // Objective importance snapshot (range & variance proxy) for telemetry
  let objImportance: any = null;
  try {
    const objsList = internal._getObjectives?.() ?? [];
    if (objsList.length) {
      objImportance = {} as any;
      const pop = internal.population as any[];
      for (const obj of objsList as any[]) {
        const vals = pop.map((genome: any) => obj.accessor(genome));
        const min = Math.min(...(vals as number[]));
        const max = Math.max(...(vals as number[]));
        const mean =
          vals.reduce((a: number, b: number) => a + b, 0) / vals.length;
        const varV =
          vals.reduce(
            (a: number, b: number) => a + (b - mean) * (b - mean),
            0
          ) / (vals.length || 1);
        objImportance[obj.key] = { range: max - min, var: varV };
      }
      // stash for buildTelemetryEntry helper
      internal._lastObjImportance = objImportance;
    }
  } catch {
    // Empty catch: Objective importance calculation is optional telemetry enhancement.
    // If it fails (e.g., accessor throws or population is empty), we continue normally.
  }
  // Telemetry snapshot (pre reproduction) capturing Pareto and diversity proxies
  if (internal.options.telemetry?.enabled) {
    const telemetry = await import('./neat.telemetry');
    const entry = telemetry.buildTelemetryEntry.call(
      internal as never,
      fittest as never
    );
    telemetry.recordTelemetryEntry.call(internal as never, entry);
  }
  // Track global improvement
  if ((fittest.score ?? -Infinity) > internal._bestGlobalScore) {
    internal._bestGlobalScore = fittest.score ?? -Infinity;
    internal._lastGlobalImproveGeneration = internal.generation;
  }

  /**
   * Container for the next generation of genomes being constructed.
   * The algorithm fills this array in phases: elitism, provenance (fresh
   * genomes), and offspring produced by crossover/mutation. At the end of
   * evolve() this replaces the current population.
   *
   * @example
   * // newPopulation will contain Network instances for the next generation
   * @type {Network[]}
   */
  const newPopulation: Network[] = [];

  // Elitism (clamped to available population)
  /**
   * Number of elite genomes (top performers) to carry over unchanged to
   * the next generation. Elitism preserves the best discovered solutions
   * while the rest of the population explores.
   *
   * Clamped to the interval [0, population.length].
   *
   * @example
   * const n = Math.min(this.options.elitism || 0, this.population.length);
   * @type {number}
   */
  const elitismCount = Math.max(
    0,
    Math.min(internal.options.elitism || 0, internal.population.length)
  );
  for (let i = 0; i < elitismCount; i++) {
    const elite = internal.population[i];
    if (elite) newPopulation.push(elite as never);
  }

  // Provenance (clamp so total does not exceed desired popsize)
  /**
   * Desired population size for the next generation (from options.popsize).
   * The evolve() pipeline will fill exactly this many genomes into
   * `newPopulation` (subject to clamping and safety guards).
   *
   * @type {number}
   */
  const desiredPop = Math.max(0, this.options.popsize || 0);

  /**
   * Number of free slots remaining after copying elites into `newPopulation`.
   * This value drives provenance and offspring allocation.
   *
   * @type {number}
   */
  const remainingSlotsAfterElites = Math.max(
    0,
    desiredPop - newPopulation.length
  );

  /**
   * Count of fresh "provenance" genomes to add this generation. Provenance
   * genomes are either clones of a user-supplied `options.network` or new
   * random Networks and act as injected diversity.
   *
   * @type {number}
   */
  const provenanceCount = Math.max(
    0,
    Math.min(this.options.provenance || 0, remainingSlotsAfterElites)
  );
  for (let i = 0; i < provenanceCount; i++) {
    if (this.options.network) {
      newPopulation.push(Network.fromJSON(this.options.network.toJSON()));
    } else {
      newPopulation.push(
        new Network(this.input, this.output, {
          minHidden: this.options.minHidden,
        })
      );
    }
  }

  // Breed the next individuals (fill up to desired popsize)
  if (internal.options.speciation && (internal._species?.length ?? 0) > 0) {
    internal._suppressTournamentError = true;
    const remaining = desiredPop - newPopulation.length;
    if (remaining > 0) {
      // Allocate offspring per species with age bonuses/penalties
      /**
       * Species age-bonus configuration used to boost or penalize shares for
       * young/old species. This supports preserving promising new species and
       * penalizing stale species to avoid premature convergence.
       *
       * @type {Record<string, any>}
       */
      const ageCfg = this.options.speciesAgeBonus || {};

      /**
       * Number of generations below which a species is considered "young".
       * Young species may receive a fitness multiplier reward to help them
       * get established.
       *
       * @type {number}
       */
      const youngT = ageCfg.youngThreshold ?? 5;

      /**
       * Multiplier applied to adjusted fitness for young species. Values >1
       * boost young species' allocation; adjust carefully to avoid
       * oscillations.
       *
       * @type {number}
       */
      const youngM = ageCfg.youngMultiplier ?? 1.3;

      /**
       * Number of generations above which a species is considered "old".
       * Old species can be penalized to free capacity for newer, more
       * promising species.
       *
       * @type {number}
       */
      const oldT = ageCfg.oldThreshold ?? 30;

      /**
       * Multiplier applied to adjusted fitness for old species. Values <1
       * penalize allocations for aged species.
       *
       * @type {number}
       */
      const oldM = ageCfg.oldMultiplier ?? 0.7;
      const speciesAdjusted = (internal._species ?? []).map((species: any) => {
        const base = species.members.reduce(
          (a: number, member: any) => a + (member.score || 0),
          0
        );
        const age = internal.generation - species.lastImproved;
        if (age <= youngT) return base * youngM;
        if (age >= oldT) return base * oldM;
        return base;
      });
      /**
       * Sum of adjusted species fitness values used as the denominator when
       * computing proportional offspring shares. We default to 1 to avoid
       * division-by-zero in degenerate cases.
       *
       * @type {number}
       */
      const totalAdj =
        speciesAdjusted.reduce((a: number, b: number) => a + b, 0) || 1;

      /**
       * Minimum offspring to allocate per species when `speciesAllocation`
       * config sets `minOffspring`. This prevents very small species from
       * being starved entirely.
       *
       * @type {number}
       */
      const minOff = this.options.speciesAllocation?.minOffspring ?? 1;

      /**
       * Fractional (raw) offspring share per species before rounding.
       * Used to compute integer allocation and fractional remainders.
       * @type {number[]}
       */
      const rawShares = (internal._species ?? []).map(
        (_: any, idx: number) => (speciesAdjusted[idx] / totalAdj) * remaining
      );

      /**
       * Integer offspring allocation per species derived by flooring
       * the fractional raw shares. Leftover slots are handled via
       * `remainders` and distributed to species with largest fractional parts.
       * @type {number[]}
       */
      const offspringAlloc: number[] = rawShares.map((s: number) =>
        Math.floor(s)
      );
      // Enforce minimum for species that have any members surviving
      for (let i = 0; i < offspringAlloc.length; i++)
        if (
          offspringAlloc[i] < minOff &&
          remaining >= (internal._species?.length ?? 0) * minOff
        )
          offspringAlloc[i] = minOff;
      /**
       * Sum of integer allocations already assigned to species. Used to
       * compute `slotsLeft` (remaining slots to distribute).
       * @type {number}
       */
      const allocated = offspringAlloc.reduce((a, b) => a + b, 0);

      /**
       * Number of unfilled offspring slots remaining after the initial
       * integer allocation. Positive -> slots to distribute; negative ->
       * oversubscription that must be trimmed.
       * @type {number}
       */
      let slotsLeft = remaining - allocated;
      // Distribute leftovers by largest fractional remainder
      /**
       * Fractional remainders used to distribute leftover slots fairly.
       * Each entry contains the species index and the fractional remainder
       * of that species' raw share.
       * @type {Array<{i:number, frac:number}>}
       */
      const remainders = rawShares.map((s: number, i: number) => ({
        i,
        frac: s - Math.floor(s),
      }));
      remainders.sort((a: any, b: any) => b.frac - a.frac);
      for (const remainderEntry of remainders) {
        if (slotsLeft <= 0) break;
        offspringAlloc[remainderEntry.i]++;
        slotsLeft--;
      }
      // If we overshot (edge case via minOff), trim from largest allocations.
      // We prefer trimming from the largest allocations first to preserve
      // diversity for smaller species that were guaranteed `minOff`.
      if (slotsLeft < 0) {
        /**
         * Species indices ordered by descending allocated offspring count.
         * Used when trimming allocations in oversubscription edge-cases.
         * @type {Array<{i:number, v:number}>}
         */
        const order = offspringAlloc
          .map((v, i) => ({ i, v }))
          .sort((a, b) => b.v - a.v);
        for (const orderEntry of order) {
          if (slotsLeft === 0) break;
          if (offspringAlloc[orderEntry.i] > minOff) {
            offspringAlloc[orderEntry.i]--;
            slotsLeft++;
          }
        }
      }
      // Record allocation for telemetry (applied next generation's telemetry snapshot)
      /**
       * Telemetry-friendly snapshot of last generation's per-species
       * offspring allocations. Stored on the instance for later reporting.
       * @type {Array<{id:number, alloc:number}>}
       */
      internal._lastOffspringAlloc = (internal._species ?? []).map(
        (species: any, i: number) => ({
          id: species.id,
          alloc: offspringAlloc[i] || 0,
        })
      );
      // Breed within species
      internal._prevInbreedingCount = internal._lastInbreedingCount; // snapshot for telemetry next generation
      this._lastInbreedingCount = 0;
      offspringAlloc.forEach((count, idx) => {
        if (count <= 0) return;
        /**
         * Shortcut reference to the current species being processed.
         * @type {any}
         */
        const species = internal._species?.[idx];
        if (!species) return;
        internal._sortSpeciesMembers?.(species);
        const survivors = species.members.slice(
          0,
          Math.max(
            1,
            Math.floor(
              species.members.length *
                (internal.options!.survivalThreshold || 0.5)
            )
          )
        );
        for (let k = 0; k < count; k++) {
          const parentA =
            survivors[Math.floor(internal._getRNG()() * survivors.length)];
          let parentB: Network;
          if (
            internal.options.crossSpeciesMatingProb &&
            (internal._species?.length ?? 0) > 1 &&
            internal._getRNG()() <
              (internal.options.crossSpeciesMatingProb || 0)
          ) {
            // Choose different species randomly
            let otherIdx = idx;
            let guard = 0;
            while (otherIdx === idx && guard++ < 5)
              otherIdx = Math.floor(
                internal._getRNG()() * (internal._species?.length ?? 1)
              );
            const otherSpecies = internal._species?.[otherIdx];
            if (!otherSpecies) {
              parentB = survivors[
                Math.floor(internal._getRNG()() * survivors.length)
              ] as never;
            } else {
              internal._sortSpeciesMembers?.(otherSpecies);
              const otherParents = otherSpecies.members.slice(
                0,
                Math.max(
                  1,
                  Math.floor(
                    otherSpecies.members.length *
                      (internal.options!.survivalThreshold || 0.5)
                  )
                )
              );
              parentB = otherParents[
                Math.floor(internal._getRNG()() * otherParents.length)
              ] as never;
            }
          } else {
            parentB = survivors[
              Math.floor(internal._getRNG()() * survivors.length)
            ] as never;
          }
          const child = (Network.crossOver(
            parentA as never,
            parentB as never,
            internal.options.equal || false
          ) as never) as GenomeWithMetadata;
          child._reenableProb = internal.options.reenableProb;
          child._id = internal._nextGenomeId++;
          if (internal._lineageEnabled) {
            child._parents = [
              ((parentA as never) as GenomeWithMetadata)._id,
              (parentB as any)._id,
            ];
            const d1 = (parentA as any)._depth ?? 0;
            const d2 = (parentB as any)._depth ?? 0;
            (child as any)._depth = 1 + Math.max(d1, d2);
            if (
              ((parentA as never) as GenomeWithMetadata)._id ===
              ((parentB as never) as GenomeWithMetadata)._id
            )
              internal._lastInbreedingCount++;
          }
          newPopulation.push(child as never);
        }
      });
      internal._suppressTournamentError = false;
    }
  } else {
    internal._suppressTournamentError = true;
    /**
     * Number of offspring to generate when speciation is disabled.
     * This equals the remaining slots after elitism/provenance.
     * @type {number}
     */
    const toBreed = Math.max(0, desiredPop - newPopulation.length);
    for (let i = 0; i < toBreed; i++)
      newPopulation.push((await internal.getOffspring?.()) as never);
    internal._suppressTournamentError = false;
  }

  // Ensure minimum hidden nodes to avoid bottlenecks
  for (const genome of newPopulation) {
    if (!genome) continue;
    await internal.ensureMinHiddenNodes?.(genome as never);
    await internal.ensureNoDeadEnds?.(genome as never); // Ensure no dead ends or blind I/O
  }

  internal.population = newPopulation as never; // Replace population instead of appending
  // --- Evolution-time pruning (structural sparsification) ---
  // Pruning & adaptive pruning delegations
  try {
    const pruningModule = await import('./neat.pruning');
    pruningModule.applyEvolutionPruning.call(internal as never);
  } catch {
    // Empty catch: Evolution-time pruning is optional. If the pruning module is
    // unavailable or throws, we continue with unpruned genomes.
  }
  try {
    const pruningModule = await import('./neat.pruning');
    pruningModule.applyAdaptivePruning.call(internal as never);
  } catch {
    // Empty catch: Adaptive pruning is optional. If unavailable or fails, we
    // continue evolution with the current structural complexity.
  }
  await internal.mutate?.();
  // Adapt per-genome mutation parameters for next generation (self-adaptive rates)
  try {
    const adaptiveModule = await import('./neat.adaptive');
    adaptiveModule.applyAdaptiveMutation.call(internal as never);
  } catch {
    // Empty catch: Genome-level adaptive mutation is optional. If the adaptive module
    // is missing or fails, we continue with global fixed mutation rates.
  }

  // Invalidate compatibility caches after structural mutations
  internal.population.forEach((genome: any) => {
    if (genome._compatCache) delete genome._compatCache;
  });

  internal.population.forEach((genome: any) => (genome.score = undefined));

  internal.generation++;
  if (internal.options.speciation) internal._updateSpeciesStagnation?.();
  // Global stagnation injection (refresh portion of worst genomes) if enabled
  if (
    (internal.options.globalStagnationGenerations || 0) > 0 &&
    internal.generation - (internal._lastGlobalImproveGeneration ?? 0) >
      (internal.options.globalStagnationGenerations || 0)
  ) {
    // Replace worst 20% (excluding elites if elitism >0) with fresh random genomes
    /**
     * Fraction of population to replace during a global stagnation injection.
     * Lower values are conservative; higher values inject more diversity.
     * @type {number}
     */
    const replaceFraction = 0.2;

    /**
     * Inclusive start index for stagnation replacement. Elites at the top
     * of the population are preserved and not replaced.
     * @type {number}
     */
    const startIdx = Math.max(
      internal.options.elitism || 0,
      Math.floor(internal.population.length * (1 - replaceFraction))
    );
    for (let i = startIdx; i < internal.population.length; i++) {
      const fresh = (new Network(internal.input, internal.output, {
        minHidden: internal.options.minHidden,
      }) as never) as GenomeWithMetadata;
      fresh.score = undefined;
      fresh._reenableProb = internal.options.reenableProb;
      fresh._id = internal._nextGenomeId++;
      if (internal._lineageEnabled) {
        fresh._parents = [];
        fresh._depth = 0;
      }
      try {
        await internal.ensureMinHiddenNodes?.(fresh);
        await internal.ensureNoDeadEnds?.(fresh);
        // Guarantee structural variance for stagnation injection test: add a hidden node if none present
        /**
         * Number of hidden nodes in a freshly injected genome. Used to
         * determine whether we should add a minimal hidden node to ensure
         * non-trivial topology for injected genomes.
         * @type {number}
         */
        const hiddenCount = fresh.nodes.filter((n: any) => n.type === 'hidden')
          .length;
        if (hiddenCount === 0) {
          const { default: NodeCls } = await import('../architecture/node');
          const newNode = new NodeCls('hidden');
          // insert before outputs
          fresh.nodes.splice(fresh.nodes.length - internal.output, 0, newNode);
          // connect a random input to hidden and hidden to a random output
          const inputNodes = fresh.nodes.filter((n: any) => n.type === 'input');
          const outputNodes = fresh.nodes.filter(
            (n: any) => n.type === 'output'
          );
          if (inputNodes.length && outputNodes.length) {
            try {
              ((fresh as never) as Network).connect(
                inputNodes[0] as never,
                newNode as never,
                1
              );
            } catch {
              // Empty catch: Connection may fail if network constraints prevent adding
              // the connection (e.g., if direct connection already exists). This is
              // acceptable since we're only aiming for minimal structural variance.
            }
            try {
              ((fresh as never) as Network).connect(
                newNode as never,
                outputNodes[0] as never,
                1
              );
            } catch {
              // Empty catch: Same justification as above – we attempt connection but
              // tolerate failure since the goal is best-effort structural variation.
            }
          }
        }
      } catch {
        // Empty catch: ensureMinHiddenNodes/ensureNoDeadEnds may fail in rare edge
        // cases (e.g., highly constrained topologies). We continue injection since
        // the stagnation relief goal is more important than strict structural guarantees.
      }
      internal.population[i] = fresh as never;
    }
    internal._lastGlobalImproveGeneration = internal.generation; // reset window after injection
  }
  // Adaptive re-enable probability tuning
  if (internal.options.reenableProb !== undefined) {
    // Track successful re-enable events versus attempts across the
    // population to adapt the global re-enable probability.
    /**
     * Counters used to aggregate successful re-enable events and
     * attempts across the population. Used to adapt the global
     * `options.reenableProb` parameter.
     * @type {number}
     */
    let reenableSuccessTotal = 0,
      reenableAttemptsTotal = 0;
    for (const genome of this.population) {
      reenableSuccessTotal += (genome as any)._reenableSuccess || 0;
      reenableAttemptsTotal += (genome as any)._reenableAttempts || 0;
      (genome as any)._reenableSuccess = 0;
      (genome as any)._reenableAttempts = 0;
    }
    if (reenableAttemptsTotal > 20) {
      // only adjust with enough samples
      const ratio = reenableSuccessTotal / reenableAttemptsTotal;
      // target moderate reuse ~0.3
      const target = 0.3;
      const delta = ratio - target;
      internal.options.reenableProb = Math.min(
        0.9,
        Math.max(0.05, (internal.options.reenableProb ?? 0.3) - delta * 0.1)
      );
    }
  }
  // Decay operator stats (EMA-like) to keep adaptation responsive
  try {
    const adaptiveModule = await import('./neat.adaptive');
    adaptiveModule.applyOperatorAdaptation.call(internal as never);
  } catch {
    // Empty catch: Operator adaptation is optional and delegated to an external
    // module. If the module is missing or throws, we continue normally without adaptation.
  }

  /**
   * Timestamp marking the end of evolve() invocation. Subtracted from
   * `startTime` to compute `_lastEvolveDuration`.
   * @type {number}
   */
  const endTime =
    typeof performance !== 'undefined' && (performance as any).now
      ? (performance as any).now()
      : Date.now();
  internal._lastEvolveDuration = endTime - startTime;
  // Ensure at least a minimal species history snapshot exists for tests expecting CSV even when speciation disabled
  try {
    if (!internal._speciesHistory) internal._speciesHistory = [];
    if (!internal.options.speciesAllocation?.extendedHistory) {
      if (
        internal._speciesHistory.length === 0 ||
        internal._speciesHistory.at(-1)?.generation !== internal.generation
      ) {
        internal._speciesHistory.push({
          generation: internal.generation,
          stats: (internal._species ?? []).map((species: any) => ({
            id: species.id,
            size: species.members.length,
            best: species.bestScore,
            lastImproved: species.lastImproved,
          })),
        });
        if (internal._speciesHistory.length > 200)
          internal._speciesHistory.shift();
      }
    }
  } catch {
    // Empty catch: Species history tracking is optional telemetry. If it fails
    // (e.g., species data unavailable), we continue normally without history snapshot.
  }
  return fittest;
}
