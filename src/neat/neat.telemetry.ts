// Telemetry stream and recording helpers

import { NeatLike, TelemetryEntry } from './neat.types';

/**
 * Apply a telemetry selection whitelist to a telemetry entry.
 *
 * This helper inspects a per-instance Set of telemetry keys stored at
 * `this._telemetrySelect`. If present, only keys included in the set are
 * retained on the produced entry. Core fields (generation, best score and
 * species count) are always preserved.
 *
 * Example:
 * @example
 * // keep only 'gen', 'best', 'species' and 'diversity' fields
 * neat._telemetrySelect = new Set(['diversity']);
 * applyTelemetrySelect.call(neat, entry);
 *
 * @param entry - Raw telemetry object to be filtered in-place.
 * @returns The filtered telemetry object (same reference as input).
 */
export function applyTelemetrySelect(this: NeatLike, entry: any): any {
  // fast-path: nothing to do when no selection set is configured
  if (!(this as any)._telemetrySelect || !(this as any)._telemetrySelect.size)
    return entry;

  /**
   * Set of telemetry keys explicitly selected by the user for reporting.
   * Only properties whose keys are present in this set will be retained on the
   * telemetry entry (besides core fields which are always preserved).
   */
  /** Set of telemetry keys the user has chosen to keep when reporting. */
  const keep = (this as any)._telemetrySelect as Set<string>;

  /**
   * Core telemetry fields that are always preserved regardless of the
   * selection set to guarantee downstream consumers receive the minimal
   * structured snapshot required for charts and logs.
   */
  /** Core telemetry fields always preserved: gen, best, species. */
  const core = { gen: entry.gen, best: entry.best, species: entry.species };

  // Iterate over entry keys and delete any non-core keys not in the keep set.
  for (const key of Object.keys(entry)) {
    // preserve core fields always
    if (key in core) continue;
    if (!keep.has(key)) delete entry[key];
  }

  // Re-attach the core fields (ensures ordering and presence)
  return Object.assign(entry, core);
}

/**
 * Lightweight proxy for structural entropy based on degree-distribution.
 *
 * This function computes an approximate entropy of a graph topology by
 * counting node degrees and computing the entropy of the degree histogram.
 * The result is cached on the graph object for the current generation in
 * `_entropyVal` to avoid repeated expensive recomputation.
 *
 * Example:
 * @example
 * const H = structuralEntropy.call(neat, genome);
 * console.log(`Structure entropy: ${H.toFixed(3)}`);
 *
 * @param graph - A genome-like object with `nodes` and `connections` arrays.
 * @returns A non-negative number approximating structural entropy.
 */
export function structuralEntropy(this: NeatLike, graph: any): number {
  const anyG = graph as any;

  // Return cached value when available and valid for current generation
  if (
    anyG._entropyGen === (this as any).generation &&
    typeof anyG._entropyVal === 'number'
  )
    return anyG._entropyVal;

  /**
   * Mapping from each node's unique gene identifier to the degree (number of
   * incident enabled connections). Initialized to 0 for every node prior to
   * accumulation of connection endpoints.
   */
  /** Map from node geneId to degree (enabled incident connections). */
  const degreeCounts: Record<number, number> = {};

  // Initialize degree counts for every node in the graph
  for (const node of graph.nodes) degreeCounts[(node as any).geneId] = 0;

  // Accumulate degrees from enabled connections
  for (const conn of graph.connections)
    if (conn.enabled) {
      const fromId = (conn.from as any).geneId;
      const toId = (conn.to as any).geneId;
      if (degreeCounts[fromId] !== undefined) degreeCounts[fromId]++;
      if (degreeCounts[toId] !== undefined) degreeCounts[toId]++;
    }

  /**
   * Histogram where each key is an observed degree and each value is the
   * number of nodes exhibiting that degree within the current genome.
   */
  /** Histogram mapping degree -> frequency of nodes with that degree. */
  const degreeHistogram: Record<number, number> = {};

  /**
   * Number of nodes (cardinality of degreeCounts) used to normalize degree
   * frequencies into probabilities. Defaults to 1 to avoid divide-by-zero.
   */
  /** Number of nodes in the graph (falls back to 1). */
  const nodeCount = graph.nodes.length || 1;

  // Build histogram of degree frequencies
  for (const nodeId in degreeCounts) {
    const d = degreeCounts[nodeId as any];
    degreeHistogram[d] = (degreeHistogram[d] || 0) + 1;
  }

  // Compute entropy H = -sum p * log(p)
  let entropy = 0;
  for (const k in degreeHistogram) {
    const p = degreeHistogram[k as any] / nodeCount;
    if (p > 0) entropy -= p * Math.log(p + EPSILON);
  }

  // Cache result on the graph object for the current generation
  anyG._entropyGen = (this as any).generation;
  anyG._entropyVal = entropy;
  return entropy;
}

/**
 * Compute several diversity statistics used by telemetry reporting.
 *
 * This helper is intentionally conservative in runtime: when `fastMode` is
 * enabled it will automatically tune a few sampling defaults to keep the
 * computation cheap. The computed statistics are written to
 * `this._diversityStats` as an object with keys like `meanCompat` and
 * `graphletEntropy`.
 *
 * The method mutates instance-level temporary fields and reads a number of
 * runtime options from `this.options`.
 *
 * @remarks
 * - Uses random sampling of pairs and 3-node subgraphs (graphlets) to
 *   approximate diversity metrics.
 *
 * Example:
 * @example
 * // compute and store diversity stats onto the neat instance
 * neat.options.diversityMetrics = { enabled: true };
 * neat.computeDiversityStats();
 * console.log(neat._diversityStats.meanCompat);
 */
export function computeDiversityStats(this: NeatLike) {
  // Ensure the feature is enabled in options
  if (!(this as any).options.diversityMetrics?.enabled) return;

  // If running in fast mode, nudge sensible sampling defaults once
  if ((this as any).options.fastMode && !(this as any)._fastModeTuned) {
    const dm = (this as any).options.diversityMetrics;
    if (dm) {
      if (dm.pairSample == null) dm.pairSample = 20;
      if (dm.graphletSample == null) dm.graphletSample = 30;
    }
    if (
      (this as any).options.novelty?.enabled &&
      (this as any).options.novelty.k == null
    )
      (this as any).options.novelty.k = 5;
    (this as any)._fastModeTuned = true;
  }

  /** Number of random pairwise samples to draw for compatibility stats. */
  /**
   * Target number of random genome pairs sampled to estimate mean and
   * variance of compatibility distance. A smaller fixed-size sample keeps
   * runtime sub-linear in population size while still providing a stable
   * signal for diversity trend tracking.
   */
  const pairSample = (this as any).options.diversityMetrics.pairSample ?? 40;

  /** Number of 3-node graphlets to sample for motif statistics. */
  /**
   * Number of randomly selected 3-node subgraphs (graphlets) whose internal
   * enabled edge counts are tallied to approximate motif distribution and
   * structural diversity.
   */
  const graphletSample =
    (this as any).options.diversityMetrics.graphletSample ?? 60;

  /** Reference to the current population array (genomes). */
  /**
   * Array reference to the active population for the current generation.
   * This is sampled repeatedly for compatibility and motif statistics.
   */
  const population = (this as any).population;

  /** Cached population size (length of `population`). */
  /**
   * Population size scalar cached to avoid repeated property lookups in
   * inner sampling loops where micro-optimizations marginally reduce GC.
   */
  const popSize = population.length;

  // --- Pairwise compatibility sampling -------------------------------------------------
  /** Sum of compatibility distances sampled. */
  let compatSum = 0;
  /** Sum of squared compatibility distances (for variance). */
  let compatSq = 0;
  /** Number of compatibility pairs sampled. */
  let compatCount = 0;

  for (let iter = 0; iter < pairSample; iter++) {
    // If population too small, stop sampling
    if (popSize < 2) break;
    const i = Math.floor((this as any)._getRNG()() * popSize);
    let j = Math.floor((this as any)._getRNG()() * popSize);
    if (j === i) j = (j + 1) % popSize;
    const d = (this as any)._compatibilityDistance(
      population[i],
      population[j]
    );
    compatSum += d;
    compatSq += d * d;
    compatCount++;
  }

  /** Mean compatibility distance from pairwise sampling. */
  const meanCompat = compatCount ? compatSum / compatCount : 0;

  /** Sample variance of compatibility distances (floored at zero). */
  const varCompat = compatCount
    ? Math.max(0, compatSq / compatCount - meanCompat * meanCompat)
    : 0;

  // --- Structural entropy across population -------------------------------------------
  /** Structural entropies for each genome in the population. */
  const entropies = population.map((g: any) =>
    (this as any)._structuralEntropy(g)
  );

  /** Mean structural entropy across the population. */
  const meanEntropy =
    entropies.reduce((a: number, b: number) => a + b, 0) /
    (entropies.length || 1);

  /** Variance of structural entropy across the population. */
  const varEntropy = entropies.length
    ? entropies.reduce(
        (a: number, b: number) => a + (b - meanEntropy) * (b - meanEntropy),
        0
      ) / entropies.length
    : 0;

  // --- Graphlet (3-node motif) sampling -----------------------------------------------
  /** Counters for 3-node motif types (index = number of edges 0..3). */
  /**
   * Frequency counters for sampled 3-node motifs grouped by how many enabled
   * edges connect the three chosen nodes. Index corresponds to edge count.
   */
  const motifCounts = [0, 0, 0, 0];

  for (let iter = 0; iter < graphletSample; iter++) {
    const g = population[Math.floor((this as any)._getRNG()() * popSize)];
    if (!g) break;
    // skip tiny genomes
    if (g.nodes.length < 3) continue;

    /** Set of random node indices used to form a 3-node graphlet. */
    const selectedIdxs = new Set<number>();
    while (selectedIdxs.size < 3)
      selectedIdxs.add(Math.floor((this as any)._getRNG()() * g.nodes.length));

    /** Selected node objects corresponding to sampled indices. */
    const selectedNodes = Array.from(selectedIdxs).map((i) => g.nodes[i]);

    let edges = 0;
    for (const c of g.connections)
      if (c.enabled) {
        if (selectedNodes.includes(c.from) && selectedNodes.includes(c.to))
          edges++;
      }
    if (edges > 3) edges = 3;
    motifCounts[edges]++;
  }

  /** Total number of motifs sampled (for normalization). */
  const totalMotifs = motifCounts.reduce((a, b) => a + b, 0) || 1;

  /** Entropy over 3-node motif type distribution. */
  let graphletEntropy = 0;
  for (let k = 0; k < motifCounts.length; k++) {
    const p = motifCounts[k] / totalMotifs;
    if (p > 0) graphletEntropy -= p * Math.log(p);
  }

  // --- Lineage-based statistics (if enabled) -----------------------------------------
  /** Mean depth of genomes in the lineage tree (if enabled). */
  let lineageMeanDepth = 0;

  /** Mean pairwise difference in lineage depth. */
  let lineageMeanPairDist = 0;

  if ((this as any)._lineageEnabled && popSize > 0) {
    const depths = population.map((g: any) => (g as any)._depth ?? 0);
    lineageMeanDepth =
      depths.reduce((a: number, b: number) => a + b, 0) / popSize;

    /** Sum of absolute differences between sampled lineage depths. */
    let lineagePairSum = 0;
    /** Number of lineage pairs sampled. */
    let lineagePairN = 0;
    for (
      let iter = 0;
      iter < Math.min(pairSample, (popSize * (popSize - 1)) / 2);
      iter++
    ) {
      if (popSize < 2) break;
      const i = Math.floor((this as any)._getRNG()() * popSize);
      let j = Math.floor((this as any)._getRNG()() * popSize);
      if (j === i) j = (j + 1) % popSize;
      lineagePairSum += Math.abs(depths[i] - depths[j]);
      lineagePairN++;
    }
    lineageMeanPairDist = lineagePairN ? lineagePairSum / lineagePairN : 0;
  }

  // Store the computed diversity statistics on the instance for telemetry
  (this as any)._diversityStats = {
    meanCompat,
    varCompat,
    meanEntropy,
    varEntropy,
    graphletEntropy,
    lineageMeanDepth,
    lineageMeanPairDist,
  };
}

/**
 * Record a telemetry entry into the instance buffer and optionally stream it.
 *
 * Steps:
 * This method performs the following steps to persist and optionally stream telemetry:
 * 1. Apply `applyTelemetrySelect` to filter fields according to user selection.
 * 2. Ensure `this._telemetry` buffer exists and push the entry.
 * 3. If a telemetry stream callback is configured, call it.
 * 4. Trim the buffer to a conservative max size (500 entries).
 *
 * Example:
 * @example
 * // record a simple telemetry entry from inside the evolve loop
 * neat.recordTelemetryEntry({ gen: neat.generation, best: neat.population[0].score });
 * @param entry - Telemetry entry to record.
 */
export function recordTelemetryEntry(this: NeatLike, entry: TelemetryEntry) {
  try {
    applyTelemetrySelect.call(this as any, entry);
  } catch {}

  if (!(this as any)._telemetry) (this as any)._telemetry = [];
  (this as any)._telemetry.push(entry);

  try {
    if (
      (this as any).options.telemetryStream?.enabled &&
      (this as any).options.telemetryStream.onEntry
    )
      (this as any).options.telemetryStream.onEntry(entry);
  } catch {}

  // Keep the in-memory telemetry buffer bounded to avoid runaway memory usage
  if ((this as any)._telemetry.length > 500) (this as any)._telemetry.shift();
}

/**
 * Build a comprehensive telemetry entry for the current generation.
 *
 * The returned object contains a snapshot of population statistics, multi-
 * objective front sizes, operator statistics, lineage summaries and optional
 * complexity/performance metrics depending on configured telemetry options.
 *
 * This function intentionally mirrors the legacy in-loop telemetry construction
 * to preserve behavior relied upon by tests and consumers.
 *
 * Example:
 * @example
 * // build a telemetry snapshot for the current generation
 * const snapshot = neat.buildTelemetryEntry(neat.population[0]);
 * neat.recordTelemetryEntry(snapshot);
 *
 * @param fittest - The currently fittest genome (used to report `best` score).
 * @returns A TelemetryEntry object suitable for recording/streaming.
 */
export function buildTelemetryEntry(
  this: NeatLike,
  fittest: any
): TelemetryEntry {
  /**
   * Current generation index for this telemetry snapshot.
   * Anchors all reported statistics to a single evolutionary timestep.
   * @example
   * // use the generation number when inspecting recorded telemetry
   * const generation = neat.generation;
   */
  const gen = (this as any).generation;

  // ---------------------------------------------------------------------------
  // Multi-objective (MO) path: compute MO-specific telemetry when enabled.
  // Method steps:
  // 1) Compute a lightweight hypervolume-like proxy over the first Pareto
  //    front to summarize quality + parsimony.
  // 2) Collect sizes of the first few Pareto fronts to observe convergence.
  // 3) Snapshot operator statistics (success/attempt counts).
  // 4) Attach diversity, lineage and objective meta-data if available.
  // 5) Optionally attach complexity & perf metrics based on options.
  // 6) Return the assembled telemetry entry.
  // ---------------------------------------------------------------------------

  /**
   * Running accumulator for a lightweight hypervolume-like proxy.
   * This heuristic weights normalized objective score by inverse complexity
   * so smaller Pareto-optimal solutions are favored. Not a formal HV.
   */
  let hyperVolumeProxy = 0;
  if ((this as any).options.multiObjective?.enabled) {
    /**
     * Complexity dimension name used to penalize solutions inside the
     * hypervolume proxy. Expected values: 'nodes' or 'connections'.
     * @example
     * // penalize by number of connections
     * neat.options.multiObjective.complexityMetric = 'connections';
     */
    /**
     * Selected complexity metric used to penalize genomes in the hypervolume
     * proxy. Allowed values: 'nodes' | 'connections'. Defaults to 'connections'.
     * @example
     * // penalize by number of connections
     * neat.options.multiObjective.complexityMetric = 'connections';
     */
    const complexityMetric =
      (this as any).options.multiObjective.complexityMetric || 'connections';

    /**
     * Primary objective scalar values for the current population. These are
     * used to compute normalization bounds when forming the hypervolume
     * proxy so all scores lie in a comparable [0,1] range.
     */
    /**
     * Array of primary objective scalars (one per genome). Used to compute
     * normalization bounds so scores are comparable when forming the proxy.
     */
    const primaryObjectiveScores = (this as any).population.map(
      (genome: any) => genome.score || 0
    );

    /** Minimum observed primary objective score in the population. */
    const minPrimaryScore = Math.min(...primaryObjectiveScores);

    /** Maximum observed primary objective score in the population. */
    const maxPrimaryScore = Math.max(...primaryObjectiveScores);

    /**
     * Collection of Pareto front sizes for the first few ranks (0..4).
     * Recording only the early fronts keeps telemetry compact while showing
     * population partitioning across non-dominated sets.
     */
    /**
     * Sizes of the first few Pareto fronts (front 0..4). Recording only the
     * early fronts keeps telemetry compact while showing partitioning.
     */
    const paretoFrontSizes: number[] = [];

    // Collect sizes of the first few Pareto fronts
    for (let r = 0; r < 5; r++) {
      const size = (this as any).population.filter(
        (g: any) => ((g as any)._moRank ?? 0) === r
      ).length;
      if (!size) break;
      paretoFrontSizes.push(size);
    }

    // Compute a simple hypervolume proxy: normalized score weighted by inverse complexity
    // Accumulate hypervolume proxy contributions from Pareto-front genomes
    for (const genome of (this as any).population) {
      const rank = (genome as any)._moRank ?? 0;
      if (rank !== 0) continue; // only consider Pareto front 0
      /**
       * Normalized primary objective score in [0,1]. When all scores are
       * identical normalization would divide by zero, so we guard and treat
       * contributions as 0 in that degenerate case.
       */
      /**
       * Normalized primary objective score in [0,1]. Guards against
       * divide-by-zero when all scores are identical by treating contribution
       * as 0 in that degenerate case.
       */
      const normalizedScore =
        maxPrimaryScore > minPrimaryScore
          ? ((genome.score || 0) - minPrimaryScore) /
            (maxPrimaryScore - minPrimaryScore)
          : 0;

      /**
       * Genome complexity measured along the chosen complexity metric. This
       * is used to apply a parsimony penalty so simpler genomes contribute
       * proportionally more to the hypervolume proxy.
       */
      /**
       * Genome complexity measured along the chosen complexity metric. Used
       * to apply a parsimony penalty so simpler genomes contribute more.
       */
      const genomeComplexity =
        complexityMetric === 'nodes'
          ? genome.nodes.length
          : genome.connections.length;

      // Accumulate the proxy (higher is better): score scaled by inverse complexity
      hyperVolumeProxy += normalizedScore * (1 / (genomeComplexity + 1));
    }

    /**
     * Snapshot of operator statistics. Each entry is an object describing a
     * genetic operator with counts for successful applications and attempts.
     * These are useful for visualizations showing operator effectiveness.
     * @example
     * // [{ op: 'mutate.addNode', succ: 12, att: 50 }, ...]
     */
    /**
     * Snapshot of operator statistics collected from the running counters.
     * Each entry contains the operator name and its success/attempt counts.
     * Useful for diagnostics and operator effectiveness visualizations.
     * @example
     * // [{ op: 'mutate.addNode', succ: 12, att: 50 }, ...]
     */
    /**
     * Snapshot of operator statistics: an array of {op, succ, att} objects
     * where `succ` is the number of successful applications and `att` is
     * the total attempts. Helpful for diagnostics and operator visualizations.
     * @example
     * // [{ op: 'mutate.addNode', succ: 12, att: 50 }, ...]
     */
    const operatorStatsSnapshot = (Array.from(
      (this as any)._operatorStats.entries()
    ) as any[]).map(([opName, stats]: any) => ({
      op: opName,
      succ: stats.success,
      att: stats.attempts,
    }));

    /**
     * Telemetry entry being constructed for multi-objective mode. Contains
     * core metrics, the MO proxies and optional snapshots such as diversity,
     * operator stats, lineage and complexity metrics. This object is later
     * augmented conditionally based on enabled features.
     */
    /**
     * Telemetry entry assembled in multi-objective mode. Contains core
     * statistics plus MO-specific proxies and optional detailed snapshots.
     * This object is suitable for recording or streaming as-is.
     *
     * @example
     * // peek at current generation telemetry
     * console.log(entry.gen, entry.best, entry.hyper);
     */
    const entry: any = {
      gen,
      best: fittest.score,
      species: (this as any)._species.length,
      hyper: hyperVolumeProxy,
      fronts: paretoFrontSizes,
      diversity: (this as any)._diversityStats,
      ops: operatorStatsSnapshot,
    };

    if (!entry.objImportance) entry.objImportance = {};
    // objective importance snapshot already computed in evolve and stored on temp property if any
    if ((this as any)._lastObjImportance)
      entry.objImportance = (this as any)._lastObjImportance;

    /**
     * Optional snapshot of objective ages: a map objectiveKey -> age (generations).
     */
    if ((this as any)._objectiveAges?.size) {
      entry.objAges = (Array.from(
        (this as any)._objectiveAges.entries()
      ) as any[]).reduce((a: any, kv: any) => {
        a[kv[0]] = kv[1];
        return a;
      }, {} as any);
    }

    // Record pending objective lifecycle events (adds/removes) for telemetry
    if (
      (this as any)._pendingObjectiveAdds?.length ||
      (this as any)._pendingObjectiveRemoves?.length
    ) {
      entry.objEvents = [] as any[];
      for (const k of (this as any)._pendingObjectiveAdds)
        entry.objEvents.push({ type: 'add', key: k });
      for (const k of (this as any)._pendingObjectiveRemoves)
        entry.objEvents.push({ type: 'remove', key: k });
      (this as any)._objectiveEvents.push(
        ...entry.objEvents.map((e: any) => ({ gen, type: e.type, key: e.key }))
      );
      (this as any)._pendingObjectiveAdds = [];
      (this as any)._pendingObjectiveRemoves = [];
    }

    /**
     * Optional per-species offspring allocation snapshot from the most recent
     * allocation calculation. Used for tracking reproductive budgets.
     */
    if ((this as any)._lastOffspringAlloc)
      entry.speciesAlloc = (this as any)._lastOffspringAlloc.slice();
    try {
      entry.objectives = ((this as any)._getObjectives() as any[]).map(
        (o: any) => o.key
      );
    } catch {}
    if (
      ((this as any).options as any).rngState &&
      (this as any)._rngState !== undefined
    )
      entry.rng = (this as any)._rngState;

    if ((this as any)._lineageEnabled) {
      /**
       * Best genome in the population (index 0 assumed to be fittest in the
       * maintained sort order). Used to capture parent references and depth.
       */
      const bestGenome = (this as any).population[0] as any;
      const depths = (this as any).population.map(
        (g: any) => (g as any)._depth ?? 0
      );
      (this as any)._lastMeanDepth =
        depths.reduce((a: number, b: number) => a + b, 0) /
        (depths.length || 1);
      const { computeAncestorUniqueness } = require('./neat.lineage');
      const ancestorUniqueness = computeAncestorUniqueness.call(this as any);
      entry.lineage = {
        parents: Array.isArray(bestGenome._parents)
          ? bestGenome._parents.slice()
          : [],
        depthBest: bestGenome._depth ?? 0,
        meanDepth: +(this as any)._lastMeanDepth.toFixed(2),
        inbreeding: (this as any)._prevInbreedingCount,
        ancestorUniq: ancestorUniqueness,
      };
    }

    if (
      (this as any).options.telemetry?.hypervolume &&
      (this as any).options.multiObjective?.enabled
    )
      entry.hv = +hyperVolumeProxy.toFixed(4);

    if ((this as any).options.telemetry?.complexity) {
      const nodesArr = (this as any).population.map((g: any) => g.nodes.length);
      const connsArr = (this as any).population.map(
        (g: any) => g.connections.length
      );
      const meanNodes =
        nodesArr.reduce((a: number, b: number) => a + b, 0) /
        (nodesArr.length || 1);
      const meanConns =
        connsArr.reduce((a: number, b: number) => a + b, 0) /
        (connsArr.length || 1);
      const maxNodes = nodesArr.length ? Math.max(...nodesArr) : 0;
      const maxConns = connsArr.length ? Math.max(...connsArr) : 0;
      const enabledRatios = (this as any).population.map((g: any) => {
        let enabled = 0,
          disabled = 0;
        for (const c of g.connections) {
          if ((c as any).enabled === false) disabled++;
          else enabled++;
        }
        return enabled + disabled ? enabled / (enabled + disabled) : 0;
      });
      const meanEnabledRatio =
        enabledRatios.reduce((a: number, b: number) => a + b, 0) /
        (enabledRatios.length || 1);
      const growthNodes =
        (this as any)._lastMeanNodes !== undefined
          ? meanNodes - (this as any)._lastMeanNodes
          : 0;
      const growthConns =
        (this as any)._lastMeanConns !== undefined
          ? meanConns - (this as any)._lastMeanConns
          : 0;
      (this as any)._lastMeanNodes = meanNodes;
      (this as any)._lastMeanConns = meanConns;
      entry.complexity = {
        meanNodes: +meanNodes.toFixed(2),
        meanConns: +meanConns.toFixed(2),
        maxNodes,
        maxConns,
        meanEnabledRatio: +meanEnabledRatio.toFixed(3),
        growthNodes: +growthNodes.toFixed(2),
        growthConns: +growthConns.toFixed(2),
        budgetMaxNodes: (this as any).options.maxNodes,
        budgetMaxConns: (this as any).options.maxConns,
      };
    }

    if ((this as any).options.telemetry?.performance)
      entry.perf = {
        evalMs: (this as any)._lastEvalDuration,
        evolveMs: (this as any)._lastEvolveDuration,
      };
    return entry;
  }

  // Fallback path (mono-objective) retained for parity with legacy behavior.
  /**
   * Snapshot of operator statistics for mono-objective mode. Kept separate
   * from the MO snapshot to document the intent and avoid accidental
   * coupling.
   */
  const operatorStatsSnapshotMono = (Array.from(
    (this as any)._operatorStats.entries()
  ) as any[]).map(([opName, stats]: any) => ({
    op: opName,
    succ: stats.success,
    att: stats.attempts,
  }));

  /**
   * Telemetry entry object for mono-objective mode. Aligns with the
   * multi-objective structure but omits MO-only fields like `fronts`.
   */
  const entry: TelemetryEntry = {
    gen,
    best: fittest.score,
    species: (this as any)._species.length,
    hyper: hyperVolumeProxy,
    diversity: (this as any)._diversityStats,
    ops: operatorStatsSnapshotMono,
    objImportance: {},
  } as TelemetryEntry;

  if ((this as any)._lastObjImportance)
    entry.objImportance = (this as any)._lastObjImportance;
  if ((this as any)._objectiveAges?.size)
    entry.objAges = (Array.from(
      (this as any)._objectiveAges.entries()
    ) as any[]).reduce((a: any, kv: any) => {
      a[kv[0]] = kv[1];
      return a;
    }, {} as any);

  if (
    (this as any)._pendingObjectiveAdds?.length ||
    (this as any)._pendingObjectiveRemoves?.length
  ) {
    entry.objEvents = [] as any[];
    for (const k of (this as any)._pendingObjectiveAdds)
      entry.objEvents.push({ type: 'add', key: k });
    for (const k of (this as any)._pendingObjectiveRemoves)
      entry.objEvents.push({ type: 'remove', key: k });
    (this as any)._objectiveEvents.push(
      ...entry.objEvents.map((e: any) => ({ gen, type: e.type, key: e.key }))
    );
    (this as any)._pendingObjectiveAdds = [];
    (this as any)._pendingObjectiveRemoves = [];
  }

  if ((this as any)._lastOffspringAlloc)
    entry.speciesAlloc = (this as any)._lastOffspringAlloc.slice();
  try {
    entry.objectives = ((this as any)._getObjectives() as any[]).map(
      (o: any) => o.key
    );
  } catch {}
  if (
    ((this as any).options as any).rngState &&
    (this as any)._rngState !== undefined
  )
    entry.rng = (this as any)._rngState;

  if ((this as any)._lineageEnabled) {
    /**
     * Best genome in the population (index 0 assumed to be fittest in the
     * maintained sort order). Used to capture parent references and depth.
     */
    const bestGenome = (this as any).population[0] as any;

    /**
     * Array of lineage depths for each genome in the population. Depth is a
     * lightweight proxy of ancestry tree height for each genome.
     */
    const depths = (this as any).population.map(
      (g: any) => (g as any)._depth ?? 0
    );
    (this as any)._lastMeanDepth =
      depths.reduce((a: number, b: number) => a + b, 0) / (depths.length || 1);

    const { buildAnc } = require('./neat.lineage');

    /**
     * Number of lineage pairwise samples actually evaluated. Used to
     * normalize the averaged Jaccard-like ancestor uniqueness metric.
     */
    let sampledPairs = 0;

    /**
     * Running sum of Jaccard-like distances between sampled ancestor sets.
     */
    let jaccardSum = 0;

    /**
     * Maximum number of random pairs to sample when estimating ancestor
     * uniqueness. Bounds runtime while providing a stable estimate.
     */
    const samplePairs = Math.min(
      30,
      ((this as any).population.length *
        ((this as any).population.length - 1)) /
        2
    );

    for (let t = 0; t < samplePairs; t++) {
      if ((this as any).population.length < 2) break;
      const i = Math.floor(
        (this as any)._getRNG()() * (this as any).population.length
      );
      let j = Math.floor(
        (this as any)._getRNG()() * (this as any).population.length
      );
      if (j === i) j = (j + 1) % (this as any).population.length;

      /**
       * Ancestor sets for the two randomly chosen genomes used to compute a
       * Jaccard-like dissimilarity (1 - intersection/union).
       */
      const ancestorsA = buildAnc.call(
        this as any,
        (this as any).population[i] as any
      );
      const ancestorsB = buildAnc.call(
        this as any,
        (this as any).population[j] as any
      );
      if (ancestorsA.size === 0 && ancestorsB.size === 0) continue;
      let intersectionCount = 0;
      for (const id of ancestorsA) if (ancestorsB.has(id)) intersectionCount++;
      const union = ancestorsA.size + ancestorsB.size - intersectionCount || 1;

      /**
       * Jaccard-like dissimilarity between ancestor sets. A value near 1
       * indicates little shared ancestry; near 0 indicates high overlap.
       */
      const jaccardDistance = 1 - intersectionCount / union;
      jaccardSum += jaccardDistance;
      sampledPairs++;
    }

    const ancestorUniqueness = sampledPairs
      ? +(jaccardSum / sampledPairs).toFixed(3)
      : 0;
    entry.lineage = {
      parents: Array.isArray(bestGenome._parents)
        ? bestGenome._parents.slice()
        : [],
      depthBest: bestGenome._depth ?? 0,
      meanDepth: +(this as any)._lastMeanDepth.toFixed(2),
      inbreeding: (this as any)._prevInbreedingCount,
      ancestorUniq: ancestorUniqueness,
    };
  }

  if (
    (this as any).options.telemetry?.hypervolume &&
    (this as any).options.multiObjective?.enabled
  )
    entry.hv = +hyperVolumeProxy.toFixed(4);
  if ((this as any).options.telemetry?.complexity) {
    const nodesArr = (this as any).population.map((g: any) => g.nodes.length);
    const connsArr = (this as any).population.map(
      (g: any) => g.connections.length
    );
    const meanNodes =
      nodesArr.reduce((a: number, b: number) => a + b, 0) /
      (nodesArr.length || 1);
    const meanConns =
      connsArr.reduce((a: number, b: number) => a + b, 0) /
      (connsArr.length || 1);
    const maxNodes = nodesArr.length ? Math.max(...nodesArr) : 0;
    const maxConns = connsArr.length ? Math.max(...connsArr) : 0;
    const enabledRatios = (this as any).population.map((g: any) => {
      let en = 0,
        dis = 0;
      for (const c of g.connections) {
        if ((c as any).enabled === false) dis++;
        else en++;
      }
      return en + dis ? en / (en + dis) : 0;
    });
    const meanEnabledRatio =
      enabledRatios.reduce((a: number, b: number) => a + b, 0) /
      (enabledRatios.length || 1);
    const growthNodes =
      (this as any)._lastMeanNodes !== undefined
        ? meanNodes - (this as any)._lastMeanNodes
        : 0;
    const growthConns =
      (this as any)._lastMeanConns !== undefined
        ? meanConns - (this as any)._lastMeanConns
        : 0;
    (this as any)._lastMeanNodes = meanNodes;
    (this as any)._lastMeanConns = meanConns;
    entry.complexity = {
      meanNodes: +meanNodes.toFixed(2),
      meanConns: +meanConns.toFixed(2),
      maxNodes,
      maxConns,
      meanEnabledRatio: +meanEnabledRatio.toFixed(3),
      growthNodes: +growthNodes.toFixed(2),
      growthConns: +growthConns.toFixed(2),
      budgetMaxNodes: (this as any).options.maxNodes,
      budgetMaxConns: (this as any).options.maxConns,
    };
  }
  if ((this as any).options.telemetry?.performance)
    entry.perf = {
      evalMs: (this as any)._lastEvalDuration,
      evolveMs: (this as any)._lastEvolveDuration,
    };
  return entry;
}
