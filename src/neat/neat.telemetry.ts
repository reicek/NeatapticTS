// Telemetry stream and recording helpers

import type {
  NeatLike,
  TelemetryEntry,
  ObjImportance,
  SpeciesAlloc,
  DiversityStats,
  GenomeDetailed,
  NeatOptions,
  ObjEvent,
  NodeLike,
  ConnectionLike,
} from './neat.types';
import { EPSILON } from './neat.constants';
import { computeAncestorUniqueness, buildAnc } from './neat.lineage';
import type { NeatLineageContext as LineageContext } from './neat.lineage';

// --- Local telemetry helper types ---
/** Minimal genome shape used by telemetry helpers (kept local to avoid
 * scattering lightweight shapes across other type files). */
export type TelemetryGenome = {
  nodes: NodeLike[];
  connections: ConnectionLike[];
  _depth?: number;
};

export type TelemetryDiversityOptions = {
  diversityMetrics?: {
    enabled?: boolean;
    pairSample?: number;
    graphletSample?: number;
  };
  fastMode?: boolean;
  novelty?: { enabled?: boolean; k?: number };
};

/** Context view used within telemetry helpers to access optional internal
 * fields with descriptive names rather than repeated inline casts. */
interface TelemetryContext extends NeatLike {
  generation?: number;
  options?: NeatOptions &
    TelemetryDiversityOptions & {
      telemetryStream?: {
        enabled?: boolean;
        onEntry?: (entry: TelemetryEntry) => void;
      };
    };
  population?: TelemetryGenome[] | GenomeDetailed[];
  _getRNG?: () => () => number;
  _compatibilityDistance?: (a: TelemetryGenome, b: TelemetryGenome) => number;
  _structuralEntropy?: (g: TelemetryGenome) => number;
  _getObjectives?: () => { key: string }[];
  _diversityStats?: DiversityStats;
  _operatorStats?: Map<string, { success: number; attempts: number }>;
  _species?: unknown[];
  _lastObjImportance?: ObjImportance;
  _objectiveAges?: Map<string, number>;
  _pendingObjectiveAdds?: string[];
  _pendingObjectiveRemoves?: string[];
  _lastOffspringAlloc?: SpeciesAlloc[];
  _rngState?: unknown;
  _lineageEnabled?: boolean;
  _lastMeanDepth?: number;
  _prevInbreedingCount?: number;
  _lastMeanNodes?: number;
  _lastMeanConns?: number;
  _lastEvalDuration?: number;
  _lastEvolveDuration?: number;
  _telemetrySelect?: Set<string>;
  _telemetry?: TelemetryEntry[];
  _objectiveEvents?: Array<Record<string, unknown>>;
}

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

export function applyTelemetrySelect(
  this: TelemetryContext,
  entry: Record<string, unknown>,
): Record<string, unknown> {
  // Step 1: Fast-path, nothing to do when no selection set is configured
  const ctx = this as TelemetryContext;
  const selectionSet = ctx._telemetrySelect;
  if (!selectionSet || selectionSet.size === 0) return entry;

  // Step 2: Core telemetry fields always preserved
  const coreFields = ['gen', 'best', 'species'] as const;
  const core: Partial<Record<string, unknown>> = {};
  for (const field of coreFields) {
    if (Object.hasOwn(entry, field)) {
      core[field] = entry[field];
    }
  }

  // Step 3: Remove non-core keys not in the selection set
  for (const key of Object.keys(entry)) {
    if (coreFields.includes(key as (typeof coreFields)[number])) continue;
    if (!selectionSet.has(key)) {
      delete entry[key];
    }
  }

  // Step 4: Re-attach the core fields (ensures ordering and presence)
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
export function structuralEntropy(
  this: NeatLike,
  graph: {
    nodes: Array<{ geneId: number }>;
    connections: Array<{
      from: { geneId: number };
      to: { geneId: number };
      enabled: boolean;
    }>;
    [key: string]: unknown;
  },
): number {
  // Step 1: Return cached value when available and valid for current generation
  const ctx = this as TelemetryContext;
  if (
    (graph as Record<string, unknown>)._entropyGen === ctx.generation &&
    typeof (graph as Record<string, unknown>)._entropyVal === 'number'
  ) {
    return (graph as Record<string, unknown>)._entropyVal as number;
  }

  // Step 2: Mapping from each node's unique gene identifier to the degree
  const degreeCounts: Record<number, number> = {};
  for (const node of graph.nodes) degreeCounts[node.geneId] = 0;

  // Step 3: Accumulate degrees from enabled connections
  for (const connection of graph.connections) {
    if (connection.enabled) {
      const fromId = connection.from.geneId;
      const toId = connection.to.geneId;
      if (degreeCounts[fromId] !== undefined) degreeCounts[fromId]!++;
      if (degreeCounts[toId] !== undefined) degreeCounts[toId]!++;
    }
  }

  // Step 4: Build histogram of degree frequencies
  const degreeHistogram: Record<number, number> = {};
  const nodeCount = graph.nodes.length || 1;
  for (const nodeId in degreeCounts) {
    const degree = degreeCounts[Number(nodeId)];
    degreeHistogram[degree] = (degreeHistogram[degree] || 0) + 1;
  }

  // Step 5: Compute entropy H = -sum p * log(p)
  let entropy = 0;
  for (const degree in degreeHistogram) {
    const probability = degreeHistogram[degree] / nodeCount;
    if (probability > 0)
      entropy -= probability * Math.log(probability + EPSILON);
  }

  // Step 6: Cache result on the graph object for the current generation
  (graph as Record<string, unknown>)._entropyGen = ctx.generation;
  (graph as Record<string, unknown>)._entropyVal = entropy;
  return entropy;
}

/**
 * Compute several diversity statistics used by telemetry reporting.
 *
 * This helper is intentionally conservative in runtime: when `fastMode` is enabled it will automatically tune a few sampling defaults to keep the computation cheap. The computed statistics are written to `this._diversityStats` as an object with keys like `meanCompat` and `graphletEntropy`.
 *
 * @remarks
 * - Uses random sampling of pairs and 3-node subgraphs (graphlets) to approximate diversity metrics.
 * @example
 * // compute and store diversity stats onto the neat instance
 * neat.options.diversityMetrics = { enabled: true };
 * neat.computeDiversityStats();
 * console.log(neat._diversityStats.meanCompat);
 */
export function computeDiversityStats(this: NeatLike) {
  const ctx = this as TelemetryContext;
  const options = ctx.options as NeatOptions & TelemetryDiversityOptions;
  if (!options.diversityMetrics?.enabled) return;

  // Fast-mode nudges
  if (
    options.fastMode &&
    !(ctx as { _fastModeTuned?: boolean })._fastModeTuned
  ) {
    const diversityMetrics = options.diversityMetrics;
    if (diversityMetrics) {
      if (diversityMetrics.pairSample == null) diversityMetrics.pairSample = 20;
      if (diversityMetrics.graphletSample == null)
        diversityMetrics.graphletSample = 30;
    }
    if (options.novelty?.enabled && options.novelty.k == null)
      options.novelty.k = 5;
    (ctx as { _fastModeTuned?: boolean })._fastModeTuned = true;
  }

  // Sampling parameters and population reference
  const pairSample = options.diversityMetrics.pairSample ?? 40;
  const graphletSample = options.diversityMetrics.graphletSample ?? 60;
  const population = (ctx.population as TelemetryGenome[]) ?? [];
  const popSize = population.length;

  // Pairwise compatibility sampling
  let compatibilitySum = 0;
  let compatibilitySumSq = 0;
  let compatibilityCount = 0;
  const rngFactory: () => () => number =
    typeof ctx._getRNG === 'function' ? ctx._getRNG : () => Math.random;
  for (let iter = 0; iter < pairSample; iter++) {
    if (popSize < 2) break;
    const rng = rngFactory();
    const firstIndex = Math.floor(rng() * popSize);
    let secondIndex = Math.floor(rng() * popSize);
    if (secondIndex === firstIndex) secondIndex = (secondIndex + 1) % popSize;
    const distance =
      ctx._compatibilityDistance?.(
        population[firstIndex],
        population[secondIndex],
      ) ?? 0;
    compatibilitySum += distance;
    compatibilitySumSq += distance * distance;
    compatibilityCount++;
  }
  const meanCompat = compatibilityCount
    ? compatibilitySum / compatibilityCount
    : 0;
  const varCompat = compatibilityCount
    ? Math.max(
        0,
        compatibilitySumSq / compatibilityCount - meanCompat * meanCompat,
      )
    : 0;

  // Structural entropy across population
  const entropies = population.map((genome) =>
    (ctx as TelemetryContext)._structuralEntropy
      ? (ctx as TelemetryContext)._structuralEntropy!(genome as TelemetryGenome)
      : structuralEntropy.call(
          ctx,
          genome as {
            nodes: Array<{ geneId: number }>;
            connections: Array<{
              from: { geneId: number };
              to: { geneId: number };
              enabled: boolean;
            }>;
          },
        ),
  );
  const meanEntropy =
    entropies.reduce((a: number, b: number) => a + b, 0) /
    (entropies.length || 1);
  const varEntropy = entropies.length
    ? entropies.reduce(
        (a: number, b: number) => a + (b - meanEntropy) * (b - meanEntropy),
        0,
      ) / entropies.length
    : 0;

  // Graphlet (3-node motif) sampling
  const motifCounts = [0, 0, 0, 0];
  for (let iter = 0; iter < graphletSample; iter++) {
    if (popSize === 0) break;
    const rng = rngFactory();
    const genome = population[Math.floor(rng() * popSize)];
    if (!genome) break;
    if (genome.nodes.length < 3) continue;
    const selectedIndices = new Set<number>();
    while (selectedIndices.size < 3)
      selectedIndices.add(Math.floor(rng() * genome.nodes.length));
    const selectedNodes = Array.from(selectedIndices).map(
      (i) => genome.nodes[i],
    );
    let edgeCount = 0;
    for (const connection of genome.connections) {
      if (
        connection.enabled &&
        selectedNodes.includes(connection.from) &&
        selectedNodes.includes(connection.to)
      )
        edgeCount++;
    }
    if (edgeCount > 3) edgeCount = 3;
    motifCounts[edgeCount]++;
  }
  const totalMotifs = motifCounts.reduce((a, b) => a + b, 0) || 1;
  let graphletEntropy = 0;
  for (let k = 0; k < motifCounts.length; k++) {
    const probability = motifCounts[k] / totalMotifs;
    if (probability > 0) graphletEntropy -= probability * Math.log(probability);
  }

  // Lineage-based statistics (if enabled)
  let lineageMeanDepth = 0;
  let lineageMeanPairDist = 0;
  if (ctx._lineageEnabled && popSize > 0) {
    const depths = population.map((genome) => genome._depth ?? 0);
    lineageMeanDepth =
      depths.reduce((a: number, b: number) => a + b, 0) / popSize;
    let lineagePairSum = 0;
    let lineagePairN = 0;
    const pairsToSample = Math.min(pairSample, (popSize * (popSize - 1)) / 2);
    for (let iter = 0; iter < pairsToSample; iter++) {
      if (popSize < 2) break;
      const rng = rngFactory();
      const i = Math.floor(rng() * popSize);
      let j = Math.floor(rng() * popSize);
      if (j === i) j = (j + 1) % popSize;
      lineagePairSum += Math.abs(depths[i] - depths[j]);
      lineagePairN++;
    }
    lineageMeanPairDist = lineagePairN ? lineagePairSum / lineagePairN : 0;
  }

  // Store computed stats on instance
  (ctx as { _diversityStats?: object })._diversityStats = {
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
  const ctx = this as TelemetryContext;
  try {
    applyTelemetrySelect.call(ctx, entry as Record<string, unknown>);
  } catch {
    // ignored: telemetry selection errors are non-fatal and safe to drop
  }

  if (!ctx._telemetry) ctx._telemetry = [];
  ctx._telemetry.push(entry);

  try {
    const telemetryStream = ctx.options?.telemetryStream;
    if (
      telemetryStream?.enabled &&
      typeof telemetryStream.onEntry === 'function'
    ) {
      telemetryStream.onEntry(entry);
    }
  } catch {
    // ignored: telemetry stream callback errors should not disrupt evolution
  }

  // Keep the in-memory telemetry buffer bounded to avoid runaway memory usage
  if (ctx._telemetry.length > 500) ctx._telemetry.shift();
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
  fittest: Record<string, unknown>,
): TelemetryEntry {
  const ctx = this as TelemetryContext;
  /**
   * Current generation index for this telemetry snapshot.
   * Anchors all reported statistics to a single evolutionary timestep.
   * @example
   * // use the generation number when inspecting recorded telemetry
   * const generation = neat.generation;
   */
  const gen = ctx.generation ?? 0;

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
  const options = ctx.options || {};
  if (options.multiObjective?.enabled) {
    /**
     * Selected complexity metric used to penalize genomes in the hypervolume
     * proxy. Allowed values: 'nodes' | 'connections'. Defaults to 'connections'.
     * @example
     * // penalize by number of connections
     * neat.options.multiObjective.complexityMetric = 'connections';
     */
    const complexityMetric: 'nodes' | 'connections' =
      options.multiObjective?.complexityMetric || 'connections';

    /**
     * Primary objective scalar values for the current population. These are
     * used to compute normalization bounds when forming the hypervolume
     * proxy so all scores lie in a comparable [0,1] range.
     */
    const population = (ctx.population as GenomeDetailed[]) || [];
    const primaryObjectiveScores: number[] = population.map(
      (genome: GenomeDetailed) => (genome.score as number) || 0,
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
    const paretoFrontSizes: number[] = [];

    // Collect sizes of the first few Pareto fronts
    for (let r = 0; r < 5; r++) {
      const size = population.filter(
        (g) => ((g as GenomeDetailed)._moRank ?? 0) === r,
      ).length;
      if (!size) break;
      paretoFrontSizes.push(size);
    }

    // Compute a simple hypervolume proxy: normalized score weighted by inverse complexity
    // Accumulate hypervolume proxy contributions from Pareto-front genomes
    for (const genome of population) {
      const rank = genome._moRank ?? 0;
      if (rank !== 0) continue; // only consider Pareto front 0
      const normalizedScore =
        maxPrimaryScore > minPrimaryScore
          ? ((genome.score || 0) - minPrimaryScore) /
            (maxPrimaryScore - minPrimaryScore)
          : 0;
      const genomeComplexity =
        complexityMetric === 'nodes'
          ? genome.nodes.length
          : genome.connections.length;
      hyperVolumeProxy += normalizedScore * (1 / (genomeComplexity + 1));
    }

    /**
     * Snapshot of operator statistics. Each entry is an object describing a
     * genetic operator with counts for successful applications and attempts.
     * These are useful for visualizations showing operator effectiveness.
     * @example
     * // [{ op: 'mutate.addNode', succ: 12, att: 50 }, ...]
     */
    const operatorStatsSnapshot = Array.from(
      (ctx._operatorStats ?? new Map()).entries(),
    ).map(([opName, stats]) => ({
      op: opName,
      succ: stats.success,
      att: stats.attempts,
    }));

    /**
     * Telemetry entry assembled in multi-objective mode. Contains core
     * statistics plus MO-specific proxies and optional detailed snapshots.
     * This object is suitable for recording or streaming as-is.
     *
     * @example
     * // peek at current generation telemetry
     * console.log(entry.gen, entry.best, entry.hyper);
     */
    const entry: TelemetryEntry & Record<string, unknown> = {
      gen,
      best: (fittest.score as number) ?? 0,
      species: ctx._species?.length ?? 0,
      hyper: hyperVolumeProxy,
      fronts: paretoFrontSizes,
      diversity: ctx._diversityStats,
      ops: operatorStatsSnapshot,
      objImportance: {},
    };

    if (!entry.objImportance) entry.objImportance = {};
    if (ctx._lastObjImportance) {
      const lastImportance = ctx._lastObjImportance;
      if (typeof lastImportance === 'object' && lastImportance !== null)
        entry.objImportance = lastImportance;
    }

    /**
     * Optional snapshot of objective ages: a map objectiveKey -> age (generations).
     */
    if (ctx._objectiveAges?.size) {
      entry.objAges = Object.fromEntries(ctx._objectiveAges.entries());
    }

    // Record pending objective lifecycle events (adds/removes) for telemetry
    if (
      ctx._pendingObjectiveAdds?.length ||
      ctx._pendingObjectiveRemoves?.length
    ) {
      entry.objEvents = [];
      for (const k of ctx._pendingObjectiveAdds || [])
        entry.objEvents.push({ type: 'add', key: k });
      for (const k of ctx._pendingObjectiveRemoves || [])
        entry.objEvents.push({ type: 'remove', key: k });
      ctx._objectiveEvents = ctx._objectiveEvents || [];
      ctx._objectiveEvents.push(
        ...entry.objEvents.map((e: ObjEvent) => ({
          gen,
          type: e.type,
          key: e.key,
        })),
      );
      ctx._pendingObjectiveAdds = [];
      ctx._pendingObjectiveRemoves = [];
    }

    /**
     * Optional per-species offspring allocation snapshot from the most recent
     * allocation calculation. Used for tracking reproductive budgets.
     */
    if (ctx._lastOffspringAlloc) {
      const lastAlloc = ctx._lastOffspringAlloc;
      if (Array.isArray(lastAlloc)) entry.speciesAlloc = lastAlloc.slice();
    }
    try {
      entry.objectives = ctx._getObjectives?.().map((o) => o.key) || [];
    } catch {
      // ignored: objective provider not present — skip objectives snapshot
    }
    if (options.rngState && ctx._rngState !== undefined)
      entry.rng = ctx._rngState as number | undefined;

    if (ctx._lineageEnabled) {
      const bestGenome = (
        ctx.population as GenomeDetailed[]
      )[0] as GenomeDetailed;
      const depths = (ctx.population as GenomeDetailed[]).map(
        (g: GenomeDetailed) => g._depth ?? 0,
      );
      ctx._lastMeanDepth =
        depths.reduce((a: number, b: number) => a + b, 0) /
        (depths.length || 1);
      // Use dynamic import for ES2023, avoid require
      // NOTE: This assumes the import is available synchronously, otherwise refactor to async

      // Build a lightweight lineage context from the telemetry context when available
      const lineageCtx = {
        population: (ctx.population as GenomeDetailed[]) || [],
        _getRNG:
          typeof ctx._getRNG === 'function' ? ctx._getRNG : () => Math.random,
      } as LineageContext;
      const ancestorUniqueness = computeAncestorUniqueness.call(lineageCtx);
      entry.lineage = {
        parents: Array.isArray(bestGenome._parents)
          ? bestGenome._parents.slice()
          : [],
        depthBest: bestGenome._depth ?? 0,
        meanDepth: +(ctx._lastMeanDepth ?? 0).toFixed(2),
        inbreeding: ctx._prevInbreedingCount ?? 0,
        ancestorUniq: ancestorUniqueness,
      };
    }

    if (options.telemetry?.hypervolume && options.multiObjective?.enabled)
      entry.hv = +hyperVolumeProxy.toFixed(4);

    if (options.telemetry?.complexity) {
      const nodesArr = population.map((g) => g.nodes.length);
      const connsArr = population.map((g) => g.connections.length);
      const meanNodes =
        nodesArr.reduce((a: number, b: number) => a + b, 0) /
        (nodesArr.length || 1);
      const meanConns =
        connsArr.reduce((a: number, b: number) => a + b, 0) /
        (connsArr.length || 1);
      const maxNodes = nodesArr.length ? Math.max(...nodesArr) : 0;
      const maxConns = connsArr.length ? Math.max(...connsArr) : 0;
      const enabledRatios = population.map((g) => {
        let enabled = 0,
          disabled = 0;
        for (const c of g.connections) {
          if (c.enabled === false) disabled++;
          else enabled++;
        }
        return enabled + disabled ? enabled / (enabled + disabled) : 0;
      });
      const meanEnabledRatio =
        enabledRatios.reduce((a: number, b: number) => a + b, 0) /
        (enabledRatios.length || 1);
      const growthNodes =
        (this as NeatLike & { _lastMeanNodes?: number })._lastMeanNodes !==
        undefined
          ? meanNodes -
            (this as NeatLike & { _lastMeanNodes: number })._lastMeanNodes
          : 0;
      const growthConns =
        (this as NeatLike & { _lastMeanConns?: number })._lastMeanConns !==
        undefined
          ? meanConns -
            (this as NeatLike & { _lastMeanConns: number })._lastMeanConns
          : 0;
      (
        this as NeatLike & {
          _lastMeanNodes?: number;
        }
      )._lastMeanNodes = meanNodes;
      (
        this as NeatLike & {
          _lastMeanConns?: number;
        }
      )._lastMeanConns = meanConns;
      entry.complexity = {
        meanNodes: +meanNodes.toFixed(2),
        meanConns: +meanConns.toFixed(2),
        maxNodes,
        maxConns,
        meanEnabledRatio: +meanEnabledRatio.toFixed(3),
        growthNodes: +growthNodes.toFixed(2),
        growthConns: +growthConns.toFixed(2),
        budgetMaxNodes: options.maxNodes ?? 0,
        budgetMaxConns: options.maxConns ?? 0,
      };
    }

    if (options.telemetry?.performance)
      entry.perf = {
        evalMs: (this as NeatLike & { _lastEvalDuration?: number })
          ._lastEvalDuration,
        evolveMs: (this as NeatLike & { _lastEvolveDuration?: number })
          ._lastEvolveDuration,
      };
    return entry;
  }

  // Fallback path (mono-objective) retained for parity with legacy behavior.
  /**
   * Snapshot of operator statistics for mono-objective mode. Kept separate
   * from the MO snapshot to document the intent and avoid accidental
   * coupling.
   */
  const operatorStatsSnapshotMono = Array.from(
    (ctx._operatorStats ?? new Map()).entries(),
  ).map(([opName, stats]) => ({
    op: opName,
    succ: stats.success,
    att: stats.attempts,
  }));

  /**
   * Telemetry entry object for mono-objective mode. Aligns with the
   * multi-objective structure but omits MO-only fields like `fronts`.
   */
  const entry: TelemetryEntry & Record<string, unknown> = {
    gen,
    best: (fittest.score as number) ?? 0,
    species: ctx._species?.length ?? 0,
    hyper: hyperVolumeProxy,
    diversity: ctx._diversityStats,
    ops: operatorStatsSnapshotMono,
    objImportance: {},
  };

  if (ctx._lastObjImportance)
    entry.objImportance = ctx._lastObjImportance as ObjImportance;
  if (ctx._objectiveAges?.size)
    entry.objAges = Object.fromEntries(ctx._objectiveAges.entries());

  if (
    ctx._pendingObjectiveAdds?.length ||
    ctx._pendingObjectiveRemoves?.length
  ) {
    entry.objEvents = [];
    for (const k of ctx._pendingObjectiveAdds || [])
      entry.objEvents.push({ type: 'add', key: k });
    for (const k of ctx._pendingObjectiveRemoves || [])
      entry.objEvents.push({ type: 'remove', key: k });
    ctx._objectiveEvents = ctx._objectiveEvents || [];
    ctx._objectiveEvents.push(
      ...entry.objEvents.map((e: ObjEvent) => ({
        gen,
        type: e.type,
        key: e.key,
      })),
    );
    ctx._pendingObjectiveAdds = [];
    ctx._pendingObjectiveRemoves = [];
  }

  if (ctx._lastOffspringAlloc)
    entry.speciesAlloc = ctx._lastOffspringAlloc?.slice();
  try {
    entry.objectives = ctx._getObjectives?.().map((o) => o.key) || [];
  } catch {
    // ignored: optional objective provider absent
  }
  if (ctx.options?.rngState && ctx._rngState !== undefined)
    entry.rng = ctx._rngState as number | undefined;

  if (ctx._lineageEnabled) {
    const bestGenome = (
      ctx.population as GenomeDetailed[]
    )[0] as GenomeDetailed;
    const depths = (ctx.population as GenomeDetailed[]).map(
      (g: GenomeDetailed) => g._depth ?? 0,
    );
    ctx._lastMeanDepth =
      depths.reduce((a: number, b: number) => a + b, 0) / (depths.length || 1);
    // Use dynamic import for ES2023, avoid require
    // NOTE: This assumes the import is available synchronously, otherwise refactor to async

    // use imported buildAnc helper
    let sampledPairs = 0;
    let jaccardSum = 0;
    const popLength = (ctx.population as GenomeDetailed[]).length;
    const samplePairs = Math.min(30, (popLength * (popLength - 1)) / 2);
    for (let t = 0; t < samplePairs; t++) {
      if (popLength < 2) break;
      const rngFn =
        typeof ctx._getRNG === 'function' ? ctx._getRNG() : Math.random;
      const i = Math.floor(rngFn() * popLength);
      let j = Math.floor(rngFn() * popLength);
      if (j === i) j = (j + 1) % popLength;
      const lineageCtx2 = {
        population: (ctx.population as GenomeDetailed[]) || [],
        _getRNG:
          typeof ctx._getRNG === 'function' ? ctx._getRNG : () => Math.random,
      } as LineageContext;
      const ancestorsA = buildAnc.call(
        lineageCtx2,
        (ctx.population as GenomeDetailed[])[i],
      );
      const ancestorsB = buildAnc.call(
        lineageCtx2,
        (ctx.population as GenomeDetailed[])[j],
      );
      if (ancestorsA.size === 0 && ancestorsB.size === 0) continue;
      let intersectionCount = 0;
      for (const id of ancestorsA) if (ancestorsB.has(id)) intersectionCount++;
      const union = ancestorsA.size + ancestorsB.size - intersectionCount || 1;
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
      meanDepth: +(ctx._lastMeanDepth ?? 0).toFixed(2),
      inbreeding: ctx._prevInbreedingCount ?? 0,
      ancestorUniq: ancestorUniqueness,
    };
  }

  if (
    (this as NeatLike & { options: NeatOptions }).options.telemetry
      ?.hypervolume &&
    (this as NeatLike & { options: NeatOptions }).options.multiObjective
      ?.enabled
  )
    entry.hv = +hyperVolumeProxy.toFixed(4);
  if (
    (this as NeatLike & { options: NeatOptions }).options.telemetry?.complexity
  ) {
    const nodesArr = (
      this as NeatLike & {
        population: GenomeDetailed[];
      }
    ).population.map((g: GenomeDetailed) => g.nodes.length);
    const connsArr = (
      this as NeatLike & {
        population: GenomeDetailed[];
      }
    ).population.map((g: GenomeDetailed) => g.connections.length);
    const meanNodes =
      nodesArr.reduce((a: number, b: number) => a + b, 0) /
      (nodesArr.length || 1);
    const meanConns =
      connsArr.reduce((a: number, b: number) => a + b, 0) /
      (connsArr.length || 1);
    const maxNodes = nodesArr.length ? Math.max(...nodesArr) : 0;
    const maxConns = connsArr.length ? Math.max(...connsArr) : 0;
    const enabledRatios = (
      this as NeatLike & {
        population: GenomeDetailed[];
      }
    ).population.map((g: GenomeDetailed) => {
      let en = 0,
        dis = 0;
      for (const c of g.connections) {
        if (c.enabled === false) dis++;
        else en++;
      }
      return en + dis ? en / (en + dis) : 0;
    });
    const meanEnabledRatio =
      enabledRatios.reduce((a: number, b: number) => a + b, 0) /
      (enabledRatios.length || 1);
    const growthNodes =
      (this as NeatLike & { _lastMeanNodes?: number })._lastMeanNodes !==
      undefined
        ? meanNodes -
          (this as NeatLike & { _lastMeanNodes: number })._lastMeanNodes
        : 0;
    const growthConns =
      (this as NeatLike & { _lastMeanConns?: number })._lastMeanConns !==
      undefined
        ? meanConns -
          (this as NeatLike & { _lastMeanConns: number })._lastMeanConns
        : 0;
    (this as NeatLike & { _lastMeanNodes?: number })._lastMeanNodes = meanNodes;
    (this as NeatLike & { _lastMeanConns?: number })._lastMeanConns = meanConns;
    entry.complexity = {
      meanNodes: +meanNodes.toFixed(2),
      meanConns: +meanConns.toFixed(2),
      maxNodes,
      maxConns,
      meanEnabledRatio: +meanEnabledRatio.toFixed(3),
      growthNodes: +growthNodes.toFixed(2),
      growthConns: +growthConns.toFixed(2),
      budgetMaxNodes:
        (this as NeatLike & { options: NeatOptions }).options.maxNodes ?? 0,
      budgetMaxConns:
        (this as NeatLike & { options: NeatOptions }).options.maxConns ?? 0,
    };
  }
  if (
    (this as NeatLike & { options: NeatOptions }).options.telemetry?.performance
  )
    entry.perf = {
      evalMs: (this as NeatLike & { _lastEvalDuration?: number })
        ._lastEvalDuration,
      evolveMs: (this as NeatLike & { _lastEvolveDuration?: number })
        ._lastEvolveDuration,
    };
  return entry;
}
