/**
 * Telemetry recorder orchestration for generation snapshots.
 *
 * This chapter is the write-heavy counterpart to the read-oriented telemetry
 * facade. The neighboring `runtime/` and `metrics/` chapters own the small
 * mechanics and computation helpers, while this recorder chapter keeps the
 * end-to-end flow readable: shape a generation snapshot, optionally filter it,
 * and persist or stream it without destabilizing evolution.
 */

import type {
  NeatLike,
  TelemetryEntry,
  ObjImportance,
  SpeciesAlloc,
  DiversityStats,
  GenomeDetailed,
  NeatOptions,
  ObjectiveEvent,
} from '../../shared/neat.shared.types';
import {
  ensureTelemetryBuffer,
  safelyStreamTelemetryEntry,
  trimTelemetryBuffer,
} from '../runtime/telemetry.runtime';
import {
  getCachedEntropy,
  computeDegreeCounts,
  buildDegreeHistogram,
  computeEntropyFromHistogram,
  setCachedEntropy,
  getTelemetryCoreSnapshot,
  stripUnselectedTelemetryKeys,
  mergeTelemetryCoreFields,
  applyFastModeDefaults,
  computeCompatibilityStats,
  computeEntropyStats,
  computeGraphletEntropy,
  computeLineageStats,
  safelyApplyTelemetrySelect,
  computeOperatorStatsSnapshot,
  computeHyperVolumeProxy,
  computeParetoFrontSizes,
  applyObjectiveImportance,
  applyObjectiveAges,
  applyObjectiveEvents,
  applySpeciesAllocation,
  applyObjectivesSnapshot,
  applyRngState,
  applyLineageStatsMultiObjective,
  applyLineageStatsMonoObjective,
  applyHypervolumeTelemetry,
  applyComplexityStatsMultiObjective,
  applyComplexityStatsMonoObjective,
  applyPerformanceStats,
} from '../metrics/telemetry.metrics';
import type {
  TelemetryDiversityOptions,
  TelemetryGenome,
} from '../types/telemetry.types';

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
  _objectiveEvents?: ObjectiveEvent[];
  _fastModeTuned?: boolean;
}

/**
 * Create a strict baseline telemetry entry with required fields populated.
 *
 * This helper centralizes defaults so downstream telemetry producers can
 * extend the entry while keeping the strict `TelemetryEntry` contract.
 *
 * @param generationIndex Generation index for the telemetry snapshot.
 * @param bestScore Best fitness value observed in the generation.
 * @param speciesCount Number of extant species.
 * @returns A strict telemetry entry with required fields populated.
 */
export function createTelemetryEntryBase(
  generationIndex: number,
  bestScore: number,
  speciesCount: number,
): TelemetryEntry {
  return {
    gen: generationIndex,
    best: bestScore,
    species: speciesCount,
    hyper: 0,
    ops: [],
    objImportance: {},
  };
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
  const ctx = this as TelemetryContext;
  const selectionSet = ctx._telemetrySelect;
  const coreFields = ['gen', 'best', 'species'] as const;

  // Step 1: Fast-path, nothing to do when no selection set is configured
  if (!selectionSet || selectionSet.size === 0) return entry;

  // Step 2: Declarative fold over the entry with internal helpers
  const coreSnapshot = getTelemetryCoreSnapshot(entry, coreFields);
  const filteredEntry = stripUnselectedTelemetryKeys(
    entry,
    selectionSet,
    coreFields,
  );

  return mergeTelemetryCoreFields(filteredEntry, coreSnapshot);
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
  const ctx = this as TelemetryContext;

  // Step 1: Fast-path cached entropy when available for current generation
  const cachedEntropy = getCachedEntropy(ctx.generation, graph);
  if (cachedEntropy !== undefined) return cachedEntropy;

  // Step 2: Compute degree counts for enabled edges
  const degreeCounts = computeDegreeCounts(graph);

  // Step 3: Convert degree counts into a degree-frequency histogram
  const nodeCount = graph.nodes.length || 1;
  const degreeHistogram = buildDegreeHistogram(degreeCounts);

  // Step 4: Compute entropy H = -sum p * log(p)
  const entropy = computeEntropyFromHistogram(degreeHistogram, nodeCount);

  // Step 5: Cache the result on the genome object
  setCachedEntropy(ctx.generation, graph, entropy);

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

  // Step 1: Apply fast-mode defaults once per instance
  applyFastModeDefaults(ctx, options);

  // Step 2: Resolve sampling parameters and population snapshot
  const pairSample = options.diversityMetrics.pairSample ?? 40;
  const graphletSample = options.diversityMetrics.graphletSample ?? 60;
  const population = (ctx.population as TelemetryGenome[]) ?? [];
  const populationSize = population.length;
  const rngFactory: () => () => number =
    typeof ctx._getRNG === 'function' ? ctx._getRNG : () => Math.random;
  const structuralEntropyFn =
    typeof ctx._structuralEntropy === 'function'
      ? (genome: TelemetryGenome) => ctx._structuralEntropy?.(genome) ?? 0
      : (genome: TelemetryGenome) =>
          structuralEntropy.call(
            ctx,
            genome as {
              nodes: Array<{ geneId: number }>;
              connections: Array<{
                from: { geneId: number };
                to: { geneId: number };
                enabled: boolean;
              }>;
            },
          );

  // Step 3: Compute pairwise compatibility statistics
  const compatibilityStats = computeCompatibilityStats(
    population,
    populationSize,
    pairSample,
    rngFactory,
    ctx._compatibilityDistance,
  );

  // Step 4: Compute structural entropy statistics across population
  const entropyStats = computeEntropyStats(population, structuralEntropyFn);

  // Step 5: Sample graphlet motifs and compute entropy
  const graphletEntropy = computeGraphletEntropy(
    population,
    populationSize,
    graphletSample,
    rngFactory,
  );

  // Step 6: Compute lineage-based statistics when enabled
  const lineageStats = computeLineageStats(
    Boolean(ctx._lineageEnabled),
    population,
    populationSize,
    pairSample,
    rngFactory,
  );

  // Step 7: Store computed stats on instance
  (ctx as { _diversityStats?: object })._diversityStats = {
    meanCompat: compatibilityStats.meanCompat,
    varCompat: compatibilityStats.varCompat,
    meanEntropy: entropyStats.meanEntropy,
    varEntropy: entropyStats.varEntropy,
    graphletEntropy,
    lineageMeanDepth: lineageStats.lineageMeanDepth,
    lineageMeanPairDist: lineageStats.lineageMeanPairDist,
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

  // Step 1: Apply telemetry selection without failing the evolution loop
  safelyApplyTelemetrySelect(ctx, entry, applyTelemetrySelect);

  // Step 2: Ensure the telemetry buffer exists and record the entry
  const telemetryBuffer = ensureTelemetryBuffer(ctx);
  telemetryBuffer.push(entry);

  // Step 3: Stream the entry when a stream callback is configured
  safelyStreamTelemetryEntry(ctx, entry);

  // Step 4: Keep the in-memory telemetry buffer bounded
  trimTelemetryBuffer(telemetryBuffer, 500);
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
  const options = ctx.options || {};
  /**
   * Current generation index for this telemetry snapshot.
   * Anchors all reported statistics to a single evolutionary timestep.
   * @example
   * // use the generation number when inspecting recorded telemetry
   * const generation = neat.generation;
   */
  const generationIndex = ctx.generation ?? 0;
  const isMultiObjectiveEnabled = options.multiObjective?.enabled ?? false;

  // Step 1: Select multi-objective or mono-objective telemetry path.
  if (isMultiObjectiveEnabled)
    return buildMultiObjectiveEntry(ctx, options, generationIndex, fittest);

  // Step 2: Fallback path (mono-objective) retained for parity with legacy behavior.
  return buildMonoObjectiveEntry(ctx, options, generationIndex, fittest);

  /**
   * Build a telemetry entry for multi-objective mode.
   *
   * @param telemetryContext - Neat-like context with population state.
   * @param telemetryOptions - Options controlling telemetry behavior.
   * @param generation - Generation index for this snapshot.
   * @param fittestGenome - Fittest genome record with score.
   * @returns Telemetry entry object for MO mode.
   */
  function buildMultiObjectiveEntry(
    telemetryContext: TelemetryContext,
    telemetryOptions: NeatOptions & TelemetryDiversityOptions,
    generation: number,
    fittestGenome: Record<string, unknown>,
  ): TelemetryEntry {
    // Step 1: Resolve population snapshot.
    const population = (telemetryContext.population as GenomeDetailed[]) || [];

    // Step 2: Compute MO proxy metrics (hypervolume proxy + pareto fronts).
    const hyperVolumeProxy = computeHyperVolumeProxy(
      telemetryOptions,
      population,
    );
    const paretoFrontSizes = computeParetoFrontSizes(population);

    // Step 3: Snapshot operator statistics.
    const operatorStatsSnapshot = computeOperatorStatsSnapshot(
      telemetryContext._operatorStats,
    );

    // Step 4: Assemble base entry.
    const entry: TelemetryEntry & Record<string, unknown> = {
      gen: generation,
      best: (fittestGenome.score as number) ?? 0,
      species: telemetryContext._species?.length ?? 0,
      hyper: hyperVolumeProxy,
      fronts: paretoFrontSizes,
      diversity: telemetryContext._diversityStats,
      ops: operatorStatsSnapshot,
      objImportance: {},
    };

    // Step 5: Attach objective/meta snapshots.
    applyObjectiveImportance(telemetryContext, entry);
    applyObjectiveAges(telemetryContext, entry);
    applyObjectiveEvents(telemetryContext, entry, generation);
    applySpeciesAllocation(telemetryContext, entry);
    applyObjectivesSnapshot(telemetryContext, entry);
    applyRngState(telemetryContext, telemetryOptions, entry);

    // Step 6: Attach lineage stats when enabled.
    applyLineageStatsMultiObjective(telemetryContext, population, entry);

    // Step 7: Attach optional telemetry expansions.
    applyHypervolumeTelemetry(telemetryOptions, hyperVolumeProxy, entry);
    applyComplexityStatsMultiObjective(
      telemetryContext,
      telemetryOptions,
      population,
      entry,
    );
    applyPerformanceStats(telemetryContext, telemetryOptions, entry);

    // Step 8: Return the assembled entry.
    return entry;
  }

  /**
   * Build a telemetry entry for mono-objective mode.
   *
   * @param telemetryContext - Neat-like context with population state.
   * @param telemetryOptions - Options controlling telemetry behavior.
   * @param generation - Generation index for this snapshot.
   * @param fittestGenome - Fittest genome record with score.
   * @returns Telemetry entry object for mono mode.
   */
  function buildMonoObjectiveEntry(
    telemetryContext: TelemetryContext,
    telemetryOptions: NeatOptions & TelemetryDiversityOptions,
    generation: number,
    fittestGenome: Record<string, unknown>,
  ): TelemetryEntry {
    // Step 1: Resolve population snapshot.
    const populationSnapshot =
      (telemetryContext.population as GenomeDetailed[]) || [];

    // Step 2: Snapshot operator statistics.
    const operatorStatsSnapshot = computeOperatorStatsSnapshot(
      telemetryContext._operatorStats,
    );

    // Step 2: Assemble base entry.
    const entry: TelemetryEntry & Record<string, unknown> = {
      gen: generation,
      best: (fittestGenome.score as number) ?? 0,
      species: telemetryContext._species?.length ?? 0,
      hyper: 0,
      diversity: telemetryContext._diversityStats,
      ops: operatorStatsSnapshot,
      objImportance: {},
    };

    // Step 3: Attach objective/meta snapshots.
    applyObjectiveImportance(telemetryContext, entry);
    applyObjectiveAges(telemetryContext, entry);
    applyObjectiveEvents(telemetryContext, entry, generation);
    applySpeciesAllocation(telemetryContext, entry);
    applyObjectivesSnapshot(telemetryContext, entry);
    applyRngState(telemetryContext, telemetryOptions, entry);

    // Step 4: Attach lineage stats when enabled.
    applyLineageStatsMonoObjective(telemetryContext, populationSnapshot, entry);

    // Step 5: Attach optional telemetry expansions.
    applyHypervolumeTelemetry(telemetryOptions, 0, entry);
    applyComplexityStatsMonoObjective(
      telemetryContext,
      telemetryOptions,
      populationSnapshot,
      entry,
    );
    applyPerformanceStats(telemetryContext, telemetryOptions, entry);

    // Step 6: Return the assembled entry.
    return entry;
  }
}
