/**
 * Telemetry recorder orchestration for generation snapshots.
 *
 * This chapter is the write-heavy heart of telemetry. If the facade explains
 * how readers inspect telemetry after it exists, the recorder explains how one
 * generation snapshot comes into existence in the first place.
 *
 * The recorder owns the end-to-end write path:
 * - start from the current generation, fittest genome, and cached controller state
 * - build one telemetry entry that summarizes what just happened
 * - optionally filter that entry down to a caller-selected surface
 * - persist the result to the in-memory buffer and stream it out when configured
 *
 * The neighboring chapters keep this boundary clean:
 * - `metrics/` computes the evidence attached to the entry
 * - `runtime/` handles safe buffer initialization, callback streaming, and trimming
 * - `facade/` exposes the recorded history back to `Neat` callers later
 *
 * Read this chapter when you want to answer questions such as:
 * - what exactly gets captured at the end of one generation?
 * - where do multi-objective and mono-objective telemetry paths diverge?
 * - when does telemetry selection happen relative to buffering and streaming?
 * - how does the library keep telemetry useful without letting diagnostics destabilize evolution?
 *
 * ```mermaid
 * flowchart LR
 *   State["Controller state<br/>population + generation + caches"] --> Build["buildTelemetryEntry()<br/>assemble one snapshot"]
 *   Build --> Select["applyTelemetrySelect()<br/>keep core fields + optional whitelist"]
 *   Select --> Buffer["recordTelemetryEntry()<br/>push into telemetry buffer"]
 *   Buffer --> Stream["optional runtime stream callback"]
 *   Buffer --> Trim["trim buffer to bounded history"]
 *   Metrics["metrics/<br/>diversity, lineage, objectives, perf"] --> Build
 *   Runtime["runtime/<br/>buffer + stream safety"] --> Buffer
 * ```
 *
 * A useful reading order is:
 * 1. `buildTelemetryEntry()` to understand the overall snapshot shape
 * 2. `recordTelemetryEntry()` to understand the write path and safety rules
 * 3. `applyTelemetrySelect()` to understand how callers can narrow the surface
 * 4. `computeDiversityStats()` and `structuralEntropy()` for the heaviest derived signals
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
 * @example
 * ```ts
 * // keep only 'gen', 'best', 'species' and 'diversity' fields
 * neat._telemetrySelect = new Set(['diversity']);
 * applyTelemetrySelect.call(neat, entry);
 * ```
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
 * @example
 * ```ts
 * const H = structuralEntropy.call(neat, genome);
 * console.log(`Structure entropy: ${H.toFixed(3)}`);
 * ```
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
 * This helper is where the recorder prepares one of its most important cached
 * evidence blocks: structural variety. Instead of treating diversity as a
 * single opaque score, it combines several approximations so later telemetry
 * can report compatibility spread, entropy, graphlet variety, and lineage depth.
 *
 * This helper is intentionally conservative at runtime. When `fastMode` is
 * enabled it tunes sampling defaults downward so telemetry stays informative
 * without turning every generation into a quadratic metrics pass.
 *
 * @remarks
 * - Uses random sampling of pairs and 3-node subgraphs (graphlets) to approximate diversity metrics.
 * @example
 * ```ts
 * // compute and store diversity stats onto the neat instance
 * neat.options.diversityMetrics = { enabled: true };
 * neat.computeDiversityStats();
 * console.log(neat._diversityStats.meanCompat);
 * ```
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
 * This is the recorder's "commit" step. By the time this function runs, the
 * entry has already been assembled. The job here is to make recording safe and
 * predictable: honor field selection, keep the history buffer initialized,
 * optionally notify observers, and cap memory growth.
 *
 * Write order:
 * 1. apply telemetry selection without breaking the evolution loop
 * 2. append the entry to the in-memory history buffer
 * 3. stream the entry when the host opts into runtime callbacks
 * 4. trim history to a bounded window
 *
 * @example
 * ```ts
 * // record a simple telemetry entry from inside the evolve loop
 * neat.recordTelemetryEntry({ gen: neat.generation, best: neat.population[0].score });
 * ```
 *
 * @param entry - Telemetry entry to record.
 * @returns Nothing. The entry is persisted by side effect on the host.
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
 * This is the recorder's main orchestration surface. It gathers one coherent
 * explanation of the current generation by combining immediate state
 * (`generation`, `best`, `species`) with whichever optional evidence blocks the
 * controller has enabled: diversity, lineage, objective activity, Pareto-front
 * summaries, complexity growth, RNG state, and performance timing.
 *
 * This function intentionally mirrors the legacy in-loop telemetry construction
 * to preserve behavior relied upon by tests and consumers.
 *
 * @example
 * ```ts
 * // build a telemetry snapshot for the current generation
 * const snapshot = neat.buildTelemetryEntry(neat.population[0]);
 * neat.recordTelemetryEntry(snapshot);
 * ```
 *
 * The function has two internal paths:
 * - multi-objective mode adds Pareto-front and hypervolume-oriented signals
 * - mono-objective mode keeps the payload smaller while preserving the same core fields
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
