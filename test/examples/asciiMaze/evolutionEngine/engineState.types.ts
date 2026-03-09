/**
 * Shared contracts for the ASCII maze evolution engine state boundary.
 *
 * This file owns the exported type surface consumed by the engine-state facade,
 * telemetry helpers, sampling utilities, and RNG/timing adapters.
 */

/**
 * Centralised shared state for the ASCII maze evolution façade.
 *
 * Responsibilities:
 * 1. Define the scratch-buffer schema consumed by telemetry, population, and inspection helpers.
 * 2. Expose runtime toggle state (`EngineToggleState`) that drives optional phases and telemetry density.
 * 3. Provide factory and maintenance helpers (`createEngineState`, `initialiseTelemetryScratch`, `ensureVisitedHashCapacity`, `ensureRngCacheBatch`, `reseedRngState`) that size buffers and keep deterministic RNG state in sync.
 * 4. Export the project-wide singleton `engineState` so extracted modules can share the façade’s pooled resources while still accepting injected state for testing.
 *
 * Callers mutate the returned scratch instances in place to avoid per-generation allocations; higher-level modules should treat the helpers as the sole entry point for sizing or resetting shared buffers.
 */
export interface EngineScratchState {
  /** Softmax exponent scratch reused by telemetry entropy calculations. */
  exps: Float64Array;
  /** Mean accumulator reused for output bias statistics. */
  means: Float64Array;
  /** Standard deviation accumulator shared by telemetry computations. */
  standardDeviations: Float64Array;
  /** Bias scratch workspace reused for output-bias telemetry calculations. */
  biasTelemetryScratch: Float64Array;
  /** Optional kurtosis accumulator allocated when full telemetry is enabled. */
  kurtosis?: Float64Array;
  /** Second raw moment buffer for variance calculations. */
  secondMomentRaw: Float64Array;
  /** Optional third raw moment buffer for skewness metrics. */
  thirdMomentRaw?: Float64Array;
  /** Optional fourth raw moment buffer for kurtosis metrics. */
  fourthMomentRaw?: Float64Array;
  /** Directional move counters reused during exploration stats. */
  moveCounts: Int32Array;
  /** Open-address hash table used for visited coordinate detection. */
  visitedHashTable: Int32Array;
  /** Current load factor threshold guiding visited-hash resizes. */
  visitedHashLoadFactor: number;
  /** Species id scratch array reused by population sorting. */
  speciesIds: Int32Array;
  /** Species count scratch array kept parallel to {@link speciesIds}. */
  speciesCounts: Int32Array;
  /** Candidate connection objects pooled for pruning heuristics. */
  connectionCandidates: Record<string, unknown>[];
  /** Hidden-to-output connection list reused during inspection. */
  hiddenToOutputConnections: Record<string, unknown>[];
  /** Bit flags for connection enable/disable states. */
  connectionFlags: Uint8Array;
  /** Bitmap reused when detecting recurrent or gated connections. */
  connectionFlagBitmap?: Int8Array;
  /** Maze tail history reused to avoid reallocating per telemetry call. */
  tailHistoryBuffer: Array<unknown>;
  /** Sampled result buffer reused by array sampling helpers. */
  sampleResultBuffer: Array<unknown>;
  /** Sorted index scratch reused when ranking genomes. */
  sortedIndexBuffer: number[];
  /** Optional typed view of {@link sortedIndexBuffer} for faster sorts. */
  sortedIndexTypedArray?: Int32Array;
  /** Quicksort stack storing pending index ranges. */
  quicksortStack: Int32Array;
  /** Temporary population clone array used during expansion. */
  populationCloneBuffer: Record<string, unknown>[];
  /** Activation name buffer reused by inspection routines. */
  activationNameBuffer: string[];
  /** Node classification buckets reused by inspection routines. */
  nodeBuckets: [
    Record<string, unknown>[],
    Record<string, unknown>[],
    Record<string, unknown>[],
  ];
  /** Top entry objects reused when generating snapshots. */
  snapshotTopEntries: Record<string, unknown>[];
  /** Snapshot metadata object reused per persistence write. */
  snapshotReusableObject: Record<string, unknown>;
  /** Mutation operator index buffer shuffled each generation. */
  mutationOperatorIndices: Uint16Array;
  /** Object pool used when sampling individuals for telemetry. */
  samplePool: Array<unknown>;
  /** Character array reused when assembling debug strings. */
  stringAssemblyBuffer: string[];
  /** Small exploration table for low-cost duplicate detection. */
  smallExploreTable: Int32Array;
  /** Non-shared logits ring storing recent action logits. */
  logitsRing: Float32Array[];
  /** Write cursor for the non-shared logits ring. */
  logitsRingWriteCursor: number;
  /** Shared float buffer backing the logits ring when SAB is available. */
  sharedLogits?: Float32Array;
  /** Shared atomic write index for the SAB-backed logits ring. */
  sharedLogitsWriteIndex?: Int32Array;
  /** Scratch node index pool reused by topology inspection. */
  nodeIndexBuffer: Int32Array;
  /** Small profiling scratch buffer reused during timing accumulation. */
  profilingScratch?: Float64Array;
  /** Linear congruential RNG state preserved across cache refills. */
  rngState: number;
  /** Cached batch of uniform random numbers reused by the façade. */
  rngCache: Float64Array;
  /** Next unread index within {@link rngCache}. */
  rngCacheIndex: number;
  /** Allow additional properties for extensibility */
  [key: string]: unknown;
}

/**
 * Runtime switches that adjust telemetry verbosity and optional training phases.
 */
export interface EngineToggleState {
  /** Softens telemetry to a reduced metric set. */
  reducedTelemetry: boolean;
  /** Enables the most compact telemetry output footprint. */
  telemetryMinimal: boolean;
  /** Disables the Baldwin (Lamarckian) warm-start phase. */
  disableBaldwinPhase: boolean;
}

/**
 * Aggregated profiling configuration and accumulators shared across the evolution run.
 */
export interface EngineProfilingState {
  /** Indicates whether detailed profiling accumulation is active. */
  detailsEnabled: boolean;
  /** Rolling millisecond totals grouped by profiling segment key. */
  accumulators: Record<string, number>;
}

/**
 * Shared engine state instance combining pooled scratch buffers with toggle flags.
 */
export interface EngineState {
  /** Bundle of pooled scratch buffers shared by the façade. */
  scratch: EngineScratchState;
  /** Runtime toggles influencing telemetry and training behaviour. */
  toggles: EngineToggleState;
  /** Profiling configuration and accumulated timings. */
  profiling: EngineProfilingState;
  /** Indicates whether deterministic RNG mode is enabled. */
  deterministicMode: boolean;
}

/**
 * Configuration describing which telemetry scratch buffers require capacity guarantees.
 */
export interface TelemetryScratchRequest {
  /** Desired action dimensionality for logit statistics helpers. */
  actionDimension?: number;
  /** When true ensure higher-moment buffers (M3/M4/kurtosis) are allocated. */
  includeHigherMoments?: boolean;
  /** Number of output biases expected when formatting telemetry strings. */
  biasCount?: number;
  /** String assembly buffer length requirement (defaults to {@link biasCount}). */
  stringBufferLength?: number;
}

/**
 * Collection of scratch buffers handed back after initialisation for convenience.
 */
export interface TelemetryScratchHandles {
  /** Float64 scratch used for exponentiation during entropy calculations. */
  exponentScratch: Float64Array;
  /** Running mean accumulator workspace. */
  meanScratch: Float64Array;
  /** Population standard deviation workspace. */
  standardDeviationScratch: Float64Array;
  /** Second raw moment workspace. */
  secondMomentScratch: Float64Array;
  /** Optional third raw moment workspace (present when higher moments requested). */
  thirdMomentScratch?: Float64Array;
  /** Optional fourth raw moment workspace (present when higher moments requested). */
  fourthMomentScratch?: Float64Array;
  /** Optional kurtosis workspace (present when higher moments requested). */
  kurtosisScratch?: Float64Array;
  /** Bias accumulator buffer used by output-bias telemetry helpers. */
  biasScratch: Float64Array;
  /** Shared string assembly buffer reused across telemetry loggers. */
  stringBuffer: string[];
}

/** Parameters controlling the RNG cache refill process. */
export interface RngCacheParameters {
  /** Number of samples generated per congruential batch. */
  batchSize: number;
  /** Linear congruential multiplier component. */
  multiplier: number;
  /** Linear congruential increment component. */
  increment: number;
  /** Bit shift applied before scaling floats into [0,1). */
  shift: number;
  /** Scalar applied to produce [0,1) floats from shifted integers. */
  scale: number;
}

/** Handles returned after ensuring the RNG cache is ready for consumption. */
export interface RngCacheHandles {
  /** Cached uniform samples ready for reuse. */
  cache: Float64Array;
  /** Batch size associated with the cache. */
  batchSize: number;
}

/**
 * Handles exposed after ensuring the visited-coordinate hash table capacity.
 */
export interface VisitedHashScratchHandles {
  /** Cleared Int32Array hash table ready for inserts. */
  table: Int32Array;
  /** Bitmask used for wraparound during linear probing (table.length - 1). */
  slotMask: number;
}
