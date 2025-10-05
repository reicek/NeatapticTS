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
  nodeBuckets: [Record<string, unknown>[], Record<string, unknown>[], Record<string, unknown>[]];
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

/** Default logits ring length used when allocating pooled softmax buffers. */
const DEFAULT_LOGITS_RING_CAPACITY = 512;
/** Number of action outputs (N, E, S, W) represented in each logits row. */
const ACTION_OUTPUT_DIMENSION = 4;
/** Default load factor target for the visited coordinate hash table. */
export const DEFAULT_VISITED_HASH_LOAD_FACTOR = 0.7;
/** Minimum safe load factor applied when normalising visited-hash configuration. */
const MIN_VISITED_HASH_LOAD_FACTOR = 0.1;
/** Maximum safe load factor applied when normalising visited-hash configuration. */
const MAX_VISITED_HASH_LOAD_FACTOR = 0.95;
/** Default RNG cache batch size mirroring the façade constant. */
export const DEFAULT_RNG_CACHE_BATCH_SIZE = 4;

/** Default capacity reserved for species identifier scratch arrays. */
const DEFAULT_SPECIES_SCRATCH_CAPACITY = 64;
/** Default capacity reserved for connection flag buffers. */
const DEFAULT_CONNECTION_FLAG_CAPACITY = 128;
/** Default capacity reused by history and sampling scratch arrays. */
const DEFAULT_HISTORY_BUFFER_CAPACITY = 64;
/** Default capacity reserved for sorted index scratch arrays. */
const DEFAULT_SORTED_INDEX_CAPACITY = 512;
/** Default stack depth reserved for quicksort range storage. */
const DEFAULT_QUICKSORT_STACK_CAPACITY = 128;
/** Default pool size for telemetry sampling helpers. */
const DEFAULT_SAMPLE_POOL_SIZE = 40;
/** Default capacity for telemetry string assembly buffers. */
const DEFAULT_STRING_BUFFER_CAPACITY = 64;
/** Default capacity for the small exploration table scratch. */
const DEFAULT_SMALL_EXPLORE_TABLE_CAPACITY = 64;
/** Default capacity for the node index buffer used during inspection. */
const DEFAULT_NODE_INDEX_BUFFER_CAPACITY = 64;
/** Knuth-derived 32-bit constant used when seeding the RNG state. */
const RNG_GOLDEN_RATIO_SEED = 0x9e3779b9;

/**
 * Build a logits ring sized to the default capacity.
 * @returns Array of Float32Array rows sized to {@link ACTION_OUTPUT_DIMENSION}.
 */
const createLogitsRing = (): Float32Array[] => {
  const rows = new Array<Float32Array>(DEFAULT_LOGITS_RING_CAPACITY);
  for (let ringIndex = 0; ringIndex < DEFAULT_LOGITS_RING_CAPACITY; ringIndex++)
    rows[ringIndex] = new Float32Array(ACTION_OUTPUT_DIMENSION);
  return rows;
};

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

/** Compute whether detailed profiling is enabled via the environment flag. */
const resolveProfilingEnabled = (): boolean => {
  try {
    return (
      typeof process !== 'undefined' &&
      process?.env?.ASCII_MAZE_PROFILE_DETAILS === '1'
    );
  } catch {
    return false;
  }
};

/**
 * Build a fresh profiling accumulator map seeded with zero totals.
 * @returns Accumulator record keyed by profiling segment name.
 */
const createProfilingAccumulators = (): Record<string, number> => ({
  telemetry: 0,
  simplify: 0,
  snapshot: 0,
  prune: 0,
});

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
 * Build the reusable snapshot metadata payload consumed by persistence helpers.
 * @returns Snapshot placeholder populated with neutral defaults.
 */
const createSnapshotReusableObject = () => ({
  /** Snapshot generation index. */
  generation: 0,
  /** Snapshot best fitness value. */
  bestFitness: 0,
  /** Flag indicating simplify mode state. */
  simplifyMode: false,
  /** Plateau counter for simplify heuristics. */
  plateauCounter: 0,
  /** Snapshot timestamp. */
  timestamp: 0,
  /** Telemetry tail cache. */
  telemetryTail: undefined,
  /** Top entry payload. */
  top: undefined,
});

/**
 * Fabricates the default pooled scratch buffers shared by the evolution façade.
 * @returns Scratch bundle sized for the baseline maze curriculum workload.
 * @example
 * const scratch = createScratchState();
 * console.log(scratch.exps.length); // 4
 */
export const createScratchState = (): EngineScratchState => ({
  /** Initial softmax exponent workspace. */
  exps: new Float64Array(ACTION_OUTPUT_DIMENSION),
  /** Mean accumulator backing telemetry stats. */
  means: new Float64Array(ACTION_OUTPUT_DIMENSION),
  /** Standard deviation accumulator backing telemetry stats. */
  standardDeviations: new Float64Array(ACTION_OUTPUT_DIMENSION),
  /** Bias telemetry numeric workspace. */
  biasTelemetryScratch: new Float64Array(ACTION_OUTPUT_DIMENSION),
  /** Optional kurtosis buffer (enabled lazily). */
  kurtosis: undefined,
  /** Second raw moment workspace. */
  secondMomentRaw: new Float64Array(ACTION_OUTPUT_DIMENSION),
  /** Third raw moment workspace (allocated when needed). */
  thirdMomentRaw: undefined,
  /** Fourth raw moment workspace (allocated when needed). */
  fourthMomentRaw: undefined,
  /** Directional move counters for exploration stats. */
  moveCounts: new Int32Array(ACTION_OUTPUT_DIMENSION),
  /** Visited coordinate hash table reused between runs. */
  visitedHashTable: new Int32Array(0),
  /** Load factor target for the visited hash table. */
  visitedHashLoadFactor: DEFAULT_VISITED_HASH_LOAD_FACTOR,
  /** Species id scratch array. */
  speciesIds: new Int32Array(DEFAULT_SPECIES_SCRATCH_CAPACITY),
  /** Species counts scratch array. */
  speciesCounts: new Int32Array(DEFAULT_SPECIES_SCRATCH_CAPACITY),
  /** Pooled candidate connection objects. */
  connectionCandidates: [],
  /** Hidden-to-output connection cache. */
  hiddenToOutputConnections: [],
  /** Connection enable/disable bit flags. */
  connectionFlags: new Uint8Array(DEFAULT_CONNECTION_FLAG_CAPACITY),
  /** Reusable bitmap for recurrent/gated detection helpers. */
  connectionFlagBitmap: undefined,
  /** Reusable maze tail history. */
  tailHistoryBuffer: new Array(DEFAULT_HISTORY_BUFFER_CAPACITY),
  /** Reusable sample results array. */
  sampleResultBuffer: new Array(DEFAULT_HISTORY_BUFFER_CAPACITY),
  /** Scratch sorted index buffer. */
  sortedIndexBuffer: new Array(DEFAULT_SORTED_INDEX_CAPACITY),
  /** Optional typed sorted index view. */
  sortedIndexTypedArray: undefined,
  /** Quicksort range stack. */
  quicksortStack: new Int32Array(DEFAULT_QUICKSORT_STACK_CAPACITY),
  /** Temporary population clone buffer. */
  populationCloneBuffer: new Array(0),
  /** Activation name cache. */
  activationNameBuffer: new Array(0),
  /** Node classification buckets reused by inspection helpers. */
  nodeBuckets: [[], [], []],
  /** Snapshot top entries reuse pool. */
  snapshotTopEntries: new Array(0),
  /** Snapshot metadata reuse object. */
  snapshotReusableObject: createSnapshotReusableObject(),
  /** Mutation operator order buffer. */
  mutationOperatorIndices: new Uint16Array(0),
  /** Sample pool for telemetry draws. */
  samplePool: new Array(DEFAULT_SAMPLE_POOL_SIZE),
  /** String assembly scratch array. */
  stringAssemblyBuffer: new Array(DEFAULT_STRING_BUFFER_CAPACITY),
  /** Small exploration table scratch. */
  smallExploreTable: new Int32Array(DEFAULT_SMALL_EXPLORE_TABLE_CAPACITY),
  /** Non-shared logits ring rows. */
  logitsRing: createLogitsRing(),
  /** Non-shared logits ring cursor. */
  logitsRingWriteCursor: 0,
  /** Shared logits SAB buffer. */
  sharedLogits: undefined,
  /** Shared logits write index view. */
  sharedLogitsWriteIndex: undefined,
  /** Scratch node index buffer. */
  nodeIndexBuffer: new Int32Array(DEFAULT_NODE_INDEX_BUFFER_CAPACITY),
  /** Profiling scratch typed array reused by timing helpers. */
  profilingScratch: undefined,
  /** LCG state backing the fast RNG helper. */
  rngState: (Date.now() ^ RNG_GOLDEN_RATIO_SEED) >>> 0,
  /** Cached RNG batch reused to amortise congruential updates. */
  rngCache: new Float64Array(DEFAULT_RNG_CACHE_BATCH_SIZE),
  /** Index into {@link rngCache}; initialised to force a refill on first use. */
  rngCacheIndex: DEFAULT_RNG_CACHE_BATCH_SIZE,
});

/**
 * Fabricates the default runtime toggle bundle for the evolution façade.
 * @returns Toggle state with all optional phases enabled.
 * @example
 * const toggles = createToggleState();
 * console.log(toggles.reducedTelemetry); // false
 */
export const createToggleState = (): EngineToggleState => ({
  /** Reduced telemetry flag default. */
  reducedTelemetry: false,
  /** Minimal telemetry flag default. */
  telemetryMinimal: false,
  /** Baldwin phase disabled flag default. */
  disableBaldwinPhase: false,
});

/**
 * Fabricates the profiling state bundle consumed by timing helpers.
 * @returns Profiling configuration and accumulator state.
 * @example
 * const profiling = createProfilingState();
 * console.log(profiling.detailsEnabled); // false unless env flag set
 */
export const createProfilingState = (): EngineProfilingState => ({
  detailsEnabled: resolveProfilingEnabled(),
  accumulators: createProfilingAccumulators(),
});

/**
 * Fabricates a new {@link EngineState} with pre-sized scratch buffers and default toggle values.
 * @returns Initialized engine state used by the maze evolution façade.
 */
export const createEngineState = (): EngineState => ({
  scratch: createScratchState(),
  toggles: createToggleState(),
  profiling: createProfilingState(),
  deterministicMode: false,
});

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

/**
 * Ensure the visited-coordinate hash table can store the requested entry count.
 *
 * Growth policy:
 * 1. Tables expand geometrically (powers of two) while respecting the configured load factor.
 * 2. When the existing table is large enough, it is cleared in place to preserve the allocation.
 * 3. Load factors outside the (0,1) range fall back to {@link DEFAULT_VISITED_HASH_LOAD_FACTOR}.
 *
 * @param targetEntryCount Expected number of unique coordinates to store.
 * @param state Optional engine state container (defaults to the shared singleton).
 * @returns Cleared table reference plus the slot mask for probing loops.
 * @example
 * const { table, slotMask } = ensureVisitedHashCapacity(256);
 */
export const ensureVisitedHashCapacity = (
  targetEntryCount: number,
  state: EngineState = engineState,
): VisitedHashScratchHandles => {
  const scratch = state.scratch;

  // Step 1: Normalise input size and load factor to safe ranges.
  const clampedTarget = normaliseVisitedHashEntries(targetEntryCount);
  const loadFactor = normaliseVisitedHashLoad(scratch.visitedHashLoadFactor);

  // Step 2: Determine whether the existing table must grow.
  const minimumCapacity = Math.max(1, clampedTarget);
  let table = scratch.visitedHashTable;
  const capacityThreshold = table.length * loadFactor;
  if (table.length === 0 || minimumCapacity > capacityThreshold) {
    // Step 2.1: Compute required slots under the load factor and grow to the next power of two.
    const requiredSlots = Math.max(1, Math.ceil(minimumCapacity / loadFactor));
    const nextSize = nextPowerOfTwo(requiredSlots);
    table = new Int32Array(nextSize);
    scratch.visitedHashTable = table;
  } else {
    // Step 2.2: Clear the table for deterministic reuse.
    table.fill(0);
  }

  // Step 3: Derive probing mask (table length is guaranteed power of two).
  const slotMask = table.length > 0 ? table.length - 1 : 0;
  return { table, slotMask };
};

/**
 * Ensure telemetry-related scratch buffers allocate enough capacity for upcoming work.
 *
 * Growth strategy:
 * - Float64Array pools grow geometrically (powers of two) and preserve previously written prefixes.
 * - String buffers are replaced with larger arrays (also geometric) to avoid repeated reallocations.
 * - Optional higher-moment buffers (M3/M4/kurtosis) are only created when {@link TelemetryScratchRequest.includeHigherMoments}
 *   is truthy to keep reduced telemetry lightweight.
 *
 * @param request Capacity hints for upcoming telemetry computations.
 * @param state Optional engine state target (defaults to the singleton {@link engineState}).
 * @returns Handles referencing the ensured scratch buffers for immediate use.
 */
export const initialiseTelemetryScratch = (
  request: TelemetryScratchRequest = {},
  state: EngineState = engineState,
): TelemetryScratchHandles => {
  const scratch = state.scratch;
  const capacityHints = normaliseTelemetryCapacityHints(request);
  ensureTelemetryFloatPools(scratch, capacityHints);
  scratch.stringAssemblyBuffer = ensureTelemetryStringBuffer(
    scratch.stringAssemblyBuffer,
    capacityHints.stringLength,
  );

  return buildTelemetryHandles(scratch);
};

/**
 * Ensure the RNG cache contains fresh samples before consumption.
 * @param parameters Congruential generator parameters and cache batch size.
 * @param state Optional engine state container (defaults to the shared singleton).
 * @returns Handles exposing the cache and its batch size.
 */
export const ensureRngCacheBatch = (
  parameters: RngCacheParameters,
  state: EngineState = engineState,
): RngCacheHandles => {
  const scratch = state.scratch;
  const batchSize = normaliseRngBatchSize(parameters.batchSize);

  if (scratch.rngCache.length !== batchSize) {
    scratch.rngCache = new Float64Array(batchSize);
    scratch.rngCacheIndex = batchSize;
  }

  if (scratch.rngCacheIndex >= batchSize) {
    let localState = scratch.rngState >>> 0;
    const multiplier = parameters.multiplier >>> 0;
    const increment = parameters.increment >>> 0;
    for (let cacheIndex = 0; cacheIndex < batchSize; cacheIndex++) {
      localState = (localState * multiplier + increment) >>> 0;
      scratch.rngCache[cacheIndex] =
        (localState >>> parameters.shift) * parameters.scale;
    }
    scratch.rngState = localState >>> 0;
    scratch.rngCacheIndex = 0;
  }

  return {
    cache: scratch.rngCache,
    batchSize,
  };
};

/**
 * Persist a new RNG seed and schedule a cache refill on the next draw.
 * @param seed Raw numeric seed requested by deterministic mode.
 * @param state Optional engine state container (defaults to the shared singleton).
 * @returns Unsigned seed stored in scratch state for diagnostics.
 */
export const reseedRngState = (
  seed: number,
  state: EngineState = engineState,
): number => {
  const scratch = state.scratch;
  const normalisedSeed = normaliseRngSeed(seed);
  scratch.rngState = normalisedSeed;
  scratch.rngCacheIndex = scratch.rngCache.length;
  return normalisedSeed;
};

interface TelemetryCapacityHints {
  actionDimension: number;
  requiresHigherMoments: boolean;
  biasCount: number;
  stringLength: number;
}

/**
 * Derive normalised capacity hints from the raw telemetry scratch request.
 * @param request Raw capacity request supplied by callers.
 * @returns Sanitised capacity values used during buffer initialisation.
 */
const normaliseTelemetryCapacityHints = (
  request: TelemetryScratchRequest,
): TelemetryCapacityHints => {
  const normaliseSize = (value: number | undefined, fallback = 0): number => {
    if (!Number.isFinite(value as number))
      return Math.max(0, Math.floor(fallback));
    return Math.max(0, Math.floor(value as number));
  };

  const biasCount = normaliseSize(request.biasCount);
  return {
    actionDimension: normaliseSize(request.actionDimension),
    requiresHigherMoments: Boolean(request.includeHigherMoments),
    biasCount,
    stringLength: normaliseSize(request.stringBufferLength, biasCount),
  };
};

/**
 * Ensure all Float64 scratch pools required for telemetry are adequately sized.
 * @param scratch Shared scratch state mutated in place.
 * @param hints Normalised capacity hints (action dimension, bias count, etc.).
 */
const ensureTelemetryFloatPools = (
  scratch: EngineScratchState,
  hints: TelemetryCapacityHints,
): void => {
  const minActionDim = Math.max(4, hints.actionDimension);
  scratch.exps = ensureFloat64Pool(
    scratch.exps,
    minActionDim,
    4,
  ); /* exponent scratch for entropy */
  scratch.means = ensureFloat64Pool(
    scratch.means,
    hints.actionDimension,
    1,
  ); /* running mean per action */
  scratch.standardDeviations = ensureFloat64Pool(
    scratch.standardDeviations,
    hints.actionDimension,
    1,
  ); /* std aggregation buffer */
  scratch.secondMomentRaw = ensureFloat64Pool(
    scratch.secondMomentRaw,
    hints.actionDimension,
    1,
  ); /* Welford M2 accumulator */
  scratch.biasTelemetryScratch = ensureFloat64Pool(
    scratch.biasTelemetryScratch,
    hints.biasCount,
    1,
  ); /* bias stats workspace */

  if (!hints.requiresHigherMoments) return;

  scratch.thirdMomentRaw = ensureOptionalFloat64Pool(
    scratch.thirdMomentRaw,
    hints.actionDimension,
  ); /* optional skewness (M3) */
  scratch.fourthMomentRaw = ensureOptionalFloat64Pool(
    scratch.fourthMomentRaw,
    hints.actionDimension,
  ); /* optional kurtosis (M4) */
  scratch.kurtosis = ensureOptionalFloat64Pool(
    scratch.kurtosis,
    hints.actionDimension,
  ); /* derived excess kurtosis */
};

/**
 * Ensure the reusable string assembly buffer has sufficient capacity.
 * @param buffer Existing string buffer instance.
 * @param required Minimum number of slots needed for upcoming telemetry joins.
 * @returns Original buffer when large enough, otherwise a grown copy.
 */
const ensureTelemetryStringBuffer = (
  buffer: string[],
  required: number,
): string[] => ensureArrayCapacity(buffer, required);

/**
 * Build typed handles referencing the ensured telemetry scratch buffers.
 * @param scratch Scratch state containing the prepared buffers.
 * @returns Structured handles consumed by telemetry helpers.
 */
const buildTelemetryHandles = (
  scratch: EngineScratchState,
): TelemetryScratchHandles => ({
  exponentScratch: scratch.exps,
  meanScratch: scratch.means,
  standardDeviationScratch: scratch.standardDeviations,
  secondMomentScratch: scratch.secondMomentRaw,
  thirdMomentScratch: scratch.thirdMomentRaw,
  fourthMomentScratch: scratch.fourthMomentRaw,
  kurtosisScratch: scratch.kurtosis,
  biasScratch: scratch.biasTelemetryScratch,
  stringBuffer: scratch.stringAssemblyBuffer,
});

/**
 * Ensure a required Float64Array pool satisfies the requested capacity using power-of-two growth.
 * @param buffer Existing Float64Array instance reused by the engine.
 * @param required Minimum length the caller needs.
 * @param minimum Optional lower bound applied before comparison.
 * @returns Buffer with sufficient capacity (original or grown).
 */
const ensureFloat64Pool = (
  buffer: Float64Array,
  required: number,
  minimum = 0,
): Float64Array => {
  const target = Math.max(required, minimum);
  if (target <= 0 || buffer.length >= target) return buffer;
  const nextSize = nextPowerOfTwo(target);
  const next = new Float64Array(nextSize);
  if (buffer.length > 0) {
    next.set(buffer.subarray(0, Math.min(buffer.length, nextSize)));
  }
  return next;
};

/**
 * Ensure an optional Float64Array is present and sized appropriately.
 * @param buffer Optional Float64Array reference to validate.
 * @param required Minimum required capacity; zero preserves the existing buffer.
 * @returns Buffer with sufficient capacity or undefined when no allocation is required.
 */
const ensureOptionalFloat64Pool = (
  buffer: Float64Array | undefined,
  required: number,
): Float64Array | undefined => {
  if (required <= 0) return buffer;
  if (!buffer) return new Float64Array(nextPowerOfTwo(required));
  if (buffer.length >= required) return buffer;
  const nextSize = nextPowerOfTwo(required);
  const next = new Float64Array(nextSize);
  next.set(buffer.subarray(0, Math.min(buffer.length, nextSize)));
  return next;
};

/**
 * Ensure a generic Array buffer has enough slots, reusing existing entries when grown.
 * @param buffer Buffer instance to validate.
 * @param required Minimum number of elements required.
 * @returns Buffer with sufficient capacity (original or grown).
 */
const ensureArrayCapacity = <T>(buffer: T[], required: number): T[] => {
  if (required <= 0 || buffer.length >= required) return buffer;
  const nextSize = nextPowerOfTwo(required);
  const grown = new Array<T>(nextSize);
  for (let index = 0; index < buffer.length; index++) {
    grown[index] = buffer[index];
  }
  return grown;
};

/**
 * Compute the next power-of-two for geometric growth.
 * @param candidate Raw size candidate.
 * @returns Smallest power-of-two >= candidate.
 */
const nextPowerOfTwo = (candidate: number): number => {
  if (candidate <= 1) return 1;
  return 1 << Math.ceil(Math.log2(candidate));
};

/**
 * Clamp the requested target entry count to a non-negative integer.
 * @param requestedEntries Raw target entry count supplied by callers.
 * @returns Sanitised entry count used for capacity planning.
 */
const normaliseVisitedHashEntries = (requestedEntries: number): number => {
  if (!Number.isFinite(requestedEntries)) return 0;
  return Math.max(0, Math.floor(requestedEntries));
};

/**
 * Ensure the visited hash load factor falls within a sensible (0,1) range.
 * @param requestedLoadFactor Proposed load factor from configuration or state.
 * @returns Clamped load factor with a fallback to the project default.
 */
const normaliseVisitedHashLoad = (requestedLoadFactor: number): number => {
  if (!Number.isFinite(requestedLoadFactor)) {
    return DEFAULT_VISITED_HASH_LOAD_FACTOR;
  }
  if (
    requestedLoadFactor <= MIN_VISITED_HASH_LOAD_FACTOR ||
    requestedLoadFactor >= MAX_VISITED_HASH_LOAD_FACTOR
  ) {
    return DEFAULT_VISITED_HASH_LOAD_FACTOR;
  }
  return requestedLoadFactor;
};

/**
 * Clamp the RNG cache batch size to a positive integer.
 * @param requestedBatchSize Raw batch size requested by callers.
 * @returns Valid batch size (minimum of one sample per refill).
 */
const normaliseRngBatchSize = (requestedBatchSize: number): number => {
  if (!Number.isFinite(requestedBatchSize)) {
    return DEFAULT_RNG_CACHE_BATCH_SIZE;
  }
  return Math.max(1, Math.floor(requestedBatchSize));
};

/**
 * Normalise a raw numeric seed into an unsigned 32-bit value (remapping zero).
 * @param rawSeed Raw seed provided by the caller.
 * @returns Unsigned 32-bit seed suitable for the congruential generator.
 */
const normaliseRngSeed = (rawSeed: number): number => {
  const candidate = rawSeed >>> 0;
  return candidate === 0 ? 0x9e3779b9 : candidate;
};

/**
 * Singleton engine state shared by the evolution façade and extracted modules.
 */
export const engineState = createEngineState();

/**
 * Toggle the reduced telemetry mode.
 * @param isEnabled When true, trims telemetry to the essential metrics only.
 */
export const setReducedTelemetryFlag = (isEnabled: boolean): void => {
  engineState.toggles.reducedTelemetry = isEnabled;
};

/**
 * Toggle the minimal telemetry mode for JSON output.
 * @param isMinimal When true, disables verbose telemetry capture.
 */
export const setTelemetryMinimalFlag = (isMinimal: boolean): void => {
  engineState.toggles.telemetryMinimal = isMinimal;
};

/**
 * Enable or disable the Baldwin-phase warm-start pipeline.
 * @param isDisabled When true, skips the Lamarckian training stage.
 */
export const setBaldwinPhaseDisabledFlag = (isDisabled: boolean): void => {
  engineState.toggles.disableBaldwinPhase = isDisabled;
};
