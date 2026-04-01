import type {
  EngineProfilingState,
  EngineScratchState,
  EngineState,
  EngineToggleState,
  RngCacheHandles,
  RngCacheParameters,
  TelemetryScratchHandles,
  TelemetryScratchRequest,
  VisitedHashScratchHandles,
} from './engineState.types';
import {
  ACTION_OUTPUT_DIMENSION,
  DEFAULT_CONNECTION_FLAG_CAPACITY,
  DEFAULT_HISTORY_BUFFER_CAPACITY,
  DEFAULT_NODE_INDEX_BUFFER_CAPACITY,
  DEFAULT_QUICKSORT_STACK_CAPACITY,
  DEFAULT_RNG_CACHE_BATCH_SIZE,
  DEFAULT_SAMPLE_POOL_SIZE,
  DEFAULT_SMALL_EXPLORE_TABLE_CAPACITY,
  DEFAULT_SORTED_INDEX_CAPACITY,
  DEFAULT_SPECIES_SCRATCH_CAPACITY,
  DEFAULT_STRING_BUFFER_CAPACITY,
  DEFAULT_VISITED_HASH_LOAD_FACTOR,
  RNG_GOLDEN_RATIO_SEED,
} from './engineState.constants';
import {
  buildTelemetryHandles,
  createLogitsRing,
  createProfilingAccumulators,
  createSnapshotReusableObject,
  ensureTelemetryFloatPools,
  ensureTelemetryStringBuffer,
  normaliseRngBatchSize,
  normaliseRngSeed,
  normaliseTelemetryCapacityHints,
  normaliseVisitedHashEntries,
  normaliseVisitedHashLoad,
  nextPowerOfTwo,
  resolveProfilingEnabled,
} from './engineState.utils';

export type {
  EngineProfilingState,
  EngineScratchState,
  EngineState,
  EngineToggleState,
  RngCacheHandles,
  RngCacheParameters,
  TelemetryScratchHandles,
  TelemetryScratchRequest,
  VisitedHashScratchHandles,
} from './engineState.types';
export {
  DEFAULT_RNG_CACHE_BATCH_SIZE,
  DEFAULT_VISITED_HASH_LOAD_FACTOR,
} from './engineState.constants';

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
