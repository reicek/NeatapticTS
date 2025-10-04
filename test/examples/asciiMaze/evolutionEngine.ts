// Handles the main NEAT evolution loop for maze solving
// Exports: EvolutionEngine class with static methods

import { MazeMovement } from './mazeMovement';
import {
  createEngineState,
  EngineScratchState,
  EngineState,
  EngineToggleState,
  RngCacheParameters,
} from './evolutionEngine/engineState';
import {
  accumulateProfilingDuration,
  clearDeterministicMode,
  getProfilingAccumulators,
  isProfilingDetailsEnabled,
  profilingStartTimestamp,
  readHighResolutionTime,
  resolveRngParameters,
  setDeterministicMode,
} from './evolutionEngine/rngAndTiming';
import {
  ensureConnFlagsCapacity,
  ensureLogitsRingCapacity,
  ensureScratchCapacity,
  maybeShrinkScratch,
} from './evolutionEngine/scratchPools';
import {
  sampleSegmentIntoScratch,
} from './evolutionEngine/sampling';
import {
  applyCompassWarmStart,
  centerOutputBiases,
} from './evolutionEngine/populationPruning';
import {
  updatePlateauState,
  handleSimplifyState,
  getSortedIndicesByScore,
  ensureOutputIdentity,
  handleSpeciesHistory,
  maybeExpandPopulation,
  pruneSaturatedHiddenOutputs,
  antiCollapseRecovery,
  compactPopulation,
} from './evolutionEngine/populationDynamics';
import {
  logActionEntropy,
  logOutputBiasStats,
  logLogitsAndCollapse,
  logExploration,
  logDiversity,
  collectTelemetryTail,
} from './evolutionEngine/telemetryMetrics';
import {
  buildLamarckianTrainingSet,
  adjustOutputBiasesAfterTraining,
  pretrainPopulationWarmStart,
  applyLamarckianTraining,
  warmStartPopulationIfNeeded,
} from './evolutionEngine/trainingWarmStart';
import {
  normalizeRunOptions,
  prepareEnvironmentForRun,
  createAndSeedNeat,
} from './evolutionEngine/optionsAndSetup';
import {
  checkCancellation,
  prepareLoopHelpers,
  checkStopConditions,
  persistSnapshotIfNeeded,
  updateDashboardAndMaybeFlush,
  updateDashboardPeriodic,
  emitProfileSummary,
} from './evolutionEngine/evolutionLoop';
import {
  INetwork,
  IRunMazeEvolutionOptions,
} from './interfaces';

/**
 * The `EvolutionEngine` class encapsulates the entire neuro-evolution process for training agents to solve mazes.
 * It leverages the NEAT (Neuro-Evolution of Augmenting Topologies) algorithm to evolve neural networks.
 * This class is designed as a static utility, meaning you don't need to instantiate it to use its methods.
 *
 * Key Responsibilities:
 * - Orchestrating the main evolution loop (generations, evaluation, selection, reproduction).
 * - Configuring and initializing the NEAT algorithm with appropriate parameters.
 * - Managing a hybrid evolution strategy that combines genetic exploration (NEAT) with local optimization (backpropagation).
 * - Handling curriculum learning, where agents can be trained on a sequence of increasingly difficult mazes.
 * - Providing utilities for logging, visualization, and debugging the evolutionary process.
 */
export class EvolutionEngine {
  /** Shared engine state instance backing all façade helpers. */
  static #STATE: EngineState = createEngineState();

  /** Retrieve the shared scratch buffer bundle. */
  static get #scratch(): EngineScratchState {
    return EvolutionEngine.#STATE.scratch;
  }

  /** Retrieve the shared toggle configuration bundle. */
  static get #toggles(): EngineToggleState {
    return EvolutionEngine.#STATE.toggles;
  }

  /** Expose the shared engine state for extracted module consumers. */
  static get sharedState(): EngineState {
    return EvolutionEngine.#STATE;
  }

  /**
   * Pooled scratch buffer used by telemetry softmax/entropy calculations.
   * @remarks Non-reentrant: telemetry functions that use this buffer must not be
   * called concurrently (single-threaded runtime assumption holds for Node/browser).
   */
  static get #SCRATCH_EXPS(): Float64Array {
    return EvolutionEngine.#scratch.exps;
  }
  static set #SCRATCH_EXPS(buffer: Float64Array) {
    EvolutionEngine.#scratch.exps = buffer;
  }
  /** Reusable empty vector constant to avoid ephemeral allocations from `|| []` fallbacks. */
  static #EMPTY_VEC: any[] = [];
  /** Pooled stats buffers (always resident) for means & stds. */
  static get #SCRATCH_MEANS(): Float64Array {
    return EvolutionEngine.#scratch.means;
  }
  static set #SCRATCH_MEANS(buffer: Float64Array) {
    EvolutionEngine.#scratch.means = buffer;
  }
  static get #SCRATCH_STDS(): Float64Array {
    return EvolutionEngine.#scratch.standardDeviations;
  }
  static set #SCRATCH_STDS(buffer: Float64Array) {
    EvolutionEngine.#scratch.standardDeviations = buffer;
  }
  /** Bias telemetry buffer reused for output bias statistics. */
  static get #SCRATCH_BIAS_TA(): Float64Array {
    return EvolutionEngine.#scratch.biasTelemetryScratch;
  }
  static set #SCRATCH_BIAS_TA(buffer: Float64Array) {
    EvolutionEngine.#scratch.biasTelemetryScratch = buffer;
  }
  /** Kurtosis related buffers allocated lazily when first needed (non-reduced telemetry). */
  static get #SCRATCH_KURT(): Float64Array | undefined {
    return EvolutionEngine.#scratch.kurtosis;
  }
  static set #SCRATCH_KURT(buffer: Float64Array | undefined) {
    EvolutionEngine.#scratch.kurtosis = buffer;
  }
  static get #SCRATCH_M2_RAW(): Float64Array {
    return EvolutionEngine.#scratch.secondMomentRaw;
  }
  static set #SCRATCH_M2_RAW(buffer: Float64Array) {
    EvolutionEngine.#scratch.secondMomentRaw = buffer;
  }
  static get #SCRATCH_M3_RAW(): Float64Array | undefined {
    return EvolutionEngine.#scratch.thirdMomentRaw;
  }
  static set #SCRATCH_M3_RAW(buffer: Float64Array | undefined) {
    EvolutionEngine.#scratch.thirdMomentRaw = buffer;
  }
  static get #SCRATCH_M4_RAW(): Float64Array | undefined {
    return EvolutionEngine.#scratch.fourthMomentRaw;
  }
  static set #SCRATCH_M4_RAW(buffer: Float64Array | undefined) {
    EvolutionEngine.#scratch.fourthMomentRaw = buffer;
  }
  /**
   * Small integer scratch buffer used for directional move counts (N,E,S,W).
   * @remarks Non-reentrant: reused across telemetry calls.
   */
  static get #SCRATCH_COUNTS(): Int32Array {
    return EvolutionEngine.#scratch.moveCounts;
  }
  static set #SCRATCH_COUNTS(buffer: Int32Array) {
    EvolutionEngine.#scratch.moveCounts = buffer;
  }
  /**
   * Open-address hash table for visited coordinate detection (pairs packed into 32-bit int).
   * Length is always a power of two; uses linear probing. A value of 0 represents EMPTY so we offset packed values by +1.
   */
  static get #SCRATCH_VISITED_HASH(): Int32Array {
    return EvolutionEngine.#scratch.visitedHashTable;
  }
  static set #SCRATCH_VISITED_HASH(buffer: Int32Array) {
    EvolutionEngine.#scratch.visitedHashTable = buffer;
  }
  /** Load factor threshold (~0.7) for resizing visited hash. */
  static get #VISITED_HASH_LOAD(): number {
    return EvolutionEngine.#scratch.visitedHashLoadFactor;
  }
  static set #VISITED_HASH_LOAD(loadFactor: number) {
    EvolutionEngine.#scratch.visitedHashLoadFactor = loadFactor;
  }
  /** Knuth multiplicative hashing constant (32-bit golden ratio). */
  static #HASH_KNUTH_32 = 2654435761 >>> 0;
  /** Scratch species id buffer (dynamic growth). */
  static get #SCRATCH_SPECIES_IDS(): Int32Array {
    return EvolutionEngine.#scratch.speciesIds;
  }
  static set #SCRATCH_SPECIES_IDS(buffer: Int32Array) {
    EvolutionEngine.#scratch.speciesIds = buffer;
  }
  /** Scratch species count buffer parallel to ids. */
  static get #SCRATCH_SPECIES_COUNTS(): Int32Array {
    return EvolutionEngine.#scratch.speciesCounts;
  }
  static set #SCRATCH_SPECIES_COUNTS(buffer: Int32Array) {
    EvolutionEngine.#scratch.speciesCounts = buffer;
  }
  /** Reusable candidate connection object buffer. */
  static get #SCRATCH_CONN_CAND(): any[] {
    return EvolutionEngine.#scratch.connectionCandidates;
  }
  static set #SCRATCH_CONN_CAND(buffer: any[]) {
    EvolutionEngine.#scratch.connectionCandidates = buffer;
  }
  /** Reusable hidden->output connection buffer. */
  static get #SCRATCH_HIDDEN_OUT(): any[] {
    return EvolutionEngine.#scratch.hiddenToOutputConnections;
  }
  static set #SCRATCH_HIDDEN_OUT(buffer: any[]) {
    EvolutionEngine.#scratch.hiddenToOutputConnections = buffer;
  }
  /** Flags buffer for connection disabling (grown on demand). */
  static get #SCRATCH_CONN_FLAGS(): Uint8Array {
    return EvolutionEngine.#scratch.connectionFlags;
  }
  static set #SCRATCH_CONN_FLAGS(buffer: Uint8Array) {
    EvolutionEngine.#scratch.connectionFlags = buffer;
  }
  /** Scratch index buffer holding sorted indices by score (reused per generation). */
  static get #SCRATCH_SORT_IDX(): number[] {
    return EvolutionEngine.#scratch.sortedIndexBuffer;
  }
  static set #SCRATCH_SORT_IDX(buffer: number[]) {
    EvolutionEngine.#scratch.sortedIndexBuffer = buffer;
  }
  /** Optional typed-array scratch used internally to accelerate sorting without allocating each call. */
  static get #SCRATCH_SORT_IDX_TA(): Int32Array | undefined {
    return EvolutionEngine.#scratch.sortedIndexTypedArray;
  }
  static set #SCRATCH_SORT_IDX_TA(buffer: Int32Array | undefined) {
    EvolutionEngine.#scratch.sortedIndexTypedArray = buffer;
  }
  /** Scratch stack (lo,hi pairs) for quicksort on indices. */
  static get #SCRATCH_QS_STACK(): Int32Array {
    return EvolutionEngine.#scratch.quicksortStack;
  }
  static set #SCRATCH_QS_STACK(buffer: Int32Array) {
    EvolutionEngine.#scratch.quicksortStack = buffer;
  }
  /** Scratch array reused when cloning an initial population. */
  static get #SCRATCH_POP_CLONE(): any[] {
    return EvolutionEngine.#scratch.populationCloneBuffer;
  }
  static set #SCRATCH_POP_CLONE(buffer: any[]) {
    EvolutionEngine.#scratch.populationCloneBuffer = buffer;
  }
  /** Scratch string array for activation function names (printNetworkStructure). */
  static get #SCRATCH_ACT_NAMES(): string[] {
    return EvolutionEngine.#scratch.activationNameBuffer;
  }
  static set #SCRATCH_ACT_NAMES(buffer: string[]) {
    EvolutionEngine.#scratch.activationNameBuffer = buffer;
  }
  /** Reusable object buffer for snapshot top entries. */
  static get #SCRATCH_SNAPSHOT_TOP(): any[] {
    return EvolutionEngine.#scratch.snapshotTopEntries;
  }
  static set #SCRATCH_SNAPSHOT_TOP(buffer: any[]) {
    EvolutionEngine.#scratch.snapshotTopEntries = buffer;
  }
  /** Reusable snapshot object (fields overwritten each persistence). */
  static get #SCRATCH_SNAPSHOT_OBJ(): any {
    return EvolutionEngine.#scratch.snapshotReusableObject;
  }
  static set #SCRATCH_SNAPSHOT_OBJ(buffer: any) {
    EvolutionEngine.#scratch.snapshotReusableObject = buffer;
  }
  /** Pooled buffer for mutation operator indices (shuffled prefix each use). */
  static get #SCRATCH_MUTOP_IDX(): Uint16Array {
    return EvolutionEngine.#scratch.mutationOperatorIndices;
  }
  static set #SCRATCH_MUTOP_IDX(buffer: Uint16Array) {
    EvolutionEngine.#scratch.mutationOperatorIndices = buffer;
  }
  /** Number of action outputs (N,E,S,W) */
  static #ACTION_DIM = 4;
  /** Precomputed 1/ln(4) for entropy normalization (micro-optimization). */
  static #INV_LOG4 = 1 / Math.log(4);
  /** Adaptive logits ring capacity (power-of-two). */
  static #LOGITS_RING_CAP = 512;
  /** Max allowed ring capacity (safety bound). */
  static #LOGITS_RING_CAP_MAX = 8192;
  /** Indicates SharedArrayBuffer-backed ring is active. */
  static #LOGITS_RING_SHARED = false;
  /** Logits ring (fallback non-shared row-of-vectors). */
  static get #SCRATCH_LOGITS_RING(): Float32Array[] {
    return EvolutionEngine.#scratch.logitsRing;
  }
  static set #SCRATCH_LOGITS_RING(buffer: Float32Array[]) {
    EvolutionEngine.#scratch.logitsRing = buffer;
  }
  /** Shared flat logits storage when shared mode enabled (length = cap * ACTION_DIM). */
  static get #SCRATCH_LOGITS_SHARED(): Float32Array | undefined {
    return EvolutionEngine.#scratch.sharedLogits;
  }
  static set #SCRATCH_LOGITS_SHARED(buffer: Float32Array | undefined) {
    EvolutionEngine.#scratch.sharedLogits = buffer;
  }
  /** Shared atomic write index (length=1 Int32). */
  static get #SCRATCH_LOGITS_SHARED_W(): Int32Array | undefined {
    return EvolutionEngine.#scratch.sharedLogitsWriteIndex;
  }
  static set #SCRATCH_LOGITS_SHARED_W(buffer: Int32Array | undefined) {
    EvolutionEngine.#scratch.sharedLogitsWriteIndex = buffer;
  }
  /** Write cursor for non-shared ring. */
  static get #SCRATCH_LOGITS_RING_W(): number {
    return EvolutionEngine.#scratch.logitsRingWriteCursor;
  }
  static set #SCRATCH_LOGITS_RING_W(value: number) {
    EvolutionEngine.#scratch.logitsRingWriteCursor = value;
  }
  /**
   * Small node index scratch arrays reused when extracting nodes by type.
   * @remarks Non-reentrant: do not call concurrently.
   */
  static get #SCRATCH_NODE_IDX(): Int32Array {
    return EvolutionEngine.#scratch.nodeIndexBuffer;
  }
  static set #SCRATCH_NODE_IDX(buffer: Int32Array) {
    EvolutionEngine.#scratch.nodeIndexBuffer = buffer;
  }
  /**
   * Object reference scratch array used as a short sample buffer (max 40 entries).
   * Avoids allocating small arrays inside hot telemetry paths.
   */
  static get #SCRATCH_SAMPLE(): any[] {
    return EvolutionEngine.#scratch.samplePool;
  }
  static set #SCRATCH_SAMPLE(buffer: any[]) {
    EvolutionEngine.#scratch.samplePool = buffer;
  }
  /** Reusable string assembly character buffer for small joins (grown geometrically). */
  static get #SCRATCH_STR(): string[] {
    return EvolutionEngine.#scratch.stringAssemblyBuffer;
  }
  static set #SCRATCH_STR(buffer: string[]) {
    EvolutionEngine.#scratch.stringAssemblyBuffer = buffer;
  }
  /** Frozen congruential RNG parameters reused by fast random helpers. */
  static get #RNG_PARAMETERS(): RngCacheParameters {
    return resolveRngParameters();
  }
  /** Small fixed-size visited table for tiny path exploration (<32) to avoid O(n^2) duplicate scan. */
  static get #SMALL_EXPLORE_TABLE(): Int32Array {
    return EvolutionEngine.#scratch.smallExploreTable;
  }
  static set #SMALL_EXPLORE_TABLE(buffer: Int32Array) {
    EvolutionEngine.#scratch.smallExploreTable = buffer;
  }
  /** Bit mask for SMALL_EXPLORE_TABLE indices (table length - 1). */
  static get #SMALL_EXPLORE_TABLE_MASK(): number {
    return EvolutionEngine.#scratch.smallExploreTable.length - 1;
  }
  /**
   * Enable deterministic mode and optionally reseed the internal RNG via the shared state helpers.
   *
   * @param seed Optional numeric seed used to reseed the deterministic RNG sequence.
   * @returns void.
   */
  static setDeterministic(seed?: number): void {
    setDeterministicMode(EvolutionEngine.#STATE, seed);
  }
  /** Disable deterministic mode. */
  static clearDeterministic(): void {
    clearDeterministicMode(EvolutionEngine.#STATE);
  }
  /** When true, telemetry skips higher-moment stats (kurtosis) for speed. */
  static get #REDUCED_TELEMETRY(): boolean {
    return EvolutionEngine.#toggles.reducedTelemetry;
  }
  static set #REDUCED_TELEMETRY(isReduced: boolean) {
    EvolutionEngine.#toggles.reducedTelemetry = isReduced;
  }
  /** Skip most telemetry logging & higher moment stats when true (minimal mode). */
  static get #TELEMETRY_MINIMAL(): boolean {
    return EvolutionEngine.#toggles.telemetryMinimal;
  }
  static set #TELEMETRY_MINIMAL(isMinimal: boolean) {
    EvolutionEngine.#toggles.telemetryMinimal = isMinimal;
  }
  /** Disable Baldwinian refinement phase when true. */
  static get #DISABLE_BALDWIN(): boolean {
    return EvolutionEngine.#toggles.disableBaldwinPhase;
  }
  static set #DISABLE_BALDWIN(isDisabled: boolean) {
    EvolutionEngine.#toggles.disableBaldwinPhase = isDisabled;
  }
  /** Default tail history size used by telemetry */
  static #RECENT_WINDOW = 40;
  /** Default population size used when no popSize provided in cfg */
  static #DEFAULT_POPSIZE = 500;
  /** Default mutation rate (fraction of individuals mutated per generation) */
  static #DEFAULT_MUTATION_RATE = 0.2;
  /** Default mutation amount (fractional magnitude for mutation operators) */
  static #DEFAULT_MUTATION_AMOUNT = 0.3;
  /** Fraction of population reserved for elitism when computing elitism count */
  static #DEFAULT_ELITISM_FRACTION = 0.1;
  /** Fraction of population reserved for provenance when computing provenance count */
  static #DEFAULT_PROVENANCE_FRACTION = 0.2;
  /** Default minimum hidden nodes for new NEAT instances */
  /**
   * Default minimum hidden nodes enforced for each evolved network.
   * Raised from 6 -> 12 to increase representational capacity for maze scaling.
   * Adjust via code edit if future experiments need a different baseline.
   */
  static #DEFAULT_MIN_HIDDEN = 20;
  /** Default target species count for adaptive target species heuristics */
  static #DEFAULT_TARGET_SPECIES = 10;
  /** Default supervised training error threshold for local training */
  static #DEFAULT_TRAIN_ERROR = 0.01;
  /** Default supervised training learning rate for local training */
  static #DEFAULT_TRAIN_RATE = 0.001;
  /** Default supervised training momentum */
  static #DEFAULT_TRAIN_MOMENTUM = 0.2;
  /** Default small batch size used during Lamarckian training */
  static #DEFAULT_TRAIN_BATCH_SMALL = 2;
  /** Default batch size used when training the fittest network for evaluation */
  static #DEFAULT_TRAIN_BATCH_LARGE = 20;
  /** Iterations used when training the fittest network for evaluation */
  static #FITTEST_TRAIN_ITERATIONS = 1000;
  /** Saturation fraction threshold triggering hidden-output pruning */
  static #SATURATION_PRUNE_THRESHOLD = 0.5;
  /** Small threshold used in several numeric comparisons */
  static #NUMERIC_EPSILON_SMALL = 0.01;
  /** Small threshold used for std flat detection in logits */
  static #LOGSTD_FLAT_THRESHOLD = 0.005;
  /** Default entropy range for adaptive target species */
  static #DEFAULT_ENTROPY_RANGE: [number, number] = [0.3, 0.8];
  /** Default smoothing factor for adaptive target species */
  static #DEFAULT_ADAPTIVE_SMOOTH = 0.5;
  /** Default probability used for small randomized jitter (25%) */
  static #DEFAULT_JITTER_PROB = 0.25;
  /** Default probability for 50/50 decisions */
  static #DEFAULT_HALF_PROB = 0.5;
  /** Fraction of sorted parents chosen as parent pool */
  static #DEFAULT_PARENT_FRACTION = 0.25;
  /** Small std threshold to consider 'small' std */
  static #DEFAULT_STD_SMALL = 0.25;
  /** Multiplier applied when std is small */
  static #DEFAULT_STD_ADJUST_MULT = 0.7;
  /** Initial weight range lower bound used by compass warm start */
  static #W_INIT_MIN = 0.55;
  /** Initial weight random range used by compass warm start */
  static #W_INIT_RANGE = 0.25;
  /** Base value for output bias initialization */
  static #OUTPUT_BIAS_BASE = 0.05;
  /** Step per output index when initializing output biases */
  static #OUTPUT_BIAS_STEP = 0.01;
  /** Absolute clamp applied after recentring output biases (prevents runaway values) */
  static #OUTPUT_BIAS_CLAMP = 5;
  /** Bias reset half-range (bias = rand * 2*R - R) */
  static #BIAS_RESET_HALF_RANGE = 0.1;
  /** Connection weight reset half-range (weight = rand * 2*R - R) */
  static #CONN_WEIGHT_RESET_HALF_RANGE = 0.2;
  /** Log tag for action entropy telemetry lines */
  static #LOG_TAG_ACTION_ENTROPY = '[ACTION_ENTROPY]';
  /** Log tag for output bias telemetry lines */
  static #LOG_TAG_OUTPUT_BIAS = '[OUTPUT_BIAS]';
  /** Log tag for logits telemetry lines */
  static #LOG_TAG_LOGITS = '[LOGITS]';
  /** High target probability for the chosen action during supervised warm start */
  static #TRAIN_OUT_PROB_HIGH = 0.92;
  /** Low target probability for non-chosen actions during supervised warm start */
  static #TRAIN_OUT_PROB_LOW = 0.02;
  /** Progress intensity: medium (single open path typical) */
  static #PROGRESS_MEDIUM = 0.7;
  /** Progress intensity: strong forward signal */
  static #PROGRESS_STRONG = 0.9;
  /** Progress intensity: typical junction neutrality */
  static #PROGRESS_JUNCTION = 0.6;
  /** Progress intensity: four-way moderate signal */
  static #PROGRESS_FOURWAY = 0.55;
  /** Progress intensity: regressing / weak progress */
  static #PROGRESS_REGRESS = 0.4;
  /** Progress intensity: mild regression / noise */
  static #PROGRESS_MILD_REGRESS = 0.45;
  /** Minimal progress positive blip used in a corner-case sample */
  static #PROGRESS_MIN_SIGNAL = 0.001;
  /** Augmentation: base openness jitter value */
  static #AUGMENT_JITTER_BASE = 0.95;
  /** Augmentation: openness jitter range added to base */
  static #AUGMENT_JITTER_RANGE = 0.05;
  /** Augmentation: probability to jitter progress channel */
  static #AUGMENT_PROGRESS_JITTER_PROB = 0.35;
  /** Augmentation: progress delta full range */
  static #AUGMENT_PROGRESS_DELTA_RANGE = 0.1;
  /** Augmentation: progress delta half range (range/2) */
  static #AUGMENT_PROGRESS_DELTA_HALF = 0.05;
  /** Max iterations used during population pretrain */
  static #PRETRAIN_MAX_ITER = 60;
  /** Base iterations added in pretrain (8 + floor(setLen/2)) */
  static #PRETRAIN_BASE_ITER = 8;
  /** Default learning rate used during pretraining population warm-start */
  static #DEFAULT_PRETRAIN_RATE = 0.002;
  /** Default momentum used during pretraining population warm-start */
  static #DEFAULT_PRETRAIN_MOMENTUM = 0.1;
  /** Default batch size used during population pretraining */
  static #DEFAULT_PRETRAIN_BATCH = 4;
  /** Entropy threshold used in collapse heuristics */
  static #ENTROPY_COLLAPSE_THRESHOLD = 0.35;
  /** Stability threshold used in collapse heuristics */
  static #STABILITY_COLLAPSE_THRESHOLD = 0.97;
  /** Window size (consecutive generations) used to detect species collapse */
  static #SPECIES_COLLAPSE_WINDOW = 20;
  /** Max length of species history buffer */
  static #SPECIES_HISTORY_MAX = 50;
  /** Collapse streak trigger (consecutive collapsed gens before recovery) */
  static #COLLAPSE_STREAK_TRIGGER = 6;
  /** Mutation rate escalation cap during collapse recovery */
  static #COLLAPSE_MUTRATE_CAP = 0.6;
  /** Mutation amount escalation cap during collapse recovery */
  static #COLLAPSE_MUTAMOUNT_CAP = 0.8;
  /** Novelty blend factor escalation cap during collapse recovery */
  static #COLLAPSE_NOVELTY_BLEND_CAP = 0.4;
  /** Mutation rate escalation multiplier */
  static #COLLAPSE_MUTRATE_MULT = 1.5;
  /** Mutation amount escalation multiplier */
  static #COLLAPSE_MUTAMOUNT_MULT = 1.3;
  /** Novelty blend factor escalation multiplier */
  static #COLLAPSE_NOVELTY_MULT = 1.2;
  /** Small-partition cutoff for quicksort; tuned empirically (was 16). */
  static #QS_SMALL_THRESHOLD = 24;
  /** Branchless (dx,dy)->direction index map ((dx+1)*3 + (dy+1)) => 0..3 or -1. */
  static #DIR_DELTA_TO_INDEX: Int8Array = (() => {
    const map = new Int8Array(9); // 3x3 neighborhood centered at (0,0)
    map.fill(-1);
    // N (0,-1), E(1,0), S(0,1), W(-1,0)
    map[(0 + 1) * 3 + (-1 + 1)] = 0; // N
    map[(1 + 1) * 3 + (0 + 1)] = 1; // E
    map[(0 + 1) * 3 + (1 + 1)] = 2; // S
    map[(-1 + 1) * 3 + (0 + 1)] = 3; // W
    return map;
  })();

  /**
   * Populate the engine's pooled node-index scratch buffer with indices of nodes matching `type`.
   * @internal - Small helper used by various engine methods; retained for internal use.
   */
  static #getNodeIndicesByType(nodes: any[] | undefined, type: string): number {
    if (!Array.isArray(nodes) || nodes.length === 0) return 0;
    let writeCount = 0;
    let scratch = EvolutionEngine.#SCRATCH_NODE_IDX;
    for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
      const nodeRef = nodes[nodeIndex];
      if (!nodeRef || nodeRef.type !== type) continue;
      if (writeCount >= scratch.length) {
        const nextCapacity = 1 << Math.ceil(Math.log2(writeCount + 1));
        const grown = new Int32Array(nextCapacity);
        grown.set(scratch);
        EvolutionEngine.#SCRATCH_NODE_IDX = grown;
        scratch = grown;
      }
      scratch[writeCount++] = nodeIndex;
    }
    return writeCount;
  }

  /**
   * Collect enabled outgoing connections from a hidden node terminating at output nodes.
   * @internal - Small helper used by network analysis methods; retained for internal use.
   */
  static #collectHiddenToOutputConns(
    hiddenNode: any,
    nodesRef: any[],
    outputCount: number,
  ): any[] {
    if (
      !hiddenNode?.connections ||
      !Array.isArray(nodesRef) ||
      outputCount <= 0
    )
      return [];
    const maxScratch = EvolutionEngine.#SCRATCH_NODE_IDX.length;
    const effectiveOutputCount = Math.min(
      outputCount | 0,
      maxScratch,
      nodesRef.length,
    );
    if (effectiveOutputCount <= 0) return [];
    const hiddenOutBuffer = EvolutionEngine.#SCRATCH_HIDDEN_OUT;
    hiddenOutBuffer.length = 0;
    const outgoing = hiddenNode.connections.out ?? EvolutionEngine.#EMPTY_VEC;
    for (let outIndex = 0; outIndex < outgoing.length; outIndex++) {
      const candidate = outgoing[outIndex];
      if (!candidate || candidate.enabled === false) continue;
      for (
        let outputIndex = 0;
        outputIndex < effectiveOutputCount;
        outputIndex++
      ) {
        const nodeIdx = EvolutionEngine.#SCRATCH_NODE_IDX[outputIndex];
        const targetNode = nodesRef[nodeIdx];
        if (candidate.to === targetNode) {
          hiddenOutBuffer.push(candidate);
          break;
        }
      }
    }
    return hiddenOutBuffer;
  }

  /**
   * Aggregate and emit per-generation telemetry and run collapse/anti-collapse checks.
   *
   * This method centralises all higher-level telemetry duties for a single generation. It is
   * intentionally best-effort: all internal telemetry calls are wrapped in a try/catch so that
   * telemetry failures cannot interrupt the main evolution loop.
   *
   * Steps:
   * 0. Global guard: skip telemetry entirely when `#TELEMETRY_MINIMAL` is enabled.
   * 1. Emit action-entropy metrics (path-based uncertainty).
   * 2. Emit output-bias statistics for the fittest network when available.
   * 3. Compute logits-level statistics and run collapse detection / recovery heuristics.
   * 4. Emit exploration telemetry (unique coverage / progress).
   * 5. Emit diversity metrics (species richness, Simpson index, weight std).
   * 6. Optionally record profiling timings into the shared profiling accumulators.
   *
   * @param neat - NEAT instance exposing `population` and related internals used by some telemetry probes.
   * @param fittest - The current fittest genome/network (may be undefined during initialization).
   * @param genResult - Per-generation result object (expected to contain `path` and other telemetry fields).
   * @param generationIndex - Completed generation index used in telemetry labels.
   * @param writeLog - Logging helper used to emit telemetry lines (accepts a single string argument).
   * @returns void
   * @example
   * EvolutionEngine['#logGenerationTelemetry'](neat, neat.fittest, genResult, gen, msg => process.stdout.write(msg));
   * @internal
   */
  static #logGenerationTelemetry(
    neat: any,
    fittest: any,
    genResult: any,
    generationIndex: number,
    writeLog: (msg: string) => void,
  ): void {
    // Step 0: Global guard for minimal telemetry mode.
    if (EvolutionEngine.#TELEMETRY_MINIMAL) return;

    // Start profiling window if enabled.
    const profilingEnabled = isProfilingDetailsEnabled(EvolutionEngine.#STATE);
    const profilingStart = profilingEnabled
      ? profilingStartTimestamp(EvolutionEngine.#STATE)
      : 0;

    try {
      // Step 1: Action entropy telemetry
      logActionEntropy({
        state: EvolutionEngine.#STATE,
        generationResult: genResult,
        generationIndex,
        safeWrite: writeLog,
      });

      // Step 2: Output bias statistics (fittest may be undefined early on)
      logOutputBiasStats({
        state: EvolutionEngine.#STATE,
        fittest,
        generationIndex,
        safeWrite: writeLog,
      });

      // Step 3: Logits statistics and collapse detection/recovery
      logLogitsAndCollapse({
        state: EvolutionEngine.#STATE,
        neat,
        fittest,
        generationIndex,
        safeWrite: writeLog,
        actionDimension: EvolutionEngine.#ACTION_DIM,
        recentWindow: EvolutionEngine.#RECENT_WINDOW,
        reducedTelemetry: EvolutionEngine.#REDUCED_TELEMETRY,
        onCollapseRecovery: () => {
          antiCollapseRecovery(
            EvolutionEngine.#STATE,
            neat,
            generationIndex,
            writeLog,
            sampleSegmentIntoScratch,
          );
        },
      });

      // Step 4: Exploration telemetry (path uniqueness, progress)
      logExploration({
        state: EvolutionEngine.#STATE,
        generationResult: genResult,
        generationIndex,
        safeWrite: writeLog,
      });

      // Step 5: Diversity metrics (species richness, Simpson, weight std)
      logDiversity({
        state: EvolutionEngine.#STATE,
        neat,
        generationIndex,
        safeWrite: writeLog,
      });
    } catch {
      // Swallow any unexpected telemetry exception to avoid disrupting the evolution core loop.
    }

    // Step 6: Record profiling delta if profiling was enabled at entry.
    if (profilingEnabled) {
      accumulateProfilingDuration(
        EvolutionEngine.#STATE,
        'telemetry',
        profilingStartTimestamp(EvolutionEngine.#STATE) - profilingStart || 0,
      );
    }
  }

  /**
   * Run one generation: evolve, ensure output identity, update species history, maybe expand population,
   * and run Lamarckian training if configured.
   *
   * Behaviour & contract:
   *  - Performs a single NEAT generation step in a best-effort, non-throwing manner.
   *  - Measures profiling durations when `doProfile` is truthy. Profiling is optional and
   *    kept allocation-free (uses local numeric temporaries only).
   *  - Invokes the following steps in order (each step is wrapped in a try/catch so
   *    the evolution loop remains resilient to per-stage failures):
   *      1) `neat.evolve()` to produce the fittest network for this generation.
   *      2) `#ensureOutputIdentity` to normalise output activations for consumers.
   *      3) `#handleSpeciesHistory` to update species statistics and history.
   *      4) `#maybeExpandPopulation` to grow the population when configured and warranted.
   *      5) Optional Lamarckian warm-start training via `#applyLamarckianTraining`.
   *  - The method is allocation-light and reuses engine helpers / pooled buffers where
   *    appropriate. It never throws; internal errors are swallowed and optionally logged
   *    via the provided `safeWrite` function.
   *
   * Parameters (props):
   * @param neat - NEAT driver instance used for evolving the generation.
   * @param doProfile - When truthy measure timing for the evolve step (ms) using engine clock.
   * @param lamarckianIterations - Number of supervised training iterations to run per genome (0 to skip).
   * @param lamarckianTrainingSet - Array of supervised training cases used for warm-start (may be empty).
   * @param lamarckianSampleSize - Optional per-network sample size used by the warm-start routine.
   * @param safeWrite - Safe logging function; used only for best-effort diagnostic messages.
   * @param completedGenerations - Current generation index (used by expansion heuristics).
   * @param dynamicPopEnabled - Whether dynamic population expansion is enabled.
   * @param dynamicPopMax - Upper bound on population size for expansion.
   * @param plateauGenerations - Window size used by plateau detection.
   * @param plateauCounter - Current plateau counter used by expansion heuristics.
   * @param dynamicPopExpandInterval - Generation interval to attempt expansion.
   * @param dynamicPopExpandFactor - Fractional growth factor used to compute new members.
   * @param dynamicPopPlateauSlack - Minimum plateau ratio required to trigger expansion.
   *
   * @returns An object shaped { fittest, tEvolve, tLamarck } where:
   *  - `fittest` is the network returned by `neat.evolve()` (may be null on error),
   *  - `tEvolve` is the measured evolve duration in milliseconds when `doProfile` is true (0 otherwise),
   *  - `tLamarck` is the total time spent in Lamarckian training (0 when skipped).
   *
   * @example
   * // Run a single generation with profiling and optional Lamarckian warm-start
   * const { fittest, tEvolve, tLamarck } = await EvolutionEngine['#runGeneration'](
   *   neatInstance,
   *   true,   // doProfile
   *   5,      // lamarckianIterations
   *   trainingSet,
   *   16,     // lamarckianSampleSize
   *   console.log,
   *   genIndex,
   *   true,
   *   500,
   *   10,
   *   plateauCounter,
   *   5,
   *   0.1,
   *   0.75
   * );
   *
   * @internal
   */
  static async #runGeneration(
    neat: any,
    doProfile: boolean,
    lamarckianIterations: number,
    lamarckianTrainingSet: any[],
    lamarckianSampleSize: number | undefined,
    safeWrite: (msg: string) => void,
    completedGenerations: number,
    dynamicPopEnabled: boolean,
    dynamicPopMax: number,
    plateauGenerations: number,
    plateauCounter: number,
    dynamicPopExpandInterval: number,
    dynamicPopExpandFactor: number,
    dynamicPopPlateauSlack: number,
  ) {
    // Step 0: Local descriptive aliases and profiling setup.
    const profileEnabled = Boolean(doProfile);
    const clockNow = () => readHighResolutionTime(EvolutionEngine.#STATE);
    const startTime = profileEnabled ? clockNow() : 0;

    // Results we will populate. Keep names descriptive for readability in hot paths.
    let fittestNetwork: any = null;
    let evolveDuration = 0;
    let lamarckDuration = 0;

    // Step 1: Run the evolutionary step and measure time when profiling is enabled.
    try {
      // `neat` is expected to provide an async `evolve()` method that returns the fittest genome.
      fittestNetwork = await neat?.evolve();
      if (profileEnabled) evolveDuration = clockNow() - startTime;
    } catch (evolveError) {
      // Best-effort: log a short diagnostic and continue. Do not rethrow.
      try {
        safeWrite?.(`#runGeneration: evolve() threw: ${String(evolveError)}`);
      } catch {}
      // leave fittestNetwork null and continue with remaining housekeeping.
    }

    // Step 2: Ensure outputs are using identity activation where required (non-throwing).
    try {
      ensureOutputIdentity(neat);
    } catch (identityError) {
      try {
        safeWrite?.(
          `#runGeneration: ensureOutputIdentity failed: ${String(
            identityError,
          )}`,
        );
      } catch {}
    }

    // Step 3: Update species history (best-effort; internal errors are swallowed).
    try {
      const speciesHistoryRef: number[] =
        (EvolutionEngine as any)._speciesHistory ??
        EvolutionEngine.#EMPTY_VEC;
      handleSpeciesHistory(EvolutionEngine.#STATE, neat, speciesHistoryRef);
    } catch (speciesError) {
      try {
        safeWrite?.(
          `#runGeneration: handleSpeciesHistory failed: ${String(speciesError)}`,
        );
      } catch {}
    }

    // Step 4: Possibly expand the population when configured and plateau conditions are met.
    try {
      maybeExpandPopulation(
        EvolutionEngine.#STATE,
        neat,
        Boolean(dynamicPopEnabled),
        completedGenerations,
        dynamicPopMax,
        plateauGenerations,
        plateauCounter,
        dynamicPopExpandInterval,
        dynamicPopExpandFactor,
        dynamicPopPlateauSlack,
        safeWrite,
      );
    } catch (expandError) {
      try {
        safeWrite?.(
          `#runGeneration: maybeExpandPopulation failed: ${String(expandError)}`,
        );
      } catch {}
    }

    // Step 5: Optional Lamarckian warm-start training. This step may be expensive;
    // we keep it synchronous as the called helper currently returns a numeric time.
    try {
      const shouldRunLamarckian =
        Number.isFinite(lamarckianIterations) &&
        lamarckianIterations > 0 &&
        Array.isArray(lamarckianTrainingSet) &&
        lamarckianTrainingSet.length > 0;

      if (shouldRunLamarckian) {
        // The helper returns the measured time (ms) spent in training when profiling is enabled.
        lamarckDuration = applyLamarckianTraining(
          neat,
          lamarckianTrainingSet,
          lamarckianIterations,
          lamarckianSampleSize,
          safeWrite,
          doProfile,
          completedGenerations,
          EvolutionEngine.#STATE,
          {
            DEFAULT_TRAIN_ERROR: EvolutionEngine.#DEFAULT_TRAIN_ERROR,
            DEFAULT_TRAIN_RATE: EvolutionEngine.#DEFAULT_TRAIN_RATE,
            DEFAULT_TRAIN_MOMENTUM: EvolutionEngine.#DEFAULT_TRAIN_MOMENTUM,
            DEFAULT_TRAIN_BATCH_SMALL: EvolutionEngine.#DEFAULT_TRAIN_BATCH_SMALL,
          },
          (network) => {
            adjustOutputBiasesAfterTraining(
              network,
              EvolutionEngine.#STATE,
              {
                DEFAULT_STD_SMALL: EvolutionEngine.#DEFAULT_STD_SMALL,
                DEFAULT_STD_ADJUST_MULT: EvolutionEngine.#DEFAULT_STD_ADJUST_MULT,
              },
              EvolutionEngine.#SCRATCH_NODE_IDX,
              EvolutionEngine.#getNodeIndicesByType,
            );
          },
        );
      }
    } catch (lamarckError) {
      try {
        safeWrite?.(
          `#runGeneration: applyLamarckianTraining failed: ${String(
            lamarckError,
          )}`,
        );
      } catch {}
    }

    // Final: return the canonical result shape. Keep original property names for callers.
    return {
      fittest: fittestNetwork,
      tEvolve: evolveDuration,
      tLamarck: lamarckDuration,
    } as any;
  }

  /**
   * Simulate the supplied `fittest` genome/network and perform allocation-light postprocessing.
   *
   * Behaviour & contract:
   *  - Runs the simulation via `MazeMovement.simulateAgent` and attaches compact telemetry
   *    (saturation fraction, action entropy) directly onto the `fittest` object (in-place).
   *  - When per-step logits are returned the helper attempts to copy them into the engine's pooled
   *    ring buffers to avoid per-run allocations. Two copy modes are supported:
   *      1) Shared SAB-backed flat Float32Array with an atomic Int32 write index (cross-worker safe).
   *      2) Local in-process per-row Float32Array ring (`#SCRATCH_LOGITS_RING`).
   *  - Best-effort: all mutation and buffer-copy steps are guarded; failures are swallowed so the
   *    evolution loop is not interrupted. Use `safeWrite` for optional diagnostic messages.
   *
   * Steps (high level):
   *  1) Run the simulator and capture wall-time when `doProfile` is truthy.
   *  2) Attach compact telemetry fields to `fittest` and ensure legacy `_lastStepOutputs` exists.
   *  3) If per-step logits are available, ensure ring capacity and copy them into the selected ring.
   *  4) Optionally prune saturated hidden->output connections and emit telemetry via `#logGenerationTelemetry`.
   *  5) Return the raw simulation result and elapsed simulation time (ms when profiling enabled).
   *
   * Notes on pooling / reentrancy:
   *  - The local ring `#SCRATCH_LOGITS_RING` is not re-entrant; callers must avoid concurrent writes.
   *  - When `#LOGITS_RING_SHARED` is true we prefer the SAB-backed path which uses Atomics and is safe
   *    for cross-thread producers.
   *
   * @param fittest Genome/network considered the generation's best; may be mutated with metadata.
   * @param encodedMaze Maze descriptor used by the simulator.
   * @param startPosition Start co-ordinates passed as-is to the simulator.
   * @param exitPosition Exit co-ordinates passed as-is to the simulator.
   * @param distanceMap Optional precomputed distance map consumed by the simulator.
   * @param maxSteps Optional maximum simulation steps; may be undefined to allow default.
   * @param doProfile When truthy measure and return the simulation time in milliseconds.
   * @param safeWrite Optional logger used for non-fatal diagnostic messages.
   * @param logEvery Emit telemetry every `logEvery` generations (0 disables periodic telemetry).
   * @param completedGenerations Current generation index used for conditional telemetry.
   * @param neat NEAT driver instance passed to telemetry hooks.
   * @returns An object { generationResult, simTime } where simTime is ms when profiling is enabled.
   * @example
   * const { generationResult, simTime } = EvolutionEngine['#simulateAndPostprocess'](
   *   bestGenome, maze, start, exit, distMap, 1000, true, console.log, 10, genIdx, neat
   * );
   * @internal
   */
  static #simulateAndPostprocess(
    fittest: any,
    encodedMaze: any,
    startPosition: any,
    exitPosition: any,
    distanceMap: any,
    maxSteps: number | undefined,
    doProfile: boolean,
    safeWrite: (msg: string) => void,
    logEvery: number,
    completedGenerations: number,
    neat: any,
  ): { generationResult: any; simTime: number } {
    // Step 1: Run simulator and optionally capture elapsed time.
    const startTime = doProfile
      ? readHighResolutionTime(EvolutionEngine.#STATE)
      : 0;
    const simResult = MazeMovement.simulateAgent(
      fittest,
      encodedMaze,
      startPosition,
      exitPosition,
      distanceMap,
      maxSteps,
    );

    // Best-effort: attach legacy buffer refs and compact telemetry onto the genome.
    try {
      if (!(fittest as any)._lastStepOutputs) {
        (fittest as any)._lastStepOutputs =
          EvolutionEngine.#SCRATCH_LOGITS_RING;
      }
    } catch (legacyBufferAttachmentError) {
      EvolutionEngine.#swallowError(legacyBufferAttachmentError);
    }

    try {
      (fittest as any)._saturationFraction = simResult?.saturationFraction ?? 0;
      (fittest as any)._actionEntropy = simResult?.actionEntropy ?? 0;
    } catch (telemetryAssignError) {
      EvolutionEngine.#swallowError(telemetryAssignError);
    }

    // Step 3: If the simulator returned per-step logits, copy them into the pooled ring buffers.
    try {
      const perStepLogits: number[][] | undefined = (simResult as any)
        ?.stepOutputs;
      if (Array.isArray(perStepLogits) && perStepLogits.length > 0) {
        // Ensure the ring can hold the incoming sequence to avoid overflow resize churn.
        const logitsRingResult = ensureLogitsRingCapacity({
          state: EvolutionEngine.#STATE,
          desiredRecentSteps: perStepLogits.length,
          currentCapacity: EvolutionEngine.#LOGITS_RING_CAP,
          minimumCapacity: 128,
          maximumCapacity: EvolutionEngine.#LOGITS_RING_CAP_MAX,
          actionDimension: EvolutionEngine.#ACTION_DIM,
          sharedModeEnabled: EvolutionEngine.#LOGITS_RING_SHARED,
        });
        EvolutionEngine.#LOGITS_RING_CAP = logitsRingResult.capacity;
        EvolutionEngine.#LOGITS_RING_SHARED =
          logitsRingResult.sharedModeEnabled;

        const useSharedSAB =
          EvolutionEngine.#LOGITS_RING_SHARED &&
          EvolutionEngine.#SCRATCH_LOGITS_SHARED &&
          EvolutionEngine.#SCRATCH_LOGITS_SHARED_W;

        const actionDim = EvolutionEngine.#ACTION_DIM;

        if (useSharedSAB) {
          // Shared flat Float32Array layout: [ idx(Int32), floats... ] with atomic index at view[0].
          const sharedBuffer = EvolutionEngine
            .#SCRATCH_LOGITS_SHARED as Float32Array;
          const atomicIndexView = EvolutionEngine
            .#SCRATCH_LOGITS_SHARED_W as Int32Array;
          const capacityMask = EvolutionEngine.#LOGITS_RING_CAP - 1;

          for (
            let stepIndex = 0;
            stepIndex < perStepLogits.length;
            stepIndex++
          ) {
            const logitsVector = perStepLogits[stepIndex];
            if (!Array.isArray(logitsVector)) continue;

            // Reserve a slot atomically and compute its base offset in the flat buffer.
            const currentWriteIndex =
              Atomics.load(atomicIndexView, 0) & capacityMask;
            const baseOffset = currentWriteIndex * actionDim;
            const copyLength = Math.min(actionDim, logitsVector.length);
            for (let dimIndex = 0; dimIndex < copyLength; dimIndex++) {
              sharedBuffer[baseOffset + dimIndex] = logitsVector[dimIndex] ?? 0;
            }

            // Advance the atomic write pointer (wrap safely using 31-bit mask to avoid negative values).
            Atomics.store(
              atomicIndexView,
              0,
              (Atomics.load(atomicIndexView, 0) + 1) & 0x7fffffff,
            );
          }
        } else {
          // Fallback: local per-row ring of Float32Array rows stored in `#SCRATCH_LOGITS_RING`.
          const ringCapacityMask = EvolutionEngine.#LOGITS_RING_CAP - 1;

          for (
            let stepIndex = 0;
            stepIndex < perStepLogits.length;
            stepIndex++
          ) {
            const logitsVector = perStepLogits[stepIndex];
            if (!Array.isArray(logitsVector)) continue;

            const writePos =
              EvolutionEngine.#SCRATCH_LOGITS_RING_W & ringCapacityMask;
            const targetRow = EvolutionEngine.#SCRATCH_LOGITS_RING[writePos];
            const copyLength = Math.min(actionDim, logitsVector.length);

            // Copy into the pooled Float32Array row (no allocation).
            for (let dimIndex = 0; dimIndex < copyLength; dimIndex++) {
              targetRow[dimIndex] = logitsVector[dimIndex] ?? 0;
            }

            // Advance the non-shared ring write cursor.
            EvolutionEngine.#SCRATCH_LOGITS_RING_W =
              (EvolutionEngine.#SCRATCH_LOGITS_RING_W + 1) & 0x7fffffff;
          }
        }
      }
    } catch (logitsPostprocessError) {
      EvolutionEngine.#swallowError(logitsPostprocessError);
    }

    // Step 4: Optionally prune saturated outputs and emit telemetry (best-effort).
    try {
      if (
        simResult?.saturationFraction &&
        simResult.saturationFraction >
          EvolutionEngine.#SATURATION_PRUNE_THRESHOLD
      ) {
        pruneSaturatedHiddenOutputs(
          EvolutionEngine.#STATE,
          fittest,
          EvolutionEngine.#getNodeIndicesByType,
          EvolutionEngine.#collectHiddenToOutputConns,
        );
      }
    } catch (saturationPruneError) {
      EvolutionEngine.#swallowError(saturationPruneError);
    }

    try {
      if (
        !EvolutionEngine.#TELEMETRY_MINIMAL &&
        logEvery > 0 &&
        completedGenerations % logEvery === 0
      ) {
        EvolutionEngine.#logGenerationTelemetry(
          neat,
          fittest,
          simResult,
          completedGenerations,
          safeWrite,
        );
      }
    } catch (telemetryDispatchError) {
      EvolutionEngine.#swallowError(telemetryDispatchError);
    }

    const elapsed = doProfile
      ? readHighResolutionTime(EvolutionEngine.#STATE) - startTime
      : 0;
    return { generationResult: simResult, simTime: elapsed } as any;
  }

  /**
   * Runs the NEAT neuro-evolution process for an agent to solve a given ASCII maze.
   *
   * This is the core function of the `EvolutionEngine`. It sets up and runs the evolutionary
   * algorithm to train a population of neural networks. Each network acts as the "brain" for an
   * agent, controlling its movement through the maze from a start point 'S' to an exit 'E'.
   *
   * This hybrid approach, combining the global search of evolution with the local search of backpropagation,
   * can significantly accelerate learning and lead to more robust solutions.
   *
   * @param options - A comprehensive configuration object for the maze evolution process.
   * @returns A Promise that resolves with an object containing the best network found, its simulation result, and the final NEAT instance.
   */
  static async runMazeEvolution(options: IRunMazeEvolutionOptions) {
    // 1) Normalise and validate options (descriptive names, defaulting).
    const opts = normalizeRunOptions(
      options,
      (seed: number) => EvolutionEngine.setDeterministic(seed),
      (enabled: boolean) => {
        EvolutionEngine.#REDUCED_TELEMETRY = enabled;
      },
      (enabled: boolean) => {
        EvolutionEngine.#TELEMETRY_MINIMAL = enabled;
      },
      (disabled: boolean) => {
        EvolutionEngine.#DISABLE_BALDWIN = disabled;
      },
    );

    // 2) Prepare maze, encoded maps and fitness context. This reuses pooled buffers where possible.
    const {
      encodedMaze,
      startPosition,
      exitPosition,
      distanceMap,
      inputSize,
      outputSize,
      fitnessContext,
    } = prepareEnvironmentForRun(opts, EvolutionEngine.#STATE.scratch);

    // 3) Create and seed NEAT instance via a descriptive helper.
    const {
      neat,
      scratchPopClone,
      scratchSample,
    } = createAndSeedNeat(
      opts,
      inputSize,
      outputSize,
      fitnessContext,
      EvolutionEngine.#SCRATCH_POP_CLONE,
      EvolutionEngine.#SCRATCH_SAMPLE,
    );
    EvolutionEngine.#SCRATCH_POP_CLONE = scratchPopClone;
    EvolutionEngine.#SCRATCH_SAMPLE = scratchSample;

    // 4) Ensure internal scratch/pooling capacity is sufficient for the configured population & network sizes.
    ensureScratchCapacity(EvolutionEngine.#STATE, {
      populationSize: opts.popSize,
      inputSize,
      outputSize,
    });

    // 5) Lamarckian warm-start (pretrain generation 0) when training cases exist.
    const lamarckianTrainingSet = buildLamarckianTrainingSet(
      EvolutionEngine.#STATE,
      {
        TRAIN_OUT_PROB_HIGH: EvolutionEngine.#TRAIN_OUT_PROB_HIGH,
        TRAIN_OUT_PROB_LOW: EvolutionEngine.#TRAIN_OUT_PROB_LOW,
        PROGRESS_MEDIUM: EvolutionEngine.#PROGRESS_MEDIUM,
        PROGRESS_STRONG: EvolutionEngine.#PROGRESS_STRONG,
        PROGRESS_JUNCTION: EvolutionEngine.#PROGRESS_JUNCTION,
        PROGRESS_FOURWAY: EvolutionEngine.#PROGRESS_FOURWAY,
        PROGRESS_REGRESS: EvolutionEngine.#PROGRESS_REGRESS,
        PROGRESS_MIN_SIGNAL: EvolutionEngine.#PROGRESS_MIN_SIGNAL,
        PROGRESS_MILD_REGRESS: EvolutionEngine.#PROGRESS_MILD_REGRESS,
        DEFAULT_JITTER_PROB: EvolutionEngine.#DEFAULT_JITTER_PROB,
        AUGMENT_JITTER_BASE: EvolutionEngine.#AUGMENT_JITTER_BASE,
        AUGMENT_JITTER_RANGE: EvolutionEngine.#AUGMENT_JITTER_RANGE,
        AUGMENT_PROGRESS_JITTER_PROB: EvolutionEngine.#AUGMENT_PROGRESS_JITTER_PROB,
        AUGMENT_PROGRESS_DELTA_RANGE: EvolutionEngine.#AUGMENT_PROGRESS_DELTA_RANGE,
        AUGMENT_PROGRESS_DELTA_HALF: EvolutionEngine.#AUGMENT_PROGRESS_DELTA_HALF,
        RNG_PARAMETERS: EvolutionEngine.#RNG_PARAMETERS,
      },
    );
    warmStartPopulationIfNeeded(
      neat,
      lamarckianTrainingSet,
      EvolutionEngine.#STATE,
      (neatInstance, trainingSet) => {
        pretrainPopulationWarmStart(
          neatInstance,
          trainingSet,
          {
            PRETRAIN_MAX_ITER: EvolutionEngine.#PRETRAIN_MAX_ITER,
            PRETRAIN_BASE_ITER: EvolutionEngine.#PRETRAIN_BASE_ITER,
            DEFAULT_TRAIN_ERROR: EvolutionEngine.#DEFAULT_TRAIN_ERROR,
            DEFAULT_PRETRAIN_RATE: EvolutionEngine.#DEFAULT_PRETRAIN_RATE,
            DEFAULT_PRETRAIN_MOMENTUM: EvolutionEngine.#DEFAULT_PRETRAIN_MOMENTUM,
            DEFAULT_TRAIN_BATCH_SMALL: EvolutionEngine.#DEFAULT_TRAIN_BATCH_SMALL,
          },
          applyCompassWarmStart,
          centerOutputBiases,
        );
      },
    );

    // 6) Prepare loop helpers and run the full evolution loop inside a private helper.
    const loopHelpers = prepareLoopHelpers(opts, EvolutionEngine.#STATE.scratch);

    // Lightweight profiling (opt-in): set env ASCII_MAZE_PROFILE=1 to enable
    const doProfile = !!(
      typeof process !== 'undefined' &&
      typeof process.env !== 'undefined' &&
      process.env.ASCII_MAZE_PROFILE === '1'
    );

    const runResult = await EvolutionEngine.#runEvolutionLoop(
      neat,
      opts,
      lamarckianTrainingSet,
      encodedMaze,
      startPosition,
      exitPosition,
      distanceMap,
      loopHelpers,
      doProfile,
    );

    // Unpack results from the loop helper
    const {
      bestNetwork,
      bestResult,
      completedGenerations,
      totalEvolveMs,
      totalLamarckMs,
      totalSimMs,
    } = runResult;

    // Emit profiling summary when enabled (use loopHelpers.safeWrite to avoid duplicating writer resolution)
    if (doProfile && completedGenerations > 0) {
      emitProfileSummary(
        EvolutionEngine.#STATE,
        loopHelpers.safeWrite,
        completedGenerations,
        totalEvolveMs,
        totalLamarckMs,
        totalSimMs,
        isProfilingDetailsEnabled,
        getProfilingAccumulators,
      );
    }

    // Final return: best network, its simulation result, the NEAT instance and exit reason
    return {
      bestNetwork,
      bestResult,
      neat,
      exitReason: (bestResult as any)?.exitReason ?? 'incomplete',
    };
  }

  /**
   * Internal evolution loop that executes generations until a stop condition or cancellation.
   *
   * Behaviour & contract:
   *  - Runs generations in a resilient, best-effort manner; internal errors are swallowed
   *    so a single failure cannot abort the whole run.
   *  - When `doProfile` is truthy the loop accumulates timing into a pooled Float64Array
   *    to avoid per-iteration allocations. The pooled buffer is reused across calls.
   *  - The helper performs side-effects (dashboard updates, persistence) in a non-fatal
   *    fashion and yields to the host when requested via `helpers.flushToFrame`.
   *
   * Props / Parameters:
   * @param neat - NEAT driver instance used to perform evolution and mutation operations.
   * @param opts - Normalised run options (produced by `#normalizeRunOptions`).
   * @param lamarckianTrainingSet - Optional supervised training cases used for Lamarckian warm-start.
   * @param encodedMaze - Encoded maze representation consumed by simulators.
   * @param startPosition - Start coordinates for the simulated agent.
   * @param exitPosition - Exit coordinates for the simulated agent.
   * @param distanceMap - Optional precomputed distance map to speed simulation.
   * @param helpers - Helper utilities: { flushToFrame, fs, path, safeWrite }.
   * @param doProfile - When truthy collect and return millisecond timings in the result.
   *
   * @returns Promise resolving to an object:
   *  { bestNetwork, bestResult, neat, completedGenerations, totalEvolveMs, totalLamarckMs, totalSimMs }
   *  - `bestNetwork`: best genome/network found (may be undefined)
   *  - `bestResult`: simulation result associated with the best network
   *  - `completedGenerations`: number of generations executed
   *  - `totalEvolveMs`, `totalLamarckMs`, `totalSimMs`: accumulated timings (ms)
   *
   * @example
   * // Private helper invoked by `runMazeEvolution` (use bracket notation to access private method in tests)
   * const runSummary = await EvolutionEngine['#runEvolutionLoop'](
   *   neatDriver,
   *   opts,
   *   trainingCasesArray,
   *   encodedMaze,
   *   startPos,
   *   exitPos,
   *   distanceMap,
   *   { flushToFrame: () => Promise.resolve(), fs: null, path: null, safeWrite: console.log },
   *   true // doProfile
   * );
   * console.log(`completed generations: ${runSummary.completedGenerations}`);
   *
   * @internal
   */
  static async #runEvolutionLoop(
    neat: any,
    opts: any,
    lamarckianTrainingSet: any[],
    encodedMaze: any,
    startPosition: any,
    exitPosition: any,
    distanceMap: any,
    helpers: {
      flushToFrame: () => Promise<void>;
      fs: any;
      path: any;
      safeWrite: (msg: string) => void;
    },
    doProfile: boolean,
  ) {
    const { flushToFrame, fs, path, safeWrite } = helpers;

    /**
     * State: descriptive local names improve readability for future maintainers.
     */
    let bestNetworkSoFar: INetwork | undefined = opts.initialBestNetwork;
    let bestFitnessSoFar = -Infinity;
    let bestRunResult: any = undefined;
    let stagnantGenerationsCount = 0;
    let completedGenerations = 0;
    let plateauCounter = 0;
    let simplifyMode = false;
    let simplifyRemaining = 0;
    let lastBestFitnessForPlateau = -Infinity;
    let lastCompactionGeneration = 0;

    // Profiling accumulators are stored in a pooled Float64Array to avoid
    // per-run object creation. Layout: [0]=evolveMs, [1]=lamarckMs, [2]=simMs, [3]=reserved
    const scratchBundle = EvolutionEngine.#STATE.scratch;
    const profileScratch: Float64Array =
      scratchBundle.profilingScratch ??
      (scratchBundle.profilingScratch = new Float64Array(4));
    profileScratch[0] = 0; // total evolve ms
    profileScratch[1] = 0; // total lamarck ms
    profileScratch[2] = 0; // total sim ms

    // Main evolution loop: resilient and best-effort. Uses descriptive names
    // and keeps allocations to a minimum.
    while (true) {
      // Step 1: cooperative cancellation check (non-allocating, safe)
      const cancelReason = checkCancellation(opts, bestRunResult);
      if (cancelReason) break;

      // Step 2: perform one generation and collect per-stage timings when enabled
      const generationOutcome = await EvolutionEngine.#runGeneration(
        neat,
        doProfile,
        opts.lamarckianIterations,
        lamarckianTrainingSet,
        opts.lamarckianSampleSize,
        safeWrite,
        completedGenerations,
        opts.dynamicPopEnabled,
        opts.dynamicPopMax,
        opts.plateauGenerations,
        plateauCounter,
        opts.dynamicPopExpandInterval,
        opts.dynamicPopExpandFactor,
        opts.dynamicPopPlateauSlack,
      );

      const fittest = generationOutcome.fittest;
      if (doProfile) {
        // Use pooled scratch to accumulate totals (avoid creating new numbers/objects)
        profileScratch[0] += Number(generationOutcome.tEvolve ?? 0);
        profileScratch[1] += Number(generationOutcome.tLamarck ?? 0);
      }

      // Step 3: optional Lamarckian refinement (best-effort)
      if (!EvolutionEngine.#DISABLE_BALDWIN) {
        try {
          fittest.train(lamarckianTrainingSet, {
            iterations: EvolutionEngine.#FITTEST_TRAIN_ITERATIONS,
            error: EvolutionEngine.#DEFAULT_TRAIN_ERROR,
            rate: EvolutionEngine.#DEFAULT_TRAIN_RATE,
            momentum: EvolutionEngine.#DEFAULT_TRAIN_MOMENTUM,
            batchSize: EvolutionEngine.#DEFAULT_TRAIN_BATCH_LARGE,
            allowRecurrent: true,
          });
        } catch {
          // ignore training errors - non-fatal
        }
      }

      // Step 4: update per-generation counters and plateau/simplify state
      const fitnessScore = fittest.score ?? 0;
      completedGenerations += 1;

      ({ plateauCounter, lastBestFitnessForPlateau } = updatePlateauState(
        fitnessScore,
        lastBestFitnessForPlateau,
        plateauCounter,
        opts.plateauImprovementThreshold,
      ));

      ({ simplifyMode, simplifyRemaining, plateauCounter } = handleSimplifyState(
        EvolutionEngine.#STATE,
        neat,
        plateauCounter,
        opts.plateauGenerations,
        opts.simplifyDuration,
        simplifyMode,
        simplifyRemaining,
        opts.simplifyStrategy,
        opts.simplifyPruneFraction,
      ));

      // Step 5: simulate the fittest genome and optionally capture sim time
      const simulationResult = EvolutionEngine.#simulateAndPostprocess(
        fittest,
        encodedMaze,
        startPosition,
        exitPosition,
        distanceMap,
        opts.agentSimConfig?.maxSteps,
        doProfile,
        safeWrite,
        opts.reportingConfig?.logEvery ?? 10,
        completedGenerations,
        neat,
      );
      const generationResult = simulationResult.generationResult;
      if (doProfile) profileScratch[2] += Number(simulationResult.simTime ?? 0);

      // Step 6: update best-so-far and dashboard periodically
      if (fitnessScore > bestFitnessSoFar) {
        bestFitnessSoFar = fitnessScore;
        bestNetworkSoFar = fittest;
        bestRunResult = generationResult;
        stagnantGenerationsCount = 0;
        try {
          await updateDashboardAndMaybeFlush(
            opts.mazeConfig.maze,
            generationResult,
            fittest,
            completedGenerations,
            neat,
            opts.reportingConfig?.dashboardManager,
            flushToFrame,
          );
        } catch {
          // best-effort: ignore dashboard errors
        }
      } else {
        stagnantGenerationsCount += 1;
        if (
          completedGenerations % (opts.reportingConfig?.logEvery ?? 10) ===
          0
        ) {
          try {
            await updateDashboardPeriodic(
              opts.mazeConfig.maze,
              bestRunResult,
              bestNetworkSoFar,
              completedGenerations,
              neat,
              opts.reportingConfig?.dashboardManager,
              flushToFrame,
            );
          } catch {
            // best-effort
          }
        }
      }

      // Step 7: persist snapshot if configured (best-effort)
      persistSnapshotIfNeeded(
        EvolutionEngine.#STATE,
        fs,
        path,
        opts.persistDir,
        opts.persistTopK,
        completedGenerations,
        opts.persistEvery,
        neat,
        bestFitnessSoFar,
        simplifyMode,
        plateauCounter,
        EvolutionEngine.#SCRATCH_SNAPSHOT_OBJ,
        EvolutionEngine.#SCRATCH_SNAPSHOT_TOP,
        collectTelemetryTail,
        getSortedIndicesByScore,
        isProfilingDetailsEnabled,
        profilingStartTimestamp,
        accumulateProfilingDuration,
      );

      // Step 8: check stop conditions
      const stopReason = await checkStopConditions(
        bestRunResult,
        bestNetworkSoFar,
        opts.mazeConfig.maze,
        completedGenerations,
        neat,
        opts.reportingConfig?.dashboardManager,
        flushToFrame,
        opts.minProgressToPass,
        opts.autoPauseOnSolve,
        opts.stopOnlyOnSolve,
        stagnantGenerationsCount,
        opts.maxStagnantGenerations,
        opts.maxGenerations,
      );
      if (stopReason) break;

      // Step 9: periodic memory compaction and scratch shrinking
      if (
        opts.memoryCompactionInterval > 0 &&
        completedGenerations - lastCompactionGeneration >=
          opts.memoryCompactionInterval
      ) {
        const removedDisabled = compactPopulation(EvolutionEngine.#STATE, neat);
        if (removedDisabled > 0) {
          const currentPopulationSize = Array.isArray(neat?.population)
            ? neat.population.length
            : 0;
          maybeShrinkScratch(EvolutionEngine.#STATE, currentPopulationSize);
          safeWrite(
            `[COMPACT] gen=${completedGenerations} removedDisabledConns=${removedDisabled}\n`,
          );
        }
        lastCompactionGeneration = completedGenerations;
      }

      // Step 10: optionally yield to host between generations
      if (opts.reportingConfig?.paceEveryGeneration) {
        try {
          await flushToFrame();
        } catch {
          // ignore host-yield failures
        }
      }
    }

    // Prepare totals to return (read from pooled scratch to avoid ephemeral numbers earlier)
    const totalEvolveMs = Number(profileScratch[0] || 0);
    const totalLamarckMs = Number(profileScratch[1] || 0);
    const totalSimMs = Number(profileScratch[2] || 0);

    return {
      bestNetwork: bestNetworkSoFar,
      bestResult: bestRunResult,
      neat,
      completedGenerations,
      totalEvolveMs,
      totalLamarckMs,
      totalSimMs,
    } as any;
  }

  /**
   * Print a concise, human-readable summary of a network's topology and runtime metadata.
   *
   * This method is intentionally a thin orchestrator: heavy lifting is delegated to
   * small private helper methods so callers can quickly understand the high-level
   * structure without digging through implementation details.
   *
   * Props / Parameters:
   * @param network - The network (genome) to inspect. Expected shape: { nodes: any[], connections: any[] }.
   *
   * Returns: void (logs to the console). The function never throws and will tolerate
   * partially-formed network objects.
   *
   * Example:
   * // Print a neat summary of the best evolved network for debugging
   * EvolutionEngine.printNetworkStructure(bestNetwork);
   */
  static printNetworkStructure(network: INetwork) {
    // Orchestrator: gather lightweight facts and delegate formatting to helpers.
    try {
      console.log('Network Structure:');

      // Nodes classification
      const { nodeList, inputNodes, hiddenNodes, outputNodes } =
        EvolutionEngine.#classifyNodes(network);
      console.log('Nodes:', nodeList.length);
      console.log('  Input nodes:', inputNodes.length);
      console.log('  Hidden nodes:', hiddenNodes.length);
      console.log('  Output nodes:', outputNodes.length);

      // Activation function names (reuses SCRATCH_ACT_NAMES pool)
      const activationNames = EvolutionEngine.#gatherActivationNames(network);
      console.log('Activation functions:', activationNames);

      // Connections summary and recurrent/gated detection
      const connectionsList = Array.isArray(network?.connections)
        ? network.connections
        : [];
      console.log('Connections:', connectionsList.length);
      const hasRecurrentOrGated =
        EvolutionEngine.#detectRecurrentOrGated(connectionsList);
      console.log('Has recurrent/gated connections:', hasRecurrentOrGated);
    } catch (inspectError) {
      EvolutionEngine.#swallowError(inspectError);
      // Best-effort logging: swallow and surface a minimal message.
      // Avoid throwing from a debug helper.

      console.log(
        'printNetworkStructure: failed to inspect network (partial data)',
      );
    }
  }

  /**
   * Classify nodes into input / hidden / output buckets.
   *
   * Behavior & contract:
   *  - Allocation-light: returns references into the original node array (no cloning).
   *  - Tolerates missing network or sparse node arrays (holes preserved by skipping).
   *  - Reuses a small pooled buckets structure across calls to reduce per-call allocations.
   *
   * Props:
   * @param network - Network-like object with an optional `nodes` array.
   * @returns An object { nodeList, inputNodes, hiddenNodes, outputNodes } where each
   *          bucket is a (pooled) array referencing nodes from the original `nodes`.
   *
   * Example:
   * const { nodeList, inputNodes, hiddenNodes, outputNodes } = EvolutionEngine['#classifyNodes'](someNet);
   * console.log(`inputs=${inputNodes.length} hidden=${hiddenNodes.length} outputs=${outputNodes.length}`);
   */
  static #classifyNodes(network: INetwork) {
    // Orchestrator: normalize inputs then delegate to the fast, allocation-light classifier.
    const normalizedNodeList = EvolutionEngine.#normalizeNodesArray(network);
    return EvolutionEngine.#classifyNodesFromArray(normalizedNodeList);
  }

  /**
   * Normalize the incoming network into a safe node list reference.
   * Small helper to keep the main method focused on orchestration.
   */
  static #normalizeNodesArray(network: INetwork): any[] {
    // Fast-guard: accept only arrays, fall back to empty array when missing.
    return Array.isArray(network?.nodes) ? network.nodes : [];
  }

  /**
   * Classify a node array into input / hidden / output buckets using pooled arrays.
   *
   * Description:
   * Performs a single in-place pass over `nodesArray` and places references into
   * three pooled buckets attached to the class to avoid per-call allocations.
   * The returned buckets are backed by pooled arrays and are reused by subsequent
   * callers; do not mutate them if you intend to reuse the engine pools.
   *
   * Implementation details:
   * - The pooled buckets live on the shared engine state and are lazily initialised.
   * - Buckets are cleared by setting `.length = 0` which preserves allocated capacity.
   * - The method is allocation-light and suitable for hot paths.
   *
   * @internal
   * @param nodesArray - Array-like collection of node objects (each node may have a `type` property).
   * @returns An object with the following properties:
   *  - `{ nodeList }` - The original (normalized) node array reference used for classification.
   *  - `{ inputNodes }` - Pooled array containing all nodes whose `type` is `'input'`.
   *  - `{ hiddenNodes }` - Pooled array containing all non-input/non-output nodes.
   *  - `{ outputNodes }` - Pooled array containing all nodes whose `type` is `'output'`.
   *
   * @example
   * // Use for quick inspection without allocating new arrays per-call
   * const { nodeList, inputNodes, hiddenNodes, outputNodes } = EvolutionEngine['#classifyNodesFromArray'](net.nodes || []);
   * console.log(`inputs=${inputNodes.length} hidden=${hiddenNodes.length} outputs=${outputNodes.length}`);
   */
  static #classifyNodesFromArray(nodesArray: any[]): {
    nodeList: any[];
    inputNodes: any[];
    hiddenNodes: any[];
    outputNodes: any[];
  } {
    // Step 1: Normalise the incoming node list to a safe, non-null array reference.
    const nodeList: any[] = Array.isArray(nodesArray) ? nodesArray : [];

    // Step 2: Lazily create / reuse the pooled buckets structure on the class.
    const scratchBundle = EvolutionEngine.#STATE.scratch;
    let pooledBuckets = scratchBundle.nodeBuckets;
    if (!Array.isArray(pooledBuckets?.[0])) {
      pooledBuckets = scratchBundle.nodeBuckets = [[], [], []];
    }

    // Descriptive bucket aliases for readability.
    const inputBucket = pooledBuckets[0];
    const hiddenBucket = pooledBuckets[1];
    const outputBucket = pooledBuckets[2];

    // Step 3: Clear buckets in-place (cheap; preserves allocated capacity where possible).
    inputBucket.length = 0;
    hiddenBucket.length = 0;
    outputBucket.length = 0;

    // Step 4: Single-pass classification. Keep the loop small and optimiser-friendly.
    for (let nodeIndex = 0; nodeIndex < nodeList.length; nodeIndex++) {
      const node = nodeList[nodeIndex];
      // Tolerate holes and malformed entries quickly.
      if (!node) continue;

      // Normalise the node type to a stable string and classify deterministically.
      const nodeType = String(node.type ?? 'hidden');
      if (nodeType === 'input') {
        inputBucket.push(node);
      } else if (nodeType === 'output') {
        outputBucket.push(node);
      } else {
        // Treat everything else as hidden (includes undefined/custom types).
        hiddenBucket.push(node);
      }
    }

    // Step 5: Return references (note: returned arrays are pooled and reused by subsequent callers).
    return {
      nodeList,
      inputNodes: inputBucket,
      hiddenNodes: hiddenBucket,
      outputNodes: outputBucket,
    };
  }

  /**
   * Populate and return a pooled array of activation (squash) function names for `network.nodes`.
   *
   * Behaviour & contract:
   * - Reuses a private pooled string array (`#SCRATCH_ACT_NAMES`) to avoid per-call allocations.
   * - Grows the pool capacity using a power-of-two strategy (next power-of-two) to reduce resize frequency.
   * - Fills the pooled array with readable names and trims `.length` to the exact node count before returning.
   *
   * Steps (high level):
   * 1) Normalise the incoming node list to a safe array reference.
   * 2) Lazily create the shared pooled names array when first used.
   * 3) Grow the pooled array to a power-of-two capacity when current capacity is insufficient.
   * 4) Populate the used prefix with function `name` when available or a stable string fallback.
   * 5) Trim the pooled array to `nodesCount` and return it (note: the returned array is reused; callers must not mutate it).
   *
   * @internal
   * @param network Network-like object with an optional `nodes` array.
   * @returns Pooled array of activation function names (length === number of nodes).
   * @example
   * // Obtain activation names for quick inspection without allocating a new array each call
   * const names = EvolutionEngine['#gatherActivationNames'](someNet);
   * console.log(names.join(','));
   */
  static #gatherActivationNames(network: INetwork): string[] {
    // Step 1: Safe normalisation of the node list reference.
    const nodesArray: any[] = Array.isArray(network?.nodes)
      ? network.nodes
      : [];
    const nodesCount = nodesArray.length;

    // Step 2: Lazily ensure the shared pool exists. Use a cast to bypass private-field creation quirks.
    const scratchBundle = EvolutionEngine.#STATE.scratch;
    if (!Array.isArray(scratchBundle.activationNameBuffer)) {
      scratchBundle.activationNameBuffer = [];
    }

    const pooledNames: string[] = scratchBundle.activationNameBuffer;

    // Helper: compute next power-of-two for growth (keeps growth jumps friendly to the allocator).
    const nextPowerOfTwo = (value: number): number => {
      let power = 1;
      while (power < value) power <<= 1;
      return power;
    };

    // Step 3: Grow pooled capacity to the next power-of-two when necessary.
    if (pooledNames.length < nodesCount) {
      const targetCapacity = nextPowerOfTwo(Math.max(1, nodesCount));
      pooledNames.length = targetCapacity;
    }

    // Step 4: Populate the used prefix with readable names.
    for (let idx = 0; idx < nodesCount; idx++) {
      const nodeEntry = nodesArray[idx];
      const squashCandidate = nodeEntry?.squash;

      // Prefer explicit function name when available; fall back to a stable string.
      if (typeof squashCandidate === 'function') {
        // Some anonymous functions may have an empty .name; normalise to 'anonymous' then.
        pooledNames[idx] =
          squashCandidate.name && squashCandidate.name.length
            ? squashCandidate.name
            : 'anonymous';
      } else {
        pooledNames[idx] = String(squashCandidate ?? 'unknown');
      }
    }

    // Step 5: Trim to exact length for consumer readability (non-allocating when shrinking a pre-sized array).
    pooledNames.length = nodesCount;
    return pooledNames;
  }

  /**
   * Fast, allocation-aware detector for recurrent or gated connections.
   *
   * Steps:
   * 1) Fast-guard invalid inputs (non-array / empty -> false).
   * 2) For small connection lists use a plain loop (lowest overhead).
   * 3) For large lists rent a pooled Int8Array via `ensureConnFlagsCapacity` and
   *    reuse it as a tiny scratch bitmap to reduce allocations and improve cache locality.
   * 4) Early-return on first detection (gated or recurrent), otherwise mark seen indices
   *    in the scratch buffer and return false when complete.
   * 5) Defensive fallback: if the pooled allocation fails, revert to the plain loop.
   *
   * @param connectionsList Array of connection-like objects with optional `from`, `to`, `gater`.
   * @returns true when any connection is recurrent (from === to) or gated (gater truthy), false otherwise.
   * @example
   * const hasSpecial = EvolutionEngine['#detectRecurrentOrGated'](network.connections);
   */
  static #detectRecurrentOrGated(connectionsList: any[]): boolean {
    // Step 1: Validate input quickly
    if (!Array.isArray(connectionsList) || connectionsList.length === 0)
      return false;

    const connectionCount = connectionsList.length;
    const SMALL_LIST_THRESHOLD = 128; // tuned threshold for typed-array trade-off

    // Step 2: Small-list fast path: direct inspection avoids typed-array overhead
    if (connectionCount < SMALL_LIST_THRESHOLD) {
      for (let i = 0; i < connectionCount; i++) {
        const connection = connectionsList[i];
        if (!connection) continue; // tolerate sparse arrays
        if (connection.gater) return true; // gated connection detected
        if (connection.from === connection.to) return true; // recurrent self-connection
      }
      return false;
    }

    // Step 3: Large-list path: attempt to rent a pooled Int8Array for scratch flags
    try {
      const scratchFlags = ensureConnFlagsCapacity(
        EvolutionEngine.#STATE,
        connectionCount,
      );

      // Step 5: Fallback to plain loop when pool allocation fails
      if (!scratchFlags) {
        for (let i = 0; i < connectionCount; i++) {
          const connection = connectionsList[i];
          if (!connection) continue;
          if (connection.gater || connection.from === connection.to)
            return true;
        }
        return false;
      }

      // Initialize only the used prefix for deterministic behavior (cheap for Int8Array)
      scratchFlags.fill(0, 0, connectionCount);

      // Step 4: Iterate, early-return on detection, and mark seen indices in scratch buffer
      for (let i = 0; i < connectionCount; i++) {
        const connection = connectionsList[i];
        if (!connection) continue;
        if (connection.gater) return true;
        if (connection.from === connection.to) return true;
        scratchFlags[i] = 1; // mark index as visited in the pooled scratch
      }

      return false;
    } catch {
      // Robust degradation: on any runtime error, use the safe plain loop.
      for (let i = 0; i < connectionCount; i++) {
        const connection = connectionsList[i];
        if (!connection) continue;
        if (connection.gater || connection.from === connection.to) return true;
      }
      return false;
    }
  }

  /**
   * Ensure the pooled connection-flag Int8Array has at least `minCapacity` entries.
   * Returns the pooled buffer or `null` when allocation fails.
   *
   * Implementation details / contract:
   * - Reuses an engine-level pooled Int8Array stored in the shared scratch state to avoid
   *   repeated allocations when analyzing large connection lists.
   * - Grows the pooled buffer lazily using a power-of-two strategy (nextPow2) to
   *   keep resize frequency low and preserve cache-friendliness.
   * - When growing, copies the preserved prefix (old length) into the new buffer so
   *   callers can rely on stable scratch contents across resizes.
   * - Defensive guards ensure we only accept sensible integer capacities and return
   *   `null` when memory allocation fails.
   *
   * @param minCapacity Minimum required capacity (integer >= 0).
   * @returns The pooled Int8Array with capacity >= `minCapacity`, or `null` if allocation failed.
   * @example
   * const flags = ensureConnFlagsCapacity(EvolutionEngine.sharedState, 1024);
   * if (flags) { // use flags as temporary Int8Array
   *   // use flags as temporary Int8Array
   * }
   */

  /** Utility to explicitly mark swallowed errors for lint compliance. */
  static #swallowError(error: unknown): void {
    void error;
  }
}
