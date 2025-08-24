// Handles the main NEAT evolution loop for maze solving
// Exports: EvolutionEngine class with static methods

import { Neat, Network, methods } from '../../../src/neataptic';
import { MazeUtils } from './mazeUtils';
import { MazeMovement } from './mazeMovement';
import { FitnessEvaluator } from './fitness';
import {
  INetwork,
  IFitnessEvaluationContext,
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
  /**
   * Pooled scratch buffer used by telemetry softmax/entropy calculations.
   * @remarks Non-reentrant: telemetry functions that use this buffer must not be
   * called concurrently (single-threaded runtime assumption holds for Node/browser).
   */
  static #SCRATCH_EXPS = new Float64Array(4);
  /** Pooled stats buffers (always resident) for means & stds. */
  static #SCRATCH_MEANS = new Float64Array(4);
  static #SCRATCH_STDS = new Float64Array(4);
  /** Kurtosis related buffers allocated lazily when first needed (non-reduced telemetry). */
  static #SCRATCH_KURT: Float64Array | undefined;
  static #SCRATCH_M2_RAW = new Float64Array(4);
  static #SCRATCH_M3_RAW: Float64Array | undefined;
  static #SCRATCH_M4_RAW: Float64Array | undefined;
  /**
   * Small integer scratch buffer used for directional move counts (N,E,S,W).
   * @remarks Non-reentrant: reused across telemetry calls.
   */
  static #SCRATCH_COUNTS = new Int32Array(4);
  /**
   * Open-address hash table for visited coordinate detection (pairs packed into 32-bit int).
   * Length is always a power of two; uses linear probing. A value of 0 represents EMPTY so we offset packed values by +1.
   */
  static #SCRATCH_VISITED_HASH = new Int32Array(0);
  /** Load factor threshold (~0.7) for resizing visited hash. */
  static #VISITED_HASH_LOAD = 0.7;
  /** Knuth multiplicative hashing constant (32-bit golden ratio). */
  static #HASH_KNUTH_32 = 2654435761 >>> 0;
  /** Scratch species id buffer (dynamic growth). */
  static #SCRATCH_SPECIES_IDS = new Int32Array(64);
  /** Scratch species count buffer parallel to ids. */
  static #SCRATCH_SPECIES_COUNTS = new Int32Array(64);
  /** Reusable candidate connection object buffer. */
  static #SCRATCH_CONN_CAND: any[] = [];
  /** Reusable hidden->output connection buffer. */
  static #SCRATCH_HIDDEN_OUT: any[] = [];
  /** Flags buffer for connection disabling (grown on demand). */
  static #SCRATCH_CONN_FLAGS = new Uint8Array(128);
  /** Scratch tail buffer reused by #getTail (grows geometrically). */
  static #SCRATCH_TAIL: any[] = new Array(64);
  /** Scratch sample result buffer reused by #sampleArray (ephemeral return). */
  static #SCRATCH_SAMPLE_RESULT: any[] = new Array(64);
  /** Scratch index buffer holding sorted indices by score (reused per generation). */
  static #SCRATCH_SORT_IDX: number[] = new Array(512);
  /** Scratch stack (lo,hi pairs) for quicksort on indices. */
  static #SCRATCH_QS_STACK = new Int32Array(128);
  /** Scratch array reused when cloning an initial population. */
  static #SCRATCH_POP_CLONE: any[] = new Array(0);
  /** Scratch string array for activation function names (printNetworkStructure). */
  static #SCRATCH_ACT_NAMES: string[] = new Array(0);
  /** Reusable object buffer for snapshot top entries. */
  static #SCRATCH_SNAPSHOT_TOP: any[] = new Array(0);
  /** Reusable snapshot object (fields overwritten each persistence). */
  static #SCRATCH_SNAPSHOT_OBJ: any = {
    generation: 0,
    bestFitness: 0,
    simplifyMode: false,
    plateauCounter: 0,
    timestamp: 0,
    telemetryTail: undefined,
    top: undefined,
  };
  /** Pooled buffer for mutation operator indices (shuffled prefix each use). */
  static #SCRATCH_MUTOP_IDX = new Uint16Array(0);
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
  static #SCRATCH_LOGITS_RING: Float32Array[] = (() => {
    const cap = 512;
    const rows: Float32Array[] = new Array(cap);
    for (let i = 0; i < cap; i++)
      rows[i] = new Float32Array(EvolutionEngine.#ACTION_DIM);
    return rows;
  })();
  /** Shared flat logits storage when shared mode enabled (length = cap * ACTION_DIM). */
  static #SCRATCH_LOGITS_SHARED: Float32Array | undefined;
  /** Shared atomic write index (length=1 Int32). */
  static #SCRATCH_LOGITS_SHARED_W: Int32Array | undefined;
  /** Write cursor for non-shared ring. */
  static #SCRATCH_LOGITS_RING_W = 0;
  /** Internal helper: allocate a non-shared ring with specified capacity. */
  static #allocateLogitsRing(cap: number): Float32Array[] {
    const rows: Float32Array[] = new Array(cap);
    for (let i = 0; i < cap; i++)
      rows[i] = new Float32Array(EvolutionEngine.#ACTION_DIM);
    return rows;
  }
  /** Attempt to initialize SharedArrayBuffer-backed ring if environment isolated (COOP+COEP). */
  static #initSharedLogitsRing(cap: number) {
    try {
      if (typeof SharedArrayBuffer === 'undefined') return;
      if ((globalThis as any).crossOriginIsolated !== true) return; // must be true in browsers
      const actionDim = EvolutionEngine.#ACTION_DIM;
      const totalFloats = cap * actionDim;
      const sab = new SharedArrayBuffer(4 + totalFloats * 4);
      EvolutionEngine.#SCRATCH_LOGITS_SHARED_W = new Int32Array(sab, 0, 1);
      EvolutionEngine.#SCRATCH_LOGITS_SHARED = new Float32Array(
        sab,
        4,
        totalFloats
      );
      Atomics.store(EvolutionEngine.#SCRATCH_LOGITS_SHARED_W, 0, 0);
      EvolutionEngine.#LOGITS_RING_SHARED = true;
    } catch {
      EvolutionEngine.#LOGITS_RING_SHARED = false;
      EvolutionEngine.#SCRATCH_LOGITS_SHARED = undefined;
      EvolutionEngine.#SCRATCH_LOGITS_SHARED_W = undefined;
    }
  }
  /** Ensure ring has capacity for desired recent steps (grow/shrink heuristics). */
  static #ensureLogitsRingCapacity(desiredRecentSteps: number) {
    let cap = EvolutionEngine.#LOGITS_RING_CAP;
    let target = cap;
    if (
      desiredRecentSteps > (cap * 3) / 4 &&
      cap < EvolutionEngine.#LOGITS_RING_CAP_MAX
    ) {
      // grow: next pow2 >= desired*2
      let next = 1;
      while (
        next < desiredRecentSteps * 2 &&
        next < EvolutionEngine.#LOGITS_RING_CAP_MAX
      )
        next <<= 1;
      target = Math.min(next, EvolutionEngine.#LOGITS_RING_CAP_MAX);
    } else if (desiredRecentSteps < cap / 4 && cap > 128) {
      // shrink while still leaving 2x headroom
      let shrink = cap;
      while (shrink > 128 && desiredRecentSteps * 2 <= shrink / 2) shrink >>= 1;
      target = Math.max(shrink, 128);
    }
    if (target !== cap) {
      EvolutionEngine.#LOGITS_RING_CAP = target;
      EvolutionEngine.#SCRATCH_LOGITS_RING_W = 0;
      EvolutionEngine.#SCRATCH_LOGITS_RING = EvolutionEngine.#allocateLogitsRing(
        target
      );
      if (EvolutionEngine.#LOGITS_RING_SHARED)
        EvolutionEngine.#initSharedLogitsRing(target);
    }
  }
  /**
   * Small node index scratch arrays reused when extracting nodes by type.
   * @remarks Non-reentrant: do not call concurrently.
   */
  static #SCRATCH_NODE_IDX = new Int32Array(64);
  /**
   * Object reference scratch array used as a short sample buffer (max 40 entries).
   * Avoids allocating small arrays inside hot telemetry paths.
   */
  static #SCRATCH_SAMPLE: any[] = new Array(40);
  /** Reusable string assembly character buffer for small joins (grown geometrically). */
  static #SCRATCH_STR: string[] = new Array(64);
  /** Internal 32-bit state for fast LCG RNG (mul 1664525 + 1013904223). */
  static #RNG_STATE = (Date.now() ^ 0x9e3779b9) >>> 0;
  /** Detailed profiling enable flag (set ASCII_MAZE_PROFILE_DETAILS=1). */
  static #PROFILE_ENABLED = (() => {
    try {
      return (
        typeof process !== 'undefined' &&
        process?.env?.ASCII_MAZE_PROFILE_DETAILS === '1'
      );
    } catch {
      return false;
    }
  })();
  /** Accumulators for detailed profiling (ms). */
  static #PROFILE_ACCUM: Record<string, number> = {
    telemetry: 0,
    simplify: 0,
    snapshot: 0,
    prune: 0,
  };
  /** Small fixed-size visited table for tiny path exploration (<32) to avoid O(n^2) duplicate scan. */
  static #SMALL_EXPLORE_TABLE = new Int32Array(64);
  /** Bit mask for SMALL_EXPLORE_TABLE indices (table length - 1). */
  static #SMALL_EXPLORE_TABLE_MASK = 64 - 1;
  static #PROFILE_T0(): number {
    return EvolutionEngine.#now();
  }
  static #PROFILE_ADD(key: string, delta: number) {
    if (!EvolutionEngine.#PROFILE_ENABLED) return;
    EvolutionEngine.#PROFILE_ACCUM[key] =
      (EvolutionEngine.#PROFILE_ACCUM[key] || 0) + delta;
  }
  /** RNG cache (batched 4 draws) to amortize state writes in tight loops. */
  static #RNG_CACHE = new Float64Array(4);
  static #RNG_CACHE_INDEX = 4; // force initial refill
  /** Fast LCG producing float in [0,1). Non-crypto. Uses 4-value batch cache. @internal */
  static #fastRandom(): number {
    if (EvolutionEngine.#RNG_CACHE_INDEX >= 4) {
      let state = EvolutionEngine.#RNG_STATE >>> 0;
      for (let fillIndex = 0; fillIndex < 4; fillIndex++) {
        state = (state * 1664525 + 1013904223) >>> 0;
        EvolutionEngine.#RNG_CACHE[fillIndex] = (state >>> 9) * (1 / 0x800000);
      }
      EvolutionEngine.#RNG_STATE = state >>> 0;
      EvolutionEngine.#RNG_CACHE_INDEX = 0;
    }
    return EvolutionEngine.#RNG_CACHE[EvolutionEngine.#RNG_CACHE_INDEX++];
  }
  /** Deterministic mode flag (enables reproducible seeded RNG). */
  static #DETERMINISTIC = false;
  /** High-resolution time helper. */
  static #now(): number {
    return globalThis.performance?.now?.() ?? Date.now();
  }
  /**
   * Enable deterministic mode and optionally re-seed RNG.
   * @param seed Optional 32-bit seed (unsigned). Zero is remapped to a non-zero constant.
   */
  static setDeterministic(seed?: number): void {
    EvolutionEngine.#DETERMINISTIC = true;
    if (typeof seed === 'number' && Number.isFinite(seed)) {
      const s = seed >>> 0 || 0x9e3779b9;
      EvolutionEngine.#RNG_STATE = s;
      EvolutionEngine.#RNG_CACHE_INDEX = 4; // force refill
    }
  }
  /** Disable deterministic mode. */
  static clearDeterministic(): void {
    EvolutionEngine.#DETERMINISTIC = false;
  }
  /** When true, telemetry skips higher-moment stats (kurtosis) for speed. */
  static #REDUCED_TELEMETRY = false;
  /** Skip most telemetry logging & higher moment stats when true (minimal mode). */
  static #TELEMETRY_MINIMAL = false;
  /** Disable Baldwinian refinement phase when true. */
  static #DISABLE_BALDWIN = false;
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
   * Populate internal node index scratch with indices of nodes of given type.
   * Returns the count of matching nodes.
   * @internal
   */
  static #getNodeIndicesByType(nodes: any[] | undefined, type: string): number {
    if (!nodes || !nodes.length) return 0;
    let count = 0;
    for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
      const node = nodes[nodeIndex];
      if (node && node.type === type) {
        if (count >= EvolutionEngine.#SCRATCH_NODE_IDX.length) {
          const nextSize = 1 << Math.ceil(Math.log2(count + 1));
          const grown = new Int32Array(nextSize);
          grown.set(EvolutionEngine.#SCRATCH_NODE_IDX);
          EvolutionEngine.#SCRATCH_NODE_IDX = grown;
        }
        EvolutionEngine.#SCRATCH_NODE_IDX[count++] = nodeIndex;
      }
    }
    return count;
  }

  /**
   * Compute exploration statistics for a path taken by an agent.
   *
   * Returned metrics:
   * - unique: number of distinct grid cells visited
   * - pathLen: total number of steps in path (array length)
   * - ratio: unique / pathLen (0 when pathLen == 0)
   *
   * Implementation details:
   * 1. Coordinates are packed into a single 32-bit integer: (x & 0xffff) << 16 | (y & 0xffff)
   * 2. Two code paths:
   *    a. Tiny path (< 32): use a fixed 64-slot Int32Array (#SMALL_EXPLORE_TABLE) cleared per call.
   *    b. Larger path: use a growable power-of-two scratch Int32Array (#SCRATCH_VISITED_HASH) sized
   *       for target load factor (#VISITED_HASH_LOAD). Table is zero-filled when reused.
   * 3. Open-addressing with linear probing; 0 denotes empty. We store packed+1 to disambiguate zero.
   * 4. Knuth multiplicative hashing constant extracted as a private static (#HASH_KNUTH_32) for clarity.
   * 5. All operations avoid heap allocation after initial buffer growth.
   *
   * Complexity: O(n) expected; worst-case O(n * alpha) with alpha ~ probe length (low under target load).
   * Determinism: Fully deterministic given identical path contents.
   * Reentrancy: Non-reentrant due to shared hash buffer.
   *
   * @param path - Array of [x,y] coordinates visited in order.
   * @returns { unique, pathLen, ratio }
   * @example
   * const stats = EvolutionEngine['#computeExplorationStats']([[0,0],[1,0],[1,0]]); // pseudo internal usage
   * @internal
   */
  static #computeExplorationStats(
    path: ReadonlyArray<[number, number]>
  ): { unique: number; pathLen: number; ratio: number } {
    const pathLength = path?.length || 0;
    if (pathLength === 0) return { unique: 0, pathLen: 0, ratio: 0 };

    // Fast tiny-path path (<32) using small fixed table (size 64, mask 63).
    if (pathLength < 32) {
      const distinctTiny = EvolutionEngine.#countDistinctCoordinatesTiny(
        path,
        pathLength
      );
      return {
        unique: distinctTiny,
        pathLen: pathLength,
        ratio: distinctTiny / pathLength,
      };
    }

    const distinct = EvolutionEngine.#countDistinctCoordinatesHashed(
      path,
      pathLength
    );
    return {
      unique: distinct,
      pathLen: pathLength,
      ratio: distinct / pathLength,
    };
  }

  /**
   * Count distinct packed coordinates for tiny paths (< 32) using a fixed-size table.
   * @param path - Coordinate path reference.
   * @param pathLength - Precomputed length (avoid repeated length lookups).
   * @returns number of unique coordinates.
   * @remarks Non-reentrant (shared tiny table). O(n) expected, tiny constant factors.
   */
  static #countDistinctCoordinatesTiny(
    path: ReadonlyArray<[number, number]>,
    pathLength: number
  ): number {
    /**
     * Fast path for very small collections of coordinates.
     * Rationale: A 64-slot table fits comfortably in L1; clearing via fill(0) is cheaper
     * than maintaining a generation-tag array. Alternative tagging was evaluated but
     * rejected for clarity and negligible gain at this scale.
     */
    if (pathLength === 0) return 0;
    const table = EvolutionEngine.#SMALL_EXPLORE_TABLE;
    table.fill(0); // predictable, tiny cost
    let distinctCount = 0;
    const mask = EvolutionEngine.#SMALL_EXPLORE_TABLE_MASK; // 63
    for (
      let coordinateIndex = 0;
      coordinateIndex < pathLength;
      coordinateIndex++
    ) {
      const coordinate = path[coordinateIndex];
      // Pack (x,y) into 32 bits: high 16 = x, low 16 = y (wrapping via & 0xffff) to avoid negative hashing issues.
      const packed =
        ((coordinate[0] & 0xffff) << 16) | (coordinate[1] & 0xffff);
      // Multiplicative hash spreads lower entropy; then linear probe.
      let hash = Math.imul(packed, EvolutionEngine.#HASH_KNUTH_32) >>> 0;
      const storedValue = (packed + 1) | 0; // +1 so 0 remains EMPTY sentinel
      while (true) {
        const slot = hash & mask;
        const slotValue = table[slot];
        if (slotValue === 0) {
          // empty slot -> insert
          table[slot] = storedValue;
          distinctCount++;
          break;
        }
        if (slotValue === storedValue) break; // already recorded coordinate
        hash = (hash + 1) | 0; // linear probe step
      }
    }
    return distinctCount;
  }

  /**
   * Count distinct coordinates for larger paths using a dynamically sized open-address hash table.
   * Ensures the table grows geometrically to keep probe chains short.
   *
   * Design rationale:
   * - Uses a single shared Int32Array (#SCRATCH_VISITED_HASH) grown geometrically (power-of-two) so
   *   subsequent calls amortize allocation costs. We target a load factor below `#VISITED_HASH_LOAD` (≈0.7).
   * - Multiplicative hashing (Knuth constant) provides a cheap, decent dispersion for packed 32-bit coordinates.
   * - Linear probing keeps memory contiguous (cache friendly) and avoids per-slot pointer overhead.
   * - We pack (x,y) into 32 bits allowing negative coordinates via masking (& 0xffff) with natural wrap —
   *   acceptable for maze sizes far below 65k.
   * - Instead of tombstones we only ever insert during this scan, so deletion complexity is avoided.
   *
   * Complexity: Expected O(n) with short probe sequences; worst-case O(n^2) only under adversarial clustering
   * (not expected with maze coordinate distributions and resizing policy).
   * Memory: One Int32Array reused; size grows but never shrinks (acceptable for long-running evolution sessions).
   * Determinism: Fully deterministic for the same input path.
   * Reentrancy: Non-reentrant due to shared buffer reuse.
   *
   * @param path - Coordinate path reference.
   * @param pathLength - Path length (number of coordinate entries).
   * @returns Number of unique coordinates encountered.
   * @remarks Non-reentrant (shared hash table). Expected O(n) with low probe lengths.
   */
  static #countDistinctCoordinatesHashed(
    path: ReadonlyArray<[number, number]>,
    pathLength: number
  ): number {
    // Step 1: Compute target raw capacity (2x pathLength) to keep effective load factor low after inserts.
    const targetCapacity = pathLength << 1; // aim ~0.5 raw load pre-threshold
    // Step 2: Acquire / resize shared table if needed under load factor heuristic.
    let table = EvolutionEngine.#SCRATCH_VISITED_HASH;
    if (
      table.length === 0 ||
      targetCapacity > table.length * EvolutionEngine.#VISITED_HASH_LOAD
    ) {
      const needed = Math.ceil(
        targetCapacity / EvolutionEngine.#VISITED_HASH_LOAD
      );
      const pow2 = 1 << Math.ceil(Math.log2(needed));
      table = EvolutionEngine.#SCRATCH_VISITED_HASH = new Int32Array(pow2);
    } else {
      table.fill(0);
    }
    const mask = table.length - 1;
    let distinct = 0;
    // Step 3: Iterate coordinates, pack & insert into open-address table.
    for (let index = 0; index < pathLength; index++) {
      const coordinate = path[index];
      // Step 3.1: Pack (x,y) into 32 bits (high 16 bits: x, low 16 bits: y) with masking for wrap safety.
      const packed =
        ((coordinate[0] & 0xffff) << 16) | (coordinate[1] & 0xffff);
      // Step 3.2: Derive initial hash via multiplicative method; shift to unsigned domain.
      let hash = Math.imul(packed, EvolutionEngine.#HASH_KNUTH_32) >>> 0;
      // Step 3.3: Offset stored value by +1 so zero remains EMPTY sentinel.
      const storeVal = (packed + 1) | 0;
      // Step 3.4: Probe until empty slot (insert) or duplicate found (ignore).
      while (true) {
        const slot = hash & mask;
        const slotValue = table[slot];
        if (slotValue === 0) {
          table[slot] = storeVal; // insert
          distinct++;
          break;
        }
        if (slotValue === storeVal) break; // duplicate coordinate encountered
        hash = (hash + 1) | 0; // linear probe advance
      }
    }
    // Step 4: Return distinct count.
    return distinct;
  }

  /**
   * Compute core diversity metrics for the current NEAT population.
   *
   * Metrics returned:
   * - speciesUniqueCount: number of distinct species IDs present (genomes without a species get id -1)
   * - simpson: Simpson diversity index (1 - sum(p_i^2)) over species proportions
   * - wStd: standard deviation of enabled connection weights from a sampled subset of genomes
   *
   * Implementation notes / performance:
   * 1. Species counting uses two class‑static Int32Array scratch buffers (#SCRATCH_SPECIES_IDS / COUNTS)
   *    and linear search because typical species cardinality << population size, making a hash map slower.
   * 2. Weight distribution sampling uses a pooled object scratch buffer (#SCRATCH_SAMPLE) filled by
   *    #sampleIntoScratch (with replacement) to avoid per‑call allocations.
   * 3. Variance is computed with Welford's single‑pass algorithm for numerical stability and zero extra storage.
   * 4. All buffers grow geometrically (power of two) and never shrink to preserve amortized O(1) reallocation cost.
   *
   * Determinism: Sampling relies on the engine RNG; in deterministic mode (#DETERMINISTIC set) results are reproducible.
   * Reentrancy: Non‑reentrant due to shared scratch buffers.
   * Complexity: O(P + S + W) where P = population length, S = speciesUniqueCount (S <= P),
   *             and W = total enabled connections visited in the sampled genomes.
   * Memory: No per‑invocation heap allocations after initial buffer growth.
   *
   * @param neat - NEAT instance exposing a `population` array.
   * @param sampleSize - Upper bound on number of genomes to sample for weight variance (default 40).
   * @returns Object containing { speciesUniqueCount, simpson, wStd }.
   * @example
   * const diversity = EvolutionEngine['#computeDiversityMetrics'](neat, 32); // internal call example (pseudo)
   * @internal
   * @remarks Non-reentrant (shared scratch arrays).
   */
  static #computeDiversityMetrics(neat: any, sampleSize = 40) {
    /**
     * Step 0: Acquire population reference & early-exit for empty populations.
     * Using a local constant helps engines keep the reference monomorphic.
     */
    const populationRef: any[] = neat.population || [];
    const populationLength = populationRef.length | 0;
    if (populationLength === 0) {
      return { speciesUniqueCount: 0, simpson: 0, wStd: 0 };
    }

    // Step 1: Ensure species scratch buffers are large enough (power-of-two growth strategy).
    let speciesIds = EvolutionEngine.#SCRATCH_SPECIES_IDS;
    let speciesCounts = EvolutionEngine.#SCRATCH_SPECIES_COUNTS;
    if (populationLength > speciesIds.length) {
      const nextSize = 1 << Math.ceil(Math.log2(populationLength));
      EvolutionEngine.#SCRATCH_SPECIES_IDS = new Int32Array(nextSize);
      EvolutionEngine.#SCRATCH_SPECIES_COUNTS = new Int32Array(nextSize);
      speciesIds = EvolutionEngine.#SCRATCH_SPECIES_IDS;
      speciesCounts = EvolutionEngine.#SCRATCH_SPECIES_COUNTS;
    }

    // Step 2: Build species id -> count arrays (linear scan with small-cardinality assumption).
    let speciesUniqueCount = 0;
    let totalPopulationCount = 0;
    for (
      let populationIndex = 0;
      populationIndex < populationLength;
      populationIndex++
    ) {
      const genome = populationRef[populationIndex];
      const speciesId =
        (genome && genome.species != null ? genome.species : -1) | 0;
      let existingSpeciesIndex = -1;
      // Small species cardinality expected (<= population size, usually far lower), so linear search is cache-friendly.
      for (
        let speciesScanIndex = 0;
        speciesScanIndex < speciesUniqueCount;
        speciesScanIndex++
      ) {
        if (speciesIds[speciesScanIndex] === speciesId) {
          existingSpeciesIndex = speciesScanIndex;
          break;
        }
      }
      if (existingSpeciesIndex === -1) {
        speciesIds[speciesUniqueCount] = speciesId;
        speciesCounts[speciesUniqueCount] = 1;
        speciesUniqueCount++;
      } else {
        speciesCounts[existingSpeciesIndex]++;
      }
      totalPopulationCount++; // track total directly to avoid second pass
    }
    if (totalPopulationCount === 0) totalPopulationCount = 1; // safety (should not happen)

    // Step 3: Compute Simpson diversity index (1 - sum(p_i^2)).
    let simpsonAccumulator = 0;
    for (
      let speciesIndex = 0;
      speciesIndex < speciesUniqueCount;
      speciesIndex++
    ) {
      const proportion = speciesCounts[speciesIndex] / totalPopulationCount;
      simpsonAccumulator += proportion * proportion;
    }
    const simpson = 1 - simpsonAccumulator;

    // Step 4: Sample genomes (with replacement) into pooled scratch for weight distribution statistics.
    const boundedSampleSize =
      sampleSize > 0 ? Math.min(populationLength, sampleSize | 0) : 0;
    const sampledLength = boundedSampleSize
      ? EvolutionEngine.#sampleIntoScratch(populationRef, boundedSampleSize)
      : 0;

    // Step 5: Welford single-pass variance for enabled connection weights across sampled genomes.
    let weightMean = 0;
    let weightM2 = 0;
    let enabledWeightCount = 0;
    for (let sampleIndex = 0; sampleIndex < sampledLength; sampleIndex++) {
      const sampledGenome = EvolutionEngine.#SCRATCH_SAMPLE[sampleIndex];
      const connections = sampledGenome.connections || [];
      for (
        let connectionIndex = 0;
        connectionIndex < connections.length;
        connectionIndex++
      ) {
        const connection = connections[connectionIndex];
        if (connection && connection.enabled !== false) {
          enabledWeightCount++;
          const delta = connection.weight - weightMean;
          weightMean += delta / enabledWeightCount;
          weightM2 += delta * (connection.weight - weightMean);
        }
      }
    }
    const weightStdDev = enabledWeightCount
      ? Math.sqrt(weightM2 / enabledWeightCount)
      : 0;

    // Step 6: Return metrics object (pooled scalars only; callers treat as immutable snapshot).
    return { speciesUniqueCount, simpson, wStd: weightStdDev };
  }

  /**
   * Sample `k` items (with replacement) from a source array using the engine RNG.
   *
   * Characteristics:
   * - With replacement: the same element can appear multiple times.
   * - Uses a reusable pooled array (#SCRATCH_SAMPLE_RESULT) that grows geometrically (power-of-two) and is reused
   *   across calls to avoid allocations. Callers MUST copy (`slice()` / spread) if they need to retain the result
   *   beyond the next invocation.
   * - Deterministic under deterministic engine mode (shared RNG state); otherwise non-deterministic.
   * - Returns an empty array (length 0) for invalid inputs (`k <= 0`, non-array, or empty source). The returned
   *   empty array is a fresh literal to avoid accidental aliasing with the scratch buffer.
   *
   * Complexity: O(k). Memory: O(k) only during first growth; subsequent calls reuse storage.
   * Reentrancy: Non-reentrant (shared scratch buffer reused) — do not call concurrently.
   *
   * @param src - Source array to sample from.
   * @param k - Number of samples requested (fractional values truncated via floor). Negative treated as 0.
   * @returns Pooled ephemeral array of length `k` (or 0) containing sampled elements (with replacement).
   * @example
   * // Internal usage (pseudo):
   * const batch = EvolutionEngine['#sampleArray'](population, 16); // do not store long-term
   * const stableCopy = [...batch]; // copy if needed later
   * @internal
   * @remarks Non-reentrant; result must be treated as ephemeral.
   */
  static #sampleArray<T>(src: T[], k: number): T[] {
    // Step 1: Validate inputs & normalize sample size.
    if (!Array.isArray(src) || k <= 0) return [];
    const sampleCount = Math.floor(k);
    const sourceLength = src.length | 0;
    if (sourceLength === 0 || sampleCount === 0) return [];

    // Step 2: Ensure pooled output buffer capacity (power-of-two growth for amortized O(1) reallocation).
    if (sampleCount > EvolutionEngine.#SCRATCH_SAMPLE_RESULT.length) {
      const nextSize = 1 << Math.ceil(Math.log2(sampleCount));
      EvolutionEngine.#SCRATCH_SAMPLE_RESULT = new Array(nextSize);
    }
    const out = EvolutionEngine.#SCRATCH_SAMPLE_RESULT as T[];

    // Step 3: Fill buffer with sampled elements (with replacement) using fast RNG.
    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
      out[sampleIndex] =
        src[(EvolutionEngine.#fastRandom() * sourceLength) | 0];
    }

    // Step 4: Truncate logical length to exactly sampleCount (allows buffer reuse if previously larger).
    out.length = sampleCount;
    return out; // pooled ephemeral array — copy if persistence required
  }

  /**
   * Apply simplify pruning to every genome in the population, catching per-genome failures.
   * Centralizes the simplify loop previously inlined in the main evolution loop.
   * @internal
   */
  static #applySimplifyPruningToPopulation(
    neat: any,
    simplifyStrategy: string,
    simplifyPruneFraction: number
  ) {
    // Step 0: Defensive normalization & early exit.
    if (
      !neat ||
      !Array.isArray(neat.population) ||
      neat.population.length === 0
    )
      return;
    const populationRef: any[] = neat.population;
    const pruneFraction = Number.isFinite(simplifyPruneFraction)
      ? Math.max(0, Math.min(1, simplifyPruneFraction))
      : 0;
    if (pruneFraction === 0) return; // nothing to do if fraction is zero

    // Step 1: Iterate genomes and attempt pruning. Failures are isolated to avoid halting the batch.
    for (
      let genomeIndex = 0;
      genomeIndex < populationRef.length;
      genomeIndex++
    ) {
      const genome = populationRef[genomeIndex];
      if (!genome) continue; // skip null/undefined placeholders defensively
      try {
        EvolutionEngine.#pruneWeakConnectionsForGenome(
          genome,
          simplifyStrategy,
          pruneFraction
        );
      } catch (e) {
        // Intentionally swallow per-genome errors: pruning is a best-effort optimization.
      }
    }
  }

  /**
   * Decide whether to start the simplify/pruning phase.
   * Returns the number of generations the simplify phase should last (0 = do not start).
   * @internal
   */
  static #maybeStartSimplify(
    plateauCounter: number,
    plateauGenerations: number,
    simplifyDuration: number
  ): number {
    try {
      if (plateauCounter >= plateauGenerations) {
        // Only run simplify on non-browser hosts (mirrors previous behavior)
        if (typeof window === 'undefined') return simplifyDuration;
      }
    } catch (e) {
      /* ignore */
    }
    return 0;
  }

  /**
   * Run one simplify generation (centralized logic). Returns the remaining simplify generations.
   * @internal
   */
  static #runSimplifyCycle(
    neat: any,
    simplifyRemaining: number,
    simplifyStrategy: string,
    simplifyPruneFraction: number
  ): number {
    /**
     * Run a single simplify/pruning generation if conditions permit.
     * Simplify is disabled in browser contexts (heuristic: presence of window) to avoid blocking UI threads
     * since pruning can be comparatively expensive and is mainly a server/offline concern.
     */
    // Step 0: Quick exits for zero remaining or missing population.
    if (!simplifyRemaining || !neat || !Array.isArray(neat.population))
      return 0;

    // Step 1: Environment gate (skip when running in browser-like context).
    try {
      if (typeof window !== 'undefined') return simplifyRemaining; // maintain existing behavior
    } catch {
      // Accessing window threw (non-browser host lacking global); continue.
    }

    // Step 2: Optionally record profiling start (high precision micro-timer disabled when profiling off).
    const profilingEnabled = EvolutionEngine.#PROFILE_ENABLED;
    const profileStart = profilingEnabled ? EvolutionEngine.#PROFILE_T0() : 0;

    // Step 3: Apply pruning across the population (best-effort; internal errors isolated per genome).
    EvolutionEngine.#applySimplifyPruningToPopulation(
      neat,
      simplifyStrategy,
      simplifyPruneFraction
    );

    // Step 4: Record profiling delta if enabled.
    if (profilingEnabled) {
      const elapsed = EvolutionEngine.#PROFILE_T0() - profileStart || 0;
      EvolutionEngine.#PROFILE_ADD('simplify', elapsed);
    }

    // Step 5: Decrement remaining generations.
    return simplifyRemaining - 1;
  }

  /**
   * Apply Lamarckian (supervised) training to the current population.
   * Returns the elapsed time (ms) spent in training when profiling is enabled.
   * @internal
   */
  static #applyLamarckianTraining(
    neat: any,
    lamarckianTrainingSet: any[],
    lamarckianIterations: number,
    lamarckianSampleSize: number | undefined,
    safeWrite: (msg: string) => void,
    doProfile: boolean,
    completedGenerations: number
  ): number {
    /**
     * Perform lightweight supervised (Lamarckian) refinement on each network in the current population.
     * The goal is to nudge weights toward local optima without erasing evolutionary diversity.
     *
     * Steps:
     * 1. Early validation & fast exits.
     * 2. Optional sampling of training set (with replacement) to cap per-gen cost.
     * 3. Iterate networks, run a bounded training pass, adjust output biases heuristically.
     * 4. Collect gradient norm statistics when provided by the network implementation.
     * 5. Emit aggregate gradient telemetry & return elapsed time (profiling mode only).
     *
     * Determinism: Dependent on RNG usage inside `network.train`. Our sampling step is deterministic when
     * the engine is in deterministic mode because it relies on `#sampleArray` which uses the shared RNG.
     * Reentrancy: Non-reentrant (shared RNG + ephemeral sampling buffer). Do not call concurrently.
     * Complexity: O(P * (I * B)) where P=population size, I=lamarckianIterations, B=batch cost.
     * Memory: Reuses pooled sample array; no per-network heap allocation beyond library-internal training.
     */

    // Step 1: Validate inputs & early exits.
    if (
      !neat ||
      !Array.isArray(neat.population) ||
      neat.population.length === 0
    )
      return 0;
    if (
      !Array.isArray(lamarckianTrainingSet) ||
      lamarckianTrainingSet.length === 0
    )
      return 0;
    if (!Number.isFinite(lamarckianIterations) || lamarckianIterations <= 0)
      return 0;

    const profileStart = doProfile ? EvolutionEngine.#now() : 0;

    // Step 2: Optionally down-sample the training set (with replacement) to reduce cost for large sets.
    let trainingSetRef = lamarckianTrainingSet;
    if (
      lamarckianSampleSize &&
      lamarckianSampleSize > 0 &&
      lamarckianSampleSize < lamarckianTrainingSet.length
    ) {
      trainingSetRef = EvolutionEngine.#sampleArray(
        lamarckianTrainingSet,
        lamarckianSampleSize
      );
    }

    // Step 3: Iterate networks performing bounded refinement.
    let gradientNormSum = 0;
    let gradientNormSamples = 0;
    const populationRef = neat.population as any[];
    for (
      let networkIndex = 0;
      networkIndex < populationRef.length;
      networkIndex++
    ) {
      const network: any = populationRef[networkIndex];
      if (!network) continue;
      try {
        network.train(trainingSetRef, {
          iterations: lamarckianIterations, // small to preserve diversity pressure
          error: EvolutionEngine.#DEFAULT_TRAIN_ERROR,
          rate: EvolutionEngine.#DEFAULT_TRAIN_RATE,
          momentum: EvolutionEngine.#DEFAULT_TRAIN_MOMENTUM,
          batchSize: EvolutionEngine.#DEFAULT_TRAIN_BATCH_SMALL,
          allowRecurrent: true,
          cost: methods.Cost.softmaxCrossEntropy,
        });
        // Step 3.1: Adjust output biases to maintain exploration post-training.
        EvolutionEngine.#adjustOutputBiasesAfterTraining(network);

        // Step 3.2: Collect gradient norm metric (if provided) for telemetry.
        try {
          const getStats = (network as any).getTrainingStats;
          if (typeof getStats === 'function') {
            const stats = getStats();
            const gradNorm = stats && stats.gradNorm;
            if (Number.isFinite(gradNorm)) {
              gradientNormSum += gradNorm;
              gradientNormSamples++;
            }
          }
        } catch {
          // Silently ignore stat retrieval errors; not critical.
        }
      } catch {
        // Per-network training failure is non-fatal; continue others.
      }
    }

    // Step 4: Emit aggregate gradient telemetry if any samples collected.
    if (gradientNormSamples > 0) {
      safeWrite(
        `[GRAD] gen=${completedGenerations} meanGradNorm=${(
          gradientNormSum / gradientNormSamples
        ).toFixed(4)} samples=${gradientNormSamples}\n`
      );
    }

    // Step 5: Return elapsed time when profiling; else zero to avoid misleading consumers.
    return doProfile ? EvolutionEngine.#now() - profileStart : 0;
  }

  /**
   * Log per-generation telemetry (action entropy, logits, exploration, diversity) and run collapse checks.
   * @internal
   */
  static #logGenerationTelemetry(
    neat: any,
    fittest: any,
    generationResult: any,
    completedGenerations: number,
    safeWrite: (msg: string) => void
  ) {
    if (EvolutionEngine.#TELEMETRY_MINIMAL) return; // Step 0: global guard
    const profileStart = EvolutionEngine.#PROFILE_ENABLED
      ? EvolutionEngine.#PROFILE_T0()
      : 0;
    try {
      // Step 1: Action entropy telemetry
      EvolutionEngine.#logActionEntropy(
        generationResult,
        completedGenerations,
        safeWrite
      );
      // Step 2: Output bias statistics
      EvolutionEngine.#logOutputBiasStats(
        fittest,
        completedGenerations,
        safeWrite
      );
      // Step 3: Logits statistics & collapse detection
      EvolutionEngine.#logLogitsAndCollapse(
        neat,
        fittest,
        completedGenerations,
        safeWrite
      );
      // Step 4: Exploration telemetry
      EvolutionEngine.#logExploration(
        generationResult,
        completedGenerations,
        safeWrite
      );
      // Step 5: Diversity metrics
      EvolutionEngine.#logDiversity(neat, completedGenerations, safeWrite);
    } catch {
      // Swallow any unexpected telemetry exception to avoid disrupting evolution core loop.
    }
    if (EvolutionEngine.#PROFILE_ENABLED) {
      EvolutionEngine.#PROFILE_ADD(
        'telemetry',
        EvolutionEngine.#PROFILE_T0() - profileStart || 0
      );
    }
  }

  /** Log action entropy summary line. @internal */
  static #logActionEntropy(
    generationResult: any,
    gen: number,
    safeWrite: (msg: string) => void
  ) {
    try {
      const stats = EvolutionEngine.#computeActionEntropy(
        generationResult.path
      );
      safeWrite(
        `${
          EvolutionEngine.#LOG_TAG_ACTION_ENTROPY
        } gen=${gen} entropyNorm=${stats.entropyNorm.toFixed(3)} uniqueMoves=${
          stats.uniqueMoves
        } pathLen=${stats.pathLen}\n`
      );
    } catch {
      /* ignore */
    }
  }

  /** Log output bias statistics if output nodes present. @internal */
  static #logOutputBiasStats(
    fittest: any,
    gen: number,
    safeWrite: (msg: string) => void
  ) {
    try {
      const nodes = fittest?.nodes || [];
      const outputCount = EvolutionEngine.#getNodeIndicesByType(
        nodes,
        'output'
      );
      if (outputCount > 0) {
        const biasStats = EvolutionEngine.#computeOutputBiasStats(
          nodes,
          outputCount
        );
        safeWrite(
          `${
            EvolutionEngine.#LOG_TAG_OUTPUT_BIAS
          } gen=${gen} mean=${biasStats.mean.toFixed(
            3
          )} std=${biasStats.std.toFixed(3)} biases=${biasStats.biasesStr}\n`
        );
      }
    } catch {
      /* ignore */
    }
  }

  /** Log logits statistics, detect collapse, and trigger anti-collapse recovery if needed. @internal */
  static #logLogitsAndCollapse(
    neat: any,
    fittest: any,
    gen: number,
    safeWrite: (msg: string) => void
  ) {
    try {
      const history: number[][] = fittest?._lastStepOutputs || [];
      if (!history.length) return;
      const recent = EvolutionEngine.#getTail<number[]>(
        history,
        EvolutionEngine.#RECENT_WINDOW
      );
      const stats: any = EvolutionEngine.#computeLogitStats(recent);
      safeWrite(
        `${EvolutionEngine.#LOG_TAG_LOGITS} gen=${gen} means=${
          stats.meansStr
        } stds=${stats.stdsStr} kurt=${
          stats.kurtStr
        } entMean=${stats.entMean.toFixed(
          3
        )} stability=${stats.stability.toFixed(3)} steps=${recent.length}\n`
      );
      // Collapse detection
      (EvolutionEngine as any)._collapseStreak =
        (EvolutionEngine as any)._collapseStreak || 0;
      let allBelowThreshold = true;
      const stds = stats.stds as ArrayLike<number>;
      for (let index = 0; index < stds.length; index++) {
        if (!(stds[index] < EvolutionEngine.#LOGSTD_FLAT_THRESHOLD)) {
          allBelowThreshold = false;
          break;
        }
      }
      const collapsed =
        allBelowThreshold &&
        (stats.entMean < EvolutionEngine.#ENTROPY_COLLAPSE_THRESHOLD ||
          stats.stability > EvolutionEngine.#STABILITY_COLLAPSE_THRESHOLD);
      if (collapsed) (EvolutionEngine as any)._collapseStreak++;
      else (EvolutionEngine as any)._collapseStreak = 0;
      if (
        (EvolutionEngine as any)._collapseStreak ===
        EvolutionEngine.#COLLAPSE_STREAK_TRIGGER
      ) {
        EvolutionEngine.#antiCollapseRecovery(neat, gen, safeWrite);
      }
    } catch {
      /* ignore */
    }
  }

  /** Log exploration statistics (unique path coverage ratio + progress). @internal */
  static #logExploration(
    generationResult: any,
    gen: number,
    safeWrite: (msg: string) => void
  ) {
    try {
      const expl = EvolutionEngine.#computeExplorationStats(
        generationResult.path
      );
      safeWrite(
        `[EXPLORE] gen=${gen} unique=${expl.unique} pathLen=${
          expl.pathLen
        } ratio=${expl.ratio.toFixed(
          3
        )} progress=${generationResult.progress.toFixed(
          1
        )} satFrac=${(generationResult as any).saturationFraction?.toFixed(
          3
        )}\n`
      );
    } catch {
      /* ignore */
    }
  }

  /** Log diversity metrics (species richness, Simpson index, weight std). @internal */
  static #logDiversity(
    neat: any,
    gen: number,
    safeWrite: (msg: string) => void
  ) {
    try {
      const diversity = EvolutionEngine.#computeDiversityMetrics(neat);
      safeWrite(
        `[DIVERSITY] gen=${gen} species=${
          diversity.speciesUniqueCount
        } simpson=${diversity.simpson.toFixed(
          3
        )} weightStd=${diversity.wStd.toFixed(3)}\n`
      );
    } catch {
      /* ignore */
    }
  }

  /**
   * Persist a snapshot of the state to disk when persistence is enabled and conditions met.
   * @internal
   */
  static #persistSnapshotIfNeeded(
    fs: any,
    pathModule: any,
    persistDir: string | undefined,
    persistTopK: number,
    completedGenerations: number,
    persistEvery: number,
    neat: any,
    bestFitness: number,
    simplifyMode: boolean,
    plateauCounter: number
  ) {
    // Step 0: Preconditions & scheduling cadence.
    if (!fs || !persistDir || persistEvery <= 0) return;
    if (completedGenerations % persistEvery !== 0) return; // respect cadence
    if (!neat || !Array.isArray(neat.population)) return;

    try {
      // Step 1: Profile start (optional).
      const profileStart = EvolutionEngine.#PROFILE_ENABLED
        ? EvolutionEngine.#PROFILE_T0()
        : 0;

      // Step 2: Populate snapshot scalar metadata.
      const snapshot = EvolutionEngine.#SCRATCH_SNAPSHOT_OBJ;
      snapshot.generation = completedGenerations;
      snapshot.bestFitness = bestFitness;
      snapshot.simplifyMode = simplifyMode;
      snapshot.plateauCounter = plateauCounter;
      snapshot.timestamp = Date.now();
      snapshot.telemetryTail = EvolutionEngine.#collectTelemetryTail(neat, 5);

      // Step 3: Gather top-K genomes metadata (score sorted) reusing pooled buffer.
      const populationRef: any[] = neat.population || [];
      const sortedIndices = EvolutionEngine.#getSortedIndicesByScore(
        populationRef
      );
      const topLimit = Math.min(persistTopK, sortedIndices.length);
      let topBuffer = EvolutionEngine.#SCRATCH_SNAPSHOT_TOP;
      if (topBuffer.length < topLimit) topBuffer.length = topLimit; // grow in place
      for (let rank = 0; rank < topLimit; rank++) {
        let entry = topBuffer[rank];
        if (!entry) entry = topBuffer[rank] = {};
        const genome = populationRef[sortedIndices[rank]];
        // Minimal invariant fields for later analysis/UI.
        entry.idx = sortedIndices[rank];
        entry.score = genome?.score;
        entry.nodes = genome?.nodes?.length;
        entry.connections = genome?.connections?.length;
        entry.json = genome?.toJSON ? genome.toJSON() : undefined;
      }
      topBuffer.length = topLimit;
      snapshot.top = topBuffer;

      // Step 4: Serialize compact JSON (no spacing) to minimize disk & parse overhead.
      const filePath = pathModule.join(
        persistDir,
        `snapshot_gen${completedGenerations}.json`
      );
      fs.writeFileSync(filePath, JSON.stringify(snapshot));

      // Step 5: Profiling delta.
      if (EvolutionEngine.#PROFILE_ENABLED) {
        EvolutionEngine.#PROFILE_ADD(
          'snapshot',
          EvolutionEngine.#PROFILE_T0() - profileStart || 0
        );
      }
    } catch {
      // Swallow persistence errors silently (I/O not critical for core evolution loop stability).
    }
  }

  /** Collect a short telemetry tail (last N entries) if NEAT instance exposes getTelemetry(). @internal */
  static #collectTelemetryTail(neat: any, tailLength: number) {
    if (!neat || typeof neat.getTelemetry !== 'function') return undefined;
    try {
      const telemetryValue = neat.getTelemetry();
      if (Array.isArray(telemetryValue)) {
        return EvolutionEngine.#getTail<any>(telemetryValue, tailLength);
      }
      return telemetryValue;
    } catch {
      return undefined;
    }
  }

  /**
   * Compute decision stability over a recent sequence of output vectors.
   *
   * Definition: stability = (# of consecutive pairs with identical argmax) / (total consecutive pairs).
   * For a sequence of N decisions there are (N-1) consecutive pairs; we skip the first vector when counting pairs.
   * Returns 0 when fewer than 2 vectors.
   *
   * Performance notes:
   * - Typical RECENT_WINDOW is small (<= 40) and ACTION_DIM is a fixed 4 for ASCII maze (N,E,S,W); cost is negligible so
   *   caching / memoization would add overhead with no throughput benefit. We therefore compute on demand.
   * - Specialized unrolled path for ACTION_DIM === 4 (the only case here) reduces loop overhead & branch mispredictions.
   * - No allocations; operates purely on provided arrays. If ACTION_DIM ever changes, the generic path still works.
   *
   * Complexity: O(R * D) where R = recent.length, D = ACTION_DIM (constant 4).
   * Determinism: Deterministic given identical input vectors.
   * Reentrancy: Pure function (no shared state touched).
   *
   * @param recent - Array of recent output activation vectors (each length = ACTION_DIM expected).
   * @returns Stability ratio in [0,1].
   * @internal
   */
  static #computeDecisionStability(recent: number[][]): number {
    const sequenceLength = recent?.length || 0;
    if (sequenceLength < 2) return 0; // Need at least one pair
    let stablePairCount = 0;
    let pairCount = 0;
    let previousArgmax = -1;

    // If action dimension is known (4) use unrolled argmax for a slight micro-optimization.
    const actionDim = EvolutionEngine.#ACTION_DIM; // Fixed 4 for ASCII maze (sync with MazeMovement)
    const unroll = actionDim === 4; // Always true currently; kept for defensive future flexibility

    for (let rowIndex = 0; rowIndex < sequenceLength; rowIndex++) {
      const row = recent[rowIndex];
      if (!row || row.length === 0) continue; // defensive skip
      let argmax: number;
      if (unroll && row.length >= 4) {
        // Unrolled argmax for length >=4 (uses first 4 entries; assume exactly ACTION_DIM used downstream)
        let bestVal = row[0];
        argmax = 0;
        const v1 = row[1];
        if (v1 > bestVal) {
          bestVal = v1;
          argmax = 1;
        }
        const v2 = row[2];
        if (v2 > bestVal) {
          bestVal = v2;
          argmax = 2;
        }
        const v3 = row[3];
        if (v3 > bestVal) {
          /* bestVal = v3; */ argmax = 3;
        }
      } else {
        // Generic argmax loop
        argmax = 0;
        let bestVal = row[0];
        for (
          let outputIndex = 1;
          outputIndex < actionDim && outputIndex < row.length;
          outputIndex++
        ) {
          const candidate = row[outputIndex];
          if (candidate > bestVal) {
            bestVal = candidate;
            argmax = outputIndex;
          }
        }
      }
      if (previousArgmax !== -1) {
        pairCount++;
        if (previousArgmax === argmax) stablePairCount++;
      }
      previousArgmax = argmax;
    }
    return pairCount ? stablePairCount / pairCount : 0;
  }

  /**
   * Compute the (base-e) softmax entropy of a single activation vector and normalize it to [0,1].
   *
   * Educational step-by-step outline (softmax + entropy):
   * 1. Defensive checks & determine effective length k: If vector empty or length < 2, entropy is 0 (no uncertainty).
   * 2. Numerical stabilisation: Subtract the maximum activation before exponentiating (log-sum-exp trick) to avoid overflow.
   * 3. Exponentiation: Convert centered logits to unnormalized positive scores e^{x_i - max}.
   * 4. Normalization: Sum the scores and divide each by the sum to obtain probabilities p_i.
   * 5. Shannon entropy: H = -Σ p_i log(p_i); ignore p_i == 0 (contributes 0 by continuity).
   * 6. Normalization to [0,1]: Divide by log(k); for k=4 we use a cached inverse (#INV_LOG4) to avoid division & log.
   *
   * Implementation details:
   * - Provides a specialized unrolled fast path for the very common ACTION_DIM=4 case.
   * - Uses the caller-provided scratch buffer `buf` to store exponentials (avoids per-call allocation).
   * - Previously a small bug omitted the p1 contribution in the 4-way path; fixed here.
   * - Pure & deterministic: no side effects on shared state.
   *
   * @param v Activation vector (logits) whose softmax entropy we want.
   * @param buf Scratch Float64Array with capacity >= v.length used to hold exponentials.
   * @returns Normalized entropy in [0,1]. 0 => fully certain (one probability ~1), 1 => uniform distribution.
   * @internal
   */
  static #softmaxEntropyFromVector(
    v: number[] | undefined,
    buf: Float64Array
  ): number {
    // Step 1: Defensive checks.
    if (!v || !v.length) return 0;
    const k = Math.min(v.length, buf.length);
    if (k <= 1) return 0; // Single outcome => zero entropy

    // Fast path: unrolled 4-way softmax (common case) with explicit operations and cached normalization.
    if (k === 4) {
      // Step 2: Find max for numerical stability.
      const v0 = v[0] || 0;
      const v1 = v[1] || 0;
      const v2 = v[2] || 0;
      const v3 = v[3] || 0;
      let maxVal = v0;
      if (v1 > maxVal) maxVal = v1;
      if (v2 > maxVal) maxVal = v2;
      if (v3 > maxVal) maxVal = v3;
      // Step 3: Exponentiate centered values.
      const e0 = Math.exp(v0 - maxVal);
      const e1 = Math.exp(v1 - maxVal);
      const e2 = Math.exp(v2 - maxVal);
      const e3 = Math.exp(v3 - maxVal);
      // Step 4: Normalize to probabilities.
      const sum = e0 + e1 + e2 + e3 || 1; // guard against pathological underflow
      const p0 = e0 / sum;
      const p1 = e1 / sum;
      const p2 = e2 / sum;
      const p3 = e3 / sum;
      // Step 5: Entropy accumulation (include all four probabilities; p_i log p_i = 0 if p_i=0).
      let entropyAccumulator = 0;
      if (p0 > 0) entropyAccumulator += -p0 * Math.log(p0);
      if (p1 > 0) entropyAccumulator += -p1 * Math.log(p1); // (bug fix: previously omitted)
      if (p2 > 0) entropyAccumulator += -p2 * Math.log(p2);
      if (p3 > 0) entropyAccumulator += -p3 * Math.log(p3);
      // Step 6: Normalize by log(4) using cached inverse.
      return entropyAccumulator * EvolutionEngine.#INV_LOG4;
    }

    // Generic path for k != 4.
    // Step 2: Find maximum for stability.
    let maxVal = -Infinity;
    for (let actionIndex = 0; actionIndex < k; actionIndex++) {
      const value = v[actionIndex] || 0;
      if (value > maxVal) maxVal = value;
    }
    // Step 3 + 4: Exponentiate centered logits and accumulate sum; store in scratch buffer.
    let sum = 0;
    for (let actionIndex = 0; actionIndex < k; actionIndex++) {
      const expValue = Math.exp((v[actionIndex] || 0) - maxVal);
      buf[actionIndex] = expValue;
      sum += expValue;
    }
    if (!sum) sum = 1; // Guard (extreme underflow) -> uniform probabilities (entropy 0)
    // Step 5: Accumulate entropy.
    let entropyAccumulator = 0;
    for (let actionIndex = 0; actionIndex < k; actionIndex++) {
      const probability = buf[actionIndex] / sum;
      if (probability > 0)
        entropyAccumulator += -probability * Math.log(probability);
    }
    // Step 6: Normalize by log(k).
    const denom = Math.log(k);
    return denom > 0 ? entropyAccumulator / denom : 0;
  }

  /**
   * Join the first `len` numeric entries of an array-like into a comma-separated string
   * with fixed decimal precision.
   *
   * Steps (educational):
   * 1. Normalize parameters: clamp `len` to the provided array-like length; clamp `digits` to [0, 20].
   * 2. Grow (geometric) the shared scratch string array if capacity is insufficient.
   * 3. Format each numeric value via `toFixed(digits)` directly into the scratch slots (no interim allocations).
   * 4. Temporarily set the scratch logical length to `len` and `Array.prototype.join(',')` to build the output.
   * 5. Restore the original scratch length (capacity preserved) and return the joined string.
   *
   * Complexity: O(n) time, O(1) additional space (amortized) beyond the shared scratch array.
   * Determinism: Deterministic for identical inputs (`toFixed` stable for finite numbers).
   * Reentrancy: Non-reentrant (shared `#SCRATCH_STR` buffer). Do not invoke concurrently from parallel contexts.
   *
   * @param arrLike Array-like source of numbers (only indices < len are read).
   * @param len Intended number of elements to serialize; clamped to available length.
   * @param digits Fixed decimal places (0–20). Values outside are clamped.
   * @returns Comma-separated string or empty string when `len <= 0` after normalization.
   */
  static #joinNumberArray(
    arrLike: ArrayLike<number>,
    len: number,
    digits = 3
  ): string {
    // Step 1: Parameter normalization.
    if (!arrLike) return '';
    const availableLength = arrLike.length >>> 0; // unsigned coercion
    if (!Number.isFinite(len) || len <= 0 || availableLength === 0) return '';
    const effectiveLength = len > availableLength ? availableLength : len;
    let fixedDigits = digits;
    if (!Number.isFinite(fixedDigits)) fixedDigits = 0;
    if (fixedDigits < 0) fixedDigits = 0;
    else if (fixedDigits > 20) fixedDigits = 20; // per spec bounds for toFixed

    // Step 2: Ensure scratch capacity (geometric growth: next power of two).
    if (effectiveLength > EvolutionEngine.#SCRATCH_STR.length) {
      const nextSize = 1 << Math.ceil(Math.log2(effectiveLength));
      EvolutionEngine.#SCRATCH_STR = new Array(nextSize);
    }

    // Step 3: Populate scratch with formatted numbers.
    const stringScratch = EvolutionEngine.#SCRATCH_STR;
    for (let valueIndex = 0; valueIndex < effectiveLength; valueIndex++) {
      // Cast to number explicitly; treat missing entries as 0 (consistent with original implicit coercion semantics).
      const rawValue = (arrLike[valueIndex] as number) ?? 0;
      stringScratch[valueIndex] = Number.isFinite(rawValue)
        ? rawValue.toFixed(fixedDigits)
        : 'NaN';
    }

    // Step 4: Join using a temporarily truncated logical length.
    const priorLength = stringScratch.length;
    stringScratch.length = effectiveLength;
    const joined = stringScratch.join(',');

    // Step 5: Restore logical length (capacity retained for reuse) and return.
    stringScratch.length = priorLength;
    return joined;
  }

  /**
   * Extract the last `count` items from an input array into a pooled scratch buffer (no new allocation).
   *
   * Educational steps:
   * 1. Validate & normalize parameters: non-array, empty, non-positive, or non-finite counts -> empty result. Clamp
   *    `count` to the source length and floor fractional values.
   * 2. Ensure pooled scratch buffer (#SCRATCH_TAIL) has sufficient capacity; grow geometrically (next power-of-two)
   *    so resizes are amortized and rare.
   * 3. Copy the tail slice elements into the scratch buffer with a tight loop (avoids `slice()` allocation).
   * 4. Set the logical length of the scratch buffer to the number of copied elements and return it.
   *
   * Complexity: O(k) where k = min(count, source length). Amortized O(1) extra space beyond the shared buffer.
   * Determinism: Deterministic given identical input array contents and `count`.
   * Reentrancy: Non-reentrant (returns a shared pooled buffer). Callers MUST copy (`slice()` / spread) if they need
   *             to retain contents across another engine call that may reuse the scratch.
   * Mutability: Returned array is mutable; mutations will affect the scratch buffer for subsequent calls.
   *
   * @param arr Source array reference.
   * @param n Desired tail length (floored; negative / non-finite treated as 0).
   * @returns Pooled ephemeral array containing the last `min(n, arr.length)` elements (or empty array literal when 0).
   * @internal
   */
  static #getTail<T>(arr: T[] | undefined, n: number): T[] {
    // Step 1: Validate & normalize parameters.
    if (
      !Array.isArray(arr) ||
      arr.length === 0 ||
      !Number.isFinite(n) ||
      n <= 0
    )
      return [];
    const desired = Math.floor(n);
    const takeCount = desired >= arr.length ? arr.length : desired;
    if (takeCount === 0) return [];

    // Step 2: Ensure scratch capacity via geometric growth (next power of two >= takeCount).
    if (takeCount > EvolutionEngine.#SCRATCH_TAIL.length) {
      const nextSize = 1 << Math.ceil(Math.log2(takeCount));
      EvolutionEngine.#SCRATCH_TAIL = new Array(nextSize);
    }

    // Step 3: Copy tail slice into scratch buffer.
    const tailBuffer = EvolutionEngine.#SCRATCH_TAIL as T[];
    const startIndex = arr.length - takeCount;
    for (let elementIndex = 0; elementIndex < takeCount; elementIndex++) {
      tailBuffer[elementIndex] = arr[startIndex + elementIndex]!;
    }

    // Step 4: Set logical length & return pooled (ephemeral) tail view.
    tailBuffer.length = takeCount;
    return tailBuffer;
  }

  /** Delegate to MazeUtils.pushHistory to keep bounded history semantics. @internal */
  static #pushHistory<T>(buf: T[] | undefined, v: T, maxLen: number): T[] {
    return MazeUtils.pushHistory(buf as any, v as any, maxLen) as T[];
  }

  /**
   * Re-center (demean) output node biases in-place and clamp them to a safe absolute range.
   *
   * Steps:
   * 1. Collect indices of output nodes into the shared scratch index buffer (#SCRATCH_NODE_IDX).
   * 2. Compute mean and (sample) standard deviation of their biases via Welford's online algorithm (single pass).
   * 3. Subtract the mean from each bias (re-centering) and clamp to ±#OUTPUT_BIAS_CLAMP to avoid extreme drift.
   * 4. Persist lightweight stats on the network object as `_outputBiasStats` for telemetry/logging.
   *
   * Rationale: Keeping output biases centered reduces systematic preference for any action direction emerging
   * from cumulative mutation drift, improving exploration signal quality without resetting useful relative offsets.
   *
   * Complexity: O(O) where O = number of output nodes. Memory: no new allocations (shared scratch indices reused).
   * Determinism: Deterministic given identical input network state. Reentrancy: Non-reentrant (shared scratch buffer).
   * Defensive Behavior: Silently no-ops if network shape unexpected; swallows errors to avoid destabilizing evolution loop.
   *
   * @param network Network whose output node biases will be recentred.
   * @internal
   */
  static #centerOutputBiases(network: any): void {
    try {
      const nodeList = network?.nodes || [];
      const totalNodeCount = nodeList.length | 0;
      if (totalNodeCount === 0) return;

      // Step 1: Collect output node indices.
      let outputNodeCount = 0;
      for (let nodeIndex = 0; nodeIndex < totalNodeCount; nodeIndex++) {
        const candidateNode = nodeList[nodeIndex];
        if (candidateNode && candidateNode.type === 'output') {
          EvolutionEngine.#SCRATCH_NODE_IDX[outputNodeCount++] = nodeIndex;
        }
      }
      if (outputNodeCount === 0) return;

      // Step 2: Welford online mean & variance (M2 accumulator).
      let meanBias = 0;
      let sumSquaredDiffs = 0; // M2
      for (let outputIndex = 0; outputIndex < outputNodeCount; outputIndex++) {
        const nodeIdx = EvolutionEngine.#SCRATCH_NODE_IDX[outputIndex];
        const biasValue = Number(nodeList[nodeIdx].bias) || 0;
        const sampleCount = outputIndex + 1;
        const delta = biasValue - meanBias;
        meanBias += delta / sampleCount;
        sumSquaredDiffs += delta * (biasValue - meanBias);
      }
      const stdBias = outputNodeCount
        ? Math.sqrt(sumSquaredDiffs / outputNodeCount)
        : 0;

      // Step 3: Recenter & clamp.
      const clampAbs = EvolutionEngine.#OUTPUT_BIAS_CLAMP;
      for (let outputIndex = 0; outputIndex < outputNodeCount; outputIndex++) {
        const nodeIdx = EvolutionEngine.#SCRATCH_NODE_IDX[outputIndex];
        const original = Number(nodeList[nodeIdx].bias) || 0;
        let adjusted = original - meanBias;
        if (adjusted > clampAbs) adjusted = clampAbs;
        else if (adjusted < -clampAbs) adjusted = -clampAbs;
        nodeList[nodeIdx].bias = adjusted;
      }

      // Step 4: Persist stats for optional telemetry.
      (network as any)._outputBiasStats = { mean: meanBias, std: stdBias };
    } catch {
      // swallow errors (best-effort maintenance routine)
    }
  }

  /**
   * Prune (disable) a fraction of the weakest enabled connections in a genome according to a strategy.
   *
   * High‑level flow:
   * 1. Validate inputs & normalize prune fraction (clamp into [0,1]); exit early for degenerate cases.
   * 2. Collect enabled connections into a pooled candidate buffer (no fresh allocations, reused scratch array).
   * 3. Compute how many to prune: floor(enabled * fraction) but ensure at least 1 when fraction > 0.
   * 4. Order candidates per strategy (e.g. prefer recurrent links first) using small‑array insertion sorts / partitions.
   * 5. Disable (set enabled = false) the selected weakest connections in place (no structural array mutation).
   *
   * Rationale:
   * - Periodic pruning combats structural bloat / drift, helping convergence and interpretability.
   * - In‑place flagging preserves indices referenced elsewhere (e.g. mutation bookkeeping) and avoids churn.
   * - Strategy indirection lets us bias removal (recurrent first, etc.) without duplicating the core pruning loop.
   *
   * Strategies supported (case‑sensitive):
   * - "weakRecurrentPreferred" : Two‑phase ordering. Recurrent OR gater connections are partitioned to the front, each partition then sorted by |weight| ascending so recurrent/gater weakest go first.
   * - (any other string / default) : All enabled connections globally ordered by |weight| ascending.
   *
   * Determinism: Deterministic for a fixed genome state and strategy (sorting by stable numeric key, insertion sort chosen for small E ensures predictable behavior). If two connections share identical |weight| their relative order may depend on initial array order (stable enough for internal use).
   * Reentrancy: Uses class‑pooled scratch buffers (#SCRATCH_CONN_CAND, #SCRATCH_CONN_FLAGS); not safe for concurrent calls.
   * Complexity: O(E log E) worst via native sort fallback for larger batches; predominantly O(E^2) small‑E insertion sorts (E = enabled connections) with tiny constants typical for NEAT genomes.
   * Memory: O(1) additional heap (scratch arrays reused, geometric growth elsewhere handled centrally).
   * Failure Handling: Silently swallows per‑genome errors (best‑effort maintenance; evolution loop stability prioritized).
   *
   * @param genome Genome object whose `connections` array will be examined. Each connection is expected to expose:
   *  - weight: number (signed; magnitude used for weakness ordering)
   *  - enabled: boolean flag (connections with enabled === false are ignored; flag is flipped to disable)
   *  - recurrent / gater (optional): truthy flags used by the recurrent‑preferential strategy
   *  Additional properties (from, to, etc.) are ignored here.
   * @param simplifyStrategy Strategy key controlling candidate ordering. Recognized: "weakRecurrentPreferred" (recurrent first); any other value falls back to pure weakest‑by‑|weight| ordering.
   * @param simplifyPruneFraction Fraction of enabled connections to prune (0..1). Values outside range are clamped. A positive fraction that produces a 0 count still forces pruning of 1 connection (progress guarantee). 0 or non‑finite => no‑op.
   *
   * @example
   * // Internally during simplify phase:
   * EvolutionEngine['#pruneWeakConnectionsForGenome'](genome, 'weakRecurrentPreferred', 0.15);
   *
   * @internal
   */
  static #pruneWeakConnectionsForGenome(
    genome: any,
    simplifyStrategy: string,
    simplifyPruneFraction: number
  ): void {
    try {
      if (!genome || !Array.isArray(genome.connections)) return; // Step 1: validate genome structure
      const rawFraction = Number.isFinite(simplifyPruneFraction)
        ? simplifyPruneFraction
        : 0;
      if (rawFraction <= 0) return; // nothing requested

      // Step 2: Collect enabled connections.
      const allConnections = genome.connections as any[];
      let candidateConnections = EvolutionEngine.#collectEnabledConnections(
        allConnections
      );
      const enabledConnectionCount = candidateConnections.length;
      if (enabledConnectionCount === 0) return; // no work

      // Step 3: Determine prune count (at least one, but never exceed enabled connections).
      const clampedFraction =
        rawFraction >= 1 ? 1 : rawFraction < 0 ? 0 : rawFraction;
      let pruneTarget = Math.floor(enabledConnectionCount * clampedFraction);
      if (clampedFraction > 0 && pruneTarget === 0) pruneTarget = 1; // ensure progress when fraction > 0
      if (pruneTarget <= 0) return; // safety

      // Step 4: Order / partition candidates per strategy.
      candidateConnections = EvolutionEngine.#sortCandidatesByStrategy(
        candidateConnections,
        simplifyStrategy
      );

      // Step 5: Disable smallest enabled connections.
      EvolutionEngine.#disableSmallestEnabledConnections(
        candidateConnections,
        Math.min(pruneTarget, candidateConnections.length)
      );
    } catch {
      // Swallow per-genome pruning errors (non-critical maintenance operation)
    }
  }

  /**
   * Collect all currently enabled connections from a genome connection array into a pooled scratch buffer.
   *
   * Steps:
   * 1. Validate & early exit: non-array or empty -> return new empty literal (avoids exposing scratch).
   * 2. Reset pooled scratch buffer (#SCRATCH_CONN_CAND) logical length to 0 (capacity retained for reuse).
   * 3. Linear scan: push each connection whose `enabled !== false` (treats missing / undefined as enabled for legacy compatibility).
   * 4. Return the pooled scratch array (EPHEMERAL) containing references to the enabled connection objects.
   *
   * Rationale:
   * - Centralizes enabled filtering logic so pruning / analytics share identical semantics.
   * - Reuses a single array instance to avoid transient garbage during frequent simplify phases.
   * - Keeps semantics liberal (`enabled !== false`) to treat absent flags as enabled (historical behavior preserved).
   *
   * Determinism: Deterministic for a fixed input ordering (stable one-pass filter, no reordering).
   * Reentrancy: Not reentrant — returns a shared pooled buffer; callers must copy if they need persistence across nested engine calls.
   * Complexity: O(E) where E = total connections scanned.
   * Memory: O(1) additional heap; capacity grows geometrically elsewhere and is retained.
   *
   * @param connectionsSource Array of connection objects; each may expose `enabled` boolean (false => filtered out).
   * @returns Pooled ephemeral array of enabled connections (DO NOT mutate length; copy if storing long-term).
   * @example
   * const enabled = EvolutionEngine['#collectEnabledConnections'](genome.connections);
   * const stableCopy = enabled.slice(); // only if retention needed
   * @internal
   */
  static #collectEnabledConnections(connectionsSource: any[]): any[] {
    // Step 1: Validate input & fast exit.
    if (!Array.isArray(connectionsSource) || connectionsSource.length === 0)
      return [];

    // Step 2: Reset pooled buffer.
    const candidateBuffer = EvolutionEngine.#SCRATCH_CONN_CAND;
    candidateBuffer.length = 0;

    // Step 3: Linear scan & collect enabled connections.
    for (
      let connectionIndex = 0;
      connectionIndex < connectionsSource.length;
      connectionIndex++
    ) {
      const candidateConnection = connectionsSource[connectionIndex];
      if (candidateConnection && candidateConnection.enabled !== false)
        candidateBuffer.push(candidateConnection);
    }

    // Step 4: Return pooled ephemeral result.
    return candidateBuffer;
  }

  /**
   * Collect enabled outgoing connections from a hidden node that terminate at any output node.
   *
   * Educational steps:
   * 1. Validate inputs & early-exit: invalid node / connections / zero outputs => fresh empty array literal
   *    (avoid exposing pooled scratch for erroneous calls).
   * 2. Clamp effective output count to available scratch index capacity and `nodesRef` length (defensive bounds).
   * 3. Reset pooled result buffer (#SCRATCH_HIDDEN_OUT) by setting length=0 (capacity retained for reuse).
   * 4. Iterate enabled outgoing connections; for each, linearly scan output indices (tiny constant factor) to
   *    detect whether its `to` endpoint references one of the output nodes; push on first match and break.
   * 5. Return pooled buffer (EPHEMERAL). Callers MUST copy if they need persistence beyond next engine helper.
   *
   * Rationale:
   * - Output node count is small (typically 4), so an inner linear scan is faster and allocation-free compared
   *   to constructing a Set/Map each time.
   * - Pooled buffer eliminates per-call garbage during simplify/pruning phases where this helper can be hot.
   * - Defensive clamping prevents out-of-bounds reads if caller overstates `outputCount`.
   *
   * Determinism: Deterministic given stable ordering of hiddenNode.connections.out and scratch index ordering.
   * Reentrancy: Not reentrant (shared scratch buffer). Do not call concurrently.
   * Complexity: O(E * O) where E = enabled outgoing connections, O = outputCount (tiny constant).
   * Memory: O(1) additional heap (buffer reused). No new allocations after initial growth elsewhere.
   * Failure Handling: Returns [] on invalid inputs instead of exposing scratch to minimize accidental mutation.
   *
   * @param hiddenNode Hidden node object with structure `{ connections: { out: Connection[] } }`.
   * @param nodesRef Full node array; indices stored in `#SCRATCH_NODE_IDX` resolve into this array.
   * @param outputCount Declared number of output nodes (will be clamped to safe range).
   * @returns Pooled ephemeral array of connections from `hiddenNode` to any output node.
   * @example
   * const outs = EvolutionEngine['#collectHiddenToOutputConns'](hNode, nodes, outputCount);
   * const stable = outs.slice(); // copy if retention required
   * @internal
   */
  static #collectHiddenToOutputConns(
    hiddenNode: any,
    nodesRef: any[],
    outputCount: number
  ): any[] {
    // Step 1: Input validation & quick exits.
    if (
      !hiddenNode ||
      !hiddenNode.connections ||
      !Array.isArray(nodesRef) ||
      nodesRef.length === 0 ||
      !Number.isFinite(outputCount) ||
      outputCount <= 0
    ) {
      return [];
    }

    // Step 2: Clamp effective output count.
    const maxScratch = EvolutionEngine.#SCRATCH_NODE_IDX.length;
    const effectiveOutputCount = Math.min(
      outputCount | 0,
      maxScratch,
      nodesRef.length
    );
    if (effectiveOutputCount <= 0) return [];

    // Step 3: Reset pooled result buffer.
    const hiddenOutBuffer = EvolutionEngine.#SCRATCH_HIDDEN_OUT;
    hiddenOutBuffer.length = 0;

    // Step 4: Enumerate enabled outgoing connections.
    const outgoing = hiddenNode.connections.out || [];
    for (let outIndex = 0; outIndex < outgoing.length; outIndex++) {
      const candidate = outgoing[outIndex];
      if (!candidate || candidate.enabled === false) continue;
      // Step 4a: Scan output node indices (tiny O constant) for a match.
      for (
        let outputIndex = 0;
        outputIndex < effectiveOutputCount;
        outputIndex++
      ) {
        const nodeIdx = EvolutionEngine.#SCRATCH_NODE_IDX[outputIndex];
        const targetNode = nodesRef[nodeIdx];
        if (candidate.to === targetNode) {
          hiddenOutBuffer.push(candidate);
          break; // proceed to next connection
        }
      }
    }

    // Step 5: Return pooled (ephemeral) result buffer.
    return hiddenOutBuffer;
  }

  /**
   * Order (in-place) a buffer of enabled candidate connections according to a pruning strategy.
   *
   * Strategies:
   *  - "weakRecurrentPreferred": Two-phase ordering. First, a linear partition brings recurrent OR gater
   *    connections ( (from === to) || gater ) to the front preserving relative order within each group (stable-ish
   *    via single forward pass + conditional swaps). Then each partition (recurrent/gater segment and the remainder)
   *    is insertion-sorted independently by ascending absolute weight (|weight| smallest first).
   *  - (default / anything else): Entire array insertion-sorted by ascending absolute weight.
   *
   * Returned array is the same reference mutated in place (allows fluent internal usage). This helper never allocates
   * a new array: it relies on small-N characteristics of genomes during simplify phases, making insertion sort's
   * O(n^2) worst-case acceptable with very low constants (n typically << 128 for enabled connection subsets).
   *
   * Steps (generic):
   * 1. Validate input; return as-is for non-arrays / empty arrays.
   * 2. If strategy is recurrent-preferential: partition then insertion-sort both segments.
   * 3. Else: insertion-sort the entire buffer.
   * 4. Return mutated buffer reference.
   *
   * Rationale:
   * - Partition + localized sorts avoid a full compare function invocation for every cross-group pair when we wish
   *   to bias pruning away from recurrent loops (or explicitly target them first).
   * - Insertion sort avoids engine-level allocations and outperforms generic sort for the very small candidate sets
   *   typically encountered during pruning sessions.
   *
   * Determinism: Deterministic given a stable initial ordering and identical connection properties. Ties (equal |weight|)
   * retain relative order within each partition thanks to forward-scanning partition & stable insertion strategy for
   * equal magnitudes (we only shift while strictly greater).
   * Complexity: O(n^2) worst-case (n = candidates length) but n is small; partition pass is O(n).
   * Memory: O(1) additional (in-place swaps only). No allocations.
   * Reentrancy: Pure w.r.t. global engine state; mutates only the provided array reference.
   *
   * @param candidateConnections Array of connection objects (mutated in-place) to order for pruning selection.
   * @param strategyKey Strategy discriminator string ("weakRecurrentPreferred" or fallback ordering).
   * @returns Same reference to `candidateConnections`, ordered per strategy.
   * @example
   * EvolutionEngine['#sortCandidatesByStrategy'](buf, 'weakRecurrentPreferred');
   * @internal
   */
  static #sortCandidatesByStrategy(
    candidateConnections: any[],
    strategyKey: string
  ): any[] {
    // Step 1: Validate input.
    if (
      !Array.isArray(candidateConnections) ||
      candidateConnections.length === 0
    )
      return candidateConnections;

    // Step 2: Strategy-specific handling.
    if (strategyKey === 'weakRecurrentPreferred') {
      // 2a. Partition recurrent/gater to front (stable-ish: single forward scan + conditional swap when out-of-place).
      let partitionWriteIndex = 0;
      for (
        let scanIndex = 0;
        scanIndex < candidateConnections.length;
        scanIndex++
      ) {
        const connectionCandidate = candidateConnections[scanIndex];
        if (
          connectionCandidate &&
          (connectionCandidate.from === connectionCandidate.to ||
            connectionCandidate.gater)
        ) {
          if (scanIndex !== partitionWriteIndex) {
            const tmpConnection = candidateConnections[partitionWriteIndex];
            candidateConnections[partitionWriteIndex] =
              candidateConnections[scanIndex];
            candidateConnections[scanIndex] = tmpConnection;
          }
          partitionWriteIndex++;
        }
      }
      // 2b. Insertion sort recurrent/gater partition by |weight| ascending.
      EvolutionEngine.#insertionSortByAbsWeight(
        candidateConnections,
        0,
        partitionWriteIndex
      );
      // 2c. Insertion sort remainder partition similarly.
      EvolutionEngine.#insertionSortByAbsWeight(
        candidateConnections,
        partitionWriteIndex,
        candidateConnections.length
      );
      return candidateConnections;
    }

    // Step 3: Fallback simple ordering (whole array) by |weight|.
    EvolutionEngine.#insertionSortByAbsWeight(
      candidateConnections,
      0,
      candidateConnections.length
    );
    return candidateConnections;
  }

  /**
   * In‑place insertion sort of a slice of a candidate connection buffer by ascending absolute weight.
   *
   * Design / behavior:
   * - Stable for ties: when two connections have identical |weight| their original relative order is preserved
   *   because we only shift elements whose |weight| is strictly greater than the candidate being inserted.
   * - Bounds are treated as a half‑open interval [startIndex, endExclusive). Out‑of‑range / degenerate ranges
   *   (null buffer, start >= endExclusive, slice length < 2) return immediately.
   * - Missing / non‑finite weights are coerced to 0 via `Math.abs(candidate?.weight || 0)` keeping semantics
   *   consistent with liberal pruning logic elsewhere.
   * - Chosen because candidate sets during simplify phases are typically small (tens, rarely > 128); insertion
   *   sort outperforms generic `Array.prototype.sort` for small N while avoiding allocation and comparator indirection.
   *
   * Complexity: O(k^2) worst‑case (k = endExclusive - startIndex) with tiny constants for typical k.
   * Memory: O(1) extra (element temp + indices). No allocations.
   * Determinism: Fully deterministic for a fixed input slice content.
   * Reentrancy: Pure (mutates only provided buffer slice; no shared global scratch accessed).
   *
   * @param connectionsBuffer Buffer containing connection objects with a numeric `weight` property.
   * @param startIndex Inclusive start index of the slice to sort (negative values coerced to 0).
   * @param endExclusive Exclusive end index (values > buffer length clamped). If <= startIndex the call is a no‑op.
   * @internal
   */
  static #insertionSortByAbsWeight(
    connectionsBuffer: any[],
    startIndex: number,
    endExclusive: number
  ): void {
    // Step 0: Defensive normalization & fast exits.
    if (!Array.isArray(connectionsBuffer)) return;
    const length = connectionsBuffer.length;
    if (length === 0) return;
    let from = Number.isFinite(startIndex) ? startIndex | 0 : 0;
    let to = Number.isFinite(endExclusive) ? endExclusive | 0 : 0;
    if (from < 0) from = 0;
    if (to > length) to = length;
    if (to - from < 2) return; // nothing to order

    // Step 1: Standard insertion sort (stable for equal absolute weights).
    for (let scanIndex = from + 1; scanIndex < to; scanIndex++) {
      const candidateConnection = connectionsBuffer[scanIndex];
      const candidateAbsWeight = Math.abs(
        candidateConnection && Number.isFinite(candidateConnection.weight)
          ? candidateConnection.weight
          : 0
      );
      let shiftIndex = scanIndex - 1;
      // Shift while strictly greater (preserves order of equals => stability).
      while (shiftIndex >= from) {
        const probe = connectionsBuffer[shiftIndex];
        const probeAbsWeight = Math.abs(
          probe && Number.isFinite(probe.weight) ? probe.weight : 0
        );
        if (probeAbsWeight <= candidateAbsWeight) break;
        connectionsBuffer[shiftIndex + 1] = probe;
        shiftIndex--;
      }
      connectionsBuffer[shiftIndex + 1] = candidateConnection;
    }
  }

  /**
   * Maximum candidate length (inclusive) at which bulk pruning still prefers insertion sort
   * over native Array.prototype.sort for determinism and lower overhead on very small arrays.
   * @internal
   */
  static #PRUNE_BULK_INSERTION_MAX = 64;

  /**
   * Disable (set enabled = false) the weakest enabled connections up to a target count.
   *
   * Two operating modes chosen adaptively by `pruneCount` vs active candidate size:
   * 1. Bulk mode (pruneCount >= activeEnabled/2): Fully order candidates by |weight| then disable
   *    the first pruneCount entries. Uses insertion sort for small N (<= #PRUNE_BULK_INSERTION_MAX)
   *    else a single native sort call.
   * 2. Sparse mode (pruneCount < activeEnabled/2): Repeated selection of current minimum |weight|
   *    without fully sorting (multi-pass partial selection). After each disable we swap the last
   *    active element into the removed slot (shrinking window) to avoid O(n) splices.
   *
   * Rationale:
   * - Avoids the O(n log n) cost of a full sort when disabling only a small fraction (selection
   *   becomes O(k * n) but k << n). When pruning many, a single ordering is cheaper.
   * - In-place modification preserves object identity and external references.
   * - Liberal enabled check (enabled !== false) matches collection semantics.
   * - Uses a reusable `Uint8Array` (#SCRATCH_CONN_FLAGS) resized geometrically (currently only
   *   cleared here—reserved for potential future marking/telemetry without reallocation).
   *
   * Complexity:
   * - Bulk: O(n^2) for insertion path small n, else O(n log n) via native sort.
   * - Sparse: O(k * n) worst (k = pruneCount) with shrinking n after each removal (average slightly better).
   * Memory: O(1) additional (reuses shared scratch flags; no new arrays created).
   * Determinism: Deterministic for identical candidate ordering and weights (native sort comparator is pure).
   * Reentrancy: Not safe for concurrent calls (shared scratch flags buffer reused & zeroed).
   * Failure Handling: Silently no-ops on invalid inputs.
   *
   * @param candidateConnections Array containing candidate connection objects (each with `weight` & `enabled`).
   * @param pruneCount Number of weakest enabled connections to disable (clamped to [0, candidateConnections.length]).
   * @internal
   */
  static #disableSmallestEnabledConnections(
    candidateConnections: any[],
    pruneCount: number
  ): void {
    // Step 0: Defensive validation & normalization.
    if (!Array.isArray(candidateConnections) || !candidateConnections.length)
      return;
    if (!Number.isFinite(pruneCount) || pruneCount <= 0) return;
    const totalCandidates = candidateConnections.length;
    if (pruneCount >= totalCandidates) pruneCount = totalCandidates;

    // Step 1: Prepare / grow scratch flags (reserved for potential future marking / metrics).
    let connectionFlags = EvolutionEngine.#SCRATCH_CONN_FLAGS;
    if (totalCandidates > connectionFlags.length) {
      EvolutionEngine.#SCRATCH_CONN_FLAGS = new Uint8Array(totalCandidates);
      connectionFlags = EvolutionEngine.#SCRATCH_CONN_FLAGS;
    } else {
      connectionFlags.fill(0, 0, totalCandidates);
    }

    // Step 2: Count / compact enabled connections to front to reduce later scans (sparse mode benefit).
    let activeEnabledCount = 0;
    for (let scanIndex = 0; scanIndex < totalCandidates; scanIndex++) {
      const connectionRef = candidateConnections[scanIndex];
      if (connectionRef && connectionRef.enabled !== false) {
        if (scanIndex !== activeEnabledCount)
          candidateConnections[activeEnabledCount] = connectionRef;
        activeEnabledCount++;
      }
    }
    if (activeEnabledCount === 0) return; // Nothing to disable.
    if (pruneCount >= activeEnabledCount) pruneCount = activeEnabledCount; // Clamp again after compaction.

    // Step 3: Choose operating mode based on fraction.
    if (pruneCount >= activeEnabledCount >>> 1) {
      // --- Bulk mode ---
      if (activeEnabledCount <= EvolutionEngine.#PRUNE_BULK_INSERTION_MAX) {
        EvolutionEngine.#insertionSortByAbsWeight(
          candidateConnections,
          0,
          activeEnabledCount
        );
      } else {
        candidateConnections
          .slice(0, activeEnabledCount) // sort only active slice; slice() to avoid comparing undefined tail beyond active
          .sort(
            (firstConnection: any, secondConnection: any) =>
              Math.abs(firstConnection?.weight || 0) -
              Math.abs(secondConnection?.weight || 0)
          )
          .forEach((sortedConnection, sortedIndex) => {
            candidateConnections[sortedIndex] = sortedConnection;
          });
      }
      const disableLimit = pruneCount;
      for (let disableIndex = 0; disableIndex < disableLimit; disableIndex++) {
        const connectionRef = candidateConnections[disableIndex];
        if (connectionRef && connectionRef.enabled !== false)
          connectionRef.enabled = false;
      }
      return;
    }

    // Step 4: Sparse mode (iterative selection of minimum absolute weight among remaining active slice).
    let remainingToDisable = pruneCount;
    let activeSliceLength = activeEnabledCount;
    while (remainingToDisable > 0 && activeSliceLength > 0) {
      // 4a. Find index of current minimum |weight| in [0, activeSliceLength).
      let minIndex = 0;
      let minAbsWeight = Math.abs(
        candidateConnections[0] &&
          Number.isFinite(candidateConnections[0].weight)
          ? candidateConnections[0].weight
          : 0
      );
      for (let probeIndex = 1; probeIndex < activeSliceLength; probeIndex++) {
        const probe = candidateConnections[probeIndex];
        const probeAbs = Math.abs(
          probe && Number.isFinite(probe.weight) ? probe.weight : 0
        );
        if (probeAbs < minAbsWeight) {
          minAbsWeight = probeAbs;
          minIndex = probeIndex;
        }
      }
      // 4b. Disable selected connection.
      const targetConnection = candidateConnections[minIndex];
      if (targetConnection && targetConnection.enabled !== false)
        targetConnection.enabled = false;
      // 4c. Shrink active window by swapping last active-1 element into freed slot.
      const lastActiveIndex = --activeSliceLength;
      candidateConnections[minIndex] = candidateConnections[lastActiveIndex];
      remainingToDisable--;
    }
  }

  /**
   * Compute directional action entropy (normalized to [0,1]) and number of distinct move directions
   * observed along a path of integer coordinates.
   *
   * Behaviour & notes:
   * - Uses the pooled `#SCRATCH_COUNTS` Int32Array (length 4) to avoid per-call allocations. Callers
   *   MUST NOT rely on preserved contents across engine helpers.
   * - Treats a direction as the delta between consecutive coordinates. Only unit deltas in the 8-neighbour
   *   Moore neighbourhood with a mapped index are considered; others are ignored.
   * - Returns entropy normalized by log(4) using `#INV_LOG4` so results are in [0,1]. For empty/degenerate
   *   inputs the entropy is 0 and uniqueMoves is 0.
   *
   * Complexity: O(L) where L = pathArr.length. Memory: O(1) (no allocations). Determinism: deterministic for
   * identical inputs. Reentrancy: Non-reentrant due to shared scratch buffer use.
   *
   * @param pathArr Array-like sequence of [x,y] coordinates visited (expected integers).
   * @returns Object with { entropyNorm, uniqueMoves, pathLen }.
   * @internal
   */
  static #computeActionEntropy(
    pathArr: ReadonlyArray<[number, number]>
  ): { entropyNorm: number; uniqueMoves: number; pathLen: number } {
    // Defensive guards.
    if (!Array.isArray(pathArr) || pathArr.length < 2)
      return { entropyNorm: 0, uniqueMoves: 0, pathLen: pathArr?.length | 0 };

    const counts = EvolutionEngine.#SCRATCH_COUNTS;
    // Manual reset for small fixed-size buffer (faster than fill for hot path)
    counts[0] = 0;
    counts[1] = 0;
    counts[2] = 0;
    counts[3] = 0;

    let totalMoves = 0;
    const dirMap = EvolutionEngine.#DIR_DELTA_TO_INDEX; // 3x3 lookup mapping

    // Scan consecutive pairs and tally mapped directions.
    for (let i = 1; i < pathArr.length; i++) {
      const cur = pathArr[i];
      const prev = pathArr[i - 1];
      if (!cur || !prev) continue;
      const dx = cur[0] - prev[0];
      const dy = cur[1] - prev[1];
      // Only unit/neighbour moves are considered (dx,dy in -1..1)
      if (dx < -1 || dx > 1 || dy < -1 || dy > 1) continue;
      const key = (dx + 1) * 3 + (dy + 1); // maps to 0..8
      const dirIdx = dirMap[key];
      if (dirIdx >= 0) {
        counts[dirIdx]++;
        totalMoves++;
      }
    }

    if (totalMoves === 0)
      return { entropyNorm: 0, uniqueMoves: 0, pathLen: pathArr.length };

    // Compute normalized Shannon entropy and unique move count.
    let entropy = 0;
    let uniqueMoves = 0;
    // counts length is 4 (ACTION_DIM); iterate explicitly for tiny fixed size.
    for (let k = 0; k < 4; k++) {
      const c = counts[k];
      if (c > 0) {
        const p = c / totalMoves;
        entropy += -p * Math.log(p);
        uniqueMoves++;
      }
    }

    const entropyNorm = entropy * EvolutionEngine.#INV_LOG4;
    return { entropyNorm, uniqueMoves, pathLen: pathArr.length };
  }

  /**
   * Compute summary statistics over a sliding window of recent logit vectors.
   * Uses class-level scratch buffers to avoid allocations and supports a
   * reduced telemetry mode which skips higher moments (kurtosis).
   *
   * Contract:
   * - Input: `recent` is an array of numeric vectors (each vector length >= actionDim
   *   is not required; missing entries are treated as 0).
   * - Output: an object containing formatted strings, aggregated arrays and
   *   scalar summaries.
   *
   * @param recent - Array of recent logit vectors, newest last.
   * @returns An object { meansStr, stdsStr, kurtStr, entMean, stability, steps, means, stds }
   * @internal
   */
  static #computeLogitStats(recent: number[][]) {
    // Defensive input checks
    if (!Array.isArray(recent) || recent.length === 0)
      return {
        meansStr: '',
        stdsStr: '',
        kurtStr: '',
        entMean: 0,
        stability: 0,
        steps: 0,
        means: EvolutionEngine.#SCRATCH_MEANS,
        stds: EvolutionEngine.#SCRATCH_STDS,
      } as any;

    const reducedTelemetry = EvolutionEngine.#REDUCED_TELEMETRY;
    const actionDim = Math.max(0, EvolutionEngine.#ACTION_DIM);
    const meansBuf = EvolutionEngine.#SCRATCH_MEANS;
    const stdsBuf = EvolutionEngine.#SCRATCH_STDS;

    // Ensure higher-moment buffers exist when full telemetry is enabled
    if (!reducedTelemetry && !EvolutionEngine.#SCRATCH_KURT) {
      EvolutionEngine.#SCRATCH_KURT = new Float64Array(actionDim);
      EvolutionEngine.#SCRATCH_M3_RAW = new Float64Array(actionDim);
      EvolutionEngine.#SCRATCH_M4_RAW = new Float64Array(actionDim);
    }

    const kurtBuf = EvolutionEngine.#SCRATCH_KURT; // may be undefined in reduced mode
    const m2Buf = EvolutionEngine.#SCRATCH_M2_RAW;
    const m3Buf = EvolutionEngine.#SCRATCH_M3_RAW;
    const m4Buf = EvolutionEngine.#SCRATCH_M4_RAW;

    // Step 1: Reset the reusable buffers for the active dimension range.
    // Use .fill for clear intent and to keep the hot path allocation-free.
    meansBuf.fill(0, 0, actionDim);
    m2Buf.fill(0, 0, actionDim);
    stdsBuf.fill(0, 0, actionDim);
    // Only initialize/clear higher-order buffers when full telemetry is enabled.
    if (!reducedTelemetry) {
      m3Buf!.fill(0, 0, actionDim);
      m4Buf!.fill(0, 0, actionDim);
      kurtBuf!.fill(0, 0, actionDim);
    }

    const sampleCount = recent.length;
    let entropyAggregate = 0;

    // Step 2: Accumulate running statistics and entropy.
    // Reduced telemetry: compute only mean/std and entropy (skip higher moments to save CPU).
    if (reducedTelemetry) {
      for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
        // 2a: For each sample, perform online Welford-style updates for mean & M2.
        const vec = recent[sampleIndex] || ([] as number[]);
        const currentSampleNumber = sampleIndex + 1;
        for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
          const x = vec[dimIndex] ?? 0; // treat missing entries as 0
          const delta = x - meansBuf[dimIndex];
          // update mean incrementally: mean += delta / n
          meansBuf[dimIndex] += delta / currentSampleNumber;
          // accumulate second central moment contribution (M2)
          const delta2 = x - meansBuf[dimIndex];
          m2Buf[dimIndex] += delta * delta2;
        }
        // 2b: Also accumulate softmax entropy for the sample (kept in a separate scratch buffer).
        entropyAggregate += EvolutionEngine.#softmaxEntropyFromVector(
          vec,
          EvolutionEngine.#SCRATCH_EXPS
        );
      }
      for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
        const variance = m2Buf[dimIndex] / sampleCount;
        stdsBuf[dimIndex] = variance > 0 ? Math.sqrt(variance) : 0;
      }
    } else if (actionDim === 4) {
      // Step 2 (fast): Unrolled fast path for the common 4-action case to reduce loop overhead.
      // This is equivalent to the generic Welford-style accumulation but manually unrolled to avoid
      // per-dimension loop overhead and repeated bounds checks. The math follows Pébay's incremental
      // higher-moment update formulas (M2/M3/M4) to compute variance and excess kurtosis in one pass.
      let mean0 = 0,
        mean1 = 0,
        mean2 = 0,
        mean3 = 0;
      let M20 = 0,
        M21 = 0,
        M22 = 0,
        M23 = 0;
      let M30 = 0,
        M31 = 0,
        M32 = 0,
        M33 = 0;
      let M40 = 0,
        M41 = 0,
        M42 = 0,
        M43 = 0;

      for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
        const vec = recent[sampleIndex] || ([] as number[]);
        const x0 = vec[0] ?? 0;
        const x1 = vec[1] ?? 0;
        const x2 = vec[2] ?? 0;
        const x3 = vec[3] ?? 0;
        const currentSampleNumber = sampleIndex + 1;

        // Dimension 0
        let delta = x0 - mean0;
        let deltaN = delta / currentSampleNumber;
        let deltaN2 = deltaN * deltaN;
        let term1 = delta * deltaN * (currentSampleNumber - 1);
        M40 +=
          term1 *
            deltaN2 *
            (currentSampleNumber * currentSampleNumber -
              3 * currentSampleNumber +
              3) +
          6 * deltaN2 * M20 -
          4 * deltaN * M30;
        M30 += term1 * deltaN * (currentSampleNumber - 2) - 3 * deltaN * M20;
        M20 += term1;
        mean0 += deltaN;

        // Dimension 1
        delta = x1 - mean1;
        deltaN = delta / currentSampleNumber;
        deltaN2 = deltaN * deltaN;
        term1 = delta * deltaN * (currentSampleNumber - 1);
        M41 +=
          term1 *
            deltaN2 *
            (currentSampleNumber * currentSampleNumber -
              3 * currentSampleNumber +
              3) +
          6 * deltaN2 * M21 -
          4 * deltaN * M31;
        M31 += term1 * deltaN * (currentSampleNumber - 2) - 3 * deltaN * M21;
        M21 += term1;
        mean1 += deltaN;

        // Dimension 2
        delta = x2 - mean2;
        deltaN = delta / currentSampleNumber;
        deltaN2 = deltaN * deltaN;
        term1 = delta * deltaN * (currentSampleNumber - 1);
        M42 +=
          term1 *
            deltaN2 *
            (currentSampleNumber * currentSampleNumber -
              3 * currentSampleNumber +
              3) +
          6 * deltaN2 * M22 -
          4 * deltaN * M32;
        M32 += term1 * deltaN * (currentSampleNumber - 2) - 3 * deltaN * M22;
        M22 += term1;
        mean2 += deltaN;

        // Dimension 3
        delta = x3 - mean3;
        deltaN = delta / currentSampleNumber;
        deltaN2 = deltaN * deltaN;
        term1 = delta * deltaN * (currentSampleNumber - 1);
        M43 +=
          term1 *
            deltaN2 *
            (currentSampleNumber * currentSampleNumber -
              3 * currentSampleNumber +
              3) +
          6 * deltaN2 * M23 -
          4 * deltaN * M33;
        M33 += term1 * deltaN * (currentSampleNumber - 2) - 3 * deltaN * M23;
        M23 += term1;
        mean3 += deltaN;

        // 2c: Entropy integrated in the same pass to avoid a second loop over samples.
        entropyAggregate += EvolutionEngine.#softmaxEntropyFromVector(
          vec,
          EvolutionEngine.#SCRATCH_EXPS
        );
      }

      meansBuf[0] = mean0;
      meansBuf[1] = mean1;
      meansBuf[2] = mean2;
      meansBuf[3] = mean3;

      const invSample = 1 / sampleCount;
      const var0 = M20 * invSample;
      const var1 = M21 * invSample;
      const var2 = M22 * invSample;
      const var3 = M23 * invSample;
      stdsBuf[0] = var0 > 0 ? Math.sqrt(var0) : 0;
      stdsBuf[1] = var1 > 0 ? Math.sqrt(var1) : 0;
      stdsBuf[2] = var2 > 0 ? Math.sqrt(var2) : 0;
      stdsBuf[3] = var3 > 0 ? Math.sqrt(var3) : 0;

      if (!reducedTelemetry) {
        kurtBuf![0] = var0 > 1e-18 ? (sampleCount * M40) / (M20 * M20) - 3 : 0;
        kurtBuf![1] = var1 > 1e-18 ? (sampleCount * M41) / (M21 * M21) - 3 : 0;
        kurtBuf![2] = var2 > 1e-18 ? (sampleCount * M42) / (M22 * M22) - 3 : 0;
        kurtBuf![3] = var3 > 1e-18 ? (sampleCount * M43) / (M23 * M23) - 3 : 0;
      }
    } else {
      // Generic path supporting arbitrary action dimension
      for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
        const vec = recent[sampleIndex] || ([] as number[]);
        const currentSampleNumber = sampleIndex + 1;
        for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
          const x = vec[dimIndex] ?? 0;
          const delta = x - meansBuf[dimIndex];
          const deltaN = delta / currentSampleNumber;
          const deltaN2 = deltaN * deltaN;
          const term1 = delta * deltaN * (currentSampleNumber - 1);
          if (!reducedTelemetry)
            m4Buf![dimIndex] +=
              term1 *
                deltaN2 *
                (currentSampleNumber * currentSampleNumber -
                  3 * currentSampleNumber +
                  3) +
              6 * deltaN2 * m2Buf[dimIndex] -
              4 * deltaN * (m3Buf ? m3Buf[dimIndex] : 0);
          if (!reducedTelemetry)
            m3Buf![dimIndex] +=
              term1 * deltaN * (currentSampleNumber - 2) -
              3 * deltaN * m2Buf[dimIndex];
          m2Buf[dimIndex] += term1;
          meansBuf[dimIndex] += deltaN;
        }
        entropyAggregate += EvolutionEngine.#softmaxEntropyFromVector(
          vec,
          EvolutionEngine.#SCRATCH_EXPS
        );
      }

      for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
        const variance = m2Buf[dimIndex] / sampleCount;
        stdsBuf[dimIndex] = variance > 0 ? Math.sqrt(variance) : 0;
        if (!reducedTelemetry) {
          const m4v = m4Buf![dimIndex];
          kurtBuf![dimIndex] =
            variance > 1e-18
              ? (sampleCount * m4v) / (m2Buf[dimIndex] * m2Buf[dimIndex]) - 3
              : 0;
        }
      }
    }

    const entropyMean = entropyAggregate / sampleCount;
    const stability = EvolutionEngine.#computeDecisionStability(recent);
    const meansStr = EvolutionEngine.#joinNumberArray(meansBuf, actionDim, 3);
    const stdsStr = EvolutionEngine.#joinNumberArray(stdsBuf, actionDim, 3);
    const kurtStr = reducedTelemetry
      ? ''
      : EvolutionEngine.#joinNumberArray(kurtBuf!, actionDim, 2);

    return {
      meansStr,
      stdsStr,
      kurtStr,
      entMean: entropyMean,
      stability,
      steps: sampleCount,
      means: meansBuf,
      stds: stdsBuf,
    } as any;
  }

  /**
   * Compute summary statistics for output node biases.
   * Uses class-level scratch buffers and avoids intermediate allocations.
   * @param nodes - full node list from a network
   * @param outputCount - number of output nodes (must be <= nodes.length)
   * @returns an object with mean, std and a comma-separated biases string
   * @internal
   */
  static #computeOutputBiasStats(nodes: any[], outputCount: number) {
    if (!nodes || outputCount <= 0) return { mean: 0, std: 0, biasesStr: '' };
    // Single-pass Welford for mean/std
    let mean = 0;
    let M2 = 0;
    for (let outIndex = 0; outIndex < outputCount; outIndex++) {
      const nodeIndex = EvolutionEngine.#SCRATCH_NODE_IDX[outIndex];
      const biasValue = nodes[nodeIndex]?.bias ?? 0;
      const count = outIndex + 1;
      const delta = biasValue - mean;
      mean += delta / count;
      M2 += delta * (biasValue - mean);
    }
    const std = outputCount ? Math.sqrt(M2 / outputCount) : 0;
    // build compact biases string using reusable string buffer
    if (outputCount > EvolutionEngine.#SCRATCH_STR.length) {
      const nextSize = 1 << Math.ceil(Math.log2(outputCount));
      EvolutionEngine.#SCRATCH_STR = new Array(nextSize);
    }
    const sBuf = EvolutionEngine.#SCRATCH_STR;
    for (let bi = 0; bi < outputCount; bi++) {
      const idx = EvolutionEngine.#SCRATCH_NODE_IDX[bi];
      sBuf[bi] = (nodes[idx]?.bias ?? 0).toFixed(2);
    }
    const prevLenBias = sBuf.length;
    sBuf.length = outputCount;
    const biasesStr = sBuf.join(',');
    sBuf.length = prevLenBias;
    return { mean, std, biasesStr };
  }

  /**
   * Expand population by creating up to `targetAdd` children from top parents.
   * Uses neat-managed spawn when available, otherwise falls back to clone/mutate/add.
   * Avoids short-lived allocations by reusing class-level scratch buffers.
   * @internal
   */
  static #expandPopulation(
    neat: any,
    targetAdd: number,
    safeWrite: (msg: string) => void,
    completedGenerations: number
  ) {
    const populationRef = neat.population || [];
    const sortedIdx = EvolutionEngine.#getSortedIndicesByScore(populationRef);
    const parentCount = Math.max(
      2,
      Math.ceil(sortedIdx.length * EvolutionEngine.#DEFAULT_PARENT_FRACTION)
    );
    const parentPoolSize = Math.min(parentCount, sortedIdx.length);
    for (let addIndex = 0; addIndex < targetAdd; addIndex++) {
      const pickIndex = (EvolutionEngine.#fastRandom() * parentPoolSize) | 0;
      const parent = populationRef[sortedIdx[pickIndex]];
      try {
        if (typeof neat.spawnFromParent === 'function') {
          const mutateCount =
            1 +
            (EvolutionEngine.#fastRandom() < EvolutionEngine.#DEFAULT_HALF_PROB
              ? 1
              : 0);
          const child = neat.spawnFromParent(parent, mutateCount);
          neat.population.push(child);
        } else {
          const clone = parent.clone ? parent.clone() : parent;
          const mutateCount =
            1 +
            (EvolutionEngine.#fastRandom() < EvolutionEngine.#DEFAULT_HALF_PROB
              ? 1
              : 0);
          try {
            const mutOps = EvolutionEngine.#getMutationOps(neat);
            const opLen = mutOps.length | 0;
            if (opLen) {
              if (EvolutionEngine.#SCRATCH_MUTOP_IDX.length < opLen) {
                const nextSize = 1 << Math.ceil(Math.log2(opLen));
                EvolutionEngine.#SCRATCH_MUTOP_IDX = new Uint16Array(nextSize);
              }
              const idxBuf = EvolutionEngine.#SCRATCH_MUTOP_IDX;
              for (let fillIndex = 0; fillIndex < opLen; fillIndex++)
                idxBuf[fillIndex] = fillIndex;
              const applyCount = Math.min(mutateCount, opLen);

              // Apply mutation ops: small-count partial unroll, otherwise partial shuffle selection.
              if (applyCount <= 2) {
                if (applyCount >= 1) clone.mutate(mutOps[idxBuf[0]]);
                if (applyCount >= 2) clone.mutate(mutOps[idxBuf[1]]);
              } else {
                for (let pick = 0; pick < applyCount; pick++) {
                  const r =
                    pick +
                    ((EvolutionEngine.#fastRandom() * (opLen - pick)) | 0);
                  const tmp = idxBuf[pick];
                  idxBuf[pick] = idxBuf[r];
                  idxBuf[r] = tmp;
                  clone.mutate(mutOps[idxBuf[pick]]);
                }
              }
            }

            // Clear runtime-only score and attempt to register the clone with neat.
            clone.score = undefined;
            try {
              if (typeof neat.addGenome === 'function') {
                neat.addGenome(clone, [(parent as any)._id]);
              } else {
                if (typeof neat._invalidateGenomeCaches === 'function')
                  neat._invalidateGenomeCaches(clone);
                neat.population.push(clone);
              }
            } catch (e) {
              // Best-effort fallback: ensure clone is in population even if addGenome fails.
              try {
                neat.population.push(clone);
              } catch (e2) {
                /* swallow push failure */
              }
            }
          } catch (e) {
            /* ignore per-child failures */
          }
        }
      } catch (e) {
        /* ignore per-child failures */
      }
    }
    neat.options.popsize = neat.population.length;
    safeWrite(
      `[DYNAMIC_POP] Expanded population to ${neat.population.length} at gen ${completedGenerations}\n`
    );
  }

  /**
   * Return indices of population sorted descending by score using pooled index buffer.
   * Uses iterative quicksort on the indices to avoid allocating a copy of the population.
   * @internal
   */
  static #getSortedIndicesByScore(population: any[]): number[] {
    const len = population.length | 0;
    if (len === 0) return [];
    if (EvolutionEngine.#SCRATCH_SORT_IDX.length < len) {
      const nextSize = 1 << Math.ceil(Math.log2(len));
      EvolutionEngine.#SCRATCH_SORT_IDX = new Array(nextSize);
    }
    const idx = EvolutionEngine.#SCRATCH_SORT_IDX;
    for (let i = 0; i < len; i++) idx[i] = i;
    idx.length = len; // trim view
    // Iterative quicksort using pooled stack (each frame = lo,hi)
    let stack = EvolutionEngine.#SCRATCH_QS_STACK;
    if (stack.length < 2)
      stack = EvolutionEngine.#SCRATCH_QS_STACK = new Int32Array(128);
    let sp = 0; // stack pointer (next free slot)
    // push initial range
    stack[sp++] = 0;
    stack[sp++] = len - 1;
    while (sp > 0) {
      const hi = stack[--sp];
      const lo = stack[--sp];
      if (lo >= hi) continue;
      if (hi - lo <= EvolutionEngine.#QS_SMALL_THRESHOLD) {
        // small partition: insertion sort directly on idx slice
        for (let a = lo + 1; a <= hi; a++) {
          const iv = idx[a];
          const ivScore = population[iv]?.score ?? -Infinity;
          let b = a - 1;
          while (
            b >= lo &&
            (population[idx[b]]?.score ?? -Infinity) < ivScore
          ) {
            idx[b + 1] = idx[b];
            b--;
          }
          idx[b + 1] = iv;
        }
        continue;
      }
      let i = lo;
      let j = hi;
      // Median-of-three pivot selection to reduce degeneracy
      const mid = (lo + hi) >> 1;
      let aIndex = idx[lo];
      let bIndex = idx[mid];
      let cIndex = idx[hi];
      let aScore = population[aIndex]?.score ?? -Infinity;
      let bScore = population[bIndex]?.score ?? -Infinity;
      let cScore = population[cIndex]?.score ?? -Infinity;
      // order a,b
      if (aScore < bScore) {
        let tmp = aIndex;
        aIndex = bIndex;
        bIndex = tmp;
        let ts = aScore;
        aScore = bScore;
        bScore = ts;
      }
      // order a,c
      if (aScore < cScore) {
        let tmp = aIndex;
        aIndex = cIndex;
        cIndex = tmp;
        let ts = aScore;
        aScore = cScore;
        cScore = ts;
      }
      // order b,c (b now holds median after this ordering)
      if (bScore < cScore) {
        let tmp = bIndex;
        bIndex = cIndex;
        cIndex = tmp;
        let ts = bScore;
        bScore = cScore;
        cScore = ts;
      }
      const pivotIndex = bIndex;
      const pivotScore = bScore;
      while (i <= j) {
        while (true) {
          const li = idx[i];
          if ((population[li]?.score ?? -Infinity) <= pivotScore) break;
          i++;
        }
        while (true) {
          const rj = idx[j];
          if ((population[rj]?.score ?? -Infinity) >= pivotScore) break;
          j--;
        }
        if (i <= j) {
          const t = idx[i];
          idx[i] = idx[j];
          idx[j] = t;
          i++;
          j--;
        }
      }
      // push larger partition first to limit depth
      const leftSize = j - lo;
      const rightSize = hi - i;
      if (leftSize > rightSize) {
        if (lo < j) {
          if (sp + 2 > stack.length) {
            const bigger = new Int32Array(stack.length * 2);
            bigger.set(stack);
            EvolutionEngine.#SCRATCH_QS_STACK = stack = bigger;
          }
          stack[sp++] = lo;
          stack[sp++] = j;
        }
        if (i < hi) {
          if (sp + 2 > stack.length) {
            const bigger = new Int32Array(stack.length * 2);
            bigger.set(stack);
            EvolutionEngine.#SCRATCH_QS_STACK = stack = bigger;
          }
          stack[sp++] = i;
          stack[sp++] = hi;
        }
      } else {
        if (i < hi) {
          if (sp + 2 > stack.length) {
            const bigger = new Int32Array(stack.length * 2);
            bigger.set(stack);
            EvolutionEngine.#SCRATCH_QS_STACK = stack = bigger;
          }
          stack[sp++] = i;
          stack[sp++] = hi;
        }
        if (lo < j) {
          if (sp + 2 > stack.length) {
            const bigger = new Int32Array(stack.length * 2);
            bigger.set(stack);
            EvolutionEngine.#SCRATCH_QS_STACK = stack = bigger;
          }
          stack[sp++] = lo;
          stack[sp++] = j;
        }
      }
    }
    return idx;
  }

  /** Cached reference to mutation ops array (invalidated if reference changes). */
  static #CACHED_MUTATION_OPS: any[] | null = null;
  /** Return cached mutation operations array to avoid repeated options lookup in hot paths. @internal */
  static #getMutationOps(neat: any): any[] {
    try {
      const current = neat?.options?.mutation;
      if (current && EvolutionEngine.#CACHED_MUTATION_OPS !== current) {
        EvolutionEngine.#CACHED_MUTATION_OPS = current;
      }
      return (EvolutionEngine.#CACHED_MUTATION_OPS as any[]) || [];
    } catch {
      return [];
    }
  }

  /**
   * Ensure all output nodes use identity activation so caller can apply softmax externally.
   * @internal
   */
  static #ensureOutputIdentity(neat: any) {
    try {
      const popRef = neat.population || [];
      for (let gi = 0; gi < popRef.length; gi++) {
        const g: any = popRef[gi];
        const nodesRef = g.nodes || [];
        for (let ni = 0; ni < nodesRef.length; ni++) {
          const n: any = nodesRef[ni];
          if (n && n.type === 'output') n.squash = methods.Activation.identity;
        }
      }
    } catch {
      /* ignore */
    }
  }

  /**
   * Update global species history and escalate mutation/novelty params when species collapse detected.
   * Returns true when a collapse (20 consecutive single-species entries) was observed.
   * @internal
   */
  static #handleSpeciesHistory(neat: any): boolean {
    try {
      (EvolutionEngine as any)._speciesHistory =
        (EvolutionEngine as any)._speciesHistory || [];
      const populationList: any[] = (neat as any).population || [];
      // Reuse species scratch arrays to count unique species without Set allocation.
      let speciesIds = EvolutionEngine.#SCRATCH_SPECIES_IDS;
      let speciesCounts = EvolutionEngine.#SCRATCH_SPECIES_COUNTS;
      if (populationList.length > speciesIds.length) {
        const nextSize = 1 << Math.ceil(Math.log2(populationList.length));
        EvolutionEngine.#SCRATCH_SPECIES_IDS = new Int32Array(nextSize);
        EvolutionEngine.#SCRATCH_SPECIES_COUNTS = new Int32Array(nextSize);
        speciesIds = EvolutionEngine.#SCRATCH_SPECIES_IDS;
        speciesCounts = EvolutionEngine.#SCRATCH_SPECIES_COUNTS;
      }
      let uniqueSpeciesCount = 0;
      for (
        let genomeIndex = 0;
        genomeIndex < populationList.length;
        genomeIndex++
      ) {
        const genome = populationList[genomeIndex];
        if (!genome || genome.species == null) continue;
        const sid = genome.species | 0;
        let found = -1;
        for (
          let speciesIndex = 0;
          speciesIndex < uniqueSpeciesCount;
          speciesIndex++
        ) {
          if (speciesIds[speciesIndex] === sid) {
            found = speciesIndex;
            break;
          }
        }
        if (found === -1) {
          speciesIds[uniqueSpeciesCount] = sid;
          speciesCounts[uniqueSpeciesCount] = 1;
          uniqueSpeciesCount++;
        } else {
          speciesCounts[found]++;
        }
      }
      const speciesCount = uniqueSpeciesCount || 1;
      (EvolutionEngine as any)._speciesHistory = EvolutionEngine.#pushHistory<number>(
        (EvolutionEngine as any)._speciesHistory,
        speciesCount,
        EvolutionEngine.#SPECIES_HISTORY_MAX
      );
      const _speciesHistory: number[] =
        (EvolutionEngine as any)._speciesHistory || [];
      const recent: number[] = EvolutionEngine.#getTail<number>(
        _speciesHistory,
        EvolutionEngine.#SPECIES_COLLAPSE_WINDOW
      );
      const collapsed =
        recent.length === EvolutionEngine.#SPECIES_COLLAPSE_WINDOW &&
        recent.every((c: number) => c === 1);
      if (collapsed) {
        const neatAny: any = neat as any;
        if (typeof neatAny.mutationRate === 'number')
          neatAny.mutationRate = Math.min(
            EvolutionEngine.#COLLAPSE_MUTRATE_CAP,
            neatAny.mutationRate * EvolutionEngine.#COLLAPSE_MUTRATE_MULT
          );
        if (typeof neatAny.mutationAmount === 'number')
          neatAny.mutationAmount = Math.min(
            EvolutionEngine.#COLLAPSE_MUTAMOUNT_CAP,
            neatAny.mutationAmount * EvolutionEngine.#COLLAPSE_MUTAMOUNT_MULT
          );
        if (neatAny.config && neatAny.config.novelty) {
          neatAny.config.novelty.blendFactor = Math.min(
            EvolutionEngine.#COLLAPSE_NOVELTY_BLEND_CAP,
            neatAny.config.novelty.blendFactor *
              EvolutionEngine.#COLLAPSE_NOVELTY_MULT
          );
        }
      }
      return collapsed;
    } catch {
      return false;
    }
  }

  /**
   * Possibly expand the population when configured and plateau conditions met.
   * Delegates to #expandPopulation when growth is required.
   * @internal
   */
  static #maybeExpandPopulation(
    neat: any,
    dynamicPopEnabled: boolean,
    completedGenerations: number,
    dynamicPopMax: number,
    plateauGenerations: number,
    plateauCounter: number,
    dynamicPopExpandInterval: number,
    dynamicPopExpandFactor: number,
    dynamicPopPlateauSlack: number,
    safeWrite: (msg: string) => void
  ) {
    try {
      if (!dynamicPopEnabled || completedGenerations <= 0) return;
      if (!neat.population || neat.population.length >= dynamicPopMax) return;
      const plateauRatio =
        plateauGenerations > 0 ? plateauCounter / plateauGenerations : 0;
      const genTrigger = completedGenerations % dynamicPopExpandInterval === 0;
      if (genTrigger && plateauRatio >= dynamicPopPlateauSlack) {
        const currentSize = neat.population.length;
        const targetAdd = Math.min(
          Math.max(1, Math.floor(currentSize * dynamicPopExpandFactor)),
          dynamicPopMax - currentSize
        );
        if (targetAdd > 0)
          EvolutionEngine.#expandPopulation(
            neat,
            targetAdd,
            safeWrite,
            completedGenerations
          );
      }
    } catch {
      /* ignore */
    }
  }

  /**
   * Update dashboard when a new best network is found and yield to the frame when requested.
   * This helper is async because it may await `flushToFrame`.
   * @internal
   */
  static async #updateDashboardAndMaybeFlush(
    maze: any,
    result: any,
    network: any,
    completedGenerations: number,
    neat: any,
    dashboardManager: any,
    flushToFrame: () => Promise<void>
  ) {
    try {
      if (dashboardManager && typeof dashboardManager.update === 'function') {
        try {
          dashboardManager.update(
            maze,
            result,
            network,
            completedGenerations,
            neat
          );
        } catch {}
      }
      try {
        await flushToFrame();
      } catch {}
    } catch {}
  }

  /**
   * Periodic dashboard update for non-best case. Async to allow flush-to-frame.
   * @internal
   */
  static async #updateDashboardPeriodic(
    maze: any,
    bestResult: any,
    bestNetwork: any,
    completedGenerations: number,
    neat: any,
    dashboardManager: any,
    flushToFrame: () => Promise<void>
  ) {
    try {
      if (
        dashboardManager &&
        typeof dashboardManager.update === 'function' &&
        bestNetwork &&
        bestResult
      ) {
        try {
          dashboardManager.update(
            maze,
            bestResult,
            bestNetwork,
            completedGenerations,
            neat
          );
        } catch {}
        try {
          await flushToFrame();
        } catch {}
      }
    } catch {}
  }

  /**
   * Adjust output biases after local training using the same heuristic the original inline code used.
   * @internal
   */
  static #adjustOutputBiasesAfterTraining(network: any) {
    try {
      const nodesRef = network.nodes || [];
      const outCount = EvolutionEngine.#getNodeIndicesByType(
        nodesRef,
        'output'
      );
      if (outCount > 0) {
        // Welford mean/std in one pass
        let mean = 0;
        let M2 = 0;
        for (let outIndex = 0; outIndex < outCount; outIndex++) {
          const b = nodesRef[EvolutionEngine.#SCRATCH_NODE_IDX[outIndex]].bias;
          const n = outIndex + 1;
          const delta = b - mean;
          mean += delta / n;
          M2 += delta * (b - mean);
        }
        const std = outCount ? Math.sqrt(M2 / outCount) : 0;
        for (let oi = 0; oi < outCount; oi++) {
          const idx = EvolutionEngine.#SCRATCH_NODE_IDX[oi];
          let adjusted = nodesRef[idx].bias - mean;
          if (std < EvolutionEngine.#DEFAULT_STD_SMALL)
            adjusted *= EvolutionEngine.#DEFAULT_STD_ADJUST_MULT;
          nodesRef[idx].bias = Math.max(-5, Math.min(5, adjusted));
        }
      }
    } catch {
      /* ignore */
    }
  }

  /**
   * Warm-start helper: connect openness input bits and compass fan-out with light weights.
   * Mirrors the inlined pre-train adjustments performed after supervised training.
   * @internal
   */
  static #applyCompassWarmStart(net: any) {
    try {
      const nodesRef = net.nodes || [];
      const outCount = EvolutionEngine.#getNodeIndicesByType(
        nodesRef,
        'output'
      );
      const inCount = EvolutionEngine.#getNodeIndicesByType(nodesRef, 'input');
      for (let dirIndex = 0; dirIndex < 4; dirIndex++) {
        const inIdx =
          dirIndex + 1 < inCount
            ? EvolutionEngine.#SCRATCH_NODE_IDX[dirIndex + 1]
            : -1;
        const outIdx =
          dirIndex < outCount
            ? EvolutionEngine.#SCRATCH_NODE_IDX[dirIndex]
            : -1;
        const inNode = inIdx === -1 ? undefined : nodesRef[inIdx];
        const outNode = outIdx === -1 ? undefined : nodesRef[outIdx];
        if (!inNode || !outNode) continue;
        let conn: any = undefined;
        for (
          let connIndex = 0;
          connIndex < net.connections.length;
          connIndex++
        ) {
          const c = net.connections[connIndex];
          if (c.from === inNode && c.to === outNode) {
            conn = c;
            break;
          }
        }
        const w =
          EvolutionEngine.#fastRandom() * EvolutionEngine.#W_INIT_RANGE +
          EvolutionEngine.#W_INIT_MIN; // 0.55..0.8
        if (!conn) net.connect(inNode, outNode, w);
        else conn.weight = w;
      }
      const compassIdx =
        inCount > 0 ? EvolutionEngine.#SCRATCH_NODE_IDX[0] : -1;
      const compassNode = compassIdx === -1 ? undefined : nodesRef[compassIdx];
      if (compassNode) {
        for (let outIndex = 0; outIndex < outCount; outIndex++) {
          const outNode = nodesRef[EvolutionEngine.#SCRATCH_NODE_IDX[outIndex]];
          let conn: any = undefined;
          for (
            let connIndex = 0;
            connIndex < net.connections.length;
            connIndex++
          ) {
            const c = net.connections[connIndex];
            if (c.from === compassNode && c.to === outNode) {
              conn = c;
              break;
            }
          }
          const base =
            EvolutionEngine.#OUTPUT_BIAS_BASE +
            outIndex * EvolutionEngine.#OUTPUT_BIAS_STEP;
          if (!conn) net.connect(compassNode, outNode, base);
          else conn.weight = base;
        }
      }
    } catch {
      /* ignore */
    }
  }

  /**
   * Build the supervised training set used for Lamarckian warm-start training.
   * @returns Array of `{ input, output }` training cases.
   * @internal
   */
  static #buildLamarckianTrainingSet(): {
    input: number[];
    output: number[];
  }[] {
    const ds: { input: number[]; output: number[] }[] = [];
    const OUT = (direction: number) => {
      const out: number[] = [0, 0, 0, 0];
      for (let dirIndex = 0; dirIndex < 4; dirIndex++) {
        out[dirIndex] =
          dirIndex === direction
            ? EvolutionEngine.#TRAIN_OUT_PROB_HIGH
            : EvolutionEngine.#TRAIN_OUT_PROB_LOW;
      }
      return out;
    };
    const add = (inp: number[], dir: number) =>
      ds.push({ input: inp, output: OUT(dir) });

    // Single open path with good progress
    add([0, 1, 0, 0, 0, EvolutionEngine.#PROGRESS_MEDIUM], 0);
    add([0.25, 0, 1, 0, 0, EvolutionEngine.#PROGRESS_MEDIUM], 1);
    add([0.5, 0, 0, 1, 0, EvolutionEngine.#PROGRESS_MEDIUM], 2);
    add([0.75, 0, 0, 0, 1, EvolutionEngine.#PROGRESS_MEDIUM], 3);
    // Strong progress
    add([0, 1, 0, 0, 0, EvolutionEngine.#PROGRESS_STRONG], 0);
    add([0.25, 0, 1, 0, 0, EvolutionEngine.#PROGRESS_STRONG], 1);
    // Two-way junctions
    add([0, 1, 0.6, 0, 0, EvolutionEngine.#PROGRESS_JUNCTION], 0);
    add([0, 1, 0, 0.6, 0, EvolutionEngine.#PROGRESS_JUNCTION], 0);
    add([0.25, 0.6, 1, 0, 0, EvolutionEngine.#PROGRESS_JUNCTION], 1);
    add([0.25, 0, 1, 0.6, 0, EvolutionEngine.#PROGRESS_JUNCTION], 1);
    add([0.5, 0, 0.6, 1, 0, EvolutionEngine.#PROGRESS_JUNCTION], 2);
    add([0.5, 0, 0, 1, 0.6, EvolutionEngine.#PROGRESS_JUNCTION], 2);
    add([0.75, 0, 0, 0.6, 1, EvolutionEngine.#PROGRESS_JUNCTION], 3);
    add([0.75, 0.6, 0, 0, 1, EvolutionEngine.#PROGRESS_JUNCTION], 3);
    // Four-way junctions
    add([0, 1, 0.8, 0.5, 0.4, EvolutionEngine.#PROGRESS_FOURWAY], 0);
    add([0.25, 0.7, 1, 0.6, 0.5, EvolutionEngine.#PROGRESS_FOURWAY], 1);
    add([0.5, 0.6, 0.55, 1, 0.65, EvolutionEngine.#PROGRESS_FOURWAY], 2);
    add([0.75, 0.5, 0.45, 0.7, 1, EvolutionEngine.#PROGRESS_FOURWAY], 3);
    // Regressing cases
    add([0, 1, 0.3, 0, 0, EvolutionEngine.#PROGRESS_REGRESS], 0);
    add([0.25, 0.5, 1, 0.4, 0, EvolutionEngine.#PROGRESS_REGRESS], 1);
    add([0.5, 0, 0.3, 1, 0.2, EvolutionEngine.#PROGRESS_REGRESS], 2);
    add([0.75, 0, 0.5, 0.4, 1, EvolutionEngine.#PROGRESS_REGRESS], 3);
    add(
      [
        0,
        0,
        0,
        EvolutionEngine.#PROGRESS_MIN_SIGNAL,
        0,
        EvolutionEngine.#PROGRESS_MILD_REGRESS,
      ],
      2
    );

    // Mild augmentation (jitter openness & progress)
    for (let dsi = 0; dsi < ds.length; dsi++) {
      const caseEntry = ds[dsi];
      for (let dirIndex = 1; dirIndex <= 4; dirIndex++) {
        if (
          caseEntry.input[dirIndex] === 1 &&
          EvolutionEngine.#fastRandom() < EvolutionEngine.#DEFAULT_JITTER_PROB
        )
          caseEntry.input[dirIndex] =
            EvolutionEngine.#AUGMENT_JITTER_BASE +
            EvolutionEngine.#fastRandom() *
              EvolutionEngine.#AUGMENT_JITTER_RANGE;
      }
      if (
        EvolutionEngine.#fastRandom() <
        EvolutionEngine.#AUGMENT_PROGRESS_JITTER_PROB
      )
        caseEntry.input[5] = Math.min(
          1,
          Math.max(
            0,
            caseEntry.input[5] +
              (EvolutionEngine.#fastRandom() *
                EvolutionEngine.#AUGMENT_PROGRESS_DELTA_RANGE -
                EvolutionEngine.#AUGMENT_PROGRESS_DELTA_HALF)
          )
        );
    }
    return ds;
  }

  /**
   * Pretrain population with the provided supervised training set and apply warm-start heuristics.
   * @internal
   */
  static #pretrainPopulationWarmStart(neat: any, lamarckianTrainingSet: any[]) {
    try {
      const populationRef = neat.population || [];
      for (
        let networkIndex = 0;
        networkIndex < populationRef.length;
        networkIndex++
      ) {
        const net: any = populationRef[networkIndex];
        try {
          net.train(lamarckianTrainingSet, {
            iterations: Math.min(
              EvolutionEngine.#PRETRAIN_MAX_ITER,
              EvolutionEngine.#PRETRAIN_BASE_ITER +
                Math.floor(lamarckianTrainingSet.length / 2)
            ),
            error: EvolutionEngine.#DEFAULT_TRAIN_ERROR,
            rate: EvolutionEngine.#DEFAULT_PRETRAIN_RATE,
            momentum: EvolutionEngine.#DEFAULT_PRETRAIN_MOMENTUM,
            batchSize: EvolutionEngine.#DEFAULT_PRETRAIN_BATCH,
            allowRecurrent: true,
            cost: methods.Cost.softmaxCrossEntropy,
          });
          try {
            EvolutionEngine.#applyCompassWarmStart(net);
          } catch {}
          EvolutionEngine.#centerOutputBiases(net);
        } catch {
          /* ignore training errors */
        }
      }
    } catch {
      /* ignore */
    }
  }

  /**
   * Create a cooperative frame-yielding function used by the evolution loop.
   * @internal
   * @returns function that resolves on next frame / tick
   */
  static #makeFlushToFrame(): () => Promise<void> {
    return () => {
      const rafPromise = () =>
        new Promise<void>((resolve) =>
          (globalThis as any).requestAnimationFrame
            ? (globalThis as any).requestAnimationFrame(() => resolve())
            : setTimeout(() => resolve(), 0)
        );
      const immediatePromise = () =>
        new Promise<void>((resolve) =>
          typeof setImmediate === 'function'
            ? setImmediate(resolve)
            : setTimeout(resolve, 0)
        );

      if (
        typeof window !== 'undefined' &&
        typeof (window as any).requestAnimationFrame === 'function'
      ) {
        return new Promise<void>(async (resolve) => {
          const check = async () => {
            if ((window as any).asciiMazePaused) {
              await rafPromise();
              setTimeout(check, 0);
            } else {
              rafPromise().then(() => resolve());
            }
          };
          check();
        });
      }
      if (typeof setImmediate === 'function') {
        return new Promise<void>(async (resolve) => {
          const check = async () => {
            if ((globalThis as any).asciiMazePaused) {
              await immediatePromise();
              setTimeout(check, 0);
            } else {
              immediatePromise().then(() => resolve());
            }
          };
          check();
        });
      }
      return new Promise<void>((resolve) => setTimeout(resolve, 0));
    };
  }

  /**
   * Initialize persistence helpers (fs + path) and ensure directory exists if possible.
   * @internal
   */
  static #initPersistence(
    persistDir: string | undefined
  ): { fs: any; path: any } {
    let fs: any = null;
    let path: any = null;
    try {
      if (typeof window === 'undefined' && typeof require === 'function') {
        fs = require('fs');
        path = require('path');
      }
    } catch {}
    if (fs && persistDir && !fs.existsSync(persistDir)) {
      try {
        fs.mkdirSync(persistDir, { recursive: true });
      } catch (e) {
        /* ignore */
      }
    }
    return { fs, path };
  }

  /**
   * Build a safe writer function that tries Node stdout, dashboard logger, then console.log.
   * @internal
   */
  static #makeSafeWriter(dashboardManager: any): (msg: string) => void {
    return (msg: string) => {
      try {
        if (
          typeof process !== 'undefined' &&
          process &&
          process.stdout &&
          typeof process.stdout.write === 'function'
        ) {
          process.stdout.write(msg);
          return;
        }
      } catch {
        /* ignore */
      }
      try {
        if (dashboardManager && (dashboardManager as any).logFunction) {
          try {
            (dashboardManager as any).logFunction(msg);
            return;
          } catch {}
        }
      } catch {}
      if (typeof console !== 'undefined' && console.log)
        console.log(msg.trim());
    };
  }

  /**
   * Construct a configured Neat instance using the project's recommended defaults.
   * @internal
   * @param inputCount - number of network inputs
   * @param outputCount - number of network outputs
   * @param fitnessCallback - fitness evaluation callback
   * @param cfg - small configuration bag to tweak high-level features
   * @returns a new Neat instance
   */
  static #createNeat(
    inputCount: number,
    outputCount: number,
    fitnessCallback: (net: Network) => number,
    cfg: any
  ): any {
    const neatInstance = new Neat(inputCount, outputCount, fitnessCallback, {
      popsize: cfg.popSize || EvolutionEngine.#DEFAULT_POPSIZE,
      mutation: [
        methods.mutation.ADD_NODE,
        methods.mutation.SUB_NODE,
        methods.mutation.ADD_CONN,
        methods.mutation.SUB_CONN,
        methods.mutation.MOD_BIAS,
        methods.mutation.MOD_ACTIVATION,
        methods.mutation.MOD_CONNECTION,
        methods.mutation.ADD_LSTM_NODE,
      ],
      mutationRate: EvolutionEngine.#DEFAULT_MUTATION_RATE,
      mutationAmount: EvolutionEngine.#DEFAULT_MUTATION_AMOUNT,
      elitism: Math.max(
        1,
        Math.floor(
          (cfg.popSize || EvolutionEngine.#DEFAULT_POPSIZE) *
            EvolutionEngine.#DEFAULT_ELITISM_FRACTION
        )
      ),
      provenance: Math.max(
        1,
        Math.floor(
          (cfg.popSize || EvolutionEngine.#DEFAULT_POPSIZE) *
            EvolutionEngine.#DEFAULT_PROVENANCE_FRACTION
        )
      ),
      allowRecurrent: cfg.allowRecurrent !== false,
      minHidden: EvolutionEngine.#DEFAULT_MIN_HIDDEN,
      adaptiveMutation: cfg.adaptiveMutation || {
        enabled: true,
        strategy: 'twoTier',
      },
      multiObjective: cfg.multiObjective || {
        enabled: true,
        complexityMetric: 'nodes',
        autoEntropy: true,
      },
      telemetry: cfg.telemetry || {
        enabled: true,
        performance: true,
        complexity: true,
        hypervolume: true,
      },
      lineageTracking: cfg.lineageTracking === true,
      novelty: cfg.novelty || { enabled: true, blendFactor: 0.15 },
      targetSpecies:
        cfg.targetSpecies || EvolutionEngine.#DEFAULT_TARGET_SPECIES,
      adaptiveTargetSpecies: cfg.adaptiveTargetSpecies || {
        enabled: true,
        entropyRange: EvolutionEngine.#DEFAULT_ENTROPY_RANGE,
        speciesRange: [6, 14],
        smooth: EvolutionEngine.#DEFAULT_ADAPTIVE_SMOOTH,
      },
    });
    return neatInstance;
  }

  /**
   * Seed the NEAT population from optional initial population and an optional initial best network.
   * @internal
   */
  static #seedInitialPopulation(
    neat: any,
    initialPopulation: any[] | undefined,
    initialBestNetwork: any | undefined,
    targetPopSize: number
  ) {
    if (Array.isArray(initialPopulation) && initialPopulation.length > 0) {
      const srcLen = initialPopulation.length;
      if (EvolutionEngine.#SCRATCH_POP_CLONE.length < srcLen) {
        EvolutionEngine.#SCRATCH_POP_CLONE.length = srcLen;
      }
      const pooled = EvolutionEngine.#SCRATCH_POP_CLONE;
      for (let pi = 0; pi < srcLen; pi++) {
        pooled[pi] = (initialPopulation[pi] as Network).clone();
      }
      // Reuse pooled array directly; mark logical length and avoid slice allocation.
      pooled.length = srcLen;
      neat.population = pooled;
    }
    if (initialBestNetwork) {
      try {
        neat.population = neat.population || [];
        neat.population[0] = (initialBestNetwork as Network).clone();
      } catch {}
    }
    // ensure options.popsize reflects actual population length
    try {
      neat.options = neat.options || {};
      neat.options.popsize = neat.population
        ? neat.population.length
        : targetPopSize;
    } catch {}
  }

  /**
   * Check cooperative cancellation sources (legacy cancellation object or AbortSignal).
   * @internal
   */
  static #checkCancellation(options: any, bestResult: any): string | undefined {
    try {
      if (
        options?.cancellation &&
        typeof options.cancellation.isCancelled === 'function' &&
        options.cancellation.isCancelled()
      ) {
        if (bestResult) (bestResult as any).exitReason = 'cancelled';
        return 'cancelled';
      }
      if (options?.signal?.aborted) {
        if (bestResult) (bestResult as any).exitReason = 'aborted';
        return 'aborted';
      }
    } catch {}
    return undefined;
  }

  /**
   * Sample `k` items from `src` into the pooled SCRATCH_SAMPLE buffer (with replacement).
   * Returns the number of items written into the scratch buffer.
   * @internal
   * @remarks Non-reentrant: uses shared `#SCRATCH_SAMPLE` buffer.
   */
  static #sampleIntoScratch<T>(src: T[], k: number): number {
    if (!Array.isArray(src) || k <= 0) return 0;
    const sampleCount = Math.floor(k);
    const maxBuf = EvolutionEngine.#SCRATCH_SAMPLE;
    const srcLen = src.length || 0;
    if (srcLen === 0) return 0;
    const writeCount = Math.min(sampleCount, maxBuf.length);
    // Loop unrolled in blocks of 4 leveraging RNG cache refill cost amortization
    let wi = 0;
    const fastRand = EvolutionEngine.#fastRandom;
    const bound = writeCount & ~3; // largest multiple of 4
    while (wi < bound) {
      maxBuf[wi++] = src[(fastRand() * srcLen) | 0];
      maxBuf[wi++] = src[(fastRand() * srcLen) | 0];
      maxBuf[wi++] = src[(fastRand() * srcLen) | 0];
      maxBuf[wi++] = src[(fastRand() * srcLen) | 0];
    }
    while (wi < writeCount) {
      maxBuf[wi++] = src[(fastRand() * srcLen) | 0];
    }
    return writeCount;
  }

  /**
   * Sample up to k items (with replacement) from src segment [segmentStart, end) into scratch buffer.
   * Avoids temporary slice allocation for segment sampling.
   * @internal
   */
  static #sampleSegmentIntoScratch<T>(
    src: T[],
    segmentStart: number,
    k: number
  ): number {
    if (!Array.isArray(src) || k <= 0) return 0;
    const len = src.length | 0;
    if (segmentStart >= len) return 0;
    const segLen = len - segmentStart;
    if (segLen <= 0) return 0;
    const maxBuf = EvolutionEngine.#SCRATCH_SAMPLE;
    const writeCount = Math.min(Math.floor(k), maxBuf.length);
    let wi = 0;
    const fastRand = EvolutionEngine.#fastRandom;
    const base = segmentStart;
    const bound = writeCount & ~3;
    while (wi < bound) {
      maxBuf[wi++] = src[base + ((fastRand() * segLen) | 0)];
      maxBuf[wi++] = src[base + ((fastRand() * segLen) | 0)];
      maxBuf[wi++] = src[base + ((fastRand() * segLen) | 0)];
      maxBuf[wi++] = src[base + ((fastRand() * segLen) | 0)];
    }
    while (wi < writeCount) {
      maxBuf[wi++] = src[base + ((fastRand() * segLen) | 0)];
    }
    return writeCount;
  }

  /**
   * Run one generation: evolve, ensure output identity, update species history, maybe expand population,
   * and run Lamarckian training if configured.
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
    dynamicPopPlateauSlack: number
  ) {
    const t0 = doProfile ? EvolutionEngine.#now() : 0;
    const fittest = await neat.evolve();
    const tEvolve = doProfile ? EvolutionEngine.#now() - t0 : 0;
    EvolutionEngine.#ensureOutputIdentity(neat);
    EvolutionEngine.#handleSpeciesHistory(neat);
    EvolutionEngine.#maybeExpandPopulation(
      neat,
      dynamicPopEnabled,
      completedGenerations,
      dynamicPopMax,
      plateauGenerations,
      plateauCounter,
      dynamicPopExpandInterval,
      dynamicPopExpandFactor,
      dynamicPopPlateauSlack,
      safeWrite
    );
    let tLamarck = 0;
    if (
      lamarckianIterations > 0 &&
      lamarckianTrainingSet &&
      lamarckianTrainingSet.length
    ) {
      tLamarck = EvolutionEngine.#applyLamarckianTraining(
        neat,
        lamarckianTrainingSet,
        lamarckianIterations,
        lamarckianSampleSize,
        safeWrite,
        doProfile,
        completedGenerations
      );
    }
    return { fittest, tEvolve, tLamarck } as any;
  }

  /**
   * Update plateau detection state based on the latest fitness.
   * @internal
   */
  static #updatePlateauState(
    fitness: number,
    lastBestFitnessForPlateau: number,
    plateauCounter: number,
    plateauImprovementThreshold: number
  ): { plateauCounter: number; lastBestFitnessForPlateau: number } {
    if (fitness > lastBestFitnessForPlateau + plateauImprovementThreshold) {
      lastBestFitnessForPlateau = fitness;
      plateauCounter = 0;
    } else {
      plateauCounter++;
    }
    return { plateauCounter, lastBestFitnessForPlateau };
  }

  /**
   * Handle simplify entry and per-generation advance. Keeps caller variables small.
   * @internal
   */
  static #handleSimplifyState(
    neat: any,
    plateauCounter: number,
    plateauGenerations: number,
    simplifyDuration: number,
    simplifyMode: boolean,
    simplifyRemaining: number,
    simplifyStrategy: string,
    simplifyPruneFraction: number
  ): {
    simplifyMode: boolean;
    simplifyRemaining: number;
    plateauCounter: number;
  } {
    if (!simplifyMode) {
      const dur = EvolutionEngine.#maybeStartSimplify(
        plateauCounter,
        plateauGenerations,
        simplifyDuration
      );
      if (dur > 0) {
        simplifyMode = true;
        simplifyRemaining = dur;
        plateauCounter = 0;
      }
    }
    if (simplifyMode) {
      simplifyRemaining = EvolutionEngine.#runSimplifyCycle(
        neat,
        simplifyRemaining,
        simplifyStrategy,
        simplifyPruneFraction
      );
      if (simplifyRemaining <= 0) simplifyMode = false;
    }
    return { simplifyMode, simplifyRemaining, plateauCounter };
  }

  /**
   * Simulate the fittest network and perform post-simulation bookkeeping (attach telemetry, pruning, logging).
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
    neat: any
  ): { generationResult: any; simTime: number } {
    const t2 = doProfile ? EvolutionEngine.#now() : 0;
    const generationResult = MazeMovement.simulateAgent(
      fittest,
      encodedMaze,
      startPosition,
      exitPosition,
      distanceMap,
      maxSteps
    );
    try {
      // Initialize legacy array reference to ring buffer view if absent
      if (!(fittest as any)._lastStepOutputs) {
        (fittest as any)._lastStepOutputs =
          EvolutionEngine.#SCRATCH_LOGITS_RING;
      }
    } catch {}
    (fittest as any)._saturationFraction = generationResult.saturationFraction;
    (fittest as any)._actionEntropy = generationResult.actionEntropy;
    try {
      const stepOutputs: number[][] | undefined = (generationResult as any)
        .stepOutputs;
      if (Array.isArray(stepOutputs) && stepOutputs.length) {
        EvolutionEngine.#ensureLogitsRingCapacity(stepOutputs.length);
        if (
          EvolutionEngine.#LOGITS_RING_SHARED &&
          EvolutionEngine.#SCRATCH_LOGITS_SHARED &&
          EvolutionEngine.#SCRATCH_LOGITS_SHARED_W
        ) {
          const shared = EvolutionEngine.#SCRATCH_LOGITS_SHARED;
          const idxView = EvolutionEngine.#SCRATCH_LOGITS_SHARED_W;
          const capMask = EvolutionEngine.#LOGITS_RING_CAP - 1;
          const actionDim = EvolutionEngine.#ACTION_DIM;
          for (let si = 0; si < stepOutputs.length; si++) {
            const vec = stepOutputs[si];
            if (!Array.isArray(vec)) continue;
            const current = Atomics.load(idxView, 0) & capMask;
            const base = current * actionDim;
            const copyLen = Math.min(actionDim, vec.length);
            for (let di = 0; di < copyLen; di++)
              shared[base + di] = vec[di] ?? 0;
            Atomics.store(
              idxView,
              0,
              (Atomics.load(idxView, 0) + 1) & 0x7fffffff
            );
          }
        } else {
          for (let si = 0; si < stepOutputs.length; si++) {
            const vec = stepOutputs[si];
            if (!Array.isArray(vec)) continue;
            const w =
              EvolutionEngine.#SCRATCH_LOGITS_RING_W &
              (EvolutionEngine.#LOGITS_RING_CAP - 1);
            const target = EvolutionEngine.#SCRATCH_LOGITS_RING[w];
            const alen = Math.min(EvolutionEngine.#ACTION_DIM, vec.length);
            for (let ai = 0; ai < alen; ai++) target[ai] = vec[ai] ?? 0;
            EvolutionEngine.#SCRATCH_LOGITS_RING_W =
              (EvolutionEngine.#SCRATCH_LOGITS_RING_W + 1) & 0x7fffffff;
          }
        }
      }
    } catch {}
    if (
      generationResult.saturationFraction &&
      generationResult.saturationFraction >
        EvolutionEngine.#SATURATION_PRUNE_THRESHOLD
    ) {
      EvolutionEngine.#pruneSaturatedHiddenOutputs(fittest);
    }
    if (
      !EvolutionEngine.#TELEMETRY_MINIMAL &&
      completedGenerations % logEvery === 0
    ) {
      EvolutionEngine.#logGenerationTelemetry(
        neat,
        fittest,
        generationResult,
        completedGenerations,
        safeWrite
      );
    }
    const tDelta = doProfile ? EvolutionEngine.#now() - t2 : 0;
    return { generationResult, simTime: tDelta } as any;
  }

  /**
   * Check stop conditions (solved, stagnation, maxGenerations) and run dashboard updates/pausing when needed.
   * Returns a reason string when evolution should stop, otherwise undefined.
   * @internal
   */
  static async #checkStopConditions(
    bestResult: any,
    bestNetwork: any,
    maze: any,
    completedGenerations: number,
    neat: any,
    dashboardManager: any,
    flushToFrame: () => Promise<void>,
    minProgressToPass: number,
    autoPauseOnSolve: boolean,
    stopOnlyOnSolve: boolean,
    stagnantGenerations: number,
    maxStagnantGenerations: number,
    maxGenerations: number
  ): Promise<string | undefined> {
    // Solved
    if (bestResult?.success && bestResult.progress >= minProgressToPass) {
      if (bestNetwork && bestResult) {
        try {
          dashboardManager.update(
            maze,
            bestResult,
            bestNetwork,
            completedGenerations,
            neat
          );
        } catch {}
        try {
          await flushToFrame();
        } catch {}
      }
      if (autoPauseOnSolve) {
        try {
          if (typeof window !== 'undefined') {
            (window as any).asciiMazePaused = true;
            window.dispatchEvent(
              new CustomEvent('asciiMazeSolved', {
                detail: {
                  maze,
                  generations: completedGenerations,
                  progress: bestResult?.progress,
                },
              })
            );
          }
        } catch {}
      }
      if (bestResult) (bestResult as any).exitReason = 'solved';
      return 'solved';
    }
    // Stagnation
    if (
      !stopOnlyOnSolve &&
      stagnantGenerations >= maxStagnantGenerations &&
      isFinite(maxStagnantGenerations)
    ) {
      if (bestNetwork && bestResult) {
        try {
          dashboardManager.update(
            maze,
            bestResult,
            bestNetwork,
            completedGenerations,
            neat
          );
        } catch {}
        try {
          await flushToFrame();
        } catch {}
      }
      if (bestResult) (bestResult as any).exitReason = 'stagnation';
      return 'stagnation';
    }
    // Max generations
    if (
      !stopOnlyOnSolve &&
      completedGenerations >= maxGenerations &&
      isFinite(maxGenerations)
    ) {
      if (bestResult) (bestResult as any).exitReason = 'maxGenerations';
      return 'maxGenerations';
    }
    return undefined;
  }

  /**
   * Expose selected private helpers for test environment only.
   * @internal
   * @remarks Only returns accessors when NODE_ENV === 'test'.
   */
  static _testExpose() {
    try {
      if (
        typeof process !== 'undefined' &&
        process &&
        process.env &&
        process.env.NODE_ENV === 'test'
      ) {
        return {
          sampleArray: (s: any[], k: number) =>
            EvolutionEngine.#sampleArray(s, k),
          pruneWeakConnectionsForGenome: (g: any, s: string, f: number) =>
            EvolutionEngine.#pruneWeakConnectionsForGenome(g, s, f),
          computeLogitStats: (r: number[][]) =>
            EvolutionEngine.#computeLogitStats(r),
        };
      }
    } catch {}
    return undefined;
  }

  /**
   * Prune weak outgoing connections from hidden nodes to outputs when saturation is detected.
   * Mirrors the inline saturation pruning logic but centralized for reuse/testing.
   * @internal
   */
  static #pruneSaturatedHiddenOutputs(genome: any) {
    try {
      const t0 = EvolutionEngine.#PROFILE_ENABLED
        ? EvolutionEngine.#PROFILE_T0()
        : 0;
      const nodesRef = genome.nodes || [];
      const outCount = EvolutionEngine.#getNodeIndicesByType(
        nodesRef,
        'output'
      );
      const hiddenCount = EvolutionEngine.#getNodeIndicesByType(
        nodesRef,
        'hidden'
      );
      for (let hi = 0; hi < hiddenCount; hi++) {
        const hiddenNode =
          nodesRef[EvolutionEngine.#SCRATCH_NODE_IDX[outCount + hi]];
        const outs = EvolutionEngine.#collectHiddenToOutputConns(
          hiddenNode,
          nodesRef,
          outCount
        );
        const outsLen = outs.length;
        if (outsLen >= 2) {
          // Single-pass Welford on abs weights (cap by scratch length)
          const limit = Math.min(outsLen, EvolutionEngine.#SCRATCH_EXPS.length);
          let mean = 0;
          let M2 = 0;
          for (let wi = 0; wi < limit; wi++) {
            const w = Math.abs((outs[wi] as any).weight) || 0;
            const n = wi + 1;
            const delta = w - mean;
            mean += delta / n;
            M2 += delta * (w - mean);
          }
          const variance = limit ? M2 / limit : 0;
          if (mean < 0.5 && variance < EvolutionEngine.#NUMERIC_EPSILON_SMALL) {
            const disableCount = Math.max(1, Math.floor(outsLen / 2));
            const flags = EvolutionEngine.#SCRATCH_NODE_IDX;
            for (let fi = 0; fi < outsLen; fi++) flags[fi] = 0;
            for (let di = 0; di < disableCount; di++) {
              let minIdx = -1;
              let minW = Infinity;
              for (let j = 0; j < outsLen; j++) {
                if (flags[j]) continue;
                const conn = outs[j] as any;
                if (!conn || conn.enabled === false) {
                  flags[j] = 1;
                  continue;
                }
                const aw = Math.abs(conn.weight);
                if (aw < minW) {
                  minW = aw;
                  minIdx = j;
                }
              }
              if (minIdx >= 0) {
                (outs[minIdx] as any).enabled = false;
                flags[minIdx] = 1;
              } else break;
            }
            for (let fi = 0; fi < outsLen; fi++) flags[fi] = 0;
          }
        }
      }
      if (EvolutionEngine.#PROFILE_ENABLED) {
        EvolutionEngine.#PROFILE_ADD(
          'prune',
          EvolutionEngine.#PROFILE_T0() - t0 || 0
        );
      }
    } catch {
      /* soft-fail */
    }
  }

  /**
   * Anti-collapse recovery: reinitialize a fraction of non-elite population's output biases & weights.
   * @internal
   */
  static #antiCollapseRecovery(
    neat: any,
    completedGenerations: number,
    safeWrite: (msg: string) => void
  ) {
    try {
      const eliteCount = neat.options.elitism || 0;
      const pop = neat.population || [];
      const reinitBuf = EvolutionEngine.#SCRATCH_SAMPLE;
      const nonEliteStart = eliteCount | 0;
      const nonEliteLen = pop.length - nonEliteStart;
      const targetK = Math.min(
        nonEliteLen,
        Math.floor(nonEliteLen * 0.3),
        reinitBuf.length
      );
      const reinitLen = EvolutionEngine.#sampleSegmentIntoScratch(
        pop,
        nonEliteStart,
        targetK
      );
      let connReset = 0,
        biasReset = 0;
      for (let rti = 0; rti < reinitLen; rti++) {
        const g: any = reinitBuf[rti];
        const deltas = EvolutionEngine.#reinitializeGenomeOutputsAndWeights(g);
        connReset += deltas.connReset;
        biasReset += deltas.biasReset;
      }
      safeWrite(
        `[ANTICOLLAPSE] gen=${completedGenerations} reinitGenomes=${reinitLen} connReset=${connReset} biasReset=${biasReset}\n`
      );
    } catch {
      /* ignore */
    }
  }

  /**
   * Reinitialize output node biases and weights targeting outputs for a single genome.
   * Returns counts of modified connections and biases.
   * @internal
   */
  static #reinitializeGenomeOutputsAndWeights(
    genome: any
  ): { connReset: number; biasReset: number } {
    const nodesList = genome.nodes || [];
    let outputsLen = 0;
    const sampleBuf = EvolutionEngine.#SCRATCH_SAMPLE;
    for (let ni = 0; ni < nodesList.length; ni++) {
      const n = nodesList[ni];
      if (n && n.type === 'output') {
        if (outputsLen < sampleBuf.length) sampleBuf[outputsLen++] = n;
      }
    }
    let biasReset = 0;
    for (let oi = 0; oi < outputsLen; oi++) {
      (sampleBuf[oi] as any).bias =
        EvolutionEngine.#fastRandom() *
          (2 * EvolutionEngine.#BIAS_RESET_HALF_RANGE) -
        EvolutionEngine.#BIAS_RESET_HALF_RANGE;
      biasReset++;
    }
    let connReset = 0;
    const conns = genome.connections || [];
    for (let ci = 0; ci < conns.length; ci++) {
      const c = conns[ci];
      for (let oi = 0; oi < outputsLen; oi++) {
        if (c.to === sampleBuf[oi]) {
          c.weight =
            EvolutionEngine.#fastRandom() *
              (2 * EvolutionEngine.#CONN_WEIGHT_RESET_HALF_RANGE) -
            EvolutionEngine.#CONN_WEIGHT_RESET_HALF_RANGE;
          connReset++;
          break;
        }
      }
    }
    return { connReset, biasReset };
  }
  /** Compact one genome's disabled connections in-place. @internal */
  static #compactGenomeConnections(genome: any): number {
    try {
      const list: any[] = genome.connections || [];
      let write = 0;
      for (let read = 0; read < list.length; read++) {
        const c = list[read];
        if (c && c.enabled !== false) {
          if (read !== write) list[write] = c;
          write++;
        }
      }
      const removed = list.length - write;
      if (removed > 0) list.length = write;
      return removed;
    } catch {
      return 0;
    }
  }
  /** Compact entire population; returns total removed disabled connections. @internal */
  static #compactPopulation(neat: any): number {
    try {
      const pop: any[] = neat.population || [];
      let total = 0;
      for (let i = 0; i < pop.length; i++)
        total += EvolutionEngine.#compactGenomeConnections(pop[i]);
      return total;
    } catch {
      return 0;
    }
  }
  /** Shrink oversize scratch buffers after compaction if they exceed heuristic threshold. @internal */
  static #maybeShrinkScratch(neat: any) {
    try {
      const popSize = (neat.population && neat.population.length) || 0;
      if (popSize && EvolutionEngine.#SCRATCH_SORT_IDX.length > popSize * 8) {
        const nextSize = 1 << Math.ceil(Math.log2(Math.max(8, popSize)));
        EvolutionEngine.#SCRATCH_SORT_IDX = new Array(nextSize);
      }
    } catch {
      /* ignore */
    }
  }

  /**
   * Runs the NEAT neuro-evolution process for an agent to solve a given ASCII maze.
   *
   * This is the core function of the `EvolutionEngine`. It sets up and runs the evolutionary
   * algorithm to train a population of neural networks. Each network acts as the "brain" for an
   * agent, controlling its movement through the maze from a start point 'S' to an exit 'E'.
   *
   * The process involves several key steps:
   * 1.  **Initialization**: Sets up the maze, NEAT parameters, and the initial population of networks.
   * 2.  **Generational Loop**: Iterates through generations, performing the following for each:
   *     a. **Evaluation**: Each network's performance (fitness) is measured by how well its agent navigates the maze.
   *        Fitness is typically based on progress towards the exit, speed, and efficiency.
   *     b. **Lamarckian Refinement**: Each individual in the population undergoes a brief period of supervised training
   *        (backpropagation) on a set of ideal sensory-action pairs. This helps to fine-tune promising behaviors.
   *     c. **Selection & Reproduction**: The NEAT algorithm selects the fittest individuals to become parents for the
   *        next generation. It uses genetic operators (crossover and mutation) to create offspring.
   * 3.  **Termination**: The loop continues until a solution is found (an agent successfully reaches the exit) or other
   *     stopping criteria are met (e.g., maximum generations, stagnation).
   *
   * This hybrid approach, combining the global search of evolution with the local search of backpropagation,
   * can significantly accelerate learning and lead to more robust solutions.
   *
   * @param options - A comprehensive configuration object for the maze evolution process.
   * @returns A Promise that resolves with an object containing the best network found, its simulation result, and the final NEAT instance.
   */
  static async runMazeEvolution(options: IRunMazeEvolutionOptions) {
    // --- Step 1: Destructure and Default Configuration ---
    // Extract all the necessary configuration objects from the main options parameter.
    const {
      mazeConfig,
      agentSimConfig,
      evolutionAlgorithmConfig,
      reportingConfig,
      fitnessEvaluator,
    } = options;
    const { maze } = mazeConfig;
    const {
      logEvery = 10,
      dashboardManager,
      paceEveryGeneration,
    } = reportingConfig as any;

    // Extract evolution parameters, providing sensible defaults for any that are not specified.
    const {
      allowRecurrent = true, // Allow networks to have connections that loop back, enabling memory.
      popSize = 500, // The number of neural networks in each generation.
      maxStagnantGenerations = 500, // Stop evolution if the best fitness doesn't improve for this many generations.
      minProgressToPass = 95, // The percentage of progress required to consider the maze "solved".
      maxGenerations = Infinity, // A safety cap on the total number of generations to prevent infinite loops.
      randomSeed, // An optional seed for the random number generator to ensure reproducible results.
      initialPopulation, // An optional population of networks to start with.
      initialBestNetwork, // An optional pre-trained network to seed the population.
      lamarckianIterations = 10, // The number of backpropagation steps for each individual per generation.
      lamarckianSampleSize, // If set, use a random subset of the training data for Lamarckian learning.
      plateauGenerations = 40, // Number of generations to wait for improvement before considering the population to be on a plateau.
      plateauImprovementThreshold = 1e-6, // The minimum fitness improvement required to reset the plateau counter.
      simplifyDuration = 30, // The number of generations to run the network simplification process.
      simplifyPruneFraction = 0.05, // The fraction of weak connections to prune during simplification.
      simplifyStrategy = 'weakWeight', // The strategy for choosing which connections to prune.
      persistEvery = 25, // Save a snapshot of the best networks every N generations.
      persistDir = './ascii_maze_snapshots', // The directory to save snapshots in.
      persistTopK = 3, // The number of top-performing networks to save in each snapshot.
      dynamicPopEnabled = true, // Enable dynamic adjustment of the population size.
      dynamicPopMax: dynamicPopMaxCfg, // The maximum population size for dynamic adjustments.
      dynamicPopExpandInterval = 25, // The number of generations between population size expansions.
      dynamicPopExpandFactor = 0.15, // The factor by which to expand the population size.
      dynamicPopPlateauSlack = 0.6, // A slack factor for plateau detection when dynamic population is enabled.
      stopOnlyOnSolve = false, // If true, ignore stagnation/maxGenerations and run until solved.
      autoPauseOnSolve = true, // Browser demo will override to false to keep running continuously
      deterministic = false,
      memoryCompactionInterval = 50,
      telemetryReduceStats = false,
      telemetryMinimal = false,
      disableBaldwinianRefinement = false,
    } = evolutionAlgorithmConfig;
    // Deterministic seeding if requested
    if (deterministic || typeof randomSeed === 'number') {
      EvolutionEngine.setDeterministic(
        typeof randomSeed === 'number' ? randomSeed : 0x12345678
      );
    }
    EvolutionEngine.#REDUCED_TELEMETRY = !!telemetryReduceStats;
    EvolutionEngine.#TELEMETRY_MINIMAL = !!telemetryMinimal;
    EvolutionEngine.#DISABLE_BALDWIN = !!disableBaldwinianRefinement;

    // Determine the maximum population size, with a fallback if not explicitly configured.
    const dynamicPopMax =
      typeof dynamicPopMaxCfg === 'number'
        ? dynamicPopMaxCfg
        : Math.max(popSize, 120);

    // --- Step 2: Maze and Environment Setup ---
    // Encode the maze into a numerical format (0 for walls, 1 for paths) for efficient processing.
    const encodedMaze = MazeUtils.encodeMaze(maze);
    // Locate the starting 'S' and exit 'E' positions within the maze.
    const startPosition = MazeUtils.findPosition(maze, 'S');
    const exitPosition = MazeUtils.findPosition(maze, 'E');
    // Pre-calculate the distance from every point in the maze to the exit. This is a crucial
    // optimization and provides a rich source of information for the fitness function.
    const distanceMap = MazeUtils.buildDistanceMap(encodedMaze, exitPosition);

    // Define the structure of the neural network: 6 inputs and 4 outputs.
    // Inputs: [compassScalar, openN, openE, openS, openW, progressDelta]
    // Outputs: [moveN, moveE, moveS, moveW]
    const inputSize = 6;
    const outputSize = 4;

    // Select the fitness evaluator function. Use the provided one or a default.
    const currentFitnessEvaluator =
      fitnessEvaluator || FitnessEvaluator.defaultFitnessEvaluator;

    // --- Step 3: Fitness Evaluation Context ---
    // Bundle all the necessary environmental data into a context object. This object will be
    // passed to the fitness function, so it has all the information it needs to evaluate a network.
    const fitnessContext: IFitnessEvaluationContext = {
      encodedMaze,
      startPosition,
      exitPosition,
      agentSimConfig,
      distanceMap,
    };

    // Create the fitness callback function that NEAT will use. This function takes a network,
    // runs the simulation, and returns a single numerical score representing its fitness.
    const neatFitnessCallback = (network: Network): number => {
      return currentFitnessEvaluator(network, fitnessContext);
    };

    // --- Step 4: NEAT Algorithm Initialization ---
    // Create and seed the NEAT instance.
    const neat = EvolutionEngine.#createNeat(
      inputSize,
      outputSize,
      neatFitnessCallback,
      {
        popSize,
        allowRecurrent,
        adaptiveMutation: { enabled: true, strategy: 'twoTier' },
        multiObjective: {
          enabled: true,
          complexityMetric: 'nodes',
          autoEntropy: true,
        },
        telemetry: {
          enabled: true,
          performance: true,
          complexity: true,
          hypervolume: true,
        },
        lineageTracking: true,
        novelty: { enabled: true, blendFactor: 0.15 },
        targetSpecies: 10,
        adaptiveTargetSpecies: {
          enabled: true,
          entropyRange: [0.3, 0.8],
          speciesRange: [6, 14],
          smooth: 0.5,
        },
      }
    );
    EvolutionEngine.#seedInitialPopulation(
      neat,
      initialPopulation,
      initialBestNetwork,
      popSize
    );

    // --- Step 5: Evolution State Tracking ---
    // Initialize variables to track the progress of the evolution.
    let bestNetwork: INetwork | undefined =
      evolutionAlgorithmConfig.initialBestNetwork;
    let bestFitness = -Infinity;
    let bestResult: any;
    let stagnantGenerations = 0;
    let completedGenerations = 0;
    let plateauCounter = 0;
    let simplifyMode = false;
    let simplifyRemaining = 0;
    let lastBestFitnessForPlateau = -Infinity;

    // --- Step 6: Filesystem, Persistence and Helpers Setup ---
    const { fs, path } = EvolutionEngine.#initPersistence(persistDir);
    const flushToFrame = EvolutionEngine.#makeFlushToFrame();

    // --- Step 7: Lamarckian Learning Setup ---
    // Define the supervised training set for the Lamarckian refinement process.
    // This dataset consists of idealized sensory inputs and the corresponding optimal actions.
    // It helps to quickly teach the networks basic, correct behaviors.
    /**
     * @const {Array<Object>} lamarckianTrainingSet
     * Encodes idealized agent perceptions and the optimal action for each case.
     * This is used for local search (backpropagation) to refine networks between generations.
     *
     * Input format: `[compassScalar, openN, openE, openS, openW, progressDelta]`
     * - `compassScalar`: Direction to the exit (0=N, 0.25=E, 0.5=S, 0.75=W).
     * - `openN/E/S/W`: Whether the path is open in that direction (1=open, 0=wall).
     * - `progressDelta`: Change in distance to the exit ( >0.5 is good, <0.5 is bad).
     *
     * Output format: A one-hot encoded array representing the desired move `[N, E, S, W]`.
     */
    const lamarckianTrainingSet = EvolutionEngine.#buildLamarckianTrainingSet();

    // --- Pre-train generation 0 population on supervised compass dataset (Lamarckian warm start) ---
    if (lamarckianTrainingSet.length) {
      EvolutionEngine.#pretrainPopulationWarmStart(neat, lamarckianTrainingSet);
    }

    // Lightweight profiling (opt-in): set env ASCII_MAZE_PROFILE=1 to enable
    const doProfile =
      typeof process !== 'undefined' &&
      typeof process.env !== 'undefined' &&
      process.env.ASCII_MAZE_PROFILE === '1';
    let tEvolveTotal = 0;
    let tLamarckTotal = 0;
    let tSimTotal = 0;
    let lastCompactionGen = 0;

    // Safe writer: prefer Node stdout when available, else dashboard logger, else console.log
    const safeWrite = EvolutionEngine.#makeSafeWriter(dashboardManager);

    while (true) {
      // Cooperative cancellation check
      const cancelReason = EvolutionEngine.#checkCancellation(
        options,
        bestResult
      );
      if (cancelReason) break;

      // Run generation-level work (evolve, species/dynamic-pop, and optional Lamarckian training)
      const genRes = await EvolutionEngine.#runGeneration(
        neat,
        doProfile,
        lamarckianIterations,
        lamarckianTrainingSet,
        lamarckianSampleSize,
        safeWrite,
        completedGenerations,
        dynamicPopEnabled,
        dynamicPopMax,
        plateauGenerations,
        plateauCounter,
        dynamicPopExpandInterval,
        dynamicPopExpandFactor,
        dynamicPopPlateauSlack
      );
      const fittest = genRes.fittest;
      if (doProfile) {
        tEvolveTotal += genRes.tEvolve || 0;
        tLamarckTotal += genRes.tLamarck || 0;
      }

      // 3. Baldwinian refinement: further train the fittest individual for evaluation only.
      //    This improves its performance for this generation's evaluation, but only the Lamarckian-trained
      //    weights are inherited by offspring. (If you want pure Lamarckian, remove this step.)
      // Baldwinian refinement phase: extra training applied only to the current
      // generation's fittest individual to improve evaluation performance.
      // These refinements are NOT inherited (keeps genetic search exploratory),
      // unlike Lamarckian updates earlier. Disable if pure Lamarckian desired.
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
          /* ignore refinement failure */
        }
      }

      // 4. Evaluate and track progress
      const fitness = fittest.score ?? 0;
      completedGenerations++;

      // Plateau detection logic
      ({
        plateauCounter,
        lastBestFitnessForPlateau,
      } = EvolutionEngine.#updatePlateauState(
        fitness,
        lastBestFitnessForPlateau,
        plateauCounter,
        plateauImprovementThreshold
      ));

      // Simplify handling
      ({
        simplifyMode,
        simplifyRemaining,
      } = EvolutionEngine.#handleSimplifyState(
        neat,
        plateauCounter,
        plateauGenerations,
        simplifyDuration,
        simplifyMode,
        simplifyRemaining,
        simplifyStrategy,
        simplifyPruneFraction
      ));

      // Simulate and postprocess
      const simRes = EvolutionEngine.#simulateAndPostprocess(
        fittest,
        encodedMaze,
        startPosition,
        exitPosition,
        distanceMap,
        agentSimConfig.maxSteps,
        doProfile,
        safeWrite,
        logEvery,
        completedGenerations,
        neat
      );
      const generationResult = simRes.generationResult;
      if (doProfile) tSimTotal += simRes.simTime;

      // If new best, update tracking and dashboard
      if (fitness > bestFitness) {
        bestFitness = fitness;
        bestNetwork = fittest;
        bestResult = generationResult;
        stagnantGenerations = 0;
        await EvolutionEngine.#updateDashboardAndMaybeFlush(
          maze,
          generationResult,
          fittest,
          completedGenerations,
          neat,
          dashboardManager,
          flushToFrame
        );
      } else {
        stagnantGenerations++;
        if (completedGenerations % logEvery === 0) {
          await EvolutionEngine.#updateDashboardPeriodic(
            maze,
            bestResult,
            bestNetwork,
            completedGenerations,
            neat,
            dashboardManager,
            flushToFrame
          );
        }
      }

      // Persistence snapshot
      EvolutionEngine.#persistSnapshotIfNeeded(
        fs,
        path,
        persistDir,
        persistTopK,
        completedGenerations,
        persistEvery,
        neat,
        bestFitness,
        simplifyMode,
        plateauCounter
      );
      // Check stop conditions via helper (solved, stagnation, max generations)
      const stopReason = await EvolutionEngine.#checkStopConditions(
        bestResult,
        bestNetwork,
        maze,
        completedGenerations,
        neat,
        dashboardManager,
        flushToFrame,
        minProgressToPass,
        autoPauseOnSolve,
        stopOnlyOnSolve,
        stagnantGenerations,
        maxStagnantGenerations,
        maxGenerations
      );
      if (stopReason) break;
      // Periodic memory compaction: remove disabled connections & optionally shrink scratch arrays
      if (
        memoryCompactionInterval > 0 &&
        completedGenerations - lastCompactionGen >= memoryCompactionInterval
      ) {
        const removedDisabled = EvolutionEngine.#compactPopulation(neat);
        if (removedDisabled > 0) {
          EvolutionEngine.#maybeShrinkScratch(neat);
          safeWrite(
            `[COMPACT] gen=${completedGenerations} removedDisabledConns=${removedDisabled}\n`
          );
        }
        lastCompactionGen = completedGenerations;
      }
      // Optional per-generation pacing to yield back to browser for smoother UI (demo mode)
      if (paceEveryGeneration) {
        try {
          await flushToFrame();
        } catch {
          /* ignore pacing errors */
        }
      }
    }

    if (doProfile && completedGenerations > 0) {
      const gen = completedGenerations;
      const avgEvolve = (tEvolveTotal / gen).toFixed(2);
      const avgLamarck = (tLamarckTotal / gen).toFixed(2);
      const avgSim = (tSimTotal / gen).toFixed(2);
      // Direct stdout to avoid jest buffering suppression
      safeWrite(
        `\n[PROFILE] Generations=${gen} avg(ms): evolve=${avgEvolve} lamarck=${avgLamarck} sim=${avgSim} totalPerGen=${(
          +avgEvolve +
          +avgLamarck +
          +avgSim
        ).toFixed(2)}\n`
      );
      if (EvolutionEngine.#PROFILE_ENABLED) {
        const det = EvolutionEngine.#PROFILE_ACCUM;
        const denom = gen || 1;
        safeWrite(
          `[PROFILE_DETAIL] avgTelemetry=${(det.telemetry / denom).toFixed(
            2
          )} avgSimplify=${(det.simplify / denom).toFixed(2)} avgSnapshot=${(
            det.snapshot / denom
          ).toFixed(2)} avgPrune=${(det.prune / denom).toFixed(2)}\n`
        );
      }
    }

    // Return the best network, its result, and the NEAT instance
    return {
      bestNetwork,
      bestResult,
      neat,
      exitReason: (bestResult as any)?.exitReason ?? 'incomplete',
    };
  }

  /**
   * Prints the structure of a given neural network to the console.
   *
   * This is useful for debugging and understanding the evolved architectures.
   * It prints the number of nodes, their types, activation functions, and connection details.
   *
   * @param network - The neural network to inspect.
   * @returns void
   */
  static printNetworkStructure(network: INetwork) {
    // Print high-level network structure and statistics
    console.log('Network Structure:');
    console.log('Nodes: ', network.nodes?.length); // Total number of nodes
    const inputNodes: any[] = [];
    const outputNodes: any[] = [];
    const hiddenNodes: any[] = [];
    const nodeList = network.nodes || [];
    for (let nodeIndex = 0; nodeIndex < nodeList.length; nodeIndex++) {
      const node = nodeList[nodeIndex];
      if (!node) continue;
      if (node.type === 'input') inputNodes.push(node);
      else if (node.type === 'output') outputNodes.push(node);
      else if (node.type === 'hidden') hiddenNodes.push(node);
    }
    console.log('Input nodes: ', inputNodes?.length); // Number of input nodes
    console.log('Hidden nodes: ', hiddenNodes?.length); // Number of hidden nodes
    console.log('Output nodes: ', outputNodes?.length); // Number of output nodes
    // Format activation function names without intermediate array
    const nodesList = network.nodes || [];
    if (EvolutionEngine.#SCRATCH_ACT_NAMES.length < nodesList.length) {
      EvolutionEngine.#SCRATCH_ACT_NAMES.length = nodesList.length;
    }
    const actNames = EvolutionEngine.#SCRATCH_ACT_NAMES;
    for (let ni = 0; ni < nodesList.length; ni++) {
      const n: any = nodesList[ni];
      actNames[ni] = n?.squash?.name || String(n?.squash);
    }
    actNames.length = nodesList.length;
    // actNames already trimmed; log directly without slice
    console.log('Activation functions: ', actNames);
    console.log('Connections: ', network.connections?.length); // Number of connections
    // Whether there are recurrent/gated connections (avoid .some callback allocation)
    let hasRecurrent = false;
    const connsList = network.connections || [];
    for (let connIndex = 0; connIndex < connsList.length; connIndex++) {
      const c = connsList[connIndex];
      if (c && (c.gater || c.from === c.to)) {
        hasRecurrent = true;
        break;
      }
    }
    console.log('Has recurrent/gated connections: ', hasRecurrent);
  }
}
