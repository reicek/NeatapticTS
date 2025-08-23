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
   * Compute simple exploration statistics from a path: unique visited cells and ratio.
   * @internal
   */
  static #computeExplorationStats(
    path: ReadonlyArray<[number, number]>
  ): { unique: number; pathLen: number; ratio: number } {
    const pathLength = path?.length || 0;
    if (!pathLength) return { unique: 0, pathLen: 0, ratio: 0 };
    // Strategy: use open-addressed hash to get O(n) instead of O(n^2) worst-case.
    // For short paths (< 32) the previous quadratic scan is cheap, so keep a fast path.
    if (pathLength < 32) {
      // Tiny open-address table (size 64) with linear probing; stores packed+1 (0 = empty).
      const table = EvolutionEngine.#SMALL_EXPLORE_TABLE;
      table.fill(0);
      let distinctTiny = 0;
      const maskTiny = table.length - 1; // 63
      for (let pathIndex = 0; pathIndex < pathLength; pathIndex++) {
        const point = path[pathIndex];
        const packed = ((point[0] & 0xffff) << 16) | (point[1] & 0xffff);
        let h = Math.imul(packed, 2654435761) >>> 0;
        const storeVal = (packed + 1) | 0;
        while (true) {
          const slot = h & maskTiny;
          const v = table[slot];
          if (v === 0) {
            table[slot] = storeVal;
            distinctTiny++;
            break;
          }
          if (v === storeVal) break; // duplicate
          h = (h + 1) | 0;
        }
      }
      return {
        unique: distinctTiny,
        pathLen: pathLength,
        ratio: distinctTiny ? distinctTiny / pathLength : 0,
      };
    }
    // Hash path
    const targetCap = pathLength << 1; // aim for load ~0.5
    let table = EvolutionEngine.#SCRATCH_VISITED_HASH;
    if (
      table.length === 0 ||
      targetCap > table.length * EvolutionEngine.#VISITED_HASH_LOAD
    ) {
      // compute new power-of-two size >= targetCap / load
      const needed = Math.ceil(targetCap / EvolutionEngine.#VISITED_HASH_LOAD);
      const pow2 = 1 << Math.ceil(Math.log2(needed));
      table = EvolutionEngine.#SCRATCH_VISITED_HASH = new Int32Array(pow2);
    } else {
      table.fill(0);
    }
    const mask = table.length - 1;
    let distinct = 0;
    for (let pi = 0; pi < pathLength; pi++) {
      const cp = path[pi];
      const packed = ((cp[0] & 0xffff) << 16) | (cp[1] & 0xffff);
      let h = Math.imul(packed, 2654435761) >>> 0; // Knuth multiplicative hashing (32-bit)
      // probe (packed+1) to distinguish EMPTY(0)
      const storeVal = (packed + 1) | 0;
      while (true) {
        const slot = h & mask;
        const v = table[slot];
        if (v === 0) {
          table[slot] = storeVal;
          distinct++;
          break;
        } else if (v === storeVal) {
          break; // duplicate
        }
        h = (h + 1) | 0; // linear probe
      }
    }
    const ratio = distinct ? distinct / pathLength : 0;
    return { unique: distinct, pathLen: pathLength, ratio };
  }

  /**
   * Compute simple diversity metrics for a NEAT instance: species count, Simpson index and sample weight std.
   * Uses class-level scratch sample to avoid small allocations.
   * @internal
   */
  static #computeDiversityMetrics(neat: any, sampleSize = 40) {
    const populationRef: any[] = neat.population || [];
    let speciesIds = EvolutionEngine.#SCRATCH_SPECIES_IDS;
    let speciesCounts = EvolutionEngine.#SCRATCH_SPECIES_COUNTS;
    if (populationRef.length > speciesIds.length) {
      const nextSize = 1 << Math.ceil(Math.log2(populationRef.length));
      EvolutionEngine.#SCRATCH_SPECIES_IDS = new Int32Array(nextSize);
      EvolutionEngine.#SCRATCH_SPECIES_COUNTS = new Int32Array(nextSize);
      speciesIds = EvolutionEngine.#SCRATCH_SPECIES_IDS;
      speciesCounts = EvolutionEngine.#SCRATCH_SPECIES_COUNTS;
    }
    let speciesUniqueCount = 0;
    for (let pi = 0; pi < populationRef.length; pi++) {
      const genome = populationRef[pi];
      const speciesId =
        (genome && genome.species != null ? genome.species : -1) | 0;
      let foundIndex = -1;
      for (let si = 0; si < speciesUniqueCount; si++) {
        if (speciesIds[si] === speciesId) {
          foundIndex = si;
          break;
        }
      }
      if (foundIndex === -1) {
        speciesIds[speciesUniqueCount] = speciesId;
        speciesCounts[speciesUniqueCount] = 1;
        speciesUniqueCount++;
      } else {
        speciesCounts[foundIndex]++;
      }
    }
    let total = 0;
    for (let si = 0; si < speciesUniqueCount; si++) total += speciesCounts[si];
    total = total || 1;
    let simpsonAcc = 0;
    for (let si = 0; si < speciesUniqueCount; si++) {
      const proportion = speciesCounts[si] / total;
      simpsonAcc += proportion * proportion;
    }
    const simpson = 1 - simpsonAcc;
    // Weight variance sample
    const sampleLength = Math.min(populationRef.length, sampleSize);
    const sampledLen = EvolutionEngine.#sampleIntoScratch(
      populationRef,
      sampleLength
    );
    // Single-pass Welford variance over sampled enabled connection weights
    let weightMean = 0;
    let weightM2 = 0;
    let weightCount = 0;
    for (let sampleIndex = 0; sampleIndex < sampledLen; sampleIndex++) {
      const sampleGenome = EvolutionEngine.#SCRATCH_SAMPLE[sampleIndex];
      const conns = sampleGenome.connections || [];
      for (
        let connectionIndex = 0;
        connectionIndex < conns.length;
        connectionIndex++
      ) {
        const conn = conns[connectionIndex];
        if (conn && conn.enabled !== false) {
          // Welford update
          weightCount++;
          const delta = conn.weight - weightMean;
          weightMean += delta / weightCount;
          weightM2 += delta * (conn.weight - weightMean);
        }
      }
    }
    const wStd = weightCount ? Math.sqrt(weightM2 / weightCount) : 0;
    return { speciesUniqueCount, simpson, wStd };
  }

  /**
   * Sample `k` items from `src` with replacement. Uses SCRATCH_SAMPLE when k is small to avoid allocations.
   * @internal
   */
  static #sampleArray<T>(src: T[], k: number): T[] {
    if (!Array.isArray(src) || k <= 0) return [];
    const sampleCount = Math.floor(k);
    const srcLen = src.length | 0;
    if (srcLen === 0) return [];
    if (sampleCount > EvolutionEngine.#SCRATCH_SAMPLE_RESULT.length) {
      const nextSize = 1 << Math.ceil(Math.log2(sampleCount));
      EvolutionEngine.#SCRATCH_SAMPLE_RESULT = new Array(nextSize);
    }
    const out = EvolutionEngine.#SCRATCH_SAMPLE_RESULT as T[];
    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
      out[sampleIndex] = src[(EvolutionEngine.#fastRandom() * srcLen) | 0];
    }
    out.length = sampleCount;
    return out; // pooled ephemeral array; copy if you need to retain
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
    const popRef = neat.population || [];
    for (let gidx = 0; gidx < popRef.length; gidx++) {
      try {
        EvolutionEngine.#pruneWeakConnectionsForGenome(
          popRef[gidx],
          simplifyStrategy,
          simplifyPruneFraction
        );
      } catch {
        /* ignore per-genome failure */
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
    } catch {
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
    if (!simplifyRemaining) return 0;
    try {
      // Skip in browsers
      if (typeof window !== 'undefined') return simplifyRemaining;
    } catch {
      return simplifyRemaining;
    }
    const t0 = EvolutionEngine.#PROFILE_ENABLED
      ? EvolutionEngine.#PROFILE_T0()
      : 0;
    EvolutionEngine.#applySimplifyPruningToPopulation(
      neat,
      simplifyStrategy,
      simplifyPruneFraction
    );
    if (EvolutionEngine.#PROFILE_ENABLED) {
      EvolutionEngine.#PROFILE_ADD(
        'simplify',
        EvolutionEngine.#PROFILE_T0() - t0 || 0
      );
    }
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
    const t1 = doProfile ? EvolutionEngine.#now() : 0;
    // Optional sampling to cut cost
    let trainingSetRef = lamarckianTrainingSet;
    if (
      lamarckianSampleSize &&
      lamarckianSampleSize < lamarckianTrainingSet.length
    ) {
      trainingSetRef = EvolutionEngine.#sampleArray(
        lamarckianTrainingSet,
        lamarckianSampleSize
      );
    }
    let gradNormSum = 0;
    let gradSamples = 0;
    const pop = neat.population || [];
    for (let np = 0; np < pop.length; np++) {
      const network: any = pop[np];
      try {
        network.train(trainingSetRef, {
          iterations: lamarckianIterations, // Small to preserve diversity
          error: EvolutionEngine.#DEFAULT_TRAIN_ERROR,
          rate: EvolutionEngine.#DEFAULT_TRAIN_RATE,
          momentum: EvolutionEngine.#DEFAULT_TRAIN_MOMENTUM,
          batchSize: EvolutionEngine.#DEFAULT_TRAIN_BATCH_SMALL,
          allowRecurrent: true, // allow recurrent connections
          cost: methods.Cost.softmaxCrossEntropy,
        });
        // Re-center output biases after local refinement
        EvolutionEngine.#adjustOutputBiasesAfterTraining(network);
        // Capture gradient norm stats if available
        try {
          if (typeof (network as any).getTrainingStats === 'function') {
            const ts = (network as any).getTrainingStats();
            if (ts && Number.isFinite(ts.gradNorm)) {
              gradNormSum += ts.gradNorm;
              gradSamples++;
            }
          }
        } catch {
          /* ignore */
        }
      } catch {
        /* ignore per-network training failures */
      }
    }
    if (gradSamples > 0) {
      safeWrite(
        `[GRAD] gen=${completedGenerations} meanGradNorm=${(
          gradNormSum / gradSamples
        ).toFixed(4)} samples=${gradSamples}\n`
      );
    }
    const tDelta = doProfile ? EvolutionEngine.#now() - t1 : 0;
    return tDelta;
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
    if (EvolutionEngine.#TELEMETRY_MINIMAL) return; // skip heavy telemetry
    const t0 = EvolutionEngine.#PROFILE_ENABLED
      ? EvolutionEngine.#PROFILE_T0()
      : 0;
    try {
      // Action entropy
      const ae = EvolutionEngine.#computeActionEntropy(generationResult.path);
      safeWrite(
        `${
          EvolutionEngine.#LOG_TAG_ACTION_ENTROPY
        } gen=${completedGenerations} entropyNorm=${ae.entropyNorm.toFixed(
          3
        )} uniqueMoves=${ae.uniqueMoves} pathLen=${ae.pathLen}\n`
      );

      // Output bias stats
      try {
        const nodesRef2 = fittest.nodes || [];
        const outCount2 = EvolutionEngine.#getNodeIndicesByType(
          nodesRef2,
          'output'
        );
        if (outCount2 > 0) {
          const bstats = EvolutionEngine.#computeOutputBiasStats(
            nodesRef2,
            outCount2
          );
          safeWrite(
            `${
              EvolutionEngine.#LOG_TAG_OUTPUT_BIAS
            } gen=${completedGenerations} mean=${bstats.mean.toFixed(
              3
            )} std=${bstats.std.toFixed(3)} biases=${bstats.biasesStr}\n`
          );
        }
      } catch {}

      // Logits and collapse detection
      try {
        const lastHist: number[][] = (fittest as any)._lastStepOutputs || [];
        if (lastHist.length) {
          const recent: number[][] = EvolutionEngine.#getTail<number[]>(
            lastHist,
            EvolutionEngine.#RECENT_WINDOW
          );
          const stats: any = EvolutionEngine.#computeLogitStats(recent);
          safeWrite(
            `${
              EvolutionEngine.#LOG_TAG_LOGITS
            } gen=${completedGenerations} means=${stats.meansStr} stds=${
              stats.stdsStr
            } kurt=${stats.kurtStr} entMean=${stats.entMean.toFixed(
              3
            )} stability=${stats.stability.toFixed(3)} steps=${recent.length}\n`
          );
          // Anti-collapse trigger
          (EvolutionEngine as any)._collapseStreak =
            (EvolutionEngine as any)._collapseStreak || 0;
          let allBelow = true;
          const stds = stats.stds as ArrayLike<number>;
          for (let si = 0; si < stds.length; si++) {
            if (!(stds[si] < EvolutionEngine.#LOGSTD_FLAT_THRESHOLD)) {
              allBelow = false;
              break;
            }
          }
          const collapsed =
            allBelow &&
            (stats.entMean < EvolutionEngine.#ENTROPY_COLLAPSE_THRESHOLD ||
              stats.stability > EvolutionEngine.#STABILITY_COLLAPSE_THRESHOLD);
          if (collapsed) (EvolutionEngine as any)._collapseStreak++;
          else (EvolutionEngine as any)._collapseStreak = 0;
          if (
            (EvolutionEngine as any)._collapseStreak ===
            EvolutionEngine.#COLLAPSE_STREAK_TRIGGER
          ) {
            EvolutionEngine.#antiCollapseRecovery(
              neat,
              completedGenerations,
              safeWrite
            );
          }
        }
      } catch {}

      // Exploration & Diversity
      try {
        const expl = EvolutionEngine.#computeExplorationStats(
          generationResult.path
        );
        safeWrite(
          `[EXPLORE] gen=${completedGenerations} unique=${
            expl.unique
          } pathLen=${expl.pathLen} ratio=${expl.ratio.toFixed(
            3
          )} progress=${generationResult.progress.toFixed(
            1
          )} satFrac=${(generationResult as any).saturationFraction?.toFixed(
            3
          )}\n`
        );
      } catch {}
      try {
        const dv = EvolutionEngine.#computeDiversityMetrics(neat);
        safeWrite(
          `[DIVERSITY] gen=${completedGenerations} species=${
            dv.speciesUniqueCount
          } simpson=${dv.simpson.toFixed(3)} weightStd=${dv.wStd.toFixed(3)}\n`
        );
      } catch {}
    } catch {}
    if (EvolutionEngine.#PROFILE_ENABLED) {
      EvolutionEngine.#PROFILE_ADD(
        'telemetry',
        EvolutionEngine.#PROFILE_T0() - t0 || 0
      );
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
    if (!fs || !persistDir || persistEvery <= 0) return;
    try {
      const t0 = EvolutionEngine.#PROFILE_ENABLED
        ? EvolutionEngine.#PROFILE_T0()
        : 0;
      const snap = EvolutionEngine.#SCRATCH_SNAPSHOT_OBJ;
      snap.generation = completedGenerations;
      snap.bestFitness = bestFitness;
      snap.simplifyMode = simplifyMode;
      snap.plateauCounter = plateauCounter;
      snap.timestamp = Date.now();
      snap.telemetryTail = (() => {
        if (!neat.getTelemetry) return undefined;
        try {
          const telemetryValue = neat.getTelemetry();
          if (Array.isArray(telemetryValue))
            return EvolutionEngine.#getTail<any>(telemetryValue, 5);
          return telemetryValue;
        } catch {
          return undefined;
        }
      })();
      const populationRef = neat.population || [];
      const sortedIdx = EvolutionEngine.#getSortedIndicesByScore(populationRef);
      const limit = Math.min(persistTopK, sortedIdx.length);
      let topBuf = EvolutionEngine.#SCRATCH_SNAPSHOT_TOP;
      if (topBuf.length < limit) {
        topBuf.length = limit;
      }
      for (let rank = 0; rank < limit; rank++) {
        let entry = topBuf[rank];
        if (!entry) entry = topBuf[rank] = {};
        const genome = populationRef[sortedIdx[rank]];
        entry.idx = sortedIdx[rank];
        entry.score = genome.score;
        entry.nodes = genome.nodes.length;
        entry.connections = genome.connections.length;
        entry.json = genome.toJSON ? genome.toJSON() : undefined;
      }
      topBuf.length = limit;
      snap.top = topBuf;
      const file = pathModule.join(
        persistDir,
        `snapshot_gen${completedGenerations}.json`
      );
      // Compact JSON to reduce allocation size and parse cost
      fs.writeFileSync(file, JSON.stringify(snap));
      if (EvolutionEngine.#PROFILE_ENABLED) {
        EvolutionEngine.#PROFILE_ADD(
          'snapshot',
          EvolutionEngine.#PROFILE_T0() - t0 || 0
        );
      }
    } catch {
      /* ignore persistence errors */
    }
  }

  /** Compute decision stability: fraction of consecutive identical argmaxes. @internal */
  static #computeDecisionStability(recent: number[][]): number {
    let stableCount = 0;
    let transitionCount = 0;
    let previousArgmax = -1;
    for (let ri = 0; ri < recent.length; ri++) {
      const v = recent[ri];
      let argmax = 0;
      let best = v[0];
      for (let outIdx = 1; outIdx < EvolutionEngine.#ACTION_DIM; outIdx++) {
        if (v[outIdx] > best) {
          best = v[outIdx];
          argmax = outIdx;
        }
      }
      if (previousArgmax === argmax) stableCount++;
      if (previousArgmax !== -1) transitionCount++;
      previousArgmax = argmax;
    }
    return transitionCount ? stableCount / transitionCount : 0;
  }

  /**
   * Compute normalized softmax entropy for a single output vector `v`.
   * Uses provided scratch `buf` to store exponentials to avoid allocations.
   * @internal
   */
  static #softmaxEntropyFromVector(
    v: number[] | undefined,
    buf: Float64Array
  ): number {
    if (!v || !v.length) return 0;
    const k = Math.min(v.length, buf.length);
    if (k === 4) {
      // Unrolled version for common 4-action case
      const v0 = v[0] || 0;
      const v1 = v[1] || 0;
      const v2 = v[2] || 0;
      const v3 = v[3] || 0;
      let maxVal = v0;
      if (v1 > maxVal) maxVal = v1;
      if (v2 > maxVal) maxVal = v2;
      if (v3 > maxVal) maxVal = v3;
      const e0 = Math.exp(v0 - maxVal);
      const e1 = Math.exp(v1 - maxVal);
      const e2 = Math.exp(v2 - maxVal);
      const e3 = Math.exp(v3 - maxVal);
      const sum = e0 + e1 + e2 + e3 || 1;
      const p0 = e0 / sum;
      const p1 = e1 / sum;
      const p2 = e2 / sum;
      const p3 = e3 / sum;
      let e = 0;
      if (p0 > 0) e += -p0 * Math.log(p0);
      if (p2 > 0) e += -p2 * Math.log(p2);
      if (p3 > 0) e += -p3 * Math.log(p3);
      return e * EvolutionEngine.#INV_LOG4;
    }
    let maxVal = -Infinity;
    for (let i = 0; i < k; i++) {
      const val = v[i] || 0;
      if (val > maxVal) maxVal = val;
    }
    let sum = 0;
    for (let i = 0; i < k; i++) {
      const ex = Math.exp((v[i] || 0) - maxVal);
      buf[i] = ex;
      sum += ex;
    }
    if (!sum) sum = 1;
    let e = 0;
    for (let i = 0; i < k; i++) {
      const p = buf[i] / sum;
      if (p > 0) e += -p * Math.log(p);
    }
    return e / Math.log(k);
  }

  static #joinNumberArray(
    arrLike: ArrayLike<number>,
    len: number,
    digits = 3
  ): string {
    if (len <= 0) return '';
    // ensure string scratch capacity
    if (len > EvolutionEngine.#SCRATCH_STR.length) {
      const nextSize = 1 << Math.ceil(Math.log2(len));
      EvolutionEngine.#SCRATCH_STR = new Array(nextSize);
    }
    const buf = EvolutionEngine.#SCRATCH_STR;
    for (let i = 0; i < len; i++)
      buf[i] = (arrLike[i] as number).toFixed(digits);
    const prevLen = buf.length;
    buf.length = len;
    const out = buf.join(',');
    buf.length = prevLen; // restore capacity (logical view only)
    return out;
  }

  /** Extract last n items from an array (small, allocation-aware helper). @internal */
  static #getTail<T>(arr: T[] | undefined, n: number): T[] {
    if (!Array.isArray(arr) || n <= 0) return [];
    const take = Math.min(n, arr.length);
    if (take > EvolutionEngine.#SCRATCH_TAIL.length) {
      const nextSize = 1 << Math.ceil(Math.log2(take));
      EvolutionEngine.#SCRATCH_TAIL = new Array(nextSize);
    }
    const tailBuf = EvolutionEngine.#SCRATCH_TAIL as T[];
    const start = arr.length - take;
    for (let i = 0; i < take; i++) tailBuf[i] = arr[start + i]!;
    tailBuf.length = take;
    return tailBuf; // pooled ephemeral array
  }

  /** Delegate to MazeUtils.pushHistory to keep bounded history semantics. @internal */
  static #pushHistory<T>(buf: T[] | undefined, v: T, maxLen: number): T[] {
    return MazeUtils.pushHistory(buf as any, v as any, maxLen) as T[];
  }

  /** Re-center output node biases for a network (defensive, best-effort). @internal */
  static #centerOutputBiases(net: any): void {
    try {
      const nodes = net.nodes || [];
      // collect output nodes indices without allocating an intermediate array
      let outCount = 0;
      for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
        if (nodes[nodeIndex] && nodes[nodeIndex].type === 'output') {
          EvolutionEngine.#SCRATCH_NODE_IDX[outCount++] = nodeIndex;
        }
      }
      if (outCount === 0) return;
      // Single-pass Welford mean/std
      let mean = 0;
      let M2 = 0;
      for (let outIndex = 0; outIndex < outCount; outIndex++) {
        const b = nodes[EvolutionEngine.#SCRATCH_NODE_IDX[outIndex]].bias;
        const n = outIndex + 1;
        const delta = b - mean;
        mean += delta / n;
        M2 += delta * (b - mean);
      }
      const std = outCount ? Math.sqrt(M2 / outCount) : 0;
      for (let oi = 0; oi < outCount; oi++) {
        const idx = EvolutionEngine.#SCRATCH_NODE_IDX[oi];
        nodes[idx].bias = Math.max(-5, Math.min(5, nodes[idx].bias - mean));
      }
      (net as any)._outputBiasStats = { mean, std };
    } catch {
      /* ignore */
    }
  }

  /** Prune weakest connections for a genome according to the chosen strategy. @internal */
  static #pruneWeakConnectionsForGenome(
    genome: any,
    simplifyStrategy: string,
    simplifyPruneFraction: number
  ): void {
    try {
      const genomeConns = genome.connections || [];
      let candidates: any[] = EvolutionEngine.#collectEnabledConnections(
        genomeConns
      );
      const enabledCount = candidates.length;
      if (enabledCount === 0) return;
      const pruneCount = Math.max(
        1,
        Math.floor(enabledCount * simplifyPruneFraction)
      );
      // Delegate candidate ordering to a dedicated helper for clarity and reuse
      candidates = EvolutionEngine.#sortCandidatesByStrategy(
        candidates || [],
        simplifyStrategy
      );
      EvolutionEngine.#disableSmallestEnabledConnections(
        candidates,
        Math.min(pruneCount, candidates.length)
      );
    } catch {
      /* ignore per-genome failures */
    }
  }

  /** Collect enabled connections from an array of connections. @internal */
  static #collectEnabledConnections(conns: any[]): any[] {
    if (!Array.isArray(conns) || conns.length === 0) return [];
    const candBuf = EvolutionEngine.#SCRATCH_CONN_CAND;
    candBuf.length = 0;
    for (
      let connectionIndex = 0;
      connectionIndex < conns.length;
      connectionIndex++
    ) {
      const connection = conns[connectionIndex];
      if (connection && connection.enabled !== false) candBuf.push(connection);
    }
    return candBuf;
  }

  /**
   * Collect outgoing connections from a hidden node that target any output nodes.
   * Returns an array of matching connections (possibly using pooled buffers internally).
   * @internal
   */
  static #collectHiddenToOutputConns(
    hiddenNode: any,
    nodesRef: any[],
    outputCount: number
  ): any[] {
    /**
     * Implementation notes:
     * - Reuses class-level pooled object buffer `#SCRATCH_SAMPLE` to avoid small allocations.
     * - Caller must ensure `EvolutionEngine.#SCRATCH_NODE_IDX` contains output node indices
     *   (0..outputCount-1) before calling this helper.
     */
    const outputBuffer = EvolutionEngine.#SCRATCH_HIDDEN_OUT;
    outputBuffer.length = 0;
    if (!hiddenNode || !hiddenNode.connections) return [];
    const outgoingConnections = hiddenNode.connections.out || [];
    for (
      let connectionIndex = 0;
      connectionIndex < outgoingConnections.length;
      connectionIndex++
    ) {
      const connection = outgoingConnections[connectionIndex];
      if (connection && connection.enabled !== false) {
        for (let outputIndex = 0; outputIndex < outputCount; outputIndex++) {
          const outputNodeIdx = EvolutionEngine.#SCRATCH_NODE_IDX[outputIndex];
          if (connection.to === nodesRef[outputNodeIdx]) {
            outputBuffer.push(connection);
            break;
          }
        }
      }
    }
    return outputBuffer;
  }

  /**
   * Sort candidate connections according to the chosen strategy.
   * For 'weakRecurrentPreferred', recurrent/gater connections are prioritized
   * for removal. Returns a newly ordered array (no external allocations when possible).
   * @internal
   */
  static #sortCandidatesByStrategy(candidates: any[], strategy: string): any[] {
    if (!Array.isArray(candidates) || candidates.length === 0)
      return candidates;
    if (strategy === 'weakRecurrentPreferred') {
      // In-place stable-ish partition then insertion sort slices (candidate arrays are small)
      let recurrentWrite = 0;
      for (
        let candidateIndex = 0;
        candidateIndex < candidates.length;
        candidateIndex++
      ) {
        const candidateConnection = candidates[candidateIndex];
        if (
          candidateConnection &&
          (candidateConnection.from === candidateConnection.to ||
            candidateConnection.gater)
        ) {
          if (candidateIndex !== recurrentWrite) {
            const tmp = candidates[recurrentWrite];
            candidates[recurrentWrite] = candidates[candidateIndex];
            candidates[candidateIndex] = tmp;
          }
          recurrentWrite++;
        }
      }
      EvolutionEngine.#insertionSortByAbsWeight(candidates, 0, recurrentWrite);
      EvolutionEngine.#insertionSortByAbsWeight(
        candidates,
        recurrentWrite,
        candidates.length
      );
      return candidates;
    }
    EvolutionEngine.#insertionSortByAbsWeight(candidates, 0, candidates.length);
    return candidates;
  }

  /** Insertion sort helper for small candidate arrays (avoids allocations). @internal */
  static #insertionSortByAbsWeight(buffer: any[], start: number, end: number) {
    for (let i = start + 1; i < end; i++) {
      const value = buffer[i];
      const weightAbs = Math.abs(value.weight);
      let j = i - 1;
      while (j >= start && Math.abs(buffer[j].weight) > weightAbs) {
        buffer[j + 1] = buffer[j];
        j--;
      }
      buffer[j + 1] = value;
    }
  }

  /** Disable the smallest enabled connections from candidates up to `count`. @internal */
  static #disableSmallestEnabledConnections(candidates: any[], count: number) {
    if (!Array.isArray(candidates) || count <= 0) return;
    let flags = EvolutionEngine.#SCRATCH_CONN_FLAGS;
    if (candidates.length > flags.length) {
      EvolutionEngine.#SCRATCH_CONN_FLAGS = new Uint8Array(candidates.length);
      flags = EvolutionEngine.#SCRATCH_CONN_FLAGS;
    } else {
      flags.fill(0, 0, candidates.length);
    }
    const n = candidates.length;
    if (count >= n >>> 1) {
      // If pruning many, just sort by abs weight once (reuse insertion for small n else fallback native)
      if (n <= 64) {
        EvolutionEngine.#insertionSortByAbsWeight(candidates, 0, n);
      } else {
        candidates.sort((a, b) => Math.abs(a.weight) - Math.abs(b.weight));
      }
      for (let i = 0, lim = Math.min(count, n); i < lim; i++) {
        const c = candidates[i];
        if (c && c.enabled !== false) c.enabled = false;
      }
      return;
    }
    // Prune a small number: maintain a partial selection using multi-pass with shrinking upper bound (selection algorithm)
    let remaining = count;
    let active = 0;
    // compact enabled candidates into front to reduce later scans
    for (let i = 0; i < n; i++) {
      const c = candidates[i];
      if (c && c.enabled !== false) {
        if (i !== active) candidates[active] = c;
        active++;
      }
    }
    while (remaining > 0 && active > 0) {
      let minIdx = 0;
      let minW = Math.abs(candidates[0].weight);
      for (let j = 1; j < active; j++) {
        const aw = Math.abs(candidates[j].weight);
        if (aw < minW) {
          minW = aw;
          minIdx = j;
        }
      }
      const target = candidates[minIdx];
      target.enabled = false;
      // remove minIdx by swapping last active-1
      const last = --active;
      candidates[minIdx] = candidates[last];
      remaining--;
    }
  }

  /** Compute directional action entropy and unique move count from a path. @internal */
  static #computeActionEntropy(
    pathArr: ReadonlyArray<[number, number]>
  ): { entropyNorm: number; uniqueMoves: number; pathLen: number } {
    const counts = EvolutionEngine.#SCRATCH_COUNTS;
    // Manual unroll for fixed length=4 avoids call overhead of fill()
    counts[0] = 0;
    counts[1] = 0;
    counts[2] = 0;
    counts[3] = 0;
    let totalMoves = 0;
    // Branchless (dx,dy)->index mapping via 3x3 Int8Array
    const dirMap = EvolutionEngine.#DIR_DELTA_TO_INDEX;
    for (let pathIndex = 1; pathIndex < pathArr.length; pathIndex++) {
      const currentPoint = pathArr[pathIndex];
      const previousPoint = pathArr[pathIndex - 1];
      const deltaX = currentPoint[0] - previousPoint[0];
      const deltaY = currentPoint[1] - previousPoint[1];
      // Pack into (dx+1)*3 + (dy+1) domain 0..8. Out-of-range deltas ignored.
      if (deltaX < -1 || deltaX > 1 || deltaY < -1 || deltaY > 1) continue;
      const key = (deltaX + 1) * 3 + (deltaY + 1);
      const directionIndex = dirMap[key];
      if (directionIndex >= 0) {
        counts[directionIndex]++;
        totalMoves++;
      }
    }
    totalMoves = totalMoves || 1;
    let entropy = 0;
    let uniqueMoves = 0;
    for (let i = 0; i < counts.length; i++) {
      const prob = counts[i] / totalMoves;
      if (prob > 0) entropy += -prob * Math.log(prob);
      if (counts[i] > 0) uniqueMoves++;
    }
    const entropyNorm = entropy * EvolutionEngine.#INV_LOG4;
    return { entropyNorm, uniqueMoves, pathLen: pathArr.length };
  }

  /** Compute statistics over recent logits: means, stds, kurtosis, mean entropy and stability. @internal */
  static #computeLogitStats(recent: number[][]) {
    const reduced = EvolutionEngine.#REDUCED_TELEMETRY;
    const actionDim = EvolutionEngine.#ACTION_DIM;
    const meansBuf = EvolutionEngine.#SCRATCH_MEANS;
    const stdsBuf = EvolutionEngine.#SCRATCH_STDS;
    // Lazy allocate higher-moment buffers only when needed
    if (!reduced && !EvolutionEngine.#SCRATCH_KURT) {
      EvolutionEngine.#SCRATCH_KURT = new Float64Array(actionDim);
      EvolutionEngine.#SCRATCH_M3_RAW = new Float64Array(actionDim);
      EvolutionEngine.#SCRATCH_M4_RAW = new Float64Array(actionDim);
    }
    const kurtBuf = EvolutionEngine.#SCRATCH_KURT; // may be undefined in reduced mode
    const m2Buf = EvolutionEngine.#SCRATCH_M2_RAW;
    const m3Buf = EvolutionEngine.#SCRATCH_M3_RAW;
    const m4Buf = EvolutionEngine.#SCRATCH_M4_RAW;
    // Reset base buffers
    for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
      meansBuf[dimIndex] = 0;
      m2Buf[dimIndex] = 0;
      stdsBuf[dimIndex] = 0;
      if (!reduced) {
        m3Buf![dimIndex] = 0;
        m4Buf![dimIndex] = 0;
        kurtBuf![dimIndex] = 0;
      }
    }
    const stepCount = recent.length;
    if (!stepCount) {
      return {
        meansStr: '',
        stdsStr: '',
        kurtStr: '',
        entMean: 0,
        stability: 0,
        steps: 0,
        means: meansBuf,
        stds: stdsBuf,
      } as any;
    }
    let entropyAggregate = 0;
    // Reduced mode: skip higher moments (kurtosis) to save CPU
    if (reduced) {
      // Compute mean/std only (generic path)
      for (let stepIndex = 0; stepIndex < stepCount; stepIndex++) {
        const vec = recent[stepIndex];
        const n = stepIndex + 1;
        for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
          const x = vec[dimIndex] || 0;
          const delta = x - meansBuf[dimIndex];
          meansBuf[dimIndex] += delta / n;
          const delta2 = x - meansBuf[dimIndex];
          m2Buf[dimIndex] += delta * delta2;
        }
        entropyAggregate += EvolutionEngine.#softmaxEntropyFromVector(
          vec,
          EvolutionEngine.#SCRATCH_EXPS
        );
      }
      for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
        const variance = m2Buf[dimIndex] / stepCount;
        stdsBuf[dimIndex] = variance > 0 ? Math.sqrt(variance) : 0;
      }
    } else if (actionDim === 4) {
      // Unrolled fast path for 4 outputs (dominant case)
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
      for (let stepIndex = 0; stepIndex < stepCount; stepIndex++) {
        const vec = recent[stepIndex];
        const x0 = vec[0] || 0;
        const x1 = vec[1] || 0;
        const x2 = vec[2] || 0;
        const x3 = vec[3] || 0;
        const n = stepIndex + 1;
        // Dimension 0
        let delta = x0 - mean0;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1);
        M40 +=
          term1 * delta_n2 * (n * n - 3 * n + 3) +
          6 * delta_n2 * M20 -
          4 * delta_n * M30;
        M30 += term1 * delta_n * (n - 2) - 3 * delta_n * M20;
        M20 += term1;
        mean0 += delta_n;
        // Dimension 1
        delta = x1 - mean1;
        delta_n = delta / n;
        delta_n2 = delta_n * delta_n;
        term1 = delta * delta_n * (n - 1);
        M41 +=
          term1 * delta_n2 * (n * n - 3 * n + 3) +
          6 * delta_n2 * M21 -
          4 * delta_n * M31;
        M31 += term1 * delta_n * (n - 2) - 3 * delta_n * M21;
        M21 += term1;
        mean1 += delta_n;
        // Dimension 2
        delta = x2 - mean2;
        delta_n = delta / n;
        delta_n2 = delta_n * delta_n;
        term1 = delta * delta_n * (n - 1);
        M42 +=
          term1 * delta_n2 * (n * n - 3 * n + 3) +
          6 * delta_n2 * M22 -
          4 * delta_n * M32;
        M32 += term1 * delta_n * (n - 2) - 3 * delta_n * M22;
        M22 += term1;
        mean2 += delta_n;
        // Dimension 3
        delta = x3 - mean3;
        delta_n = delta / n;
        delta_n2 = delta_n * delta_n;
        term1 = delta * delta_n * (n - 1);
        M43 +=
          term1 * delta_n2 * (n * n - 3 * n + 3) +
          6 * delta_n2 * M23 -
          4 * delta_n * M33;
        M33 += term1 * delta_n * (n - 2) - 3 * delta_n * M23;
        M23 += term1;
        mean3 += delta_n;
        // Entropy (integrated same pass)
        entropyAggregate += EvolutionEngine.#softmaxEntropyFromVector(
          vec,
          EvolutionEngine.#SCRATCH_EXPS
        );
      }
      meansBuf[0] = mean0;
      meansBuf[1] = mean1;
      meansBuf[2] = mean2;
      meansBuf[3] = mean3;
      const invN = 1 / stepCount;
      const var0 = M20 * invN;
      const var1 = M21 * invN;
      const var2 = M22 * invN;
      const var3 = M23 * invN;
      stdsBuf[0] = var0 > 0 ? Math.sqrt(var0) : 0;
      stdsBuf[1] = var1 > 0 ? Math.sqrt(var1) : 0;
      stdsBuf[2] = var2 > 0 ? Math.sqrt(var2) : 0;
      stdsBuf[3] = var3 > 0 ? Math.sqrt(var3) : 0;
      // Excess kurtosis
      if (!reduced) {
        kurtBuf![0] = var0 > 1e-18 ? (stepCount * M40) / (M20 * M20) - 3 : 0;
        kurtBuf![1] = var1 > 1e-18 ? (stepCount * M41) / (M21 * M21) - 3 : 0;
        kurtBuf![2] = var2 > 1e-18 ? (stepCount * M42) / (M22 * M22) - 3 : 0;
        kurtBuf![3] = var3 > 1e-18 ? (stepCount * M43) / (M23 * M23) - 3 : 0;
      }
    } else {
      // Generic path (rare in this maze task but kept for completeness)
      for (let stepIndex = 0; stepIndex < stepCount; stepIndex++) {
        const vec = recent[stepIndex];
        const n = stepIndex + 1;
        for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
          const x = vec[dimIndex] || 0;
          const delta = x - meansBuf[dimIndex];
          const delta_n = delta / n;
          const delta_n2 = delta_n * delta_n;
          const term1 = delta * delta_n * (n - 1);
          // Update higher moments (Pébay 2008)
          if (!reduced)
            m4Buf![dimIndex] +=
              term1 * delta_n2 * (n * n - 3 * n + 3) +
              6 * delta_n2 * m2Buf[dimIndex] -
              4 * delta_n * (m3Buf ? m3Buf[dimIndex] : 0);
          if (!reduced)
            m3Buf![dimIndex] +=
              term1 * delta_n * (n - 2) - 3 * delta_n * m2Buf[dimIndex];
          m2Buf[dimIndex] += term1;
          meansBuf[dimIndex] += delta_n;
        }
        entropyAggregate += EvolutionEngine.#softmaxEntropyFromVector(
          vec,
          EvolutionEngine.#SCRATCH_EXPS
        );
      }
      for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
        const variance = m2Buf[dimIndex] / stepCount;
        stdsBuf[dimIndex] = variance > 0 ? Math.sqrt(variance) : 0;
        if (!reduced) {
          const m4v = m4Buf![dimIndex];
          kurtBuf![dimIndex] =
            variance > 1e-18
              ? (stepCount * m4v) / (m2Buf[dimIndex] * m2Buf[dimIndex]) - 3
              : 0;
        }
      }
    }
    const entropyMean = entropyAggregate / stepCount;
    const stability = EvolutionEngine.#computeDecisionStability(recent);
    const meansStr = EvolutionEngine.#joinNumberArray(meansBuf, actionDim, 3);
    const stdsStr = EvolutionEngine.#joinNumberArray(stdsBuf, actionDim, 3);
    const kurtStr = reduced
      ? ''
      : EvolutionEngine.#joinNumberArray(kurtBuf!, actionDim, 2);
    return {
      meansStr,
      stdsStr,
      kurtStr,
      entMean: entropyMean,
      stability,
      steps: stepCount,
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
              // Partial unroll for first two selections (common case mutateCount <=2)
              if (applyCount > 0) {
                const r0 = (EvolutionEngine.#fastRandom() * opLen) | 0;
                const tmp0 = idxBuf[0];
                idxBuf[0] = idxBuf[r0];
                idxBuf[r0] = tmp0;
                try {
                  clone.mutate(mutOps[idxBuf[0]]);
                } catch {}
              }
              if (applyCount > 1) {
                const r1 =
                  1 + ((EvolutionEngine.#fastRandom() * (opLen - 1)) | 0);
                const tmp1 = idxBuf[1];
                idxBuf[1] = idxBuf[r1];
                idxBuf[r1] = tmp1;
                try {
                  clone.mutate(mutOps[idxBuf[1]]);
                } catch {}
              }
              for (let sel = 2; sel < applyCount; sel++) {
                const r =
                  sel + ((EvolutionEngine.#fastRandom() * (opLen - sel)) | 0);
                const tmp = idxBuf[sel];
                idxBuf[sel] = idxBuf[r];
                idxBuf[r] = tmp;
                try {
                  clone.mutate(mutOps[idxBuf[sel]]);
                } catch {}
              }
            }
          } catch {
            /* ignore mutation sequence */
          }
          clone.score = undefined;
          try {
            if (typeof neat.addGenome === 'function') {
              neat.addGenome(clone, [(parent as any)._id]);
            } else {
              if (neat._nextGenomeId !== undefined)
                clone._id = neat._nextGenomeId++;
              if (neat._lineageEnabled) {
                clone._parents = [(parent as any)._id];
                clone._depth = ((parent as any)._depth ?? 0) + 1;
              }
              if (typeof neat._invalidateGenomeCaches === 'function')
                neat._invalidateGenomeCaches(clone);
              neat.population.push(clone);
            }
          } catch {
            try {
              neat.population.push(clone);
            } catch {}
          }
        }
      } catch {
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
      } catch {
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
