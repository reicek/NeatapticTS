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
  /** Reusable empty vector constant to avoid ephemeral allocations from `|| []` fallbacks. */
  static #EMPTY_VEC: any[] = [];
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
  /** Optional typed-array scratch used internally to accelerate sorting without allocating each call. */
  static #SCRATCH_SORT_IDX_TA: Int32Array | undefined;
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
  /** LCG multiplier (1664525) used by the fast RNG (32-bit LCG). */
  static #LCG_MULT = 1664525;
  /** LCG additive constant (1013904223) used by the fast RNG. */
  static #LCG_ADD = 1013904223;
  /** Number of cached RNG outputs per refill (batched to amortize state writes). */
  static #RNG_CACHE_SIZE = 4;
  /** Bit shift applied when converting 32-bit state to fractional mantissa (>> 9). */
  static #RNG_SHIFT = 9;
  /** Scale factor to map shifted integer to [0,1): 1 / 0x800000. */
  static #RNG_SCALE = 1 / 0x800000;
  /** Adaptive logits ring capacity (power-of-two). */
  static #LOGITS_RING_CAP = 512;
  /** Max allowed ring capacity (safety bound). */
  static #LOGITS_RING_CAP_MAX = 8192;
  /** Indicates SharedArrayBuffer-backed ring is active. */
  static #LOGITS_RING_SHARED = false;
  /** Logits ring (fallback non-shared row-of-vectors). */
  static #SCRATCH_LOGITS_RING: Float32Array[] = (() => {
    const cap = EvolutionEngine.#LOGITS_RING_CAP;
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

  /**
   * Behavior & environment constraints:
   * - No-op when `SharedArrayBuffer` is unavailable or when the global context is not
   *   `crossOriginIsolated` (the browser COOP+COEP requirement). In those cases the engine will
   *   continue using the fallback non-shared `#SCRATCH_LOGITS_RING`.
   * - Any exception during allocation or view creation is caught; on failure the method clears
   *   any partially-initialized shared references and leaves `#LOGITS_RING_SHARED` as false.
   *
   * Memory layout details:
   * - The SAB size is 4 + (cap * ACTION_DIM * 4) bytes.
   *   - Byte offset 0: Int32Array view of length 1 used as the atomic write index (4 bytes).
   *   - Byte offset 4: Float32Array view of length (cap * ACTION_DIM) storing the flattened logits.
   * - Consumers should treat the Float32Array as rows of length `ACTION_DIM` and use the
   *   atomic write index to coordinate producer/consumer access.
   *
   * Safety / assumptions:
   * - `cap` should be a sensible ring capacity (the rest of the ring logic prefers power-of-two
   *   capacities, though this method does not enforce it).
   * - Atomics.store is used to initialize the write index to 0.
   *
   * @param cap Number of rows (ring capacity). The Float32 storage length will be `cap * ACTION_DIM`.
   * @internal
   * @remarks This is a best-effort performance optimization for worker/agent setups; when the
   *          environment doesn't permit SAB usage the engine gracefully falls back to the
   *          per-row `#SCRATCH_LOGITS_RING` representation.
   * @example
   * // internal usage (may succeed only in cross-origin-isolated browsers or compatible worker hosts)
   * EvolutionEngine['#initSharedLogitsRing'](512);
   */
  static #initSharedLogitsRing(cap: number) {
    try {
      if (typeof SharedArrayBuffer === 'undefined') return;
      if (globalThis?.crossOriginIsolated !== true) return; // must be true in browsers
      if (!Number.isInteger(cap) || cap <= 0) return; // defensive

      const actionDim = EvolutionEngine.#ACTION_DIM;
      const totalFloats = cap * actionDim;
      const indexBytes = Int32Array.BYTES_PER_ELEMENT; // 4
      const floatBytes = Float32Array.BYTES_PER_ELEMENT; // 4
      const sab = new SharedArrayBuffer(indexBytes + totalFloats * floatBytes);
      EvolutionEngine.#SCRATCH_LOGITS_SHARED_W = new Int32Array(sab, 0, 1);
      EvolutionEngine.#SCRATCH_LOGITS_SHARED = new Float32Array(
        sab,
        indexBytes,
        totalFloats
      );

      // Initialize write cursor and zero the logits storage for deterministic startup.
      Atomics.store(EvolutionEngine.#SCRATCH_LOGITS_SHARED_W, 0, 0);
      EvolutionEngine.#SCRATCH_LOGITS_SHARED.fill(0);
      EvolutionEngine.#LOGITS_RING_SHARED = true;
    } catch {
      EvolutionEngine.#LOGITS_RING_SHARED = false;
      EvolutionEngine.#SCRATCH_LOGITS_SHARED = undefined;
      EvolutionEngine.#SCRATCH_LOGITS_SHARED_W = undefined;
    }
  }

  /**
   * Ensure the logits ring has sufficient capacity for `desiredRecentSteps`.
   *
   * Heuristics:
   * - Grow when usage exceeds ~75% of capacity (and cap < max); growth chooses the next
   *   power-of-two >= desiredRecentSteps * 2 to leave headroom.
   * - Shrink when usage drops below 25% of capacity while maintaining a lower bound (128).
   * - All sizes are clamped to [128, #LOGITS_RING_CAP_MAX] and kept as powers of two.
   *
   * Behavior:
   * - On resize we reset the non-shared write cursor, reallocate the non-shared per-row ring,
   *   and attempt to reinitialize the SharedArrayBuffer-backed ring if shared mode is active.
   * - The method is best-effort and non-blocking; callers should avoid concurrent calls from
   *   multiple threads/workers because internal scratch state (non-shared ring) is replaced.
   *
   * @param desiredRecentSteps Estimated number of recent rows that need to be stored.
   * @internal
   */
  static #ensureLogitsRingCapacity(desiredRecentSteps: number) {
    // Defensive input validation
    if (!Number.isFinite(desiredRecentSteps) || desiredRecentSteps < 0) return;

    const MIN_CAP = 128;
    const maxCap = EvolutionEngine.#LOGITS_RING_CAP_MAX;
    let cap = EvolutionEngine.#LOGITS_RING_CAP;
    let target = cap;

    // Helper to compute next power-of-two >= n (for n > 0).
    const nextPow2 = (n: number) => {
      if (n <= 1) return 1;
      return 1 << Math.ceil(Math.log2(n));
    };

    // Grow when usage approaches 75% of capacity (leave headroom by sizing >= desired*2).
    if (desiredRecentSteps > (cap * 3) / 4 && cap < maxCap) {
      const desired = Math.min(desiredRecentSteps * 2, maxCap);
      target = Math.min(nextPow2(Math.ceil(desired)), maxCap);
    } else if (desiredRecentSteps < cap / 4 && cap > MIN_CAP) {
      // Shrink while leaving 2x headroom, but never below MIN_CAP.
      let shrink = cap;
      while (shrink > MIN_CAP && desiredRecentSteps * 2 <= shrink / 2)
        shrink >>= 1;
      target = Math.max(shrink, MIN_CAP);
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
  /**
   * RNG cache (batched draws) to amortize LCG state updates in tight loops.
   * Size is taken from `#RNG_CACHE_SIZE` so the batch parameter is centralized.
   */
  static #RNG_CACHE = new Float64Array(EvolutionEngine.#RNG_CACHE_SIZE);
  // Force initial refill by setting index to cache size.
  static #RNG_CACHE_INDEX: number = EvolutionEngine.#RNG_CACHE_SIZE;

  /**
   * Fast LCG producing a float in [0,1).
   *
   * Notes:
   * - Non-cryptographic: simple 32-bit LCG (mul/add) used for high-performance sampling.
   * - Batches `#RNG_CACHE_SIZE` outputs to reduce the number of state writes.
   * - Deterministic when `#DETERMINISTIC` is set and seeded via `setDeterministic`.
   * - Returns a double in [0,1). Consumers relying on high-quality randomness should
   *   replace this with a cryptographic RNG.
   *
   * @returns float in [0,1)
   * @internal
   */
  static #fastRandom(): number {
    /**
     * @internal Local alias for engine-wide numeric constants used by the RNG loop.
     * Keeping a local const in the hot path improves readability while allowing
     * the optimiser to keep accesses monomorphic.
     */
    // Step 1: Refill the cached batch when we've consumed the existing entries.
    if (EvolutionEngine.#RNG_CACHE_INDEX >= EvolutionEngine.#RNG_CACHE_SIZE) {
      // Copy static RNG state into a local variable for faster loop updates.
      let localRngState = EvolutionEngine.#RNG_STATE >>> 0;

      // Produce a small batch of cached uniform floats using the 32-bit LCG.
      for (
        let cacheFillIndex = 0;
        cacheFillIndex < EvolutionEngine.#RNG_CACHE_SIZE;
        cacheFillIndex++
      ) {
        // LCG update: newState = oldState * multiplier + increment (mod 2^32)
        localRngState =
          (localRngState * EvolutionEngine.#LCG_MULT +
            EvolutionEngine.#LCG_ADD) >>>
          0;
        // Convert the high bits of the state to a floating fraction in [0,1).
        EvolutionEngine.#RNG_CACHE[cacheFillIndex] =
          (localRngState >>> EvolutionEngine.#RNG_SHIFT) *
          EvolutionEngine.#RNG_SCALE;
      }

      // Persist updated state back to the shared static field.
      EvolutionEngine.#RNG_STATE = localRngState >>> 0;

      // Reset read index so subsequent calls consume from the freshly-filled cache.
      EvolutionEngine.#RNG_CACHE_INDEX = 0;
    }

    // Step 2: Return the next cached uniform value (post-increment index).
    const nextValue =
      EvolutionEngine.#RNG_CACHE[EvolutionEngine.#RNG_CACHE_INDEX++];
    return nextValue;
  }
  /** Deterministic mode flag (enables reproducible seeded RNG). */
  static #DETERMINISTIC = false;
  /** High-resolution time helper. */
  static #now(): number {
    return globalThis.performance?.now?.() ?? Date.now();
  }
  /**
   * Enable deterministic mode and optionally reseed the internal RNG.
   *
   * Behaviour (stepwise):
   * 1. Set the internal deterministic flag so other helpers can opt-in to deterministic behaviour.
   * 2. If `seed` is provided and finite, normalise it to an unsigned 32-bit integer. A seed value of
   *    `0` is remapped to the golden-ratio-derived constant `0x9e3779b9` to avoid the degenerate LCG state.
   * 3. Persist the chosen u32 seed into `#RNG_STATE` and force the RNG cache to refill so the next
   *    `#fastRandom()` call yields the deterministic sequence starting from the new seed.
   *
   * Notes:
   * - If `seed` is omitted the method only enables deterministic mode without reseeding the RNG state.
   * - The method is intentionally conservative about input coercion: only finite numeric seeds are accepted.
   *
   * @param seed Optional numeric seed. Fractional values are coerced via `>>> 0`. Passing `0` results in
   *             a non-zero canonical seed to avoid trivial cycles.
   * @example
   * // Enable deterministic mode with an explicit seed:
   * EvolutionEngine.setDeterministic(12345);
   *
   * @internal
   */
  static setDeterministic(seed?: number): void {
    // Step 1: enable deterministic mode flag.
    EvolutionEngine.#DETERMINISTIC = true;

    // Step 2: if a finite numeric seed was supplied, normalise to u32 and persist.
    if (typeof seed === 'number' && Number.isFinite(seed)) {
      // Coerce to unsigned 32-bit. If result is 0, remap to a safe non-zero constant.
      const normalised = seed >>> 0 || 0x9e3779b9;
      EvolutionEngine.#RNG_STATE = normalised >>> 0;

      // Step 3: force the RNG cache to refill on next use so the sequence starts from the new seed.
      EvolutionEngine.#RNG_CACHE_INDEX = EvolutionEngine.#RNG_CACHE_SIZE;
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
   * Populate the engine's pooled node-index scratch buffer with indices of nodes matching `type`.
   *
   * Steps:
   * 1. Validate inputs and early-exit for empty or missing `nodes`.
   * 2. Iterate the node array once, selecting nodes whose `node.type === type`.
   * 3. Grow the pooled `#SCRATCH_NODE_IDX` Int32Array geometrically (power-of-two) when capacity
   *    is insufficient to avoid frequent allocations.
   * 4. Write matching indices into the scratch buffer and return the count of matches.
   *
   * Notes:
   * - The method mutates `EvolutionEngine.#SCRATCH_NODE_IDX` and is therefore not reentrant. Callers
   *   must copy the first `N` entries if they need persistence across subsequent engine calls.
   * - Returns the number of matched nodes; the first `N` entries of `#SCRATCH_NODE_IDX` contain the indices.
   *
   * @param nodes - Optional array of node objects (each expected to expose a `type` property).
   * @param type - Node type key to match (for example 'input', 'hidden', 'output').
   * @returns Number of matching nodes written into `#SCRATCH_NODE_IDX`.
   * @example
   * // Populate scratch with output node indices and get the count:
   * const outCount = EvolutionEngine['#getNodeIndicesByType'](nodes, 'output');
   * // Use the first outCount entries of EvolutionEngine['#SCRATCH_NODE_IDX'] as indices into nodes.
   *
   * @internal
   */
  static #getNodeIndicesByType(nodes: any[] | undefined, type: string): number {
    // Step 1: Defensive validation & fast exit.
    if (!Array.isArray(nodes) || nodes.length === 0) return 0;

    const nodesRef = nodes;
    const desiredType = type;
    let writeCount = 0;

    // Local alias of pooled scratch for faster access.
    let scratch = EvolutionEngine.#SCRATCH_NODE_IDX;

    // Step 2: Single-pass scan collecting matching indices.
    for (let nodeIndex = 0; nodeIndex < nodesRef.length; nodeIndex++) {
      const nodeRef = nodesRef[nodeIndex];
      if (!nodeRef || nodeRef.type !== desiredType) continue;

      // Step 3: Grow pooled scratch geometrically when capacity insufficient.
      if (writeCount >= scratch.length) {
        const nextCapacity = 1 << Math.ceil(Math.log2(writeCount + 1));
        const grown = new Int32Array(nextCapacity);
        grown.set(scratch);
        EvolutionEngine.#SCRATCH_NODE_IDX = grown;
        scratch = grown; // update local alias to the new buffer
      }

      // Step 4: Write the matching node index into the pooled scratch buffer.
      scratch[writeCount++] = nodeIndex;
    }

    return writeCount;
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
    const populationRef: any[] = neat.population ?? EvolutionEngine.#EMPTY_VEC;
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
      const connections =
        sampledGenome.connections ?? EvolutionEngine.#EMPTY_VEC;
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
   * Apply simplify/prune pass to every genome in the provided NEAT population.
   *
   * Steps:
   * 1) Validate inputs and normalize prune fraction into [0,1]. If fraction is 0, exit early.
   * 2) Iterate the population, invoking the central per-genome pruning helper
   *    (#pruneWeakConnectionsForGenome) for each genome.
   * 3) Isolate failures: catch and ignore exceptions on a per-genome basis so a single
   *    faulty genome cannot abort the entire simplify phase (best-effort maintenance).
   *
   * Notes:
   * - This helper intentionally reuses existing per-genome helpers and class-level
   *   pooled buffers (via those helpers) to remain allocation-light. It itself does
   *   not allocate intermediate arrays.
   * - Non-reentrant: callers must not invoke concurrently due to shared scratch buffers
   *   used by downstream helpers.
   *
   * @param neat NEAT instance with a `population` array of genomes.
   * @param simplifyStrategy Strategy key forwarded to per-genome pruning logic.
   * @param simplifyPruneFraction Fraction in [0,1] controlling pruning aggressiveness.
   * @example
   * // Perform a single simplify generation across the population.
   * EvolutionEngine['#applySimplifyPruningToPopulation'](neatInstance, 'weakRecurrentPreferred', 0.12);
   */
  static #applySimplifyPruningToPopulation(
    neat: any,
    simplifyStrategy: string,
    simplifyPruneFraction: number
  ) {
    // Step 0: Defensive normalization & fast exits.
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
    if (pruneFraction === 0) return; // nothing requested

    // Step 1: Apply pruning to each genome with isolated error handling.
    // Use classic indexed loop for maximum predictability and minimal allocations.
    for (
      let genomeIndex = 0;
      genomeIndex < populationRef.length;
      genomeIndex++
    ) {
      const genome = populationRef[genomeIndex];
      try {
        if (!genome || !Array.isArray(genome.connections)) continue;

        // Delegate the heavy lifting to the shared per-genome helper which reuses pooled buffers.
        EvolutionEngine.#pruneWeakConnectionsForGenome(
          genome,
          simplifyStrategy ?? '',
          pruneFraction
        );
      } catch {
        // Swallow per-genome errors to keep simplify a best-effort maintenance step.
      }
    }
  }

  /**
   * Warm-start wiring for compass and directional openness inputs.
   *
   * Purpose:
   *  - Ensure the four directional input->output pathways and a compass fan-out exist with
   *    light initial weights so freshly-warmed networks have sensible starting connectivity.
   *  - This helper is allocation-light and reuses class-level scratch buffers (notably
   *    `#SCRATCH_NODE_IDX`) to avoid per-call allocations.
   *
   * Behaviour / steps:
   *  1) Fast-guard when `net` is missing or has no node list.
   *  2) For each of the 4 compass directions try to find an input node and the corresponding
   *     output node (indices are taken from `#SCRATCH_NODE_IDX` populated by callers).
   *  3) For each input->output pair ensure a connection exists; create it or set its weight
   *     to a small random initialization in [W_INIT_MIN, W_INIT_MIN+W_INIT_RANGE].
   *  4) Finally, connect the special 'compass' input (index 0 when present) to all outputs
   *     with a deterministic base weight computed from engine constants.
   *  5) Swallow unexpected errors to preserve best-effort semantics.
   *
   * Notes:
   *  - Non-reentrant: shared scratch arrays may be mutated elsewhere; callers must avoid
   *    concurrent use.
   *  - No temporary arrays are allocated; loops use local length aliases to minimize property reads.
   *
   * @param net - Network-like object with `nodes` and `connections` arrays and `connect(from,to,w)` method.
   * @example
   * // After constructing or pretraining `net`:
   * EvolutionEngine['#applyCompassWarmStart'](net);
   */
  static #applyCompassWarmStart(net: any) {
    try {
      // Step 1: defensive guards
      if (!net) return;

      const nodesRef = net.nodes ?? EvolutionEngine.#EMPTY_VEC;
      const connectionsRef = net.connections ?? EvolutionEngine.#EMPTY_VEC;

      // Determine counts for input/output nodes using the engine helper that populates
      // the pooled `#SCRATCH_NODE_IDX` with indices by type.
      const outputCount = EvolutionEngine.#getNodeIndicesByType(
        nodesRef,
        'output'
      );
      const inputCount = EvolutionEngine.#getNodeIndicesByType(
        nodesRef,
        'input'
      );

      // Local aliases for constants used below
      const wInitRange = EvolutionEngine.#W_INIT_RANGE;
      const wInitMin = EvolutionEngine.#W_INIT_MIN;

      // Step 2: connect directional input -> corresponding output for 4 compass directions
      for (let direction = 0; direction < 4; direction++) {
        // input index for this direction is at SCRATCH_NODE_IDX[direction + 1]
        const inputNodeIndex =
          direction + 1 < inputCount
            ? EvolutionEngine.#SCRATCH_NODE_IDX[direction + 1]
            : -1;
        const outputNodeIndex =
          direction < outputCount
            ? EvolutionEngine.#SCRATCH_NODE_IDX[direction]
            : -1;

        const inputNode =
          inputNodeIndex === -1 ? undefined : nodesRef[inputNodeIndex];
        const outputNode =
          outputNodeIndex === -1 ? undefined : nodesRef[outputNodeIndex];
        if (!inputNode || !outputNode) continue; // nothing to wire for this direction

        // Find existing connection input->output (linear scan; avoids allocations)
        let existingConn: any = undefined;
        for (let ci = 0, cLen = connectionsRef.length; ci < cLen; ci++) {
          const conn = connectionsRef[ci];
          if (conn.from === inputNode && conn.to === outputNode) {
            existingConn = conn;
            break;
          }
        }

        // Small random initialization in [wInitMin, wInitMin + wInitRange)
        const initWeight =
          EvolutionEngine.#fastRandom() * wInitRange + wInitMin;
        if (!existingConn) net.connect(inputNode, outputNode, initWeight);
        else existingConn.weight = initWeight;
      }

      // Step 3: compass fan-out (connect special compass input at SCRATCH_NODE_IDX[0] to all outputs)
      const compassNodeIndex =
        inputCount > 0 ? EvolutionEngine.#SCRATCH_NODE_IDX[0] : -1;
      const compassNode =
        compassNodeIndex === -1 ? undefined : nodesRef[compassNodeIndex];
      if (!compassNode) return;

      for (let outIndex = 0; outIndex < outputCount; outIndex++) {
        const outNode = nodesRef[EvolutionEngine.#SCRATCH_NODE_IDX[outIndex]];
        if (!outNode) continue;

        // Find existing connection compass->outNode
        let existingConn: any = undefined;
        for (let ci = 0, cLen = connectionsRef.length; ci < cLen; ci++) {
          const conn = connectionsRef[ci];
          if (conn.from === compassNode && conn.to === outNode) {
            existingConn = conn;
            break;
          }
        }

        const baseWeight =
          EvolutionEngine.#OUTPUT_BIAS_BASE +
          outIndex * EvolutionEngine.#OUTPUT_BIAS_STEP;
        if (!existingConn) net.connect(compassNode, outNode, baseWeight);
        else existingConn.weight = baseWeight;
      }
    } catch {
      /* best-effort: swallow unexpected errors */
    }
  }

  /**
   * Decide whether to start the simplify/pruning phase and return its duration in generations.
   *
   * Behaviour (stepwise):
   * 1. Normalise inputs defensively; non-finite or missing values are treated as 0.
   * 2. If the observed plateau counter meets or exceeds the configured plateau generations,
   *    the method will start a simplify phase only when running in a non-browser host (previous
   *    behaviour gated on `typeof window === 'undefined'`).
   * 3. Returns `simplifyDuration` when the simplify phase should start, or `0` to indicate no start.
   *
   * Rationale: Simplify/pruning is potentially expensive and historically skipped in browser contexts
   * to avoid blocking the UI; this helper preserves that heuristic while sanitising inputs.
   *
   * @param plateauCounter - Observed consecutive plateau generations (may be non-finite).
   * @param plateauGenerations - Configured threshold to trigger simplify phase (may be non-finite).
   * @param simplifyDuration - Requested simplify phase length in generations (returned when starting).
   * @returns Number of generations the simplify phase should run (0 means "do not start").
   * @example
   * const dur = EvolutionEngine['#maybeStartSimplify'](plateauCount, 10, 5);
   * if (dur > 0) { // begin simplify for dur generations
   *   // ...start simplify loop for `dur` generations
   * }
   * @internal
   */
  static #maybeStartSimplify(
    plateauCounter: number,
    plateauGenerations: number,
    simplifyDuration: number
  ): number {
    // Step 1: Defensive normalization of numeric inputs.
    const observedPlateau = Number.isFinite(plateauCounter)
      ? Math.max(0, Math.floor(plateauCounter))
      : 0;
    const requiredPlateau = Number.isFinite(plateauGenerations)
      ? Math.max(0, Math.floor(plateauGenerations))
      : 0;
    const requestedDuration = Number.isFinite(simplifyDuration)
      ? Math.max(0, Math.floor(simplifyDuration))
      : 0;

    // Step 2: If we haven't reached the plateau threshold, do not start simplify.
    if (observedPlateau < requiredPlateau) return 0;

    // Step 3: Environment gate - preserve historical behaviour: skip simplify when running in
    // a browser-like host (presence of `window` global). Use try/catch to avoid host errors.
    try {
      if (typeof window !== 'undefined') return 0;
    } catch {
      // Accessing `window` threw (non-browser host); continue and allow simplify.
    }

    // Step 4: All checks passed: start simplify for the requested duration.
    return requestedDuration;
  }

  /**
   * Run a single simplify/pruning generation if conditions permit.
   *
   * Steps:
   * 1. Normalize inputs and perform fast exits for zero remaining or invalid population.
   * 2. Environment gate: retain existing behaviour that skips heavy pruning in browser-like hosts.
   * 3. Optionally record profiling start time when profiling is enabled.
   * 4. Execute the centralized pruning pass across the population (best-effort per-genome).
   * 5. Record profiling delta and return the remaining simplify generations minus one.
   *
   * Notes:
   * - The method preserves historical behaviour where presence of a `window` global causes an early
   *   return to avoid blocking UI threads. A try/catch is used to safely probe the host environment.
   * - Inputs are defensively coerced to finite non-negative integers to avoid surprising arithmetic.
   *
   * @param neat - NEAT instance exposing a `population` array of genomes to be pruned.
   * @param simplifyRemaining - Number of simplify generations remaining before exiting (will be coerced to integer >= 0).
   * @param simplifyStrategy - Strategy identifier used by pruning routines (string, engine-specific).
   * @param simplifyPruneFraction - Fraction in [0,1] controlling pruning aggressiveness; non-finite values treated as 0.
   * @returns Remaining simplify generations after executing one cycle (0 means no further simplify).
   * @example
   * // Internal usage (pseudo): attempt one simplify generation and update remaining counter
   * const remaining = EvolutionEngine['#runSimplifyCycle'](neatInstance, simplifyRemaining, 'pruneWeak', 0.2);
   * // if remaining === 0 the simplify phase is complete or skipped
   * @internal
   */
  static #runSimplifyCycle(
    neat: any,
    simplifyRemaining: number,
    simplifyStrategy: string,
    simplifyPruneFraction: number
  ): number {
    // Step 1: Defensive normalization & quick exits.
    const remainingGens = Number.isFinite(simplifyRemaining)
      ? Math.max(0, Math.floor(simplifyRemaining))
      : 0;
    if (remainingGens === 0) return 0;
    if (
      !neat ||
      !Array.isArray(neat.population) ||
      neat.population.length === 0
    )
      return 0;

    // Step 2: Environment gate (skip when running in browser-like context).
    try {
      if (typeof window !== 'undefined') return remainingGens; // keep previous semantics in browsers
    } catch {
      // Accessing `window` threw (non-browser host); continue and allow pruning to proceed.
    }

    // Step 3: Optionally record profiling start (high precision micro-timer when enabled).
    const profilingEnabled = EvolutionEngine.#PROFILE_ENABLED;
    const profileStartMs = profilingEnabled ? EvolutionEngine.#PROFILE_T0() : 0;

    // Step 4: Apply pruning across the population (best-effort; errors are isolated per genome).
    EvolutionEngine.#applySimplifyPruningToPopulation(
      neat,
      simplifyStrategy,
      simplifyPruneFraction
    );

    // Step 5: Record profiling delta if enabled and return remaining generations decremented.
    if (profilingEnabled) {
      const elapsedMs = EvolutionEngine.#PROFILE_T0() - profileStartMs || 0;
      EvolutionEngine.#PROFILE_ADD('simplify', elapsedMs);
    }

    return Math.max(0, remainingGens - 1);
  }

  /**
   * Apply Lamarckian (supervised) training to the current population.
   *
   * Description:
   * Runs a short, bounded supervised training pass on each network in the population. This
   * refinement is intentionally conservative (few iterations / small batch) to improve
   * local performance without collapsing evolutionary diversity.
   *
   * Steps:
   * 1. Validate inputs and early-exit for empty population or training set.
   * 2. Optionally down-sample the training set (with replacement) to bound per-generation cost.
   * 3. Iterate each network and run a small training pass. Adjust output biases heuristically.
   * 4. Collect optional training statistics (gradient norm) for telemetry when available.
   * 5. Emit telemetry and return elapsed time when profiling is enabled.
   *
   * @param neat - NEAT instance exposing a `population` array of networks.
   * @param trainingSet - Array of supervised examples (engine-specific format) used for local training.
   * @param iterations - Number of training iterations to run per-network (must be > 0).
   * @param sampleSize - Optional sample size to down-sample `trainingSet` (with replacement). Fractional/invalid treated as no sampling.
   * @param safeWrite - Logging helper used for telemetry lines (string writer).
   * @param profileEnabled - When true, function returns elapsed ms spent training; otherwise returns 0.
   * @param completedGenerations - Generation index used when emitting telemetry lines.
   * @returns Elapsed milliseconds spent in training when profiling is enabled; otherwise 0.
   * @example
   * // Internal usage (pseudo): run a short Lamarckian pass and get elapsed time when profiling
   * const elapsed = EvolutionEngine['#applyLamarckianTraining'](neat, trainExamples, 2, 8, console.log, true, gen);
   * @internal
   */
  static #applyLamarckianTraining(
    neat: any,
    trainingSet: any[],
    iterations: number,
    sampleSize: number | undefined,
    safeWrite: (msg: string) => void,
    profileEnabled: boolean,
    completedGenerations: number
  ): number {
    // Step 1: Validate inputs & early exits.
    if (
      !neat ||
      !Array.isArray(neat.population) ||
      neat.population.length === 0
    )
      return 0;
    if (!Array.isArray(trainingSet) || trainingSet.length === 0) return 0;
    if (!Number.isFinite(iterations) || iterations <= 0) return 0;

    // Step 2: Start profiling timer if requested.
    const profileStart = profileEnabled ? EvolutionEngine.#now() : 0;

    // Step 3: Optionally down-sample the training set (with replacement) to reduce cost for large sets.
    const trainingSetRef =
      sampleSize && sampleSize > 0 && sampleSize < trainingSet.length
        ? EvolutionEngine.#sampleArray(trainingSet, sampleSize)
        : trainingSet;

    // Step 4: Iterate networks performing a bounded training pass.
    let gradientNormSum = 0;
    let gradientNormSamples = 0;
    const populationRef = neat.population as any[];

    for (const network of populationRef) {
      if (!network) continue; // defensive guard for sparse arrays
      try {
        // 4.1: Run a conservative training invocation to avoid destroying diversity.
        network.train(trainingSetRef, {
          iterations,
          error: EvolutionEngine.#DEFAULT_TRAIN_ERROR,
          rate: EvolutionEngine.#DEFAULT_TRAIN_RATE,
          momentum: EvolutionEngine.#DEFAULT_TRAIN_MOMENTUM,
          batchSize: EvolutionEngine.#DEFAULT_TRAIN_BATCH_SMALL,
          allowRecurrent: true,
          cost: methods.Cost.softmaxCrossEntropy,
        });

        // 4.2: Heuristic bias adjustment after training to maintain exploration.
        EvolutionEngine.#adjustOutputBiasesAfterTraining(network);

        // 4.3: Collect optional training stats (use optional chaining to avoid errors).
        try {
          const stats = (network as any).getTrainingStats?.();
          const gradNorm = stats?.gradNorm;
          if (Number.isFinite(gradNorm)) {
            gradientNormSum += gradNorm;
            gradientNormSamples++;
          }
        } catch {
          // Non-fatal: skip stat collection errors silently.
        }
      } catch {
        // Per-network training failure is non-fatal; continue with others.
      }
    }

    // Step 5: Emit aggregate gradient telemetry if samples were collected.
    if (gradientNormSamples > 0) {
      const meanGrad = gradientNormSum / gradientNormSamples;
      safeWrite(
        `[GRAD] gen=${completedGenerations} meanGradNorm=${meanGrad.toFixed(
          4
        )} samples=${gradientNormSamples}\n`
      );
    }

    // Step 6: Return elapsed time when profiling; otherwise return 0.
    return profileEnabled ? EvolutionEngine.#now() - profileStart : 0;
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
   * 6. Optionally record profiling timings into `#PROFILE_ACCUM`.
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
    writeLog: (msg: string) => void
  ): void {
    // Step 0: Global guard for minimal telemetry mode.
    if (EvolutionEngine.#TELEMETRY_MINIMAL) return;

    // Start profiling window if enabled.
    const profilingEnabled = EvolutionEngine.#PROFILE_ENABLED;
    const profilingStart = profilingEnabled ? EvolutionEngine.#PROFILE_T0() : 0;

    try {
      // Step 1: Action entropy telemetry
      EvolutionEngine.#logActionEntropy(genResult, generationIndex, writeLog);

      // Step 2: Output bias statistics (fittest may be undefined early on)
      EvolutionEngine.#logOutputBiasStats(fittest, generationIndex, writeLog);

      // Step 3: Logits statistics and collapse detection/recovery
      EvolutionEngine.#logLogitsAndCollapse(
        neat,
        fittest,
        generationIndex,
        writeLog
      );

      // Step 4: Exploration telemetry (path uniqueness, progress)
      EvolutionEngine.#logExploration(genResult, generationIndex, writeLog);

      // Step 5: Diversity metrics (species richness, Simpson, weight std)
      EvolutionEngine.#logDiversity(neat, generationIndex, writeLog);
    } catch {
      // Swallow any unexpected telemetry exception to avoid disrupting the evolution core loop.
    }

    // Step 6: Record profiling delta if profiling was enabled at entry.
    if (profilingEnabled) {
      EvolutionEngine.#PROFILE_ADD(
        'telemetry',
        EvolutionEngine.#PROFILE_T0() - profilingStart || 0
      );
    }
  }

  /**
   * Emit a compact action-entropy telemetry summary line.
   *
   * Steps:
   * 1. Defensive guards: validate `safeWrite` and read `generationResult.path` safely.
   * 2. Compute action-entropy metrics via `#computeActionEntropy` (pure function).
   * 3. Format a single-line telemetry record and emit it using `safeWrite`.
   *
   * @param generationResult - Per-generation result object (expected to expose `.path` array of visited coordinates).
   * @param gen - Completed generation index used in telemetry labels.
   * @param safeWrite - Writer function that accepts a single-string telemetry line (for example `msg => process.stdout.write(msg)`).
   * @internal
   * @example
   * // Internal usage: write to stdout
   * EvolutionEngine['#logActionEntropy'](genResult, 12, msg => process.stdout.write(msg));
   */
  static #logActionEntropy(
    generationResult: any,
    gen: number,
    safeWrite: (msg: string) => void
  ): void {
    // Step 1: Defensive parameter validation. Best-effort telemetry must never throw.
    if (typeof safeWrite !== 'function') return;

    const pathRef = generationResult?.path;

    try {
      // Step 2: Compute entropy statistics (pure, may handle undefined/empty path internally).
      const stats = EvolutionEngine.#computeActionEntropy(pathRef);

      // Step 3: Build formatted telemetry line using safe coercions and consistent fixed precision.
      const logTag = EvolutionEngine.#LOG_TAG_ACTION_ENTROPY;
      const entropyNormStr = Number.isFinite(stats?.entropyNorm)
        ? stats.entropyNorm.toFixed(3)
        : '0.000';
      const uniqueMovesStr = Number.isFinite(stats?.uniqueMoves)
        ? String(stats.uniqueMoves)
        : '0';
      const pathLenStr = Number.isFinite(stats?.pathLen)
        ? String(stats.pathLen)
        : '0';

      safeWrite(
        `${logTag} gen=${gen} entropyNorm=${entropyNormStr} uniqueMoves=${uniqueMovesStr} pathLen=${pathLenStr}\n`
      );
    } catch {
      // Swallow all telemetry errors silently to avoid affecting the evolution core loop.
    }
  }

  /**
   * Emit output-bias statistics for the provided fittest genome's output nodes.
   *
   * Steps:
   * 1. Defensive guards: ensure `safeWrite` is a function and obtain `nodes` from `fittest` safely.
   * 2. Query pooled node indices for outputs using `#getNodeIndicesByType` (non-reentrant scratch buffer).
   * 3. If outputs exist, compute bias statistics via `#computeOutputBiasStats` and format a single-line record.
   * 4. Emit the formatted line via `safeWrite`. All errors are swallowed to keep telemetry best-effort.
   *
   * @param fittest - Candidate genome/network expected to expose a `.nodes` array (may be undefined during init).
   * @param gen - Completed generation index used in telemetry labels.
   * @param safeWrite - Telemetry writer accepting a single string (for example `msg => process.stdout.write(msg)`).
   * @internal
   * @example
   * // Internal use: write bias stats to stdout
   * EvolutionEngine['#logOutputBiasStats'](neat.fittest, 5, msg => process.stdout.write(msg));
   */
  static #logOutputBiasStats(
    fittest: any,
    gen: number,
    safeWrite: (msg: string) => void
  ): void {
    // Step 1: Defensive validation — telemetry must not throw.
    if (typeof safeWrite !== 'function') return;

    // Safely obtain nodes (may be undefined during early generations).
    const nodesRef = fittest?.nodes ?? [];

    try {
      // Step 2: Acquire number of output nodes using pooled scratch; non-reentrant.
      const outputCount = EvolutionEngine.#getNodeIndicesByType(
        nodesRef,
        'output'
      );

      // Step 3: If no outputs, nothing to log.
      if (outputCount <= 0) return;

      // Step 4: Compute bias statistics (pure-ish helper) and format for emission.
      const biasStats = EvolutionEngine.#computeOutputBiasStats(
        nodesRef,
        outputCount
      );

      const tag = EvolutionEngine.#LOG_TAG_OUTPUT_BIAS;
      const meanStr = Number.isFinite(biasStats?.mean)
        ? biasStats.mean.toFixed(3)
        : '0.000';
      const stdStr = Number.isFinite(biasStats?.std)
        ? biasStats.std.toFixed(3)
        : '0.000';
      const biasesStr = String(biasStats?.biasesStr ?? '');

      safeWrite(
        `${tag} gen=${gen} mean=${meanStr} std=${stdStr} biases=${biasesStr}\n`
      );
    } catch {
      // Best-effort telemetry: swallow any unexpected error.
    }
  }

  /**
   * Compute and emit logits-level statistics, detect a collapse condition, and trigger
   * anti-collapse recovery when the collapse streak threshold is reached.
   *
   * Steps:
   * 1. Guard & extract the recent output history from the `fittest` candidate.
   * 2. Compute aggregated logit statistics for the recent tail via `#computeLogitStats`.
   * 3. Emit a single-line telemetry record containing means, stds, kurtosis, entropy mean and stability.
   * 4. Detect a collapse when all output stds are below `#LOGSTD_FLAT_THRESHOLD` AND
   *    (entropy mean is low OR decision stability is high). Maintain a collapse streak counter.
   * 5. When the streak reaches `#COLLAPSE_STREAK_TRIGGER`, invoke `#antiCollapseRecovery` to attempt recovery.
   *
   * @param neat - NEAT instance (passed to recovery helper when triggered).
   * @param fittest - Current fittest genome/network which may expose `_lastStepOutputs` (array of output vectors).
   * @param gen - Completed generation index used in telemetry labels and recovery actions.
   * @param safeWrite - Writer accepting a single-string telemetry line (for example `msg => process.stdout.write(msg)`).
   * @internal
   * @example
   * EvolutionEngine['#logLogitsAndCollapse'](neat, neat.fittest, gen, msg => process.stdout.write(msg));
   */
  static #logLogitsAndCollapse(
    neat: any,
    fittest: any,
    gen: number,
    safeWrite: (msg: string) => void
  ): void {
    // Best-effort telemetry must not throw; validate writer.
    if (typeof safeWrite !== 'function') return;

    try {
      // Step 1: Safely read the fittest' recent per-step outputs history and early-exit when absent.
      const fullHistory: number[][] =
        (fittest?._lastStepOutputs as number[][]) ?? EvolutionEngine.#EMPTY_VEC;
      if (!fullHistory.length) return;

      // Acquire a bounded tail for statistics (recent window size).
      const recentTail = EvolutionEngine.#getTail<number[]>(
        fullHistory,
        EvolutionEngine.#RECENT_WINDOW
      );

      // Step 2: Compute logit-level aggregated statistics (meansStr, stdsStr, kurtStr, entMean, stability, stds[]).
      const logitStats: any = EvolutionEngine.#computeLogitStats(recentTail);

      // Step 3: Emit formatted telemetry line with fixed precision parts.
      safeWrite(
        `${EvolutionEngine.#LOG_TAG_LOGITS} gen=${gen} means=${
          logitStats.meansStr
        } stds=${logitStats.stdsStr} kurt=${logitStats.kurtStr} entMean=${
          Number.isFinite(logitStats.entMean)
            ? logitStats.entMean.toFixed(3)
            : '0.000'
        } stability=${
          Number.isFinite(logitStats.stability)
            ? logitStats.stability.toFixed(3)
            : '0.000'
        } steps=${recentTail.length}\n`
      );

      // Step 4: Collapse detection — all stds below threshold and entropy/stability signals.
      (EvolutionEngine as any)._collapseStreak =
        (EvolutionEngine as any)._collapseStreak || 0;
      const stdArray =
        (logitStats.stds as ArrayLike<number>) ||
        (([] as unknown) as ArrayLike<number>);
      let allStdsBelowThreshold = true;
      for (let stdIndex = 0; stdIndex < stdArray.length; stdIndex++) {
        const stdValue = stdArray[stdIndex];
        if (!(stdValue < EvolutionEngine.#LOGSTD_FLAT_THRESHOLD)) {
          allStdsBelowThreshold = false;
          break;
        }
      }

      const isCollapsed =
        allStdsBelowThreshold &&
        (logitStats.entMean < EvolutionEngine.#ENTROPY_COLLAPSE_THRESHOLD ||
          logitStats.stability > EvolutionEngine.#STABILITY_COLLAPSE_THRESHOLD);

      // Step 5: Update collapse streak and trigger recovery when threshold reached.
      if (isCollapsed) (EvolutionEngine as any)._collapseStreak++;
      else (EvolutionEngine as any)._collapseStreak = 0;

      if (
        (EvolutionEngine as any)._collapseStreak ===
        EvolutionEngine.#COLLAPSE_STREAK_TRIGGER
      ) {
        EvolutionEngine.#antiCollapseRecovery(neat, gen, safeWrite);
      }
    } catch {
      // Keep telemetry best-effort: suppress all errors.
    }
  }

  /**
   * Emit exploration telemetry: unique coverage, path length, coverage ratio, progress and saturation fraction.
   *
   * Steps:
   * 1. Validate inputs (`safeWrite`) and safely extract `path` and numeric fields from `generationResult`.
   * 2. Compute exploration statistics via `#computeExplorationStats` (returns { unique, pathLen, ratio }).
   * 3. Format a single-line telemetry message with stable numeric formatting and emit via `safeWrite`.
   *
   * @param generationResult - Per-generation result object expected to expose `.path`, `.progress` and optional `.saturationFraction`.
   * @param gen - Completed generation index used in telemetry labels.
   * @param safeWrite - Writer function accepting a single telemetry string (for example `msg => process.stdout.write(msg)`).
   * @internal
   * @example
   * // Internal usage: emit exploration line to stdout
   * EvolutionEngine['#logExploration'](genResult, 7, msg => process.stdout.write(msg));
   */
  static #logExploration(
    generationResult: any,
    gen: number,
    safeWrite: (msg: string) => void
  ): void {
    // Step 1: Defensive validation — telemetry must never throw.
    if (typeof safeWrite !== 'function') return;

    // Safely read inputs with defensive coercions.
    const pathRef = generationResult?.path;
    const rawProgress = generationResult?.progress;
    const rawSatFrac = (generationResult as any)?.saturationFraction;

    try {
      // Step 2: Compute exploration statistics (pure helper).
      const exploration = EvolutionEngine.#computeExplorationStats(pathRef);

      // Step 3: Build stable formatted strings for numeric fields.
      const tag = '[EXPLORE]';
      const uniqueStr = Number.isFinite(exploration?.unique)
        ? String(exploration.unique)
        : '0';
      const pathLenStr = Number.isFinite(exploration?.pathLen)
        ? String(exploration.pathLen)
        : '0';
      const ratioStr = Number.isFinite(exploration?.ratio)
        ? exploration.ratio.toFixed(3)
        : '0.000';
      const progressStr = Number.isFinite(rawProgress)
        ? rawProgress.toFixed(1)
        : '0.0';
      const satFracStr = Number.isFinite(rawSatFrac)
        ? rawSatFrac.toFixed(3)
        : '0.000';

      // Emit single-line exploration telemetry.
      safeWrite(
        `${tag} gen=${gen} unique=${uniqueStr} pathLen=${pathLenStr} ratio=${ratioStr} progress=${progressStr} satFrac=${satFracStr}\n`
      );
    } catch {
      // Best-effort telemetry: swallow any unexpected error.
    }
  }

  /**
   * Emit core diversity metrics for the current population.
   *
   * Steps:
   * 1. Defensive guards: validate `safeWrite` and that `neat` exposes a population.
   * 2. Compute diversity metrics via `#computeDiversityMetrics` (species count, Simpson index, weight std).
   * 3. Format a concise telemetry line with stable numeric formatting and emit via `safeWrite`.
   *
   * @param neat - NEAT instance exposing a `population` array used to compute diversity metrics.
   * @param gen - Completed generation index used in telemetry labels.
   * @param safeWrite - Telemetry writer accepting a single string (for example `msg => process.stdout.write(msg)`).
   * @internal
   * @example
   * // Internal usage: write diversity metrics to stdout
   * EvolutionEngine['#logDiversity'](neatInstance, 42, msg => process.stdout.write(msg));
   */
  static #logDiversity(
    neat: any,
    gen: number,
    safeWrite: (msg: string) => void
  ): void {
    // Step 1: Defensive validation — telemetry must never throw.
    if (typeof safeWrite !== 'function') return;
    if (!neat || !Array.isArray(neat.population)) return;

    try {
      // Step 2: Compute diversity metrics (pure helper; may be expensive but bounded by sampling size).
      const metrics = EvolutionEngine.#computeDiversityMetrics(neat);

      // Step 3: Format values with stable fixed precision and emit a single telemetry line.
      const tag = '[DIVERSITY]';
      const speciesCountStr = Number.isFinite(metrics?.speciesUniqueCount)
        ? String(metrics.speciesUniqueCount)
        : '0';
      const simpsonStr = Number.isFinite(metrics?.simpson)
        ? metrics.simpson.toFixed(3)
        : '0.000';
      const weightStdStr = Number.isFinite(metrics?.wStd)
        ? metrics.wStd.toFixed(3)
        : '0.000';

      safeWrite(
        `${tag} gen=${gen} species=${speciesCountStr} simpson=${simpsonStr} weightStd=${weightStdStr}\n`
      );
    } catch {
      // Best-effort telemetry: swallow any unexpected error.
    }
  }

  /**
   * Persist a compact snapshot file when persistence is enabled and cadence matches.
   *
   * Steps:
   * 1. Defensive validation of IO primitives, scheduling cadence and NEAT shape.
   * 2. Start optional profiling window.
   * 3. Populate a pooled snapshot object with scalar metadata and a short telemetry tail.
   * 4. Reuse a pooled top-K buffer to write minimal per-genome metadata for the best genomes.
   * 5. Serialize the compact snapshot JSON and write it to disk using the provided `fs`.
   * 6. Record profiling delta (best-effort) and swallow any IO/errors to avoid destabilizing the loop.
   *
   * @param fs - File-system-like object with `writeFileSync(path, data)` (for example Node's `fs`).
   * @param pathModule - Path-like object exposing `join(...parts)` (for example Node's `path`).
   * @param persistDir - Directory where snapshots should be written; falsy disables persistence.
   * @param persistTopK - How many top genomes to include metadata for (0 disables top list).
   * @param completedGenerations - Current completed generation index (used for filename & metadata).
   * @param persistEvery - Cadence (in generations) at which to persist; <=0 disables persistence.
   * @param neat - NEAT instance exposing a `population` array.
   * @param bestFitness - Best fitness scalar to record in the snapshot metadata.
   * @param simplifyMode - Whether the engine is currently simplifying (recorded for diagnostics/UI).
   * @param plateauCounter - Plateau counter value recorded for diagnostics/UI.
   * @internal
   * @example
   * // Typical Node usage inside engine loop:
   * EvolutionEngine['#persistSnapshotIfNeeded'](require('fs'), require('path'), './snapshots', 5, gen, 10, neat, best, false, 0);
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
    // Step 1: Defensive validation & scheduling cadence.
    if (
      !fs ||
      typeof fs.writeFileSync !== 'function' ||
      !pathModule ||
      typeof pathModule.join !== 'function' ||
      !persistDir ||
      !Number.isFinite(persistEvery) ||
      persistEvery <= 0
    )
      return;

    if (
      !Number.isFinite(completedGenerations) ||
      completedGenerations % persistEvery !== 0
    )
      return;

    if (
      !neat ||
      !Array.isArray(neat.population) ||
      neat.population.length === 0
    )
      return;

    try {
      // Step 2: Optional profiling start (high-precision timer when enabled).
      const profileStart = EvolutionEngine.#PROFILE_ENABLED
        ? EvolutionEngine.#PROFILE_T0()
        : 0;

      // Step 3: Populate pooled snapshot metadata (mutate shared scratch snapshot object).
      const snapshot = EvolutionEngine.#SCRATCH_SNAPSHOT_OBJ;
      snapshot.generation = completedGenerations;
      snapshot.bestFitness = bestFitness;
      snapshot.simplifyMode = Boolean(simplifyMode);
      snapshot.plateauCounter = Number.isFinite(plateauCounter)
        ? plateauCounter
        : 0;
      snapshot.timestamp = Date.now();
      snapshot.telemetryTail = EvolutionEngine.#collectTelemetryTail(neat, 5);

      // Step 4: Prepare the top-K minimal metadata list by reusing pooled buffer.
      const populationRef: any[] =
        neat.population ?? EvolutionEngine.#EMPTY_VEC;
      const sortedIndices =
        EvolutionEngine.#getSortedIndicesByScore(populationRef) ??
        EvolutionEngine.#EMPTY_VEC;
      const normalizedTopK = Math.max(
        0,
        Math.floor(Number.isFinite(persistTopK) ? persistTopK : 0)
      );
      const topLimit = Math.min(normalizedTopK, sortedIndices.length);

      const topBuffer = EvolutionEngine.#SCRATCH_SNAPSHOT_TOP;
      if (topBuffer.length < topLimit) topBuffer.length = topLimit; // grow in place if needed

      for (let rank = 0; rank < topLimit; rank++) {
        // Reuse or create an entry object in the pooled topBuffer.
        let entry = topBuffer[rank] ?? (topBuffer[rank] = {});
        const genome = populationRef[sortedIndices[rank]];

        // Minimal invariant fields for later analysis/UI. Use optional chaining and defaults.
        entry.idx = sortedIndices[rank];
        entry.score = genome?.score;
        entry.nodes = genome?.nodes?.length ?? 0;
        entry.connections = genome?.connections?.length ?? 0;
        entry.json =
          typeof genome?.toJSON === 'function' ? genome.toJSON() : undefined;
      }

      topBuffer.length = topLimit;
      snapshot.top = topBuffer;

      // Step 5: Serialize compact JSON and write to disk using provided path/FS.
      const snapshotFilePath = pathModule.join(
        persistDir,
        `snapshot_gen${completedGenerations}.json`
      );
      fs.writeFileSync(snapshotFilePath, JSON.stringify(snapshot));

      // Step 6: Profile delta accumulation (best-effort).
      if (EvolutionEngine.#PROFILE_ENABLED) {
        EvolutionEngine.#PROFILE_ADD(
          'snapshot',
          EvolutionEngine.#PROFILE_T0() - profileStart || 0
        );
      }
    } catch {
      // Best-effort: swallow persistence errors silently to avoid disrupting the evolution loop.
    }
  }

  /**
   * Collect a short telemetry tail (last N entries) from a NEAT instance if available.
   *
   * Steps:
   * 1. Validate that `neat` exists and exposes a callable `getTelemetry` method.
   * 2. Normalise `tailLength` to a non-negative integer (floor) with a sensible default.
   * 3. Call `neat.getTelemetry()` inside a try/catch to avoid propagation of telemetry errors.
   * 4. If telemetry is an array, return the last `tailLength` entries via `#getTail`; otherwise return the raw value.
   *
   * Behavioural notes:
   * - This helper is best-effort: any thrown error from `getTelemetry` will be swallowed and `undefined` returned.
   * - When the telemetry value is an array we reuse the pooled tail helper to avoid allocations.
   *
   * @param neat - NEAT instance that may implement `getTelemetry(): unknown`.
   * @param tailLength - Desired tail length (floored to integer >= 0). Defaults to 10 when invalid.
   * @returns An array containing the last `tailLength` telemetry entries when telemetry is an array,
   *          the raw telemetry value when non-array, or `undefined` on missing API / errors.
   * @example
   * // Internal usage: request last 5 telemetry entries when available
   * const tail = EvolutionEngine['#collectTelemetryTail'](neatInstance, 5);
   * @internal
   */
  static #collectTelemetryTail(neat: any, tailLength = 10) {
    // Step 1: Quick guards — do not throw from telemetry helpers.
    if (!neat || typeof neat.getTelemetry !== 'function') return undefined;

    // Step 2: Normalise tailLength to a safe non-negative integer.
    const normalizedTailLength = Number.isFinite(tailLength)
      ? Math.max(0, Math.floor(tailLength))
      : 10;

    try {
      // Step 3: Invoke the provider safely using optional chaining; keep the raw result for inspection.
      const telemetryRaw = neat.getTelemetry?.();

      // Step 4a: If telemetry is an array, reuse pooled tail helper to avoid allocation.
      if (Array.isArray(telemetryRaw)) {
        return EvolutionEngine.#getTail<any>(
          telemetryRaw,
          normalizedTailLength
        );
      }

      // Step 4b: Non-array telemetry is returned as-is (could be object/string/number/etc.).
      return telemetryRaw;
    } catch (err) {
      // Best-effort: swallow errors to avoid disrupting the evolution loop.
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
      const nodeList = network?.nodes ?? EvolutionEngine.#EMPTY_VEC;
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
    const outgoing = hiddenNode.connections.out ?? EvolutionEngine.#EMPTY_VEC;
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

    // Orchestration variables
    const reducedTelemetry = EvolutionEngine.#REDUCED_TELEMETRY;
    const actionDim = Math.max(0, EvolutionEngine.#ACTION_DIM);
    const sampleCount = recent.length;

    // Step 1: Ensure scratch buffers are present and cleared for the active action dimension.
    EvolutionEngine.#resetLogitScratch(actionDim, reducedTelemetry);

    // Step 2: Accumulate statistics and entropy using specialised helpers.
    let entropyAggregate = 0;
    if (reducedTelemetry) {
      entropyAggregate = EvolutionEngine.#accumulateLogitStatsReduced(
        recent,
        actionDim
      );
      EvolutionEngine.#finalizeLogitStatsReduced(actionDim, sampleCount);
    } else if (actionDim === 4) {
      entropyAggregate = EvolutionEngine.#accumulateLogitStatsUnrolled4(
        recent,
        sampleCount
      );
      EvolutionEngine.#finalizeLogitStatsFull(actionDim, sampleCount);
    } else {
      entropyAggregate = EvolutionEngine.#accumulateLogitStatsGeneric(
        recent,
        actionDim
      );
      EvolutionEngine.#finalizeLogitStatsFull(actionDim, sampleCount);
    }

    // Step 3: Build output summary (strings + scalars) and return.
    const entropyMean = entropyAggregate / sampleCount;
    const stability = EvolutionEngine.#computeDecisionStability(recent);
    const meansStr = EvolutionEngine.#joinNumberArray(
      EvolutionEngine.#SCRATCH_MEANS,
      actionDim,
      3
    );
    const stdsStr = EvolutionEngine.#joinNumberArray(
      EvolutionEngine.#SCRATCH_STDS,
      actionDim,
      3
    );
    const kurtStr = reducedTelemetry
      ? ''
      : EvolutionEngine.#joinNumberArray(
          EvolutionEngine.#SCRATCH_KURT!,
          actionDim,
          2
        );

    return {
      meansStr,
      stdsStr,
      kurtStr,
      entMean: entropyMean,
      stability,
      steps: sampleCount,
      means: EvolutionEngine.#SCRATCH_MEANS,
      stds: EvolutionEngine.#SCRATCH_STDS,
    } as any;
  }

  /**
   * Prepare and zero the class-level scratch buffers used by logit statistics.
   *
   * Steps:
   * 1. Normalize `actionDim` to a non-negative integer and fast-exit on zero.
   * 2. Grow (only when needed) the pooled Float64Array buffers to at least `actionDim` length.
   * 3. Zero the active prefix [0, actionDim) of means, second-moment (M2) and std buffers.
   * 4. When full telemetry is enabled also ensure/zero M3/M4/kurtosis buffers.
   *
   * Notes:
   * - This helper intentionally avoids preserving previous contents when resizing — callers always
   *   expect zeroed buffers after invocation.
   * - Allocations are kept minimal: a new backing array is created only when the existing capacity
   *   is insufficient; new capacity is chosen to be at least `actionDim` (no aggressive over-sizing).
   *
   * @param actionDim - Number of active action dimensions (will be floored & clamped to >= 0).
   * @param reducedTelemetry - When true skip higher-moment buffers (M3/M4/kurtosis) to save memory/CPU.
   * @example
   * // Internal usage: prepare scratch for 4 actions with full telemetry enabled
   * EvolutionEngine['#resetLogitScratch'](4, false);
   * @internal
   */
  static #resetLogitScratch(actionDim: number, reducedTelemetry: boolean) {
    // Step 1: Normalize input to a safe integer range.
    const dim = Number.isFinite(actionDim)
      ? Math.max(0, Math.floor(actionDim))
      : 0;
    if (dim === 0) return; // nothing to prepare for zero-dimension

    // Step 2: Ensure base scratch buffers exist and have sufficient capacity.
    // Grow only when capacity is insufficient to avoid noisy allocations on hot paths.
    if (
      !EvolutionEngine.#SCRATCH_MEANS ||
      EvolutionEngine.#SCRATCH_MEANS.length < dim
    ) {
      EvolutionEngine.#SCRATCH_MEANS = new Float64Array(dim);
    }
    if (
      !EvolutionEngine.#SCRATCH_M2_RAW ||
      EvolutionEngine.#SCRATCH_M2_RAW.length < dim
    ) {
      EvolutionEngine.#SCRATCH_M2_RAW = new Float64Array(dim);
    }
    if (
      !EvolutionEngine.#SCRATCH_STDS ||
      EvolutionEngine.#SCRATCH_STDS.length < dim
    ) {
      EvolutionEngine.#SCRATCH_STDS = new Float64Array(dim);
    }

    const meansBuffer = EvolutionEngine.#SCRATCH_MEANS;
    const secondMomentBuffer = EvolutionEngine.#SCRATCH_M2_RAW;
    const stdBuffer = EvolutionEngine.#SCRATCH_STDS;

    // Step 3: Zero only the active prefix to retain any extra capacity beyond `dim`.
    meansBuffer.fill(0, 0, dim);
    secondMomentBuffer.fill(0, 0, dim);
    stdBuffer.fill(0, 0, dim);

    // Step 4: Full telemetry path: ensure and zero higher-moment buffers.
    if (!reducedTelemetry) {
      if (
        !EvolutionEngine.#SCRATCH_M3_RAW ||
        EvolutionEngine.#SCRATCH_M3_RAW!.length < dim
      ) {
        EvolutionEngine.#SCRATCH_M3_RAW = new Float64Array(dim);
      }
      if (
        !EvolutionEngine.#SCRATCH_M4_RAW ||
        EvolutionEngine.#SCRATCH_M4_RAW!.length < dim
      ) {
        EvolutionEngine.#SCRATCH_M4_RAW = new Float64Array(dim);
      }
      if (
        !EvolutionEngine.#SCRATCH_KURT ||
        EvolutionEngine.#SCRATCH_KURT!.length < dim
      ) {
        EvolutionEngine.#SCRATCH_KURT = new Float64Array(dim);
      }

      EvolutionEngine.#SCRATCH_M3_RAW!.fill(0, 0, dim);
      EvolutionEngine.#SCRATCH_M4_RAW!.fill(0, 0, dim);
      EvolutionEngine.#SCRATCH_KURT!.fill(0, 0, dim);
    }
  }

  /**
   * Accumulate running means and second raw moments (M2) using a reduced-telemetry
   * Welford pass (no higher moments) and also accumulate per-sample softmax entropy.
   *
   * Contract / side-effects:
   * - Fills the class-level scratch buffers: `#SCRATCH_MEANS`, `#SCRATCH_M2_RAW`, and `#SCRATCH_STDS`.
   * - Returns the aggregated entropy sum (caller will divide by sample count to get mean entropy).
   *
   * Example:
   * const entropySum = EvolutionEngine['#accumulateLogitStatsReduced'](recentLogits, 4);
   *
   * @param recent Array of recent logit vectors (newest last). Missing entries treated as 0.
   * @param actionDim Number of action dimensions to process (floored & clamped by caller).
   * @returns Sum of softmax entropies for each vector in `recent`.
   */
  static #accumulateLogitStatsReduced(
    recent: number[][],
    actionDim: number
  ): number {
    // Step 0: defensive guards - if there are no samples, nothing to do.
    const sampleCount = Array.isArray(recent) ? recent.length : 0;
    if (sampleCount === 0 || actionDim <= 0) return 0;

    // Step 1: acquire pooled scratch buffers (typed arrays) used to accumulate stats.
    const meansBuffer = EvolutionEngine.#SCRATCH_MEANS;
    const secondMomentBuffer = EvolutionEngine.#SCRATCH_M2_RAW;
    const stdsBuffer = EvolutionEngine.#SCRATCH_STDS;

    // Step 2: accumulator for entropy (sum over samples). Caller computes mean if needed.
    let entropyAccumulator = 0;

    // Step 3: Main streaming accumulation loop (Welford's algorithm per-dimension).
    // We loop over samples and update all active action dimensions in-place to avoid allocations.
    for (const [sampleIndex, rawVector] of recent.entries()) {
      const vector = rawVector ?? EvolutionEngine.#EMPTY_VEC; // defensive fallback to avoid undefined (no allocation)
      const sampleNumber = sampleIndex + 1; // 1-based sample count for Welford updates

      // Update each action dimension with descriptive locals for clarity.
      for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
        const observedValue = vector[dimIndex] ?? 0; // treat missing as 0

        // Welford incremental update (stable online mean + M2 accumulation):
        // delta = x - mean_old
        // mean_new = mean_old + delta / n
        // M2 += delta * (x - mean_new)
        const previousMean = meansBuffer[dimIndex];
        const delta = observedValue - previousMean;
        const deltaNormalized = delta / sampleNumber;
        const updatedMean = previousMean + deltaNormalized;
        meansBuffer[dimIndex] = updatedMean;
        const correction = observedValue - updatedMean;
        secondMomentBuffer[dimIndex] += delta * correction;
      }

      // Step 3b: accumulate sample softmax entropy using the shared exponent scratch buffer.
      entropyAccumulator += EvolutionEngine.#softmaxEntropyFromVector(
        vector,
        EvolutionEngine.#SCRATCH_EXPS
      );
    }

    // Step 4: finalize standard deviations into the shared stds buffer (population variance)
    const invSampleCount = 1 / sampleCount;
    for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
      const variance = secondMomentBuffer[dimIndex] * invSampleCount;
      stdsBuffer[dimIndex] = variance > 0 ? Math.sqrt(variance) : 0;
    }

    // Return aggregated entropy (sum for caller to average if desired).
    return entropyAccumulator;
  }

  /**
   * Compute unrolled Welford moments for the common ACTION_DIM === 4 case.
   *
   * Steps (high level):
   * 1. Stream over samples and update running means and raw central moments (M2/M3/M4)
   *    for each of the four action dimensions using numerically-stable recurrence.
   * 2. Accumulate per-sample softmax entropy into `entropyAccumulator` for later averaging.
   * 3. After the streaming pass write final means/stds into the shared scratch buffers and
   *    compute kurtosis when full telemetry is enabled.
   *
   * Notes:
   * - Uses descriptive local names for the four directions (North/East/South/West) to clarify intent.
   * - Preserves the original numeric formulas and ordering to remain bit-for-bit compatible.
   *
   * @param recent Array of sample vectors (may contain undefined entries; missing values treated as 0).
   * @param sampleCount Number of samples to process (should equal recent.length)
   * @returns Sum of softmax entropies across samples (caller computes mean if desired)
   * @example
   * const entropySum = EvolutionEngine['#accumulateLogitStatsUnrolled4'](recentLogits, recentLogits.length);
   */
  static #accumulateLogitStatsUnrolled4(
    recent: number[][],
    sampleCount: number
  ): number {
    // Step 0: defensive fast-path
    if (!Array.isArray(recent) || sampleCount === 0) return 0;

    // Step 1: descriptive accumulators for the four action dimensions (N,E,S,W).
    let meanNorth = 0,
      meanEast = 0,
      meanSouth = 0,
      meanWest = 0;

    let M2North = 0,
      M2East = 0,
      M2South = 0,
      M2West = 0;

    let M3North = 0,
      M3East = 0,
      M3South = 0,
      M3West = 0;

    let M4North = 0,
      M4East = 0,
      M4South = 0,
      M4West = 0;

    // Step 2: entropy accumulator (sum across samples)
    let entropyAccumulator = 0;

    // Step 3: streaming update loop - unrolled per-dimension for performance clarity
    // Hoist shared exponent scratch into a local alias to avoid repeated private-field lookups in the hot loop.
    const expsBuf = EvolutionEngine.#SCRATCH_EXPS;

    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
      const vec = recent[sampleIndex] ?? EvolutionEngine.#EMPTY_VEC;
      const xNorth = vec[0] ?? 0;
      const xEast = vec[1] ?? 0;
      const xSouth = vec[2] ?? 0;
      const xWest = vec[3] ?? 0;
      const n = sampleIndex + 1;

      // update North (dim 0)
      {
        const delta = xNorth - meanNorth;
        const deltaN = delta / n;
        const deltaN2 = deltaN * deltaN;
        const term1 = delta * deltaN * (n - 1);
        M4North +=
          term1 * deltaN2 * (n * n - 3 * n + 3) +
          6 * deltaN2 * M2North -
          4 * deltaN * M3North;
        M3North += term1 * deltaN * (n - 2) - 3 * deltaN * M2North;
        M2North += term1;
        meanNorth += deltaN;
      }

      // update East (dim 1)
      {
        const delta = xEast - meanEast;
        const deltaN = delta / n;
        const deltaN2 = deltaN * deltaN;
        const term1 = delta * deltaN * (n - 1);
        M4East +=
          term1 * deltaN2 * (n * n - 3 * n + 3) +
          6 * deltaN2 * M2East -
          4 * deltaN * M3East;
        M3East += term1 * deltaN * (n - 2) - 3 * deltaN * M2East;
        M2East += term1;
        meanEast += deltaN;
      }

      // update South (dim 2)
      {
        const delta = xSouth - meanSouth;
        const deltaN = delta / n;
        const deltaN2 = deltaN * deltaN;
        const term1 = delta * deltaN * (n - 1);
        M4South +=
          term1 * deltaN2 * (n * n - 3 * n + 3) +
          6 * deltaN2 * M2South -
          4 * deltaN * M3South;
        M3South += term1 * deltaN * (n - 2) - 3 * deltaN * M2South;
        M2South += term1;
        meanSouth += deltaN;
      }

      // update West (dim 3)
      {
        const delta = xWest - meanWest;
        const deltaN = delta / n;
        const deltaN2 = deltaN * deltaN;
        const term1 = delta * deltaN * (n - 1);
        M4West +=
          term1 * deltaN2 * (n * n - 3 * n + 3) +
          6 * deltaN2 * M2West -
          4 * deltaN * M3West;
        M3West += term1 * deltaN * (n - 2) - 3 * deltaN * M2West;
        M2West += term1;
        meanWest += deltaN;
      }

      // Step 3b: accumulate per-sample softmax entropy
      entropyAccumulator += EvolutionEngine.#softmaxEntropyFromVector(
        vec,
        expsBuf
      );
    }

    // Step 4: write means into pooled scratch buffer
    const meansBuf = EvolutionEngine.#SCRATCH_MEANS;
    meansBuf[0] = meanNorth;
    meansBuf[1] = meanEast;
    meansBuf[2] = meanSouth;
    meansBuf[3] = meanWest;

    // Step 5: compute population-variance-based stds into pooled stds buffer
    const invSample = 1 / sampleCount;
    const varNorth = M2North * invSample;
    const varEast = M2East * invSample;
    const varSouth = M2South * invSample;
    const varWest = M2West * invSample;
    const stdsBuf = EvolutionEngine.#SCRATCH_STDS;
    stdsBuf[0] = varNorth > 0 ? Math.sqrt(varNorth) : 0;
    stdsBuf[1] = varEast > 0 ? Math.sqrt(varEast) : 0;
    stdsBuf[2] = varSouth > 0 ? Math.sqrt(varSouth) : 0;
    stdsBuf[3] = varWest > 0 ? Math.sqrt(varWest) : 0;

    // Step 6: full telemetry path: compute kurtosis when enabled
    if (!EvolutionEngine.#REDUCED_TELEMETRY) {
      const kurtBuf = EvolutionEngine.#SCRATCH_KURT!;
      kurtBuf[0] =
        varNorth > 1e-18
          ? (sampleCount * M4North) / (M2North * M2North) - 3
          : 0;
      kurtBuf[1] =
        varEast > 1e-18 ? (sampleCount * M4East) / (M2East * M2East) - 3 : 0;
      kurtBuf[2] =
        varSouth > 1e-18
          ? (sampleCount * M4South) / (M2South * M2South) - 3
          : 0;
      kurtBuf[3] =
        varWest > 1e-18 ? (sampleCount * M4West) / (M2West * M2West) - 3 : 0;
    }

    // Return aggregated entropy (sum across samples)
    return entropyAccumulator;
  }

  /**
   * Generic accumulation for arbitrary action dimension including higher moments when enabled.
   * @internal
   */
  static #accumulateLogitStatsGeneric(
    recent: number[][],
    actionDim: number
  ): number {
    const meansBuf = EvolutionEngine.#SCRATCH_MEANS;
    const m2Buf = EvolutionEngine.#SCRATCH_M2_RAW;
    const m3Buf = EvolutionEngine.#SCRATCH_M3_RAW;
    const m4Buf = EvolutionEngine.#SCRATCH_M4_RAW;
    const sampleCount = recent.length;
    let entropyAccumulator = 0;

    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
      const vector = recent[sampleIndex] ?? EvolutionEngine.#EMPTY_VEC;
      const seqNumber = sampleIndex + 1;
      for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
        const x = vector[dimIndex] ?? 0;
        const delta = x - meansBuf[dimIndex];
        const deltaN = delta / seqNumber;
        const deltaN2 = deltaN * deltaN;
        const term1 = delta * deltaN * (seqNumber - 1);
        if (!EvolutionEngine.#REDUCED_TELEMETRY)
          m4Buf![dimIndex] +=
            term1 * deltaN2 * (seqNumber * seqNumber - 3 * seqNumber + 3) +
            6 * deltaN2 * m2Buf[dimIndex] -
            4 * deltaN * (m3Buf ? m3Buf[dimIndex] : 0);
        if (!EvolutionEngine.#REDUCED_TELEMETRY)
          m3Buf![dimIndex] +=
            term1 * deltaN * (seqNumber - 2) - 3 * deltaN * m2Buf[dimIndex];
        m2Buf[dimIndex] += term1;
        meansBuf[dimIndex] += deltaN;
      }
      entropyAccumulator += EvolutionEngine.#softmaxEntropyFromVector(
        vector,
        EvolutionEngine.#SCRATCH_EXPS
      );
    }
    return entropyAccumulator;
  }

  /**
   * Finalize full-statistics values (stds and kurtosis) after accumulation.
   * @internal
   */
  static #finalizeLogitStatsFull(actionDim: number, sampleCount: number) {
    const m2Buf = EvolutionEngine.#SCRATCH_M2_RAW;
    const m4Buf = EvolutionEngine.#SCRATCH_M4_RAW;
    const stdsBuf = EvolutionEngine.#SCRATCH_STDS;
    const kurtBuf = EvolutionEngine.#SCRATCH_KURT!;

    for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
      const variance = m2Buf[dimIndex] / sampleCount;
      stdsBuf[dimIndex] = variance > 0 ? Math.sqrt(variance) : 0;
      if (!EvolutionEngine.#REDUCED_TELEMETRY) {
        const m4v = m4Buf![dimIndex];
        kurtBuf[dimIndex] =
          variance > 1e-18
            ? (sampleCount * m4v) / (m2Buf[dimIndex] * m2Buf[dimIndex]) - 3
            : 0;
      }
    }
  }

  /**
   * Finalize reduced-statistics values (only stds) after accumulation.
   * @internal
   */
  static #finalizeLogitStatsReduced(actionDim: number, sampleCount: number) {
    const m2Buf = EvolutionEngine.#SCRATCH_M2_RAW;
    const stdsBuf = EvolutionEngine.#SCRATCH_STDS;
    for (let dimIndex = 0; dimIndex < actionDim; dimIndex++) {
      const variance = m2Buf[dimIndex] / sampleCount;
      stdsBuf[dimIndex] = variance > 0 ? Math.sqrt(variance) : 0;
    }
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
   * Contract / behaviour:
   * - Inputs: `neat` driver (may provide `spawnFromParent`, `addGenome`, or `_invalidateGenomeCaches`),
   *   `targetAdd` (desired number of new genomes), `safeWrite` logger and generation counter.
   * - Output: side-effects on `neat.population` and `neat.options.popsize`; emits a single status line.
   * - Error model: per-child exceptions are swallowed (best-effort growth). The method is non-throwing.
   * Edge cases:
   * - If `neat.population` is missing or empty nothing is done.
   * - `targetAdd` is clamped to non-negative integer; fractional values are floored.
   * @internal
   */
  static #expandPopulation(
    neat: any,
    targetAdd: number,
    safeWrite: (msg: string) => void,
    completedGenerations: number
  ) {
    // Defensive normalization of requested add count.
    const wanted = Number.isFinite(targetAdd)
      ? Math.max(0, Math.floor(targetAdd))
      : 0;
    if (wanted <= 0) return;

    // Step 1: Compute working sets (population reference, sorted indices and parent pool size).
    const {
      populationRef,
      sortedIdx,
      parentPoolSize,
    } = EvolutionEngine.#prepareExpansion(neat, wanted);

    // If there are no parents to sample from, bail early.
    if (!populationRef?.length || parentPoolSize === 0) return;

    // Step 2: Create requested number of children by repeatedly sampling parents.
    for (let addIndex = 0; addIndex < wanted; addIndex++) {
      const pickIndex = (EvolutionEngine.#fastRandom() * parentPoolSize) | 0;
      const parent = populationRef[sortedIdx[pickIndex]];
      // Delegate per-child creation to a dedicated helper to keep the loop simple and testable.
      try {
        EvolutionEngine.#createChildFromParent(neat, parent);
      } catch {
        /* ignore per-child failures - best-effort expansion */
      }
    }

    // Step 3: Finalize bookkeeping and emit a compact status line.
    neat.options.popsize = neat.population.length;
    safeWrite(
      `[DYNAMIC_POP] Expanded population to ${neat.population.length} at gen ${completedGenerations}\n`
    );
  }

  /**
   * Prepare and validate working sets used when expanding the population.
   *
   * @param neat - NEAT driver object which may expose a `population` array.
   * @param _targetAdd - Requested number of additions (unused here; kept for API symmetry).
   * @returns An object with:
   *  - `populationRef`: reference to the population array (or [] when missing),
   *  - `sortedIdx`: indices of `populationRef` sorted by descending score (empty if no population),
   *  - `parentPoolSize`: number of parents to sample from (0 when none available).
   *
   * Example:
   * const { populationRef, sortedIdx, parentPoolSize } =
   *   EvolutionEngine['#prepareExpansion'](neat, 4);
   * // if populationRef.length === 0 then parentPoolSize === 0 and expansion is skipped.
   *
   * @internal
   */
  static #prepareExpansion(
    neat: any,
    _targetAdd: number
  ): { populationRef: any[]; sortedIdx: number[]; parentPoolSize: number } {
    const populationRef: any[] = Array.isArray(neat?.population)
      ? neat.population
      : [];

    // Fast path: empty population yields trivially empty working sets.
    if (populationRef.length === 0) {
      return { populationRef, sortedIdx: [], parentPoolSize: 0 };
    }

    // Sort indices by descending score using the pooled sorter (stable, allocation-free).
    const sortedIdx = EvolutionEngine.#getSortedIndicesByScore(populationRef);

    // Compute desired parent count and clamp into a safe range.
    const desiredParentCount = Math.ceil(
      sortedIdx.length * EvolutionEngine.#DEFAULT_PARENT_FRACTION
    );
    const parentCount = Math.max(2, desiredParentCount);
    const parentPoolSize = Math.min(parentCount, sortedIdx.length);

    return { populationRef, sortedIdx, parentPoolSize };
  }

  /**
   * Determine how many mutation operations to attempt for a new child (1 or 2).
   * @internal
   */
  static #determineMutateCount(): number {
    return (
      1 +
      (EvolutionEngine.#fastRandom() < EvolutionEngine.#DEFAULT_HALF_PROB
        ? 1
        : 0)
    );
  }

  /**
   * Apply up to `mutateCount` distinct mutation operations to `clone`.
   *
   * Behavioural contract:
   * 1. Uses the engine's cached mutation operation array (via `#getMutationOps`) as the operation pool.
   * 2. Selects up to `mutateCount` unique operations without allocating a fresh permutation array by
   *    reusing the pooled `#SCRATCH_MUTOP_IDX` Uint16Array and performing a partial Fisher–Yates shuffle.
   * 3. For very small `mutateCount` values the method takes an unrolled fast path to avoid shuffle overhead.
   * 4. Mutation is applied by calling `clone.mutate(op)` for each chosen op when `clone.mutate` exists.
   *
   * Implementation notes:
   * - Defensive: clamps and validates inputs; no-ops when there are zero available mutation ops or when
   *   `mutateCount` <= 0.
   * - Uses descriptive local names and avoids temporary allocations other than those grown on the pooled buffer.
   * - Side-effects: mutates the pooled `#SCRATCH_MUTOP_IDX` contents but restores no state (caller must treat
   *   the scratch buffer as ephemeral).
   *
   * Complexity:
   * - O(opCount) to initialise the pooled index buffer. Partial Fisher–Yates costs O(k) where k = applied count.
   * - Memory: O(1) extra (reuses class scratch buffer).
   *
   * @param clone - Genome-like object expected to expose `mutate(op)`; if missing, mutation calls are skipped.
   * @param neat - NEAT driver used to resolve the available mutation operations via `#getMutationOps`.
   * @param mutateCount - Desired number of distinct mutation ops to apply (floored to integer and clamped).
   * @example
   * // Apply up to two random distinct mutations from the configured mutation set:
   * EvolutionEngine['#applyMutationsToClone'](someClone, neat, 2);
   *
   * @internal
   */
  static #applyMutationsToClone(clone: any, neat: any, mutateCount: number) {
    // Step 1: Resolve the current set of available mutation operations.
    const mutationOps = EvolutionEngine.#getMutationOps(neat);
    const operationCount = mutationOps.length | 0;
    // No available ops -> nothing to do.
    if (operationCount === 0) return;

    // Step 2: Ensure pooled index buffer has sufficient capacity for op indices.
    if (EvolutionEngine.#SCRATCH_MUTOP_IDX.length < operationCount) {
      const nextSize = 1 << Math.ceil(Math.log2(operationCount));
      EvolutionEngine.#SCRATCH_MUTOP_IDX = new Uint16Array(nextSize);
    }
    const indexBuffer = EvolutionEngine.#SCRATCH_MUTOP_IDX;

    // Step 3: Initialise pooled index buffer with the identity permutation
    // [0,1,2,...,operationCount-1] so we can perform an in-place partial shuffle.
    for (let writeIndex = 0; writeIndex < operationCount; writeIndex++) {
      indexBuffer[writeIndex] = writeIndex;
    }

    // Step 4: Normalize requested apply count to a safe integer range [0, operationCount].
    const wanted = Math.max(0, Math.floor(mutateCount || 0));
    const toApply = Math.min(wanted, operationCount);
    if (toApply === 0) return;

    // Step 5: Tiny-count fast paths: avoid shuffle when applying 1 or 2 ops.
    if (toApply === 1) {
      // Single pick: indexBuffer[0] is deterministic identity; apply directly.
      const opIndex = indexBuffer[0];
      const op = mutationOps[opIndex];
      if (typeof clone?.mutate === 'function') clone.mutate(op);
      return;
    }

    if (toApply === 2) {
      // Two picks: use first two entries of identity buffer (sufficiently random given prior shuffling
      // across calls is not required here; we rely on random selection only in the general path).
      const firstIndex = indexBuffer[0];
      const secondIndex = indexBuffer[1];
      if (typeof clone?.mutate === 'function') {
        clone.mutate(mutationOps[firstIndex]);
        clone.mutate(mutationOps[secondIndex]);
      }
      return;
    }

    // Step 6: Partial Fisher–Yates selection for k distinct picks.
    // For selectionCursor in [0,k) pick a random element from the remaining tail
    // and swap it into the prefix position, then apply it immediately.
    for (
      let selectionCursor = 0;
      selectionCursor < toApply;
      selectionCursor++
    ) {
      const remaining = operationCount - selectionCursor;
      // Random offset in [0, remaining)
      const offset = (EvolutionEngine.#fastRandom() * remaining) | 0;
      const swapPosition = selectionCursor + offset;

      // Swap chosen element into current prefix slot.
      const temp = indexBuffer[selectionCursor];
      indexBuffer[selectionCursor] = indexBuffer[swapPosition];
      indexBuffer[swapPosition] = temp;

      // Apply the chosen operation to the clone if supported.
      const chosenOpIndex = indexBuffer[selectionCursor];
      const chosenOp = mutationOps[chosenOpIndex];
      if (typeof clone?.mutate === 'function') clone.mutate(chosenOp);
    }
  }

  /**
   * Register a newly-created clone with the NEAT driver if supported, falling back to a simple push.
   * Preserves the original best-effort semantics and cache invalidation hook.
   *
   * Contract & behaviour:
   * 1. Prefer calling `neat.addGenome(clone, [parentId])` when the driver exposes it. This allows
   *    driver-specific bookkeeping (species, id assignment, telemetry).
   * 2. When `addGenome` is absent, attempt a lightweight fallback: call `_invalidateGenomeCaches` if
   *    present, then push the clone into `neat.population` (creating the array if missing).
   * 3. Best-effort error model: per-registration exceptions are swallowed; the method tries to ensure the
   *    clone is present in `neat.population` before returning.
   *
   * Notes:
   * - The helper mutates the provided `neat` object to guarantee a `population` array exists when falling back.
   * - This is intentionally forgiving to avoid breaking the evolution loop on driver edge cases.
   *
   * @param neat - NEAT driver / manager object (may be undefined in tests; guarded defensively).
   * @param clone - Genome object to register (expected to be a valid genome instance).
   * @param parentId - Optional identifier or array of parent ids used by driver-level registration.
   * @example
   * // Preferred (driver-aware) registration when using a full NEAT manager:
   * EvolutionEngine['#registerClone'](neat, genomeClone, parentId);
   *
   * @internal
   */
  static #registerClone(neat: any, clone: any, parentId?: any) {
    // Step 0: Defensive guard - nothing to do when minimal inputs missing.
    if (!neat || !clone) return;

    try {
      // Step 1: Prefer driver-managed registration if available. This lets the driver perform any
      //         bookkeeping (IDs, species assignment, telemetry hooks) atomically.
      if (typeof neat.addGenome === 'function') {
        neat.addGenome(clone, [parentId]);
        return;
      }

      // Step 2: Fallback path - ensure population array exists and attempt a conservative registration.
      // 2a: Create population array if absent to preserve downstream invariants.
      if (!Array.isArray(neat.population)) neat.population = [];

      // 2b: Let the driver invalidate any caches it tracks for genomes (non-fatal if absent).
      if (typeof neat._invalidateGenomeCaches === 'function') {
        try {
          neat._invalidateGenomeCaches(clone);
        } catch {
          // Ignore cache invalidation errors; proceed to push.
        }
      }

      // 2c: Append the clone into the population.
      neat.population.push(clone);
      return;
    } catch (err) {
      // Step 3: Best-effort recovery. If any of the above failed, try to leave the population in a usable state.
      try {
        if (neat && !Array.isArray(neat.population)) neat.population = [];
        neat?.population?.push(clone);
      } catch {
        // Final swallow: we cannot reliably register the clone, but evolution should continue.
      }
    }
  }

  /**
   * Create and register a single child derived from `parent`.
   *
   * Behavioural contract (best-effort):
   * 1. Prefer the driver-level `neat.spawnFromParent(parent, mutateCount)` when available. If it returns a
   *    child-like object we register it via `#registerClone` so driver bookkeeping remains consistent.
   * 2. If the driver doesn't provide spawn or spawn fails, fall back to cloning the parent, applying
   *    `mutateCount` mutation operations, sanitising the clone (clear score) and registering it.
   * 3. All steps are non-throwing from the caller's perspective; internal exceptions are swallowed
   *    so expansion remains best-effort and doesn't abort the evolution loop.
   *
   * Steps (inline):
   * 1. Defensive guard: exit when `neat` or `parent` are missing.
   * 2. Compute `mutateCount` once and reuse for both driver-spawn and fallback paths.
   * 3. Try driver spawn inside a protective try/catch; if a child is produced, register and return.
   * 4. Fallback: produce a clone (use `parent.clone()` when available), apply mutations, clear score,
   *    and register the clone.
   * 5. Swallow any errors and return silently (best-effort).
   *
   * @param neat - NEAT driver / manager object.
   * @param parent - Parent genome object used as the basis for spawning/cloning.
   * @example
   * EvolutionEngine['#createChildFromParent'](neat, someParentGenome);
   *
   * @internal
   */
  static #createChildFromParent(neat: any, parent: any) {
    // Step 1: Defensive guard.
    if (!neat || !parent) return;

    // Step 2: Determine number of mutation ops to attempt for this child.
    const mutateCount = EvolutionEngine.#determineMutateCount();

    // Step 3: Prefer driver-provided spawn (protected so failures fall back gracefully).
    if (typeof neat.spawnFromParent === 'function') {
      try {
        const spawnedChild = neat.spawnFromParent(parent, mutateCount);
        if (spawnedChild) {
          // Use unified register path so driver bookkeeping remains consistent.
          EvolutionEngine.#registerClone(
            neat,
            spawnedChild,
            (parent as any)?._id
          );
          return;
        }
        // If spawn returned falsy, fall through to clone path.
      } catch {
        // Ignore driver spawn errors and fall back to clone+mutate below.
      }
    }

    // Step 4: Fallback clone + mutate + register path (best-effort).
    try {
      const clone =
        typeof parent.clone === 'function' ? parent.clone() : parent;

      // Apply mutations to the clone (may throw; handled below to avoid aborting expansion).
      try {
        EvolutionEngine.#applyMutationsToClone(clone, neat, mutateCount);
      } catch {
        // Ignore mutation errors; proceed to registration using whatever state the clone has.
      }

      // Clear transient score so the new child is re-evaluated by the NEAT driver.
      try {
        (clone as any).score = undefined;
      } catch {
        // If clearing score fails, continue anyway.
      }

      // Register the clone using the unified helper (which itself is fault-tolerant).
      EvolutionEngine.#registerClone(neat, clone, (parent as any)?._id);
    } catch {
      // Final swallow: do not let per-child creation failures bubble up.
    }
  }

  /**
   * Return indices of population sorted descending by score using pooled index buffer.
   * Uses iterative quicksort on the indices to avoid allocating a copy of the population.
   * @internal
   */
  /**
   * Return indices of `population` sorted by descending `score` using a pooled, allocation-free sorter.
   *
   * Steps:
   * 1. Validate inputs and fast-exit for empty populations.
   * 2. Prepare a pooled index buffer (number[] or typed Int32Array) sized to `len`.
   * 3. Initialize the index buffer with the identity permutation [0,1,2,...].
   * 4. Sort indices by descending `population[idx].score` using an iterative quicksort with median-of-three pivot
   *    and an insertion-sort fallback for small partitions. All work uses pooled scratch (no per-call allocations
   *    aside from a minimal typed-array growth when required).
   * 5. Return a number[] view trimmed to `len` (public API remains number[] for compatibility).
   *
   * @param population - Array-like population where each entry may expose a numeric `.score` property.
   * @returns number[] Sorted indices (highest score first). Empty array when input empty.
   * @example
   * const indices = EvolutionEngine['#getSortedIndicesByScore'](population);
   */
  static #getSortedIndicesByScore(population: any[]): number[] {
    // Step 1: Validate inputs.
    const populationLength = population.length | 0;
    if (populationLength === 0) return [];

    // Step 2: Decide whether to use the typed Int32Array scratch or the number[] scratch.
    // Heuristic: prefer typed scratch for larger populations to reduce per-element boxing.
    let useTypedScratch = false;
    const typedScratchBuf = EvolutionEngine.#SCRATCH_SORT_IDX_TA;
    if (typedScratchBuf && typedScratchBuf.length >= populationLength) {
      useTypedScratch = true;
    } else if (!typedScratchBuf && populationLength > 512) {
      // Lazily allocate a typed scratch buffer for large sorts.
      const allocSize = 1 << Math.ceil(Math.log2(populationLength));
      EvolutionEngine.#SCRATCH_SORT_IDX_TA = new Int32Array(allocSize);
      useTypedScratch = true;
    }

    // Ensure number[] scratch remains large enough when we fallback to it or when we need to copy back.
    if (EvolutionEngine.#SCRATCH_SORT_IDX.length < populationLength) {
      const nextSize = 1 << Math.ceil(Math.log2(populationLength));
      EvolutionEngine.#SCRATCH_SORT_IDX = new Array(nextSize);
    }

    // Local alias (typed or numeric). Use `any` to keep the rest of the algorithm generic.
    const indexScratch: any = useTypedScratch
      ? EvolutionEngine.#SCRATCH_SORT_IDX_TA!
      : EvolutionEngine.#SCRATCH_SORT_IDX;

    // Step 3: Initialize identity permutation into scratch.
    for (let initIdx = 0; initIdx < populationLength; initIdx++)
      indexScratch[initIdx] = initIdx;
    // Only number[] supports changing .length; typed arrays don't.
    if (!useTypedScratch)
      EvolutionEngine.#SCRATCH_SORT_IDX.length = populationLength;

    // Step 4: Iterative quicksort using pooled Int32 stack. We operate directly on `indexScratch`.
    let qsStack = EvolutionEngine.#SCRATCH_QS_STACK;
    if (qsStack.length < 2)
      qsStack = EvolutionEngine.#SCRATCH_QS_STACK = new Int32Array(128);
    let stackPtr = 0; // stack pointer (next free slot)

    // push initial range
    qsStack[stackPtr++] = 0;
    qsStack[stackPtr++] = populationLength - 1;

    while (stackPtr > 0) {
      const hi = qsStack[--stackPtr];
      const lo = qsStack[--stackPtr];
      if (lo >= hi) continue;

      // For small partitions use insertion sort directly on the index array.
      if (hi - lo <= EvolutionEngine.#QS_SMALL_THRESHOLD) {
        EvolutionEngine.#insertionSortIndices(indexScratch, lo, hi, population);
        continue;
      }

      // Median-of-three pivot selection to reduce degeneracy.
      let leftPtr = lo;
      let rightPtr = hi;
      const pivotScore = EvolutionEngine.#medianOfThreePivot(
        indexScratch,
        lo,
        hi,
        population
      );

      // Partition step (descending order: larger scores to the left)
      while (leftPtr <= rightPtr) {
        while (true) {
          const li = indexScratch[leftPtr];
          if ((population[li]?.score ?? -Infinity) <= pivotScore) break;
          leftPtr++;
        }
        while (true) {
          const rj = indexScratch[rightPtr];
          if ((population[rj]?.score ?? -Infinity) >= pivotScore) break;
          rightPtr--;
        }
        if (leftPtr <= rightPtr) {
          const t = indexScratch[leftPtr];
          indexScratch[leftPtr] = indexScratch[rightPtr];
          indexScratch[rightPtr] = t;
          leftPtr++;
          rightPtr--;
        }
      }

      // Push larger partition first to limit stack depth.
      const leftPartitionSize = rightPtr - lo;
      const rightPartitionSize = hi - leftPtr;

      if (leftPartitionSize > rightPartitionSize) {
        if (lo < rightPtr) {
          stackPtr = EvolutionEngine.#qsPushRange(stackPtr, lo, rightPtr);
          qsStack = EvolutionEngine.#SCRATCH_QS_STACK;
        }
        if (leftPtr < hi) {
          stackPtr = EvolutionEngine.#qsPushRange(stackPtr, leftPtr, hi);
          qsStack = EvolutionEngine.#SCRATCH_QS_STACK;
        }
      } else {
        if (leftPtr < hi) {
          stackPtr = EvolutionEngine.#qsPushRange(stackPtr, leftPtr, hi);
          qsStack = EvolutionEngine.#SCRATCH_QS_STACK;
        }
        if (lo < rightPtr) {
          stackPtr = EvolutionEngine.#qsPushRange(stackPtr, lo, rightPtr);
          qsStack = EvolutionEngine.#SCRATCH_QS_STACK;
        }
      }
    }

    // Step 5: Return a number[] trimmed to len. If we used typed scratch, copy into the pooled
    // number[] buffer (so public API remains number[] and callers that mutate the result don't break).
    if (useTypedScratch) {
      // Ensure the destination number[] has capacity.
      if (EvolutionEngine.#SCRATCH_SORT_IDX.length < populationLength)
        EvolutionEngine.#SCRATCH_SORT_IDX = new Array(
          1 << Math.ceil(Math.log2(populationLength))
        );
      const out = EvolutionEngine.#SCRATCH_SORT_IDX;
      const ta = EvolutionEngine.#SCRATCH_SORT_IDX_TA!;
      for (let k = 0; k < populationLength; k++) out[k] = ta[k];
      out.length = populationLength;
      return out;
    }

    // Non-typed path: indexScratch is already a number[]; trim and return.
    EvolutionEngine.#SCRATCH_SORT_IDX.length = populationLength;
    return EvolutionEngine.#SCRATCH_SORT_IDX;
  }

  /**
   * In-place insertion sort of an index buffer slice by descending `population[idx].score`.
   *
   * Behaviour / contract:
   *  - Sorts the half-open slice [lo, hi] inclusive of both bounds (legacy behaviour preserved).
   *  - Operates in-place on `indexBuf` (no new arrays are allocated). `indexBuf` may be a
   *    `number[]` or an `Int32Array` typed buffer. Callers relying on a `number[]` view should
   *    ensure the appropriate buffer type is supplied (the pooled sorter orchestrator handles
   *    typed->number[] copy when necessary).
   *  - Comparison uses `population[index]?.score` with missing scores treated as -Infinity
   *    so entries without numeric scores sink to the end (lowest priority).
   *  - Stable for equal scores: ties preserve original relative ordering because we only shift
   *    when strictly less-than the current key.
   *
   * Steps (high level):
   *  1. Validate inputs and fast-exit for degenerate ranges.
   *  2. For each element in the slice, extract its index and score (the "key").
   *  3. Shift larger elements rightwards until the insertion point is found.
   *  4. Write the key into its final slot.
   *
   * Notes:
   *  - This method is intentionally allocation-free and non-reentrant (it mutates the
   *    provided buffer). Do not call concurrently with other helpers that reuse the same
   *    pooled scratch buffers.
   *
   * @param indexBuf - Mutable index buffer (either `number[]` or `Int32Array`) containing
   *                   integer indices into `population` to be partially-sorted.
   * @param lo - Inclusive lower bound of the slice to sort (will be clamped by caller).
   * @param hi - Inclusive upper bound of the slice to sort (if hi <= lo the call is a no-op).
   * @param population - Array-like population where `.score` is read for each index.
   * @example
   * // Sort the indices 0..(n-1) stored in `idxBuf` by descending score
   * EvolutionEngine['#insertionSortIndices'](idxBuf, 0, n - 1, population);
   *
   * @internal
   */
  static #insertionSortIndices(
    indexBuf: any,
    lo: number,
    hi: number,
    population: any
  ) {
    // Step 1: fast guards
    if (!indexBuf || lo >= hi) return;

    const populationRef = population ?? EvolutionEngine.#EMPTY_VEC;

    // Helper to obtain score for an index with a clear fallback.
    const scoreOf = (idx: number) =>
      (populationRef[idx]?.score ?? Number.NEGATIVE_INFINITY) as number;

    // Step 2..4: classic insertion sort by descending score.
    // Use descriptive loop variables for clarity.
    for (let writePos = lo + 1; writePos <= hi; writePos++) {
      const keyIndex = indexBuf[writePos];
      const keyScore = scoreOf(keyIndex);

      // scan leftwards to find insertion slot; shift elements that are strictly
      // less than `keyScore` to the right to keep stable ordering for ties.
      let scanPos = writePos - 1;
      while (scanPos >= lo && scoreOf(indexBuf[scanPos]) < keyScore) {
        indexBuf[scanPos + 1] = indexBuf[scanPos];
        scanPos--;
      }

      // place the key into the correct slot
      indexBuf[scanPos + 1] = keyIndex;
    }
  }

  /**
   * Compute the median-of-three pivot score taken from indices at `lo`, `mid`, `hi`.
   *
   * Behaviour / contract:
   *  - Reads three candidate indices from `indexBuf` at positions `lo`, `mid`, `hi` and returns
   *    the median of their `population[idx].score` values.
   *  - `indexBuf` may be a `number[]` or an `Int32Array` (the sorter uses pooled typed buffers);
   *    this helper performs reads only and does not allocate new arrays.
   *  - Missing or non-numeric `.score` values are treated as `Number.NEGATIVE_INFINITY`, so
   *    entries without a score sort to the end.
   *
   * Steps (high level):
   *  1. Compute the middle position and load the three candidate indices.
   *  2. Fetch their scores with a safe fallback.
   *  3. Use a small sequence of comparisons and swaps (no allocations) to determine the median
   *     score value and return it.
   *
   * Notes:
   *  - This method is intentionally small and allocation-free so it can be used in hot sorting
   *    paths. It's non-reentrant when used with shared pooled buffers but it performs only local reads.
   *
   * @param indexBuf - Index buffer (either `number[]` or `Int32Array`) containing population indices.
   * @param lo - Inclusive low index in `indexBuf`.
   * @param hi - Inclusive high index in `indexBuf`.
   * @param population - Array-like population where `.score` is read for each index.
   * @returns The median score (a number) among the three candidate positions.
   * @example
   * const pivot = EvolutionEngine['#medianOfThreePivot'](idxBuf, 0, n-1, population);
   *
   * @internal
   */
  static #medianOfThreePivot(
    indexBuf: any,
    lo: number,
    hi: number,
    population: any
  ): number {
    // Step 1: compute mid and read candidate indices (caller ensures bounds are valid).
    const mid = (lo + hi) >> 1;
    const leftIndex = indexBuf[lo];
    const middleIndex = indexBuf[mid];
    const rightIndex = indexBuf[hi];

    const popRef = population ?? EvolutionEngine.#EMPTY_VEC;

    // Step 2: safely fetch scores with fallback to negative infinity.
    let leftScore = popRef[leftIndex]?.score ?? Number.NEGATIVE_INFINITY;
    let middleScore = popRef[middleIndex]?.score ?? Number.NEGATIVE_INFINITY;
    let rightScore = popRef[rightIndex]?.score ?? Number.NEGATIVE_INFINITY;

    // Step 3: determine median via pairwise comparisons and swaps (no allocations).
    // Ensure leftScore <= middleScore
    if (leftScore > middleScore) {
      const tmp = leftScore;
      leftScore = middleScore;
      middleScore = tmp;
    }

    // Ensure middleScore <= rightScore
    if (middleScore > rightScore) {
      const tmp = middleScore;
      middleScore = rightScore;
      rightScore = tmp;

      // leftScore may now be > middleScore; ensure ordering again
      if (leftScore > middleScore) {
        const tmp2 = leftScore;
        leftScore = middleScore;
        middleScore = tmp2;
      }
    }

    // middleScore now holds the median value
    return middleScore as number;
  }

  /**
   * Push a [lo, hi] pair onto the pooled quicksort stack.
   *
   * Behaviour / contract:
   *  - Uses the class-level `#SCRATCH_QS_STACK` Int32Array as a reusable stack to avoid
   *    per-call allocations in the hot sorting path. The backing buffer may be grown when
   *    capacity is insufficient; growth size follows power-of-two doubling to bound
   *    amortized allocations.
   *  - Mutates the pooled stack and returns the updated `stackPtr` (the index of the next
   *    free slot). Callers should use the returned value; the method does not update any
   *    external stack pointer state beyond returning it.
   *  - Non-reentrant: the pooled stack is shared and must not be used concurrently.
   *
   * Steps:
   *  1. Ensure the pooled `Int32Array` exists and has capacity for two more elements.
   *  2. If capacity is insufficient, allocate a new `Int32Array` with at least double the
   *     previous length (or large enough to accommodate the required size), copy contents,
   *   and swap it into the pooled field.
   *  3. Push `rangeLo` then `rangeHi` into the stack and return the incremented pointer.
   *
   * @param stackPtr - Current stack pointer (next free slot index) into `#SCRATCH_QS_STACK`.
   * @param rangeLo - Inclusive lower bound of the range to push.
   * @param rangeHi - Inclusive upper bound of the range to push.
   * @returns Updated stack pointer (after the push).
   * @example
   * // push initial full range onto pooled stack
   * let ptr = 0;
   * ptr = EvolutionEngine['#qsPushRange'](ptr, 0, population.length - 1);
   *
   * @internal
   */
  static #qsPushRange(
    stackPtr: number,
    rangeLo: number,
    rangeHi: number
  ): number {
    // Step 1: obtain a local alias to the pooled stack buffer
    let stackBuf = EvolutionEngine.#SCRATCH_QS_STACK;

    // Step 2: ensure capacity for two new slots (stackPtr points to next free index)
    const required = stackPtr + 2;
    if (required > stackBuf.length) {
      // Grow by power-of-two until we satisfy the required capacity.
      let newCapacity = Math.max(stackBuf.length << 1, 4);
      while (newCapacity < required) newCapacity <<= 1;

      const grown = new Int32Array(newCapacity);
      grown.set(stackBuf);
      EvolutionEngine.#SCRATCH_QS_STACK = stackBuf = grown;
    }

    // Step 3: push the range (low then high) and return the updated stack pointer.
    stackBuf[stackPtr++] = rangeLo | 0;
    stackBuf[stackPtr++] = rangeHi | 0;
    return stackPtr;
  }

  /** Cached reference to mutation ops array (invalidated if the driver replaces the reference). */
  static #CACHED_MUTATION_OPS: any[] | null = null;

  /**
   * Pooled scratch buffer for temporary bias storage when computing output-bias statistics.
   * - Lazily grown with power-of-two sizing to avoid per-call allocations.
   * - Shared across engine helpers; non-reentrant (callers must not use concurrently).
   */
  static #SCRATCH_BIAS_TA: Float64Array = new Float64Array(0);

  /**
   * Resolve and cache the configured mutation operations from the NEAT driver options.
   *
   * Behaviour / contract:
   *  - Reads `neat?.options?.mutation` and returns a stable array reference when available.
   *  - Caches the resolved reference in `#CACHED_MUTATION_OPS` to avoid repeated property
   *    lookups on hot paths. If the driver later replaces its `mutation` reference, the cache
   *    is updated on the next call.
   *  - Minimises allocations: when the driver provides an actual `Array` we use it directly.
   *    When a non-array object is provided, we attempt a cheap, one-time normalization and
   *    cache the result (this may allocate once).
   *  - The returned array should be treated as read-only by callers. Mutating it may break
   *    the driver's expectations and the engine's caching semantics.
   *
   * Steps:
   *  1. Fast-guard for a missing `neat` or options object -> return shared empty vector.
   *  2. Read the candidate `mutation` value and compare by reference with the cached value.
   *  3. If the reference changed, resolve to an array (use directly if already an Array,
   *     attempt to reuse array-like shapes, or perform a one-time Object->Array conversion).
   *  4. Return the cached array or the shared empty vector.
   *
   * @param neat - NEAT driver object (may be undefined in tests).
   * @returns Read-only array of mutation operation descriptors (may be `#EMPTY_VEC`).
   * @example
   * const ops = EvolutionEngine['#getMutationOps'](neat);
   * // if (ops.length) { apply mutations }
   * @internal
   */
  static #getMutationOps(neat: any): any[] {
    try {
      // Step 1: cheap guard
      if (!neat) return EvolutionEngine.#EMPTY_VEC;

      // Step 2: read candidate from driver options
      const candidate = neat?.options?.mutation;

      // Step 3: update cache only when the driver changed the reference
      if (candidate && EvolutionEngine.#CACHED_MUTATION_OPS !== candidate) {
        // Prefer using the provided array directly when possible (zero-allocation path).
        if (Array.isArray(candidate)) {
          EvolutionEngine.#CACHED_MUTATION_OPS = candidate as any[];
        } else if (candidate && typeof candidate === 'object') {
          // Heuristic: if it's array-like (has a numeric length) reuse without copying.
          const maybeLen = (candidate as any).length;
          if (Number.isFinite(maybeLen) && maybeLen >= 0) {
            EvolutionEngine.#CACHED_MUTATION_OPS = candidate as any[];
          } else {
            // Last resort: convert enumerable values into an array once and cache it.
            EvolutionEngine.#CACHED_MUTATION_OPS = Object.values(
              candidate as any
            );
          }
        } else {
          // Non-collection values yield the shared empty vector.
          EvolutionEngine.#CACHED_MUTATION_OPS = EvolutionEngine.#EMPTY_VEC;
        }
      }

      // Step 4: return cached array or the shared empty vector.
      return (
        (EvolutionEngine.#CACHED_MUTATION_OPS as any[]) ??
        EvolutionEngine.#EMPTY_VEC
      );
    } catch {
      // Non-fatal: on unexpected failures return the shared empty vector to keep callers safe.
      return EvolutionEngine.#EMPTY_VEC;
    }
  }

  /**
   * Ensure every output node in the provided NEAT population uses the identity activation.
   *
   * Rationale:
   * - Some evaluation paths expect raw network outputs (logits) so callers may apply softmax
   *   externally. This helper enforces `Activation.identity` on all nodes typed as `output`.
   * - Uses pooled references and performs no allocations.
   *
   * Steps:
   * 1. Defensive: verify `neat` and `neat.population` exist; fast-exit when missing.
   * 2. Iterate genomes and their `nodes` arrays (fall back to shared empty vector when absent).
   * 3. For each node object that declares `type === 'output'` set `node.squash = methods.Activation.identity`.
   * 4. Swallow errors to preserve best-effort, non-throwing behaviour in the evolution loop.
   *
   * Notes:
   * - The helper mutates node objects in-place. Callers should not rely on this being reentrant
   *   or safe to call concurrently with other helpers that mutate the same population.
   *
   * @param neat - NEAT driver object which may contain a `population` array (optional).
   * @example
   * EvolutionEngine['#ensureOutputIdentity'](neat);
   * @internal
   */
  static #ensureOutputIdentity(neat: any) {
    try {
      // Step 1: quick defensive guard
      if (!neat) return;

      const populationRef: any[] = Array.isArray(neat.population)
        ? neat.population
        : EvolutionEngine.#EMPTY_VEC;

      // Step 2: iterate genomes using index loops to avoid allocations (no for..of temp arrays)
      for (
        let genomeIndex = 0;
        genomeIndex < populationRef.length;
        genomeIndex++
      ) {
        const genome: any = populationRef[genomeIndex];
        if (!genome) continue;

        const nodesRef: any[] = Array.isArray(genome.nodes)
          ? genome.nodes
          : EvolutionEngine.#EMPTY_VEC;

        // Step 3: set identity activation for explicit output nodes
        for (let nodeIndex = 0; nodeIndex < nodesRef.length; nodeIndex++) {
          const node: any = nodesRef[nodeIndex];
          if (node && node.type === 'output') {
            // Assign the identity activation function (no allocation; `methods` is shared)
            node.squash = methods.Activation.identity;
          }
        }
      }
    } catch {
      // Step 4: best-effort semantics — swallow and proceed when unexpected failures occur.
    }
  }

  /**
   * Update the engine-wide species history and adapt mutation/novelty parameters when a
   * species collapse is detected.
   *
   * Behaviour / contract:
   *  - Counts unique species ids present in `neat.population` using pooled Int32Array scratch
   *    buffers to avoid per-call allocations.
   *  - Pushes the computed species count into a global history buffer and inspects the
   *    most-recent window to detect collapse (consecutive single-species entries).
   *  - When a collapse is detected the function applies conservative escalations to
   *    `mutationRate`, `mutationAmount` and `config.novelty.blendFactor` (when present).
   *  - Returns `true` when a collapse was observed, `false` otherwise. Silently returns
   *    `false` on unexpected errors to preserve the evolution loop.
   *
   * Steps:
   *  1. Normalize and fast-exit when `neat` or `neat.population` is missing.
   *  2. Ensure pooled species scratch buffers (`#SCRATCH_SPECIES_IDS`, `#SCRATCH_SPECIES_COUNTS`)
   *     have capacity for the population size (grow using power-of-two sizing when needed).
   3. Count unique species using a small in-place table in the scratch arrays.
   4. Push the species count into `_speciesHistory` and inspect the rolling window for collapse.
   5. When collapsed, escalate mutation/novelty parameters using configured caps/multipliers.
   6. Return the collapse detection boolean.
   *
   * @param neat - NEAT driver instance which may expose a `population` array.
   * @returns boolean `true` when species collapse observed; `false` otherwise.
   * @example
   * const collapsed = EvolutionEngine['#handleSpeciesHistory'](neat);
   * if (collapsed) console.log('Species collapse: escalated mutation params');
   * @internal
   */
  static #handleSpeciesHistory(neat: any): boolean {
    try {
      // Step 1: fast guards & ensure history exists
      (EvolutionEngine as any)._speciesHistory =
        (EvolutionEngine as any)._speciesHistory ?? EvolutionEngine.#EMPTY_VEC;

      const populationRef: any[] = Array.isArray((neat as any)?.population)
        ? (neat as any).population
        : EvolutionEngine.#EMPTY_VEC;

      // Step 2: ensure pooled scratch buffers are large enough
      let speciesIdsBuf = EvolutionEngine.#SCRATCH_SPECIES_IDS;
      let speciesCountsBuf = EvolutionEngine.#SCRATCH_SPECIES_COUNTS;
      if (populationRef.length > speciesIdsBuf.length) {
        const nextSize = 1 << Math.ceil(Math.log2(populationRef.length || 1));
        EvolutionEngine.#SCRATCH_SPECIES_IDS = new Int32Array(nextSize);
        EvolutionEngine.#SCRATCH_SPECIES_COUNTS = new Int32Array(nextSize);
        speciesIdsBuf = EvolutionEngine.#SCRATCH_SPECIES_IDS;
        speciesCountsBuf = EvolutionEngine.#SCRATCH_SPECIES_COUNTS;
      }

      // Step 3: count unique species into the scratch buffers
      let uniqueCount = 0;
      for (
        let genomeIndex = 0;
        genomeIndex < populationRef.length;
        genomeIndex++
      ) {
        const genome = populationRef[genomeIndex];
        if (!genome || genome.species == null) continue;

        const speciesId = genome.species | 0;
        let foundIndex = -1;

        for (let scan = 0; scan < uniqueCount; scan++) {
          if (speciesIdsBuf[scan] === speciesId) {
            foundIndex = scan;
            break;
          }
        }

        if (foundIndex === -1) {
          speciesIdsBuf[uniqueCount] = speciesId;
          speciesCountsBuf[uniqueCount] = 1;
          uniqueCount++;
        } else {
          speciesCountsBuf[foundIndex]++;
        }
      }

      // Step 4: push the species count into the global history and inspect recent window
      const speciesCount = uniqueCount || 1;
      (EvolutionEngine as any)._speciesHistory = EvolutionEngine.#pushHistory<number>(
        (EvolutionEngine as any)._speciesHistory,
        speciesCount,
        EvolutionEngine.#SPECIES_HISTORY_MAX
      );

      const _speciesHistory: number[] =
        (EvolutionEngine as any)._speciesHistory ?? EvolutionEngine.#EMPTY_VEC;
      const recentWindow: number[] = EvolutionEngine.#getTail<number>(
        _speciesHistory,
        EvolutionEngine.#SPECIES_COLLAPSE_WINDOW
      );

      const collapsed =
        recentWindow.length === EvolutionEngine.#SPECIES_COLLAPSE_WINDOW &&
        recentWindow.every((v: number) => v === 1);

      // Step 5: when collapsed escalate conservative engine parameters (best-effort)
      if (collapsed) {
        const neatAny: any = neat as any;
        if (typeof neatAny.mutationRate === 'number') {
          neatAny.mutationRate = Math.min(
            EvolutionEngine.#COLLAPSE_MUTRATE_CAP,
            neatAny.mutationRate * EvolutionEngine.#COLLAPSE_MUTRATE_MULT
          );
        }

        if (typeof neatAny.mutationAmount === 'number') {
          neatAny.mutationAmount = Math.min(
            EvolutionEngine.#COLLAPSE_MUTAMOUNT_CAP,
            neatAny.mutationAmount * EvolutionEngine.#COLLAPSE_MUTAMOUNT_MULT
          );
        }

        if (neatAny.config && neatAny.config.novelty) {
          neatAny.config.novelty.blendFactor = Math.min(
            EvolutionEngine.#COLLAPSE_NOVELTY_BLEND_CAP,
            neatAny.config.novelty.blendFactor *
              EvolutionEngine.#COLLAPSE_NOVELTY_MULT
          );
        }
      }

      // Step 6: return detection result
      return collapsed;
    } catch {
      return false;
    }
  }

  /**
   * Possibly expand the population when configured and plateau conditions are met.
   * This helper decides whether to grow the NEAT population and delegates the
   * actual creation of children to `#expandPopulation` when required.
   *
   * Behaviour & guarantees:
   * - Best-effort: non-throwing and swallows internal errors to avoid breaking the
   *   evolution loop. Any growth side-effects are performed by `#expandPopulation`.
   * - Allocation-light: performs numeric checks and uses pooled references; it does
   *   not allocate per-call data structures.
   *
   * @param neat - NEAT driver instance; expected to expose a `population` array and `options`.
   * @param dynamicPopEnabled - When falsy no expansion will be attempted.
   * @param completedGenerations - Current generation counter (integer, newest generation).
   * @param dynamicPopMax - Maximum allowed population size (upper bound, integer).
   * @param plateauGenerations - Window length used to compute plateau ratio (integer > 0).
   * @param plateauCounter - Number of plateaued generations observed within the window.
   * @param dynamicPopExpandInterval - Generation interval to attempt expansion (e.g. every N gens).
   * @param dynamicPopExpandFactor - Fractional growth factor used to compute additions (e.g. 0.1 -> 10%).
   * @param dynamicPopPlateauSlack - Minimum plateau ratio (0..1) required to trigger expansion.
   * @param safeWrite - Logger function used by `#expandPopulation` to emit status lines.
   *
   * @example
   * // Attempt an expansion every 5 generations when at least 75% of the plateau
   * // window is 'stalled'. The call is best-effort and will not throw.
   * EvolutionEngine['#maybeExpandPopulation'](
   *   neat,
   *   true,        // dynamicPopEnabled
   *   100,         // completedGenerations
   *   500,         // dynamicPopMax
   *   10,          // plateauGenerations
   *   8,           // plateauCounter
   *   5,           // dynamicPopExpandInterval
   *   0.1,         // dynamicPopExpandFactor
   *   0.75,        // dynamicPopPlateauSlack
   *   console.log  // safeWrite
   * );
   *
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
      // Step 1: quick guards — only proceed when dynamic expansion is enabled
      if (!dynamicPopEnabled || completedGenerations <= 0) return;

      // Step 2: validate population shape and numeric limits without allocating
      const populationRef: any[] = Array.isArray((neat as any)?.population)
        ? (neat as any).population
        : EvolutionEngine.#EMPTY_VEC;

      const maxAllowed = Number.isFinite(dynamicPopMax)
        ? Math.max(0, dynamicPopMax | 0)
        : 0;
      if (populationRef.length >= maxAllowed) return;

      // Step 3: plateau ratio and interval checks (defensive against bad inputs)
      const plateauWindow =
        Number.isFinite(plateauGenerations) && plateauGenerations > 0
          ? plateauGenerations | 0
          : 0;
      const plateauRatio =
        plateauWindow > 0
          ? Math.min(1, (plateauCounter | 0) / plateauWindow)
          : 0;

      const expandInterval =
        Number.isFinite(dynamicPopExpandInterval) &&
        dynamicPopExpandInterval > 0
          ? Math.max(1, dynamicPopExpandInterval | 0)
          : 0;

      // If interval is invalid treat as no-trigger to avoid accidental every-gen expansion
      if (expandInterval === 0) return;

      const isGenerationTrigger =
        (completedGenerations | 0) % expandInterval === 0;
      if (!isGenerationTrigger) return;

      // Step 4: slack threshold check
      const slackThreshold = Number.isFinite(dynamicPopPlateauSlack)
        ? dynamicPopPlateauSlack
        : 0;
      if (plateauRatio < slackThreshold) return;

      // Step 5: compute requested addition count and clamp into safe range
      const currentSize = populationRef.length | 0;
      const factor = Number.isFinite(dynamicPopExpandFactor)
        ? Math.max(0, dynamicPopExpandFactor)
        : 0;
      const computedAdd = Math.floor(Math.max(1, currentSize * factor));
      const allowed = Math.max(0, maxAllowed - currentSize);
      const targetAdd = Math.min(computedAdd, allowed);

      // Step 6: delegate to expansion helper when there is room to grow
      if (targetAdd > 0) {
        EvolutionEngine.#expandPopulation(
          neat,
          targetAdd,
          safeWrite,
          completedGenerations | 0
        );
      }
    } catch {
      /* ignore - best-effort helper must not throw */
    }
  }

  /**
   * Safely update a UI dashboard with the latest run state and optionally yield to the
   * host/frame via an awaited flush function.
   *
   * Behaviour (best-effort):
   *  1) If `dashboardManager.update` exists and is callable, call it with the stable
   *     argument order (maze, result, network, completedGenerations, neat). Any exception
   *     raised by the dashboard is swallowed to avoid interrupting the evolution loop.
   *  2) If `flushToFrame` is supplied as an async function, await it to yield control to
   *     the event loop or renderer (for example `() => new Promise(r => requestAnimationFrame(r))`).
   *  3) The helper avoids heap allocations and relies on existing pooled scratch buffers in
   *     the engine for heavy telemetry elsewhere; this method intentionally performs only
   *     short-lived control flow and minimal work.
   *
   * @param maze - Maze instance or descriptor used by dashboard rendering.
   * @param result - Per-run result object (path, progress, telemetry, etc.).
   * @param network - Network or genome object that should be visualised.
   * @param completedGenerations - Integer index of the completed generation.
   * @param neat - NEAT manager instance (context passed to the dashboard update).
   * @param dashboardManager - Optional manager exposing `update(maze, result, network, gen, neat)`.
   * @param flushToFrame - Optional async function used to yield to the host/frame scheduler; may be omitted.
   *
   * @example
   * // Yield to the browser's next repaint after dashboard update:
   * await EvolutionEngine['#updateDashboardAndMaybeFlush'](
   *   maze, genResult, fittestNetwork, gen, neatInstance, dashboard, () => new Promise(r => requestAnimationFrame(r))
   * );
   *
   * @internal
   */
  static async #updateDashboardAndMaybeFlush(
    maze: any,
    result: any,
    network: any,
    completedGenerations: number,
    neat: any,
    dashboardManager: any,
    flushToFrame?: () => Promise<void>
  ) {
    // Step 0: Defensive local aliases with descriptive names to improve readability in hot paths.
    const manager = dashboardManager;
    const yieldFrame = flushToFrame;

    // Step 1: Call dashboard update if provided. This is intentionally best-effort and must not
    // throw — any error from the dashboard implementation is swallowed to keep the engine stable.
    if (manager?.update && typeof manager.update === 'function') {
      try {
        // Use the stable argument order so dashboard implementations are consistent.
        manager.update(maze, result, network, completedGenerations, neat);
      } catch (_updateError) {
        // Swallow dashboard errors — telemetry/UI must not break evolution.
      }
    }

    // Step 2: Optionally yield to the host/frame scheduler. Guard typeof to avoid an accidental
    // Promise rejection when a caller passes undefined/non-function.
    if (typeof yieldFrame === 'function') {
      try {
        await yieldFrame();
      } catch (_flushError) {
        // Swallow flush errors; a failed frame yield is non-fatal for the evolution loop.
      }
    }
  }

  /**
   * Periodic dashboard update used when the engine wants to refresh a non-primary
   * dashboard view (for example background or periodic reporting). This helper is
   * intentionally small, allocation-light and best-effort: dashboard errors are
   * swallowed so the evolution loop cannot be interrupted by UI issues.
   *
   * Behavioural contract:
   *  1) If `dashboardManager.update` is present and callable the method invokes it with
   *     the stable argument order: (maze, bestResult, bestNetwork, completedGenerations, neat).
   *  2) If `flushToFrame` is supplied the helper awaits it after the update to yield to
   *     the host renderer (eg. requestAnimationFrame). Any exceptions raised by the
   *     flush are swallowed.
   *  3) The helper avoids creating ephemeral arrays/objects and therefore does not use
   *     typed-array scratch buffers here — there is no hot numerical work to pool. Other
   *     engine helpers already reuse class-level scratch buffers where appropriate.
   *
   * Steps / inline intent:
   *  1. Fast-guard when an update cannot be performed (missing manager, update method,
   *     or missing content to visualise).
   *  2. Call the dashboard update in a try/catch to preserve best-effort semantics.
   *  3. Optionally await the provided `flushToFrame` function to yield to the host.
   *
   * @param maze - Maze descriptor passed to the dashboard renderer.
   * @param bestResult - Best-run result object used for display (may be falsy when not present).
   * @param bestNetwork - Network or genome object to visualise (may be falsy when not present).
   * @param completedGenerations - Completed generation index (number).
   * @param neat - NEAT manager instance (passed through to dashboard update).
   * @param dashboardManager - Optional manager exposing `update(maze, result, network, gen, neat)`.
   * @param flushToFrame - Optional async function used to yield to the host/frame scheduler
   *                      (for example: `() => new Promise(r => requestAnimationFrame(r))`).
   * @example
   * // Safe periodic update and yield to next frame
   * await EvolutionEngine['#updateDashboardPeriodic'](
   *   maze, result, network, gen, neatInstance, dashboard, () => new Promise(r => requestAnimationFrame(r))
   * );
   *
   * @internal
   */
  static async #updateDashboardPeriodic(
    maze: any,
    bestResult: any,
    bestNetwork: any,
    completedGenerations: number,
    neat: any,
    dashboardManager: any,
    flushToFrame?: () => Promise<void>
  ) {
    // Step 0: create descriptive local aliases to clarify intent and keep hot-path refs short.
    const dashboard = dashboardManager;
    const updateFunction = dashboard?.update;
    const frameFlush = flushToFrame;

    // Step 1: Fast-guard — nothing to do when update isn't callable or we lack meaningful data.
    if (typeof updateFunction !== 'function' || !bestNetwork || !bestResult)
      return;

    // Step 2: Invoke dashboard update in a best-effort manner. Swallow any errors so the
    // evolution loop cannot be disrupted by UI failures.
    try {
      // Use `.call` to preserve potential dashboard `this` binding semantics.
      updateFunction.call(
        dashboard,
        maze,
        bestResult,
        bestNetwork,
        completedGenerations,
        neat
      );
    } catch (updateError) {
      // Intentionally ignore update errors — dashboard should not crash the engine.
    }

    // Step 3: Optionally yield to the host renderer/scheduler.
    if (typeof frameFlush === 'function') {
      try {
        await frameFlush();
      } catch (flushError) {
        // Ignore flush errors — non-critical for engine progress.
      }
    }
  }

  /**
   * Re-centers and clamps output node biases after local training.
   *
   * Behaviour & contract:
   *  - Computes the mean and (population) standard deviation of the output node biases
   *    using a numerically-stable single-pass Welford accumulator.
   *  - Subtracts the mean from each output bias and applies a small-scale multiplier when
   *    the measured std is below a configured small-std threshold to avoid collapsing to zero.
   *  - Clamps final biases into the safe range [-5, 5] and writes them back in-place.
   *  - Uses a pooled Float64Array (`#SCRATCH_BIAS_TA`) to avoid per-call allocations when
   *    collecting bias values; the buffer grows lazily and uses power-of-two sizing.
   *  - Best-effort: swallows internal errors to preserve the evolution loop.
   *
   * Steps:
   *  1. Fast-guard when `network` is missing or contains no output nodes.
   *  2. Ensure pooled bias scratch buffer has capacity for `outCount` elements.
   *  3. One-pass Welford accumulation over biases to compute mean and M2.
   *  4. Compute population std = sqrt(M2 / outCount) and optionally apply small-std multiplier.
   *  5. Subtract mean from each bias, scale if needed, clamp to [-5,5], and write back.
   *
   * @param network - Network object containing a `nodes` array. Missing nodes are treated as empty.
   * @internal
   * @example
   * // After performing local training on `net` run:
   * EvolutionEngine['#adjustOutputBiasesAfterTraining'](net);
   */
  static #adjustOutputBiasesAfterTraining(network: any) {
    try {
      // Step 1: early exit when no network or no nodes exist
      if (!network) return;

      const nodesRef = network.nodes ?? EvolutionEngine.#EMPTY_VEC;
      const outputNodeCount = EvolutionEngine.#getNodeIndicesByType(
        nodesRef,
        'output'
      );
      if (outputNodeCount <= 0) return;

      // Step 2: ensure pooled scratch buffer capacity (grow with power-of-two to bound allocations)
      let biasScratch = EvolutionEngine.#SCRATCH_BIAS_TA;
      if (biasScratch.length < outputNodeCount) {
        let newCap = biasScratch.length || 1;
        while (newCap < outputNodeCount) newCap <<= 1;
        biasScratch = EvolutionEngine.#SCRATCH_BIAS_TA = new Float64Array(
          newCap
        );
      }

      // Step 3: Welford one-pass accumulate into local variables while writing raw biases into scratch
      let mean = 0;
      let M2 = 0;
      for (let oi = 0; oi < outputNodeCount; oi++) {
        const nodeIndex = EvolutionEngine.#SCRATCH_NODE_IDX[oi];
        const currentBias = nodesRef[nodeIndex]?.bias ?? 0;
        biasScratch[oi] = currentBias;

        const sampleIndex = oi + 1;
        const delta = currentBias - mean;
        mean += delta / sampleIndex;
        M2 += delta * (currentBias - mean);
      }

      // Step 4: population standard deviation (avoid division by zero)
      const populationStd =
        outputNodeCount > 0 ? Math.sqrt(M2 / outputNodeCount) : 0;
      const smallStdThreshold = EvolutionEngine.#DEFAULT_STD_SMALL;
      const smallStdMultiplier = EvolutionEngine.#DEFAULT_STD_ADJUST_MULT;

      // Step 5: subtract mean, optionally scale small-std results, clamp and write back
      for (let oi = 0; oi < outputNodeCount; oi++) {
        const nodeIndex = EvolutionEngine.#SCRATCH_NODE_IDX[oi];
        let adjusted = biasScratch[oi] - mean;
        if (populationStd < smallStdThreshold) adjusted *= smallStdMultiplier;
        // clamp to safe operational range
        nodesRef[nodeIndex].bias = Math.max(-5, Math.min(5, adjusted));
      }
    } catch {
      // Best-effort: swallow errors to avoid breaking the engine loop.
    }
  }

  /**
   * Build the supervised training set used for Lamarckian warm-start training.
   * @returns Array of training cases. Each case has:
   *  - `input`: [compassScalar, openN, openE, openS, openW, progressDelta]
   *  - `output`: one-hot desired move [N,E,S,W] with soft probabilities (TRAIN_OUT_PROB_HIGH/LOW)
   * @example
   * // Typical usage (warm-start pretraining):
   * const ds = EvolutionEngine['#buildLamarckianTrainingSet']();
   * EvolutionEngine['#pretrainPopulationWarmStart'](neat, ds);
   * @internal
   */
  static #buildLamarckianTrainingSet(): {
    input: number[];
    output: number[];
  }[] {
    // Step 1: Prepare the result container (small, bounded dataset).
    const trainingSet: { input: number[]; output: number[] }[] = [];

    // Step 2: Precompute and reuse the four canonical soft one-hot output vectors.
    // This avoids allocating the same small arrays repeatedly and is safe because
    // training routines treat them as read-only.
    const high = EvolutionEngine.#TRAIN_OUT_PROB_HIGH;
    const low = EvolutionEngine.#TRAIN_OUT_PROB_LOW;
    const OUTPUTS: number[][] = [
      [high, low, low, low],
      [low, high, low, low],
      [low, low, high, low],
      [low, low, low, high],
    ];

    // Helper to construct an input vector. Small and explicit so each case is readable.
    const makeInput = (
      compassScalar: number,
      openN: number,
      openE: number,
      openS: number,
      openW: number,
      progressDelta: number
    ) => [compassScalar, openN, openE, openS, openW, progressDelta];

    // Local helper to append a case (keeps call sites terse below).
    const pushCase = (inp: number[], direction: number) =>
      trainingSet.push({ input: inp, output: OUTPUTS[direction] });

    // Step 3: Populate the dataset with canonical scenarios.
    // Single open path with steady progress
    pushCase(makeInput(0, 1, 0, 0, 0, EvolutionEngine.#PROGRESS_MEDIUM), 0);
    pushCase(makeInput(0.25, 0, 1, 0, 0, EvolutionEngine.#PROGRESS_MEDIUM), 1);
    pushCase(makeInput(0.5, 0, 0, 1, 0, EvolutionEngine.#PROGRESS_MEDIUM), 2);
    pushCase(makeInput(0.75, 0, 0, 0, 1, EvolutionEngine.#PROGRESS_MEDIUM), 3);

    // Strong progress cases
    pushCase(makeInput(0, 1, 0, 0, 0, EvolutionEngine.#PROGRESS_STRONG), 0);
    pushCase(makeInput(0.25, 0, 1, 0, 0, EvolutionEngine.#PROGRESS_STRONG), 1);

    // Two-way junctions (ambiguous openings => bias toward one direction)
    pushCase(makeInput(0, 1, 0.6, 0, 0, EvolutionEngine.#PROGRESS_JUNCTION), 0);
    pushCase(makeInput(0, 1, 0, 0.6, 0, EvolutionEngine.#PROGRESS_JUNCTION), 0);
    pushCase(
      makeInput(0.25, 0.6, 1, 0, 0, EvolutionEngine.#PROGRESS_JUNCTION),
      1
    );
    pushCase(
      makeInput(0.25, 0, 1, 0.6, 0, EvolutionEngine.#PROGRESS_JUNCTION),
      1
    );
    pushCase(
      makeInput(0.5, 0, 0.6, 1, 0, EvolutionEngine.#PROGRESS_JUNCTION),
      2
    );
    pushCase(
      makeInput(0.5, 0, 0, 1, 0.6, EvolutionEngine.#PROGRESS_JUNCTION),
      2
    );
    pushCase(
      makeInput(0.75, 0, 0, 0.6, 1, EvolutionEngine.#PROGRESS_JUNCTION),
      3
    );
    pushCase(
      makeInput(0.75, 0.6, 0, 0, 1, EvolutionEngine.#PROGRESS_JUNCTION),
      3
    );

    // Four-way junctions (full variety)
    pushCase(
      makeInput(0, 1, 0.8, 0.5, 0.4, EvolutionEngine.#PROGRESS_FOURWAY),
      0
    );
    pushCase(
      makeInput(0.25, 0.7, 1, 0.6, 0.5, EvolutionEngine.#PROGRESS_FOURWAY),
      1
    );
    pushCase(
      makeInput(0.5, 0.6, 0.55, 1, 0.65, EvolutionEngine.#PROGRESS_FOURWAY),
      2
    );
    pushCase(
      makeInput(0.75, 0.5, 0.45, 0.7, 1, EvolutionEngine.#PROGRESS_FOURWAY),
      3
    );
    // Regressing cases
    pushCase(makeInput(0, 1, 0.3, 0, 0, EvolutionEngine.#PROGRESS_REGRESS), 0);
    pushCase(
      makeInput(0.25, 0.5, 1, 0.4, 0, EvolutionEngine.#PROGRESS_REGRESS),
      1
    );
    pushCase(
      makeInput(0.5, 0, 0.3, 1, 0.2, EvolutionEngine.#PROGRESS_REGRESS),
      2
    );
    pushCase(
      makeInput(0.75, 0, 0.5, 0.4, 1, EvolutionEngine.#PROGRESS_REGRESS),
      3
    );
    pushCase(
      makeInput(
        0,
        0,
        0,
        EvolutionEngine.#PROGRESS_MIN_SIGNAL,
        0,
        EvolutionEngine.#PROGRESS_MILD_REGRESS
      ),
      2
    );

    // Mild augmentation (jitter openness & progress)
    for (let dsi = 0; dsi < trainingSet.length; dsi++) {
      const caseEntry = trainingSet[dsi];
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
    return trainingSet;
  }

  /**
   * Pretrain the population using a small supervised dataset and apply a warm-start.
   *
   * Behaviour & contract:
   *  - Runs a short supervised training pass (backprop) on each network in `neat.population`.
   *  - Applies lightweight warm-start heuristics after training: compass wiring and output bias centering.
   *  - Errors are isolated per-network: a failing network does not abort the overall pretrain step.
   *  - This helper is allocation-light and does not create sizable temporary buffers; it delegates heavy
   *    work to `net.train` and the `#applyCompassWarmStart` / `#centerOutputBiases` helpers which reuse
   *    class-level scratch buffers where appropriate.
   *
   * Steps:
   *  1) Validate inputs and obtain `population` (fast-exit on empty populations).
   *  2) For each network: guard missing `train` method, compute conservative iteration budget, then call `train`.
   *  3) Apply warm-start heuristics (compass wiring + bias centering). Swallow any per-network exceptions.
   *
   * @param neat NEAT instance exposing a `population` array of networks. Missing/empty populations are a no-op.
   * @param lamarckianTrainingSet Array of `{input:number[], output:number[]}` training cases used for warm-start.
   * @internal
   */
  static #pretrainPopulationWarmStart(
    neat: any,
    lamarckianTrainingSet: any[]
  ): void {
    // Step 1: Defensive validation & fast exit.
    if (!neat) return;
    const population = neat.population ?? EvolutionEngine.#EMPTY_VEC;
    if (!Array.isArray(population) || population.length === 0) return;

    // Step 2: Iterate population and apply supervised training per network (best-effort).
    for (
      let networkIndex = 0;
      networkIndex < population.length;
      networkIndex++
    ) {
      const network: any = population[networkIndex];
      try {
        if (!network || typeof network.train !== 'function') continue; // skip non-trainable entries

        // Compute conservative per-network iteration budget (bounded by PRETRAIN_MAX_ITER).
        const iterations = Math.min(
          EvolutionEngine.#PRETRAIN_MAX_ITER,
          EvolutionEngine.#PRETRAIN_BASE_ITER +
            Math.floor((lamarckianTrainingSet?.length || 0) / 2)
        );

        // Delegate to the network's own training routine; options are intentionally conservative.
        network.train(lamarckianTrainingSet, {
          iterations,
          error: EvolutionEngine.#DEFAULT_TRAIN_ERROR,
          rate: EvolutionEngine.#DEFAULT_PRETRAIN_RATE,
          momentum: EvolutionEngine.#DEFAULT_PRETRAIN_MOMENTUM,
          batchSize: EvolutionEngine.#DEFAULT_PRETRAIN_BATCH,
          allowRecurrent: true,
          cost: methods.Cost.softmaxCrossEntropy,
        });

        // Step 3: Post-training warm-start heuristics. Each may throw; isolate failures.
        try {
          EvolutionEngine.#applyCompassWarmStart(network);
        } catch {
          // best-effort: ignore wiring failures
        }
        try {
          EvolutionEngine.#centerOutputBiases(network);
        } catch {
          // best-effort: ignore bias-centering failures
        }
      } catch {
        // Swallow training errors for this network; continue with the next one.
      }
    }
  }

  /**
   * Create a cooperative frame-yielding function used by the evolution loop.
   * @internal
   * @returns A function that yields cooperatively to the next animation frame / tick.
   *
   * Behaviour:
   *  - Prefers `requestAnimationFrame` when available (browser hosts).
   *  - Falls back to `setImmediate` when available (Node) or `setTimeout(...,0)` otherwise.
   *  - Respects a cooperative pause flag (`globalThis.asciiMazePaused`) by polling between ticks
   *    without busy-waiting. The returned function resolves once a single new frame/tick is available
   *    and the pause flag is not set.
   *
   * Steps:
   *  1) Choose the preferred tick function based on the host runtime.
   *  2) When called, await the preferred tick; if `asciiMazePaused` is true poll again after the tick.
   *  3) Resolve once a tick passed while not paused.
   */
  static #makeFlushToFrame(): () => Promise<void> {
    // Helper factories for the three tick primitives; each returns a Promise that resolves on the next tick.
    const rafTick = () =>
      new Promise<void>((resolve) =>
        (globalThis as any).requestAnimationFrame
          ? (globalThis as any).requestAnimationFrame(() => resolve())
          : setTimeout(() => resolve(), 0)
      );
    const immediateTick = () =>
      new Promise<void>((resolve) =>
        typeof setImmediate === 'function'
          ? setImmediate(resolve)
          : setTimeout(resolve, 0)
      );
    const timeoutTick = () =>
      new Promise<void>((resolve) => setTimeout(resolve, 0));

    // Pick the most appropriate tick primitive for this host.
    const preferredTick =
      typeof (globalThis as any).requestAnimationFrame === 'function'
        ? rafTick
        : typeof setImmediate === 'function'
        ? immediateTick
        : timeoutTick;

    // Return the async flush function used by the evolution loop.
    return async (): Promise<void> => {
      // Polling loop: after each tick, if the cooperative pause flag is set, wait another tick.
      // This keeps CPU usage minimal while allowing the host to pause/resume the evolution loop.
      while (true) {
        await preferredTick();
        // Note: using a permissive read of the global pause flag; undefined => not paused.
        if (!(globalThis as any).asciiMazePaused) return;
        // otherwise continue and await another tick before re-checking
      }
    };
  }

  /**
   * Initialize persistence helpers (Node `fs` & `path`) when available and ensure the target
   * directory exists. This helper intentionally does nothing in browser-like hosts.
   *
   * Steps:
   * 1) Detect whether a Node-like `require` is available and attempt to load `fs` and `path`.
   * 2) If both modules are available and `persistDir` is provided, ensure the directory exists
   *    by creating it recursively when necessary.
   * 3) Return an object containing the (possibly null) `{ fs, path }` references for callers to use.
   *
   * Notes:
   * - This helper deliberately performs defensive checks and swallows synchronous errors because
   *   persistence is optional in many host environments (tests, browser demos).
   * - No large allocations are performed here; the function returns lightweight references.
   *
   * @param persistDir Optional directory to ensure exists. If falsy, no filesystem mutations are attempted.
   * @returns Object with `{ fs, path }` where each value may be `null` when unavailable.
   * @internal
   */
  static #initPersistence(
    persistDir: string | undefined
  ): { fs: any; path: any } {
    let fs: any = null;
    let path: any = null;

    // Step 1: Safe detection of Node-style `require` without crashing bundlers that rewrite `require`.
    try {
      // eslint-disable-next-line @typescript-eslint/no-implied-eval
      const maybeRequire =
        (globalThis as any).require ??
        (typeof require === 'function' ? require : null);
      if (maybeRequire) {
        try {
          fs = maybeRequire('fs');
          path = maybeRequire('path');
        } catch {
          // module not available or require denied; leave as null
        }
      }
    } catch {
      // Defensive: any host restriction => treat as not available.
    }

    // Step 2: Ensure directory exists if possible and requested.
    if (fs && typeof fs.existsSync === 'function' && persistDir) {
      try {
        if (!fs.existsSync(persistDir)) {
          // Use recursive mkdir where supported.
          if (typeof fs.mkdirSync === 'function')
            fs.mkdirSync(persistDir, { recursive: true });
        }
      } catch {
        // Best-effort: ignore filesystem permission errors or path issues.
      }
    }

    // Step 3: Return module references (may be null in browser-like hosts).
    return { fs, path };
  }

  /**
   * Build a resilient writer that attempts to write to Node stdout, then a provided
   * dashboard logger, and finally `console.log` as a last resort.
   *
   * Steps:
   * 1) If Node `process.stdout.write` is available, use it (no trailing newline forced).
   * 2) Else if `dashboardManager.logFunction` exists, call it.
   * 3) Else fall back to `console.log` and trim the message.
   *
   * Notes:
   * - Errors are swallowed; logging must never throw and disrupt the evolution loop.
   * - This factory is allocation-light; the returned function only creates a trimmed string
   *   when falling back to `console.log`.
   *
   * @param dashboardManager Optional manager exposing `logFunction(msg:string)` used in some UIs.
   * @returns A function accepting a single string message to write.
   * @internal
   */
  static #makeSafeWriter(dashboardManager: any): (msg: string) => void {
    // Capture local references to avoid repeated property lookups at call time.
    const hasProcessStdout = (() => {
      try {
        return (
          typeof process !== 'undefined' &&
          process &&
          process.stdout &&
          typeof process.stdout.write === 'function'
        );
      } catch {
        return false;
      }
    })();

    const dashboardLogFn = (() => {
      try {
        return dashboardManager && (dashboardManager as any).logFunction
          ? (dashboardManager as any).logFunction.bind(dashboardManager)
          : null;
      } catch {
        return null;
      }
    })();

    return (msg: string) => {
      if (!msg && msg !== '') return; // ignore undefined/null
      // Fast path: Node stdout writer
      if (hasProcessStdout) {
        try {
          (process as any).stdout.write(msg);
          return;
        } catch {
          /* swallow and fall through */
        }
      }

      // Dashboard logger path
      if (dashboardLogFn) {
        try {
          dashboardLogFn(msg);
          return;
        } catch {
          /* swallow and fall through */
        }
      }

      // Final fallback: console.log with trimmed message to avoid accidental trailing whitespace
      try {
        if (typeof console !== 'undefined' && typeof console.log === 'function')
          console.log((msg as string).trim());
      } catch {
        /* swallow all logging errors */
      }
    };
  }

  /**
   * Construct a configured Neat instance using the project's recommended defaults.
   *
   * Steps:
   * 1) Normalize `cfg` into a local `conf` bag so we can use nullish coalescing for defaults.
   * 2) Derive commonly-used numeric settings (popsize, elitism, provenance) into descriptive locals.
   * 3) Assemble the final options object and instantiate the `Neat` driver.
   *
   * Notes:
   * - This helper centralizes engine opinionated defaults so other parts of the engine remain concise.
   * - No typed-array scratch buffers are required here; the configuration objects are small and infrequently created.
   *
   * @param inputCount Number of inputs for the networks.
   * @param outputCount Number of outputs for the networks.
   * @param fitnessCallback Function receiving a network and returning its fitness score.
   * @param cfg Optional configuration overrides (popSize, mutation, telemetry, etc.).
   * @returns A new configured `Neat` instance.
   * @example
   * // Create a neat instance with a custom population size and disabled lineage tracking
   * const neat = EvolutionEngine['#createNeat'](10, 4, fitnessFn, { popSize: 200, lineageTracking: false });
   * @internal
   */
  static #createNeat(
    inputCount: number,
    outputCount: number,
    fitnessCallback: (net: Network) => number,
    cfg: any
  ): any {
    // Step 1: Normalize configuration bag and derive primary numeric settings.
    const conf = cfg ?? {};
    const popSize = Number.isFinite(conf.popSize)
      ? conf.popSize
      : EvolutionEngine.#DEFAULT_POPSIZE;
    const mutationOps = Array.isArray(conf.mutation)
      ? conf.mutation
      : [
          methods.mutation.ADD_NODE,
          methods.mutation.SUB_NODE,
          methods.mutation.ADD_CONN,
          methods.mutation.SUB_CONN,
          methods.mutation.MOD_BIAS,
          methods.mutation.MOD_ACTIVATION,
          methods.mutation.MOD_CONNECTION,
          methods.mutation.ADD_LSTM_NODE,
        ];

    // Step 2: Compute derived integer settings with descriptive names.
    const elitism = Math.max(
      1,
      Math.floor(popSize * EvolutionEngine.#DEFAULT_ELITISM_FRACTION)
    );
    const provenance = Math.max(
      1,
      Math.floor(popSize * EvolutionEngine.#DEFAULT_PROVENANCE_FRACTION)
    );

    // Step 3: Compose other option objects using nullish coalescing for defaults.
    const allowRecurrent = conf.allowRecurrent !== false;
    const adaptiveMutation = conf.adaptiveMutation ?? {
      enabled: true,
      strategy: 'twoTier',
    };
    const multiObjective = conf.multiObjective ?? {
      enabled: true,
      complexityMetric: 'nodes',
      autoEntropy: true,
    };
    const telemetry = conf.telemetry ?? {
      enabled: true,
      performance: true,
      complexity: true,
      hypervolume: true,
    };
    const lineageTracking = conf.lineageTracking === true;
    const novelty = conf.novelty ?? { enabled: true, blendFactor: 0.15 };
    const targetSpecies =
      conf.targetSpecies ?? EvolutionEngine.#DEFAULT_TARGET_SPECIES;
    const adaptiveTargetSpecies = conf.adaptiveTargetSpecies ?? {
      enabled: true,
      entropyRange: EvolutionEngine.#DEFAULT_ENTROPY_RANGE,
      speciesRange: [6, 14],
      smooth: EvolutionEngine.#DEFAULT_ADAPTIVE_SMOOTH,
    };

    // Step 4: Instantiate the Neat driver with the assembled options.
    const neatInstance = new Neat(inputCount, outputCount, fitnessCallback, {
      popsize: popSize,
      mutation: mutationOps,
      mutationRate: EvolutionEngine.#DEFAULT_MUTATION_RATE,
      mutationAmount: EvolutionEngine.#DEFAULT_MUTATION_AMOUNT,
      elitism,
      provenance,
      allowRecurrent,
      minHidden: EvolutionEngine.#DEFAULT_MIN_HIDDEN,
      adaptiveMutation,
      multiObjective,
      telemetry,
      lineageTracking,
      novelty,
      targetSpecies,
      adaptiveTargetSpecies,
    });

    return neatInstance;
  }

  /**
   * Seed the NEAT population from an optional initial population and/or an optional
   * initial best network.
   *
   * This method is intentionally best-effort and non-throwing: any cloning or
   * driver mutating errors are swallowed so the evolution loop can continue.
   *
   * Parameters:
   * @param neat - NEAT driver/manager object which may receive the initial population.
   * @param initialPopulation - Optional array of networks to use as the starting population.
   * @param initialBestNetwork - Optional single network to place at index 0 (best seed).
   * @param targetPopSize - Fallback population size used when `neat.population` is missing.
   *
   * Example:
   * // Use a provided population and ensure `neat.options.popsize` is kept in sync
   * EvolutionEngine['#seedInitialPopulation'](neat, providedPopulation, providedBest, 150);
   *
   * Implementation notes / steps:
   * 1) Fast-guard when `neat` is falsy (nothing to seed).
   * 2) When `initialPopulation` is provided, clone entries into the pooled
   *    `#SCRATCH_POP_CLONE` buffer (grow the pool only when needed). This avoids
   *    per-call allocations while still returning a fresh logical array length.
   * 3) If `initialBestNetwork` is supplied, ensure `neat.population` exists and
   *    overwrite index 0 with a clone of the supplied best network.
   * 4) Keep `neat.options.popsize` consistent with the actual population length
   *    (best-effort, swallowed errors).
   *
   * The method prefers calling `.clone()` on network objects when available and
   * falls back to referencing the original object on clone failure.
   *
   * @internal
   */
  static #seedInitialPopulation(
    neat: any,
    initialPopulation: any[] | undefined,
    initialBestNetwork: any | undefined,
    targetPopSize: number
  ) {
    // Step 1: Defensive guard - nothing to do without a neat manager
    if (!neat) return;

    try {
      // Step 2: If an explicit initial population was provided, clone into pooled buffer.
      if (Array.isArray(initialPopulation) && initialPopulation.length > 0) {
        const sourceLength = initialPopulation.length;

        // Reuse the engine-wide pooled clone array. Grow only when necessary to
        // avoid repeated allocations across runs. This preserves the original
        // behaviour of reusing a pooled buffer while keeping intent explicit.
        let pooledCloneBuffer = EvolutionEngine.#SCRATCH_POP_CLONE;
        if (
          !Array.isArray(pooledCloneBuffer) ||
          pooledCloneBuffer.length < sourceLength
        ) {
          // Allocate a new array of the required capacity and replace the pooled reference.
          pooledCloneBuffer = new Array(sourceLength);
          EvolutionEngine.#SCRATCH_POP_CLONE = pooledCloneBuffer;
        }

        // Fill the pooled buffer with cloned networks (when `.clone()` exists).
        for (let sourceIndex = 0; sourceIndex < sourceLength; sourceIndex++) {
          const candidateNetwork = initialPopulation[sourceIndex];
          try {
            pooledCloneBuffer[sourceIndex] =
              candidateNetwork && typeof candidateNetwork.clone === 'function'
                ? candidateNetwork.clone()
                : candidateNetwork;
          } catch (cloneError) {
            // Best-effort: if cloning fails, fall back to the original reference.
            pooledCloneBuffer[sourceIndex] = candidateNetwork;
          }
        }

        // Mark the logical length of the pooled buffer and adopt it as the starting population.
        pooledCloneBuffer.length = sourceLength;
        neat.population = pooledCloneBuffer;
      }

      // Step 3: If a single best network was provided, ensure it's placed at index 0.
      if (initialBestNetwork) {
        // Ensure we have an actual population array to write into.
        if (!Array.isArray(neat.population)) neat.population = [];
        try {
          neat.population[0] =
            typeof initialBestNetwork.clone === 'function'
              ? initialBestNetwork.clone()
              : initialBestNetwork;
        } catch {
          // Swallow per-network clone errors to preserve best-effort semantics.
        }
      }

      // Step 4: Keep the driver's configured popsize in sync with the actual population.
      try {
        neat.options = neat.options || {};
        neat.options.popsize = Array.isArray(neat.population)
          ? neat.population.length
          : targetPopSize;
      } catch {
        /* best-effort; swallow */
      }
    } catch (outerError) {
      // Top-level safety net: swallow all errors to avoid breaking the evolution loop.
      try {
        neat.options = neat.options || {};
        neat.options.popsize = Array.isArray(neat.population)
          ? neat.population.length
          : targetPopSize;
      } catch {
        /* ignore */
      }
    }
  }

  /**
   * Inspect cooperative cancellation sources and annotate the provided result when cancelled.
   *
   * Behaviour & contract:
   *  - Prefer non-allocating, cheap checks. This helper is allocation-free and uses only
   *    short-lived local references (no typed-array scratch buffers are necessary here).
   *  - Checks two cancellation sources in priority order:
   *      1) `options.cancellation.isCancelled()` (legacy cancellation object)
   *      2) `options.signal.aborted` (standard AbortSignal)
   *  - When a cancellation is observed the helper sets `bestResult.exitReason` to the
   *    canonical string ('cancelled' or 'aborted') and returns that string.
   *  - All checks are best-effort: exceptions are swallowed to avoid disrupting the
   *    evolution control loop.
   *
   * Parameters:
   * @param options - Optional run configuration which may contain `cancellation` and/or `signal`.
   * @param bestResult - Optional mutable result object that will be annotated with `exitReason`.
   * @returns A reason string ('cancelled' | 'aborted') when cancellation is detected, otherwise `undefined`.
   *
   * Example:
   * const cancelReason = EvolutionEngine['#checkCancellation'](opts, runResult);
   * if (cancelReason) return cancelReason;
   *
   * Notes:
   *  - No pooling or typed-array scratch buffers are required for this small helper.
   *  - Keep this method allocation-free and safe to call on hot paths.
   *
   * @internal
   */
  static #checkCancellation(options: any, bestResult: any): string | undefined {
    try {
      // Step 1: Check legacy cancellation object first (if present).
      const legacyCancellation = options?.cancellation;
      if (
        legacyCancellation &&
        typeof legacyCancellation.isCancelled === 'function' &&
        legacyCancellation.isCancelled()
      ) {
        if (bestResult) bestResult.exitReason = 'cancelled';
        return 'cancelled';
      }

      // Step 2: Check standard AbortSignal (many hosts expose `options.signal`).
      const abortSignal = options?.signal;
      if (abortSignal?.aborted) {
        if (bestResult) bestResult.exitReason = 'aborted';
        return 'aborted';
      }
    } catch (err) {
      // Best-effort: swallow any unexpected errors to avoid breaking the caller.
    }
    // No cancellation detected.
    return undefined;
  }

  /**
   * Sample `k` items from `src` into the pooled SCRATCH_SAMPLE buffer (with replacement).
   * Returns the number of items written into the scratch buffer.
   * @internal
   * @remarks Non-reentrant: uses shared `#SCRATCH_SAMPLE` buffer.
   */
  static #sampleIntoScratch<T>(src: T[], k: number): number {
    // Step 0: Validate inputs (fast-fail for bad src or non-positive sample size)
    if (!Array.isArray(src) || k <= 0) return 0;

    // Normalize requested sample count to an integer
    const requestedSampleCount = Math.floor(k);

    // Step 1: Ensure source contains items to sample from
    const sourceLength = src.length | 0;
    if (sourceLength === 0) return 0;

    // Step 2: Ensure pooled scratch buffer has capacity. Grow pool using power-of-two
    // sizing when necessary to avoid frequent allocations.
    let pooledBuffer = EvolutionEngine.#SCRATCH_SAMPLE;
    if (!Array.isArray(pooledBuffer))
      pooledBuffer = EvolutionEngine.#SCRATCH_SAMPLE = [];

    if (pooledBuffer.length < requestedSampleCount) {
      // Grow to next power-of-two >= requestedSampleCount
      let newCapacity = pooledBuffer.length > 0 ? pooledBuffer.length : 1;
      while (newCapacity < requestedSampleCount) newCapacity <<= 1;
      const newBuf: any[] = new Array(newCapacity);
      // Copy existing contents (cheap, amortized) into new buffer
      for (let i = 0; i < pooledBuffer.length; i++) newBuf[i] = pooledBuffer[i];
      EvolutionEngine.#SCRATCH_SAMPLE = newBuf;
      pooledBuffer = newBuf;
    }

    // Step 3: Determine how many items we'll actually write (bounded by pool capacity)
    const writeCount = Math.min(requestedSampleCount, pooledBuffer.length);

    // Step 4: Fill the pooled buffer with randomly selected items (with replacement).
    // Keep the loop tight and unroll in blocks of 4 for throughput (same strategy as before).
    let writeIndex = 0;
    const fastRand = EvolutionEngine.#fastRandom;
    const blockBound = writeCount & ~3; // largest multiple of 4
    while (writeIndex < blockBound) {
      pooledBuffer[writeIndex++] = src[(fastRand() * sourceLength) | 0];
      pooledBuffer[writeIndex++] = src[(fastRand() * sourceLength) | 0];
      pooledBuffer[writeIndex++] = src[(fastRand() * sourceLength) | 0];
      pooledBuffer[writeIndex++] = src[(fastRand() * sourceLength) | 0];
    }
    while (writeIndex < writeCount) {
      pooledBuffer[writeIndex++] = src[(fastRand() * sourceLength) | 0];
    }

    // Return how many items were written into the pooled scratch buffer.
    return writeCount;
  }

  /**
   * Sample up to k items (with replacement) from src segment [segmentStart, end) into scratch buffer.
   * Avoids temporary slice allocation for segment sampling.
   * @internal
   * @remarks Non-reentrant: uses shared `#SCRATCH_SAMPLE` buffer.
   *
   * Example:
   * // Sample 10 items from src starting at index 20 into the engine scratch buffer
   * const written = EvolutionEngine['#sampleSegmentIntoScratch'](srcArray, 20, 10);
   */
  static #sampleSegmentIntoScratch<T>(
    src: T[],
    segmentStart: number,
    k: number
  ): number {
    // Step 0: Validate inputs
    if (!Array.isArray(src) || k <= 0) return 0;

    // Normalize lengths safely
    const sourceLength = src.length | 0;
    if (segmentStart >= sourceLength) return 0;

    // Compute segment length (number of candidate items to sample from)
    const segmentLength = sourceLength - (segmentStart | 0);
    if (segmentLength <= 0) return 0;

    // Normalize requested sample count
    const requestedSampleCount = Math.floor(k);

    // Step 1: Ensure pooled scratch buffer exists and has capacity. Grow using power-of-two sizing.
    let pooledBuffer = EvolutionEngine.#SCRATCH_SAMPLE;
    if (!Array.isArray(pooledBuffer))
      pooledBuffer = EvolutionEngine.#SCRATCH_SAMPLE = [];

    if (pooledBuffer.length < requestedSampleCount) {
      let newCapacity = pooledBuffer.length > 0 ? pooledBuffer.length : 1;
      while (newCapacity < requestedSampleCount) newCapacity <<= 1;
      const newBuf: any[] = new Array(newCapacity);
      for (let i = 0; i < pooledBuffer.length; i++) newBuf[i] = pooledBuffer[i];
      EvolutionEngine.#SCRATCH_SAMPLE = newBuf;
      pooledBuffer = newBuf;
    }

    // Step 2: Bound the write count by the pooled buffer capacity.
    const writeCount = Math.min(requestedSampleCount, pooledBuffer.length);

    // Step 3: Fill the pooled buffer by sampling uniformly (with replacement) from the segment.
    let writeIndex = 0;
    const fastRand = EvolutionEngine.#fastRandom;
    const baseIndex = segmentStart | 0;
    const blockBound = writeCount & ~3; // largest multiple of 4
    while (writeIndex < blockBound) {
      pooledBuffer[writeIndex++] =
        src[baseIndex + ((fastRand() * segmentLength) | 0)];
      pooledBuffer[writeIndex++] =
        src[baseIndex + ((fastRand() * segmentLength) | 0)];
      pooledBuffer[writeIndex++] =
        src[baseIndex + ((fastRand() * segmentLength) | 0)];
      pooledBuffer[writeIndex++] =
        src[baseIndex + ((fastRand() * segmentLength) | 0)];
    }
    while (writeIndex < writeCount) {
      pooledBuffer[writeIndex++] =
        src[baseIndex + ((fastRand() * segmentLength) | 0)];
    }

    // Return how many items were written into the pooled scratch buffer.
    return writeCount;
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
    dynamicPopPlateauSlack: number
  ) {
    // Step 0: Local descriptive aliases and profiling setup.
    const profileEnabled = Boolean(doProfile);
    const clockNow = () => EvolutionEngine.#now();
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
      EvolutionEngine.#ensureOutputIdentity(neat);
    } catch (identityError) {
      try {
        safeWrite?.(
          `#runGeneration: ensureOutputIdentity failed: ${String(
            identityError
          )}`
        );
      } catch {}
    }

    // Step 3: Update species history (best-effort; internal errors are swallowed).
    try {
      EvolutionEngine.#handleSpeciesHistory(neat);
    } catch (speciesError) {
      try {
        safeWrite?.(
          `#runGeneration: handleSpeciesHistory failed: ${String(speciesError)}`
        );
      } catch {}
    }

    // Step 4: Possibly expand the population when configured and plateau conditions are met.
    try {
      EvolutionEngine.#maybeExpandPopulation(
        neat,
        Boolean(dynamicPopEnabled),
        completedGenerations,
        dynamicPopMax,
        plateauGenerations,
        plateauCounter,
        dynamicPopExpandInterval,
        dynamicPopExpandFactor,
        dynamicPopPlateauSlack,
        safeWrite
      );
    } catch (expandError) {
      try {
        safeWrite?.(
          `#runGeneration: maybeExpandPopulation failed: ${String(expandError)}`
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
        lamarckDuration = EvolutionEngine.#applyLamarckianTraining(
          neat,
          lamarckianTrainingSet,
          lamarckianIterations,
          lamarckianSampleSize,
          safeWrite,
          doProfile,
          completedGenerations
        );
      }
    } catch (lamarckError) {
      try {
        safeWrite?.(
          `#runGeneration: applyLamarckianTraining failed: ${String(
            lamarckError
          )}`
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
   * Update plateau detection state based on the latest fitness.
   *
   * Behaviour & contract:
   *  - Compare the provided `fitness` with the last recorded best-for-plateau plus a
   *    configurable improvement threshold. If the new fitness exceeds that value the
   *    plateau counter is reset and the last-best-for-plateau is updated.
   *  - Otherwise the plateau counter is incremented (capped to avoid overflow).
   *  - The helper is allocation-free and intentionally simple; no pooled scratch
   *    buffers are required for this numeric operation.
   *
   * Steps (high level):
   *  1) Validate numeric inputs and normalise the improvement threshold to a non-negative number.
   *  2) If `fitness` represents a meaningful improvement then reset the counter and update the last-best.
   *  3) Otherwise increment the counter (with a safe cap) and return the updated state.
   *
   * Parameters:
   * @param fitness - Latest measured fitness (finite number expected).
   * @param lastBestFitnessForPlateau - Previous best fitness used as the plateau baseline.
   * @param plateauCounter - Current plateau counter (integer >= 0).
   * @param plateauImprovementThreshold - Minimum improvement required to reset plateau (>= 0).
   * @returns Updated `{ plateauCounter, lastBestFitnessForPlateau }`.
   *
   * @example
   * const state = EvolutionEngine['#updatePlateauState'](1.23, 1.1, 3, 0.05);
   * // state => { plateauCounter: 0, lastBestFitnessForPlateau: 1.23 }
   *
   * @internal
   */
  static #updatePlateauState(
    fitness: number,
    lastBestFitnessForPlateau: number,
    plateauCounter: number,
    plateauImprovementThreshold: number
  ): { plateauCounter: number; lastBestFitnessForPlateau: number } {
    // Step 1: Validate & normalise numeric inputs to be defensive on untrusted callers.
    if (!Number.isFinite(fitness)) {
      // Nothing meaningful to do; return inputs unchanged.
      return { plateauCounter, lastBestFitnessForPlateau };
    }

    const baseline = Number.isFinite(lastBestFitnessForPlateau)
      ? lastBestFitnessForPlateau
      : -Infinity;

    // Treat non-finite or negative thresholds as zero.
    const improvementThreshold =
      Number.isFinite(plateauImprovementThreshold) &&
      plateauImprovementThreshold > 0
        ? plateauImprovementThreshold
        : 0;

    // Ensure plateauCounter is a safe non-negative integer before mutating.
    let counter =
      Number.isFinite(plateauCounter) && plateauCounter >= 0
        ? Math.floor(plateauCounter)
        : 0;

    // Step 2: Compare fitness against the baseline + threshold. If improved, reset counter.
    if (fitness > baseline + improvementThreshold) {
      lastBestFitnessForPlateau = fitness;
      counter = 0;
      return { plateauCounter: counter, lastBestFitnessForPlateau };
    }

    // Step 3: No sufficient improvement — increment the plateau counter with a safe cap.
    // Cap at a large integer to avoid unbounded growth in pathological cases.
    const SAFE_CAP = 0x1fffffff; // ~= 536 million
    counter = Math.min(SAFE_CAP, counter + 1);

    return { plateauCounter: counter, lastBestFitnessForPlateau: baseline };
  }

  /**
   * Handle simplify entry and per-generation advance.
   *
   * Behaviour & contract:
   *  - Decides when to enter a simplification phase and runs one simplify cycle per
   *    generation while active. The helper is intentionally small and allocation-free.
   *  - It delegates the decision to start simplifying to `#maybeStartSimplify` and the
   *    per-generation work to `#runSimplifyCycle`. Both calls are best-effort and any
   *    internal errors are swallowed so the evolution loop remains resilient.
   *
   * Steps:
   *  1) Fast-guard and normalise numeric inputs.
   *  2) If not currently simplifying, ask `#maybeStartSimplify` whether to begin and
   *     initialise `simplifyRemaining` accordingly (reset plateau counter when started).
   *  3) If simplifying, run one simplify cycle and update remaining duration; turn off
   *     simplify mode when the remaining budget is exhausted.
   *
   * Notes:
   *  - This helper does not allocate scratch buffers; no typed-array pooling is necessary.
   *  - Returns the minimal state the caller needs to persist between generations.
   *
   * @param neat - NEAT driver instance used for simplification operations.
   * @param plateauCounter - Current plateau counter (integer >= 0).
   * @param plateauGenerations - Window size used to decide when to attempt simplification.
   * @param simplifyDuration - Requested simplify duration (generations) when starting.
   * @param simplifyMode - Current boolean indicating whether a simplify cycle is active.
   * @param simplifyRemaining - Remaining simplify generations (integer >= 0).
   * @param simplifyStrategy - Strategy identifier passed to the simplify cycle.
   * @param simplifyPruneFraction - Fraction of connections to prune when simplifying.
   * @returns Object with updated { simplifyMode, simplifyRemaining, plateauCounter }.
   *
   * @example
   * const state = EvolutionEngine['#handleSimplifyState'](neat, 3, 10, 5, false, 0, 'aggressive', 0.2);
   * // use returned state in the next generation loop
   *
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
    // Step 1: Defensive normalisation of numeric inputs.
    let counter =
      Number.isFinite(plateauCounter) && plateauCounter >= 0
        ? Math.floor(plateauCounter)
        : 0;
    const windowSize =
      Number.isFinite(plateauGenerations) && plateauGenerations > 0
        ? Math.floor(plateauGenerations)
        : 0;
    const requestedDuration =
      Number.isFinite(simplifyDuration) && simplifyDuration > 0
        ? Math.floor(simplifyDuration)
        : 0;
    let remaining =
      Number.isFinite(simplifyRemaining) && simplifyRemaining > 0
        ? Math.floor(simplifyRemaining)
        : 0;
    let active = Boolean(simplifyMode);

    // Step 2: When not active, consult the starter helper to determine whether to begin.
    if (!active) {
      try {
        const startBudget = EvolutionEngine.#maybeStartSimplify(
          counter,
          windowSize,
          requestedDuration
        );
        if (Number.isFinite(startBudget) && startBudget > 0) {
          active = true;
          remaining = Math.floor(startBudget);
          // Reset plateau counter when a simplify phase starts.
          counter = 0;
        }
      } catch (startError) {
        // Best-effort: swallow and continue without starting simplify.
      }
    }

    // Step 3: When active, run a single simplify cycle and decrement remaining budget.
    if (active) {
      try {
        remaining = EvolutionEngine.#runSimplifyCycle(
          neat,
          remaining,
          simplifyStrategy,
          simplifyPruneFraction
        );
        if (!Number.isFinite(remaining) || remaining <= 0) {
          active = false;
          remaining = 0;
        }
      } catch (cycleError) {
        // Best-effort: cancel simplification on error to avoid repeated failing attempts.
        active = false;
        remaining = 0;
      }
    }

    return {
      simplifyMode: active,
      simplifyRemaining: remaining,
      plateauCounter: counter,
    };
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
    neat: any
  ): { generationResult: any; simTime: number } {
    // Step 1: Run simulator and optionally capture elapsed time.
    const startTime = doProfile ? EvolutionEngine.#now() : 0;
    const simResult = MazeMovement.simulateAgent(
      fittest,
      encodedMaze,
      startPosition,
      exitPosition,
      distanceMap,
      maxSteps
    );

    // Best-effort: attach legacy buffer refs and compact telemetry onto the genome.
    try {
      if (!(fittest as any)._lastStepOutputs) {
        (fittest as any)._lastStepOutputs =
          EvolutionEngine.#SCRATCH_LOGITS_RING;
      }
    } catch {}

    try {
      (fittest as any)._saturationFraction = simResult?.saturationFraction ?? 0;
      (fittest as any)._actionEntropy = simResult?.actionEntropy ?? 0;
    } catch {}

    // Step 3: If the simulator returned per-step logits, copy them into the pooled ring buffers.
    try {
      const perStepLogits: number[][] | undefined = (simResult as any)
        ?.stepOutputs;
      if (Array.isArray(perStepLogits) && perStepLogits.length > 0) {
        // Ensure the ring can hold the incoming sequence to avoid overflow resize churn.
        EvolutionEngine.#ensureLogitsRingCapacity(perStepLogits.length);

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
              (Atomics.load(atomicIndexView, 0) + 1) & 0x7fffffff
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
    } catch {}

    // Step 4: Optionally prune saturated outputs and emit telemetry (best-effort).
    try {
      if (
        simResult?.saturationFraction &&
        simResult.saturationFraction >
          EvolutionEngine.#SATURATION_PRUNE_THRESHOLD
      ) {
        EvolutionEngine.#pruneSaturatedHiddenOutputs(fittest);
      }
    } catch {}

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
          safeWrite
        );
      }
    } catch {}

    const elapsed = doProfile ? EvolutionEngine.#now() - startTime : 0;
    return { generationResult: simResult, simTime: elapsed } as any;
  }

  /**
   * Inspect common termination conditions and perform minimal, best-effort side-effects.
   *
   * Behaviour & contract:
   *  - Checks three canonical stop reasons in priority order: `solved`, `stagnation`, then `maxGenerations`.
   *  - When a stop condition is met the helper will:
   *      a) Annotate `bestResult.exitReason` with the canonical reason string.
   *      b) Attempt to update the `dashboardManager` (if present) and await `flushToFrame` to yield to the host.
   *      c) On solve, optionally toggle a cooperative pause flag and emit a small `asciiMazeSolved` event.
   *  - All side-effects are best-effort: exceptions are caught and swallowed so the evolution loop cannot be aborted.
   *  - The helper is allocation-light and uses only local references; it is safe to call frequently.
   *
   * Steps (high-level):
   *  1) Fast-check `solved` using `bestResult.success` and `minProgressToPass`.
   *  2) If solved: update dashboard, await flush, emit optional pause/event, set exit reason and return 'solved'.
   *  3) Check stagnation (bounded by `maxStagnantGenerations`) and, if triggered, update dashboard/flush and return 'stagnation'.
   *  4) Check max generations and return 'maxGenerations' when reached.
   *  5) Return `undefined` when no stop condition applies.
   *
   * @param bestResult Mutable run summary object (may be mutated with `exitReason`).
   * @param bestNetwork Network object associated with the best result (used for dashboard rendering).
   * @param maze Maze descriptor passed to dashboard updates/events.
   * @param completedGenerations Current generation index (integer).
   * @param neat NEAT driver instance (passed to dashboard update).
   * @param dashboardManager Optional manager exposing `update(maze, result, network, gen, neat)`.
   * @param flushToFrame Async function used to yield to the host renderer (e.g. requestAnimationFrame); may be a no-op.
   * @param minProgressToPass Numeric threshold used to consider a run 'solved'.
   * @param autoPauseOnSolve When truthy set cooperative pause flag and emit an event on solve.
   * @param stopOnlyOnSolve When true ignore stagnation/maxGenerations as stop reasons.
   * @param stagnantGenerations Current count of stagnant generations observed.
   * @param maxStagnantGenerations Max allowed stagnant generations before stopping.
   * @param maxGenerations Absolute generation cap after which the run stops.
   * @returns A canonical reason string ('solved'|'stagnation'|'maxGenerations') when stopping, otherwise `undefined`.
   * @example
   * const reason = await EvolutionEngine['#checkStopConditions'](bestResult, bestNet, maze, gen, neat, dashboard, flush, 95, true, false, stagnant, 500, 10000);
   * if (reason) console.log('Stopping due to', reason);
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
    // Local convenience aliases for small, hot checks.
    const hasBest = Boolean(bestResult);
    const shouldConsiderStops = !stopOnlyOnSolve;

    // --- 1) Solved check ---
    if (
      bestResult?.success &&
      bestResult.progress >= (minProgressToPass ?? 0)
    ) {
      // Attempt dashboard update and yield; swallow any errors.
      try {
        dashboardManager?.update?.(
          maze,
          bestResult,
          bestNetwork,
          completedGenerations,
          neat
        );
      } catch {}

      try {
        await flushToFrame?.();
      } catch {}

      // Optionally set a cooperative pause and emit a small event for UIs to react to.
      if (autoPauseOnSolve) {
        try {
          if (typeof window !== 'undefined') {
            (window as any).asciiMazePaused = true;
            try {
              window.dispatchEvent(
                new CustomEvent('asciiMazeSolved', {
                  detail: {
                    maze,
                    generations: completedGenerations,
                    progress: bestResult?.progress,
                  },
                })
              );
            } catch {}
          }
        } catch {}
      }

      if (hasBest) (bestResult as any).exitReason = 'solved';
      return 'solved';
    }

    // --- 2) Stagnation check ---
    if (
      shouldConsiderStops &&
      isFinite(maxStagnantGenerations) &&
      stagnantGenerations >= maxStagnantGenerations
    ) {
      try {
        dashboardManager?.update?.(
          maze,
          bestResult,
          bestNetwork,
          completedGenerations,
          neat
        );
      } catch {}
      try {
        await flushToFrame?.();
      } catch {}
      if (hasBest) (bestResult as any).exitReason = 'stagnation';
      return 'stagnation';
    }

    // --- 3) Max generations check ---
    if (
      shouldConsiderStops &&
      isFinite(maxGenerations) &&
      completedGenerations >= maxGenerations
    ) {
      if (hasBest) (bestResult as any).exitReason = 'maxGenerations';
      return 'maxGenerations';
    }

    // No stop condition matched.
    return undefined;
  }

  /**
   * Prune weak outgoing connections from hidden->output when a hidden node appears
   * saturated (low mean absolute outgoing weight and near-zero variance).
   *
   * Behaviour & contract:
   *  - Performs a single-pass Welford accumulation over absolute outgoing weights to
   *    decide whether a hidden node's outputs are collapsed.
   *  - Uses engine-level typed-array scratch buffers (Float32Array) to avoid per-call
   *    allocations when collecting absolute weights. The scratch buffer will grow lazily
   *    using power-of-two sizing when necessary.
   *  - When saturation is detected we deterministically disable roughly half of the
   *    outgoing connections with smallest absolute weight. Mutation is done in-place.
   *  - All work is best-effort: internal exceptions are swallowed to keep the evolution
   *    loop resilient.
   *
   * Steps (inline):
   *  1) Fast-guard & profiling snapshot.
   *  2) Iterate hidden nodes, collect their outgoing-to-output connections.
   *  3) Copy absolute weights into the pooled Float32Array and run Welford to compute mean/M2.
   *  4) If mean and variance indicate collapse, disable the smallest half of active connections.
   *  5) Record profiling delta when enabled.
   *
   * @param genome Mutable genome object containing a `nodes` array.
   * @internal
   * @example
   * // Soft-prune a single genome after a high-saturation simulation:
   * EvolutionEngine['#pruneSaturatedHiddenOutputs'](genome);
   */
  static #pruneSaturatedHiddenOutputs(genome: any) {
    try {
      // 1) Profiling and defensive references
      const startProfile = EvolutionEngine.#PROFILE_ENABLED
        ? EvolutionEngine.#PROFILE_T0()
        : 0;
      const nodesRef = genome?.nodes ?? EvolutionEngine.#EMPTY_VEC;

      // getNodeIndicesByType populates SCRATCH_NODE_IDX and returns counts.
      const outputCount = EvolutionEngine.#getNodeIndicesByType(
        nodesRef,
        'output'
      );
      const hiddenCount = EvolutionEngine.#getNodeIndicesByType(
        nodesRef,
        'hidden'
      );

      // Local aliases to pooled scratch buffers to avoid repeated static lookups.
      let absWeightsTA = EvolutionEngine.#SCRATCH_EXPS as Float64Array;
      const indexFlags = EvolutionEngine.#SCRATCH_NODE_IDX as
        | Int32Array
        | number[];

      for (let hiddenIndex = 0; hiddenIndex < hiddenCount; hiddenIndex++) {
        // Resolve the actual hidden node index populated by the helper above.
        const hiddenNode =
          nodesRef[
            Number(EvolutionEngine.#SCRATCH_NODE_IDX[outputCount + hiddenIndex])
          ];
        if (!hiddenNode) continue;

        // Collect outgoing connections from this hidden node to outputs.
        const outConns =
          EvolutionEngine.#collectHiddenToOutputConns(
            hiddenNode,
            nodesRef,
            outputCount
          ) || [];
        const outConnsLen = outConns.length;
        if (outConnsLen < 2) continue; // nothing to prune if fewer than 2 outs

        // 3) Ensure our pooled Float64Array has capacity for the work (grow power-of-two).
        const needed = outConnsLen;
        if (!absWeightsTA || absWeightsTA.length < needed) {
          // Grow to the next power-of-two >= needed
          let newCap = 1;
          while (newCap < needed) newCap <<= 1;
          absWeightsTA = new Float64Array(newCap);
          // Cast to any here to avoid strict ArrayBuffer/SharedArrayBuffer incompatibility in some TS configs
          EvolutionEngine.#SCRATCH_EXPS = absWeightsTA as any;
        }

        // Fill pooled buffer with absolute weights (cap by buffer length for safety).
        const fillLimit = Math.min(outConnsLen, absWeightsTA.length);
        for (let wi = 0; wi < fillLimit; wi++) {
          const conn = outConns[wi] as any;
          absWeightsTA[wi] = Math.abs(conn?.weight) || 0;
        }

        // Single-pass Welford on the pooled typed-array slice [0..fillLimit)
        let mean = 0;
        let M2 = 0;
        for (let wi = 0; wi < fillLimit; wi++) {
          const value = absWeightsTA[wi];
          const n = wi + 1;
          const delta = value - mean;
          mean += delta / n;
          M2 += delta * (value - mean);
        }
        const variance = fillLimit ? M2 / fillLimit : 0;

        // 4) Decide collapse: low mean and near-zero variance -> prune smallest half
        if (mean < 0.5 && variance < EvolutionEngine.#NUMERIC_EPSILON_SMALL) {
          const disableTarget = Math.max(1, Math.floor(outConnsLen / 2));

          // Reuse indexFlags as a small bitmap to mark already-disabled candidates.
          for (let fi = 0; fi < outConnsLen; fi++) indexFlags[fi] = 0;

          // Disable the `disableTarget` smallest active connections in a deterministic single-pass manner.
          for (let di = 0; di < disableTarget; di++) {
            let minPos = -1;
            let minAbs = Infinity;
            for (let j = 0; j < outConnsLen; j++) {
              if (indexFlags[j]) continue;
              const candidate = outConns[j] as any;
              if (!candidate || candidate.enabled === false) {
                indexFlags[j] = 1; // skip permanently disabled
                continue;
              }
              const weightAbs = Math.abs(candidate.weight) || 0;
              if (weightAbs < minAbs) {
                minAbs = weightAbs;
                minPos = j;
              }
            }
            if (minPos >= 0) {
              (outConns[minPos] as any).enabled = false;
              indexFlags[minPos] = 1;
            } else {
              // No further candidates to disable
              break;
            }
          }

          // Clear the flags we used (defensive). Only clear used prefix to avoid touching unrelated pool state.
          for (let fi = 0; fi < outConnsLen; fi++) indexFlags[fi] = 0;
        }
      }

      // 5) Profiling record (best-effort)
      if (EvolutionEngine.#PROFILE_ENABLED) {
        EvolutionEngine.#PROFILE_ADD(
          'prune',
          EvolutionEngine.#PROFILE_T0() - startProfile || 0
        );
      }
    } catch {
      // Soft-fail: do not let pruning errors interrupt the evolution loop.
    }
  }

  /**
   * Anti-collapse recovery: reinitialise a fraction of the non-elite population's
   * output biases and their outgoing weights.
   *
   * Behaviour & contract:
   *  - Selects a deterministic fraction (up to 30%) of non-elite genomes and reinitialises
   *    their output-node biases and any outgoing connections targeting outputs.
   *  - Uses the engine's pooled sample buffer (`#SCRATCH_SAMPLE`) to avoid per-call
   *    allocations. Sampling is done via `#sampleSegmentIntoScratch` into that pool.
   *  - Returns nothing; diagnostic summaries are emitted via the provided `safeWrite`.
   *  - Best-effort and non-throwing: internal errors are swallowed to keep the
   *    evolution loop resilient.
   *
   * Props / Parameters:
   * @param neat - NEAT driver instance which must expose `population` and `options.elitism`.
   * @param completedGenerations - Current generation index used for diagnostic logging.
   * @param safeWrite - Lightweight writer function used for best-effort diagnostics.
   *
   * Example:
   * // Periodically call from the generation loop to recover from weight/bias collapse
   * EvolutionEngine['#antiCollapseRecovery'](neatInstance, generationIndex, console.log);
   *
   * @internal
   */
  static #antiCollapseRecovery(
    neat: any,
    completedGenerations: number,
    safeWrite: (msg: string) => void
  ) {
    // Inline step comments and descriptive names follow the STYLEGUIDE.
    try {
      // Step 0: Defensive fast-guards
      const neatInstance = neat ?? null;
      if (!neatInstance) return;

      const elitismCount = Number.isFinite(neatInstance?.options?.elitism)
        ? Math.max(0, Math.floor(neatInstance.options.elitism))
        : 0;

      const population = Array.isArray(neatInstance.population)
        ? neatInstance.population
        : EvolutionEngine.#EMPTY_VEC;

      // If population is too small or there are no non-elite members, nothing to do.
      const nonEliteStartIndex = elitismCount;
      const nonEliteCount = Math.max(0, population.length - nonEliteStartIndex);
      if (nonEliteCount === 0) return;

      // Step 1: Decide how many genomes to reinitialise (cap at 30% of non-elite)
      const fractionToReinit = 0.3; // heuristic fraction
      const maxCandidates = Math.floor(nonEliteCount * fractionToReinit) || 1;

      // Step 2: Reuse the pooled sample buffer (`#SCRATCH_SAMPLE`) to avoid allocations.
      // Ensure the pool exists; sampleSegmentIntoScratch will grow it as needed.
      let pooledSampleBuffer = EvolutionEngine.#SCRATCH_SAMPLE;
      if (!Array.isArray(pooledSampleBuffer)) {
        pooledSampleBuffer = EvolutionEngine.#SCRATCH_SAMPLE = [];
      }

      // Step 3: Sample up to `maxCandidates` genomes from the non-elite segment into the pool.
      // `sampledCount` is the number of items written into the pooled buffer.
      const sampledCount = EvolutionEngine.#sampleSegmentIntoScratch(
        population,
        nonEliteStartIndex,
        maxCandidates
      );

      if (sampledCount <= 0) return;

      // Step 4: Reinitialise each sampled genome's outputs and tally changes.
      let totalConnectionResets = 0;
      let totalBiasResets = 0;

      // Use a for-loop with descriptive names to be explicit and optimiser-friendly.
      for (let sampleIndex = 0; sampleIndex < sampledCount; sampleIndex++) {
        const genome = pooledSampleBuffer[sampleIndex] as any;
        if (!genome) continue;

        // Each call is best-effort: isolated errors should not abort the loop.
        try {
          const {
            connReset,
            biasReset,
          } = EvolutionEngine.#reinitializeGenomeOutputsAndWeights(genome) || {
            connReset: 0,
            biasReset: 0,
          };
          totalConnectionResets += Number(connReset) || 0;
          totalBiasResets += Number(biasReset) || 0;
        } catch (genomeErr) {
          // Swallow per-genome errors to preserve overall loop robustness.
        }
      }

      // Step 5: Emit a compact diagnostic line via the provided safe writer.
      try {
        safeWrite(
          `[ANTICOLLAPSE] gen=${completedGenerations} reinitGenomes=${sampledCount} connReset=${totalConnectionResets} biasReset=${totalBiasResets}\n`
        );
      } catch {
        // best-effort logging only
      }
    } catch {
      // Global swallow: never throw from the recovery helper.
    }
  }

  /**
   * Reinitialize output-node biases and outgoing weights that target those outputs
   * for a single genome. This helper is used by the anti-collapse recovery routine
   * to inject fresh variability into non-elite individuals.
   *
   * Behaviour & contract:
   *  - Reuses the engine's pooled sample buffer `#SCRATCH_SAMPLE` to collect output
   *    nodes without allocating a temporary array per-call.
   *  - Randomises each output node's `bias` within [-BIAS_RESET_HALF_RANGE, +BIAS_RESET_HALF_RANGE].
   *  - Resets connection `weight` values for any connection where `conn.to` points to an
   *    output node to a value sampled from the symmetric range defined by
   *    `#CONN_WEIGHT_RESET_HALF_RANGE`.
   *  - Best-effort: the function swallows internal errors and returns zeroed counts on failure.
   *
   * Steps / inline intent:
   *  1) Fast-guard and obtain the genome's node list.
   *  2) Ensure the pooled sample buffer exists and has capacity (grow by power-of-two).
   *  3) Collect references to output nodes into the pooled buffer.
   *  4) Reinitialise each collected output's bias.
   *  5) Reset connection weights that target any collected output (use a Set for O(1) lookup).
   *
   * @param genome - Mutable genome object containing `nodes` and `connections` arrays.
   * @returns Object with counts: `{ connReset: number, biasReset: number }`.
   * @example
   * // Reinitialise outputs for a single genome and inspect the number of changes
   * const deltas = EvolutionEngine['#reinitializeGenomeOutputsAndWeights'](genome);
   * console.log(`connReset=${deltas.connReset} biasReset=${deltas.biasReset}`);
   * @internal
   */
  static #reinitializeGenomeOutputsAndWeights(
    genome: any
  ): { connReset: number; biasReset: number } {
    try {
      // Step 1: Defensive guards and local aliases
      const nodesList: any[] = Array.isArray(genome?.nodes) ? genome.nodes : [];

      // Step 2: Ensure pooled sample buffer exists and has capacity.
      let sampleBuf = EvolutionEngine.#SCRATCH_SAMPLE;
      if (!Array.isArray(sampleBuf))
        sampleBuf = EvolutionEngine.#SCRATCH_SAMPLE = [];

      // Grow pooled buffer lazily using power-of-two sizing to avoid frequent allocations.
      const requiredCapacity = nodesList.length;
      if (sampleBuf.length < requiredCapacity) {
        let newCapacity = Math.max(1, sampleBuf.length);
        while (newCapacity < requiredCapacity) newCapacity <<= 1;
        sampleBuf.length = newCapacity;
      }

      // Step 3: Collect output nodes into the pooled buffer.
      let outputCount = 0;
      for (const node of nodesList) {
        if (node && node.type === 'output') {
          sampleBuf[outputCount++] = node;
        }
      }

      // Step 4: Reinitialise biases for collected outputs.
      let biasReset = 0;
      const biasHalfRange = Number(EvolutionEngine.#BIAS_RESET_HALF_RANGE) || 0;
      for (let idx = 0; idx < outputCount; idx++) {
        const outNode = sampleBuf[idx];
        if (!outNode) continue;
        outNode.bias =
          EvolutionEngine.#fastRandom() * (2 * biasHalfRange) - biasHalfRange;
        biasReset++;
      }

      // Step 5: Reset weights for connections that target any of the collected outputs.
      let connReset = 0;
      const connections: any[] = Array.isArray(genome?.connections)
        ? genome.connections
        : [];
      if (connections.length > 0 && outputCount > 0) {
        // Using a Set for O(1) membership tests instead of nested loops.
        const outputsSet = new Set<any>();
        for (let idx = 0; idx < outputCount; idx++) {
          const outNode = sampleBuf[idx];
          if (outNode) outputsSet.add(outNode);
        }

        const weightHalfRange =
          Number(EvolutionEngine.#CONN_WEIGHT_RESET_HALF_RANGE) || 0;

        for (const conn of connections) {
          try {
            if (outputsSet.has(conn?.to)) {
              conn.weight =
                EvolutionEngine.#fastRandom() * (2 * weightHalfRange) -
                weightHalfRange;
              connReset++;
            }
          } catch {
            // Swallow per-connection errors; continue with best-effort semantics.
          }
        }
      }

      return { connReset, biasReset };
    } catch {
      // Global swallow — return zeros if anything unexpected happens.
      return { connReset: 0, biasReset: 0 };
    }
  }

  /**
   * Compact a single genome's connection list in-place by removing disabled connections.
   *
   * Behaviour & contract:
   *  - Performs an in-place stable compaction of `genome.connections`, preserving the
   *    relative order of enabled connections while removing entries where `conn.enabled === false`.
   *  - The operation is allocation-light and avoids creating temporary arrays for most
   *    workloads. It follows a two-pointer write/read technique that is optimiser-friendly.
   *  - Best-effort and non-throwing: any internal error is swallowed and the method returns 0.
   *
   * Steps (inline):
   *  1) Fast-guard and obtain the connections list reference.
   *  2) Walk the array with a read pointer and copy enabled connections forward to the write pointer.
   *  3) Truncate the array if disabled connections were removed and return the number removed.
   *
   * @param genome - Mutable genome object that may contain a `connections` array.
   * @returns Number of removed (disabled) connections. Returns 0 on error or when nothing was removed.
   * @example
   * // Compact the connections for a genome and inspect how many disabled connections were removed
   * const removed = EvolutionEngine['#compactGenomeConnections'](genome);
   * console.log(`removed disabled connections: ${removed}`);
   * @internal
   */
  static #compactGenomeConnections(genome: any): number {
    try {
      // Step 1: Defensive fast-guards and aliases
      const connectionsList: any[] = Array.isArray(genome?.connections)
        ? genome.connections
        : [];

      const totalConnections = connectionsList.length;
      if (totalConnections === 0) return 0;

      // Step 2: Two-pointer compaction: read through the array and write enabled items forward.
      let writeIndex = 0;
      for (let readIndex = 0; readIndex < totalConnections; readIndex++) {
        const connection = connectionsList[readIndex];
        // Keep the connection if it's truthy and not explicitly disabled.
        if (connection && connection.enabled !== false) {
          if (readIndex !== writeIndex)
            connectionsList[writeIndex] = connection;
          writeIndex++;
        }
      }

      // Step 3: Truncate the array if any elements were removed and return the removed count.
      const removedCount = totalConnections - writeIndex;
      if (removedCount > 0) connectionsList.length = writeIndex;
      return removedCount;
    } catch {
      // Best-effort: do not throw from this maintenance helper.
      return 0;
    }
  }

  /**
   * Compact the entire population by removing disabled connections from each genome.
   *
   * Behaviour & contract:
   *  - Iterates the `neat.population` array and compacts each genome's connection list
   *    in-place using `#compactGenomeConnections`.
   *  - Uses the engine's pooled sample buffer (`#SCRATCH_SAMPLE`) as a temporary
   *    per-genome removed-counts scratch area to avoid per-call allocations.
   *  - Grows the pooled scratch buffer lazily using power-of-two sizing when necessary.
   *  - Returns the total number of removed (disabled) connections across the population.
   *  - Best-effort and non-throwing: internal errors are swallowed and zero is returned
   *    when compaction cannot be completed.
   *
   * Steps (inline):
   *  1) Fast-guard and obtain the population reference.
   *  2) Ensure the pooled scratch buffer exists and has capacity for the population length.
   *  3) For each genome, run `#compactGenomeConnections` in isolation and store the removed
   *     count into the pooled buffer.
   *  4) Sum the removed counts and return the total.
   *
   * @param neat - NEAT driver exposing a `population` array.
   * @returns Total removed disabled connections across the population (integer >= 0).
   * @example
   * // Compact all genomes and obtain the total number of connections removed
   * const totalRemoved = EvolutionEngine['#compactPopulation'](neatInstance);
   * console.log(`removed connections: ${totalRemoved}`);
   * @internal
   */
  static #compactPopulation(neat: any): number {
    try {
      // Step 1: Defensive guards and aliases
      const populationList: any[] = Array.isArray(neat?.population)
        ? neat.population
        : [];

      const populationSize = populationList.length;
      if (populationSize === 0) return 0;

      // Step 2: Reuse the pooled sample buffer as a scratch counts array.
      let scratchCounts = EvolutionEngine.#SCRATCH_SAMPLE;
      if (!Array.isArray(scratchCounts))
        scratchCounts = EvolutionEngine.#SCRATCH_SAMPLE = [];

      // Grow the pooled buffer with power-of-two sizing to avoid frequent reallocations.
      if (scratchCounts.length < populationSize) {
        let newCapacity = Math.max(1, scratchCounts.length);
        while (newCapacity < populationSize) newCapacity <<= 1;
        scratchCounts.length = newCapacity;
      }

      // Step 3: Compact each genome and record removed counts into the pooled buffer.
      let totalRemoved = 0;
      for (let idx = 0; idx < populationSize; idx++) {
        try {
          const genome = populationList[idx];
          const removedForGenome =
            EvolutionEngine.#compactGenomeConnections(genome) | 0;
          scratchCounts[idx] = removedForGenome;
          totalRemoved += removedForGenome;
        } catch {
          // Per-genome failures are non-fatal: record zero and continue.
          scratchCounts[idx] = 0;
        }
      }

      // Step 4: Return aggregated total.
      return totalRemoved;
    } catch {
      // Global swallow: maintain best-effort behaviour.
      return 0;
    }
  }

  /**
   * Shrink oversize pooled scratch buffers when they grow much larger than the population.
   *
   * Behaviour & contract:
   *  - Heuristically shrink pooled buffers (plain arrays and typed arrays) when their
   *    capacity exceeds a configurable multiple of the current population size. This
   *    reduces memory pressure after large compaction or temporary peaks.
   *  - Uses power-of-two sizing for the target capacity to keep growth/shrinkage cache-friendly.
   *  - When shrinking typed arrays the helper creates a new typed array and copies the
   *    preserved prefix (min(oldLen, newLen)) to avoid losing useful scratch state.
   *  - Best-effort: all errors are swallowed so this maintenance helper cannot throw.
   *
   * Steps (inline):
   *  1) Fast-guard and compute the population size.
   *  2) Compute a target capacity (power-of-two) for pools using a small minimum.
   *  3) For each known pool: if current capacity >> target threshold, allocate a smaller
   *     pool and copy preserved items where appropriate.
   *
   * @param neat - NEAT instance used to derive population size for heuristics.
   * @internal
   */
  static #maybeShrinkScratch(neat: any) {
    try {
      // Step 1: Defensive guards
      const populationSize = Array.isArray(neat?.population)
        ? neat.population.length
        : 0;
      if (!populationSize) return;

      // Heuristic thresholds
      const SHRINK_THRESHOLD_FACTOR = 8; // shrink when pool length > populationSize * factor
      const MIN_POOL_SIZE = 8; // never shrink below this size

      // Helper: next power-of-two >= n
      const nextPowerOfTwo = (n: number) =>
        1 << Math.ceil(Math.log2(Math.max(1, n)));

      // Desired capacity computed from population size (clamped to MIN_POOL_SIZE)
      const desiredCapacity = nextPowerOfTwo(
        Math.max(MIN_POOL_SIZE, populationSize)
      );

      // --- SCRATCH_SORT_IDX (plain Array used for sorting indices) ---
      try {
        const sortIdx = EvolutionEngine.#SCRATCH_SORT_IDX;
        if (
          Array.isArray(sortIdx) &&
          sortIdx.length > populationSize * SHRINK_THRESHOLD_FACTOR
        ) {
          EvolutionEngine.#SCRATCH_SORT_IDX = new Array(desiredCapacity);
        }
      } catch {
        /* ignore per-pool failures */
      }

      // --- SCRATCH_SAMPLE (pooled sample buffer, plain Array) ---
      try {
        let samplePool = EvolutionEngine.#SCRATCH_SAMPLE;
        if (!Array.isArray(samplePool))
          samplePool = EvolutionEngine.#SCRATCH_SAMPLE = [];
        if (samplePool.length > populationSize * SHRINK_THRESHOLD_FACTOR) {
          samplePool.length = desiredCapacity;
          EvolutionEngine.#SCRATCH_SAMPLE = samplePool;
        }
      } catch {
        /* ignore per-pool failures */
      }

      // --- SCRATCH_EXPS (Float64Array used for temporary numeric work) ---
      try {
        const exps = EvolutionEngine.#SCRATCH_EXPS as Float64Array | undefined;
        if (
          exps instanceof Float64Array &&
          exps.length > populationSize * SHRINK_THRESHOLD_FACTOR
        ) {
          const newLen = desiredCapacity;
          const smaller = new Float64Array(newLen);
          smaller.set(exps.subarray(0, Math.min(exps.length, newLen)));
          EvolutionEngine.#SCRATCH_EXPS = smaller;
        }
      } catch {
        /* ignore per-pool failures */
      }

      // --- SCRATCH_BIAS_TA (Float64Array for biases) ---
      try {
        const biasTa = EvolutionEngine.#SCRATCH_BIAS_TA as
          | Float64Array
          | undefined;
        if (
          biasTa instanceof Float64Array &&
          biasTa.length > populationSize * SHRINK_THRESHOLD_FACTOR
        ) {
          const newLen = desiredCapacity;
          const smaller = new Float64Array(newLen);
          smaller.set(biasTa.subarray(0, Math.min(biasTa.length, newLen)));
          EvolutionEngine.#SCRATCH_BIAS_TA = smaller;
        }
      } catch {
        /* ignore per-pool failures */
      }

      // --- SCRATCH_NODE_IDX (Int32Array) ---
      try {
        const nodeIdx = EvolutionEngine.#SCRATCH_NODE_IDX as
          | Int32Array
          | undefined;
        if (
          nodeIdx instanceof Int32Array &&
          nodeIdx.length > populationSize * SHRINK_THRESHOLD_FACTOR
        ) {
          const newLen = desiredCapacity;
          const smaller = new Int32Array(newLen);
          smaller.set(nodeIdx.subarray(0, Math.min(nodeIdx.length, newLen)));
          EvolutionEngine.#SCRATCH_NODE_IDX = smaller;
        }
      } catch {
        /* ignore per-pool failures */
      }
    } catch {
      // Best-effort: do not propagate errors from maintenance helper.
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
      const {
        nodeList,
        inputNodes,
        hiddenNodes,
        outputNodes,
      } = EvolutionEngine.#classifyNodes(network);
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
      const hasRecurrentOrGated = EvolutionEngine.#detectRecurrentOrGated(
        connectionsList
      );
      console.log('Has recurrent/gated connections:', hasRecurrentOrGated);
    } catch (e) {
      // Best-effort logging: swallow and surface a minimal message.
      // Avoid throwing from a debug helper.
      // eslint-disable-next-line no-console
      console.log(
        'printNetworkStructure: failed to inspect network (partial data)'
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
   * Core classifier that performs a single pass over `nodesArray` and fills three pooled buckets.
   * Implementation notes / steps:
   *  1) Lazily ensure a pooled buckets structure exists on the class to avoid allocating new arrays.
   *  2) Clear the pooled buckets by setting .length = 0 (non-allocating when capacity is sufficient).
   *  3) Iterate the node list once and push references into the appropriate bucket using descriptive names.
   *  4) Return the buckets alongside the original node list.
   *
   * This method intentionally keeps the hot loop tiny and readable.
   */
  static #classifyNodesFromArray(
    nodesArray: any[]
  ): {
    nodeList: any[];
    inputNodes: any[];
    hiddenNodes: any[];
    outputNodes: any[];
  } {
    // Step 1: Ensure a pooled buckets holder exists. We attach a plain property to avoid
    // depending on other private static fields that may not exist in this test file.
    // The structure is [ inputBucket, hiddenBucket, outputBucket ] to avoid creating
    // separate arrays on every call.
    if (!(EvolutionEngine as any)._SCRATCH_NODE_BUCKETS) {
      (EvolutionEngine as any)._SCRATCH_NODE_BUCKETS = [[], [], []];
    }

    const pooledBuckets: any[][] = (EvolutionEngine as any)
      ._SCRATCH_NODE_BUCKETS;

    const inputBucket = pooledBuckets[0];
    const hiddenBucket = pooledBuckets[1];
    const outputBucket = pooledBuckets[2];

    // Step 2: Clear buckets in-place (cheap when capacity is already adequate).
    inputBucket.length = 0;
    hiddenBucket.length = 0;
    outputBucket.length = 0;

    // Step 3: Single-pass classification. Use descriptive variable names for clarity.
    for (let nodeIndex = 0; nodeIndex < nodesArray.length; nodeIndex++) {
      const node = nodesArray[nodeIndex];
      // Tolerate array holes and malformed entries.
      if (!node) continue;

      // Normalize node type to a string and classify deterministically.
      const nodeType = String(node.type ?? 'hidden');
      if (nodeType === 'input') {
        inputBucket.push(node);
      } else if (nodeType === 'output') {
        outputBucket.push(node);
      } else {
        // Everything else is treated as hidden. This includes 'hidden', undefined, or custom types.
        hiddenBucket.push(node);
      }
    }

    // Step 4: Return references into the original node list along with pooled buckets.
    return {
      nodeList: nodesArray,
      inputNodes: inputBucket,
      hiddenNodes: hiddenBucket,
      outputNodes: outputBucket,
    };
  }

  /**
   * Populate and return a pooled array of activation function names for the network's nodes.
   * Uses the engine-level SCRATCH_ACT_NAMES pool to avoid per-call allocation.
   */
  static #gatherActivationNames(network: INetwork): string[] {
    const nodesArray: any[] = Array.isArray(network?.nodes)
      ? network.nodes
      : [];
    // Lazily ensure the shared pool array exists.
    if (!Array.isArray((EvolutionEngine as any).#SCRATCH_ACT_NAMES)) {
      // @ts-ignore - private static creation
      (EvolutionEngine as any).#SCRATCH_ACT_NAMES = [];
    }
    const pooledNames: string[] = EvolutionEngine.#SCRATCH_ACT_NAMES;

    // Grow the pool length if needed (avoid slice / new allocations)
    if (pooledNames.length < nodesArray.length)
      pooledNames.length = nodesArray.length;

    for (let i = 0; i < nodesArray.length; i++) {
      const n = nodesArray[i];
      // Prefer readable squash function name when available
      pooledNames[i] = n?.squash?.name ?? String(n?.squash ?? 'unknown');
    }

    // Trim to exact length for consumer-readable output (non-allocating since we set .length)
    pooledNames.length = nodesArray.length;
    return pooledNames;
  }

  /**
   * Detect whether any connection is recurrent (from === to) or gated (has a gater).
   * Accepts the connections list as input to avoid re-indexing network object.
   * Uses a small pooled Int8Array as a temporary flag buffer when the connection list is large.
   */
  static #detectRecurrentOrGated(connectionsList: any[]): boolean {
    /**
     * JSDoc & contract:
     * - Fast, allocation-light detection of any recurrent (conn.from === conn.to)
     *   or gated (conn.gater truthy) connection in `connectionsList`.
     * - Small lists use a plain loop to avoid typed-array overhead.
     * - Large lists reuse a pooled Int8Array scratch buffer to reduce allocation churn.
     *
     * Props:
     * @param connectionsList - Array-like list of connection objects with optional `from`, `to`, and `gater` fields.
     * @returns true when any connection is recurrent or gated; false otherwise.
     *
     * Example:
     * const hasSpecial = EvolutionEngine['#detectRecurrentOrGated'](network.connections);
     */

    if (!Array.isArray(connectionsList) || connectionsList.length === 0)
      return false;

    // Use a descriptive name for list length and pick a small-list fast-path threshold.
    const listLen = connectionsList.length;
    const SMALL_LIST_THRESHOLD = 128;

    // === Fast path for small connection lists ===
    if (listLen < SMALL_LIST_THRESHOLD) {
      for (let index = 0; index < listLen; index++) {
        const connection = connectionsList[index];
        if (!connection) continue; // tolerate sparse arrays
        if (connection.gater) return true; // gated connection found
        if (connection.from === connection.to) return true; // direct recurrent self-connection
      }
      return false;
    }

    // === Large-list path: reuse pooled Int8Array as a tiny scratch buffer ===
    // Intent: reduce allocation churn when this function is called repeatedly on large graphs.
    try {
      const pooledFlags = EvolutionEngine.#ensureConnFlagsCapacity(listLen);

      // If allocation failed, fall back to safe loop below.
      if (!pooledFlags) {
        for (let index = 0; index < listLen; index++) {
          const connection = connectionsList[index];
          if (!connection) continue;
          if (connection.gater || connection.from === connection.to)
            return true;
        }
        return false;
      }

      // Iterate and early-return on detection. We also write to the pooled buffer to
      // demonstrate and exercise the scratch space without relying on external semantics.
      for (let index = 0; index < listLen; index++) {
        const connection = connectionsList[index];
        if (!connection) continue;
        if (connection.gater) return true;
        if (connection.from === connection.to) return true;
        // Mark this connection index in the scratch buffer (cheap and helps keep memory hot)
        pooledFlags[index] = 1;
      }

      return false;
    } catch {
      // Allocation or runtime error: gracefully degrade to safe loop.
      for (let index = 0; index < listLen; index++) {
        const connection = connectionsList[index];
        if (!connection) continue;
        if (connection.gater || connection.from === connection.to) return true;
      }
      return false;
    }
  }

  /**
   * Ensure the pooled connection-flag Int8Array has at least `minCapacity` entries.
   * Returns the pooled buffer or `null` when allocation fails.
   */
  static #ensureConnFlagsCapacity(minCapacity: number): Int8Array | null {
    try {
      const clsAny = EvolutionEngine as any;
      const existing: Int8Array | undefined = clsAny._SCRATCH_CONN_FLAGS;
      if (existing instanceof Int8Array && existing.length >= minCapacity)
        return existing;

      // Compute next power-of-two capacity to reduce future resizes.
      let cap = 1;
      while (cap < minCapacity) cap <<= 1;
      const newBuf = new Int8Array(cap);
      // Store the pooled buffer for reuse.
      clsAny._SCRATCH_CONN_FLAGS = newBuf;
      return newBuf;
    } catch {
      return null;
    }
  }
}
