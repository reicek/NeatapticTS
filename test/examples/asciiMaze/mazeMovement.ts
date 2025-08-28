/**
 * Maze Movement - Handles agent movement and simulation logic (Simplified)
 *
 * This module contains functions for agent movement and simulation in the maze environment,
 * focusing on simple navigation based primarily on neural network decisions.
 *
 * The agent movement system demonstrates:
 * - Decision making based on neural network outputs
 * - Basic reward calculations for reinforcement learning
 * - Simple goal-seeking behavior
 * - Simulation of movement with collision detection
 */
import { INetwork } from './interfaces';
import { MazeUtils } from './mazeUtils';
import { MazeVision } from './mazeVision';

/**
 * Internal aggregate state used during a single agent simulation run.
 *
 * Purpose:
 * - Hold all derived runtime values, counters and diagnostic stats used by the
 *   MazeMovement simulation helpers. This shape is intentionally rich so tests
 *   and visualisers can inspect intermediate state when debugging.
 *
 * Notes:
 * - This interface is internal to the mazeMovement module and is not exported.
 * - Property descriptions are explicit to surface helpful tooltips in editors.
 */
interface SimulationState {
  /** Current mutable agent position as [x, y]. */
  position: [number, number];

  /** Number of simulation steps executed so far (increments each loop). */
  steps: number;

  /** Number of entries in the recorded path (index into pooled PathX/PathY). */
  pathLength: number;

  /** Count of distinct cells visited during this run. */
  visitedUniqueCount: number;

  /** True when a precomputed distance map was supplied to the simulation. */
  hasDistanceMap: boolean;

  /** Optional precomputed distance map (rows × cols) used for fast heuristics. */
  distanceMap?: number[][];

  /** Minimum observed distance-to-exit reached so far (lower is better). */
  minDistanceToExit: number;

  /** Accumulated shaping reward derived from forward progress signals. */
  progressReward: number;

  /** Bonus accumulated when entering previously unvisited cells. */
  newCellExplorationBonus: number;

  /** Accumulated penalty from invalid moves, loops and other negative signals. */
  invalidMovePenalty: number;

  /** Index of the previous action/direction taken (-1 for no-move). */
  prevAction: number;

  /** Steps elapsed since the last observed improvement toward the goal. */
  stepsSinceImprovement: number;

  /** Last global distance-to-exit used for long-term improvement checks. */
  lastDistanceGlobal: number;

  /** Number of steps flagged as 'saturated' (network overconfident/flat outputs). */
  saturatedSteps: number;

  /** Recent positions sliding window used to detect local oscillation/stagnation. */
  recentPositions: [number, number][];

  /** Penalty applied when agent is oscillating in a tight local region. */
  localAreaPenalty: number;

  /** Counters of moves taken per direction index (N,E,S,W). */
  directionCounts: number[];

  /** Ring buffer storing recent visited cell indices for A↔B loop detection. */
  moveHistoryRing: Int32Array;

  /** Current number of populated entries in `moveHistoryRing`. */
  moveHistoryLength: number;
  /** Index pointer (head) into the circular moveHistoryRing. */
  moveHistoryHead: number;
  /** Current linearized cell index for the agent position. */
  currentCellIndex: number;
  /** Penalty accumulated for short A<->B oscillation detection. */
  loopPenalty: number;
  /** Penalty applied for returning to any recent cell (memory-based). */
  memoryPenalty: number;
  /** Dynamic revisit penalty scaled by per-cell visit counts. */
  revisitPenalty: number;
  /** Visit count at the current cell (derived from VisitCounts pool). */
  visitsAtCurrent: number;
  /** Current distance-to-goal measured at agent position. */
  distHere: number;
  /** Per-step perception/vision vector built for the network. */
  vision: number[];
  /** Network action statistics (softmax, entropy, etc.) populated each step. */
  actionStats: any;
  /** Currently selected direction index (0..3) or #-NO_MOVE. */
  direction: number;
  /** Whether the agent moved on the last executed action. */
  moved: boolean;
  /** Distance value measured before executing the current action (previous step). */
  prevDistance: number;

  /** When true the simulation loop should terminate early due to safety triggers. */
  earlyTerminate: boolean;
}

/**
 * MazeMovement provides static methods for agent movement and simulation.
 */
export class MazeMovement {
  /**
   * Maximum number of simulation steps before terminating (safety cap)
   * @internal
   */
  static #DEFAULT_MAX_STEPS = 3000;

  /**
   * Number of recent moves tracked for oscillation detection
   * @internal
   */
  static #MOVE_HISTORY_LENGTH = 6;

  // Named private constants to replace magic numbers and document intent.
  /** Reward scale applied to shaping terms (smaller reduces selection pressure) */
  static #REWARD_SCALE = 0.5;
  /** Strong penalty multiplier for short A->B oscillations */
  static #LOOP_PENALTY = 10; // multiplied by rewardScale
  /** Penalty applied when returning to a recent cell (memory-based) */
  static #MEMORY_RETURN_PENALTY = 2; // multiplied by rewardScale
  /** Per-visit penalty for repeated visits to same cell */
  static #REVISIT_PENALTY_PER_VISIT = 0.2; // per extra visit, multiplied by rewardScale
  /** Visits threshold to trigger termination/harsh penalty */
  static #VISIT_TERMINATION_THRESHOLD = 10;
  /** Extremely harsh penalty for invalid moves (used sparingly) */
  static #INVALID_MOVE_PENALTY_HARSH = 1000;
  /** Mild penalty for invalid moves to preserve learning signal */
  static #INVALID_MOVE_PENALTY_MILD = 10;

  // Saturation / collapse thresholds and penalties
  /** Probability threshold indicating overconfidence (near-deterministic) */
  static #OVERCONFIDENT_PROB = 0.985;
  /** Secondary-probability threshold used with overconfidence detection */
  static #SECOND_PROB_LOW = 0.01;
  /** Threshold for flat-collapse detection using log-std of outputs */
  static #LOGSTD_FLAT_THRESHOLD = 0.01;
  /** Penalty when network appears overconfident */
  static #OVERCONFIDENT_PENALTY = 0.25; // * rewardScale
  /** Penalty for flat collapse (no variance in outputs) */
  static #FLAT_COLLAPSE_PENALTY = 0.35; // * rewardScale
  /** Minimum saturations before applying bias adjustments */
  static #SATURATION_ADJUST_MIN = 6;
  /** Interval (in steps) used for saturation bias adjustment checks */
  static #SATURATION_ADJUST_INTERVAL = 5;
  /** Clamp for adaptive bias adjustments */
  static #BIAS_CLAMP = 5;
  /** Scaling factor used when adjusting biases to mitigate saturation */
  static #BIAS_ADJUST_FACTOR = 0.5;

  // Convenience thresholds and tuning knobs (centralized to avoid magic literals)
  /** Warmup steps where exploration is encouraged */
  static #EPSILON_WARMUP_STEPS = 10;
  /** Steps-stagnant threshold to consider very stagnant (high epsilon) */
  static #EPSILON_STAGNANT_HIGH_THRESHOLD = 12;
  /** Steps-stagnant threshold to consider moderate stagnation */
  static #EPSILON_STAGNANT_MED_THRESHOLD = 6;
  /** Saturation count that triggers epsilon-increase behavior */
  static #EPSILON_SATURATION_TRIGGER = 3;
  /** Length used to detect tiny A->B oscillations */
  static #OSCILLATION_DETECT_LENGTH = 4;
  /** Saturation penalty trigger (>=) */
  static #SATURATION_PENALTY_TRIGGER = 5;
  /** Period (in steps) to escalate saturation penalty */
  static #SATURATION_PENALTY_PERIOD = 10;
  /** Start step for global break bonus when breaking long stagnation */
  static #GLOBAL_BREAK_BONUS_START = 10;
  /** Per-step bonus for global break beyond the start threshold */
  static #GLOBAL_BREAK_BONUS_PER_STEP = 0.01;
  /** Cap for the global break bonus */
  static #GLOBAL_BREAK_BONUS_CAP = 0.5;
  /** Number of steps since improvement to begin repetition penalty scaling */
  static #REPETITION_PENALTY_START = 4;
  /** Weight for entropy bonus on failed runs */
  static #ENTROPY_BONUS_WEIGHT = 4;

  // Vision input layout indices (groups used by hasGuidance checks)
  /** Start index of LOS group within vision vector */
  static #VISION_LOS_START = 8;
  /** Start index of gradient group within vision vector */
  static #VISION_GRAD_START = 12;
  /** Number of elements in each vision group (LOS / Gradient) */
  static #VISION_GROUP_LEN = 4;

  // Proximity/exploration tuning
  /** Distance (in cells) within which greedy proximity moves are prioritized */
  static #PROXIMITY_GREEDY_DISTANCE = 2;
  /** Distance threshold to reduce epsilon exploration near goal */
  static #PROXIMITY_SUPPRESS_EXPLOR_DIST = 5;
  /** Initial epsilon for epsilon-greedy exploration */
  static #EPSILON_INITIAL = 0.35;
  /** Epsilon used when the agent is highly stagnant */
  static #EPSILON_STAGNANT_HIGH = 0.5;
  /** Epsilon used for moderate stagnation */
  static #EPSILON_STAGNANT_MED = 0.25;
  /** Epsilon used when network saturations are detected */
  static #EPSILON_SATURATIONS = 0.3;
  /** Minimum epsilon allowed when near the goal */
  static #EPSILON_MIN_NEAR_GOAL = 0.05;
  /** Streak length used to trigger forced exploration */
  static #NO_MOVE_STREAK_THRESHOLD = 5;

  // Local area stagnation
  /** Size of the recent-positions sliding window for local stagnation detection */
  static #LOCAL_WINDOW = 30;
  /** Max span (in cells) considered "local" for oscillation penalties */
  static #LOCAL_AREA_SPAN_THRESHOLD = 5;
  /** Steps without improvement before local-area stagnation penalty applies */
  static #LOCAL_AREA_STAGNATION_STEPS = 8;
  /** Amount applied to local area penalty when tight oscillation detected (multiplied by rewardScale) */
  static #LOCAL_AREA_PENALTY_AMOUNT = 0.05;

  // Progress reward shaping
  /** Base reward for making forward progress toward the exit */
  static #PROGRESS_REWARD_BASE = 0.3;
  /** Additional progress reward scaled by network confidence */
  static #PROGRESS_REWARD_CONF_SCALE = 0.7;
  /** Multiplier applied per step-since-improvement for extra reward shaping */
  static #PROGRESS_STEPS_MULT = 0.02;
  /** Maximum steps-based progress contribution (times rewardScale) */
  static #PROGRESS_STEPS_MAX = 0.5; // times rewardScale
  /** Scale applied to raw distance-delta when shaping reward */
  static #DISTANCE_DELTA_SCALE = 2.0;
  /** Base confidence factor for distance-delta shaping */
  static #DISTANCE_DELTA_CONF_BASE = 0.4;
  /** Additional confidence scale applied to distance-delta shaping */
  static #DISTANCE_DELTA_CONF_SCALE = 0.6;
  /** Base penalty applied when a move increases distance to goal (multiplied by rewardScale) */
  static #PROGRESS_AWAY_BASE_PENALTY = 0.05;
  /** Additional scaling applied to away penalty proportional to network confidence */
  static #PROGRESS_AWAY_CONF_SCALE = 0.15;

  // Entropy tuning
  /** Entropy value above which the action distribution is considered too uniform */
  static #ENTROPY_HIGH_THRESHOLD = 0.95;
  /** Entropy value below which the distribution is considered confident */
  static #ENTROPY_CONFIDENT_THRESHOLD = 0.55;
  /** Required gap between top two probs to treat as confident */
  static #ENTROPY_CONFIDENT_DIFF = 0.25;
  /** Small penalty applied when entropy is persistently high */
  static #ENTROPY_PENALTY = 0.03; // * rewardScale
  /** Tiny bonus for clear decisions that aid exploration */
  static #EXPLORATION_BONUS_SMALL = 0.015; // * rewardScale
  /** Base repetition/backtrack penalty applied when repeating same action without improvement */
  static #REPETITION_PENALTY_BASE = 0.05;
  /** Penalty for making the direct opposite move (when it doesn't improve) */
  static #BACK_MOVE_PENALTY = 0.2;

  // Saturation penalties
  /** Base penalty applied when saturation is detected */
  static #SATURATION_PENALTY_BASE = 0.05; // * rewardScale
  /** Escalating penalty applied periodically when saturation persists */
  static #SATURATION_PENALTY_ESCALATE = 0.1; // * rewardScale when escalation applies

  // Deep stagnation
  /** Steps without improvement that trigger deep-stagnation handling */
  static #DEEP_STAGNATION_THRESHOLD = 40;
  /** Penalty applied when deep stagnation is detected (non-browser environments) */
  static #DEEP_STAGNATION_PENALTY = 2; // * rewardScale
  // Action/output dimension and softmax/entropy tuning
  /** Number of cardinal actions (N,E,S,W) */
  static #ACTION_DIM = 4;
  /** Natural log of ACTION_DIM; used to normalize entropy calculations */
  static #LOG_ACTIONS = Math.log(MazeMovement.#ACTION_DIM);
  /**
   * Pooled scratch buffers used by `selectDirection` to avoid per-call
   * allocations on the softmax/entropy hot path.
   *
   * @remarks
   * - These are class-private and reused across calls; `selectDirection` is
   *   therefore not reentrant and should not be called concurrently.
   */
  static #SCRATCH_CENTERED = new Float64Array(4);
  static #SCRATCH_EXPS = new Float64Array(4);
  /** Small pooled scratch for temporary integer coordinate coercion. */
  static #COORD_SCRATCH = new Int32Array(2);
  /** Representation for 'no move' direction */
  static #NO_MOVE = -1;
  /** Minimum standard deviation used to prevent division by zero */
  static #STD_MIN = 1e-6;
  /** Thresholds for collapse ratio decisions based on std */
  static #COLLAPSE_STD_THRESHOLD = 0.01;
  /** Secondary threshold used when std indicates medium collapse */
  static #COLLAPSE_STD_MED = 0.03;
  /** Collapse ratio constants used for adaptive temperature */
  /** Full collapse ratio used when std is extremely low */
  static #COLLAPSE_RATIO_FULL = 1;
  /** Partial collapse ratio used for medium collapse */
  static #COLLAPSE_RATIO_HALF = 0.5;
  /** Base and scale used to compute softmax temperature */
  static #TEMPERATURE_BASE = 1;
  /** Scale factor applied when computing adaptive softmax temperature */
  static #TEMPERATURE_SCALE = 1.2;

  // Network history and randomness
  /** History length for recent output snapshots (used for variance diagnostics) */
  static #OUTPUT_HISTORY_LENGTH = 80;
  /**
   * Number of outputs snapshots to keep for variance diagnostics.
   * Larger values smooth variance estimates at the cost of memory.
   */
  /** Small randomness added to fitness to break ties stably */
  static #FITNESS_RANDOMNESS = 0.01;

  // Success fitness constants
  /** Base fitness given for successful maze completion */
  static #SUCCESS_BASE_FITNESS = 650;
  /** Scale applied for remaining steps on success to reward efficiency */
  static #STEP_EFFICIENCY_SCALE = 0.2;
  /** Weight for action-entropy bonus on successful runs */
  static #SUCCESS_ACTION_ENTROPY_SCALE = 5;
  /** Minimum clamp for any successful-run fitness */
  static #MIN_SUCCESS_FITNESS = 150;

  // Exploration / revisiting tuning
  /** Bonus reward for discovering a previously unvisited cell */
  static #NEW_CELL_EXPLORATION_BONUS = 0.3;
  /** Strong penalty factor for revisiting cells */
  static #REVISIT_PENALTY_STRONG = 0.5;

  // Progress shaping constants
  /** Exponent used in non-linear progress shaping */
  static #PROGRESS_POWER = 1.3;
  /** Scale used to convert shaped progress into fitness contribution */
  static #PROGRESS_SCALE = 500;

  /** Node type string used in network node objects */
  static #NODE_TYPE_OUTPUT = 'output';

  /** Direction deltas for cardinal moves: N, E, S, W */
  static #DIRECTION_DELTAS: readonly [number, number][] = [
    [0, -1], // North
    [1, 0], // East
    [0, 1], // South
    [-1, 0], // West
  ];
  /** Lookup table for opposite directions (index -> opposite index). */
  static #OPPOSITE_DIR: readonly number[] = [2, 3, 0, 1];

  // ---------------------------------------------------------------------------
  // Pooled / reusable typed-array buffers (non‑reentrant) for simulation state
  // ---------------------------------------------------------------------------
  /** Visited flag per cell (0/1). Reused across simulations. @remarks Non-reentrant. */
  static #VisitedFlags: Uint8Array | null = null;
  /** Visit counts per cell (clamped). @remarks Non-reentrant. */
  static #VisitCounts: Uint16Array | null = null;
  /** Path X coordinates (index-aligned with #PathY). */
  static #PathX: Int32Array | null = null;
  /** Path Y coordinates (index-aligned with #PathX). */
  static #PathY: Int32Array | null = null;
  /** Capacity (cells) currently allocated for grid‑dependent arrays. */
  static #GridCapacity = 0;
  /** Capacity (steps) currently allocated for path arrays. */
  static #PathCapacity = 0;
  /** Cached maze width for index calculations. */
  static #CachedWidth = 0;
  /** Cached maze height for bounds validation. */
  static #CachedHeight = 0;

  /** Pooled softmax output (returned as a cloned plain array). */
  static #SOFTMAX = new Float64Array(4);

  /**
   * Seedable PRNG state (Mulberry32 style) stored in a pooled Uint32Array.
   * - When `null`, the implementation falls back to `Math.random()`.
   * - Using a typed-array for the single-word state avoids repeated
   *   heap allocations when reseeding and makes in-place updates explicit.
   */
  static #PRNGState: Uint32Array | null = null;

  // ---------------------------------------------------------------------------
  // Internal mutable run-scoped state (replaces (MazeMovement as any).foo uses)
  // ---------------------------------------------------------------------------
  /** Rolling saturation counter used for adaptive penalties */
  static #StateSaturations = 0;
  /** Consecutive steps with no movement to trigger forced exploration */
  static #StateNoMoveStreak = 0;
  /** Previous distance value supplied to vision builder */
  static #StatePrevDistanceStep: number | undefined = undefined;

  /**
   * Determine whether a proposed move target is valid: inside maze bounds
   * and not a wall. This function accepts either a coordinate tuple
   * (`[x,y]`) or separate numeric `x, y` arguments.
   *
   * Behaviour / rationale:
   * - Centralises argument handling for two public overloads so callers
   *   can use whichever form is more convenient.
   * - Defers the actual bounds and wall test to `#isCellOpen` which
   *   contains defensive checks and cached-dimension micro-optimisations.
   * - Uses a tiny pooled `Int32Array` (#COORD_SCRATCH) when coercing
   *   numeric args to 32-bit integers to avoid short-lived temporaries in
   *   hot loops.
   *
   * Steps:
   * 1) Normalize arguments into integer `x` and `y` coordinates.
   * 2) Delegate to the private `#isCellOpen` helper which performs the
   *    actual maze bounds and wall checks.
   *
   * @param encodedMaze - 2D read-only numeric maze (-1 === wall)
   * @param position - optional tuple [x,y] OR numeric `x` parameter
   * @param y - optional numeric `y` parameter when `x` and `y` passed separately
   * @returns `true` when the coordinates are within bounds and not a wall
   * @example
   * // tuple-form
   * MazeMovement.isValidMove(encodedMaze, [3, 2]);
   * // numeric-form
   * MazeMovement.isValidMove(encodedMaze, 3, 2);
   */
  static isValidMove(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    positionOrX: any,
    yMaybe?: any
  ): boolean {
    // Step 1: Normalize inputs to integer coordinates using the pooled scratch
    if (Array.isArray(positionOrX)) {
      // Destructure the tuple and coerce to 32-bit ints
      const [rawX, rawY] = positionOrX as [number, number];
      // Store into pooled scratch to avoid creating new temporaries
      MazeMovement.#COORD_SCRATCH[0] = rawX | 0;
      MazeMovement.#COORD_SCRATCH[1] = rawY | 0;
      return MazeMovement.#isCellOpen(
        encodedMaze,
        MazeMovement.#COORD_SCRATCH[0],
        MazeMovement.#COORD_SCRATCH[1]
      );
    }

    // Numeric-form: coerce both args into pooled scratch and delegate
    const rawX = positionOrX as number;
    const rawY = yMaybe as number;
    MazeMovement.#COORD_SCRATCH[0] = rawX | 0;
    MazeMovement.#COORD_SCRATCH[1] = rawY | 0;
    return MazeMovement.#isCellOpen(
      encodedMaze,
      MazeMovement.#COORD_SCRATCH[0],
      MazeMovement.#COORD_SCRATCH[1]
    );
  }

  /**
   * Generate a pseudo-random number in [0,1).
   *
   * Behaviour
   * - When `MazeMovement.#PRNGState` contains a single-word `Uint32Array`
   *   the method uses a tiny, allocation-free Mulberry32-like generator
   *   that mutates that pooled state in-place to produce deterministic
   *   results for testing and reproducible simulations.
   * - When `#PRNGState` is `null` the method falls back to the host
   *   JavaScript engine's `Math.random()`.
   *
   * Example
   * // (internal/private field shown for illustration)
   * MazeMovement['#PRNGState'] = new Uint32Array([123456789]);
   * const r = MazeMovement.#rand(); // deterministic in [0,1)
   *
   * Implementation steps (each step has an inline comment in the body):
   * 1) fast-path: fallback to Math.random when no pooled seed present
   * 2) advance the pooled uint32 state by a large odd constant (wraps)
   * 3) apply integer scrambles (xors + Math.imul) to mix bits
   * 4) final mix and convert the 32-bit integer to a float in [0,1)
   *
   * @returns number in range [0,1)
   */
  static #rand(): number {
    // Fast-path: if no pooled deterministic state is present, use engine RNG
    const pooledState = MazeMovement.#PRNGState;
    if (pooledState == null || pooledState.length === 0) {
      return Math.random();
    }

    // Step 1: advance pooled state in-place by a large odd increment and
    // keep everything in uint32 land using >>> 0. Using a pooled Uint32Array
    // avoids allocating a new seed object on every call.
    const current = (pooledState[0] + 0x6d2b79f5) >>> 0;
    pooledState[0] = current;

    // Step 2: perform integer scrambles using imul/xor/shifts to mix bits.
    // Local descriptive names help readers (and JITs) reason about the math.
    let mixed = current;
    // multiply/xor mix stage 1
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1) >>> 0;
    // multiply/xor mix stage 2
    mixed =
      (mixed ^ (mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61))) >>> 0;

    // Step 3: final avalanche and convert to float in [0,1) by dividing
    // by 2^32. >>> 0 ensures an unsigned 32-bit integer before the division.
    const final32 = (mixed ^ (mixed >>> 14)) >>> 0;
    return final32 / 4294967296; // 2^32
  }

  /**
   * Convert 2D coordinates (x,y) to a linear index into pooled grid buffers.
   *
   * Purpose:
   * - All pooled typed-arrays (visited flags, visit counts, etc.) are
   *   indexed using this linear index: index = y * width + x.
   * - Using `Math.imul` provides a fast 32-bit integer multiplication which
   *   avoids potential floating point rounding for large grids and is
   *   slightly faster on some engines.
   *
   * Steps:
   * 1) Compute the row stride (number of cells in full rows above `y`).
   * 2) Add the column offset `x` to produce the final linear index.
   *
   * @param x - Column coordinate (0-based)
   * @param y - Row coordinate (0-based)
   * @returns Linearized cell index used for indexing pooled arrays
   * @example
   * // For a maze width of 10, (x=3,y=2) -> index = 2*10 + 3 = 23
   * MazeMovement.#CachedWidth = 10; // (normally set by #initBuffers)
   * const idx = MazeMovement.#index(3, 2); // 23
   */
  static #index(x: number, y: number): number {
    // Step 1: compute number of cells spanned by full rows above `y`.
    const rowStride = Math.imul(y, MazeMovement.#CachedWidth);

    // Step 2: add the column offset to obtain a compact linear index.
    const linearIndex = rowStride + x;

    // Return the index (intended to be used with pooled typed arrays).
    return linearIndex;
  }

  /**
   * Ensure pooled typed-array buffers are allocated and sized for the
   * provided maze dimensions and maximum path length.
   *
   * Behaviour & rationale:
   * - Reuses existing pooled arrays when possible to avoid repeated
   *   heap allocations during many fast simulations.
   * - When growing, allocates the next power-of-two capacity to amortize
   *   future resizes (common pooling strategy).
   * - Only the actively used portion of pooled buffers is cleared to keep
   *   clears cheap for large, reused buffers.
   *
   * Steps:
   * 1) Compute required cell count for grid buffers.
   * 2) Grow or reuse `#VisitedFlags` and `#VisitCounts` as needed.
   * 3) Grow or reuse path buffers `#PathX` / `#PathY` for `maxSteps+1` entries.
   * 4) Cache width/height for index arithmetic used by other helpers.
   *
   * @param width - maze width (columns)
   * @param height - maze height (rows)
   * @param maxSteps - maximum expected path length (safety bound)
   * @example
   * MazeMovement.#initBuffers(32, 20, 1500);
   */
  static #initBuffers(width: number, height: number, maxSteps: number) {
    // Step 1: required cell count for the grid
    const requiredCellCount = width * height;

    // Step 2: ensure grid buffers large enough; grow to next power-of-two when needed
    if (!this.#VisitedFlags || requiredCellCount > this.#GridCapacity) {
      const newCellCapacity = MazeMovement.#nextPow2(requiredCellCount);
      // Allocate new pooled typed arrays
      this.#VisitedFlags = new Uint8Array(newCellCapacity);
      this.#VisitCounts = new Uint16Array(newCellCapacity);
      // Record the new pool capacity
      this.#GridCapacity = newCellCapacity;
    } else {
      // Fast-clear only the active region; keep remainder for reuse
      this.#VisitedFlags.fill(0, 0, requiredCellCount);
      this.#VisitCounts!.fill(0, 0, requiredCellCount);
    }

    // Step 3: ensure path buffers sized for maxSteps+1 entries (path includes start)
    const requiredPathEntries = maxSteps + 1;
    if (!this.#PathX || requiredPathEntries > this.#PathCapacity) {
      const newPathCapacity = MazeMovement.#nextPow2(requiredPathEntries);
      this.#PathX = new Int32Array(newPathCapacity);
      this.#PathY = new Int32Array(newPathCapacity);
      this.#PathCapacity = newPathCapacity;
    }

    // Step 4: cache dimensions used by indexing helpers
    this.#CachedWidth = width;
    this.#CachedHeight = height;
  }

  /**
   * Return the smallest power-of-two integer >= `n`.
   *
   * Implementation notes:
   * - Fast-path for typical 32-bit ranges uses `Math.clz32` and bit ops
   *   which are very fast on modern JS engines.
   * - For extremely large values (outside 32-bit unsigned range) a safe
   *   fallback iteratively doubles to avoid incorrect 32-bit shifts.
   *
   * Steps:
   * 1) Handle trivial and boundary cases (n <= 1).
   * 2) For n within 32-bit range, compute next power using leading-zero count.
   * 3) For larger n, fall back to a safe doubling loop.
   *
   * @param n - Target minimum integer (expected positive)
   * @returns The smallest power of two >= n
   * @example
   * MazeMovement.#nextPow2(13) === 16
   */
  static #nextPow2(n: number): number {
    // Step 1: sanitize input and handle trivial cases
    const requested = Math.max(1, Math.floor(n));
    if (requested <= 1) return 1;

    // Step 2: fast 32-bit path using clz32 when safe
    if (requested <= 0xffffffff) {
      // values are treated as unsigned 32-bit; compute next power-of-two
      const v = (requested - 1) >>> 0; // ensure uint32
      const leadingZeros = Math.clz32(v);
      const exponent = 32 - leadingZeros;
      // shifting by 32 is undefined, clamp exponent to [0,31]
      const clampedExp = Math.min(31, Math.max(0, exponent));
      const power = 1 << clampedExp;
      // If the computed power is less than requested (edge case), double once
      return power >= requested ? power : power << 1;
    }

    // Step 3: safe fallback for very large numbers — doubling loop (rare)
    let power = 1;
    while (power < requested) power = power * 2;
    return power;
  }

  /**
   * Materialize the current path stored in the pooled `#PathX` / `#PathY`
   * buffers into a fresh, mutable array of [x,y] tuples.
   *
   * Rationale:
   * - Internal path coordinate buffers are pooled to minimize allocations
   *   during many fast simulations. Callers often require an independent
   *   array (for inspection, serialization, or mutation) so we copy the
   *   active prefix into a new plain JS array of tuples.
   *
   * Steps:
   * 1) Normalize the requested length and early-return an empty array for 0.
   * 2) Read local references to the pooled typed-arrays to reduce repeated
   *    global/property lookups in the hot loop.
   * 3) Allocate the result array with the known length and fill it with
   *    [x,y] tuples copied from the pooled Int32Arrays.
   *
   * @param length - number of path entries to materialize (usually `state.pathLength`)
   * @returns A newly allocated array of `[x, y]` tuples with `length` entries.
   * @example
   * // produce an independent copy of the active path
   * const pathSnapshot = MazeMovement.#materializePath(state.pathLength);
   */
  static #materializePath(length: number): [number, number][] {
    // Step 1: sanitize and fast-return for empty paths
    const entries = Math.max(0, Math.floor(length));
    if (entries === 0) return [];

    // Step 2: local references to pooled buffers (faster in a tight loop)
    const pathX = MazeMovement.#PathX!;
    const pathY = MazeMovement.#PathY!;

    // Step 3: allocate output array of known size and populate
    const out = new Array<[number, number]>(entries);
    for (let index = 0; index < entries; index++) {
      // Read int32 entries into descriptive locals before creating tuple
      const x = pathX[index];
      const y = pathY[index];
      out[index] = [x, y];
    }
    return out;
  }

  /**
   * Return the opposite cardinal direction for a given action index.
   *
   * Rationale:
   * - Using a centered lookup (add half the action-space and wrap) keeps the
   *   implementation independent of the exact `#ACTION_DIM` value and avoids
   *   branchy conditionals when the action space changes.
   * - Special-case the `#NO_MOVE` sentinel so callers that pass `-1` preserve
   *   the 'no move' semantics instead of producing a wrapped numeric result.
   *
   * Steps:
   * 1) If `direction` equals `#NO_MOVE` return `#NO_MOVE` immediately.
   * 2) Coerce the input to a 32-bit integer and normalize into [0, ACTION_DIM).
   * 3) Add the half-span (ACTION_DIM >> 1) and wrap with modulo to compute
   *    the opposite index.
   *
   * @param direction - action index (0=N,1=E,2=S,3=W) or `#NO_MOVE` (-1)
   * @returns Opposite action index, or `#NO_MOVE` when input was `#NO_MOVE`.
   * @example
   * MazeMovement.#opposite(0) === 2; // North -> South
   */
  static #opposite(direction: number): number {
    // Step 1: preserve the no-move sentinel
    if (direction === MazeMovement.#NO_MOVE) return MazeMovement.#NO_MOVE;

    // Step 2: coerce to 32-bit integer and normalize into [0, ACTION_DIM)
    const coerced = direction | 0; // fast int coercion
    const dim = MazeMovement.#ACTION_DIM;
    let normalized = coerced % dim;
    if (normalized < 0) normalized += dim; // handle negative remainders

    // Step 3: compute opposite by adding half the action-space and wrapping
    const halfSpan = dim >> 1; // integer division by 2
    return (normalized + halfSpan) % dim;
  }

  /**
   * Sum a contiguous group of `#VISION_GROUP_LEN` elements in the vision
   * vector starting at `start`.
   *
   * Behaviour and rationale:
   * - This helper is a hot-path primitive used by perception checks. It
   *   avoids allocations and keeps the loop minimal for performance.
   * - The implementation is defensive: it bounds-checks the input so a
   *   malformed `start` or shorter-than-expected `vision` arrays won't throw.
   *
   * Steps:
   * 1) Sanitize `start` and compute the clamped `end` index using the
   *    configured `#VISION_GROUP_LEN`.
   * 2) Iterate linearly and accumulate into a numeric accumulator.
   * 3) Return the numeric sum.
   *
   * @param vision - flat array of numeric vision inputs
   * @param start - start index of the group to sum
   * @returns numeric sum of the group (0 for empty/out-of-range input)
   * @example
   * // Sum the LOS group starting at index 8
   * const losSum = MazeMovement.#sumVisionGroup(visionVector, MazeMovement.#VISION_LOS_START);
   */
  static #sumVisionGroup(vision: number[], start: number) {
    // Step 1: sanitize and clamp inputs (use descriptive names for clarity)
    const groupLength = MazeMovement.#VISION_GROUP_LEN;
    const sanitizedStart = Math.max(0, start | 0);
    const clampedEnd = Math.min(vision.length, sanitizedStart + groupLength);
    if (sanitizedStart >= clampedEnd) return 0;

    // Step 2: reuse pooled scratch buffer to avoid per-call allocations.
    // NOTE: #SCRATCH_CENTERED is a pooled Float64Array sized to at least
    // `#VISION_GROUP_LEN` and this class is non-reentrant in hot paths.
    const pooledScratch = MazeMovement.#SCRATCH_CENTERED;

    // Step 3: accumulate values into a local numeric accumulator while
    // copying into the pooled scratch. Copying documents intent and keeps
    // micro-benchmarks stable across engines (no hidden temporaries).
    let sumAccumulator = 0;
    let writeIndex = 0;
    for (let readIndex = sanitizedStart; readIndex < clampedEnd; readIndex++) {
      const value = vision[readIndex] ?? 0;
      pooledScratch[writeIndex++] = value;
      sumAccumulator += value;
    }

    // Step 4: return the numeric sum. We intentionally do not clear the
    // pooled scratch — consumers that rely on it should overwrite contents.
    return sumAccumulator;
  }

  /**
   * Compute an adaptive epsilon used for epsilon-greedy exploration.
   *
   * Behaviour:
   * - Epsilon controls random exploratory moves. This helper centralizes
   *   the tuning logic so callers can keep the hot loop small.
   * - The returned value is intentionally conservative (often 0) unless
   *   particular conditions (warmup, stagnation, or saturations) are met.
   * - When the agent is near the goal (`distHere` small) exploration is
   *   suppressed by clamping epsilon to a small minimum.
   *
   * Steps:
   * 1) Compute boolean predicates for warmup/stagnation/saturation cases.
   * 2) Select the base epsilon from the highest-priority matching case.
   * 3) If proximate to goal, clamp epsilon to `#EPSILON_MIN_NEAR_GOAL`.
   * 4) Return the chosen epsilon.
   *
   * @param stepNumber - global step index inside the simulation loop
   * @param stepsSinceImprovement - number of steps since last improvement
   * @param distHere - current distance-to-goal (used to suppress exploration)
   * @param saturations - rolling saturation count used for bias adjustments
   * @returns epsilon value in [0,1] used for epsilon-greedy exploration
   * @example
   * // Typical usage inside simulation loop
   * const eps = MazeMovement.#computeEpsilon(step, state.stepsSinceImprovement, state.distHere, MazeMovement.#StateSaturations);
   */
  static #computeEpsilon(
    stepNumber: number,
    stepsSinceImprovement: number,
    distHere: number,
    saturations: number
  ): number {
    // Step 1: evaluate predicates with descriptive names for clarity
    const isWarmup = stepNumber < MazeMovement.#EPSILON_WARMUP_STEPS;
    const isHighlyStagnant =
      stepsSinceImprovement > MazeMovement.#EPSILON_STAGNANT_HIGH_THRESHOLD;
    const isModeratelyStagnant =
      stepsSinceImprovement > MazeMovement.#EPSILON_STAGNANT_MED_THRESHOLD;
    const isSaturationTriggered =
      saturations > MazeMovement.#EPSILON_SATURATION_TRIGGER;

    // Step 2: choose the most relevant base epsilon (priority order)
    let chosenEpsilon = 0;
    // Use a switch(true) so each predicate is a case and priority is explicit
    switch (true) {
      case isWarmup:
        chosenEpsilon = MazeMovement.#EPSILON_INITIAL;
        break;
      case isHighlyStagnant:
        chosenEpsilon = MazeMovement.#EPSILON_STAGNANT_HIGH;
        break;
      case isModeratelyStagnant:
        chosenEpsilon = MazeMovement.#EPSILON_STAGNANT_MED;
        break;
      case isSaturationTriggered:
        chosenEpsilon = MazeMovement.#EPSILON_SATURATIONS;
        break;
      default:
        // leave chosenEpsilon at default 0
        break;
    }

    // Step 3: suppress exploration near the goal by clamping down
    if (distHere <= MazeMovement.#PROXIMITY_SUPPRESS_EXPLOR_DIST) {
      // Use Math.min to prefer the smaller (less exploratory) epsilon
      chosenEpsilon = Math.min(
        chosenEpsilon,
        MazeMovement.#EPSILON_MIN_NEAR_GOAL
      );
    }

    // Step 4: return the decided epsilon
    return chosenEpsilon;
  }

  /**
   * Check whether a cell at (x, y) is inside the maze bounds and not a wall.
   *
   * Behaviour / rationale:
   * - Prefers cached maze dimensions (when they match the provided maze)
   *   to avoid repeated nested property accesses inside hot loops.
   * - Defensively guards against malformed inputs (empty rows / missing data)
   *   and treats those as non-open (equivalent to wall/out-of-bounds).
   *
   * Steps:
   * 1) Resolve maze width/height (prefer cached values when appropriate).
   * 2) Perform fast, descriptive bounds checks.
   * 3) Read the cell once and compare against the wall sentinel (-1).
   *
   * @param encodedMaze - 2D read-only numeric maze representation (-1 == wall)
   * @param x - zero-based column index to test
   * @param y - zero-based row index to test
   * @returns true when the cell exists and is not a wall
   * @example
   * // Typical usage inside simulation loop
   * const open = MazeMovement.#isCellOpen(encodedMaze, x, y);
   */
  static #isCellOpen(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    x: number,
    y: number
  ): boolean {
    // Step 1: resolve provided maze dimensions and grab a stable first-row
    const providedRowCount = encodedMaze?.length ?? 0;
    const firstRow = encodedMaze?.[0];
    const providedColumnCount = firstRow?.length ?? 0;

    // Step 2: prefer cached dimensions when they match the provided maze
    const cachedColumnCount = MazeMovement.#CachedWidth;
    const cachedRowCount = MazeMovement.#CachedHeight;

    const mazeColumnCount =
      cachedColumnCount > 0 &&
      cachedRowCount === providedRowCount &&
      cachedColumnCount === providedColumnCount
        ? cachedColumnCount
        : providedColumnCount;
    const mazeRowCount =
      cachedRowCount > 0 &&
      cachedColumnCount === providedColumnCount &&
      cachedRowCount === providedRowCount
        ? cachedRowCount
        : providedRowCount;

    // Step 3: coerce coordinates into the pooled scratch Int32Array to avoid
    // creating temporary boxed numbers on hot paths.
    MazeMovement.#COORD_SCRATCH[0] = x | 0;
    MazeMovement.#COORD_SCRATCH[1] = y | 0;
    const col = MazeMovement.#COORD_SCRATCH[0];
    const row = MazeMovement.#COORD_SCRATCH[1];

    // Step 4: fast bounds checks with clear descriptive names
    if (row < 0 || row >= mazeRowCount) return false;
    if (col < 0 || col >= mazeColumnCount) return false;

    // Step 5: defensive single-read of the row and cell value test
    const targetRow = encodedMaze[row];
    if (!targetRow) return false; // malformed row -> treat as wall/out-of-bounds
    const cellValue = targetRow[col];
    return cellValue !== -1;
  }

  /**
   * Unified distance lookup for a cell coordinate.
   *
   * Behaviour / rationale:
   * - Fast-path: when a `distanceMap` is supplied and contains a finite
   *   numeric entry for the coordinate, that value is returned immediately.
   * - Defensive: performs robust bounds checking and uses cached maze
   *   dimensions (when they match the provided maze) to avoid repeated
   *   nested property lookups in hot code paths.
   * - Fallback: when no finite distance is available, returns `Infinity` to
   *   indicate unknown/unreachable distance (preserves previous behaviour).
   *
   * Steps:
   * 1) Coerce incoming coordinates to 32-bit integers.
   * 2) Fast-path check for a finite value in the optional `distanceMap`.
   * 3) Validate bounds using cached dimensions when they align with the
   *    provided maze to reduce property access overhead.
   * 4) If no distance found, return `Infinity` (unknown/unreachable).
   *
   * @param encodedMaze - 2D read-only numeric maze representation
   * @param coords - readonly tuple [x, y] of zero-based coordinates
   * @param distanceMap - optional precomputed distance map (same shape as maze)
   * @returns finite distance number when available, otherwise `Infinity`
   * @example
   * const d = MazeMovement.#distanceAt(encodedMaze, [3,2], distanceMap);
   */
  static #distanceAt(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    [x, y]: readonly [number, number],
    distanceMap?: number[][]
  ): number {
    // Step 1: coerce coordinates to 32-bit integers for consistent indexing
    const xCoord = x | 0;
    const yCoord = y | 0;

    // Step 2: fast-path: return from provided distanceMap when present
    if (
      distanceMap &&
      distanceMap[yCoord] !== undefined &&
      Number.isFinite(distanceMap[yCoord][xCoord])
    ) {
      return distanceMap[yCoord][xCoord];
    }

    // Step 3: bounds validation — prefer cached sizes when they match the maze
    const providedHeight = encodedMaze?.length ?? 0;
    const firstRow = encodedMaze?.[0];
    const providedWidth = firstRow?.length ?? 0;

    const cachedWidth = MazeMovement.#CachedWidth;
    const cachedHeight = MazeMovement.#CachedHeight;

    const mazeWidth =
      cachedWidth > 0 &&
      cachedHeight === providedHeight &&
      cachedWidth === providedWidth
        ? cachedWidth
        : providedWidth;
    const mazeHeight =
      cachedHeight > 0 &&
      cachedWidth === providedWidth &&
      cachedHeight === providedHeight
        ? cachedHeight
        : providedHeight;

    if (xCoord < 0 || xCoord >= mazeWidth) return Infinity;
    if (yCoord < 0 || yCoord >= mazeHeight) return Infinity;

    // Step 4: no precomputed distance found — preserve historical fallback
    // (treat as unknown/unreachable). A BFS fallback could be added here
    // if callers require an on-demand computation, but that is intentionally
    // omitted to avoid expensive work in hot paths.
    return Infinity;
  }

  // ...existing code...

  /**
   * Moves the agent in the specified direction if the move is valid.
   *
   * Handles collision detection with walls and maze boundaries,
   * preventing the agent from making invalid moves.
   *
   * @param encodedMaze - 2D array representation of the maze.
   * @param position - Current [x, y] position of the agent.
   * @param direction - Direction index (0=North, 1=East, 2=South, 3=West, -1=No move).
   * @returns { [number, number] } New position after movement, or original position if move was invalid.
   */
  static moveAgent(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    position: readonly [number, number],
    direction: number
  ): [number, number] {
    // If direction is -1, do not move — return a mutable copy for callers that expect a mutable tuple
    if (direction === MazeMovement.#NO_MOVE) {
      return [position[0], position[1]] as [number, number];
    }
    // Copy current position
    /**
     * Next position candidate for the agent after moving
     */
    // Create a mutable copy of the readonly input position for local mutation
    const nextPosition: [number, number] = [position[0], position[1]] as [
      number,
      number
    ];
    // Update position based on direction using the centralized deltas table
    if (direction >= 0 && direction < MazeMovement.#ACTION_DIM) {
      const [dx, dy] = MazeMovement.#DIRECTION_DELTAS[direction];
      nextPosition[0] += dx;
      nextPosition[1] += dy;
    }
    // Check if the new position is valid
    if (MazeMovement.isValidMove(encodedMaze, nextPosition)) {
      return nextPosition;
    } else {
      // If invalid, stay in place — return a mutable copy to satisfy return type
      return [position[0], position[1]] as [number, number];
    }
  }

  /**
   * Choose an action index from network outputs.
   *
   * Behaviour:
   * - Centers the raw outputs (logits), computes an adaptive temperature
   *   based on collapse heuristics, performs a numerically-stable softmax
   *   into pooled scratch buffers, and returns argmax plus diagnostics.
   * - Reuses pooled typed-array scratch buffers to avoid per-call
   *   allocations; the method is therefore non-reentrant.
   *
   * Steps (implemented inline with comments):
   * 1) Validate inputs and early-return a safe default for malformed inputs.
   * 2) Center logits and compute variance/std for adaptive temperature.
   * 3) Compute softmax in pooled buffers with numerical-stability trick.
   * 4) Determine argmax (best action) and second-best probability.
   * 5) Compute normalized entropy and return a defensive copy of softmax.
   *
   * @param outputs - Array of raw network outputs (logits), expected length === #ACTION_DIM
   * @returns An object with:
   *  - direction: chosen action index (0..#ACTION_DIM-1) or #NO_MOVE on invalid input
   *  - softmax: fresh array copy of probabilities (length #ACTION_DIM)
   *  - entropy: normalized entropy in [0,1]
   *  - maxProb: probability of the chosen action
   *  - secondProb: probability of the runner-up action
   * @example
   * const result = MazeMovement.selectDirection([0.2, 1.4, -0.1, 0]);
   * // result.direction -> 1 (for example)
   */
  static selectDirection(
    outputs: number[]
  ): {
    direction: number;
    softmax: number[];
    entropy: number;
    maxProb: number;
    secondProb: number;
  } {
    // Step 1: validate inputs and provide safe default
    const actionCount = MazeMovement.#ACTION_DIM;
    if (!Array.isArray(outputs) || outputs.length !== actionCount) {
      return {
        direction: MazeMovement.#NO_MOVE,
        softmax: Array.from(MazeMovement.#SOFTMAX),
        entropy: 0,
        maxProb: 0,
        secondProb: 0,
      };
    }

    // Local references to pooled scratch buffers for clarity and perf.
    const centered = MazeMovement.#SCRATCH_CENTERED;
    const exps = MazeMovement.#SCRATCH_EXPS;
    const softmaxPooled = MazeMovement.#SOFTMAX;

    // Step 2: center logits and compute variance (numerically simple loop)
    let sum = 0;
    for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
      sum += outputs[actionIndex];
    }
    const meanOutput = sum / actionCount;

    let varianceAccumulator = 0;
    for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
      const delta = outputs[actionIndex] - meanOutput;
      centered[actionIndex] = delta; // write into pooled centered buffer
      varianceAccumulator += delta * delta;
    }
    varianceAccumulator /= actionCount;
    let standardDeviation = Math.sqrt(varianceAccumulator);
    if (
      !Number.isFinite(standardDeviation) ||
      standardDeviation < MazeMovement.#STD_MIN
    ) {
      standardDeviation = MazeMovement.#STD_MIN;
    }

    // Adaptive collapse ratio -> temperature
    const collapseRatio =
      standardDeviation < MazeMovement.#COLLAPSE_STD_THRESHOLD
        ? MazeMovement.#COLLAPSE_RATIO_FULL
        : standardDeviation < MazeMovement.#COLLAPSE_STD_MED
        ? MazeMovement.#COLLAPSE_RATIO_HALF
        : 0;
    const temperature =
      MazeMovement.#TEMPERATURE_BASE +
      MazeMovement.#TEMPERATURE_SCALE * collapseRatio;

    // Step 3: softmax numerically stable: subtract maxCentered before exp
    let maxCentered = -Infinity;
    for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
      const v = centered[actionIndex];
      if (v > maxCentered) maxCentered = v;
    }

    let expSum = 0;
    for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
      const value = Math.exp(
        (centered[actionIndex] - maxCentered) / temperature
      );
      exps[actionIndex] = value;
      expSum += value;
    }
    if (expSum === 0) expSum = 1; // defensive

    // Step 4: compute probabilities in pooled softmax buffer and find top-two
    let chosenDirection = 0;
    let bestProb = -Infinity;
    let runnerUpProb = 0;
    for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
      const prob = exps[actionIndex] / expSum;
      softmaxPooled[actionIndex] = prob;
      if (prob > bestProb) {
        runnerUpProb = bestProb;
        bestProb = prob;
        chosenDirection = actionIndex;
      } else if (prob > runnerUpProb) {
        runnerUpProb = prob;
      }
    }

    // Step 5: compute normalized entropy (divide by log(actionCount))
    let entropy = 0;
    for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
      const p = softmaxPooled[actionIndex];
      if (p > 0) entropy += -p * Math.log(p);
    }
    entropy /= MazeMovement.#LOG_ACTIONS;

    // Defensive: return a copy of pooled softmax so callers cannot mutate
    return {
      direction: chosenDirection,
      softmax: Array.from(softmaxPooled),
      entropy,
      maxProb: bestProb,
      secondProb: runnerUpProb,
    };
  }

  /**
   * Simulates the agent navigating the maze using its neural network.
   *
   * Runs a complete simulation of an agent traversing a maze,
   * using its neural network for decision making. This implementation focuses
   * on a minimalist approach, putting more responsibility on the neural network.
   *
   * @param network - Neural network controlling the agent.
   * @param encodedMaze - 2D array representation of the maze.
   * @param startPos - Starting position [x,y] of the agent.
   * @param exitPos - Exit/goal position [x,y] of the maze.
   * @param maxSteps - Maximum steps allowed before terminating (default 3000).
   * @returns Object containing:
   *   - success: Boolean indicating if exit was reached.
   *   - steps: Number of steps taken.
   *   - path: Array of positions visited.
   *   - fitness: Calculated fitness score for evolution.
   *   - progress: Percentage progress toward exit (0-100).
   */
  static simulateAgent(
    network: INetwork,
    encodedMaze: number[][],
    startPos: readonly [number, number],
    exitPos: readonly [number, number],
    distanceMap?: number[][],
    maxSteps = MazeMovement.#DEFAULT_MAX_STEPS
  ): {
    success: boolean;
    steps: number;
    path: readonly [number, number][];
    fitness: number;
    progress: number;
    saturationFraction?: number;
    actionEntropy?: number;
  } {
    const state = MazeMovement.#initRunState(
      encodedMaze,
      startPos,
      distanceMap,
      maxSteps
    );

    while (state.steps < maxSteps) {
      state.steps++;
      // Record cell visit & derive penalties for loops / memory / revisits
      MazeMovement.#recordVisitAndUpdatePenalties(state, encodedMaze);

      // Build perception & compute current distance for exploration logic
      MazeMovement.#buildVisionAndDistance(
        state,
        encodedMaze,
        exitPos,
        distanceMap
      );

      // Neural net activation & saturation handling
      MazeMovement.#decideDirection(state, network, encodedMaze, distanceMap);

      // Proximity greedy override
      MazeMovement.#maybeApplyProximityGreedy(state, encodedMaze, distanceMap);

      // Epsilon exploration
      MazeMovement.#maybeApplyEpsilonExploration(state, encodedMaze);

      // Force exploration if stuck
      MazeMovement.#maybeForceExploration(state, encodedMaze);

      // Execute move & update rewards
      MazeMovement.#executeMoveAndRewards(state, encodedMaze, distanceMap);

      // Post‑action repetition / entropy / saturation penalties
      MazeMovement.#applyPostActionPenalties(state);

      // Deep stagnation termination
      if (MazeMovement.#maybeTerminateDeepStagnation(state)) break;

      // Success check
      if (
        state.position[0] === exitPos[0] &&
        state.position[1] === exitPos[1]
      ) {
        return MazeMovement.#finalizeSuccess(state, maxSteps);
      }
    }

    return MazeMovement.#finalizeFailure(
      state,
      encodedMaze,
      startPos,
      exitPos,
      distanceMap
    );
  }

  // ---------------------------------------------------------------------------
  // Private helper methods (refactored from large simulateAgent body)
  // ---------------------------------------------------------------------------

  /** Internal aggregate simulation state (not exported). */
  static #initRunState(
    encodedMaze: number[][],
    startPos: readonly [number, number],
    distanceMap: number[][] | undefined,
    maxSteps: number
  ): SimulationState {
    // Reset global mutable counters reused across runs
    MazeMovement.#StateSaturations = 0;
    MazeMovement.#StateNoMoveStreak = 0;
    MazeMovement.#StatePrevDistanceStep = undefined;
    const height = encodedMaze.length;
    const width = encodedMaze[0].length;
    const hasDistanceMap =
      Array.isArray(distanceMap) && distanceMap.length === height;
    MazeMovement.#initBuffers(width, height, maxSteps);
    // Seed path with start position
    const position: [number, number] = [startPos[0], startPos[1]];
    MazeMovement.#PathX![0] = position[0];
    MazeMovement.#PathY![0] = position[1];
    const historyCapacity = MazeMovement.#MOVE_HISTORY_LENGTH;
    const state: SimulationState = {
      position,
      steps: 0,
      pathLength: 1,
      visitedUniqueCount: 0,
      hasDistanceMap,
      distanceMap,
      minDistanceToExit: hasDistanceMap
        ? distanceMap![position[1]]?.[position[0]] ?? Infinity
        : MazeMovement.#distanceAt(encodedMaze, position, distanceMap),
      progressReward: 0,
      newCellExplorationBonus: 0,
      invalidMovePenalty: 0,
      prevAction: MazeMovement.#NO_MOVE,
      stepsSinceImprovement: 0,
      lastDistanceGlobal: MazeMovement.#distanceAt(
        encodedMaze,
        position,
        distanceMap
      ),
      saturatedSteps: 0,
      recentPositions: [] as [number, number][],
      localAreaPenalty: 0,
      directionCounts: [0, 0, 0, 0] as number[],
      moveHistoryRing: new Int32Array(historyCapacity),
      moveHistoryLength: 0,
      moveHistoryHead: 0,
      currentCellIndex: 0,
      loopPenalty: 0,
      memoryPenalty: 0,
      revisitPenalty: 0,
      visitsAtCurrent: 0,
      distHere: Infinity,
      vision: [] as number[],
      actionStats: null as any,
      direction: MazeMovement.#NO_MOVE,
      moved: false,
      prevDistance: Infinity,
      earlyTerminate: false,
    };
    return state;
  }

  /**
   * Push a cell index into the circular move-history ring buffer.
   *
   * Behaviour / rationale:
   * - The history is stored in a preallocated `Int32Array` (`moveHistoryRing`) to
   *   avoid allocations. This helper updates the head pointer and length in-place.
   * - The method is deliberately allocation-free and fast; callers use the
   *   ring to detect tiny oscillations like A->B->A->B.
   *
   * Steps:
   * 1) Read local references to the ring, head and length for faster hot-path access.
   * 2) Write the provided `cellIndex` at the current head slot.
   * 3) Advance the head index modulo the ring capacity and store it back on state.
   * 4) If the ring was not yet full, increment the stored length.
   *
   * @param state - simulation state containing `moveHistoryRing`, `moveHistoryHead`, `moveHistoryLength`
   * @param cellIndex - linearized cell index to push into history
   * @returns void
   * @example
   * MazeMovement.#pushHistory(state, currentCellIndex);
   */
  static #pushHistory(state: SimulationState, cellIndex: number) {
    // Step 1: local references for perf and clearer names
    const ring = state.moveHistoryRing;
    let headIndex = state.moveHistoryHead | 0; // coerce to int
    const currentLength = state.moveHistoryLength;
    const capacity = ring.length;

    // Defensive: if capacity is zero nothing to do (shouldn't happen normally)
    if (capacity === 0) return;

    // Step 2: write the new entry into the ring at the current head
    ring[headIndex] = cellIndex;

    // Step 3: advance the head (wrap using modulo) and store back on state
    headIndex = (headIndex + 1) % capacity;
    state.moveHistoryHead = headIndex;

    // Step 4: if ring wasn't full yet, increment the recorded length
    if (currentLength < capacity) state.moveHistoryLength = currentLength + 1;
  }

  /**
   * Return the nth-most-recent entry from the circular move history.
   *
   * Behaviour:
   * - `n` is 1-based: `1` returns the last pushed entry, `2` the one
   *   before that, etc. Returns `undefined` when `n` is out of range.
   * - Uses only preallocated `Int32Array` ring storage and integer
   *   arithmetic; allocation-free and safe for hot paths.
   *
   * Steps:
   * 1) Coerce inputs to 32-bit integers and validate `n` against stored length.
   * 2) Compute the wrapped index by subtracting `n` from the head and
   *    normalizing into `[0, capacity)` via addition + modulo.
   * 3) Return the ring value at the computed slot or `undefined` when invalid.
   *
   * @param state - simulation state containing `moveHistoryRing`, `moveHistoryHead`, `moveHistoryLength`
   * @param nth - 1-based index from the end (1 === last pushed)
   * @returns linearized cell index when present, otherwise `undefined`
   * @example
   * const last = MazeMovement.#nthFromHistoryEnd(state, 1);
   */
  static #nthFromHistoryEnd(
    state: SimulationState,
    nth: number
  ): number | undefined {
    // Step 1: coerce arguments and validate
    const requested = nth | 0;
    const length = state.moveHistoryLength | 0;
    if (requested <= 0 || requested > length) return undefined;

    // Step 2: local refs and capacity (fast-path locals reduce property loads)
    const ring = state.moveHistoryRing;
    const capacity = ring.length;
    if (capacity === 0) return undefined; // defensive: empty ring
    let head = state.moveHistoryHead | 0;

    // Compute wrapped index: head - requested (1-based) then normalize
    let rawIndex = head - requested;
    // Normalize negative values into [0, capacity) without using slow division
    rawIndex = ((rawIndex % capacity) + capacity) % capacity;

    // Step 3: return the stored value (Int32Array read)
    return ring[rawIndex];
  }

  /**
   * Record a visit to the current cell and derive shaping penalties.
   *
   * Behaviour / rationale:
   * - Updates pooled visit flags and visit counts (allocation-free).
   * - Pushes the cell into the fixed-size circular `moveHistoryRing` and
   *   derives three shaping penalties used to discourage trivial oscillation
   *   and revisiting behavior: loopPenalty, memoryPenalty, revisitPenalty.
   * - May mark the run `earlyTerminate` when a visit count exceeds a hard threshold.
   *
   * Steps:
   * 1) Compute linearized cell index and mark it visited (unique-visit accounting).
   * 2) Increment the per-cell visit counter and push into the circular history.
   * 3) Detect tiny A↔B oscillations (A->B->A->B) and apply loop penalty.
   * 4) Scan recent history (excluding last entry) for returning-to-recent-cell
   *    and apply memory-return penalty if found.
   * 5) Compute revisit penalty scaling with visit counts and enforce termination
   *    when visits exceed `#VISIT_TERMINATION_THRESHOLD`.
   *
   * @param state - current simulation state (modified in-place)
   * @param encodedMaze - read-only maze (unused directly here but kept for symmetry)
   * @returns void
   * @example
   * MazeMovement.#recordVisitAndUpdatePenalties(state, encodedMaze);
   */
  static #recordVisitAndUpdatePenalties(
    state: SimulationState,
    encodedMaze: number[][]
  ) {
    // Step 0: local references and descriptive names for hot-path perf
    const visitedFlags = MazeMovement.#VisitedFlags!;
    const visitCounts = MazeMovement.#VisitCounts!;
    const rewardScale = MazeMovement.#REWARD_SCALE;

    // Step 1: linearize current position and update unique-visit tracking
    const cellIndex = MazeMovement.#index(state.position[0], state.position[1]);
    state.currentCellIndex = cellIndex;
    if (!visitedFlags[cellIndex]) {
      visitedFlags[cellIndex] = 1;
      state.visitedUniqueCount++;
    }

    // Step 2: increment visit count and record into move-history ring
    visitCounts[cellIndex] = (visitCounts[cellIndex] + 1) as number;
    MazeMovement.#pushHistory(state, cellIndex);
    const visitsAtCell = (state.visitsAtCurrent = visitCounts[cellIndex]);

    // Step 3: loop detection (A->B->A->B) using the small fixed-length ring
    state.loopPenalty = 0;
    if (state.moveHistoryLength >= MazeMovement.#OSCILLATION_DETECT_LENGTH) {
      const last = MazeMovement.#nthFromHistoryEnd(state, 1)!;
      const secondLast = MazeMovement.#nthFromHistoryEnd(state, 2);
      const thirdLast = MazeMovement.#nthFromHistoryEnd(state, 3);
      const fourthLast = MazeMovement.#nthFromHistoryEnd(state, 4);
      // detect pattern (A, B, A, B) where positions alternate
      if (
        last === thirdLast &&
        secondLast !== undefined &&
        fourthLast !== undefined &&
        secondLast === fourthLast
      ) {
        state.loopPenalty = -MazeMovement.#LOOP_PENALTY * rewardScale;
      }
    }

    // Step 4: memory-return penalty — returning to any recent cell (excluding immediate previous)
    state.memoryPenalty = 0;
    if (state.moveHistoryLength > 1) {
      for (let offset = 2; offset <= state.moveHistoryLength; offset++) {
        const recentIndex = MazeMovement.#nthFromHistoryEnd(state, offset);
        if (recentIndex === cellIndex) {
          state.memoryPenalty =
            -MazeMovement.#MEMORY_RETURN_PENALTY * rewardScale;
          break;
        }
      }
    }

    // Step 5: revisit penalty (scaled by extra visits beyond the first)
    state.revisitPenalty = 0;
    if (visitsAtCell > 1) {
      state.revisitPenalty =
        -MazeMovement.#REVISIT_PENALTY_PER_VISIT *
        (visitsAtCell - 1) *
        rewardScale;
    }

    // Enforce harsh termination penalty if a cell is visited too often
    if (visitsAtCell > MazeMovement.#VISIT_TERMINATION_THRESHOLD) {
      state.invalidMovePenalty -=
        MazeMovement.#INVALID_MOVE_PENALTY_HARSH * rewardScale;
      state.earlyTerminate = true;
    }
  }

  /**
   * Build vision inputs and compute the current-cell distance used by
   * proximity and epsilon logic.
   *
   * Behaviour / rationale:
   * - Delegates perception construction to MazeVision.buildInputs6 and
   *   stores the resulting vision vector on `state.vision`.
   * - Updates the rolling previous-distance value (`#StatePrevDistanceStep`) so
   *   the next step's vision builder receives the correct prior distance.
   * - Minimizes allocations: when the builder returns an Array we assign it
   *   directly; otherwise we perform a single, explicit conversion.
   *
   * Steps:
   * 1) Early-exit if the run is marked `earlyTerminate`.
   * 2) Localize and coerce the current position, choose between `distanceMap`
   *    lookup or computed distance for the pre-move distance value.
   * 3) Call `MazeVision.buildInputs6(...)` with the prior distance and store
   *    the returned vision array on `state.vision` (single copy only when needed).
   * 4) Update `#StatePrevDistanceStep` and `state.distHere` for downstream logic.
   *
   * @param state - Simulation state object (mutated in-place)
   * @param encodedMaze - Read-only 2D maze array (rows of numeric columns)
   * @param exitPos - Exit coordinate tuple [x, y] used by the vision builder
   * @param distanceMap - Optional precomputed distance map aligned to `encodedMaze`
   * @returns void
   * @example
   * // inside the simulation loop
   * MazeMovement.#buildVisionAndDistance(state, encodedMaze, exitPos, distanceMap);
   */
  static #buildVisionAndDistance(
    state: SimulationState,
    encodedMaze: number[][],
    exitPos: readonly [number, number],
    distanceMap?: number[][]
  ) {
    // Early-exit when run already marked for termination.
    if (state.earlyTerminate) return;

    // Step 1: localize frequently-used values for clarity & perf
    const currentPosition = state.position;
    const posX = currentPosition[0] | 0;
    const posY = currentPosition[1] | 0;
    const hasPrecomputedDistances = state.hasDistanceMap;

    // Step 2: determine the "pre-move" distance used by the vision builder
    // - When a distance map exists, prefer the direct table lookup (may be undefined)
    // - Otherwise fall back to the unified distance accessor (fast, defensive)
    const preMoveDistance = hasPrecomputedDistances
      ? distanceMap![posY]?.[posX] ?? undefined
      : MazeMovement.#distanceAt(encodedMaze, currentPosition, distanceMap);

    // Step 3: build perception inputs. MazeVision.buildInputs6 is the canonical
    // builder; it accepts the previous-step distance and returns a plain JS array
    // (or a typed-array-compatible structure). We keep the result as-is to avoid
    // double-copying; callers expect `state.vision` to be a regular array of numbers.
    // NOTE: MazeVision may internally reuse pools — prefer that over forcing a copy here.
    const visionInputs = MazeVision.buildInputs6(
      encodedMaze,
      currentPosition,
      exitPos,
      distanceMap,
      MazeMovement.#StatePrevDistanceStep,
      preMoveDistance,
      state.prevAction
    );

    // Step 4: store results into simulation state. We intentionally assign the
    // builder's result directly to avoid an extra allocation; if MazeVision
    // returns a typed array the activation code should accept it — this keeps
    // hot-path overhead minimal. If you later observe mutation issues, convert
    // to a defensive copy here.
    state.vision = (Array.isArray(visionInputs)
      ? visionInputs
      : Array.from(visionInputs as Iterable<number>)) as number[];

    // Step 5: update the rolling previous-distance and the current-cell distance
    // used by proximity / epsilon logic. Use the cached distance map when present
    // otherwise compute via #distanceAt which is defensive and fast for small inputs.
    MazeMovement.#StatePrevDistanceStep = preMoveDistance;
    state.distHere = hasPrecomputedDistances
      ? distanceMap![posY]?.[posX] ?? Infinity
      : MazeMovement.#distanceAt(encodedMaze, currentPosition, distanceMap);
  }

  /**
   * Activate the neural network, record its outputs for history, choose an action
   * using the pooled softmax path, and update saturation/bias diagnostics.
   *
   * Behaviour & rationale:
   * - Keeps hot-path allocation minimal: we avoid creating unnecessary temporaries
   *   used only by downstream selection logic. `MazeMovement.selectDirection`
   *   accepts typed-arrays and reuses pooled scratch buffers internally so we
   *   pass the raw outputs directly for selection.
   * - `MazeUtils.pushHistory` requires a plain JS Array for correct bounded
   *   history semantics; we therefore make a single explicit shallow copy sized
   *   to the action count to record the outputs. This copy is the only
   *   unavoidable allocation required to preserve historical state safely.
   *
   * Steps (inline):
   * 1) Early-exit if the run is flagged `earlyTerminate`.
   * 2) Activate the network to receive raw outputs (logits).
   * 3) Copy the outputs into a fresh, fixed-length JS Array and push into the
   *    network's `_lastStepOutputs` history (bounded by `#OUTPUT_HISTORY_LENGTH`).
   * 4) Call `selectDirection(outputs)` which uses pooled scratch buffers to
   *    compute a numerically-stable softmax and returns argmax + diagnostics.
   * 5) Apply saturation and bias adjustments and store the chosen direction on
   *    the simulation `state`.
   *
   * @param state - simulation state (mutated in-place)
   * @param network - neural network implementing `activate(vision): number[]`
   * @param encodedMaze - read-only maze grid (kept for symmetry with callers)
   * @param distanceMap - optional precomputed distance map aligned to the maze
   * @returns void
   *
   * @example
   * // inside the simulation loop
   * MazeMovement.#decideDirection(state, network, encodedMaze, distanceMap);
   */
  static #decideDirection(
    state: SimulationState,
    network: INetwork,
    encodedMaze: number[][],
    distanceMap?: number[][]
  ) {
    // Step 1: fast-path bail when run flagged for early termination
    if (state.earlyTerminate) return;

    try {
      // Step 2: activate the network to obtain raw outputs (logits). We keep
      // the reference as-is because `selectDirection` can operate on typed
      // arrays and internally uses pooled scratch buffers for softmax.
      const networkOutputs = network.activate(state.vision) as number[];

      // Step 3: record a shallow, fixed-length plain-Array copy into the
      // network's history. `MazeUtils.pushHistory` expects Array semantics so
      // we must supply a real Array; create it deterministically sized to the
      // action count to avoid intermediate temporaries like spread operators.
      const outputsLength = (networkOutputs && networkOutputs.length) | 0;
      const outputsHistoryCopy: number[] = new Array(outputsLength);
      for (let copyIndex = 0; copyIndex < outputsLength; copyIndex++) {
        outputsHistoryCopy[copyIndex] = networkOutputs[copyIndex];
      }
      (network as any)._lastStepOutputs = MazeUtils.pushHistory(
        (network as any)._lastStepOutputs,
        outputsHistoryCopy,
        MazeMovement.#OUTPUT_HISTORY_LENGTH
      );

      // Step 4: select action using pooled softmax / scratch buffers.
      const selectedActionStats = MazeMovement.selectDirection(networkOutputs);
      state.actionStats = selectedActionStats;

      // Step 5: apply saturation/bias adjustments (may mutate network internals)
      MazeMovement.#applySaturationAndBiasAdjust(
        state,
        networkOutputs,
        network
      );

      // Finalize: store chosen direction on the simulation state
      state.direction = selectedActionStats.direction;
    } catch (error) {
      // Defensive: keep behaviour identical to previous implementation
      console.error('Error activating network:', error);
      state.direction = MazeMovement.#NO_MOVE;
    }
  }

  /**
   * Proximity greedy override: when the agent is within a configured
   * proximity to the exit prefer the immediate neighbor that minimises
   * the distance-to-exit (ties favour the current chosen direction).
   *
   * Behaviour & rationale:
   * - This is a deterministic short-circuit: when close to the goal we bias
   *   the policy to a local greedy choice to avoid aimless dithering.
   * - Uses pooled scratch (`#COORD_SCRATCH`) for temporary integer coords to
   *   avoid creating short-lived boxed numbers in hot loops.
   *
   * Steps:
   * 1) Early-exit when run marked for termination.
   * 2) When within `#PROXIMITY_GREEDY_DISTANCE` evaluate each neighbour.
   * 3) Skip invalid moves and compute neighbor distance via `#distanceAt`.
   * 4) Keep the neighbour with the smallest distance and assign it into
   *    `state.direction` if a better candidate is found.
   *
   * @param state - simulation state (modified in-place)
   * @param encodedMaze - read-only maze grid for move validity checks
   * @param distanceMap - optional precomputed distance map
   * @example
   * // inside the simulation loop
   * MazeMovement.#maybeApplyProximityGreedy(state, encodedMaze, distanceMap);
   */
  static #maybeApplyProximityGreedy(
    state: SimulationState,
    encodedMaze: number[][],
    distanceMap?: number[][]
  ) {
    // Step 1: guard
    if (state.earlyTerminate) return;

    // Only apply greedy override when agent is sufficiently close to the exit
    if (state.distHere > MazeMovement.#PROXIMITY_GREEDY_DISTANCE) return;

    // Step 2: evaluate neighbours and pick locally-minimal distance
    let chosenDirection = state.direction;
    let minimalNeighborDistance = Infinity;

    // Local alias to pooled coord scratch to avoid boxed temporaries
    const coordScratch = MazeMovement.#COORD_SCRATCH;

    for (
      let directionIndex = 0;
      directionIndex < MazeMovement.#ACTION_DIM;
      directionIndex++
    ) {
      const [deltaX, deltaY] = MazeMovement.#DIRECTION_DELTAS[directionIndex];

      // compute neighbour coordinates using integer arithmetic
      const neighbourX = (state.position[0] + deltaX) | 0;
      const neighbourY = (state.position[1] + deltaY) | 0;

      // write into pooled scratch (documents intent and may help some engines)
      coordScratch[0] = neighbourX;
      coordScratch[1] = neighbourY;

      // Step 3: skip invalid moves quickly
      if (!MazeMovement.isValidMove(encodedMaze, neighbourX, neighbourY))
        continue;

      // Step 4: get the distance for this neighbour; prefer provided map when present
      const neighbourDistance = MazeMovement.#distanceAt(
        encodedMaze,
        [neighbourX, neighbourY],
        distanceMap
      );

      // Keep the best (smallest) neighbour distance
      if (neighbourDistance < minimalNeighborDistance) {
        minimalNeighborDistance = neighbourDistance;
        chosenDirection = directionIndex;
      }
    }

    // Assign chosen direction back to state (preserves previous when none found)
    if (chosenDirection !== undefined && chosenDirection !== state.direction) {
      state.direction = chosenDirection;
    }
  }

  /**
   * Epsilon-greedy exploration override.
   *
   * Behaviour:
   * - Occasionally (probability `epsilon`) choose a random neighbouring
   *   valid action to encourage exploration. The helper prefers moves that are
   *   not the immediate previous action to reduce trivial back-and-forth.
   * - Uses pooled scratch storage and cached locals to keep the hot loop
   *   allocation-free and reduce property loads.
   *
   * Steps:
   * 1) Early-exit when the run is flagged `earlyTerminate`.
   * 2) Compute the adaptive epsilon via `#computeEpsilon`.
   * 3) With probability `epsilon` try up to `#ACTION_DIM` random candidate
   *    directions, skipping the previous action.
   * 4) For each candidate, test move validity and accept the first valid one.
   *
   * @param state - simulation state (mutated in-place)
   * @param encodedMaze - read-only maze used for move validity checks
   * @example
   * MazeMovement.#maybeApplyEpsilonExploration(state, encodedMaze);
   */
  static #maybeApplyEpsilonExploration(
    state: SimulationState,
    encodedMaze: number[][]
  ) {
    // Step 1: guard
    if (state.earlyTerminate) return;

    // Step 2: adaptive epsilon (small, often zero)
    const epsilon = MazeMovement.#computeEpsilon(
      state.steps,
      state.stepsSinceImprovement,
      state.distHere,
      MazeMovement.#StateSaturations
    );

    // Fast-path: only run the random trials when exploration is triggered
    if (!(MazeMovement.#rand() < epsilon)) return;

    // Cache locals for fewer property loads in the hot loop
    const actionCount = MazeMovement.#ACTION_DIM;
    const currentPrevAction = state.prevAction;
    const currentPosX = state.position[0] | 0;
    const currentPosY = state.position[1] | 0;
    const coordScratch = MazeMovement.#COORD_SCRATCH;

    // Step 3: attempt up to `actionCount` random candidate directions
    for (let attempt = 0; attempt < actionCount; attempt++) {
      // integer random selection without temporary arrays
      const randomDirection = (MazeMovement.#rand() * actionCount) | 0;
      if (randomDirection === currentPrevAction) continue; // prefer change

      const [directionDeltaX, directionDeltaY] = MazeMovement.#DIRECTION_DELTAS[
        randomDirection
      ];

      // compute candidate target coordinates (coerced to 32-bit ints)
      const candidateX = (currentPosX + directionDeltaX) | 0;
      const candidateY = (currentPosY + directionDeltaY) | 0;

      // write into pooled scratch (no functional dependency but documents intent)
      coordScratch[0] = candidateX;
      coordScratch[1] = candidateY;

      // Step 4: accept the first valid move
      if (MazeMovement.isValidMove(encodedMaze, candidateX, candidateY)) {
        state.direction = randomDirection;
        break;
      }
    }
  }

  /**
   * Force exploration when the agent has been unable to move for a while.
   *
   * Behaviour & rationale:
   * - Tracks a streak of `#NO_MOVE` decisions and when the configured
   *   threshold is reached chooses a random valid neighbour to escape
   *   potential deadlocks.
   * - Uses pooled scratch (`#COORD_SCRATCH`) and cached locals to reduce
   *   allocations and repeated property lookups in the hot loop.
   *
   * Steps:
   * 1) Early-exit when the run is already marked for termination.
   * 2) Maintain the global no-move streak counter (`#StateNoMoveStreak`).
   * 3) When the threshold is exceeded, try up to `#ACTION_DIM` random
   *    candidate directions and pick the first valid neighbour.
   * 4) Reset the no-move streak counter after forcing exploration.
   *
   * @param state - simulation state (mutated in-place)
   * @param encodedMaze - read-only maze used for move validity tests
   * @example
   * // inside simulation loop to recover from stuck states
   * MazeMovement.#maybeForceExploration(state, encodedMaze);
   */
  static #maybeForceExploration(
    state: SimulationState,
    encodedMaze: number[][]
  ) {
    // Step 1: guard
    if (state.earlyTerminate) return;

    // Step 2: update the rolling no-move streak counter
    if (state.direction === MazeMovement.#NO_MOVE) {
      MazeMovement.#StateNoMoveStreak++;
    } else {
      MazeMovement.#StateNoMoveStreak = 0;
    }

    // Only trigger forced exploration when the configured threshold is reached
    if (
      MazeMovement.#StateNoMoveStreak < MazeMovement.#NO_MOVE_STREAK_THRESHOLD
    )
      return;

    // Cache locals for speed in the hot loop
    const actionCount = MazeMovement.#ACTION_DIM;
    const currentPosX = state.position[0] | 0;
    const currentPosY = state.position[1] | 0;
    const coordScratch = MazeMovement.#COORD_SCRATCH;

    // Step 3: try up to `actionCount` random candidate directions
    for (let attemptIndex = 0; attemptIndex < actionCount; attemptIndex++) {
      // integer random selection (faster than Math.floor in tight loops)
      const candidateDirection = (MazeMovement.#rand() * actionCount) | 0;
      const [deltaX, deltaY] = MazeMovement.#DIRECTION_DELTAS[
        candidateDirection
      ];

      // compute candidate coordinates
      const candidateX = (currentPosX + deltaX) | 0;
      const candidateY = (currentPosY + deltaY) | 0;
      coordScratch[0] = candidateX;
      coordScratch[1] = candidateY;

      if (MazeMovement.isValidMove(encodedMaze, candidateX, candidateY)) {
        state.direction = candidateDirection;
        break;
      }
    }

    // Step 4: reset the global no-move streak counter after forcing exploration
    MazeMovement.#StateNoMoveStreak = 0;
  }

  /**
   * Execute the currently selected move (if valid) and update all
   * progress/exploration rewards and local penalties.
   *
   * Behavioural contract:
   * - Reads `state.direction` and attempts to move the agent by the
   *   matching delta from `#DIRECTION_DELTAS` when the action is valid.
   * - Updates `state.prevDistance`, `state.moved`, `state.pathLength`,
   *   `state.minDistanceToExit` and the various reward/penalty fields.
   * - Reuses pooled buffers (e.g. `#COORD_SCRATCH`, `#PathX`, `#PathY`) to
   *   avoid per-step allocations and keep the hot path allocation-free.
   *
   * Steps (step-level comments are present in the implementation):
   * 1) Early-exit if the run is already marked for termination.
   * 2) Record the pre-move distance into `state.prevDistance`.
   * 3) Compute the candidate target coordinates using pooled scratch.
   * 4) If the candidate cell is valid, update `state.position` and mark
   *    `state.moved = true`.
   * 5) When moved: append to the pooled path buffers, update local-area
   *    penalties, compute distance delta and apply progress/exploration
   *    shaping.
   * 6) When not moved: apply a mild invalid-move penalty.
   * 7) Apply the global distance-improvement bonus (separate helper).
   *
   * @param state - simulation state (mutated in-place)
   * @param encodedMaze - read-only 2D maze array
   * @param distanceMap - optional precomputed distance map aligned to maze
   * @example
   * // Typical usage inside the simulation loop
   * MazeMovement.#executeMoveAndRewards(state, encodedMaze, distanceMap);
   */
  static #executeMoveAndRewards(
    state: SimulationState,
    encodedMaze: number[][],
    distanceMap?: number[][]
  ) {
    // Step 1: early-exit when run already slated for termination
    if (state.earlyTerminate) return;

    // Step 2: capture pre-move distance for shaping calculations
    const previousDistance = MazeMovement.#distanceAt(
      encodedMaze,
      state.position,
      distanceMap
    );
    state.prevDistance = previousDistance;

    // Step 3: attempt to move using pooled direction deltas and coord scratch
    state.moved = false;
    const chosenAction = state.direction;
    if (chosenAction >= 0 && chosenAction < MazeMovement.#ACTION_DIM) {
      const [deltaX, deltaY] = MazeMovement.#DIRECTION_DELTAS[chosenAction];

      // Compute candidate coordinates (coerce to 32-bit ints) and reuse scratch
      const candidateX = (state.position[0] + deltaX) | 0;
      const candidateY = (state.position[1] + deltaY) | 0;
      const coordScratch = MazeMovement.#COORD_SCRATCH;
      coordScratch[0] = candidateX;
      coordScratch[1] = candidateY;

      // Validate the target cell and commit the move if valid
      if (MazeMovement.isValidMove(encodedMaze, candidateX, candidateY)) {
        state.position[0] = candidateX;
        state.position[1] = candidateY;
        state.moved = true;
      }
    }

    // Step 4: bookkeeping and reward/penalty updates
    const rewardScale = MazeMovement.#REWARD_SCALE;
    const pooledPathX = MazeMovement.#PathX!;
    const pooledPathY = MazeMovement.#PathY!;

    if (state.moved) {
      // Append the new position into the pooled path buffers
      const writeIndex = state.pathLength | 0;
      pooledPathX[writeIndex] = state.position[0];
      pooledPathY[writeIndex] = state.position[1];
      state.pathLength = writeIndex + 1;

      // Track recent local positions using the utility pushHistory (mutates in-place)
      MazeUtils.pushHistory(
        state.recentPositions,
        [state.position[0], state.position[1]] as [number, number],
        MazeMovement.#LOCAL_WINDOW
      );

      // Local-area stagnation penalty application (may mutate state)
      MazeMovement.#maybeApplyLocalAreaPenalty(state, rewardScale);

      // Resolve the post-move distance using precomputed map when available
      const currentDistance = state.hasDistanceMap
        ? state.distanceMap?.[state.position[1]]?.[state.position[0]] ??
          Infinity
        : MazeMovement.#distanceAt(
            encodedMaze,
            state.position,
            state.distanceMap
          );

      // Compute improvement/worsening and apply progress shaping
      const distanceDelta = previousDistance - currentDistance; // positive -> improvement
      const improved = distanceDelta > 0;
      const worsened = !improved && currentDistance > previousDistance;
      MazeMovement.#applyProgressShaping(
        state,
        distanceDelta,
        improved,
        worsened,
        rewardScale
      );

      // Exploration and revisit adjustments for the just-visited cell
      MazeMovement.#applyExplorationVisitAdjustment(state, rewardScale);

      // Update direction statistics & best-seen distance
      if (state.direction >= 0) state.directionCounts[state.direction]++;
      state.minDistanceToExit = Math.min(
        state.minDistanceToExit,
        currentDistance
      );
    } else {
      // Mild invalid-move penalty when the agent attempted an invalid move
      state.invalidMovePenalty -=
        MazeMovement.#INVALID_MOVE_PENALTY_MILD * rewardScale;
    }

    // Step 5: apply global distance-improvement bonus (may mutate state)
    MazeMovement.#applyGlobalDistanceImprovementBonus(
      state,
      encodedMaze,
      rewardScale
    );

    // Note: repetition/backtrack penalties and prevAction update are applied
    // later in the post-action penalties stage (#applyPostActionPenalties).
  }

  /**
   * Finalize per-step penalties after an action has been executed.
   *
   * Responsibilities:
   * - Apply repetition and backtrack penalties that depend on previous action
   *   and stagnation counters.
   * - Update the `prevAction` when a movement occurred.
   * - Apply entropy-based guidance shaping and periodic saturation penalties.
   * - Aggregate earlier-computed local penalties (loop/memory/revisit) into
   *   the run's `invalidMovePenalty` accumulator.
   *
   * Implementation notes:
   * - Uses pooled scratch storage (`#COORD_SCRATCH`) for a tiny, allocation-free
   *   temporary accumulator. The scratch is short-lived and reused across hot
   *   paths to minimise GC pressure.
   * - Variable names are intentionally descriptive to aid readability in hot
   *   loops and profiling traces.
   *
   * @param state - simulation state object mutated in-place
   * @returns void
   * @example
   * // call after moving/deciding action to finalize penalties for the step
   * MazeMovement.#applyPostActionPenalties(state);
   */
  static #applyPostActionPenalties(state: SimulationState) {
    // Step 1: fast-path guard — do nothing when run already flagged for termination
    if (state.earlyTerminate) return;

    // Local alias for the global reward/scale constant used by lower-level helpers
    const scale = MazeMovement.#REWARD_SCALE;

    // Step 2: apply repetition & backtrack penalties (may mutate state.invalidMovePenalty)
    MazeMovement.#applyRepetitionAndBacktrackPenalties(state, scale);

    // Step 3: update prevAction only when a movement actually happened
    if (state.moved) state.prevAction = state.direction;

    // Step 4: entropy-guidance shaping adjusts bonuses/penalties based on
    // the network's confidence and available perceptual cues
    MazeMovement.#applyEntropyGuidanceShaping(state, scale);

    // Step 5: periodic saturation escalation penalties
    MazeMovement.#applySaturationPenaltyCycle(state, scale);

    // Step 6: aggregate small, earlier-computed local penalties (loop/memory/revisit)
    // Use a tiny pooled scratch to avoid creating a transient Number object.
    const coordScratch = MazeMovement.#COORD_SCRATCH;
    // store aggregated penalty temporarily in scratch[0]
    coordScratch[0] =
      (state.loopPenalty || 0) +
      (state.memoryPenalty || 0) +
      (state.revisitPenalty || 0);
    // fold aggregated penalty into the global invalid-move accumulator
    state.invalidMovePenalty += coordScratch[0];
  }

  /**
   * Apply a local-area stagnation penalty when the agent is oscillating
   * within a small window without making progress.
   *
   * Behaviour:
   * - Examines the fixed-size `state.recentPositions` window and computes
   *   the bounding box (min/max X and Y). If the bounding box span is
   *   small and the run has been stagnant for configured steps, apply a
   *   local-area penalty to discourage dithering.
   * - Uses an existing pooled scratch (`#COORD_SCRATCH`) as a tiny,
   *   allocation-free temporary to reduce GC pressure in hot loops.
   *
   * Steps:
   * 1) Fast-path: ensure we have the full `#LOCAL_WINDOW` of recent positions.
   * 2) Iterate the recent positions to compute min/max X/Y using integer
   *    arithmetic for speed.
   * 3) Compute a simple span metric and apply the penalty when thresholds
   *    are exceeded.
   *
   * @param state - simulation state mutated in-place
   * @param rewardScale - global reward scale applied to penalty magnitudes
   * @example
   * // called after moving to decide if a local-area penalty is warranted
   * MazeMovement.#maybeApplyLocalAreaPenalty(state, MazeMovement.#REWARD_SCALE);
   */
  static #maybeApplyLocalAreaPenalty(
    state: SimulationState,
    rewardScale: number
  ) {
    // Step 1: require the full local history window to compute meaningful span
    const recentWindow = state.recentPositions;
    if (recentWindow.length !== MazeMovement.#LOCAL_WINDOW) return;

    // Step 2: compute bounding box using integer-coerced coordinates
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    // Use a simple index loop for faster iteration in some engines
    for (let idx = 0, len = recentWindow.length; idx < len; idx++) {
      const pair = recentWindow[idx];
      const rx = pair[0] | 0;
      const ry = pair[1] | 0;
      if (rx < minX) minX = rx;
      if (rx > maxX) maxX = rx;
      if (ry < minY) minY = ry;
      if (ry > maxY) maxY = ry;
    }

    // Small allocation-free write into pooled scratch to keep values live in a
    // typed-array for consumers or debuggers that prefer seeing typed storage.
    const coordScratch = MazeMovement.#COORD_SCRATCH;
    coordScratch[0] = minX;
    coordScratch[1] = minY;

    // Step 3: compute span metric and apply penalty if agent is stuck locally
    const span = maxX - minX + (maxY - minY);
    if (
      span <= MazeMovement.#LOCAL_AREA_SPAN_THRESHOLD &&
      state.stepsSinceImprovement > MazeMovement.#LOCAL_AREA_STAGNATION_STEPS
    ) {
      state.localAreaPenalty -=
        MazeMovement.#LOCAL_AREA_PENALTY_AMOUNT * rewardScale;
    }
  }

  /**
   * Apply shaping rewards/penalties based on the change in distance-to-goal.
   *
   * Behaviour:
   * - When the agent improved (distance decreased) grant progressive rewards
   *   scaled by confidence and stagnation duration.
   * - When the agent worsened (distance increased) apply a penalty scaled by
   *   confidence.
   * - When there is no change, increment the stagnation counter.
   *
   * Steps:
   * 1) Read confidence from `state.actionStats.maxProb` with a sensible default.
   * 2) When improved: apply step-based bonus, a base progress reward and a
   *    distance-delta contribution that is confidence-weighted.
   * 3) When worsened: apply an away penalty and increment the stagnation counter.
   * 4) When unchanged: increment `stepsSinceImprovement`.
   *
   * @param state - simulation state (mutated in-place)
   * @param distanceDelta - positive when the agent moved closer to goal
   * @param improved - boolean indicating whether distanceDelta > 0
   * @param worsened - boolean indicating whether distance increased
   * @param rewardScale - global reward scaling constant
   * @example
   * MazeMovement.#applyProgressShaping(state, prevDist - currDist, improved, worsened, MazeMovement.#REWARD_SCALE);
   */
  static #applyProgressShaping(
    state: SimulationState,
    distanceDelta: number,
    improved: boolean,
    worsened: boolean,
    rewardScale: number
  ) {
    // Step 1: derive confidence from last action statistics (fallbacks chosen
    // to preserve previous semantics used by the original implementation).
    const currentConfidence =
      state.actionStats?.maxProb ?? (improved ? 1 : 0.5);

    if (improved) {
      // Step 2.a: compute the base progress reward influenced by confidence
      const confidenceScaledBase =
        (MazeMovement.#PROGRESS_REWARD_BASE +
          MazeMovement.#PROGRESS_REWARD_CONF_SCALE * currentConfidence) *
        rewardScale;

      // Step 2.b: grant an additional warmup bonus proportional to how long
      // the agent has been without improvement (clamped by a configured max)
      if (state.stepsSinceImprovement > 0) {
        const stepBonus = Math.min(
          state.stepsSinceImprovement *
            MazeMovement.#PROGRESS_STEPS_MULT *
            rewardScale,
          MazeMovement.#PROGRESS_STEPS_MAX * rewardScale
        );
        state.progressReward += stepBonus;
      }

      // Apply the primary base progress reward and reset stagnation counter
      state.progressReward += confidenceScaledBase;
      state.stepsSinceImprovement = 0;

      // Step 2.c: distance-delta contribution scaled by confidence factors
      const distanceContribution =
        distanceDelta *
        MazeMovement.#DISTANCE_DELTA_SCALE *
        (MazeMovement.#DISTANCE_DELTA_CONF_BASE +
          MazeMovement.#DISTANCE_DELTA_CONF_SCALE * currentConfidence);
      state.progressReward += distanceContribution;
    } else if (worsened) {
      // Step 3: moving away from goal -> apply a penalty influenced by confidence
      const awayPenalty =
        (MazeMovement.#PROGRESS_AWAY_BASE_PENALTY +
          MazeMovement.#PROGRESS_AWAY_CONF_SCALE * currentConfidence) *
        rewardScale;
      state.progressReward -= awayPenalty;
      state.stepsSinceImprovement++;
    } else {
      // Step 4: no distance change -> increment stagnation counter
      state.stepsSinceImprovement++;
    }
  }

  /**
   * Apply exploration bonuses or revisit penalties for the cell that was
   * just visited.
   *
   * Behaviour:
   * - If a cell was visited for the first time in the run, award a
   *   `NEW_CELL_EXPLORATION_BONUS` scaled by `rewardScale`.
   * - If the cell has been visited before, apply a revisit penalty to
   *   discourage repetitive revisits to the same tile.
   *
   * Steps:
   * 1) Read the visit count for the current cell from `state.visitsAtCurrent`.
   * 2) Compute the adjustment (bonus or penalty) using the configured
   *    constants and `rewardScale`.
   * 3) Apply the adjustment to `state.newCellExplorationBonus` using a
   *    tiny pooled scratch (`#COORD_SCRATCH`) to avoid creating a transient
   *    Number wrapper on hot paths.
   *
   * @param state - simulation state mutated in-place
   * @param rewardScale - global reward scaling constant used to scale magnitudes
   * @example
   * MazeMovement.#applyExplorationVisitAdjustment(state, MazeMovement.#REWARD_SCALE);
   */
  static #applyExplorationVisitAdjustment(
    state: SimulationState,
    rewardScale: number
  ) {
    // Step 1: cache the visit count as a 32-bit integer for consistent semantics
    const visitsAtThisCell = state.visitsAtCurrent | 0;

    // Step 2: compute adjustment amount using named constants for clarity
    const positiveBonus =
      MazeMovement.#NEW_CELL_EXPLORATION_BONUS * rewardScale;
    const revisitPenalty = MazeMovement.#REVISIT_PENALTY_STRONG * rewardScale;

    // Step 3: use pooled scratch to hold the computed adjustment (allocation-free)
    const scratch = MazeMovement.#COORD_SCRATCH;
    scratch[0] = visitsAtThisCell === 1 ? positiveBonus : -revisitPenalty;

    // Apply the adjustment to the state's exploration bonus accumulator
    state.newCellExplorationBonus += scratch[0];
  }

  /**
   * Global distance-improvement bonus.
   *
   * Purpose:
   * - When the run breaks a long stagnation by improving the global
   *   distance-to-exit, grant a capped, step-scaled bonus to
   *   `state.progressReward` to encourage escapes from local minima.
   *
   * Behaviour / steps (inlined and commented):
   * 1) Resolve the current global distance-to-exit (prefer precomputed map).
   * 2) If the current global distance strictly improved over the last
   *    recorded global distance, compute a scaled bonus based on how many
   *    steps the agent had been without improvement and apply it (capped).
   * 3) Reset the run's `stepsSinceImprovement` when an improvement occurs.
   * 4) Store the current distance as `lastDistanceGlobal` for the next step.
   *
   * Notes:
   * - Uses the pooled `#COORD_SCRATCH` buffer for a tiny, allocation-free
   *   temporary storage to reduce GC pressure in hot loops.
   * - Local variable names are intentionally descriptive for readability.
   *
   * @param state - Mutable simulation state for the current run.
   * @param encodedMaze - Readonly maze grid (rows of numeric columns).
   * @param rewardScale - Global scalar applied to reward magnitudes.
   * @example
   * // Called from the move-execution path to potentially reward breaking
   * // prolonged stagnation when the agent finally decreases its global
   * // distance-to-exit.
   * MazeMovement.#applyGlobalDistanceImprovementBonus(state, maze, 1.0);
   */
  static #applyGlobalDistanceImprovementBonus(
    state: SimulationState,
    encodedMaze: number[][],
    rewardScale: number
  ) {
    // Step 1: fast-path locals & pooled scratch to minimise property loads
    const coordScratch = MazeMovement.#COORD_SCRATCH;

    // Resolve current global distance; prefer precomputed distance map when present.
    const posX = state.position[0] | 0;
    const posY = state.position[1] | 0;
    const currentGlobalDistance = state.hasDistanceMap
      ? state.distanceMap?.[posY]?.[posX] ?? Infinity
      : MazeMovement.#distanceAt(
          encodedMaze,
          state.position,
          state.distanceMap
        );

    // Store into pooled scratch[0] (keeps a typed-slot live for debugging/inspect).
    coordScratch[0] = currentGlobalDistance as number;

    // Step 2: compare against the previously-seen global distance
    const previousGlobalDistance = state.lastDistanceGlobal ?? Infinity;
    if (currentGlobalDistance < previousGlobalDistance) {
      // Improvement detected: compute an improvement bonus when the run
      // had been stagnant for more than the configured threshold.
      const stagnationSteps = (state.stepsSinceImprovement | 0) as number;
      if (stagnationSteps > MazeMovement.#GLOBAL_BREAK_BONUS_START) {
        const bonusSteps =
          stagnationSteps - MazeMovement.#GLOBAL_BREAK_BONUS_START;
        const uncappedBonus =
          bonusSteps * MazeMovement.#GLOBAL_BREAK_BONUS_PER_STEP * rewardScale;
        const cappedBonus = Math.min(
          uncappedBonus,
          MazeMovement.#GLOBAL_BREAK_BONUS_CAP * rewardScale
        );
        // Apply the computed bonus to the progress reward accumulator.
        state.progressReward += cappedBonus;
      }

      // Step 3: reset stagnation counter because we just improved globally.
      state.stepsSinceImprovement = 0;
    }

    // Step 4: persist the current distance for the next comparison step.
    state.lastDistanceGlobal = currentGlobalDistance;
  }

  /**
   * Apply repetition and backtrack penalties.
   *
   * Purpose:
   * - Penalise repeated identical actions when the agent has been stagnant
   *   for longer than the configured repetition threshold.
   * - Penalise immediate backtrack moves (opposite of the previous action)
   *   when the agent is not currently improving.
   *
   * Steps (inline):
   * 1) Guard against early termination.
   * 2) If the agent repeated the same action and stagnation exceeded the
   *    configured start threshold, compute a scaled repetition penalty and
   *    fold it into `state.invalidMovePenalty`.
   * 3) If the agent moved directly back (opposite direction) and the run
   *    is stagnant, apply a fixed backtrack penalty.
   *
   * Notes:
   * - Uses the pooled `#COORD_SCRATCH` buffer for tiny temporary values to
   *   keep the hot path allocation-free and to avoid creating transient
   *   Number objects.
   *
   * @param state - Mutable simulation state for the current run.
   * @param rewardScale - Global scalar applied to penalty magnitudes.
   * @example
   * // Called during post-action penalty finalization
   * MazeMovement.#applyRepetitionAndBacktrackPenalties(state, MazeMovement.#REWARD_SCALE);
   */
  static #applyRepetitionAndBacktrackPenalties(
    state: SimulationState,
    rewardScale: number
  ) {
    // Step 1: fast-path guard
    if (state.earlyTerminate) return;

    // Local descriptive aliases (minimise repeated property loads)
    const previousAction = state.prevAction;
    const currentAction = state.direction;
    const stagnationSteps = state.stepsSinceImprovement | 0;

    // Pooled tiny scratch to hold temporary penalty values (allocation-free)
    const scratch = MazeMovement.#COORD_SCRATCH;

    // Step 2: repetition penalty — when repeating the same action for too long
    const repetitionStartThreshold = MazeMovement.#REPETITION_PENALTY_START;
    if (
      previousAction === currentAction &&
      stagnationSteps > repetitionStartThreshold
    ) {
      const repetitionMultiplier = stagnationSteps - repetitionStartThreshold;
      const repetitionBase = MazeMovement.#REPETITION_PENALTY_BASE;
      const computedRepetitionPenalty =
        repetitionBase * repetitionMultiplier * rewardScale;

      // store negative penalty in scratch[0] then fold into the accumulator
      scratch[0] = -computedRepetitionPenalty;
      state.invalidMovePenalty += scratch[0];
    }

    // Step 3: backtrack penalty — penalise immediate opposite-direction moves
    if (
      previousAction >= 0 &&
      currentAction >= 0 &&
      stagnationSteps > 0 &&
      currentAction === MazeMovement.#OPPOSITE_DIR[previousAction]
    ) {
      const backtrackPenalty = MazeMovement.#BACK_MOVE_PENALTY * rewardScale;
      scratch[1] = -backtrackPenalty;
      state.invalidMovePenalty += scratch[1];
    }
  }

  /**
   * Entropy-guided shaping: apply small penalties or exploration bonuses
   * based on the network's action entropy and whether perceptual guidance
   * (line-of-sight or gradient cues) is present.
   *
   * Behaviour / steps:
   * 1) Early-exit when there are no recorded action statistics.
   * 2) Compute whether the current perception provides guidance.
   * 3) If entropy is very high, apply a small penalty to discourage
   *    aimless, highly-uncertain behaviour.
   * 4) If perception provides guidance and the network is confident
   *    (low entropy and clear max-vs-second gap), award a small
   *    exploration bonus to encourage exploiting the useful cue.
   *
   * Implementation notes:
   * - Uses descriptive local names and the pooled `#COORD_SCRATCH` typed
   *   array for tiny temporaries to avoid transient allocation on hot paths.
   * - Preserves existing numeric thresholds and multipliers.
   *
   * @param state - Mutable simulation state for the current run.
   * @param rewardScale - Global scalar applied to penalty/bonus magnitudes.
   * @example
   * // Called as part of per-step penalty finalization
   * MazeMovement.#applyEntropyGuidanceShaping(state, MazeMovement.#REWARD_SCALE);
   */
  static #applyEntropyGuidanceShaping(
    state: SimulationState,
    rewardScale: number
  ) {
    // Step 1: require action stats
    if (state.earlyTerminate || !state.actionStats) return;

    // Local copies for clarity and fewer property loads
    const { entropy, maxProb, secondProb } = state.actionStats;
    const entropyHighThreshold = MazeMovement.#ENTROPY_HIGH_THRESHOLD;
    const entropyConfidentThreshold = MazeMovement.#ENTROPY_CONFIDENT_THRESHOLD;
    const confidentDiffThreshold = MazeMovement.#ENTROPY_CONFIDENT_DIFF;

    // Step 2: detect whether perceptual guidance exists (LOS or gradient cues)
    const hasLineOfSightGuidance =
      MazeMovement.#sumVisionGroup(
        state.vision,
        MazeMovement.#VISION_LOS_START
      ) > 0;
    const hasGradientGuidance =
      MazeMovement.#sumVisionGroup(
        state.vision,
        MazeMovement.#VISION_GRAD_START
      ) > 0;
    const hasGuidance = hasLineOfSightGuidance || hasGradientGuidance;

    // Pooled scratch for tiny temporary values (avoid boxed Number allocations)
    const scratch = MazeMovement.#COORD_SCRATCH;

    // Step 3: high-entropy penalty (discourage dithering/ambivalence)
    if (entropy > entropyHighThreshold) {
      scratch[0] = -MazeMovement.#ENTROPY_PENALTY * rewardScale;
      state.invalidMovePenalty += scratch[0];
      return; // high-entropy is dominant; bail early
    }

    // Step 4: confident + guided => small exploration bonus
    const maxMinusSecond = (maxProb ?? 0) - (secondProb ?? 0);
    if (
      hasGuidance &&
      entropy < entropyConfidentThreshold &&
      maxMinusSecond > confidentDiffThreshold
    ) {
      scratch[0] = MazeMovement.#EXPLORATION_BONUS_SMALL * rewardScale;
      state.newCellExplorationBonus += scratch[0];
    }
  }

  /**
   * Periodic saturation penalty cycle.
   *
   * Purpose:
   * - When the global saturation counter (`#StateSaturations`) exceeds a
   *   trigger, apply a base saturation penalty to discourage chronic
   *   overconfidence. On configured periods apply an additional escalate
   *   penalty to increase pressure over time.
   *
   * Behaviour / steps:
   * 1) Early-exit when saturations have not reached the configured trigger.
   * 2) Apply the base saturation penalty scaled by `rewardScale`.
   * 3) If the saturations counter aligns with the configured period, apply
   *    an extra escalate penalty (also scaled by `rewardScale`).
   *
   * Implementation notes:
   * - Uses the pooled `#COORD_SCRATCH` typed array as a tiny allocation-free
   *   temporary for computed penalty values to keep the hot path GC-friendly.
   * - Local descriptive names improve readability without changing logic.
   *
   * @param state - Mutable simulation state (penalties are accumulated here)
   * @param rewardScale - Global scalar used to scale penalty magnitudes
   * @example
   * MazeMovement.#applySaturationPenaltyCycle(state, MazeMovement.#REWARD_SCALE);
   */
  static #applySaturationPenaltyCycle(
    state: SimulationState,
    rewardScale: number
  ) {
    // Step 1: quick-exit when under the configured trigger
    const saturations = MazeMovement.#StateSaturations;
    const triggerThreshold = MazeMovement.#SATURATION_PENALTY_TRIGGER;
    if (saturations < triggerThreshold) return;

    // Pooled tiny scratch to hold negative penalty values (avoid boxed numbers)
    const scratch = MazeMovement.#COORD_SCRATCH;

    // Step 2: apply base saturation penalty (negative value folded into accumulator)
    const basePenalty = MazeMovement.#SATURATION_PENALTY_BASE * rewardScale;
    scratch[0] = -basePenalty;
    state.invalidMovePenalty += scratch[0];

    // Step 3: periodic escalation on configured period boundaries
    const period = MazeMovement.#SATURATION_PENALTY_PERIOD;
    if (period > 0 && saturations % period === 0) {
      const escalatePenalty =
        MazeMovement.#SATURATION_PENALTY_ESCALATE * rewardScale;
      scratch[1] = -escalatePenalty;
      state.invalidMovePenalty += scratch[1];
    }
  }

  /**
   * Detect saturation/overconfidence, apply shaping penalties, and
   * optionally perform adaptive output-node bias dampening.
   *
   * Behaviour / steps:
   * 1) Read action confidence statistics and decide whether the network is
   *    overconfident (sharp winner) or has flat logit collapse (low variance).
   * 2) Update a rolling `#StateSaturations` counter and the run-local
   *    `state.saturatedSteps` when either condition holds.
   * 3) Apply fixed penalties for overconfidence and flat collapse scaled by
   *    `rewardScale`.
   * 4) When chronic saturation persists, periodically adjust output-node
   *    biases to dampen runaway confidence (best-effort; errors are swallowed).
   *
   * Implementation notes:
   * - Uses descriptive local variables for readability and fewer property loads.
   * - Reuses the pooled `#COORD_SCRATCH` typed-array for tiny temporaries to
   *   avoid boxed Number allocations on hot paths.
   * - Preserves existing numeric thresholds and update semantics.
   *
   * @param state - Mutable simulation state for the current run.
   * @param outputs - Raw network logits for the current activation.
   * @param network - The neural network instance (used for optional bias adjust).
   * @example
   * MazeMovement.#applySaturationAndBiasAdjust(state, outputs, network);
   */
  static #applySaturationAndBiasAdjust(
    state: SimulationState,
    outputs: number[],
    network: INetwork
  ) {
    // Step 0: locals & pooled scratch
    const rewardScale = MazeMovement.#REWARD_SCALE;
    const scratch = MazeMovement.#COORD_SCRATCH;

    // Defensive: require actionStats to compute confidence; callers normally set this.
    const actionStats = state.actionStats;
    if (!actionStats) return;

    // Step 1: overconfidence detection (max probability vs second-best)
    const maxProbability = actionStats.maxProb ?? 0;
    const secondProbability = actionStats.secondProb ?? 0;
    const isOverConfident =
      maxProbability > MazeMovement.#OVERCONFIDENT_PROB &&
      secondProbability < MazeMovement.#SECOND_PROB_LOW;

    // Step 1b: detect flat collapse using logits variance (population std-dev)
    const actionCount = MazeMovement.#ACTION_DIM;
    // compute mean logit
    let sumLogits = 0;
    for (let i = 0; i < outputs.length; i++) sumLogits += outputs[i];
    const meanLogit = sumLogits / actionCount;

    // compute variance (avoid intermediate arrays)
    let varianceAccumulator = 0;
    for (let i = 0; i < outputs.length; i++) {
      const delta = outputs[i] - meanLogit;
      varianceAccumulator += delta * delta;
    }
    const variance = varianceAccumulator / actionCount;
    const stdDev = Math.sqrt(variance);
    const isFlatCollapsed = stdDev < MazeMovement.#LOGSTD_FLAT_THRESHOLD;

    // Step 2: update rolling saturation counter and saturated steps
    let saturationCounter = MazeMovement.#StateSaturations;
    if (isOverConfident || isFlatCollapsed) {
      saturationCounter++;
      state.saturatedSteps++;
    } else if (saturationCounter > 0) {
      saturationCounter--;
    }
    MazeMovement.#StateSaturations = saturationCounter;

    // Step 3: fold in penalties using pooled scratch to avoid boxed temporaries
    if (isOverConfident) {
      scratch[0] = -MazeMovement.#OVERCONFIDENT_PENALTY * rewardScale;
      state.invalidMovePenalty += scratch[0];
    }
    if (isFlatCollapsed) {
      scratch[0] = -MazeMovement.#FLAT_COLLAPSE_PENALTY * rewardScale;
      state.invalidMovePenalty += scratch[0];
    }

    // Step 4: adaptive bias dampening when chronic saturation persists
    const shouldAdjustBiases =
      MazeMovement.#StateSaturations > MazeMovement.#SATURATION_ADJUST_MIN &&
      state.steps % MazeMovement.#SATURATION_ADJUST_INTERVAL === 0;

    if (shouldAdjustBiases) {
      try {
        const outputNodes = (network as any).nodes?.filter(
          (node: any) => node.type === MazeMovement.#NODE_TYPE_OUTPUT
        );
        if (outputNodes && outputNodes.length > 0) {
          // compute mean bias (simple loop to avoid higher-order helpers)
          let biasSum = 0;
          for (let i = 0; i < outputNodes.length; i++)
            biasSum += outputNodes[i].bias;
          const meanBias = biasSum / outputNodes.length;

          // adjust each node bias towards zero by removing a scaled meanBias
          const adjustFactor = MazeMovement.#BIAS_ADJUST_FACTOR;
          const clamp = MazeMovement.#BIAS_CLAMP;
          for (let i = 0; i < outputNodes.length; i++) {
            const node = outputNodes[i];
            const adjusted = node.bias - meanBias * adjustFactor;
            // clamp adjusted bias into allowed range
            node.bias = Math.max(-clamp, Math.min(clamp, adjusted));
          }
        }
      } catch {
        // Best-effort: swallow errors (network shapes vary in tests)
      }
    }
  }

  /**
   * Check deep stagnation and optionally mark the run for termination.
   *
   * Purpose:
   * - If the run has been without improvement for longer than
   *   `#DEEP_STAGNATION_THRESHOLD` we may apply a deep-stagnation penalty
   *   and terminate the run in non-browser environments (node / CI). The
   *   method avoids allocations by reusing the pooled `#COORD_SCRATCH`.
   *
   * Steps:
   * 1) Fast-path: compare `state.stepsSinceImprovement` against the
   *    configured threshold.
   * 2) Detect whether we are running outside a browser (only then apply
   *    the penalty and return `true`).
   * 3) Use `#COORD_SCRATCH[0]` to hold the negative penalty (allocation-free)
   *    and fold it into `state.invalidMovePenalty`.
   * 4) Return `true` when we applied the penalty (indicating termination),
   *    otherwise preserve and return `state.earlyTerminate`.
   *
   * @param state - mutable simulation state (mutated in-place when penalty applies)
   * @returns boolean - `true` when the run should be terminated (penalty applied),
   *          otherwise the existing `state.earlyTerminate` value.
   * @example
   * // inside the simulation loop
   * if (MazeMovement.#maybeTerminateDeepStagnation(state)) break;
   */
  static #maybeTerminateDeepStagnation(state: SimulationState): boolean {
    // Step 1: quick guard using 32-bit coercion for stable comparisons
    const stagnationSteps = state.stepsSinceImprovement | 0;
    if (stagnationSteps <= MazeMovement.#DEEP_STAGNATION_THRESHOLD)
      return state.earlyTerminate;

    // Step 2: prepare locals and pooled scratch for allocation-free penalty write
    const rewardScale = MazeMovement.#REWARD_SCALE;
    const scratch = MazeMovement.#COORD_SCRATCH;

    // Step 3: apply penalty and request termination only when not running in a
    // browser environment (preserve original behaviour that avoids applying
    // the penalty when `window` exists). Keep a try/catch as a defensive
    // fallback in case environment detection throws in unusual hosts.
    try {
      const runningOutsideBrowser = typeof window === 'undefined';
      if (runningOutsideBrowser) {
        scratch[0] = -MazeMovement.#DEEP_STAGNATION_PENALTY * rewardScale;
        state.invalidMovePenalty += scratch[0];
        return true;
      }
    } catch {
      // Best-effort fallback: if detection failed, still apply the penalty.
      scratch[0] = -MazeMovement.#DEEP_STAGNATION_PENALTY * rewardScale;
      state.invalidMovePenalty += scratch[0];
      return true;
    }

    // Step 4: no change to termination state in browser-like hosts
    return state.earlyTerminate;
  }

  /**
   * Compute the normalized action entropy from recorded direction counts.
   *
   * Behaviour / rationale:
   * - Converts direction visit counts into a probability distribution and
   *   computes the Shannon entropy. The result is normalised by
   *   `#LOG_ACTIONS` so the returned value lies in a stable range used by
   *   the rest of the scoring heuristics.
   * - The implementation is allocation-free and uses the pooled
   *   `#COORD_SCRATCH` typed-array as a tiny scratch accumulator to avoid
   *   creating transient Number objects on hot paths.
   *
   * Steps:
   * 1) Sum the provided `directionCounts` and fall back to 1 to avoid
   *    division-by-zero.
   * 2) Iterate counts, skip zeros, compute per-action probability and
   *    accumulate -p * log(p) into a pooled accumulator.
   * 3) Normalise the accumulated entropy by `#LOG_ACTIONS` and return it.
   *
   * @param directionCounts - array of non-negative integers counting how often each action was chosen
   * @returns normalised entropy number used in fitness shaping
   * @example
   * const entropy = MazeMovement.#computeActionEntropyFromCounts(state.directionCounts);
   */
  static #computeActionEntropyFromCounts(directionCounts: number[]): number {
    // Step 1: sum counts (coerce to number) and avoid dividing by zero
    const totalCount =
      directionCounts.reduce((sum, value) => sum + (value | 0), 0) || 1;

    // Use pooled scratch to hold the running entropy accumulator (allocation-free)
    const scratch = MazeMovement.#COORD_SCRATCH;
    scratch[0] = 0;

    // Local alias for performance-sensitive globals
    const logFn = Math.log;

    // Step 2: accumulate entropy = -sum(p * log(p)) skipping zero-counts
    for (let i = 0, len = directionCounts.length; i < len; i++) {
      const count = directionCounts[i] | 0;
      if (count === 0) continue;
      const probability = count / totalCount;
      scratch[0] -= probability * logFn(probability);
    }

    // Step 3: normalise by the project's LOG_ACTIONS constant and return
    return scratch[0] / MazeMovement.#LOG_ACTIONS;
  }

  /**
   * Build and return the finalized result object for a successful run.
   *
   * Behaviour / rationale:
   * - Aggregates progress, exploration and penalty terms into a single
   *   fitness score. The final fitness is clamped by `#MIN_SUCCESS_FITNESS`.
   * - Returns a compact result object including steps, materialized path,
   *   a progress metric and a normalised action-entropy value used by scoring.
   * - Uses the pooled `#COORD_SCRATCH` for a tiny, allocation-free saturation
   *   fraction calculation to reduce transient allocations on hot code paths.
   *
   * Steps:
   * 1) Compute step efficiency (how many steps under the maximum were used).
   * 2) Compute action entropy from recorded direction counts.
   * 3) Aggregate fitness components (base, efficiency, rewards, penalties).
   * 4) Materialize the executed path and compute saturation fraction.
   * 5) Clamp the fitness to the configured minimum and return the result.
   *
   * @param state - simulation state containing run accumulators and diagnostics
   * @param maxSteps - configured maximum steps for the run (used to compute efficiency)
   * @returns result object describing success, steps, path, fitness, progress and diagnostics
   * @example
   * const result = MazeMovement.#finalizeSuccess(state, maxSteps);
   */
  static #finalizeSuccess(state: SimulationState, maxSteps: number) {
    // Step 1: compute steps and efficiency (coerce to 32-bit ints for stability)
    const stepsTaken = state.steps | 0;
    const stepEfficiency = (maxSteps | 0) - stepsTaken;

    // Step 2: entropy of the action distribution (normalised by #LOG_ACTIONS)
    const actionEntropy = MazeMovement.#computeActionEntropyFromCounts(
      state.directionCounts
    );

    // Step 3: aggregate fitness components using descriptive locals
    const baseFitness =
      MazeMovement.#SUCCESS_BASE_FITNESS +
      stepEfficiency * MazeMovement.#STEP_EFFICIENCY_SCALE +
      state.progressReward +
      state.newCellExplorationBonus +
      state.invalidMovePenalty;

    const totalFitness =
      baseFitness + actionEntropy * MazeMovement.#SUCCESS_ACTION_ENTROPY_SCALE;

    // Step 4: materialize the path and compute saturation fraction using pooled scratch
    const pathMaterialized = MazeMovement.#materializePath(state.pathLength);
    const scratch = MazeMovement.#COORD_SCRATCH;
    scratch[0] = stepsTaken ? state.saturatedSteps / stepsTaken : 0;
    const saturationFraction = scratch[0];

    // Step 5: ensure final fitness meets the configured minimum for successes
    const finalFitness = Math.max(
      MazeMovement.#MIN_SUCCESS_FITNESS,
      totalFitness
    );

    return {
      success: true,
      steps: stepsTaken,
      path: pathMaterialized,
      fitness: finalFitness,
      progress: 100,
      saturationFraction,
      actionEntropy,
    };
  }

  /**
   * Build and return the finalized result object for a failed run.
   *
   * Behaviour / rationale:
   * - Computes shaped progress, exploration contributions, entropy bonus and
   *   aggregates penalties into a single fitness score. For failures the
   *   fitness is transformed to avoid negative-heavy values using the same
   *   heuristic as the original implementation.
   * - Uses the pooled `#COORD_SCRATCH` for a tiny, allocation-free
   *   saturation fraction calculation.
   *
   * Steps:
   * 1) Materialize the executed path and determine the last visited cell.
   * 2) Compute progress (via distance map or geometry), then shape it.
   * 3) Aggregate exploration, reward and penalty contributions including
   *    an entropy-derived bonus.
   * 4) Mix in small random noise and transform negative raw fitness using
   *    the project's stabilizing mapping.
   * 5) Return the failure result object with diagnostics.
   *
   * @param state - simulation state containing run accumulators and diagnostics
   * @param encodedMaze - read-only maze grid (rows of numeric columns)
   * @param startPos - starting coordinate tuple [x, y]
   * @param exitPos - exit coordinate tuple [x, y]
   * @param distanceMap - optional precomputed distance map aligned to maze
   * @returns failure result object with fitness, path and diagnostics
   * @example
   * const result = MazeMovement.#finalizeFailure(state, maze, startPos, exitPos, distanceMap);
   */
  static #finalizeFailure(
    state: SimulationState,
    encodedMaze: number[][],
    startPos: readonly [number, number],
    exitPos: readonly [number, number],
    distanceMap?: number[][]
  ) {
    // Step 1: materialize path and compute last visited position
    const pathX = MazeMovement.#PathX!;
    const pathY = MazeMovement.#PathY!;
    const lastIndex = (state.pathLength | 0) - 1;
    const lastPos: [number, number] = [
      pathX[lastIndex] ?? 0,
      pathY[lastIndex] ?? 0,
    ];

    // Step 2: compute progress using an optional distance map or geometry
    const progress = distanceMap
      ? MazeUtils.calculateProgressFromDistanceMap(
          distanceMap,
          lastPos,
          startPos
        )
      : MazeUtils.calculateProgress(encodedMaze, lastPos, startPos, exitPos);
    const progressFraction = progress / 100;
    const shapedProgress =
      Math.pow(progressFraction, MazeMovement.#PROGRESS_POWER) *
      MazeMovement.#PROGRESS_SCALE;

    // Step 3: aggregate exploration and entropy-derived components
    const explorationScore = state.visitedUniqueCount * 1.0;
    const actionEntropy = MazeMovement.#computeActionEntropyFromCounts(
      state.directionCounts
    );
    const entropyBonus = actionEntropy * MazeMovement.#ENTROPY_BONUS_WEIGHT;

    // Placeholders for future heuristics (preserve original behaviour)
    const saturationPenalty = 0;
    const outputVarPenalty = 0;

    // Aggregate base fitness components
    const baseFitness =
      shapedProgress +
      explorationScore +
      state.progressReward +
      state.newCellExplorationBonus +
      state.invalidMovePenalty +
      entropyBonus +
      state.localAreaPenalty +
      saturationPenalty +
      outputVarPenalty;

    // Step 4: add a small random factor and stabilise negative values
    const raw =
      baseFitness + MazeMovement.#rand() * MazeMovement.#FITNESS_RANDOMNESS;
    const fitness = raw >= 0 ? raw : -Math.log1p(1 - raw);

    // Step 5: produce materialized path and saturation fraction (allocation-free)
    const pathMaterialized = MazeMovement.#materializePath(state.pathLength);
    const scratch = MazeMovement.#COORD_SCRATCH;
    const stepsTaken = state.steps | 0;
    scratch[0] = stepsTaken ? state.saturatedSteps / stepsTaken : 0;
    const saturationFraction = scratch[0];

    return {
      success: false,
      steps: state.steps,
      path: pathMaterialized,
      fitness,
      progress,
      saturationFraction,
      actionEntropy,
    };
  }
}
