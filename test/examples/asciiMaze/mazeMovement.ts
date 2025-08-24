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

/** Internal (non-exported) aggregate state for a single simulation run. */
interface SimulationState {
  position: [number, number];
  steps: number;
  pathLength: number;
  visitedUniqueCount: number;
  hasDistanceMap: boolean;
  distanceMap?: number[][];
  minDistanceToExit: number;
  progressReward: number;
  newCellExplorationBonus: number;
  invalidMovePenalty: number;
  prevAction: number;
  stepsSinceImprovement: number;
  lastDistanceGlobal: number;
  saturatedSteps: number;
  recentPositions: [number, number][];
  localAreaPenalty: number;
  directionCounts: number[];
  moveHistoryRing: Int32Array;
  moveHistoryLength: number;
  moveHistoryHead: number;
  currentCellIndex: number;
  loopPenalty: number;
  memoryPenalty: number;
  revisitPenalty: number;
  visitsAtCurrent: number;
  distHere: number;
  vision: number[];
  actionStats: any;
  direction: number;
  moved: boolean;
  prevDistance: number;
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

  /** Seedable PRNG state (Mulberry32 style). Undefined => Math.random(). */
  static #PRNGState: number | null = null;

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
   * Enable deterministic pseudo-randomness for simulations executed after this call.
   *
   * Uses a lightweight Mulberry32 generator so repeated runs with the same seed
   * produce identical stochastic choices (epsilon exploration, tie‑breaking, etc.).
   * Because internal buffers are shared, calls are not reentrant / thread‑safe.
   *
   * @param seed 32-bit unsigned integer seed. If 0 is provided a fixed default constant is used.
   * @returns void
   * @example
   * // Ensure reproducible simulation ordering
   * MazeMovement.seedDeterministic(1234);
   * const result = MazeMovement.simulateAgent(network, maze, start, exit);
   */
  static seedDeterministic(seed: number): void {
    MazeMovement.#PRNGState = seed >>> 0 || 0x9e3779b9;
  }

  /**
   * Disable deterministic seeding and return to Math.random based randomness.
   *
   * @returns void
   * @example
   * MazeMovement.clearDeterministicSeed();
   */
  static clearDeterministicSeed(): void {
    MazeMovement.#PRNGState = null;
  }

  /** Generate a random float in [0,1). Deterministic when seed set. */
  static #rand(): number {
    if (MazeMovement.#PRNGState == null) return Math.random();
    // Mulberry32
    let t = (MazeMovement.#PRNGState += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Encode (x,y) -> linear cell index. */
  static #index(x: number, y: number): number {
    return y * MazeMovement.#CachedWidth + x;
  }

  /** Ensure pooled buffers sized for given maze & path. */
  static #initBuffers(width: number, height: number, maxSteps: number) {
    const cellCount = width * height;
    // Grow grid buffers if needed (never shrink synchronously to preserve reuse)
    if (!this.#VisitedFlags || cellCount > this.#GridCapacity) {
      const nextCellCap = MazeMovement.#nextPow2(cellCount);
      this.#VisitedFlags = new Uint8Array(nextCellCap);
      this.#VisitCounts = new Uint16Array(nextCellCap);
      this.#GridCapacity = nextCellCap;
    } else {
      // Fast clear only required portion (rest retains old zeros or stale data not addressed)
      this.#VisitedFlags.fill(0, 0, cellCount);
      this.#VisitCounts!.fill(0, 0, cellCount);
    }
    // Grow path buffers
    if (!this.#PathX || maxSteps + 1 > this.#PathCapacity) {
      const nextPathCap = MazeMovement.#nextPow2(maxSteps + 1);
      this.#PathX = new Int32Array(nextPathCap);
      this.#PathY = new Int32Array(nextPathCap);
      this.#PathCapacity = nextPathCap;
    }
    this.#CachedWidth = width;
    this.#CachedHeight = height;
  }

  /** Next power of two helper (>= n). */
  static #nextPow2(n: number): number {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
  }

  /** Materialize the path arrays into an array<tuple>. */
  static #materializePath(length: number): [number, number][] {
    const px = MazeMovement.#PathX!;
    const py = MazeMovement.#PathY!;
    const out = new Array(length) as [number, number][];
    for (let positionIndex = 0; positionIndex < length; positionIndex++) {
      out[positionIndex] = [px[positionIndex], py[positionIndex]];
    }
    return out;
  }

  /** Return opposite cardinal direction (works for ACTION_DIM even if it changes) */
  static #opposite(direction: number): number {
    return (
      (direction + MazeMovement.#ACTION_DIM / 2) % MazeMovement.#ACTION_DIM
    );
  }

  /** Sum a contiguous group in the vision vector starting at `start`. */
  static #sumVisionGroup(vision: number[], start: number) {
    // Manual unrolled/loop sum to avoid intermediate slice allocation (hot path)
    let total = 0;
    const end = start + MazeMovement.#VISION_GROUP_LEN;
    for (let visionIndex = start; visionIndex < end; visionIndex++)
      total += vision[visionIndex];
    return total;
  }

  /**
   * Compute adaptive epsilon used for epsilon-greedy exploration.
   * Extracted from the main loop to shrink hot path complexity and enable reuse.
   */
  static #computeEpsilon(
    stepNumber: number,
    stepsSinceImprovement: number,
    distHere: number,
    saturations: number
  ): number {
    let epsilon = 0;
    switch (true) {
      case stepNumber < MazeMovement.#EPSILON_WARMUP_STEPS:
        epsilon = MazeMovement.#EPSILON_INITIAL;
        break;
      case stepsSinceImprovement >
        MazeMovement.#EPSILON_STAGNANT_HIGH_THRESHOLD:
        epsilon = MazeMovement.#EPSILON_STAGNANT_HIGH;
        break;
      case stepsSinceImprovement > MazeMovement.#EPSILON_STAGNANT_MED_THRESHOLD:
        epsilon = MazeMovement.#EPSILON_STAGNANT_MED;
        break;
      case saturations > MazeMovement.#EPSILON_SATURATION_TRIGGER:
        epsilon = MazeMovement.#EPSILON_SATURATIONS;
        break;
      default:
        break;
    }
    if (distHere <= MazeMovement.#PROXIMITY_SUPPRESS_EXPLOR_DIST)
      epsilon = Math.min(epsilon, MazeMovement.#EPSILON_MIN_NEAR_GOAL);
    return epsilon;
  }

  /**
   * Helper: is a cell (x,y) within bounds and not a wall?
   */
  static #isCellOpen(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    x: number,
    y: number
  ): boolean {
    return (
      y >= 0 &&
      y < encodedMaze.length &&
      x >= 0 &&
      x < encodedMaze[0].length &&
      encodedMaze[y][x] !== -1
    );
  }

  /**
   * Helper: unified distance lookup. Prefer distance map when available,
   * otherwise fall back to BFS. Keeps callers simple and avoids repeated
   * ternary expressions across the file.
   */
  static #distanceAt(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    [x, y]: readonly [number, number],
    distanceMap?: number[][]
  ): number {
    return Number.isFinite(distanceMap?.[y]?.[x])
      ? distanceMap![y][x]
      : Infinity;
  }

  /**
   * Checks if a move is valid (within maze bounds and not a wall cell).
   *
   * @param encodedMaze - 2D array representation of the maze (cells: -1=wall, 0+=open).
   * @param coords - [x, y] coordinates to check for validity.
   * @returns {boolean} True if the position is within bounds and not a wall.
   */
  static isValidMove(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    position: readonly [number, number]
  ): boolean;
  static isValidMove(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    x: number,
    y: number
  ): boolean;
  static isValidMove(
    encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
    positionOrX: any,
    yMaybe?: any
  ): boolean {
    if (Array.isArray(positionOrX)) {
      const [x, y] = positionOrX as [number, number];
      return MazeMovement.#isCellOpen(encodedMaze, x, y);
    }
    return MazeMovement.#isCellOpen(
      encodedMaze,
      positionOrX as number,
      yMaybe as number
    );
  }

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
   * Selects the direction with the highest output value from the neural network.
   * Applies softmax to interpret outputs as probabilities, then uses argmax.
   * Also computes entropy and confidence statistics for analysis.
   *
   * @param outputs - Array of output values from the neural network (length 4).
   * @returns {object} Direction index, softmax probabilities, entropy, and confidence stats.
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
    // Handle invalid or missing outputs
    if (!outputs || outputs.length !== MazeMovement.#ACTION_DIM) {
      return {
        direction: MazeMovement.#NO_MOVE,
        softmax: [0, 0, 0, 0],
        entropy: 0,
        maxProb: 0,
        secondProb: 0,
      };
    }
    // Center logits to prevent mean bias drift. Use pooled buffers to avoid
    // allocating intermediate arrays on every call.
    const meanLogit =
      (outputs[0] + outputs[1] + outputs[2] + outputs[3]) /
      MazeMovement.#ACTION_DIM;
    // variance
    let varianceSum = 0;
    for (let k = 0; k < MazeMovement.#ACTION_DIM; k++) {
      const delta = outputs[k] - meanLogit;
      varianceSum += delta * delta;
      MazeMovement.#SCRATCH_CENTERED[k] = delta;
    }
    varianceSum /= MazeMovement.#ACTION_DIM;
    let stdDev = Math.sqrt(varianceSum);
    if (!Number.isFinite(stdDev) || stdDev < MazeMovement.#STD_MIN)
      stdDev = MazeMovement.#STD_MIN;

    const collapseRatio =
      stdDev < MazeMovement.#COLLAPSE_STD_THRESHOLD
        ? MazeMovement.#COLLAPSE_RATIO_FULL
        : stdDev < MazeMovement.#COLLAPSE_STD_MED
        ? MazeMovement.#COLLAPSE_RATIO_HALF
        : 0;
    const temperature =
      MazeMovement.#TEMPERATURE_BASE +
      MazeMovement.#TEMPERATURE_SCALE * collapseRatio;

    // Find max of centered logits for numerical stability
    let maxCentered = -Infinity;
    for (let k = 0; k < MazeMovement.#ACTION_DIM; k++) {
      const v = MazeMovement.#SCRATCH_CENTERED[k];
      if (v > maxCentered) maxCentered = v;
    }

    // Exponentiate into pooled buffer and sum
    let expSum = 0;
    for (let k = 0; k < MazeMovement.#ACTION_DIM; k++) {
      const expVal = Math.exp(
        (MazeMovement.#SCRATCH_CENTERED[k] - maxCentered) / temperature
      );
      MazeMovement.#SCRATCH_EXPS[k] = expVal;
      expSum += expVal;
    }
    if (!expSum) expSum = 1;
    // Find direction with highest probability and compute softmax in-place
    let direction = 0;
    let maxProb = -Infinity;
    let secondProb = 0;
    const pooled = MazeMovement.#SOFTMAX; // pooled softmax output
    for (let k = 0; k < MazeMovement.#ACTION_DIM; k++) {
      const prob = MazeMovement.#SCRATCH_EXPS[k] / expSum;
      pooled[k] = prob;
      if (prob > maxProb) {
        secondProb = maxProb;
        maxProb = prob;
        direction = k;
      } else if (prob > secondProb) {
        secondProb = prob;
      }
    }
    // Compute entropy (uncertainty measure)
    let entropy = 0;
    for (let k = 0; k < MazeMovement.#ACTION_DIM; k++) {
      const prob = pooled[k];
      if (prob > 0) entropy += -prob * Math.log(prob);
    }
    entropy /= MazeMovement.#LOG_ACTIONS; // Normalize to [0,1]
    // Return a copy so caller cannot mutate pooled buffer (educational safety)
    return {
      direction,
      softmax: Array.from(pooled),
      entropy,
      maxProb,
      secondProb,
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

  /** Push a cell index into circular history (A->B loop detection). */
  static #pushHistory(state: SimulationState, cellIndex: number) {
    const { moveHistoryRing, moveHistoryHead, moveHistoryLength } = state;
    const capacity = moveHistoryRing.length;
    moveHistoryRing[moveHistoryHead] = cellIndex;
    state.moveHistoryHead = (moveHistoryHead + 1) % capacity;
    if (moveHistoryLength < capacity) state.moveHistoryLength++;
  }

  /** nth (1-based) from history end; 1 == last; undefined if not present. */
  static #nthFromHistoryEnd(
    state: SimulationState,
    n: number
  ): number | undefined {
    if (n > state.moveHistoryLength) return undefined;
    const capacity = state.moveHistoryRing.length;
    const index = (state.moveHistoryHead - n + capacity) % capacity;
    return state.moveHistoryRing[index];
  }

  /** Record visit + compute loop/memory/revisit penalties and optionally early terminate. */
  static #recordVisitAndUpdatePenalties(
    state: SimulationState,
    encodedMaze: number[][]
  ) {
    const visitedFlags = MazeMovement.#VisitedFlags!;
    const visitCountsTyped = MazeMovement.#VisitCounts!;
    const rewardScale = MazeMovement.#REWARD_SCALE;
    // Record current cell
    const cellIndex = MazeMovement.#index(state.position[0], state.position[1]);
    state.currentCellIndex = cellIndex;
    if (!visitedFlags[cellIndex]) {
      visitedFlags[cellIndex] = 1;
      state.visitedUniqueCount++;
    }
    visitCountsTyped[cellIndex]++;
    MazeMovement.#pushHistory(state, cellIndex);
    const visits = (state.visitsAtCurrent = visitCountsTyped[cellIndex]);

    // Loop detection (A->B->A->B)
    state.loopPenalty = 0;
    if (state.moveHistoryLength >= MazeMovement.#OSCILLATION_DETECT_LENGTH) {
      const last = MazeMovement.#nthFromHistoryEnd(state, 1)!;
      const secondLast = MazeMovement.#nthFromHistoryEnd(state, 2);
      const thirdLast = MazeMovement.#nthFromHistoryEnd(state, 3);
      const fourthLast = MazeMovement.#nthFromHistoryEnd(state, 4);
      if (
        last === thirdLast &&
        secondLast !== undefined &&
        fourthLast !== undefined &&
        secondLast === fourthLast
      ) {
        state.loopPenalty -= MazeMovement.#LOOP_PENALTY * rewardScale;
      }
    }
    // Memory penalty: returning to any recent cell (excluding immediate previous)
    state.memoryPenalty = 0;
    if (state.moveHistoryLength > 1) {
      for (let scan = 2; scan <= state.moveHistoryLength; scan++) {
        const candidateIndex = MazeMovement.#nthFromHistoryEnd(state, scan);
        if (candidateIndex === cellIndex) {
          state.memoryPenalty -=
            MazeMovement.#MEMORY_RETURN_PENALTY * rewardScale;
          break;
        }
      }
    }
    // Revisit penalty (dynamic scaling)
    state.revisitPenalty = 0;
    if (visits > 1) {
      state.revisitPenalty -=
        MazeMovement.#REVISIT_PENALTY_PER_VISIT * (visits - 1) * rewardScale;
    }
    if (visits > MazeMovement.#VISIT_TERMINATION_THRESHOLD) {
      state.invalidMovePenalty -=
        MazeMovement.#INVALID_MOVE_PENALTY_HARSH * rewardScale;
      state.earlyTerminate = true;
    }
  }

  /** Build vision inputs & compute distHere for proximity / epsilon logic. */
  static #buildVisionAndDistance(
    state: SimulationState,
    encodedMaze: number[][],
    exitPos: readonly [number, number],
    distanceMap?: number[][]
  ) {
    if (state.earlyTerminate) return;
    const hasDistanceMap = state.hasDistanceMap;
    const prevDistLocal = hasDistanceMap
      ? distanceMap![state.position[1]]?.[state.position[0]] ?? undefined
      : MazeMovement.#distanceAt(encodedMaze, state.position, distanceMap);
    const distCurrentLocal = prevDistLocal; // same pre-move
    state.vision = MazeVision.buildInputs6(
      encodedMaze,
      state.position,
      exitPos,
      distanceMap,
      MazeMovement.#StatePrevDistanceStep,
      distCurrentLocal,
      state.prevAction
    );
    MazeMovement.#StatePrevDistanceStep = distCurrentLocal;
    state.distHere = hasDistanceMap
      ? distanceMap![state.position[1]]?.[state.position[0]] ?? Infinity
      : MazeMovement.#distanceAt(encodedMaze, state.position, distanceMap);
  }

  /** Activate network, compute direction, update saturation counters & penalties. */
  static #decideDirection(
    state: SimulationState,
    network: INetwork,
    encodedMaze: number[][],
    distanceMap?: number[][]
  ) {
    if (state.earlyTerminate) return;
    try {
      const outputs = network.activate(state.vision) as number[];
      (network as any)._lastStepOutputs = MazeUtils.pushHistory(
        (network as any)._lastStepOutputs,
        [...outputs],
        MazeMovement.#OUTPUT_HISTORY_LENGTH
      );
      state.actionStats = MazeMovement.selectDirection(outputs);
      // Extracted saturation + bias handling
      MazeMovement.#applySaturationAndBiasAdjust(state, outputs, network);
      state.direction = state.actionStats.direction;
    } catch (error) {
      console.error('Error activating network:', error);
      state.direction = MazeMovement.#NO_MOVE;
    }
  }

  /** Greedy override when close to exit: choose direction minimizing distance. */
  static #maybeApplyProximityGreedy(
    state: SimulationState,
    encodedMaze: number[][],
    distanceMap?: number[][]
  ) {
    if (state.earlyTerminate) return;
    if (state.distHere <= MazeMovement.#PROXIMITY_GREEDY_DISTANCE) {
      let bestDirection = state.direction;
      let bestDistance = Infinity;
      for (
        let directionIndex = 0;
        directionIndex < MazeMovement.#ACTION_DIM;
        directionIndex++
      ) {
        const [deltaX, deltaY] = MazeMovement.#DIRECTION_DELTAS[directionIndex];
        const testX = state.position[0] + deltaX;
        const testY = state.position[1] + deltaY;
        if (!MazeMovement.isValidMove(encodedMaze, testX, testY)) continue;
        const candidateDistance = MazeMovement.#distanceAt(
          encodedMaze,
          [testX, testY],
          distanceMap
        );
        if (candidateDistance < bestDistance) {
          bestDistance = candidateDistance;
          bestDirection = directionIndex;
        }
      }
      if (bestDirection != null) state.direction = bestDirection;
    }
  }

  /** Epsilon-greedy exploration override. */
  static #maybeApplyEpsilonExploration(
    state: SimulationState,
    encodedMaze: number[][]
  ) {
    if (state.earlyTerminate) return;
    const epsilon = MazeMovement.#computeEpsilon(
      state.steps,
      state.stepsSinceImprovement,
      state.distHere,
      MazeMovement.#StateSaturations
    );
    if (MazeMovement.#rand() < epsilon) {
      for (
        let trialIndex = 0;
        trialIndex < MazeMovement.#ACTION_DIM;
        trialIndex++
      ) {
        const candidateDirection = Math.floor(
          MazeMovement.#rand() * MazeMovement.#ACTION_DIM
        );
        if (candidateDirection === state.prevAction) continue;
        const [deltaX, deltaY] = MazeMovement.#DIRECTION_DELTAS[
          candidateDirection
        ];
        const testX = state.position[0] + deltaX;
        const testY = state.position[1] + deltaY;
        if (MazeMovement.isValidMove(encodedMaze, testX, testY)) {
          state.direction = candidateDirection;
          break;
        }
      }
    }
  }

  /** Force exploration if no-move streak triggered. */
  static #maybeForceExploration(
    state: SimulationState,
    encodedMaze: number[][]
  ) {
    if (state.earlyTerminate) return;
    if (state.direction === MazeMovement.#NO_MOVE)
      MazeMovement.#StateNoMoveStreak++;
    else MazeMovement.#StateNoMoveStreak = 0;
    if (
      MazeMovement.#StateNoMoveStreak >= MazeMovement.#NO_MOVE_STREAK_THRESHOLD
    ) {
      for (
        let attemptIndex = 0;
        attemptIndex < MazeMovement.#ACTION_DIM;
        attemptIndex++
      ) {
        const candidateDirection = Math.floor(
          MazeMovement.#rand() * MazeMovement.#ACTION_DIM
        );
        const [deltaX, deltaY] = MazeMovement.#DIRECTION_DELTAS[
          candidateDirection
        ];
        const testX = state.position[0] + deltaX;
        const testY = state.position[1] + deltaY;
        if (MazeMovement.isValidMove(encodedMaze, testX, testY)) {
          state.direction = candidateDirection;
          break;
        }
      }
      MazeMovement.#StateNoMoveStreak = 0;
    }
  }

  /** Execute move and compute progress / exploration rewards. */
  static #executeMoveAndRewards(
    state: SimulationState,
    encodedMaze: number[][],
    distanceMap?: number[][]
  ) {
    if (state.earlyTerminate) return;
    state.prevDistance = MazeMovement.#distanceAt(
      encodedMaze,
      state.position,
      distanceMap
    );
    // Move
    state.moved = false;
    if (state.direction >= 0 && state.direction < MazeMovement.#ACTION_DIM) {
      const [deltaX, deltaY] = MazeMovement.#DIRECTION_DELTAS[state.direction];
      const newX = state.position[0] + deltaX;
      const newY = state.position[1] + deltaY;
      if (MazeMovement.isValidMove(encodedMaze, newX, newY)) {
        state.position[0] = newX;
        state.position[1] = newY;
        state.moved = true;
      }
    }
    const rewardScale = MazeMovement.#REWARD_SCALE;
    const pathX = MazeMovement.#PathX!;
    const pathY = MazeMovement.#PathY!;
    if (state.moved) {
      pathX[state.pathLength] = state.position[0];
      pathY[state.pathLength] = state.position[1];
      state.pathLength++;
      MazeUtils.pushHistory(
        state.recentPositions,
        [state.position[0], state.position[1]] as [number, number],
        MazeMovement.#LOCAL_WINDOW
      );
      MazeMovement.#maybeApplyLocalAreaPenalty(state, rewardScale);
      const currentDistance = state.hasDistanceMap
        ? state.distanceMap![state.position[1]]?.[state.position[0]] ?? Infinity
        : MazeMovement.#distanceAt(
            encodedMaze,
            state.position,
            state.distanceMap
          );
      const distanceDelta = state.prevDistance - currentDistance; // positive if improved
      const improved = distanceDelta > 0;
      const worsened = !improved && currentDistance > state.prevDistance;
      MazeMovement.#applyProgressShaping(
        state,
        distanceDelta,
        improved,
        worsened,
        rewardScale
      );
      MazeMovement.#applyExplorationVisitAdjustment(state, rewardScale);
      if (state.direction >= 0) state.directionCounts[state.direction]++;
      state.minDistanceToExit = Math.min(
        state.minDistanceToExit,
        currentDistance
      );
    } else {
      state.invalidMovePenalty -=
        MazeMovement.#INVALID_MOVE_PENALTY_MILD * rewardScale;
    }
    // Global distance improvement bonus
    MazeMovement.#applyGlobalDistanceImprovementBonus(
      state,
      encodedMaze,
      rewardScale
    );
    // Repetition / back-move penalties now in post action stage
  }

  /** Apply repetition / entropy / saturation penalties & update prevAction. */
  static #applyPostActionPenalties(state: SimulationState) {
    if (state.earlyTerminate) return;
    const rewardScale = MazeMovement.#REWARD_SCALE;
    MazeMovement.#applyRepetitionAndBacktrackPenalties(state, rewardScale);
    if (state.moved) state.prevAction = state.direction;
    MazeMovement.#applyEntropyGuidanceShaping(state, rewardScale);
    MazeMovement.#applySaturationPenaltyCycle(state, rewardScale);
    // Aggregate penalties captured earlier
    state.invalidMovePenalty +=
      state.loopPenalty + state.memoryPenalty + state.revisitPenalty;
  }

  /** Apply local area stagnation penalty if oscillating tightly without improvements. */
  static #maybeApplyLocalAreaPenalty(
    state: SimulationState,
    rewardScale: number
  ) {
    if (state.recentPositions.length !== MazeMovement.#LOCAL_WINDOW) return;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const [rx, ry] of state.recentPositions) {
      if (rx < minX) minX = rx;
      if (rx > maxX) maxX = rx;
      if (ry < minY) minY = ry;
      if (ry > maxY) maxY = ry;
    }
    const span = maxX - minX + (maxY - minY);
    if (
      span <= MazeMovement.#LOCAL_AREA_SPAN_THRESHOLD &&
      state.stepsSinceImprovement > MazeMovement.#LOCAL_AREA_STAGNATION_STEPS
    ) {
      state.localAreaPenalty -=
        MazeMovement.#LOCAL_AREA_PENALTY_AMOUNT * rewardScale;
    }
  }

  /** Progress shaping rewards / penalties based on distance delta. */
  static #applyProgressShaping(
    state: SimulationState,
    distanceDelta: number,
    improved: boolean,
    worsened: boolean,
    rewardScale: number
  ) {
    if (improved) {
      const confidence = state.actionStats?.maxProb ?? 1;
      const baseProgress =
        (MazeMovement.#PROGRESS_REWARD_BASE +
          MazeMovement.#PROGRESS_REWARD_CONF_SCALE * confidence) *
        rewardScale;
      if (state.stepsSinceImprovement > 0) {
        state.progressReward += Math.min(
          state.stepsSinceImprovement *
            MazeMovement.#PROGRESS_STEPS_MULT *
            rewardScale,
          MazeMovement.#PROGRESS_STEPS_MAX * rewardScale
        );
      }
      state.progressReward += baseProgress;
      state.stepsSinceImprovement = 0;
      state.progressReward +=
        distanceDelta *
        MazeMovement.#DISTANCE_DELTA_SCALE *
        (MazeMovement.#DISTANCE_DELTA_CONF_BASE +
          MazeMovement.#DISTANCE_DELTA_CONF_SCALE * confidence);
    } else if (worsened) {
      const confidence = state.actionStats?.maxProb ?? 0.5;
      state.progressReward -=
        (MazeMovement.#PROGRESS_AWAY_BASE_PENALTY +
          MazeMovement.#PROGRESS_AWAY_CONF_SCALE * confidence) *
        rewardScale;
      state.stepsSinceImprovement++;
    } else {
      state.stepsSinceImprovement++;
    }
  }

  /** Exploration bonus / revisit penalties for the just-visited cell. */
  static #applyExplorationVisitAdjustment(
    state: SimulationState,
    rewardScale: number
  ) {
    if (state.visitsAtCurrent === 1) {
      state.newCellExplorationBonus +=
        MazeMovement.#NEW_CELL_EXPLORATION_BONUS * rewardScale;
    } else {
      state.newCellExplorationBonus -=
        MazeMovement.#REVISIT_PENALTY_STRONG * rewardScale;
    }
  }

  /** Global distance improvement bonus for breaking long stagnation. */
  static #applyGlobalDistanceImprovementBonus(
    state: SimulationState,
    encodedMaze: number[][],
    rewardScale: number
  ) {
    const currentDistanceGlobal = state.hasDistanceMap
      ? state.distanceMap![state.position[1]]?.[state.position[0]] ?? Infinity
      : MazeMovement.#distanceAt(
          encodedMaze,
          state.position,
          state.distanceMap
        );
    if (currentDistanceGlobal < state.lastDistanceGlobal) {
      if (state.stepsSinceImprovement > MazeMovement.#GLOBAL_BREAK_BONUS_START)
        state.progressReward += Math.min(
          (state.stepsSinceImprovement -
            MazeMovement.#GLOBAL_BREAK_BONUS_START) *
            MazeMovement.#GLOBAL_BREAK_BONUS_PER_STEP *
            rewardScale,
          MazeMovement.#GLOBAL_BREAK_BONUS_CAP * rewardScale
        );
      state.stepsSinceImprovement = 0;
    }
    state.lastDistanceGlobal = currentDistanceGlobal;
  }

  /** Repetition and backward (opposite) move penalties. */
  static #applyRepetitionAndBacktrackPenalties(
    state: SimulationState,
    rewardScale: number
  ) {
    if (
      state.prevAction === state.direction &&
      state.stepsSinceImprovement > MazeMovement.#REPETITION_PENALTY_START
    ) {
      state.invalidMovePenalty -=
        MazeMovement.#REPETITION_PENALTY_BASE *
        (state.stepsSinceImprovement - MazeMovement.#REPETITION_PENALTY_START) *
        rewardScale;
    }
    if (
      state.prevAction >= 0 &&
      state.direction >= 0 &&
      state.stepsSinceImprovement > 0 &&
      state.direction === MazeMovement.#OPPOSITE_DIR[state.prevAction]
    ) {
      state.invalidMovePenalty -= MazeMovement.#BACK_MOVE_PENALTY * rewardScale;
    }
  }

  /** Entropy / guidance shaping (confidence vs ambiguity). */
  static #applyEntropyGuidanceShaping(
    state: SimulationState,
    rewardScale: number
  ) {
    if (!state.actionStats) return;
    const { entropy, maxProb, secondProb } = state.actionStats;
    const hasGuidance =
      MazeMovement.#sumVisionGroup(
        state.vision,
        MazeMovement.#VISION_LOS_START
      ) > 0 ||
      MazeMovement.#sumVisionGroup(
        state.vision,
        MazeMovement.#VISION_GRAD_START
      ) > 0;
    switch (true) {
      case entropy > MazeMovement.#ENTROPY_HIGH_THRESHOLD:
        state.invalidMovePenalty -= MazeMovement.#ENTROPY_PENALTY * rewardScale;
        break;
      case hasGuidance &&
        entropy < MazeMovement.#ENTROPY_CONFIDENT_THRESHOLD &&
        maxProb - secondProb > MazeMovement.#ENTROPY_CONFIDENT_DIFF:
        state.newCellExplorationBonus +=
          MazeMovement.#EXPLORATION_BONUS_SMALL * rewardScale;
        break;
      default:
        break;
    }
  }

  /** Saturation penalty application (periodic escalation). */
  static #applySaturationPenaltyCycle(
    state: SimulationState,
    rewardScale: number
  ) {
    if (
      MazeMovement.#StateSaturations < MazeMovement.#SATURATION_PENALTY_TRIGGER
    )
      return;
    state.invalidMovePenalty -=
      MazeMovement.#SATURATION_PENALTY_BASE * rewardScale;
    if (
      MazeMovement.#StateSaturations %
        MazeMovement.#SATURATION_PENALTY_PERIOD ===
      0
    ) {
      state.invalidMovePenalty -=
        MazeMovement.#SATURATION_PENALTY_ESCALATE * rewardScale;
    }
  }

  /** Handle saturation/overconfidence detection, penalties and adaptive bias adjustment. */
  static #applySaturationAndBiasAdjust(
    state: SimulationState,
    outputs: number[],
    network: INetwork
  ) {
    const rewardScale = MazeMovement.#REWARD_SCALE;
    // Overconfidence detection (probability based)
    const overConfident =
      state.actionStats.maxProb > MazeMovement.#OVERCONFIDENT_PROB &&
      state.actionStats.secondProb < MazeMovement.#SECOND_PROB_LOW;
    // Centered logits variance (std dev) for flat collapse detection
    const meanLogit =
      outputs.reduce((accumulator, value) => accumulator + value, 0) /
      MazeMovement.#ACTION_DIM;
    let varianceSum = 0;
    for (const logit of outputs) {
      const delta = logit - meanLogit;
      varianceSum += delta * delta;
    }
    varianceSum /= MazeMovement.#ACTION_DIM;
    const logStd = Math.sqrt(varianceSum);
    const flatCollapsed = logStd < MazeMovement.#LOGSTD_FLAT_THRESHOLD;
    // Update rolling saturation counter
    let saturationCounter = MazeMovement.#StateSaturations;
    if (overConfident || flatCollapsed) {
      saturationCounter++;
      state.saturatedSteps++;
    } else if (saturationCounter > 0) {
      saturationCounter--;
    }
    MazeMovement.#StateSaturations = saturationCounter;
    // Penalties
    if (overConfident) {
      state.invalidMovePenalty -=
        MazeMovement.#OVERCONFIDENT_PENALTY * rewardScale;
    }
    if (flatCollapsed) {
      state.invalidMovePenalty -=
        MazeMovement.#FLAT_COLLAPSE_PENALTY * rewardScale;
    }
    // Adaptive bias dampening when chronic saturation persists
    if (
      MazeMovement.#StateSaturations > MazeMovement.#SATURATION_ADJUST_MIN &&
      state.steps % MazeMovement.#SATURATION_ADJUST_INTERVAL === 0
    ) {
      try {
        const outputNodes = (network as any).nodes?.filter(
          (node: any) => node.type === MazeMovement.#NODE_TYPE_OUTPUT
        );
        if (outputNodes?.length) {
          const meanBias =
            outputNodes.reduce(
              (total: number, node: any) => total + node.bias,
              0
            ) / outputNodes.length;
          for (const node of outputNodes) {
            node.bias = Math.max(
              -MazeMovement.#BIAS_CLAMP,
              Math.min(
                MazeMovement.#BIAS_CLAMP,
                node.bias - meanBias * MazeMovement.#BIAS_ADJUST_FACTOR
              )
            );
          }
        }
      } catch {
        // Swallow bias adjustment errors (network shape may differ in some tests)
      }
    }
  }

  /** Check deep stagnation and optionally terminate. */
  static #maybeTerminateDeepStagnation(state: SimulationState): boolean {
    if (state.stepsSinceImprovement > MazeMovement.#DEEP_STAGNATION_THRESHOLD) {
      const rewardScale = MazeMovement.#REWARD_SCALE;
      try {
        if (typeof window === 'undefined') {
          state.invalidMovePenalty -=
            MazeMovement.#DEEP_STAGNATION_PENALTY * rewardScale;
          return true;
        }
      } catch {
        state.invalidMovePenalty -=
          MazeMovement.#DEEP_STAGNATION_PENALTY * rewardScale;
        return true;
      }
    }
    // Early termination from visit threshold
    return state.earlyTerminate;
  }

  /** Compute entropy from direction counts (shared by success & failure finalization). */
  static #computeActionEntropyFromCounts(directionCounts: number[]): number {
    const total = directionCounts.reduce((s, v) => s + v, 0) || 1;
    let entropySum = 0;
    for (const count of directionCounts) {
      if (!count) continue;
      const probability = count / total;
      entropySum -= probability * Math.log(probability);
    }
    return entropySum / MazeMovement.#LOG_ACTIONS;
  }

  /** Build and return result object for success scenario. */
  static #finalizeSuccess(state: SimulationState, maxSteps: number) {
    const stepEfficiency = maxSteps - state.steps;
    const actionEntropy = MazeMovement.#computeActionEntropyFromCounts(
      state.directionCounts
    );
    const fitness =
      MazeMovement.#SUCCESS_BASE_FITNESS +
      stepEfficiency * MazeMovement.#STEP_EFFICIENCY_SCALE +
      state.progressReward +
      state.newCellExplorationBonus +
      state.invalidMovePenalty +
      actionEntropy * MazeMovement.#SUCCESS_ACTION_ENTROPY_SCALE;
    const pathMaterialized = MazeMovement.#materializePath(state.pathLength);
    return {
      success: true,
      steps: state.steps,
      path: pathMaterialized,
      fitness: Math.max(MazeMovement.#MIN_SUCCESS_FITNESS, fitness),
      progress: 100,
      saturationFraction: state.steps ? state.saturatedSteps / state.steps : 0,
      actionEntropy,
    };
  }

  /** Build and return result object for failure scenario. */
  static #finalizeFailure(
    state: SimulationState,
    encodedMaze: number[][],
    startPos: readonly [number, number],
    exitPos: readonly [number, number],
    distanceMap?: number[][]
  ) {
    const pathX = MazeMovement.#PathX!;
    const pathY = MazeMovement.#PathY!;
    const lastPos: [number, number] = [
      pathX[state.pathLength - 1] ?? 0,
      pathY[state.pathLength - 1] ?? 0,
    ];
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
    const explorationScore = state.visitedUniqueCount * 1.0;
    const actionEntropy = MazeMovement.#computeActionEntropyFromCounts(
      state.directionCounts
    );
    const entropyBonus = actionEntropy * MazeMovement.#ENTROPY_BONUS_WEIGHT;
    const saturationPenalty = 0; // placeholder kept for future heuristics
    const outputVarPenalty = 0; // placeholder
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
    const raw =
      baseFitness + MazeMovement.#rand() * MazeMovement.#FITNESS_RANDOMNESS;
    const fitness = raw >= 0 ? raw : -Math.log1p(1 - raw);
    const pathMaterialized = MazeMovement.#materializePath(state.pathLength);
    return {
      success: false,
      steps: state.steps,
      path: pathMaterialized,
      fitness,
      progress,
      saturationFraction: state.steps ? state.saturatedSteps / state.steps : 0,
      actionEntropy,
    };
  }
}
