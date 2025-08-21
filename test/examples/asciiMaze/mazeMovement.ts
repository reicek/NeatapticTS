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

  // Near-miss penalty multiplier
  /** Penalty multiplier for near-miss (reaching distance 1 to goal) */
  static #NEAR_MISS_PENALTY = 30; // * rewardScale
  // Action/output dimension and softmax/entropy tuning
  /** Number of cardinal actions (N,E,S,W) */
  static #ACTION_DIM = 4;
  /** Natural log of ACTION_DIM; used to normalize entropy calculations */
  static #LOG_ACTIONS = Math.log(MazeMovement.#ACTION_DIM);
  /** Minimum path length required to compute action entropy */
  static #MIN_PATH_FOR_ENTROPY = 2;
  /** Minimum action total to avoid divide-by-zero fallbacks */
  static #MIN_ACTION_TOTAL = 1;
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

  /** Return the nth element from the end (1-based). Undefined when not available. */
  static #nthFromEnd<T>(arr: T[] | null | undefined, n: number): T | undefined {
    if (!arr || arr.length < n) return undefined;
    return arr[arr.length - n];
  }

  /** Return opposite cardinal direction (works for ACTION_DIM even if it changes) */
  static #opposite(direction: number): number {
    return (
      (direction + MazeMovement.#ACTION_DIM / 2) % MazeMovement.#ACTION_DIM
    );
  }

  /** Map a (dx,dy) delta to a cardinal direction index, or #NO_MOVE when unknown */
  static #deltaToDirection(dx: number, dy: number): number {
    for (let i = 0; i < MazeMovement.#DIRECTION_DELTAS.length; i++) {
      const [ddx, ddy] = MazeMovement.#DIRECTION_DELTAS[i];
      if (ddx === dx && ddy === dy) return i;
    }
    return MazeMovement.#NO_MOVE;
  }

  /**
   * Return the last element of an array or undefined.
   * Delegates to `MazeUtils.safeLast` to centralize boundary-safe trailing access.
   * @internal
   */
  static #last<T>(arr?: T[] | null): T | undefined {
    // Delegate to MazeUtils.safeLast to centralize trailing-element access
    return MazeUtils.safeLast(arr as any) as T | undefined;
  }

  /** Sum a contiguous group in the vision vector starting at `start`. */
  static #sumVisionGroup(vision: number[], start: number) {
    return vision
      .slice(start, start + MazeMovement.#VISION_GROUP_LEN)
      .reduce((a, b) => a + b, 0);
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
      : MazeUtils.bfsDistance(encodedMaze, [x, y], [x, y]);
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
    [x, y]: readonly [number, number]
  ): boolean {
    // Delegate to the private helper which centralizes the bounds/wall check
    return MazeMovement.#isCellOpen(encodedMaze, x, y);
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
    // Center logits to prevent mean bias drift
    /**
     * Mean of the output logits
     */
    const meanLogit =
      (outputs[0] + outputs[1] + outputs[2] + outputs[3]) /
      MazeMovement.#ACTION_DIM;
    /**
     * Variance sum of the outputs (for adaptive temperature)
     * @type {number}
     */
    let varianceSum = 0;
    for (const outputVal of outputs)
      varianceSum += (outputVal - meanLogit) * (outputVal - meanLogit);
    varianceSum /= MazeMovement.#ACTION_DIM;
    /**
     * Standard deviation of the outputs
     * @type {number}
     */
    let stdDev = Math.sqrt(varianceSum);
    if (!Number.isFinite(stdDev) || stdDev < MazeMovement.#STD_MIN)
      stdDev = MazeMovement.#STD_MIN;
    /**
     * Centered logits (mean subtracted)
     */
    const centered = outputs.map((outputVal) => outputVal - meanLogit);
    /**
     * Ratio for adaptive temperature (higher if variance is tiny)
     */
    const collapseRatio =
      stdDev < MazeMovement.#COLLAPSE_STD_THRESHOLD
        ? MazeMovement.#COLLAPSE_RATIO_FULL
        : stdDev < MazeMovement.#COLLAPSE_STD_MED
        ? MazeMovement.#COLLAPSE_RATIO_HALF
        : 0;
    /**
     * Softmax temperature (adaptive)
     */
    const temperature =
      MazeMovement.#TEMPERATURE_BASE +
      MazeMovement.#TEMPERATURE_SCALE * collapseRatio;
    /**
     * Maximum centered logit value
     */
    const maxCentered = Math.max(...centered);
    /**
     * Exponentiated logits for softmax
     */
    const exps = centered.map((v) => Math.exp((v - maxCentered) / temperature));
    /**
     * Sum of exponentiated logits (softmax denominator)
     */
    const expSum = exps.reduce((acc, val) => acc + val, 0) || 1;
    /**
     * Softmax probability vector
     */
    const softmax = exps.map((expVal) => expVal / expSum);
    // Find direction with highest probability
    let direction = 0;
    let maxProb = -Infinity;
    let secondProb = 0;
    softmax.forEach((prob, index) => {
      // Prefer switch/case for multi-branch comparisons. Use switch(true)
      // when branches are predicates evaluated in order.
      switch (true) {
        case prob > maxProb: {
          secondProb = maxProb;
          maxProb = prob;
          direction = index;
          break;
        }
        case prob > secondProb: {
          secondProb = prob;
          break;
        }
        default:
          break;
      }
    });
    // Compute entropy (uncertainty measure)
    let entropy = 0;
    softmax.forEach((prob) => {
      if (prob > 0) entropy += -prob * Math.log(prob);
    });
    entropy /= MazeMovement.#LOG_ACTIONS; // Normalize to [0,1]
    return { direction, softmax, entropy, maxProb, secondProb };
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
    /**
     * Current position of the agent [x, y]
     * @type {[number, number]}
     */
    let position = [startPos[0], startPos[1]] as [number, number];
    /**
     * Number of steps taken so far
     * @type {number}
     */
    let steps = 0;
    /**
     * Path of positions visited by the agent
     * @type {Array<[number, number]>}
     */
    let path = [[position[0], position[1]] as [number, number]];
    /**
     * Set of visited positions (as string keys)
     * @type {Set<string>}
     */
    let visitedPositions = new Set<string>();
    /**
     * Map of visit counts per cell
     * @type {Map<string, number>}
     */
    let visitCounts = new Map<string, number>();
    /**
     * Short-term memory for last N positions (for loop/oscillation detection)
     * @type {string[]}
     */
    let moveHistory: string[] = [];
    /**
     * Number of positions to keep in moveHistory
     * @type {number}
     */
    const MOVE_HISTORY_LENGTH = MazeMovement.#MOVE_HISTORY_LENGTH;
    /**
     * Closest distance to exit found so far
     * @type {number}
     */
    let minDistanceToExit = MazeMovement.#distanceAt(
      encodedMaze,
      position,
      distanceMap
    );

    /**
     * Reward scaling factor for all reward/penalty calculations
     * @type {number}
     */
    const rewardScale = MazeMovement.#REWARD_SCALE;

    /**
     * Accumulated reward for progress toward exit
     * @type {number}
     */
    let progressReward = 0;
    /**
     * Bonus for exploring new cells
     * @type {number}
     */
    let newCellExplorationBonus = 0;
    /**
     * Penalty for invalid moves or stagnation
     * @type {number}
     */
    let invalidMovePenalty = 0;

    /**
     * Last direction taken (0-3 or -1)
     * @type {number}
     */
    let prevAction = MazeMovement.#NO_MOVE;
    /**
     * Steps since last improvement in distance to exit
     * @type {number}
     */
    let stepsSinceImprovement = 0;
    /**
     * Initial global distance to exit
     * @type {number}
     */
    const startDistanceGlobal = MazeMovement.#distanceAt(
      encodedMaze,
      position,
      distanceMap
    );
    /**
     * Last global distance to exit
     * @type {number}
     */
    let lastDistanceGlobal = startDistanceGlobal;
    /**
     * Number of saturated steps (output collapse or overconfidence)
     * @type {number}
     */
    let saturatedSteps = 0;
    /**
     * Window size for local area stagnation detection (use class private constant)
     * @type {number}
     */
    // ... use MazeMovement.#LOCAL_WINDOW where needed
    /**
     * Recent positions for local area stagnation
     * @type {Array<[number, number]>}
     */
    const recentPositions: [number, number][] = [];
    /**
     * Penalty for local area stagnation
     * @type {number}
     */
    let localAreaPenalty = 0;

    // Main simulation loop: agent moves until maxSteps or exit is reached
    let lastProgressRatio = 0;
    while (steps < maxSteps) {
      steps++;

      // --- Step 1: Record current position as visited ---
      /**
       * String key for the agent's current position
       */
      const currentPosKey = `${position[0]},${position[1]}`;
      visitedPositions.add(currentPosKey);
      visitCounts.set(currentPosKey, (visitCounts.get(currentPosKey) || 0) + 1);
      // Record recent move into a small bounded history used for oscillation detection
      moveHistory = MazeUtils.pushHistory(
        moveHistory,
        currentPosKey,
        MOVE_HISTORY_LENGTH
      );

      // --- Step 2: Calculate percent of maze explored so far ---
      /**
       * Percent of maze explored so far
       */
      const percentExplored =
        visitedPositions.size / (encodedMaze.length * encodedMaze[0].length);

      // --- Step 3: Oscillation/loop detection (A->B->A->B) ---
      /**
       * Penalty for oscillation/looping
       */
      let loopPenalty = 0;
      if (moveHistory.length >= MazeMovement.#OSCILLATION_DETECT_LENGTH) {
        const last = MazeMovement.#last(moveHistory)!;
        const thirdLast = MazeMovement.#nthFromEnd(moveHistory, 3);
        const secondLast = MazeMovement.#nthFromEnd(moveHistory, 2);
        const fourthLast = MazeMovement.#nthFromEnd(moveHistory, 4);
        if (last === thirdLast && secondLast === fourthLast) {
          // detected A->B->A->B oscillation
          loopPenalty -= MazeMovement.#LOOP_PENALTY * rewardScale; // Strong penalty for 2-step loop
        }
      }
      /**
       * Memory loop indicator input (1 if loop detected, else 0)
       */
      const loopFlag = loopPenalty < 0 ? 1 : 0;

      // --- Step 4: Penalty for returning to any cell in recent history ---
      /**
       * Penalty for returning to a cell in recent history
       */
      let memoryPenalty = 0;
      if (moveHistory.length > 1) {
        const idx = moveHistory.indexOf(currentPosKey);
        if (idx !== -1 && idx < moveHistory.length - 1) {
          memoryPenalty -= MazeMovement.#MEMORY_RETURN_PENALTY * rewardScale;
        }
      }

      // --- Step 5: Dynamic penalty for multiple visits ---
      /**
       * Penalty for revisiting cells
       */
      let revisitPenalty = 0;
      /**
       * Number of times the current cell has been visited
       */
      const visits = visitCounts.get(currentPosKey) || 1;
      if (visits > 1) {
        revisitPenalty -=
          MazeMovement.#REVISIT_PENALTY_PER_VISIT * (visits - 1) * rewardScale; // Penalty increases with each revisit
      }

      // --- Step 6: Early termination if a cell is visited too many times ---
      if (visits > MazeMovement.#VISIT_TERMINATION_THRESHOLD) {
        invalidMovePenalty -=
          MazeMovement.#INVALID_MOVE_PENALTY_HARSH * rewardScale;
        break;
      }

      // --- Step 7: Build agent's perception (vision inputs) ---
      /**
       * Previous local distance to exit
       */
      const prevDistLocal = distanceMap
        ? distanceMap[position[1]]?.[position[0]] ?? undefined
        : MazeMovement.#distanceAt(encodedMaze, position, distanceMap);
      /**
       * Current local distance to exit (same as prev before move)
       */
      const distCurrentLocal = prevDistLocal;
      /**
       * Vision input vector for the agent
       */
      const vision = MazeVision.buildInputs6(
        encodedMaze,
        position,
        exitPos,
        distanceMap,
        (MazeMovement as any)._prevDistanceStep,
        distCurrentLocal,
        prevAction
      );
      (MazeMovement as any)._prevDistanceStep = distCurrentLocal;

      // --- Step 8: Get distance at current location (for proximity logic) ---
      /**
       * Distance at current location (pre-action) for proximity exploitation logic
       */
      const distHere = MazeMovement.#distanceAt(
        encodedMaze,
        position,
        distanceMap
      );

      // --- Step 9: Neural network decision making ---
      /**
       * Chosen direction for this step
       */
      let direction;
      /**
       * Action statistics (softmax, entropy, etc.)
       */
      let actionStats: any = null;
      try {
        // Activate the network with vision inputs
        /**
         * Output vector from the neural network
         */
        const outputs = network.activate(vision) as number[];
        // Track outputs for variance diagnostics using a bounded sliding window
        (network as any)._lastStepOutputs = MazeUtils.pushHistory(
          (network as any)._lastStepOutputs,
          [...outputs],
          MazeMovement.#OUTPUT_HISTORY_LENGTH
        );
        // Select direction and compute stats
        actionStats = MazeMovement.selectDirection(outputs);
        // Detect output saturation (overconfidence or flat collapse)
        (MazeMovement as any)._saturations =
          (MazeMovement as any)._saturations || 0;
        const overConfident =
          actionStats.maxProb > MazeMovement.#OVERCONFIDENT_PROB &&
          actionStats.secondProb < MazeMovement.#SECOND_PROB_LOW;
        // Recompute std on centered logits
        const logitsMean =
          outputs.reduce((s, v) => s + v, 0) / MazeMovement.#ACTION_DIM;
        let logVar = 0;
        for (const o of outputs) logVar += Math.pow(o - logitsMean, 2);
        logVar /= MazeMovement.#ACTION_DIM;
        const logStd = Math.sqrt(logVar);
        const flatCollapsed = logStd < MazeMovement.#LOGSTD_FLAT_THRESHOLD;
        const saturatedNow = overConfident || flatCollapsed;
        if (saturatedNow) {
          (MazeMovement as any)._saturations++;
          saturatedSteps++;
        } else {
          (MazeMovement as any)._saturations = Math.max(
            0,
            (MazeMovement as any)._saturations - 1
          );
        }
        // Penalties for saturation
        if (overConfident)
          invalidMovePenalty -=
            MazeMovement.#OVERCONFIDENT_PENALTY * rewardScale;
        if (flatCollapsed)
          invalidMovePenalty -=
            MazeMovement.#FLAT_COLLAPSE_PENALTY * rewardScale;
        // Adaptive bias anti-saturation: gently reduce output biases if chronic
        try {
          if (
            (MazeMovement as any)._saturations >
              MazeMovement.#SATURATION_ADJUST_MIN &&
            steps % MazeMovement.#SATURATION_ADJUST_INTERVAL === 0
          ) {
            const outs = (network as any).nodes?.filter(
              (n: any) => n.type === MazeMovement.#NODE_TYPE_OUTPUT
            );
            if (outs?.length) {
              const mean =
                outs.reduce((a: number, n: any) => a + n.bias, 0) / outs.length;
              outs.forEach((n: any) => {
                n.bias = Math.max(
                  -MazeMovement.#BIAS_CLAMP,
                  Math.min(
                    MazeMovement.#BIAS_CLAMP,
                    n.bias - mean * MazeMovement.#BIAS_ADJUST_FACTOR
                  )
                );
              });
            }
          }
        } catch {
          /* ignore */
        }
        direction = actionStats.direction;
      } catch (error) {
        console.error('Error activating network:', error);
        direction = MazeMovement.#NO_MOVE; // Fallback: don't move
      }

      // --- Step 10: Proximity exploitation (greedy move if near exit) ---
      if (distHere <= MazeMovement.#PROXIMITY_GREEDY_DISTANCE) {
        /**
         * Best direction found (minimizing distance to exit)
         */
        let bestDirection = direction;
        /**
         * Best distance found
         */
        let bestDistance = Infinity;
        for (
          let dirIndex = 0;
          dirIndex < MazeMovement.#ACTION_DIM;
          dirIndex++
        ) {
          const testPos = MazeMovement.moveAgent(
            encodedMaze,
            position,
            dirIndex
          );
          if (testPos[0] === position[0] && testPos[1] === position[1])
            continue; // invalid
          /**
           * Distance value for candidate direction
           */
          const candidateDistance = MazeMovement.#distanceAt(
            encodedMaze,
            testPos,
            distanceMap
          );
          if (candidateDistance < bestDistance) {
            bestDistance = candidateDistance;
            bestDirection = dirIndex;
          }
        }
        if (bestDirection != null) direction = bestDirection;
      }

      // Epsilon-greedy exploration: encourage divergence early & when stagnant
      const stepsStagnant = stepsSinceImprovement;
      let epsilon = 0;
      // Select epsilon using a switch over ordered predicates for clarity
      switch (true) {
        case steps < MazeMovement.#EPSILON_WARMUP_STEPS:
          epsilon = MazeMovement.#EPSILON_INITIAL;
          break;
        case stepsStagnant > MazeMovement.#EPSILON_STAGNANT_HIGH_THRESHOLD:
          epsilon = MazeMovement.#EPSILON_STAGNANT_HIGH;
          break;
        case stepsStagnant > MazeMovement.#EPSILON_STAGNANT_MED_THRESHOLD:
          epsilon = MazeMovement.#EPSILON_STAGNANT_MED;
          break;
        case (MazeMovement as any)._saturations >
          MazeMovement.#EPSILON_SATURATION_TRIGGER:
          epsilon = MazeMovement.#EPSILON_SATURATIONS;
          break;
        default:
          break;
      }
      // Suppress exploration when near goal to encourage completion
      if (distHere <= MazeMovement.#PROXIMITY_SUPPRESS_EXPLOR_DIST)
        epsilon = Math.min(epsilon, MazeMovement.#EPSILON_MIN_NEAR_GOAL);
      if (Math.random() < epsilon) {
        // pick a random valid direction differing from previous when possible
        /**
         * Candidate directions for random exploration
         */
        const candidateDirections = [0, 1, 2, 3].filter(
          (dir) => dir !== prevAction
        );
        while (candidateDirections.length) {
          /**
           * Random index into candidateDirections
           */
          const randomIndex = Math.floor(
            Math.random() * candidateDirections.length
          );
          /**
           * Candidate direction value
           */
          const candidateDirection = candidateDirections.splice(
            randomIndex,
            1
          )[0];
          /**
           * Test position for candidate direction
           */
          const testPos = MazeMovement.moveAgent(
            encodedMaze,
            position,
            candidateDirection
          );
          if (testPos[0] !== position[0] || testPos[1] !== position[1]) {
            direction = candidateDirection;
            break;
          }
        }
      }

      // --- Anti-stagnation: if we haven't moved in several attempts, force exploratory move
      // Track consecutive failed moves
      (MazeMovement as any)._noMoveStreak =
        (MazeMovement as any)._noMoveStreak || 0;
      if (direction === MazeMovement.#NO_MOVE)
        (MazeMovement as any)._noMoveStreak++;
      if (
        (MazeMovement as any)._noMoveStreak >=
        MazeMovement.#NO_MOVE_STREAK_THRESHOLD
      ) {
        // pick a random cardinal direction until a valid move found (epsilon-greedy style)
        for (let attempt = 0; attempt < MazeMovement.#ACTION_DIM; attempt++) {
          /**
           * Candidate direction for forced exploration
           */
          const candidateDirection = Math.floor(
            Math.random() * MazeMovement.#ACTION_DIM
          );
          /**
           * Test position for candidate direction
           */
          const testPos = MazeMovement.moveAgent(
            encodedMaze,
            position,
            candidateDirection
          );
          if (testPos[0] !== position[0] || testPos[1] !== position[1]) {
            direction = candidateDirection;
            break;
          }
        }
        (MazeMovement as any)._noMoveStreak = 0;
      }

      // Save previous state for reward calculation
      /**
       * Previous position before move
       */
      const prevPosition = [position[0], position[1]] as [number, number];
      /**
       * Previous distance to exit before move
       */
      const prevDistance = MazeMovement.#distanceAt(
        encodedMaze,
        position,
        distanceMap
      );

      // --- ACTION: Move based on network decision
      position = MazeMovement.moveAgent(encodedMaze, position, direction);
      /**
       * Whether the agent actually moved this step
       */
      const moved =
        prevPosition[0] !== position[0] || prevPosition[1] !== position[1];

      // Record movement and update rewards/penalties
      if (moved) {
        path.push([position[0], position[1]] as [number, number]);
        // Add to a small bounded window of recent positions used for local stagnation
        // detection. Use the shared helper to preserve in-place semantics and avoid
        // repeated O(n) shift() operations when trimming the head.
        MazeUtils.pushHistory(
          recentPositions,
          [position[0], position[1]] as [number, number],
          MazeMovement.#LOCAL_WINDOW
        );
        if (recentPositions.length === MazeMovement.#LOCAL_WINDOW) {
          /**
           * Minimum and maximum X/Y in recent positions
           */
          let minX = Infinity,
            maxX = -Infinity,
            minY = Infinity,
            maxY = -Infinity;
          for (const [rx, ry] of recentPositions) {
            if (rx < minX) minX = rx;
            if (rx > maxX) maxX = rx;
            if (ry < minY) minY = ry;
            if (ry > maxY) maxY = ry;
          }
          /**
           * Span of recent positions (for local oscillation detection)
           */
          const span = maxX - minX + (maxY - minY);
          // Penalize tight oscillation in a small neighborhood when no improvements recently
          if (
            span <= MazeMovement.#LOCAL_AREA_SPAN_THRESHOLD &&
            stepsSinceImprovement > MazeMovement.#LOCAL_AREA_STAGNATION_STEPS
          ) {
            localAreaPenalty -=
              MazeMovement.#LOCAL_AREA_PENALTY_AMOUNT * rewardScale; // accumulate gradually
          }
        }

        // Calculate current distance to exit
        /**
         * Current distance to exit after move
         */
        const currentDistance = MazeMovement.#distanceAt(
          encodedMaze,
          position,
          distanceMap
        );

        // Reward for getting closer to exit, penalty for moving away
        /**
         * Change in distance to exit (positive if improved)
         */
        const distanceDelta = prevDistance - currentDistance; // positive if improved
        // Use switch(true) to clearly express ordered predicates for distance changes
        switch (true) {
          case distanceDelta > 0: {
            // Confidence shaping if available
            const conf = actionStats?.maxProb ?? 1;
            progressReward +=
              (MazeMovement.#PROGRESS_REWARD_BASE +
                MazeMovement.#PROGRESS_REWARD_CONF_SCALE * conf) *
              rewardScale;
            if (stepsSinceImprovement > 0)
              progressReward += Math.min(
                stepsSinceImprovement *
                  MazeMovement.#PROGRESS_STEPS_MULT *
                  rewardScale,
                MazeMovement.#PROGRESS_STEPS_MAX * rewardScale
              );
            stepsSinceImprovement = 0;
            // Additional proportional reward to create gradient
            progressReward +=
              distanceDelta *
              MazeMovement.#DISTANCE_DELTA_SCALE *
              (MazeMovement.#DISTANCE_DELTA_CONF_BASE +
                MazeMovement.#DISTANCE_DELTA_CONF_SCALE * conf); // scale by confidence
            break;
          }
          case currentDistance > prevDistance: {
            const conf = actionStats?.maxProb ?? 0.5;
            progressReward -=
              (MazeMovement.#PROGRESS_AWAY_BASE_PENALTY +
                MazeMovement.#PROGRESS_AWAY_CONF_SCALE * conf) *
              rewardScale;
            stepsSinceImprovement++;
            break;
          }
          default: {
            stepsSinceImprovement++;
            break;
          }
        }

        // Bonus for exploring new cells, penalty for revisiting
        if (visits === 1) {
          newCellExplorationBonus +=
            MazeMovement.#NEW_CELL_EXPLORATION_BONUS * rewardScale;
        } else {
          newCellExplorationBonus -=
            MazeMovement.#REVISIT_PENALTY_STRONG * rewardScale; // Stronger penalty for revisiting
        }

        // Track closest approach to exit
        minDistanceToExit = Math.min(minDistanceToExit, currentDistance);
      } else {
        // Penalty for invalid move (collision or out of bounds)
        // Previously this was extremely punitive (-1000 * scale) causing all genomes to bottom-out at the clamp
        // which destroyed selection pressure. Keep it mild so progress/exploration dominate.
        invalidMovePenalty -=
          MazeMovement.#INVALID_MOVE_PENALTY_MILD * rewardScale; // mild penalty now
        // No tolerance for invalid moves; break if needed
        steps === maxSteps;
      }
      // Update global distance improvement memory
      /**
       * Current global distance to exit
       */
      const currentDistanceGlobal = MazeMovement.#distanceAt(
        encodedMaze,
        position,
        distanceMap
      );
      if (currentDistanceGlobal < lastDistanceGlobal) {
        // bonus for breaking a long stagnation globally
        if (stepsSinceImprovement > MazeMovement.#GLOBAL_BREAK_BONUS_START)
          progressReward += Math.min(
            (stepsSinceImprovement - MazeMovement.#GLOBAL_BREAK_BONUS_START) *
              MazeMovement.#GLOBAL_BREAK_BONUS_PER_STEP *
              rewardScale,
            MazeMovement.#GLOBAL_BREAK_BONUS_CAP * rewardScale
          );
        stepsSinceImprovement = 0;
      }
      lastDistanceGlobal = currentDistanceGlobal;
      // Repetition penalty: if repeating same action without improvement
      if (
        prevAction === direction &&
        stepsSinceImprovement > MazeMovement.#REPETITION_PENALTY_START
      ) {
        invalidMovePenalty -=
          MazeMovement.#REPETITION_PENALTY_BASE *
          (stepsSinceImprovement - MazeMovement.#REPETITION_PENALTY_START) *
          rewardScale;
      }
      // Penalize backward (opposite) moves strongly if they do not improve
      if (prevAction >= 0 && direction >= 0) {
        // Opposite direction to previous action (use helper in case ACTION_DIM changes)
        if (
          direction === MazeMovement.#opposite(prevAction) &&
          stepsSinceImprovement > 0
        ) {
          invalidMovePenalty -= MazeMovement.#BACK_MOVE_PENALTY * rewardScale;
        }
      }
      // Only record previous action if movement succeeded to avoid mismatches
      if (moved) {
        prevAction = direction; // record last successful move for back-direction suppression
      }

      // Encourage decisiveness: slight penalty for very high entropy (uniform outputs),
      // slight bonus for confident low-entropy when some guidance signal (gradient or LOS) exists.
      if (actionStats) {
        const { entropy, maxProb, secondProb } = actionStats;
        // Compute presence of directional guidance (any non-zero gradient or LOS)
        const hasGuidance =
          MazeMovement.#sumVisionGroup(vision, MazeMovement.#VISION_LOS_START) >
            0 ||
          MazeMovement.#sumVisionGroup(
            vision,
            MazeMovement.#VISION_GRAD_START
          ) > 0;
        // Use switch(true) to express ordered predicate logic clearly
        switch (true) {
          case entropy > MazeMovement.#ENTROPY_HIGH_THRESHOLD: {
            invalidMovePenalty -= MazeMovement.#ENTROPY_PENALTY * rewardScale; // discourage persistent ambiguity
            break;
          }
          case hasGuidance &&
            entropy < MazeMovement.#ENTROPY_CONFIDENT_THRESHOLD &&
            maxProb - secondProb > MazeMovement.#ENTROPY_CONFIDENT_DIFF: {
            newCellExplorationBonus +=
              MazeMovement.#EXPLORATION_BONUS_SMALL * rewardScale; // tiny shaping bonus for clear decision
            break;
          }
          default:
            break;
        }
        // Penalty for prolonged saturation (uninformative all-ones behavior)
        if (
          (MazeMovement as any)._saturations >=
          MazeMovement.#SATURATION_PENALTY_TRIGGER
        ) {
          invalidMovePenalty -=
            MazeMovement.#SATURATION_PENALTY_BASE * rewardScale;
          if (
            (MazeMovement as any)._saturations %
              MazeMovement.#SATURATION_PENALTY_PERIOD ===
            0
          ) {
            invalidMovePenalty -=
              MazeMovement.#SATURATION_PENALTY_ESCALATE * rewardScale; // escalating periodically when saturated
          }
        }
      }

      // Early termination on deep stagnation (disabled for browser demo to allow full exploration)
      if (stepsSinceImprovement > MazeMovement.#DEEP_STAGNATION_THRESHOLD) {
        try {
          if (typeof window === 'undefined') {
            invalidMovePenalty -=
              MazeMovement.#DEEP_STAGNATION_PENALTY * rewardScale;
            break; // keep for non-browser environments (tests / Node)
          }
        } catch {
          // if window check failed, proceed with default behavior
          invalidMovePenalty -=
            MazeMovement.#DEEP_STAGNATION_PENALTY * rewardScale;
          break;
        }
      }

      // Apply oscillation/loop/memory/revisit penalties
      invalidMovePenalty += loopPenalty + memoryPenalty + revisitPenalty;

      // --- SUCCESS CHECK: Exit reached
      if (position[0] === exitPos[0] && position[1] === exitPos[1]) {
        // Calculate fitness for successful completion
        /**
         * Step efficiency (remaining steps)
         */
        const stepEfficiency = maxSteps - steps;
        // Action entropy bonus for successful runs (promote balanced yet decisive policies)
        /**
         * Action entropy for the successful path
         */
        const { actionEntropy } = MazeMovement.computeActionEntropy(path);
        /**
         * Fitness score for successful completion
         */
        const fitness =
          MazeMovement.#SUCCESS_BASE_FITNESS +
          stepEfficiency * MazeMovement.#STEP_EFFICIENCY_SCALE +
          progressReward +
          newCellExplorationBonus +
          invalidMovePenalty +
          actionEntropy * MazeMovement.#SUCCESS_ACTION_ENTROPY_SCALE;

        return {
          success: true,
          steps,
          path,
          fitness: Math.max(MazeMovement.#MIN_SUCCESS_FITNESS, fitness),
          progress: 100,
          saturationFraction: steps ? saturatedSteps / steps : 0,
          actionEntropy,
        };
      }
    }

    // --- FAILURE CASE: Did not reach exit
    /**
     * Progress percentage toward exit (0-100)
     */
    const lastPos = MazeMovement.#last(path) ?? [0, 0];
    const progress = distanceMap
      ? MazeUtils.calculateProgressFromDistanceMap(
          distanceMap,
          lastPos,
          startPos
        )
      : MazeUtils.calculateProgress(encodedMaze, lastPos, startPos, exitPos);

    // Fitness for unsuccessful attempts: emphasize progress & exploration with moderated penalties
    /**
     * Fractional progress toward exit (0..1)
     */
    const progressFrac = progress / 100;
    /**
     * Shaped progress score (concave for early gradient)
     */
    const shapedProgress =
      Math.pow(progressFrac, MazeMovement.#PROGRESS_POWER) *
      MazeMovement.#PROGRESS_SCALE;
    /**
     * Exploration score (number of unique cells visited)
     */
    const explorationScore = visitedPositions.size * 1.0; // increase weight so exploration differentiates genomes
    /**
     * Aggregated penalty
     */
    const penalty = invalidMovePenalty; // already aggregated
    // Action entropy based on path
    /**
     * Action entropy for the failed path
     */
    const { actionEntropy } = MazeMovement.computeActionEntropy(path);
    /**
     * Bonus for action entropy
     */
    const entropyBonus = actionEntropy * MazeMovement.#ENTROPY_BONUS_WEIGHT; // weight
    /**
     * Additional optional penalties initialized to zero. In earlier
     * revisions these were computed from output-variance, saturation and
     * near-miss heuristics. Initialize them here so the final fitness
     * calculation remains stable while we modernize incrementally.
     */
    let saturationPenalty = 0;
    let outputVarPenalty = 0;
    let nearMissPenalty = 0;
    const satFrac = steps ? saturatedSteps / steps : 0;

    if (
      minDistanceToExit ===
      MazeMovement.#NEAR_MISS_PENALTY /* placeholder check kept for semantics */
    )
      nearMissPenalty -= MazeMovement.#NEAR_MISS_PENALTY * rewardScale;

    /**
     * Base fitness score before final adjustment
     */
    const base =
      shapedProgress +
      explorationScore +
      progressReward +
      newCellExplorationBonus +
      penalty +
      entropyBonus +
      localAreaPenalty +
      saturationPenalty +
      outputVarPenalty +
      nearMissPenalty;
    const raw = base + Math.random() * MazeMovement.#FITNESS_RANDOMNESS;
    /**
     * Final fitness score (nonlinear squash for negatives)
     */
    const fitness = raw >= 0 ? raw : -Math.log1p(1 - raw);
    return {
      success: false,
      steps,
      path,
      fitness,
      progress,
      saturationFraction: satFrac,
      actionEntropy,
    };
  }

  /**
   * Computes the entropy of the agent's action distribution from its path.
   * Higher entropy means more diverse movement; lower means repetitive.
   *
   * @param path - Array of [x, y] positions visited by the agent.
   * @returns {object} actionEntropy (0..1)
   */
  static computeActionEntropy(path: readonly [number, number][]) {
    if (!path || path.length < this.#MIN_PATH_FOR_ENTROPY)
      return { actionEntropy: 0 };
    /**
     * Counts of each direction taken (N, E, S, W)
     * @type {number[]}
     */
    const directionCounts = [0, 0, 0, 0];
    for (let stepIndex = 1; stepIndex < path.length; stepIndex++) {
      const deltaX = path[stepIndex][0] - path[stepIndex - 1][0];
      const deltaY = path[stepIndex][1] - path[stepIndex - 1][1];
      // Map the delta to a direction index using the centralized deltas table
      const dir = MazeMovement.#deltaToDirection(deltaX, deltaY);
      if (dir >= 0 && dir < directionCounts.length) directionCounts[dir]++;
    }
    /**
     * Total number of actions taken
     */
    const actionTotal =
      directionCounts.reduce((acc, val) => acc + val, 0) ||
      this.#MIN_ACTION_TOTAL;
    let entropySum = 0;
    directionCounts.forEach((count) => {
      if (count > 0) {
        const probability = count / actionTotal;
        entropySum += -probability * Math.log(probability);
      }
    });
    /**
     * Normalized entropy of the action distribution (0=deterministic, 1=uniform)
     * @type {number}
     */
    const actionEntropy = entropySum / this.#LOG_ACTIONS;
    return { actionEntropy };
  }
}
