/**
 * Shared type surface for the dedicated mazeMovement module.
 *
 * Step 2 moves internal simulation contracts here first so later helper files
 * can depend on one narrow typed surface.
 */

/**
 * Initialized pooled buffers shared across maze movement simulations.
 *
 * These pools are reused between runs to keep the hot path allocation-light
 * while preserving a narrow typed seam for service helpers.
 */
export interface MazeMovementBufferPools {
  /** Reused per-cell visited flags aligned to the current maze dimensions. */
  visitedFlags: Uint8Array;
  /** Reused per-cell visit counters aligned to the current maze dimensions. */
  visitCounts: Uint16Array;
  /** Reused X-coordinate path buffer for the active simulation. */
  pathX: Int32Array;
  /** Reused Y-coordinate path buffer for the active simulation. */
  pathY: Int32Array;
  /** Cached maze width used by index helpers. */
  cachedWidth: number;
  /** Cached maze height used by bounds helpers. */
  cachedHeight: number;
}

/**
 * Mutable run-scoped service state shared by the mazeMovement facade.
 *
 * The dedicated services module owns these counters so later runtime, policy,
 * and shaping helpers can depend on one explicit mutable surface instead of
 * directly reaching into class-private state.
 */
export interface MazeMovementRunServiceState {
  /** Rolling saturation counter used by adaptive penalties and epsilon logic. */
  saturations: number;
  /** Consecutive no-move counter used to force exploration. */
  noMoveStreak: number;
  /** Previous distance value fed into the next vision-builder call. */
  prevDistanceStep: number | undefined;
}

/**
 * Diagnostic telemetry produced when selecting a direction from network logits.
 *
 * Encapsulates the chosen direction along with entropy and probability data so
 * downstream helpers can apply shaping rewards and penalties without
 * rederiving softmax statistics on hot paths.
 */
export interface DirectionSelectionStats {
  /** Chosen action index (0..#ACTION_DIM-1) or -1 when no move is selected. */
  direction: number;
  /** Defensive copy of per-action softmax probabilities. */
  softmax: number[];
  /** Normalised entropy of the action distribution in [0,1]. */
  entropy: number;
  /** Probability assigned to the chosen action. */
  maxProb: number;
  /** Probability assigned to the runner-up action. */
  secondProb: number;
}

/**
 * Internal aggregate state used during a single agent simulation run.
 *
 * Purpose:
 * - Hold all derived runtime values, counters and diagnostic stats used by the
 *   MazeMovement simulation helpers. This shape is intentionally rich so tests
 *   and visualisers can inspect intermediate state when debugging.
 *
 * Notes:
 * - This interface remains internal to the mazeMovement module boundary.
 * - Property descriptions are explicit to surface helpful tooltips in editors.
 */
export interface SimulationState {
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
  actionStats: DirectionSelectionStats | null;
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
 * Result shape returned by `MazeMovement.simulateAgent`.
 *
 * This contract matches the legacy inline return annotation so callers can
 * keep depending on the current fields while the dedicated module boundary is
 * being extracted.
 */
export interface MazeMovementSimulationResult {
  /** Whether the run reached the maze exit. */
  success: boolean;
  /** Number of steps executed before success or failure finalization. */
  steps: number;
  /** Materialized path snapshot for the finished run. */
  path: readonly [number, number][];
  /** Final shaped fitness for the run. */
  fitness: number;
  /** Progress percentage toward the exit. */
  progress: number;
  /** Optional fraction of saturated steps observed during the run. */
  saturationFraction?: number;
  /** Optional action-entropy summary derived from direction counts. */
  actionEntropy?: number;
}
