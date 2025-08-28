// Interfaces for ASCII Maze Neuroevolution System
// This file centralizes shared interfaces and types for consistency and maintainability.

/**
 * Interface for dashboard manager abstraction.
 * Used for dependency inversion and testability.
 */
export interface IDashboardManager {
  /**
   * Update the dashboard with the latest simulation/evolution state.
   *
   * Callers:
   * - Evolution loop: supplies a snapshot after each generation or on important
   *   events (e.g., when a solution is found).
   *
   * Responsibilities:
   * 1. Present a human-readable maze layout (maze array) and agent outcome.
   * 2. Optionally visualise or serialize the supplied network structure.
   * 3. Optionally use neatInstance for richer telemetry (population stats,
   *    training snapshots), but implementations should not mutate neatInstance.
   *
   * @param maze - The current maze layout represented as an array of ASCII strings.
   *               Each string is a row; implementations may join or render rows
   *               line-by-line for presentation.
   * @param result - An opaque result object produced by the agent run (fitness,
   *                 steps taken, solved flag, progress metrics, etc.). Consumers
   *                 should treat this as read-only.
   * @param network - The network instance used for the run. Implementations may
   *                  inspect its nodes/connections for visualization but must
   *                  not assume any concrete runtime shape beyond the INetwork
   *                  interface.
   * @param generation - The current generation number (0-based or 1-based as
   *                     used by the caller). Display purposes only.
   * @param neatInstance - Optional reference to the overall evolutionary engine
   *                       instance (may be undefined in headless modes). Use for
   *                       advanced telemetry only; do not mutate the engine.
   * @returns void
   *
   * Notes:
   * - Implementations should avoid heavy synchronous work: if rendering is
   *   expensive consider batching or yielding (requestAnimationFrame) in UI contexts.
   * - For tests, a minimal implementation is a no-op or a simple recorder:
   *   { lastUpdate: null, update(maze, result, network, generation) { this.lastUpdate = {maze,result,network,generation} } }
   */
  update(
    maze: string[],
    result: any,
    network: INetwork,
    generation: number,
    neatInstance?: any // optional Neat instance for advanced telemetry display
  ): void;
}

/**
 * Maze configuration used by the ASCII Maze evolution and simulation helpers.
 *
 * Purpose
 * - Provide a minimal, serialisable description of a 2D maze suitable for
 *   human-editable examples, automated tests, and demo pages.
 *
 * Semantics
 * - The maze is represented as an array of strings; each string is one row.
 * - All strings should have the same length (rows aligned). Rows are read from
 *   top (index 0) to bottom (last index).
 * - Typical characters used by examples:
 *   - '#' or 'X' : wall / obstacle
 *   - ' ' (space) or '.' : open floor
 *   - 'S' : start position (first occurrence often used)
 *   - 'E' or 'X' : exit / goal (first occurrence used)
 *   - other characters may be interpreted by custom fitness evaluators
 *
 * Validation notes
 * - Implementations should validate row length consistency and presence of a
 *   start/exit pair when running experiments that require them.
 * - For headless tests, a minimal maze can be a 1×N/ N×1 open corridor.
 *
 * Example
 * const sample: IMazeConfig = {
 *   maze: [
 *     "#########",
 *     "#S  .  E#",
 *     "#   #   #",
 *     "#########"
 *   ]
 * };
 *
 * @remarks
 * - This shape intentionally remains small and textual so learners can edit
 *   mazes inline in examples and tests without needing binary formats.
 */
export interface IMazeConfig {
  /**
   * The maze layout, each element is a row string. Rows should be equal length.
   *
   * @example
   * ["#####",
   *  "#S E#",
   *  "#####"]
   */
  maze: string[];
}

/**
 * Agent simulation configuration.
 *
 * Purpose
 * - Configure per-episode simulation parameters used by the maze runner and
 *   fitness evaluators. This small surface lets examples/tests control episode
 *   length, determinism, and other runtime constraints without modifying the
 *   simulation engine.
 *
 * Semantics
 * - Settings apply to a single agent episode/run (not global evolution state).
 * - Runners should honour these values deterministically when possible so tests
 *   remain reproducible.
 *
 * Typical usage
 * - For short smoke tests use small maxSteps (e.g. 50).
 * - For harder mazes or thorough evaluation increase maxSteps to allow longer
 *   trajectories before timing out (e.g. 500–2000).
 *
 * Example
 * const simCfg: IAgentSimulationConfig = { maxSteps: 200 };
 *
 * @remarks
 * - Implementations may provide additional optional fields later (e.g. stepTimeout,
 *   deterministicActionNoise) but this interface focuses on the minimal set
 *   required by educational examples.
 */
export interface IAgentSimulationConfig {
  /**
   * Maximum number of discrete steps the agent is allowed to take in a single
   * episode before the runner terminates the simulation and returns a timeout
   * / failure result.
   *
   * Runners should:
   * 1. Count only environment steps that change agent state (do not double-count internal bookkeeping).
   * 2. Terminate the episode when this limit is reached and report progress/fitness.
   *
   * @example
   * // allow up to 500 steps for a difficult maze
   * { maxSteps: 500 }
   */
  maxSteps: number;
}

/**
 * Configuration options for the evolutionary algorithm used in the ASCII Maze demos.
 *
 * Purpose
 * - Collect tunable parameters that control population, stopping criteria,
 *   optional refinement phases, telemetry, and persistence. This object is
 *   passed to the evolution runner so examples, demos and tests can customise
 *   behaviour without touching algorithm code.
 *
 * Design notes
 * - Keep sensible defaults in the runner; consumers should override only the
 *   values they need for experiments.
 * - Booleans toggle optional features (Baldwinian/Lamarckian refinements,
 *   deterministic RNG, telemetry), numeric options tune population/termination.
 *
 * Typical workflow
 * 1. For quick experiments use small popSize (e.g. 50) and low maxGenerations (e.g. 200).
 * 2. For reproducible comparisons set randomSeed and deterministic=true.
 * 3. Enable persistEvery/persistTopK to snapshot progress for later analysis.
 *
 * Example
 * const cfg: IEvolutionAlgorithmConfig = {
 *   popSize: 150,
 *   allowRecurrent: true,
 *   lamarckianIterations: 10,
 *   deterministic: true,
 *   persistEvery: 50,
 *   maxGenerations: 1000
 * };
 *
 * @remarks
 * - This interface is intentionally descriptive to help learners understand
 *   what each toggle means and to encourage reproducible experiments.
 */
export interface IEvolutionAlgorithmConfig {
  /** Allow recurrent connections in genomes (networks may be stateful). */
  allowRecurrent?: boolean;
  /** Population size used each generation. Larger values increase search breadth but cost CPU. */
  popSize?: number;
  /** Maximum number of generations without improvement before considering stagnation. */
  maxStagnantGenerations?: number;
  /** Minimum measurable progress (fitness delta) required to count as improvement. */
  minProgressToPass?: number;
  /** Safety cap on total generations. Set to limit wall-clock runs in demos/tests. */
  maxGenerations?: number;
  /**
   * If true, ignore stagnation/generation caps and only stop when the agent
   * actually solves the maze (success flag true and progress threshold met).
   */
  stopOnlyOnSolve?: boolean;
  /**
   * If true (default historically), the engine sets window.asciiMazePaused=true
   * after a solution is found. Browser demos may disable this to continue runs.
   */
  autoPauseOnSolve?: boolean;
  /** Optional numeric seed for RNG. Use with deterministic=true for reproducible runs. */
  randomSeed?: number;
  /** Optionally provide an initial population of networks instead of random seeding. */
  initialPopulation?: INetwork[];
  /** Optionally supply a known-good starting network (copied/cloned by the engine). */
  initialBestNetwork?: INetwork;
  /** Per-individual local refinement iterations (Baldwinian/Lamarckian style). */
  lamarckianIterations?: number;
  /** Subsample size used for refinement training patterns to speed up Lamarckian steps. */
  lamarckianSampleSize?: number;
  // Adaptive simplify/pruning phase triggers
  /** Generations of low improvement that trigger simplify/prune mode. */
  plateauGenerations?: number;
  /** Minimum fitness delta to consider a generation an improvement (default ~1e-6). */
  plateauImprovementThreshold?: number;
  /** Number of generations to remain in simplify/prune mode once triggered. */
  simplifyDuration?: number;
  /**
   * Fraction (0-1) of the weakest connections to prune each simplify generation.
   * Use small values (0.01–0.1) to avoid destabilising behaviour.
   */
  simplifyPruneFraction?: number;
  /** Pruning heuristic: 'weakWeight' or prefer pruning recurrent links first. */
  simplifyStrategy?: 'weakWeight' | 'weakRecurrentPreferred';
  // Persistence
  /** Save a snapshot every N generations (set to undefined to disable). */
  persistEvery?: number;
  /** Directory path where snapshots will be saved (demo/runtime provided). */
  persistDir?: string;
  /** Number of top genomes to persist when saving snapshots. */
  persistTopK?: number;
  // Dynamic population growth controls
  /** If true, allow the engine to expand population size dynamically on plateaus. */
  dynamicPopEnabled?: boolean;
  /** Upper bound for dynamically expanded population. */
  dynamicPopMax?: number;
  /** Interval (generations) between population expansion events when enabled. */
  dynamicPopExpandInterval?: number;
  /** Multiplicative factor to expand the population when a growth event occurs. */
  dynamicPopExpandFactor?: number;
  /** Slack (fitness) tolerance used to decide when plateau-driven expansion should kick in. */
  dynamicPopPlateauSlack?: number;
  /**
   * If true, enables deterministic RNG seeding. If randomSeed is omitted a fixed
   * fallback seed will be used so runs are reproducible.
   */
  deterministic?: boolean;
  /**
   * Optional generation interval to trigger memory compaction (removal of
   * disabled connections). Default value is handled by the engine.
   */
  memoryCompactionInterval?: number;
  /** If true, reduce telemetry math to lower CPU usage (skip kurtosis, higher moments). */
  telemetryReduceStats?: boolean;
  /** If true, disable higher-cost per-generation telemetry (entropy, diversity, etc.). */
  telemetryMinimal?: boolean;
  /** If true, skip Baldwinian refinement phase (extra training on the fittest). */
  disableBaldwinianRefinement?: boolean;
}

/**
 * Context passed to a fitness evaluator when scoring a network on a maze.
 *
 * Purpose
 * - Provide all derived and raw information needed to deterministically evaluate
 *   a single agent episode in a maze so fitness functions can be simple and
 *   focused (no hidden IO or global state required).
 *
 * Semantics & conventions
 * - Coordinate system: row-major arrays with origin at the top-left of the maze.
 *   Positions are expressed as a readonly tuple [rowIndex, colIndex] where both
 *   indices are zero-based.
 * - Encodings: the numeric encoding of cells (encodedMaze) is intentionally
 *   engine-specific. Common encoders map open floor to 0 and walls/obstacles to 1,
 *   but evaluators must consult the caller's encoder or accept common defaults.
 * - Determinism: values in this context should not be mutated by evaluators.
 *
 * Typical responsibilities of a fitness evaluator using this context
 * 1. Use encodedMaze and distanceMap (when present) to compute heuristics such as
 *    distance-to-goal, reachable area, or dead-ends.
 * 2. Simulate or replay agent steps (using agentSimConfig limits) and combine
 *    behavioural metrics into a single scalar fitness score.
 * 3. Prefer read-only access; when caching is required produce ephemeral copies.
 *
 * @example
 * // Example shape of a simple context:
 * const ctx: IFitnessEvaluationContext = {
 *   encodedMaze: [
 *     [1,1,1,1,1],
 *     [1,0,0,0,1],
 *     [1,0,1,0,1],
 *     [1,0,0,2,1],
 *     [1,1,1,1,1]
 *   ],
 *   startPosition: [1,1],
 *   exitPosition: [3,3],
 *   agentSimConfig: { maxSteps: 200 },
 *   // distanceMap optional: same dimensions as encodedMaze; distance in steps to exit
 * };
 *
 * @remarks
 * - This interface is tailored for educational examples: keep data small and
 *   explicit so learners can inspect values while debugging fitness functions.
 */
export interface IFitnessEvaluationContext {
  /**
   * Row-major numeric representation of the maze.
   *
   * - Shape: encodedMaze.length === number of rows; encodedMaze[0].length === number of columns.
   * - Cell codes: encoding is chosen by the maze encoder. Common conventions:
   *   - 0: open/traversable
   *   - 1: wall/obstacle
   *   - 2: goal/exit (optional if exitPosition is provided separately)
   *
   * Use-case tips:
   * - Iteration order is [row][col] (y, x). When mapping to x/y visuals swap indices as needed.
   */
  encodedMaze: number[][];

  /**
   * Start position for the agent as a readonly tuple [rowIndex, colIndex].
   *
   * - First element: row (0 = top).
   * - Second element: column (0 = left).
   *
   * Example: [2, 4] means third row, fifth column.
   */
  startPosition: readonly [number, number];

  /**
   * Exit/goal position for the episode as a readonly tuple [rowIndex, colIndex].
   *
   * - Evaluators commonly compute Manhattan or shortest-path distance between
   *   the agent and this coordinate when deriving progress-based fitness.
   */
  exitPosition: readonly [number, number];

  /**
   * Simulation controls for a single agent episode.
   *
   * - Provides deterministic limits such as maxSteps so evaluators and runners
   *   behave reproducibly in tests and examples.
   * - Evaluators should respect maxSteps when simulating trajectories.
   */
  agentSimConfig: IAgentSimulationConfig;

  /**
   * Optional cached distance map (same dimensions as encodedMaze).
   *
   * - Typical content: non-negative step counts indicating the shortest-path
   *   distance from each cell to the exitPosition. Unreachable cells may be
   *   represented by Infinity or a large sentinel value.
   * - Purpose: supply fast heuristics (e.g., progress-to-goal) without recomputing
   *   an expensive BFS on every fitness evaluation. Consumers must treat this
   *   as read-only.
   *
   * Example usage:
   * const dist = context.distanceMap?.[agentRow]?.[agentCol] ?? Infinity;
   */
  distanceMap?: number[][]; // optional cached distance map for performance
}

/**
 * Signature for a fitness evaluator used to score a network on a single maze episode.
 *
 * Purpose
 * - Provide a deterministic, side-effect-free function that maps a network's
 *   behaviour (when run in a given environment context) to a single numeric
 *   fitness value used by evolutionary algorithms.
 *
 * Responsibilities & conventions
 * 1. Determinism: for the same `network` state and `context` inputs the function
 *    should return the same numeric score. If stochasticity is needed, it must
 *    be driven by values inside `context` (so tests/experiments can reproduce runs).
 * 2. Read-only: evaluators must not mutate `network` or `context`. If temporary
 *    changes are required (e.g. calling `network.clear()`), operate on a clone
 *    or ensure the original is restored before returning.
 * 3. Scalar meaning: higher numbers typically denote better performance unless
 *    the caller documents otherwise. The evaluator should document its range
 *    and interpretation (e.g., [0..1], arbitrary positive rewards, penalties).
 *
 * Implementation tips
 * - Use `context.distanceMap` when available to compute progress-to-goal quickly.
 * - Combine multiple signals (distance-to-goal, steps-used, collisions) into a
 *   single scalar using a consistent weighting scheme and document it.
 * - Keep fitness values numerically stable (avoid huge magnitudes) so aggregation
 *   statistics remain meaningful to telemetry code.
 *
 * @returns {number} A finite numeric fitness score. By default larger values indicate better performance;
 *          concrete evaluators should document score range and interpretation.
 *
 * @example
 * // Minimal evaluator structure:
 * const evaluator: FitnessEvaluatorFn = (network, context) => {
 *   // 1) use context to build inputs
 *   // 2) run network.activate(inputs) to get actions
 *   // 3) simulate agent up to context.agentSimConfig.maxSteps
 *   // 4) compute and return a single numeric score
 *   return 0; // replace with real computation
 * };
 */
export type FitnessEvaluatorFn = (
  /** Read-only episode context (see IFitnessEvaluationContext). */
  network: INetwork,
  /**
   * The candidate network to evaluate.
   *    Expected contract:
   *      - Call network.activate(inputs) for inference; do not assume concrete internals.
   *      - Optional helpers: network.clear(), network.clone(), network.propagate() may exist --
   *        if used, operate on a clone or restore state to avoid mutating the original.
   */
  context: IFitnessEvaluationContext
) => number;

/**
 * Reporting configuration used to control logging, dashboard updates and UI pacing
 * for an evolution run.
 *
 * Purpose:
 * - Give experiment authors a small, well-documented surface to control
 *   telemetry frequency, the dashboard implementation and an optional label
 *   so runs can be identified in logs or persisted snapshots.
 *
 * Design notes:
 * - Documentation is intentionally explicit so IDEs (VSCode) surface property
 *   descriptions inline while composing experiments, helping learners understand
 *   intent without reading library source.
 *
 * @example
 * const reporting: IReportingConfig = {
 *   logEvery: 10,
 *   dashboardManager: myDashboard,
 *   label: "small-experiment",
 *   paceEveryGeneration: true
 * };
 */
export interface IReportingConfig {
  /**
   * How frequently (in generations) to emit logs or telemetry updates.
   *
   * Behaviour:
   * - If undefined, the runner may fall back to a sensible default (for example: 1).
   * - When set to N > 0 the runner should produce a log/telemetry entry every Nth generation.
   *
   * @example
   * // Log every 5 generations
   * { logEvery: 5 }
   */
  logEvery?: number;

  /**
   * Dashboard manager instance responsible for receiving per-generation updates.
   *
   * Responsibilities of implementations:
   * 1. Display a human-readable maze and the agent result snapshot.
   * 2. Optionally visualise or serialize the network structure (read-only).
   * 3. Avoid mutating the supplied objects and avoid expensive synchronous work.
   *
   * The runner will call `dashboardManager.update(maze, result, network, generation, neatInstance?)`
   * with a snapshot for presentation or recording.
   */
  dashboardManager: IDashboardManager;

  /**
   * Optional human-readable label identifying this run in logs, snapshot files
   * and UI elements.
   *
   * Use-cases:
   * - Distinguish runs when persisting multiple experiments to the same directory.
   * - Tag telemetry streams for easier filtering in dashboards.
   *
   * @example
   * { label: "pop150_seed42" }
   */
  label?: string;

  /**
   * When true, the evolution loop yields control to the browser after each
   * generation (for example via requestAnimationFrame) to keep the UI
   * responsive during long-running experiments.
   *
   * Notes:
   * - Default may be false in headless or CI environments.
   * - Enabling this will affect wall-clock timing of runs and should be used
   *   primarily for interactive demos.
   */
  paceEveryGeneration?: boolean;
}

/**
 * Main options for running a single maze-evolution experiment.
 *
 * Purpose:
 * - Aggregate all tunable inputs required to configure one run of the ASCII
 *   maze evolutionary system (maze, simulation limits, algorithm hyper-parameters,
 *   reporting and optional evaluator hooks).
 *
 * Design notes:
 * - This object is intentionally explicit and documented so learners can adjust
 *   experiments from examples or IDE autocompletion without reading implementation code.
 * - Fields that are optional may be left undefined to use sensible defaults provided by the runner.
 *
 * Example:
 * const opts: IRunMazeEvolutionOptions = {
 *   mazeConfig: { maze: ["#####", "#S E#", "#####"] },
 *   agentSimConfig: { maxSteps: 200 },
 *   evolutionAlgorithmConfig: { popSize: 100, maxGenerations: 500, deterministic: true },
 *   reportingConfig: { logEvery: 5, dashboardManager: myDashboard, paceEveryGeneration: true },
 *   fitnessEvaluator: myEvaluator,
 *   cancellation: { isCancelled: () => false }
 * };
 */
export interface IRunMazeEvolutionOptions {
  /**
   * Minimal, human-editable maze definition used for the experiment.
   *
   * - Provides the ASCII layout used for every episode/evaluation in the run.
   * - Runners should validate row lengths and presence of start/exit when required.
   */
  mazeConfig: IMazeConfig;

  /**
   * Per-episode simulation controls.
   *
   * - Controls deterministic limits such as maxSteps so single-episode behaviour
   *   is reproducible for tests and examples.
   * - The runner must honour these limits when simulating agents for fitness evaluation.
   */
  agentSimConfig: IAgentSimulationConfig;

  /**
   * All tunable hyper-parameters and behavioural flags for the evolutionary algorithm.
   *
   * - Controls population size, stopping criteria, determinism, refinement iterations,
   *   persistence settings and other algorithm-level options.
   * - Use the runner defaults for any omitted optional fields.
   */
  evolutionAlgorithmConfig: IEvolutionAlgorithmConfig;

  /**
   * Reporting and telemetry configuration.
   *
   * - Controls logging cadence, dashboard wiring and optional UI pacing.
   * - The dashboardManager will receive per-generation snapshots for presentation or recording.
   */
  reportingConfig: IReportingConfig;

  /**
   * Optional fitness evaluator function used to score candidate networks on the maze.
   *
   * - If omitted the runner may fall back to a library-provided default evaluator.
   * - When provided the function must be deterministic for given inputs and must not mutate
   *   the provided network or context objects (operate on clones if necessary).
   *
   * @example
   * // A small evaluator that prefers shorter path-to-goal and fewer collisions
   * const fn: FitnessEvaluatorFn = (network, context) => { return score; };
   */
  fitnessEvaluator?: FitnessEvaluatorFn;

  /**
   * Optional cooperative cancellation token.
   *
   * - Runner should periodically call `cancellation.isCancelled()` and exit early
   *   (returning the best-so-far result) when it returns true.
   * - This is a lightweight, platform-agnostic cancellation protocol useful in tests.
   */
  cancellation?: { isCancelled: () => boolean };

  /**
   * Optional standard AbortSignal (Web/DOM-style).
   *
   * - When provided the runner should periodically inspect `signal.aborted` and
   *   terminate gracefully, returning partial results if applicable.
   * - This is the modern, standardized cancellation primitive for browser/node environments.
   */
  signal?: AbortSignal;
}

/**
 * Visualization node used by ASCII/graph renderers to present a network node.
 *
 * Purpose:
 * - Provide a compact, editor-friendly descriptor for a single neuron/node so
 *   UI components and examples can show topology, activations and simple stats.
 *
 * Design notes:
 * - Fields are intentionally small and descriptive so learners can inspect nodes
 *   in tooltips or serialized snapshots without needing the full Network object.
 *
 * @example
 * const node: IVisualizationNode = {
 *   uuid: "node-123",
 *   id: 0,
 *   type: "input",
 *   activation: 0.0,
 *   bias: 0.1,
 *   label: "x0"
 * };
 */
export interface IVisualizationNode {
  /**
   * Stable identifier for this visualization node (unique within the visualisation).
   *
   * - Use a string UUID or any unique string that ties back to the network's node.
   * - Useful for diffing frames or matching gater/connection endpoints.
   */
  uuid: string;

  /**
   * Numeric index useful for compact lookups or ordering in arrays.
   *
   * - Not required to be globally unique across different networks; unique within this network snapshot is sufficient.
   */
  id: number;

  /**
   * Semantic type of the node (for example: "input" | "hidden" | "output" | "constant").
   *
   * - Visualisers can use this to apply different styling or grouping.
   */
  type: string;

  /**
   * Current activation value produced by the node (post-squash).
   *
   * - Range depends on the node's activation function. Visualisers commonly
   *   map this value to color/height in charts.
   */
  activation: number;

  /**
   * Optional bias term associated with the node (if available).
   *
   * - Visualisers may render this as a small numeric badge or tooltip detail.
   * - When undefined the node has no reported bias in this snapshot.
   */
  bias?: number;

  /**
   * When true, this node represents an averaged/aggregate node (for visual summaries).
   *
   * - Example: showing mean activations across an ensemble or a temporal average.
   * - Visualisers may draw these nodes differently (dashed outline, faded color).
   */
  isAverage?: boolean;

  /**
   * Number of samples included in the averaged value when `isAverage` is true.
   *
   * - Helps consumers interpret how many underlying values were combined.
   * - Undefined when `isAverage` is false or when not applicable.
   */
  avgCount?: number;

  /**
   * Optional short human-readable label for the node (e.g. "x0", "hidden-2").
   *
   * - Primarily used in tooltips and exported diagrams to make topology understandable.
   */
  label?: string;
}

/**
 * Visualization connection used by ASCII/graph renderers to present an edge between two nodes.
 *
 * Purpose:
 * - Provide a compact, editor-friendly descriptor for a network connection so UI components
 *   and examples can show topology, weights and gating information.
 *
 * Design notes:
 * - Fields intentionally mirror the minimal information needed by visualisers: endpoints,
 *   optional gater, numeric weight and enabled state. This keeps snapshots small while
 *   providing useful tooltip content in IDEs and UIs.
 *
 * @example
 * const conn: IVisualizationConnection = {
 *   fromUUID: "node-1",
 *   toUUID: "node-2",
 *   gaterUUID: null,
 *   weight: -0.42,
 *   enabled: true
 * };
 */
export interface IVisualizationConnection {
  /**
   * UUID of the source node for this connection (matches IVisualizationNode.uuid).
   *
   * - Used to map the connection to its visual source endpoint.
   */
  fromUUID: string;

  /**
   * UUID of the destination node for this connection (matches IVisualizationNode.uuid).
   *
   * - Used to map the connection to its visual target endpoint.
   */
  toUUID: string;

  /**
   * Optional UUID of a gater node that modulates this connection, or null when none.
   *
   * - When present the gater's activation is typically used to scale or gate the connection.
   * - Visualisers can highlight gated connections or show the gater relationship in tooltips.
   */
  gaterUUID?: string | null;

  /**
   * Numeric weight of the connection.
   *
   * - Sign indicates excitatory (positive) or inhibitory (negative) influence.
   * - Magnitude can be visualised as line thickness; visualisers should clamp/scale values for display.
   */
  weight: number;

  /**
   * Whether this connection is currently enabled (active) in the network topology.
   *
   * - Disabled connections are typically drawn faded, dashed, or omitted.
   */
  enabled: boolean;
}

/**
 * Type representing a node activation (squash) function with optional metadata.
 *
 * Purpose:
 * - Document the runtime contract for activation functions used by network nodes
 *   and provide optional human-friendly metadata that UI/tooling can surface.
 *
 * Call signature:
 * - (input: number, derivate?: boolean) => number
 *   - `input`: numeric input to the activation function.
 *   - `derivate` (optional): when true the function should return the derivative
 *     of the activation with respect to `input`. Implementations may ignore
 *     this parameter if derivatives are not supported.
 *
 * Semantics:
 * - When `derivate` is omitted or false the function returns the activated value.
 * - When `derivate` is true the function returns the derivative value (useful for backpropagation).
 * - Implementations should return finite numbers and avoid throwing for typical numeric inputs.
 *
 * Optional metadata properties:
 * - `name?: string` - a short human-friendly label for UI/tooling (e.g. "tanh", "relu").
 * - `originalName?: string` - canonical/original identifier (useful for serialization or debugging).
 *
 * Example:
 * const relu: ActivationFunctionWithName = (x, derivate = false) =>
 *   derivate ? (x > 0 ? 1 : 0) : Math.max(0, x);
 * relu.name = "relu";
 * relu.originalName = "ReLU";
 */
export type ActivationFunctionWithName = ((
  input: number,
  derivate?: boolean
) => number) & {
  name?: string;
  originalName?: string;
};

/**
 * Structure describing a single network node for visualization, serialization and tooling.
 *
 * Purpose:
 * - Provide a compact, self-documenting snapshot of a node's static and runtime
 *   metadata so examples, visualisers and tests can inspect networks without
 *   depending on a concrete Network class.
 *
 * Key ideas:
 * - Keep shape small and serialisable: useful for JSON snapshots, tooltips and diagrams.
 * - Distinguish static descriptors (type, name, index) from runtime values (activation).
 *
 * Example:
 * const node: INodeStruct = {
 *   type: "hidden",
 *   bias: -0.1,
 *   squash: tanhFunction,
 *   activation: 0.234,
 *   name: "hidden-3",
 *   index: 12
 * };
 */
export interface INodeStruct {
  /**
   * Semantic node type. Typical values: 'input', 'hidden', 'output', 'constant'.
   *
   * - Visualisers and serialization code can use this to group or style nodes.
   */
  type: string;

  /**
   * Optional numeric bias term applied at the node's aggregation stage.
   *
   * - When omitted the node is treated as having no explicit bias in this snapshot.
   */
  bias?: number;

  /**
   * Optional activation (squash) function used by this node.
   *
   * - Provides a callable used for inference/visualisation and optional name metadata.
   * - May be undefined for lightweight snapshots where function references are not serialised.
   */
  squash?: ActivationFunctionWithName;

  /**
   * Runtime activation value produced by the node (post-squash).
   *
   * - Useful for visualisers, debugging and test assertions that inspect network behaviour.
   * - Range depends on the chosen activation function.
   */
  activation?: number;

  /**
   * Optional human-friendly label for the node (e.g. "x0", "hidden-2").
   *
   * - Used in tooltips, exported diagrams and debug logs to make topology easier to read.
   */
  name?: string;

  /**
   * Optional numeric index for compact lookups or ordering.
   *
   * - Conventionally unique within a single network snapshot; not necessarily globally unique.
   * - Useful when mapping connections that refer to numeric indices rather than uuids.
   */
  index?: number;

  /**
   * Catch-all for implementation-specific extras.
   *
   * - Allows concrete Network implementations to expose additional fields without
   *   breaking the generic interface. Prefer explicit properties where feasible.
   */
  [key: string]: any;
}

/**
 * Lightweight, educational abstraction of a neural network used by the ASCII
 * Maze examples. This interface is intentionally permissive so it can describe
 * multiple concrete implementations (the library's Network class, clones,
 * simplified mocks for tests, or externally serialized networks).
 *
 * The goal of these comments is to make it easy for learners to:
 * - see the minimal runtime surface required by the maze demos,
 * - understand typical semantics of each member (inputs/outputs, side-effects),
 * - know which methods are optional for simple inference vs. training/debugging.
 *
 * Typical usage:
 *  - For inference only: implement activate(inputs) -> outputs.
 *  - For training/refinement: also implement propagate(...) and optionally clone().
 *  - For visualization/analysis: provide nodes/connections arrays and input/output descriptors.
 */
export interface INetwork {
  /**
   * Performs a forward pass of the network using the provided input values and
   * returns the resulting output values.
   *
   * Semantics:
   * - inputs: an array of numbers matching the network's expected input size.
   * - returns: an array of numbers representing the network's outputs (one or more).
   *
   * Typical responsibilities:
   * - apply input values to input nodes,
   * - propagate activations through layers/recurrent links according to the
   *   network's topology and activation functions,
   * - return the values produced by the output nodes.
   *
   * This method must be synchronous and pure with respect to its return value
   * (side-effects like internal state updates are allowed but callers expect
   * a deterministic output for the same inputs unless stateful recurrence exists).
   */
  activate: (inputs: number[]) => number[];

  /**
   * Optional supervised-learning update method (backpropagation style).
   *
   * Parameters:
   * - rate: learning rate (how large weight updates should be),
   * - momentum: momentum term for weight updates (typical 0.0 - 1.0),
   * - update: if true, actually apply weight updates; if false, compute but do not apply,
   * - target: array of expected output values used to compute output-layer errors.
   *
   * Implementations:
   * - Concrete Network classes may implement this to support gradient-based
   *   refinement (Baldwinian/Lamarckian steps in evolution).
   * - Optional on the interface because many evolutionary workflows only need
   *   activation (inference).
   */
  propagate?: (
    rate: number,
    momentum: number,
    update: boolean,
    target: number[]
  ) => void;

  /**
   * Optional: clears internal dynamic state such as recurrent activations or
   * transient caches so the network behaves as if it's freshly initialised.
   *
   * Use when running sequences or episodic environments to avoid state leakage
   * between episodes.
   */
  clear?: () => void;

  /**
   * Optional: produce a deep copy of the network object.
   *
   * Semantics:
   * - The clone should be independent: mutations to the clone must not affect the original.
   * - Clones are useful for experimentation, evaluation, or storing snapshots.
   */
  clone?: () => INetwork;

  /**
   * Optional structural description of the network's nodes.
   *
   * Typical contents:
   * - Each node entry describes its type ('input'|'hidden'|'output'|'constant'),
   *   bias, activation function reference (squash), and runtime activation value.
   * - Useful for visualization, introspection, and deterministic reproducible
   *   comparisons between networks.
   *
   * Note: implementations may omit nodes to save memory; visualisers should
   * handle missing or partial descriptions.
   */
  nodes?: INodeStruct[];

  /**
   * Optional structural description of the network's connections.
   *
   * Each connection typically contains:
   * - from/to: references to INodeStruct describing the endpoints,
   * - weight: numeric weight applied during activation propagation,
   * - gater: optional node that gates this connection (null when unused),
   * - enabled: whether this connection is currently active.
   *
   * Use-cases:
   * - visualization of topology and weights,
   * - pruning or serialization routines that iterate connections,
   * - analysis of recurrent links vs feed-forward paths.
   */
  connections?: {
    from: INodeStruct;
    to: INodeStruct;
    weight: number;
    gater?: INodeStruct | null;
    enabled?: boolean;
    [key: string]: any;
  }[];

  /**
   * Optional description of the input "layer".
   *
   * Can be either:
   * - a number: the expected size of the input (used for validation/documentation),
   * - an array of INodeStruct describing input nodes (used by visualisers or low-level code).
   */
  input?: number | INodeStruct[];

  /**
   * Optional description of the output "layer".
   *
   * Same conventions as `input`.
   */
  output?: number | INodeStruct[];

  // Add any other methods/properties from the concrete Network class that are used by the asciiMaze example
}

/**
 * Represents the outcome of a single logical step, phase or checkpoint in the evolution process.
 *
 * Purpose:
 * - Provide a small, stable shape that runners and reporters can use to communicate
 *   incremental results (for example per-generation or per-refinement-phase snapshots).
 *
 * Semantics:
 * - `success` is a boolean flag indicating whether the step achieved its primary goal
 *   (for example: a candidate solved the maze or a phase completed without error).
 * - `progress` is a numeric measure of improvement or completion. Its exact scale is
 *   context-dependent (for example: normalized [0..1], raw fitness delta, number of improved individuals).
 *   Consumers should consult the caller's conventions; treat the field as monotonic within a phase when possible.
 *
 * Usage notes:
 * - Reporters and dashboards can use `success` to highlight terminal events and `progress`
 *   to render progress bars or compute trend lines.
 * - When `success` is true, `progress` typically represents the achieved score; when false it may be a relative improvement metric.
 *
 * @example
 * // A generation that produced an improved best individual:
 * const step: IEvolutionStepResult = { success: false, progress: 0.034 };
 */
export interface IEvolutionStepResult {
  /**
   * True when the reported step achieved its primary success condition.
   *
   * - Example meanings: a solution was found, a refinement phase completed successfully, or a save completed.
   */
  success: boolean;

  /**
   * Numeric progress indicator associated with the step.
   *
   * - Interpretation depends on the caller: commonly a positive fitness value, a relative delta, or a normalized fraction [0..1].
   * - Should be finite (not NaN/Infinity); use caller conventions to interpret magnitude/sign.
   */
  progress: number;
}

/**
 * Represents the overall result of an evolution function call.
 *
 * Purpose:
 * - Provide a small, explicit container that callers receive when an evolution
 *   run completes or is terminated. This wrapper makes it clear which step
 *   result represents the final/returned summary so reporters and callers can
 *   consistently read outcome data.
 *
 * Semantics:
 * - `finalResult` typically holds the best-known IEvolutionStepResult at the
 *   time the run finished (normal completion, early stop, or cancellation).
 * - Runners should populate `finalResult` with a succinct summary that is safe
 *   to serialize and display (do not embed large transient structures).
 *
 * Interpretation notes:
 * - Consumers can use `finalResult.success` to decide whether the run met its
 *   main goal (for example: a solution was found).
 * - `finalResult.progress` contains the numeric progress/score according to
 *   the experiment's convention (see IEvolutionStepResult). Its scale may be
 *   normalized ([0..1]) or raw fitness/metric values depending on the runner.
 *
 * Example:
 * const result: IEvolutionFunctionResult = {
 *   finalResult: { success: true, progress: 0.987 }
 * };
 */
export interface IEvolutionFunctionResult {
  /**
   * The final summary step result produced by the evolution run.
   *
   * - Should be set by the runner to reflect the best/terminal IEvolutionStepResult.
   * - Useful for UI, logging, persistence and programmatic checks after the run returns.
   */
  finalResult: IEvolutionStepResult;
}
