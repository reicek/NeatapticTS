/**
 * Shared type definitions for the NGE collective multi-agent system (Phase G).
 *
 * These types are the shared vocabulary for the shared-field, evaluation, and metrics
 * sub-modules. Import individual sub-modules directly for their implementation functions.
 *
 * ## Type relationships
 *
 * ```mermaid
 * graph TD
 *   SF["SharedField\nwidth · height · cells: Float32Array"] -->|"bound into"| CEC
 *   CEC["CollectiveEvaluationContext\nagentCount · generationTick · field"] -->|"passed to"| CTR
 *   CTR["CollectiveTickResult\nagentFitness[ ] · evaluationOrder[ ]"]
 *   AE["AgentEvaluator\n(agentIndex, field) => number"] -->|"invoked by"| CEC
 *   OSP["OpponentSnapshotPool\ncapacity · snapshots[ ]"] -->|"contains"| OS
 *   OS["OpponentSnapshot\nagentId · snapshot · frozenAt"]
 * ```
 *
 * ### Key invariants
 *
 * - `SharedField.cells` is row-major: cell `(x, y)` maps to index `y × width + x`.
 * - `CollectiveEvaluationContext.generationTick` starts at `0` and is immutable within a tick;
 *   it advances only via `resetCollectiveEvaluationState`.
 * - `CollectiveTickResult.agentFitness` and `evaluationOrder` are parallel arrays of equal length.
 * - `OpponentSnapshot.snapshot` is a deep-cloned, frozen payload — post-registration mutations
 *   to the original object are never reflected.
 */

/**
 * A row-major Float32Array-backed 2D pheromone/signal field shared across all agents
 * in one collective evaluation tick.
 *
 * Cell `(x, y)` maps to flat index `y * width + x`.
 * All cells are initialized to `0` on creation.
 *
 * @example
 * ```ts
 * const field: SharedField = createSharedField(10, 10);
 * ```
 */
export interface SharedField {
  /** Number of columns in the 2D grid. */
  width: number;
  /** Number of rows in the 2D grid. */
  height: number;
  /**
   * Row-major backing store. Cell `(x, y)` maps to index `y * width + x`.
   * All values are initialized to `0` on field creation.
   */
  cells: Float32Array;
}

/**
 * Persistent state carried across collective evaluation ticks.
 * Returned by `createCollectiveEvaluationContext` and updated via `resetCollectiveEvaluationState`.
 */
export interface CollectiveEvaluationContext {
  /** Total number of agents participating in collective evaluation. */
  agentCount: number;
  /** Number of completed evaluation generations. Starts at `0`. */
  generationTick: number;
  /** Live shared field visible to all agent evaluators within the current tick. */
  field: SharedField;
}

/**
 * Immutable result produced by one collective evaluation tick via `runCollectiveEvaluationTick`.
 */
export interface CollectiveTickResult {
  /** Fitness value returned by each agent evaluator, ordered by agent index. */
  agentFitness: number[];
  /** Sequence in which agents were evaluated within the tick. */
  evaluationOrder: number[];
}

/**
 * A frozen point-in-time snapshot of an opponent agent for rolling tournament evaluation.
 * Payloads are deep-cloned at registration time so post-registration mutations are not reflected.
 */
export interface OpponentSnapshot {
  /** Stable identifier for the agent whose state was captured. */
  agentId: string;
  /** Immutable deep clone of the agent payload at registration time. */
  snapshot: Readonly<Record<string, unknown>>;
  /** Generation tick at which the snapshot was registered. */
  frozenAt: number;
}

/**
 * Rolling circular buffer of opponent snapshots with a fixed capacity.
 * The oldest snapshot is evicted in FIFO order when the pool reaches capacity.
 */
export interface OpponentSnapshotPool {
  /** Maximum number of snapshots retained at any time. */
  capacity: number;
  /** Ordered list of stored snapshots, oldest first. */
  snapshots: OpponentSnapshot[];
}

/**
 * Evaluator function invoked once per agent per collective tick.
 *
 * Because `writeCell` mutates the shared field in-place, sequential evaluators
 * within the same tick can observe writes committed by prior agents.
 *
 * @param agentIndex - Zero-based index of the agent being evaluated.
 * @param field - Live shared field; sequential evaluators see writes from prior agents.
 * @returns Scalar fitness value for the evaluated agent.
 */
export type AgentEvaluator = (agentIndex: number, field: SharedField) => number;
