import type { IAgentSimulationConfig } from './evolutionEngine/evolutionEngine.types';
import type { INetwork } from './interfaces';

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
 * @example
 * ```ts
 * const ctx: IFitnessEvaluationContext = {
 *   encodedMaze: [
 *     [1, 1, 1, 1, 1],
 *     [1, 0, 0, 0, 1],
 *     [1, 0, 1, 0, 1],
 *     [1, 0, 0, 2, 1],
 *     [1, 1, 1, 1, 1],
 *   ],
 *   startPosition: [1, 1],
 *   exitPosition: [3, 3],
 *   agentSimConfig: { maxSteps: 200 },
 * };
 * ```
 */
export interface IFitnessEvaluationContext {
  /** Row-major numeric representation of the maze. */
  encodedMaze: number[][];

  /** Start position for the agent as a readonly tuple [rowIndex, colIndex]. */
  startPosition: readonly [number, number];

  /** Exit/goal position for the episode as a readonly tuple [rowIndex, colIndex]. */
  exitPosition: readonly [number, number];

  /** Simulation controls for a single agent episode. */
  agentSimConfig: IAgentSimulationConfig;

  /** Optional cached distance map aligned to `encodedMaze`. */
  distanceMap?: number[][];
}

/**
 * Signature for a fitness evaluator used to score a network on a single maze episode.
 *
 * @param network - The candidate network to evaluate.
 * @param context - Read-only episode context used to score the candidate.
 * @returns A finite numeric fitness score where larger values usually mean better performance.
 */
export type FitnessEvaluatorFn = (
  network: INetwork,
  context: IFitnessEvaluationContext,
) => number;
