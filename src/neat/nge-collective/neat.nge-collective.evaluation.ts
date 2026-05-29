import { clearField } from './neat.nge-collective.shared-field';
import type {
  AgentEvaluator,
  CollectiveEvaluationContext,
  CollectiveTickResult,
  SharedField,
} from './neat.nge-collective.types';

export type { CollectiveEvaluationContext, CollectiveTickResult };

/**
 * @module neat.nge-collective.evaluation
 *
 * Sequential multi-agent evaluation orchestration for NGE Phase G.
 *
 * This module wires together the `SharedField` substrate and a list of `AgentEvaluator`
 * functions into a single deterministic evaluation tick. The design is intentionally
 * sequential: evaluators are called in declared order `[0, 1, ..., N-1]` so that earlier
 * agents can deposit signals that later agents read within the same tick.
 *
 * ## Tick lifecycle
 *
 * ```mermaid
 * sequenceDiagram
 *   participant Caller
 *   participant Tick as runCollectiveEvaluationTick
 *   participant Eval0 as AgentEvaluator[0]
 *   participant Eval1 as AgentEvaluator[1]
 *   participant Field as SharedField
 *
 *   Caller->>Tick: context, evaluators
 *   Tick->>Eval0: (0, field)
 *   Eval0->>Field: writeCell(field, x, y, value)
 *   Eval0-->>Tick: fitness₀
 *   Tick->>Eval1: (1, field)
 *   Eval1->>Field: readCell(field, x, y)  ← sees Eval0's write
 *   Eval1-->>Tick: fitness₁
 *   Tick-->>Caller: { agentFitness, evaluationOrder }
 * ```
 *
 * ## Generation reset
 *
 * `resetCollectiveEvaluationState` returns a new `CollectiveEvaluationContext` with the
 * generation tick incremented by one and the shared field zeroed via `clearField`. The
 * caller should apply `applyDecay` / `applyDiffusion` on the field *before* calling reset
 * if persistence-across-generations semantics are needed.
 */

/**
 * Creates a new collective evaluation context with an initial generation tick of zero.
 *
 * @param agentCount - Number of agents participating in collective evaluation.
 * @param field - Shared field visible to all agent evaluators within the current tick.
 * @returns A new `CollectiveEvaluationContext` ready for the first evaluation tick.
 *
 * @example
 * ```ts
 * const field = createSharedField(10, 10);
 * const context = createCollectiveEvaluationContext(4, field);
 * // context.agentCount === 4, context.generationTick === 0
 * ```
 */
export function createCollectiveEvaluationContext(
  agentCount: number,
  field: SharedField,
): CollectiveEvaluationContext {
  return { agentCount, generationTick: 0, field };
}

/**
 * Runs one sequential collective evaluation tick.
 *
 * Evaluators are called in declared order `[0, 1, ..., N-1]`. Because the shared field is
 * passed by reference and `writeCell` mutates in-place, each evaluator observes writes
 * committed by all earlier evaluators within the same tick.
 *
 * @param context - Current collective evaluation context carrying field and agent count.
 * @param evaluators - Ordered array of evaluator functions, one per agent.
 * @returns Tick result containing per-agent fitness values and the evaluation order.
 *
 * @example
 * ```ts
 * const result = runCollectiveEvaluationTick(context, [
 *   (_idx, field) => { writeCell(field, 0, 0, 1.0); return 10; },
 *   (_idx, field) => readCell(field, 0, 0) * 5,
 * ]);
 * // result.agentFitness === [10, 5]
 * ```
 */
export function runCollectiveEvaluationTick(
  context: CollectiveEvaluationContext,
  evaluators: AgentEvaluator[],
): CollectiveTickResult {
  // Step 1: Collect fitness values in declared agent order.
  const agentFitness: number[] = [];
  const evaluationOrder: number[] = [];

  for (let agentIndex = 0; agentIndex < context.agentCount; agentIndex++) {
    const evaluator = evaluators[agentIndex];
    if (evaluator === undefined) continue;

    // Step 2: Invoke each evaluator with the live shared field reference.
    const fitness = evaluator(agentIndex, context.field);
    agentFitness.push(fitness);
    evaluationOrder.push(agentIndex);
  }

  return { agentFitness, evaluationOrder };
}

/**
 * Resets the collective evaluation state after a completed generation.
 *
 * Returns a **new** `CollectiveEvaluationContext` with the generation tick incremented by one
 * and the shared field zeroed via `clearField`. The original context is **not** mutated.
 *
 * @param context - Context from the generation that just completed.
 * @returns A new context ready for the next generation's evaluation tick.
 *
 * @example
 * ```ts
 * const nextContext = resetCollectiveEvaluationState(context);
 * // nextContext.generationTick === context.generationTick + 1
 * // all cells in nextContext.field === 0
 * ```
 */
export function resetCollectiveEvaluationState(
  context: CollectiveEvaluationContext,
): CollectiveEvaluationContext {
  // Step 1: Advance the generation tick and zero the shared field for the next generation.
  return {
    agentCount: context.agentCount,
    generationTick: context.generationTick + 1,
    field: clearField(context.field),
  };
}
