/**
 * NGE Collective Intelligence boundary — Phase G public surface.
 *
 * This module is the entry point for all multi-agent collective evaluation primitives
 * introduced in NGE Phase G. It re-exports three cooperating sub-modules as a single
 * cohesive boundary so consumers need only one import path.
 *
 * **Design contract:** classic NEAT behavior is entirely unaffected when this module
 * is not imported. No global state is mutated at import time.
 *
 * ## Architecture
 *
 * ```mermaid
 * graph TD
 *   A["shared-field\nFloat32Array-backed 2D grid\ndecay · diffusion · read/write"] --> B
 *   B["evaluation\nCollectiveEvaluationContext\nrunCollectiveEvaluationTick\nresetCollectiveEvaluationState"] --> C
 *   C["metrics\ncomputeRoleDivergenceMetric\ncreateOpponentSnapshotPool\naddOpponentSnapshot"]
 *   B -->|"passes SharedField\nby reference"| A
 * ```
 *
 * ## Sub-modules
 *
 * ### shared-field
 * A `Float32Array`-backed 2D grid (`SharedField`) shared across all agents during one
 * evaluation tick. Agents write signals with `writeCell` and read them with `readCell`.
 * Because the backing array is passed by reference, sequential evaluators within the same
 * tick observe each other's writes — exactly the stigmergy contract required by the ant-hive
 * and racing benchmarks. Between ticks, `applyDecay` and `applyDiffusion` evolve field
 * dynamics; `clearField` resets it for the next generation.
 *
 * ### evaluation
 * `createCollectiveEvaluationContext` initialises a generation counter and binds the shared
 * field. `runCollectiveEvaluationTick` invokes each `AgentEvaluator` in declared order and
 * returns per-agent fitness plus the evaluation order. `resetCollectiveEvaluationState`
 * advances the generation tick and zeros the field, ready for the next generation.
 *
 * ### metrics
 * `computeRoleDivergenceMetric` measures structural divergence between two agents using the
 * L1 (Manhattan) distance over their module-size distributions — a zero score means identical
 * compositions. `createOpponentSnapshotPool` and `addOpponentSnapshot` maintain a rolling
 * fixed-capacity buffer of deep-cloned opponent payloads for tournament evaluation; the oldest
 * snapshot is evicted FIFO when the pool reaches capacity.
 *
 * ## Determinism contract
 * Same agent count + same evaluator array + same initial field ⇒ identical
 * `CollectiveTickResult` for every call. `applyDecay` and `applyDiffusion` are both
 * deterministic pure functions that produce new `SharedField` instances without mutating
 * the source.
 *
 * @example
 * ```ts
 * import {
 *   createSharedField,
 *   writeCell,
 *   readCell,
 *   applyDecay,
 *   createCollectiveEvaluationContext,
 *   runCollectiveEvaluationTick,
 *   resetCollectiveEvaluationState,
 *   computeRoleDivergenceMetric,
 *   createOpponentSnapshotPool,
 *   addOpponentSnapshot,
 * } from './neat.nge-collective';
 *
 * // 1. Create a 10×10 pheromone field shared across 3 agents.
 * const field = createSharedField(10, 10);
 * const context = createCollectiveEvaluationContext(3, field);
 *
 * // 2. Run one evaluation tick — agent 0 writes a signal; agent 1 reads it.
 * const result = runCollectiveEvaluationTick(context, [
 *   (_idx, f) => { writeCell(f, 0, 0, 1.0); return 10; },
 *   (_idx, f) => readCell(f, 0, 0) * 5,   // sees agent 0's write
 *   () => 0,
 * ]);
 * // result.agentFitness === [10, 5, 0]
 *
 * // 3. Advance to the next generation with decay applied.
 * const decayed = applyDecay(context.field, 0.95);
 * const nextContext = resetCollectiveEvaluationState({ ...context, field: decayed });
 *
 * // 4. Measure role divergence between two agents' module distributions.
 * const divergence = computeRoleDivergenceMetric([10, 5, 0], [0, 5, 10]); // 20
 *
 * // 5. Maintain a rolling opponent pool for tournament selection.
 * let pool = createOpponentSnapshotPool(5);
 * pool = addOpponentSnapshot(pool, 'agent:alpha', { fitness: 42 }, nextContext.generationTick);
 * ```
 */

// --- Shared types ---
export type {
  AgentEvaluator,
  CollectiveEvaluationContext,
  CollectiveTickResult,
  OpponentSnapshot,
  OpponentSnapshotPool,
  SharedField,
} from './neat.nge-collective.types';

// --- Shared field primitives ---
export {
  applyDecay,
  applyDiffusion,
  clearField,
  createSharedField,
  readCell,
  writeCell,
} from './neat.nge-collective.shared-field';

// --- Evaluation orchestration ---
export {
  createCollectiveEvaluationContext,
  resetCollectiveEvaluationState,
  runCollectiveEvaluationTick,
} from './neat.nge-collective.evaluation';

// --- Observability metrics ---
export {
  addOpponentSnapshot,
  computeRoleDivergenceMetric,
  createOpponentSnapshotPool,
} from './neat.nge-collective.metrics';
