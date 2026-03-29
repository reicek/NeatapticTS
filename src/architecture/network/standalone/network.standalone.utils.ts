/**
 * Standalone code-generation boundary for turning a live `Network` into a
 * self-contained inference function.
 *
 * This chapter exists for the moment when the graph has finished evolving or
 * training and the next question is no longer "how do I mutate it?" but "how
 * do I ship or inspect its forward pass without the whole runtime?" The
 * standalone generator answers by snapshotting node state, connection order,
 * gating terms, and activation functions into a plain JavaScript source string
 * that can be persisted, reviewed, or evaluated elsewhere.
 *
 * The folder is intentionally split like a miniature compiler pipeline.
 * `setup` validates the network and seeds stable indexes, `graph` and `loop`
 * collect the execution lines, `activation` and `coverage` normalize emitted
 * function bodies, and `finalize` folds everything into the finished source
 * string. That keeps the README focused on the transformation stages instead
 * of a flat shelf of string helpers.
 *
 * ```mermaid
 * flowchart LR
 *   Runtime[Runtime Network] --> Snapshot[Seed stable indexes and state]
 *   Snapshot --> Emit[Emit computation lines]
 *   Emit --> Assemble[Assemble source string]
 *   Assemble --> ActivateFn[Standalone activator]
 * ```
 *
 * The important constraint is scope. The generated artifact is for forward
 * inference only. It keeps fixed weights, gating, and simple recurrent
 * self-connections, but it intentionally omits training-time randomness,
 * backpropagation, and any runtime machinery that requires the full library
 * around it.
 *
 * Example: emit a portable source string and persist it with the rest of a
 * model package.
 *
 * ```ts
 * const source = network.standalone();
 * console.log(source.slice(0, 120));
 * ```
 *
 * Example: evaluate the emitted source in a sandbox and compare inference
 * results against the original runtime.
 *
 * ```ts
 * const source = network.standalone();
 * const activate = new Function(`return ${source}`)() as (
 *   input: number[],
 * ) => number[];
 * const outputValues = activate([0.2, 0.8]);
 * ```
 */
import type Network from '../../network/network';
import { assembleStandaloneSource } from './network.standalone.utils.finalize';
import { collectOutputIndexes } from './network.standalone.utils.graph';
import {
  appendAllNodeComputationLines,
  appendInputSeedLine,
  appendOutputReturnLine,
} from './network.standalone.utils.loop';
import {
  asStandaloneProps,
  createGenerationContext,
  ensureOutputNodesExist,
  seedNodeIndexesAndState,
} from './network.standalone.utils.setup';

/**
 * Generate a standalone JavaScript source string that returns an `activate(input:number[])` function.
 *
 * Implementation Steps:
 *  1. Validate presence of output nodes (must produce something observable).
 *  2. Assign stable sequential indices to nodes (used as array offsets in generated code).
 *  3. Collect initial activation/state values into typed array initializers for warm starting.
 *  4. For each non-input node, build a line computing S[i] (pre-activation sum with bias) and A[i]
 *     (post-activation output). Gating multiplies activation by gate activations; self-connection adds
 *     recurrent term S[i] * weight before activation.
 *  5. De-duplicate activation functions: each unique squash name is emitted once; references become
 *     indices into array F of function references for compactness.
 *  6. Emit an IIFE producing the activate function with internal arrays A (activations) and S (states).
 *
 * @param net Network instance to snapshot.
 * @returns Source string (ES5-compatible) – safe to eval in sandbox to obtain activate function.
 * @throws If network lacks output nodes.
 */
export function generateStandalone(net: Network): string {
  // Step 1: Convert to the internal network view and validate required structure.
  const standaloneProps = asStandaloneProps(net);
  ensureOutputNodesExist(standaloneProps);

  // Step 2: Initialize generation context and seed runtime arrays.
  const generationContext = createGenerationContext(standaloneProps);
  seedNodeIndexesAndState(generationContext);

  // Step 3: Build activate(input) body using tiny orchestration helpers.
  appendInputSeedLine(generationContext);
  appendAllNodeComputationLines(generationContext);
  const outputIndexes = collectOutputIndexes(generationContext);
  appendOutputReturnLine(generationContext, outputIndexes);

  // Step 4: Fold all generated parts into final standalone source.
  return assembleStandaloneSource(generationContext);
}
