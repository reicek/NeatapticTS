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
 * Standalone forward pass code generator.
 *
 * Purpose:
 *  Transforms a dynamic Network instance (object graph with Nodes / Connections / gating metadata)
 *  into a self-contained JavaScript function string that, when evaluated, returns an `activate(input)`
 *  function capable of performing forward propagation without the original library runtime.
 *
 * Why generate code?
 *  - Deployment: Embed a compact, dependency‑free inference function in environments where bundling
 *    the full evolutionary framework is unnecessary (e.g. model cards, edge scripts, CI sanity checks).
 *  - Performance: Remove dynamic indirection (property lookups, virtual dispatch) by specializing
 *    the computation graph into straight‑line code and simple loops; JS engines can optimize this.
 *  - Readability: Emitted source is human-readable so users can inspect weighted sums and activations.
 *
 * Features Supported:
 *  - Standard feed‑forward connections with optional gating (multiplicative modulation).
 *  - Single self-connection per node (handled as recurrent term S[i] * weight before activation).
 *  - Arbitrary activation functions: built‑in ones are emitted via canonical snippets; custom user
 *    functions are stringified and sanitized via stripCoverage(). Arrow or anonymous functions are
 *    normalized into named `function <name>(...)` forms for clarity and stable ordering.
 *
 * Not Supported / Simplifications:
 *  - No dynamic dropout, noise injection, or stochastic depth—those would require runtime randomness.
 *  - Assumes all node indices are stable and sequential (enforced prior to generation).
 *  - Gradient / backprop logic intentionally omitted (forward inference only).
 */

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
