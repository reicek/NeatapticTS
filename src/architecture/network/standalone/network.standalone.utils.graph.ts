import type Node from '../../node';
import type {
  NodeWithIndex,
  StandaloneGenerationContext as GenerationContext,
} from '../network.types';
import {
  ACTIVATION_PRECISION_F16,
  SINGLE_TERM_FALLBACK,
} from './network.standalone.utils.types';

/**
 * Build the generated pre-activation summation expression for one node.
 *
 * The expression combines inbound weighted terms and optional recurrent
 * self-connection terms, then folds them into a single JavaScript expression
 * emitted into standalone network source.
 *
 * @param generationContext Standalone code-generation context.
 * @param currentNode Current node.
 * @param nodeTraversalIndex Node index.
 * @returns String expression used for generated `S[index]` assignment.
 * @example
 * ```ts
 * const sumExpression = buildNodeSumExpression(context, node, nodeIndex);
 * // Example output: "A[2] * 0.5 + S[4] * 0.9"
 * ```
 */
export function buildNodeSumExpression(
  generationContext: GenerationContext,
  currentNode: Node,
  nodeTraversalIndex: number,
): string {
  const incomingTerms = collectIncomingTerms(generationContext, currentNode);
  const selfConnectionTerms = collectSelfConnectionTerms(
    generationContext,
    currentNode,
    nodeTraversalIndex,
  );
  const allTerms = mergeTermCollections(incomingTerms, selfConnectionTerms);
  return foldTermsIntoExpression(allTerms);
}

/**
 * Collect the output-node index sequence used by generated return paths.
 *
 * The returned array preserves traversal order so emitted output selectors map
 * consistently to the public standalone activation result vector.
 *
 * @param generationContext Mutable generation context.
 * @returns Output indexes used for result array emission.
 * @example
 * ```ts
 * const outputIndexes = collectOutputIndexes(context);
 * // outputIndexes can be passed to formatOutputArrayValues
 * ```
 */
export function collectOutputIndexes(
  generationContext: GenerationContext,
): number[] {
  return [...generationContext.outputNodeIndexes];
}

/**
 * Format output activation selectors for the generated return expression.
 *
 * Each output index is translated into a storage-buffer read expression and
 * joined as a comma-separated selector list for emitted array literals.
 *
 * @param generationContext Mutable generation context.
 * @param outputIndexes Output node indexes.
 * @returns Comma-separated `A[index]` selector list.
 * @example
 * ```ts
 * const selectorList = formatOutputArrayValues(context, [3, 4]);
 * // Example output: "A[3],A[4]"
 * ```
 */
export function formatOutputArrayValues(
  generationContext: GenerationContext,
  outputIndexes: number[],
): string {
  const outputTerms: string[] = [];
  for (const outputIndex of outputIndexes) {
    outputTerms.push(
      buildStoredValueReadExpression(generationContext, 'A', outputIndex),
    );
  }
  return outputTerms.join(',');
}

/**
 * Collect feed-forward inbound connection terms for a node.
 *
 * @param currentNode Current node.
 * @returns Weighted term expressions.
 */
function collectIncomingTerms(
  generationContext: GenerationContext,
  currentNode: Node,
): string[] {
  const terms: string[] = [];

  for (const incomingConnection of currentNode.connections.in) {
    const fromNodeIndex = getOptionalNodeIndex(incomingConnection.from);
    if (typeof fromNodeIndex !== 'number') {
      continue;
    }

    let connectionTerm = `${buildStoredValueReadExpression(
      generationContext,
      'A',
      fromNodeIndex,
    )} * ${incomingConnection.weight}`;
    connectionTerm = appendGateMultiplier(
      generationContext,
      connectionTerm,
      incomingConnection.gater,
    );
    terms.push(connectionTerm);
  }

  return terms;
}

/**
 * Collect recurrent self-connection term for a node when present.
 *
 * @param currentNode Current node.
 * @param nodeTraversalIndex Node index used for self-state reference.
 * @returns Zero or one recurrent term expressions.
 */
function collectSelfConnectionTerms(
  generationContext: GenerationContext,
  currentNode: Node,
  nodeTraversalIndex: number,
): string[] {
  if (currentNode.connections.self.length === 0) {
    return [];
  }

  const selfConnection = currentNode.connections.self[0];
  let connectionTerm = `${buildStoredValueReadExpression(
    generationContext,
    'S',
    nodeTraversalIndex,
  )} * ${selfConnection.weight}`;
  connectionTerm = appendGateMultiplier(
    generationContext,
    connectionTerm,
    selfConnection.gater,
  );
  return [connectionTerm];
}

/**
 * Append a gate activation multiplier to a connection term when a gate exists.
 *
 * @param connectionTerm Base connection term.
 * @param gateNode Optional gate node.
 * @returns Term with optional gate multiplier.
 */
function appendGateMultiplier(
  generationContext: GenerationContext,
  connectionTerm: string,
  gateNode: Node | null,
): string {
  const gateNodeIndex = getOptionalNodeIndex(gateNode);
  if (typeof gateNodeIndex !== 'number') {
    return connectionTerm;
  }

  return `${connectionTerm} * ${buildStoredValueReadExpression(
    generationContext,
    'A',
    gateNodeIndex,
  )}`;
}

/**
 * Resolve optional generated node index from a node reference.
 *
 * @param nodeReference Optional node reference.
 * @returns Node index when available.
 */
function getOptionalNodeIndex(nodeReference: Node | null): number | undefined {
  if (!nodeReference) {
    return undefined;
  }

  const indexedNode = nodeReference as Partial<NodeWithIndex>;
  return indexedNode.index;
}

/**
 * Merge two term lists into a single ordered list.
 *
 * @param firstTerms First term collection.
 * @param secondTerms Second term collection.
 * @returns Combined term collection.
 */
function mergeTermCollections(
  firstTerms: string[],
  secondTerms: string[],
): string[] {
  const mergedTerms = [...firstTerms];
  for (const termValue of secondTerms) {
    mergedTerms.push(termValue);
  }
  return mergedTerms;
}

/**
 * Fold a term collection into a summation expression.
 *
 * @param allTerms Term collection.
 * @returns Summation expression or fallback zero literal.
 */
function foldTermsIntoExpression(allTerms: string[]): string {
  if (allTerms.length === 0) {
    return SINGLE_TERM_FALLBACK;
  }

  return allTerms.join(' + ');
}

/**
 * Build one storage read expression for generated standalone buffers.
 *
 * @param generationContext Mutable generation context.
 * @param bufferName Generated buffer variable name.
 * @param nodeIndex Indexed storage slot.
 * @returns Native array read or float16 decode expression.
 */
function buildStoredValueReadExpression(
  generationContext: GenerationContext,
  bufferName: 'A' | 'S',
  nodeIndex: number,
): string {
  if (
    generationContext.resolvedActivationPrecision === ACTIVATION_PRECISION_F16
  ) {
    return `${resolveStandaloneBufferName(bufferName)}[${nodeIndex}]`;
  }

  return `${bufferName}[${nodeIndex}]`;
}

/**
 * Resolve the generated working-buffer variable name for one standalone buffer.
 *
 * @param bufferName Persistent standalone storage name.
 * @returns Working-buffer name used during one float16 activation call.
 */
function resolveStandaloneBufferName(bufferName: 'A' | 'S'): 'WA' | 'WS' {
  return bufferName === 'A' ? 'WA' : 'WS';
}
