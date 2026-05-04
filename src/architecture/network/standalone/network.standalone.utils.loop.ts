import type { StandaloneGenerationContext as GenerationContext } from '../network.types';
import { MASK_MULTIPLIER_IDENTITY } from './network.standalone.utils.types';
import {
  ensureActivationFunctionIndex,
  resolveSquashName,
} from './network.standalone.utils.activation';
import {
  buildNodeSumExpression,
  formatOutputArrayValues,
} from './network.standalone.utils.graph';

/**
 * Append the generated input-copy loop to the standalone body.
 *
 * @param generationContext Mutable generation context.
 * @returns Void.
 */
export function appendInputSeedLine(
  generationContext: GenerationContext,
): void {
  generationContext.inputNodeIndexes.forEach((nodeIndex, inputIndex) => {
    generationContext.bodyLines.push(`A[${nodeIndex}] = input[${inputIndex}];`);
  });
}

/**
 * Append compute lines for all non-input nodes.
 *
 * @param generationContext Mutable generation context.
 * @returns Void.
 */
export function appendAllNodeComputationLines(
  generationContext: GenerationContext,
): void {
  for (const nodeTraversalIndex of generationContext.activationNodeIndexes) {
    appendSingleNodeComputationLines(generationContext, nodeTraversalIndex);
  }
}

/**
 * Append generated return line for output activations.
 *
 * @param generationContext Mutable generation context.
 * @param outputIndexes Output node indexes.
 * @returns Void.
 */
export function appendOutputReturnLine(
  generationContext: GenerationContext,
  outputIndexes: number[],
): void {
  generationContext.bodyLines.push(
    `return [${formatOutputArrayValues(outputIndexes)}];`,
  );
}

/**
 * Append state and activation lines for one node.
 *
 * @param generationContext Mutable generation context.
 * @param nodeTraversalIndex Node index currently being emitted.
 * @returns Void.
 */
function appendSingleNodeComputationLines(
  generationContext: GenerationContext,
  nodeTraversalIndex: number,
): void {
  const currentNode =
    generationContext.standaloneProps.nodes[nodeTraversalIndex];
  const squashName = resolveSquashName(currentNode, nodeTraversalIndex);
  const activationFunctionIndex = ensureActivationFunctionIndex(
    generationContext,
    squashName,
    currentNode.squash,
    nodeTraversalIndex,
  );
  const sumExpression = buildNodeSumExpression(currentNode, nodeTraversalIndex);
  appendStateLine(
    generationContext,
    nodeTraversalIndex,
    sumExpression,
    currentNode.bias,
  );
  appendActivationLine(
    generationContext,
    nodeTraversalIndex,
    activationFunctionIndex,
    currentNode.mask,
  );
}

/**
 * Append generated state assignment line for one node.
 *
 * @param generationContext Mutable generation context.
 * @param nodeTraversalIndex Node index.
 * @param sumExpression Generated sum expression.
 * @param biasValue Node bias.
 * @returns Void.
 */
function appendStateLine(
  generationContext: GenerationContext,
  nodeTraversalIndex: number,
  sumExpression: string,
  biasValue: number,
): void {
  generationContext.bodyLines.push(
    `S[${nodeTraversalIndex}] = ${sumExpression} + ${biasValue};`,
  );
}

/**
 * Append generated activation assignment line for one node.
 *
 * @param generationContext Mutable generation context.
 * @param nodeTraversalIndex Node index.
 * @param activationFunctionIndex Function table index.
 * @param maskValue Multiplicative mask.
 * @returns Void.
 */
function appendActivationLine(
  generationContext: GenerationContext,
  nodeTraversalIndex: number,
  activationFunctionIndex: number,
  maskValue: number,
): void {
  const maskSuffix = buildMaskSuffix(maskValue);
  generationContext.bodyLines.push(
    `A[${nodeTraversalIndex}] = F[${activationFunctionIndex}](S[${nodeTraversalIndex}])${maskSuffix};`,
  );
}

/**
 * Build optional activation mask suffix for generated assignment line.
 *
 * @param maskValue Multiplicative mask value.
 * @returns Empty suffix for identity, otherwise multiplicative fragment.
 */
function buildMaskSuffix(maskValue: number): string {
  if (maskValue === MASK_MULTIPLIER_IDENTITY) {
    return '';
  }

  return ` * ${maskValue}`;
}
